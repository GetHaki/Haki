"""Evaluation pipeline CLI (sprint 10).

For each selected question, two systems are compared in the SAME protocol
(same answer model, same answer prompt, same judge, temperature 0):

- **haki**: ingest the history sessions as events (subject = the dataset
  question's user, occurred_at = dataset dates), consolidate, fetch a
  ContextPacket via POST /v1/context, then answer with the packet injected;
- **baseline**: full-context reference — the whole history (or the most
  recent sessions fitting `baseline_max_context_tokens`) in the prompt.

The judge (LLM-as-judge, versioned prompt) labels each answer
correct/incorrect/abstained and flags answers relying on outdated
(superseded) information -> contradiction leakage.

Usage:
    uv run python -m eval.run --config eval/configs/longmemeval_s.json \
        --subset 15 --types knowledge-update,temporal-reasoning

Requires: the Haki API running with HAKI_LLM_PROVIDER=openai and
HAKI_ADMIN_KEY set (pass the same value via HAKI_EVAL_ADMIN_KEY), Postgres
up, and the dataset downloaded (`uv run python -m eval.download <config>`).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval import datasets, metrics
from eval.env import llm_settings
from eval.haki_client import HakiClient, cleanup_project
from eval.llm import ChatClient
from eval.report import write_reports

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_dataset(config: dict) -> Path:
    dataset_path = ROOT / config["dataset"]["file"]
    if not dataset_path.exists():
        raise SystemExit(
            f"dataset manquant: {dataset_path} — lance "
            f"`uv run python -m eval.download <config>` d'abord."
        )
    expected = config["dataset"].get("sha256")
    actual = sha256_file(dataset_path)
    if expected and expected != actual:
        raise SystemExit(
            f"checksum mismatch pour {dataset_path}:\n  attendu {expected}\n  obtenu  {actual}\n"
            "Le run n'est PAS reproductible avec ce fichier — retélécharge-le."
        )
    return dataset_path


def question_events(question: datasets.Question, org_id: str, project_id: str, run_id: str) -> list[dict]:
    subject = (question.history_id or question.qid)[:128]
    events = []
    for i, session in enumerate(question.sessions):
        events.append(
            {
                "org_id": org_id,
                "project_id": project_id,
                "subject_type": "user",
                "subject_id": subject,
                "kind": "chat_session",
                "occurred_at": session.date.astimezone(timezone.utc).isoformat(),
                "payload": {
                    "session_id": session.session_id,
                    "date": session.date.astimezone(timezone.utc).isoformat(),
                    "messages": [
                        {"role": m.speaker, "content": m.content} for m in session.messages
                    ],
                },
                "idempotency_key": f"{run_id}:{subject}:{i}",
            }
        )
    return events


def parse_judge_output(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"label": "incorrect", "outdated": False, "judge_reason": f"unparseable: {text[:120]}"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "incorrect", "outdated": False, "judge_reason": f"unparseable: {text[:120]}"}
    label = str(data.get("label", "incorrect")).lower()
    if label not in metrics.LABELS:
        label = "incorrect"
    return {
        "label": label,
        "outdated": bool(data.get("relies_on_outdated_information", False)),
        "judge_reason": str(data.get("reason", ""))[:300],
    }


def cost_usd(prompt_tokens: int, completion_tokens: int, prices: dict) -> float:
    return (
        prompt_tokens * prices.get("input", 0.0) / 1_000_000
        + completion_tokens * prices.get("output", 0.0) / 1_000_000
    )


async def answer_with_memory(
    client: ChatClient, answer_prompt: str, question: datasets.Question, memory: str
) -> tuple[str, int, int]:
    prompt = answer_prompt.format(
        question_date=question.question_date or "unknown",
        memory=memory,
        question=question.question,
    )
    result = await client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    return result.content, result.prompt_tokens, result.completion_tokens


async def judge(
    client: ChatClient, judge_prompt: str, question: datasets.Question, answer: str
) -> tuple[dict, int, int]:
    extra = ""
    if question.qtype in metrics.LEAKAGE_TYPES:
        extra = (
            "Note: this question tests a knowledge UPDATE. The history states an earlier "
            "value and a later replacement; the gold answer is the LATEST value. An answer "
            "with the earlier value is incorrect and relies on outdated information."
        )
    if question.abstention_expected:
        extra += (
            "\nNote: the gold answer indicates this question is NOT answerable from the "
            "history; a system that invents an answer is incorrect, a proper abstention "
            "is the expected behavior."
        )
    prompt = judge_prompt.format(
        question_date=question.question_date or "unknown",
        question=question.question,
        gold=question.answer,
        answer=answer,
        extra=extra.strip(),
    )
    result = await client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    verdict = parse_judge_output(result.content)
    return verdict, result.prompt_tokens, result.completion_tokens


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset_path = check_dataset(config)
    questions = datasets.LOADERS[config["dataset"]["loader"]](dataset_path)
    types = args.types.split(",") if args.types else config.get("selection", {}).get("types")
    subset = args.subset if args.subset is not None else config.get("selection", {}).get("subset")
    selected = datasets.select(questions, subset=subset, types=types)
    if not selected:
        print("aucune question sélectionnée (filtres trop restrictifs ?)")
        return 1

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config["selection"] = {"subset": subset, "types": types}
    config["run_id"] = run_id
    config["dataset"]["sha256"] = config["dataset"].get("sha256") or sha256_file(dataset_path)

    llm = llm_settings()
    prices = config.get("prices_per_mtok", {}).get(config["answer_model"], {})
    answer_client = ChatClient(llm["base_url"], llm["api_key"], config["answer_model"])
    judge_client = ChatClient(llm["base_url"], llm["api_key"], config["judge_model"])
    answer_prompt = (ROOT / config["prompts"]["answer"]).read_text(encoding="utf-8")
    judge_prompt = (ROOT / config["prompts"]["judge"]).read_text(encoding="utf-8")

    systems = set(args.systems.split(","))
    project_id = f"{config.get('haki_project_prefix', 'prj_eval')}_{run_id}"
    org_id = config.get("haki_org", "org_eval")
    haki: HakiClient | None = None
    api_key = None

    if "haki" in systems:
        haki = HakiClient(args.api_url)
        if not await haki.health():
            print(f"API Haki injoignable sur {args.api_url} — lance uvicorn d'abord.")
            return 1
        api_key = await haki.create_project_key(org_id, project_id, label=f"eval {run_id}")
        print(f"projet eval: {project_id}")

    records: list[dict] = []
    ingested: set[str] = set()
    started = time.perf_counter()
    try:
        for index, question in enumerate(selected, start=1):
            subject = (question.history_id or question.qid)[:128]
            record: dict = {
                "qid": question.qid,
                "qtype": question.qtype,
                "abstention_expected": question.abstention_expected,
                "question": question.question,
                "gold_answer": question.answer,
                "n_sessions": len(question.sessions),
                "history_tokens_est": datasets.estimate_tokens(
                    datasets.render_transcript(question.sessions)
                ),
                "systems": {},
            }
            print(
                f"[{index}/{len(selected)}] {question.qid} ({question.qtype}, "
                f"{len(question.sessions)} sessions, ~{record['history_tokens_est']} tok)"
            )

            if "haki" in systems:
                t0 = time.perf_counter()
                ingest_s = 0.0
                ingested_now = False
                if subject not in ingested:
                    events = question_events(question, org_id, project_id, run_id)
                    await haki.capture(api_key, events)
                    await haki.consolidate_until_idle(api_key, project_id)
                    ingested.add(subject)
                    ingest_s = time.perf_counter() - t0
                    ingested_now = True

                body, latency_ms = await haki.context(
                    api_key,
                    project_id,
                    subject,
                    question.question,
                    config.get("context_budget_tokens", 900),
                )
                facts = body["packet"]["facts"]
                episodes = body["packet"].get("episodes", [])
                memory = (
                    "Known facts about the user (from the memory system):\n"
                    + datasets.render_facts(facts)
                    + "\nDated events from the source history:\n"
                    + datasets.render_episodes(episodes)
                )
                answer, ptok, ctok = await answer_with_memory(
                    answer_client, answer_prompt, question, memory
                )
                verdict, jptok, jctok = await judge(judge_client, judge_prompt, question, answer)
                record["systems"]["haki"] = {
                    "answer": answer,
                    **verdict,
                    "packet_facts": len(facts),
                    "packet_episodes": len(episodes),
                    "packet": [
                        {
                            "predicate": f.get("predicate"),
                            "value": f.get("value"),
                            "valid_from": f.get("valid_from"),
                        }
                        for f in facts
                    ],
                    "packet_warnings": body["packet"].get("warnings", []),
                    "context_tokens": body.get("token_count"),
                    "trace_id": body.get("trace_id"),
                    "latency_ms": round(latency_ms, 1),
                    "ingest_seconds": round(ingest_s, 1),
                    # Server-side extraction is one LLM pass per session event
                    # (documented estimate: history tokens in, ~5% out), paid
                    # once per ingested history, not per question.
                    "extraction_tokens_est": record["history_tokens_est"] if ingested_now else 0,
                    "cost_usd": cost_usd(ptok + jptok, ctok + jctok, prices)
                    + (
                        cost_usd(
                            record["history_tokens_est"],
                            int(record["history_tokens_est"] * 0.05),
                            prices,
                        )
                        if ingested_now
                        else 0.0
                    ),
                }
                print(
                    f"  haki: {verdict['label']}"
                    f"{' OUTDATED' if verdict['outdated'] else ''} "
                    f"(packet {body.get('token_count')} tok, {len(facts)} faits, "
                    f"context {latency_ms:.0f} ms, ingest {ingest_s:.0f} s)"
                )

            if "baseline" in systems:
                kept, truncated = datasets.truncate_sessions(
                    question.sessions, config.get("baseline_max_context_tokens", 100_000)
                )
                transcript = datasets.render_transcript(kept)
                memory = "Conversation history (most recent sessions first kept, chronological order):\n" + transcript
                answer, ptok, ctok = await answer_with_memory(
                    answer_client, answer_prompt, question, memory
                )
                verdict, jptok, jctok = await judge(judge_client, judge_prompt, question, answer)
                record["systems"]["baseline"] = {
                    "answer": answer,
                    **verdict,
                    "sessions_used": len(kept),
                    "truncated": truncated,
                    "context_tokens": ptok,
                    "latency_ms": None,
                    "cost_usd": cost_usd(ptok + jptok, ctok + jctok, prices),
                }
                print(
                    f"  base: {verdict['label']}"
                    f"{' OUTDATED' if verdict['outdated'] else ''} "
                    f"({len(kept)}/{len(question.sessions)} sessions, {ptok} tok)"
                )

            records.append(record)
            # Incremental, crash-safe: a valid report of completed questions
            # exists even if the run dies (flaky network, Ctrl-C).
            write_reports(
                RESULTS_DIR, config["name"], run_id, config, records,
                metrics.aggregate(records),
            )
    finally:
        await answer_client.close()
        await judge_client.close()
        if haki is not None:
            await haki.close()

    summary = metrics.aggregate(records)
    json_path, md_path = write_reports(
        RESULTS_DIR, config["name"], run_id, config, records, summary
    )
    elapsed = time.perf_counter() - started
    print(f"\nrapport: {json_path}\n         {md_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"durée totale: {elapsed / 60:.1f} min")

    if "haki" in systems and not args.keep_data:
        deleted = await cleanup_project(project_id)
        print(f"projet {project_id} nettoyé: {deleted}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Haki public benchmark harness")
    parser.add_argument("--config", required=True, help="eval/configs/<benchmark>.json")
    parser.add_argument("--subset", type=int, default=None, help="N premières questions (déterministe)")
    parser.add_argument("--types", default=None, help="filtre CSV, ex: knowledge-update,temporal-reasoning")
    parser.add_argument("--systems", default="haki,baseline", help="haki,baseline | haki | baseline")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--keep-data", action="store_true", help="ne pas nettoyer le projet eval")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
