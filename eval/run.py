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


# Mechanism E4 (15 aout, Sprint 1): mirrored per-speaker stores. Real
# effect measured (AFA paper, cited in research/Haki_Livre_Construction_
# 2026-08-15.md): named-entity attribution in a store SHARED between two
# speakers falls to 35.7% (below chance -- the system actively drifts
# to the wrong person); with one store per speaker, 61.3% (+25 points).
# The convention read from the official harness code (mem0, OmniMemEval):
# two mirror stores per conversation -- for speaker A's store, A's own
# turns are role "user" and B's are "assistant" (and the reverse for B's
# store); the real speaker name is kept as an explicit prefix in the
# TEXT of every turn, so "who said what" survives the role normalization.
#
# Opt-in (config["speaker_mirror"]): the default (unset) ingestion path
# above is untouched, so every existing config/run stays reproducible
# exactly as before. Only acts on questions carrying `speakers` (LoCoMo;
# LongMemEval's "speaker" is already the normalized user/assistant role,
# no shared-store ambiguity to fix there -- see the Question docstring).
def mirror_subject(history_id: str, speaker: str) -> str:
    return f"{history_id}__{speaker}"[:128]


def question_events_mirrored(
    question: datasets.Question,
    speaker: str,
    org_id: str,
    project_id: str,
    run_id: str,
) -> list[dict]:
    subject = mirror_subject(question.history_id or question.qid, speaker)
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
                        {
                            "role": "user" if m.speaker == speaker else "assistant",
                            "content": f"{m.speaker}: {m.content}",
                        }
                        for m in session.messages
                    ],
                },
                "idempotency_key": f"{run_id}:{subject}:{i}",
            }
        )
    return events


def target_speakers(question: datasets.Question) -> list[str]:
    """Which mirror store(s) to query for this question: the ONE speaker
    explicitly named in the question text (whole-word, case-insensitive),
    or every speaker if zero or both are named -- "route to the subject
    the question names, both otherwise" (the book's own phrasing)."""
    named = [
        speaker
        for speaker in question.speakers
        if re.search(rf"\b{re.escape(speaker)}\b", question.question, re.IGNORECASE)
    ]
    return named if len(named) == 1 else question.speakers


def merge_packets(bodies: list[dict]) -> dict:
    """Merge several /v1/context response bodies (one per queried mirror
    store) into one packet: facts and episodes deduplicated by id (mirror
    stores are disjoint by construction, but a defensive dedup costs
    nothing), token_count summed, warnings/status of the LAST body kept
    (informational only, never affects scoring)."""
    if len(bodies) == 1:
        return bodies[0]
    seen_fact_ids: set[str] = set()
    seen_episode_ids: set[str] = set()
    facts: list[dict] = []
    episodes: list[dict] = []
    token_count = 0
    trace_ids = []
    for body in bodies:
        packet = body["packet"]
        for fact in packet.get("facts", []):
            if fact["id"] not in seen_fact_ids:
                seen_fact_ids.add(fact["id"])
                facts.append(fact)
        for episode in packet.get("episodes", []):
            if episode["event_id"] not in seen_episode_ids:
                seen_episode_ids.add(episode["event_id"])
                episodes.append(episode)
        token_count += body.get("token_count", 0)
        trace_ids.append(body.get("trace_id"))
    merged = dict(bodies[-1])
    merged["packet"] = {**bodies[-1]["packet"], "facts": facts, "episodes": episodes}
    merged["token_count"] = token_count
    merged["trace_ids"] = trace_ids
    return merged


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


def parse_mem0_judge_output(text: str) -> dict:
    """15 aout, Sprint 0 calibration: mirrors mem0's own judge parsing
    (evaluation/metrics/llm_judge.py -- `json.loads(extract_json(...))["label"]`,
    label in {"CORRECT", "WRONG"}). Deliberately simpler than
    `parse_judge_output` above: mem0's judge has no "abstained" concept and
    no `relies_on_outdated_information` field -- forcing either through
    this parser would silently invent signal mem0's own protocol never
    produces, defeating the point of calibrating against it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"label": "incorrect", "judge_reason": f"unparseable: {text[:120]}"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "incorrect", "judge_reason": f"unparseable: {text[:120]}"}
    label = str(data.get("label", "WRONG")).strip().upper()
    return {
        "label": "correct" if label == "CORRECT" else "incorrect",
        "judge_reason": text[:300],
    }


async def answer_mem0_baseline(
    client: ChatClient, system_prompt: str, user_template: str, question: str, context: str
) -> tuple[str, int, int]:
    """15 aout, Sprint 0 calibration: mirrors mem0's own
    `RAGManager.generate_response` (evaluation/src/rag.py) exactly -- a
    system message plus a user message with question BEFORE context (kept
    even though it reads backwards, because that is the calibration
    target's actual protocol), temperature 0."""
    prompt = user_template.format(question=question, context=context)
    result = await client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return result.content, result.prompt_tokens, result.completion_tokens


# 15 aout, Sprint 0 calibration: verbatim from the OFFICIAL LongMemEval repo
# (xiaowu0162/LongMemEval, src/evaluation/evaluate_qa.py::get_anscheck_prompt)
# -- five distinct judge templates dispatched by question type, plus a
# SEPARATE abstention template used whenever the qid carries "_abs"
# regardless of type (mirrors `abstention='_abs' in entry['question_id']`
# in the source). Deliberately five templates, not one generic judge: the
# temporal-reasoning one explicitly forgives off-by-one day errors, the
# knowledge-update one explicitly accepts an answer that also restates the
# superseded value alongside the correct one, the preference one grades
# against a rubric instead of an exact answer -- collapsing these into one
# prompt would stop this from being the calibration target's own
# instrument.
_LME_JUDGE_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}"
    "\n\nModel Response: {response}\n\nIs the model response correct? Answer yes "
    "or no only."
)
LME_JUDGE_TEMPLATES: dict[str, str] = {
    "single-session-user": _LME_JUDGE_STANDARD,
    "single-session-assistant": _LME_JUDGE_STANDARD,
    "multi-session": _LME_JUDGE_STANDARD,
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct "
        "answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a "
        "subset of the information required by the answer, answer no. In "
        "addition, do not penalize off-by-one errors for the number of days. If "
        "the question asks for the number of days/weeks/months, etc., and the "
        "model makes off-by-one errors (e.g., predicting 19 days when the answer "
        "is 18), the model's response is still correct. \n\nQuestion: {question}"
        "\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the "
        "model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be "
        "considered as correct as long as the updated answer is the required "
        "answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel "
        "Response: {response}\n\nIs the model response correct? Answer yes or "
        "no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized "
        "response, and a response from a model. Please answer yes if the "
        "response satisfies the desired response. Otherwise, answer no. The "
        "model does not need to reflect all the points in the rubric. The "
        "response is correct as long as it recalls and utilizes the user's "
        "personal information correctly.\n\nQuestion: {question}\n\nRubric: "
        "{answer}\n\nModel Response: {response}\n\nIs the model response "
        "correct? Answer yes or no only."
    ),
}
LME_JUDGE_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: "
    "{response}\n\nDoes the model correctly identify the question as "
    "unanswerable? Answer yes or no only."
)


# 15 aout: run_generation.py computes `max_retrieval_length = model_max_
# length - gen_length - 1000` and truncates the RENDERED history string to
# that many tokens, keeping the FIRST N (tokens[:max_retrieval_length] --
# the earliest sessions, not the most recent ones; an odd choice but
# faithfully reproduced since fidelity to the calibration target is the
# point, not second-guessing it). model2maxlength['gpt-4o-mini-2024-07-06']
# = 128000, gen_length defaults to 500 (non-CoT) -- both mirrored below.
# Estimated in chars/4 (datasets.estimate_tokens), consistent with the rest
# of this harness's token accounting, not a real tokenizer -- close enough
# to keep every real LongMemEval-S haystack (~115K tokens) under the limit
# without a new dependency; only matters when a haystack is unusually long.
LME_MODEL_MAX_TOKENS = 128_000
LME_GEN_LENGTH = 500
LME_MAX_RETRIEVAL_TOKENS = LME_MODEL_MAX_TOKENS - LME_GEN_LENGTH - 1000


async def answer_lme_baseline(
    client: ChatClient, user_template: str, history: str, question_date: str, question: str
) -> tuple[str, int, int]:
    """15 aout, Sprint 0 calibration: mirrors the official LongMemEval
    full-context baseline (src/generation/run_generation.py,
    `retriever_type="orig-session"`, `history_format="json"`,
    `useronly=false`, no chain-of-note) -- single user message, no system
    message (the official script never sends one), temperature 0."""
    max_chars = LME_MAX_RETRIEVAL_TOKENS * datasets.CHARS_PER_TOKEN
    if len(history) > max_chars:
        history = history[:max_chars]
    prompt = user_template.format(history=history, question_date=question_date, question=question)
    result = await client.chat(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=LME_GEN_LENGTH
    )
    return result.content, result.prompt_tokens, result.completion_tokens


async def judge_lme(
    client: ChatClient, question: datasets.Question, answer: str
) -> tuple[dict, int, int]:
    """15 aout, Sprint 0 calibration: mirrors `get_anscheck_prompt` +
    the official call site (evaluate_qa.py) exactly -- template dispatch by
    qtype, `_abs` in the qid overrides to the abstention template
    regardless of qtype (matching `abstention='_abs' in entry['question_id']`
    verbatim), gpt-4o-2024-08-06 model REQUIRED by the harness (see
    print_qa_metrics.py's own assertion -- config must set judge_model to
    it in mem0_calibration mode for LongMemEval, not gpt-4o-mini), single
    user message, max_tokens=10, label = 'yes' in response.lower()."""
    if question.abstention_expected and "_abs" in question.qid:
        template = LME_JUDGE_ABSTENTION_TEMPLATE
    else:
        template = LME_JUDGE_TEMPLATES.get(question.qtype, _LME_JUDGE_STANDARD)
    prompt = template.format(question=question.question, answer=question.answer, response=answer)
    result = await client.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
    label = "correct" if "yes" in result.content.strip().lower() else "incorrect"
    return {"label": label, "judge_reason": result.content.strip()[:120]}, result.prompt_tokens, result.completion_tokens


async def judge_mem0(
    client: ChatClient, judge_prompt: str, question: datasets.Question, answer: str
) -> tuple[dict, int, int]:
    """15 aout, Sprint 0 calibration: mirrors mem0's own
    `evaluate_llm_judge` (evaluation/metrics/llm_judge.py) exactly -- a
    single user message (no system message, unlike Haki's own `judge()`
    above), no per-qtype `extra` injection (mem0's ACCURACY_PROMPT has none
    -- inventing one would stop this from being the same instrument),
    `response_format=json_object`."""
    prompt = judge_prompt.format(
        question=question.question,
        gold_answer=question.answer,
        generated_answer=answer,
    )
    result = await client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    verdict = parse_mem0_judge_output(result.content)
    return verdict, result.prompt_tokens, result.completion_tokens


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
        # 13 aout, LongMemEval diagnostic (run 31705865474): this used to
        # unconditionally assert "the gold answer is the LATEST value" --
        # true for MOST knowledge-update questions, but not all: some
        # explicitly ask about a past/previous state ("what was my goal
        # BEFORE I updated it", "in the first three months"), where the
        # gold answer IS the earlier value on purpose. The old wording
        # made the judge override a correct historical answer because it
        # contradicted the judge's own assumption rather than the actual
        # gold text given below -- verified: gold "level 100", system
        # answer "100" (an exact match) still got labeled incorrect.
        # Deference to the literal {gold} value is the fix, not deleting
        # the heuristic (it is still the right default for most cases).
        extra = (
            "Note: this question type usually tests a knowledge UPDATE, where the "
            "history states an earlier value and a later replacement, and the gold "
            "answer below reflects the LATEST value -- an answer using the earlier "
            "value is then incorrect and relies on outdated information. BUT always "
            "defer to the actual gold answer given above this note, not to this "
            "general rule: if the gold answer itself IS the earlier/historical value "
            "(the question explicitly asks about a past state, e.g. 'before I "
            "changed it', 'when I started', 'in the first [period]'), then matching "
            "that earlier value is correct and does NOT rely on outdated information."
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
    if args.shard_count and args.shard_count > 1:
        selected = datasets.shard(selected, args.shard_index, args.shard_count)
    if not selected:
        print("aucune question sélectionnée (filtres trop restrictifs ?)")
        return 1

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config["selection"] = {
        "subset": subset,
        "types": types,
        "shard": f"{args.shard_index}/{args.shard_count}" if args.shard_count and args.shard_count > 1 else None,
    }
    config["run_id"] = run_id
    config["dataset"]["sha256"] = config["dataset"].get("sha256") or sha256_file(dataset_path)

    llm = llm_settings()
    prices = config.get("prices_per_mtok", {}).get(config["answer_model"], {})
    answer_client = ChatClient(llm["base_url"], llm["api_key"], config["answer_model"])
    judge_client = ChatClient(llm["base_url"], llm["api_key"], config["judge_model"])
    answer_prompt = (ROOT / config["prompts"]["answer"]).read_text(encoding="utf-8")
    judge_prompt = (ROOT / config["prompts"]["judge"]).read_text(encoding="utf-8")

    # 15 aout, Sprint 0 -- "mem0_calibration" protocol: reproduces the exact
    # published mem0 LoCoMo/LongMemEval protocol (system+user baseline
    # prompt with question BEFORE context, flat "{ts} | {speaker}: {text}"
    # transcript, no truncation, ACCURACY_PROMPT judge on BOTH systems) so
    # a run against THIS harness is directly comparable to the published
    # numbers (Mem0 66.88%, full-context 72.90%, etc. -- see
    # research/Haki_Livre_Construction_2026-08-15.md Partie 2). Every
    # other config is completely unaffected: this only activates when the
    # config explicitly opts in.
    calibration = config.get("protocol") == "mem0_calibration"
    # Two distinct calibration targets share the same protocol flag: LoCoMo
    # (mem0's own harness, ACCURACY_PROMPT judge) and LongMemEval (the
    # dataset's OWN official harness, 5 qtype-dispatched judge templates --
    # see judge_lme). Which one applies is entirely determined by the
    # dataset loader already selected in the config; no separate flag to
    # keep in sync.
    calibration_dataset = config["dataset"]["loader"] if calibration else None
    mem0_system_prompt = mem0_user_prompt = mem0_judge_prompt = lme_baseline_prompt = None
    if calibration_dataset == "locomo":
        mem0_system_prompt = (ROOT / "eval/prompts/mem0_baseline_system.txt").read_text(encoding="utf-8")
        mem0_user_prompt = (ROOT / "eval/prompts/mem0_baseline_user.txt").read_text(encoding="utf-8")
        mem0_judge_prompt = (ROOT / "eval/prompts/mem0_judge.txt").read_text(encoding="utf-8")
    elif calibration_dataset == "longmemeval":
        lme_baseline_prompt = (ROOT / "eval/prompts/lme_baseline_user.txt").read_text(encoding="utf-8")

    systems = set(args.systems.split(","))
    project_id = args.reuse_project or f"{config.get('haki_project_prefix', 'prj_eval')}_{run_id}"
    org_id = config.get("haki_org", "org_eval")
    haki: HakiClient | None = None
    api_key = None

    if "haki" in systems:
        haki = HakiClient(args.api_url)
        if not await haki.health():
            print(f"API Haki injoignable sur {args.api_url} — lance uvicorn d'abord.")
            return 1
        api_key = await haki.create_project_key(org_id, project_id, label=f"eval {run_id}")
        if args.reuse_project:
            print(
                f"projet eval REUTILISE: {project_id} — ingestion sautee entierement, "
                "en confiance que ce projet a deja ete peuple par un run precedent sur "
                "EXACTEMENT le meme sous-ensemble de questions"
            )
        else:
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
                "retrieval_need": datasets.classify_retrieval_need(question.question),
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
                # Mechanism E4: mirrored per-speaker stores, opt-in and only
                # for a question that actually carries named speakers
                # (LoCoMo). Every existing config/run keeps the single
                # shared-store path below unchanged.
                mirrored = bool(config.get("speaker_mirror")) and bool(question.speakers)
                if mirrored:
                    for speaker in question.speakers:
                        mirror = mirror_subject(question.history_id or question.qid, speaker)
                        if mirror not in ingested and not args.reuse_project:
                            events = question_events_mirrored(
                                question, speaker, org_id, project_id, run_id
                            )
                            await haki.capture(api_key, events)
                            await haki.consolidate_until_idle(api_key, project_id)
                            ingested.add(mirror)
                            ingested_now = True
                    ingest_s = time.perf_counter() - t0

                    bodies = []
                    latencies = []
                    for speaker in target_speakers(question):
                        mirror = mirror_subject(question.history_id or question.qid, speaker)
                        one_body, one_latency = await haki.context(
                            api_key,
                            project_id,
                            mirror,
                            question.question,
                            config.get("context_budget_tokens", 900),
                            as_of=question.as_of,
                        )
                        bodies.append(one_body)
                        latencies.append(one_latency)
                    body = merge_packets(bodies)
                    latency_ms = sum(latencies)
                else:
                    if subject not in ingested and not args.reuse_project:
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
                        as_of=question.as_of,
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
                if calibration_dataset == "locomo":
                    verdict, jptok, jctok = await judge_mem0(
                        judge_client, mem0_judge_prompt, question, answer
                    )
                elif calibration_dataset == "longmemeval":
                    verdict, jptok, jctok = await judge_lme(judge_client, question, answer)
                else:
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
                    # Split, not summed: cost_query_usd is paid on EVERY
                    # question (answer + judge) and is what actually scales
                    # with traffic; cost_ingest_usd is paid ONCE per history
                    # (extraction) and amortizes over however many questions
                    # get asked about that same subject afterwards. Reporting
                    # only their sum (the old cost_usd) let a single-question-
                    # per-history benchmark like LongMemEval_S make Haki look
                    # more expensive than a full-context baseline, when the
                    # two costs behave completely differently at scale —
                    # flagged externally (12-13 aout) after the episode-
                    # budget-reservation chantier flipped this comparison.
                    "cost_query_usd": cost_usd(ptok + jptok, ctok + jctok, prices),
                    "cost_ingest_usd": (
                        cost_usd(
                            record["history_tokens_est"],
                            int(record["history_tokens_est"] * 0.05),
                            prices,
                        )
                        if ingested_now
                        else 0.0
                    ),
                }
                record["systems"]["haki"]["cost_usd"] = (
                    record["systems"]["haki"]["cost_query_usd"]
                    + record["systems"]["haki"]["cost_ingest_usd"]
                )
                print(
                    f"  haki: {verdict['label']}"
                    f"{' OUTDATED' if verdict.get('outdated') else ''} "
                    f"(packet {body.get('token_count')} tok, {len(facts)} faits, "
                    f"context {latency_ms:.0f} ms, ingest {ingest_s:.0f} s)"
                )

            if "baseline" in systems:
                if calibration_dataset == "locomo":
                    # No truncation: the mem0 full-context baseline this
                    # calibrates against never truncates (chunk_size=-1 in
                    # RAGManager) -- see Sprint 0 in
                    # research/Haki_Livre_Construction_2026-08-15.md.
                    kept, truncated = question.sessions, False
                    transcript = datasets.render_mem0_transcript(kept)
                    answer, ptok, ctok = await answer_mem0_baseline(
                        answer_client, mem0_system_prompt, mem0_user_prompt,
                        question.question, transcript,
                    )
                    verdict, jptok, jctok = await judge_mem0(
                        judge_client, mem0_judge_prompt, question, answer
                    )
                elif calibration_dataset == "longmemeval":
                    # "orig-session", topk effectively unbounded (the
                    # official baseline's own topk_context=1000 config
                    # exceeds any real haystack session count) -- no
                    # truncation here either.
                    kept, truncated = question.sessions, False
                    history = datasets.render_lme_session_history(kept)
                    answer, ptok, ctok = await answer_lme_baseline(
                        answer_client, lme_baseline_prompt, history,
                        question.question_date or "unknown", question.question,
                    )
                    verdict, jptok, jctok = await judge_lme(judge_client, question, answer)
                else:
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
                    # No ingestion step for the full-context baseline -- the
                    # whole cost is per-query, every time.
                    "cost_query_usd": cost_usd(ptok + jptok, ctok + jctok, prices),
                    "cost_ingest_usd": 0.0,
                    "cost_usd": cost_usd(ptok + jptok, ctok + jctok, prices),
                }
                print(
                    f"  base: {verdict['label']}"
                    f"{' OUTDATED' if verdict.get('outdated') else ''} "
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

    if "haki" in systems and not args.keep_data and not args.reuse_project:
        deleted = await cleanup_project(project_id)
        print(f"projet {project_id} nettoyé: {deleted}")
    elif args.reuse_project:
        print(f"projet reutilise {project_id} laisse intact (jamais nettoye par --reuse-project)")
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
    parser.add_argument(
        "--reuse-project",
        default=None,
        help=(
            "reutilise un projet Haki deja peuple par un run precedent (ingestion "
            "sautee entierement) au lieu d'un projet fraichement genere -- pour "
            "retester un parametre de LECTURE seule (ex. context_budget_tokens, "
            "qui n'intervient qu'a la requete /v1/context) sans repayer "
            "l'extraction, de loin le plus gros poste de cout sur un dataset ou "
            "chaque question a son propre historique (LongMemEval). Implique "
            "--keep-data. Le sous-ensemble de questions DOIT etre exactement le "
            "meme que celui qui a peuple ce projet -- aucune verification cote "
            "harnais, un ecart se traduirait par des paquets de contexte vides "
            "plutot qu'une erreur explicite."
        ),
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="index de ce shard (0-based), avec --shard-count"
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="nombre total de shards — repartit par history_id (jamais une conversation coupee en deux), pour paralleliser un run complet sur plusieurs jobs cloud bornes a 6h chacun",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
