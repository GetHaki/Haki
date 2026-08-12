"""Lexical recall of what /v1/context actually serves, against the source
text the answer depends on (app/context episode-budget-reservation chantier,
12 aout 2026).

No published benchmark paper reports this number for a facts+episodes
memory system. Definition here: for each question, take the union of
unique words (>=4 chars) in its `evidence_sessions` (falls back to all
`sessions` when a dataset doesn't label evidence) and compare against the
union of unique words in the served packet (rendered facts + rendered
episodes, same text the answer prompt receives) -- recall = overlap /
evidence words. A rough proxy (word overlap, not semantic equivalence) but
cheap, deterministic, and directly measures whether the two-tier-retrieval
chantier is closing the gap to the raw-text ceiling or just moving tokens
around.

No LLM calls -- build_context() only (DB), reuses an already-ingested eval
project instead of paying for extraction again.

Usage:
    uv run python scripts/check_retrieval_recall.py [project_id]

Requires: Postgres up, and `project_id` (default:
prj_eval_lme_lme_ku_n15_after_fixes) already ingested by a prior
`uv run python -m eval.run --config eval/configs/longmemeval_s_bigcontext.json
--subset 15 --types knowledge-update --keep-data` (or --reuse-project) run.
"""

import asyncio
import re
import sys

from app.context import build_context
from app.db import async_session
from eval import datasets

DEFAULT_PROJECT_ID = "prj_eval_lme_lme_ku_n15_after_fixes"
DATASET_FILE = "eval/data/longmemeval_s_cleaned.json"
BUDGET_TOKENS = 4000

WORD_RE = re.compile(r"[a-zA-Z0-9À-ÿ]{4,}")


def words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text)}


async def main(project_id: str) -> None:
    questions = datasets.LOADERS["longmemeval"](DATASET_FILE)
    selected = datasets.select(questions, subset=15, types=["knowledge-update"])

    recalls: list[float] = []
    for question in selected:
        subject = (question.history_id or question.qid)[:128]
        evidence = question.evidence_sessions or question.sessions
        evidence_words = words(datasets.render_transcript(evidence))

        async with async_session() as session:
            packet, token_count, _trace_id = await build_context(
                session,
                project_id=project_id,
                subject_id=subject,
                query=question.question,
                budget_tokens=BUDGET_TOKENS,
            )

        served_words = words(
            datasets.render_facts(packet["facts"])
            + "\n"
            + datasets.render_episodes(packet.get("episodes", []))
        )

        if not evidence_words:
            continue
        recall = len(evidence_words & served_words) / len(evidence_words)
        recalls.append(recall)
        print(
            f"{question.qid}: recall={recall:.1%} "
            f"(evidence {len(evidence_words)} mots, servi {len(served_words)}, "
            f"{len(packet['facts'])} facts, {len(packet.get('episodes', []))} episodes, {token_count} tok)"
        )

    mean_recall = sum(recalls) / len(recalls)
    print(f"\n--- recall moyen ({len(recalls)} questions): {mean_recall:.1%} ---")
    print(f"min={min(recalls):.1%} max={max(recalls):.1%}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROJECT_ID))
