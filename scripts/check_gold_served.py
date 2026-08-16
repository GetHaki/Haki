"""Free, zero-LLM "gold-servi" gate (Livre de construction, Sprint 1 gate:
gold-served >= 45%). Same definition as scripts/classify_coverage_gaps.py's
token-overlap heuristic (>=60% of the gold answer's own tokens present in
what was actually served), but computed against a FRESH packet regenerated
right now (current code, current mechanisms) instead of a frozen old run's
JSON -- reuses scripts/check_retrieval_recall.py's pattern: build_context()
called directly against an already-ingested project, no LLM call, no new
extraction cost.

Caveat found while building this (16 aout): the book's ~900-failure
diagnostic set (14 aout) was ingested on GitHub Actions' ephemeral Postgres
and no longer exists locally or anywhere else -- re-measuring it for free
would need re-ingestion, which is not free. This script runs the same
metric on the largest already-ingested local sample with real gold
answers instead: LongMemEval knowledge-update n=15
(prj_eval_lme_lme_ku_n15_after_fixes). Smaller n, same methodology,
directionally informative -- not a substitute for the real gate on the
full set.

Usage:
    uv run python scripts/check_gold_served.py [project_id]
"""

import asyncio
import json
import re
import sys
from collections import Counter

from app.context import build_context
from app.db import async_session
from eval import datasets

DEFAULT_PROJECT_ID = "prj_eval_lme_lme_ku_n15_after_fixes"
DATASET_FILE = "eval/data/longmemeval_s_cleaned.json"
BUDGET_TOKENS = 4000
GOLD_COVERAGE_THRESHOLD = 0.6  # same threshold as classify_coverage_gaps.py

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "was",
    "were", "are", "be", "for", "with", "her", "his", "she", "he", "it",
    "they", "their", "i", "you", "your", "my", "me", "we", "us", "that",
    "this", "as", "by", "from", "not", "no", "yes",
}


def tokens(text) -> set[str]:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return {t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS and len(t) > 1}


def packet_text(facts: list[dict], episodes: list[dict]) -> str:
    parts = []
    for fact in facts:
        parts.append(fact.get("predicate", ""))
        parts.append(json.dumps(fact.get("value", {}), ensure_ascii=False))
    for ep in episodes:
        parts.append(json.dumps(ep, ensure_ascii=False))
    return " ".join(parts)


async def main(project_id: str) -> None:
    questions = datasets.LOADERS["longmemeval"](DATASET_FILE)
    selected = datasets.select(questions, subset=15, types=["knowledge-update"])

    buckets = Counter()
    rows = []
    for question in selected:
        subject = (question.history_id or question.qid)[:128]
        gold_tokens = tokens(question.answer)
        if not gold_tokens:
            continue

        async with async_session() as session:
            packet, _token_count, _trace_id = await build_context(
                session,
                project_id=project_id,
                subject_id=subject,
                query=question.question,
                budget_tokens=BUDGET_TOKENS,
            )

        served = tokens(packet_text(packet["facts"], packet.get("episodes", [])))
        coverage_ratio = len(gold_tokens & served) / len(gold_tokens)
        gold_served = coverage_ratio >= GOLD_COVERAGE_THRESHOLD
        buckets["gold_served" if gold_served else "gold_not_served"] += 1
        rows.append((question.qid, gold_served, coverage_ratio))
        print(f"{question.qid}: gold_served={gold_served} (coverage={coverage_ratio:.1%})")

    total = sum(buckets.values())
    rate = buckets["gold_served"] / total if total else 0.0
    print(f"\n--- gold-servi ({total} questions, n=15 local sample): {rate:.1%} ---")
    print("Livre de construction, gate Sprint 1: gold-servi >= 45% (sur les ~900 echecs "
          "du diagnostic complet, indisponible localement -- voir docstring).")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROJECT_ID))
