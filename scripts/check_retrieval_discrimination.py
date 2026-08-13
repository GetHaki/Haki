"""Bug 2 diagnostic (11 aout): does /v1/context actually discriminate by
relevance, or does a subject sometimes get the SAME packet regardless of
the query?

Original symptom (found rereading old LoCoMo eval files, 2 aout): five
totally different questions on one subject returned the same 42-fact
packet, identical id hash. Hypothesis at the time, not yet confirmed: the
subject's total active facts already fit under the token budget (42 x
~16 tok =~ 662 < 900) -- nothing is ever excluded by score in that
regime, so every query gets literally everything, independent of
relevance. Open question: does this persist on a subject whose fact
count genuinely EXCEEDS the budget, where the score should start to
matter?

This script runs the SAME few unrelated queries against a real subject at
a real budget and prints a short id-hash + count per query, for both
regimes if you pass two (project, subject) pairs. Free -- build_context()
direct against an already-ingested project (local embedder), no LLM call.

Usage:
    uv run python scripts/check_retrieval_discrimination.py \
        <project_id> <subject_id> [budget_tokens]
"""

import asyncio
import hashlib
import sys

from app.context import build_context
from app.db import async_session

QUERIES = [
    "What is the person's favorite food?",
    "Where did the person go on vacation last year?",
    "What did the person study in college?",
    "How many pets does the person have?",
    "What is the person's job title?",
]


async def main(project_id: str, subject_id: str, budget_tokens: int) -> None:
    hashes = set()
    for query in QUERIES:
        async with async_session() as session:
            packet, tok, _trace_id = await build_context(
                session,
                project_id=project_id,
                subject_id=subject_id,
                query=query,
                budget_tokens=budget_tokens,
            )
        ids = sorted(f["id"] for f in packet["facts"])
        h = hashlib.sha256("".join(ids).encode()).hexdigest()[:12]
        hashes.add(h)
        print(f"{h}  n_facts={len(ids):3d}  tok={tok:4d}  q={query!r}")

    if len(hashes) == 1:
        print(
            "\nIDENTICAL packet for every query -- expected when the subject's "
            "eligible facts already fit under the budget (nothing is ever "
            "excluded by score, so ranking never gets a chance to matter). "
            "Not a discrimination bug by itself; only worth investigating "
            "further if this subject's active fact count clearly EXCEEDS "
            "what budget_tokens can hold."
        )
    else:
        print(f"\n{len(hashes)}/{len(QUERIES)} distinct packets -- discrimination is active.")


if __name__ == "__main__":
    project_id = sys.argv[1]
    subject_id = sys.argv[2]
    budget_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    asyncio.run(main(project_id, subject_id, budget_tokens))
