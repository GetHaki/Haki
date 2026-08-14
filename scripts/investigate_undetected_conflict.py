"""13 aout, fix 2: root-cause AND verify the fix for the undetected-conflict
case found in the LongMemEval run (qid ba61f0b9, run 31705865474) -- two
ACTIVE facts under the same predicate ("women_roles_on_rachels_team", 5
people vs 6 people, different dates), never flagged as a conflict, never
marked `contested`. The GH Actions Postgres that produced them is gone --
this re-ingests the EXACT real haystack (45 sessions) locally and inspects
the resulting Fact/ConflictSet rows directly to first find, then (after the
`_find_qualifier_ambiguous_active_fact` fix landed in
app/consolidator/__init__.py) confirm, why/whether the conflict is caught.

SUBJECT_ID is a fresh id (not the original `ba61f0b9`) so this re-ingestion
runs the WHOLE pipeline through the current code from a clean slate --
reprocessing old jobs would not retroactively open a conflict for facts
already created under the pre-fix code.

Usage: uv run python scripts/investigate_undetected_conflict.py
Requires: local Postgres up, local API server running with
HAKI_LLM_PROVIDER=openai on :8100, HAKI_VERIFY_KEY env var set to a valid
key for prj_verify_conflict_gap. Real cost: 45 sessions extraction, a
few tens of cents.
"""

import asyncio
import os

from app.db import async_session
from app.models import ConflictSet, Fact
from eval import datasets
from haki import HakiClient
from sqlalchemy import select

DATASET_FILE = "eval/data/longmemeval_s_cleaned.json"
API_URL = "http://127.0.0.1:8100"
PROJECT_ID = "prj_verify_conflict_gap"
ORG_ID = "org_verify"
SUBJECT_ID = "ba61f0b9_fix2verify"


def ingest_events(question) -> list[dict]:
    events = []
    for i, session in enumerate(question.sessions):
        events.append(
            {
                "org_id": ORG_ID,
                "project_id": PROJECT_ID,
                "subject_type": "user",
                "subject_id": SUBJECT_ID,
                "kind": "chat_session",
                "occurred_at": session.date.astimezone().isoformat(),
                "payload": {
                    "session_id": session.session_id,
                    "date": session.date.astimezone().isoformat(),
                    "messages": [
                        {"role": m.speaker, "content": m.content} for m in session.messages
                    ],
                },
                "idempotency_key": f"investigate-conflict-gap:{SUBJECT_ID}:{i}",
            }
        )
    return events


async def main() -> None:
    questions = datasets.LOADERS["longmemeval"](DATASET_FILE)
    q = next(qq for qq in questions if qq.qid == "ba61f0b9")
    print(f"ingesting {len(q.sessions)} sessions for {q.qid}...")

    api_key = os.environ["HAKI_VERIFY_KEY"]
    client = HakiClient(API_URL, api_key=api_key, timeout=600.0)
    body = client.capture(ingest_events(q), idempotency_key="investigate-conflict-gap-batch")
    print("capture:", body["status"], body.get("consolidation_job_id"))
    result = client.consolidate()
    print("consolidate:", result)
    client.close()

    print("\n=== facts mentioning 'rachel' or 'team' (predicate, status, qualifiers) ===")
    async with async_session() as session:
        rows = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.project_id == PROJECT_ID, Fact.subject_id == SUBJECT_ID
                    )
                )
            )
            .scalars()
            .all()
        )
        for f in rows:
            blob = str(f.value).lower() + " " + f.predicate.lower()
            if "rachel" in blob or "team" in blob or "women" in blob:
                print("---")
                print("id:", f.id)
                print("predicate:", f.predicate)
                print("qualifiers:", f.qualifiers)
                print("status:", f.status)
                print("value:", f.value)
                print("valid_from:", f.valid_from, "| recorded_from:", f.recorded_from)
                print("supersedes_id:", f.supersedes_id)

        print("\n=== ConflictSet rows for this subject ===")
        conflicts = (
            (
                await session.execute(
                    select(ConflictSet).where(
                        ConflictSet.project_id == PROJECT_ID, ConflictSet.subject_id == SUBJECT_ID
                    )
                )
            )
            .scalars()
            .all()
        )
        if not conflicts:
            print("(none)")
        for c in conflicts:
            print("---")
            print("id:", c.id, "| status:", c.status)
            print("fact_ids:", c.fact_ids)
            print("reason:", c.reason)


if __name__ == "__main__":
    asyncio.run(main())
