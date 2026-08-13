"""13 aout, "phase 2 benchmarks": real-data verification of the entity-
aware retrieval boost (app.context.ENTITY_MATCH_BOOST/ENTITY_MISMATCH_
PENALTY) -- does it actually fix the failure mode found in run
gh-31698210575 (LoCoMo single-hop questions naming one of two speakers
in a shared-subject conversation getting the OTHER speaker's facts
served instead)?

Ingests conv-26 (Caroline/Melanie) from the real LoCoMo dataset for real
(extraction cost, a few cents), then calls build_context() directly for
the exact questions that failed in the run, TWICE per question: once
with the entity boost disabled (module constants monkey-patched to 1.0,
a true no-op -- same code path, not a different version), once with it
enabled (real values) -- a genuine A/B on the SAME freshly-ingested data,
not a frozen-packet replay like the chain-of-note probe (this changes
RETRIEVAL itself, which the frozen JSON can't exercise).

Usage: uv run python scripts/verify_entity_aware_retrieval.py
Requires: local Postgres (docker compose up), a local API server running
with HAKI_LLM_PROVIDER=openai (uvicorn app.main:app --port 8100), and
HAKI_LLM_API_KEY set. Real cost: ~19 sessions extraction for one
conversation, a few cents.
"""

import asyncio
import json
import os

import app.context as context_module
from app.db import async_session
from eval import datasets
from haki import HakiClient

DATASET_FILE = "eval/data/locomo10.json"
API_URL = "http://127.0.0.1:8100"
PROJECT_ID = "prj_verify_entity_boost"
ORG_ID = "org_verify"
BUDGET_TOKENS = 4000

# The exact failing questions from run gh-31698210575 (conv-26), with the
# gold answer for a quick eyeball check -- this script inspects WHICH
# facts get packed, not automated judging (that's what the LoCoMo run
# itself already measured; this is a targeted mechanism check).
TARGET_QUESTIONS = [
    ("conv-26_q15", "What activities does Melanie partake in?", "pottery, camping, painting, swimming"),
    ("conv-26_q32", "What LGBTQ+ events has Caroline participated in?", "Pride parade, school speech, support group"),
    ("conv-26_q51", "What has Melanie painted?", "Horse, sunset, sunrise"),
]


def ingest_events(question) -> list[dict]:
    events = []
    for i, session in enumerate(question.sessions):
        events.append(
            {
                "org_id": ORG_ID,
                "project_id": PROJECT_ID,
                "subject_type": "user",
                "subject_id": question.history_id,
                "kind": "chat_session",
                "occurred_at": session.date.astimezone().isoformat(),
                "payload": {
                    "session_id": session.session_id,
                    "date": session.date.astimezone().isoformat(),
                    "messages": [
                        {"role": m.speaker, "content": m.content} for m in session.messages
                    ],
                },
                "idempotency_key": f"verify-entity-boost:{question.history_id}:{i}",
            }
        )
    return events


def person_breakdown(packet_facts: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in packet_facts:
        person = (f.get("value") or {}).get("person", "(tracked subject)")
        counts[person] = counts.get(person, 0) + 1
    return counts


async def main() -> None:
    questions = datasets.LOADERS["locomo"](DATASET_FILE)
    conv26 = next(q for q in questions if q.history_id == "conv-26")

    api_key = os.environ["HAKI_VERIFY_KEY"]
    client = HakiClient(API_URL, api_key=api_key, timeout=300.0)
    print("=== ingesting conv-26 (real extraction cost) ===")
    body = client.capture(ingest_events(conv26), idempotency_key="verify-entity-boost-batch")
    print("capture:", body["status"], body.get("consolidation_job_id"))
    result = client.consolidate()
    print("consolidate:", result)

    print("\n=== A/B per question: boost disabled vs enabled, same ingested data ===")
    async with async_session() as db_session:
        for qid, question_text, gold in TARGET_QUESTIONS:
            print(f"\n--- {qid}: {question_text!r} (gold: {gold}) ---")
            for label, boost, penalty in [("DISABLED (no-op)", 1.0, 1.0), ("ENABLED (real)", 1.3, 0.3)]:
                context_module.ENTITY_MATCH_BOOST = boost
                context_module.ENTITY_MISMATCH_PENALTY = penalty
                packet, _tokens, _trace_id = await context_module.build_context(
                    db_session,
                    project_id=PROJECT_ID,
                    subject_id="conv-26",
                    query=question_text,
                    budget_tokens=BUDGET_TOKENS,
                )
                counts = person_breakdown(packet["facts"])
                print(f"  [{label}] facts by 'person' tag: {counts}")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
