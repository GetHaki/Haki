"""Live probe (real extractor, real cost, single run) for the reasoning-
field chantier (12 aout): does putting `reasoning` before the decision
fields actually help the extractor reuse an existing predicate under
realistic clutter and topic-shifting, the exact failure mode documented
earlier this session ("invents a different predicate instead of reusing
personal_best_5k" when the subject has ~85 active facts and the message
changes topic several times)?

In-process (ASGI transport, same recipe as tests/conftest.py) against the
`haki_test` database -- no server to start, no HTTP. Seeds 12 unrelated
facts via FakeProvider (free), then runs ONE real OpenAIProvider
extraction on a topic-shifting message that should supersede
`personal_best_5k` under mild distraction.

Usage: uv run python scripts/probe_reasoning_field.py
Requires Postgres up (haki_test, same as the test suite) and
HAKI_LLM_API_KEY in the environment/.env.
"""

import asyncio
import os

os.environ.setdefault(
    "HAKI_TEST_BASE_DATABASE_URL", "postgresql+asyncpg://haki_app:haki@localhost:5433/haki"
)
os.environ.setdefault(
    "HAKI_TEST_BASE_MIGRATION_URL", "postgresql+asyncpg://haki:haki@localhost:5433/haki"
)
_base = os.environ["HAKI_TEST_BASE_DATABASE_URL"]
os.environ["HAKI_DATABASE_URL"] = _base.rsplit("/", 1)[0] + "/haki_test"
os.environ["HAKI_EMBED_PROVIDER"] = "fake"
os.environ["HAKI_LLM_PROVIDER"] = "fake"  # overridden per-call below, not globally
os.environ["HAKI_AUTH_REQUIRED"] = "false"
os.environ.setdefault("HAKI_REDIS_URL", "redis://localhost:6379/15")

import json
import uuid

from dotenv import load_dotenv

load_dotenv()

import app.ledger  # noqa: E402,F401 - import before app.consolidator to avoid a pre-existing circular import
from app.consolidator import run_pending_consolidations  # noqa: E402
from app.db import async_session  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Fact, FactStatus  # noqa: E402
from app.providers.fake import FakeProvider, mock_fact  # noqa: E402
from app.providers.openai import OpenAIProvider  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

ORG = "org_probe"
PROJECT = "prj_probe_reasoning"
SUBJECT = "usr_probe"

SEED_FACTS = [
    mock_fact("personal_best_5k", {"time": "27:12"}, subject_id=SUBJECT),
    mock_fact("coffee_preference", {"drink": "oat milk latte"}, subject_id=SUBJECT),
    mock_fact("home_city", {"city": "Lyon"}, subject_id=SUBJECT),
    mock_fact("employer", {"company": "Nimbus Analytics"}, subject_id=SUBJECT),
    mock_fact("pet_name", {"name": "Miso", "species": "cat"}, subject_id=SUBJECT),
    mock_fact("favorite_book", {"title": "The Overstory"}, subject_id=SUBJECT),
    mock_fact("sleep_schedule", {"bedtime": "23:00"}, subject_id=SUBJECT),
    mock_fact("gym_membership", {"gym": "FitZone", "since": "2024-01"}, subject_id=SUBJECT),
    mock_fact("language_preference", {"language": "fr"}, subject_id=SUBJECT),
    mock_fact("weekend_hobby", {"hobby": "pottery"}, subject_id=SUBJECT),
    mock_fact("phone_model", {"model": "Pixel 9"}, subject_id=SUBJECT),
    mock_fact("commute_mode", {"mode": "bike"}, subject_id=SUBJECT),
]

PROBE_MESSAGE = (
    "Busy week! Finally switched my coffee order to oat milk lattes, been "
    "meaning to for months. Also finished the 5K on Saturday in 25:40, "
    "which felt great after all that training -- way better than my old "
    "time. Oh, and I started reading a book my sister recommended, still "
    "early days on it though."
)


async def main() -> int:
    async with async_session() as session:
        await session.execute(delete(Fact).where(Fact.project_id == PROJECT))
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        seed_event = {
            "org_id": ORG,
            "project_id": PROJECT,
            "subject_type": "user",
            "subject_id": SUBJECT,
            "kind": "conversation.message",
            "occurred_at": "2023-05-01T10:00:00Z",
            "payload": {"role": "user", "content": "...", "mock_facts": SEED_FACTS},
        }
        resp = await client.post(
            "/v1/capture",
            json={"idempotency_key": f"seed-{uuid.uuid4()}", "events": [seed_event]},
        )
        resp.raise_for_status()

    async with async_session() as session:
        await run_pending_consolidations(session, extractor=FakeProvider(), embedder=FakeProvider())
        await session.commit()

    async with async_session() as session:
        seeded = (
            await session.execute(
                select(Fact.predicate).where(Fact.project_id == PROJECT, Fact.status == FactStatus.active)
            )
        ).scalars().all()
    print(f"seeded {len(seeded)} active facts: {sorted(seeded)}")
    assert len(seeded) == len(SEED_FACTS), "seeding failed, aborting probe"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        probe_event = {
            "org_id": ORG,
            "project_id": PROJECT,
            "subject_type": "user",
            "subject_id": SUBJECT,
            "kind": "conversation.message",
            "occurred_at": "2023-05-15T10:00:00Z",
            "payload": {"role": "user", "content": PROBE_MESSAGE},
        }
        resp = await client.post(
            "/v1/capture",
            json={"idempotency_key": f"probe-{uuid.uuid4()}", "events": [probe_event]},
        )
        resp.raise_for_status()

    async with async_session() as session:
        await run_pending_consolidations(session, extractor=OpenAIProvider(), embedder=FakeProvider())
        await session.commit()

    async with async_session() as session:
        facts = (
            await session.execute(
                select(Fact).where(Fact.project_id == PROJECT, Fact.subject_id == SUBJECT)
            )
        ).scalars().all()

    print("\n--- facts after probe event ---")
    for f in sorted(facts, key=lambda f: (f.predicate, f.status.value)):
        print(f"  {f.predicate:30s} {f.status.value:12s} {json.dumps(f.value, ensure_ascii=False)}")

    active = [f for f in facts if f.status is FactStatus.active]
    superseded_5k = [f for f in facts if f.predicate == "personal_best_5k" and f.status is FactStatus.superseded]
    new_5k_active = [f for f in active if f.predicate == "personal_best_5k" and "25:40" in json.dumps(f.value)]
    stray_predicates = [
        f for f in active
        if "25:40" in json.dumps(f.value) and f.predicate != "personal_best_5k"
    ]

    ok = bool(new_5k_active) and bool(superseded_5k) and not stray_predicates
    print(f"\nVERDICT: {'OK -- reused personal_best_5k via supersede' if ok else 'FAIL -- see facts above'}")
    if stray_predicates:
        print(f"  invented predicate(s) instead of reusing personal_best_5k: {[f.predicate for f in stray_predicates]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
