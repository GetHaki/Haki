"""Non-duplication guarantees under real concurrency (chantier toctou-dedup).

Two consolidations of the SAME subject running as genuinely concurrent
transactions (`asyncio.gather`, two independent sessions against the real
test Postgres) must never both observe "no duplicate" and both insert --
the exact race class of mem0 #6515. Plus the DB backstop (partial unique
index, migration 0015) holding even outside the consolidator.
"""

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.consolidator import run_pending_consolidations
from app.db import async_session
from app.models import ConflictSet, Fact, FactStatus, Job, JobStatus
from app.providers.fake import FakeProvider, mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker


async def _run_worker_in_own_session() -> int:
    async with async_session() as session:
        done = await run_pending_consolidations(
            session, extractor=FakeProvider(), embedder=FakeProvider()
        )
        await session.commit()
        return done


async def _all_jobs_done() -> bool:
    async with async_session() as session:
        statuses = (await session.execute(select(Job.status))).scalars().all()
    return bool(statuses) and all(status is JobStatus.done for status in statuses)


async def test_concurrent_consolidations_do_not_duplicate_the_same_fact(client):
    """The course TOCTOU this chantier fixes: two jobs re-asserting the same
    value for the same subject+predicate, consolidated by two genuinely
    concurrent transactions, must leave exactly one active fact -- serialized
    by the advisory lock, never a duplicate."""
    await capture(
        client, [make_memory_event([mock_fact("favorite_city", {"city": "paris"})])]
    )
    await capture(
        client, [make_memory_event([mock_fact("favorite_city", {"city": "paris"})])]
    )

    # No job claiming exists (deliberately out of scope for this chantier --
    # both concurrent sessions see the same pending jobs and each attempts
    # both; only the advisory lock's serialization of the WRITE PHASE is
    # under test here, not job scheduling).
    await asyncio.gather(_run_worker_in_own_session(), _run_worker_in_own_session())
    assert await _all_jobs_done()

    facts = await facts_for("usr_42", "favorite_city")
    assert len(facts) == 1
    fact = facts[0]
    assert fact.status is FactStatus.active
    assert fact.reinforcement_count == 1
    assert len(fact.source_event_ids) == 2


async def test_concurrent_conflicting_creates_never_leave_two_active_facts(client):
    """Same race, but with a genuinely different value: the lock still
    serializes the two writers, so the outcome is exactly the sequential one
    (one active fact, one open conflict) -- never two active facts, whichever
    writer happens to win the lock."""
    await capture(
        client, [make_memory_event([mock_fact("contact_channel", {"channel": "email"})])]
    )
    await capture(
        client, [make_memory_event([mock_fact("contact_channel", {"channel": "chat"})])]
    )

    await asyncio.gather(_run_worker_in_own_session(), _run_worker_in_own_session())
    assert await _all_jobs_done()

    facts = await facts_for("usr_42", "contact_channel")
    assert len(facts) == 2
    active = [f for f in facts if f.status is FactStatus.active]
    assert len(active) == 1

    async with async_session() as session:
        conflicts = list(
            (await session.execute(select(ConflictSet))).scalars().all()
        )
    assert len(conflicts) == 1
    assert conflicts[0].status == "open"
    assert set(conflicts[0].fact_ids) == {f.id for f in facts}


async def test_db_rejects_a_second_active_fact_for_same_subject_predicate():
    """DB backstop (migration 0015): the partial unique index holds even
    completely outside the consolidator -- any future write path that
    forgets the advisory lock fails loudly at the database, never silently."""
    async with async_session() as session:
        first = Fact(
            org_id="org_acme",
            project_id="prj_support",
            subject_id="usr_backstop",
            predicate="favorite_city",
            value={"city": "paris"},
            status=FactStatus.active,
        )
        session.add(first)
        await session.flush()

        second = Fact(
            org_id="org_acme",
            project_id="prj_support",
            subject_id="usr_backstop",
            predicate="favorite_city",
            value={"city": "lyon"},
            status=FactStatus.active,
        )
        session.add(second)
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()


async def test_two_active_facts_with_same_predicate_for_different_subjects_are_allowed():
    """Anti-over-blocking guard: the unique index is scoped per subject, not
    global to the project -- two different subjects each get their own
    active fact for the same predicate without any conflict."""
    async with async_session() as session:
        session.add(
            Fact(
                org_id="org_acme",
                project_id="prj_support",
                subject_id="usr_scope_a",
                predicate="favorite_city",
                value={"city": "paris"},
                status=FactStatus.active,
            )
        )
        session.add(
            Fact(
                org_id="org_acme",
                project_id="prj_support",
                subject_id="usr_scope_b",
                predicate="favorite_city",
                value={"city": "lyon"},
                status=FactStatus.active,
            )
        )
        await session.flush()
        await session.commit()
