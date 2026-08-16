"""RLS non-disclosure (sprint 6, migration 0006).

THE guarantee of this sprint: with haki.project_id set on the transaction,
a query that FORGETS the project filter in the code (plain `select(Fact)`,
no .where) still only returns the rows of that project — PostgreSQL
Row-Level Security enforces what the application code could miss.
"""

from datetime import datetime, timezone

from sqlalchemy import select, text

from app.db import async_session
from app.ledger import create_fact
from app.models import Event, Fact, ForgetReceipt


async def _seed_two_projects() -> None:
    """Rows for prj_a and prj_b, written WITHOUT any RLS context (the
    permissive mode: haki.project_id unset)."""
    async with async_session() as session:
        for project_id in ("prj_a", "prj_b"):
            session.add(
                Event(
                    org_id="org_x",
                    project_id=project_id,
                    subject_type="user",
                    subject_id="usr_1",
                    kind="conversation.message",
                    occurred_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
                    payload={"role": "user", "content": f"hello from {project_id}"},
                    hash=f"sha256:{project_id}",
                    idempotency_key=f"rls-test:{project_id}",
                )
            )
            await create_fact(
                session,
                org_id="org_x",
                project_id=project_id,
                subject_id="usr_1",
                predicate="plan",
                value={"tier": project_id},
            )
        await session.commit()


async def test_rls_never_discloses_other_project():
    await _seed_two_projects()

    # Permissive mode (no setting): both projects are visible. This is the
    # documented dev-open behavior, asserted so it cannot change silently.
    async with async_session() as session:
        all_facts = (await session.execute(select(Fact))).scalars().all()
    assert {f.project_id for f in all_facts} == {"prj_a", "prj_b"}

    # RLS context of prj_a, then queries WITHOUT any project filter in the
    # code — simulating a forgotten .where(Fact.project_id == ...).
    async with async_session() as session:
        await session.execute(
            text("SELECT set_config('haki.project_id', 'prj_a', true)")
        )
        facts = (await session.execute(select(Fact))).scalars().all()
        events = (await session.execute(select(Event))).scalars().all()

    assert facts, "expected at least the prj_a rows"
    assert {f.project_id for f in facts} == {"prj_a"}
    assert {e.project_id for e in events} == {"prj_a"}


async def test_rls_empty_string_setting_is_permissive():
    """Postgres leaves a custom GUC at '' (not NULL) after a SET LOCAL
    reverts at transaction end — and pooled connections reuse it. '' must
    mean "no context" exactly like NULL, or every pooled connection that
    ever served an authenticated request would hide ALL rows (bug found
    live in the sprint-6 security demo)."""
    await _seed_two_projects()
    async with async_session() as session:
        await session.execute(
            text("SELECT set_config('haki.project_id', '', false)")
        )
        facts = (await session.execute(select(Fact))).scalars().all()
    assert {f.project_id for f in facts} == {"prj_a", "prj_b"}


async def test_rls_never_discloses_other_project_forget_receipts():
    """forget_receipts (migration 0005) predates RLS (migration 0006) and
    was never retrofitted -- gap found and closed by security review
    (16 aout, migration 0022). Same guarantee as test_rls_never_discloses_
    other_project above, for this one table."""
    async with async_session() as session:
        for project_id in ("prj_a", "prj_b"):
            session.add(
                ForgetReceipt(
                    project_id=project_id,
                    scope="subject",
                    subject_id="usr_1",
                    mode="delete",
                    counters={"events_deleted": 1},
                )
            )
        await session.commit()

    async with async_session() as session:
        await session.execute(
            text("SELECT set_config('haki.project_id', 'prj_a', true)")
        )
        receipts = (await session.execute(select(ForgetReceipt))).scalars().all()

    assert receipts, "expected at least the prj_a receipt"
    assert {r.project_id for r in receipts} == {"prj_a"}


async def test_rls_blocks_cross_project_insert():
    import pytest

    from sqlalchemy.exc import ProgrammingError

    await _seed_two_projects()
    async with async_session() as session:
        await session.execute(
            text("SELECT set_config('haki.project_id', 'prj_a', true)")
        )
        with pytest.raises(ProgrammingError, match="row-level security"):
            await create_fact(
                session,
                org_id="org_x",
                project_id="prj_b",  # outside the key's project
                subject_id="usr_1",
                predicate="plan",
                value={"tier": "intrusion"},
            )
