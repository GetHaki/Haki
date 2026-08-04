"""Fact status lifecycle: legal transitions pass, illegal ones are refused."""

import pytest

from app.db import async_session
from app.ledger import (
    IllegalTransitionError,
    create_fact,
    get_fact,
    transition_fact_status,
)
from app.models import FactStatus


async def make_fact():
    async with async_session() as session:
        fact = await create_fact(
            session,
            org_id="org_acme",
            project_id="prj_support",
            subject_id="usr_42",
            predicate="invoice_language",
            value={"language": "fr"},
        )
        await session.commit()
        return fact.id


async def test_candidate_to_active_is_allowed():
    fact_id = await make_fact()
    async with async_session() as session:
        fact = await transition_fact_status(session, fact_id, FactStatus.active)
        await session.commit()
    assert fact.status is FactStatus.active
    assert fact.version == 2


async def test_active_to_superseded_is_allowed():
    fact_id = await make_fact()
    async with async_session() as session:
        await transition_fact_status(session, fact_id, FactStatus.active)
        fact = await transition_fact_status(session, fact_id, FactStatus.superseded)
        await session.commit()
    assert fact.status is FactStatus.superseded


async def test_superseded_to_active_is_refused():
    fact_id = await make_fact()
    async with async_session() as session:
        await transition_fact_status(session, fact_id, FactStatus.active)
        await transition_fact_status(session, fact_id, FactStatus.superseded)
        with pytest.raises(IllegalTransitionError):
            await transition_fact_status(session, fact_id, FactStatus.active)
        await session.commit()

    async with async_session() as session:
        fact = await get_fact(session, fact_id)
    assert fact.status is FactStatus.superseded


async def test_deleted_is_terminal():
    fact_id = await make_fact()
    async with async_session() as session:
        await transition_fact_status(session, fact_id, FactStatus.deleted)
        await session.commit()

    for target in (FactStatus.active, FactStatus.candidate, FactStatus.disabled):
        async with async_session() as session:
            with pytest.raises(IllegalTransitionError):
                await transition_fact_status(session, fact_id, target)

    async with async_session() as session:
        fact = await get_fact(session, fact_id)
    assert fact.status is FactStatus.deleted
