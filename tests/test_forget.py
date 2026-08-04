"""POST /v1/forget (real database, FakeProvider):

fact disable/delete via the Ledger lifecycle, subject disable (all
active/candidate facts), subject delete (real erasure of facts, conflicts,
events, traces), validation errors, receipt journaling, and context no
longer serving a deleted subject.
"""

import uuid

from sqlalchemy import select

from app.db import async_session
from app.models import (
    ConflictSet,
    ContextTrace,
    Event,
    Fact,
    FactStatus,
    ForgetReceipt,
)
from app.providers.fake import mock_fact
from test_consolidator import capture, make_memory_event, run_worker

PROJECT = "prj_support"
SUBJECT = "usr_forget"


async def seed_fact(
    client, predicate: str, value: dict, subject_id: str = SUBJECT
) -> dict:
    """Capture + consolidate one active fact; return the /v1/context view."""
    await capture(
        client,
        [make_memory_event([mock_fact(predicate, value, subject_id=subject_id)], subject_id)],
    )
    assert await run_worker() == 1
    return await context_facts(client, subject_id, query=predicate)


async def context_facts(client, subject_id: str, query: str) -> dict:
    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject_id,
            "query": query,
            "budget_tokens": 900,
        },
    )
    assert response.status_code == 200
    return response.json()


async def forget(client, **payload) -> tuple[int, dict]:
    response = await client.post("/v1/forget", json=payload)
    return response.status_code, response.json()


async def active_fact_id(predicate: str, subject_id: str = SUBJECT) -> uuid.UUID:
    async with async_session() as session:
        fact = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.subject_id == subject_id, Fact.predicate == predicate
                    )
                )
            )
            .scalars()
            .one()
        )
        return fact.id


async def test_fact_disable_transitions_to_disabled_and_hides_from_context(client):
    packet = await seed_fact(client, "code_language", {"language": "typescript"})
    fact_id = packet["packet"]["facts"][0]["id"]

    status, body = await forget(
        client, project_id=PROJECT, fact_id=fact_id, mode="disable"
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["mode"] == "disable"
    assert body["scope"] == "fact"
    assert body["facts_disabled"] == 1
    assert uuid.UUID(body["forget_id"])

    async with async_session() as session:
        fact = await session.get(Fact, uuid.UUID(fact_id))
    assert fact.status is FactStatus.disabled

    packet = await context_facts(client, SUBJECT, query="code_language")
    assert packet["packet"]["facts"] == []


async def test_fact_delete_transitions_to_deleted_with_recorded_to(client):
    await seed_fact(client, "code_language", {"language": "typescript"})
    fact_id = await active_fact_id("code_language")

    status, body = await forget(
        client, project_id=PROJECT, fact_id=str(fact_id), mode="delete"
    )
    assert status == 200
    assert body["facts_deleted"] == 1

    async with async_session() as session:
        fact = await session.get(Fact, fact_id)
    assert fact.status is FactStatus.deleted
    assert fact.recorded_to is not None  # bitemporal erasure


async def test_subject_disable_disables_all_active_and_candidate_facts(client):
    await seed_fact(client, "code_language", {"language": "typescript"})
    # A second fact left in `candidate` state (conflicting value).
    await capture(
        client,
        [make_memory_event(
            [mock_fact("code_language", {"language": "python"}, subject_id=SUBJECT)],
            SUBJECT,
        )],
    )
    assert await run_worker() == 1

    status, body = await forget(client, project_id=PROJECT, subject_id=SUBJECT, mode="disable")
    assert status == 200
    assert body["scope"] == "subject"
    assert body["facts_disabled"] == 2  # the active one + the conflicting candidate

    async with async_session() as session:
        statuses = {
            fact.status
            for fact in (
                (await session.execute(select(Fact).where(Fact.subject_id == SUBJECT)))
                .scalars()
                .all()
            )
        }
    assert statuses == {FactStatus.disabled}


async def test_subject_delete_erases_everything_and_context_serves_nothing(client):
    packet = await seed_fact(client, "code_language", {"language": "typescript"})
    trace_id = packet["trace_id"]

    status, body = await forget(client, project_id=PROJECT, subject_id=SUBJECT, mode="delete")
    assert status == 200
    assert body["mode"] == "delete"
    assert body["events_deleted"] == 1
    assert body["facts_deleted"] == 1
    assert body["traces_deleted"] == 1  # the context trace created above
    assert "conflict_sets_deleted" in body

    async with async_session() as session:
        for model in (Event, Fact, ConflictSet, ContextTrace):
            rows = (
                (
                    await session.execute(
                        select(model).where(
                            model.project_id == PROJECT, model.subject_id == SUBJECT
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert rows == [], model.__tablename__

    # Context serves nothing of the subject anymore.
    packet = await context_facts(client, SUBJECT, query="code_language")
    assert packet["packet"]["facts"] == []
    # And the old trace is gone with the subject (same 404 as unknown id).
    response = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": PROJECT, "subject_id": SUBJECT},
    )
    assert response.status_code == 404


async def test_validation_errors(client):
    # Neither target.
    status, body = await forget(client, project_id=PROJECT, mode="disable")
    assert status == 422
    assert body["error"]["type"] in ("invalid_payload", "invalid_forget_scope")

    # Both targets.
    status, body = await forget(
        client,
        project_id=PROJECT,
        fact_id=str(uuid.uuid4()),
        subject_id=SUBJECT,
        mode="disable",
    )
    assert status == 422
    assert body["error"]["type"] in ("invalid_payload", "invalid_forget_scope")

    # Unknown mode.
    status, body = await forget(
        client, project_id=PROJECT, subject_id=SUBJECT, mode="purge"
    )
    assert status == 422

    # Unknown fact.
    status, body = await forget(
        client, project_id=PROJECT, fact_id=str(uuid.uuid4()), mode="disable"
    )
    assert status == 404
    assert body["error"]["type"] == "fact_not_found"


async def test_fact_from_another_project_is_not_found(client):
    await seed_fact(client, "code_language", {"language": "typescript"})
    fact_id = await active_fact_id("code_language")

    status, body = await forget(
        client, project_id="prj_other", fact_id=str(fact_id), mode="disable"
    )
    assert status == 404
    assert body["error"]["type"] == "fact_not_found"

    async with async_session() as session:
        fact = await session.get(Fact, fact_id)
    assert fact.status is FactStatus.active  # untouched


async def test_receipt_is_journaled(client):
    await seed_fact(client, "code_language", {"language": "typescript"})
    fact_id = await active_fact_id("code_language")

    status, body = await forget(
        client, project_id=PROJECT, fact_id=str(fact_id), mode="delete"
    )
    assert status == 200

    async with async_session() as session:
        receipt = await session.get(ForgetReceipt, uuid.UUID(body["forget_id"]))
    assert receipt is not None
    assert receipt.project_id == PROJECT
    assert receipt.scope == "fact"
    assert receipt.fact_id == fact_id
    assert receipt.mode == "delete"
    assert receipt.counters == {"facts_deleted": 1}
    assert receipt.created_at is not None
