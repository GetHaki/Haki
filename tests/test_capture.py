"""Capture: idempotence, ack, and consolidation job creation."""

import uuid

from sqlalchemy import func, select

from app.db import async_session
from app.models import Event, Job


def make_event(subject_id: str = "usr_42", content: str = "Use French for invoices."):
    return {
        "org_id": "org_acme",
        "project_id": "prj_support",
        "subject_type": "user",
        "subject_id": subject_id,
        "actor_type": "agent",
        "actor_id": "agt_support",
        "agent_id": "agt_support",
        "thread_id": "thread_456",
        "run_id": "run_123",
        "kind": "conversation.message",
        "occurred_at": "2026-07-26T12:03:41Z",
        "payload": {"role": "user", "content": content},
        "source": {"system": "intercom", "record_id": "msg_9"},
        "classification": ["customer-data"],
        "retention_policy": "customer-90d",
    }


def make_batch(idempotency_key: str):
    return {
        "idempotency_key": idempotency_key,
        "events": [make_event(), make_event(content="Second message.")],
    }


async def count_events() -> int:
    async with async_session() as session:
        return await session.scalar(select(func.count()).select_from(Event))


async def test_capture_same_batch_twice_is_idempotent(client):
    key = f"batch-{uuid.uuid4()}"
    first = await client.post("/v1/capture", json=make_batch(key))
    assert first.status_code == 202

    rows_before = await count_events()
    second = await client.post("/v1/capture", json=make_batch(key))
    rows_after = await count_events()

    assert second.status_code == 202
    assert rows_after == rows_before
    assert all(e["deduplicated"] for e in second.json()["events"])

    first_ids = sorted(e["id"] for e in first.json()["events"])
    second_ids = sorted(e["id"] for e in second.json()["events"])
    assert first_ids == second_ids


async def test_capture_ack_contains_ids_and_consolidation_job(client):
    response = await client.post("/v1/capture", json=make_batch(f"batch-{uuid.uuid4()}"))
    assert response.status_code == 202

    body = response.json()
    assert body["status"] == "accepted"
    assert len(body["events"]) == 2
    assert all(not e["deduplicated"] for e in body["events"])
    assert body["policy"] == "default"

    job_id = uuid.UUID(body["consolidation_job_id"])
    async with async_session() as session:
        job = await session.get(Job, job_id)
    assert job is not None
    assert job.kind == "consolidate"


async def test_capture_rejects_oversized_payload(client):
    """No size bound existed at all before this (found by security review,
    16 aout) -- an authenticated key holder could post an arbitrarily large
    or deeply nested JSON blob. See app.schemas.capture.MAX_JSON_BYTES."""
    from app.schemas.capture import MAX_JSON_BYTES

    event = make_event()
    event["payload"] = {"content": "x" * (MAX_JSON_BYTES + 1)}

    response = await client.post(
        "/v1/capture",
        json={"idempotency_key": f"batch-{uuid.uuid4()}", "events": [event]},
    )
    assert response.status_code == 422
    assert await count_events() == 0


async def test_capture_accepts_payload_under_the_size_bound(client):
    """Regression guard: the size bound must not reject ordinary payloads
    -- a real conversational session comfortably fits under it."""
    from app.schemas.capture import MAX_JSON_BYTES

    event = make_event()
    event["payload"] = {"content": "x" * (MAX_JSON_BYTES // 2)}

    response = await client.post(
        "/v1/capture",
        json={"idempotency_key": f"batch-{uuid.uuid4()}", "events": [event]},
    )
    assert response.status_code == 202
