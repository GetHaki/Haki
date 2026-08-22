"""Haki Gateway behaviors (sprint 7): memory injection, pass-through,
degradation, upstream error propagation, idempotent capture, auth.

The upstream LLM provider is mocked (hermetic tests): `forward_upstream`
is replaced by a fake returning a valid chat.completion JSON. The database
is real, as everywhere else.
"""

import uuid
from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import select

from app import ledger
from app.db import async_session
from app.models import Event, Job, JobStatus
from app.providers.fake import mock_fact
from app.schemas import EventIn
from tests.test_consolidator import run_worker

CHAT_URL = "/gateway/v1/chat/completions"
ASSISTANT_TEXT = "Voici votre facture en français."


def chat_body(text: str = "rédige ma facture") -> dict:
    return {"model": "fake-model", "messages": [{"role": "user", "content": text}]}


def upstream_response(content: str = ASSISTANT_TEXT) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.fixture
def fake_upstream(monkeypatch):
    """Replace the gateway's upstream call; records every forwarded payload.

    Returns (calls, responder): mutate responder["fn"] to change the fake
    upstream answer (e.g. a 500).
    """
    calls: list[dict] = []
    responder = {"fn": lambda payload: upstream_response()}

    async def fake(payload: dict) -> httpx.Response:
        calls.append(payload)
        return responder["fn"](payload)

    monkeypatch.setattr("app.gateway.forward_upstream", fake)
    return calls, responder


async def seed_active_fact(
    project_id: str, org_id: str, predicate: str, value: dict, subject_id: str = "usr_42"
) -> None:
    """Seed one active fact through the real path: event -> consolidation."""
    event = EventIn(
        org_id=org_id,
        project_id=project_id,
        subject_type="user",
        subject_id=subject_id,
        kind="conversation.message",
        occurred_at=datetime.now(timezone.utc),
        payload={
            "role": "user",
            "content": "...",
            "mock_facts": [mock_fact(predicate, value, subject_id=subject_id)],
        },
        idempotency_key=f"seed-{uuid.uuid4()}",
    )
    async with async_session() as session:
        results = await ledger.write_events(session, [event])
        await ledger.create_consolidation_job(
            session, project_id=project_id, event_ids=[e.id for e, _ in results]
        )
        await session.commit()
    assert await run_worker() == 1


async def events_for(project_id: str, subject_id: str) -> list[Event]:
    async with async_session() as session:
        stmt = select(Event).where(
            Event.project_id == project_id, Event.subject_id == subject_id
        )
        return list((await session.execute(stmt)).scalars().all())


def auth(key: str, **headers: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", **headers}


# -- 1. memory active: injection, byte-identical response, capture ------------


async def test_gateway_injects_memory_and_captures(
    client, auth_required, make_api_key, fake_upstream
):
    key = await make_api_key(project_id="prj_a", org_id="org_a")
    await seed_active_fact("prj_a", "org_a", "invoice_language", {"language": "fr"})
    calls, _ = fake_upstream

    response = await client.post(
        CHAT_URL,
        json=chat_body(),
        headers=auth(key, **{"X-Haki-Subject-Id": "usr_42"}),
    )

    assert response.status_code == 200
    # The upstream body is returned unchanged.
    assert response.json()["choices"][0]["message"]["content"] == ASSISTANT_TEXT
    assert response.headers["x-haki-memory"] == "active"
    assert "x-haki-trace-id" in response.headers
    assert "x-haki-context-ms" in response.headers

    # The forwarded payload carries the memory block first in a system message.
    upstream_payload = calls[0]
    system = upstream_payload["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("<haki_memory>")
    assert "invoice_language" in system["content"]
    assert "</haki_memory>" in system["content"]

    # The turn was captured and a consolidation job enqueued.
    turns = [
        e
        for e in await events_for("prj_a", "usr_42")
        if e.kind == "conversation.turn"
    ]
    assert len(turns) == 1
    assert turns[0].payload["messages"][0]["content"] == "rédige ma facture"
    assert turns[0].payload["messages"][1]["content"] == ASSISTANT_TEXT
    # M8: a turn through the authenticated gateway proxy is a direct
    # end-user message — trusted, explicit (not derived).
    assert turns[0].origin_trust == "trusted"
    async with async_session() as session:
        jobs = list((await session.execute(select(Job))).scalars().all())
    # One pending consolidate job for the captured turn (the seed's job was
    # already processed by the worker).
    pending = [j for j in jobs if j.status == JobStatus.pending]
    assert len(pending) == 1
    assert pending[0].kind == "consolidate"


# -- 2. no subject: pass-through, disabled, no capture -------------------------


async def test_gateway_without_subject_forwards_without_memory(
    client, auth_required, make_api_key, fake_upstream
):
    key = await make_api_key(project_id="prj_a")
    await seed_active_fact("prj_a", "org_a", "invoice_language", {"language": "fr"})

    response = await client.post(CHAT_URL, json=chat_body(), headers=auth(key))

    assert response.status_code == 200
    assert response.headers["x-haki-memory"] == "disabled"
    assert "x-haki-trace-id" not in response.headers
    upstream_payload = fake_upstream[0][0]
    assert all(m["role"] != "system" for m in upstream_payload["messages"])
    # No capture at all (only the seed event exists, and it is not a turn).
    assert all(
        e.kind != "conversation.turn" for e in await events_for("prj_a", "usr_42")
    )


# -- 3. build_context failure: degraded, response still returned ---------------


async def test_gateway_degraded_when_build_context_fails(
    client, auth_required, make_api_key, fake_upstream, monkeypatch
):
    key = await make_api_key(project_id="prj_a")

    async def boom(*args, **kwargs):
        raise RuntimeError("database is down")

    monkeypatch.setattr("app.api.routes.gateway.build_context", boom)

    response = await client.post(
        CHAT_URL,
        json=chat_body(),
        headers=auth(key, **{"X-Haki-Subject-Id": "usr_42"}),
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ASSISTANT_TEXT
    assert response.headers["x-haki-memory"] == "degraded"
    # Forwarded without any memory block.
    upstream_payload = fake_upstream[0][0]
    assert all(m["role"] != "system" for m in upstream_payload["messages"])


# -- 4. upstream error: status and body propagated unchanged -------------------


async def test_gateway_upstream_error_propagated(
    client, auth_required, make_api_key, fake_upstream
):
    key = await make_api_key(project_id="prj_a")
    error_body = {"error": {"message": "model overloaded", "type": "server_error"}}
    fake_upstream[1]["fn"] = lambda payload: httpx.Response(500, json=error_body)

    response = await client.post(
        CHAT_URL,
        json=chat_body(),
        headers=auth(key, **{"X-Haki-Subject-Id": "usr_42"}),
    )

    assert response.status_code == 500
    assert response.json() == error_body
    # No turn captured on an upstream error.
    assert all(
        e.kind != "conversation.turn" for e in await events_for("prj_a", "usr_42")
    )


# -- 5. idempotence: same key twice -> one captured event ----------------------


async def test_gateway_capture_is_idempotent(
    client, auth_required, make_api_key, fake_upstream
):
    key = await make_api_key(project_id="prj_a")
    headers = auth(
        key,
        **{"X-Haki-Subject-Id": "usr_42", "X-Haki-Idempotency-Key": "gw-idem-1"},
    )

    first = await client.post(CHAT_URL, json=chat_body(), headers=headers)
    second = await client.post(CHAT_URL, json=chat_body(), headers=headers)
    assert first.status_code == second.status_code == 200

    turns = [
        e
        for e in await events_for("prj_a", "usr_42")
        if e.kind == "conversation.turn"
    ]
    assert len(turns) == 1


# -- 6. auth: key mandatory, everything bound to the key's project --------------


async def test_gateway_requires_api_key(client, auth_required):
    response = await client.post(CHAT_URL, json=chat_body())
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"


async def test_gateway_memory_and_capture_stay_in_the_keys_project(
    client, auth_required, make_api_key, fake_upstream
):
    key_a = await make_api_key(project_id="prj_a", org_id="org_a")
    # The fact lives in ANOTHER project: key A must never see it.
    await seed_active_fact("prj_b", "org_b", "invoice_language", {"language": "fr"})

    response = await client.post(
        CHAT_URL,
        json=chat_body(),
        headers=auth(key_a, **{"X-Haki-Subject-Id": "usr_42"}),
    )

    assert response.status_code == 200
    assert response.headers["x-haki-memory"] == "active"  # context built, empty
    upstream_payload = fake_upstream[0][0]
    assert all(m["role"] != "system" for m in upstream_payload["messages"])
    # The capture lands in project A (the key's project), never in B.
    turns_a = [
        e for e in await events_for("prj_a", "usr_42") if e.kind == "conversation.turn"
    ]
    assert len(turns_a) == 1
    assert all(
        e.kind != "conversation.turn" for e in await events_for("prj_b", "usr_42")
    )


# -- streaming: memory injected, user turn captured, assistant turn not -------
#
# Until 22 aout streaming was a pure pass-through: no memory, no capture,
# and the only signal was an X-Haki-Memory: disabled header that no
# OpenAI-compatible SDK reads. Streaming is the DEFAULT mode of nearly
# every integration, so a large share of gateway traffic had no memory at
# all and no way to find out.


async def test_gateway_streaming_injects_memory_into_the_request(
    client, auth_required, make_api_key, monkeypatch
):
    """Injection never needed the response.

    The memory block goes into the REQUEST; only reading the assistant's
    reply back needs the stream. Conflating the two is what cost streaming
    callers their memory.
    """
    key = await make_api_key(project_id="prj_a", org_id="org_a")
    await seed_active_fact("prj_a", "org_a", "invoice_language", {"language": "fr"})
    sent: dict = {}

    async def fake_open_stream(payload: dict):
        sent["payload"] = payload

        async def chunks():
            yield b"data: {}\n\n"

        stream_client = httpx.AsyncClient()
        return (
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=chunks(),
            ),
            stream_client,
        )

    monkeypatch.setattr("app.gateway.open_upstream_stream", fake_open_stream)

    response = await client.post(
        CHAT_URL,
        json={**chat_body(), "stream": True},
        headers=auth(key, **{"X-Haki-Subject-Id": "usr_42"}),
    )

    assert response.status_code == 200
    assert response.content == b"data: {}\n\n"
    assert response.headers["x-haki-memory"] == "active"
    system = sent["payload"]["messages"][0]
    assert system["role"] == "system"
    assert "invoice_language" in system["content"]


async def test_gateway_streaming_captures_the_user_turn_and_says_so(
    client, auth_required, make_api_key, monkeypatch
):
    """The subject said it whatever the model replies.

    The assistant side genuinely cannot be stored without buffering the
    stream -- i.e. without giving up the property the caller asked for --
    so the header states it rather than leaving it to be discovered.
    """
    key = await make_api_key(project_id="prj_a", org_id="org_a")
    await seed_active_fact("prj_a", "org_a", "invoice_language", {"language": "fr"})

    async def fake_open_stream(payload: dict):
        async def chunks():
            yield b"data: {}\n\n"

        return (
            httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=chunks()
            ),
            httpx.AsyncClient(),
        )

    monkeypatch.setattr("app.gateway.open_upstream_stream", fake_open_stream)

    response = await client.post(
        CHAT_URL,
        json={**chat_body(), "stream": True},
        headers=auth(key, **{"X-Haki-Subject-Id": "usr_42"}),
    )
    assert response.status_code == 200
    assert response.headers["x-haki-capture"] == "user_turn_only"

    turns = [
        e for e in await events_for("prj_a", "usr_42") if e.kind == "conversation.turn"
    ]
    assert len(turns) == 1
    messages = turns[0].payload["messages"]
    assert messages[0]["content"] == "rédige ma facture"
    # No null assistant message: a turn with no content is noise for the
    # chunker and for the extractor alike.
    assert len(messages) == 1
