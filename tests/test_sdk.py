"""SDK tests (sprint 3): HakiClient against the ASGI app (httpx ASGITransport),
runtime helpers, and the verify scenario simulated through client calls.

The fake providers are forced by conftest (hermetic tests).
"""

import uuid

import httpx
import pytest

from haki import AsyncHakiClient, HakiApiError, HakiClient
from haki.runtime import build_prompt_context, capture_turn

from app.main import app
from tests.test_consolidator import make_memory_event
from app.providers.fake import mock_fact


def sdk_client() -> AsyncHakiClient:
    return AsyncHakiClient(
        "http://test", transport=httpx.ASGITransport(app=app)
    )


async def test_capture_context_inspect_roundtrip():
    async with sdk_client() as client:
        body = await client.capture(
            [make_memory_event([mock_fact("invoice_language", {"language": "fr"})])],
            idempotency_key=f"sdk-{uuid.uuid4()}",
        )
        assert body["status"] == "accepted"
        assert body["consolidation_job_id"]

        result = await client.consolidate()
        assert result["processed"] == 1

        response = await client.context(
            subject_id="usr_42", query="invoice_language", project_id="prj_support"
        )
        facts = response["packet"]["facts"]
        assert [f["value"] for f in facts] == [{"language": "fr"}]
        assert response["token_count"] > 0

        trace = await client.inspect(
            response["trace_id"], project_id="prj_support", subject_id="usr_42"
        )
        assert trace["query"] == "invoice_language"
        assert trace["decisions"][0]["action"] == "included"

        timeline = await client.timeline("usr_42", "prj_support")
        assert len(timeline["events"]) == 1


async def test_consolidate_endpoint_processes_pending_jobs():
    """POST /v1/consolidate runs pending jobs synchronously (dev/ops endpoint)."""
    async with sdk_client() as client:
        await client.capture(
            [make_memory_event([mock_fact("plan", {"tier": "pro"})])],
            idempotency_key=f"sdk-{uuid.uuid4()}",
        )
        assert (await client.consolidate())["processed"] == 1
        # Nothing left pending.
        assert (await client.consolidate())["processed"] == 0

        response = await client.context(
            subject_id="usr_42", query="plan", project_id="prj_support"
        )
        assert [f["value"] for f in response["packet"]["facts"]] == [{"tier": "pro"}]


def test_build_prompt_context_contains_value_dates_and_sources():
    packet = {
        "facts": [
            {
                "id": "f1",
                "predicate": "invoice_language",
                "value": {"language": "fr"},
                "confidence": 0.9,
                "valid_from": "2026-07-28T10:00:00+00:00",
                "source_event_ids": ["evt-123"],
            }
        ],
        "warnings": ["open_conflict: 1 fact(s) hidden pending conflict resolution"],
    }
    block = build_prompt_context(packet)
    assert "<haki_memory>" in block and "</haki_memory>" in block
    assert "invoice_language" in block
    assert "fr" in block
    assert "2026-07-28" in block
    assert "evt-123" in block
    assert "open_conflict" in block
    # Empty packet -> empty block, safe to prepend.
    assert build_prompt_context({"facts": [], "warnings": []}) == ""


def test_capture_turn_builds_a_well_formed_event():
    class StubClient:
        def __init__(self):
            self.calls = []

        def capture(self, events, idempotency_key=None):
            self.calls.append((events, idempotency_key))
            return {"status": "accepted"}

    stub = StubClient()
    capture_turn(
        stub, "usr_42", "prj_support", "Bonjour", "Bonjour !", thread_id="thr_1"
    )
    events, _ = stub.calls[0]
    event = events[0]
    assert event["subject_id"] == "usr_42"
    assert event["project_id"] == "prj_support"
    assert event["thread_id"] == "thr_1"
    assert event["payload"]["messages"] == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Bonjour !"},
    ]
    assert event["idempotency_key"].startswith("turn-")


async def test_verify_scenario_simulated_via_client_calls():
    """The `haki verify` flow, driven through client calls (no CLI subprocess):
    capture a preference -> consolidate -> NEW thread -> context recalls it."""
    subject_id = f"usr_verify_{uuid.uuid4().hex[:8]}"
    event = make_memory_event(
        [mock_fact("invoice_language", {"language": "fr"}, subject_id=subject_id)],
        subject_id=subject_id,
    )
    event["thread_id"] = "thr_first"

    async with sdk_client() as client:
        await client.capture([event], idempotency_key=f"verify-{uuid.uuid4()}")
        assert (await client.consolidate())["processed"] == 1

        # New thread: the memory must survive the thread boundary.
        response = await client.context(
            subject_id=subject_id,
            query="invoice_language",
            project_id="prj_support",
            purpose="new thread thr_second",
        )
        values = [f["value"] for f in response["packet"]["facts"]]
        assert {"language": "fr"} in values
        assert response["trace_id"]


def test_sync_client_maps_typed_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "error": {
                    "type": "missing_scope",
                    "message": "subject_id query parameter is required",
                    "field": "subject_id",
                }
            },
        )

    client = HakiClient("http://test", transport=httpx.MockTransport(handler))
    with pytest.raises(HakiApiError) as excinfo:
        client.timeline("usr_42", "prj_support")
    assert excinfo.value.error_type == "missing_scope"
    assert excinfo.value.status_code == 422
    assert "missing_scope" in str(excinfo.value)
    client.close()
