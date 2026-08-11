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


def test_build_prompt_context_renders_nothing_for_no_relevant_memory_packet():
    """M3 recall gate: a packet the gate emptied (status ok, empty_reason
    set) renders as "" -- injecting a "no relevant memory" block would
    itself be a distractor. The signal is for the caller, not the prompt."""
    packet = {
        "facts": [],
        "episodes": [],
        "warnings": [],
        "status": "ok",
        "empty_reason": "no_relevant_memory",
    }
    assert build_prompt_context(packet) == ""


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


async def test_consolidate_subject_leaves_other_subjects_pending():
    """POST /v1/consolidate/subject is the scoped counterpart: it must
    process ONLY the named subject's jobs. The unscoped endpoint drains
    every project on the server — the reason `haki verify` moved off it."""
    mine = f"usr_scoped_{uuid.uuid4().hex[:8]}"
    other = f"usr_other_{uuid.uuid4().hex[:8]}"

    async with sdk_client() as client:
        for subject in (mine, other):
            await client.capture(
                [
                    make_memory_event(
                        [mock_fact("plan", {"tier": "pro"}, subject_id=subject)],
                        subject_id=subject,
                    )
                ],
                idempotency_key=f"sdk-{uuid.uuid4()}",
            )

        scoped = await client.consolidate_subject(
            project_id="prj_support", subject_id=mine
        )
        assert scoped["processed"] == 1
        # Idempotent: nothing left pending for that subject.
        assert (
            await client.consolidate_subject(project_id="prj_support", subject_id=mine)
        )["processed"] == 0

        # The other subject's job was untouched and is still waiting.
        assert (
            await client.consolidate_subject(project_id="prj_support", subject_id=other)
        )["processed"] == 1


async def test_facts_lists_superseded_values_the_packet_never_serves():
    """GET /v1/facts?status=superseded is how a caller sees what a newer
    value replaced — the context packet deliberately serves only the
    current one."""
    subject_id = f"usr_facts_{uuid.uuid4().hex[:8]}"

    async with sdk_client() as client:
        await client.capture(
            [
                make_memory_event(
                    [
                        mock_fact(
                            "invoice_language", {"language": "fr"}, subject_id=subject_id
                        )
                    ],
                    subject_id=subject_id,
                )
            ],
            idempotency_key=f"sdk-{uuid.uuid4()}",
        )
        await client.consolidate_subject(
            project_id="prj_support", subject_id=subject_id
        )
        await client.capture(
            [
                make_memory_event(
                    [
                        mock_fact(
                            "invoice_language",
                            {"language": "en"},
                            subject_id=subject_id,
                            action="supersede",
                            supersedes_predicate="invoice_language",
                        )
                    ],
                    subject_id=subject_id,
                    occurred_at="2026-07-28T11:00:00Z",
                )
            ],
            idempotency_key=f"sdk-{uuid.uuid4()}",
        )
        await client.consolidate_subject(
            project_id="prj_support", subject_id=subject_id
        )

        superseded = (
            await client.facts(
                project_id="prj_support", subject_id=subject_id, status="superseded"
            )
        )["facts"]
        assert [f["value"] for f in superseded] == [{"language": "fr"}]
        assert superseded[0]["valid_to"] is not None

        every_status = (
            await client.facts(project_id="prj_support", subject_id=subject_id)
        )["facts"]
        assert {"language": "fr"} in [f["value"] for f in every_status]
        assert {"language": "en"} in [f["value"] for f in every_status]


async def test_verify_scenario_simulated_via_client_calls():
    """The `haki verify` flow, driven through client calls (no CLI
    subprocess): a preference, then a change of mind in the SAME thread ->
    scoped consolidation -> a NEW thread recalls the CURRENT value only,
    with the replaced one still on file as superseded."""
    subject_id = f"usr_verify_{uuid.uuid4().hex[:8]}"
    first = make_memory_event(
        [mock_fact("invoice_language", {"language": "fr"}, subject_id=subject_id)],
        subject_id=subject_id,
    )
    first["thread_id"] = "thr_first"
    second = make_memory_event(
        [
            mock_fact(
                "invoice_language",
                {"language": "en"},
                subject_id=subject_id,
                action="supersede",
                supersedes_predicate="invoice_language",
            )
        ],
        subject_id=subject_id,
        occurred_at="2026-07-28T11:00:00Z",
    )
    second["thread_id"] = "thr_first"  # same thread: this is a correction

    async with sdk_client() as client:
        await client.capture([first], idempotency_key=f"verify-{uuid.uuid4()}")
        await client.consolidate_subject(
            project_id="prj_support", subject_id=subject_id
        )
        await client.capture([second], idempotency_key=f"verify-{uuid.uuid4()}")
        await client.consolidate_subject(
            project_id="prj_support", subject_id=subject_id
        )

        # New thread: the memory must survive the thread boundary, and only
        # the current value may be served.
        response = await client.context(
            subject_id=subject_id,
            query="invoice_language",
            project_id="prj_support",
            purpose="new thread thr_second",
        )
        values = [f["value"] for f in response["packet"]["facts"]]
        assert values == [{"language": "en"}]
        assert response["trace_id"]

        replaced = (
            await client.facts(
                project_id="prj_support", subject_id=subject_id, status="superseded"
            )
        )["facts"]
        assert [f["value"] for f in replaced] == [{"language": "fr"}]


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
