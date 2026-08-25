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


def test_build_prompt_context_contains_value_dates_and_a_reference():
    packet = {
        "facts": [
            {
                "id": "f1",
                "ref": "F1",
                "predicate": "invoice_language",
                "value": {"language": "fr"},
                "confidence": 0.9,
                "valid_from": "2026-07-28T10:00:00+00:00",
                "valid_from_short": "2026-07-28 10:00",
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
    # The block cites the packet reference, not the uuid (22 aout): a uuid4
    # is 35 o200k tokens against 2, and at ~46 identifiers per packet that
    # was 23 % of everything sent to the model. The id itself is still in
    # the packet, which is where "with proof" has to be resolvable.
    assert "[F1]" in block
    assert "evt-123" not in block
    assert "sources:" not in block
    assert "10:00:00" not in block
    assert "open_conflict" in block
    # Empty packet -> empty block, safe to prepend.
    assert build_prompt_context({"facts": [], "warnings": []}) == ""


def test_build_prompt_context_states_facts_outrank_episodes_on_conflict():
    """13 aout, Bug 3 (temporal tie-break): once episodes carry raw
    historical text alongside facts (key merging, 13 aout), an episode can
    mention a value that has since been superseded. The rendered prompt
    must say explicitly that a fact is the already-resolved current truth
    and wins over a conflicting episode mention -- the guard the 11 aout
    oracle test showed gpt-4o-mini needs spelled out, not left implicit."""
    packet = {
        "facts": [
            {
                "id": "f1",
                "predicate": "wells_fargo_pre_approval",
                "value": {"amount": "$400,000"},
                "valid_from": "2023-11-30T00:00:00+00:00",
                "source_event_ids": ["evt-2"],
            }
        ],
        "episodes": [
            {
                "event_id": "evt-1",
                "kind": "conversation.turn",
                "occurred_at": "2023-08-11T00:00:00+00:00",
                "excerpt": "user: I got pre-approved for $350,000 from Wells Fargo.",
            }
        ],
        "warnings": [],
    }
    block = build_prompt_context(packet)
    assert "the FACT is the current, correct answer" in block
    assert "CURRENT, resolved truth" in block


def test_build_prompt_context_marks_context_window_neighbors():
    """Mechanism F2 (15 aout): a `context_neighbor` episode (added for
    surrounding context, not independently matched to the query) is
    marked as such in the rendered prompt, so the agent does not treat it
    as an equally strong retrieval match to the ordinary episode above
    it."""
    packet = {
        "facts": [],
        "episodes": [
            {
                "event_id": "evt-1",
                "ref": "E1",
                "kind": "chat_session",
                "occurred_at": "2023-05-07T12:00:00+00:00",
                "occurred_at_short": "2023-05-07 12:00",
                "excerpt": "user: Zolgorvex mentioned a favorite pastime.",
                "context_neighbor": False,
            },
            {
                "event_id": "evt-2",
                "ref": "E2",
                "kind": "chat_session",
                "occurred_at": "2023-05-07T13:00:00+00:00",
                "occurred_at_short": "2023-05-07 13:00",
                "excerpt": "user: Unrelated later chat.",
                "context_neighbor": True,
            },
        ],
        "warnings": [],
    }
    block = build_prompt_context(packet)
    lines = block.splitlines()
    ordinary_line = next(line for line in lines if "[E1]" in line)
    neighbor_line = next(line for line in lines if "[E2]" in line)
    assert "surrounding context" not in ordinary_line
    assert "surrounding context" in neighbor_line


def test_build_prompt_context_renders_dual_dates_and_temporal_range():
    """Mechanism F1 (15 aout): every dated packet item carries its ISO date
    AND an exact, precomputed offset from the question ("N days before the
    question") — the reader verifies a number already in the text instead
    of computing one from two ISO dates. A fact whose source text used a
    relative time expression ("last week") also carries the resolved
    temporal_range."""
    packet = {
        "facts": [
            {
                "id": "f1",
                "predicate": "hiking_trip",
                "value": {"trail": "Congress Trail"},
                "confidence": 0.9,
                "valid_from": "2023-06-04T00:00:00+00:00",
                "valid_from_relative": "21 days (3 weeks) before the question",
                "temporal_range": {"start": "2023-06-18", "end": "2023-06-25"},
                "source_event_ids": ["evt-1"],
            }
        ],
        "episodes": [
            {
                "event_id": "evt-2",
                "kind": "chat_session",
                "occurred_at": "2023-06-04T00:00:00+00:00",
                "occurred_at_relative": "21 days (3 weeks) before the question",
                "excerpt": "user: went hiking last week",
                "context_neighbor": False,
            }
        ],
        "warnings": [],
    }
    block = build_prompt_context(packet)
    assert "21 days (3 weeks) before the question" in block
    assert "2023-06-18 to 2023-06-25" in block


def test_build_prompt_context_marks_contested_facts_and_their_tie_break():
    """13 aout, "stop hiding real conflicts": a genuine two-sided
    disagreement is now served (app.context), both facts sharing a
    conflict_id, instead of an empty packet. The rendered prompt must
    flag each side CONTESTED (not silently presented as equally certain
    like an ordinary fact) and carry the explicit exception to the
    "never compare dates yourself" rule -- for a contested pair, and only
    a contested pair, the caller applies the same temporal tie-break Bug 3
    verified."""
    packet = {
        "facts": [
            {
                "id": "f1",
                "predicate": "language",
                "value": {"lang": "fr"},
                "valid_from": "2026-07-28T10:00:00+00:00",
                "source_event_ids": ["evt-1"],
                "contested": True,
                "conflict_id": "c1",
            },
            {
                "id": "f2",
                "predicate": "language",
                "value": {"lang": "en"},
                "valid_from": "2026-07-29T10:00:00+00:00",
                "source_event_ids": ["evt-2"],
                "contested": True,
                "conflict_id": "c1",
            },
        ],
        "warnings": ["open_conflict: 2 fact(s) served with an unresolved conflicting value"],
    }
    block = build_prompt_context(packet)
    # Header exception sentence (1) + dedicated chain-of-note paragraph
    # (2, only emitted when a fact is actually contested) + once per
    # contested fact (2) -- five total.
    assert block.count("CONTESTED") == 5
    assert "find every CONTESTED fact that shares the same conflict id" in block
    assert "conflict c1" in block
    assert "compare 'valid from' dates yourself" in block
    # An ordinary (non-contested) fact never gets the per-fact marker (the
    # header sentence always explains the exception, so "CONTESTED" alone
    # is not a useful signal here -- the marker phrase is).
    ordinary = build_prompt_context(
        {
            "facts": [
                {
                    "id": "f3",
                    "predicate": "plan",
                    "value": {"tier": "pro"},
                    "valid_from": "2026-07-28T10:00:00+00:00",
                    "source_event_ids": ["evt-3"],
                }
            ],
            "warnings": [],
        }
    )
    assert "— CONTESTED (conflict" not in ordinary


def test_build_prompt_context_flags_auto_reclassified_facts():
    """16 aout, reclassification safety net: a fact activated by the
    automatic overflow reclassification (mechanism C) is served like any
    other -- but flagged, so the reader can judge whether 3 "occurrences"
    actually look like updates to one attribute instead of genuinely
    distinct ones."""
    packet = {
        "facts": [
            {
                "id": "f1",
                "predicate": "office_city",
                "value": {"city": "Dakar"},
                "valid_from": "2026-07-28T10:00:00+00:00",
                "source_event_ids": ["evt-1"],
                "auto_reclassified": True,
            },
            {
                "id": "f2",
                "predicate": "plan",
                "value": {"tier": "pro"},
                "valid_from": "2026-07-28T10:00:00+00:00",
                "source_event_ids": ["evt-2"],
            },
        ],
        "warnings": [],
    }
    block = build_prompt_context(packet)
    assert block.count("AUTO-RECLASSIFIED") == 1
    lines = block.splitlines()
    assert any("office_city" in line and "AUTO-RECLASSIFIED" in line for line in lines)
    assert not any(line.startswith("- plan:") and "AUTO-RECLASSIFIED" in line for line in lines)


def test_build_prompt_context_carries_a_past_value_exception():
    """13 aout, LongMemEval diagnostic (run 31705865474): the "always
    prefer most recent" guard is wrong when the question itself asks
    about a PAST value ("what was my goal before I changed it") -- a real
    case (Apex Legends level goal, qid 9bbe84a2) showed gpt-4o-mini can
    answer this correctly, but the guard text gave it no explicit
    permission to pick the earlier date on purpose. Both the contested-
    facts chain-of-note and the facts-vs-episodes tie-break need the same
    carve-out, or a future prompt tightening could silently break it."""
    contested_packet = {
        "facts": [
            {
                "id": "f1",
                "predicate": "apex_legends_level_goal",
                "value": {"goal": 100},
                "valid_from": "2023-06-16T00:00:00+00:00",
                "source_event_ids": ["evt-1"],
                "contested": True,
                "conflict_id": "c1",
            },
            {
                "id": "f2",
                "predicate": "apex_legends_level_goal",
                "value": {"goal": 150},
                "valid_from": "2023-09-30T00:00:00+00:00",
                "source_event_ids": ["evt-2"],
                "contested": True,
                "conflict_id": "c1",
            },
        ],
        "warnings": [],
    }
    block = build_prompt_context(contested_packet)
    assert "EXPLICIT past-state marker" in block
    assert "Ordinary past-tense phrasing alone" in block
    assert "answer with the EARLIER dated value instead" in block

    episodes_packet = {
        "facts": [],
        "episodes": [
            {
                "event_id": "evt-1",
                "kind": "conversation.turn",
                "occurred_at": "2023-06-16T00:00:00+00:00",
                "excerpt": "user: my Apex Legends goal is level 100.",
            }
        ],
        "warnings": [],
    }
    block2 = build_prompt_context(episodes_packet)
    assert "UNLESS the question explicitly asks about a past/previous state" in block2


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
