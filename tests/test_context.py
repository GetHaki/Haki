"""Context Assembler behaviors (real database, FakeProvider):

token budget, scope isolation, trace inspection with reason codes.
"""

import pytest

from app.config import settings
from app.consolidator import _search_text
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker


@pytest.fixture
def recall_floor(monkeypatch):
    """Enable the recall gate for one test (HAKI_RECALL_MAX_DISTANCE=0.5).

    FakeEmbedder distances: identical text = 0.0, different texts ~= 1.0
    (sha256-derived vectors never cluster) — 0.5 splits the two regimes
    with ample margin.
    """
    monkeypatch.setattr(settings, "recall_max_distance", 0.5)
    return settings


async def test_budget_packs_best_scored_facts_and_traces_over_budget(client):
    # Three facts with large values (~50 estimated tokens each).
    big = {"detail": "x" * 200}
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("topic_a", big),
                    mock_fact("topic_b", big),
                    mock_fact("topic_c", big),
                ]
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "topic",
            # 160, not 60: facts are now capped at budget_tokens * (1 -
            # EPISODE_MIN_BUDGET_SHARE) so episodes always get a real share
            # (see app/context/__init__.py) — at ~55 estimated tokens per
            # fact here, a facts share of 80 (half of 160) still fits
            # exactly one of the three and excludes the other two, which is
            # what this test is actually about.
            "budget_tokens": 160,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_count"] <= 160
    assert 1 <= len(body["packet"]["facts"]) < 3

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(d["reason_code"] == "over_budget" for d in decisions)
    included = [d for d in decisions if d["action"] == "included"]
    assert len(included) == len(body["packet"]["facts"])


async def test_episodes_get_a_guaranteed_share_even_with_many_active_facts(client):
    """EPISODE_MIN_BUDGET_SHARE: a subject with enough active facts to
    consume the whole budget on their own must still get an episode served,
    not zero. Facts alone would exhaust budget_tokens on this data (6 facts
    x ~55 tokens = ~330 > 300); the reservation caps facts at
    facts_budget_cap (150) so the episode loop, which still checks against
    the FULL budget_tokens, has real room left."""
    big = {"detail": "x" * 200}
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("topic_a", big),
                    mock_fact("topic_b", big),
                    mock_fact("topic_c", big),
                    mock_fact("topic_d", big),
                    mock_fact("topic_e", big),
                    mock_fact("topic_f", big),
                ]
            ),
            {
                "org_id": "org_acme",
                "project_id": "prj_support",
                "subject_type": "user",
                "subject_id": "usr_42",
                "kind": "chat_session",
                "occurred_at": "2026-07-29T10:00:00Z",
                "payload": {"messages": [{"role": "user", "content": "I visited Lisbon in July."}]},
            },
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "topic",
            "budget_tokens": 300,
        },
    )
    assert response.status_code == 200
    body = response.json()
    packet = body["packet"]
    assert body["token_count"] <= 300
    # Facts alone would fit up to 5 of the 6 (275 tokens) if unreserved --
    # the cap must leave some out.
    assert len(packet["facts"]) < 6
    # The episode still made it in: the reservation did its job.
    assert len(packet["episodes"]) == 1
    assert "Lisbon" in packet["episodes"][0]["excerpt"]


async def test_budget_zero_or_negative_is_a_typed_error(client):
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "anything",
            "budget_tokens": 0,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "budget_exceeded"


async def test_context_never_leaks_across_subjects(client):
    await capture(
        client,
        [make_memory_event([mock_fact("plan", {"tier": "pro"}, subject_id="usr_a")], subject_id="usr_a")],
    )
    await capture(
        client,
        [make_memory_event([mock_fact("plan", {"tier": "free"}, subject_id="usr_b")], subject_id="usr_b")],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_a", "query": "plan"},
    )
    served = response.json()["packet"]["facts"]
    assert len(served) == 1
    assert served[0]["value"] == {"tier": "pro"}


async def test_inspect_trace_never_leaks_across_scopes(client):
    await capture(
        client,
        [make_memory_event([mock_fact("plan", {"tier": "pro"})])],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "plan"},
    )
    trace_id = response.json()["trace_id"]

    # Right scope: full trace with decisions.
    ok = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    assert ok.status_code == 200
    assert ok.json()["query"] == "plan"
    assert ok.json()["decisions"][0]["action"] == "included"

    # Other subject, same project: 404, never a leak of the trace.
    other_subject = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": "prj_support", "subject_id": "usr_99"},
    )
    assert other_subject.status_code == 404
    assert other_subject.json()["error"]["type"] == "trace_not_found"

    # Other project: 404 as well.
    other_project = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": "prj_other", "subject_id": "usr_42"},
    )
    assert other_project.status_code == 404

    # Missing scope: typed error.
    missing = await client.get(f"/v1/inspect/{trace_id}")
    assert missing.status_code == 422
    assert missing.json()["error"]["type"] == "missing_scope"


async def test_recall_floor_disabled_by_default_keeps_greedy_packing(client):
    """M3 retro-compat: with no HAKI_RECALL_MAX_DISTANCE set (0.0 = gate
    off), an off-topic query still serves whatever fits the budget, exactly
    as before this chantier — documents the behavior the gate corrects."""
    await capture(
        client, [make_memory_event([mock_fact("language", {"lang": "fr"})])]
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "recette de lasagnes maison",
            "purpose": "test",
        },
    )
    body = response.json()
    assert len(body["packet"]["facts"]) == 1
    assert body["packet"]["empty_reason"] is None
    assert body["packet"]["status"] == "ok"


async def test_recall_floor_returns_honest_empty_packet_for_off_topic_query(
    recall_floor, client
):
    await capture(
        client, [make_memory_event([mock_fact("language", {"lang": "fr"})])]
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "recette de lasagnes maison",
            "purpose": "test",
        },
    )
    body = response.json()
    assert body["packet"]["facts"] == []
    assert body["packet"]["episodes"] == []
    assert body["packet"]["status"] == "ok"
    assert body["packet"]["warnings"] == []
    assert body["packet"]["empty_reason"] == "no_relevant_memory"
    assert body["token_count"] == 0

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(
        d["action"] == "excluded" and d["reason_code"] == "below_relevance_floor"
        for d in decisions
    )
    assert not any(d["action"] == "included" for d in decisions)
    assert trace.json()["packet"]["empty_reason"] == "no_relevant_memory"


async def test_recall_floor_serves_relevant_query_unchanged(recall_floor, client):
    await capture(
        client, [make_memory_event([mock_fact("language", {"lang": "fr"})])]
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": _search_text("language", {"lang": "fr"}),
            "purpose": "test",
        },
    )
    body = response.json()
    assert len(body["packet"]["facts"]) == 1
    assert body["packet"]["empty_reason"] is None
    assert body["packet"]["status"] == "ok"

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(
        d["action"] == "included" and d["reason_code"] == "top_score"
        for d in decisions
    )


async def test_recall_floor_empty_subject_is_not_no_relevant_memory(
    recall_floor, client
):
    """Honesty of the signal: a subject with zero facts at all is a
    different case from "has facts, none relevant" -- no candidate was
    ever rejected by the gate, so empty_reason must stay None."""
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_never_captured",
            "query": "anything",
        },
    )
    body = response.json()
    assert body["packet"]["facts"] == []
    assert body["packet"]["empty_reason"] is None


async def test_recall_floor_gates_episodes_too(recall_floor, client):
    await capture(
        client, [make_memory_event([mock_fact("language", {"lang": "fr"})])]
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "recette de lasagnes maison",
        },
    )
    body = response.json()
    assert body["packet"]["episodes"] == []

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(
        d.get("episode_id") and d["reason_code"] == "below_relevance_floor"
        for d in decisions
    )
