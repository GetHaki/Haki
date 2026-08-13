"""Context Assembler behaviors (real database, FakeProvider):

token budget, scope isolation, trace inspection with reason codes.
"""

import pytest
from sqlalchemy import select

from app.config import settings
from app.consolidator import _search_text
from app.context import episode_text
from app.db import async_session
from app.models import Fact
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
            "budget_tokens": 60,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_count"] <= 60
    assert 1 <= len(body["packet"]["facts"]) < 3

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(d["reason_code"] == "over_budget" for d in decisions)
    included = [d for d in decisions if d["action"] == "included"]
    assert len(included) == len(body["packet"]["facts"])


async def test_tied_score_facts_pack_deterministically_across_repeated_calls(client):
    """13 aout, "Bug 2" root cause (11 aout: five different questions
    returned an identical packet -- budget headroom explained SOME cases,
    but not a HIGH-volume subject where discrimination should matter, see
    scripts/check_retrieval_discrimination.py): facts written in the same
    batch share coalesce(valid_from, recorded_from) down to the minute, so
    once similarity is ALSO tied (a real case: two facts sharing an
    embedding, e.g. from the semantic-fallback/alias mechanism, or simply
    two facts an embedder happens to score identically for a given query)
    their scores tie EXACTLY on every axis -- and without a secondary sort
    key, which one wins a tight budget cutoff is left to Postgres's query
    plan, not to anything meaningful, and is NOT guaranteed to repeat
    across otherwise-identical calls.

    FakeEmbedder does not naturally tie different predicates' similarity
    (verified: topic_a/b/c score 0.068/0.251/0.055 for query "topic", not
    tied) -- topic_b and topic_c are forced to share topic_b's embedding
    here, the same technique tests/test_semantic_supersession.py uses to
    simulate what a real embedder can and does produce naturally."""
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

    async with async_session() as session:
        facts = (
            (
                await session.execute(
                    select(Fact).where(Fact.project_id == "prj_support", Fact.subject_id == "usr_42")
                )
            )
            .scalars()
            .all()
        )
        by_predicate = {f.predicate: f for f in facts}
        tied_embedding = by_predicate["topic_b"].embedding
        by_predicate["topic_c"].embedding = tied_embedding
        await session.commit()

    # topic_a keeps its own (higher) natural score and is always included;
    # budget fits exactly topic_a + one of the tied pair {topic_b, topic_c}
    # (~55 tokens each, 3x > 110 >= 2x) -- which ONE of the tied pair wins
    # is the actual thing under test.
    ids_by_call = []
    for _ in range(5):
        response = await client.post(
            "/v1/context",
            json={
                "project_id": "prj_support",
                "subject_id": "usr_42",
                "query": "topic",
                "budget_tokens": 110,
            },
        )
        assert response.status_code == 200
        packet_facts = response.json()["packet"]["facts"]
        assert len(packet_facts) == 2
        ids_by_call.append(tuple(sorted(f["id"] for f in packet_facts)))

    assert len(set(ids_by_call)) == 1, f"non-deterministic packet across repeated calls: {ids_by_call}"


async def test_unified_pool_lets_a_highly_relevant_episode_outrank_low_relevance_facts(client):
    """Key merging (13 aout): no more fixed floor for episodes (removed
    EPISODE_MIN_BUDGET_SHARE) -- they win real budget space by outscoring
    less relevant facts in ONE ranked pool, not by a guaranteed share. Six
    facts irrelevant to the query (FakeEmbedder: different texts ~1.0
    cosine distance, near-zero similarity) would alone exceed a 300-token
    budget (~55 tokens each); one episode whose content EXACTLY matches
    the query (FakeEmbedder: identical text -> distance 0, near-max
    similarity) must still win a slot despite competing against 6 higher-
    count candidates -- proof the merge is real, not order-dependent."""
    big = {"detail": "x" * 200}
    episode_payload = {"messages": [{"role": "user", "content": "I visited Lisbon in July."}]}
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
                "payload": episode_payload,
            },
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": episode_text("chat_session", episode_payload),
            "budget_tokens": 300,
        },
    )
    assert response.status_code == 200
    body = response.json()
    packet = body["packet"]
    assert body["token_count"] <= 300
    # Six irrelevant facts (~55 tokens each) don't all fit in 300 tokens
    # regardless of the episode -- budget alone forces some out.
    assert len(packet["facts"]) < 6
    # The episode wins a slot on merit (top score), not on a reserved share.
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
