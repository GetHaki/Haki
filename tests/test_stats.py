"""GET /v1/stats/overview + the timing/hit-rate fields build_context now
writes on every ContextTrace — every number here traces back to real rows,
never a placeholder."""

from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

PROJECT = "prj_support"
SUBJECT = "usr_42"


async def test_context_trace_has_real_timing_and_fact_count(client):
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "fr"})])])
    assert await run_worker() == 1

    ctx = await client.post(
        "/v1/context",
        json={"project_id": PROJECT, "subject_id": SUBJECT, "query": "langue préférée ?"},
    )
    trace_id = ctx.json()["trace_id"]

    inspected = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": PROJECT, "subject_id": SUBJECT},
    )
    body = inspected.json()
    assert body["fact_count"] == 1
    assert isinstance(body["duration_ms"], int) and body["duration_ms"] >= 0
    assert set(body["stage_timings"]) >= {"embed", "retrieval", "episodes"}
    assert all(isinstance(v, int) for v in body["stage_timings"].values())


async def test_stats_overview_reflects_real_facts_recalls_and_hit_rate(client):
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "fr"})])])
    assert await run_worker() == 1

    # One recall that hits (the fact above), one that misses (no facts for
    # this made-up subject) — hit_rate must land at exactly 0.5.
    await client.post(
        "/v1/context",
        json={"project_id": PROJECT, "subject_id": SUBJECT, "query": "langue ?"},
    )
    await client.post(
        "/v1/context",
        json={"project_id": PROJECT, "subject_id": "usr_nothing", "query": "anything"},
    )

    response = await client.get("/v1/stats/overview", params={"project_id": PROJECT})
    assert response.status_code == 200
    body = response.json()

    assert body["active_facts"] == 1
    assert body["recall_count"] == 2
    assert body["hit_rate"] == 0.5
    assert body["recall_p50_ms"] is not None
    assert body["context_tokens_served"] > 0
    # make_memory_event uses a fixed historical occurred_at, so it may or
    # may not fall in the rolling 7-day window depending on when this runs
    # — only the shape is asserted here, not a specific bucket's count.
    assert len(body["events_this_week"]) == 7
    assert all(d["count"] >= 0 for d in body["events_this_week"])


async def test_stats_overview_with_no_history_returns_none_not_zero(client):
    response = await client.get(
        "/v1/stats/overview", params={"project_id": "prj_totally_empty"}
    )
    body = response.json()
    assert body["active_facts"] == 0
    assert body["recall_count"] == 0
    assert body["hit_rate"] is None
    assert body["recall_p50_ms"] is None


async def test_stats_overview_requires_project_id(client):
    response = await client.get("/v1/stats/overview")
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "missing_scope"
