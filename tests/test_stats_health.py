"""GET /v1/stats/health — memory-health score over a bounded 30-day window.

Every metric traces back to real rows (facts, events, context_traces,
conflict_sets, persisted job results); a metric with an empty denominator
is None, never a fake 0 or 100 — the "never a false 100" guarantee is
locked by test_health_all_null_on_empty_project.
"""

import uuid

from app.db import async_session
from app.models import ContextTrace
from app.providers.fake import mock_fact
from test_consolidator import capture, make_memory_event, run_worker

PROJECT = "prj_support"
SUBJECT = "usr_42"


async def insert_trace(project_id: str, subject_id: str, fact_ids: list[uuid.UUID]) -> None:
    """Directly-inserted trace carrying arbitrary fact ids in its packet —
    the real context assembler filters status=active so it can never
    produce a leaked packet on demand; this crafts exactly the shape the
    metric is meant to detect."""
    async with async_session() as session:
        session.add(
            ContextTrace(
                project_id=project_id,
                subject_id=subject_id,
                query="crafted",
                packet={
                    "facts": [{"id": str(f)} for f in fact_ids],
                    "episodes": [],
                    "warnings": [],
                    "status": "ok",
                },
                decisions=[],
                token_count=0,
                fact_count=len(fact_ids),
            )
        )
        await session.commit()


async def health(client, **params) -> dict:
    response = await client.get("/v1/stats/health", params=params)
    assert response.status_code == 200
    return response.json()


async def test_health_requires_project_id(client):
    response = await client.get("/v1/stats/health")
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "missing_scope"


async def test_health_all_null_on_empty_project(client):
    body = await health(client, project_id="prj_never_seen")
    assert body["health_score"] is None
    assert body["injection_rate"] is None
    assert body["fact_density"] is None
    assert body["write_rejection_rate"] is None
    assert body["contradiction_leakage"] is None
    assert body["staleness"] is None
    assert body["open_conflicts"] == 0
    assert body["rejection_breakdown"] == {}
    assert all(c["value"] is None for c in body["components"])


async def test_injection_rate_over_window(client):
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "fr"})])])
    assert await run_worker() == 1

    hit = await client.post(
        "/v1/context",
        json={"project_id": PROJECT, "subject_id": SUBJECT, "query": "language"},
    )
    assert hit.status_code == 200
    miss = await client.post(
        "/v1/context",
        json={"project_id": PROJECT, "subject_id": "usr_nothing", "query": "language"},
    )
    assert miss.status_code == 200

    body = await health(client, project_id=PROJECT)
    assert body["traces_in_window"] == 2
    assert body["packets_with_facts"] == 1
    assert body["injection_rate"] == 0.5


async def test_fact_density_active_over_events(client):
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "fr"})])])
    await capture(client, [make_memory_event([])])
    assert await run_worker() == 2

    body = await health(client, project_id=PROJECT)
    assert body["active_facts"] == 1
    assert body["events_total"] == 2
    assert body["fact_density"] == 0.5


async def test_write_rejection_rate_and_breakdown_from_job_results(client):
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("lang", {"l": "fr"}),
                    mock_fact(
                        "noise", {}, action="reject", reject_reason="system_noise"
                    ),
                ]
            )
        ],
    )
    assert await run_worker() == 1

    body = await health(client, project_id=PROJECT)
    assert body["candidates_total"] == 2
    assert body["write_rejection_rate"] == 0.5
    assert body["rejection_breakdown"] == {"system_noise": 1}


async def test_write_rejection_breakdown_counts_unclassified(client):
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("lang", {"l": "fr"}),
                    {"value": {}},  # invalid: no predicate -> Pydantic validation fails
                ]
            )
        ],
    )
    assert await run_worker() == 1

    body = await health(client, project_id=PROJECT)
    assert body["write_rejection_rate"] == 0.5
    assert body["rejection_breakdown"] == {"unclassified": 1}


async def test_contradiction_leakage_detected_on_constructed_case(client):
    await capture(client, [make_memory_event([mock_fact("plan", {"tier": "free"})])])
    assert await run_worker() == 1

    served_before = await client.post(
        "/v1/context",
        json={"project_id": PROJECT, "subject_id": SUBJECT, "query": "plan"},
    )
    assert served_before.status_code == 200
    fact_id = uuid.UUID(served_before.json()["packet"]["facts"][0]["id"])

    await capture(
        client,
        [make_memory_event([mock_fact("plan", {"tier": "pro"}, action="supersede")])],
    )
    assert await run_worker() == 1

    await insert_trace(PROJECT, SUBJECT, [fact_id])

    body = await health(client, project_id=PROJECT)
    assert body["packets_with_facts"] == 2
    assert body["leaked_packets"] == 1
    assert body["contradiction_leakage"] == 0.5


async def test_contradiction_leakage_counts_conflict_resolution_path(client):
    await capture(client, [make_memory_event([mock_fact("home_city", {"city": "Lyon"})])])
    assert await run_worker() == 1
    await capture(
        client, [make_memory_event([mock_fact("home_city", {"city": "Berlin"})])]
    )
    assert await run_worker() == 1

    from sqlalchemy import select

    from app.models import ConflictSet

    async with async_session() as session:
        conflict = (
            (
                await session.execute(
                    select(ConflictSet).where(
                        ConflictSet.project_id == PROJECT,
                        ConflictSet.subject_id == SUBJECT,
                        ConflictSet.status == "open",
                    )
                )
            )
            .scalars()
            .first()
        )
    assert conflict is not None
    keep_id, loser_id = conflict.fact_ids[0], conflict.fact_ids[1]

    resolve = await client.post(
        f"/v1/conflicts/{conflict.id}/resolve",
        json={"project_id": PROJECT, "keep_fact_id": str(keep_id)},
    )
    assert resolve.status_code == 200

    await insert_trace(PROJECT, SUBJECT, [loser_id])

    body = await health(client, project_id=PROJECT)
    assert body["leaked_packets"] == 1


async def test_health_score_formula_renormalized(client):
    # Subject A: one fact, one hit -> injection 1.0, leakage 0 -> integrity 1.0.
    await capture(
        client, [make_memory_event([mock_fact("plan", {"tier": "free"})], subject_id="usr_a")]
    )
    assert await run_worker() == 1
    ctx = await client.post(
        "/v1/context", json={"project_id": PROJECT, "subject_id": "usr_a", "query": "plan"}
    )
    assert ctx.status_code == 200

    # Subject B: one open conflict -> open_conflicts == 1 -> hygiene 0.9.
    await capture(
        client, [make_memory_event([mock_fact("home_city", {"city": "Lyon"})], subject_id="usr_b")]
    )
    assert await run_worker() == 1
    await capture(
        client,
        [make_memory_event([mock_fact("home_city", {"city": "Berlin"})], subject_id="usr_b")],
    )
    assert await run_worker() == 1

    body = await health(client, project_id=PROJECT)
    assert body["open_conflicts"] == 1
    assert body["health_score"] == 97.5

    by_name = {c["name"]: c for c in body["components"]}
    assert len(body["components"]) == 4
    assert by_name["staleness"]["value"] is None
    assert by_name["injection"]["weight"] == 0.30
    assert by_name["contradiction_integrity"]["weight"] == 0.30
    assert by_name["conflict_hygiene"]["weight"] == 0.20
    assert by_name["staleness"]["weight"] == 0.20


async def test_staleness_always_null(client):
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "fr"})])])
    assert await run_worker() == 1
    await client.post(
        "/v1/context",
        json={"project_id": PROJECT, "subject_id": SUBJECT, "query": "language"},
    )

    body = await health(client, project_id=PROJECT)
    assert body["staleness"] is None
    by_name = {c["name"]: c for c in body["components"]}
    assert by_name["staleness"]["value"] is None


async def test_score_null_when_only_informational_metrics_measurable(client):
    await capture(client, [make_memory_event([])])
    assert await run_worker() == 1

    body = await health(client, project_id=PROJECT)
    assert body["events_total"] == 1
    assert body["fact_density"] == 0.0
    assert body["health_score"] is None
