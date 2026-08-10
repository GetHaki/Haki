"""Typologie + volatilite (M2) : extraction avec classe, horloge de
fraicheur, exclusion des volatiles perimes, retrocompat des faits pre-M2
(vraie base, FakeProvider)."""

import uuid
from datetime import datetime, timedelta, timezone

from app.db import async_session
from app.models import ContextTrace, Fact, FactStatus
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


async def test_extraction_stores_llm_proposed_kind_and_volatility(client):
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "current_project",
                        {"name": "Atlas"},
                        fact_kind="attribute",
                        volatility="volatile",
                    ),
                    mock_fact(
                        "invoice_language",
                        {"language": "fr"},
                        fact_kind="preference",
                    ),
                ]
            )
        ],
    )
    await run_worker()

    project_facts = await facts_for("usr_42", "current_project")
    assert project_facts[0].fact_kind == "attribute"
    assert project_facts[0].volatility == "volatile"

    invoice_facts = await facts_for("usr_42", "invoice_language")
    assert invoice_facts[0].fact_kind == "preference"
    assert invoice_facts[0].volatility == "stable"


async def test_extraction_without_classes_defaults_to_attribute_stable(client):
    await capture(
        client, [make_memory_event([mock_fact("birthplace", {"city": "Dakar"})])]
    )
    await run_worker()

    facts = await facts_for("usr_42", "birthplace")
    assert len(facts) == 1
    assert facts[0].fact_kind == "attribute"
    assert facts[0].volatility == "stable"
    assert facts[0].last_reinforced_at is None


async def test_invalid_volatility_candidate_is_rejected_batch_survives(client):
    valid = mock_fact("language", {"lang": "fr"})
    invalid = {
        "subject_id": "usr_42",
        "predicate": "mood",
        "value": {"mood": "curious"},
        "confidence": 0.9,
        "action": "create",
        "evidence_span": "feeling curious today",
        "volatility": "hourly",
    }
    body = await capture(client, [make_memory_event([valid, invalid])])

    assert await run_worker() == 1

    facts = await facts_for("usr_42")
    assert [f.predicate for f in facts] == ["language"]

    async with async_session() as session:
        from app.models import Job

        job = await session.get(Job, uuid.UUID(body["consolidation_job_id"]))
    assert job.payload["result"]["rejected"] == 1
    assert job.payload["result"]["created"] == 1


async def test_expired_volatile_fact_is_never_served_as_current(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=_iso_days_ago(365),
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "current_project",
        },
    )
    body = response.json()
    assert body["packet"]["facts"] == []
    assert body["packet"]["status"] == "degraded"
    assert any(w.startswith("volatility_expired:") for w in body["packet"]["warnings"])

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(
        d.get("action") == "excluded" and d.get("reason_code") == "volatility_expired"
        for d in decisions
    )


async def test_expired_ephemeral_fact_is_excluded_like_volatile(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("mood_today", {"mood": "happy"}, volatility="ephemeral")],
                occurred_at=_iso_days_ago(30),
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "mood_today"},
    )
    body = response.json()
    assert body["packet"]["facts"] == []
    assert body["packet"]["status"] == "degraded"
    assert any(w.startswith("volatility_expired:") for w in body["packet"]["warnings"])


async def test_volatile_fact_within_horizon_is_served_current(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=_iso_days_ago(10),
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "current_project",
            "purpose": "test",
        },
    )
    body = response.json()
    assert body["packet"]["status"] == "ok"
    [fact] = body["packet"]["facts"]
    assert fact["volatility"] == "volatile"
    assert fact["freshness"] == "current"
    assert fact["last_confirmed"] is not None


async def test_slow_fact_past_horizon_served_with_unconfirmed_marker(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("employer", {"name": "Dicken AI"}, volatility="slow")],
                occurred_at=_iso_days_ago(800),
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "employer",
            "purpose": "test",
        },
    )
    body = response.json()
    [fact] = body["packet"]["facts"]
    assert fact["freshness"] == "unconfirmed"
    assert fact["last_confirmed"] is not None
    assert not any(
        w.startswith("volatility_expired:") for w in body["packet"]["warnings"]
    )
    assert body["packet"]["status"] == "ok"


async def test_duplicate_reassertion_refreshes_confirmed_at(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=_iso_days_ago(180),
            )
        ],
    )
    await run_worker()

    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=_iso_days_ago(1),
            )
        ],
    )
    await run_worker()

    [fact] = await facts_for("usr_42", "current_project")
    assert fact.last_reinforced_at is not None
    assert (datetime.now(timezone.utc) - fact.last_reinforced_at) < timedelta(days=2)

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "current_project",
        },
    )
    body = response.json()
    [served] = body["packet"]["facts"]
    assert served["freshness"] == "current"


async def test_replayed_old_event_never_rewinds_confirmed_at(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=_iso_days_ago(180),
            )
        ],
    )
    await run_worker()

    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=_iso_days_ago(1),
            )
        ],
    )
    await run_worker()
    [fact_after_recent] = await facts_for("usr_42", "current_project")
    recent_reinforced_at = fact_after_recent.last_reinforced_at

    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=_iso_days_ago(500),
            )
        ],
    )
    await run_worker()

    [fact_after_old_replay] = await facts_for("usr_42", "current_project")
    assert fact_after_old_replay.last_reinforced_at == recent_reinforced_at


async def test_pre_m2_facts_keep_their_exact_behavior(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("birthplace", {"city": "Dakar"})],
                occurred_at=_iso_days_ago(365 * 5),
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "birthplace",
            "purpose": "test",
        },
    )
    body = response.json()
    assert body["packet"]["status"] == "ok"
    [fact] = body["packet"]["facts"]
    assert fact["freshness"] == "current"


async def test_pre_m2_trace_packet_still_inspectable(client):
    pre_m2_packet = {
        "facts": [
            {
                "id": str(uuid.uuid4()),
                "predicate": "old_predicate",
                "value": {"x": 1},
                "confidence": 0.9,
                "valid_from": "2026-01-01T00:00:00Z",
                "source_event_ids": [],
            }
        ],
        "episodes": [],
        "warnings": [],
        "status": "ok",
    }
    async with async_session() as session:
        trace = ContextTrace(
            project_id="prj_support",
            subject_id="usr_42",
            query="old query",
            purpose=None,
            packet=pre_m2_packet,
            decisions=[],
            token_count=10,
        )
        session.add(trace)
        await session.flush()
        trace_id = trace.id
        await session.commit()

    response = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    assert response.status_code == 200
    [fact] = response.json()["packet"]["facts"]
    assert fact["fact_kind"] is None
    assert fact["volatility"] is None
    assert fact["freshness"] is None


async def test_supersede_inherits_volatility_when_extractor_omits_it(client):
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")]
            )
        ],
    )
    await run_worker()

    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "current_project",
                        {"name": "Zenith"},
                        action="supersede",
                    )
                ]
            )
        ],
    )
    await run_worker()

    facts = await facts_for("usr_42", "current_project")
    active = next(f for f in facts if f.status is FactStatus.active)
    assert active.value == {"name": "Zenith"}
    assert active.volatility == "volatile"


async def test_facts_endpoint_exposes_kind_volatility_last_reinforced_at(client):
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "current_project",
                        {"name": "Atlas"},
                        fact_kind="attribute",
                        volatility="volatile",
                    )
                ]
            )
        ],
    )
    await run_worker()

    response = await client.get(
        "/v1/facts",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    assert response.status_code == 200
    [fact] = response.json()["facts"]
    assert fact["fact_kind"] == "attribute"
    assert fact["volatility"] == "volatile"
    assert "last_reinforced_at" in fact
