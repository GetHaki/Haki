"""Typologie + volatilite (M2) : extraction avec classe, horloge de
fraicheur, degradation (pas exclusion, 14 aout -- mecanisme D) des
volatiles perimes, retrocompat des faits pre-M2 (vraie base, FakeProvider)."""

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


async def test_unknown_volatility_candidate_falls_back_to_default_not_rejected(client):
    """Regression guard (17 aout): a real model emitting an out-of-enum
    volatility (or fact_kind/memory_form) used to destroy the WHOLE
    candidate -- see app.providers.base's field_validator. Both facts must
    now be created; the one with the bad tag simply falls back to the
    field's own default ("stable") instead of being lost."""
    valid = mock_fact("language", {"lang": "fr"})
    unknown_volatility = {
        "subject_id": "usr_42",
        "predicate": "mood",
        "value": {"mood": "curious"},
        "confidence": 0.9,
        "action": "create",
        "evidence_span": "feeling curious today",
        "volatility": "hourly",
    }
    body = await capture(client, [make_memory_event([valid, unknown_volatility])])

    assert await run_worker() == 1

    facts = await facts_for("usr_42")
    assert sorted(f.predicate for f in facts) == ["language", "mood"]
    mood_fact = next(f for f in facts if f.predicate == "mood")
    assert mood_fact.volatility == "stable"  # the documented default, not "hourly"

    async with async_session() as session:
        from app.models import Job

        job = await session.get(Job, uuid.UUID(body["consolidation_job_id"]))
    assert job.payload["result"]["rejected"] == 0
    assert job.payload["result"]["created"] == 2


async def test_malformed_candidate_is_rejected_batch_survives(client):
    """A candidate that is genuinely unrecoverable (here: an action outside
    create/supersede/reject -- no documented default to fall back to,
    unlike fact_kind/volatility/memory_form) must still be dropped without
    crashing the rest of the batch."""
    valid = mock_fact("language", {"lang": "fr"})
    malformed = {
        "subject_id": "usr_42",
        "predicate": "mood",
        "value": {"mood": "curious"},
        "confidence": 0.9,
        "action": "update",  # not create/supersede/reject
        "evidence_span": "feeling curious today",
    }
    body = await capture(client, [make_memory_event([valid, malformed])])

    assert await run_worker() == 1

    facts = await facts_for("usr_42")
    assert [f.predicate for f in facts] == ["language"]

    async with async_session() as session:
        from app.models import Job

        job = await session.get(Job, uuid.UUID(body["consolidation_job_id"]))
    assert job.payload["result"]["rejected"] == 1
    assert job.payload["result"]["created"] == 1


async def test_expired_volatile_fact_is_served_flagged_stale(client):
    """14 aout, mecanisme D (research/Diagnostic_Couverture_2026-08-14.md):
    a volatile fact past its horizon used to be excluded outright -- an
    "expired" fact is not FALSE, it is UNCERTAIN, and hiding it left the
    agent knowing neither the value nor that a value existed. Now served,
    same honest-degradation treatment "unconfirmed" already got for slow
    facts, no packet-level warning (per-fact, not a packet problem)."""
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
            "purpose": "test",
        },
    )
    body = response.json()
    [fact] = body["packet"]["facts"]
    assert fact["freshness"] == "stale"
    assert fact["last_confirmed"] is not None
    assert body["packet"]["status"] == "ok"
    assert not any(w.startswith("volatility_expired:") for w in body["packet"]["warnings"])

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(
        d.get("action") == "included" and d.get("fact_id") == fact["id"] for d in decisions
    )


async def test_expired_ephemeral_fact_is_served_stale_like_volatile(client):
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
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "mood_today",
            "purpose": "test",
        },
    )
    body = response.json()
    [fact] = body["packet"]["facts"]
    assert fact["freshness"] == "stale"
    assert body["packet"]["status"] == "ok"


async def test_as_of_reads_freshness_from_the_callers_point_of_view(client):
    """14 aout, mecanisme D: the same volatile fact, ingested "365 days ago"
    relative to the real wall clock (so it would read "stale" by default --
    see test_expired_volatile_fact_is_served_flagged_stale above), reads
    "current" when the caller passes `as_of` close to when the fact was
    actually recorded -- exactly the LoCoMo/LongMemEval case: a conversation
    dated years in the past, queried from its OWN point in time rather than
    today's real clock."""
    occurred_at = _iso_days_ago(365)
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("current_project", {"name": "Atlas"}, volatility="volatile")],
                occurred_at=occurred_at,
            )
        ],
    )
    await run_worker()

    # Without as_of: real wall clock, 365 days later -> stale.
    default_response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "current_project"},
    )
    [default_fact] = default_response.json()["packet"]["facts"]
    assert default_fact["freshness"] == "stale"

    # With as_of a few days after occurred_at (the fact's own point of
    # view): well within the volatility horizon -> current.
    as_of = (datetime.fromisoformat(occurred_at) + timedelta(days=5)).isoformat()
    as_of_response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "current_project",
            "as_of": as_of,
        },
    )
    [as_of_fact] = as_of_response.json()["packet"]["facts"]
    assert as_of_fact["freshness"] == "current"


async def test_facts_from_after_as_of_are_excluded_from_scope(client):
    """Found 20 Aug 2026 via external code audit, confirmed by direct
    inspection: build_context's scope_filters checked valid_to > now but
    never checked valid_from <= now. A fact whose effective date (valid_from,
    or recorded_from when unset) is AFTER the caller's as_of doesn't just
    slip into an EARLY packet -- the recency term
    exp(-(now - valid_from) / TAU) goes POSITIVE (exponential growth, not
    decay) for a negative elapsed time, so the fact's score can run away and
    dominate the ranked pool ahead of every genuinely relevant fact. This
    matters most on exactly the workload that exercises `as_of` on purpose:
    LoCoMo/LongMemEval questions asked from a point in time inside the
    conversation, not the real wall clock."""
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("future_project", {"name": "Atlas"})],
                occurred_at="2023-06-01T00:00:00Z",
            )
        ],
    )
    await run_worker()

    # as_of BEFORE the fact's own date: from this point of view the fact
    # doesn't exist yet and must not be served -- let alone dominate.
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "future_project",
            "as_of": "2023-01-01T00:00:00Z",
        },
    )
    assert response.status_code == 200
    assert response.json()["packet"]["facts"] == []


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
