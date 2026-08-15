"""Temporal grounding (mechanism F1, 15 aout Sprint 2):

1. Write-time: a relative time expression in the source text ("last
   week") is resolved into an ISO range (`temporal_range`) anchored on the
   event's `occurred_at`, and persisted onto the fact -- never left as raw
   text, never silently collapsed onto `valid_from` (the MESSAGE's own
   timestamp, not necessarily the described event's).
2. Render-time: every dated packet item (fact, episode) carries an exact,
   precomputed offset from `as_of` ("N days before the question") next to
   its ISO date -- see `_relative_to_now` in app.context. Deterministic
   Python arithmetic, so the reader never has to compute a duration itself
   (Test-of-Time Arithmetic: an LLM given the raw dates gets that right
   only 13.5-16% of the time).
"""

import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.db import async_session
from app.models import Fact
from app.providers.base import ExtractedFact
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

ORG = "org_acme"
PROJECT = "prj_support"


# -- Write-time: schema validation -------------------------------------------


def test_temporal_range_requires_both_bounds():
    with pytest.raises(ValidationError, match="temporal_range"):
        ExtractedFact(
            subject_id="usr_42",
            predicate="hiking_trip",
            value={"trail": "Congress Trail"},
            confidence=0.9,
            action="create",
            evidence_span="went hiking last week",
            temporal_range={"start": "2023-06-18"},  # missing "end"
        )


def test_temporal_range_rejects_non_iso_bounds():
    with pytest.raises(ValidationError, match="ISO 8601"):
        ExtractedFact(
            subject_id="usr_42",
            predicate="hiking_trip",
            value={"trail": "Congress Trail"},
            confidence=0.9,
            action="create",
            evidence_span="went hiking last week",
            temporal_range={"start": "last week", "end": "2023-06-25"},
        )


def test_temporal_range_accepts_a_valid_iso_range():
    candidate = ExtractedFact(
        subject_id="usr_42",
        predicate="hiking_trip",
        value={"trail": "Congress Trail"},
        confidence=0.9,
        action="create",
        evidence_span="went hiking last week",
        temporal_range={"start": "2023-06-18", "end": "2023-06-25"},
    )
    assert candidate.temporal_range == {"start": "2023-06-18", "end": "2023-06-25"}


def test_temporal_range_is_optional():
    # No relative time expression in the source text -> no temporal_range,
    # still validates -- a provider that never heard of this field (or a
    # fact with an absolute date already in `value`) keeps working
    # unchanged.
    candidate = ExtractedFact(
        subject_id="usr_42",
        predicate="native_language",
        value={"language": "fr"},
        confidence=0.9,
        action="create",
        evidence_span="I grew up speaking French",
    )
    assert candidate.temporal_range is None


# -- Write-time: consolidator persistence ------------------------------------


async def test_temporal_range_persisted_on_create(client):
    subject = f"usr_f1_{uuid.uuid4().hex[:8]}"
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "hiking_trip",
                        {"trail": "Congress Trail"},
                        subject_id=subject,
                        evidence_span="went hiking last week",
                        memory_form="event",
                        temporal_range={"start": "2023-06-18", "end": "2023-06-25"},
                    )
                ],
                subject_id=subject,
                occurred_at="2023-06-25T13:22:00Z",
            )
        ],
    )
    await run_worker()

    async with async_session() as session:
        fact = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.subject_id == subject, Fact.predicate == "hiking_trip"
                    )
                )
            )
            .scalars()
            .one()
        )
    assert fact.temporal_range == {"start": "2023-06-18", "end": "2023-06-25"}
    # valid_from stays the MESSAGE's own timestamp -- distinct from
    # temporal_range, which is when the DESCRIBED event happened.
    assert fact.valid_from.isoformat().startswith("2023-06-25")


async def test_temporal_range_absent_when_not_extracted(client):
    subject = f"usr_f1_{uuid.uuid4().hex[:8]}"
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("native_language", {"language": "fr"}, subject_id=subject)],
                subject_id=subject,
            )
        ],
    )
    await run_worker()

    async with async_session() as session:
        fact = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.subject_id == subject, Fact.predicate == "native_language"
                    )
                )
            )
            .scalars()
            .one()
        )
    assert fact.temporal_range is None


# -- Render-time: dual-date, exact offset ------------------------------------


async def test_packet_fact_carries_exact_offset_and_temporal_range(client):
    """The gate case from the book (Partie 5.4): a duration question
    ("how many weeks ago") is answerable because the exact day/week count
    is already in the served text -- computed in Python, not left for the
    reader to derive from two ISO dates."""
    subject = f"usr_f1_{uuid.uuid4().hex[:8]}"
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "hiking_trip",
                        {"trail": "Congress Trail"},
                        subject_id=subject,
                        evidence_span="went hiking last week",
                        memory_form="event",
                        temporal_range={"start": "2023-06-18", "end": "2023-06-25"},
                    )
                ],
                subject_id=subject,
                # Exactly 21 days before as_of below -> "3 weeks".
                occurred_at="2023-06-04T00:00:00Z",
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject,
            "query": "hiking_trip",
            "budget_tokens": 500,
            "as_of": "2023-06-25T00:00:00Z",
        },
    )
    assert response.status_code == 200
    facts = response.json()["packet"]["facts"]
    assert len(facts) == 1
    fact = facts[0]
    assert fact["valid_from_relative"] == "21 days (3 weeks) before the question"
    assert fact["temporal_range"] == {"start": "2023-06-18", "end": "2023-06-25"}


async def test_packet_episode_carries_exact_offset(client):
    subject = f"usr_f1_{uuid.uuid4().hex[:8]}"
    event = {
        "org_id": ORG,
        "project_id": PROJECT,
        "subject_type": "user",
        "subject_id": subject,
        "kind": "chat_session",
        "occurred_at": "2023-06-04T00:00:00Z",
        "payload": {
            "messages": [
                {"role": "user", "content": "Went zorlaxifying yesterday, loved it."}
            ],
            "mock_facts": [
                mock_fact(
                    "zorlaxifying_trip",
                    {"note": "loved it"},
                    subject_id=subject,
                    evidence_span="Went zorlaxifying yesterday",
                )
            ],
        },
    }
    await capture(client, [event])
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject,
            "query": "zorlaxifying",
            "budget_tokens": 500,
            "as_of": "2023-06-25T00:00:00Z",
        },
    )
    assert response.status_code == 200
    episodes = response.json()["packet"]["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["occurred_at_relative"] == "21 days (3 weeks) before the question"
