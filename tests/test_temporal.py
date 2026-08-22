"""When a fact is ABOUT: `observed_at`, and the range it inherits.

Three instants get confused constantly in a memory system:

    recorded_from   when Haki learned it
    valid_from      when it became true (the message's own timestamp)
    observed_at     when the fact HAPPENED

"I got pre-approved back in August", said on 30 November: the first two
are November, and the answer to "when did you get pre-approved?" is
August. Temporal reasoning is the category every published memory system
is worst at, and a date stored as free-form JSON under an arbitrary key is
a large part of why.
"""

from datetime import datetime, timezone

import pytest

from app.consolidator.temporal import observed_at_of, parse_iso_instant
from app.models import FactStatus
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker

AUGUST = {"start": "2023-08-01", "end": "2023-08-31"}


# --------------------------------------------------------------------------
# Deriving the instant
# --------------------------------------------------------------------------

def test_a_resolved_range_anchors_the_fact_at_its_start():
    """The extractor said so explicitly; nothing else competes with that."""
    assert observed_at_of({"state": "pre-approved"}, AUGUST) == datetime(
        2023, 8, 1, tzinfo=timezone.utc
    )


def test_a_single_date_in_the_value_is_taken_whatever_its_key():
    """Key-agnostic on purpose.

    The extraction prompt lets the extractor put an absolute date in
    `value` without prescribing the key. Matching on a list of key names
    ("date", "when", "start_date"...) silently misses the ones nobody
    thought of; what is matched is the VALUE looking like an ISO date.
    """
    for value in (
        {"date": "2023-06-02"},
        {"purchased_on": "2023-06-02"},
        {"detail": {"when": "2023-06-02T09:30:00Z"}},
    ):
        assert observed_at_of(value, None) is not None, value


def test_two_dates_in_one_value_produce_nothing():
    """Exact or absent, like source_chunk_id.

    A confidently wrong date is worse than no date: it gets rendered to
    the reader as fact. Ambiguity resolves to NULL.
    """
    assert observed_at_of({"start": "2023-06-02", "end": "2023-06-09"}, None) is None


def test_a_fact_about_no_instant_gets_none():
    """The common case, and a real answer rather than a failure.

    Inventing an instant for "I have a dog" would make the column
    meaningless for the facts that genuinely have one.
    """
    assert observed_at_of({"pet": "dog"}, None) is None


@pytest.mark.parametrize(
    "raw",
    [
        "not a date",
        "chapter 2023-01-01 of the manual",  # anchored regex, not a search
        "20230102",
        "2023-13-45",
        None,
        42,
    ],
)
def test_things_that_are_not_dates_are_not_dates(raw):
    assert parse_iso_instant(raw) is None


def test_a_bare_date_becomes_midnight_utc():
    """Picking the start is the only choice that keeps a date and a
    datetime comparable to each other."""
    assert parse_iso_instant("2023-06-02") == datetime(2023, 6, 2, tzinfo=timezone.utc)
    assert parse_iso_instant("2023-06-02T14:00:00+02:00") == datetime(
        2023, 6, 2, 12, tzinfo=timezone.utc
    )


# --------------------------------------------------------------------------
# The write path
# --------------------------------------------------------------------------

async def test_a_supersede_that_omits_the_range_does_not_destroy_it(client):
    """The bug this chantier started from.

    `fact_kind` and `volatility` were inherited on supersede;
    `temporal_range` was not, and the extraction prompt never asks a
    status-only update to restate it. So "pre-approved" -> "approved"
    silently stopped being about August and became about the message's own
    timestamp -- on the field the temporal questions depend on.
    """
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "mortgage_status",
                        {"state": "pre-approved"},
                        temporal_range=AUGUST,
                    )
                ]
            )
        ],
    )
    await run_worker()
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("mortgage_status", {"state": "approved"}, action="supersede")],
                occurred_at="2026-08-01T10:00:00Z",
            )
        ],
    )
    await run_worker()

    active = [
        fact
        for fact in await facts_for("usr_42", "mortgage_status")
        if fact.status is FactStatus.active
    ]
    assert len(active) == 1
    assert active[0].temporal_range == AUGUST
    assert active[0].observed_at == datetime(2023, 8, 1, tzinfo=timezone.utc)


async def test_observed_at_is_distinct_from_valid_from(client):
    """The whole point of the column, end to end.

    The message is from July 2026; the fact is about June 2023. Storing
    only one of the two makes "when did that happen?" unanswerable.
    """
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("bought_kayak", {"item": "kayak", "date": "2023-06-02"})],
                occurred_at="2026-07-28T10:00:00Z",
            )
        ],
    )
    await run_worker()
    [fact] = await facts_for("usr_42", "bought_kayak")
    assert fact.observed_at == datetime(2023, 6, 2, tzinfo=timezone.utc)
    assert fact.valid_from.year == 2026


async def test_the_packet_carries_the_observed_date_in_one_place(client):
    """A reader had to dig it out of free-form `value` JSON or out of
    `temporal_range`, in two different shapes. Now it is one field, always
    the same one, with the same dual-date rendering as valid_from."""
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("bought_kayak", {"item": "kayak", "date": "2023-06-02"})],
                occurred_at="2026-07-28T10:00:00Z",
            )
        ],
    )
    await run_worker()
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "kayak",
            "budget_tokens": 2000,
            "as_of": "2026-07-29T10:00:00Z",
        },
    )
    assert response.status_code == 200
    [fact] = [
        f for f in response.json()["packet"]["facts"] if f["predicate"] == "bought_kayak"
    ]
    assert fact["observed_at"].startswith("2023-06-02")
    assert fact["observed_at_relative"]
    assert fact["valid_from"].startswith("2026-07-28")


async def test_a_fact_about_no_instant_leaves_the_field_empty(client):
    await capture(client, [make_memory_event([mock_fact("pet", {"kind": "dog"})])])
    await run_worker()
    [fact] = await facts_for("usr_42", "pet")
    assert fact.observed_at is None
