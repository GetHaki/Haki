"""Memory form (mechanism C, 15 aout): a fact identity is either a "state"
(a scalar attribute that changes over time — the default) or an "event" (an
accumulating occurrence — volunteered somewhere, tried a restaurant — where
a new mention is never a contradiction of the previous ones).

The reclassification path (a 3rd competing "state" value dissolving its
conflict into "event") is covered by
tests/test_fact_identity_qualifiers.py::test_a_third_value_reclassifies_the_identity_as_event
and its two neighbors. This module covers the form itself: an extractor
that declares "event" up front, and the sticky/inherited nature of the
field once an identity has one.
"""

from sqlalchemy import select

from app.db import async_session
from app.models import ConflictSet, Fact, FactStatus
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker


async def _facts(subject_id: str, predicate: str) -> list[Fact]:
    async with async_session() as session:
        rows = await session.execute(
            select(Fact)
            .where(Fact.subject_id == subject_id, Fact.predicate == predicate)
            .order_by(Fact.recorded_from)
        )
        return list(rows.scalars().all())


async def _open_conflicts(subject_id: str) -> list[ConflictSet]:
    async with async_session() as session:
        rows = await session.execute(
            select(ConflictSet)
            .where(ConflictSet.subject_id == subject_id, ConflictSet.status == "open")
        )
        return list(rows.scalars().all())


async def test_event_form_declared_upfront_never_opens_a_conflict(client):
    """An extractor that already knows a predicate is an accumulating
    occurrence (memory_form="event") gets the fast path immediately —
    no waiting for a 3rd value to trigger reclassification, no conflict
    ever opened, however many differing values arrive."""
    subject = "usr_event_form_1"
    for city, at in [
        ("Dakar", "2026-07-28T10:00:00Z"),
        ("Abidjan", "2026-07-28T11:00:00Z"),
        ("Lome", "2026-07-28T12:00:00Z"),
        ("Accra", "2026-07-28T13:00:00Z"),
    ]:
        await capture(
            client,
            [
                make_memory_event(
                    [
                        mock_fact(
                            "city_visited",
                            {"city": city},
                            subject_id=subject,
                            memory_form="event",
                        )
                    ],
                    subject_id=subject,
                    occurred_at=at,
                )
            ],
        )
        await run_worker()

    assert await _open_conflicts(subject) == []
    facts = await _facts(subject, "city_visited")
    assert len(facts) == 4
    assert all(fact.status is FactStatus.active for fact in facts)
    assert all(fact.memory_form == "event" for fact in facts)


async def test_event_form_never_fuses_two_occurrences_with_the_same_value(client):
    """Regression guard (bug found by code review, 16 aout): _find_duplicate
    ran before memory_form was computed for the candidate and had no
    visibility into it, so a second "event" candidate with the exact same
    value as the first (e.g. the same volunteering activity done twice) was
    silently reinforced onto the first fact instead of becoming its own
    active fact — defeating the one guarantee memory_form="event" exists to
    make ("a new mention is never a contradiction OR a duplicate of the
    previous ones")."""
    subject = "usr_event_form_dup"
    for at in ["2026-07-28T10:00:00Z", "2026-08-04T10:00:00Z"]:
        await capture(
            client,
            [
                make_memory_event(
                    [
                        mock_fact(
                            "volunteered_at",
                            {"place": "food bank"},
                            subject_id=subject,
                            memory_form="event",
                        )
                    ],
                    subject_id=subject,
                    occurred_at=at,
                )
            ],
        )
        await run_worker()

    assert await _open_conflicts(subject) == []
    facts = await _facts(subject, "volunteered_at")
    assert len(facts) == 2
    assert all(fact.status is FactStatus.active for fact in facts)
    assert all(fact.reinforcement_count == 0 for fact in facts)


async def test_active_fact_lookup_is_deterministic_with_multiple_actives(client):
    """Regression guard (bug found by code review, 16 aout): _active_fact
    had no ORDER BY, so once memory_form="event" allows several
    simultaneously-active facts under one identity, which one it returned
    depended on arbitrary DB row order — and the M8 trust/quarantine check
    in _apply_candidate adjudicates against exactly that arbitrary pick.
    Now pinned to the most recently recorded one, repeatably."""
    subject = "usr_event_form_order"
    for at in ["2026-07-28T10:00:00Z", "2026-08-04T10:00:00Z", "2026-08-11T10:00:00Z"]:
        await capture(
            client,
            [
                make_memory_event(
                    [
                        mock_fact(
                            "volunteered_at",
                            {"place": "food bank", "session": at},
                            subject_id=subject,
                            memory_form="event",
                        )
                    ],
                    subject_id=subject,
                    occurred_at=at,
                )
            ],
        )
        await run_worker()

    facts = await _facts(subject, "volunteered_at")
    assert len(facts) == 3
    most_recent = max(facts, key=lambda f: f.recorded_from)

    from app.consolidator import _active_fact

    async with async_session() as session:
        for _ in range(3):
            found = await _active_fact(
                session,
                project_id=most_recent.project_id,
                subject_id=subject,
                predicate="volunteered_at",
                qualifiers=None,
            )
            assert found is not None
            assert found.id == most_recent.id


async def test_memory_form_is_inherited_from_the_matched_identity_not_the_new_candidate(client):
    """Once an identity has a settled memory_form, a single later
    candidate's own guess must never flip it back — extraction is
    non-deterministic, so a candidate that forgets to restate "event" (or
    even actively guesses "state") on a supersede-free create still
    inherits the identity's real form. The only sanctioned way to move
    "state" -> "event" is the conflict-overflow reclassification, never a
    silent per-candidate toggle."""
    subject = "usr_event_form_2"
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "city_visited",
                        {"city": "Dakar"},
                        subject_id=subject,
                        memory_form="event",
                    )
                ],
                subject_id=subject,
                occurred_at="2026-07-28T10:00:00Z",
            )
        ],
    )
    await run_worker()

    # Second candidate under the same identity omits memory_form entirely
    # (as most candidates do — only the extractor's first mention of a
    # predicate typically restates its form).
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("city_visited", {"city": "Abidjan"}, subject_id=subject)],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()

    assert await _open_conflicts(subject) == []
    facts = await _facts(subject, "city_visited")
    assert len(facts) == 2
    assert all(fact.status is FactStatus.active for fact in facts)
    assert all(fact.memory_form == "event" for fact in facts)


async def test_state_form_is_the_default_and_still_conflicts_on_two_values(client):
    """Regression guard: an ordinary 2-value contradiction under the
    default memory_form ("state") still opens a genuine open conflict —
    mechanism C only changes behavior at the 3rd competing value, never
    the 2nd."""
    subject = "usr_event_form_3"
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("favorite_color", {"color": "blue"}, subject_id=subject)],
                subject_id=subject,
                occurred_at="2026-07-28T10:00:00Z",
            )
        ],
    )
    await run_worker()
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("favorite_color", {"color": "green"}, subject_id=subject)],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()

    conflicts = await _open_conflicts(subject)
    assert len(conflicts) == 1
    facts = await _facts(subject, "favorite_color")
    assert len(facts) == 2
    assert all(fact.memory_form == "state" for fact in facts)
