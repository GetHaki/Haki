"""A fact's identity is (subject, predicate, qualifiers) — not the predicate
string on its own.

The first real-scale eval run (LongMemEval knowledge-update) classified
66-80% of the open conflicts it produced as false positives: two facts with
no referential relationship merged because identity was GUESSED from a
string. The semantic fallback matches nearby meanings, not identical facts,
so it paired `lower_quartile` with `upper_quartile` and two different book
authors. Failures ran in both directions — same fact re-extracted under a
different name and never recognized, distinct facts merged — which is what
rules out "the threshold is slightly off" as the explanation: a loose
threshold only ever produces false positives.

The fix has two halves that only work together. The extraction prompt puts
the condition in `qualifiers` instead of burying it in the predicate name
(`wake_up_time` + {"day_type": "weekday"}, not `wake_up_time_weekday`), and
this module refuses to match across differing qualifiers. Half a fix would
be worse than none: moving the qualifier out of the name WITHOUT the guard
makes two facts that used to have distinct predicates collide on one.

`FakeProvider.embed` is a pure hash (see the note in
test_semantic_supersession.py), so the semantic tests here force the
embedding collision explicitly rather than hoping two strings cluster.
"""

from sqlalchemy import select

from app.db import async_session
from app.models import ConflictSet, Fact, FactStatus, Job
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker
from tests.test_semantic_supersession import _collide_embedding


async def _facts(subject_id: str, predicate: str) -> list[Fact]:
    async with async_session() as session:
        rows = await session.execute(
            select(Fact)
            .where(Fact.subject_id == subject_id, Fact.predicate == predicate)
            .order_by(Fact.recorded_from)
        )
        return list(rows.scalars().all())


async def _conflicts(subject_id: str) -> list[ConflictSet]:
    async with async_session() as session:
        rows = await session.execute(
            select(ConflictSet).where(ConflictSet.subject_id == subject_id)
        )
        return list(rows.scalars().all())


async def test_same_predicate_different_qualifiers_are_two_facts(client):
    """The case the eval surfaced, in its post-fix shape: one measure, two
    conditions. Both must stay active — neither superseded, and no conflict
    opened, because they never contradicted each other in the first place."""
    subject = "usr_qualifiers_1"
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "wake_up_time",
                        {"time": "06:30"},
                        subject_id=subject,
                        qualifiers={"day_type": "weekday"},
                    )
                ],
                subject_id=subject,
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
                        "wake_up_time",
                        {"time": "09:15"},
                        subject_id=subject,
                        qualifiers={"day_type": "weekend"},
                    )
                ],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()

    facts = await _facts(subject, "wake_up_time")
    assert len(facts) == 2
    assert all(fact.status is FactStatus.active for fact in facts)
    assert [fact.qualifiers["day_type"] for fact in facts] == ["weekday", "weekend"]
    assert await _conflicts(subject) == []


async def test_an_update_lands_on_the_matching_qualifier_only(client):
    """Two conditions on file, then a new value for ONE of them. It must
    supersede its own reading and leave the other untouched — the whole
    point of keeping qualifiers as a key rather than a label."""
    subject = "usr_qualifiers_2"
    for day_type, time, at in (
        ("weekday", "06:30", "2026-07-28T10:00:00Z"),
        ("weekend", "09:15", "2026-07-28T11:00:00Z"),
    ):
        await capture(
            client,
            [
                make_memory_event(
                    [
                        mock_fact(
                            "wake_up_time",
                            {"time": time},
                            subject_id=subject,
                            qualifiers={"day_type": day_type},
                        )
                    ],
                    subject_id=subject,
                    occurred_at=at,
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
                        "wake_up_time",
                        {"time": "07:00"},
                        subject_id=subject,
                        qualifiers={"day_type": "weekday"},
                        action="supersede",
                        supersedes_predicate="wake_up_time",
                    )
                ],
                subject_id=subject,
                occurred_at="2026-07-28T12:00:00Z",
            )
        ],
    )
    await run_worker()

    by_status = {
        (fact.qualifiers["day_type"], fact.value["time"]): fact.status
        for fact in await _facts(subject, "wake_up_time")
    }
    assert by_status == {
        ("weekday", "06:30"): FactStatus.superseded,
        ("weekday", "07:00"): FactStatus.active,
        ("weekend", "09:15"): FactStatus.active,  # untouched by the update
    }


async def test_same_value_under_different_qualifiers_is_not_a_duplicate(client):
    """Dedup is qualifier-aware too. "8am on weekdays" and "8am at the
    weekend" share a predicate AND a value; collapsing them as a duplicate
    would drop one of them outright, which no later correction could
    recover."""
    subject = "usr_qualifiers_3"
    for day_type, at in (
        ("weekday", "2026-07-28T10:00:00Z"),
        ("weekend", "2026-07-28T11:00:00Z"),
    ):
        await capture(
            client,
            [
                make_memory_event(
                    [
                        mock_fact(
                            "gym_time",
                            {"time": "08:00"},
                            subject_id=subject,
                            qualifiers={"day_type": day_type},
                        )
                    ],
                    subject_id=subject,
                    occurred_at=at,
                )
            ],
        )
        await run_worker()

    facts = await _facts(subject, "gym_time")
    assert len(facts) == 2
    assert all(fact.status is FactStatus.active for fact in facts)


async def test_semantic_fallback_never_crosses_differing_qualifiers(client):
    """The hard guard, on the path that caused the damage. Even with the
    embeddings made IDENTICAL — the most favourable case the fallback can
    ever see — a candidate carrying different qualifiers must not be
    adjudicated against that fact. Distance is evidence about meaning; it
    is not evidence that two readings describe the same thing."""
    subject = "usr_qualifiers_4"
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "commute_duration",
                        {"minutes": 25},
                        subject_id=subject,
                        qualifiers={"mode": "bike"},
                    )
                ],
                subject_id=subject,
            )
        ],
    )
    await run_worker()
    [existing] = await _facts(subject, "commute_duration")
    await _collide_embedding(existing, "trip_duration", {"minutes": 40})

    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "trip_duration",
                        {"minutes": 40},
                        subject_id=subject,
                        qualifiers={"mode": "car"},
                    )
                ],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()

    async with async_session() as session:
        rows = await session.execute(
            select(Fact).where(Fact.subject_id == subject, Fact.status == FactStatus.active)
        )
        active = list(rows.scalars().all())
    assert {fact.qualifiers["mode"] for fact in active} == {"bike", "car"}
    assert await _conflicts(subject) == []


async def test_semantic_fallback_still_matches_when_qualifiers_agree(client):
    """The guard must not cost the sprint-10 fix it sits on top of: with the
    SAME qualifiers, a drifted predicate is still matched and superseded.
    A guard that blocked everything would pass the test above for the wrong
    reason."""
    subject = "usr_qualifiers_5"
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "commute_duration",
                        {"minutes": 25},
                        subject_id=subject,
                        qualifiers={"mode": "bike"},
                    )
                ],
                subject_id=subject,
            )
        ],
    )
    await run_worker()
    [existing] = await _facts(subject, "commute_duration")
    await _collide_embedding(existing, "trip_duration", {"minutes": 40})

    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "trip_duration",
                        {"minutes": 40},
                        subject_id=subject,
                        qualifiers={"mode": "bike"},
                        action="supersede",
                        supersedes_predicate="trip_duration",
                    )
                ],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()

    async with async_session() as session:
        rows = await session.execute(select(Fact).where(Fact.subject_id == subject))
        facts = {fact.predicate: fact.status for fact in rows.scalars().all()}
    assert facts == {
        "commute_duration": FactStatus.superseded,
        "trip_duration": FactStatus.active,
    }


async def test_attribution_is_provenance_and_never_splits_a_fact(client):
    """`attributed_to` is stamped by the consolidator to record WHO said a
    thing, not a condition under which it holds. It must stay out of the
    identity key: the same claim reported by a third party has to meet the
    subject's own fact — reinforcing or contradicting it — instead of
    quietly becoming a second, parallel active fact under the same
    predicate."""
    subject = "usr_qualifiers_6"
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("employer", {"name": "Acme"}, subject_id=subject)],
                subject_id=subject,
            )
        ],
    )
    await run_worker()

    third_party = make_memory_event(
        [mock_fact("employer", {"name": "Acme"}, subject_id=subject)],
        subject_id=subject,
        occurred_at="2026-07-28T11:00:00Z",
    )
    third_party["origin_trust"] = "third_party"
    third_party["actor_id"] = "melanie"
    await capture(client, [third_party])
    await run_worker()

    facts = await _facts(subject, "employer")
    assert len(facts) == 1, "a third-party echo must reinforce, not fork the fact"
    assert facts[0].status is FactStatus.active
    assert facts[0].reinforcement_count >= 1


# -- conflict sets stay pairs --------------------------------------------


async def _open_conflicts(subject_id: str) -> list[ConflictSet]:
    async with async_session() as session:
        rows = await session.execute(
            select(ConflictSet)
            .where(ConflictSet.subject_id == subject_id, ConflictSet.status == "open")
            .order_by(ConflictSet.created_at)
        )
        return list(rows.scalars().all())


async def _last_job_result() -> dict:
    """The result payload of the most recent consolidation job. Each test
    starts on a truncated `jobs` table (see conftest), so "most recent"
    unambiguously means "the one this test just ran"."""
    async with async_session() as session:
        rows = await session.execute(select(Job).order_by(Job.created_at))
        return list(rows.scalars().all())[-1].payload["result"]


async def _assert_value(client, subject: str, value: dict, at: str) -> None:
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("office_city", value, subject_id=subject)],
                subject_id=subject,
                occurred_at=at,
            )
        ],
    )
    await run_worker()


async def test_a_third_value_never_joins_an_existing_conflict(client):
    """A conflict is a disagreement between TWO competing values of one
    fact. Letting a third join turns the set into a magnet: the eval run
    found 3+ member sets in ~24% of conflicts, almost always accumulation
    behind one bad match, with every member blocked from the packet
    together. The third candidate is held in its own set instead — still
    unserved, still visible and resolvable, but the original pair stays
    readable as a pair."""
    subject = "usr_conflict_cap_1"
    await _assert_value(client, subject, {"city": "Dakar"}, "2026-07-28T10:00:00Z")
    await _assert_value(client, subject, {"city": "Abidjan"}, "2026-07-28T11:00:00Z")
    await _assert_value(client, subject, {"city": "Lome"}, "2026-07-28T12:00:00Z")

    conflicts = await _open_conflicts(subject)
    assert [len(conflict.fact_ids) for conflict in conflicts] == [2, 1]
    assert "held" in conflicts[1].reason


async def test_capping_is_counted_separately_from_ordinary_conflicts(client):
    """`conflict_capped` is the number to watch after any change to the
    matching rules — repeated capping on one subject is the signature of a
    bad match upstream, and it must not hide inside the conflicts total."""
    subject = "usr_conflict_cap_2"
    await _assert_value(client, subject, {"city": "Dakar"}, "2026-07-28T10:00:00Z")

    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("office_city", {"city": "Abidjan"}, subject_id=subject)],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()
    pair_result = await _last_job_result()
    assert pair_result["conflicts"] == 1
    assert pair_result["conflict_capped"] == 0

    await _assert_value(client, subject, {"city": "Lome"}, "2026-07-28T12:00:00Z")
    capped_result = await _last_job_result()
    assert capped_result["conflicts"] == 1
    assert capped_result["conflict_capped"] == 1


async def test_a_held_third_value_is_still_kept_out_of_the_packet(client):
    """Held apart must not mean quietly dropped OR quietly served: the
    third value is a real row in its OWN single-member open conflict (a
    held/quarantined candidate, not a two-sided disagreement), and that
    stays hard-blocked (context reason_code 'conflict_open') even after
    13 aout's "stop hiding real conflicts" change — only a genuine 2-member
    conflict is now served, contested. The original pair (Dakar vs Abidjan)
    IS one, so it's served, both sides marked contested; Lome never joined
    a pair (capped), so it stays hidden."""
    subject = "usr_conflict_cap_3"
    await _assert_value(client, subject, {"city": "Dakar"}, "2026-07-28T10:00:00Z")
    await _assert_value(client, subject, {"city": "Abidjan"}, "2026-07-28T11:00:00Z")
    await _assert_value(client, subject, {"city": "Lome"}, "2026-07-28T12:00:00Z")

    async with async_session() as session:
        rows = await session.execute(
            select(Fact).where(Fact.subject_id == subject, Fact.predicate == "office_city")
        )
        values = sorted(fact.value["city"] for fact in rows.scalars().all())
    assert values == ["Abidjan", "Dakar", "Lome"], "nothing may be silently discarded"

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": subject,
            "query": "office_city",
        },
    )
    assert response.status_code == 200
    served = response.json()["packet"]["facts"]
    assert {f["value"]["city"] for f in served} == {"Dakar", "Abidjan"}
    assert all(f["contested"] for f in served)
    assert "Lome" not in {f["value"]["city"] for f in served}
