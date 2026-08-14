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


async def test_unqualified_and_qualified_disagreement_opens_a_conflict(client):
    """13 aout, fix 2 (LongMemEval `ba61f0b9`): the guard above is right to
    keep two non-empty, genuinely distinct conditions apart — but an empty
    qualifier set is not a third condition, it is "no condition narrows
    this". An unconditional fact and a qualified one under the SAME
    predicate, disagreeing on the value, must not silently coexist as two
    active facts the way weekday/weekend correctly does above — this is the
    real case the LongMemEval run missed: "5 women on the team" (no
    qualifier) followed by "6 women on Rachel's team" (team_name
    qualifier), never flagged, both served as if uncontested."""
    subject = "usr_qualifiers_conflict"
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "women_on_team",
                        {"count": 5},
                        subject_id=subject,
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
                        "women_on_team",
                        {"count": 6},
                        subject_id=subject,
                        qualifiers={"team_name": "Rachel's team"},
                    )
                ],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()

    facts = await _facts(subject, "women_on_team")
    assert len(facts) == 2
    new = next(f for f in facts if f.value == {"count": 6})
    assert new.status is FactStatus.candidate

    conflicts = await _conflicts(subject)
    assert len(conflicts) == 1
    assert conflicts[0].status == "open"
    assert set(conflicts[0].fact_ids) == {f.id for f in facts}


async def test_unqualified_and_qualified_same_value_is_not_a_conflict(client):
    """The narrow fix must not fire when the values actually agree — a
    general statement later reaffirmed with more specific context is not a
    disagreement, and stays out of scope for this fix (falls through to the
    ordinary duplicate/reinforcement paths, unaffected by it)."""
    subject = "usr_qualifiers_agree"
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("women_on_team", {"count": 5}, subject_id=subject)],
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
                        "women_on_team",
                        {"count": 5},
                        subject_id=subject,
                        qualifiers={"team_name": "Rachel's team"},
                    )
                ],
                subject_id=subject,
                occurred_at="2026-07-28T11:00:00Z",
            )
        ],
    )
    await run_worker()

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


async def test_a_third_value_reclassifies_the_identity_as_event(client):
    """Mechanism C (15 aout): a conflict is a disagreement between TWO
    competing values of one fact. A third candidate no longer joins the set
    or gets held apart in a single-member quarantine — the eval run found
    3+ member sets in ~24% of conflicts, and the real diagnostic case
    (Maria, research/Diagnostic_Couverture_2026-08-14.md) showed this was
    almost always proof the predicate was never a scalar in the first
    place (5 genuinely distinct volunteering occurrences, capped one by
    one). The whole identity is reclassified `memory_form="event"`, the
    conflict is dissolved, and every member — old pair and newcomer alike —
    is activated."""
    subject = "usr_conflict_cap_1"
    await _assert_value(client, subject, {"city": "Dakar"}, "2026-07-28T10:00:00Z")
    await _assert_value(client, subject, {"city": "Abidjan"}, "2026-07-28T11:00:00Z")
    await _assert_value(client, subject, {"city": "Lome"}, "2026-07-28T12:00:00Z")

    assert await _open_conflicts(subject) == []

    facts = await _facts(subject, "office_city")
    assert len(facts) == 3
    assert all(fact.status is FactStatus.active for fact in facts)
    assert all(fact.memory_form == "event" for fact in facts)


async def test_reclassification_is_counted_separately_from_ordinary_conflicts(client):
    """`reclassified_event` is the number to watch after any change to the
    matching rules — repeated reclassification on one subject is the
    signature of a predicate that should never have been extracted as a
    scalar in the first place, and it must not hide inside the conflicts
    total."""
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
    assert pair_result["reclassified_event"] == 0

    await _assert_value(client, subject, {"city": "Lome"}, "2026-07-28T12:00:00Z")
    reclassified_result = await _last_job_result()
    assert reclassified_result["conflicts"] == 0
    assert reclassified_result["reclassified_event"] == 1


async def test_a_reclassified_third_value_is_served_alongside_the_others(client):
    """Once reclassified to `event`, none of the three values is a
    contradiction of the others any more — all three are ordinary active
    facts and all three are SERVED, uncontested (no open conflict remains
    to mark them as such). This is the entire point of mechanism C: the
    old capped-and-hidden Lome never reached the packet; the reclassified
    one does."""
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
    assert {f["value"]["city"] for f in served} == {"Dakar", "Abidjan", "Lome"}
    assert not any(f["contested"] for f in served)
