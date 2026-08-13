"""Semantic fallback matching in the Consolidator (sprint-10 fix).

Regression tests for the three named contradiction-leakage cases from the
sprint-10 eval audit (yoga_frequency / bike_count-bikes_owned /
5k_personal_best_time-goal_personal_best_time): an LLM-generated predicate
string is not a reliable join key on natural language, so an update whose
predicate drifts from the original must still be matched against the
existing active fact via embedding similarity, not left to silently coexist
with it.

`FakeProvider.embed` is a pure hash of the input text (see app/providers/
fake.py): it does NOT cluster semantically similar text, so these tests
force the collision explicitly — they set the PRE-EXISTING fact's embedding
to whatever FakeProvider would compute for the SECOND candidate's search
text, simulating what a real embedder would naturally produce for two
restatements of the same concept. This tests the matching mechanism
(`_resolve_existing_fact`) directly and deterministically, independent of
embedding quality.

Also covers the predicate-alias tier (13 aout, 11 aout diagnostic's
proposed order — "canonical key first, alias table second, semantic
fallback last"): a successful semantic match is learned as a
PredicateAlias row, and a pre-registered alias resolves a candidate on its
own, without needing (or getting) any embedding-distance help.
"""

from sqlalchemy import select

import uuid

from app.consolidator import _search_text
from app.db import async_session
from app.models import ConflictSet, Fact, FactStatus, Job, PredicateAlias
from app.providers.fake import FakeProvider, mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker


async def _collide_embedding(fact: Fact, predicate: str, value: dict) -> None:
    """Overwrite `fact`'s stored embedding with the one FakeProvider would
    compute for (predicate, value) — simulates a real embedder finding the
    two search texts semantically close."""
    [target_embedding] = await FakeProvider().embed([_search_text(predicate, value)])
    async with async_session() as session:
        row = await session.get(Fact, fact.id)
        row.embedding = target_embedding
        await session.commit()


async def test_semantic_match_supersedes_despite_drifted_predicate(client):
    """5k_personal_best_time -> goal_personal_best_time (named case): the
    extractor recognizes an update (action=supersede) but names the wrong
    predicate. The semantic fallback must still find and supersede the
    original fact instead of leaving both active."""
    await capture(
        client,
        [make_memory_event([mock_fact("personal_best_5k", {"time": "27:12"})])],
    )
    await run_worker()
    [old_fact] = await facts_for("usr_42", "personal_best_5k")
    await _collide_embedding(old_fact, "goal_personal_best_time", {"time": "25:50"})

    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "goal_personal_best_time",
                        {"time": "25:50"},
                        action="supersede",
                        supersedes_predicate="goal_personal_best_time",
                    )
                ]
            )
        ],
    )
    await run_worker()

    all_facts = await facts_for("usr_42")
    assert len(all_facts) == 2
    old = next(f for f in all_facts if f.id == old_fact.id)
    new = next(f for f in all_facts if f.id != old_fact.id)
    assert old.status is FactStatus.superseded
    assert new.status is FactStatus.active
    assert new.supersedes_id == old.id

    # Only the fresh value is ever served — the stale time never leaks.
    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "personal best 5k time"},
    )
    served = response.json()["packet"]["facts"]
    assert [f["value"] for f in served] == [{"time": "25:50"}]


async def test_semantic_match_opens_conflict_instead_of_silent_duplicate(client):
    """bike_count -> bikes_owned (named case): the extractor emits "create"
    (fails to recognize the update) under a different predicate with a
    different value. Before the fix, both stayed active in parallel and the
    stale count kept winning. The fix must at minimum route this into an
    open conflict (both hidden) instead of silently serving the old value
    as current."""
    await capture(
        client, [make_memory_event([mock_fact("bike_count", {"count": 3})])]
    )
    await run_worker()
    [old_fact] = await facts_for("usr_42", "bike_count")
    await _collide_embedding(old_fact, "bikes_owned", {"count": 4})

    await capture(
        client, [make_memory_event([mock_fact("bikes_owned", {"count": 4})])]
    )
    await run_worker()

    async with async_session() as session:
        conflicts = list((await session.execute(select(ConflictSet))).scalars().all())
    assert len(conflicts) == 1
    assert conflicts[0].status == "open"

    all_facts = await facts_for("usr_42")
    assert len(all_facts) == 2
    # The stale fact is NOT left uniquely active: it is now a member of the
    # open conflict, so it is never served — this is the core fix, whether
    # or not the new candidate reached `active` status.
    assert {f.id for f in all_facts} == set(conflicts[0].fact_ids)

    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "how many bikes"},
    )
    body = response.json()
    assert body["packet"]["facts"] == []
    assert any("open_conflict" in w for w in body["packet"]["warnings"])


async def test_semantic_predicate_variant_same_value_reinforces_without_new_row(client):
    """Chantier toctou-dedup: a candidate whose predicate is a semantic
    variant of an existing fact's predicate, but carries the SAME value, must
    reinforce that fact -- not create a second row that used to sit as an
    orphan `candidate` (the bug this chantier's restructuring of
    `_apply_candidate` fixes)."""
    await capture(
        client, [make_memory_event([mock_fact("favorite_color", {"color": "blue"})])]
    )
    await run_worker()
    [old_fact] = await facts_for("usr_42", "favorite_color")
    await _collide_embedding(old_fact, "preferred_color", {"color": "blue"})

    body = await capture(
        client, [make_memory_event([mock_fact("preferred_color", {"color": "blue"})])]
    )
    await run_worker()

    all_facts = await facts_for("usr_42")
    assert len(all_facts) == 1
    fact = all_facts[0]
    assert fact.id == old_fact.id
    assert fact.status is FactStatus.active
    assert fact.reinforcement_count == 1

    async with async_session() as session:
        job = await session.get(Job, uuid.UUID(body["consolidation_job_id"]))
    assert job.payload["result"]["reinforced"] == 1


async def test_semantic_match_learns_a_predicate_alias(client):
    """13 aout chantier: a successful semantic-fallback match under a
    DIFFERENT predicate string must be recorded as a predicate_aliases row
    -- turning a repeated embedding-distance guess into a persisted,
    deterministic fact about identity, exactly the 11 aout diagnostic's
    proposed second tier ("canonical key first, alias table second,
    semantic fallback last")."""
    await capture(
        client,
        [make_memory_event([mock_fact("personal_best_5k", {"time": "27:12"})])],
    )
    await run_worker()
    [old_fact] = await facts_for("usr_42", "personal_best_5k")
    await _collide_embedding(old_fact, "goal_personal_best_time", {"time": "25:50"})

    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "goal_personal_best_time",
                        {"time": "25:50"},
                        action="supersede",
                        supersedes_predicate="goal_personal_best_time",
                    )
                ]
            )
        ],
    )
    await run_worker()

    async with async_session() as session:
        aliases = list(
            (await session.execute(select(PredicateAlias))).scalars().all()
        )
    assert len(aliases) == 1
    alias = aliases[0]
    assert alias.project_id == "prj_support"
    assert alias.subject_id == "usr_42"
    assert alias.alias_predicate == "goal_personal_best_time"
    assert alias.canonical_predicate == "personal_best_5k"
    assert alias.confidence is not None and 0.0 < alias.confidence <= 1.0


async def test_registered_predicate_alias_resolves_without_embedding_collision(client):
    """The alias tier must work on its own merits, not as a side effect of
    embedding luck: a PredicateAlias registered ahead of time resolves a
    candidate correctly even when its NATURAL (non-colliding) FakeProvider
    embedding is nowhere near the existing fact -- proving step 2
    (registered alias) fires before step 3 (semantic fallback) would ever
    have had a chance to."""
    await capture(
        client,
        [make_memory_event([mock_fact("personal_best_5k", {"time": "27:12"})])],
    )
    await run_worker()
    [old_fact] = await facts_for("usr_42", "personal_best_5k")

    async with async_session() as session:
        session.add(
            PredicateAlias(
                project_id="prj_support",
                subject_id="usr_42",
                alias_predicate="goal_time",
                canonical_predicate="personal_best_5k",
                confidence=0.95,
            )
        )
        await session.commit()

    # No _collide_embedding call: "goal_time" keeps FakeProvider's natural,
    # uncorrelated embedding for this search text -- a real embedder-driven
    # semantic fallback would have no reason to succeed here.
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "goal_time",
                        {"time": "25:50"},
                        action="supersede",
                        supersedes_predicate="goal_time",
                    )
                ]
            )
        ],
    )
    await run_worker()

    all_facts = await facts_for("usr_42")
    assert len(all_facts) == 2
    old = next(f for f in all_facts if f.id == old_fact.id)
    new = next(f for f in all_facts if f.id != old_fact.id)
    assert old.status is FactStatus.superseded
    assert new.status is FactStatus.active
    assert new.supersedes_id == old.id
    assert new.predicate == "goal_time"


async def test_unrelated_facts_are_not_falsely_merged(client):
    """False-positive guard: two genuinely unrelated facts, extracted with
    FakeProvider's natural (non-colliding) hash embeddings, must stay
    independent — the semantic fallback must not over-merge distinct
    concepts just because neither predicate matches an existing one."""
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("invoice_language", {"language": "fr"}),
                    mock_fact("favorite_color", {"color": "blue"}),
                ]
            )
        ],
    )
    await run_worker()

    facts = await facts_for("usr_42")
    assert len(facts) == 2
    assert {f.status for f in facts} == {FactStatus.active}
    async with async_session() as session:
        conflicts = list((await session.execute(select(ConflictSet))).scalars().all())
    assert conflicts == []
