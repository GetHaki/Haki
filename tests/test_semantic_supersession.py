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
"""

from sqlalchemy import select

import uuid

from app.consolidator import _search_text
from app.db import async_session
from app.models import ConflictSet, Fact, FactStatus, Job
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
