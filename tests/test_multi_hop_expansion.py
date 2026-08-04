"""Multi-hop expansion in the Context Assembler (sprint 10).

A second, deterministic full-text pass seeded by entities found in the
facts already packed — no LLM call, no extra embedding, bounded and one hop
deep. Targets evidence that never matches the ORIGINAL query's wording but
becomes relevant once a first fact names a shared entity (e.g. two facts
about "Michael" linked only by that name, not by similarity to the
question).
"""

import uuid

from sqlalchemy import func

from app.context import _candidate_entities, _expand_via_entities
from app.db import async_session
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker


def test_candidate_entities_extracts_capitalized_tokens_excluding_query_and_stopwords():
    texts = [
        'wedding_attendee {"name": "Michael", "role": "best man"}',
        'wedding_attendee {"name": "Michael", "role": "best man"}',  # repeated: ranked first
        'trip_destination {"city": "Lisbon", "companion": "Sarah"}',
    ]
    entities = _candidate_entities(texts, exclude={"the", "best"}, limit=2)
    assert entities[0] == "Michael"  # most frequent
    assert "Lisbon" in entities or "Sarah" in entities
    assert "The" not in entities  # stopword, even though capitalized


def test_candidate_entities_respects_the_limit():
    texts = ['a {"x": "Alice"} b {"y": "Bob"} c {"z": "Carla"} d {"w": "Diane"}']
    assert len(_candidate_entities(texts, exclude=set(), limit=2)) == 2


async def test_expand_via_entities_finds_fact_absent_from_original_query(client):
    """Direct test of the DB helper: fact B shares zero words with the
    original query but shares the entity "Michael" with fact A. The
    helper must find fact B via a second full-text pass on that entity,
    without ever revisiting fact A."""
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("wedding_attendee", {"name": "Michael", "role": "best man"}),
                    mock_fact("event_engagement_party", {"person": "Michael", "date": "2023-02-14"}),
                ]
            )
        ],
    )
    await run_worker()

    [fact_a] = await facts_for("usr_42", "wedding_attendee")
    [fact_b] = await facts_for("usr_42", "event_engagement_party")

    async with async_session() as session:
        extra = await _expand_via_entities(
            session,
            project_id="prj_support",
            subject_id="usr_42",
            seed_texts=['wedding_attendee {"name": "Michael", "role": "best man"}'],
            query_words={"who", "is", "the", "best", "man"},
            exclude_ids={fact_a.id},
            now=func.now(),
        )

    assert [row.id for row in extra] == [fact_b.id]


async def test_expand_via_entities_never_revisits_excluded_ids(client):
    """A fact that matches the seeded entity but is already excluded
    (already packed, or blocked by an open conflict) must not reappear."""
    await capture(
        client,
        [make_memory_event([mock_fact("wedding_attendee", {"name": "Michael", "role": "best man"})])],
    )
    await run_worker()
    [fact_a] = await facts_for("usr_42", "wedding_attendee")

    async with async_session() as session:
        extra = await _expand_via_entities(
            session,
            project_id="prj_support",
            subject_id="usr_42",
            seed_texts=['wedding_attendee {"name": "Michael", "role": "best man"}'],
            query_words=set(),
            exclude_ids={fact_a.id},  # the only match is already excluded
            now=func.now(),
        )

    assert extra == []


async def test_expand_via_entities_bounded_to_max_entities(client):
    """More candidate entities than MULTI_HOP_MAX_ENTITIES exist: only the
    top entities (by frequency) seed a search, never an unbounded fan-out."""
    from app.context import MULTI_HOP_MAX_ENTITIES

    facts = [
        mock_fact(f"trip_{i}", {"companion": name})
        for i, name in enumerate(["Alice", "Bob", "Carla", "Diane", "Elena"])
    ]
    await capture(client, [make_memory_event(facts)])
    await run_worker()

    async with async_session() as session:
        extra = await _expand_via_entities(
            session,
            project_id="prj_support",
            subject_id="usr_42",
            # One seed text per name -> 5 distinct single-occurrence entities.
            seed_texts=[f'x {{"n": "{name}"}}' for name in ["Alice", "Bob", "Carla", "Diane", "Elena"]],
            query_words=set(),
            exclude_ids=set(),
            now=func.now(),
        )

    # At most MULTI_HOP_MAX_ENTITIES distinct entities are queried, each
    # matching exactly one fact here, so the result is bounded accordingly.
    assert len(extra) <= MULTI_HOP_MAX_ENTITIES


async def test_build_context_wires_expansion_end_to_end(client):
    """End-to-end wiring check through the real API: with only two facts in
    scope and a generous budget, the trivial top-K union already includes
    both regardless of query wording (small-scope edge case — confirmed
    directly by the isolated _expand_via_entities tests above, which prove
    the mechanism itself against a fact genuinely absent from a query
    match). This test only asserts the weaker, still-meaningful property
    that survives that edge case: multi-hop-linked facts end up served
    together end-to-end, with no crash and a coherent trace."""
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("wedding_attendee", {"name": "Michael", "role": "best man"}),
                    mock_fact("event_engagement_party", {"person": "Michael", "date": "2023-02-14"}),
                ]
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "who is the best man",
            "budget_tokens": 900,
        },
    )
    assert response.status_code == 200
    body = response.json()
    served_predicates = {f["predicate"] for f in body["packet"]["facts"]}
    assert "wedding_attendee" in served_predicates
    assert "event_engagement_party" in served_predicates

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    reason_codes = {d.get("reason_code") for d in trace.json()["decisions"] if d.get("fact_id")}
    assert reason_codes <= {"top_score", "multi_hop_expansion"}
