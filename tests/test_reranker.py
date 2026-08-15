"""Reranker (mechanism F-R, 15 aout Sprint 2): a cross-encoder re-scoring
pass over the top RERANK_TOP_K candidates from the unified pool.

Calls `build_context()` directly rather than through the HTTP client (the
pattern every other context test uses): `reranker` is a provider singleton
cached at module scope in app.providers (same pattern as the embedder),
and enabling it via a settings monkeypatch would leak a provider instance
into that global cache for the rest of the test session. Direct injection
via build_context()'s own `reranker` parameter sidesteps that entirely --
no settings/singleton state touched, nothing to reset.

Isolation technique: force every candidate fact to share one embedding AND
one recency timestamp, and use a query with NO lexical overlap with any
candidate's rendered text -- similarity, full-text and recency all tie at
exactly zero, so the pre-rerank order is an arbitrary (but real, checkable)
tie-break. A `_ScriptedReranker` -- not FakeProvider's word-overlap rerank,
which duplicates full-text's OWN word-overlap signal on the exact same
rendered text and so can never be shown to diverge from it -- then scores
candidates from a fixed lookup table, independent of the query, proving
build_context() actually USES the reranker's scores to reorder the pool
rather than the reranker call being a no-op wired in but never consulted.
"""

from app.consolidator import _search_text
from app.context import _render, build_context
from app.db import async_session
from app.models import Fact
from app.providers.fake import FakeProvider, mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker

PROJECT = "prj_support"


class _ScriptedReranker:
    """Test-only reranker: a fixed document -> score lookup, independent
    of the query. Documents not in the table score 0.0."""

    def __init__(self, scores_by_document: dict[str, float]) -> None:
        self._scores = scores_by_document

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores.get(doc, 0.0) for doc in documents]


async def _collide_embedding(fact: Fact, predicate: str, value: dict) -> None:
    [target_embedding] = await FakeProvider().embed([_search_text(predicate, value)])
    async with async_session() as session:
        row = await session.get(Fact, fact.id)
        row.embedding = target_embedding
        await session.commit()


async def _tie_facts(facts: list[Fact]) -> None:
    """Force every fact to the same embedding and the same recency, so the
    hybrid formula ties exactly (a neutral query then also zeroes the
    full-text axis for all of them -- see NEUTRAL_QUERY)."""
    anchor = facts[0]
    for fact in facts[1:]:
        await _collide_embedding(fact, anchor.predicate, anchor.value)
    async with async_session() as session:
        row_anchor = await session.get(Fact, anchor.id)
        for fact in facts[1:]:
            row = await session.get(Fact, fact.id)
            row.valid_from = row_anchor.valid_from
            row.recorded_from = row_anchor.recorded_from
        await session.commit()


NEUTRAL_QUERY = "zzznope yyynope xxxnope"


async def test_reranker_reorders_a_tied_shortlist(client):
    subject = "usr_rerank_1"
    for predicate, blob in (("topic_first", "blobalpha"), ("topic_second", "blobbeta")):
        await capture(
            client,
            [
                make_memory_event(
                    [mock_fact(predicate, {"blob": blob}, subject_id=subject)],
                    subject_id=subject,
                    occurred_at="2026-07-28T10:00:00Z",
                )
            ],
        )
        await run_worker()

    facts = await facts_for(subject)
    await _tie_facts(facts)
    second = next(f for f in facts if f.predicate == "topic_second")

    reranker = _ScriptedReranker({_render(second.predicate, second.value): 99.0})
    async with async_session() as session:
        packet, _tokens, _trace = await build_context(
            session,
            project_id=PROJECT,
            subject_id=subject,
            query=NEUTRAL_QUERY,
            reranker=reranker,
        )
    order = [f["predicate"] for f in packet["facts"]]
    assert order[0] == "topic_second", (
        f"scripted reranker gave topic_second the only nonzero score, it "
        f"must be promoted to first regardless of the tied hybrid order; "
        f"got {order}"
    )


async def test_reranker_disabled_leaves_the_hybrid_tie_break_alone(client):
    """Same tied setup, no reranker passed: build_context() must not error
    or silently change behavior -- baseline stays whatever the hybrid
    formula's own tie-break produces."""
    subject = "usr_rerank_3"
    for predicate, blob in (("topic_first", "blobalpha"), ("topic_second", "blobbeta")):
        await capture(
            client,
            [
                make_memory_event(
                    [mock_fact(predicate, {"blob": blob}, subject_id=subject)],
                    subject_id=subject,
                    occurred_at="2026-07-28T10:00:00Z",
                )
            ],
        )
        await run_worker()
    facts = await facts_for(subject)
    await _tie_facts(facts)

    async with async_session() as session:
        packet, _tokens, _trace = await build_context(
            session, project_id=PROJECT, subject_id=subject, query=NEUTRAL_QUERY
        )
    assert {f["predicate"] for f in packet["facts"]} == {"topic_first", "topic_second"}


async def test_reranker_leaves_the_tail_beyond_rerank_top_k_untouched(client, monkeypatch):
    """RERANK_TOP_K bounds the cross-encoder pass to the top of the hybrid
    shortlist -- a candidate that never made that shortlist keeps its
    original tie-break position even when the reranker's own score table
    would have strongly preferred it over everything in the reranked
    head."""
    import app.context as context_module

    monkeypatch.setattr(context_module, "RERANK_TOP_K", 2)

    subject = "usr_rerank_2"
    blobs = {"topic_a": "aaa111", "topic_b": "bbb222", "topic_c": "ccc333"}
    for predicate, blob in blobs.items():
        await capture(
            client,
            [
                make_memory_event(
                    [mock_fact(predicate, {"blob": blob}, subject_id=subject)],
                    subject_id=subject,
                    occurred_at="2026-07-28T10:00:00Z",
                )
            ],
        )
        await run_worker()

    facts = await facts_for(subject)
    await _tie_facts(facts)

    async with async_session() as session:
        baseline_packet, _tokens, _trace = await build_context(
            session, project_id=PROJECT, subject_id=subject, query=NEUTRAL_QUERY
        )
    baseline_order = [f["predicate"] for f in baseline_packet["facts"]]
    assert set(baseline_order) == set(blobs)
    tail_predicate = baseline_order[2]
    tail_fact = next(f for f in facts if f.predicate == tail_predicate)

    # The scripted reranker would send this fact straight to the top --
    # but it never gets the chance, since it fell outside RERANK_TOP_K=2.
    reranker = _ScriptedReranker({_render(tail_fact.predicate, tail_fact.value): 999.0})
    async with async_session() as session:
        reranked_packet, _tokens, _trace = await build_context(
            session,
            project_id=PROJECT,
            subject_id=subject,
            query=NEUTRAL_QUERY,
            reranker=reranker,
        )
    reranked_order = [f["predicate"] for f in reranked_packet["facts"]]
    assert reranked_order[-1] == tail_predicate, (
        f"{tail_predicate} was excluded from the RERANK_TOP_K=2 shortlist "
        f"(baseline position 3) -- it must never be promoted even with a "
        f"score of 999; got {reranked_order}"
    )


async def test_reranker_never_runs_when_pool_is_empty(client):
    """No candidates at all -- build_context() with a reranker passed must
    not error out trying to rerank zero documents."""
    async with async_session() as session:
        packet, _tokens, _trace = await build_context(
            session,
            project_id=PROJECT,
            subject_id="usr_rerank_nobody",
            query="anything",
            reranker=FakeProvider(),
        )
    assert packet["facts"] == []
    assert packet["episodes"] == []
