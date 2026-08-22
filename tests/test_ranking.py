"""How the candidate pool is ordered: the properties, proven without a database.

The behaviour these pin -- a strong lexical match surviving a mediocre
vector score, and a recent-but-irrelevant candidate NOT displacing a
relevant one -- is the whole reason the ordering changed on 21 Aug. It is
far more legible here than through an end-to-end packet assertion where a
dozen other mechanisms are also in play. tests/test_context.py covers the
wiring; this covers the arithmetic.
"""

import pytest
from sqlalchemy import select

from app.config import settings
from app.context.ranking import (
    W_FULLTEXT,
    W_SIMILARITY,
    legacy_weighted_sum,
    normalize,
    relevance,
)
from app.db import async_session
from app.models import Fact
from app.providers.fake import FakeProvider, mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker


def test_normalize_maps_the_pool_onto_zero_one():
    assert normalize([0.7, 0.9, 0.8]) == pytest.approx([0.0, 1.0, 0.5])


def test_a_flat_axis_normalizes_to_zeros_and_therefore_moves_nothing():
    """An axis that separates nothing must not contribute anything.

    This is what makes it safe to always include the lexical axis: when the
    tsquery matches every candidate equally (or none of them), the axis
    collapses to a constant and the ordering is decided by the other one.
    """
    assert normalize([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]
    assert normalize([]) == []


def test_the_applied_weight_of_the_lexical_axis_equals_its_declared_weight():
    """The defect, stated as an equality.

    Take a pool where similarity is constant, so the lexical axis alone
    decides. Moving one candidate from worst to best on that axis should
    change its score by the axis' weight -- that is what a weight means.

    Raw, it changes it by 0.25 x (max - min) of a `ts_rank_cd`, and
    `ts_rank_cd` spans about 0.01 to 0.1 in practice: roughly 0.02, i.e.
    under a tenth of the declared 0.25. The declared weights were never
    the applied ones, and the applied ones moved with the corpus, the
    query length and the embedder. Normalized, the two coincide exactly.
    """
    similarity = [0.75] * 4
    fulltext = [0.01, 0.04, 0.07, 0.10]  # a realistic ts_rank_cd spread

    scores = relevance(similarity, fulltext)
    assert scores[-1] - scores[0] == pytest.approx(W_FULLTEXT)

    raw = legacy_weighted_sum(similarity, fulltext, [0.0] * 4, (0.6, 0.25, 0.15))
    applied = raw[-1] - raw[0]
    assert applied == pytest.approx(0.0225)
    assert applied < W_FULLTEXT / 10


def test_the_lexical_axis_cannot_outweigh_the_vector_axis_on_its_own():
    """Normalization gives the lexical axis its declared weight, not more.

    A candidate that is best on full-text and worst on similarity must not
    automatically win: it carries W_FULLTEXT, and W_SIMILARITY is larger.
    """
    scores = relevance([0.0, 1.0], [1.0, 0.0])
    assert scores == pytest.approx([W_FULLTEXT, W_SIMILARITY])
    assert scores[1] > scores[0]


def test_relevance_ignores_recency_entirely():
    """Recency is a tie-break in build_context, never a relevance term.

    Weighted 0.15 against a similarity spread of ~0.1 in practice, it used
    to pull the most recent candidates to the top of the pool whatever the
    query -- and they then ate the token budget. The signature is the
    guarantee: `relevance` cannot see it.
    """
    with pytest.raises(TypeError):
        relevance([0.5], [0.5], [1.0])  # type: ignore[call-arg]


def test_relevance_rejects_mismatched_axis_lengths():
    with pytest.raises(ValueError):
        relevance([0.1, 0.2], [0.3])


async def _flatten_embeddings(*predicates: str) -> None:
    """Give several facts the SAME stored embedding.

    FakeProvider derives embeddings from a sha256, so similarities are
    noise rather than signal; flattening them removes that noise from the
    experiment and leaves exactly the two things under test — the lexical
    axis and recency — to decide the ordering.
    """
    [shared] = await FakeProvider().embed(["a fixed vector for this test"])
    async with async_session() as session:
        rows = (
            (await session.execute(select(Fact).where(Fact.predicate.in_(predicates))))
            .scalars()
            .all()
        )
        for row in rows:
            row.embedding = shared
        await session.commit()


async def _rank_two_facts(client, monkeypatch, mode: str) -> list[str]:
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("sport_equipment", {"item": "kayak"})],
                occurred_at="2026-01-05T10:00:00Z",
            ),
            make_memory_event(
                [mock_fact("favourite_colour", {"colour": "vermillion"})],
                occurred_at="2026-08-01T10:00:00Z",
            ),
        ],
    )
    await run_worker()
    await _flatten_embeddings("sport_equipment", "favourite_colour")

    monkeypatch.setattr(settings, "ranking", mode)
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "which kayak does he own?",
            "budget_tokens": 2000,
        },
    )
    assert response.status_code == 200
    return [fact["predicate"] for fact in response.json()["packet"]["facts"]]


async def test_a_recent_but_irrelevant_fact_no_longer_outranks_a_relevant_one(
    client, monkeypatch
):
    """The recency defect, end to end, with the vector axis neutralised.

    Two facts of equal semantic distance to the query. One matches it
    lexically and is months old; the other has nothing to do with it and is
    the most recent thing in the scope.

    Under the old sum, recency (0.15) beat the lexical term (0.25 x a
    ts_rank_cd of ~0.02) and the fresh irrelevant fact came first. Recency
    is now a tie-break: it decides between candidates the query cannot
    separate, and never displaces one it can.
    """
    ranked = await _rank_two_facts(client, monkeypatch, "normalized")
    assert ranked[0] == "sport_equipment", (
        f"the recent irrelevant fact outranked the one the query matches: {ranked}"
    )


async def test_the_legacy_ranking_still_shows_the_old_behaviour(client, monkeypatch):
    """Pins what the rollback path actually rolls back to.

    If this ever starts agreeing with the test above, the two modes have
    converged and `settings.ranking` has stopped being a rollback — delete
    it rather than keep a switch that switches nothing.
    """
    ranked = await _rank_two_facts(client, monkeypatch, "legacy")
    assert ranked[0] == "favourite_colour"
