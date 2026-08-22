"""How the unified candidate pool is ordered.

The bug
-------
Until 21 Aug the pool was ordered by

    0.6 * cosine_similarity + 0.25 * ts_rank_cd + 0.15 * exp(-dt/tau)

Two independent defects in that expression, both measured on the LoCoMo
calibration set (gold evidence actually served under a 900-token budget,
identical candidate sets, only the ordering changing):

1. **The three terms do not share a scale.** A cosine similarity lives in
   [-1, 1] and concentrates between 0.6 and 0.9; `ts_rank_cd` is unbounded
   in principle and lands around 0.01-0.1; the recency term is an
   exponential that saturates at 1.0. The declared 0.25 on the lexical
   axis is therefore not the applied one -- the applied one is closer to
   0.25 x 0.05, and it moves with the corpus, the query length and the
   embedder. It is worst exactly where the pool is most heterogeneous:
   a fact's `search_text` is ~50 characters, an episode's index text is up
   to 4 000, so their `ts_rank_cd` distributions have nothing in common
   and the two sources were never comparable.

2. **Recency was a relevance term.** It is not one. Weighted 0.15 against
   a similarity spread of ~0.1 in practice, it systematically pulled the
   most recent candidates to the top of the pool whether or not they had
   anything to do with the query, and they then ate the token budget.

Why not RRF
-----------
Reciprocal Rank Fusion is the standard answer to "these scores are not
comparable": no parameter to fit, robust out of domain. It was the first
thing tried here and it looked like a large win (+25 points) -- that
measurement turned out to be an artifact of the measuring harness (ties on
one axis broken by array position, which correlates with LoCoMo's session
order, i.e. an accidental recency prior baked into the *measurement*, not
the retrieval). Re-measured with proper competition ranking for ties, RRF
loses to the normalized convex combination below by a wide margin.

The reason is structural, not a measurement quirk: RRF gives every axis
equal weight by construction, so it promotes the WEAKEST axis as much as
the strongest -- and one of the three axes here (recency) is noise, not
signal. This is exactly the failure mode Bruch, Gai & Ingber describe
(TOIS 2023, arXiv 2210.11934): a tuned convex combination of normalized
scores beats RRF in domain and out of domain. This module is that convex
combination, with recency demoted to a tie-break rather than folded into
the tuned combination (it is not a relevance axis at all, see above).

What this costs
----------------
Min-max normalization is computed over the candidate pool of ONE query, so
a candidate's score depends on which other candidates came back. Two
consequences, both deliberate:

- The score is comparable WITHIN a response and meaningless ACROSS
  responses. Nothing persists it, and the recall gate deliberately runs on
  the raw cosine distance instead (see `settings.recall_max_distance`) --
  a floor on a pool-relative score would mean nothing at all.
- A pool where every candidate ties on an axis normalizes to all-zeros on
  that axis, which is the correct behaviour: an axis that does not
  separate anything should not move anything.

The weights (0.65 similarity / 0.35 full-text) are deliberately not
fitted tightly -- see app/context/__init__.py's build_context for how they
are applied; this module only supplies the arithmetic, kept separately so
it is testable without a database.
"""

from __future__ import annotations

from collections.abc import Sequence

# Relevance weights, applied to NORMALIZED axes so they mean what they say.
W_SIMILARITY = 0.65
W_FULLTEXT = 0.35


def normalize(values: Sequence[float]) -> list[float]:
    """Min-max onto [0, 1] over this pool. A flat axis becomes all zeros."""
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0] * len(values)
    span = high - low
    return [(value - low) / span for value in values]


def relevance(
    similarity: Sequence[float],
    fulltext: Sequence[float],
    *,
    w_similarity: float = W_SIMILARITY,
    w_fulltext: float = W_FULLTEXT,
) -> list[float]:
    """The ordering key: a convex combination of the two relevance axes.

    Recency is deliberately absent -- it is a tie-break applied by the
    caller (build_context), never folded into relevance itself; see the
    module docstring. Both inputs must hold one value per candidate, in
    the same order; the result follows that order.
    """
    if len(similarity) != len(fulltext):
        raise ValueError("similarity and fulltext must hold one value per candidate")
    normalized_similarity = normalize(similarity)
    normalized_fulltext = normalize(fulltext)
    return [
        w_similarity * s + w_fulltext * f
        for s, f in zip(normalized_similarity, normalized_fulltext)
    ]


def legacy_weighted_sum(
    similarity: Sequence[float],
    fulltext: Sequence[float],
    recency: Sequence[float],
    weights: tuple[float, float, float],
) -> list[float]:
    """The pre-21-Aug ordering, kept behind `settings.ranking`.

    A rollback path for a ranking change landing in production, and a way
    to reproduce old eval numbers exactly -- not an alternative anyone
    should pick on the merits. Delete it once the retrieval bench has run
    in CI for a while: two ranking implementations is debt, and this one
    is deliberately temporary.
    """
    w_similarity, w_fulltext, w_recency = weights
    return [
        w_similarity * s + w_fulltext * f + w_recency * r
        for s, f, r in zip(similarity, fulltext, recency)
    ]
