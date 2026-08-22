"""Entity-affinity re-ranking (app.context._entity_adjusted_score),
recalibrated 22 aout: ENTITY_MATCH_BOOST 1.3 -> 1.0 (removed, it cost 1.3
points on its own -- a name in the query is a FILTER, not a promotion),
ENTITY_MISMATCH_PENALTY 0.3 -> 0.7 (0.3 was a near-exclusion on a score
bounded in [0, 1], not the re-rank the mechanism is meant to be), and the
match rule itself: TOKEN OVERLAP instead of exact string equality, since a
two-word tagged name like "John Smith" can never equal a single capitalised
token from `_query_entities` and was demoted as "somebody else" on every
single query that ever named them.

Unit-level on purpose: `_entity_adjusted_score`/`_query_entities` are pure
functions of (score, value, query_entities) with no database dependency --
exactly what they are for.
"""

from app.context import (
    _ENTITY_TOKEN_RE,
    ENTITY_MATCH_BOOST,
    ENTITY_MISMATCH_PENALTY,
    _entity_adjusted_score,
    _query_entities,
)


def test_a_named_person_is_never_promoted_above_a_more_relevant_fact():
    """The boost is gone (1.0): a query naming a person must never multiply
    a matching fact's score above what it already was."""
    assert ENTITY_MATCH_BOOST == 1.0
    score = 0.8
    assert _entity_adjusted_score(score, {"person": "John"}, {"John"}) == score


def test_a_fact_about_somebody_else_is_demoted_but_not_excluded():
    """A named mismatch costs exactly ENTITY_MISMATCH_PENALTY (0.7) -- a
    re-rank, not a hard exclusion: a fact about someone else can still
    surface if nothing else is remotely relevant."""
    score = 0.8
    adjusted = _entity_adjusted_score(score, {"person": "Diego"}, {"Maria"})
    assert adjusted == score * ENTITY_MISMATCH_PENALTY
    assert adjusted > 0


def test_a_two_word_name_is_recognised_when_the_query_names_them():
    """The bug this fix exists for: the old rule compared the WHOLE tagged
    name against `query_entities`, a set that only ever holds single
    capitalised tokens (that is all `_query_entities` extracts from a
    sentence) -- so a fact tagged "John Smith" could not match ANY query,
    ever. Token overlap fixes it: "Smith" alone, as a query would actually
    tag it, now matches.
    """
    score = 0.8
    assert _entity_adjusted_score(score, {"person": "John Smith"}, {"Smith"}) == score
    assert _entity_adjusted_score(score, {"person": "John Smith"}, {"John"}) == score


def test_the_two_matching_rules_agree_on_every_single_token_name():
    """The common LoCoMo case: every speaker name is one token, and a
    one-token name tokenises to itself -- token overlap and plain equality
    must agree on both a match and a mismatch."""
    for person, query_entities, should_match in [
        ("Maria", {"Maria"}, True),
        ("Diego", {"Diego"}, True),
        ("Maria", {"Diego"}, False),
        ("Fatima", {"Diego", "Fatima"}, True),
    ]:
        tokens = set(_ENTITY_TOKEN_RE.findall(person))
        assert tokens == {person}, "test setup assumption: a single real token"
        overlap_match = bool(tokens & query_entities)
        equality_match = person in query_entities
        assert overlap_match == equality_match == should_match


def test_a_fact_that_names_nobody_is_left_alone():
    score = 0.8
    # No query entities at all: the query named nobody, mechanism inert.
    assert _entity_adjusted_score(score, {"person": "John"}, set()) == score
    # A fact with no "person" key: nothing to compare against.
    assert _entity_adjusted_score(score, {"other": "x"}, {"John"}) == score
    # Not even a dict value.
    assert _entity_adjusted_score(score, "not-a-dict", {"John"}) == score


def test_a_name_too_short_for_the_tokeniser_still_matches_exactly():
    """`_ENTITY_TOKEN_RE` needs a capital followed by two-or-more lowercase
    letters, so a two-letter name like "Al" tokenises to nothing --
    `tokens` is empty and `_entity_adjusted_score` must fall back to plain
    equality rather than treating an empty overlap as a mismatch."""
    assert _ENTITY_TOKEN_RE.findall("Al") == []
    score = 0.8
    assert _entity_adjusted_score(score, {"person": "Al"}, {"Al"}) == score
    assert (
        _entity_adjusted_score(score, {"person": "Al"}, {"Bo"})
        == score * ENTITY_MISMATCH_PENALTY
    )


def test_query_entities_extracts_capitalised_tokens_only():
    """Sanity on the other half of the mechanism: what actually lands in
    query_entities from a real sentence, since every test above passes it
    in by hand."""
    assert _query_entities("What did John Smith say about the trip?") == {
        "John",
        "Smith",
    }
    assert _query_entities("what did she say") == set()
