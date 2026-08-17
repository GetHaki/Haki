"""Real-model bug found by external GTM-driven measurement (17 aout): a
live gpt-4o-mini extraction run on real LoCoMo conversations repeatedly
emitted fact_kind="event" -- not one of fact_kind's own three values
("attribute"|"preference"|"instruction"), but memory_form's. Before this
fix, ExtractedFact's strict `pattern=` constraint on fact_kind/volatility/
memory_form turned a single misclassified field into total data loss: the
whole candidate (predicate, value, evidence_span) was rejected, not just
the field. Since all three fields already have a documented, safe default
for "the provider never set it", an out-of-enum value now falls back to
that same default instead of destroying the candidate -- logged, not
silent, so a real prompt-quality regression stays visible.
"""

import logging

from app.providers.base import ExtractedFact


def _candidate(**overrides) -> ExtractedFact:
    fields = {
        "subject_id": "usr_42",
        "predicate": "event_school_speech",
        "value": {"date": "2023-06-02", "description": "gave a speech"},
        "confidence": 0.9,
        "action": "create",
        "evidence_span": "I gave a speech at my daughter's school",
    }
    fields.update(overrides)
    return ExtractedFact(**fields)


def test_unknown_fact_kind_falls_back_to_default_instead_of_rejecting():
    candidate = _candidate(fact_kind="event")
    assert candidate.fact_kind is None  # None -> consolidator applies "attribute"
    assert candidate.predicate == "event_school_speech"  # the fact itself survives


def test_unknown_volatility_falls_back_to_default_instead_of_rejecting():
    candidate = _candidate(volatility="permanent")  # not a real VOLATILITY_CLASSES value
    assert candidate.volatility is None


def test_unknown_memory_form_falls_back_to_default_instead_of_rejecting():
    candidate = _candidate(memory_form="recurring")  # not "state" or "event"
    assert candidate.memory_form is None


def test_valid_enum_values_still_pass_through_unchanged():
    candidate = _candidate(fact_kind="preference", volatility="volatile", memory_form="event")
    assert candidate.fact_kind == "preference"
    assert candidate.volatility == "volatile"
    assert candidate.memory_form == "event"


def test_unknown_enum_value_is_logged_not_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="haki.providers.base"):
        _candidate(fact_kind="event")
    assert "fact_kind" in caplog.text
    assert "event" in caplog.text
