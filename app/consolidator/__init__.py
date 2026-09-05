"""Memory Consolidator (PRD semaines 3-4).

Takes pending `consolidate` jobs, extracts memory candidates through the
configured LLM provider, validates them, and applies the fact lifecycle:

- same value as an existing fact for the same subject+predicate -> duplicate
  (replayed event, no new row) or reinforcement (a NEW event re-asserting
  the same value strengthens the existing active fact instead);
- action "supersede" -> the current active fact becomes `superseded` (Ledger
  transition), the new fact becomes `active` with `supersedes_id`;
- action "create" with a different value than an active fact of the same
  identity, memory_form "state" -> both facts enter an OPEN conflict set;
  the new fact stays `candidate` and is never served while the conflict is
  open. A conflict set holds at most CONFLICT_SET_MAX_MEMBERS facts: a
  third competing value no longer joins or gets held apart -- it
  reclassifies the WHOLE identity to memory_form "event" (mechanism C,
  15 aout), dissolves the conflict, and activates every member, since a
  predicate that keeps producing new competing values was never a scalar
  to begin with;
- action "create" on an identity whose memory_form is "event" -> never a
  contradiction, whatever else is already active under it; goes straight
  to `active`, exactly like a brand-new identity would;
- action "reject" -> the candidate is counted (`rejected`,
  `rejected_with_reason`) and logged, NEVER becomes a Fact;
- otherwise -> candidate promoted to `active`.

Write gate (M1 — "porte d'ecriture"): a candidate never reaches the ledger
on trust alone. The extraction prompt itself asks for a verbatim
`evidence_span` and an explicit `reject_reason` taxonomy for anything that
is not a genuine, sourced, novel fact (see app.providers.base.
REJECT_REASONS and the WRITE GATE section of the extraction prompt). On top
of that, this module enforces two rules the provider cannot be trusted to
self-police on prompt compliance alone: a candidate whose embedding closely
matches a fact already SERVED to the same subject in a recent context
packet (context_traces) is rejected as `echo_of_context` in post-validation,
whatever action the provider assigned — see `_echo_reject_reason`. This is
what stops a served fact from being echoed back by the agent, re-extracted,
and re-stored without bound. Separately, a candidate whose own text matches
a small set of high-precision "instruction addressed to the agent" trigger
phrases is rejected as `imperative_directive` regardless of the provider's
verdict — see `_imperative_directive_reason` for that check and its
documented (narrow, not exhaustive) scope.

"Same fact" above is resolved by `_resolve_existing_fact`, on the key
(subject, predicate, identity qualifiers): exact predicate match first,
then a registered predicate_aliases synonym for this subject (13 aout —
see PredicateAlias), then a semantic fallback (cosine distance on the
already-computed fact embedding) only if both miss. An LLM-generated
predicate is not a reliable join key on natural language on its own — this
is the write-time adjudication step, decoupled from extraction, that keeps
a same-concept update from silently coexisting with the fact it was meant
to replace. Every semantic-fallback match under a different predicate
string is learned as a new alias, so the same subject's next occurrence of
that synonym pair resolves deterministically instead of re-rolling the
embedding-distance dice.

Qualifiers are part of that key, and the guard is hard: different
qualifiers are never the same fact, however close the embeddings are. The
first real-scale eval run traced 66-80% of open conflicts to the opposite
behavior — the fallback taking the nearest active fact on distance alone,
pairing `lower_quartile` with `upper_quartile`. The extraction prompt is
the other half of the same fix: a qualifier belongs in the `qualifiers`
field, never buried in the predicate name.

That guard's blind spot (13 aout, LongMemEval `ba61f0b9`): it also hides a
genuine disagreement between an UNCONDITIONAL fact (empty qualifiers) and a
QUALIFIED one under the same predicate — an empty qualifier set is not a
third condition to keep apart from the others, so `_resolve_existing_fact`
returning None there left two contradicting active facts with nobody ever
comparing them. `_find_qualifier_ambiguous_active_fact` is the narrow,
exact-predicate-only second net for exactly that shape, feeding its match
into the ordinary create/contradiction path so it becomes a normal, served
`contested` conflict instead of two silently-coexisting active facts.

Guarantees:
- a provider/DB exception fails the job (`failed`, error in payload) and
  NEVER deletes or alters the source events; the job stays replayable —
  failed jobs are picked up again on the next worker run;
- idempotence: dedup is content-based (same subject_id + predicate +
  canonical value among non-deleted facts), so processing the same job
  twice never creates duplicate facts and a NEW event re-asserting an
  active fact reinforces it (counter + date) instead of creating a row.
  Concurrency: the whole write phase is serialized per (project_id,
  subject_id) with a transaction-scoped advisory lock, and a partial
  unique index (migration 0015, widened to include identity qualifiers by
  0019) makes two ACTIVE facts with the same identity impossible at the DB
  level even if some future write path forgets the lock — legitimate
  re-assertions are absorbed as duplicates/reinforcements BEFORE
  insertion, so the index only ever fires on a genuine race.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import ValidationError
from sqlalchemy import Text, and_, cast, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app import metrics
from app.ledger.core import acquire_subject_write_lock, create_fact, transition_fact_status
from app.context import episode_text
from app.consolidator.temporal import _dates_in, observed_at_of, resolve_relative_range
from app.context.chunking import chunk_payload
from app.models import (
    ConflictSet,
    ContextTrace,
    EpisodeChunk,
    Event,
    Fact,
    FactStatus,
    Job,
    JobStatus,
    PredicateAlias,
)
from app.models.event import ORIGIN_TRUST_RANK
from app.providers import (
    REJECT_REASONS,
    Embedder,
    ExtractedFact,
    Extractor,
    get_embedder,
    get_extractor,
)

logger = logging.getLogger(__name__)

# Semantic fallback for supersession/dedup matching (sprint 10 fix): an
# LLM-generated predicate string is not a reliable join key on natural
# language — "bike_count" vs "bikes_owned" for the same concept never match
# by strict equality, which is exactly why a stale fact kept winning silently
# (measured contradiction leakage 87.5% on the sprint-10 eval sample). Below
# this cosine distance, an active fact is treated as the same concept as the
# candidate even when its predicate string differs.
#
# Calibrated empirically against the REAL local embedder (fastembed,
# paraphrase-multilingual-MiniLM-L12-v2) on the named sprint-10 regression
# pairs, not guessed from a literature prior — see
# scripts/check_semantic_threshold.py. Measured cosine distance: same-concept
# pairs (bike_count/bikes_owned, personal_best_5k/goal_personal_best_time,
# favorite_color/preferred_color) cluster at 0.09-0.23; unrelated pairs
# (including lexically-similar ones like bike_count/car_count) stay above
# 0.35. 0.28 sits in that gap with margin on both sides.
SEMANTIC_MATCH_MAX_DISTANCE = 0.28

# A conflict is a disagreement between two competing values of ONE fact.
# Nothing in the product's model needs a three-way one, and the eval run
# that motivated the identity fix above found 3+ member sets in ~24% of
# conflicts, almost always as accumulation rather than genuine multi-way
# disagreement: one bad match turns a set into a magnet that every later
# loosely-matching candidate joins, and every member is blocked from the
# context packet together.
CONFLICT_SET_MAX_MEMBERS = 2

# existing_facts sent to the extractor (extraction-time context, distinct
# from the consolidator's own identity matching below): capped and
# semantically filtered once a subject has accumulated more active facts
# than this. Found via a real eval re-ingestion (12 aout): with 85 active
# facts and a topic-switching session in view, the real extractor invented a
# NEW predicate ("marathon_goal_time") instead of reusing the existing
# "personal_best_5k" sitting right there in the list — a needle-in-a-
# haystack failure at extraction time, not a matching bug (_resolve_
# existing_fact already has a semantic fallback for this at APPLICATION
# time; this is the same underlying problem one step earlier, where the
# extractor first decides the predicate name and the create/supersede
# action). Below the threshold, every active fact is sent unfiltered — the
# common case for a new or lightly-used subject, unaffected by this at all.
EXISTING_FACTS_FILTER_THRESHOLD = 40
EXISTING_FACTS_TOP_K = 40


# Qualifier keys that describe WHERE a fact came from rather than WHEN or
# under WHAT CONDITIONS it holds. They are stamped by this module (M8
# attribution), not declared by the extractor, and they must stay out of
# the identity key: two people asserting the same thing is one fact,
# reinforced — not two competing active facts.
_PROVENANCE_QUALIFIERS = frozenset({"attributed_to"})


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _identity_qualifiers(qualifiers: dict[str, Any] | None) -> dict[str, Any]:
    """The part of `qualifiers` that takes part in a fact's identity.

    A fact is identified by (subject, predicate, qualifiers) — not by the
    predicate string alone. The eval run that motivated this found the
    qualifier hidden INSIDE the predicate name instead
    (`wake_up_time_weekday` vs `wake_up_time_weekend`), which left nothing
    downstream able to tell two conditions of the same measure apart: they
    read as two unrelated strings to exact matching, and as the same
    concept to the semantic fallback. Mirrors the DB index of migration
    0019 — the two must agree, or the write path and its backstop disagree
    about what "the same fact" means.
    """
    return {
        key: value
        for key, value in (qualifiers or {}).items()
        if key not in _PROVENANCE_QUALIFIERS
    }


def _identity_qualifiers_match(fact: Fact, qualifiers: dict[str, Any] | None) -> bool:
    return _canonical(_identity_qualifiers(fact.qualifiers)) == _canonical(
        _identity_qualifiers(qualifiers)
    )


def _identity_qualifiers_sql():
    """SQL mirror of `_identity_qualifiers`, for filtering inside a query.

    Must stay in step with the expression indexed by migration 0019 — the
    write path and its DB backstop have to agree on what identifies a fact,
    or one of them starts rejecting rows the other considers distinct.
    """
    expression = Fact.qualifiers
    for key in sorted(_PROVENANCE_QUALIFIERS):
        expression = expression.op("-")(literal(key, Text))
    return expression


def _search_text(predicate: str, value: dict[str, Any]) -> str:
    return f"{predicate} {_canonical(value)}"


# Deterministic post-validation net for "imperative_directive" (added after
# M1, alongside the anti-echo net below). The extraction prompt already asks
# the provider to self-classify an instruction-addressed-to-the-agent as
# action="reject"/reject_reason="imperative_directive" (WRITE GATE section,
# app/providers/openai.py _SYSTEM_PROMPT), but prompt compliance alone is not
# guaranteed — the very lesson this sprint drew from the supersede
# value-merge bug, which is why the anti-echo rule below also does not rely
# on the provider alone. This is a SECOND, independent check that catches a
# small set of high-precision trigger phrases regardless of what action the
# provider assigned.
#
# Deliberately narrow, NOT a general prompt-injection classifier — Haki does
# not yet ingest untrusted third-party documents, only the agent's own
# conversation events (see module docstring), so a full defense-in-depth
# system would be over-engineering for the current threat model. Documented
# residual risk, left for future refinement rather than pretended away:
#   - false negatives: any phrasing not in this list slips through
#     untouched (a different verb, a translation, a synonym, indirection via
#     a quoted third party) — this is a last-resort net, not a filter that
#     catches every directive;
#   - false positives: a candidate that happens to quote one of these exact
#     phrases as part of a genuine fact (e.g. the subject reporting "my
#     manager's email said to always ignore previous instructions from
#     IT") would be wrongly rejected. Accepted for now because an
#     occasional over-rejection is far cheaper than a directive silently
#     entering the ledger and being replayed into a future context packet
#     as if it were memory.
# Both English and French patterns are included since the extraction prompt
# and Haki's own docs are bilingual.
_IMPERATIVE_DIRECTIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignor[ez]\s+(all\s+|toutes?\s+les\s+)?(the\s+)?(previous|prior|above|pr[eé]c[eé]dentes?)\s+instructions?",
        r"disregard\s+(your|the|all)\s+(previous\s+)?instructions?",
        r"consid[eè]re[sz]?\s+toujours\b",
        r"r[eé]ponds?\s+toujours\s+en\s+donnant\s+la\s+priorit[eé]",
        r"n'?oublie\s+jamais\s+de\s+toujours\b",
        r"you\s+must\s+always\b",
        r"always\s+(treat|trust|obey|prioriti[sz]e)\b",
        r"from\s+now\s+on,?\s+(always|never|you\s+must)\b",
    )
)


def _imperative_directive_reason(candidate: ExtractedFact) -> str | None:
    """Second, deterministic pass for "imperative_directive": scans the
    candidate's own text (predicate, value, qualifiers, evidence_span) for a
    small set of high-precision command-to-the-agent phrases. Independent of
    whatever action/reject_reason the provider itself assigned — see the
    module-level note above for what this does and does not catch."""
    text = " ".join(
        [
            candidate.predicate,
            _canonical(candidate.value),
            _canonical(candidate.qualifiers),
            candidate.evidence_span or "",
        ]
    )
    if any(pattern.search(text) for pattern in _IMPERATIVE_DIRECTIVE_PATTERNS):
        return "imperative_directive"
    return None


def _untrusted_instruction_reason(event: Event, candidate: ExtractedFact) -> str | None:
    """Deterministic authority check (M8), independent of prompt
    compliance: a durable instruction (fact_kind="instruction") may only
    be born from the subject's own channel or the agent's own tooling
    (origin rank >= semi_trusted). Ingested content and third parties have
    no authority to steer future behavior — the write-time blind spot
    compositional/dormant attacks exploit (a legitimate-looking
    "preference" planted in a document, triggered turns later).
    Complements _imperative_directive_reason: that one catches orders
    aimed AT the agent whatever the origin; this one catches instructions
    whose ORIGIN disqualifies them even when perfectly phrased."""
    if candidate.fact_kind != "instruction":
        return None
    rank = ORIGIN_TRUST_RANK.get(event.origin_trust or "trusted", 0)
    if rank >= ORIGIN_TRUST_RANK["semi_trusted"]:
        return None
    return "untrusted_instruction"


async def _active_fact(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    predicate: str,
    qualifiers: dict[str, Any] | None,
) -> Fact | None:
    stmt = (
        select(Fact)
        .where(
            Fact.project_id == project_id,
            Fact.subject_id == subject_id,
            Fact.predicate == predicate,
            Fact.status == FactStatus.active,
        )
        .order_by(Fact.recorded_from.desc())
    )
    for fact in (await session.execute(stmt)).scalars().all():
        if _identity_qualifiers_match(fact, qualifiers):
            return fact
    return None


async def _find_duplicate(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    predicate: str,
    value: dict[str, Any],
    qualifiers: dict[str, Any] | None,
) -> Fact | None:
    """Content-based dedup: the already-memorized (non-deleted) fact with
    this exact predicate, the same identity qualifiers and an identical
    canonical value, or None. Prefers an ACTIVE match so the caller can
    reinforce it; a non-active match (e.g. a superseded value being
    re-asserted) is still returned and counted as a plain duplicate,
    exactly as before.

    Qualifiers matter here as much as in supersession: "8am on weekdays"
    and "8am at the weekend" carry the same predicate and the same value
    and are not the same fact, so collapsing them as duplicates would lose
    one of them outright.

    memory_form="event" is excluded from matching entirely (mechanism C,
    15 aout): an event-form identity is "never fused" by definition -- two
    occurrences that happen to share the exact same value are still two
    occurrences (e.g. the same volunteering activity done twice), not one
    reinforced fact. Without this guard, this function ran BEFORE
    memory_form is computed for the candidate in _apply_candidate and had
    no visibility into it at all, so a second identical-value "event"
    candidate was silently reinforced onto the first instead of becoming
    its own active fact -- defeating the one guarantee memory_form="event"
    exists to make."""
    stmt = select(Fact).where(
        Fact.project_id == project_id,
        Fact.subject_id == subject_id,
        Fact.predicate == predicate,
        Fact.status != FactStatus.deleted,
        Fact.memory_form != "event",
    )
    canonical = _canonical(value)
    matches = [
        fact
        for fact in (await session.execute(stmt)).scalars().all()
        if _canonical(fact.value) == canonical
        and _identity_qualifiers_match(fact, qualifiers)
    ]
    if not matches:
        return None
    active = [fact for fact in matches if fact.status is FactStatus.active]
    return (active or matches)[0]


async def _resolve_existing_fact(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    predicate: str,
    qualifiers: dict[str, Any] | None,
    embedding: list[float],
) -> Fact | None:
    """Find the active fact a candidate should be adjudicated against.

    This is the "adjudicate against the existing" step, decoupled from
    extraction (Control-Plane Placement, arXiv:2606.15903), in the exact
    order the 11 aout diagnostic specified — "canonical key first, alias
    table second, semantic fallback last": (1) exact predicate match (fast
    path — extraction was lexically consistent, the common case); (2) a
    registered predicate_aliases entry for this exact subject, a
    deterministic memory of a synonym pair already confirmed once (see
    PredicateAlias); (3) only if both miss, a semantic match among the
    subject's active facts on the candidate's already-computed embedding —
    no extra LLM call. This closes the gap where the extractor recognizes
    an update but mints a slightly different predicate string than the one
    already on file (e.g. "personal_best_5k" vs "goal_personal_best_time"),
    which previously left both facts active in parallel with the stale one
    still served as current. A successful semantic match (step 3) is
    recorded as a new alias so the SAME pair resolves deterministically at
    step 2 next time, instead of re-rolling the embedding-distance dice on
    every future event for this subject.

    ALL THREE paths are gated on identity qualifiers, and the gate is
    hard: different qualifiers are never the same fact, however close the
    embeddings are (or however confidently an alias was learned — an alias
    is a predicate-NAME synonym, not a qualifier override). Without it the
    semantic fallback takes the single nearest active fact on cosine
    distance alone, which is how the eval run produced conflicts between
    `lower_quartile` and `upper_quartile`, or between two different book
    authors — near neighbours in meaning, not the same fact. The exact and
    alias paths need it just as much: once qualifiers live in their own
    field rather than inside the predicate name, "wake up time on
    weekdays" and "wake up time at the weekend" share a predicate, and
    matching on the string alone (or an alias of it) would merge them.
    """
    exact = await _active_fact(
        session,
        project_id=project_id,
        subject_id=subject_id,
        predicate=predicate,
        qualifiers=qualifiers,
    )
    if exact is not None:
        return exact

    alias = (
        await session.execute(
            select(PredicateAlias.canonical_predicate).where(
                PredicateAlias.project_id == project_id,
                PredicateAlias.subject_id == subject_id,
                PredicateAlias.alias_predicate == predicate,
            )
        )
    ).scalar_one_or_none()
    if alias is not None:
        aliased = await _active_fact(
            session,
            project_id=project_id,
            subject_id=subject_id,
            predicate=alias,
            qualifiers=qualifiers,
        )
        if aliased is not None:
            return aliased

    stmt = (
        select(Fact, Fact.embedding.cosine_distance(embedding).label("distance"))
        .where(
            Fact.project_id == project_id,
            Fact.subject_id == subject_id,
            Fact.status == FactStatus.active,
            Fact.embedding.is_not(None),
            # Filtered in SQL, not after the fact: ordering by distance and
            # then discarding an incompatible winner would hide a
            # compatible second-nearest behind it. Bound as a dict, not as
            # canonical JSON text — a string bound to a JSONB parameter is
            # serialized again and compares as the JSON *string* "{}",
            # which silently matches nothing.
            _identity_qualifiers_sql()
            == literal(_identity_qualifiers(qualifiers), JSONB),
        )
        .order_by(Fact.embedding.cosine_distance(embedding))
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None or row.distance > SEMANTIC_MATCH_MAX_DISTANCE:
        return None
    matched = row[0]
    if matched.predicate != predicate:
        # Learn it: the NEXT event for this subject using this exact
        # synonym pair resolves deterministically at step 2 instead of
        # depending on embedding luck again. First discovery wins — never
        # overwritten by a later, possibly noisier, semantic match.
        await session.execute(
            pg_insert(PredicateAlias)
            .values(
                project_id=project_id,
                subject_id=subject_id,
                alias_predicate=predicate,
                canonical_predicate=matched.predicate,
                confidence=1 - row.distance,
            )
            .on_conflict_do_nothing(
                index_elements=["project_id", "subject_id", "alias_predicate"]
            )
        )
    return matched


async def _find_qualifier_ambiguous_active_fact(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    predicate: str,
    qualifiers: dict[str, Any] | None,
    value: dict[str, Any],
) -> Fact | None:
    """13 aout, fix 2 (LongMemEval ba61f0b9): a narrow second net for the one
    case `_resolve_existing_fact`'s hard qualifier guard is not meant to
    catch by design -- an UNCONDITIONAL fact (empty qualifiers, "this holds
    for the subject in general") and a QUALIFIED fact under the exact same
    predicate (a specific condition), asserting a DIFFERENT value. The hard
    guard is right to keep "wake up time weekday" and "wake up time weekend"
    apart -- two non-empty, genuinely different conditions -- but an empty
    qualifier set is not a third condition to keep apart from the others; it
    is the extractor's documented way of saying "no condition narrows this"
    (see app.providers.openai's qualifier guidance). Two active facts, one
    unconditional and one narrowed to a specific case, disagreeing on the
    SAME predicate, is exactly the shape a conflict is for -- and today it
    was invisible to `_apply_candidate`'s conflict logic, because that logic
    only runs when `_resolve_existing_fact` returns a match, and that
    function returns None the moment identity qualifiers differ at all
    (real case: `women_representation_on_team` 5 vs 6, `{}` vs {"team_name":
    "Rachel's team"} -- both stayed active side by side, neither ever
    flagged).

    Deliberately narrower than loosening the hard guard itself: exact
    predicate match only (no alias chase, no semantic fallback -- widening
    the pool is not needed here since the predicate string already matches
    verbatim), exactly one side empty, values must actually differ. The
    caller feeds the result into the ordinary create/contradiction path in
    `_apply_candidate`, which opens a ConflictSet exactly as it already does
    for a same-qualifier contradiction -- held for human review, served as
    `contested`, never silently trusted either way.
    """
    candidate_identity = _identity_qualifiers(qualifiers)
    stmt = select(Fact).where(
        Fact.project_id == project_id,
        Fact.subject_id == subject_id,
        Fact.predicate == predicate,
        Fact.status == FactStatus.active,
    )
    for fact in (await session.execute(stmt)).scalars().all():
        fact_identity = _identity_qualifiers(fact.qualifiers)
        if bool(fact_identity) == bool(candidate_identity):
            # Both empty (the exact-match path above already catches this)
            # or both non-empty (the hard guard's actual, still-intact
            # target) -- not this narrow case.
            continue
        if _canonical(fact.value) == _canonical(value):
            # Same value: not a contradiction, e.g. a general statement
            # later reaffirmed with more specific context. Leave it to the
            # ordinary duplicate/reinforcement paths, unchanged by this fix.
            continue
        return fact
    return None


async def _relevant_existing_facts(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    query_embedding: list[float] | None,
) -> list[Fact]:
    """Active facts to show the extractor ahead of one event: all of them,
    unless the subject has accumulated more than EXISTING_FACTS_FILTER_
    THRESHOLD — see that constant for why a full list stops being useful
    context past a point and starts being noise the model loses the right
    entry in. Above the threshold, ranked by cosine distance to the event's
    own embedding: the same signal _resolve_existing_fact already uses to
    match a candidate back to an active fact after extraction, applied one
    step earlier so the extractor can find it in the first place.
    """
    all_active = (
        (
            await session.execute(
                select(Fact).where(
                    Fact.project_id == project_id,
                    Fact.subject_id == subject_id,
                    Fact.status == FactStatus.active,
                )
            )
        )
        .scalars()
        .all()
    )
    if len(all_active) <= EXISTING_FACTS_FILTER_THRESHOLD or query_embedding is None:
        return list(all_active)

    stmt = (
        select(Fact)
        .where(
            Fact.project_id == project_id,
            Fact.subject_id == subject_id,
            Fact.status == FactStatus.active,
            Fact.embedding.is_not(None),
        )
        .order_by(Fact.embedding.cosine_distance(query_embedding))
        .limit(EXISTING_FACTS_TOP_K)
    )
    return list((await session.execute(stmt)).scalars().all())


# Anti-echo write gate (M1 — "porte d'ecriture"): how many of the scope's
# most recently SERVED context packets are checked against a new candidate.
# Bounded so the query stays cheap regardless of how long the subject has
# been active — this guards against a feedback loop (a fact gets served to
# the agent, echoed back in its own words, re-extracted, re-stored, served
# again...), not against every fact ever served to the subject.
ANTI_ECHO_TRACE_LOOKBACK = 20

# Distance threshold for "this candidate is the same concept as an
# already-served fact". Starts equal to SEMANTIC_MATCH_MAX_DISTANCE (same
# calibration source) but is kept as its own named constant on purpose: echo
# detection and supersession matching ask a similar question ("same
# concept?") against different candidate pools (served facts vs. active
# facts) and may need independent tuning once real echo traffic is observed.
ANTI_ECHO_MAX_DISTANCE = SEMANTIC_MATCH_MAX_DISTANCE


async def _recently_served_fact_ids(
    session: AsyncSession, *, project_id: str, subject_id: str
) -> set[uuid.UUID]:
    """Fact ids that appeared in one of this scope's last served
    ContextPackets (context_traces.packet["facts"][*]["id"])."""
    stmt = (
        select(ContextTrace.packet)
        .where(
            ContextTrace.project_id == project_id,
            ContextTrace.subject_id == subject_id,
        )
        .order_by(ContextTrace.created_at.desc())
        .limit(ANTI_ECHO_TRACE_LOOKBACK)
    )
    served: set[uuid.UUID] = set()
    for (packet,) in (await session.execute(stmt)).all():
        for fact in (packet or {}).get("facts", []):
            fact_id = fact.get("id")
            if fact_id:
                served.add(uuid.UUID(fact_id))
    return served


async def _echo_reject_reason(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    embedding: list[float],
    candidate_value: dict | None = None,
) -> str | None:
    """Post-validation half of the anti-echo write gate (M1): a candidate
    whose embedding closely matches a fact already SERVED to this subject in
    a recent context packet is a reformulation of information the agent
    already has, not new information from the world — it must never
    re-enter the ledger. Left unimplemented as a pre-LLM filter (the
    simpler, equally correct place for this sprint): it runs once per
    candidate, after extraction, using the embedding already computed for
    the write path, no extra LLM call. Returns "echo_of_context" when a
    served fact is found within ANTI_ECHO_MAX_DISTANCE, else None.

    B14: a candidate carrying a NOVEL canonical value skips the gate
    outright. An echo repeats the same information (same value); a genuine
    update or contradiction asserts a different one -- and the two
    populations are not separable by distance (rephrased-same-value pairs
    land at 0.002-0.187, genuine value updates at 0.030-0.158, fully
    overlapping). Before this, every `create` update to a recently-served
    fact was destroyed here, the conflict never opened, and the old value
    stayed served alone. (`supersede` already skipped the gate for the same
    reason -- an echo is never a replacement.)
    """
    served_ids = await _recently_served_fact_ids(
        session, project_id=project_id, subject_id=subject_id
    )
    if not served_ids:
        return None
    stmt = (
        select(Fact.embedding.cosine_distance(embedding).label("distance"))
        .where(Fact.id.in_(served_ids), Fact.embedding.is_not(None))
        .order_by(Fact.embedding.cosine_distance(embedding))
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    if candidate_value is not None:
        served_values = await _served_canonical_values(session, served_ids)
        if _canonical(candidate_value) not in served_values:
            return None
    if row.distance <= ANTI_ECHO_MAX_DISTANCE:
        return "echo_of_context"
    return None


async def _served_canonical_values(
    session: AsyncSession, served_ids: set[uuid.UUID]
) -> set[str]:
    """Canonical values of recently-served facts (B14 echo pre-filter)."""
    rows = (
        await session.execute(select(Fact.value).where(Fact.id.in_(served_ids)))
    ).all()
    return {_canonical((value or {})) for (value,) in rows}


async def _open_conflict_set(
    session: AsyncSession, *, project_id: str, subject_id: str, fact_id: uuid.UUID
) -> ConflictSet | None:
    stmt = select(ConflictSet).where(
        ConflictSet.project_id == project_id,
        ConflictSet.subject_id == subject_id,
        ConflictSet.status == "open",
    )
    for conflict in (await session.execute(stmt)).scalars().all():
        if fact_id in list(conflict.fact_ids):
            return conflict
    return None


def _reinforce_or_count_duplicate(
    fact: Fact, *, event: Event, result: dict[str, int]
) -> None:
    """M1d — write-time reinforcement: a NEW source event re-asserting the
    exact same canonical value strengthens the existing ACTIVE fact
    (counter + business-time date + provenance) instead of creating a row.

    Two deliberate guards keep this conservative:
    - replay idempotence: an event already recorded in source_event_ids
      (job reprocessing) counts as a plain duplicate, never a second
      reinforcement;
    - only an ACTIVE fact is reinforced — re-asserting a superseded/
      candidate value stays a plain duplicate, as before.

    Value equality is REQUIRED by every caller: measured with the real
    local embedder (see scripts/check_semantic_threshold.py), rephrased-
    same-value pairs (0.002-0.187) and genuine value updates (0.030-0.158)
    fully overlap in cosine distance, so no threshold can authorize a
    merge — a false merge silently destroys an update, which is strictly
    worse than a duplicate.

    Anti-clock-poisoning guard (M8): the freshness clock (last_reinforced_
    at) only advances when the reasserting event's origin rank is >= the
    fact's own — otherwise a lower-trust source (an untrusted document, a
    third party) could keep a volatile fact looking "fresh" forever just
    by repeating it. The counter and source_event_ids still record the
    reinforcement either way; only the CLOCK is gated.
    """
    if fact.status is not FactStatus.active or event.id in fact.source_event_ids:
        result["duplicates"] += 1
        return
    fact.reinforcement_count += 1
    event_rank = ORIGIN_TRUST_RANK.get(event.origin_trust or "trusted", 0)
    fact_rank = ORIGIN_TRUST_RANK.get(fact.origin_trust or "trusted", 3)
    if (
        event_rank >= fact_rank
        and (fact.last_reinforced_at is None or event.occurred_at > fact.last_reinforced_at)
    ):
        # Business time, monotonic: out-of-order replays never move it back.
        fact.last_reinforced_at = event.occurred_at
    # ARRAY column without a Mutable wrapper: reassign, never append in place.
    fact.source_event_ids = [*fact.source_event_ids, event.id]
    result["reinforced"] += 1


async def _write_episode_chunks(
    session: AsyncSession, events: list[Event], embedder: Embedder
) -> int:
    """Cut every not-yet-chunked event into retrievable slices and embed them.

    Idempotent: an event that already has chunks is skipped entirely rather
    than re-chunked, so a replayed job is a no-op here. That check is on
    (event_id) and not on content because chunking is deterministic --
    same payload, same chunks -- so "has chunks" and "has the right chunks"
    are the same statement. Re-chunking after a change to
    app.context.chunking is a backfill, not a side effect of replay: see
    scripts/backfill_episode_chunks.py.

    Returns the number of chunks written, for the caller's metrics.
    """
    if not events:
        return 0
    already_chunked = set(
        (
            await session.execute(
                select(EpisodeChunk.event_id).where(
                    EpisodeChunk.event_id.in_([event.id for event in events])
                )
            )
        )
        .scalars()
        .all()
    )
    pending: list[tuple[Event, int, str]] = []
    for event in events:
        if event.id in already_chunked:
            continue
        for ordinal, text in enumerate(chunk_payload(event.kind, event.payload)):
            pending.append((event, ordinal, text))
    if not pending:
        return 0

    # One batched call for every chunk of every event in the job: the
    # embedder's own batching is what keeps this from becoming N round
    # trips, and the local ONNX provider caches identical texts.
    embeddings = await embedder.embed([text for _, _, text in pending])
    session.add_all(
        [
            EpisodeChunk(
                event_id=event.id,
                ordinal=ordinal,
                project_id=event.project_id,
                subject_id=event.subject_id,
                occurred_at=event.occurred_at,
                origin_trust=event.origin_trust,
                text=text,
                # index_text == text until the fact-to-chunk link lands
                # (K = V + fact); the two columns stay distinct so that a
                # fact folded into the index can never leak into what the
                # agent reads as a verbatim quote.
                index_text=text,
                embedding=embedding,
            )
            for (event, ordinal, text), embedding in zip(pending, embeddings)
        ]
    )
    await session.flush()
    return len(pending)


# A verbatim quote from an LLM is verbatim in intention, not in bytes:
# whitespace collapses, quotes get curled, case drifts. Matching is done on
# a normalised form, and only an EXACT containment counts as a match. The
# alternative -- accepting the best fuzzy candidate -- would attribute a
# fact to a turn it did not come from and silently pollute that turn's
# index, which is worse than attributing nothing: `source_chunk_id` stays
# NULL and every consumer falls back to what it did before.
_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize_for_span_match(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", (text or "").replace("’", "'")).strip().casefold()


async def _resolve_source_chunk(
    session: AsyncSession, event: Event, evidence_span: str | None
) -> uuid.UUID | None:
    """Which turn of this event the extractor quoted, if it can be known.

    Two ways to know, both exact, no heuristic:

    1. The event has a single chunk. Then there is nothing to disambiguate
       -- whatever the fact came from, it came from there. This is the
       common case for `conversation.message` events, i.e. most of the
       product's traffic.
    2. The normalised evidence span is contained in exactly one chunk.

    Anything else returns None. A fact with no attributable turn is a fact
    that enriches no chunk's index and whose source turn is guessed by the
    context window's fallback -- both strictly the pre-21-aout behaviour.
    """
    chunks = (
        (
            await session.execute(
                select(EpisodeChunk.id, EpisodeChunk.text)
                .where(EpisodeChunk.event_id == event.id)
                .order_by(EpisodeChunk.ordinal)
            )
        )
        .all()
    )
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0].id
    span = _normalize_for_span_match(evidence_span or "")
    if not span:
        return None
    matches = [
        chunk.id for chunk in chunks if span in _normalize_for_span_match(chunk.text)
    ]
    return matches[0] if len(matches) == 1 else None


async def _merge_facts_into_chunk_index(
    session: AsyncSession, event: Event, embedder: Embedder
) -> None:
    """Key merging (E3), at the granularity that makes it work.

    A chunk's `index_text` becomes its verbatim text concatenated with the
    facts extracted FROM THAT CHUNK -- LongMemEval's K = V + fact: the
    compressed signal folded into the raw key at INDEX time, so a turn can
    be found through what was understood of it and still be SERVED as its
    own words. `text` is never touched, which is what keeps a fact from
    leaking into a quote.

    Until this mechanism, key merging ran against the whole event, so a
    fact extracted from turn 3 of a twenty-turn session was folded into the
    index of all twenty -- and into an embedding that only ever saw the
    first 128 tokens of the concatenation anyway.

    Facts with no `source_chunk_id` enrich nothing: attributing them to
    every chunk of the event is exactly the pollution this replaces.
    """
    facts = (
        (
            await session.execute(
                select(Fact)
                .where(
                    Fact.source_event_ids.any(event.id),
                    Fact.source_chunk_id.is_not(None),
                )
                # Fully specified order, on purpose (22 aout). Without it
                # Postgres returns the rows in whatever order the plan and the
                # physical heap happen to produce -- which shifts as soon as a
                # tuple is rewritten -- so the SAME facts on the SAME chunk
                # produced a DIFFERENT `index_text`, hence a different embedding
                # and a different ts_rank_cd cover density. Two identical
                # ingestions of one conversation then retrieved differently
                # (measured: 86 of 231 packets differed between two runs of
                # eval.retrieval_bench with nothing changed), and the equality
                # guard below could never fire on a replayed job, so every
                # replay paid for a re-embedding that changed nothing.
                #
                # `recorded_from` first because chronological is the one order
                # that means something here; the rest of the key exists so that
                # the SAME SET of facts always renders to the SAME STRING --
                # predicate and value are content, `id` only makes it total.
                .order_by(Fact.recorded_from, Fact.predicate, cast(Fact.value, Text), Fact.id)
            )
        )
        .scalars()
        .all()
    )
    if not facts:
        return
    by_chunk: dict[uuid.UUID, list[Fact]] = {}
    for fact in facts:
        by_chunk.setdefault(fact.source_chunk_id, []).append(fact)

    chunks = (
        (
            await session.execute(
                select(EpisodeChunk).where(EpisodeChunk.id.in_(by_chunk.keys()))
            )
        )
        .scalars()
        .all()
    )
    updated = []
    for chunk in chunks:
        facts_text = "; ".join(
            _search_text(fact.predicate, fact.value) for fact in by_chunk[chunk.id]
        )
        index_text = f"{chunk.text} FACTS: {facts_text}"
        if index_text == chunk.index_text:
            # Replayed job, or a re-assertion that changed nothing:
            # re-embedding identical text costs a call for no change.
            continue
        chunk.index_text = index_text
        updated.append(chunk)
    if updated:
        embeddings = await embedder.embed([chunk.index_text for chunk in updated])
        for chunk, embedding in zip(updated, embeddings):
            chunk.embedding = embedding


async def _apply_candidate(
    session: AsyncSession,
    candidate: ExtractedFact,
    *,
    embedding: list[float],
    event: Event,
    result: dict[str, int],
) -> None:
    # Scope (subject_id) always comes from the EVENT, never from the LLM's
    # candidate.subject_id, even though the extraction schema still carries
    # that field (kept for backward compatibility with existing providers).
    # "The model never chooses scopes" is a documented security invariant
    # (README - Securite) that this write path did not actually enforce:
    # a candidate whose subject_id drifted from the event's (e.g. the
    # extractor naming a person instead of reusing the event subject_id)
    # silently created a fact under an orphan subject_id that no /v1/context
    # call could ever reach again - a real, confirmed data-loss bug found
    # while auditing the sprint-10 eval results at scale.
    # Resolved once for this candidate: both write paths below carry it,
    # and it is what makes key merging chunk-accurate (see
    # _resolve_source_chunk / _merge_facts_into_chunk_index).
    source_chunk_id = await _resolve_source_chunk(session, event, candidate.evidence_span)

    duplicate = await _find_duplicate(
        session,
        project_id=event.project_id,
        subject_id=event.subject_id,
        predicate=candidate.predicate,
        value=candidate.value,
        qualifiers=candidate.qualifiers,
    )
    if duplicate is not None:
        # A value coming BACK after being superseded is an update, not a
        # duplicate (21 aout). A -> B -> A: `_find_duplicate` matches the
        # superseded A (it searches every non-deleted status),
        # `_reinforce_or_count_duplicate` sees a non-active fact, counts a
        # duplicate and returns -- and B stays active. The subject said
        # they went back to A and the ledger kept serving B, with a counter
        # as the only trace. This is exactly the knowledge-update failure
        # mode LongMemEval's category of the same name is built to catch.
        #
        # It is only an update if something else is actually active for
        # this identity. A superseded value re-asserted with nothing active
        # to replace stays a plain duplicate, as before.
        superseded_return = duplicate.status is not FactStatus.active and (
            await _resolve_existing_fact(
                session,
                project_id=event.project_id,
                subject_id=event.subject_id,
                predicate=candidate.supersedes_predicate or candidate.predicate,
                qualifiers=candidate.qualifiers,
                embedding=embedding,
            )
        ) is not None
        if not superseded_return:
            _reinforce_or_count_duplicate(duplicate, event=event, result=result)
            await session.flush()
            return
        # Treated as the supersede it is, whatever the extractor labelled
        # it: the extractor sees one message, not the identity's history,
        # so it has no way to know this value was held before.
        candidate = candidate.model_copy(update={"action": "supersede"})

    target_predicate = candidate.supersedes_predicate or candidate.predicate
    # The candidate's own qualifiers, before this module stamps provenance
    # onto them below: identity is what the extractor declared about WHEN
    # and under WHAT CONDITIONS the fact holds, never who reported it.
    existing = await _resolve_existing_fact(
        session,
        project_id=event.project_id,
        subject_id=event.subject_id,
        predicate=target_predicate,
        qualifiers=candidate.qualifiers,
        embedding=embedding,
    )
    if existing is None and candidate.action == "create":
        existing = await _find_qualifier_ambiguous_active_fact(
            session,
            project_id=event.project_id,
            subject_id=event.subject_id,
            predicate=target_predicate,
            qualifiers=candidate.qualifiers,
            value=candidate.value,
        )

    if (
        candidate.action == "create"
        and existing is not None
        and existing.memory_form != "event"
        and _canonical(existing.value) == _canonical(candidate.value)
    ):
        # Same value under a semantically-equivalent predicate string
        # (exact-predicate matches were already caught by _find_duplicate):
        # reinforcement, and — unlike before — NO row is created first, so
        # no orphan `candidate` row is left behind.
        #
        # `existing.memory_form != "event"` (bug found by code review, 16
        # aout): _resolve_existing_fact/_active_fact above have no
        # memory_form filter (unlike _find_duplicate), so this short-
        # circuit used to fire for an "event"-form identity too whenever a
        # later occurrence happened to carry the exact same value as an
        # earlier one -- silently reinforcing instead of falling through to
        # the memory_form=="event" branch below (always active, never
        # fused). Excluded here so that branch is reached instead.
        _reinforce_or_count_duplicate(existing, event=event, result=result)
        await session.flush()
        return

    # Typology/volatility (M2): the candidate's own classes win; on a
    # supersede of an existing fact, omitted classes are inherited rather
    # than silently reset to the defaults (a status-only update must not
    # "promote" a volatile fact to stable just because the extractor did not
    # restate its class — same reasoning as the value carry-forward below).
    # Computed here (before the trust check) so a quarantined fact keeps
    # its declared/inherited typology instead of silently defaulting.
    inherit = candidate.action == "supersede" and existing is not None
    fact_kind = candidate.fact_kind or (existing.fact_kind if inherit else "attribute")
    volatility = candidate.volatility or (existing.volatility if inherit else "stable")
    # `temporal_range` inherits for exactly the same reason (21 aout) and
    # was the one field of the three that did not. A status-only update
    # ("pre-approved" -> "approved") whose candidate does not restate the
    # range it was anchored on -- and the extraction prompt does not ask it
    # to -- silently destroyed the anchoring: the fact stopped being about
    # August and became about the message's own timestamp. Same class of
    # loss as the value carry-forward below, on the field the temporal
    # questions depend on.
    temporal_range = candidate.temporal_range or (
        existing.temporal_range if inherit else None
    )
    # Last resort, and the case that turns out to be the common one
    # (23 aout): the extractor was asked to resolve relative expressions
    # into `temporal_range` and, measured on a real provider, does it for
    # 1.8 % of facts. "next month" comes back as the string the subject
    # said. Resolved here instead, against the event's own timestamp --
    # see app.consolidator.temporal.resolve_relative_range for why only
    # expressions with an exact calendar meaning are accepted.
    #
    # Third in line on purpose: an extractor-resolved range says what the
    # extractor understood, and a plain ISO date already inside `value` is
    # more precise than any phrase. This only fills a hole.
    if temporal_range is None and not _dates_in(candidate.value):
        temporal_range = resolve_relative_range(candidate.value, event.occurred_at)

    # Memory form (mechanism C, 15 aout): unlike fact_kind/volatility above,
    # ALWAYS inherited whenever an existing identity was matched -- on
    # "create" too, not just "supersede". This is deliberate: once an
    # identity (subject, predicate, qualifiers) has an active fact, its
    # form is settled, and a single later candidate's own guess must never
    # flip it back and forth between runs (extraction is non-deterministic
    # -- observed directly this session). The ONLY sanctioned way an
    # identity moves from "state" to "event" is the conflict-overflow
    # reclassification below -- a structural, one-way, loud transition,
    # never a silent per-candidate toggle. A brand-new identity (existing
    # is None) takes the candidate's own declared form, default "state".
    memory_form = existing.memory_form if existing is not None else (candidate.memory_form or "state")

    # Provenance authority (M8). Attribution first: a fact born from a
    # third party is stamped with who actually said it — deterministic,
    # not prompt-dependent (the prompt also asks for a named value, but
    # the qualifier is what the packet exposes as attributed_to).
    qualifiers = candidate.qualifiers
    if event.origin_trust == "third_party":
        qualifiers = {**qualifiers, "attributed_to": event.actor_id or "third_party"}

    event_rank = ORIGIN_TRUST_RANK.get(event.origin_trust or "trusted", 0)
    existing_rank = (
        ORIGIN_TRUST_RANK.get(existing.origin_trust or "trusted", 3)
        if existing is not None
        else None
    )
    if (existing_rank is not None and event_rank < existing_rank) or (
        event.origin_trust == "untrusted"
    ):
        # Origin holdback ("quarantine" without a new status): the
        # candidate never auto-activates. Two triggers, same mechanics:
        # (a) a strictly lower-ranked origin trying to displace/contradict
        # a higher-ranked fact — the higher-ranked fact stays SERVED (a
        # third party or a poisoned document must not be able to hide the
        # subject's own memory behind an open conflict), the candidate is
        # held alone; (b) any untrusted-origin candidate, even on a brand
        # new predicate — ingested content never enters served memory
        # without human resolution (write-time filtering is structurally
        # blind to compositional/dormant payloads; provenance is the
        # backstop). Held = status stays `candidate` + a single-member
        # open ConflictSet: never served (context blocks open-conflict
        # facts), visible on the console's Conflicts page, resolvable via
        # POST /v1/conflicts/{id}/resolve (keep -> active) or discardable
        # via POST /v1/feedback rating=incorrect (-> disputed).
        held_fact = await create_fact(
            session,
            org_id=event.org_id,
            project_id=event.project_id,
            subject_id=event.subject_id,
            predicate=candidate.predicate,
            value=candidate.value,
            subject_type=event.subject_type,
            agent_id=event.agent_id,
            qualifiers=qualifiers,
            confidence=candidate.confidence,
            valid_from=event.occurred_at,
            source_event_ids=[event.id],
            fact_kind=fact_kind,
            volatility=volatility,
            origin_trust=event.origin_trust or "trusted",
            memory_form=memory_form,
            temporal_range=temporal_range,
            observed_at=observed_at_of(candidate.value, temporal_range),
            evidence_span=candidate.evidence_span,
            source_chunk_id=source_chunk_id,
        )
        held_fact.embedding = embedding
        held_fact.search_text = _search_text(candidate.predicate, candidate.value)
        if existing_rank is not None and event_rank < existing_rank:
            reason = (
                f"lower_trust_origin: '{event.origin_trust}' candidate for "
                f"predicate '{candidate.predicate}' held for review — it "
                f"cannot displace the '{existing.origin_trust}' fact "
                f"{existing.id}, which stays served"
            )
        else:
            reason = (
                f"untrusted_origin: candidate for predicate "
                f"'{candidate.predicate}' from an untrusted event held for "
                "human review before it can ever be served"
            )
        session.add(
            ConflictSet(
                project_id=event.project_id,
                subject_id=event.subject_id,
                fact_ids=[held_fact.id],
                status="open",
                kind="quarantine",
                reason=reason,
            )
        )
        await session.flush()
        result["quarantined"] += 1
        logger.info(
            "consolidator: quarantined candidate fact %s (%s)", held_fact.id, reason
        )
        return

    value = candidate.value
    if candidate.action == "supersede" and existing is not None:
        # Carry forward descriptive fields the extractor didn't re-state.
        # A status-only update ("researching" -> "completed") must not
        # silently drop the fact's subject/target just because the LLM's
        # new value only mentions what changed — verified failure case:
        # `adoption_agency_research` went from {target: "adoption
        # agencies", status: "researching"} to a bare {status: "completed"}
        # on supersede, and the target was only recoverable from the now-
        # superseded (never served) version. Keys the candidate DOES set
        # always win — that is the actual update.
        value = {**existing.value, **candidate.value}
        # ...and an explicit null REMOVES a key (21 aout). Without it, the
        # carry-forward above has no inverse: once a key is in a value, no
        # later update can take it out, so a fact accumulates keys that
        # stopped being true and are served as if they still were. Making
        # the merge opt-in instead would give back the loss it was written
        # to fix (a status-only update dropping the fact's target, the
        # measured adoption_agency_research case); a delete sentinel adds
        # the missing direction without taking that away.
        #
        # `{"key": null}` carried no meaning before this -- it stored a
        # null, which says nothing -- so nothing that used to work changes.
        value = {key: item for key, item in value.items() if item is not None}

    fact = await create_fact(
        session,
        org_id=event.org_id,
        project_id=event.project_id,
        subject_id=event.subject_id,
        predicate=candidate.predicate,
        value=value,
        subject_type=event.subject_type,
        agent_id=event.agent_id,
        qualifiers=qualifiers,
        confidence=candidate.confidence,
        valid_from=event.occurred_at,
        source_event_ids=[event.id],
        fact_kind=fact_kind,
        volatility=volatility,
        origin_trust=event.origin_trust or "trusted",
        memory_form=memory_form,
        temporal_range=temporal_range,
        # Computed from the FINAL value: the supersede carry-forward above
        # may have restored a date the candidate did not restate.
        observed_at=observed_at_of(value, temporal_range),
        evidence_span=candidate.evidence_span,
        source_chunk_id=source_chunk_id,
    )
    fact.embedding = embedding
    fact.search_text = _search_text(candidate.predicate, value)

    if candidate.action == "supersede":
        if existing is not None:
            # Ledger transition active -> superseded; the replaced fact stays
            # in history and is never served as current again.
            await transition_fact_status(session, existing.id, FactStatus.superseded)
            # B11, same monotonic-clock idiom as last_reinforced_at above: a
            # late-ingested event carries an OLDER occurred_at than the fact
            # it replaces, and writing it raw as valid_to would make
            # valid_to < valid_from -- a negative validity interval that
            # corrupts duration rendering and date arbitration downstream.
            # Clamp to valid_from (zero-length, never negative); the new
            # fact still takes event.occurred_at as its own valid_from.
            if existing.valid_from is not None and event.occurred_at < existing.valid_from:
                existing.valid_to = existing.valid_from
            else:
                existing.valid_to = event.occurred_at
            fact.supersedes_id = existing.id
            result["superseded"] += 1
        else:
            # B7: the extractor labeled this "supersede" but no existing fact
            # was found to replace -- an extractor labeling error, not a
            # supersession. The value itself is novel, so activating it as a
            # fresh fact loses nothing (semantically identical to "create"),
            # but it is counted separately: a rising number here means the
            # extraction prompt mislabels updates and needs attention.
            result["supersede_without_existing"] += 1
        await transition_fact_status(session, fact.id, FactStatus.active)
        result["created"] += 1
        return

    # action == "create"
    if memory_form == "event":
        # Mechanism C (15 aout): an accumulating occurrence is never a
        # contradiction of the others under the same identity, however
        # many are already active -- go straight to active, exactly like
        # a brand-new identity (existing is None) would. This is the
        # entire fix for the diagnosed failure (research/Diagnostic_
        # Couverture_2026-08-14.md, cas Maria): 5 genuinely different
        # volunteering occurrences no longer compete for one "current"
        # slot or get capped into invisible quarantine.
        await transition_fact_status(session, fact.id, FactStatus.active)
        result["created"] += 1
        return

    if existing is not None and _canonical(existing.value) != _canonical(candidate.value):
        # Contradiction: both facts enter an open conflict set; the new fact
        # stays `candidate` until resolution (PRD lifecycle rule).
        conflict = await _open_conflict_set(
            session,
            project_id=event.project_id,
            subject_id=event.subject_id,
            fact_id=existing.id,
        )
        if conflict is None:
            conflict = ConflictSet(
                project_id=event.project_id,
                subject_id=event.subject_id,
                fact_ids=[existing.id, fact.id],
                status="open",
                kind="contradiction",
                reason=(
                    f"predicate '{candidate.predicate}': "
                    f"{_canonical(existing.value)} vs {_canonical(candidate.value)}"
                ),
            )
            session.add(conflict)
        elif len(conflict.fact_ids) >= CONFLICT_SET_MAX_MEMBERS:
            # Cap reached with memory_form still "state" (an "event"
            # candidate never reaches this branch at all -- see the early
            # return above). Mechanism C (15 aout): a 3rd competing value
            # under one identity is not "one bad match" to quarantine, it
            # is PROOF this predicate was never a scalar in the first
            # place -- the eval run that motivated the original cap found
            # 3+ member sets in ~24% of cases, and the real diagnostic
            # case (research/Diagnostic_Couverture_2026-08-14.md, Maria)
            # showed exactly this: 5 genuinely distinct volunteering
            # occurrences, capped into invisible quarantine one by one.
            #
            # Reclassify the WHOLE identity as memory_form "event",
            # dissolve the conflict, activate every member -- zero LLM
            # calls, zero new threshold, reversible by an operator later
            # if this really was a bad match (flip memory_form back to
            # "state" and the ordinary conflict path resumes). A genuine
            # scalar never reaches a 3rd competing value in the first
            # place (its updates go through "supersede", not "create"),
            # so this never fires on a real scalar.
            other_id = next(fid for fid in conflict.fact_ids if fid != existing.id)
            other = await session.get(Fact, other_id)
            # Safety net (found by code review, 16 aout): this reclassification
            # trusts a heuristic ("a genuine scalar never reaches a 3rd
            # competing value") that non-deterministic extraction can defeat.
            # Rather than a new calibrated threshold or reintroducing the
            # quarantine this branch replaces, mark all 3 members instead --
            # the project's existing honest-degradation convention (contested
            # facts, "unconfirmed"/"stale" freshness): never hidden, always
            # flagged so the reader can judge whether this looks like 3
            # genuine occurrences or 3 updates to one attribute. See
            # Fact.reclassified_at and app.context._packet_fact.
            reclassified_at = datetime.now(timezone.utc)
            existing.memory_form = "event"
            existing.reclassified_at = reclassified_at
            fact.memory_form = "event"
            fact.reclassified_at = reclassified_at
            await transition_fact_status(session, fact.id, FactStatus.active)
            if other is not None:
                other.memory_form = "event"
                other.reclassified_at = reclassified_at
                if other.status is not FactStatus.active:
                    await transition_fact_status(session, other.id, FactStatus.active)
            # The triggering (3rd) fact was never a member of this conflict
            # set -- it only existed as the incoming candidate. Record it so
            # fact_ids reflects all 3 reclassified facts, matching what the
            # log line below already claims.
            conflict.fact_ids = [*conflict.fact_ids, fact.id]
            conflict.status = "reclassified_event"
            conflict.resolved_at = datetime.now(timezone.utc)
            await session.flush()
            result["created"] += 1
            result["reclassified_event"] += 1
            # Loud on purpose, like the capping it replaces: this is a
            # structural change to how a predicate behaves for this
            # subject from now on, worth watching after any change to the
            # matching rules.
            logger.warning(
                "consolidator: conflict set %s for subject %s reclassified as "
                "memory_form=event (predicate '%s') -- %d fact(s) activated, "
                "conflict dissolved",
                conflict.id,
                event.subject_id,
                candidate.predicate,
                len([f for f in (existing, other, fact) if f is not None]),
            )
            return
        else:
            conflict.fact_ids = [*conflict.fact_ids, fact.id]
        await session.flush()
        result["conflicts"] += 1
        return

    # existing is None here: the semantic-same-value short-circuit above
    # already returned for any "create" whose value matches an existing
    # fact, and the contradiction branch above handles a differing value.
    await transition_fact_status(session, fact.id, FactStatus.active)
    result["created"] += 1


def _record_rejection(
    result: dict[str, Any], reason: str, *, job_id: uuid.UUID, source: str
) -> None:
    """Count a candidate classified against the write-gate taxonomy (M1)
    that will never become a Fact — either the provider itself said
    action="reject" (source="provider") or the anti-echo post-validation
    rule caught it (source="anti-echo"). `rejected` is the same aggregate
    counter used for plain Pydantic validation failures; `rejected_with_
    reason` is the breakdown, keyed by `reason` (one of REJECT_REASONS)."""
    result["rejected"] += 1
    result["rejected_with_reason"][reason] += 1
    logger.info(
        "consolidator: rejected candidate (job %s, reason=%s, source=%s)",
        job_id,
        reason,
        source,
    )


async def _process_job(
    session: AsyncSession, job: Job, extractor: Extractor, embedder: Embedder,
    *,
    only_event_ids: set[uuid.UUID] | None = None,
) -> dict[str, Any]:
    event_ids = [uuid.UUID(e) for e in job.payload.get("event_ids", [])]
    events = (
        (
            await session.execute(
                select(Event)
                .where(Event.id.in_(event_ids))
                .order_by(Event.occurred_at, Event.id)
            )
        )
        .scalars()
        .all()
    )
    if only_event_ids is not None:
        # B12: a subject-scoped run (POST /v1/consolidate/subject) must never
        # extract another subject's events just because they share this job's
        # batch. The remainder stays in the payload for the global
        # worker/ops run (see _run_jobs) -- nothing is skipped, nothing is
        # processed under the wrong scope.
        events = [e for e in events if e.id in only_event_ids]

    result: dict[str, Any] = {
        "created": 0,
        "superseded": 0,
        "conflicts": 0,
        # Subset of `conflicts`: candidates held apart because the conflict
        # on their fact was already full. Worth its own counter — it is the
        # number to watch after any change to the matching rules.
        "conflict_capped": 0,
        # Mechanism C (15 aout): a capped conflict automatically
        # reclassified as memory_form="event" instead of held apart --
        # counted separately from "created" (also incremented) because it
        # is a structural signal worth watching, like conflict_capped was
        # before it (this replaces most, not all, of what conflict_capped
        # used to count -- only whichever candidate happens to still be
        # memory_form "state" at cap time goes this way instead now).
        "reclassified_event": 0,
        # B7: "supersede" with no existing fact to replace (extractor
        # mislabel) -- still counted in "created" above, tracked here too.
        "supersede_without_existing": 0,
        "duplicates": 0,
        "reinforced": 0,
        "quarantined": 0,
        "rejected": 0,
        "rejected_with_reason": {reason: 0 for reason in REJECT_REASONS},
    }

    # Episodic memory (sprint 10): embed each processed event once, up
    # front (derived data, re-computable — one of the only post-insert
    # writes allowed on events). Events already embedded (replayed job) are
    # skipped. Independent of extraction order, so done before the
    # per-event loop below. `index_text` (mechanism E1a/E3, 15 aout) starts
    # here as the same plain kind+payload text already embedded — the
    # per-event loop below overwrites both, for THIS SAME event only, once
    # its own extracted facts are known, folding them into index_text (true
    # key merging) and re-embedding from it instead of the raw payload
    # alone. An event that ends up with no applied candidate keeps this
    # baseline value for both — correct, since "kind+payload with no facts
    # appended" is exactly what index_text degrades to anyway.
    unembedded = [event for event in events if event.embedding is None]
    if unembedded:
        event_embeddings = await embedder.embed(
            [episode_text(event.kind, event.payload) for event in unembedded]
        )
        for event, embedding in zip(unembedded, event_embeddings):
            event.embedding = embedding
            event.index_text = episode_text(event.kind, event.payload)

    # Episode chunks (21 aout, migration 0027): the unit episodic retrieval
    # actually reads. An event is indexed and served in turn-sized slices
    # instead of whole -- see app.context.chunking for why (a whole session
    # cost 810 of the eval's 900-token budget, and the embedder saw 12.4 %
    # of it).
    #
    # Derived data, like the event embedding above: reconstructible from
    # the payload at any time, which is what makes it safe to write here on
    # an append-only table. Idempotent by (event_id, ordinal) so a replayed
    # job re-chunks nothing.
    #
    # Note on cost: this adds one embedding call per CHUNK on top of the one
    # per event above. The event-level embedding is no longer read by
    # retrieval after this change -- its only remaining readers are the E3
    # key-merging pass below and scripts/backfill_episode_index_text.py --
    # and dropping it is a follow-up, deliberately not bundled here.
    await _write_episode_chunks(session, events, embedder)

    # Locks acquired up front, before any extraction — earlier than before
    # (see the loop below for why) but the lock's actual job is unchanged:
    # serializing two JOBS for the same subject across workers, never
    # events within this one job. Sorted: a job spanning several subjects
    # always locks them in the same order (deadlock avoidance).
    for project_id, subject_id in sorted(
        {(event.project_id, event.subject_id) for event in events}
    ):
        await acquire_subject_write_lock(
            session, project_id=project_id, subject_id=subject_id
        )

    # Extraction and application are interleaved per event, in
    # chronological order — NOT extract-the-whole-batch-then-apply-the-
    # whole-batch as before. A job's event_ids can span many sessions for
    # ONE subject (a bulk import, a backfill, an eval harness ingesting a
    # subject's whole history through one capture() call): with the old
    # two-phase split, event N's "active facts" snapshot was queried
    # before ANY candidate from this same job had been applied, so it
    # never included what event N-1 of the SAME job was about to create.
    # A real provider given an empty existing_facts view for event N has
    # no way to recognize a same-predicate value change as an update and
    # correctly emit action="supersede" — confirmed against real eval
    # data: a personal record updated 7 days later, both mentions landing
    # in one capture() batch, was misclassified as a contradiction instead
    # of a supersession on every re-ingestion run, even after the
    # fact-identity (qualifiers) fix, because that fix addresses
    # identity matching, not this — the extractor never saw the earlier
    # value to compare against in the first place. Interleaving makes
    # event N's active-facts query see everything already applied from
    # events 1..N-1 of this same job, same as it always has across
    # separate jobs. Trade-off: the per-subject lock above is now held
    # across the (slower) extraction round-trips too, instead of only
    # across the apply phase — a wider contention window, accepted
    # because correctness here matters more than that narrower window.
    for event in events:
        active_facts = await _relevant_existing_facts(
            session,
            project_id=event.project_id,
            subject_id=event.subject_id,
            query_embedding=event.embedding,
        )
        existing = [
            {
                "predicate": fact.predicate,
                "value": fact.value,
                "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
            }
            for fact in active_facts
        ]

        # Validate before embedding: an invalid candidate is rejected and
        # logged, never crashes the job. A candidate the provider itself
        # marked action="reject" (write gate M1) is a well-formed
        # observation the extractor deliberately screened out (echo,
        # noise, unsourced inference...) — counted the same way and never
        # reaches embedding/application.
        to_apply: list[ExtractedFact] = []
        for raw in await extractor.extract_facts([event], existing=existing):
            try:
                candidate = ExtractedFact.model_validate(raw)
            except ValidationError as exc:
                # No reason code: the candidate never even parsed, so
                # there is no taxonomy to classify it against.
                result["rejected"] += 1
                logger.warning(
                    "consolidator: rejected invalid candidate (job %s): %s",
                    job.id,
                    exc,
                )
                continue
            if candidate.action == "reject":
                _record_rejection(
                    result, candidate.reject_reason, job_id=job.id, source="provider"
                )
                continue
            directive_reason = _imperative_directive_reason(candidate)
            if directive_reason is not None:
                _record_rejection(
                    result, directive_reason, job_id=job.id, source="directive-filter"
                )
                continue
            trust_reason = _untrusted_instruction_reason(event, candidate)
            if trust_reason is not None:
                _record_rejection(
                    result, trust_reason, job_id=job.id, source="trust-gate"
                )
                continue
            to_apply.append(candidate)

        if not to_apply:
            continue

        texts = [_search_text(fact.predicate, fact.value) for fact in to_apply]
        embeddings = await embedder.embed(texts)

        for candidate, embedding in zip(to_apply, embeddings):
            # Anti-echo write gate (M1), post-validation: a candidate that
            # only reformulates a fact already SERVED to this subject in a
            # recent context packet is rejected here, before it can ever
            # reach the ledger — this is what stops a served fact from
            # being echoed back, re-extracted, and re-stored without
            # bound.
            #
            # ...but ONLY for a candidate that claims to add something
            # (21 aout). An action="supersede" candidate says explicitly
            # that it REPLACES a value, which is the one thing an echo can
            # never be: an agent repeating a fact back produces the same
            # fact, not a replacement for it.
            #
            # This matters because the two populations are not separable by
            # distance. This module's own measurement says so
            # (_reinforce_or_count_duplicate, from
            # scripts/check_semantic_threshold.py): rephrased-same-value
            # pairs land at 0.002-0.187 and genuine value updates at
            # 0.030-0.158 -- fully overlapping, and both under the 0.28
            # gate. So every legitimate update to a fact that had been
            # served in the last 20 packets was being destroyed here, with
            # a counter as its only trace.
            echo_reason = (
                await _echo_reject_reason(
                    session,
                    project_id=event.project_id,
                    subject_id=event.subject_id,
                    embedding=embedding,
                    candidate_value=candidate.value,
                )
                if candidate.action != "supersede"
                else None
            )
            if echo_reason is not None:
                _record_rejection(result, echo_reason, job_id=job.id, source="anti-echo")
                continue
            await _apply_candidate(
                session, candidate, embedding=embedding, event=event, result=result
            )

        # True key merging (mechanism E3, 15 aout): now that every candidate
        # from THIS event has been applied, fold whichever facts it actually
        # touched (created, held, or reinforced -- read back from the source
        # of truth, source_event_ids, rather than tracked through every
        # _apply_candidate branch) into index_text and re-embed from it.
        # Skipped when nothing was touched (e.g. every candidate was
        # echo-rejected above) -- index_text/embedding already hold the
        # correct plain-payload baseline set in the up-front pass, and
        # re-embedding identical text would only cost an extra call for no
        # change.
        touched_facts = (
            (
                await session.execute(
                    select(Fact).where(Fact.source_event_ids.any(event.id))
                )
            )
            .scalars()
            .all()
        )
        if touched_facts:
            facts_text = "; ".join(
                _search_text(fact.predicate, fact.value) for fact in touched_facts
            )
            event.index_text = (
                f"{episode_text(event.kind, event.payload)} FACTS: {facts_text}"
            )
            event.embedding = (await embedder.embed([event.index_text]))[0]
        # The same mechanism at the granularity retrieval actually reads
        # (21 aout, migration 0028). The event-level pass above is kept for
        # now -- it is what scripts/backfill_episode_index_text.py and the
        # /v1/timeline lineage still look at -- but nothing in the
        # retrieval path reads it any more, and removing it is a follow-up.
        await _merge_facts_into_chunk_index(session, event, embedder)
    return result


# How long a claim stays valid before another worker may take the job back.
# A worker killed mid-job (deploy, OOM, SIGKILL) never gets to mark it
# failed, so without a reclaim window the job would sit in `running`
# forever -- trading a duplicate-processing bug for a stuck-job bug. Two
# minutes is far longer than any observed consolidation (seconds, dominated
# by one LLM call) and far shorter than a human noticing a stalled queue.
STALE_CLAIM_AFTER = timedelta(minutes=2)


async def claim_jobs(session: AsyncSession, job_ids: list[uuid.UUID]) -> list[Job]:
    """Take exclusive ownership of jobs, atomically. Returns what was won.

    `FOR UPDATE SKIP LOCKED` inside the subquery is what makes two workers
    safe: each locks the rows it selects, and skips rows another
    transaction already holds instead of blocking on them. The UPDATE then
    flips the winner's rows to `running`, so they are invisible to the next
    poll of either worker.

    Until 22 aout nothing did this. `JobStatus.running` existed and was
    never assigned; every caller selected `pending` rows and processed them.
    Two workers -- or one worker and one POST /v1/consolidate -- therefore
    picked up the same job and extracted the same events twice. The
    per-subject advisory lock serialised the writes, but the second
    extraction is a second LLM call, non-deterministic even at temperature
    0: it can produce a slightly different value and open a conflict set
    against the fact the first pass just wrote. A customer sees that.

    Losing a job to another worker is not an error and not logged as one:
    it is the mechanism working.
    """
    if not job_ids:
        return []
    claimable = (
        select(Job.id)
        .where(
            Job.id.in_(job_ids),
            or_(
                Job.status.in_([JobStatus.pending, JobStatus.failed]),
                and_(
                    Job.status == JobStatus.running,
                    Job.started_at < func.now() - STALE_CLAIM_AFTER,
                ),
            ),
        )
        .order_by(Job.created_at)
        .with_for_update(skip_locked=True)
    )
    claimed_ids = (
        (
            await session.execute(
                update(Job)
                .where(Job.id.in_(claimable.scalar_subquery()))
                .values(status=JobStatus.running, started_at=func.now())
                .returning(Job.id)
            )
        )
        .scalars()
        .all()
    )
    if not claimed_ids:
        return []
    return list(
        (
            await session.execute(
                select(Job).where(Job.id.in_(claimed_ids)).order_by(Job.created_at)
            )
        )
        .scalars()
        .all()
    )


async def run_pending_consolidations(
    session: AsyncSession,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Process pending (and previously failed) consolidate jobs.

    Returns the number of jobs completed successfully. Failed jobs keep their
    payload (plus an "error" key) and are retried on the next run.
    """
    candidates = (
        (
            await session.execute(
                select(Job.id)
                .where(
                    Job.kind == "consolidate",
                    or_(
                        Job.status.in_([JobStatus.pending, JobStatus.failed]),
                        and_(
                            Job.status == JobStatus.running,
                            Job.started_at < func.now() - STALE_CLAIM_AFTER,
                        ),
                    ),
                )
                .order_by(Job.created_at)
            )
        )
        .scalars()
        .all()
    )
    jobs = await claim_jobs(session, list(candidates))
    return await _run_jobs(session, jobs, extractor, embedder)


async def run_pending_consolidations_for_subject(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Console Playground "Write": process only the pending jobs touching
    THIS project/subject, leaving every other tenant's pending work alone
    — unlike POST /v1/consolidate (dev/ops, unscoped, no rate limit by
    design), this is reachable with a normal customer hk_ key, so it must
    never let one caller's Playground clicks process (or repeatedly re-walk)
    the entire global queue."""
    subject_event_ids = {
        row[0]
        for row in (
            await session.execute(
                select(Event.id).where(
                    Event.project_id == project_id, Event.subject_id == subject_id
                )
            )
        ).all()
    }
    if not subject_event_ids:
        return 0

    candidate_jobs = (
        (
            await session.execute(
                select(Job)
                .where(
                    Job.kind == "consolidate",
                    or_(
                        Job.status.in_([JobStatus.pending, JobStatus.failed]),
                        and_(
                            Job.status == JobStatus.running,
                            Job.started_at < func.now() - STALE_CLAIM_AFTER,
                        ),
                    ),
                )
                .order_by(Job.created_at)
            )
        )
        .scalars()
        .all()
    )
    # Filter to this subject FIRST, then claim: claiming a job only to
    # discover it belongs to another subject would take it away from the
    # worker that should have it. Claimed jobs are then processed with
    # only_event_ids (B12): a mixed batch's other subjects stay queued,
    # never extracted under this caller's scope.
    jobs = await claim_jobs(
        session,
        [
            job.id
            for job in candidate_jobs
            if subject_event_ids.intersection(
                uuid.UUID(e) for e in job.payload.get("event_ids", [])
            )
        ],
    )
    return await _run_jobs(
        session, jobs, extractor, embedder, only_event_ids=subject_event_ids
    )


async def _run_jobs(
    session: AsyncSession,
    jobs: list[Job],
    extractor: Extractor | None,
    embedder: Embedder | None,
    *,
    only_event_ids: set[uuid.UUID] | None = None,
) -> int:
    extractor = extractor or get_extractor()
    embedder = embedder or get_embedder()
    done = 0
    for job in jobs:
        try:
            # Savepoint per job: a failure rolls back only this job's writes.
            async with session.begin_nested():
                result = await _process_job(
                    session, job, extractor, embedder, only_event_ids=only_event_ids
                )
                if only_event_ids is not None:
                    remaining = [
                        e
                        for e in job.payload.get("event_ids", [])
                        if uuid.UUID(e) not in only_event_ids
                    ]
                else:
                    remaining = []
                if remaining:
                    # B12: other subjects' events stay queued for the global
                    # worker/ops run -- back to pending (the claim flipped
                    # this job to running), payload narrowed to what is left.
                    # The partial result is kept for observability; the next
                    # run overwrites it with its own full counts.
                    job.payload = {
                        **job.payload,
                        "event_ids": remaining,
                        "result": result,
                    }
                    job.status = JobStatus.pending
                    job.started_at = None
                else:
                    job.status = JobStatus.done
                    job.finished_at = datetime.now(timezone.utc)
                    job.payload = {**job.payload, "result": result}
            done += 1
            metrics.increment("consolidator.job.done")
        except Exception as exc:  # provider or DB failure: job stays replayable
            logger.exception("consolidator: job %s failed", job.id)
            job.status = JobStatus.failed
            job.finished_at = datetime.now(timezone.utc)
            job.payload = {**job.payload, "error": f"{type(exc).__name__}: {exc}"}
            await session.flush()
            metrics.increment("consolidator.job.failed")
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                # The provider's rate limit is a per-minute token/request
                # budget: every remaining job in THIS batch would hit the
                # exact same wall within the same second (confirmed via a
                # real Groq free-tier run — a 19-job batch burned its 6000
                # tokens/min budget on job #3, then failed jobs #4-19
                # instantly, wasting the caller's retry/backoff entirely).
                # Stop the batch here; the caller's own retry already backs
                # off between /v1/consolidate calls, which only helps if we
                # don't immediately re-exhaust the budget on the next job.
                break
    return done
