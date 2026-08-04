"""Memory Consolidator (PRD semaines 3-4).

Takes pending `consolidate` jobs, extracts memory candidates through the
configured LLM provider, validates them, and applies the fact lifecycle:

- same value as an existing fact for the same subject+predicate -> duplicate
  (no new row; this is also what makes reprocessing a job idempotent);
- action "supersede" -> the current active fact becomes `superseded` (Ledger
  transition), the new fact becomes `active` with `supersedes_id`;
- action "create" with a different value than an active fact of the same
  predicate -> both facts enter an OPEN conflict set; the new fact stays
  `candidate` and is never served while the conflict is open;
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

"Same predicate" above is resolved by `_resolve_existing_fact`: exact string
match first, then a semantic fallback (cosine distance on the already-
computed fact embedding) when no exact match exists. An LLM-generated
predicate is not a reliable join key on natural language on its own — this
is the write-time adjudication step, decoupled from extraction, that keeps a
same-concept update from silently coexisting with the fact it was meant to
replace.

Guarantees:
- a provider/DB exception fails the job (`failed`, error in payload) and
  NEVER deletes or alters the source events; the job stays replayable —
  failed jobs are picked up again on the next worker run;
- idempotence: dedup is content-based (same subject_id + predicate +
  canonical value among non-deleted facts => duplicate), so processing the
  same job twice never creates duplicate facts. Chosen over a unique
  constraint because several events may legitimately re-assert the same
  fact and we want them counted as duplicates, not rejected by SQL.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import metrics
from app.ledger.core import create_fact, transition_fact_status
from app.context import episode_text
from app.models import ConflictSet, ContextTrace, Event, Fact, FactStatus, Job, JobStatus
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


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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


async def _active_fact(
    session: AsyncSession, *, project_id: str, subject_id: str, predicate: str
) -> Fact | None:
    stmt = select(Fact).where(
        Fact.project_id == project_id,
        Fact.subject_id == subject_id,
        Fact.predicate == predicate,
        Fact.status == FactStatus.active,
    )
    return (await session.execute(stmt)).scalars().first()


async def _is_duplicate(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    predicate: str,
    value: dict[str, Any],
) -> bool:
    """Content-based dedup: an identical value already memorized (not deleted)."""
    stmt = select(Fact).where(
        Fact.project_id == project_id,
        Fact.subject_id == subject_id,
        Fact.predicate == predicate,
        Fact.status != FactStatus.deleted,
    )
    canonical = _canonical(value)
    for fact in (await session.execute(stmt)).scalars().all():
        if _canonical(fact.value) == canonical:
            return True
    return False


async def _resolve_existing_fact(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    predicate: str,
    embedding: list[float],
) -> Fact | None:
    """Find the active fact a candidate should be adjudicated against.

    This is the "adjudicate against the existing" step, decoupled from
    extraction (Control-Plane Placement, arXiv:2606.15903): exact predicate
    match first (fast path — extraction was lexically consistent, the common
    case). If none, fall back to a semantic match among the subject's active
    facts, using the candidate's already-computed embedding — no extra LLM
    call. This closes the gap where the extractor recognizes an update but
    mints a slightly different predicate string than the one already on
    file (e.g. "personal_best_5k" vs "goal_personal_best_time"), which
    previously left both facts active in parallel with the stale one still
    served as current.
    """
    exact = await _active_fact(
        session, project_id=project_id, subject_id=subject_id, predicate=predicate
    )
    if exact is not None:
        return exact

    stmt = (
        select(Fact, Fact.embedding.cosine_distance(embedding).label("distance"))
        .where(
            Fact.project_id == project_id,
            Fact.subject_id == subject_id,
            Fact.status == FactStatus.active,
            Fact.embedding.is_not(None),
        )
        .order_by(Fact.embedding.cosine_distance(embedding))
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None or row.distance > SEMANTIC_MATCH_MAX_DISTANCE:
        return None
    return row[0]


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
    if row is not None and row.distance <= ANTI_ECHO_MAX_DISTANCE:
        return "echo_of_context"
    return None


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
    if await _is_duplicate(
        session,
        project_id=event.project_id,
        subject_id=event.subject_id,
        predicate=candidate.predicate,
        value=candidate.value,
    ):
        result["duplicates"] += 1
        return

    target_predicate = candidate.supersedes_predicate or candidate.predicate
    existing = await _resolve_existing_fact(
        session,
        project_id=event.project_id,
        subject_id=event.subject_id,
        predicate=target_predicate,
        embedding=embedding,
    )

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

    fact = await create_fact(
        session,
        org_id=event.org_id,
        project_id=event.project_id,
        subject_id=event.subject_id,
        predicate=candidate.predicate,
        value=value,
        subject_type=event.subject_type,
        agent_id=event.agent_id,
        qualifiers=candidate.qualifiers,
        confidence=candidate.confidence,
        valid_from=event.occurred_at,
        source_event_ids=[event.id],
    )
    fact.embedding = embedding
    fact.search_text = _search_text(candidate.predicate, value)

    if candidate.action == "supersede":
        if existing is not None:
            # Ledger transition active -> superseded; the replaced fact stays
            # in history and is never served as current again.
            await transition_fact_status(session, existing.id, FactStatus.superseded)
            existing.valid_to = event.occurred_at
            fact.supersedes_id = existing.id
            result["superseded"] += 1
        await transition_fact_status(session, fact.id, FactStatus.active)
        result["created"] += 1
        return

    # action == "create"
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
                reason=(
                    f"predicate '{candidate.predicate}': "
                    f"{_canonical(existing.value)} vs {_canonical(candidate.value)}"
                ),
            )
            session.add(conflict)
        else:
            conflict.fact_ids = [*conflict.fact_ids, fact.id]
        await session.flush()
        result["conflicts"] += 1
        return

    if existing is not None:
        # Same predicate, same value (e.g. not caught above because the
        # existing fact is active): duplicate.
        result["duplicates"] += 1
        return

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
    session: AsyncSession, job: Job, extractor: Extractor, embedder: Embedder
) -> dict[str, Any]:
    event_ids = [uuid.UUID(e) for e in job.payload.get("event_ids", [])]
    events = (
        (await session.execute(select(Event).where(Event.id.in_(event_ids))))
        .scalars()
        .all()
    )

    result: dict[str, Any] = {
        "created": 0,
        "superseded": 0,
        "conflicts": 0,
        "duplicates": 0,
        "rejected": 0,
        "rejected_with_reason": {reason: 0 for reason in REJECT_REASONS},
    }

    # Extraction per event: exact provenance (source_event_ids) for every
    # candidate. The subject's currently active facts are passed along so
    # the provider can emit "supersede" on a change of mind instead of
    # piling up contradictions. A provider exception propagates -> job
    # failed, events intact.
    candidates: list[tuple[Event, Any]] = []
    for event in events:
        active_facts = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.project_id == event.project_id,
                        Fact.subject_id == event.subject_id,
                        Fact.status == FactStatus.active,
                    )
                )
            )
            .scalars()
            .all()
        )
        existing = [
            {
                "predicate": fact.predicate,
                "value": fact.value,
                "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
            }
            for fact in active_facts
        ]
        for raw in await extractor.extract_facts([event], existing=existing):
            candidates.append((event, raw))

    # Validate everything before embedding: an invalid candidate is rejected
    # and logged, never crashes the batch. A candidate the provider itself
    # marked action="reject" (write gate M1) is a well-formed observation the
    # extractor deliberately screened out (echo, noise, unsourced
    # inference...) — it is counted the same way and never reaches
    # embedding/application.
    to_apply: list[tuple[Event, ExtractedFact]] = []
    for event, raw in candidates:
        try:
            candidate = ExtractedFact.model_validate(raw)
        except ValidationError as exc:
            # No reason code: the candidate never even parsed, so there is
            # no taxonomy to classify it against.
            result["rejected"] += 1
            logger.warning(
                "consolidator: rejected invalid candidate (job %s): %s", job.id, exc
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
        to_apply.append((event, candidate))

    texts = [_search_text(fact.predicate, fact.value) for _, fact in to_apply]
    embeddings = await embedder.embed(texts) if texts else []

    # Episodic memory (sprint 10): embed each processed event once (derived
    # data, re-computable — the only post-insert write allowed on events).
    # Events already embedded (replayed job) are skipped.
    unembedded = [event for event in events if event.embedding is None]
    if unembedded:
        event_embeddings = await embedder.embed(
            [episode_text(event.kind, event.payload) for event in unembedded]
        )
        for event, embedding in zip(unembedded, event_embeddings):
            event.embedding = embedding

    for (event, candidate), embedding in zip(to_apply, embeddings):
        # Anti-echo write gate (M1), post-validation: a candidate that only
        # reformulates a fact already SERVED to this subject in a recent
        # context packet is rejected here, before it can ever reach the
        # ledger — this is what stops a served fact from being echoed back,
        # re-extracted, and re-stored without bound.
        echo_reason = await _echo_reject_reason(
            session,
            project_id=event.project_id,
            subject_id=event.subject_id,
            embedding=embedding,
        )
        if echo_reason is not None:
            _record_rejection(result, echo_reason, job_id=job.id, source="anti-echo")
            continue
        await _apply_candidate(
            session, candidate, embedding=embedding, event=event, result=result
        )
    return result


async def run_pending_consolidations(
    session: AsyncSession,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Process pending (and previously failed) consolidate jobs.

    Returns the number of jobs completed successfully. Failed jobs keep their
    payload (plus an "error" key) and are retried on the next run.
    """
    extractor = extractor or get_extractor()
    embedder = embedder or get_embedder()
    jobs = (
        (
            await session.execute(
                select(Job)
                .where(
                    Job.kind == "consolidate",
                    Job.status.in_([JobStatus.pending, JobStatus.failed]),
                )
                .order_by(Job.created_at)
            )
        )
        .scalars()
        .all()
    )

    done = 0
    for job in jobs:
        try:
            # Savepoint per job: a failure rolls back only this job's writes.
            async with session.begin_nested():
                result = await _process_job(session, job, extractor, embedder)
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
