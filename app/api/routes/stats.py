"""GET /v1/stats/overview and GET /v1/stats/health — real numbers for the
console Overview page.

Every figure here is derived from data already written for other reasons
(facts, events, context_traces, conflict_sets, consolidation job results) —
nothing is a guess or a placeholder. `recall_p50_ms`/`recall_p99_ms`/
`hit_rate`/every health metric are None (not 0) when there is no data yet
to measure, so the console can render "no data yet" instead of a
misleading zero or a fake 100.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import ApiError
from app.models import ConflictSet, ContextTrace, Event, Fact, FactStatus
from app.schemas.stats import (
    DailyCount,
    HealthComponent,
    HealthStatsResponse,
    OverviewStatsResponse,
)

router = APIRouter()

_WEEK = 7
_HEALTH_WINDOW_DAYS = 30
# One open conflict costs 10 points of its component (floor 0). Deliberate,
# documented heuristic — not a measured/calibrated constant.
_CONFLICT_PENALTY_PER_OPEN = 0.1


@router.get("/stats/overview", response_model=OverviewStatsResponse)
async def overview_stats(
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> OverviewStatsResponse:
    if not project_id:
        raise ApiError(
            type="missing_scope",
            message="project_id query parameter is required",
            field="project_id",
        )

    fact_scope = [Fact.project_id == project_id, Fact.status == FactStatus.active]
    trace_scope = [ContextTrace.project_id == project_id]
    event_scope = [Event.project_id == project_id]
    if subject_id:
        fact_scope.append(Fact.subject_id == subject_id)
        trace_scope.append(ContextTrace.subject_id == subject_id)
        event_scope.append(Event.subject_id == subject_id)

    active_facts = await session.scalar(
        select(func.count()).select_from(Fact).where(*fact_scope)
    )

    since = datetime.now(timezone.utc) - timedelta(days=_WEEK)
    # The SAME expression object is reused in both select() and group_by():
    # calling func.date_trunc("day", ...) twice binds "day" as two distinct
    # parameters, and Postgres then refuses the query ("must appear in the
    # GROUP BY clause") because it can't see they're identical (found live
    # running this test against real Postgres, not a guess).
    day_bucket = func.date_trunc("day", Event.occurred_at)
    day_rows = dict(
        (
            await session.execute(
                select(day_bucket, func.count())
                .where(*event_scope, Event.occurred_at >= since)
                .group_by(day_bucket)
            )
        ).all()
    )
    events_this_week = []
    for i in range(_WEEK - 1, -1, -1):
        day = (datetime.now(timezone.utc) - timedelta(days=i)).date()
        count = next(
            (c for d, c in day_rows.items() if d.date() == day), 0
        )
        events_this_week.append(DailyCount(date=day.isoformat(), count=count))

    recall_count = await session.scalar(
        select(func.count()).select_from(ContextTrace).where(*trace_scope)
    )
    p50 = p99 = injection_rate = None
    if recall_count:
        p50, p99 = (
            await session.execute(
                select(
                    func.percentile_cont(0.5).within_group(ContextTrace.duration_ms),
                    func.percentile_cont(0.99).within_group(ContextTrace.duration_ms),
                ).where(*trace_scope, ContextTrace.duration_ms.is_not(None))
            )
        ).one()
        hits = await session.scalar(
            select(func.count())
            .select_from(ContextTrace)
            .where(*trace_scope, ContextTrace.fact_count > 0)
        )
        injection_rate = hits / recall_count

    context_tokens_served = await session.scalar(
        select(func.coalesce(func.sum(ContextTrace.token_count), 0)).where(*trace_scope)
    )

    return OverviewStatsResponse(
        active_facts=active_facts or 0,
        events_this_week=events_this_week,
        recall_p50_ms=p50,
        recall_p99_ms=p99,
        hit_rate=injection_rate,
        injection_rate=injection_rate,
        context_tokens_served=context_tokens_served or 0,
        recall_count=recall_count or 0,
    )


@router.get("/stats/health", response_model=HealthStatsResponse)
async def health_stats(
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> HealthStatsResponse:
    """Memory-health metrics over a bounded 30-day window.

    Every figure is derived from rows already written for other reasons
    (facts, events, context_traces, conflict_sets, consolidation job
    results persisted in jobs.payload["result"]). A metric whose
    denominator is empty is None, never a fake 0 or 100; the health score
    is a weighted mean over the MEASURED components only and is None when
    nothing is measurable. write_rejection_rate is project-wide (job
    results are batch-level, not per-subject). `jobs` is not under RLS:
    the payload->>'project_id' filter below is the sole isolation barrier
    on that table (the auth middleware's project_id-vs-key check is a
    second, independent barrier on top).
    """
    if not project_id:
        raise ApiError(
            type="missing_scope",
            message="project_id query parameter is required",
            field="project_id",
        )
    since = datetime.now(timezone.utc) - timedelta(days=_HEALTH_WINDOW_DAYS)

    trace_scope = [ContextTrace.project_id == project_id, ContextTrace.created_at >= since]
    fact_scope = [Fact.project_id == project_id]
    active_fact_scope = [
        Fact.project_id == project_id,
        Fact.status == FactStatus.active,
    ]
    event_scope = [Event.project_id == project_id]
    conflict_scope = [ConflictSet.project_id == project_id, ConflictSet.status == "open"]
    if subject_id:
        trace_scope.append(ContextTrace.subject_id == subject_id)
        fact_scope.append(Fact.subject_id == subject_id)
        active_fact_scope.append(Fact.subject_id == subject_id)
        event_scope.append(Event.subject_id == subject_id)
        conflict_scope.append(ConflictSet.subject_id == subject_id)

    # (a) injection_rate: packet->'facts' (not fact_count, NULL on pre-0014
    # traces) is present on every trace, so it never silently miscounts.
    traces_in_window = await session.scalar(
        select(func.count()).select_from(ContextTrace).where(*trace_scope)
    )
    packets_with_facts = await session.scalar(
        select(func.count())
        .select_from(ContextTrace)
        .where(*trace_scope, func.jsonb_array_length(ContextTrace.packet["facts"]) > 0)
    )
    traces_in_window = traces_in_window or 0
    packets_with_facts = packets_with_facts or 0
    injection_rate = packets_with_facts / traces_in_window if traces_in_window else None

    # (b) fact_density: informational only, never fed into the score (no
    # universal target value).
    active_facts = await session.scalar(
        select(func.count()).select_from(Fact).where(*active_fact_scope)
    )
    total_facts = await session.scalar(
        select(func.count()).select_from(Fact).where(*fact_scope)
    )
    events_total = await session.scalar(
        select(func.count()).select_from(Event).where(*event_scope)
    )
    active_facts = active_facts or 0
    total_facts = total_facts or 0
    events_total = events_total or 0
    fact_density = active_facts / events_total if events_total else None

    # (c) write_rejection_rate + breakdown, from consolidation job results
    # already persisted in jobs.payload["result"] — batch-level, so
    # subject_id does NOT narrow this one (documented, not simulated).
    # Every processed candidate increments exactly one of created/
    # conflicts/duplicates/reinforced/quarantined/rejected (superseded is a
    # side effect of a candidate already counted in created — never added
    # to the denominator).
    job_totals = (
        await session.execute(
            text(
                """
                SELECT
                    COALESCE(SUM((payload #>> '{result,created}')::int), 0) AS created,
                    COALESCE(SUM((payload #>> '{result,conflicts}')::int), 0) AS conflicts,
                    COALESCE(SUM((payload #>> '{result,duplicates}')::int), 0) AS duplicates,
                    COALESCE(SUM((payload #>> '{result,reinforced}')::int), 0) AS reinforced,
                    COALESCE(SUM((payload #>> '{result,quarantined}')::int), 0) AS quarantined,
                    COALESCE(SUM((payload #>> '{result,rejected}')::int), 0) AS rejected
                FROM jobs
                WHERE kind = 'consolidate'
                  AND status = 'done'
                  AND payload -> 'result' IS NOT NULL
                  AND payload ->> 'project_id' = :project_id
                  AND created_at >= :since
                """
            ),
            {"project_id": project_id, "since": since},
        )
    ).one()
    rejected_total = job_totals.rejected
    candidates_total = (
        job_totals.created
        + job_totals.conflicts
        + job_totals.duplicates
        + job_totals.reinforced
        + job_totals.quarantined
        + rejected_total
    )
    write_rejection_rate = rejected_total / candidates_total if candidates_total else None

    breakdown_rows = (
        await session.execute(
            text(
                """
                SELECT r.key, SUM((r.value)::int) AS n
                FROM jobs
                CROSS JOIN LATERAL
                    jsonb_each_text(payload #> '{result,rejected_with_reason}') AS r(key, value)
                WHERE kind = 'consolidate' AND status = 'done'
                  AND payload -> 'result' IS NOT NULL
                  AND payload ->> 'project_id' = :project_id
                  AND created_at >= :since
                GROUP BY r.key
                """
            ),
            {"project_id": project_id, "since": since},
        )
    ).all()
    rejection_breakdown = {key: n for key, n in breakdown_rows if n > 0}
    unclassified = rejected_total - sum(rejection_breakdown.values())
    if unclassified > 0:
        rejection_breakdown["unclassified"] = unclassified

    # (d) contradiction_leakage: % of served packets (window) containing a
    # fact that was ALREADY superseded at the time it was served. No
    # dedicated system timestamp exists for "moment of supersession" — it
    # is reconstructed as the min of: the winning fact's recorded_from
    # (consolidator path: winner created + loser transitioned in the same
    # transaction) and the resolved_at of any resolved conflict_set
    # containing the fact (resolution path: the LOSER points at the
    # winner, so supersedes_id is unusable in that direction there).
    # Documented as a LOWER BOUND: a fact that cycled superseded ->
    # disputed -> active is correctly no longer counted; a fact erased via
    # the right to be forgotten drops out of the join entirely (erasure
    # takes precedence); the consolidator path approximates the instant by
    # the winner's recorded_from (same transaction, millisecond-scale
    # skew).
    subject_clause = "AND ct.subject_id = :subject_id" if subject_id else ""
    leakage_params: dict[str, object] = {"project_id": project_id, "since": since}
    if subject_id:
        leakage_params["subject_id"] = subject_id
    leaked_packets = (
        await session.scalar(
            text(
                f"""
                WITH served AS (
                    SELECT ct.id AS trace_id,
                           ct.created_at,
                           (fact_elem ->> 'id')::uuid AS fact_id
                    FROM context_traces ct
                    CROSS JOIN LATERAL jsonb_array_elements(ct.packet -> 'facts') AS fact_elem
                    WHERE ct.project_id = :project_id
                      AND ct.created_at >= :since
                      {subject_clause}
                ),
                superseded_moments AS (
                    SELECT fact_id, MIN(at) AS superseded_at FROM (
                        SELECT s.supersedes_id AS fact_id, s.recorded_from AS at
                        FROM facts s
                        WHERE s.project_id = :project_id AND s.supersedes_id IS NOT NULL
                        UNION ALL
                        SELECT unnest(c.fact_ids) AS fact_id, c.resolved_at AS at
                        FROM conflict_sets c
                        WHERE c.project_id = :project_id
                          AND c.status = 'resolved'
                          AND c.resolved_at IS NOT NULL
                    ) u
                    GROUP BY fact_id
                )
                SELECT COUNT(DISTINCT served.trace_id)
                FROM served
                JOIN facts f
                  ON f.id = served.fact_id AND f.status = 'superseded'
                JOIN superseded_moments sm
                  ON sm.fact_id = served.fact_id AND sm.superseded_at <= served.created_at
                """
            ),
            leakage_params,
        )
    ) or 0
    contradiction_leakage = leaked_packets / packets_with_facts if packets_with_facts else None

    # (e) staleness: wired to the volatility horizon column when it lands
    # (typology-volatility); None until then, never a fake 0.
    staleness = None

    # (f) open_conflicts: same definition as GET /v1/conflicts, a direct
    # count — no internal HTTP call.
    open_conflicts = await session.scalar(
        select(func.count()).select_from(ConflictSet).where(*conflict_scope)
    )
    open_conflicts = open_conflicts or 0

    # (g) health_score: weighted mean over MEASURED components only.
    contradiction_integrity = (
        1 - contradiction_leakage if contradiction_leakage is not None else None
    )
    conflict_hygiene = (
        max(0.0, 1.0 - open_conflicts * _CONFLICT_PENALTY_PER_OPEN) if total_facts else None
    )
    components = [
        HealthComponent(name="injection", value=injection_rate, weight=0.30),
        HealthComponent(
            name="contradiction_integrity", value=contradiction_integrity, weight=0.30
        ),
        HealthComponent(name="conflict_hygiene", value=conflict_hygiene, weight=0.20),
        HealthComponent(name="staleness", value=staleness, weight=0.20),
    ]
    measured = [c for c in components if c.value is not None]
    health_score = (
        round(100 * sum(c.weight * c.value for c in measured) / sum(c.weight for c in measured), 1)
        if measured
        else None
    )

    return HealthStatsResponse(
        window_days=_HEALTH_WINDOW_DAYS,
        injection_rate=injection_rate,
        fact_density=fact_density,
        write_rejection_rate=write_rejection_rate,
        rejection_breakdown=rejection_breakdown,
        contradiction_leakage=contradiction_leakage,
        staleness=staleness,
        open_conflicts=open_conflicts,
        health_score=health_score,
        components=components,
        traces_in_window=traces_in_window,
        packets_with_facts=packets_with_facts,
        leaked_packets=leaked_packets,
        candidates_total=candidates_total,
        active_facts=active_facts,
        events_total=events_total,
    )
