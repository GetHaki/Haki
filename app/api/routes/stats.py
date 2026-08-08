"""GET /v1/stats/overview — real numbers for the console Overview page.

Every figure here is derived from data already written for other reasons
(facts, events, context_traces) — nothing is a guess or a placeholder.
`recall_p50_ms`/`recall_p99_ms`/`hit_rate` are None (not 0) when there is
no recall history yet to measure, so the console can render "no data yet"
instead of a misleading zero.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import ApiError
from app.models import ContextTrace, Event, Fact, FactStatus
from app.schemas.stats import DailyCount, OverviewStatsResponse

router = APIRouter()

_WEEK = 7


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
    p50 = p99 = hit_rate = None
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
        hit_rate = hits / recall_count

    context_tokens_served = await session.scalar(
        select(func.coalesce(func.sum(ContextTrace.token_count), 0)).where(*trace_scope)
    )

    return OverviewStatsResponse(
        active_facts=active_facts or 0,
        events_this_week=events_this_week,
        recall_p50_ms=p50,
        recall_p99_ms=p99,
        hit_rate=hit_rate,
        context_tokens_served=context_tokens_served or 0,
        recall_count=recall_count or 0,
    )
