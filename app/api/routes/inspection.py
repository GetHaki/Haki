"""Read-only listings for the console: facts (all statuses) and recent traces.

Both endpoints are scoped like /v1/timeline: the project scope is mandatory
(and bound to the API key by the auth middleware), subject scope narrows it.
They mutate nothing.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import ApiError
from app.models import ContextTrace, Fact, FactStatus
from app.schemas import (
    FactListResponse,
    FactOut,
    TraceListResponse,
    TraceSummaryOut,
)

router = APIRouter()

FACTS_LIMIT = 200
TRACES_LIMIT = 50


@router.get("/facts", response_model=FactListResponse)
async def list_facts(
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    status: FactStatus | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> FactListResponse:
    """Facts of one subject, all statuses (active, superseded, disputed, …)."""
    if not project_id:
        raise ApiError(
            type="missing_scope",
            message="project_id query parameter is required",
            field="project_id",
        )
    if not subject_id:
        raise ApiError(
            type="missing_scope",
            message="subject_id query parameter is required",
            field="subject_id",
        )
    stmt = (
        select(Fact)
        .where(Fact.project_id == project_id, Fact.subject_id == subject_id)
        .order_by(Fact.recorded_from.desc())
        .limit(FACTS_LIMIT)
    )
    if status is not None:
        stmt = stmt.where(Fact.status == status)
    facts = (await session.execute(stmt)).scalars().all()
    return FactListResponse(facts=[FactOut.model_validate(f) for f in facts])


@router.get("/traces", response_model=TraceListResponse)
async def list_traces(
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> TraceListResponse:
    """Most recent context traces of a project (newest first, 50 max)."""
    if not project_id:
        raise ApiError(
            type="missing_scope",
            message="project_id query parameter is required",
            field="project_id",
        )
    stmt = (
        select(ContextTrace)
        .where(ContextTrace.project_id == project_id)
        .order_by(ContextTrace.created_at.desc())
        .limit(TRACES_LIMIT)
    )
    if subject_id:
        stmt = stmt.where(ContextTrace.subject_id == subject_id)
    traces = (await session.execute(stmt)).scalars().all()
    return TraceListResponse(
        traces=[TraceSummaryOut.model_validate(t) for t in traces]
    )
