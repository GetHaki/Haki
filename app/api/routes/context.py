import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app import policy
from app.context import build_context, get_trace
from app.db import get_session
from app.errors import ApiError
from app.schemas import ContextRequest, ContextResponse, TraceResponse

router = APIRouter()


@router.post("/context", response_model=ContextResponse)
async def context(
    request: ContextRequest, session: AsyncSession = Depends(get_session)
) -> ContextResponse:
    # Policy Engine (rule 3): purpose recommended — warning, not an error.
    purpose_warning = policy.context_purpose_warning(
        purpose=request.purpose,
        project_id=request.project_id,
        subject_id=request.subject_id,
    )
    packet, token_count, trace_id = await build_context(
        session,
        project_id=request.project_id,
        subject_id=request.subject_id,
        query=request.query,
        purpose=request.purpose,
        budget_tokens=request.budget_tokens,
        extra_warnings=[purpose_warning] if purpose_warning else None,
    )
    await session.commit()
    return ContextResponse(packet=packet, token_count=token_count, trace_id=trace_id)


@router.get("/inspect/{trace_id}", response_model=TraceResponse)
async def inspect(
    trace_id: uuid.UUID,
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> TraceResponse:
    # Scope is mandatory and checked against the trace: a trace never leaks
    # outside the (project_id, subject_id) it was created for.
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
    trace = await get_trace(
        session, trace_id=trace_id, project_id=project_id, subject_id=subject_id
    )
    return TraceResponse(
        trace_id=trace.id,
        project_id=trace.project_id,
        subject_id=trace.subject_id,
        query=trace.query,
        purpose=trace.purpose,
        packet=trace.packet,
        decisions=trace.decisions,
        token_count=trace.token_count,
        duration_ms=trace.duration_ms,
        stage_timings=trace.stage_timings,
        fact_count=trace.fact_count,
    )
