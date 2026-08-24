import uuid

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger, policy
from app.context import build_context, get_trace
from app.db import get_session
from app.errors import ApiError
from app.rate_limit import key_or_ip, limiter
from app.schemas import ContextRequest, ContextResponse, TraceResponse

router = APIRouter()


# Found by security review (16 aout): every other write/consolidate route on
# this router already had a rate limit (capture: 300/min, consolidate:
# 20/min) but this one didn't. build_context makes no LLM call (local
# embeddings only) so a leaked key can't run up a bill here -- but it IS an
# expensive DB query (hybrid scoring, multi-hop expansion), unthrottled, on
# every request. 120/minute is generous for real conversational use (one
# call per turn) while still bounding an application-level DoS from a
# single leaked key or IP.
@router.post("/context", response_model=ContextResponse)
@limiter.limit("120/minute", key_func=key_or_ip)
async def context(
    request: Request, body: ContextRequest, session: AsyncSession = Depends(get_session)
) -> ContextResponse:
    # Identity resolution (M4), read path: NEVER writes. An unknown alias is
    # a loud typed 404 — returning an empty "ok" packet here would be
    # exactly the silent cold-start bug this layer exists to kill.
    subject_id = body.subject_id
    if body.subject_alias is not None:
        alias_row, _ = await ledger.resolve_alias(
            session,
            project_id=body.project_id,
            alias_kind=body.subject_alias.kind,
            alias_value=body.subject_alias.value,
            register=False,
        )
        if alias_row is None:
            raise ApiError(
                type="alias_not_found",
                message="no subject is registered for this alias in this project",
                field="subject_alias",
                status_code=404,
            )
        subject_id = alias_row.canonical_subject_id

    # Policy Engine (rule 3): purpose recommended — warning, not an error.
    purpose_warning = policy.context_purpose_warning(
        purpose=body.purpose,
        project_id=body.project_id,
        subject_id=subject_id,
    )
    packet, token_count, trace_id = await build_context(
        session,
        project_id=body.project_id,
        subject_id=subject_id,
        query=body.query,
        purpose=body.purpose,
        budget_tokens=body.budget_tokens,
        extra_warnings=[purpose_warning] if purpose_warning else None,
        as_of=body.as_of,
        exclude_ids=body.exclude_ids,
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
