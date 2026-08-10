import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger
from app.db import get_session
from app.errors import ApiError
from app.models import ConflictSet, FactStatus
from app.schemas import (
    ConflictListResponse,
    ConflictSetOut,
    ResolveConflictRequest,
    ResolveConflictResponse,
)

router = APIRouter()


@router.get("/conflicts", response_model=ConflictListResponse)
async def list_conflicts(
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> ConflictListResponse:
    """Open conflict sets of a project (optionally narrowed to one subject)."""
    if not project_id:
        raise ApiError(
            type="missing_scope",
            message="project_id query parameter is required",
            field="project_id",
        )
    stmt = (
        select(ConflictSet)
        .where(ConflictSet.project_id == project_id, ConflictSet.status == "open")
        .order_by(ConflictSet.created_at)
    )
    if subject_id:
        stmt = stmt.where(ConflictSet.subject_id == subject_id)
    conflicts = (await session.execute(stmt)).scalars().all()
    oldest_open_seconds = None
    if conflicts:
        # `conflicts` is already ordered by created_at ascending, so the
        # first row is the oldest still-open one.
        age = datetime.now(timezone.utc) - conflicts[0].created_at
        oldest_open_seconds = age.total_seconds()
    return ConflictListResponse(
        conflicts=[ConflictSetOut.model_validate(c) for c in conflicts],
        open_count=len(conflicts),
        oldest_open_seconds=oldest_open_seconds,
    )


@router.post("/conflicts/{conflict_id}/resolve", response_model=ResolveConflictResponse)
async def resolve_conflict(
    conflict_id: uuid.UUID,
    request: ResolveConflictRequest,
    session: AsyncSession = Depends(get_session),
) -> ResolveConflictResponse:
    """Human resolution of a conflict set (PRD — conflict sets).

    The kept fact becomes `active`; every OTHER fact of the set becomes
    `superseded` with supersedes_id pointing at the kept one; the set is
    marked resolved. From then on /v1/context serves the kept fact
    normally (it is no longer blocked by conflict_open).
    """
    conflict = await session.get(ConflictSet, conflict_id)
    if conflict is None or conflict.project_id != request.project_id:
        # Same 404 as an unknown id: no cross-project leak.
        raise ApiError(
            type="conflict_not_found",
            message=f"Conflict set {conflict_id} does not exist",
            field="conflict_id",
            status_code=404,
        )
    if conflict.status != "open":
        raise ApiError(
            type="conflict_already_resolved",
            message=f"Conflict set {conflict_id} is already {conflict.status}",
            field="conflict_id",
            status_code=409,
        )
    if request.keep_fact_id not in list(conflict.fact_ids):
        raise ApiError(
            type="fact_not_in_conflict",
            message=f"Fact {request.keep_fact_id} does not belong to conflict set {conflict_id}",
            field="keep_fact_id",
        )

    # Serialize with the consolidator's write phase (and with a concurrent
    # resolve of the same subject): activation must never race a
    # consolidation creating an active fact for the same predicate — the
    # partial unique index (migration 0012) would turn that race into a
    # 500 instead of a clean sequential outcome.
    await ledger.acquire_subject_write_lock(
        session, project_id=conflict.project_id, subject_id=conflict.subject_id
    )
    await session.refresh(conflict)
    if conflict.status != "open":
        raise ApiError(
            type="conflict_already_resolved",
            message=f"Conflict set {conflict_id} is already {conflict.status}",
            field="conflict_id",
            status_code=409,
        )

    superseded: list[uuid.UUID] = []
    for fact_id in conflict.fact_ids:
        if fact_id == request.keep_fact_id:
            fact = await ledger.get_fact(session, fact_id)
            if fact.status is not FactStatus.active:
                # candidate/disputed -> active (Ledger lifecycle).
                fact = await ledger.transition_fact_status(
                    session, fact_id, FactStatus.active
                )
            continue
        fact = await ledger.get_fact(session, fact_id)
        if fact.status is FactStatus.superseded:
            superseded.append(fact_id)
            continue
        fact = await ledger.transition_fact_status(
            session, fact_id, FactStatus.superseded
        )
        fact.supersedes_id = request.keep_fact_id
        superseded.append(fact_id)

    conflict.status = "resolved"
    conflict.resolved_at = datetime.now(timezone.utc)
    await session.commit()
    return ResolveConflictResponse(
        conflict_id=conflict.id,
        status=conflict.status,
        kept_fact_id=request.keep_fact_id,
        superseded_fact_ids=superseded,
        resolved_at=conflict.resolved_at,
    )
