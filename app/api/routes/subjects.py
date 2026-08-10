"""POST /v1/subjects/resolve and /v1/subjects/merge (M4 identity layer).

Resolution maps N channel identifiers to 1 canonical subject per project.
It is always requested by the CLIENT backend — the model never chooses a
scope, so the MCP tools expose neither endpoint nor any alias argument.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger
from app.db import get_session
from app.schemas import (
    MergeSubjectsRequest,
    MergeSubjectsResponse,
    ResolveSubjectRequest,
    ResolveSubjectResponse,
)

router = APIRouter()


@router.post("/subjects/resolve", response_model=ResolveSubjectResponse)
async def resolve_subject(
    request: ResolveSubjectRequest, session: AsyncSession = Depends(get_session)
) -> ResolveSubjectResponse:
    row, created = await ledger.resolve_alias(
        session,
        project_id=request.project_id,
        alias_kind=request.alias_kind,
        alias_value=request.alias_value,
        canonical_subject_id=request.canonical_subject_id,
        register=True,
    )
    await session.commit()
    return ResolveSubjectResponse(
        project_id=row.project_id,
        alias_kind=row.alias_kind,
        alias_value=row.alias_value,
        canonical_subject_id=row.canonical_subject_id,
        created=created,
        self_registered=created and request.canonical_subject_id is None,
    )


@router.post("/subjects/merge", response_model=MergeSubjectsResponse)
async def merge_subjects_endpoint(
    request: MergeSubjectsRequest, session: AsyncSession = Depends(get_session)
) -> MergeSubjectsResponse:
    receipt, counters = await ledger.merge_subjects(
        session,
        project_id=request.project_id,
        source_subject_id=request.source_subject_id,
        target_subject_id=request.target_subject_id,
    )
    await session.commit()
    return MergeSubjectsResponse(
        merge_id=receipt.id,
        project_id=receipt.project_id,
        source_subject_id=receipt.source_subject_id,
        target_subject_id=receipt.target_subject_id,
        **counters,
    )
