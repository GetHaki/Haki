from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger
from app.db import get_session
from app.errors import ApiError
from app.schemas import EventOut, TimelineResponse

router = APIRouter()


@router.get("/timeline", response_model=TimelineResponse)
async def timeline(
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> TimelineResponse:
    # Scope filtering is mandatory: never serve a cross-subject timeline.
    if not subject_id:
        raise ApiError(
            type="missing_scope",
            message="subject_id query parameter is required",
            field="subject_id",
        )
    if not project_id:
        raise ApiError(
            type="missing_scope",
            message="project_id query parameter is required",
            field="project_id",
        )
    events = await ledger.list_timeline(
        session, project_id=project_id, subject_id=subject_id
    )
    return TimelineResponse(events=[EventOut.model_validate(e) for e in events])
