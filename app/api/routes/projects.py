"""GET /v1/projects and GET /v1/subjects (console "Projects & subjects" view).

Projects have no dedicated table: a project is just a project_id string
that some ApiKey of the caller's org was minted for (self-hosted/curl
callers can already have several — POST /v1/keys accepts any project_id).
Listing them means listing distinct ApiKey.project_id for the caller's own
org_id, which spans MULTIPLE projects — the normal request-scoped session
(SET LOCAL haki.project_id, one project only) would RLS-hide every project
except the caller's own, so this aggregation deliberately uses the same
RLS-bypass session as the dev/ops consolidate endpoint, with org_id read
straight off the resolved caller key (never client input) as the only
scope filter.

GET /v1/subjects stays single-project (the caller's own, enforced by
ApiKeyAuthMiddleware's scope binding) and uses the normal RLS-scoped
session like /v1/facts and /v1/timeline.
"""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import STATE_KEY
from app.db import get_session, get_session_ops
from app.errors import ApiError
from app.models import ApiKey, ContextTrace, Fact, FactStatus
from app.schemas.projects import (
    ProjectListResponse,
    ProjectOut,
    SubjectListResponse,
    SubjectOut,
)

router = APIRouter()


def _caller_key(request: Request) -> ApiKey:
    key = getattr(request.state, STATE_KEY, None)
    if key is None:
        raise ApiError(
            type="unauthorized",
            message="missing or invalid API key",
            field="Authorization",
            status_code=401,
        )
    return key


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    request: Request, session: AsyncSession = Depends(get_session_ops)
) -> ProjectListResponse:
    org_id = _caller_key(request).org_id

    project_ids = sorted(
        {
            row[0]
            for row in (
                await session.execute(
                    select(ApiKey.project_id).where(ApiKey.org_id == org_id).distinct()
                )
            ).all()
        }
    )

    projects: list[ProjectOut] = []
    for project_id in project_ids:
        active_facts = await session.scalar(
            select(func.count())
            .select_from(Fact)
            .where(Fact.project_id == project_id, Fact.status == FactStatus.active)
        )
        subjects = await session.scalar(
            select(func.count(func.distinct(Fact.subject_id))).where(
                Fact.project_id == project_id, Fact.status != FactStatus.deleted
            )
        )
        projects.append(
            ProjectOut(
                project_id=project_id,
                active_facts=active_facts or 0,
                subjects=subjects or 0,
            )
        )
    return ProjectListResponse(projects=projects)


@router.get("/subjects", response_model=SubjectListResponse)
async def list_subjects(
    project_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> SubjectListResponse:
    if not project_id:
        raise ApiError(
            type="missing_scope",
            message="project_id query parameter is required",
            field="project_id",
        )

    fact_rows = (
        await session.execute(
            select(
                Fact.subject_id,
                func.count().label("facts"),
                func.max(Fact.recorded_from).label("last_fact"),
            )
            .where(Fact.project_id == project_id, Fact.status != FactStatus.deleted)
            .group_by(Fact.subject_id)
        )
    ).all()

    recall_rows = dict(
        (
            await session.execute(
                select(ContextTrace.subject_id, func.count())
                .where(ContextTrace.project_id == project_id)
                .group_by(ContextTrace.subject_id)
            )
        ).all()
    )
    last_recall_rows = dict(
        (
            await session.execute(
                select(ContextTrace.subject_id, func.max(ContextTrace.created_at))
                .where(ContextTrace.project_id == project_id)
                .group_by(ContextTrace.subject_id)
            )
        ).all()
    )

    subjects = []
    for subject_id, facts, last_fact in fact_rows:
        last_recall = last_recall_rows.get(subject_id)
        last_seen = max(filter(None, [last_fact, last_recall])) if (last_fact or last_recall) else last_fact
        subjects.append(
            SubjectOut(
                subject_id=subject_id,
                facts=facts,
                recalls=recall_rows.get(subject_id, 0),
                last_seen=last_seen,
            )
        )
    subjects.sort(key=lambda s: s.last_seen, reverse=True)
    return SubjectListResponse(subjects=subjects)
