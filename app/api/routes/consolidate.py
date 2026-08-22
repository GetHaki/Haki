"""POST /v1/consolidate — dev/ops endpoint (sprint 3).

Triggers `run_pending_consolidations` synchronously: pending and failed
`consolidate` jobs are processed immediately instead of waiting for a worker
run. Useful for local development, end-to-end tests, self-hosted n8n flows
and the `haki verify` CLI scenario. V1: no rate limiting (documented as a
dev/ops endpoint, to be protected with auth when principals land).
"""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import constant_time_bearer_match
from app.config import settings
from app.consolidator import run_pending_consolidations, run_pending_consolidations_for_subject
from app.db import get_session, get_session_ops
from app.errors import ApiError
from app.rate_limit import key_or_ip, limiter

router = APIRouter()


@router.post("/consolidate")
async def consolidate(
    request: Request, session: AsyncSession = Depends(get_session_ops)
) -> dict[str, int]:
    """Process the pending consolidation queue ACROSS every project.

    Ops session without RLS context -- by design, and that is exactly why
    it is admin-gated (22 aout). Before, any valid customer `hk_` key
    reached it, and the auth middleware's project binding did not apply
    because this session carries no RLS context: one tenant could drain
    (and pay the LLM cost of) every other tenant's queue, and see nothing
    of it. The scoped, rate-limited `/consolidate/subject` below is the
    endpoint a customer key is meant to use.

    When HAKI_ADMIN_KEY is unset the endpoint stays open, exactly like
    /v1/keys: that is the documented self-hosted/local bootstrap, where
    "every project" and "my project" are the same thing. docs/DEPLOY.md
    already treats a set admin key as the signal for "this is a real
    deployment".
    """
    if settings.admin_key and not constant_time_bearer_match(
        request.headers.get("authorization"), settings.admin_key
    ):
        raise ApiError(
            type="unauthorized",
            message=(
                "POST /v1/consolidate processes every project's queue and "
                "requires the admin key. Use POST /v1/consolidate/subject "
                "for a single subject with a normal key."
            ),
            field="Authorization",
            status_code=401,
        )
    processed = await run_pending_consolidations(session)
    await session.commit()
    return {"processed": processed}


@router.post("/consolidate/subject")
# Customer-facing (normal hk_ key, normal RLS-scoped session) — unlike
# POST /v1/consolidate this only touches ONE project/subject's pending
# jobs, so it's safe to rate-limit and expose to the console's Playground
# "Write" panel: a real, synchronous-feeling extraction instead of asking
# the user to trust that a background worker will get to it eventually.
@limiter.limit("20/minute", key_func=key_or_ip)
async def consolidate_subject(
    request: Request,
    project_id: str | None = Query(default=None),
    subject_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
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
    processed = await run_pending_consolidations_for_subject(
        session, project_id=project_id, subject_id=subject_id
    )
    await session.commit()
    return {"processed": processed}
