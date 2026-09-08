"""POST /v1/retention/purge — dev/ops endpoint (sprint 16), same shape as
POST /v1/consolidate: manual trigger, not the console directly. The worker
loop also runs the same purge on its own cadence
(HAKI_RETENTION_PURGE_INTERVAL_SECONDS, app/worker.py), so opt-in orgs are
purged even when nobody calls this. Only organizations with retention_days
set are touched (app/retention.py documents exactly what is and isn't
purged, and why).

Security audit H7 (8 sept): like /v1/consolidate, this runs an OPS session
(no RLS) and its blast radius is every opt-in organization — a customer
hk_ key must not reach it. Admin-gated when HAKI_ADMIN_KEY is set, with
the same self-hosted exception: without an admin key the endpoint stays
open, because on the documented single-server bootstrap "every org" and
"my org" are the same thing.
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import constant_time_bearer_match
from app.config import settings
from app.db import get_session_ops
from app.errors import ApiError
from app.retention import purge_all_organizations

router = APIRouter()


@router.post("/retention/purge")
async def purge(
    request: Request, session: AsyncSession = Depends(get_session_ops)
) -> dict[str, int]:
    if settings.admin_key and not constant_time_bearer_match(
        request.headers.get("authorization"), settings.admin_key
    ):
        raise ApiError(
            type="unauthorized",
            message=(
                "POST /v1/retention/purge erases expired data across every "
                "opt-in organization and requires the admin key."
            ),
            field="Authorization",
            status_code=401,
        )
    totals = await purge_all_organizations(session)
    await session.commit()
    return totals
