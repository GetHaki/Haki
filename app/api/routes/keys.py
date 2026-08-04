"""API key management (sprint 6): create, list (masked), revoke.

Access rules, deliberately simple for V1 (documented in the README):

- `HAKI_ADMIN_KEY` set   -> every operation requires
  `Authorization: Bearer <HAKI_ADMIN_KEY>` (admin mode);
- unset                  -> bootstrap: the FIRST key creation is free (the
  api_keys table is empty); afterwards a valid `hk_` key manages the keys
  of ITS OWN project (create bound to its project, list and revoke scoped
  to it — a revoke on another project's key returns the same 404 as an
  unknown id, no cross-project leak).

The clear key (`hk_...`) is returned exactly once, at creation. Listings
only ever show the 8-char prefix.
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import bearer_token, generate_key, hash_key, resolve_api_key
from app.config import settings
from app.db import get_session
from app.errors import ApiError
from app.models import ApiKey
from app.schemas import (
    CreateKeyRequest,
    KeyCreatedResponse,
    KeyListResponse,
    KeyOut,
    KeyRevokedResponse,
)

router = APIRouter()


def _unauthorized() -> ApiError:
    return ApiError(
        type="unauthorized",
        message="missing or invalid credentials for key management",
        field="Authorization",
        status_code=401,
    )


async def _caller(request: Request) -> tuple[bool, ApiKey | None]:
    """(is_admin, valid caller key or None)."""
    if settings.admin_key:
        expected = f"Bearer {settings.admin_key}"
        return request.headers.get("authorization") == expected, None
    token = bearer_token(list(request.headers.raw))
    return False, await resolve_api_key(token)


@router.post("/keys", response_model=KeyCreatedResponse, status_code=201)
async def create_key(
    body: CreateKeyRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> KeyCreatedResponse:
    is_admin, caller = await _caller(request)
    if settings.admin_key and not is_admin:
        raise _unauthorized()

    if is_admin:
        org_id, project_id = body.org_id, body.project_id
    elif caller is not None:
        # A key creates keys for its own scope only (policy rule 2).
        if body.project_id != caller.project_id or body.org_id != caller.org_id:
            raise ApiError(
                type="forbidden_scope",
                message="an API key can only create keys for its own org/project",
                field="project_id",
                status_code=403,
            )
        org_id, project_id = caller.org_id, caller.project_id
    else:
        # Documented bootstrap: the very first key is free to create.
        count = await session.scalar(select(func.count()).select_from(ApiKey))
        if count:
            raise _unauthorized()
        org_id, project_id = body.org_id, body.project_id

    clear = generate_key()
    key = ApiKey(
        key_hash=hash_key(clear),
        prefix=clear[:8],
        org_id=org_id,
        project_id=project_id,
        label=body.label,
    )
    session.add(key)
    await session.commit()
    return KeyCreatedResponse(
        id=key.id,
        key=clear,
        prefix=key.prefix,
        org_id=key.org_id,
        project_id=key.project_id,
        label=key.label,
        created_at=key.created_at,
    )


@router.get("/keys", response_model=KeyListResponse)
async def list_keys(
    request: Request, session: AsyncSession = Depends(get_session)
) -> KeyListResponse:
    is_admin, caller = await _caller(request)
    if settings.admin_key and not is_admin:
        raise _unauthorized()
    stmt = select(ApiKey).order_by(ApiKey.created_at)
    if not is_admin:
        if caller is None:
            raise _unauthorized()
        stmt = stmt.where(ApiKey.project_id == caller.project_id)
    keys = (await session.execute(stmt)).scalars().all()
    return KeyListResponse(keys=[KeyOut.model_validate(k) for k in keys])


@router.delete("/keys/{key_id}", response_model=KeyRevokedResponse)
async def revoke_key(
    key_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> KeyRevokedResponse:
    is_admin, caller = await _caller(request)
    if settings.admin_key and not is_admin:
        raise _unauthorized()
    if not is_admin and caller is None:
        raise _unauthorized()

    try:
        uid = UUID(key_id)
    except ValueError:
        uid = None
    key = await session.get(ApiKey, uid) if uid else None
    # Same 404 as an unknown id: another project's keys are invisible.
    if key is None or (not is_admin and key.project_id != caller.project_id):
        raise ApiError(
            type="key_not_found",
            message=f"API key {key_id} does not exist",
            field="key_id",
            status_code=404,
        )
    key.revoked_at = datetime.now(timezone.utc)
    await session.commit()
    return KeyRevokedResponse(id=key.id)
