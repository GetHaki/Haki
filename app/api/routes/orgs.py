"""Self-serve organization provisioning (sprint 11).

Called ONLY by the console's Next.js backend, using an already-verified
Clerk identity — never by a browser directly, never by a customer's own
`hk_` key. This is what lets a SECOND human sign up without asking the
founder to mint a key by hand (the gap found, and worked around manually,
while auditing production-readiness).

Auth is a single shared secret (`HAKI_CONSOLE_SERVICE_KEY`), deliberately
simpler than the customer-facing API-key model: the console is one trusted
caller, not a multi-tenant surface. Excluded from `ApiKeyAuthMiddleware`
(app/auth.py) the same way `/v1/keys` already is.
"""

import uuid

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import constant_time_bearer_match, generate_key, hash_key
from app.config import settings
from app.db import get_session
from app.errors import ApiError
from app.models import ApiKey, Organization
from app.rate_limit import limiter
from app.schemas.orgs import (
    OrgSettingsResponse,
    ProvisionOrgRequest,
    ProvisionOrgResponse,
    UpdateOrgSettingsRequest,
)
from app.schemas.team import (
    AcceptInviteRequest,
    AcceptInviteResponse,
    InviteRequest,
    InviteResponse,
    MemberListResponse,
    MemberOut,
    RemoveMemberRequest,
)
from app.teams import (
    accept_invite,
    create_invite,
    list_members,
    remove_member,
    resolve_org_for_user_ref,
)

router = APIRouter()


def _unauthorized() -> ApiError:
    return ApiError(
        type="unauthorized",
        message="missing or invalid console service credentials",
        field="Authorization",
        status_code=401,
    )


def _require_console_auth(request: Request) -> None:
    if not settings.console_service_key:
        raise _unauthorized()
    if not constant_time_bearer_match(
        request.headers.get("authorization"), settings.console_service_key
    ):
        raise _unauthorized()


async def _org_by_owner_ref(session: AsyncSession, owner_ref: str) -> Organization:
    """The organization an owner_ref belongs to — owner OR member.

    Security audit H2 (8 sept): this used to look at owner_ref alone,
    which silently broke invited members' settings/billing pages (they
    resolve via app.teams membership, not ownership) and made "who does
    this ref belong to" the caller's claim. resolve_org_for_user_ref is
    the single resolution point: ownership first, membership second,
    None otherwise."""
    org = await resolve_org_for_user_ref(session, owner_ref)
    if org is None:
        raise ApiError(
            type="org_not_found",
            message="no organization for this owner_ref",
            field="owner_ref",
            status_code=404,
        )
    return org


def _key_caller_org_id(request: Request) -> str | None:
    """The org of the middleware-resolved hk_ key, if any ("org_<uuid>")."""
    key = (request.scope.get("state") or {}).get("haki_api_key")
    org_id = str(getattr(key, "org_id", "") or "")
    return org_id if org_id.startswith("org_") else None


async def _resolve_caller_org(
    session: AsyncSession, request: Request, owner_ref: str | None
) -> Organization:
    """Org resolution for the two caller kinds (sprint 18):

    - console service secret + owner_ref (account flow): unchanged —
      ownership first, membership second via resolve_org_for_user_ref.
    - raw hk_ key (API-key login): the org is derived from the
      authenticated key itself; owner_ref must be absent (a present one
      cannot be honored — it lives in the Clerk namespace — and silently
      ignoring a caller-supplied scope would repeat the H2 mistake, so it
      is a 403, same spirit as the C4 org check in app/auth.py).

    Data operations only (settings read/write, member listing): invite,
    accept and remove keep the pure service-secret trust model below.
    """
    try:
        _require_console_auth(request)
    except ApiError as service_exc:
        key_org_id = _key_caller_org_id(request)
        if key_org_id is None:
            raise service_exc
        if owner_ref is not None:
            raise ApiError(
                type="forbidden",
                message="owner_ref is not honored with API-key auth: omit it, the org comes from the key",
                field="owner_ref",
                status_code=403,
            )
        try:
            org_uuid = uuid.UUID(key_org_id[len("org_"):])
        except ValueError:
            raise ApiError(
                type="unauthorized",
                message="a project API key is required",
                field="Authorization",
                status_code=401,
            ) from None
        org = await session.get(Organization, org_uuid)
        if org is None:
            raise ApiError(
                type="org_not_found",
                message="no organization for this API key",
                field="Authorization",
                status_code=404,
            )
        return org
    if not owner_ref:
        raise ApiError(
            type="invalid_payload",
            message="owner_ref is required",
            field="owner_ref",
            status_code=422,
        )
    return await _org_by_owner_ref(session, owner_ref)


@router.post("/orgs/provision", response_model=ProvisionOrgResponse, status_code=201)
# 60/minute per client IP. Found live in production: the original 5/minute
# broke real sign-ups within hours, because every real user's provisioning
# call arrives from the SAME IP (the console's own server — see the module
# docstring in app/rate_limit.py). IP isn't a precise per-caller bucket
# for this endpoint, but the actual security boundary here is the shared
# service-secret check just below, not this limit — 60/minute is generous
# headroom against real signup bursts while still bounding the damage if
# that secret ever leaked.
@limiter.limit("60/minute")
async def provision_org(
    body: ProvisionOrgRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ProvisionOrgResponse:
    _require_console_auth(request)

    # Checks BOTH ownership and membership (app.teams): a user who joined
    # someone else's org via an invite must resolve back to that SAME org
    # here — every sign-in calls this endpoint, and looking at owner_ref
    # alone would silently create a second, empty organization for them
    # instead of minting a key on the team they actually joined.
    org = await resolve_org_for_user_ref(session, body.owner_ref)
    org_created = False
    if org is None:
        org = Organization(name=body.name, owner_ref=body.owner_ref)
        session.add(org)
        await session.flush()  # assigns org.id, needed for the ids below
        org_created = True

    org_id_str = f"org_{org.id}"
    project_id_str = f"prj_{org.id}_default"

    clear = generate_key()
    key = ApiKey(
        key_hash=hash_key(clear),
        prefix=clear[:8],
        org_id=org_id_str,
        project_id=project_id_str,
        label="console-provisioned",
    )
    session.add(key)
    await session.commit()

    return ProvisionOrgResponse(
        org_id=org_id_str,
        project_id=project_id_str,
        api_key=clear,
        org_created=org_created,
        created_at=org.created_at,
    )


@router.get("/orgs/settings", response_model=OrgSettingsResponse)
async def get_org_settings(
    request: Request,
    owner_ref: str | None = Query(default=None, min_length=1, max_length=256),
    session: AsyncSession = Depends(get_session),
) -> OrgSettingsResponse:
    org = await _resolve_caller_org(session, request, owner_ref)
    return OrgSettingsResponse(
        org_id=f"org_{org.id}", name=org.name, retention_days=org.retention_days
    )


@router.patch("/orgs/settings", response_model=OrgSettingsResponse)
async def update_org_settings(
    body: UpdateOrgSettingsRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> OrgSettingsResponse:
    org = await _resolve_caller_org(session, request, body.owner_ref)
    if body.name is not None:
        org.name = body.name
    if body.clear_retention:
        org.retention_days = None
    elif body.retention_days is not None:
        org.retention_days = body.retention_days
    await session.commit()
    return OrgSettingsResponse(
        org_id=f"org_{org.id}", name=org.name, retention_days=org.retention_days
    )


@router.get("/orgs/members", response_model=MemberListResponse)
async def get_members(
    request: Request,
    owner_ref: str | None = Query(default=None, min_length=1, max_length=256),
    session: AsyncSession = Depends(get_session),
) -> MemberListResponse:
    # Same two-caller rule as settings above (key callers see their own
    # org's roster); invite/accept/remove below stay service-secret-only.
    org = await _resolve_caller_org(session, request, owner_ref)
    members = await list_members(session, org)
    return MemberListResponse(members=[MemberOut(**m) for m in members])


@router.post("/orgs/members/invite", response_model=InviteResponse, status_code=201)
async def invite_member(
    body: InviteRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> InviteResponse:
    _require_console_auth(request)
    org = await resolve_org_for_user_ref(session, body.owner_ref)
    if org is None:
        raise ApiError(
            type="org_not_found",
            message="no organization for this owner_ref",
            field="owner_ref",
            status_code=404,
        )
    invite = await create_invite(session, org=org, inviter_ref=body.owner_ref, role=body.role)
    await session.commit()
    return InviteResponse(token=invite.token, role=invite.role, expires_at=invite.expires_at)


@router.post("/orgs/members/accept", response_model=AcceptInviteResponse)
async def accept_member_invite(
    body: AcceptInviteRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> AcceptInviteResponse:
    _require_console_auth(request)
    org = await accept_invite(session, token=body.token, accepter_ref=body.owner_ref)
    await session.commit()
    return AcceptInviteResponse(org_id=f"org_{org.id}")


@router.post("/orgs/members/{user_ref}/remove")
async def delete_member(
    user_ref: str,
    body: RemoveMemberRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, bool]:
    _require_console_auth(request)
    org = await resolve_org_for_user_ref(session, body.owner_ref)
    if org is None:
        raise ApiError(
            type="org_not_found",
            message="no organization for this owner_ref",
            field="owner_ref",
            status_code=404,
        )
    await remove_member(session, org=org, actor_ref=body.owner_ref, target_ref=user_ref)
    await session.commit()
    return {"removed": True}
