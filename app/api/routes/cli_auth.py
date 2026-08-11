"""CLI device-code auth flow (sprint 14) — the "gh auth login" /
"vercel login" pattern.

Today a Cloud (Clerk) user has NO way to get their raw API key onto their
terminal: it is shown exactly once, at console provisioning time
(POST /v1/orgs/provision, see app/api/routes/orgs.py), and only its hash is
ever stored server-side afterwards. This flow closes that gap without a new
customer-facing secret and without a new Postgres table:

1. POST /v1/cli/device/start (public, IP rate-limited, no auth) — the CLI
   gets a `device_code` (opaque, 64 hex chars, never shown to the human) and
   a short `user_code` (XXXX-XXXX, shown to the human, typed at
   `<HAKI_CONSOLE_BASE_URL>/cli-auth`).
2. The CLI polls POST /v1/cli/device/poll with the device_code every
   `interval` seconds.
3. Meanwhile the human, already logged into the console with their Clerk
   session, approves the user_code there; the console's Next.js backend
   calls POST /v1/cli/device/approve — protected by
   `Authorization: Bearer HAKI_CONSOLE_SERVICE_KEY`, the exact same trusted
   -caller pattern as POST /v1/orgs/provision — with the real
   api_key/org_id/project_id to hand to that CLI.
4. The NEXT poll after approval consumes the code: it returns 'approved'
   with the key exactly once, then the device_code entry is deleted from
   Redis — a device_code leaked after that point (log, proxy, shoulder
   surfing) cannot replay the poll to fetch the key again. This is the
   single most important security property of the whole flow; see
   tests/test_cli_auth.py for an explicit test of it.

Storage: Redis only (app/redis_client.py; docker-compose.yml has shipped
the service since the start of the project, unused until now), native TTL
= expires_in (600s), no Postgres table — the entire flow is inherently
short-lived and disposable, nothing here needs to survive past expiry.

Excluded from `ApiKeyAuthMiddleware` (app/auth.py), same as /v1/keys and
/v1/orgs: /device/start and /device/poll are intentionally public (a
terminal has no `hk_` key yet — that is the whole point), and /device/approve
authenticates itself with the console service secret, not a customer key.
"""

import json
import time
from secrets import choice, token_hex
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from redis.asyncio import Redis

from app.config import settings
from app.errors import ApiError
from app.redis_client import get_redis
from app.schemas.cli_auth import (
    DeviceApproveRequest,
    DeviceApproveResponse,
    DevicePollRequest,
    DevicePollResponse,
    DeviceStartResponse,
)

router = APIRouter()

EXPIRES_IN = 600  # seconds — also the Redis TTL on both keys below
POLL_INTERVAL = 3  # seconds — hint returned to the CLI, not enforced here

# 32-char alphabet excluding 0/O and 1/I (contract requirement): a human
# reads and types this on a second device, ambiguous glyphs cause real
# support tickets.
_USER_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

_DEVICE_KEY = "cli:device:{}"
_USER_CODE_KEY = "cli:usercode:{}"

# Reasonable public-endpoint rate limit (contract: "rate-limite
# raisonnablement par IP"): 20 device sessions/minute/IP is generous for a
# human typing `haki login` a few times while debugging, and blunts a
# trivial script that would otherwise fill Redis with pending codes.
_RATE_LIMIT_KEY = "cli:ratelimit:start:{}"
_RATE_LIMIT_MAX = 20
_RATE_LIMIT_WINDOW = 60  # seconds

# RFC 8628 §5.4 — the user_code is short enough to guess (32**8), so wrong
# guesses must be capped. Approving a code hands the APPROVER's key to
# whatever terminal is holding it, so a successful guess against someone
# else's pending session plants an attacker-owned key in a victim's CLI —
# every write that terminal makes then lands in the attacker's project.
# Counting per approver (not per IP) is what makes this usable: every
# approval reaches the API from the console backend's single address, so an
# IP counter would be one shared bucket that any user could exhaust for
# everyone. Only FAILED attempts count — a person approving several of
# their own terminals in a row is normal and must stay unthrottled.
_APPROVE_ATTEMPTS_KEY = "cli:ratelimit:approve:{}"
_APPROVE_ATTEMPTS_MAX = 10
_APPROVE_ATTEMPTS_WINDOW = 300  # seconds


def _generate_user_code() -> str:
    raw = "".join(choice(_USER_CODE_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


def _normalize_user_code(raw: str) -> str:
    """Reshape what a human typed back into the exact XXXX-XXXX key.

    The code is compared as a literal Redis key, so "abcd efgh", "ABCDEFGH"
    and "abcd-efgh" would otherwise be three different wrong codes — each
    one also burning an attempt against the §5.4 cap below. Normalizing
    here rather than in the console means every approving surface gets the
    same forgiveness, and the rule is covered by the API's own tests.

    Only cosmetics are forgiven: separators and case. Anything that is not
    eight alphanumerics is passed through untouched, to be rejected as the
    unknown code it is.
    """
    cleaned = "".join(char for char in raw.upper() if char.isalnum())
    if len(cleaned) != 8:
        return raw
    return f"{cleaned[:4]}-{cleaned[4:]}"


async def _check_rate_limit(redis: Redis, request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    key = _RATE_LIMIT_KEY.format(ip)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, _RATE_LIMIT_WINDOW)
    if count > _RATE_LIMIT_MAX:
        raise ApiError(
            type="rate_limited",
            message="too many device-code requests from this address, try again shortly",
            status_code=429,
        )


async def _check_approve_attempts(redis: Redis, approver_ref: str | None) -> str:
    """Raise 429 once this approver has burned through its wrong guesses.

    Returns the Redis key so a failed attempt can be counted against it.
    `approver_ref` is the console's identifier for the signed-in human; an
    absent one falls back to a single shared bucket rather than to no limit
    at all — a caller that declines to identify itself gets the strictest
    treatment, never the loosest.
    """
    key = _APPROVE_ATTEMPTS_KEY.format(approver_ref or "anonymous")
    attempts = await redis.get(key)
    if attempts is not None and int(attempts) >= _APPROVE_ATTEMPTS_MAX:
        raise ApiError(
            type="rate_limited",
            message=(
                "too many incorrect codes, wait a few minutes and run "
                "`haki login` again for a fresh one"
            ),
            status_code=429,
        )
    return key


async def _count_failed_attempt(redis: Redis, key: str) -> None:
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, _APPROVE_ATTEMPTS_WINDOW)


@router.post("/cli/device/start", response_model=DeviceStartResponse, status_code=201)
async def device_start(
    request: Request, redis: Redis = Depends(get_redis)
) -> DeviceStartResponse:
    await _check_rate_limit(redis, request)

    device_code = token_hex(32)  # 64 hex chars, unguessable, single-use

    # NX-claim a user_code: astronomically unlikely to collide (32**8
    # possibilities) with another pending code, but never silently overwrite
    # one if it somehow does.
    user_code = _generate_user_code()
    user_code_key = _USER_CODE_KEY.format(user_code)
    for _ in range(5):
        claimed = await redis.set(user_code_key, device_code, nx=True, ex=EXPIRES_IN)
        if claimed:
            break
        user_code = _generate_user_code()
        user_code_key = _USER_CODE_KEY.format(user_code)
    else:
        raise ApiError(
            type="internal_error",
            message="could not allocate a user code, please retry",
            status_code=500,
        )

    payload = {
        "status": "pending",
        "user_code": user_code,
        "expires_at": time.time() + EXPIRES_IN,
    }
    await redis.set(
        _DEVICE_KEY.format(device_code), json.dumps(payload), ex=EXPIRES_IN
    )

    verification_uri = f"{settings.console_base_url}/cli-auth"
    return DeviceStartResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=(
            f"{verification_uri}?code={quote(user_code, safe='')}"
        ),
        expires_in=EXPIRES_IN,
        interval=POLL_INTERVAL,
    )


@router.post("/cli/device/poll", response_model=DevicePollResponse)
async def device_poll(
    body: DevicePollRequest, redis: Redis = Depends(get_redis)
) -> DevicePollResponse:
    key = _DEVICE_KEY.format(body.device_code)
    raw = await redis.get(key)
    if raw is None:
        raise ApiError(
            type="device_code_not_found",
            message="unknown device_code",
            field="device_code",
            status_code=404,
        )

    payload = json.loads(raw)

    if payload["status"] == "pending" and time.time() >= payload["expires_at"]:
        await redis.delete(key)  # logically expired: don't wait on the TTL
        return DevicePollResponse(status="expired")

    if payload["status"] == "approved":
        # Consume-on-read: this is the ONLY poll response that will ever
        # carry the api_key. Deleting here means a repeat poll with the
        # same device_code (attacker replay, or the CLI polling once too
        # many times) gets 404, never a second 'approved'.
        await redis.delete(key)
        return DevicePollResponse(
            status="approved",
            api_key=payload["api_key"],
            org_id=payload["org_id"],
            project_id=payload["project_id"],
        )

    return DevicePollResponse(status="pending")


def _unauthorized() -> ApiError:
    return ApiError(
        type="unauthorized",
        message="missing or invalid console service credentials",
        field="Authorization",
        status_code=401,
    )


@router.post("/cli/device/approve", response_model=DeviceApproveResponse)
async def device_approve(
    body: DeviceApproveRequest, request: Request, redis: Redis = Depends(get_redis)
) -> DeviceApproveResponse:
    if not settings.console_service_key:
        raise _unauthorized()
    if request.headers.get("authorization") != f"Bearer {settings.console_service_key}":
        raise _unauthorized()

    attempts_key = await _check_approve_attempts(redis, body.approver_ref)

    user_code = _normalize_user_code(body.user_code)
    device_code = await redis.get(_USER_CODE_KEY.format(user_code))
    if device_code is None:
        await _count_failed_attempt(redis, attempts_key)
        raise ApiError(
            type="user_code_not_found",
            message="unknown or expired user_code",
            field="user_code",
            status_code=404,
        )

    key = _DEVICE_KEY.format(device_code)
    raw = await redis.get(key)
    if raw is None:
        # The user_code mapping was still alive but the device session
        # behind it is already gone (expired, or already consumed by a
        # poll) — Gone, not just "not found", to tell the console apart
        # from a plain typo.
        raise ApiError(
            type="user_code_expired",
            message="this user_code's device session has expired",
            field="user_code",
            status_code=410,
        )

    payload = json.loads(raw)
    payload.update(
        status="approved",
        api_key=body.api_key,
        org_id=body.org_id,
        project_id=body.project_id,
    )
    # keepttl: approving must not grant the CLI extra time beyond the
    # original expires_in countdown.
    await redis.set(key, json.dumps(payload), keepttl=True)

    return DeviceApproveResponse(ok=True)
