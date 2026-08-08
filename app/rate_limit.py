"""Basic rate limiting (production-readiness fix ahead of opening sign-ups).

Haki runs as a single instance today — the Dockerfile's own comment is
explicit about this: migrations run at container start, and running more
than one instance would race on `alembic upgrade head`. slowapi's default
in-memory backend is the right amount of infrastructure for that shape:
counters just need to survive the lifetime of the process they protect.
Redis already exists in this codebase (app/redis_client.py, the CLI
device-code auth flow, docker-compose.yml) but wiring it into the limiter
too would be pure extra coupling — a shared backend only earns its keep
once there is more than one process to share it between.

Callers are identified the same way app/auth.py already does: by API key
where one is resolved, by client IP otherwise (unauthenticated requests,
dev-open mode, and the console-only provisioning endpoint — see
app/api/routes/orgs.py for why that one's limit has to stay generous even
though IP isn't a precise bucket there: every real signup shares the
console server's one IP, and this slowapi version calls key_func()
synchronously, so a body-derived per-user key isn't available without a
much larger rework of how the limiter is wired in).
"""

import time

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth import STATE_KEY, bearer_token
from app.errors import error_body

# headers_enabled=True is deliberately NOT passed here: slowapi's own
# auto-injection (Limiter._inject_headers, called from inside the
# @limiter.limit wrapper for every decorated endpoint) requires the
# endpoint to return a raw starlette Response — every route in this API
# returns its Pydantic response_model instead and lets FastAPI serialize
# it, so that auto-injection crashes with "parameter `response` must be
# an instance of starlette.responses.Response" on every successful (non
# rate-limited) call. Retry-After is set by hand below, only on the 429
# path, where a real JSONResponse already exists.
limiter = Limiter(key_func=get_remote_address)


def key_or_ip(request: Request) -> str:
    """Rate-limit bucket for a request: the resolved API key's PREFIX
    (never the clear secret — the same 8 chars already shown in `GET
    /v1/keys` listings) when `ApiKeyAuthMiddleware` bound one to this
    request (protected /v1/* routes, see app/auth.py), else the bearer
    token's own prefix (key management routes resolve their caller inside
    the route body, not in middleware, so nothing is on `request.state`
    yet), else the client IP (no key at all: dev-open mode, or a request
    that is about to 401 anyway)."""
    key = getattr(request.state, STATE_KEY, None)
    if key is not None:
        return key.prefix
    token = bearer_token(request.headers.raw)
    if token and token.startswith("hk_"):
        return token[:8]
    return get_remote_address(request)


async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Same `{"error": {"type", "message", "field"}}` envelope as every
    other error in this API (app/errors.py) instead of slowapi's default
    plain-text-ish body, plus a Retry-After header (built by hand — see
    the comment on `limiter` above for why slowapi's own header injection
    isn't used)."""
    retry_after = 60
    current_limit = getattr(request.state, "view_rate_limit", None)
    if current_limit is not None:
        try:
            reset_at, _ = limiter.limiter.get_window_stats(current_limit[0], *current_limit[1])
            retry_after = max(1, int(reset_at - time.time()) + 1)
        except Exception:
            pass
    return JSONResponse(
        status_code=429,
        content=error_body("rate_limited", f"rate limit exceeded: {exc.detail}", None),
        headers={"Retry-After": str(retry_after)},
    )
