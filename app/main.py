from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp
from mcp.server.transport_security import TransportSecuritySettings
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import api_router
from app.auth import ApiKeyAuthMiddleware
from app.config import settings
from app.db import install_tcp_nodelay, verify_fts_config
from app.errors import error_body, register_error_handlers
from app.mcp_server import mcp as mcp_server
from app.rate_limit import limiter, rate_limit_exceeded_handler

logger = logging.getLogger("haki.main")

# Configure the MCP streamable-HTTP session manager (json responses, DNS
# rebinding protection scoped explicitly -- see settings.mcp_public_host).
# The returned Starlette app is not mounted as-is: its inner route is
# "/mcp", and mounting it under FastAPI would force a "/mcp/"
# trailing-slash redirect that MCP clients may not follow. Instead the raw
# ASGI handler is mounted on exactly "/mcp".
#
# transport_security is passed explicitly (not left to the SDK's default):
# left unset, mcp.server.lowlevel.server.streamable_http_app() only
# auto-enables the Host-header allowlist when its own `host` argument
# defaults to "127.0.0.1" -- which it always does here, since Haki mounts
# the app rather than running it standalone. That silently restricted
# every request's Host header to localhost variants, so any real deployed
# domain (confirmed live on api.gethaki.space) got a blanket 421 "Invalid
# Host header" on /mcp/, before any Haki code ever ran.
_mcp_allowed_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_mcp_allowed_origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
if settings.mcp_public_host:
    _mcp_allowed_hosts.append(settings.mcp_public_host)
    _mcp_allowed_origins.append(f"https://{settings.mcp_public_host}")

mcp_server.streamable_http_app(
    streamable_http_path="/mcp",
    json_response=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=_mcp_allowed_hosts,
        allowed_origins=_mcp_allowed_origins,
    ),
)
mcp_session_manager = mcp_server._lowlevel_server.session_manager


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    # See app.db.install_tcp_nodelay (standard TCP_NODELAY practice for
    # Postgres clients; asyncpg exposes no socket option).
    install_tcp_nodelay()
    logger.info(
        "startup: llm_provider=%s embed_provider=%s fts_config=%s auth_required=%s",
        settings.llm_provider,
        settings.embed_provider,
        settings.fts_config,
        settings.auth_required,
    )
    # Refuses to serve when the queried and indexed text search
    # configurations disagree — see app.db.verify_fts_config.
    await verify_fts_config()
    # `fake` is the DEFAULT extractor, which is right for tests and wrong
    # for anything else: FakeProvider reads `payload["mock_facts"]` and
    # returns [] otherwise. A self-hosted install that forgets
    # HAKI_LLM_PROVIDER therefore extracts nothing, raises nothing, and
    # builds a silently empty memory -- the worst possible default for a
    # product whose argument is reliability. Loud at startup instead.
    #
    # Paired with auth_required rather than with a "is this production?"
    # guess: HAKI_AUTH_REQUIRED=false is already documented as OPEN dev
    # mode, never for production, and warns two lines below. "Open dev mode
    # AND a fake extractor" is a coherent development box; "authentication
    # on AND a fake extractor" is someone about to store nothing for real
    # users.
    if settings.llm_provider == "fake" and settings.auth_required:
        raise RuntimeError(
            "HAKI_LLM_PROVIDER=fake extracts nothing outside tests: "
            "FakeProvider only reads mock_facts from the payload, so every "
            "captured event would produce an empty memory, silently. Set "
            "HAKI_LLM_PROVIDER=openai (and HAKI_LLM_API_KEY), or set "
            "HAKI_AUTH_REQUIRED=false for the documented local dev mode."
        )
    if not settings.auth_required:
        logger.warning(
            "HAKI_AUTH_REQUIRED=false: OPEN dev mode — every /v1 endpoint is "
            "unauthenticated and RLS stays permissive. Never use in production."
        )
    # A mounted sub-app's lifespan never runs: the MCP session manager task
    # group must be started explicitly, or every /mcp request fails with
    # "StreamableHTTPSessionManager is not running".
    async with mcp_session_manager.run():
        yield


# Swagger/ReDoc UIs are disabled whenever HAKI_ADMIN_KEY is set — the same
# signal docs/DEPLOY.md already treats as "this is a real deployment, not a
# local docker-compose". Self-hosters running without an admin key (local
# dev, first-time trial) keep interactive docs; api.gethaki.space, which
# always sets the admin key, does not expose them publicly.
_docs_enabled = settings.admin_key is None
app = FastAPI(
    title="Haki",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
)
register_error_handlers(app)
app.include_router(api_router)

# Basic rate limiting (production-readiness fix): in-memory backend, see
# app/rate_limit.py for why that's the right call for this single-instance
# deployment. Per-route limits are declared with @limiter.limit(...) on
# the individual endpoints (provisioning, key creation, capture).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# API key auth + project scope binding on /v1/* (sprint 6). No-op when
# HAKI_AUTH_REQUIRED=false (open dev mode).
app.add_middleware(ApiKeyAuthMiddleware)


@app.middleware("http")
async def mcp_dev_auth(request: Request, call_next) -> Response:
    """Legacy single-secret gate for the MCP endpoint (sprint 4).

    A real customer `hk_` key is let through here unconditionally: it is
    validated per-tool-call against the api_keys table by
    app.mcp_server._resolve_scope (invalid/revoked -> a clear ToolError,
    not a silent fallback) — checking it AGAIN here against the single
    shared HAKI_API_KEY secret would reject every real Cloud customer,
    since their key is never equal to that one shared value (this is
    exactly the bug found live: multi-tenant MCP was unusable whenever
    HAKI_API_KEY was configured in production).

    When HAKI_API_KEY is set, any OTHER bearer token on /mcp (not an hk_
    key — i.e. no Authorization header, or one that isn't the legacy
    self-hosted single-server config) must match it exactly. Unset = open
    mode for that path (local development only, documented in the
    README). Full OAuth lands in a later sprint.
    """
    if settings.api_key and request.url.path.startswith("/mcp"):
        auth_header = request.headers.get("authorization") or ""
        token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
        if not token.startswith("hk_") and auth_header != f"Bearer {settings.api_key}":
            return JSONResponse(
                status_code=401,
                content=error_body(
                    "unauthorized", "invalid or missing bearer token", "Authorization"
                ),
            )
    return await call_next(request)


app.mount("/mcp", StreamableHTTPASGIApp(mcp_session_manager))
