from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp

from app.api.routes import api_router
from app.auth import ApiKeyAuthMiddleware
from app.config import settings
from app.db import install_tcp_nodelay
from app.errors import error_body, register_error_handlers
from app.mcp_server import mcp as mcp_server

logger = logging.getLogger("haki.main")

# Configure the MCP streamable-HTTP session manager (json responses, DNS
# rebinding protection for localhost). The returned Starlette app is not
# mounted as-is: its inner route is "/mcp", and mounting it under FastAPI
# would force a "/mcp/" trailing-slash redirect that MCP clients may not
# follow. Instead the raw ASGI handler is mounted on exactly "/mcp".
mcp_server.streamable_http_app(streamable_http_path="/mcp", json_response=True)
mcp_session_manager = mcp_server._lowlevel_server.session_manager


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    # See app.db.install_tcp_nodelay (standard TCP_NODELAY practice for
    # Postgres clients; asyncpg exposes no socket option).
    install_tcp_nodelay()
    logger.info(
        "startup: llm_provider=%s embed_provider=%s auth_required=%s",
        settings.llm_provider,
        settings.embed_provider,
        settings.auth_required,
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


app = FastAPI(title="Haki", version="0.1.0", lifespan=lifespan)
register_error_handlers(app)
app.include_router(api_router)

# API key auth + project scope binding on /v1/* (sprint 6). No-op when
# HAKI_AUTH_REQUIRED=false (open dev mode).
app.add_middleware(ApiKeyAuthMiddleware)


@app.middleware("http")
async def mcp_dev_auth(request: Request, call_next) -> Response:
    """Dev auth for the MCP endpoint (sprint 4).

    When HAKI_API_KEY is set, /mcp requires `Authorization: Bearer <key>`.
    Unset = open mode (local development only, documented in the README).
    Full OAuth lands in a later sprint.
    """
    if settings.api_key and request.url.path.startswith("/mcp"):
        if request.headers.get("authorization") != f"Bearer {settings.api_key}":
            return JSONResponse(
                status_code=401,
                content=error_body(
                    "unauthorized", "invalid or missing bearer token", "Authorization"
                ),
            )
    return await call_next(request)


app.mount("/mcp", StreamableHTTPASGIApp(mcp_session_manager))
