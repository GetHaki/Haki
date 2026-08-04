"""API key authentication (sprint 6).

`ApiKeyAuthMiddleware` is a pure ASGI middleware on /v1/* and /gateway/v1/*
(key management endpoints /v1/keys excluded: they have their own
bootstrap/admin logic in app/api/routes/keys.py). When
`HAKI_AUTH_REQUIRED=true` (default):

- `Authorization: Bearer hk_...` is mandatory; the key is resolved by its
  sha256 hash and must not be revoked -> 401 unauthorized otherwise ;
- scope binding (policy rule 2): every project_id found in the body or the
  query must equal the key's project -> 403 forbidden_scope, generic
  message, no hint about other projects ;
- the resolved key is stored in scope["state"]["haki_api_key"]; the
  get_session dependency turns it into `SET LOCAL haki.project_id` so RLS
  (migration 0006) enforces the same isolation in SQL — even if the code
  forgets a project filter, other projects' rows never leave the server.

When `HAKI_AUTH_REQUIRED=false` everything is open (documented dev mode,
startup warning in app.main) and no RLS context is set: the policies are
permissive when haki.project_id is unset.
"""

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs

from sqlalchemy import select
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import policy
from app.config import settings
from app.db import async_session
from app.errors import ApiError, error_body
from app.models import ApiKey

STATE_KEY = "haki_api_key"


def generate_key() -> str:
    """New clear key, shown exactly once at creation (only the hash is stored)."""
    return "hk_" + uuid.uuid4().hex


def hash_key(clear: str) -> str:
    return hashlib.sha256(clear.encode()).hexdigest()


def bearer_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name.lower() == b"authorization":
            text = value.decode().strip()
            if text.lower().startswith("bearer "):
                return text[7:].strip()
            return None
    return None


async def resolve_api_key(token: str | None) -> ApiKey | None:
    """Valid (existing, non-revoked) key for a clear token, else None."""
    if not token or not token.startswith("hk_"):
        return None
    async with async_session() as session:
        key = (
            await session.execute(
                select(ApiKey).where(ApiKey.key_hash == hash_key(token))
            )
        ).scalars().first()
    if key is None or key.revoked_at is not None:
        return None
    return key


def _body_project_ids(payload: Any) -> list[str | None]:
    """project_id candidates in a JSON body: top-level plus per-event
    (capture batches carry one project_id per event)."""
    if not isinstance(payload, dict):
        return []
    candidates: list[str | None] = [payload.get("project_id")]
    events = payload.get("events")
    if isinstance(events, list):
        candidates.extend(
            event.get("project_id") for event in events if isinstance(event, dict)
        )
    return candidates


class ApiKeyAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def _deny(self, scope: Scope, receive: Receive, send: Send, exc: ApiError) -> None:
        response = JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.type, exc.message, exc.field),
        )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        # Protected paths: /v1/* (key management /v1/keys and org
        # provisioning /v1/orgs each have their own auth logic — a service
        # secret, not a customer hk_ key; same for /v1/billing — console
        # service secret — and /v1/webhooks — HMAC signature, sprint 12)
        # and the gateway /gateway/v1/* (sprint 7 — the chat-completions
        # body carries no project_id, the key's project is the scope).
        # /v1/cli/device/* (sprint 14) is excluded too: start/poll are
        # deliberately public (a terminal has no hk_ key yet — that's the
        # whole point of the flow) and approve authenticates itself with
        # the console service secret, same pattern as /v1/orgs — see
        # app/api/routes/cli_auth.py.
        protected = path.startswith("/gateway/v1/") or (
            path.startswith("/v1/")
            and not path.startswith("/v1/keys")
            and not path.startswith("/v1/orgs")
            and not path.startswith("/v1/billing")
            and not path.startswith("/v1/webhooks")
            and not path.startswith("/v1/cli/device")
        )
        if scope["type"] != "http" or not settings.auth_required or not protected:
            await self.app(scope, receive, send)
            return

        action = f"{scope['method']} {path}"

        key = await resolve_api_key(bearer_token(scope["headers"]))
        if key is None:
            await self._deny(
                scope,
                receive,
                send,
                ApiError(
                    type="unauthorized",
                    message="missing, invalid or revoked API key",
                    field="Authorization",
                    status_code=401,
                ),
            )
            return

        # Scope binding candidates: query string first.
        candidates: list[str | None] = [
            value
            for value in parse_qs(scope.get("query_string", b"").decode()).get(
                "project_id", []
            )
        ]

        # Then the JSON body (buffered and replayed downstream).
        body = b""
        replay: Callable[[], Awaitable[Message]] | None = None
        if scope["method"] in ("POST", "PUT", "PATCH"):
            more = True
            while more:
                message = await receive()
                if message["type"] != "http.request":
                    break
                body += message.get("body", b"")
                more = message.get("more_body", False)
            try:
                candidates.extend(_body_project_ids(json.loads(body)) if body else [])
            except ValueError:
                pass  # malformed JSON: 422 invalid_payload downstream

            sent = False

            async def replay() -> Message:
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                # After the buffered body, delegate to the real receive: a
                # synthetic empty http.request here would busy-loop any
                # downstream consumer waiting for http.disconnect (found via
                # the gateway's StreamingResponse, sprint 7).
                return await receive()

        try:
            policy.check_project_scope(key.project_id, candidates, action=action)
        except ApiError as exc:
            await self._deny(scope, receive, send, exc)
            return

        scope.setdefault("state", {})[STATE_KEY] = key
        await self.app(scope, replay or receive, send)
