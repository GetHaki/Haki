"""POST /gateway/v1/chat/completions — Haki Gateway (sprint 7).

OpenAI-compatible endpoint: a client only changes `base_url` and keeps its
own flow. Memory identity arrives via X-Haki-* headers (never via the
model-controlled body). See app/gateway/__init__.py for the full contract
(flow, degradation, streaming pass-through).
"""

import json
import logging
import time

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from haki.runtime import build_prompt_context
from sqlalchemy.ext.asyncio import AsyncSession

from app import gateway, metrics
from app.auth import STATE_KEY
from app.context import build_context
from app.db import get_session
from app.errors import ApiError, error_body

logger = logging.getLogger("haki.gateway")

router = APIRouter()

# Dev-open mode (HAKI_AUTH_REQUIRED=false) has no key to bind: the project
# comes from X-Haki-Project-Id or this documented default.
OPEN_MODE_PROJECT = "prj_gateway_dev"
OPEN_MODE_ORG = "org_gateway_dev"


@router.post("/chat/completions")
async def chat_completions(
    request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    raw_body = await request.body()
    try:
        payload = json.loads(raw_body)
    except ValueError:
        raise ApiError(
            type="invalid_payload", message="request body must be valid JSON", field="body"
        ) from None
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ApiError(
            type="invalid_payload",
            message="'messages' must be a list of chat messages",
            field="messages",
        )

    api_key = (request.scope.get("state") or {}).get(STATE_KEY)
    if api_key is not None:
        project_id, org_id = api_key.project_id, api_key.org_id
    else:
        project_id = request.headers.get("x-haki-project-id", OPEN_MODE_PROJECT)
        org_id = OPEN_MODE_ORG

    subject_id = request.headers.get("x-haki-subject-id")
    thread_id = request.headers.get("x-haki-thread-id")
    run_id = request.headers.get("x-haki-run-id")
    purpose = request.headers.get("x-haki-purpose")
    idempotency_key = request.headers.get(
        "x-haki-idempotency-key"
    ) or gateway.default_idempotency_key(raw_body)

    # Streaming: pure pass-through, no memory, no capture (documented choice).
    if payload.get("stream"):
        metrics.increment("gateway.memory.disabled")
        upstream, stream_client = await gateway.open_upstream_stream(payload)
        return StreamingResponse(
            gateway.stream_body(upstream, stream_client),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers={"X-Haki-Memory": "disabled"},
        )

    memory = "disabled"
    trace_id = None
    context_ms: float | None = None
    user_text = gateway.last_user_message(payload["messages"]) if subject_id else None

    if subject_id and user_text:
        try:
            start = time.perf_counter()
            packet, _tokens, trace_id = await build_context(
                session,
                project_id=project_id,
                subject_id=subject_id,
                query=user_text,
                purpose=purpose,
            )
            # Persist the decision trace on its own: it must survive a later
            # capture failure.
            await session.commit()
            context_ms = (time.perf_counter() - start) * 1000
            block = build_prompt_context(packet)
            if block:
                payload = {
                    **payload,
                    "messages": gateway.inject_memory_block(
                        payload["messages"], block
                    ),
                }
            memory = "active"
        except Exception:
            memory = "degraded"
            logger.exception(
                "gateway build_context failed (project=%s subject=%s): "
                "forwarding without memory",
                project_id,
                subject_id,
            )

    metrics.increment(f"gateway.memory.{memory}")

    try:
        upstream = await gateway.forward_upstream(payload)
    except httpx.HTTPError as exc:
        logger.warning("gateway upstream call failed: %s", exc)
        return JSONResponse(
            status_code=502,
            content=error_body(
                "upstream_unavailable", "the LLM provider could not be reached", None
            ),
            headers={"X-Haki-Memory": memory},
        )

    # Capture AFTER the response, best-effort: a capture failure never
    # changes what the client receives.
    if memory == "active" and 200 <= upstream.status_code < 300:
        try:
            await gateway.capture_turn(
                session,
                org_id=org_id,
                project_id=project_id,
                subject_id=subject_id,
                user_text=user_text,
                assistant_text=gateway.assistant_message(upstream.content),
                model=payload.get("model"),
                trace_id=trace_id,
                thread_id=thread_id,
                run_id=run_id,
                idempotency_key=idempotency_key,
            )
        except Exception:
            logger.exception(
                "gateway capture failed (project=%s subject=%s): response "
                "returned anyway",
                project_id,
                subject_id,
            )

    headers = {
        "X-Haki-Memory": memory,
        "content-type": upstream.headers.get("content-type", "application/json"),
    }
    if trace_id is not None:
        headers["X-Haki-Trace-Id"] = str(trace_id)
    if context_ms is not None:
        headers["X-Haki-Context-Ms"] = f"{context_ms:.1f}"
    return Response(
        content=upstream.content, status_code=upstream.status_code, headers=headers
    )
