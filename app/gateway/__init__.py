"""Haki Gateway (sprint 7) — OpenAI-compatible proxy with automatic memory.

Flow for `POST /gateway/v1/chat/completions` (non-streaming, memory active):

1. the API key middleware (extended to /gateway/v1/*) resolves the `hk_`
   key -> project_id ; the Haki key is NEVER sent upstream (only the
   HAKI_LLM_* credentials are) ;
2. memory identity comes from headers, never from the model-controlled
   body: X-Haki-Subject-Id (required for memory), X-Haki-Thread-Id,
   X-Haki-Run-Id, X-Haki-Purpose, X-Haki-Idempotency-Key (default: sha256
   of the raw body) ;
3. the last user message keys a `build_context` call; the ContextPacket is
   rendered with the SDK's `build_prompt_context` (single implementation,
   no divergent copy) and prepended to the system message inside a
   `<haki_memory>...</haki_memory>` block ;
4. the body is forwarded to `{HAKI_LLM_BASE_URL}/chat/completions`
   (timeout 60 s) ;
5. AFTER the response, the user/assistant turn is captured as a
   `conversation.turn` event (idempotent) and a consolidation job is
   enqueued — consolidation stays off the hot path ;
6. the upstream response is returned byte-identical (status + body), plus
   X-Haki-Memory / X-Haki-Trace-Id / X-Haki-Context-Ms headers.

Degradation (the agent is never blocked by Haki):

- no X-Haki-Subject-Id (or no user message): forwarded without memory,
  `X-Haki-Memory: disabled`, no capture — an existing OpenAI client never
  breaks ;
- build_context failure (DB down, ...): forwarded without memory,
  `X-Haki-Memory: degraded`, structured log ;
- capture failure: best-effort, logged, the response is still returned ;
- upstream errors: status and body propagated unchanged.

Streaming (`stream: true`): deliberate pass-through (option (a) of the
sprint brief). The request is proxied as a raw SSE stream with
`X-Haki-Memory: disabled` — no injection, no capture. A proxy that would
inject memory but could not capture the final answer would break the
runtime rule "no final answer without a Haki pass after"; buffering the
whole stream would kill the reason clients stream. Documented in the
README.

Known limit (per research/Haki_Memory_Runtime.md): the gateway sees model
calls, not the tools an agent runs locally between calls — those must be
captured via the SDK/API.
"""

import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import httpx
from haki.runtime import build_prompt_context
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger
from app.config import settings
from app.schemas import EventIn

logger = logging.getLogger("haki.gateway")

UPSTREAM_TIMEOUT = 60.0

__all__ = [
    "assistant_message",
    "build_prompt_context",
    "capture_turn",
    "content_to_text",
    "default_idempotency_key",
    "forward_upstream",
    "inject_memory_block",
    "last_user_message",
    "open_upstream_stream",
    "stream_body",
]


# -- request shaping ----------------------------------------------------------


def content_to_text(content: Any) -> str | None:
    """OpenAI message content -> plain text (string or list of parts)."""
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(text for text in parts if text) or None
    return None


def last_user_message(messages: list[dict[str, Any]]) -> str | None:
    """The last user message text: it keys the memory query."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return content_to_text(message.get("content"))
    return None


def inject_memory_block(
    messages: list[dict[str, Any]], block: str
) -> list[dict[str, Any]]:
    """Prepend the <haki_memory> block at the START of the system message.

    A system message is created (first position) when absent. Input messages
    are copied, never mutated.
    """
    if not block:
        return messages
    out = [dict(message) for message in messages]
    for message in out:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [{"type": "text", "text": block}, *content]
            else:
                existing = content if isinstance(content, str) else ""
                message["content"] = f"{block}\n\n{existing}" if existing else block
            return out
    return [{"role": "system", "content": block}, *out]


def default_idempotency_key(raw_body: bytes) -> str:
    """Idempotency default when X-Haki-Idempotency-Key is absent: body hash."""
    return "gw-" + hashlib.sha256(raw_body).hexdigest()


# -- upstream -----------------------------------------------------------------


def _upstream_headers() -> dict[str, str]:
    # Only the upstream credentials: the caller's Haki key is never forwarded.
    headers = {"content-type": "application/json"}
    if settings.llm_api_key:
        headers["authorization"] = f"Bearer {settings.llm_api_key}"
    return headers


def _upstream_url() -> str:
    return f"{settings.llm_base_url.rstrip('/')}/chat/completions"


async def forward_upstream(payload: dict[str, Any]) -> httpx.Response:
    """POST the (possibly memory-augmented) body to the real provider."""
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        return await client.post(
            _upstream_url(), json=payload, headers=_upstream_headers()
        )


async def open_upstream_stream(
    payload: dict[str, Any],
) -> tuple[httpx.Response, httpx.AsyncClient]:
    """Open a streaming request upstream. Caller must close via stream_body."""
    client = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT, read=None))
    request = client.build_request(
        "POST", _upstream_url(), json=payload, headers=_upstream_headers()
    )
    response = await client.send(request, stream=True)
    return response, client


async def stream_body(
    response: httpx.Response, client: httpx.AsyncClient
) -> AsyncGenerator[bytes, None]:
    """Raw SSE pass-through; closes the upstream response and client at end."""
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


def assistant_message(response_body: bytes) -> str | None:
    """choices[0].message.content of a chat.completion response, if parseable."""
    try:
        data = json.loads(response_body)
    except ValueError:
        return None
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message") or {}
    return content_to_text(message.get("content"))


# -- capture (after the response, best-effort) --------------------------------


async def capture_turn(
    session: AsyncSession,
    *,
    org_id: str,
    project_id: str,
    subject_id: str,
    user_text: str,
    assistant_text: str | None,
    model: str | None,
    trace_id: Any,
    thread_id: str | None,
    run_id: str | None,
    idempotency_key: str,
) -> None:
    """Persist the exchange as one idempotent conversation.turn event and
    enqueue its consolidation job (consolidation runs off the hot path)."""
    event = EventIn(
        org_id=org_id,
        project_id=project_id,
        subject_type="user",
        subject_id=subject_id,
        thread_id=thread_id,
        run_id=run_id,
        # M8: direct end-user turn through the authenticated proxy.
        origin_trust="trusted",
        kind="conversation.turn",
        occurred_at=datetime.now(timezone.utc),
        payload={
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "model": model,
            "trace_id": str(trace_id) if trace_id else None,
        },
        source={"channel": "gateway"},
        idempotency_key=idempotency_key,
    )
    results = await ledger.write_events(session, [event])
    new_event_ids = [event.id for event, dedup in results if not dedup]
    if new_event_ids:
        await ledger.create_consolidation_job(
            session, project_id=project_id, event_ids=new_event_ids
        )
    await session.commit()
