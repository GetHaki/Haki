"""Haki Python SDK client (sync by default, httpx).

Wraps the Haki HTTP API with readable, typed errors. Sync is the default
because the main consumers are user scripts and agent hooks; an async
variant is provided for async codebases.
"""

import asyncio
import random
import time
from typing import Any

import httpx

from haki.errors import HakiApiError, HakiConnectionError

# Statuses worth a second attempt: rate limiting and transient upstream
# failures. Anything else (400/401/403/404/409/422, ...) is definitive —
# retrying it burns the caller's time and, worse, its rate-limit budget.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# How many extra attempts after the first one. 3 retries = 4 tries total:
# enough to ride out a rolling restart or a burst-limited minute, small
# enough that a genuinely dead server fails fast (~3.5 s of backoff).
_MAX_RETRIES = 3
# Base backoff between attempts: 0.5 s, 1 s, 2 s, plus jitter so a fleet
# of agents woken by the same outage does not retry in lockstep.
_BACKOFF_BASE_SECONDS = 0.5


def _retry_delay(attempt: int, response: httpx.Response | None) -> float:
    """Seconds to wait before attempt number `attempt` (1-based retry)."""
    if response is not None and response.status_code == 429:
        # The server tells us how long to wait — believe it, within reason.
        try:
            asked = float(response.headers.get("retry-after", ""))
            if 0 <= asked <= 60:
                return asked
        except (TypeError, ValueError):
            pass
    return _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.25)


def _raise_for_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    error_type = message = field = None
    payload: dict[str, Any] = {}
    try:
        payload = response.json()
        error = payload.get("error") or {}
        error_type = error.get("type")
        message = error.get("message")
        field = error.get("field")
    except ValueError:
        pass
    raise HakiApiError(
        message or f"HTTP {response.status_code}: {response.text[:200]}",
        status_code=response.status_code,
        error_type=error_type,
        field=field,
        payload=payload,
    )


class HakiClient:
    """Synchronous client for the Haki API.

    `transport` is injectable for tests (e.g. httpx.ASGITransport(app=app)).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            # Fail fast on connect/write/pool, generous on read: the server
            # answers memory reads in ~100 ms but a consolidation pass can
            # legitimately hold the socket for seconds.
            timeout=httpx.Timeout(10.0, read=timeout, write=10.0, pool=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "HakiClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(
        self, method: str, path: str, *, retry: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """One HTTP call with retries on transient failures.

        Retries 429/500/502/503/504 and network errors with exponential
        backoff (+ Retry-After on 429). Pass `retry=False` for calls that
        must run exactly once — today only `create_key`, where a retry
        after an ambiguous timeout would mint a second key and orphan the
        first (the clear value is returned once and never listable).
        Every other route is safe to replay: reads are reads, capture
        deduplicates by content, and forget/feedback/resolve are
        idempotent by ledger transition.
        """
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.HTTPError as exc:
                last_error = HakiConnectionError(
                    f"cannot reach Haki at {self._http.base_url}: {exc}"
                )
                response = None
            else:
                if response.is_success or response.status_code not in _RETRYABLE_STATUS:
                    _raise_for_error(response)
                    return response.json()
                last_error = None
            if not retry or attempt >= _MAX_RETRIES:
                break
            time.sleep(_retry_delay(attempt + 1, response))
        if last_error is not None:
            raise last_error
        _raise_for_error(response)
        return response.json()

    # -- API ---------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def capture(
        self, events: list[dict[str, Any]], idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Ingest events (contract B.1). Idempotent per idempotency_key."""
        return self._request(
            "POST",
            "/v1/capture",
            json={"events": events, "idempotency_key": idempotency_key},
        )

    def context(
        self,
        subject_id: str,
        query: str,
        project_id: str,
        *,
        purpose: str | None = None,
        budget_tokens: int = 3000,
        exclude_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble a ContextPacket. Returns {packet, token_count, trace_id}.

        `exclude_ids` asks for the NEXT PAGE of the same ranked list: pass
        what an earlier packet already gave you (see haki.runtime.seen) and
        those items are dropped before ranking. Ask again with the SAME
        query -- rewriting it with what you just read is measurably worse.
        This is a further page, not a second hop.
        """
        return self._request(
            "POST",
            "/v1/context",
            json={
                "project_id": project_id,
                "subject_id": subject_id,
                "query": query,
                "purpose": purpose,
                "budget_tokens": budget_tokens,
                "exclude_ids": exclude_ids,
            },
        )

    def inspect(
        self, trace_id: str, *, project_id: str, subject_id: str
    ) -> dict[str, Any]:
        """Full decision trace of a context call (scope mandatory)."""
        return self._request(
            "GET",
            f"/v1/inspect/{trace_id}",
            params={"project_id": project_id, "subject_id": subject_id},
        )

    def timeline(self, subject_id: str, project_id: str) -> dict[str, Any]:
        """Events of one subject, ordered by occurred_at."""
        return self._request(
            "GET",
            "/v1/timeline",
            params={"project_id": project_id, "subject_id": subject_id},
        )

    def consolidate(self) -> dict[str, Any]:
        """Process pending/failed consolidation jobs now. Returns {processed}.

        Dev/ops endpoint: it drains the pending jobs of EVERY project on the
        server, on a session without RLS scoping. Fine on a local dev
        server, wrong against a shared one — prefer `consolidate_subject`
        whenever the subject is known.
        """
        return self._request("POST", "/v1/consolidate")

    def consolidate_subject(
        self, *, project_id: str, subject_id: str
    ) -> dict[str, Any]:
        """Consolidate one subject's pending jobs now. Returns {processed}.

        The scoped counterpart of `consolidate()`: same synchronous
        "extraction happened, look now" behavior, but bounded to one
        project/subject, so a caller never triggers work on another
        tenant's data and never waits behind it.
        """
        return self._request(
            "POST",
            "/v1/consolidate/subject",
            params={"project_id": project_id, "subject_id": subject_id},
        )

    def facts(
        self, *, project_id: str, subject_id: str, status: str | None = None
    ) -> dict[str, Any]:
        """Memorized facts of one subject. Returns {facts: [...]}.

        Unlike the context packet, this lists facts in EVERY status —
        `status="superseded"` is how a caller sees what a newer value
        replaced, which the packet deliberately never serves.
        """
        params = {"project_id": project_id, "subject_id": subject_id}
        if status is not None:
            params["status"] = status
        return self._request("GET", "/v1/facts", params=params)

    def forget(
        self,
        *,
        project_id: str,
        mode: str = "disable",
        subject_id: str | None = None,
        fact_id: str | None = None,
    ) -> dict[str, Any]:
        """Forget one fact or one subject (exactly one target required).

        Returns {status, mode, scope, forget_id, ...counters}.
        """
        return self._request(
            "POST",
            "/v1/forget",
            json={
                "project_id": project_id,
                "subject_id": subject_id,
                "fact_id": fact_id,
                "mode": mode,
            },
        )

    def feedback(
        self,
        *,
        project_id: str,
        rating: str,
        trace_id: str | None = None,
        fact_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Quality observation on a trace or a fact (exactly one target).
        rating: useful|irrelevant|incorrect. `incorrect` on a fact
        transitions it to `disputed`. Returns {status, feedback_id, fact_status?}.
        """
        return self._request(
            "POST",
            "/v1/feedback",
            json={
                "project_id": project_id,
                "trace_id": trace_id,
                "fact_id": fact_id,
                "rating": rating,
                "comment": comment,
            },
        )

    def resolve_conflict(
        self, conflict_id: str, *, project_id: str, keep_fact_id: str
    ) -> dict[str, Any]:
        """Resolve an open conflict set: keep one fact, supersede the others."""
        return self._request(
            "POST",
            f"/v1/conflicts/{conflict_id}/resolve",
            json={"project_id": project_id, "keep_fact_id": keep_fact_id},
        )

    def create_key(
        self, *, project_id: str, org_id: str, label: str | None = None
    ) -> dict[str, Any]:
        """Create an API key. The clear key is in the response ONCE — store it."""
        return self._request(
            "POST",
            "/v1/keys",
            json={"org_id": org_id, "project_id": project_id, "label": label},
            retry=False,  # never replay: a retry could mint an orphan key
        )

    def list_keys(self) -> dict[str, Any]:
        """Masked key listing (prefix only)."""
        return self._request("GET", "/v1/keys")

    def revoke_key(self, key_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/keys/{key_id}")

    # -- CLI device-code auth (`haki login`) --------------------------------
    # No Authorization header is sent for these two: /device/start and
    # /device/poll are intentionally public (a terminal has no hk_ key yet).

    def cli_device_start(self) -> dict[str, Any]:
        """POST /v1/cli/device/start. Returns {device_code, user_code,
        verification_uri, expires_in, interval}."""
        return self._request("POST", "/v1/cli/device/start")

    def cli_device_poll(self, device_code: str) -> dict[str, Any]:
        """POST /v1/cli/device/poll. Returns {status: pending|approved|expired,
        api_key?, org_id?, project_id?} — api_key is only ever present once,
        on the first 'approved' response (the server consumes the code)."""
        return self._request(
            "POST", "/v1/cli/device/poll", json={"device_code": device_code}
        )


class AsyncHakiClient:
    """Async variant of HakiClient (same methods, awaited)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(10.0, read=timeout, write=10.0, pool=10.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncHakiClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _request(
        self, method: str, path: str, *, retry: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """Async mirror of HakiClient._request (same retry contract)."""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._http.request(method, path, **kwargs)
            except httpx.HTTPError as exc:
                last_error = HakiConnectionError(
                    f"cannot reach Haki at {self._http.base_url}: {exc}"
                )
                response = None
            else:
                if response.is_success or response.status_code not in _RETRYABLE_STATUS:
                    _raise_for_error(response)
                    return response.json()
                last_error = None
            if not retry or attempt >= _MAX_RETRIES:
                break
            await asyncio.sleep(_retry_delay(attempt + 1, response))
        if last_error is not None:
            raise last_error
        _raise_for_error(response)
        return response.json()

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def capture(
        self, events: list[dict[str, Any]], idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/capture",
            json={"events": events, "idempotency_key": idempotency_key},
        )

    async def context(
        self,
        subject_id: str,
        query: str,
        project_id: str,
        *,
        purpose: str | None = None,
        budget_tokens: int = 3000,
        exclude_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/context",
            json={
                "project_id": project_id,
                "subject_id": subject_id,
                "query": query,
                "purpose": purpose,
                "budget_tokens": budget_tokens,
                "exclude_ids": exclude_ids,
            },
        )

    async def inspect(
        self, trace_id: str, *, project_id: str, subject_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/v1/inspect/{trace_id}",
            params={"project_id": project_id, "subject_id": subject_id},
        )

    async def timeline(self, subject_id: str, project_id: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/v1/timeline",
            params={"project_id": project_id, "subject_id": subject_id},
        )

    async def consolidate(self) -> dict[str, Any]:
        return await self._request("POST", "/v1/consolidate")

    async def consolidate_subject(
        self, *, project_id: str, subject_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/consolidate/subject",
            params={"project_id": project_id, "subject_id": subject_id},
        )

    async def facts(
        self, *, project_id: str, subject_id: str, status: str | None = None
    ) -> dict[str, Any]:
        params = {"project_id": project_id, "subject_id": subject_id}
        if status is not None:
            params["status"] = status
        return await self._request("GET", "/v1/facts", params=params)

    async def forget(
        self,
        *,
        project_id: str,
        mode: str = "disable",
        subject_id: str | None = None,
        fact_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/forget",
            json={
                "project_id": project_id,
                "subject_id": subject_id,
                "fact_id": fact_id,
                "mode": mode,
            },
        )

    async def feedback(
        self,
        *,
        project_id: str,
        rating: str,
        trace_id: str | None = None,
        fact_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/feedback",
            json={
                "project_id": project_id,
                "trace_id": trace_id,
                "fact_id": fact_id,
                "rating": rating,
                "comment": comment,
            },
        )

    async def resolve_conflict(
        self, conflict_id: str, *, project_id: str, keep_fact_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/conflicts/{conflict_id}/resolve",
            json={"project_id": project_id, "keep_fact_id": keep_fact_id},
        )

    async def create_key(
        self, *, project_id: str, org_id: str, label: str | None = None
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/keys",
            json={"org_id": org_id, "project_id": project_id, "label": label},
            retry=False,  # never replay: a retry could mint an orphan key
        )

    async def list_keys(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/keys")

    async def revoke_key(self, key_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/v1/keys/{key_id}")

    # -- CLI device-code auth (`haki login`) --------------------------------

    async def cli_device_start(self) -> dict[str, Any]:
        return await self._request("POST", "/v1/cli/device/start")

    async def cli_device_poll(self, device_code: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/v1/cli/device/poll", json={"device_code": device_code}
        )
