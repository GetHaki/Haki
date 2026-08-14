"""HTTP client for the Haki API + eval-project cleanup.

The harness drives Haki exactly like a real integration: capture events,
trigger consolidation, fetch a ContextPacket. Each run lives in its own
project (`prj_eval_<dataset>_<run_id>`) with its own `hk_` key, created via
the admin key (the API is started with HAKI_ADMIN_KEY for eval runs).

Cleanup deletes every row of the run project directly in Postgres (owner
role, bypasses RLS): there is deliberately no delete-project endpoint in
the API (forget is per-subject and part of the product, not of test
hygiene).
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CLEANUP_DSN = "postgresql://haki:haki@localhost:5433/haki"


class HakiClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, admin_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_key = admin_key or os.environ.get("HAKI_EVAL_ADMIN_KEY")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=1800.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, retries: int = 4, **kwargs: Any) -> httpx.Response:
        """POST with retry on transient transport errors (flaky networks)."""
        delay = 2.0
        for attempt in range(retries + 1):
            try:
                return await self._client.post(path, **kwargs)
            except httpx.HTTPError:
                if attempt >= retries:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def create_project_key(self, org_id: str, project_id: str, label: str) -> str:
        headers = {}
        if self.admin_key:
            headers["Authorization"] = f"Bearer {self.admin_key}"
        response = await self._client.post(
            "/v1/keys",
            json={"org_id": org_id, "project_id": project_id, "label": label},
            headers=headers,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "key creation refused: start the API with HAKI_ADMIN_KEY set and pass "
                "the same value to the harness via HAKI_EVAL_ADMIN_KEY "
                "(or use an empty api_keys table for the documented bootstrap)."
            )
        response.raise_for_status()
        return response.json()["key"]

    def _auth(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}"}

    async def capture(self, key: str, events: list[dict[str, Any]]) -> dict:
        response = await self._post(
            "/v1/capture", json={"events": events}, headers=self._auth(key)
        )
        response.raise_for_status()
        return response.json()

    async def consolidate_until_idle(
        self, key: str, project_id: str, max_rounds: int = 40, dsn: str | None = None
    ) -> int:
        """POST /v1/consolidate until the project has NO pending/failed
        consolidate job left. `processed` counts only successful jobs — a
        job that keeps failing (transient LLM/network error) returns
        {"processed": 0}, so the SQL check is the real idle condition.

        Backs off exponentially (2s -> 30s cap) instead of a fixed 2s delay:
        a free-tier LLM provider's per-minute rate limit doesn't reset in
        the ~60s a flat retry schedule used to allow, so every rate-limited
        session simply exhausted max_rounds and failed the whole run."""
        total = 0
        delay = 2.0
        for _ in range(max_rounds):
            response = await self._post("/v1/consolidate", headers=self._auth(key))
            response.raise_for_status()
            total += int(response.json().get("processed", 0))
            remaining = await pending_consolidation_jobs(project_id, dsn)
            if remaining == 0:
                return total
            await asyncio.sleep(delay)
            delay = min(delay * 1.6, 30.0)
        raise RuntimeError(
            f"consolidation did not go idle after {max_rounds} rounds "
            f"({remaining} jobs still pending/failed — check the API logs)"
        )

    async def context(
        self,
        key: str,
        project_id: str,
        subject_id: str,
        query: str,
        budget_tokens: int,
        as_of: datetime | None = None,
    ) -> tuple[dict, float]:
        """POST /v1/context; returns (response body, latency ms).

        `as_of` (14 aout, mecanisme D): the question's own point in time,
        so a conversation dated in the past does not have every volatile
        fact judged stale against today's real wall clock -- see
        research/Diagnostic_Couverture_2026-08-14.md.
        """
        started = time.perf_counter()
        body: dict[str, Any] = {
            "project_id": project_id,
            "subject_id": subject_id,
            "query": query,
            "budget_tokens": budget_tokens,
        }
        if as_of is not None:
            body["as_of"] = as_of.isoformat()
        response = await self._post(
            "/v1/context",
            json=body,
            headers=self._auth(key),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        return response.json(), latency_ms


async def pending_consolidation_jobs(project_id: str, dsn: str | None = None) -> int:
    """Consolidate jobs still pending/failed for a project (owner role)."""
    import asyncpg

    dsn = dsn or os.environ.get("HAKI_EVAL_CLEANUP_DSN", DEFAULT_CLEANUP_DSN)
    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            "SELECT count(*) FROM jobs WHERE kind = 'consolidate' "
            "AND status IN ('pending', 'failed', 'running') "
            "AND payload->>'project_id' = $1",
            project_id,
        )
    finally:
        await conn.close()


async def cleanup_project(project_id: str, dsn: str | None = None) -> dict[str, int]:
    """Delete every row of an eval project. Owner role: bypasses RLS."""
    import asyncpg

    dsn = dsn or os.environ.get("HAKI_EVAL_CLEANUP_DSN", DEFAULT_CLEANUP_DSN)
    conn = await asyncpg.connect(dsn)
    deleted: dict[str, int] = {}
    try:
        for table in (
            "context_traces",
            "conflict_sets",
            "feedback",
            "forget_receipts",
            "facts",
            "events",
            "api_keys",
        ):
            result = await conn.execute(
                f"DELETE FROM {table} WHERE project_id = $1", project_id
            )
            deleted[table] = int(result.split()[-1])
        result = await conn.execute(
            "DELETE FROM jobs WHERE payload->>'project_id' = $1", project_id
        )
        deleted["jobs"] = int(result.split()[-1])
    finally:
        await conn.close()
    return deleted
