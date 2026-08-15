"""Consolidation worker (sprint 2: real extraction via Memory Consolidator).

Processes pending `consolidate` jobs: LLM extraction through the configured
extractor (HAKI_LLM_PROVIDER) + embeddings through the configured embedder
(HAKI_EMBED_PROVIDER), dedupe, supersession, conflict sets. Failed jobs
keep their events and are retried on the next poll.

Sprint 16 fix: this used to run ONE pass and exit — nothing else in the
Dockerfile/deployment ever invoked it again, so every captured event queued
a job that no process would ever process. It now loops forever, polling
every HAKI_WORKER_POLL_SECONDS (default 5s). docker-entrypoint.sh starts
this alongside uvicorn in the same container. Still the documented future
entry point of a real Celery/queue worker (see README — migration path) if
this ever needs to scale past one container.

Run: uv run python -m app.worker
"""

import asyncio
import logging

from app.config import settings
from app.db import async_session
from app.ledger import run_pending_consolidations

logger = logging.getLogger("haki.worker")


async def _run_once() -> int:
    async with async_session() as session:
        done = await run_pending_consolidations(session)
        await session.commit()
    return done


async def main() -> None:
    poll_seconds = settings.worker_poll_seconds
    logger.info("consolidation worker started (poll interval %ss)", poll_seconds)
    while True:
        try:
            done = await _run_once()
            if done:
                logger.info("consolidation: %s job(s) processed", done)
        except Exception:
            # One bad job (or a transient DB blip) must never take the
            # whole worker process down — log it and keep polling.
            logger.exception("consolidation worker: error in poll loop")
        await asyncio.sleep(poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
