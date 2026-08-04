"""Consolidation worker (sprint 2: real extraction via Memory Consolidator).

Processes pending `consolidate` jobs: LLM extraction through the configured
extractor (HAKI_LLM_PROVIDER) + embeddings through the configured embedder
(HAKI_EMBED_PROVIDER), dedupe, supersession, conflict sets. Failed jobs
keep their events and are retried on the next run. This module stays the
future entry point of the Celery worker (see README — migration path).

Run: uv run python -m app.worker
"""

import asyncio

from app.db import async_session
from app.ledger import run_pending_consolidations


async def main() -> None:
    async with async_session() as session:
        done = await run_pending_consolidations(session)
        await session.commit()
    print(f"consolidation: {done} job(s) processed")


if __name__ == "__main__":
    asyncio.run(main())
