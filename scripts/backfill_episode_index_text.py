"""One-time backfill for events.index_text on episodes embedded before
migration 0022_episode_index_text (15 aout, mechanisms E1a/E3).

Bug found by code review (b47c275~1..HEAD): the up-front backfill pass in
`run_consolidation` (app/consolidator/__init__.py) only sets `index_text`
for events with `embedding is None`. Episodes already embedded before
0022 keep `index_text` NULL forever -- migration 0022's own docstring
expected them to be "repopulated whenever the subject is revisited by a
new consolidation pass", but a job's `events` list only ever contains that
job's own newly-captured events, never a subject's historical episodes --
so that expectation never actually fires. A NULL `index_text` means a
zero full-text axis (`EPISODE_W_FULLTEXT`) forever, capping the episode's
max achievable score at similarity+recency only (0.75 of 1.0) and ranking
it permanently below any episode indexed after 0022, or any fact.

Sets the same baseline `index_text` the up-front pass would have set at
consolidation time -- kind+payload only, no fact concatenation (this
script cannot know which facts a given event's candidates ultimately
touched; the real E3 key-merged text is only ever computed once, at
consolidation time). Re-embeds from that text with the local embedder --
no LLM/OpenRouter call, purely local ONNX inference, zero cost.

Usage:
    uv run python scripts/backfill_episode_index_text.py [--dry-run] [--batch-size 200]
"""

import argparse
import asyncio

from sqlalchemy import func, select

from app.context import episode_text
from app.db import async_session
from app.models import Event
from app.providers import get_embedder


async def main(dry_run: bool, batch_size: int) -> None:
    if dry_run:
        async with async_session() as session:
            count = (
                await session.execute(
                    select(func.count()).where(
                        Event.index_text.is_(None), Event.embedding.is_not(None)
                    )
                )
            ).scalar_one()
        print(f"Would backfill {count} episode(s). Re-run without --dry-run to apply.")
        return

    embedder = get_embedder()
    total = 0
    while True:
        async with async_session() as session:
            stmt = (
                select(Event)
                .where(Event.index_text.is_(None), Event.embedding.is_not(None))
                .order_by(Event.id)
                .limit(batch_size)
            )
            batch = (await session.execute(stmt)).scalars().all()
            if not batch:
                break

            texts = [episode_text(event.kind, event.payload) for event in batch]
            embeddings = await embedder.embed(texts)
            for event, text, embedding in zip(batch, texts, embeddings):
                event.index_text = text
                event.embedding = embedding
            await session.commit()

            total += len(batch)
            print(f"backfilled {total} episodes so far...")

    print(f"\nBackfilled {total} episode(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="count only, no writes")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main(args.dry_run, args.batch_size))
