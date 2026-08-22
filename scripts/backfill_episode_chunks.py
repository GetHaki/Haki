"""Cut already-ingested events into episode chunks.

    uv run python scripts/backfill_episode_chunks.py [--batch 200] [--dry-run]

Migration 0024 creates `episode_chunks` EMPTY. Until this has run, every
event ingested before the upgrade is invisible to episodic retrieval --
its facts still work, its verbatim text does not. Run it once per
environment right after `alembic upgrade head`.

No LLM call: chunking is deterministic and the embeddings come from the
configured embedder (local ONNX by default, so also free). What it costs
is one embedding per chunk, which is the same total text as before spread
over more calls.

Idempotent and resumable. Events that already have chunks are skipped, so
interrupting and re-running picks up where it stopped. To re-chunk after a
change to `app.context.chunking`, pass --rechunk: it deletes and rebuilds
the chunks of the events it processes, in the same transaction, so no
event is ever left without episodes.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.context.chunking import chunk_payload
from app.models import EpisodeChunk, Event
from app.providers import get_embedder


async def _run(batch_size: int, dry_run: bool, rechunk: bool) -> int:
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    embedder = get_embedder()
    total_events = 0
    total_chunks = 0
    try:
        while True:
            async with session_factory() as session:
                already = select(EpisodeChunk.event_id).where(
                    EpisodeChunk.event_id == Event.id
                )
                query = select(Event).order_by(Event.occurred_at, Event.id).limit(batch_size)
                if not rechunk:
                    query = query.where(~already.exists())
                else:
                    query = query.offset(total_events)
                events = (await session.execute(query)).scalars().all()
                if not events:
                    break

                pending: list[tuple[Event, int, str]] = []
                for event in events:
                    for ordinal, text in enumerate(chunk_payload(event.kind, event.payload)):
                        pending.append((event, ordinal, text))

                print(
                    f"{len(events)} events -> {len(pending)} chunks "
                    f"(total so far: {total_events} events, {total_chunks} chunks)",
                    flush=True,
                )
                total_events += len(events)
                total_chunks += len(pending)
                if dry_run:
                    continue

                embeddings = await embedder.embed([text for _, _, text in pending])
                if rechunk:
                    await session.execute(
                        delete(EpisodeChunk).where(
                            EpisodeChunk.event_id.in_([event.id for event in events])
                        )
                    )
                session.add_all(
                    [
                        EpisodeChunk(
                            event_id=event.id,
                            ordinal=ordinal,
                            project_id=event.project_id,
                            subject_id=event.subject_id,
                            occurred_at=event.occurred_at,
                            origin_trust=event.origin_trust,
                            text=text,
                            index_text=text,
                            embedding=embedding,
                        )
                        for (event, ordinal, text), embedding in zip(pending, embeddings)
                    ]
                )
                await session.commit()

        async with session_factory() as session:
            remaining = await session.scalar(
                select(func.count())
                .select_from(Event)
                .where(
                    ~select(EpisodeChunk.event_id)
                    .where(EpisodeChunk.event_id == Event.id)
                    .exists()
                )
            )
    finally:
        await engine.dispose()

    print(
        f"done: {total_events} events, {total_chunks} chunks"
        + (" (dry run, nothing written)" if dry_run else "")
    )
    if remaining:
        print(f"warning: {remaining} events still have no chunk")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=200, help="events per transaction")
    parser.add_argument("--dry-run", action="store_true", help="count without writing")
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="rebuild chunks for events that already have them (after a chunker change)",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.batch, args.dry_run, args.rechunk))


if __name__ == "__main__":
    raise SystemExit(main())
