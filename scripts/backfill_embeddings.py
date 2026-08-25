"""Re-embed existing rows after an embedding model/dimension change.

Migration 0029 drops and recreates the `embedding` columns as `vector(1024)`
and sets every row to NULL -- it cannot compute new vectors itself, that
needs the embedder loaded at runtime. This script does that, resumably:

    uv run python -m scripts.backfill_embeddings --status
    uv run python -m scripts.backfill_embeddings
    uv run python -m scripts.backfill_embeddings --batch 500

Resumable by construction: each pass selects `WHERE embedding IS NULL`, so
an interrupted run just leaves fewer rows to pick up next time -- no
checkpoint file, no `--resume` flag, nothing to get out of sync. The
`embedding_space.backfilled_at` timestamp is stamped only after a pass
that embeds every remaining row (a "clean" pass): `verify_embedding_space`
(`app/db.py`) reads that column at startup and only WARNS, never refuses
to boot, while it is NULL -- so a half-finished backfill degrades ranking
(rows with a NULL embedding are invisible to the vector axis, not to the
FTS one) instead of taking the API down.

A row whose source text is empty (`search_text`/`index_text` blank or
whitespace-only) cannot be embedded and would otherwise make the
`WHERE embedding IS NULL` selection pick the same row forever. It gets an
explicit zero vector instead -- distinguishable from "not yet processed"
(NULL) and inert on cosine similarity (a zero vector has no direction, so
it never wins a top-k competition, it also never blocks the pass from
completing).

`embedding_space` (migration 0028) is a single unmodelled row -- read and
written here with raw SQL, the same way `app.db.verify_embedding_space`
reads it, rather than through an ORM model that would exist for this one
table alone.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import func, select, text, update

import app.ledger  # noqa: F401  (import order: see eval/retrieval_bench.py)
from app.db import async_session
from app.models import EpisodeChunk, Event, Fact
from app.providers import EMBEDDING_DIM, get_embedder

# (model, text column) -- the three tables carrying an `embedding` column.
TARGETS = ((Fact, "search_text"), (EpisodeChunk, "index_text"), (Event, "index_text"))
_ZERO_VECTOR = [0.0] * EMBEDDING_DIM


async def _pending_count(session, model) -> int:
    result = await session.execute(
        select(func.count()).select_from(model).where(model.embedding.is_(None))
    )
    return int(result.scalar_one())


async def status() -> None:
    async with async_session() as session:
        row = (
            await session.execute(
                text("SELECT model, dim, backfilled_at FROM embedding_space WHERE id = 1")
            )
        ).first()
        if row is not None:
            print(
                f"embedding_space: model={row.model!r} dim={row.dim} "
                f"backfilled_at={row.backfilled_at}"
            )
        else:
            print("embedding_space: no row yet (migration not applied?)")
        for model, _column in TARGETS:
            pending = await _pending_count(session, model)
            print(f"  {model.__tablename__:<16} pending={pending}")


async def _backfill_one(session, model, column: str, batch: int) -> int:
    """Embed up to `batch` rows still NULL. Returns how many it touched."""
    text_col = getattr(model, column)
    rows = (
        await session.execute(
            select(model.id, text_col).where(model.embedding.is_(None)).limit(batch)
        )
    ).all()
    if not rows:
        return 0

    embedder = get_embedder()
    real_rows = [(row_id, value) for row_id, value in rows if value and value.strip()]
    empty_ids = [row_id for row_id, value in rows if not (value and value.strip())]

    if real_rows:
        vectors = await embedder.embed([value for _row_id, value in real_rows])
        for (row_id, _value), vector in zip(real_rows, vectors):
            await session.execute(
                update(model).where(model.id == row_id).values(embedding=vector)
            )
    for row_id in empty_ids:
        await session.execute(
            update(model).where(model.id == row_id).values(embedding=_ZERO_VECTOR)
        )
    await session.commit()
    return len(rows)


async def backfill(batch: int) -> None:
    total = 0
    async with async_session() as session:
        for model, column in TARGETS:
            while True:
                touched = await _backfill_one(session, model, column, batch)
                total += touched
                if touched:
                    print(f"  {model.__tablename__}: +{touched} embedded ({total} total)")
                if touched < batch:
                    break

        remaining = sum([await _pending_count(session, model) for model, _column in TARGETS])
        if remaining == 0:
            await session.execute(
                text("UPDATE embedding_space SET backfilled_at = now() WHERE id = 1")
            )
            await session.commit()
            print("clean pass: every row embedded, embedding_space.backfilled_at stamped")
        else:
            print(f"{remaining} row(s) still pending -- run again to continue")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status", action="store_true", help="report pending counts, write nothing"
    )
    parser.add_argument(
        "--batch", type=int, default=200, help="rows per table per round-trip (default 200)"
    )
    args = parser.parse_args()
    if args.status:
        asyncio.run(status())
    else:
        asyncio.run(backfill(args.batch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
