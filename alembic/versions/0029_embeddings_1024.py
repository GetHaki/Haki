"""Embedding columns widen to vector(1024) for a retrieval-trained model

Revision ID: 0029_embeddings_1024
Revises: 0028_embedding_space
Create Date: 2026-08-24

The default embedder becomes intfloat/multilingual-e5-large (1024 dims),
replacing paraphrase-multilingual-MiniLM-L12-v2 (384).

WHY, measured -- retrieval bench, LoCoMo, budget 3000, paired McNemar on
the per-question dumps (eval/compare_dumps.py), NOT on the gap between two
percentages:

    same 1 525 LoCoMo questions through both models, McNemar exact on
    the ones whose outcome changed:

        any        88.5 -> 92.7 %    82 won,  17 lost    p < 0.0001
        complete   75.2 -> 81.1 %   117 won,  27 lost    p < 0.0001

    Every category improves on both metrics. The largest single move is
    multi-hop COMPLETE, 40.8 -> 50.7 %: the axis on which a packed context
    was losing to replaying the whole history. On the 231-question slice
    this was first measured on, `complete` was +3.9 at p = 0.09 -- not
    enough to justify invalidating every stored vector, which is why the
    full run exists.

The gain is NOT the extra dimensions. Ablation on the same bench with
paraphrase-multilingual-mpnet-base-v2 -- the same FAMILY as the old
default, twice the size, 768 dims -- scored 86.6 % / 77.1 %, BELOW the
384-dimensional default it doubles. Model size buys nothing here. What
buys the gain is that e5 is trained for retrieval on (query, passage)
pairs and applies a different prefix on each side, which
app/providers/local.py has applied since 22 aout and which the old default
(a symmetric paraphrase model) has no use for. Anyone tempted to revisit
this should re-read that ablation before assuming "bigger is better".

WHAT THIS COSTS, so that it is in the history and not only in a review:

- one query embedding goes from 6.3 ms to 96.0 ms p50 (2-core container,
  same machine, MiniLM measured before AND after e5 at 6.3 / 6.3 to rule
  out drift); `build_context` goes from ~50 ms to ~125 ms p50 / 181 ms
  p95 on that machine -- a ratio of ~2.5x, which is the part that
  transfers to other hardware. The absolute numbers do not;
- the model is 2.24 GB of weights on disk, ~1.56 GB resident per process
  (measured, VmRSS after a warm embed -- not shared across workers: PSS
  across three concurrent processes was 1 529 MB private of 1 561 MB
  resident). A deployment running N uvicorn workers pays it N times: this
  raises the floor on what it takes to self-host Haki, and that is a
  product decision as much as a technical one;
- every stored vector becomes meaningless and MUST be recomputed. A
  384-dimensional vector cannot be cast to 1024, and even if it could, a
  vector from another model is not comparable to one from this one.

Hence the shape of this migration: the columns are dropped and recreated
(NULL everywhere), exactly as 0003 did going from 1536 to 384, and
`embedding_space.backfilled_at` is set to NULL so that
`app.db.verify_embedding_space` warns at every start until
`scripts/backfill_embeddings.py` has finished. The upgrade path is:

    alembic upgrade head
    uv run python -m scripts.backfill_embeddings

Between those two commands the dense axis is empty and retrieval runs on
the lexical axis alone. That window is DEGRADED, not broken, and it is
loud -- which is the whole reason 0028 exists.

The hnsw indexes are recreated empty, which is also why the backfill is
fast to index: pgvector builds hnsw incrementally on INSERT/UPDATE.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0029_embeddings_1024"
down_revision: str | None = "0028_embedding_space"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_MODEL = "intfloat/multilingual-e5-large"
_NEW_DIM = 1024
_OLD_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_OLD_DIM = 384

# (table, index name) -- facts has an hnsw index since 0003, events since
# 0007, episode_chunks since 0024.
_VECTOR_TABLES = (
    ("facts", "ix_facts_embedding_hnsw"),
    ("episode_chunks", "ix_episode_chunks_embedding_hnsw"),
    ("events", "ix_events_embedding_hnsw"),
)


def _rebuild(dim: int) -> None:
    for table, index in _VECTOR_TABLES:
        op.execute(f"DROP INDEX IF EXISTS {index}")
        # DROP + ADD rather than ALTER TYPE: there is no meaningful cast
        # between two vector widths, and pgvector rejects the ALTER on a
        # non-empty column anyway.
        op.execute(f"ALTER TABLE {table} DROP COLUMN embedding")
        op.execute(f"ALTER TABLE {table} ADD COLUMN embedding vector({dim})")
        op.execute(
            f"CREATE INDEX {index} ON {table} USING hnsw (embedding vector_cosine_ops)"
        )


def upgrade() -> None:
    _rebuild(_NEW_DIM)
    op.execute(
        sa.text(
            "UPDATE embedding_space SET model = :model, dim = :dim, "
            "backfilled_at = NULL, updated_at = now() WHERE id = 1"
        ).bindparams(model=_NEW_MODEL, dim=_NEW_DIM)
    )


def downgrade() -> None:
    _rebuild(_OLD_DIM)
    op.execute(
        sa.text(
            "UPDATE embedding_space SET model = :model, dim = :dim, "
            "backfilled_at = NULL, updated_at = now() WHERE id = 1"
        ).bindparams(model=_OLD_MODEL, dim=_OLD_DIM)
    )
