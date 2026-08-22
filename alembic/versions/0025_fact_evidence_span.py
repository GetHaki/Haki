"""Keep the proof the write gate already demanded, and link it to its turn

Revision ID: 0025_fact_evidence_span
Revises: 0024_episode_chunks
Create Date: 2026-08-21

The extraction prompt has required a verbatim `evidence_span` for every
create/supersede since the write gate (M1) landed: a candidate that cannot
quote its source must be emitted as action="reject" with reason
"no_evidence_span" instead. The consolidator asked for it, validated it,
used it as one input to a hash -- and then discarded it. The gate demanded
the proof and kept none of it.

Two columns:

`evidence_span` -- the quote itself. Provenance a caller can verify
("here is the fact, here is the sentence it came from"), and the only
per-fact signal available for measuring extraction quality: today an
extraction failure is visible in aggregate accuracy and nowhere else.

`source_chunk_id` -- the episode chunk (migration 0024) that the span was
found in, resolved at write time. This is the exact fact-to-turn link that
two mechanisms have been approximating:

  - the context window's `fact_source_turn`, which knew the source EVENT
    and had to guess which of its turns by word overlap;
  - key merging (E3), which folded a fact into the index of the whole
    event, so a fact from turn 3 of a twenty-turn session polluted all
    twenty. At chunk granularity the same mechanism becomes LongMemEval's
    K = V + fact -- measured there at +9.4 % recall@k and +5.4 % accuracy
    (arXiv 2410.10813, Table 3).

ON DELETE SET NULL on the chunk reference, deliberately not CASCADE:
chunks are derived data that scripts/backfill_episode_chunks.py can
rebuild at any time, and losing a rebuildable pointer must never take a
ledger fact with it.

Both columns are NULL for every existing fact and stay that way: the span
was never stored, so there is nothing to backfill it from. Facts written
from now on carry it. Nothing degrades in the meantime -- the mechanisms
above fall back to exactly what they did before when the link is absent.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0025_fact_evidence_span"
down_revision: str | None = "0024_episode_chunks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE facts ADD COLUMN evidence_span varchar")
    op.execute(
        "ALTER TABLE facts ADD COLUMN source_chunk_id uuid "
        "REFERENCES episode_chunks(id) ON DELETE SET NULL"
    )
    # Key merging reads this the other way round -- "which facts belong to
    # this chunk" -- once per consolidated event.
    op.execute(
        "CREATE INDEX ix_facts_source_chunk_id ON facts (source_chunk_id) "
        "WHERE source_chunk_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_source_chunk_id")
    op.execute("ALTER TABLE facts DROP COLUMN IF EXISTS source_chunk_id")
    op.execute("ALTER TABLE facts DROP COLUMN IF EXISTS evidence_span")
