"""When a fact is ABOUT, as a typed column

Revision ID: 0026_fact_observed_at
Revises: 0025_fact_evidence_span
Create Date: 2026-08-21

Three instants get confused constantly in a memory system, and until now
Haki stored only two of them:

    recorded_from   when Haki learned it     (already a column)
    valid_from      when it became true      (already a column)
    observed_at     when the fact HAPPENED   (this migration)

"I got pre-approved for my mortgage back in August", said on 30 November:
`valid_from` and `recorded_from` are both 30 November, and the answer to
"when did you get pre-approved?" is August. That August existed -- as a
free-form key inside `value` JSON for absolute dates, or inside
`temporal_range` for resolved relative ones -- in two shapes, typed as
nothing, indexed by nothing, and impossible to compare or order.

Temporal reasoning is the category every published memory system is worst
at: Mem0 55.5 % against 67.1 % on single-hop, OpenAI's memory 21.7 %
(arXiv 2504.19413, Table 1). A date that is not a date is a large part of
why.

Derived, at write time, from what the extractor already produces -- see
app/consolidator/temporal.py. Nothing is asked of the extractor that it
was not already asked, and nothing is removed from `value`: the reader
sees the same JSON it always did.

NULL is the common case and a real answer: most facts ("I have a dog")
are about no particular instant. The index is partial for that reason.

No backfill. Existing facts keep NULL: their dates are still inside
`value`, and re-deriving them would mean re-running a parser over the
whole table for a column nothing reads yet. Facts written from now on
carry it, and `scripts/` can backfill later if a temporal retrieval axis
ever justifies it -- which is deliberately NOT part of this change, since
no measurement supports one yet (the retrieval bench already reads 95.2 %
on the temporal category).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0026_fact_observed_at"
down_revision: str | None = "0025_fact_evidence_span"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE facts ADD COLUMN observed_at timestamptz")
    # Scope-first, like ix_facts_scope_status (migration 0004): any future
    # temporal query is "this subject's facts, ordered by when they
    # happened". Partial, because the column is NULL for most facts.
    op.execute(
        "CREATE INDEX ix_facts_observed_at ON facts "
        "(project_id, subject_id, observed_at) WHERE observed_at IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_observed_at")
    op.execute("ALTER TABLE facts DROP COLUMN IF EXISTS observed_at")
