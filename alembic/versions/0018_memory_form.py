"""Mechanism C: memory_form (state/event) on facts, kind on conflict_sets

Revision ID: 0018_memory_form
Revises: 0017_predicate_aliases
Create Date: 2026-08-15

Coverage diagnostic from August 14 (research/Diagnostic_Couverture_
2026-08-14.md, Maria case): the current fact model assumes one active
scalar value per (subject, predicate, qualifiers) -- correct for an
attribute ("relationship status"), wrong for a cumulative attribute
("places I've volunteered at"). Every new mention of a cumulative
attribute was being treated as a COMPETING value of the previous one,
opening a ConflictSet -- 20 ConflictSets across 60 facts for a single
subject observed, most of them never served.

`memory_form` distinguishes the two: "state" (current behavior unchanged
-- supersession/conflict) vs "event" (one row per occurrence, never
fused/superseded/put in conflict). server_default 'state' means no
existing row changes behavior.

`conflict_sets.reason` already existed (initial migration) but mixed two
populations that had never been distinguished until now: the M8
quarantine (untrusted origin, must stay hidden) and cap overflow (the
artefact this project fixes). A `kind` column now structures that
distinction instead of leaving it only in `reason` free text --
needed so the automatic reclassification (app.consolidator) can one day
query "which overflows haven't been reclassified yet" without parsing
text.

Finally, the partial unique index added by migration 0016 (at most ONE
active fact per (project_id, subject_id, predicate, qualifiers)) is
correct for "state" -- it is the TOCTOU-dedup guardrail (project #55) --
and wrong for "event": several distinct occurrences share exactly the
same identity (same qualifiers, often empty) and must be able to be
active simultaneously, by definition. Narrowed here to `memory_form =
'state'` rather than removed: the invariant stays fully in force for
scalar facts, and only stops blocking cumulative ones.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018_memory_form"
down_revision: str | None = "0017_predicate_aliases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same expression as migration 0016 -- kept identical so the index this
# migration recreates matches column-for-column, only the WHERE clause
# narrows.
_IDENTITY_KEY = "(qualifiers - 'attributed_to')"


def upgrade() -> None:
    op.add_column(
        "facts",
        sa.Column("memory_form", sa.String(length=16), nullable=False, server_default="state"),
    )
    op.add_column(
        "conflict_sets",
        sa.Column("kind", sa.String(length=32), nullable=False, server_default="contradiction"),
    )
    op.drop_index("uq_facts_active_subject_predicate", table_name="facts")
    op.execute(
        f"""
        CREATE UNIQUE INDEX uq_facts_active_subject_predicate
        ON facts (project_id, subject_id, predicate, {_IDENTITY_KEY})
        WHERE status = 'active' AND memory_form = 'state'
        """
    )


def downgrade() -> None:
    op.drop_index("uq_facts_active_subject_predicate", table_name="facts")
    op.execute(
        f"""
        CREATE UNIQUE INDEX uq_facts_active_subject_predicate
        ON facts (project_id, subject_id, predicate, {_IDENTITY_KEY})
        WHERE status = 'active'
        """
    )
    op.drop_column("conflict_sets", "kind")
    op.drop_column("facts", "memory_form")
