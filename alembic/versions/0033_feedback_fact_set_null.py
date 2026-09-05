"""Feedback keeps its history when its fact is erased (B5b).

Revision ID: 0033_feedback_fact_set_null
Revises: 0032_embeddings_1024
Create Date: 2026-09-05

`feedback.fact_id REFERENCES facts(id)` was created in 0006 with the
default NO ACTION behavior. Deleting a subject that has feedback rows
pointing at its facts (`POST /v1/forget {subject_id, mode: delete}`)
therefore died with an IntegrityError (500) instead of erasing.

The fix is ON DELETE SET NULL, not CASCADE: a feedback row is a quality
observation ("this answer was incorrect") whose value survives the
erasure of the fact it pointed at -- the link is cleared, the observation
stays. A NULL fact_id is already a supported state (trace-only feedback
sets trace_id instead of fact_id; see app/models/feedback.py).

Upgrade path: plain `alembic upgrade head`. Existing rows are untouched;
only the constraint behavior changes.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0033_feedback_fact_set_null"
down_revision: str | None = "0032_embeddings_1024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_CONSTRAINT = "fk_feedback_fact_set_null"


def upgrade() -> None:
    # The 0006 CREATE TABLE let Postgres auto-name the FK (conventionally
    # feedback_fact_id_fkey), so drop whatever constraint links
    # feedback.fact_id -> facts.id rather than guessing its name.
    op.execute(
        """
        DO $$
        DECLARE
            r RECORD;
        BEGIN
            FOR r IN
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'feedback'::regclass
                  AND confrelid = 'facts'::regclass
                  AND contype = 'f'
            LOOP
                EXECUTE format('ALTER TABLE feedback DROP CONSTRAINT %I', r.conname);
            END LOOP;
        END
        $$;
        """
    )
    op.execute(
        f"ALTER TABLE feedback ADD CONSTRAINT {_NEW_CONSTRAINT} "
        "FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE SET NULL"
    )


def downgrade() -> None:
    op.execute(f"ALTER TABLE feedback DROP CONSTRAINT IF EXISTS {_NEW_CONSTRAINT}")
    op.execute(
        "ALTER TABLE feedback ADD CONSTRAINT feedback_fact_id_fkey "
        "FOREIGN KEY (fact_id) REFERENCES facts(id)"
    )
