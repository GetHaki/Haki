"""The corpus records which embedding model produced its vectors

Revision ID: 0028_embedding_space
Revises: 0027_job_started_at
Create Date: 2026-08-24

A stored embedding is only meaningful next to other embeddings from the
SAME model. Nothing in this schema said which model that was, and the
consequences were both silent:

- two models of the same dimension (the default and
  snowflake/snowflake-arctic-embed-s are both 384) could be swapped by an
  environment variable. Every check passed, every INSERT succeeded, and
  query vectors from one embedding space were compared against stored
  vectors from another. Cosine similarity between two unrelated spaces is
  noise, so the dense axis contributed nothing and ranking silently fell
  back to lexical-only;
- a column widened by a migration without a re-embedding pass leaves rows
  whose vector is NULL, invisible to the dense axis, with no way to tell
  "not embedded yet" from "embedded correctly".

`embedding_space` is a single row -- the invariant is a property of the
corpus, not of a row, so this costs one SELECT at startup and touches
none of the eight sites that write an embedding.

    model          the model every stored vector came from
    dim            its dimension, mirroring the vector(N) columns
    backfilled_at  NULL while a re-embedding pass is incomplete

Read by app.db.verify_embedding_space (refuses to serve on a model
mismatch, warns on an incomplete backfill) and written by
scripts/backfill_embeddings.py.

The row is seeded with the model configured at migration time, because on
an existing install that IS the model the stored vectors came from, and
seeding NULL would make the first start warn about a backfill that is not
needed. A fresh install has no rows to be wrong about either way.

This migration does NOT change any dimension: see 0029.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# alembic/env.py already imports app.config to get the database URL, so
# this adds no coupling that migrations did not already have. The seed
# has to come from configuration: an install already running a non-default
# 384-dimensional model would otherwise be recorded as running the default
# and REFUSED at its next start -- a migration that bricks a boot to fix a
# bug that install does not have.
from app.config import settings

revision: str = "0028_embedding_space"
down_revision: str | None = "0027_job_started_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE embedding_space (
            id             smallint PRIMARY KEY DEFAULT 1,
            model          text NOT NULL,
            dim            integer NOT NULL,
            backfilled_at  timestamptz,
            updated_at     timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT embedding_space_singleton CHECK (id = 1)
        )
        """
    )
    # Seeded from the configured model rather than hardcoded, so that an
    # install already running a non-default 384-dimensional model is
    # described accurately instead of being told it has a mismatch. The
    # DIMENSION stays hardcoded: it is a property of the schema at this
    # revision, not of anyone's configuration.
    op.execute(
        sa.text(
            "INSERT INTO embedding_space (id, model, dim, backfilled_at) "
            "VALUES (1, :model, 384, now())"
        ).bindparams(model=settings.embed_model)
    )
    # No RLS: the row describes the deployment, not a project's data, and
    # holds nothing a tenant could read about another tenant.


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_space")
