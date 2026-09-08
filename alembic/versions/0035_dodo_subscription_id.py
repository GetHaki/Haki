"""Add dodo_subscription_id to organizations (sprint 17 — Dodo Payments
replaces GeniusPay as the subscription provider).

GeniusPay's column stays: orgs provisioned through it keep their
historical subscription id. New checkouts write dodo_subscription_id.

Revision ID: 0035
Revises: 0034
"""
from alembic import op
import sqlalchemy as sa

revision = "0035_dodo_subscription_id"
down_revision = "0034_job_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("dodo_subscription_id", sa.String(128), nullable=True),
    )
    op.create_index(
        "ix_organizations_dodo_subscription_id",
        "organizations",
        ["dodo_subscription_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_organizations_dodo_subscription_id", table_name="organizations"
    )
    op.drop_column("organizations", "dodo_subscription_id")
