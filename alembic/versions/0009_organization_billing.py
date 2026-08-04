"""sprint 12: facturation GeniusPay (colonnes d'abonnement sur organizations)

Revision ID: 0009_organization_billing
Revises: 0008_organizations
Create Date: 2026-08-04

Ajoute a `organizations` ce qu'il faut pour suivre un abonnement GeniusPay
sans table separee : le volume est faible (un abonnement actif par
organisation, V1 mono-plan), donc quatre colonnes nullable suffisent plutot
qu'une table `subscriptions` a part.

`geniuspay_subscription_id` est indexee en UNIQUE : c'est la cle de
correlation des webhooks entrants (POST /v1/webhooks/geniuspay) -- la
payload GeniusPay ne connait pas l'id Haki de l'organisation, seulement son
propre id d'abonnement, donc chaque webhook cherche l'organisation PAR
CETTE COLONNE. Nullable : la plupart des organisations n'ont jamais souscrit.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_organization_billing"
down_revision: str | None = "0008_organizations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "organizations", sa.Column("geniuspay_subscription_id", sa.String(128))
    )
    op.add_column("organizations", sa.Column("subscription_status", sa.String(32)))
    op.add_column("organizations", sa.Column("subscription_plan", sa.String(64)))
    op.add_column(
        "organizations",
        sa.Column("current_period_end", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_organizations_geniuspay_subscription_id",
        "organizations",
        ["geniuspay_subscription_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_organizations_geniuspay_subscription_id", table_name="organizations"
    )
    op.drop_column("organizations", "current_period_end")
    op.drop_column("organizations", "subscription_plan")
    op.drop_column("organizations", "subscription_status")
    op.drop_column("organizations", "geniuspay_subscription_id")
