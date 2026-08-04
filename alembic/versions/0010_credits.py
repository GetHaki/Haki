"""sprint 13: credits (facturation a l'usage, remplace les paliers)

Revision ID: 0010_credits
Revises: 0009_organization_billing
Create Date: 2026-08-04

Modele economique : 1 credit = 1 evenement accepte par POST /v1/capture.
La recuperation (GET /v1/context, /v1/inspect, ...) n'est jamais facturee.
Trois facons d'obtenir des credits : octroi gratuit mensuel (lazy, pas de
scheduler dans Haki), abonnement Cloud GeniusPay (20 000 credits par
paiement reussi), achat ponctuel ("top-up", paiement GeniusPay unique).

`organizations.credit_balance` est un solde cache, TOUJOURS mis a jour dans
la meme transaction que la ligne `credit_transactions` correspondante
(app/billing/credits.py) — jamais de derive possible entre les deux, meme
philosophie "ledger, jamais de mutation silencieuse" que le reste de Haki.

`credit_transactions` n'a pas de RLS : comme `organizations` et `api_keys`,
ce n'est pas une table isolee par projet (isolee par organisation, et
uniquement accessible via les routes billing protegees par le secret
console, ou en interne par la route capture).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_credits"
down_revision: str | None = "0009_organization_billing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column(
            "credit_balance", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "organizations",
        sa.Column("free_credits_granted_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "credit_transactions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "org_id",
            sa.Uuid(),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column("delta", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("reference", sa.String(256)),
        sa.Column("balance_after", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_credit_transactions_org_created",
        "credit_transactions",
        ["org_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_credit_transactions_org_created", table_name="credit_transactions"
    )
    op.drop_table("credit_transactions")
    op.drop_column("organizations", "free_credits_granted_at")
    op.drop_column("organizations", "credit_balance")
