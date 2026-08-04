"""sprint 11: organizations (self-serve provisioning)

Revision ID: 0008_organizations
Revises: 0007_episodic_events
Create Date: 2026-08-04

Le bootstrap "premiere cle libre" (app/api/routes/keys.py) ne fonctionne
qu'une fois : au-dela, creer une cle pour un nouvel org_id/project_id exige
HAKI_ADMIN_KEY. En pratique, un deuxieme client ne peut pas s'auto-
provisionner (trouve pendant l'audit production de sprint 10-11, confirme
en le vivant : la seule facon de donner une cle a quelqu'un d'autre etait
un bypass admin manuel).

`organizations` ferme aussi, pour tout ce qui passe par le nouveau chemin
de signup, la faille de collision `(org_id, project_id)` : id genere cote
serveur (jamais choisi par le client), donc jamais devinable/collisionnable
comme les org_id en chaine libre du chemin self-hosted/curl (qui restent
inchanges et rétrocompatibles — cette table est additive).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_organizations"
down_revision: str | None = "0007_episodic_events"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("owner_ref", sa.String(256), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_organizations_owner_ref", "organizations", ["owner_ref"], unique=True
    )


def downgrade() -> None:
    op.drop_index("ix_organizations_owner_ref", table_name="organizations")
    op.drop_table("organizations")
