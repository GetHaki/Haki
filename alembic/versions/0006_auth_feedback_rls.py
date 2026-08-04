"""sprint 6: cles API, feedback, Row-Level Security

Revision ID: 0006_auth_feedback_rls
Revises: 0005_forget_receipts
Create Date: 2026-08-01

- `api_keys` : cles API par projet. Seul le hash sha256 est stocke ; la cle
  en clair (`hk_...`) n'est montree qu'a la creation. `prefix` (8 premiers
  caracteres) sert a l'affichage masque.
- `feedback` : observations de qualite (useful|irrelevant|incorrect) liees
  a une trace ou a un fait (PRD — contrat `feedback`).
- RLS sur events, facts, context_traces, conflict_sets : isolation par
  `project_id = current_setting('haki.project_id', true)`.

Decision documentee (mode dev ouvert) : quand `haki.project_id` n'est PAS
pose dans la transaction (mode dev sans auth, worker interne, serveur MCP),
`current_setting(..., true)` vaut NULL et la policy est permissive. Quand
l'auth resout une cle, `get_session` pose `SET LOCAL haki.project_id` et la
policy devient stricte : une requete qui OUBLIE le filtre project_id dans
le code ne voit que les lignes du projet de la cle (garantie PRD de
non-divulgation, testee dans tests/test_rls.py).

`FORCE ROW LEVEL SECURITY` est indispensable : le role applicatif `haki`
est proprietaire des tables, et le proprietaire contourne RLS sans FORCE.

Deux roles (decision documentee) : les migrations tournent avec le role
proprietaire `haki` (DDL, CREATE EXTENSION) ; le RUNTIME applicatif utilise
`haki_app`, cree ici, qui n'est NI superuser NI proprietaire — un superuser
contourne RLS meme avec FORCE, donc sans ce role dedie la garantie de
non-divulgation serait factice. Mot de passe `haki` : credential de dev
local documente (meme niveau que haki/haki du docker-compose), a remplacer
en deploiement reel.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006_auth_feedback_rls"
down_revision: str | None = "0005_forget_receipts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables isolees par projet via RLS. api_keys et feedback n'y sont pas :
# api_keys doit rester lisible par le middleware d'auth sans contexte
# projet ; le binding projet de feedback est applique au niveau API.
RLS_TABLES = ("events", "facts", "context_traces", "conflict_sets")

POLICY = """
    USING (
        NULLIF(current_setting('haki.project_id', true), '') IS NULL
        OR project_id = current_setting('haki.project_id', true)
    )
"""
# NULLIF(..., '') est indispensable : quand une transaction qui avait fait un
# SET LOCAL haki.project_id se termine, Postgres ne remet PAS le GUC a NULL
# mais a '' (comportement des GUC custom apres revert/RESET). Comme les
# connexions sont mutualisees (pool), une requete ulterieure sans contexte
# lirait '' et la policy `IS NULL` serait fausse -> toutes les lignes
# cachees. '' et NULL signifient donc tous deux « pas de contexte ».

# Runtime role, non privileged: RLS applies to it (see module docstring).
# One statement per op.execute: asyncpg rejects multi-command statements.
CREATE_ROLE = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'haki_app') THEN
        CREATE ROLE haki_app LOGIN PASSWORD 'haki';
    END IF;
END
$$
"""


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE api_keys (
            id UUID PRIMARY KEY,
            key_hash VARCHAR(64) NOT NULL,
            prefix VARCHAR(16) NOT NULL,
            org_id VARCHAR(128) NOT NULL,
            project_id VARCHAR(128) NOT NULL,
            label VARCHAR(128),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            revoked_at TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX ix_api_keys_key_hash ON api_keys (key_hash)")
    op.execute(
        "CREATE INDEX ix_api_keys_project ON api_keys (project_id, created_at)"
    )

    op.execute(
        """
        CREATE TABLE feedback (
            id UUID PRIMARY KEY,
            project_id VARCHAR(128) NOT NULL,
            trace_id UUID,
            fact_id UUID REFERENCES facts(id),
            rating VARCHAR(16) NOT NULL,
            comment TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_feedback_project_fact ON feedback (project_id, fact_id)"
    )

    for table in RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY haki_project_isolation ON {table} {POLICY}")

    op.execute(CREATE_ROLE)
    op.execute("GRANT USAGE ON SCHEMA public TO haki_app")
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE "
        "ON ALL TABLES IN SCHEMA public TO haki_app"
    )
    # Future tables created by the migration role get the same grants.
    op.execute(
        "ALTER DEFAULT PRIVILEGES "
        "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLES TO haki_app"
    )


def downgrade() -> None:
    for table in RLS_TABLES:
        op.execute(f"DROP POLICY IF EXISTS haki_project_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.execute("DROP INDEX IF EXISTS ix_feedback_project_fact")
    op.execute("DROP TABLE IF EXISTS feedback")
    op.execute("DROP INDEX IF EXISTS ix_api_keys_project")
    op.execute("DROP INDEX IF EXISTS ix_api_keys_key_hash")
    op.execute("DROP TABLE IF EXISTS api_keys")
