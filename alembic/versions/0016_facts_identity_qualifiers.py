"""M-eval1: l'identite d'un fait inclut ses qualifieurs, pas seulement son predicat

Revision ID: 0016_facts_identity_qualifiers
Revises: 0015_provenance_trust
Create Date: 2026-08-11

Le premier run d'eval a echelle reelle (LongMemEval knowledge-update) a
montre 66-80% de faux positifs parmi les conflits ouverts : deux faits sans
rapport referentiel fusionnes parce que l'identite d'un fait etait devinee
sur une CHAINE DE CARACTERES (le predicat) au lieu d'etre calculee sur une
cle. Le repli semantique du consolidator matche des sujets proches, pas des
faits identiques : `lower_quartile` contre `upper_quartile`, deux auteurs
de livres differents.

Le correctif d'ensemble sort le qualifieur du NOM du predicat
(`wake_up_time_weekday` devient `wake_up_time` + `{day_type: weekday}`,
voir le prompt d'extraction) et fait des qualifieurs une partie de la cle
d'identite cote consolidator.

Cette migration est ce qui rend ce correctif possible : l'index unique
partiel de 0015 portait sur (project_id, subject_id, predicate) WHERE
status='active'. Des que les qualifieurs sortent du nom du predicat, les
deux faits ci-dessus partagent le predicat `wake_up_time` et ne peuvent
plus etre actifs en meme temps — l'index les rejetterait au niveau DB,
transformant un correctif en panne d'ecriture. Le filet reste donc en
place, mais sur la vraie cle.

`attributed_to` est exclu de la cle : il est pose par le consolidator
(M8) pour dire QUI a affirme le fait, c'est de la provenance, pas une
condition de validite. Deux affirmations de la meme chose par deux
personnes doivent rester un seul fait renforce, pas deux faits actifs
concurrents. L'operateur `jsonb - text` est immutable, donc indexable
directement — pas besoin d'une colonne generee.

Reparation prealable, meme logique que 0015 : des lignes actives peuvent
deja exister avec le meme (predicat, qualifieurs d'identite) — l'ancien
index ne les distinguait pas, mais la reecriture de l'index echouerait sur
elles. On garde la plus recente, les autres passent `superseded` sans
supersedes_id (un doublon n'est pas une lignee de remplacement).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016_facts_identity_qualifiers"
down_revision: str | None = "0015_provenance_trust"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IDENTITY_KEY = "(qualifiers - 'attributed_to')"


def upgrade() -> None:
    # 1) Any active duplicate on the NEW key would make the index creation
    # fail. Under the old index there can be at most one active row per
    # exact predicate, so this can only fire on rows written before 0015 —
    # kept anyway, because a migration that can fail on real data is worse
    # than one redundant UPDATE.
    op.execute(
        f"""
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY project_id, subject_id, predicate,
                                    {_IDENTITY_KEY}
                       ORDER BY recorded_from DESC, id DESC
                   ) AS rn
            FROM facts
            WHERE status = 'active'
        )
        UPDATE facts
        SET status = 'superseded', version = version + 1
        FROM ranked
        WHERE facts.id = ranked.id AND ranked.rn > 1
        """
    )

    # 2) Swap the backstop onto the real identity key. Dropped first: the
    # old index is a strict subset of the new one's constraint, so keeping
    # both would forbid exactly the rows this change exists to allow.
    op.drop_index("uq_facts_active_subject_predicate", table_name="facts")
    op.execute(
        f"""
        CREATE UNIQUE INDEX uq_facts_active_subject_predicate
        ON facts (project_id, subject_id, predicate, {_IDENTITY_KEY})
        WHERE status = 'active'
        """
    )


def downgrade() -> None:
    # Going back to the narrower key can fail on data this migration made
    # legal (two active facts sharing a predicate with different
    # qualifiers). Collapse them the same way the upgrade does rather than
    # leaving the downgrade to blow up mid-flight.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY project_id, subject_id, predicate
                       ORDER BY recorded_from DESC, id DESC
                   ) AS rn
            FROM facts
            WHERE status = 'active'
        )
        UPDATE facts
        SET status = 'superseded', version = version + 1
        FROM ranked
        WHERE facts.id = ranked.id AND ranked.rn > 1
        """
    )
    op.drop_index("uq_facts_active_subject_predicate", table_name="facts")
    op.create_index(
        "uq_facts_active_subject_predicate",
        "facts",
        ["project_id", "subject_id", "predicate"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
