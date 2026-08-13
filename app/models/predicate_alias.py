import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PredicateAlias(Base):
    """Learned predicate-name synonym, scoped to one subject (11 aout
    diagnostic, "l'identite d'un fait n'est pas calculee, elle est devinee
    sur une chaine" — exact canonical key first, alias table second,
    semantic fallback only after both miss, see app.consolidator.
    _resolve_existing_fact).

    Scoped by subject, not just project: a project-wide alias would risk a
    generic short predicate string ("count", "status") learned for one
    subject's vocabulary silently hijacking an unrelated subject's
    genuinely different concept of the same name. Per-subject means each
    subject "rediscovers" a synonym independently — more conservative, no
    cross-subject contamination, at the cost of not generalizing across
    subjects sharing the same convention.

    Auto-populated: app.consolidator._resolve_existing_fact inserts a row
    (INSERT ... ON CONFLICT DO NOTHING — first discovery wins, never
    silently overwritten by a later, possibly noisier, semantic match)
    every time the semantic fallback resolves a candidate to an existing
    fact under a DIFFERENT predicate string. Turns a repeated embedding-
    distance guess into a persisted, deterministic fact about identity:
    the same synonym pair for the same subject only needs to be
    "discovered" once.
    """

    __tablename__ = "predicate_aliases"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "subject_id", "alias_predicate",
            name="uq_predicate_aliases_identity",
        ),
        Index(
            "ix_predicate_aliases_lookup",
            "project_id", "subject_id", "alias_predicate",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[str] = mapped_column(String(128))
    subject_id: Mapped[str] = mapped_column(String(128))
    alias_predicate: Mapped[str] = mapped_column(String(128))
    canonical_predicate: Mapped[str] = mapped_column(String(128))
    # 1 - cosine distance at discovery time — informational (not read by
    # the lookup path, which trusts a registered alias unconditionally
    # once discovered), useful for a future audit/console view.
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
