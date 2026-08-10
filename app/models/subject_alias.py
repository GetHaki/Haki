import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class SubjectAlias(Base):
    """Channel identity -> canonical subject mapping (M4, contract: N channel
    identifiers resolve to 1 canonical subject per project).

    Uniqueness of (project_id, alias_kind, alias_value) is enforced by the
    database: two concurrent registrations of the same alias can never
    diverge. `alias_kind="subject"` rows are merge tombstones: they keep an
    old (merged-away) subject id resolvable, and feed the fragmentation
    detector in app/context.
    """

    __tablename__ = "subject_aliases"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "alias_kind", "alias_value",
            name="uq_subject_aliases_identity",
        ),
        Index("ix_subject_aliases_lookup", "project_id", "alias_value"),
        Index("ix_subject_aliases_canonical", "project_id", "canonical_subject_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[str] = mapped_column(String(128))
    alias_kind: Mapped[str] = mapped_column(String(64))
    alias_value: Mapped[str] = mapped_column(String(256))
    canonical_subject_id: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SubjectMergeReceipt(Base):
    """Timestamped receipt of one subject merge (same philosophy as
    ForgetReceipt): what was merged (source -> target), how much moved
    (counters, e.g. {events_moved, facts_moved, ...}) and exactly which
    rows moved (`moved`: per-table lists of ids) — the information a future
    guarded un-merge needs (see app/ledger/subjects.py for the honest
    reversibility contract).
    """

    __tablename__ = "subject_merge_receipts"
    __table_args__ = (
        Index("ix_subject_merge_receipts_project", "project_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[str] = mapped_column(String(128))
    source_subject_id: Mapped[str] = mapped_column(String(128))
    target_subject_id: Mapped[str] = mapped_column(String(128))
    counters: Mapped[dict] = mapped_column(JSONB, default=dict)
    moved: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
