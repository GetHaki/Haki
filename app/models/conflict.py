import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Uuid, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ConflictSet(Base):
    """Group of incompatible facts awaiting resolution (PRD — conflict set).

    While `status` is 'open', the Context Assembler never presents any of
    `fact_ids` as certain (reason_code conflict_open).
    """

    __tablename__ = "conflict_sets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[str] = mapped_column(String(128))
    subject_id: Mapped[str] = mapped_column(String(128))
    fact_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(Uuid), default=list)
    status: Mapped[str] = mapped_column(String(32), default="open")
    # Mechanism C (15 aout, migration 0018): structures what used to live
    # only in the free-text `reason` below -- "contradiction" (two facts
    # genuinely disagree) or "quarantine" (M8 untrusted/lower-trust origin,
    # a single held candidate, never a real disagreement). server_default
    # 'contradiction' is exactly what every pre-migration open set already
    # was: quarantines did not exist as a concept with its own reason
    # format until M8/13 aout, and even then only distinguishable by
    # parsing `reason` text.
    #
    # A "contradiction" set that hits CONFLICT_SET_MAX_MEMBERS on a 3rd
    # competing value is not given a distinct `kind` -- app.consolidator's
    # automatic reclassification-to-event mechanism (see
    # CONFLICT_SET_MAX_MEMBERS) fires on it directly and dissolves it on
    # the spot (status "reclassified_event"), so no set is ever observed
    # sitting in a separate "overflowing" kind.
    kind: Mapped[str] = mapped_column(String(32), default="contradiction", server_default="contradiction")
    reason: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
