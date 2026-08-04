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
    reason: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
