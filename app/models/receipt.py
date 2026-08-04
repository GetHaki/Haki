import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ForgetReceipt(Base):
    """Erasure receipt (sprint 4 — embryo of the PRD "recu d'effacement").

    One row per forget operation: what was targeted (scope), how (mode),
    and what happened (counters, e.g. {events_deleted, facts_deleted}).
    """

    __tablename__ = "forget_receipts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[str] = mapped_column(String(128))
    scope: Mapped[str] = mapped_column(String(16))  # fact | subject
    fact_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    subject_id: Mapped[str | None] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(16))  # disable | delete
    counters: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
