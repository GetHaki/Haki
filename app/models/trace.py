import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ContextTrace(Base):
    """Audit trail of one context assembly (PRD — "Confiance développeur").

    `decisions` is a list of {fact_id, action: included|excluded|blocked,
    reason_code} entries explaining every retrieval decision.
    """

    __tablename__ = "context_traces"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[str] = mapped_column(String(128))
    subject_id: Mapped[str] = mapped_column(String(128))
    query: Mapped[str] = mapped_column(Text)
    purpose: Mapped[str | None] = mapped_column(String(128))
    packet: Mapped[dict] = mapped_column(JSONB)
    decisions: Mapped[list] = mapped_column(JSONB, default=list)
    token_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
