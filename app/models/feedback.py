import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Feedback(Base):
    """Quality observation on a trace or a fact (PRD — contrat `feedback`).

    Exactly one of trace_id / fact_id is set (enforced by the API schema).
    A rating `incorrect` on a fact also transitions the fact to `disputed`
    (Ledger transition, handled by the route).
    """

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[str] = mapped_column(String(128))
    trace_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    fact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("facts.id", ondelete="SET NULL")
    )
    rating: Mapped[str] = mapped_column(String(16))  # useful|irrelevant|incorrect
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
