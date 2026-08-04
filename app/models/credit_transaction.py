import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CreditTransaction(Base):
    """Append-only credit ledger line (sprint 13 — usage-based billing).

    Every change to `Organization.credit_balance` is written here IN THE
    SAME DB transaction (see app/billing/credits.py) so the cached balance
    and this audit trail can never drift — same bitemporal-ledger
    philosophy as `Event`/`Fact` elsewhere in Haki: nothing is ever mutated
    silently, every change is an appended row with the balance it produced.

    `reason` is a plain string (enum-like, not a DB enum — same choice as
    `Organization.subscription_status`, so a new value is never silently
    dropped): signup_grant, monthly_free_grant, subscription_grant,
    topup_purchase, capture_debit, admin_adjustment.

    `reference` correlates this line to whatever caused it — a GeniusPay
    payment/subscription id for a grant or purchase, the captured Event id
    for a debit. Free-form, nullable (e.g. admin_adjustment has none).
    """

    __tablename__ = "credit_transactions"
    __table_args__ = (
        Index("ix_credit_transactions_org_created", "org_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organizations.id"), nullable=False
    )
    delta: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(32))
    reference: Mapped[str | None] = mapped_column(String(256))
    balance_after: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
