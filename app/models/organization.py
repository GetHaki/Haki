import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Organization(Base):
    """A signed-up human account (sprint 11 — self-serve provisioning).

    `id` is always server-generated, never client-chosen — unlike the
    free-string `org_id`/`project_id` still used by the self-hosted/curl
    bootstrap path (README), which is how a real client could historically
    collide with another one on a guessed name. Provisioning through this
    table sidesteps that: every org created here gets a guaranteed-unique
    id, and the api_keys it owns are minted with `org_id = f"org_{id}"`.

    `owner_ref` is the external identity provider's user id (Clerk `user_
    ...`). Unique: one human owns exactly one organization in V1 — no
    multi-member orgs yet (documented scope limit, not an oversight).

    Billing (sprint 12 — GeniusPay subscriptions, migration 0009):
    `geniuspay_subscription_id` is the GeniusPay subscription uuid, set once
    at checkout (POST /v1/billing/checkout) and used afterwards as the
    correlation key for inbound webhooks (POST /v1/webhooks/geniuspay) —
    there is no org_id in the GeniusPay payload, only the subscription id,
    so this column is looked up by value, not by primary key. `None` means
    "never subscribed". `subscription_status` mirrors GeniusPay's own
    vocabulary (active/trialing/past_due/cancelled/pending/...) rather than
    a Haki-invented enum, so a new status GeniusPay starts sending shows up
    as-is instead of being silently dropped.

    Credits (sprint 13 — usage-based billing, migration 0010): only an
    organization provisioned here (Clerk/console signup) is ever billed.
    The self-hosted/curl bootstrap path uses free-string org_ids with no
    row in this table — app/billing/credits.py resolves the org_id on a
    captured event back to a row here and, when it finds none, skips all
    credit logic (self-hosted stays free and unlimited, unchanged).
    `credit_balance` is a cached read of the sum of credit_transactions —
    ALWAYS updated in the same DB transaction as the CreditTransaction row
    that changed it (app/billing/credits.py), so the two can never drift.
    `free_credits_granted_at` tracks the lazy monthly free grant (no
    scheduler in Haki: the grant happens on the first capture more than 30
    days after the previous one, or ever).
    """

    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128))
    owner_ref: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    geniuspay_subscription_id: Mapped[str | None] = mapped_column(
        String(128), unique=True, index=True
    )
    subscription_status: Mapped[str | None] = mapped_column(String(32))
    subscription_plan: Mapped[str | None] = mapped_column(String(64))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    credit_balance: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    free_credits_granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
