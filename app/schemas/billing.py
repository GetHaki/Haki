import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from app.billing.credits import DEFAULT_CLOUD_PLAN


def _strip_phone_whitespace(value: object) -> object:
    # Real users type mobile-money numbers with spaces ("+225 07 13 47 49
    # 85") — the strict E.164-ish pattern below has no tolerance for that,
    # and the resulting 422 surfaces as an opaque "Invalid request payload"
    # with no indication the fix is just "remove the spaces" (found live).
    if isinstance(value, str):
        return "".join(value.split())
    return value


PhoneNumber = Annotated[str, BeforeValidator(_strip_phone_whitespace), Field(pattern=r"^\+\d{8,15}$")]


class CreditTransactionOut(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    delta: int
    reason: str
    reference: str | None
    balance_after: int
    created_at: datetime


class CreditsResponse(BaseModel):
    credit_balance: int
    transactions: list[CreditTransactionOut]
    # True when credit_balance is below billing_low_balance_threshold — the
    # console shows a "top up" banner. Computed here (not in the console)
    # so every client shares the same threshold.
    low_balance: bool = False


class CreditsPurchaseRequest(BaseModel):
    """Called by the console backend on behalf of an already-verified Clerk
    user — same trust model as CheckoutRequest. A one-shot GeniusPay
    payment (not a subscription), run in checkout mode: `customer_phone` is
    optional (confirmed against the live docs, 2026-08-10) — GeniusPay's
    hosted checkout page collects the customer's phone/payment method
    itself when it's omitted here."""

    owner_ref: str = Field(min_length=1, max_length=256)
    customer_phone: PhoneNumber | None = None
    customer_name: str | None = Field(default=None, max_length=128)
    customer_email: str | None = Field(default=None, max_length=256)
    credits: int = Field(ge=1000)


class CreditsPurchaseResponse(BaseModel):
    org_id: uuid.UUID
    credits: int
    amount_xof: float
    geniuspay_payment_id: str
    payment_url: str


class CheckoutRequest(BaseModel):
    """Called by the console backend on behalf of an already-verified Clerk
    user (see app/api/routes/billing.py) — never by a browser directly."""

    owner_ref: str = Field(min_length=1, max_length=256)
    # Sprint 17 (Dodo): the phone number was a GeniusPay requirement
    # (Ivorian mobile money). Dodo's hosted checkout collects the payment
    # method and billing address itself — customer_phone is now optional
    # and unused by the Dodo flow, kept for the GeniusPay handler.
    customer_phone: PhoneNumber | None = None
    customer_name: str | None = Field(default=None, max_length=128)
    customer_email: str | None = Field(default=None, max_length=256)
    plan: Literal["starter", "growth", "scale"] = DEFAULT_CLOUD_PLAN


class CheckoutResponse(BaseModel):
    org_id: uuid.UUID
    # Provider that owns this checkout ("dodo" | "geniuspay"). GeniusPay
    # kept during the transition; new checkouts are Dodo (sprint 17).
    provider: str = "dodo"
    # Dodo session id (cks_...); empty on the GeniusPay path.
    session_id: str | None = None
    geniuspay_subscription_id: str | None = None
    subscription_status: str | None = None
    subscription_plan: str
    # Hosted checkout page (Dodo's checkout_url / GeniusPay's
    # redirect_url) — the console redirects the customer here. None only
    # if the provider's response omitted it (the console falls back to
    # polling billing status).
    redirect_url: str | None = None


class BillingStatusResponse(BaseModel):
    is_subscribed: bool
    subscription_status: str | None
    subscription_plan: str | None
    current_period_end: datetime | None
    geniuspay_subscription_id: str | None


class PortalRequest(BaseModel):
    """Same trust model as CheckoutRequest: called by the console backend
    on behalf of an already-verified Clerk user — never by a browser."""

    owner_ref: str = Field(min_length=1, max_length=256)
    # The Dodo customer is located by email (the org row stores no
    # Dodo customer id); the console forwards the Clerk user's email.
    customer_email: str | None = Field(default=None, max_length=256)


class PortalResponse(BaseModel):
    org_id: uuid.UUID
    provider: str = "dodo"
    portal_url: str
