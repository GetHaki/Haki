import uuid
from datetime import datetime

from pydantic import BaseModel, Field


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


class CreditsPurchaseRequest(BaseModel):
    """Called by the console backend on behalf of an already-verified Clerk
    user — same trust model as CheckoutRequest. A one-shot GeniusPay
    payment (not a subscription): `customer_phone` is required the same way
    (mobile-money charge), independent of whether the org also has an
    active Cloud subscription."""

    owner_ref: str = Field(min_length=1, max_length=256)
    customer_phone: str = Field(pattern=r"^\+\d{8,15}$")
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
    # GeniusPay requires an Ivorian mobile-money number for the recurring
    # debit (+225 followed by 8-10 digits observed on the live account;
    # kept loose on purpose rather than hard-coding an exact digit count).
    customer_phone: str = Field(pattern=r"^\+\d{8,15}$")
    customer_name: str | None = Field(default=None, max_length=128)
    customer_email: str | None = Field(default=None, max_length=256)


class CheckoutResponse(BaseModel):
    org_id: uuid.UUID
    geniuspay_subscription_id: str
    subscription_status: str | None
    subscription_plan: str


class BillingStatusResponse(BaseModel):
    is_subscribed: bool
    subscription_status: str | None
    subscription_plan: str | None
    current_period_end: datetime | None
    geniuspay_subscription_id: str | None
