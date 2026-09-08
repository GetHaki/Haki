"""GeniusPay billing (sprint 12): console-driven checkout + webhook ingestion.

Two unrelated trust boundaries share this router:

- POST /v1/billing/checkout and GET /v1/billing/status are called ONLY by
  the console's Next.js BACKEND (never a browser directly), authenticated
  by the same shared secret as POST /v1/orgs/provision
  (HAKI_CONSOLE_SERVICE_KEY) — not by a customer hk_ key. Excluded from
  ApiKeyAuthMiddleware (app/auth.py) the same way /v1/orgs already is.
- POST /v1/webhooks/geniuspay is called by GeniusPay's own servers,
  authenticated by an HMAC signature (X-GeniusPay-Signature header) over
  the raw request body — never by any Haki credential. Also excluded from
  ApiKeyAuthMiddleware: neither auth model is "an hk_ key bound to a
  project".

Both exclusions live in app/auth.py's `protected` path check.
"""

import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import constant_time_bearer_match
from app.billing.credits import (
    ACTIVE_SUBSCRIPTION_STATUSES,
    CLOUD_PLANS,
    REASON_SUBSCRIPTION_GRANT,
    REASON_TOPUP_PURCHASE,
    grant_credits,
)
from app.billing.dodopayments import (
    DodoClient,
    DodoError,
    PRODUCT_TO_PLAN,
    verify_webhook_signature as verify_dodo_webhook_signature,
)
from app.billing.geniuspay import GeniusPayClient, GeniusPayError, verify_webhook_signature
from app.config import settings
from app.db import get_session
from app.errors import ApiError
from app.models import CreditTransaction, Organization
from app.schemas.billing import (
    BillingStatusResponse,
    CheckoutRequest,
    CheckoutResponse,
    CreditsPurchaseRequest,
    CreditsPurchaseResponse,
    CreditsResponse,
    CreditTransactionOut,
)

logger = logging.getLogger("haki.billing")

router = APIRouter()

_BILLING_CYCLE = "monthly"

# event -> subscription_status fallback, used only when the webhook payload
# itself carries no `data.subscription.status` (the documented payload
# shape always includes one, but defending against a thin/legacy payload
# costs nothing and avoids silently leaving the status stale).
_STATUS_BY_EVENT = {
    "subscription.payment_succeeded": "active",
    "subscription.payment_failed": "past_due",
    "subscription.past_due": "past_due",
    "subscription.cancelled": "cancelled",
}


def _unauthorized(message: str) -> ApiError:
    return ApiError(
        type="unauthorized", message=message, field="Authorization", status_code=401
    )


def _require_console_auth(request: Request) -> None:
    if not settings.console_service_key:
        raise _unauthorized("console billing is not configured")
    if not constant_time_bearer_match(
        request.headers.get("authorization"), settings.console_service_key
    ):
        raise _unauthorized("missing or invalid console service credentials")


async def _org_by_owner_ref(session: AsyncSession, owner_ref: str) -> Organization:
    org = (
        await session.execute(
            select(Organization).where(Organization.owner_ref == owner_ref)
        )
    ).scalars().first()
    if org is None:
        raise ApiError(
            type="org_not_found",
            message="no organization for this owner_ref",
            field="owner_ref",
            status_code=404,
        )
    return org


@router.get("/billing/status", response_model=BillingStatusResponse)
async def billing_status(
    request: Request,
    owner_ref: str = Query(min_length=1, max_length=256),
    session: AsyncSession = Depends(get_session),
) -> BillingStatusResponse:
    _require_console_auth(request)
    org = await _org_by_owner_ref(session, owner_ref)
    return BillingStatusResponse(
        is_subscribed=org.subscription_status in ACTIVE_SUBSCRIPTION_STATUSES,
        subscription_status=org.subscription_status,
        subscription_plan=org.subscription_plan,
        current_period_end=org.current_period_end,
        geniuspay_subscription_id=org.geniuspay_subscription_id,
    )


@router.post("/billing/checkout", response_model=CheckoutResponse, status_code=201)
async def checkout(
    body: CheckoutRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> CheckoutResponse:
    """Creates the subscription checkout for the caller's organization.

    Sprint 17: Dodo Payments (checkout session -> hosted checkout_url).
    The subscription id itself only materializes on the first webhook
    (subscription.active carries it); the checkout response's session_id
    is stored on the org's dodo_subscription_id so the webhook can
    correlate even before the subscription exists.

    GeniusPay path retained below behind `dodo_api_key is None` — never
    exercised against the real API in tests (mocked HTTP transport,
    tests/test_billing.py)."""
    _require_console_auth(request)
    org = await _org_by_owner_ref(session, body.owner_ref)
    plan = CLOUD_PLANS[body.plan]
    billing_page_url = f"{settings.console_base_url}/app/billing"

    if settings.dodo_api_key:
        product_id = getattr(
            settings, f"dodo_product_{body.plan}", ""
        )
        if not product_id:
            raise ApiError(
                type="billing_provider_error",
                message=f"no Dodo product configured for plan {body.plan}",
                status_code=502,
            )
        client = DodoClient()
        try:
            result = await client.create_checkout_session(
                product_id=product_id,
                customer_email=body.customer_email,
                customer_name=body.customer_name or org.name,
                return_url=billing_page_url,
                metadata={"haki_org_id": str(org.id), "haki_plan": body.plan},
            )
        except DodoError as exc:
            raise ApiError(
                type="billing_provider_error", message=str(exc), status_code=502
            ) from exc
        finally:
            await client.aclose()

        session_id = str(result.get("session_id") or "")
        checkout_url = result.get("checkout_url")
        if not session_id or not checkout_url:
            raise ApiError(
                type="billing_provider_error",
                message="Dodo response did not include a session id / checkout url",
                status_code=502,
            )

        # The subscription id arrives on the first webhook; the session id
        # correlates the org until then. Status stays pending until
        # subscription.active — same rule as the GeniusPay path: never
        # trust a synchronous response to mark a subscription paid.
        org.dodo_subscription_id = session_id
        org.subscription_status = "pending"
        org.subscription_plan = body.plan
        await session.commit()

        return CheckoutResponse(
            org_id=org.id,
            provider="dodo",
            session_id=session_id,
            subscription_status=org.subscription_status,
            subscription_plan=org.subscription_plan,
            redirect_url=checkout_url,
        )

    # --- Legacy GeniusPay path (transition only; no new merchants) ------
    if not body.customer_phone:
        raise ApiError(
            type="invalid_payload",
            message="customer_phone is required for the GeniusPay checkout path",
            field="customer_phone",
            status_code=422,
        )

    client = GeniusPayClient()

    # Single billing page for both outcomes: the org's actual subscription
    # state is authoritative via the webhook + a status poll on that page,
    # never via query params on the Stripe redirect back — success_url and
    # cancel_url only decide where the browser lands, not what Haki trusts.
    billing_page_url = f"{settings.console_base_url}/app/billing"

    client = GeniusPayClient()
    try:
        result = await client.create_subscription(
            customer_phone=body.customer_phone,
            customer_name=body.customer_name or org.name,
            customer_email=body.customer_email,
            plan_name=plan["name"],
            amount=plan["price_xof"],
            billing_cycle=_BILLING_CYCLE,
            payment_method="stripe_checkout",
            success_url=billing_page_url,
            cancel_url=billing_page_url,
            metadata={"haki_org_id": str(org.id)},
        )
    except GeniusPayError as exc:
        raise ApiError(
            type="billing_provider_error", message=str(exc), status_code=502
        ) from exc
    finally:
        await client.aclose()

    subscription = result.get("subscription") or {}
    subscription_id = str(subscription.get("uuid") or subscription.get("id") or "")
    if not subscription_id:
        raise ApiError(
            type="billing_provider_error",
            message="GeniusPay response did not include a subscription id",
            status_code=502,
        )
    redirect_url = result.get("redirect_url")

    org.geniuspay_subscription_id = subscription_id
    # NEVER trust GeniusPay's synchronous create-subscription response to
    # mark a subscription active/trialing: the card charge is confirmed
    # asynchronously (the customer completes the hosted Stripe checkout
    # page at `redirect_url`, after this API call already returned) — found
    # live in production (back when this integration was mobile-money/USSD
    # based): a real checkout returned status="active" immediately, before
    # the customer had completed the payment, which showed "abonnement
    # actif" and unlocked PAYG top-ups for an organization that had not
    # actually paid anything. Always "pending" here, no matter what
    # GeniusPay's response claims — only the webhook (POST
    # /v1/webhooks/geniuspay, subscription.payment_succeeded) may ever
    # promote an organization to an ACTIVE_SUBSCRIPTION_STATUSES value.
    org.subscription_status = "pending"
    # The dict key ("starter"/"growth"/"scale"), never GeniusPay's display
    # name — this is what the webhook below looks up CLOUD_PLANS with to
    # know how many credits each cycle payment grants.
    org.subscription_plan = body.plan
    await session.commit()

    return CheckoutResponse(
        org_id=org.id,
        geniuspay_subscription_id=subscription_id,
        subscription_status=org.subscription_status,
        subscription_plan=org.subscription_plan,
        redirect_url=redirect_url,
    )


@router.get("/billing/credits", response_model=CreditsResponse)
async def billing_credits(
    request: Request,
    owner_ref: str = Query(min_length=1, max_length=256),
    session: AsyncSession = Depends(get_session),
) -> CreditsResponse:
    """Current credit balance + the 20 most recent CreditTransaction rows.
    Read-only: unlike a capture, this never triggers the lazy monthly free
    grant (that only fires at the point of actual usage, see
    app/billing/credits.py)."""
    _require_console_auth(request)
    org = await _org_by_owner_ref(session, owner_ref)
    rows = (
        (
            await session.execute(
                select(CreditTransaction)
                .where(CreditTransaction.org_id == org.id)
                .order_by(CreditTransaction.created_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    return CreditsResponse(
        credit_balance=org.credit_balance,
        transactions=[CreditTransactionOut.model_validate(row) for row in rows],
    )


@router.post(
    "/billing/credits/purchase", response_model=CreditsPurchaseResponse, status_code=201
)
async def purchase_credits(
    body: CreditsPurchaseRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> CreditsPurchaseResponse:
    """Creates a ONE-SHOT checkout for a credit top-up — a WRITE call
    against a LIVE payment provider (real financial effect). Credits are
    granted later, by the webhook, once the provider confirms the payment
    actually completed — never here (this endpoint only starts the
    checkout and returns where to redirect the user). Never exercised
    against the real API in tests (mocked HTTP transport,
    tests/test_credits.py).

    Sprint 15: on-demand top-ups (PAYG) are gated to organizations with an
    active/trialing Cloud subscription (`ACTIVE_SUBSCRIPTION_STATUSES`,
    app/billing/credits.py) — a product decision to avoid unlimited PAYG
    access to an org that never subscribed. This affects ONLY the purchase
    of NEW credits: the lazy monthly free grant and spending credits the
    org already holds (POST /v1/capture) are untouched by this check.

    Sprint 17: the Dodo path maps `credits` to a one-time pack product
    (1000/5000/20000 — the console's quick top-ups) and redirects to the
    hosted checkout; the grant rides on `payment.succeeded` (idempotent by
    webhook-id). GeniusPay retained behind `dodo_api_key is None`."""
    _require_console_auth(request)
    org = await _org_by_owner_ref(session, body.owner_ref)

    if org.subscription_status not in ACTIVE_SUBSCRIPTION_STATUSES:
        raise ApiError(
            type="subscription_required",
            message=(
                "On-demand credit top-ups are reserved for active Cloud "
                "subscribers. Subscribe to the Cloud plan first "
                "(POST /v1/billing/checkout), then retry this purchase."
            ),
            status_code=403,
        )

    if settings.dodo_api_key:
        pack_key = f"payg_{body.credits}"
        product_id = getattr(settings, f"dodo_product_{pack_key}", "")
        if not product_id:
            raise ApiError(
                type="invalid_payload",
                message=(
                    f"no PAYG pack for {body.credits} credits — available "
                    "packs: 1000, 5000, 20000"
                ),
                field="credits",
                status_code=422,
            )
        client = DodoClient()
        try:
            result = await client.create_checkout_session(
                product_id=product_id,
                customer_email=body.customer_email,
                customer_name=body.customer_name or org.name,
                return_url=f"{settings.console_base_url}/app/billing",
                metadata={
                    "haki_org_id": str(org.id),
                    "haki_credits": body.credits,
                    "haki_pack": pack_key,
                },
            )
        except DodoError as exc:
            raise ApiError(
                type="billing_provider_error", message=str(exc), status_code=502
            ) from exc
        finally:
            await client.aclose()

        session_id = str(result.get("session_id") or "")
        checkout_url = result.get("checkout_url")
        if not session_id or not checkout_url:
            raise ApiError(
                type="billing_provider_error",
                message="Dodo response did not include a session id / checkout url",
                status_code=502,
            )

        return CreditsPurchaseResponse(
            org_id=org.id,
            credits=body.credits,
            amount_xof=0.0,  # Dodo prices in USD; XOF amount not applicable
            geniuspay_payment_id=session_id,  # correlation id for this checkout
            payment_url=checkout_url,
        )

    # --- Legacy GeniusPay path (transition only) ------------------------
    amount_xof = round(body.credits * settings.billing_credit_price_xof_per_credit, 2)

    client = GeniusPayClient()
    try:
        payment = await client.create_payment(
            customer_phone=body.customer_phone,
            customer_name=body.customer_name or org.name,
            customer_email=body.customer_email,
            amount=amount_xof,
            description=f"Haki credits top-up: {body.credits} credits",
            # The webhook (POST /v1/webhooks/geniuspay) correlates back to
            # this organization and credit count purely from this metadata
            # — a top-up has no dedicated "current payment id" column on
            # Organization the way a subscription does (an org can have
            # many top-ups over time), so it must be self-describing.
            metadata={"haki_org_id": str(org.id), "haki_credits": body.credits},
        )
    except GeniusPayError as exc:
        raise ApiError(
            type="billing_provider_error", message=str(exc), status_code=502
        ) from exc
    finally:
        await client.aclose()

    payment_id = str(payment.get("uuid") or payment.get("id") or "")
    payment_url = payment.get("payment_url") or payment.get("checkout_url") or payment.get("url")
    if not payment_id or not payment_url:
        raise ApiError(
            type="billing_provider_error",
            message="GeniusPay response did not include a payment id/url",
            status_code=502,
        )

    return CreditsPurchaseResponse(
        org_id=org.id,
        credits=body.credits,
        amount_xof=amount_xof,
        geniuspay_payment_id=payment_id,
        payment_url=payment_url,
    )


def _parse_period_end(subscription: dict) -> datetime | None:
    raw = subscription.get("current_period_end") or subscription.get("next_billing_date")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# Documented event name for a completed one-shot payment (sprint 13
# top-ups) — by analogy with subscription.* below, not yet observed on a
# real webhook delivery (no top-up has been exercised against the live API
# yet). The branch below also treats any `data.payment` whose own `status`
# already reads as completed as a completion, so an unexpected event name
# doesn't silently drop a real payment.
_TOPUP_EVENT = "payment.completed"
_TOPUP_COMPLETED_STATUSES = ("completed", "success", "succeeded", "paid")


async def _handle_topup_payment(session: AsyncSession, event: str, payment: dict) -> dict:
    """POST /v1/webhooks/geniuspay, one-shot top-up branch (sprint 13).

    Correlated purely through the payment's own metadata (`haki_org_id`,
    `haki_credits`, set at POST /v1/billing/credits/purchase) — unlike a
    subscription, a top-up has no dedicated "current payment id" column on
    Organization to look it up by (an org can make many top-ups over time).

    Idempotent by `reference` (the GeniusPay payment id): a redelivered
    webhook for a payment already credited is acknowledged without
    granting credits a second time.
    """
    status = payment.get("status")
    if event != _TOPUP_EVENT and status not in _TOPUP_COMPLETED_STATUSES:
        logger.info(
            "geniuspay webhook: payment event ignored (event=%s status=%s)", event, status
        )
        return {"received": True}

    metadata = payment.get("metadata") or {}
    org_id_raw = metadata.get("haki_org_id")
    credits_raw = metadata.get("haki_credits")
    if not org_id_raw or credits_raw is None:
        logger.warning(
            "geniuspay webhook: completed payment without haki_org_id/haki_credits metadata"
        )
        return {"received": True}
    try:
        org_uuid = uuid.UUID(str(org_id_raw))
        credits = int(credits_raw)
    except (ValueError, TypeError):
        logger.warning(
            "geniuspay webhook: malformed top-up metadata (org_id=%r credits=%r)",
            org_id_raw,
            credits_raw,
        )
        return {"received": True}
    if credits <= 0:
        logger.warning("geniuspay webhook: non-positive credits in top-up metadata (%r)", credits)
        return {"received": True}

    org = await session.get(Organization, org_uuid)
    if org is None:
        logger.warning("geniuspay webhook: no organization %s for completed top-up", org_uuid)
        return {"received": True}

    payment_id = str(payment.get("uuid") or payment.get("id") or "") or None
    if payment_id is not None:
        existing = await session.execute(
            select(CreditTransaction.id).where(
                CreditTransaction.org_id == org.id,
                CreditTransaction.reason == REASON_TOPUP_PURCHASE,
                CreditTransaction.reference == payment_id,
            )
        )
        if existing.scalar_one_or_none() is not None:
            logger.info(
                "geniuspay webhook: top-up payment %s already credited, skipping", payment_id
            )
            return {"received": True}

    await grant_credits(session, org, credits, reason=REASON_TOPUP_PURCHASE, reference=payment_id)
    await session.commit()


# --- Dodo webhook (sprint 17: Dodo replaces GeniusPay) -------------------

# event -> Organization.subscription_status. GeniusPay's own vocabulary was
# mirrored before; Dodo's vocabulary maps closely but not 1:1 (on_hold and
# past_due both mean "renewal failed, recoverable", and Dodo also has
# paused/expired, which GeniusPay never sent).
_DODO_STATUS_BY_EVENT = {
    "subscription.active": "active",
    "subscription.renewed": "active",
    "subscription.unpaused": "active",
    "subscription.plan_changed": "active",
    "subscription.past_due": "past_due",
    "subscription.on_hold": "past_due",
    "subscription.paused": "paused",
    "subscription.cancelled": "cancelled",
    "subscription.expired": "cancelled",
    "subscription.failed": "cancelled",
}


async def _handle_dodo_event(
    session: AsyncSession, event: str, data: dict, webhook_id: str
) -> dict:
    """POST /v1/webhooks/dodo — subscription lifecycle branch.

    Correlation: the checkout stores the Dodo subscription id on the org
    (dodo_subscription_id), set by subscription.active's payload the first
    time if a future checkout flow predates it. metadata.haki_org_id (set
    at checkout creation) is the primary key back to the org.

    Idempotency: the Standard-Webhooks `webhook-id` is unique per event
    delivery and Dodo redelivers with the SAME id on retries — a Credit-
    Transaction reference check makes redelivered grants no-ops. This
    closes the replay hole the GeniusPay handler documented (its payload
    had no per-cycle id; Dodo's webhook-id is that key)."""
    subscription = data.get("subscription") or {}
    if not subscription:
        logger.warning("dodo webhook: no subscription object (event=%s)", event)
        return {"received": True}

    subscription_id = str(subscription.get("subscription_id") or "")
    metadata = subscription.get("metadata") or {}
    org_id_raw = metadata.get("haki_org_id") or metadata.get("haki_org_id")

    org: Organization | None = None
    if metadata.get("haki_org_id"):
        try:
            org = await session.get(Organization, uuid.UUID(str(metadata["haki_org_id"])))
        except ValueError:
            org = None
    if org is None and subscription_id:
        org = (
            await session.execute(
                select(Organization).where(
                    Organization.dodo_subscription_id == subscription_id
                )
            )
        ).scalars().first()
    if org is None:
        logger.warning(
            "dodo webhook: no organization for subscription %s (event=%s)",
            subscription_id,
            event,
        )
        return {"received": True}

    if subscription_id and org.dodo_subscription_id != subscription_id:
        org.dodo_subscription_id = subscription_id

    status = subscription.get("status") or _DODO_STATUS_BY_EVENT.get(event)
    if status:
        org.subscription_status = status
    period_end = _parse_period_end(subscription)
    if period_end:
        org.current_period_end = period_end

    # Credit grant on money actually deducted for the cycle: Dodo's own
    # guidance is to use `subscription.renewed` (fires alongside
    # payment.succeeded) as the extend-access signal. `subscription.active`
    # fires at mandate creation — for a $0-trial plan the first REAL charge
    # comes with the trial end's renewed. Granting on renewed only, plus
    # active when there was no trial, is handled by Dodo sending renewed
    # in both cases.
    if event in ("subscription.active", "subscription.renewed"):
        # Plan resolution: prefer the product_id in the payload (which plan
        # the customer actually bought) over org.subscription_plan.
        product_id = subscription.get("product_id") or ""
        plan_key = PRODUCT_TO_PLAN.get(product_id) or org.subscription_plan or "starter"
        plan = CLOUD_PLANS.get(plan_key)
        credits = plan["monthly_credits"] if plan else 0
        if credits > 0:
            await grant_credits(
                session,
                org,
                credits,
                reason=REASON_SUBSCRIPTION_GRANT,
                # webhook-id makes redeliveries no-ops (unique per delivery)
                reference=webhook_id or subscription_id,
            )
        org.subscription_plan = plan_key
    elif event == "subscription.plan_changed":
        product_id = subscription.get("product_id") or ""
        new_plan = PRODUCT_TO_PLAN.get(product_id)
        if new_plan:
            org.subscription_plan = new_plan

    await session.commit()
    return {"received": True}


@router.post("/webhooks/dodo", status_code=200)
async def dodo_webhook(
    request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    """Dodo Payments webhook ingestion (Standard Webhooks spec).

    Signature: webhook-signature header, HMAC-SHA256 of
    `{webhook_id}.{webhook_timestamp}.{body}` with the base64 whsec_ key;
    ±5 min timestamp window. Idempotency: the webhook-id is unique per
    delivery; credit grants are referenced by it, so Dodo's automatic
    retries (up to 8) can never double-credit."""
    raw_body = await request.body()
    if not verify_dodo_webhook_signature(
        request.headers.get("webhook-id"),
        request.headers.get("webhook-timestamp"),
        request.headers.get("webhook-signature"),
        raw_body,
        settings.dodo_webhook_key,
    ):
        raise _unauthorized("invalid or missing webhook signature")

    try:
        payload = json.loads(raw_body) if raw_body else {}
    except ValueError as exc:
        raise ApiError(
            type="invalid_payload", message="malformed JSON body", status_code=422
        ) from exc

    event = payload.get("type", "")
    data = payload.get("data") or {}
    webhook_id = request.headers.get("webhook-id") or ""

    if event.startswith("subscription."):
        return await _handle_dodo_event(session, event, data, webhook_id)

    if event == "payment.succeeded":
        return await _handle_dodo_topup(session, data, webhook_id)

    # credit.* / dunning.* / dispute.* etc. are acknowledged but unused:
    # the credit grant rides on subscription.renewed + payment.succeeded,
    # and Dodo's own credit entitlements are not the ledger of record yet
    # (app/billing/credits.py is).
    logger.info("dodo webhook: event ignored (event=%s)", event)
    return {"received": True}


# PAYG packs: product_id -> credits granted on payment.succeeded.
_DODO_PAYG_PRODUCTS = {
    settings.dodo_product_payg_1000: 1000,
    settings.dodo_product_payg_5000: 5000,
    settings.dodo_product_payg_20000: 20000,
}


async def _handle_dodo_topup(session: AsyncSession, data: dict, webhook_id: str) -> dict:
    """POST /v1/webhooks/dodo, one-shot top-up branch (sprint 17).

    Correlation: the checkout stored `haki_org_id` + the pack product in
    the session metadata; the payment webhook echoes the product_cart it
    was paid for. Idempotent by webhook-id (the grant's reference): Dodo's
    automatic retries can never double-credit."""
    payment = data.get("payment") or {}
    if not payment:
        logger.warning("dodo webhook: payment.succeeded without payment object")
        return {"received": True}

    org_id_raw = (payment.get("metadata") or {}).get("haki_org_id")
    org: Organization | None = None
    if org_id_raw:
        try:
            org = await session.get(Organization, uuid.UUID(str(org_id_raw)))
        except ValueError:
            org = None
    if org is None:
        logger.warning("dodo webhook: no organization for top-up (webhook_id=%s)", webhook_id)
        return {"received": True}

    # Resolve the pack credits from the payment's product cart.
    cart = payment.get("product_cart") or payment.get("products") or []
    credits = 0
    for item in cart:
        pid = item.get("product_id") or item.get("product") or ""
        if pid in _DODO_PAYG_PRODUCTS:
            credits += _DODO_PAYG_PRODUCTS[pid]
    if credits <= 0:
        logger.warning(
            "dodo webhook: top-up payment without a known pack product (webhook_id=%s)",
            webhook_id,
        )
        return {"received": True}

    await grant_credits(
        session,
        org,
        credits,
        reason=REASON_TOPUP_PURCHASE,
        reference=webhook_id,  # unique per delivery -> retries are no-ops
    )
    await session.commit()
    return {"received": True}
