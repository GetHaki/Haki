"""Dodo Payments client (sprint 17, 8 sept 2026) — replaces GeniusPay.

Why the switch: GeniusPay bills XOF via Ivorian mobile money only, which
blocks every customer outside UEMA. Dodo is a global Merchant of Record
(190+ countries, 80+ currencies, taxes handled), with Standard-Webhooks
signatures (webhook-id = native idempotency), a hosted customer portal
(cancel + payment method — GeniusPay never had either), and test_mode.

Endpoints confirmed against the live API (8 sept):
- Auth: `Authorization: Bearer *** — Cloudflare 403/1010 on a bare
  python-urllib UA, so the client sends a normal UA header.
- POST /checkouts -> {session_id, checkout_url} (hosted checkout; the
  URL is single-use, 24h, per-customer — never cache it).
- GET /subscriptions/{id}, POST /subscriptions/{id}/cancel? — managed
  mostly through webhooks + the Dodo customer portal; only the status
  read lives here.
- Products: created via POST /products (recurring_price), the ids live
  in settings (dodo_product_starter/growth/scale).

Base URL: https://live.dodopayments.com (live) / https://test.dodopayments.com
(test). HAKI_DODO_ENV chooses; HAKI_DODO_BASE_URL overrides.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Any

import httpx

from app.config import settings


class DodoError(Exception):
    """Non-2xx response from the Dodo API."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Dodo API error {status_code}: {body}")


def _base_url() -> str:
    if settings.dodo_base_url:
        return settings.dodo_base_url
    if settings.dodo_env == "test":
        return "https://test.dodopayments.com"
    return "https://live.dodopayments.com"


class DodoClient:
    """Thin async wrapper. Pass `client=` in tests to inject an
    httpx.AsyncClient(transport=httpx.MockTransport(...))."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        key = api_key if api_key is not None else settings.dodo_api_key
        if not key:
            raise RuntimeError("HAKI_DODO_API_KEY is required")
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.dodo_base_url or self._derived_url(),
            headers={
                "Authorization": f"Bearer {key}",
                # Cloudflare 403/1010 on python-urllib's default UA (found
                # live); a normal-looking UA passes.
                "User-Agent": "Haki/1.0 (+https://gethaki.space)",
            },
            timeout=30.0,
        )

    @staticmethod
    def _derived_url() -> str:
        return (
            "https://test.dodopayments.com"
            if settings.dodo_env == "test"
            else "https://live.dodopayments.com"
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DodoClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise DodoError(response.status_code, response.text)
        if not response.content:
            return {}
        return response.json()

    # --- Checkout sessions -------------------------------------------

    async def create_checkout_session(
        self,
        *,
        product_id: str,
        customer_email: str | None = None,
        customer_name: str | None = None,
        return_url: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /checkouts — creates a hosted checkout session for the
        subscription product. Response: {session_id, checkout_url}; the
        caller redirects the customer to checkout_url. Single-use."""
        payload: dict[str, Any] = {
            "product_cart": [{"product_id": product_id, "quantity": 1}],
            "return_url": return_url,
        }
        customer: dict[str, Any] = {}
        if customer_email:
            customer["email"] = customer_email
        if customer_name:
            customer["name"] = customer_name
        if customer:
            payload["customer"] = customer
        if metadata:
            payload["metadata"] = metadata
        return await self._request("POST", "/checkouts", json=payload)

    # --- Subscriptions (reads; lifecycle is webhook-driven) -----------

    async def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        """GET /subscriptions/{id} — read-only. Statuses seen in the
        webhook stream: active, on_hold, past_due, paused, cancelled,
        expired, failed."""
        return await self._request("GET", f"/subscriptions/{subscription_id}")

    async def cancel_subscription(
        self, subscription_id: str, *, cancel_immediately: bool = True
    ) -> dict[str, Any]:
        """POST /subscriptions/{id}/cancel. WRITE call — real financial
        effect with live credentials; only an actual user-initiated
        cancel (customer portal) calls this."""
        return await self._request(
            "POST",
            f"/subscriptions/{subscription_id}/cancel",
            json={"cancel_immediately": cancel_immediately},
        )

    # --- Portal --------------------------------------------------------

    def customer_portal_url(self, customer_id: str) -> str:
        """The Dodo-hosted customer portal (cancel, update payment
        method, invoices). Linked from the console's billing page."""
        base = (
            "https://portal.dodopayments.com"
            if settings.dodo_env == "live"
            else "https://test-portal.dodopayments.com"
        )
        return f"{base}/{customer_id}"


# --- Standard Webhooks verification (the spec Dodo follows) -------------

def verify_webhook_signature(
    webhook_id: str | None,
    webhook_timestamp: str | None,
    webhook_signature: str | None,
    raw_body: bytes,
    secret: str | None,
) -> bool:
    """Standard Webhooks HMAC-SHA256: `msg_id.timestamp.body` signed with
    a base64-encoded key (whsec_...), signatures base64 too. The header
    may carry several space-separated v1,... signatures — a match on ANY
    one is a pass (per spec). Fail-closed: unset secret or missing
    headers → False. ±5 min replay window on the timestamp."""
    if not secret or not webhook_id or not webhook_timestamp or not webhook_signature:
        return False
    try:
        key = base64.b64decode(secret.removeprefix("whsec_"), validate=False)
        ts = int(webhook_timestamp)
    except (ValueError, TypeError):
        return False
    if abs(time.time() - ts) > 300:
        return False
    msg = f"{webhook_id}.{webhook_timestamp}.{raw_body.decode('utf-8', 'replace')}"
    expected = base64.b64encode(
        hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
    ).decode()
    # Multiple signatures are space-separated; each element is `v1,<b64>`
    # (spec), any match wins. Accept both bare and prefixed forms — the
    # spec prefixes with `v1,`, but tolerate senders omitting it.
    import hmac as _h

    for candidate in webhook_signature.split(" "):
        candidate = candidate.strip()
        if not candidate:
            continue
        bare = candidate.split(",", 1)[-1] if "," in candidate else candidate
        if _h.compare_digest(expected, bare) or _h.compare_digest(
            expected, candidate
        ):
            return True
    return False


DODO_WEBHOOK_PRODUCT_TO_PLAN = {
    "starter": settings.dodo_product_starter,
    "growth": settings.dodo_product_growth,
    "scale": settings.dodo_product_scale,
}

PRODUCT_TO_PLAN = {pid: plan for plan, pid in DODO_WEBHOOK_PRODUCT_TO_PLAN.items()}
