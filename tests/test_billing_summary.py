"""GET /v1/billing/summary (sprint 18): the caller's own billing numbers,
authenticated by their hk_ key — for the console's API-key login.

- key-scoped by construction (org derived from the key, no parameters);
- read-only (no checkout/grant/portal);
- every OTHER /v1/billing/* route still requires the console service
  secret (a customer key must not reach them).
"""

from datetime import datetime, timedelta, timezone

from app.auth import generate_key, hash_key
from app.db import async_session
from app.models import ApiKey, CreditTransaction, Organization


async def _make_org_with_key(
    owner_ref: str, *, plan: str | None = "scale", balance: int = 150_000
) -> tuple[Organization, str]:
    async with async_session() as session:
        org = Organization(
            owner_ref=owner_ref,
            name="Ada",
            subscription_status="active",
            subscription_plan=plan,
            current_period_end=datetime.now(timezone.utc) + timedelta(days=30),
            credit_balance=balance,
        )
        session.add(org)
        await session.flush()
        clear = generate_key()
        session.add(
            ApiKey(
                key_hash=hash_key(clear),
                prefix=clear[:8],
                org_id=f"org_{org.id}",
                project_id=f"prj_{org.id}_default",
                label="summary-test",
            )
        )
        session.add(
            CreditTransaction(
                org_id=org.id,
                delta=balance,
                reason="subscription_grant",
                reference="test",
                balance_after=balance,
            )
        )
        await session.commit()
        return org, clear


def _auth(clear: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {clear}"}


async def test_summary_returns_the_keys_own_numbers(client, auth_required):
    org, clear = await _make_org_with_key("user_summary_a")
    response = await client.get("/v1/billing/summary", headers=_auth(clear))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["org_id"] == str(org.id)
    assert body["credit_balance"] == 150_000
    assert body["low_balance"] is False
    assert body["subscription_status"] == "active"
    assert body["subscription_plan"] == "scale"
    assert body["current_period_end"] is not None
    assert len(body["transactions"]) == 1
    assert body["transactions"][0]["delta"] == 150_000


async def test_summary_flags_low_balance(client, auth_required):
    _, clear = await _make_org_with_key("user_summary_low", balance=10)
    response = await client.get("/v1/billing/summary", headers=_auth(clear))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["credit_balance"] == 10
    assert body["low_balance"] is True


async def test_summary_needs_a_key(client, auth_required):
    response = await client.get("/v1/billing/summary")
    assert response.status_code == 401


async def test_summary_is_scoped_to_the_keys_org(client, auth_required):
    org_a, clear_a = await _make_org_with_key("user_summary_scope_a", balance=1_000)
    org_b, clear_b = await _make_org_with_key("user_summary_scope_b", balance=2_000)
    assert org_a.id != org_b.id
    for clear, org, balance in ((clear_a, org_a, 1_000), (clear_b, org_b, 2_000)):
        response = await client.get("/v1/billing/summary", headers=_auth(clear))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["org_id"] == str(org.id)
        assert body["credit_balance"] == balance


async def test_other_billing_routes_still_reject_a_customer_key(client, auth_required):
    """The /v1/billing/summary exception must not open the rest of the
    billing surface: status stays on the console service secret."""
    _, clear = await _make_org_with_key("user_summary_locked")
    response = await client.get(
        "/v1/billing/status", params={"owner_ref": "user_summary_locked"}, headers=_auth(clear)
    )
    assert response.status_code == 401, response.text
