"""Credit ledger (sprint 13): grants, debits, and the lazy monthly free
grant. Applied by the capture route (app/api/routes/capture.py) and
surfaced by the console-facing balance/purchase routes
(app/api/routes/billing.py).

Bitemporal ledger philosophy, consistent with the rest of Haki: every
balance change is BOTH an update to `Organization.credit_balance` (cached,
fast read) and an appended `CreditTransaction` row (audit trail) — written
in the SAME DB transaction, so the two can never drift.

Self-hosted org_ids (free strings, no `organizations` row — the
`docker compose up` / curl bootstrap path) never reach the debit/grant
functions here: `resolve_billable_organization` returns None for them, and
the caller skips all credit logic. Self-hosted capture stays free and
unlimited, exactly as before this sprint.

Concurrency: `try_debit_credit` and `grant_credits` both row-lock the
organization (`SELECT ... FOR UPDATE`, via `Session.refresh(with_for_
update=True)` — NOT a plain `select()`, which would silently keep already-
loaded, possibly-stale attributes instead of re-reading the locked row) so
two captures racing on the same organization can never both observe a
sufficient balance and drive it negative.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import CreditTransaction, Organization

# reason values (documented "enum-like", plain str column — same choice as
# Organization.subscription_status: a value Haki doesn't yet know about is
# never silently dropped, just stored as-is).
REASON_SIGNUP_GRANT = "signup_grant"
REASON_MONTHLY_FREE_GRANT = "monthly_free_grant"
REASON_SUBSCRIPTION_GRANT = "subscription_grant"
REASON_TOPUP_PURCHASE = "topup_purchase"
REASON_CAPTURE_DEBIT = "capture_debit"
REASON_ADMIN_ADJUSTMENT = "admin_adjustment"

# A subscriber already receives billing_cloud_plan_monthly_credits per
# billing cycle (GeniusPay webhook) — the lazy free grant is only for an
# organization with no active/trialing subscription.
_ACTIVE_SUBSCRIPTION_STATUSES = ("active", "trialing")
_FREE_GRANT_INTERVAL = timedelta(days=30)


def _org_uuid_from_org_id(org_id: str) -> uuid.UUID | None:
    """A captured event's/api key's `org_id` string -> the console-
    provisioned Organization.id it refers to, or None.

    Provisioned org_ids are always exactly `org_{uuid}` (minted once in
    app/api/routes/orgs.py). Anything else — including a self-hosted org_id
    that happens to start with "org_", e.g. the "org_cursor_dev" MCP
    default — fails the UUID parse and is correctly treated as self-hosted
    (never provisioned, never billed).
    """
    prefix = "org_"
    if not org_id.startswith(prefix):
        return None
    try:
        return uuid.UUID(org_id[len(prefix) :])
    except ValueError:
        return None


async def resolve_billable_organization(
    session: AsyncSession, org_id: str
) -> Organization | None:
    """The Organization row a captured event's org_id refers to, or None
    when it was never provisioned through the Clerk/console flow
    (self-hosted — always free, never checked)."""
    org_uuid = _org_uuid_from_org_id(org_id)
    if org_uuid is None:
        return None
    return await session.get(Organization, org_uuid)


async def _lock_org(session: AsyncSession, org: Organization) -> Organization:
    """Row-locks `org` (SELECT ... FOR UPDATE) and refreshes its Python
    attributes from that locked row. `org` must already be attached to
    `session` (loaded through it earlier in the same request/transaction).

    `Session.refresh(with_for_update=True)`, deliberately not a plain
    `select(...).with_for_update()`: SQLAlchemy's identity map does not
    overwrite an already-loaded, unexpired object's attributes with a new
    query's row by default, which would make the lock pointless here (a
    concurrent transaction's already-committed balance change would stay
    invisible to this Python object even though the DB row lock was
    correctly acquired). `refresh()` unconditionally repopulates — but,
    confirmed empirically, it does NOT autoflush first the way a plain
    query does: any pending, not-yet-flushed attribute change the caller
    made on `org` before calling this (e.g. the GeniusPay webhook setting
    `org.subscription_status` right before granting subscription credits)
    would otherwise be silently discarded and overwritten by the pre-change
    DB row. The explicit `session.flush()` below closes that gap.
    """
    await session.flush()
    await session.refresh(org, with_for_update=True)
    return org


async def grant_credits(
    session: AsyncSession,
    org: Organization,
    amount: int,
    *,
    reason: str,
    reference: str | None = None,
) -> Organization:
    """Credits `org`'s balance by `amount` and appends the matching
    CreditTransaction row, atomically (row-locked)."""
    if amount <= 0:
        raise ValueError("grant_credits amount must be positive")
    locked = await _lock_org(session, org)
    locked.credit_balance += amount
    session.add(
        CreditTransaction(
            org_id=locked.id,
            delta=amount,
            reason=reason,
            reference=reference,
            balance_after=locked.credit_balance,
        )
    )
    await session.flush()
    return locked


async def try_debit_credit(
    session: AsyncSession,
    org: Organization,
    *,
    reason: str,
    reference: str | None = None,
) -> bool:
    """Debits exactly 1 credit from `org` if the balance allows it
    (row-locked read-check-write, race-safe). Returns False — and writes
    NO CreditTransaction row, changes NO balance — when the balance is
    already 0."""
    locked = await _lock_org(session, org)
    if locked.credit_balance < 1:
        return False
    locked.credit_balance -= 1
    session.add(
        CreditTransaction(
            org_id=locked.id,
            delta=-1,
            reason=reason,
            reference=reference,
            balance_after=locked.credit_balance,
        )
    )
    await session.flush()
    return True


async def maybe_grant_lazy_free_credits(session: AsyncSession, org: Organization) -> None:
    """Lazy monthly free grant (sprint 13 — Haki has no scheduler/cron, so
    this runs inline right before a credit balance check on capture).

    Grants `settings.billing_free_monthly_credits` once per rolling 30-day
    window (`org.free_credits_granted_at` unset, or older than 30 days),
    but only for an organization with no active/trialing subscription.
    """
    if org.subscription_status in _ACTIVE_SUBSCRIPTION_STATUSES:
        return
    now = datetime.now(timezone.utc)
    if (
        org.free_credits_granted_at is not None
        and now - org.free_credits_granted_at < _FREE_GRANT_INTERVAL
    ):
        return
    locked = await grant_credits(
        session, org, settings.billing_free_monthly_credits, reason=REASON_MONTHLY_FREE_GRANT
    )
    locked.free_credits_granted_at = now
    await session.flush()
