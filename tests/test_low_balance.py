"""Low-balance alert (sprint 18): the pure predicate behind
GET /v1/billing/credits' `low_balance` flag.

Pure (no DB): these run inside the normal Postgres-backed suite but need
no fixtures themselves.
"""

from app.billing.credits import is_low_balance
from app.config import settings


def test_balance_below_threshold_is_low():
    assert is_low_balance(settings.billing_low_balance_threshold - 1) is True


def test_balance_at_threshold_is_not_low():
    assert is_low_balance(settings.billing_low_balance_threshold) is False


def test_zero_balance_is_low():
    assert is_low_balance(0) is True


def test_healthy_balance_is_not_low():
    assert is_low_balance(settings.billing_low_balance_threshold + 1) is False
    assert is_low_balance(150_000) is False
