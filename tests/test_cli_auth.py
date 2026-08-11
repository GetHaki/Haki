"""CLI device-code auth flow (sprint 14): start -> poll -> approve -> poll,
the single-use consumption guarantee (the security-critical property of the
whole flow), the unknown-code 404, the console-service-key gate on approve,
and that Redis actually enforces the TTL (real Redis, no mock)."""

import re

from app.config import settings
from app.redis_client import redis_client

USER_CODE_RE = re.compile(r"^[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{4}-[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{4}$")


async def _start(client):
    response = await client.post("/v1/cli/device/start")
    assert response.status_code == 201
    return response.json()


async def test_device_start_shape(client, monkeypatch):
    monkeypatch.setattr(settings, "console_base_url", "https://console.example.com")

    body = await _start(client)

    assert len(body["device_code"]) == 64
    int(body["device_code"], 16)  # 64 hex chars, decodes cleanly
    assert USER_CODE_RE.match(body["user_code"])
    assert body["verification_uri"] == "https://console.example.com/cli-auth"
    # RFC 8628 §3.3.1: the same page, code already in the query string, so
    # an agent (or a CLI opening a browser) can relay ONE link instead of a
    # link plus a code to transcribe. The plain uri and the code stay
    # populated alongside it -- a human typing on a phone still needs them.
    assert body["verification_uri_complete"] == (
        f"https://console.example.com/cli-auth?code={body['user_code']}"
    )
    assert body["expires_in"] == 600
    assert body["interval"] == 3

    # Two starts never collide on device_code or user_code.
    other = await _start(client)
    assert other["device_code"] != body["device_code"]
    assert other["user_code"] != body["user_code"]


async def test_full_flow_start_poll_approve_poll_single_use(client, monkeypatch):
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")

    start = await _start(client)
    device_code = start["device_code"]
    user_code = start["user_code"]

    # Pending: the human hasn't approved yet.
    pending = await client.post("/v1/cli/device/poll", json={"device_code": device_code})
    assert pending.status_code == 200
    assert pending.json() == {
        "status": "pending",
        "api_key": None,
        "org_id": None,
        "project_id": None,
    }

    # The console approves on the human's behalf.
    approve = await client.post(
        "/v1/cli/device/approve",
        json={
            "user_code": user_code,
            "api_key": "hk_deadbeefcafef00d",
            "org_id": "org_1",
            "project_id": "prj_1",
        },
        headers={"Authorization": "Bearer svc_test_secret"},
    )
    assert approve.status_code == 200
    assert approve.json() == {"ok": True}

    # First poll after approval: carries the key exactly once.
    approved = await client.post("/v1/cli/device/poll", json={"device_code": device_code})
    assert approved.status_code == 200
    body = approved.json()
    assert body["status"] == "approved"
    assert body["api_key"] == "hk_deadbeefcafef00d"
    assert body["org_id"] == "org_1"
    assert body["project_id"] == "prj_1"

    # Security-critical: the SAME device_code polled again never returns
    # 'approved' (or the key) a second time — it's consumed, treated as
    # unknown from here on.
    replay = await client.post("/v1/cli/device/poll", json={"device_code": device_code})
    assert replay.status_code == 404
    assert replay.json()["error"]["type"] == "device_code_not_found"


async def test_poll_unknown_device_code_is_404(client):
    response = await client.post(
        "/v1/cli/device/poll", json={"device_code": "a" * 64}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "device_code_not_found"


async def test_approve_wrong_or_missing_service_key_is_401(client, monkeypatch):
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")
    start = await _start(client)

    no_header = await client.post(
        "/v1/cli/device/approve",
        json={
            "user_code": start["user_code"],
            "api_key": "hk_x",
            "org_id": "org_1",
            "project_id": "prj_1",
        },
    )
    assert no_header.status_code == 401

    wrong_header = await client.post(
        "/v1/cli/device/approve",
        json={
            "user_code": start["user_code"],
            "api_key": "hk_x",
            "org_id": "org_1",
            "project_id": "prj_1",
        },
        headers={"Authorization": "Bearer not-the-secret"},
    )
    assert wrong_header.status_code == 401

    # And the device_code is still just pending — a rejected approve never
    # leaked the key in.
    pending = await client.post(
        "/v1/cli/device/poll", json={"device_code": start["device_code"]}
    )
    assert pending.json()["status"] == "pending"


async def test_approve_refuses_when_service_key_unconfigured(client, monkeypatch):
    monkeypatch.setattr(settings, "console_service_key", None)
    start = await _start(client)

    response = await client.post(
        "/v1/cli/device/approve",
        json={
            "user_code": start["user_code"],
            "api_key": "hk_x",
            "org_id": "org_1",
            "project_id": "prj_1",
        },
        headers={"Authorization": "Bearer anything"},
    )
    assert response.status_code == 401


async def test_approve_unknown_user_code_is_404(client, monkeypatch):
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")

    response = await client.post(
        "/v1/cli/device/approve",
        json={
            "user_code": "ZZZZ-ZZZZ",
            "api_key": "hk_x",
            "org_id": "org_1",
            "project_id": "prj_1",
        },
        headers={"Authorization": "Bearer svc_test_secret"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "user_code_not_found"


async def test_start_and_poll_are_public_even_when_auth_required(client, auth_required):
    """/device/start and /device/poll must stay reachable with NO
    Authorization header even when HAKI_AUTH_REQUIRED=true — a terminal has
    no hk_ key yet, that's the whole point of this flow (excluded from
    ApiKeyAuthMiddleware, app/auth.py)."""
    start = await client.post("/v1/cli/device/start")
    assert start.status_code == 201

    poll = await client.post(
        "/v1/cli/device/poll", json={"device_code": start.json()["device_code"]}
    )
    assert poll.status_code == 200


async def test_redis_ttl_is_actually_set(client):
    """Real Redis (not a mock): the device_code entry carries a genuine TTL
    close to expires_in, and the user_code mapping does too."""
    start = await _start(client)

    device_ttl = await redis_client.ttl(f"cli:device:{start['device_code']}")
    user_code_ttl = await redis_client.ttl(f"cli:usercode:{start['user_code']}")

    assert 0 < device_ttl <= 600
    assert 0 < user_code_ttl <= 600


async def test_ttl_expiry_makes_a_pending_code_unknown(client):
    """Once Redis actually evicts the key (simulated here by forcing a
    1-second TTL instead of waiting 10 real minutes), the device_code
    behaves exactly like an unknown one: 404, not a stale 'pending'."""
    start = await _start(client)
    key = f"cli:device:{start['device_code']}"

    assert await redis_client.expire(key, 1)
    import asyncio

    await asyncio.sleep(1.5)

    response = await client.post(
        "/v1/cli/device/poll", json={"device_code": start["device_code"]}
    )
    assert response.status_code == 404


# -- code normalization and the RFC 8628 §5.4 attempt cap -----------------


async def _approve(client, user_code: str, *, approver: str | None = "usr_clerk_1"):
    return await client.post(
        "/v1/cli/device/approve",
        json={
            "user_code": user_code,
            "api_key": "hk_deadbeefcafef00d",
            "org_id": "org_acme",
            "project_id": "prj_support",
            "approver_ref": approver,
        },
        headers={"Authorization": "Bearer svc_test_secret"},
    )


async def test_approve_forgives_case_and_separators_in_the_typed_code(
    client, monkeypatch
):
    """The code is compared as an exact Redis key, so "abcd efgh" and
    "ABCDEFGH" would each read as a wrong code -- and burn an attempt --
    for what is only a transcription habit. Normalizing server-side (not in
    the console) means every approving surface forgives the same things."""
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")
    start = await _start(client)
    typed = start["user_code"].replace("-", " ").lower()

    assert (await _approve(client, typed)).status_code == 200

    poll = await client.post(
        "/v1/cli/device/poll", json={"device_code": start["device_code"]}
    )
    assert poll.json()["status"] == "approved"


async def test_approve_rejects_a_code_that_is_not_eight_characters(
    client, monkeypatch
):
    """Normalization forgives cosmetics, never length: a truncated or
    padded code stays the unknown code it is."""
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")
    start = await _start(client)

    assert (await _approve(client, start["user_code"][:-1])).status_code == 404


async def test_wrong_codes_are_capped_per_approver(client, monkeypatch):
    """RFC 8628 §5.4. Approving hands the APPROVER's key to whichever
    terminal holds the code, so guessing a stranger's pending code plants
    an attacker-owned key in their CLI -- every write that terminal makes
    then lands in the attacker's project. Unbounded guessing against a
    32**8 space is not acceptable."""
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")
    approver = "usr_clerk_bruteforcer"
    for attempt in range(10):
        response = await _approve(client, "ZZZZ-ZZZZ", approver=approver)
        assert response.status_code == 404, f"attempt {attempt} should be a plain miss"

    blocked = await _approve(client, "ZZZZ-ZZZZ", approver=approver)
    assert blocked.status_code == 429
    assert blocked.json()["error"]["type"] == "rate_limited"

    # And a REAL code from that approver is now refused too -- the cap is
    # on the person, not on the code they happen to be trying.
    start = await _start(client)
    assert (await _approve(client, start["user_code"], approver=approver)).status_code == 429


async def test_the_cap_is_per_approver_not_shared_by_everyone(client, monkeypatch):
    """Every approval reaches the API from the console backend's single
    address, so an IP-keyed counter would be one bucket shared by all
    users: one person fat-fingering codes would lock out the whole
    instance. Counting per approver is what keeps this safe to enable."""
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")
    noisy, quiet = "usr_clerk_noisy", "usr_clerk_quiet"
    for _ in range(11):
        await _approve(client, "ZZZZ-ZZZZ", approver=noisy)
    assert (await _approve(client, "ZZZZ-ZZZZ", approver=noisy)).status_code == 429

    start = await _start(client)
    assert (await _approve(client, start["user_code"], approver=quiet)).status_code == 200


async def test_successful_approvals_never_count_against_the_cap(client, monkeypatch):
    """Connecting several terminals in a row is normal use, not an attack:
    only FAILED attempts may accumulate."""
    monkeypatch.setattr(settings, "console_service_key", "svc_test_secret")
    approver = "usr_clerk_many_terminals"
    for _ in range(12):
        start = await _start(client)
        assert (
            await _approve(client, start["user_code"], approver=approver)
        ).status_code == 200
