"""Org settings + member listing with a raw hk_ key (sprint 18).

Data operations on one's own org work key-scoped (org derived from the
key, owner_ref must be absent); access control (invite/remove) stays on
the console service secret (route-level check).
"""

from app.auth import generate_key, hash_key
from app.db import async_session
from app.models import ApiKey, Organization


async def _make_org_with_key(owner_ref: str, *, name: str = "Ada") -> tuple[Organization, str]:
    async with async_session() as session:
        org = Organization(owner_ref=owner_ref, name=name, retention_days=90)
        session.add(org)
        await session.flush()
        clear = generate_key()
        session.add(
            ApiKey(
                key_hash=hash_key(clear),
                prefix=clear[:8],
                org_id=f"org_{org.id}",
                project_id=f"prj_{org.id}_default",
                label="orgs-key-test",
            )
        )
        await session.commit()
        return org, clear


def _auth(clear: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {clear}"}


async def test_key_gets_own_settings(client, auth_required):
    org, clear = await _make_org_with_key("user_orgkey_a")
    response = await client.get("/v1/orgs/settings", headers=_auth(clear))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["org_id"] == f"org_{org.id}"
    assert body["name"] == "Ada"
    assert body["retention_days"] == 90


async def test_key_patches_own_settings(client, auth_required):
    org, clear = await _make_org_with_key("user_orgkey_b")
    response = await client.patch(
        "/v1/orgs/settings", json={"name": "Beta"}, headers=_auth(clear)
    )
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "Beta"
    check = await client.get("/v1/orgs/settings", headers=_auth(clear))
    assert check.json()["name"] == "Beta"
    assert check.json()["org_id"] == f"org_{org.id}"


async def test_key_lists_own_members(client, auth_required):
    org, clear = await _make_org_with_key("user_orgkey_c")
    response = await client.get("/v1/orgs/members", headers=_auth(clear))
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["members"], list)


async def test_key_with_owner_ref_is_rejected(client, auth_required):
    """owner_ref lives in the Clerk namespace: with key auth it must be
    absent, never silently ignored (H2)."""
    _, clear = await _make_org_with_key("user_orgkey_d")
    for method, url, kwargs in (
        ("GET", "/v1/orgs/settings", {"params": {"owner_ref": "user_x"}}),
        ("GET", "/v1/orgs/members", {"params": {"owner_ref": "user_x"}}),
        ("PATCH", "/v1/orgs/settings", {"json": {"owner_ref": "user_x", "name": "Z"}}),
    ):
        response = await client.request(method, url, headers=_auth(clear), **kwargs)
        assert response.status_code == 403, (method, url, response.text)


async def test_key_cannot_invite_or_remove(client, auth_required):
    """Access control stays service-secret-only (route-level check)."""
    _, clear = await _make_org_with_key("user_orgkey_e")
    invite = await client.post(
        "/v1/orgs/members/invite",
        json={"owner_ref": "user_x", "role": "member"},
        headers=_auth(clear),
    )
    assert invite.status_code in (401, 403), invite.text
    remove = await client.post(
        "/v1/orgs/members/user_x/remove",
        json={"owner_ref": "user_x"},
        headers=_auth(clear),
    )
    assert remove.status_code in (401, 403, 404), remove.text


async def test_orgs_settings_needs_some_auth(client, auth_required):
    response = await client.get("/v1/orgs/settings")
    assert response.status_code == 401


async def test_service_secret_path_unchanged(client, auth_required, monkeypatch):
    """The account flow (service secret + owner_ref) works exactly as
    before, including the 422 when owner_ref is missing."""
    from app.config import settings

    monkeypatch.setattr(settings, "console_service_key", "cs_test_secret")
    headers = {"Authorization": "Bearer cs_test_secret"}
    await _make_org_with_key("user_orgkey_f", name="Zeta")
    ok = await client.get(
        "/v1/orgs/settings", params={"owner_ref": "user_orgkey_f"}, headers=headers
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["name"] == "Zeta"
    missing = await client.get("/v1/orgs/settings", headers=headers)
    assert missing.status_code == 422, missing.text
