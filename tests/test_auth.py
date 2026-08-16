"""API key auth (sprint 6): 401 unauthorized, 403 forbidden_scope,
key management (bootstrap, masked listing, revocation, admin mode),
and the documented dev-open mode.
"""

import uuid

import pytest

from app.auth import constant_time_bearer_match


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# -- constant_time_bearer_match (security review, 16 aout) -------------------
# The 4 shared-secret checks (HAKI_ADMIN_KEY, HAKI_CONSOLE_SERVICE_KEY) used
# plain str comparison before this; behavior must stay identical, only the
# comparison mechanics change.


def test_constant_time_bearer_match_accepts_exact_bearer_header():
    assert constant_time_bearer_match("Bearer svc_secret", "svc_secret") is True


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "svc_secret",  # missing "Bearer " prefix
        "Bearer svc_secre",  # truncated
        "Bearer svc_secretx",  # extra char
        "bearer svc_secret",  # wrong case on the scheme
        "Bearer other_secret",
    ],
)
def test_constant_time_bearer_match_rejects_anything_else(header):
    assert constant_time_bearer_match(header, "svc_secret") is False


def make_event(project_id: str, subject_id: str = "usr_1") -> dict:
    return {
        "org_id": "org_x",
        "project_id": project_id,
        "subject_type": "user",
        "subject_id": subject_id,
        "kind": "conversation.message",
        "occurred_at": "2026-08-01T10:00:00Z",
        "payload": {"role": "user", "content": "hello"},
    }


# -- 401 unauthorized ---------------------------------------------------------


async def test_missing_key_is_401(client, auth_required):
    for response in (
        await client.post("/v1/capture", json={"events": [make_event("prj_a")]}),
        await client.post(
            "/v1/context",
            json={"project_id": "prj_a", "subject_id": "usr_1", "query": "q"},
        ),
        await client.get(
            "/v1/timeline", params={"project_id": "prj_a", "subject_id": "usr_1"}
        ),
    ):
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "unauthorized"


async def test_invalid_key_is_401(client, auth_required):
    response = await client.post(
        "/v1/capture",
        json={"events": [make_event("prj_a")]},
        headers=auth("hk_" + "0" * 32),
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"


async def test_revoked_key_is_401(client, auth_required, make_api_key):
    key = await make_api_key(project_id="prj_a", revoked=True)
    response = await client.post(
        "/v1/capture", json={"events": [make_event("prj_a")]}, headers=auth(key)
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"


# -- 403 forbidden_scope ------------------------------------------------------


async def test_key_of_project_a_cannot_touch_project_b(
    client, auth_required, make_api_key
):
    key = await make_api_key(project_id="prj_a")

    # Capture: project_id lives in each event of the batch.
    response = await client.post(
        "/v1/capture", json={"events": [make_event("prj_b")]}, headers=auth(key)
    )
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "forbidden_scope"

    # Context: project_id in the body.
    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_b", "subject_id": "usr_1", "query": "q"},
        headers=auth(key),
    )
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "forbidden_scope"

    # Timeline: project_id in the query string.
    response = await client.get(
        "/v1/timeline",
        params={"project_id": "prj_b", "subject_id": "usr_1"},
        headers=auth(key),
    )
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "forbidden_scope"


async def test_key_of_project_a_works_on_project_a(
    client, auth_required, make_api_key
):
    key = await make_api_key(project_id="prj_a", org_id="org_x")
    response = await client.post(
        "/v1/capture", json={"events": [make_event("prj_a")]}, headers=auth(key)
    )
    assert response.status_code == 202


# -- dev-open mode ------------------------------------------------------------


async def test_dev_open_mode_allows_calls_without_key(client):
    # conftest default: HAKI_AUTH_REQUIRED=false (no `auth_required` fixture).
    response = await client.post("/v1/capture", json={"events": [make_event("prj_a")]})
    assert response.status_code == 202


# -- key management -----------------------------------------------------------


async def test_bootstrap_first_key_is_free_then_requires_auth(client, auth_required):
    # First creation: free (documented bootstrap, no HAKI_ADMIN_KEY set).
    created = await client.post(
        "/v1/keys",
        json={"org_id": "org_a", "project_id": "prj_a", "label": "bootstrap"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["key"].startswith("hk_")
    assert body["prefix"] == body["key"][:8]

    # Second creation without credentials: refused.
    denied = await client.post(
        "/v1/keys", json={"org_id": "org_a", "project_id": "prj_a"}
    )
    assert denied.status_code == 401


async def test_list_keys_is_masked_and_revocation_works(client, auth_required):
    created = await client.post(
        "/v1/keys", json={"org_id": "org_a", "project_id": "prj_a"}
    )
    key = created.json()["key"]
    key_id = created.json()["id"]

    listing = await client.get("/v1/keys", headers=auth(key))
    assert listing.status_code == 200
    row = listing.json()["keys"][0]
    assert row["prefix"] == key[:8]
    assert "key" not in row and "key_hash" not in row

    revoked = await client.delete(f"/v1/keys/{key_id}", headers=auth(key))
    assert revoked.status_code == 200

    # A revoked key no longer authenticates anything.
    response = await client.get(
        "/v1/timeline",
        params={"project_id": "prj_a", "subject_id": "usr_1"},
        headers=auth(key),
    )
    assert response.status_code == 401


async def test_admin_key_mode_protects_key_management(
    client, auth_required, monkeypatch
):
    from app.config import settings

    monkeypatch.setattr(settings, "admin_key", "adm_test_secret")

    denied = await client.post(
        "/v1/keys", json={"org_id": "org_a", "project_id": "prj_a"}
    )
    assert denied.status_code == 401

    created = await client.post(
        "/v1/keys",
        json={"org_id": "org_a", "project_id": "prj_a"},
        headers=auth("adm_test_secret"),
    )
    assert created.status_code == 201

    # A non-admin hk_ key cannot manage keys in admin mode.
    denied_list = await client.get("/v1/keys", headers=auth(created.json()["key"]))
    assert denied_list.status_code == 401


async def test_health_stays_open(client, auth_required):
    response = await client.get("/health")
    assert response.status_code == 200
