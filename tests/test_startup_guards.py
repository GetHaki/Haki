"""Configuration guards: the failures that used to be silent.

Every one of these protects against the same shape of bug -- a setting
that is wrong in a way nothing complains about, so the system runs,
returns 200s, and quietly does much less than it says. This project has
already paid for that shape three times (a dead lexical axis, a text
search configuration mismatch, an extractor that extracts nothing), which
is why each guard has a test proving it FIRES, not merely that it exists.
"""

import pytest

from app.config import settings
from app.main import lifespan


async def test_a_fake_extractor_with_auth_on_refuses_to_start(monkeypatch):
    """`fake` is the DEFAULT extractor, and it extracts nothing.

    FakeProvider only reads `payload["mock_facts"]`; a real event yields
    []. A self-hosted install that forgets HAKI_LLM_PROVIDER would capture
    events, raise nothing, and build an empty memory -- with no symptom
    until someone notices the packets are always empty.
    """
    monkeypatch.setattr(settings, "llm_provider", "fake")
    monkeypatch.setattr(settings, "auth_required", True)
    with pytest.raises(RuntimeError, match="extracts nothing"):
        async with lifespan(None):
            pass


async def test_open_dev_mode_with_a_fake_extractor_is_allowed(monkeypatch):
    """The pair that IS coherent: a development box.

    HAKI_AUTH_REQUIRED=false is documented as open dev mode, never for
    production, and already warns on its own. Refusing this combination
    would break every local run and the whole test suite.
    """
    monkeypatch.setattr(settings, "llm_provider", "fake")
    monkeypatch.setattr(settings, "auth_required", False)
    async with lifespan(None):
        pass


async def test_the_openai_embedder_is_refused_rather_than_failing_at_insert():
    """1536 dimensions against a vector(1024) column.

    It could never have worked: the mismatch was documented in a provider
    docstring, which is not where someone setting an environment variable
    looks. The failure surfaced as a database error in the middle of a
    consolidation job.
    """
    from app import providers

    original = providers._embedder
    providers._embedder = None
    try:
        with pytest.raises(RuntimeError, match="not supported"), pytest.MonkeyPatch.context() as patch:
            patch.setattr(settings, "embed_provider", "openai")
            providers.get_embedder()
    finally:
        providers._embedder = original


async def test_cross_project_consolidate_requires_the_admin_key(
    client, auth_required, make_api_key, monkeypatch
):
    """POST /v1/consolidate drains EVERY project's queue.

    It runs on an ops session with no RLS context, so the middleware's
    project binding does not apply: before 22 aout any valid customer key
    could drain -- and pay the LLM cost of -- every other tenant's queue,
    invisibly. `/consolidate/subject` is the scoped endpoint a customer key
    is meant to use.
    """
    monkeypatch.setattr(settings, "admin_key", "admin-secret")
    key = await make_api_key(project_id="prj_a")

    refused = await client.post(
        "/v1/consolidate", headers={"Authorization": f"Bearer {key}"}
    )
    assert refused.status_code == 401

    allowed = await client.post(
        "/v1/consolidate", headers={"Authorization": "Bearer admin-secret"}
    )
    assert allowed.status_code == 200
