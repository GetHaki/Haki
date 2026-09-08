"""Security audit C3 (8 sept): the RLS GUC must survive mid-request commits.

`set_config('haki.project_id', ..., true)` is transaction-local. This
codebase commits mid-request by design (the gateway commits the decision
trace before capturing the turn), so every transaction AFTER the first
commit on the same session used to start with NO GUC — the migration-0006
policies' "no context = permissive" branch then applied and post-commit
writes ran unscoped. get_session now binds an `after_begin` hook that
re-applies the GUC at the start of every transaction on the session.

The first test here is the exact regression: GUC present, commit, GUC
still present. It runs against the real Postgres (asyncpg), like the rest
of the suite.
"""
from unittest.mock import patch

import pytest
from sqlalchemy import text

from app.db import get_session


class _FakeKey:
    """Shape-compatible stand-in for the middleware's ApiKey row."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.org_id = "org_c3_test"
        self.prefix = "hk_c3"


async def _guc(session) -> str | None:
    return (
        await session.execute(text("SELECT current_setting('haki.project_id', true)"))
    ).scalar()


async def test_rls_guc_survives_mid_request_commits(migrated_database):
    """TX1: GUC set. commit(). TX2: GUC MUST still be the key's project.

    Before the C3 fix, TX2 saw '' (no context) — the permissive branch of
    every migration-0006 policy — so post-commit statements ran unscoped
    (the MCP autoconsolidation / gateway capture-turn hole)."""
    scope = {"state": {"haki_api_key": _FakeKey("prj_c3_survives")}}
    gen = get_session(type("R", (), {"scope": scope})())
    session = await gen.__anext__()

    assert await _guc(session) == "prj_c3_survives"
    await session.commit()  # the mid-request commit pattern
    assert await _guc(session) == "prj_c3_survives"
    await session.commit()
    assert await _guc(session) == "prj_c3_survives"

    await gen.aclose()


async def test_rls_guc_not_set_without_a_key(migrated_database):
    """No key (dev-open/anon surface) -> no GUC, permissive by design."""
    gen = get_session(type("R", (), {"scope": {"state": {}}})())
    session = await gen.__anext__()
    assert await _guc(session) in ("", None)
    await gen.aclose()
