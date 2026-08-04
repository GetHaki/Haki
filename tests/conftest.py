"""Test bootstrap: real Postgres (haki_test), no database mocks.

The test database lives on the docker-compose Postgres. Its schema is
dropped/recreated and migrated once per session; tables are truncated
between tests.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

# App runtime connects as haki_app (non-superuser, RLS applies — migration
# 0006); alembic migrations run as the owner role haki (DDL rights). Both
# point at the "haki_test" database, derived from a BASE url that must name
# an already-existing database (the admin connection used to CREATE
# DATABASE haki_test connects there). Override the base via
# HAKI_TEST_BASE_DATABASE_URL / HAKI_TEST_BASE_MIGRATION_URL to run this
# same suite against a different Postgres target (e.g. a real Supabase
# project, one-off empirical verification) without touching the hardcoded
# local defaults used by CI/local runs.
_BASE_DATABASE_URL = os.environ.get(
    "HAKI_TEST_BASE_DATABASE_URL",
    "postgresql+asyncpg://haki_app:haki@localhost:5433/haki",
)
_BASE_MIGRATION_URL = os.environ.get(
    "HAKI_TEST_BASE_MIGRATION_URL",
    "postgresql+asyncpg://haki:haki@localhost:5433/haki",
)


def _with_test_db(url: str) -> str:
    return url.rsplit("/", 1)[0] + "/haki_test"


TEST_DATABASE_URL = _with_test_db(_BASE_DATABASE_URL)
TEST_MIGRATION_URL = _with_test_db(_BASE_MIGRATION_URL)
os.environ["HAKI_DATABASE_URL"] = TEST_DATABASE_URL  # before any app import
# Hermetic tests: no local model download, no remote LLM call.
os.environ["HAKI_EMBED_PROVIDER"] = "fake"
os.environ["HAKI_LLM_PROVIDER"] = "fake"
# Dev-open mode by default: the pre-sprint-6 tests exercise the API without
# auth. Auth tests opt in via the `auth_required` fixture below.
os.environ["HAKI_AUTH_REQUIRED"] = "false"
# Redis db 15 (of the default 16), never db 0 — keeps the CLI device-code
# auth flow's test data (tests/test_cli_auth.py) off whatever a developer
# has running locally on the default db.
os.environ.setdefault("HAKI_REDIS_URL", "redis://localhost:6379/15")

from app.config import settings  # noqa: E402
from app.db import async_session, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.redis_client import redis_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _sync_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


@pytest.fixture(scope="session", autouse=True)
def migrated_database():
    """Create haki_test if needed, wipe its schema, run migrations."""

    async def _prepare() -> None:
        # Admin connection to CREATE DATABASE haki_test: opened against the
        # BASE migration url's own (already-existing) database, whatever
        # it's named ("haki" locally, "postgres" on Supabase) — never a
        # hardcoded name, so this works against either target.
        conn = await asyncpg.connect(_sync_dsn(_BASE_MIGRATION_URL))
        try:
            await conn.execute("CREATE DATABASE haki_test")
        except asyncpg.DuplicateDatabaseError:
            pass
        finally:
            await conn.close()

        test_conn = await asyncpg.connect(_sync_dsn(TEST_MIGRATION_URL))
        try:
            await test_conn.execute("DROP SCHEMA public CASCADE")
            await test_conn.execute("CREATE SCHEMA public")
        finally:
            await test_conn.close()

    asyncio.run(_prepare())

    env = {
        **os.environ,
        "HAKI_DATABASE_URL": TEST_DATABASE_URL,
        "HAKI_MIGRATION_DATABASE_URL": TEST_MIGRATION_URL,
    }
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    yield


@pytest.fixture(autouse=True)
async def clean_tables():
    async with async_session() as session:
        await session.execute(
            text(
                "TRUNCATE events, facts, jobs, conflict_sets, context_traces, "
                "forget_receipts, feedback, api_keys, organizations CASCADE"
            )
        )
        await session.commit()
    await redis_client.flushdb()
    yield
    # Dispose pooled connections inside the test's event loop: otherwise the
    # pool hands connections to a closed loop at teardown (pytest-asyncio
    # uses one loop per test) and asyncpg crashes with "Event loop is closed".
    await engine.dispose()
    # Same reasoning for redis-py's async connections: they're bound to the
    # loop that opened them, so drop them now rather than let the next
    # test's (new) loop try to reuse a dead one.
    await redis_client.connection_pool.disconnect()


@pytest.fixture
def auth_required(monkeypatch):
    """Enable API key auth for one test (HAKI_AUTH_REQUIRED=true)."""
    monkeypatch.setattr(settings, "auth_required", True)
    return settings


@pytest.fixture
def make_api_key():
    """Create an api_keys row directly; returns the clear key (hk_...)."""

    async def _make(
        project_id: str = "prj_a",
        org_id: str = "org_a",
        label: str | None = None,
        revoked: bool = False,
    ) -> str:
        import hashlib
        import uuid
        from datetime import datetime, timezone

        from app.models import ApiKey

        clear = "hk_" + uuid.uuid4().hex
        async with async_session() as session:
            session.add(
                ApiKey(
                    key_hash=hashlib.sha256(clear.encode()).hexdigest(),
                    prefix=clear[:8],
                    org_id=org_id,
                    project_id=project_id,
                    label=label,
                    revoked_at=datetime.now(timezone.utc) if revoked else None,
                )
            )
            await session.commit()
        return clear

    return _make


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
