import re
from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

# See app/config.py (db_disable_prepared_statement_cache) for why this is
# sometimes needed: Supabase in production is reached through the Supavisor
# pooler in transaction mode, and asyncpg's client-side prepared-statement
# cache breaks under that pooling mode (confirmed empirically) unless
# disabled per connection.
_connect_args = (
    {"statement_cache_size": 0}
    if settings.db_disable_prepared_statement_cache
    else {}
)
engine = create_async_engine(
    settings.database_url, pool_pre_ping=True, connect_args=_connect_args
)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Request-scoped session, bound to the API key's project via RLS.

    When the auth middleware resolved a key, `haki.project_id` is set LOCAL
    to this transaction (set_config ..., true): every statement then only
    sees the key's project rows, even if the code forgets a project filter
    (migration 0006 policies). In dev-open mode no key is resolved, nothing
    is set, and the policies are permissive (documented).
    """
    async with async_session() as session:
        state = request.scope.get("state") or {}
        api_key = state.get("haki_api_key")
        if api_key is not None:
            await session.execute(
                text("SELECT set_config('haki.project_id', :pid, true)"),
                {"pid": api_key.project_id},
            )
        yield session


async def get_session_ops() -> AsyncGenerator[AsyncSession, None]:
    """Session WITHOUT RLS context for the dev/ops consolidate endpoint:
    it processes pending jobs across projects by design (documented)."""
    async with async_session() as session:
        yield session


def install_tcp_nodelay() -> None:
    """Set TCP_NODELAY on every TCP connection created by this event loop.

    asyncpg does not disable the Nagle algorithm; every production Postgres
    driver does, because Nagle + delayed ACK stalls request/response
    protocols with multi-segment writes on real networks. Installed as a
    shim (the loop's `create_connection` is wrapped) since asyncpg exposes
    no socket option. Idempotent per loop.

    Note (sprint-3 latency investigation, honest record): on Windows +
    Docker Desktop, a ~40 ms penalty on large writes (e.g. the
    context_traces INSERT with its decisions JSONB: 2.9 ms vs 47.7 ms for a
    27 KB payload) was measured WITH and WITHOUT TCP_NODELAY — the same
    INSERT runs in 9 ms inside the container, so the cost is the Docker
    Desktop userspace proxy on the host<->VM path, a dev-machine artifact,
    not something this shim (or any client code) can remove.
    """
    import asyncio
    import socket

    loop = asyncio.get_running_loop()
    if getattr(loop, "_haki_nodelay_installed", False):
        return
    original = loop.create_connection

    async def create_connection(protocol_factory, *args, **kwargs):
        transport, protocol = await original(protocol_factory, *args, **kwargs)
        try:
            sock = transport.get_extra_info("socket")
            if sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass  # never break a connection over a best-effort optimization
        return transport, protocol

    loop.create_connection = create_connection
    loop._haki_nodelay_installed = True


# Text search configuration drift guard (20 Aug).
#
# `facts.search_vector` and `events.search_vector` are GENERATED columns:
# the text search configuration used to build the tsvector is frozen into
# the schema by the migration that created them, while the query side
# reads `settings.fts_config` at runtime (app.context.fts). If the two
# disagree, PostgreSQL does not complain -- 'english' stems `caroline` to
# `carolin`, 'simple' does not, so the tsquery simply matches nothing. The
# lexical axis goes dark and the only symptom is a quietly worse ranking.
#
# This project has already paid the full price of that exact failure mode
# once (the axis contributed nothing on the large majority of eval
# questions for weeks, with every test green). Hence a loud startup check
# rather than a comment.
_GENERATED_FTS_CONFIG_SQL = """
    SELECT c.relname AS table_name,
           pg_get_expr(d.adbin, d.adrelid) AS expression
    FROM pg_attrdef d
    JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
    JOIN pg_class c ON c.oid = d.adrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = current_schema()
      AND c.relname IN ('facts', 'events')
      AND a.attname = 'search_vector'
"""
_TSVECTOR_CONFIG_RE = re.compile(r"to_tsvector\(\s*'([a-z_]+)'::regconfig")


async def verify_fts_config() -> None:
    """Fail fast when the indexed and queried text search configs differ.

    Silent on a database whose schema is not migrated yet (fresh install,
    CI bootstrap): there is nothing to disagree with, and refusing to boot
    before `alembic upgrade head` would be a worse default.
    """
    expected = settings.fts_config
    async with engine.connect() as conn:
        rows = (await conn.execute(text(_GENERATED_FTS_CONFIG_SQL))).all()
    if not rows:
        return
    mismatched = {
        row.table_name: found
        for row in rows
        if (match := _TSVECTOR_CONFIG_RE.search(row.expression or ""))
        and (found := match.group(1)) != expected
    }
    if mismatched:
        detail = ", ".join(f"{table}={found}" for table, found in sorted(mismatched.items()))
        raise RuntimeError(
            f"text search configuration mismatch: HAKI_FTS_CONFIG={expected!r} but the "
            f"GENERATED search_vector columns were built with {detail}. The lexical "
            "retrieval axis would silently match nothing. Run `alembic upgrade head`, "
            "or set HAKI_FTS_CONFIG to the configuration the schema was built with."
        )
