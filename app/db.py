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
