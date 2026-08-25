import logging
import re
from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

logger = logging.getLogger("haki.db")

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


# The vector axis has the SAME failure mode as the lexical one above, and
# nothing checked it until now. Two independent ways to go dark, both
# silent:
#
# 1. The dimension. `facts.embedding` is a vector(N) fixed by a migration;
#    the embedder is chosen at runtime by HAKI_EMBED_MODEL. get_embedder
#    compares the model's dimension against the EMBEDDING_DIM CONSTANT --
#    which is the dimension the code BELIEVES the column has, not the one
#    it actually has. An install whose `alembic upgrade head` did not run
#    passes that check and fails at the first INSERT, mid-consolidation.
#
# 2. The model. This is the hole this function exists for. Two DIFFERENT
#    models of the SAME dimension are interchangeable to every check the
#    project had: snowflake-arctic-embed-s and the default are both 384,
#    so swapping HAKI_EMBED_MODEL between them starts cleanly, inserts
#    cleanly, and compares query vectors from one embedding space against
#    stored vectors from another. Cosine similarity between two unrelated
#    spaces is noise: the dense axis contributes nothing and ranking
#    silently degrades to lexical-only. No error, no test failure -- the
#    exact shape of the FTS bug that cost this project weeks.
#
# `embedding_space` records which model produced the stored vectors
# (migration 0028). One row, read once at startup, no per-row stamp and no
# change to any of the eight write sites: the invariant is a property of
# the corpus, not of a row.
_EMBEDDING_COLUMN_DIM_SQL = """
    SELECT c.relname AS table_name,
           a.atttypmod AS dim
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = current_schema()
      AND c.relname IN ('facts', 'events', 'episode_chunks')
      AND a.attname = 'embedding'
      AND NOT a.attisdropped
"""


async def verify_embedding_space() -> None:
    """Fail fast when the configured embedder does not match stored vectors.

    Silent on a database whose schema is not migrated yet (fresh install,
    CI bootstrap), for the same reason as verify_fts_config: there is
    nothing to disagree with yet.

    A pending backfill is a WARNING, not a refusal. Refusing would mean the
    service is down for the whole duration of the re-embedding, and the
    degradation is partial and recoverable -- rows already re-embedded are
    correct. A model MISMATCH is a refusal, because every comparison it
    produces is meaningless.
    """
    if settings.embed_provider != "local":
        return
    from app.providers.local import MODELS

    # An unknown model name is NOT a reason to skip this check. get_embedder
    # does raise on it, with the list of known models -- but only once
    # something asks for an embedder, and the model comparison below needs
    # nothing but the NAME. Returning early here would mean a typo in
    # HAKI_EMBED_MODEL passes startup silently.
    spec = MODELS.get(settings.embed_model)

    async with engine.connect() as conn:
        dims = {
            row.table_name: int(row.dim)
            for row in (await conn.execute(text(_EMBEDDING_COLUMN_DIM_SQL))).all()
        }
        if not dims:
            return
        try:
            state = (
                await conn.execute(
                    text(
                        "SELECT model, backfilled_at FROM embedding_space WHERE id = 1"
                    )
                )
            ).first()
        except ProgrammingError:
            # Migration 0028 not applied: the dimension check below still
            # runs, which is the half that was already possible to get wrong.
            state = None

    mismatched = (
        {table: dim for table, dim in sorted(dims.items()) if dim != spec.dim}
        if spec is not None
        else {}
    )
    # The DIMENSION first when both are wrong: the fix is ordered (widen the
    # columns, then re-embed), and this is the first half of it.
    if mismatched:
        detail = ", ".join(
            f"{table}=vector({dim})" for table, dim in mismatched.items()
        )
        raise RuntimeError(
            f"embedding dimension mismatch: HAKI_EMBED_MODEL={settings.embed_model!r} "
            f"produces {spec.dim}-dimensional vectors and the schema has {detail}. "
            "Run `alembic upgrade head`, then re-embed with "
            "`uv run python -m scripts.backfill_embeddings` -- vectors from two "
            "models are not comparable, so widening the column without a backfill "
            "leaves the dense axis matching noise."
        )

    if state is None:
        return
    if state.model and state.model != settings.embed_model:
        raise RuntimeError(
            f"embedding model mismatch: the stored vectors were produced by "
            f"{state.model!r} and HAKI_EMBED_MODEL is {settings.embed_model!r}. "
            "Both produce vectors of the same size, so nothing else would have "
            "complained: query vectors from one embedding space would be compared "
            "against stored vectors from another, and cosine similarity between "
            "two unrelated spaces is noise -- the dense axis would go dark and "
            "ranking would silently fall back to lexical only. Either restore "
            f"HAKI_EMBED_MODEL={state.model!r}, or re-embed the corpus with "
            "`uv run python -m scripts.backfill_embeddings`."
        )
    if state.backfilled_at is None:
        logger.warning(
            "embedding backfill INCOMPLETE for model %s: rows whose embedding is "
            "still NULL are invisible to the dense axis and are retrieved by the "
            "lexical axis alone. Finish with "
            "`uv run python -m scripts.backfill_embeddings`.",
            settings.embed_model,
        )


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
