"""Change the text search configuration of the lexical retrieval axis.

    uv run python scripts/set_fts_config.py french

The configuration is frozen into the GENERATED `search_vector` columns of
`facts` and `events` (migration 0023), so changing it is DDL, not a config
reload. This script runs `app.context.fts.rebuild_statements` -- the exact
same DDL a migration would run -- against the migration-owner connection,
then tells you to set HAKI_FTS_CONFIG to the same value, which
`app.db.verify_fts_config` enforces at startup.

Both tables are rewritten and both GIN indexes rebuilt, under an ACCESS
EXCLUSIVE lock. See alembic/versions/0023_search_vector_english.py for the
out-of-band procedure on a large table.

Connects with the OWNER role (HAKI_MIGRATION_DATABASE_URL), like Alembic:
the runtime role has no DDL rights in production.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.context.fts import rebuild_statements

SUPPORTED = ("simple", "english", "french")


async def _apply(config: str, statements: list[str]) -> int:
    engine = create_async_engine(settings.migration_database_url)
    try:
        async with engine.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_ts_config WHERE cfgname = :name"),
                {"name": config},
            )
        if not exists:
            print(
                f"no such text search configuration on this server: {config!r}",
                file=sys.stderr,
            )
            return 2
        async with engine.begin() as conn:
            for statement in statements:
                await conn.execute(text(statement))
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help=f"text search configuration ({'|'.join(SUPPORTED)})")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    config = args.config
    if config not in SUPPORTED:
        print(
            f"warning: {config!r} is not one of {SUPPORTED}; it must exist in "
            "pg_ts_config on the target database.",
            file=sys.stderr,
        )
    try:
        statements = rebuild_statements(config)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not args.yes:
        answer = input(
            f"Rebuild facts.search_vector and events.search_vector with "
            f"'{config}'? Both tables are rewritten. [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            return 1

    code = asyncio.run(_apply(config, statements))
    if code == 0:
        print(
            f"done. Set HAKI_FTS_CONFIG={config} on every process that talks to this "
            "database — the API refuses to start while the two disagree."
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
