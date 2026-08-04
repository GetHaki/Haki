"""Shared Redis connection (sprint 14).

Redis backs ONLY the CLI device-code auth flow
(app/api/routes/cli_auth.py) — short-lived pending/approved state with
native TTL, deliberately not a Postgres table (nothing there needs to
survive past its own expiry, and there is nothing to migrate/query
relationally). docker-compose.yml has shipped a redis service since the
start of the project; this is the first code to actually use it.

Uses redis.asyncio (redis-py's built-in async client) — NOT the separate
aioredis package, which is deprecated and merged into redis-py itself.
"""

from redis.asyncio import Redis

from app.config import settings

# One shared client for the process lifetime, same spirit as the
# SQLAlchemy engine in app.db: redis-py pools connections internally, no
# per-request connect/disconnect needed.
redis_client = Redis.from_url(settings.redis_url, decode_responses=True)


async def get_redis() -> Redis:
    """FastAPI dependency yielding the shared client."""
    return redis_client
