from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, overridable via HAKI_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="HAKI_", env_file=".env", extra="ignore"
    )

    database_url: str = "postgresql+asyncpg://haki_app:haki@localhost:5433/haki"
    # Owner role for migrations (DDL, CREATE EXTENSION). The runtime role
    # haki_app (created by migration 0006) is neither superuser nor table
    # owner, so FORCE ROW LEVEL SECURITY actually applies to it.
    migration_database_url: str = "postgresql+asyncpg://haki:haki@localhost:5433/haki"
    default_policy: str = "default"

    # Supabase in production (see docs/DEPLOY.md) is reached through the
    # Supavisor pooler in TRANSACTION mode (its direct-connection host is
    # IPv6-only, unreachable from plenty of networks/hosts). Confirmed
    # empirically: asyncpg's client-side prepared-statement cache breaks
    # under transaction-mode pooling as soon as more than one backend
    # session is involved (concurrent requests) — errors like
    # `InvalidSQLStatementNameError: prepared statement "..." does not
    # exist` or `DuplicatePreparedStatementError`, because the pooler can
    # hand successive transactions on the SAME client connection to
    # DIFFERENT physical Postgres backends, which never saw the PREPARE.
    # true = pass asyncpg's documented fix (statement_cache_size=0) to
    # every connection (see app/db.py). false (default) = unchanged local
    # behavior against docker-compose Postgres (direct connection, no
    # pooler, caching is safe and free performance).
    db_disable_prepared_statement_cache: bool = False  # HAKI_DB_DISABLE_PREPARED_STATEMENT_CACHE

    # Provider selection: extractor (LLM, off hot path) and embedder (in the
    # context hot path) are configured independently.
    llm_provider: str = "fake"  # HAKI_LLM_PROVIDER=fake|openai
    embed_provider: str = "local"  # HAKI_EMBED_PROVIDER=local|fake|openai
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_embed_model: str = "text-embedding-3-small"

    # Dev auth (sprint 4): when set, the MCP endpoint requires
    # `Authorization: Bearer <api_key>`. Unset = open mode (local dev only,
    # documented). Full OAuth lands in a later sprint.
    api_key: str | None = None

    # API key auth (sprint 6). When true (default), every /v1/* endpoint
    # except key management requires `Authorization: Bearer hk_...` and the
    # request is bound to the key's project (403 forbidden_scope otherwise,
    # RLS enforces it in SQL too). false = OPEN dev mode, never for
    # production: a warning is logged at startup.
    auth_required: bool = True
    # Optional admin key protecting /v1/keys management. When set, only
    # `Authorization: Bearer <admin_key>` may create/list/revoke keys.
    # When unset, the FIRST key creation is free (documented bootstrap);
    # afterwards a valid key manages the keys of its own project.
    admin_key: str | None = None

    # Sprint 11: secret shared with the console's Next.js BACKEND ONLY
    # (never sent to a browser) so it can call POST /v1/orgs/provision on
    # behalf of an already-verified Clerk user. Unset = the endpoint
    # refuses every request (self-serve signup off, self-hosted/curl
    # bootstrap via /v1/keys is unaffected).
    console_service_key: str | None = None  # HAKI_CONSOLE_SERVICE_KEY

    # Sprint 14: base URL of the console web app, used to build the
    # verification_uri returned by POST /v1/cli/device/start (the human
    # opens `<HAKI_CONSOLE_BASE_URL>/cli-auth` and types the user_code).
    console_base_url: str = "http://localhost:3000"  # HAKI_CONSOLE_BASE_URL

    # Sprint 14: Redis connection for the CLI device-code auth flow (see
    # app/redis_client.py, app/api/routes/cli_auth.py) — the first thing in
    # Haki to use the redis service docker-compose.yml has always shipped.
    redis_url: str = "redis://localhost:6379/0"  # HAKI_REDIS_URL

    # Sprint 12: GeniusPay (subscription billing, replaces the originally
    # planned Stripe integration). Base URL confirmed against the live
    # merchant API (see app/billing/geniuspay.py for the endpoint map).
    # Auth is X-API-Key/X-API-Secret headers, NOT Authorization: Bearer —
    # confirmed empirically, contradicts one GeniusPay doc page. Unset
    # key/secret = GeniusPayClient refuses to construct (fails loudly
    # rather than silently no-op'ing a paid feature).
    geniuspay_api_key: str | None = None  # HAKI_GENIUSPAY_API_KEY
    geniuspay_api_secret: str | None = None  # HAKI_GENIUSPAY_API_SECRET
    # HMAC secret for POST /v1/webhooks/geniuspay (header
    # X-GeniusPay-Signature). Unset = the webhook endpoint refuses every
    # request (no unverified webhook is ever processed) — generate this in
    # the GeniusPay dashboard once the endpoint is deployed and reachable.
    geniuspay_webhook_secret: str | None = None  # HAKI_GENIUSPAY_WEBHOOK_SECRET
    geniuspay_base_url: str = (
        "https://pay.genius.ci/api/v1/merchant"
    )  # HAKI_GENIUSPAY_BASE_URL

    # Single V1 plan (documented scope limit, same spirit as "one org per
    # human" — no plan picker yet). Price confirmed (sprint 13): 9900 XOF,
    # aligned with "Inside AI Starter" on the same GeniusPay merchant
    # account — overridable without a redeploy via
    # HAKI_BILLING_CLOUD_PLAN_PRICE_XOF.
    billing_cloud_plan_name: str = "Cloud"  # HAKI_BILLING_CLOUD_PLAN_NAME
    billing_cloud_plan_price_xof: int = 9900  # HAKI_BILLING_CLOUD_PLAN_PRICE_XOF

    # Credits (sprint 13 — usage-based billing, replaces the earlier
    # Free/Cloud/Scale tier idea). 1 credit = 1 event accepted by
    # POST /v1/capture; retrieval (context/inspect/facts/timeline/traces)
    # is never billed. See app/billing/credits.py for the ledger and
    # app/api/routes/capture.py for where credits are checked/debited.
    billing_free_monthly_credits: int = 1000  # HAKI_BILLING_FREE_MONTHLY_CREDITS
    billing_cloud_plan_monthly_credits: int = 20000  # HAKI_BILLING_CLOUD_PLAN_MONTHLY_CREDITS
    billing_credit_price_xof_per_credit: float = 0.65  # HAKI_BILLING_CREDIT_PRICE_XOF_PER_CREDIT

    # MCP server scope (sprint 4): memory is project- AND subject-scoped by
    # config, never chosen by the model (security invariant, README —
    # "le modele ne choisit jamais les scopes"). One MCP server instance =
    # one project = one subject; a team sharing one Cursor deployment needs
    # one server config per person (own HAKI_MCP_SUBJECT_ID), the same way
    # each install already gets its own project.
    mcp_project_id: str = "prj_cursor_dev"  # HAKI_MCP_PROJECT_ID
    mcp_org_id: str = "org_cursor_dev"  # HAKI_MCP_ORG_ID
    mcp_subject_id: str = "usr_cursor_dev"  # HAKI_MCP_SUBJECT_ID
    # Synchronous consolidation after each haki_capture so the memory is
    # recallable immediately (dev default; a worker can take over in prod).
    mcp_autoconsolidate: bool = True  # HAKI_MCP_AUTOCONSOLIDATE


settings = Settings()
