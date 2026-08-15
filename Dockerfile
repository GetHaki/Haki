# Haki API — multi-stage build on uv (matches pyproject.toml / uv.lock).
#
# `haki` (sdk/python) is an editable path dependency (pyproject.toml
# [tool.uv.sources]), so the SDK source has to be in the build context
# before `uv sync` — not just `app/`.
#
# Migrations run at container start (docker-entrypoint.sh): fine for a
# single-instance pilot deploy (the scope this was built for — see README
# deploy section). Running more than one instance would race on
# `alembic upgrade head`; split migration-run from app-start into separate
# steps/jobs before ever scaling past one instance.
#
# docker-entrypoint.sh also starts app/worker.py's consolidation loop
# alongside uvicorn (sprint 16 fix — CMD used to run uvicorn only, and
# nothing else ever invoked the worker, so captured events queued jobs no
# process would process).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first (cache layer, invalidated only when these change).
COPY pyproject.toml uv.lock ./
COPY sdk/python ./sdk/python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# App code.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app

# libgomp1: onnxruntime (fastembed's local embedder, HAKI_EMBED_PROVIDER=
# local, the default) needs it at import time — missing it is a runtime
# crash, not a build error, so it's easy to miss without this line.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r haki && useradd -r -g haki -d /app haki
COPY --from=builder --chown=haki:haki /app /app
COPY --chown=haki:haki docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
ENV PATH="/app/.venv/bin:$PATH" HOME=/app

USER haki
EXPOSE 8100
CMD ["/app/docker-entrypoint.sh"]
