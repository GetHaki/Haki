#!/bin/bash
# `wait -n` below needs bash (POSIX sh/dash has no such flag) — Debian
# bookworm-slim (this image's base) ships bash by default, so this is safe.
#
# Runs the API (uvicorn) and the consolidation worker (app/worker.py) as two
# processes in the same container — the simplest fix for the sprint-16 bug
# where nothing ever invoked the worker after the single-pass CMD switched
# to just `uvicorn`. Splitting into two separate deployed services is the
# architecturally cleaner long-term answer (see app/worker.py docstring),
# but this keeps the fix to one container, one Coolify resource, no new
# infra, on a box that's already tight on resources.
#
# Both children are started in the background so this script's own PID can
# `wait` on them and forward SIGTERM (docker stop / Coolify redeploy) to
# both — without the trap, only PID 1's own default signal handling would
# apply, and the worker would be killed abruptly rather than exiting its
# poll loop cleanly.
set -e

alembic upgrade head

python -m app.worker &
WORKER_PID=$!

# --proxy-headers/--forwarded-allow-ips: Coolify's front proxy (Traefik)
# terminates TLS and forwards plain HTTP with X-Forwarded-Proto: https.
# Without these flags uvicorn ignores that header and believes every
# request is HTTP, so any redirect it emits (e.g. the trailing-slash
# redirect on /mcp) downgrades to an http:// Location — which browsers
# and MCP clients then refuse to follow from an https:// page.
uvicorn app.main:app --host 0.0.0.0 --port 8100 --proxy-headers --forwarded-allow-ips='*' &
API_PID=$!

trap 'kill -TERM $WORKER_PID $API_PID 2>/dev/null' TERM INT

# If either process exits (crash or normal shutdown), stop the other and
# exit with its status — a silently-dead worker with the API still "up"
# would just be a quieter version of the original bug.
wait -n $WORKER_PID $API_PID
EXIT_CODE=$?
kill -TERM $WORKER_PID $API_PID 2>/dev/null
wait $WORKER_PID $API_PID 2>/dev/null
exit $EXIT_CODE
