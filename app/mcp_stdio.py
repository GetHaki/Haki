"""Stdio entrypoint for the Haki MCP server — used by MCP directories
(Glama, Smithery, ...) that build the repo's Dockerfile and probe the
server over stdio.

This is the same tool set as the HTTP server mounted on /mcp by
app.main (app/mcp_server.py) — the only difference is the transport and
the lifecycle:

- stdio (this file): one process per client session. No Postgres
  connection is opened until a tool actually needs one, and nothing
  listens on a port — the directory runner's expectations exactly.
- HTTP (app/main.py): one shared server behind the API, session manager
  task group started explicitly, auth handled by HAKI_API_KEY.

Self-hosted users configure Cursor/Claude Desktop with:
    { "command": "uvx", "args": ["--from", "git+https://github.com/GetHaki/Haki", "haki-mcp"] }
or, from a clone:  python -m app.mcp_stdio

Env (same as the HTTP surface): HAKI_DATABASE_URL points at a Postgres
with the Haki schema (alembic-upgraded). Scope comes from
HAKI_MCP_PROJECT_ID/ORG_ID/SUBJECT_ID, or from the hk_ key + headers on
the HTTP surface only — over stdio there are no headers, so the legacy
env-var scoping always applies.
"""

import anyio

from app.mcp_server import mcp


def main() -> None:
    anyio.run(mcp.run_stdio_async)


if __name__ == "__main__":
    main()
