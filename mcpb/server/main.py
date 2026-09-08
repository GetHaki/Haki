"""MCPB entrypoint — minimal stdio shim.

The bundle ships the whole repository (uv resolves app/ + sdk/python from
pyproject.toml), so this file only has to import the real server and run
it over stdio. Kept separate from app/mcp_stdio.py so the bundle's
manifest stays self-describing without depending on the repo layout.
"""

import anyio

from app.mcp_server import mcp


def main() -> None:
    anyio.run(mcp.run_stdio_async)


if __name__ == "__main__":
    main()
