"""Live MCP demo (sprint 4): real server, real LLM extraction.

Prerequisite: `uv run uvicorn app.main:app --port 8100` with
HAKI_LLM_PROVIDER=openai (real extraction) and the local embedder.

Scenario via the official MCP client (Streamable HTTP):
  1. haki_capture  — a project convention ("TypeScript strict, tests avant
     toute modification"), consolidated synchronously by the real LLM;
  2. haki_context  — "quelles conventions avant de modifier du code ?" must
     recall the convention, with source;
  3. haki_inspect  — the trace proves WHY it was served;
  4. haki_forget   — delete; the next haki_context serves nothing.
"""

import asyncio
import json

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://localhost:8100/mcp"
SUBJECT = "usr_demo_cursor"


def show(title: str, data: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))


async def main() -> None:
    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("outils MCP visibles:", sorted(t.name for t in tools.tools))

            # 1. Capture a durable convention (real LLM extraction).
            result = await session.call_tool(
                "haki_capture",
                {
                    "content": (
                        "Ce projet utilise TypeScript strict, tests avant "
                        "toute modification."
                    ),
                    "subject_id": SUBJECT,
                },
            )
            capture = result.structured_content
            show("1. haki_capture", capture)

            # 2. Context recalls the convention.
            result = await session.call_tool(
                "haki_context",
                {
                    "query": "quelles conventions avant de modifier du code ?",
                    "subject_id": SUBJECT,
                },
            )
            context = result.structured_content
            show("2. haki_context — bloc pret a injecter", context)
            assert "TypeScript" in context["context"], (
                "la convention n'est pas rappelee !"
            )
            assert "source:" in context["context"]
            print("\n>>> convention rappelee avec source: OK")

            # 3. Inspect the trace.
            result = await session.call_tool(
                "haki_inspect",
                {"trace_id": context["trace_id"], "subject_id": SUBJECT},
            )
            inspect = result.structured_content
            show("3. haki_inspect — decisions", {"decisions": inspect["decisions"]})

            # 4. Forget (real delete), then context serves nothing.
            result = await session.call_tool(
                "haki_forget", {"subject_id": SUBJECT, "mode": "delete"}
            )
            forget = result.structured_content
            show("4. haki_forget (delete)", forget)

            result = await session.call_tool(
                "haki_context",
                {
                    "query": "quelles conventions avant de modifier du code ?",
                    "subject_id": SUBJECT,
                },
            )
            after = result.structured_content
            show("5. haki_context apres oubli", after)
            assert after["facts"] == [], "la memoire n'a pas ete effacee !"
            print("\n>>> apres forget, le context ne sert plus rien: OK")


if __name__ == "__main__":
    asyncio.run(main())
