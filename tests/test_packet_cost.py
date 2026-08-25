"""The budget must be the number the caller's prompt actually costs.

Until 22 aout `budget_tokens` was charged against a stripped string --
`predicate value` for a fact, `date kind excerpt` for an episode -- while
the prompt carried a fully rendered line. Measured on eval.retrieval_bench
(LoCoMo 1-2, n=231, o200k tokenizer): `budget_tokens=2000` put a median of
4 565 tokens into the caller's prompt. 2.28x, on every call, silently.

Two things had to be true to fix it, and both are pinned here:

- ONE renderer. The same block was built independently by the Python SDK,
  the TypeScript SDK and (until P14) a third copy inside the MCP server,
  and costed from a fourth string matching none of them. The server renders
  the line now and the SDKs print it; the SDKs keep a fallback for older
  servers, and it must stay byte-identical or it silently becomes renderer
  number two again.
- The line must be CHEAP, because it is what the budget buys. A uuid4 is
  35 o200k tokens; `F3` is 2.
"""

import re
import uuid
from pathlib import Path

from app.context import cost
from app.context.cost import estimate_prose_tokens, render_line, short_timestamp
from app.db import async_session
from app.models import Fact
from app.providers.fake import mock_fact
from sdk.python.src.haki.runtime import build_prompt_context, resolve_refs
from tests.test_consolidator import run_worker

BASE_FACT = {
    "id": "f-1",
    "ref": "F1",
    "predicate": "invoice_language",
    "value": {"language": "fr"},
    "valid_from": "2026-07-28T10:00:00+00:00",
    "valid_from_short": "2026-07-28 10:00",
    "valid_from_relative": "12 days before the question",
    "source_event_ids": ["evt-1"],
}
BASE_EPISODE = {
    "event_id": "evt-1",
    "episode_id": "chk-1",
    "ref": "E1",
    "kind": "chat_session",
    "occurred_at": "2026-07-28T10:00:00+00:00",
    "occurred_at_short": "2026-07-28 10:00",
    "occurred_at_relative": "12 days before the question",
    "excerpt": "user: I want my invoices in French.",
    "context_neighbor": False,
}

MARKERS = [
    {},
    {"freshness": "unconfirmed", "last_confirmed": "2026-01-01"},
    {"freshness": "stale", "last_confirmed": "2026-01-01"},
    {"attributed_to": "her sister"},
    {"contested": True, "conflict_id": "cfl-9"},
    {"auto_reclassified": True},
    {"temporal_range": {"start": "2026-06-01", "end": "2026-06-30"}},
    {"freshness": "stale", "last_confirmed": "2026-01-01", "contested": True,
     "conflict_id": "cfl-9", "attributed_to": "her sister", "auto_reclassified": True},
]


def _sdk_line(packet_key: str, item: dict) -> str:
    """What the SDK's FALLBACK renders -- `line` deliberately withheld."""
    stripped = {k: v for k, v in item.items() if k != "line"}
    packet = {"facts": [], "episodes": [], "warnings": []}
    packet[packet_key] = [stripped]
    return next(
        line for line in build_prompt_context(packet).split("\n") if line.startswith("- ")
    )


def test_the_server_line_and_the_sdk_fallback_are_the_same_string():
    """The contract that keeps one renderer from quietly becoming two.

    String equality, not "close enough": a marker added on one side and not
    the other is then a failing test, not a budget that drifts and a prompt
    that no longer matches what was charged for it.
    """
    for extra in MARKERS:
        fact = {**BASE_FACT, **extra}
        assert render_line("fact", fact) == _sdk_line("facts", fact), extra
    for neighbor in (False, True):
        episode = {**BASE_EPISODE, "context_neighbor": neighbor}
        assert render_line("episode", episode) == _sdk_line("episodes", episode)


def test_the_rendered_line_carries_no_uuid_and_no_dead_precision():
    fact_line = render_line("fact", BASE_FACT)
    episode_line = render_line("episode", BASE_EPISODE)
    assert fact_line.startswith("- [F1] ")
    assert episode_line.startswith("- [E1] ")
    for line in (fact_line, episode_line):
        assert "evt-1" not in line
        assert "chk-1" not in line
        assert "10:00:00" not in line and "+00:00" not in line
    # The dual-date rendering that WAS measured to help stays untouched.
    assert "12 days before the question" in fact_line


def test_short_timestamp_keeps_the_day_and_the_time_of_day():
    assert short_timestamp("2026-07-28T10:32:07+00:00") == "2026-07-28 10:32"
    assert short_timestamp("2026-07-28T10:32:07Z") == "2026-07-28 10:32"
    assert short_timestamp("2026-07-28") == "2026-07-28"
    assert short_timestamp(None) is None


async def _seed(client) -> None:
    """A subject with more facts and turns than a small budget can hold."""
    await client.post(
        "/v1/capture",
        json={
            "idempotency_key": f"batch-{uuid.uuid4()}",
            "events": [
                {
                    "org_id": "org_acme",
                    "project_id": "prj_support",
                    "subject_type": "user",
                    "subject_id": "usr_cost",
                    "kind": "chat_session",
                    "occurred_at": "2026-05-01T10:00:00Z",
                    "payload": {
                        "messages": [
                            {"role": "user", "content": f"I bought a Zolgorvex kayak, model {i}."}
                            for i in range(12)
                        ],
                        "mock_facts": [
                            mock_fact(
                                f"owns_kayak_{i}",
                                {"model": i},
                                subject_id="usr_cost",
                                evidence_span=f"I bought a Zolgorvex kayak, model {i}.",
                            )
                            for i in range(12)
                        ],
                    },
                }
            ],
        },
    )
    await run_worker()


async def _packet(client, budget: int = 600) -> dict:
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_cost",
            "query": "Zolgorvex kayak",
            "budget_tokens": budget,
        },
    )
    assert response.status_code == 200
    return response.json()


async def test_the_budget_is_what_the_prompt_costs(client):
    """End to end: what was charged is what the whole block costs."""
    from app.context import estimate_tokens

    await _seed(client)
    body = await _packet(client)
    packet = body["packet"]
    items = packet["facts"] + packet["episodes"]
    assert items, "nothing packed -- the test would pass vacuously"

    charged = body["token_count"]
    assert charged <= 600
    assert charged == sum(estimate_tokens(item["line"]) for item in items)

    # And what the block costs BESIDES the items is reported, not hidden.
    expected_overhead = estimate_prose_tokens(cost.HEADER) + estimate_prose_tokens(
        cost.WRAPPER
    )
    if packet["episodes"]:
        expected_overhead += estimate_prose_tokens(cost.EPISODES_HEADER)
    if any(fact.get("contested") for fact in packet["facts"]):
        expected_overhead += estimate_prose_tokens(cost.CONTESTED_INSTRUCTIONS)
    assert packet["overhead_tokens"] == expected_overhead

    block_lines = [
        line for line in build_prompt_context(packet).split("\n") if line.startswith("- ")
    ]
    assert block_lines == [item["line"] for item in items]


async def test_an_empty_packet_costs_nothing(client):
    """The reservation is released when nothing ends up packed.

    The status line and the warnings a degraded packet still renders stay
    OUTSIDE the budget on purpose: they are the noisy-failure contract, not
    memory, and a budget must never be the reason a caller stops being told
    that retrieval was degraded.
    """
    body = await _packet(client, budget=600)
    assert body["packet"]["facts"] == [] and body["packet"]["episodes"] == []
    assert body["token_count"] == 0
    assert body["packet"]["overhead_tokens"] == 0
    block = build_prompt_context(body["packet"])
    assert not [line for line in block.split("\n") if line.startswith("- ")]


async def test_the_sdk_instructions_are_the_ones_the_budget_charged():
    """Third renderer, third drift surface -- pinned in both languages.

    The header text is canonical in app.context.cost because the budget has
    to charge it. If a copy of it drifts, the caller is charged for one
    block and served another.
    """
    block = build_prompt_context(
        {"facts": [{**BASE_FACT}], "episodes": [{**BASE_EPISODE}], "warnings": []}
    )
    assert cost.HEADER in block
    assert cost.EPISODES_HEADER in block

    typescript = Path("sdk/typescript/src/runtime.ts").read_text(encoding="utf-8")
    joined = re.sub(r'"\s*\+\s*\n\s*"', "", typescript)
    for sentence in (
        "You MUST apply them",
        "Cite an item by the reference in square brackets",
        "raw excerpts kept for citation and narrative detail",
    ):
        assert sentence in joined, sentence


async def test_every_item_carries_a_unique_reference_a_caller_can_resolve(client):
    await _seed(client)
    packet = (await _packet(client))["packet"]
    refs = [item["ref"] for item in packet["facts"] + packet["episodes"]]
    assert refs and all(refs)
    assert len(set(refs)) == len(refs)

    answer = f"I relied on {refs[0]} and on E999, which does not exist."
    resolved = resolve_refs(answer, packet)
    assert set(resolved) == {refs[0]}
    resolved_item = resolved[refs[0]]
    assert resolved_item.get("id") or resolved_item.get("event_id")


async def test_a_packet_from_an_older_server_still_renders(client):
    """The fallback is not decoration: pinned by the equality test above."""
    legacy = {
        "facts": [{k: v for k, v in BASE_FACT.items() if k not in ("ref",)}],
        "episodes": [],
        "warnings": [],
    }
    block = build_prompt_context(legacy)
    assert "invoice_language" in block
    assert "<haki_memory>" in block


async def test_facts_are_still_linked_to_their_source_turn_in_the_packet(client):
    """The uuid left the PROMPT, not the packet.

    "Reliable context, with proof" is a claim about what the caller can
    resolve, not about what the model has to copy back. Dropping the ids
    from the packet would be removing the proof; dropping them from the
    prompt is removing 23 % of its cost.
    """
    await _seed(client)
    packet = (await _packet(client))["packet"]
    for fact in packet["facts"]:
        assert fact["id"]
        assert fact["source_event_ids"]
    for episode in packet["episodes"]:
        assert episode["event_id"] and episode["episode_id"]
    async with async_session() as session:
        assert (
            await session.get(Fact, uuid.UUID(packet["facts"][0]["id"]))
        ) is not None
