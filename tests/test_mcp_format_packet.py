"""app.mcp_server._format_packet: what a model actually receives over MCP.

Since 22 aout this delegates to the SDK's `build_prompt_context` -- the
same renderer the gateway and every SDK user already get. It used to carry
its own, poorer copy (facts only: no episodes, no `stale`, no `contested`
and none of its resolution instructions, no relative dates), so half the
mechanisms shipped in August never reached a model through MCP -- the very
surface being pushed into the MCP registries.

These tests pin the two things that are MCP's own: an empty packet must
still SAY something (a tool result is not a prompt block, where an empty
string is the right answer), and the two kinds of empty must stay
distinguishable.
"""

from app.mcp_server import _format_packet

EMPTY = {"facts": [], "warnings": [], "status": "ok", "empty_reason": None}
GATED = {**EMPTY, "empty_reason": "no_relevant_memory"}


def test_an_empty_packet_still_says_something():
    """`build_prompt_context` returns "" for an ok packet with nothing in
    it -- correct for a prompt, where an empty block is a distractor, and
    wrong for a tool result, which must be able to say "nothing here"."""
    rendered = _format_packet("usr_42", EMPTY)
    assert rendered.strip()
    assert "usr_42" in rendered


def test_no_memory_and_nothing_relevant_enough_stay_distinguishable():
    """Two different situations for the caller: "this subject is unknown"
    against "this subject is known and nothing cleared the floor"."""
    no_memory = _format_packet("usr_42", EMPTY)
    gated = _format_packet("usr_42", GATED)
    assert "No memory recorded" in no_memory
    assert "relevant enough" in gated
    assert no_memory != gated


def test_episodes_reach_the_model_over_mcp():
    """The regression that motivated the change.

    An episode is what answers "what happened / when". The previous
    renderer dropped them entirely, so an MCP client never saw one.
    """
    rendered = _format_packet(
        "usr_42",
        {
            "facts": [],
            "episodes": [
                {
                    "event_id": "e1",
                    "episode_id": "c1",
                    "kind": "chat_session",
                    "occurred_at": "2026-05-01T10:00:00+00:00",
                    "excerpt": "user: I bought a Zolgorvex kayak.",
                    "context_neighbor": False,
                }
            ],
            "warnings": [],
            "status": "ok",
        },
    )
    assert "Zolgorvex" in rendered


def test_a_contested_fact_reaches_the_model_with_its_instructions():
    """A contested pair is served on purpose (13 aout) and is useless
    without the instruction telling the model how to resolve it."""
    rendered = _format_packet(
        "usr_42",
        {
            "facts": [
                {
                    "id": "f1",
                    "predicate": "employer",
                    "value": {"company": "Acme"},
                    "confidence": 0.9,
                    "valid_from": "2026-01-01T00:00:00+00:00",
                    "source_event_ids": [],
                    "fact_kind": "attribute",
                    "volatility": "stable",
                    "freshness": "current",
                    "origin_trust": "trusted",
                    "contested": True,
                    "conflict_id": "k1",
                }
            ],
            "warnings": [],
            "status": "ok",
        },
    )
    assert "contested" in rendered.lower()


def test_a_degraded_packet_is_still_announced():
    rendered = _format_packet("usr_42", {**EMPTY, "status": "degraded"})
    assert "degraded" in rendered.lower()
