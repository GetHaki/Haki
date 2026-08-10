"""Unit test for app.mcp_server._format_packet (pure function, no server,
no database) — how the M3 recall gate's empty_reason renders for Cursor."""

from app.mcp_server import _format_packet


def test_format_packet_distinguishes_no_relevant_memory_from_no_memory():
    no_memory_at_all = _format_packet(
        "usr_42", {"facts": [], "warnings": [], "status": "ok", "empty_reason": None}
    )
    assert "Aucun fait memorise" in no_memory_at_all
    assert "pertinente" not in no_memory_at_all

    gated = _format_packet(
        "usr_42",
        {
            "facts": [],
            "warnings": [],
            "status": "ok",
            "empty_reason": "no_relevant_memory",
        },
    )
    assert "Aucune memoire suffisamment pertinente" in gated
    assert "Aucun fait memorise" not in gated

    # Neither case is a degradation: no "[memoire ...]" status line.
    assert "[memoire" not in no_memory_at_all
    assert "[memoire" not in gated
