"""Cursor packaging (`haki mcp`): mcp.json snippet, deeplink round-trip,
Project Rule frontmatter."""

import json

from haki.mcp_setup import (
    decode_deeplink_config,
    deeplink,
    mcp_json_snippet,
    project_rule,
)


def test_mcp_json_snippet_is_valid_cursor_config():
    config = json.loads(mcp_json_snippet())
    assert config == {"mcpServers": {"haki": {"url": "http://localhost:8100/mcp"}}}


def test_deeplink_decodes_to_the_server_config():
    link = deeplink()
    assert link.startswith("cursor://anysphere.cursor-deeplink/mcp/install?name=haki&config=")
    assert decode_deeplink_config(link) == {"url": "http://localhost:8100/mcp"}


def test_deeplink_honors_a_custom_url():
    link = deeplink("https://memory.example.com/mcp")
    assert decode_deeplink_config(link) == {"url": "https://memory.example.com/mcp"}


def test_project_rule_frontmatter_and_tool_contract():
    rule = project_rule()
    assert rule.startswith("---\n")
    assert "alwaysApply: true" in rule
    assert "description:" in rule
    # The rule must instruct recall-before-work and capture-after-work, and
    # the secrets ban.
    assert "haki_context" in rule
    assert "haki_capture" in rule
    assert "haki_inspect" in rule
    assert "haki_forget" in rule
    assert "jamais de secrets" in rule
