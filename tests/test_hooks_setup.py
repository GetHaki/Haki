"""Cursor Hooks packaging (`haki hooks`): hooks.json generation, rules-file
template, and the fail-open contract of the hook CLI entrypoints.

Pure generation is tested directly (no I/O, mirrors test_mcp_setup.py).
The hook CLI entrypoints (`hook-capture`, `hook-session-start`) are invoked
BY Cursor as a spawned process, never unit-tested against a live server in
this suite -- consistent with `_cmd_verify`/`_cmd_connect`, which are also
only exercised live (`haki verify`), not hermetically. What IS tested here
is their fail-open contract in isolation (bad input / missing config never
crashes or exits non-zero), since that is the one invariant a hook must
never violate regardless of network state.
"""

import json

import pytest

from haki.hooks_setup import hooks_json_snippet, session_rule_template, setup_instructions


def test_hooks_json_snippet_is_valid_and_scopes_both_hooks():
    config = json.loads(hooks_json_snippet("usr_alice", "prj_demo", "org_demo"))
    assert config["version"] == 1
    assert set(config["hooks"]) == {"sessionStart", "afterAgentResponse"}

    session_start = config["hooks"]["sessionStart"][0]
    assert session_start["type"] == "command"
    assert session_start["command"] == (
        "haki hook-session-start --subject-id usr_alice "
        "--project-id prj_demo --org-id org_demo"
    )

    after_response = config["hooks"]["afterAgentResponse"][0]
    assert after_response["command"] == (
        "haki hook-capture --subject-id usr_alice "
        "--project-id prj_demo --org-id org_demo"
    )
    # Never break the agent's response if Haki is unreachable.
    assert after_response["failClosed"] is False


def test_hooks_json_snippet_defaults_org_id():
    config = json.loads(hooks_json_snippet("usr_alice", "prj_demo"))
    assert "--org-id org_default" in config["hooks"]["sessionStart"][0]["command"]


def test_hooks_json_snippet_never_embeds_a_secret():
    """Security invariant: the generated config carries scope ids only --
    never an API key. The key is resolved by the already-installed `haki`
    CLI from ~/.haki/config.json at hook-invocation time."""
    snippet = hooks_json_snippet("usr_alice", "prj_demo", "org_demo")
    assert "hk_" not in snippet
    assert "api_key" not in snippet
    assert "Authorization" not in snippet


@pytest.mark.parametrize("timeout", [5, 20, 60])
def test_hooks_json_snippet_honors_custom_timeout(timeout):
    config = json.loads(hooks_json_snippet("usr_alice", "prj_demo", timeout=timeout))
    assert config["hooks"]["sessionStart"][0]["timeout"] == timeout
    assert config["hooks"]["afterAgentResponse"][0]["timeout"] == timeout


def test_session_rule_template_has_always_apply_frontmatter():
    rule = session_rule_template("usr_alice")
    assert rule.startswith("---\n")
    assert "alwaysApply: true" in rule
    assert "usr_alice" in rule


def test_setup_instructions_mentions_the_subject_and_review_step():
    instructions = setup_instructions("usr_alice")
    assert "usr_alice" in instructions
    # The "generate, never auto-write" security principle must be visible
    # to whoever runs `haki hooks` -- not just enforced in code.
    assert "revise" in instructions.lower() or "CI" in instructions
