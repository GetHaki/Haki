"""Cursor Hooks packaging (sprint 10): guaranteed capture + session context.

Pure generation, no I/O — the `haki hooks` CLI prints, tests assert.

Design constraints from the real Cursor Hooks API (verified against the
official docs, cursor.com/docs/hooks, checked August 2026):

- Hooks are local processes Cursor spawns, JSON in/out via stdio — not an
  HTTP server. Config lives in ``<project>/.cursor/hooks.json`` (or
  ``~/.cursor/hooks.json`` for a user-wide install).
- ``sessionStart``'s native ``additional_context`` output is fire-and-forget
  (the agent loop never waits for it) AND documented as broken in current
  Cursor releases — a Cursor team member confirmed on their own forum that
  it is dropped due to a timing bug, with no workaround at the time of
  writing. We still emit it on stdout (harmless, costs nothing, may start
  working), but the RELIABLE path is the same workaround the one other
  memory product using Cursor Hooks (Hindsight) ships: write a
  ``.cursor/rules/*.mdc`` file with ``alwaysApply: true``, which Cursor's
  rule engine reads reliably regardless of the broken hook channel.
- ``beforeSubmitPrompt`` can only allow/deny a submission — its documented
  output schema has no content-injection field, unlike ``sessionStart``.
  It is not used here for recall.
- ``afterAgentResponse`` is purely observational (no known bugs) — the
  most reliable of the three, used here for guaranteed capture: no
  cooperation from the agent is required, unlike the MCP tools which the
  model may simply not call.
- The subject_id is baked into the generated command AT GENERATION TIME,
  never inferred at runtime, so the config file is fully reviewable before
  a developer commits it — same principle as ``mcp_setup.py``: generate,
  never auto-write, no secret in the file (the API key is resolved by the
  already-installed ``haki`` CLI from ``~/.haki/config.json``, never
  embedded here). Writing ``.cursor/hooks.json`` automatically (e.g. from
  an agent or on repo open) is exactly the pattern behind a real Cursor
  sandbox-escape CVE (workspace-file-writes-a-hook-config-that-then-runs-
  unapproved) — this module only ever prints, the developer pastes.
"""

import json

DEFAULT_TIMEOUT_SECONDS = 20


def hooks_json_snippet(
    subject_id: str,
    project_id: str,
    org_id: str = "org_default",
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Ready-to-paste .cursor/hooks.json content.

    Scope (subject_id/project_id/org_id) is baked into the command string
    at generation time, never inferred at runtime by the hook itself — the
    same "the model/hook never chooses scope" invariant already enforced
    server-side for the MCP tools (HAKI_MCP_SUBJECT_ID). The generated
    command carries only ids, never the API key: the key is resolved by
    the already-installed `haki` CLI from ~/.haki/config.json.

    sessionStart: best-effort native context injection + the reliable
    rules-file fallback (see session_rule_template()). afterAgentResponse:
    guaranteed capture, non-blocking (no failClosed — never breaks the
    agent's response if Haki is unreachable).
    """
    scope_args = f"--subject-id {subject_id} --project-id {project_id} --org-id {org_id}"
    config = {
        "version": 1,
        "hooks": {
            "sessionStart": [
                {
                    "command": f"haki hook-session-start {scope_args}",
                    "type": "command",
                    "timeout": timeout,
                }
            ],
            "afterAgentResponse": [
                {
                    "command": f"haki hook-capture {scope_args}",
                    "type": "command",
                    "timeout": timeout,
                    "failClosed": False,
                }
            ],
        },
    }
    return json.dumps(config, indent=2)


def session_rule_template(subject_id: str) -> str:
    """Initial content of .cursor/rules/haki-session.mdc, committed ONCE by
    the developer. `haki hook-session-start` overwrites the body below the
    frontmatter on every new Cursor session — this is the reliable
    workaround for the currently-broken native sessionStart channel.
    `alwaysApply: true` so Cursor's rule engine always includes it,
    independent of the hook's own (fire-and-forget, sometimes dropped)
    delivery."""
    return f"""---
description: Memoire Haki (sujet {subject_id}) -- reecrite par le hook sessionStart a chaque nouvelle session
alwaysApply: true
---

(vide pour l'instant -- rempli automatiquement des la premiere session Cursor
apres installation du hook `sessionStart`)
"""


def setup_instructions(subject_id: str) -> str:
    """Human-readable install steps printed by `haki hooks`."""
    return f"""1. Colle le JSON ci-dessus dans .cursor/hooks.json a la racine du projet.
2. Cree .cursor/rules/haki-session.mdc avec le contenu affiche plus bas (une seule fois --
   le hook sessionStart le reecrit ensuite a chaque session).
3. Verifie que le binaire `haki` est installe et connecte (`haki status`) --
   les hooks l'invoquent directement, ils ne portent aucune cle API.
4. Ouvre une nouvelle session Cursor sur ce projet : `haki hook-session-start`
   met a jour .cursor/rules/haki-session.mdc, `haki hook-capture` memorise
   chaque reponse de l'agent automatiquement (aucune action de ta part).
5. .cursor/hooks.json et .cursor/rules/haki-session.mdc sont du code execute
   automatiquement -- revise-les comme tu reviserais une CI avant de les
   committer (sujet {subject_id} en dur, jamais un secret)."""
