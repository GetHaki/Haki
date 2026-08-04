"""Haki CLI (stdlib argparse): connect, verify, status, keys, mcp, hooks.

  haki connect --api-url URL [--api-key KEY]   test /health, save ~/.haki/config.json
  haki verify                                  end-to-end timed scenario, exit 0/1
  haki status                                  API health
  haki keys create --project-id P --org-id O   create an API key (shown once)
  haki keys list                               masked key listing
  haki keys revoke KEY_ID                      revoke a key
  haki mcp [--mcp-url URL]                     Cursor packaging (mcp.json, deeplink, rule)
  haki hooks --subject-id S --project-id P     Cursor Hooks packaging (hooks.json, rule)
  haki hook-capture ...                        internal: invoked BY Cursor's afterAgentResponse hook
  haki hook-session-start ...                  internal: invoked BY Cursor's sessionStart hook
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from haki.client import HakiClient
from haki.errors import HakiApiError, HakiError
from haki.hooks_setup import hooks_json_snippet, session_rule_template, setup_instructions
from haki.mcp_setup import deeplink, mcp_json_snippet, project_rule
from haki.runtime import build_prompt_context

CONFIG_PATH = Path.home() / ".haki" / "config.json"

_VERIFY_PROJECT = "prj_haki_verify"
_HOOK_SESSION_RULE_PATH = Path(".cursor") / "rules" / "haki-session.mdc"
_HOOK_CONTEXT_QUERY = (
    "decisions techniques, conventions du projet, preferences, erreurs deja resolues"
)


def _save_config(api_url: str, api_key: str | None) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"api_url": api_url, "api_key": api_key}, indent=2),
        encoding="utf-8",
    )


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"no config at {CONFIG_PATH} — run `haki connect --api-url URL` first"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _client_from_config() -> HakiClient:
    config = _load_config()
    return HakiClient(config["api_url"], api_key=config.get("api_key"))


def _cmd_connect(args: argparse.Namespace) -> int:
    client = HakiClient(args.api_url, api_key=args.api_key, timeout=10)
    started = time.perf_counter()
    try:
        health = client.health()
    except HakiError as exc:
        print(f"connect FAILED: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    _save_config(args.api_url.rstrip("/"), args.api_key)
    # The API key is never printed back.
    print(f"connected to {args.api_url} ({health.get('status')}, "
          f"db {health.get('database')}) in {elapsed * 1000:.0f} ms")
    print(f"config written to {CONFIG_PATH}")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    try:
        with _client_from_config() as client:
            started = time.perf_counter()
            health = client.health()
            elapsed = time.perf_counter() - started
    except HakiError as exc:
        print(f"status FAILED: {exc}")
        return 1
    print(f"api: {health.get('status')} | database: {health.get('database')} "
          f"| health latency: {elapsed * 1000:.0f} ms")
    return 0


def _step(label: str, started: float) -> None:
    print(f"  [{time.perf_counter() - started:6.2f} s] {label}")


def _fact_recalled(facts: list[dict]) -> dict | None:
    for fact in facts:
        rendered = json.dumps(fact.get("value"), ensure_ascii=False).lower()
        if "français" in rendered or "french" in rendered or '"fr"' in rendered:
            return fact
    return None


def _cmd_verify(_args: argparse.Namespace) -> int:
    """Timed end-to-end scenario: capture -> consolidate -> new thread ->
    context recalls the fact. Exit 0 on success, 1 on any failure.

    If no API key is configured, a bootstrap key creation is attempted for
    the verify project (works on a fresh server without HAKI_ADMIN_KEY, or
    in dev-open mode); on refusal the scenario proceeds without a key —
    which fails with 401 on an auth-required server, with a clear message.
    """
    subject_id = f"usr_verify_{uuid.uuid4().hex[:12]}"
    thread_1 = f"thr_{uuid.uuid4().hex[:8]}"
    thread_2 = f"thr_{uuid.uuid4().hex[:8]}"  # new thread: memory must survive it

    print(f"haki verify — subject {subject_id}")
    try:
        config = _load_config()
        if not config.get("api_key"):
            try:
                created = HakiClient(config["api_url"]).create_key(
                    project_id=_VERIFY_PROJECT,
                    org_id="org_haki_verify",
                    label="haki verify bootstrap",
                )
                config["api_key"] = created["key"]
                _save_config(config["api_url"], config["api_key"])
                print(f"  API key created for {_VERIFY_PROJECT} (saved to config)")
            except HakiError:
                pass  # dev-open mode, or key creation needs admin rights
        with _client_from_config() as client:
            t0 = time.perf_counter()

            try:
                client.capture(
                    [
                        {
                            "org_id": "org_haki_verify",
                            "project_id": _VERIFY_PROJECT,
                            "subject_type": "user",
                            "subject_id": subject_id,
                            "thread_id": thread_1,
                            "kind": "conversation.message",
                            "occurred_at": datetime.now(timezone.utc).isoformat(),
                            "payload": {
                                "role": "user",
                                "content": "Je préfère recevoir mes factures en français.",
                            },
                            "idempotency_key": f"verify-{uuid.uuid4()}",
                        }
                    ]
                )
            except HakiApiError as exc:
                # The configured credential (e.g. an admin key) cannot write
                # data: mint a project-scoped key with it, save it, retry once.
                if exc.status_code != 401:
                    raise
                created = client.create_key(
                    project_id=_VERIFY_PROJECT,
                    org_id="org_haki_verify",
                    label="haki verify",
                )
                config["api_key"] = created["key"]
                _save_config(config["api_url"], config["api_key"])
                print(f"  API key created for {_VERIFY_PROJECT} (saved to config)")
                client.close()
                client = HakiClient(config["api_url"], api_key=created["key"])
                client.capture(
                    [
                        {
                            "org_id": "org_haki_verify",
                            "project_id": _VERIFY_PROJECT,
                            "subject_type": "user",
                            "subject_id": subject_id,
                            "thread_id": thread_1,
                            "kind": "conversation.message",
                            "occurred_at": datetime.now(timezone.utc).isoformat(),
                            "payload": {
                                "role": "user",
                                "content": "Je préfère recevoir mes factures en français.",
                            },
                            "idempotency_key": f"verify-{uuid.uuid4()}",
                        }
                    ]
                )
            _step(f"capture (thread {thread_1})", t0)

            t1 = time.perf_counter()
            result = client.consolidate()
            _step(f"consolidate: {result.get('processed')} job(s) processed", t1)

            t2 = time.perf_counter()
            response = client.context(
                subject_id=subject_id,
                query="dans quelle langue envoyer les factures ?",
                project_id=_VERIFY_PROJECT,
                purpose=f"new thread {thread_2}",
            )
            _step(f"context (new thread {thread_2})", t2)

            facts = response["packet"]["facts"]
            trace_id = response["trace_id"]
            fact = _fact_recalled(facts)
            if fact is None:
                print(f"  trace_id: {trace_id}")
                print(f"FAIL: preference not recalled ({len(facts)} fact(s) served: "
                      f"{json.dumps([f.get('value') for f in facts], ensure_ascii=False)})")
                return 1
            print(f"  recalled: {fact['predicate']} = "
                  f"{json.dumps(fact['value'], ensure_ascii=False)}")
            print(f"  trace_id: {trace_id}")
            print(f"OK — total {time.perf_counter() - t0:.2f} s")
            return 0
    except HakiError as exc:
        print(f"FAIL: {exc}")
        return 1


def _cmd_keys_create(args: argparse.Namespace) -> int:
    try:
        with _client_from_config() as client:
            created = client.create_key(
                project_id=args.project_id, org_id=args.org_id, label=args.label
            )
    except HakiError as exc:
        print(f"keys create FAILED: {exc}")
        return 1
    # The clear key is shown exactly once — it is never stored by Haki.
    print(f"key: {created['key']}")
    print(f"id: {created['id']} | prefix: {created['prefix']}… | "
          f"project: {created['project_id']} (org {created['org_id']})")
    if args.save:
        config = _load_config()
        _save_config(config["api_url"], created["key"])
        print(f"key saved to {CONFIG_PATH}")
    return 0


def _cmd_keys_list(_args: argparse.Namespace) -> int:
    try:
        with _client_from_config() as client:
            keys = client.list_keys()["keys"]
    except HakiError as exc:
        print(f"keys list FAILED: {exc}")
        return 1
    if not keys:
        print("no API key")
        return 0
    for key in keys:
        state = "revoked" if key.get("revoked_at") else "active"
        print(f"{key['id']}  {key['prefix']}…  {key['project_id']}  "
              f"{state}  {key.get('label') or ''}")
    return 0


def _cmd_keys_revoke(args: argparse.Namespace) -> int:
    try:
        with _client_from_config() as client:
            client.revoke_key(args.key_id)
    except HakiError as exc:
        print(f"keys revoke FAILED: {exc}")
        return 1
    print(f"key {args.key_id} revoked")
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Print the Cursor packaging: mcp.json snippet, install deeplink and
    the Project Rule (.cursor/rules/haki.mdc)."""
    url = args.mcp_url
    print("== 1. .cursor/mcp.json (a coller dans votre projet) ==")
    print()
    print(mcp_json_snippet(url))
    print()
    print("== 2. Deeplink « Add Haki to Cursor » (installation en un clic) ==")
    print()
    print(deeplink(url))
    print()
    print("== 3. .cursor/rules/haki.mdc (Project Rule, alwaysApply) ==")
    print()
    print(project_rule(url))
    return 0


def _cmd_hooks(args: argparse.Namespace) -> int:
    """Print the Cursor Hooks packaging: hooks.json snippet, the initial
    rules-file template, and setup instructions. Never writes a file —
    same "generate, developer pastes" pattern as `haki mcp` (README —
    Securite: a hook config is executed automatically, it must never be
    written silently, by an agent, or on repo open)."""
    print("== 1. .cursor/hooks.json (a coller a la racine du projet) ==")
    print()
    print(hooks_json_snippet(args.subject_id, args.project_id, args.org_id))
    print()
    print("== 2. .cursor/rules/haki-session.mdc (a creer UNE FOIS -- reecrit ensuite par le hook) ==")
    print()
    print(session_rule_template(args.subject_id))
    print()
    print("== 3. Installation ==")
    print()
    print(setup_instructions(args.subject_id))
    return 0


def _hook_fail_open(reason: str) -> int:
    """A Cursor hook must NEVER crash the agent loop. On any error we print
    a harmless empty JSON result and exit 0 (success) rather than let a
    traceback or non-zero code reach Cursor -- Cursor's own default is
    fail-open on non-zero exit anyway (see hooks.json `failClosed`, unset
    here), this just keeps the failure clean and silent instead of noisy.
    The reason is logged to stderr only (never mixed into the stdout JSON
    Cursor parses)."""
    print(f"haki hook: {reason}", file=sys.stderr)
    print("{}")
    return 0


def _cmd_hook_capture(args: argparse.Namespace) -> int:
    """Invoked by Cursor's afterAgentResponse hook (stdin: {"text": ...,
    "conversation_id": ..., "generation_id": ..., ...}). Captures the
    agent's final response as a Haki event -- purely observational on
    Cursor's side, so this hook cannot fail a Cursor action; it can only
    fail to remember it, which is why every error path here still exits 0."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _hook_fail_open(f"unreadable stdin payload: {exc}")

    text = (payload.get("text") or "").strip()
    if not text:
        return _hook_fail_open("empty afterAgentResponse text, nothing to capture")

    try:
        config = _load_config()
    except SystemExit as exc:
        return _hook_fail_open(str(exc))
    if not config.get("api_key"):
        return _hook_fail_open("no API key configured (run `haki connect` first)")

    conv_id = payload.get("conversation_id") or "unknown"
    gen_id = payload.get("generation_id") or uuid.uuid4().hex
    try:
        with HakiClient(config["api_url"], api_key=config["api_key"], timeout=10) as client:
            client.capture(
                [
                    {
                        "org_id": args.org_id,
                        "project_id": args.project_id,
                        "subject_type": "user",
                        "subject_id": args.subject_id,
                        "agent_id": "cursor",
                        "kind": "agent.observation",
                        "occurred_at": datetime.now(timezone.utc).isoformat(),
                        "payload": {"role": "assistant", "content": text},
                        "source": {"tool": "cursor-hook", "hook": "afterAgentResponse"},
                        # Idempotent per (conversation, generation): a hook
                        # that fires twice for the same response never
                        # creates a duplicate event.
                        "idempotency_key": f"cursor-hook:{conv_id}:{gen_id}",
                    }
                ]
            )
    except HakiError as exc:
        return _hook_fail_open(f"capture failed: {exc}")

    print("{}")
    return 0


def _cmd_hook_session_start(args: argparse.Namespace) -> int:
    """Invoked by Cursor's sessionStart hook (fire-and-forget on Cursor's
    side; stdin: {"session_id": ..., "is_background_agent": ..., ...}).
    Fetches the subject's memory and writes it to
    .cursor/rules/haki-session.mdc (alwaysApply rule) -- the reliable path,
    since the native `additional_context` stdout channel this hook ALSO
    emits is currently broken in Cursor's own agent loop (see
    hooks_setup.py docstring). Runs before the developer's first message,
    so it must be fast and must never block a session on Haki being down."""
    try:
        json.loads(sys.stdin.read() or "{}")  # consumed, not currently needed
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass  # a malformed sessionStart payload is not fatal here

    try:
        config = _load_config()
    except SystemExit as exc:
        return _hook_fail_open(str(exc))
    if not config.get("api_key"):
        return _hook_fail_open("no API key configured (run `haki connect` first)")

    try:
        with HakiClient(config["api_url"], api_key=config["api_key"], timeout=10) as client:
            response = client.context(
                subject_id=args.subject_id,
                query=_HOOK_CONTEXT_QUERY,
                project_id=args.project_id,
                purpose="cursor-hook:sessionStart",
            )
    except HakiError as exc:
        return _hook_fail_open(f"context fetch failed: {exc}")

    prompt_block = build_prompt_context(response.get("packet") or {})

    project_dir = Path(os.environ.get("CURSOR_PROJECT_DIR") or ".")
    rule_path = project_dir / _HOOK_SESSION_RULE_PATH
    try:
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        rule_path.write_text(
            f"""---
description: Memoire Haki (sujet {args.subject_id}) -- reecrite par le hook sessionStart a chaque nouvelle session
alwaysApply: true
---

{prompt_block or "(rien de memorise pour ce sujet pour le moment)"}
""",
            encoding="utf-8",
        )
    except OSError as exc:
        # The rules-file workaround failed (e.g. read-only workspace) --
        # still emit the native channel below on a best-effort basis.
        print(f"haki hook: could not write {rule_path}: {exc}", file=sys.stderr)

    # Best-effort native channel: emitted in case a future Cursor release
    # fixes the documented additional_context delivery bug. Harmless if
    # ignored (fire-and-forget on Cursor's side either way).
    print(json.dumps({"additional_context": prompt_block} if prompt_block else {}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="haki", description="Haki CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_connect = sub.add_parser("connect", help="test the API and save the config")
    p_connect.add_argument("--api-url", required=True, help="base URL of the Haki API")
    p_connect.add_argument("--api-key", default=None, help="API key (optional in V1)")
    p_connect.set_defaults(func=_cmd_connect)

    p_verify = sub.add_parser("verify", help="run the end-to-end memory scenario")
    p_verify.set_defaults(func=_cmd_verify)

    p_status = sub.add_parser("status", help="API health")
    p_status.set_defaults(func=_cmd_status)

    p_keys = sub.add_parser("keys", help="manage API keys")
    keys_sub = p_keys.add_subparsers(dest="keys_command", required=True)

    p_keys_create = keys_sub.add_parser("create", help="create an API key (shown once)")
    p_keys_create.add_argument("--project-id", required=True)
    p_keys_create.add_argument("--org-id", default="org_default")
    p_keys_create.add_argument("--label", default=None)
    p_keys_create.add_argument(
        "--save",
        action="store_true",
        help="save the created key to ~/.haki/config.json",
    )
    p_keys_create.set_defaults(func=_cmd_keys_create)

    p_keys_list = keys_sub.add_parser("list", help="masked key listing")
    p_keys_list.set_defaults(func=_cmd_keys_list)

    p_keys_revoke = keys_sub.add_parser("revoke", help="revoke a key by id")
    p_keys_revoke.add_argument("key_id")
    p_keys_revoke.set_defaults(func=_cmd_keys_revoke)

    p_mcp = sub.add_parser(
        "mcp", help="print the Cursor packaging (mcp.json, deeplink, Project Rule)"
    )
    p_mcp.add_argument(
        "--mcp-url",
        default="http://localhost:8100/mcp",
        help="URL of the Haki MCP endpoint (default: %(default)s)",
    )
    p_mcp.set_defaults(func=_cmd_mcp)

    p_hooks = sub.add_parser(
        "hooks", help="print the Cursor Hooks packaging (hooks.json, rules-file, instructions)"
    )
    p_hooks.add_argument("--subject-id", required=True, help="stable subject id for this developer/workspace")
    p_hooks.add_argument("--project-id", required=True)
    p_hooks.add_argument("--org-id", default="org_default")
    p_hooks.set_defaults(func=_cmd_hooks)

    # Internal: invoked BY Cursor itself (.cursor/hooks.json), not by a
    # human. Scope is required and comes only from the generated command
    # string (see hooks_setup.py) -- never inferred at runtime.
    p_hook_capture = sub.add_parser(
        "hook-capture", help="internal: Cursor afterAgentResponse hook entrypoint"
    )
    p_hook_capture.add_argument("--subject-id", required=True)
    p_hook_capture.add_argument("--project-id", required=True)
    p_hook_capture.add_argument("--org-id", default="org_default")
    p_hook_capture.set_defaults(func=_cmd_hook_capture)

    p_hook_session_start = sub.add_parser(
        "hook-session-start", help="internal: Cursor sessionStart hook entrypoint"
    )
    p_hook_session_start.add_argument("--subject-id", required=True)
    p_hook_session_start.add_argument("--project-id", required=True)
    p_hook_session_start.add_argument("--org-id", default="org_default")
    p_hook_session_start.set_defaults(func=_cmd_hook_session_start)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
