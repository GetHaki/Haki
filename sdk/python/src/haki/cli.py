"""Haki CLI (stdlib argparse): login, connect, verify, status, keys, mcp, hooks.

  haki login [--api-url URL]                   device-code flow: browser approval, saves config
  haki connect --api-url URL [--api-key KEY]   test /health, save ~/.haki/config.json
  haki verify                                  end-to-end timed scenario, exit 0/1
  haki status                                  API health
  haki keys create --project-id P --org-id O   create an API key (shown once)
  haki keys list                               masked key listing
  haki keys revoke KEY_ID                      revoke a key
  haki mcp [--project-id P] [--org-id O]       Cursor packaging (mcp.json, deeplink, rule)
  haki hooks --subject-id S [--project-id P]   Cursor Hooks packaging (hooks.json, rule)
  haki hook-capture ...                        internal: invoked BY Cursor's afterAgentResponse hook
  haki hook-session-start ...                  internal: invoked BY Cursor's sessionStart hook

`mcp`/`hooks` --project-id/--org-id are optional: when omitted they fall
back to the values saved by `haki login` (or `haki connect` + `haki keys
create --save`) in ~/.haki/config.json; a clear error is raised if neither
a flag nor a saved config provides them. Both also accept --write, which
(after an explicit y/N confirmation) writes the generated files to disk
instead of only printing them.
"""

import argparse
import json
import os
import sys
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from haki.client import HakiClient
from haki.errors import HakiApiError, HakiError
from haki.hooks_setup import hooks_json_snippet, session_rule_template, setup_instructions
from haki.mcp_setup import deeplink, mcp_json_snippet, project_rule
from haki.runtime import build_prompt_context

CONFIG_PATH = Path.home() / ".haki" / "config.json"

_VERIFY_PROJECT = "prj_haki_verify"
_MCP_JSON_PATH = Path(".cursor") / "mcp.json"
_MCP_RULE_PATH = Path(".cursor") / "rules" / "haki.mdc"
_HOOKS_JSON_PATH = Path(".cursor") / "hooks.json"
_HOOK_SESSION_RULE_PATH = Path(".cursor") / "rules" / "haki-session.mdc"
_HOOK_CONTEXT_QUERY = (
    "decisions techniques, conventions du projet, preferences, erreurs deja resolues"
)

# `haki login` never waits longer than expires_in (from the server) plus
# this small client-side margin -- a strict, independent upper bound so a
# server bug (never returning "expired") or a dropped connection can never
# turn the poll loop into an infinite wait.
_LOGIN_TIMEOUT_MARGIN_SECONDS = 5


def _read_raw_config() -> dict:
    """Best-effort read of ~/.haki/config.json — {} if absent/corrupt.
    Used by callers that must NOT hard-fail just because no config exists
    yet (unlike _load_config, which is for commands that require one)."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_config(
    api_url: str,
    api_key: str | None,
    *,
    org_id: str | None = None,
    project_id: str | None = None,
) -> None:
    """Write ~/.haki/config.json. Merges onto any existing file so that a
    field not passed here (e.g. org_id/project_id on a plain `haki connect`)
    is preserved rather than wiped -- retrocompatible with the pre-login
    schema {api_url, api_key}."""
    config = _read_raw_config()
    config["api_url"] = api_url
    config["api_key"] = api_key
    if org_id is not None:
        config["org_id"] = org_id
    if project_id is not None:
        config["project_id"] = project_id
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"no config at {CONFIG_PATH} — run `haki login` "
            "(or `haki connect --api-url URL`) first"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve_scope(cli_project_id: str | None, cli_org_id: str | None) -> tuple[str, str]:
    """project_id/org_id resolution for `haki mcp`/`haki hooks`: an
    explicit flag always wins; otherwise fall back to ~/.haki/config.json
    (populated by `haki login`, or `haki connect` + `haki keys create
    --save`). Raises a clear, actionable SystemExit if neither source has
    them -- these commands never guess a silent default scope."""
    config = _read_raw_config()
    project_id = cli_project_id or config.get("project_id")
    org_id = cli_org_id or config.get("org_id")
    if not project_id or not org_id:
        missing = [
            name
            for name, value in (("--project-id", project_id), ("--org-id", org_id))
            if not value
        ]
        raise SystemExit(
            f"{' et '.join(missing)} manquant(s), et absent(s) de {CONFIG_PATH} -- "
            "passe le(s) flag(s) manquant(s), ou lance `haki login` d'abord "
            "(ou `haki connect` + `haki keys create --save`)"
        )
    return project_id, org_id


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "o", "oui")


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _client_from_config() -> HakiClient:
    config = _load_config()
    return HakiClient(config["api_url"], api_key=config.get("api_key"))


def _cmd_login(args: argparse.Namespace) -> int:
    """Device-code flow (`gh auth login` / `vercel login` pattern):
    POST /v1/cli/device/start (no auth), show the user_code + verification
    URL, best-effort open the browser, then POST /v1/cli/device/poll every
    `interval` seconds until 'approved' or 'expired'. Saves
    api_url/api_key/org_id/project_id to ~/.haki/config.json on success --
    no need to run `haki connect` or `haki keys create` by hand afterwards.

    Client-side timeout is strict: the poll loop NEVER waits longer than
    the server's expires_in plus a small fixed margin, regardless of what
    the server reports (see _LOGIN_TIMEOUT_MARGIN_SECONDS) -- a server bug
    or a stuck connection can never turn this into an infinite wait.
    """
    api_url = args.api_url
    if api_url is None:
        api_url = _read_raw_config().get("api_url")
    if not api_url:
        print(
            "login FAILED: --api-url requis (aucune config existante) -- "
            "ex: haki login --api-url https://api.haki.example.com"
        )
        return 1
    api_url = api_url.rstrip("/")

    client = HakiClient(api_url, timeout=10)
    try:
        try:
            start = client.cli_device_start()
        except HakiError as exc:
            print(f"login FAILED: impossible de demarrer la connexion: {exc}")
            return 1

        device_code = start["device_code"]
        user_code = start["user_code"]
        verification_uri = start["verification_uri"]
        expires_in = float(start["expires_in"])
        interval = max(float(start["interval"]), 0.5)

        # RFC 8628 §3.3.1: the code stays printed and the plain URI stays
        # usable (typing the code on a phone is a supported path), but the
        # browser gets the link that carries the code -- one thing to
        # follow instead of two things to transcribe. `.get` because the
        # SDK is versioned independently of the server it talks to: an
        # older API returns no verification_uri_complete.
        complete_uri = start.get("verification_uri_complete") or verification_uri

        print(f"Code : {user_code}")
        print(f"Ouvrez {verification_uri} et entrez ce code pour connecter ce terminal.")

        try:
            webbrowser.open(complete_uri)
        except Exception:
            pass  # best-effort only -- never fails the login flow

        print("En attente de l'approbation...")
        deadline = time.monotonic() + expires_in + _LOGIN_TIMEOUT_MARGIN_SECONDS
        while True:
            if time.monotonic() >= deadline:
                print(
                    "login FAILED: delai depasse sans approbation -- "
                    "relancez `haki login`."
                )
                return 1

            time.sleep(interval)

            try:
                poll = client.cli_device_poll(device_code)
            except HakiError as exc:
                print(f"login FAILED: {exc}")
                return 1

            status = poll.get("status")
            if status == "approved":
                _save_config(
                    api_url,
                    poll["api_key"],
                    org_id=poll.get("org_id"),
                    project_id=poll.get("project_id"),
                )
                print(f"Connecte -- org {poll.get('org_id')}, projet {poll.get('project_id')}.")
                print(f"config ecrite dans {CONFIG_PATH}")
                return 0
            if status == "expired":
                print(
                    "login FAILED: code expire avant approbation -- "
                    "relancez `haki login`."
                )
                return 1
            # "pending": keep polling until the deadline above.
    finally:
        client.close()


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


# The two messages of the verify scenario, in the SAME thread: a stated
# preference, then a change of mind. Kept verbatim (French, "en fait X
# plutôt que Y") because this exact phrasing is what was measured to make
# the extractor classify the second message as action="supersede" rather
# than action="create" — a "create" would open a conflict set instead, and
# an open conflict is never served (app/context: reason_code
# "conflict_open"), so the scenario would fail for a reason that has
# nothing to do with what it is meant to demonstrate. Reword only with a
# real extraction provider in front of you, never on inspection alone.
_VERIFY_MESSAGE_1 = "Je préfère recevoir mes factures en français."
_VERIFY_MESSAGE_2 = "En fait, envoie-les moi en anglais plutôt, pas en français."

# What the demo fact is about. Matched loosely on the predicate because the
# extractor mints the string itself ("invoice_language", "billing_language",
# "langue_facture", …) — pinning one exact spelling would make verify fail
# on a correct system that simply named the fact differently.
_VERIFY_PREDICATE_HINT = "lang"
_VERIFY_ENGLISH_TOKENS = ('"en"', "english", "anglais")


def _stdout_mark() -> str:
    """The step marker: a tick when the output stream can encode one,
    the ASCII "OK" otherwise.

    `haki verify` is often piped or redirected, and a redirected stream
    falls back to the locale encoding (cp1252 on a French Windows, ascii
    under LC_ALL=C) — neither can encode "✔". Crashing the verification
    command on its own decoration would be an absurd failure mode.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✔".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "OK"
    return "✔"


# Width of the middle column. The captured messages are the widest thing
# that goes in it, so it is sized on them: message 1 fits whole, message 2
# is elided rather than pushing the notes column out of alignment.
_VERIFY_DETAIL_WIDTH = 50


def _verify_step(mark: str, label: str, detail: str, note: str = "") -> None:
    print(f"  {mark} {label:<11} {detail:<{_VERIFY_DETAIL_WIDTH}} {note}".rstrip())


def _quoted(message: str) -> str:
    budget = _VERIFY_DETAIL_WIDTH - 2  # the two quotes
    text = message if len(message) <= budget else message[: budget - 3] + "..."
    return f'"{text}"'


def _verify_event(subject_id: str, thread_id: str, content: str) -> dict:
    return {
        "org_id": "org_haki_verify",
        "project_id": _VERIFY_PROJECT,
        "subject_type": "user",
        "subject_id": subject_id,
        "thread_id": thread_id,
        "kind": "conversation.message",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "payload": {"role": "user", "content": content},
        "idempotency_key": f"verify-{uuid.uuid4()}",
    }


def _language_fact(facts: list[dict]) -> dict | None:
    """The scenario's fact, whatever exact predicate string it was given."""
    for fact in facts:
        if _VERIFY_PREDICATE_HINT in str(fact.get("predicate", "")).lower():
            return fact
    return None


def _says_english(fact: dict) -> bool:
    rendered = json.dumps(fact.get("value"), ensure_ascii=False).lower()
    return any(token in rendered for token in _VERIFY_ENGLISH_TOKENS)


def _render_value(fact: dict) -> str:
    return json.dumps(fact.get("value"), ensure_ascii=False)


def _valid_since(fact: dict) -> str:
    raw = fact.get("valid_from")
    return f"valid since {str(raw)[:10]}" if raw else ""


def _verify_first_capture(
    client: HakiClient, config: dict, event: dict
) -> HakiClient:
    """First capture, with the one-shot key repair verify has always done:
    if the configured credential cannot write (401 — typically an admin
    key, which manages keys but owns no project scope), mint a
    project-scoped key with it, save it, retry once. Returns the client to
    use for the rest of the scenario — a new one when a key was minted,
    otherwise the one passed in."""
    try:
        client.capture([event])
        return client
    except HakiApiError as exc:
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
        retried = HakiClient(config["api_url"], api_key=created["key"])
        retried.capture([event])
        return retried


def _cmd_verify(_args: argparse.Namespace) -> int:
    """Timed end-to-end scenario. Exit 0 on success, 1 on any failure.

    Two messages in the SAME thread — a stated preference, then a change of
    mind — then a context call from a NEW thread. What this demonstrates
    that a plain capture-then-recall cannot: the packet serves the CURRENT
    value, the replaced one is still on file marked superseded rather than
    deleted, and the whole answer carries a trace id. That is the product
    promise in one command, which is why both halves are hard requirements
    here: recalling the stale value, or losing the old one entirely, both
    exit 1.

    Consolidation runs through the SUBJECT-scoped endpoint. The unscoped
    POST /v1/consolidate this used to call drains every project's pending
    jobs on a session without RLS — on a shared server that means a verify
    run does (and waits on) other tenants' work.

    If no API key is configured, a bootstrap key creation is attempted for
    the verify project (works on a fresh server without HAKI_ADMIN_KEY, or
    in dev-open mode); on refusal the scenario proceeds without a key —
    which fails with 401 on an auth-required server, with a clear message.
    """
    subject_id = f"usr_verify_{uuid.uuid4().hex[:12]}"
    thread_1 = f"thr_{uuid.uuid4().hex[:8]}"
    thread_2 = f"thr_{uuid.uuid4().hex[:8]}"  # new thread: memory must survive it
    mark = _stdout_mark()

    print(f"haki verify — subject {subject_id}")
    print()
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

        client = _client_from_config()
        try:
            t0 = time.perf_counter()

            client = _verify_first_capture(
                client, config, _verify_event(subject_id, thread_1, _VERIFY_MESSAGE_1)
            )
            _verify_step(mark, "capture", _quoted(_VERIFY_MESSAGE_1), thread_1)

            t = time.perf_counter()
            client.consolidate_subject(
                project_id=_VERIFY_PROJECT, subject_id=subject_id
            )
            consolidate_1_s = time.perf_counter() - t
            extracted = len(
                client.facts(
                    project_id=_VERIFY_PROJECT, subject_id=subject_id, status="active"
                )["facts"]
            )
            _verify_step(
                mark,
                "consolidate",
                f"{extracted} fact(s) extracted",
                f"{consolidate_1_s:.1f}s",
            )

            client.capture(
                [_verify_event(subject_id, thread_1, _VERIFY_MESSAGE_2)]
            )
            _verify_step(
                mark, "capture", _quoted(_VERIFY_MESSAGE_2), f"{thread_1} (same thread)"
            )

            t = time.perf_counter()
            client.consolidate_subject(
                project_id=_VERIFY_PROJECT, subject_id=subject_id
            )
            consolidate_2_s = time.perf_counter() - t

            # Read the superseded side now so the consolidate line can state
            # what actually happened instead of a job count.
            replaced = _language_fact(
                client.facts(
                    project_id=_VERIFY_PROJECT,
                    subject_id=subject_id,
                    status="superseded",
                )["facts"]
            )
            _verify_step(
                mark,
                "consolidate",
                "1 supersession" if replaced else "no supersession recorded",
                f"{consolidate_2_s:.1f}s",
            )

            t = time.perf_counter()
            response = client.context(
                subject_id=subject_id,
                query="dans quelle langue envoyer les factures ?",
                project_id=_VERIFY_PROJECT,
                purpose=f"new thread {thread_2}",
            )
            _verify_step(
                mark,
                "context",
                f"NEW thread {thread_2}",
                f"{time.perf_counter() - t:.1f}s",
            )
            print()

            served = response["packet"]["facts"]
            trace_id = response["trace_id"]
            recalled = _language_fact(served)

            if recalled is None:
                print(f"    trace     {trace_id}")
                print(
                    f"FAIL: nothing recalled about the preference "
                    f"({len(served)} fact(s) served: "
                    f"{json.dumps([f.get('value') for f in served], ensure_ascii=False)})"
                )
                return 1

            print(
                f"    recalled  {recalled['predicate']} = "
                f"{_render_value(recalled)}   {_valid_since(recalled)}".rstrip()
            )
            if replaced is not None:
                print(
                    f"    hidden    {replaced['predicate']} = "
                    f"{_render_value(replaced)}   superseded"
                )
            print(f"    trace     {trace_id}")
            print()

            if not _says_english(recalled):
                print(
                    "FAIL: the update was not applied — the packet still serves "
                    f"{_render_value(recalled)} after the correction."
                )
                return 1
            if replaced is None:
                print(
                    "FAIL: the current value is served, but the value it replaced "
                    "is not on file as superseded — the update was stored as a "
                    "brand-new fact, so nothing proves the two are the same memory."
                )
                return 1

            print(
                "OK — your agent remembered across conversations, and it can "
                f"prove it.  {time.perf_counter() - t0:.1f}s"
            )
            return 0
        finally:
            client.close()
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
    the Project Rule (.cursor/rules/haki.mdc). --project-id/--org-id are
    optional (default: saved config, see _resolve_scope) and only used here
    to remind the developer which HAKI_MCP_PROJECT_ID/HAKI_MCP_ORG_ID their
    MCP server process should be configured with -- the generated mcp.json
    itself carries no scope or secret (see mcp_setup.py).

    With --write, additionally writes .cursor/mcp.json and
    .cursor/rules/haki.mdc to disk -- only after an explicit y/N
    confirmation: once mcp.json is present, Cursor connects to that MCP
    server automatically and can invoke its tools without further action."""
    project_id, org_id = _resolve_scope(args.project_id, args.org_id)
    url = args.mcp_url
    print(f"Scope resolu — project: {project_id}  org: {org_id}")
    print("(a assortir cote serveur MCP : HAKI_MCP_PROJECT_ID / HAKI_MCP_ORG_ID)")
    print()
    print("== 1. .cursor/mcp.json (a coller dans votre projet) ==")
    print()
    mcp_json = mcp_json_snippet(url)
    print(mcp_json)
    print()
    print("== 2. Deeplink « Add Haki to Cursor » (installation en un clic) ==")
    print()
    print(deeplink(url))
    print()
    print("== 3. .cursor/rules/haki.mdc (Project Rule, alwaysApply) ==")
    print()
    rule = project_rule(url)
    print(rule)

    if args.write:
        print()
        print(
            "/!\\ Une fois .cursor/mcp.json present, Cursor se connecte "
            "AUTOMATIQUEMENT a ce serveur MCP et peut invoquer ses outils."
        )
        if _confirm(f"Ecrire {_MCP_JSON_PATH} et {_MCP_RULE_PATH} maintenant ?"):
            _write_file(_MCP_JSON_PATH, mcp_json)
            _write_file(_MCP_RULE_PATH, rule)
            print(f"ecrit: {_MCP_JSON_PATH}")
            print(f"ecrit: {_MCP_RULE_PATH}")
        else:
            print("Annule — rien ecrit sur le disque.")
    return 0


def _cmd_hooks(args: argparse.Namespace) -> int:
    """Print the Cursor Hooks packaging: hooks.json snippet, the initial
    rules-file template, and setup instructions. --project-id/--org-id are
    optional (default: saved config, see _resolve_scope); --subject-id
    stays required (per-developer/workspace, never inferred).

    Without --write, never writes a file — same "generate, developer
    pastes" pattern as `haki mcp` (README — Securite: a hook config is
    executed automatically, it must never be written silently, by an
    agent, or on repo open). With --write, writes .cursor/hooks.json and
    .cursor/rules/haki-session.mdc to disk, only after an explicit y/N
    confirmation that spells out that Cursor executes this automatically."""
    project_id, org_id = _resolve_scope(args.project_id, args.org_id)
    print("== 1. .cursor/hooks.json (a coller a la racine du projet) ==")
    print()
    hooks_json = hooks_json_snippet(args.subject_id, project_id, org_id)
    print(hooks_json)
    print()
    print("== 2. .cursor/rules/haki-session.mdc (a creer UNE FOIS -- reecrit ensuite par le hook) ==")
    print()
    session_rule = session_rule_template(args.subject_id)
    print(session_rule)
    print()
    print("== 3. Installation ==")
    print()
    print(setup_instructions(args.subject_id))

    if args.write:
        print()
        print(
            "/!\\ .cursor/hooks.json est du code que Cursor EXECUTE AUTOMATIQUEMENT "
            "(haki hook-session-start a chaque session, haki hook-capture apres "
            "chaque reponse de l'agent)."
        )
        if _confirm(f"Ecrire {_HOOKS_JSON_PATH} et {_HOOK_SESSION_RULE_PATH} maintenant ?"):
            _write_file(_HOOKS_JSON_PATH, hooks_json)
            _write_file(_HOOK_SESSION_RULE_PATH, session_rule)
            print(f"ecrit: {_HOOKS_JSON_PATH}")
            print(f"ecrit: {_HOOK_SESSION_RULE_PATH}")
        else:
            print("Annule — rien ecrit sur le disque.")
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

    p_login = sub.add_parser(
        "login", help="device-code login: browser approval, saves the config"
    )
    p_login.add_argument(
        "--api-url",
        default=None,
        help="base URL of the Haki API (default: from saved config, else required)",
    )
    p_login.set_defaults(func=_cmd_login)

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
    p_mcp.add_argument(
        "--project-id", default=None, help="default: from saved config (haki login)"
    )
    p_mcp.add_argument(
        "--org-id", default=None, help="default: from saved config (haki login)"
    )
    p_mcp.add_argument(
        "--write",
        action="store_true",
        help="write .cursor/mcp.json and .cursor/rules/haki.mdc to disk (asks for confirmation)",
    )
    p_mcp.set_defaults(func=_cmd_mcp)

    p_hooks = sub.add_parser(
        "hooks", help="print the Cursor Hooks packaging (hooks.json, rules-file, instructions)"
    )
    p_hooks.add_argument("--subject-id", required=True, help="stable subject id for this developer/workspace")
    p_hooks.add_argument(
        "--project-id", default=None, help="default: from saved config (haki login)"
    )
    p_hooks.add_argument(
        "--org-id", default=None, help="default: from saved config (haki login)"
    )
    p_hooks.add_argument(
        "--write",
        action="store_true",
        help="write .cursor/hooks.json and .cursor/rules/haki-session.mdc to disk (asks for confirmation)",
    )
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
