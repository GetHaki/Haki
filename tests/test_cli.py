"""CLI-level tests: `haki login` (device-code flow) and the
--project-id/--org-id resolution + --write behavior of `haki mcp`/`haki
hooks`.

Two kinds of coverage on purpose:

- The full device-code round-trip -- `haki login` talking to the REAL
  FastAPI + Redis backend over real sockets (a genuine `uvicorn` server
  subprocess, see `live_server()`), approval simulated with a direct call
  to POST /v1/cli/device/approve (exactly as instructed), and the
  single-use consumption security property checked explicitly (a second
  poll of the same device_code must never see 'approved' again).

- The CLI's OWN client-side polling/timeout algorithm (strict upper bound
  that never waits past expires_in + a small margin, tolerance of a
  webbrowser.open() failure, immediate reaction to an 'expired' status) is
  tested against a fake stand-in for HakiClient with fully controlled, fast
  timings -- the real server's expires_in/interval (600s / 3s, see
  app/api/routes/cli_auth.py: EXPIRES_IN, POLL_INTERVAL) are fixed
  server-side constants, not meant to be raced in a unit test.

`haki mcp`/`haki hooks` --write and --project-id/--org-id resolution need
no server at all -- pure argparse + filesystem, tested directly.
"""

import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from haki import cli
from haki.client import HakiClient
from haki.errors import HakiApiError

ROOT = Path(__file__).resolve().parent.parent
_SERVICE_KEY = "test-console-service-key-for-cli-login"


# -- real backend, real sockets -------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def live_server():
    """A real `python -m uvicorn app.main:app` subprocess, run with the
    exact interpreter/environment this test itself runs under (so whatever
    is already installed there -- fastapi, uvicorn, redis -- is reused, no
    separate `uv run` project resolution involved). Readiness is probed via
    POST /v1/cli/device/start itself (Redis-only, no Postgres dependency),
    since that is the only surface this test exercises.
    """
    port = _free_port()
    env = {
        **os.environ,
        "HAKI_CONSOLE_SERVICE_KEY": _SERVICE_KEY,
        # Hermetic startup: no local embedding model download, no remote
        # LLM call (same rationale as tests/conftest.py).
        "HAKI_EMBED_PROVIDER": "fake",
        "HAKI_LLM_PROVIDER": "fake",
    }
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                response = httpx.post(f"{base_url}/v1/cli/device/start", timeout=1)
                if response.status_code == 201:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        if not ready:
            proc.terminate()
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"uvicorn server never became ready (exit={proc.poll()}):\n{output[-4000:]}"
            )
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_login_full_device_code_round_trip_against_real_server(monkeypatch, tmp_path):
    """`haki login` end to end: device/start -> human approves (simulated
    with a direct POST /v1/cli/device/approve) -> haki login's own poll
    picks it up -> config saved. Also verifies the contract's user_code
    format and the single-use consumption security property."""
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    opened_urls = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened_urls.append(url))

    # Spy on the class method so we can observe device_code/user_code the
    # CLI itself received, without scraping stdout from a background
    # thread -- delegates to the real implementation (a real HTTP call).
    captured = {}
    got_start = threading.Event()
    original_start = HakiClient.cli_device_start

    def spy_start(self):
        payload = original_start(self)
        captured.update(payload)
        got_start.set()
        return payload

    monkeypatch.setattr(HakiClient, "cli_device_start", spy_start)

    with live_server() as base_url:
        result = {}

        def run_login():
            result["code"] = cli.main(["login", "--api-url", base_url])

        thread = threading.Thread(target=run_login)
        thread.start()

        assert got_start.wait(timeout=10), "haki login never called device/start"
        user_code = captured["user_code"]
        assert re.fullmatch(r"[A-Z0-9]{4}-[A-Z0-9]{4}", user_code)
        assert not set("01OI") & set(user_code.replace("-", ""))
        assert len(captured["device_code"]) == 64
        assert all(c in "0123456789abcdef" for c in captured["device_code"])

        approve = httpx.post(
            f"{base_url}/v1/cli/device/approve",
            json={
                "user_code": user_code,
                "api_key": "hk_test_e2e",
                "org_id": "org_e2e",
                "project_id": "prj_e2e",
            },
            headers={"Authorization": f"Bearer {_SERVICE_KEY}"},
            timeout=5,
        )
        assert approve.status_code == 200
        assert approve.json() == {"ok": True}

        thread.join(timeout=20)
        assert not thread.is_alive(), "haki login did not return in time"

        # Security property (explicitly required): once consumed by a poll,
        # the SAME device_code must never yield 'approved' again -- a
        # leaked/replayed device_code cannot be used to fetch the key twice.
        replay = httpx.post(
            f"{base_url}/v1/cli/device/poll",
            json={"device_code": captured["device_code"]},
            timeout=5,
        )
        assert replay.json().get("status") != "approved"
        assert replay.status_code == 404  # consumed -> deleted from Redis

    assert result["code"] == 0
    # The browser gets the code-carrying link (RFC 8628 §3.3.1), which the
    # real server built from the real user_code -- not a fixture's guess.
    assert opened_urls == [captured["verification_uri_complete"]]
    assert opened_urls[0] == f"{captured['verification_uri']}?code={user_code}"
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved == {
        "api_url": base_url,
        "api_key": "hk_test_e2e",
        "org_id": "org_e2e",
        "project_id": "prj_e2e",
    }


def test_login_rejects_unauthorized_approve_against_real_server():
    """/v1/cli/device/approve without the console service key must be
    refused -- a customer's own hk_ key (or no credential at all) can never
    approve a device code."""
    with live_server() as base_url:
        start = httpx.post(f"{base_url}/v1/cli/device/start", timeout=5)
        assert start.status_code == 201
        user_code = start.json()["user_code"]

        approve = httpx.post(
            f"{base_url}/v1/cli/device/approve",
            json={
                "user_code": user_code,
                "api_key": "hk_should_not_be_accepted",
                "org_id": "org_x",
                "project_id": "prj_x",
            },
            headers={"Authorization": "Bearer wrong-secret"},
            timeout=5,
        )
        assert approve.status_code == 401


# -- CLI polling/timeout algorithm, fake client, no network ---------------


class _FakeDeviceClient:
    """Stand-in for HakiClient exposing exactly the two calls `_cmd_login`
    makes, with fully controlled, fast timings -- isolates the CLI's own
    poll-loop/timeout logic from the real server's fixed 600s/3s constants.
    """

    def __init__(
        self, poll_statuses, expires_in=0.3, interval=0.05, with_complete_uri=True
    ):
        self._poll_statuses = list(poll_statuses)
        self.with_complete_uri = with_complete_uri
        self.expires_in = expires_in
        self.interval = interval
        self.poll_calls = 0
        self.closed = False

    def cli_device_start(self):
        payload = {
            "device_code": "fakecode",
            "user_code": "ABCD-EFGH",
            "verification_uri": "http://localhost:3000/cli-auth",
            "expires_in": self.expires_in,
            "interval": self.interval,
        }
        if self.with_complete_uri:
            payload["verification_uri_complete"] = (
                "http://localhost:3000/cli-auth?code=ABCD-EFGH"
            )
        return payload

    def cli_device_poll(self, device_code):
        assert device_code == "fakecode"
        index = min(self.poll_calls, len(self._poll_statuses) - 1)
        self.poll_calls += 1
        status = self._poll_statuses[index]
        if status == "approved":
            return {
                "status": "approved",
                "api_key": "hk_fake",
                "org_id": "org_fake",
                "project_id": "prj_fake",
            }
        return {"status": status}

    def close(self):
        self.closed = True


def test_login_client_side_timeout_is_bounded_even_if_server_never_expires(monkeypatch, tmp_path):
    """The strict client-side requirement: even a (simulated buggy) server
    that keeps saying 'pending' forever cannot make `haki login` hang --
    it bails out on its own after expires_in + a small margin."""
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: None)
    fake = _FakeDeviceClient(poll_statuses=["pending"], expires_in=0.3, interval=0.05)
    monkeypatch.setattr(cli, "HakiClient", lambda *a, **kw: fake)

    started = time.monotonic()
    code = cli.main(["login", "--api-url", "http://fake.local"])
    elapsed = time.monotonic() - started

    assert code == 1
    assert elapsed < 0.3 + cli._LOGIN_TIMEOUT_MARGIN_SECONDS + 2  # bounded, never infinite
    assert fake.closed


def test_login_reacts_to_expired_status_immediately(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: None)
    fake = _FakeDeviceClient(poll_statuses=["expired"], expires_in=30, interval=0.05)
    monkeypatch.setattr(cli, "HakiClient", lambda *a, **kw: fake)

    started = time.monotonic()
    code = cli.main(["login", "--api-url", "http://fake.local"])
    elapsed = time.monotonic() - started

    assert code == 1
    assert elapsed < 2  # returns immediately, doesn't wait out the 30s deadline


def test_login_succeeds_and_survives_webbrowser_open_failure(monkeypatch, tmp_path):
    """webbrowser.open() is best-effort: it must never fail the flow."""
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")

    def boom(_url):
        raise RuntimeError("no display available")

    monkeypatch.setattr(cli.webbrowser, "open", boom)
    fake = _FakeDeviceClient(poll_statuses=["pending", "approved"], expires_in=5, interval=0.02)
    monkeypatch.setattr(cli, "HakiClient", lambda *a, **kw: fake)

    code = cli.main(["login", "--api-url", "http://fake.local"])

    assert code == 0
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved == {
        "api_url": "http://fake.local",
        "api_key": "hk_fake",
        "org_id": "org_fake",
        "project_id": "prj_fake",
    }


def test_login_requires_api_url_when_no_saved_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")

    code = cli.main(["login"])

    assert code == 1
    assert "--api-url" in capsys.readouterr().out


def test_login_falls_back_to_the_saved_api_url(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"api_url": "http://saved.example", "api_key": "hk_old"}))
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: None)
    fake = _FakeDeviceClient(poll_statuses=["approved"], expires_in=5, interval=0.02)
    seen = {}

    def factory(url, timeout=10):
        seen["url"] = url
        return fake

    monkeypatch.setattr(cli, "HakiClient", factory)

    code = cli.main(["login"])

    assert code == 0
    assert seen["url"] == "http://saved.example"


def test_login_start_failure_is_reported_and_returns_1(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")

    class _BoomClient:
        def cli_device_start(self):
            raise HakiApiError("too many requests", status_code=429, error_type="rate_limited")

        def close(self):
            pass

    monkeypatch.setattr(cli, "HakiClient", lambda *a, **kw: _BoomClient())

    code = cli.main(["login", "--api-url", "http://fake.local"])

    assert code == 1


# -- `haki mcp` / `haki hooks`: optional scope + --write -------------------


def test_resolve_scope_prefers_explicit_flags_over_saved_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"project_id": "prj_config", "org_id": "org_config"}))
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    assert cli._resolve_scope("prj_flag", "org_flag") == ("prj_flag", "org_flag")


def test_resolve_scope_falls_back_to_saved_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"project_id": "prj_config", "org_id": "org_config"}))
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    assert cli._resolve_scope(None, None) == ("prj_config", "org_config")


def test_resolve_scope_errors_clearly_when_absent_from_both_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")

    with pytest.raises(SystemExit) as excinfo:
        cli._resolve_scope(None, None)
    assert "--project-id" in str(excinfo.value)
    assert "--org-id" in str(excinfo.value)


def test_mcp_missing_scope_errors_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        cli.main(["mcp"])


def test_mcp_uses_saved_scope_when_flags_omitted(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"api_url": "http://api.example", "api_key": "hk_x",
                    "project_id": "prj_saved", "org_id": "org_saved"})
    )
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["mcp"])

    assert code == 0
    out = capsys.readouterr().out
    assert "prj_saved" in out
    assert "org_saved" in out


def test_mcp_without_write_never_prompts_or_writes(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")

    def must_not_be_called(*_a, **_kw):
        raise AssertionError("input() must not be called without --write")

    monkeypatch.setattr("builtins.input", must_not_be_called)

    code = cli.main(["mcp", "--project-id", "prj_x", "--org-id", "org_x"])

    assert code == 0
    assert not (project_dir / ".cursor").exists()


def test_mcp_write_writes_files_after_confirmation(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("builtins.input", lambda *_a: "y")

    code = cli.main(["mcp", "--project-id", "prj_x", "--org-id", "org_x", "--write"])

    assert code == 0
    mcp_json_path = project_dir / ".cursor" / "mcp.json"
    rule_path = project_dir / ".cursor" / "rules" / "haki.mdc"
    assert json.loads(mcp_json_path.read_text()) == {
        "mcpServers": {"haki": {"url": "http://localhost:8100/mcp"}}
    }
    assert "alwaysApply: true" in rule_path.read_text()


def test_mcp_write_declined_writes_nothing(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("builtins.input", lambda *_a: "n")

    code = cli.main(["mcp", "--project-id", "prj_x", "--org-id", "org_x", "--write"])

    assert code == 0
    assert not (project_dir / ".cursor").exists()


def test_hooks_missing_scope_errors_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        cli.main(["hooks", "--subject-id", "usr_alice"])


def test_hooks_project_id_and_org_id_are_optional_with_saved_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"project_id": "prj_saved", "org_id": "org_saved"}))
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["hooks", "--subject-id", "usr_alice"])

    assert code == 0


def test_hooks_without_write_never_prompts_or_writes(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")

    def must_not_be_called(*_a, **_kw):
        raise AssertionError("input() must not be called without --write")

    monkeypatch.setattr("builtins.input", must_not_be_called)

    code = cli.main([
        "hooks", "--subject-id", "usr_alice",
        "--project-id", "prj_x", "--org-id", "org_x",
    ])

    assert code == 0
    assert not (project_dir / ".cursor").exists()


def test_hooks_write_writes_files_after_confirmation(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("builtins.input", lambda *_a: "y")

    code = cli.main([
        "hooks", "--subject-id", "usr_alice",
        "--project-id", "prj_x", "--org-id", "org_x", "--write",
    ])

    assert code == 0
    hooks_json_path = project_dir / ".cursor" / "hooks.json"
    session_rule_path = project_dir / ".cursor" / "rules" / "haki-session.mdc"
    hooks_config = json.loads(hooks_json_path.read_text())
    assert hooks_config["hooks"]["afterAgentResponse"][0]["command"] == (
        "haki hook-capture --subject-id usr_alice --project-id prj_x --org-id org_x"
    )
    assert "alwaysApply: true" in session_rule_path.read_text()


def test_hooks_write_declined_writes_nothing(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("builtins.input", lambda *_a: "")  # empty answer -> default N

    code = cli.main([
        "hooks", "--subject-id", "usr_alice",
        "--project-id", "prj_x", "--org-id", "org_x", "--write",
    ])

    assert code == 0
    assert not (project_dir / ".cursor").exists()


# -- `haki verify` scenario, fake client, no network ----------------------


class _FakeVerifyClient:
    """Stand-in for HakiClient exposing exactly the calls `_cmd_verify`
    makes. Records them so the tests can assert on the scenario's shape
    (which endpoint, which scope, which thread) rather than on a server's
    behavior -- the server side of the same scenario is covered by
    tests/test_sdk.py against the real app."""

    def __init__(self, *, served, superseded):
        self.captures: list[dict] = []
        self.scoped_consolidations: list[tuple[str, str]] = []
        self.unscoped_consolidations = 0
        self.context_calls: list[dict] = []
        self._served = served
        self._superseded = superseded
        self.closed = False

    def capture(self, events):
        self.captures.extend(events)
        return {"status": "accepted"}

    def consolidate(self):
        self.unscoped_consolidations += 1
        return {"processed": 1}

    def consolidate_subject(self, *, project_id, subject_id):
        self.scoped_consolidations.append((project_id, subject_id))
        return {"processed": 1}

    def facts(self, *, project_id, subject_id, status=None):
        if status == "superseded":
            return {"facts": list(self._superseded)}
        if status == "active":
            return {"facts": list(self._served)}
        return {"facts": [*self._superseded, *self._served]}

    def context(self, subject_id, query, project_id, **kwargs):
        self.context_calls.append({"subject_id": subject_id, **kwargs})
        return {"packet": {"facts": list(self._served)}, "trace_id": "trc_fake"}

    def close(self):
        self.closed = True


_CURRENT_FACT = {
    "predicate": "invoice_language",
    "value": {"language": "en"},
    "valid_from": "2026-08-11T10:00:00Z",
}
_REPLACED_FACT = {"predicate": "invoice_language", "value": {"language": "fr"}}


def _run_verify(monkeypatch, tmp_path, *, served, superseded):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"api_url": "http://fake.local", "api_key": "hk_fake"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "CONFIG_PATH", config)
    fake = _FakeVerifyClient(served=served, superseded=superseded)
    monkeypatch.setattr(cli, "HakiClient", lambda *a, **kw: fake)
    return cli.main(["verify"]), fake


def test_verify_never_calls_the_unscoped_consolidate_endpoint(monkeypatch, tmp_path):
    """Regression guard: POST /v1/consolidate drains EVERY project's pending
    jobs on a session without RLS. A customer command must never trigger
    (nor wait on) another tenant's consolidation -- it consolidates its own
    subject, by scope, twice."""
    code, fake = _run_verify(
        monkeypatch, tmp_path, served=[_CURRENT_FACT], superseded=[_REPLACED_FACT]
    )

    assert code == 0
    assert fake.unscoped_consolidations == 0
    assert len(fake.scoped_consolidations) == 2
    subject_ids = {subject for _, subject in fake.scoped_consolidations}
    assert len(subject_ids) == 1  # both calls scoped to the same, single subject
    assert all(project == cli._VERIFY_PROJECT for project, _ in fake.scoped_consolidations)
    assert fake.closed


def test_verify_captures_the_correction_in_the_same_thread(monkeypatch, tmp_path):
    """The scenario only demonstrates a supersession if the second message
    is a correction WITHIN the conversation -- a new thread would make it a
    plausible independent statement instead."""
    code, fake = _run_verify(
        monkeypatch, tmp_path, served=[_CURRENT_FACT], superseded=[_REPLACED_FACT]
    )

    assert code == 0
    assert len(fake.captures) == 2
    assert fake.captures[0]["thread_id"] == fake.captures[1]["thread_id"]
    contents = [event["payload"]["content"] for event in fake.captures]
    assert contents == [cli._VERIFY_MESSAGE_1, cli._VERIFY_MESSAGE_2]
    # ... and the context call reads from a DIFFERENT thread.
    assert fake.captures[0]["thread_id"] not in fake.context_calls[0]["purpose"]


def test_verify_reports_the_current_value_and_the_one_it_replaced(monkeypatch, tmp_path, capsys):
    code, _ = _run_verify(
        monkeypatch, tmp_path, served=[_CURRENT_FACT], superseded=[_REPLACED_FACT]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert 'recalled  invoice_language = {"language": "en"}' in out
    assert 'hidden    invoice_language = {"language": "fr"}' in out
    assert "superseded" in out
    assert "trace     trc_fake" in out
    assert out.rstrip().splitlines()[-1].startswith("OK — your agent remembered")


def test_verify_fails_when_the_stale_value_is_still_served(monkeypatch, tmp_path, capsys):
    """The failure this command exists to catch: the correction was captured
    and consolidated, but the packet still serves the superseded value."""
    code, _ = _run_verify(
        monkeypatch, tmp_path, served=[_REPLACED_FACT], superseded=[_REPLACED_FACT]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "FAIL: the update was not applied" in out
    assert '{"language": "fr"}' in out


def test_verify_fails_when_no_supersession_was_recorded(monkeypatch, tmp_path, capsys):
    """Serving the right value is not enough: if the old value is not on
    file as superseded, the update was stored as an unrelated new fact and
    nothing links the two."""
    code, _ = _run_verify(
        monkeypatch, tmp_path, served=[_CURRENT_FACT], superseded=[]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "not on file as superseded" in out


def test_verify_fails_when_nothing_is_recalled(monkeypatch, tmp_path, capsys):
    """An open conflict blocks BOTH values (context reason_code
    'conflict_open'), so an empty packet is a realistic failure, not a
    theoretical one -- it must not read as a crash."""
    code, _ = _run_verify(monkeypatch, tmp_path, served=[], superseded=[])
    out = capsys.readouterr().out

    assert code == 1
    assert "FAIL: nothing recalled about the preference" in out
    assert "trace     trc_fake" in out


def test_verify_falls_back_to_ascii_when_stdout_cannot_encode_the_tick(monkeypatch):
    """`haki verify` is routinely piped, and a redirected stream falls back
    to the locale encoding -- cp1252/ascii cannot encode the tick."""
    class _AsciiStdout:
        encoding = "ascii"

    monkeypatch.setattr(cli.sys, "stdout", _AsciiStdout())
    assert cli._stdout_mark() == "OK"


def test_login_opens_the_link_that_carries_the_code(monkeypatch, tmp_path):
    """RFC 8628 §3.3.1. The browser gets verification_uri_complete so the
    code is already in the box, while the printed instructions keep the
    plain URI and the code — typing it on a phone stays a supported path."""
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    fake = _FakeDeviceClient(poll_statuses=["approved"], expires_in=5, interval=0.02)
    monkeypatch.setattr(cli, "HakiClient", lambda *a, **kw: fake)

    assert cli.main(["login", "--api-url", "http://fake.local"]) == 0
    assert opened == ["http://localhost:3000/cli-auth?code=ABCD-EFGH"]


def test_login_falls_back_to_the_plain_uri_on_an_older_server(monkeypatch, tmp_path, capsys):
    """The SDK ships and versions independently of the API it talks to: a
    server predating verification_uri_complete must still log in, not crash
    on a missing key."""
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.json")
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))
    fake = _FakeDeviceClient(
        poll_statuses=["approved"], expires_in=5, interval=0.02, with_complete_uri=False
    )
    monkeypatch.setattr(cli, "HakiClient", lambda *a, **kw: fake)

    assert cli.main(["login", "--api-url", "http://fake.local"]) == 0
    assert opened == ["http://localhost:3000/cli-auth"]
    # The code itself is always printed, whichever URI was opened.
    assert "ABCD-EFGH" in capsys.readouterr().out
