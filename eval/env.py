"""Environment helpers for the eval harness.

The harness is standalone (no `app` import): it reads the repo `.env` for
the OpenAI-compatible LLM settings (HAKI_LLM_BASE_URL / HAKI_LLM_API_KEY /
HAKI_LLM_MODEL) exactly like the API does, without ever printing secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from .env; existing environment wins."""
    env_path = Path(path) if path else ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def llm_settings() -> dict[str, str]:
    load_dotenv()
    base_url = os.environ.get("HAKI_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("HAKI_LLM_API_KEY", "")
    model = os.environ.get("HAKI_LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError(
            "HAKI_LLM_API_KEY is not set (checked environment and .env) — "
            "the eval harness needs the same OpenAI-compatible endpoint as the API."
        )
    return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model}
