"""Minimal async client for any OpenAI-compatible chat endpoint.

Used for the two answer-generation calls (Haki and baseline) and for the
LLM judge. Token usage comes from the API response (`usage`); when the
provider omits it we fall back to the chars/4 estimate and mark it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from eval.datasets import estimate_tokens


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    usage_estimated: bool = False


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 300.0,
        max_retries: int = 8,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # 15 aout, calibration mem0 (Sprint 0): mem0's own judge call passes
        # response_format={"type": "json_object"} -- optional, off by default,
        # every existing caller is unaffected.
        if response_format is not None:
            payload["response_format"] = response_format
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError:
                # Transport failure (connect/timeout): retry like a 5xx.
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            estimated = prompt_tokens is None or completion_tokens is None
            if estimated:
                prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
                completion_tokens = estimate_tokens(content)
            return ChatResult(
                content=content.strip(),
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                usage_estimated=estimated,
            )
        raise RuntimeError("unreachable")  # pragma: no cover
