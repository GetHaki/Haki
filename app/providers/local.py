"""Local embedder: fastembed (ONNX Runtime, CPU), no network in the hot path.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384 dims, multilingual — Haki's primary usage is francophone). The model
(~100 MB) is downloaded once by fastembed into its local cache on first
load, then everything runs offline.

Design notes:
- Lazy singleton: the ONNX session is loaded on first `embed` call, not at
  import time, so tests and CLI startup stay fast.
- `embed` is async but inference is CPU-bound and blocking: it runs in
  `asyncio.to_thread` so the event loop is never stalled.
- LRU cache (bounded dict, 1024 entries, key = text): context queries repeat
  a lot in practice ("resume this thread", system prompts...), so repeated
  queries cost ~0 ms after the first hit.
"""

import asyncio
from collections import OrderedDict

_CACHE_MAX_ENTRIES = 1024

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    def __init__(self) -> None:
        self._model = None  # fastembed.TextEmbedding, loaded lazily
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _load(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=_MODEL_NAME)

    def _embed_missing(self, texts: list[str]) -> dict[str, list[float]]:
        self._load()
        assert self._model is not None
        vectors = self._model.embed(list(texts))
        return {text: [float(x) for x in vector] for text, vector in zip(texts, vectors)}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Preserve order and duplicates; only unique misses hit the model.
        results: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in texts:
            if text in self._cache:
                self._cache.move_to_end(text)
                results[text] = self._cache[text]
            elif text not in results and text not in missing:
                missing.append(text)
        if missing:
            computed = await asyncio.to_thread(self._embed_missing, missing)
            for text, vector in computed.items():
                self._cache[text] = vector
                while len(self._cache) > _CACHE_MAX_ENTRIES:
                    self._cache.popitem(last=False)
                results[text] = vector
        return [results[text] for text in texts]
