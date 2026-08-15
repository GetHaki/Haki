"""Local embedder + reranker: fastembed (ONNX Runtime, CPU), no network in
the hot path.

Embedder model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384 dims, multilingual — Haki's primary usage is francophone). The model
(~100 MB) is downloaded once by fastembed into its local cache on first
load, then everything runs offline.

Reranker model (mechanism F-R, 15 aout): Xenova/ms-marco-MiniLM-L-6-v2, see
LocalReranker below.

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
import warnings
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

            # fastembed unconditionally warns on this model name, pointing
            # at fastembed==0.5.1 as the way back to its OLDER CLS-pooling
            # behaviour -- irrelevant here: fastembed is pinned to 0.8.0
            # (see pyproject.toml) and has been the only version this
            # project has ever used, so every stored embedding already
            # uses mean pooling. Following the warning's own suggestion
            # would be the actual regression.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"The model .* now uses mean pooling")
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


# Reranker (mechanism F-R, Sprint 2, 15 aout): a cross-encoder -- query and
# document run through the model TOGETHER, so it can attend to their
# interaction directly, unlike two independently-embedded vectors compared
# by cosine distance. The literature's single largest measured retrieval
# gain (SmartSearch ablation, +15.1pp, median gold rank 195 -> 8) --
# app.context applies it as a re-scoring pass over the top candidates
# already selected by the existing hybrid formula, not a replacement for
# it (a cross-encoder scores one query-document PAIR at a time, too slow
# to run over an entire scope; the hybrid formula's job is still to narrow
# a large scope down to a shortlist cheaply).
#
# Model: Xenova/ms-marco-MiniLM-L-6-v2 -- a standard MS MARCO-trained
# cross-encoder, not the exact model named in the plan (mxbai-rerank-base
# for eval / bge-m3 int8 for production): neither is in fastembed==0.8.0's
# supported list (pinned, see pyproject.toml) for THIS local ONNX runner.
# Picked for being small (~80 MB, vs. BAAI/bge-reranker-base's ~1 GB also
# offered here) and CPU-fast, consistent with this module's existing
# preference for a light multilingual embedder over a larger one --
# recalibrate/swap only if measured recall against the eval harness's own
# gold-served metric says otherwise, not on the plan's model name alone.
_RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


class LocalReranker:
    def __init__(self) -> None:
        self._model = None  # fastembed.rerank.cross_encoder.TextCrossEncoder, loaded lazily

    def _load(self) -> None:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=_RERANK_MODEL_NAME)

    def _rerank_blocking(self, query: str, documents: list[str]) -> list[float]:
        self._load()
        assert self._model is not None
        return [float(s) for s in self._model.rerank(query, documents)]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return await asyncio.to_thread(self._rerank_blocking, query, documents)
