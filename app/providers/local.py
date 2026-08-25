"""Local embedder + reranker: fastembed (ONNX Runtime, CPU), no network in
the hot path.

Embedder model: configured by HAKI_EMBED_MODEL, chosen from MODELS below,
which carries the measurements behind each entry. The default is
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dims,
multilingual — Haki's primary usage is francophone). The model (~100 MB for
the default) is downloaded once by fastembed into its local cache on first
load, then everything runs offline.

A model is not a setting like any other: every stored vector was produced by
the one in use, and two models' vectors are not comparable. Changing it is a
migration with a backfill; get_embedder refuses at startup if the configured
model's dimension does not match the `vector(N)` columns.

Reranker model (mechanism F-R, 15 aout): Xenova/ms-marco-MiniLM-L-6-v2, see
LocalReranker below.

Design notes:
- Lazy singleton: the ONNX session is loaded on first `embed` call, not at
  import time, so tests and CLI startup stay fast.
- `embed` is async but inference is CPU-bound and blocking: it runs in
  `asyncio.to_thread` so the event loop is never stalled.
- LRU cache (bounded dict, 1024 entries, key = text): context queries repeat
  a lot in practice ("resume this thread", system prompts...), so repeated
  queries cost ~0 ms after the first hit. The key is the PREFIXED text, so a
  question and a passage with the same words never collide on an asymmetric
  model, and never split into two entries on a symmetric one.
- `embed` is the PASSAGE side, `embed_query` the QUERY side -- identical for
  a symmetric model, deliberately different for a retrieval-trained one.
"""

import asyncio
import warnings
from collections import OrderedDict
from dataclasses import dataclass

from app.config import settings

_CACHE_MAX_ENTRIES = 1024


@dataclass(frozen=True)
class EmbeddingModel:
    """The three things about a model that silently break if they are wrong.

    `dim` must match the `vector(N)` columns, or every INSERT fails in the
    middle of a consolidation job. The two prefixes must match what the model
    was TRAINED with: a retrieval-trained model is asymmetric -- it embeds a
    question and a passage differently on purpose -- and dropping the prefix
    costs real recall while raising no error at all (measured below).
    """

    dim: int
    query_prefix: str
    doc_prefix: str
    note: str


# Every model here was measured on two axes, because Haki's usage is
# francophone and its published benchmark is English (22 aout):
#
#   EN  = gold served on eval.retrieval_bench, LoCoMo, budget 3000 -- does
#         the packet hold the annotated evidence. Reported as
#         any / COMPLETE: "at least one annotated turn" and "all of them".
#         Only the second one predicts whether the question is answerable.
#   FR  = MRR on the 525 PIAF questions (French, native, CC-BY) whose gold
#         passage BM25 puts outside its top 10. Restricting to those is what
#         makes it a test of FRENCH rather than of proper nouns: BM25 alone
#         answers 65.7 % of full PIAF at rank 1, above every model here, and
#         Haki already has a lexical axis doing exactly that job. What is
#         left is the part only the embedder can do.
#
#   model                             EN any/complete  FR MRR  query p50  size
#   paraphrase-multilingual-MiniLM     88.5 / 75.2 %   0.303     6.3 ms  0.22 GB
#   paraphrase-multilingual-mpnet      86.6 / 77.1 %*  0.345    25.7 ms  1.00 GB
#   intfloat/multilingual-e5-large     92.7 / 81.1 %   0.523    96.0 ms  2.24 GB
#   snowflake-arctic-embed-s (EN only) 89.6 / -    %*  0.194    14.0 ms  0.13 GB
#   BAAI/bge-small-en-v1.5   (EN only) 87.4 / -    %*  0.150       -     0.07 GB
#
#   * measured on LoCoMo conversations 1-2 only (n=231); the two rows
#     without a star are the full 10 conversations (n=1536).
#
# Four things that table says, in the order they matter.
#
# SIZE IS NOT WHAT SEPARATES THEM. mpnet-base-v2 is 4.5x the old default,
# 768 dimensions against 384 -- and scores BELOW it on English (86.6 %
# against 88.3 % on the same n=231 slice). It has the same SYMMETRIC
# paraphrase objective. e5-large wins because it is trained for RETRIEVAL
# on (query, passage) pairs and applies a different prefix on each side,
# which this module has applied since 22 aout. Anyone tempted to reach for
# a bigger model should re-read this line first.
#
# THE e5 GAIN IS MEASURED PAIRED, not by the gap between two percentages.
# Same 1 525 questions through both models, McNemar exact on the ones whose
# outcome changed (eval/compare_dumps.py, 24 aout):
#   any       88.5 -> 92.7 %   82 won, 17 lost   p < 0.0001
#   complete  75.2 -> 81.1 %  117 won, 27 lost   p < 0.0001
# Every category improves on both metrics. The largest single move is
# multi-hop COMPLETE, 40.8 -> 50.7 % -- the axis on which packed context
# was losing to replaying the whole history.
#
# IT COSTS 15x ON THE HOT PATH. One query embedding goes from 6.3 ms to
# 96.0 ms p50, and build_context from ~50 ms to ~125 ms p50 / 181 ms p95
# (2-core container; the old default was measured at 6.3 ms both BEFORE and
# AFTER the e5 run, so this is not machine drift). The RATIO is what
# transfers to other hardware; the absolute numbers are not comparable
# across machines and should never be published as a badge.
#
# AND IT CHANGES THE SCALING SHAPE, which matters more than the ratio. At
# 6 ms the embedder was noise next to Postgres and the API was I/O-bound;
# at 96 ms of pure CPU per request it DOMINATES, and throughput per worker
# becomes a function of cores. Measured resident memory (VmRSS after a
# warm embed): 1 561 MB for e5-large against 683 MB for the old default,
# and PSS across three concurrent processes (1 529 MB private of 1 561)
# says onnxruntime copies the weights into private memory rather than
# sharing the mapping -- so every worker really does pay it. Four workers
# is ~6.2 GB against ~2.7 GB. The way out, when it is needed, is to move
# the embedder into its own process pool instead of one copy per API
# worker; nothing here assumes that has been done.
#
# THE ENGLISH-ONLY MODELS ARE A TRAP for this product: they buy about a
# point of English and give back a THIRD of the French.
#
# Changing the default is a MIGRATION, never a setting flip: see 0029, and
# scripts/backfill_embeddings.py for the re-embedding pass it requires.
MODELS: dict[str, EmbeddingModel] = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": EmbeddingModel(
        dim=384,
        query_prefix="",
        doc_prefix="",
        note="multilingual, symmetric paraphrase objective, 128-token window "
        "-- the default until migration 0029, kept for a rollback",
    ),
    "intfloat/multilingual-e5-large": EmbeddingModel(
        dim=1024,
        query_prefix="query: ",
        doc_prefix="passage: ",
        note="DEFAULT since 0029: multilingual, retrieval-trained, "
        "512-token window, 2.24 GB on disk / ~1.5 GB resident, 96 ms/query",
    ),
    "snowflake/snowflake-arctic-embed-s": EmbeddingModel(
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        note="ENGLISH ONLY -- French MRR 0.194 against 0.303 for the default",
    ),
}


class LocalEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self._name = model_name or settings.embed_model
        if self._name not in MODELS:
            raise RuntimeError(
                f"HAKI_EMBED_MODEL={self._name!r} is not a model this project "
                f"has measured. Known: {', '.join(sorted(MODELS))}. Adding one "
                "means measuring it on both axes first -- see the table in "
                "app/providers/local.py."
            )
        self._spec = MODELS[self._name]
        self._model = None  # fastembed.TextEmbedding, loaded lazily
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def spec(self) -> EmbeddingModel:
        return self._spec

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
                warnings.filterwarnings(
                    "ignore", message=r"The model .* now uses mean pooling"
                )
                self._model = TextEmbedding(
                    model_name=self._name,
                    # None means fastembed's default (system temp dir); see
                    # the setting for why a container must not use it.
                    cache_dir=settings.embed_cache_dir,
                )

    def _embed_missing(self, texts: list[str]) -> dict[str, list[float]]:
        self._load()
        assert self._model is not None
        vectors = self._model.embed(list(texts))
        return {text: [float(x) for x in vector] for text, vector in zip(texts, vectors)}

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as QUERIES, not as passages.

        A retrieval-trained model is asymmetric on purpose: it is trained on
        (question, passage) pairs with a different prefix on each side, so
        embedding a question the way a passage is embedded loses recall --
        and loses it SILENTLY, which is the failure mode this project keeps
        paying for. Measured with snowflake-arctic-embed-s on the retrieval
        bench: 89.6 % with its query prefix, 87.4 % without it, same model,
        same everything else. fastembed's own `query_embed` does not apply
        it (verified: cos(query_embed(q), embed(q)) == 1.0), so nothing
        applies it unless this does.

        For the default model both prefixes are empty and this is exactly
        `embed`, cache included -- as it should be: that model is symmetric.
        """
        return await self._embed(texts, self._spec.query_prefix)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, self._spec.doc_prefix)

    async def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        if prefix:
            texts = [prefix + text for text in texts]
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
