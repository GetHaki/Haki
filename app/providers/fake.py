"""Deterministic provider for tests and local development without an API key.

`extract_facts` reads `event.payload["mock_facts"]` (a list of raw fact
dicts) when present, so tests fully control the extraction output; otherwise
it returns []. Raw dicts are passed through unvalidated on purpose: the
consolidator is responsible for Pydantic validation and rejection.

`embed` derives a deterministic EMBEDDING_DIM vector from the sha256 of the
text (digest bytes repeated, mapped to [-1, 1], L2-normalized). Same text
always yields the same vector, similar texts do NOT cluster — retrieval
tests must use identical strings for the query to match.
"""

import hashlib
import math
from typing import Any

from app.models import Event
from app.providers.base import EMBEDDING_DIM, RawCandidate


class FakeProvider:
    async def extract_facts(
        self,
        events: list[Event],
        existing: list[dict[str, Any]] | None = None,
    ) -> list[RawCandidate]:
        candidates: list[RawCandidate] = []
        for event in events:
            mock_facts = (event.payload or {}).get("mock_facts") or []
            candidates.extend(dict(fact) for fact in mock_facts)
        return candidates

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_embed_one(text) for text in texts]


def _embed_one(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 32-byte digest repeated to fill EMBEDDING_DIM floats.
    raw = [(b / 255.0) * 2.0 - 1.0 for b in (digest * (EMBEDDING_DIM // 32 + 1))]
    vector = raw[:EMBEDDING_DIM]
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def mock_fact(
    predicate: str,
    value: dict[str, Any],
    *,
    subject_id: str = "usr_42",
    action: str = "create",
    confidence: float = 0.9,
    supersedes_predicate: str | None = None,
    qualifiers: dict[str, Any] | None = None,
    evidence_span: str | None = None,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    """Helper to build a raw mock_facts entry (kept unvalidated on purpose).

    `evidence_span` and `reject_reason` back the write gate (M1): pass
    `action="reject"` with a `reject_reason` (see app.providers.base.
    REJECT_REASONS) to simulate a candidate the extractor screened out.
    """
    fact: dict[str, Any] = {
        "subject_id": subject_id,
        "predicate": predicate,
        "value": value,
        "qualifiers": qualifiers or {},
        "confidence": confidence,
        "action": action,
    }
    if supersedes_predicate is not None:
        fact["supersedes_predicate"] = supersedes_predicate
    if evidence_span is not None:
        fact["evidence_span"] = evidence_span
    if reject_reason is not None:
        fact["reject_reason"] = reject_reason
    return fact
