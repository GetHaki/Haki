"""Calibration check for SEMANTIC_MATCH_MAX_DISTANCE (app/consolidator).

The Consolidator's semantic fallback (sprint-10 contradiction-leakage fix)
was unit-tested with FakeProvider, which forces embedding collisions
directly — those tests validate the SQL matching logic but say nothing
about whether the REAL local embedder (fastembed, paraphrase-multilingual-
MiniLM-L12-v2) actually places a reworded predicate/value close enough to
trigger the fallback in production.

This script embeds the exact named regression pairs from the sprint-10 eval
audit with the real local embedder and reports their cosine distance next
to the threshold, plus a set of genuinely unrelated pairs (including
lexically-similar-but-conceptually-different ones) as a false-positive
check.

Usage: uv run python scripts/check_semantic_threshold.py
"""

import asyncio
import math

import app.ledger  # noqa: F401 - import before app.consolidator to avoid a pre-existing circular import
from app.consolidator import SEMANTIC_MATCH_MAX_DISTANCE, _search_text
from app.providers.local import LocalEmbedder

SAME_CONCEPT_PAIRS = [
    ("bike_count", {"count": 3}, "bikes_owned", {"count": 4}),
    ("personal_best_5k", {"time": "27:12"}, "goal_personal_best_time", {"time": "25:50"}),
    ("yoga_class_frequency", {"frequency": "twice a week"}, "yoga_frequency", {"frequency": "three times a week"}),
    ("invoice_language", {"language": "fr"}, "invoicing_language", {"language": "en"}),
    ("favorite_color", {"color": "blue"}, "preferred_color", {"color": "green"}),
]

UNRELATED_PAIRS = [
    ("invoice_language", {"language": "fr"}, "favorite_color", {"color": "blue"}),
    ("personal_best_5k", {"time": "27:12"}, "invoice_language", {"language": "fr"}),
    ("bike_count", {"count": 3}, "car_count", {"count": 2}),  # lexically close, different entity
    ("bike_count", {"count": 3}, "book_count", {"count": 12}),  # ditto
    ("wedding_venue_preference", {"venue": "garden"}, "wedding_ceremony_style", {"style": "traditional"}),
]


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 1 - dot / (norm_a * norm_b)


async def main() -> None:
    embedder = LocalEmbedder()
    print(f"Threshold SEMANTIC_MATCH_MAX_DISTANCE = {SEMANTIC_MATCH_MAX_DISTANCE}\n")

    print("=== Paires du MEME concept (devraient matcher: distance <= seuil) ===")
    fails = 0
    for pred_a, val_a, pred_b, val_b in SAME_CONCEPT_PAIRS:
        text_a, text_b = _search_text(pred_a, val_a), _search_text(pred_b, val_b)
        [emb_a, emb_b] = await embedder.embed([text_a, text_b])
        dist = cosine_distance(emb_a, emb_b)
        ok = dist <= SEMANTIC_MATCH_MAX_DISTANCE
        fails += not ok
        print(f"  {'OK ' if ok else 'FAIL'} distance={dist:.4f}  {pred_a!r} <-> {pred_b!r}")

    print("\n=== Paires SANS RAPPORT (devraient rester separees: distance > seuil) ===")
    for pred_a, val_a, pred_b, val_b in UNRELATED_PAIRS:
        text_a, text_b = _search_text(pred_a, val_a), _search_text(pred_b, val_b)
        [emb_a, emb_b] = await embedder.embed([text_a, text_b])
        dist = cosine_distance(emb_a, emb_b)
        ok = dist > SEMANTIC_MATCH_MAX_DISTANCE
        fails += not ok
        print(f"  {'OK ' if ok else 'FAIL'} distance={dist:.4f}  {pred_a!r} <-> {pred_b!r}")

    print(f"\n{'TOUT PASSE' if fails == 0 else f'{fails} ECHEC(S)'} — seuil {'valide' if fails == 0 else 'a recalibrer'} empiriquement.")


if __name__ == "__main__":
    asyncio.run(main())
