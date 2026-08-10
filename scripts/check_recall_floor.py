"""Calibration check for RECOMMENDED_RECALL_MAX_DISTANCE (app/context).

The M3 recall gate compares the QUERY embedding to each candidate's
embedding — natural-language question on one side, `_search_text(predicate,
value)` (facts) or `episode_text(kind, payload)` (episodes) on the other.
That axis was never calibrated: SEMANTIC_MATCH_MAX_DISTANCE (0.28) measures
fact<->fact paraphrases, whose distances are structurally smaller.

This script embeds realistic relevant pairs (a question a user would
actually ask about a stored fact, including lexical/identifier-style
queries) and off-topic pairs (including lexically-overlapping-but-
conceptually-unrelated ones, with French stopword overlap) with the real
local embedder, and reports each distance next to the threshold plus the
global margin. Run it BEFORE changing RECOMMENDED_RECALL_MAX_DISTANCE or
enabling HAKI_RECALL_MAX_DISTANCE in an environment.

Usage: uv run python scripts/check_recall_floor.py
"""

import asyncio

import app.ledger  # noqa: F401 - import before app.consolidator to avoid a pre-existing circular import
from app.consolidator import _search_text
from app.context import RECOMMENDED_RECALL_MAX_DISTANCE, episode_text
from app.providers.local import LocalEmbedder
from scripts.check_semantic_threshold import cosine_distance

# (query, predicate, value) — the query SHOULD recall the fact.
RELEVANT_FACT_PAIRS = [
    ("dans quelle langue dois-je facturer ce client ?", "invoice_language", {"language": "fr"}),
    ("what language should the invoice be in?", "invoice_language", {"language": "fr"}),
    ("combien de velos possede-t-il ?", "bike_count", {"count": 3}),
    ("quel est son record sur 5 km ?", "personal_best_5k", {"time": "27:12"}),
    ("what's their favorite color?", "favorite_color", {"color": "blue"}),
    ("a quelle frequence va-t-elle au yoga ?", "yoga_class_frequency", {"frequency": "twice a week"}),
    # identifier-style query (lexical path): must still pass the SEMANTIC gate
    ("preference invoice_language du client", "invoice_language", {"language": "fr"}),
]

# (query, predicate, value) — the query should NOT recall the fact.
OFF_TOPIC_FACT_PAIRS = [
    ("quelle est la meteo a Bamako demain ?", "invoice_language", {"language": "fr"}),
    ("recette de lasagnes maison pour six personnes", "personal_best_5k", {"time": "27:12"}),
    ("how do I reset my router password?", "favorite_color", {"color": "blue"}),
    # heavy French stopword overlap, zero conceptual overlap:
    ("quelle est la capitale de la Mongolie ?", "invoice_language", {"language": "fr"}),
    ("combien coute un billet pour Dakar ?", "yoga_class_frequency", {"frequency": "twice a week"}),
]

# Same exercise against episode texts (episodes pass the same gate).
RELEVANT_EPISODE_PAIRS = [
    ("que s'est-il passe avec la facture du client ?", "invoice.sent", {"invoice_id": "inv_42", "amount": 1200, "currency": "EUR"}),
    ("when did we last talk about the marathon?", "conversation.turn", {"messages": [{"role": "user", "content": "I finished the marathon in 4:05!"}]}),
]
OFF_TOPIC_EPISODE_PAIRS = [
    ("recette de lasagnes maison pour six personnes", "invoice.sent", {"invoice_id": "inv_42", "amount": 1200, "currency": "EUR"}),
    ("how do I reset my router password?", "conversation.turn", {"messages": [{"role": "user", "content": "I finished the marathon in 4:05!"}]}),
]


async def _check(embedder, title, pairs, text_of, should_pass, fails):
    print(title)
    worst = None
    for query, a, b in pairs:
        [q, t] = await embedder.embed([query, text_of(a, b)])
        dist = cosine_distance(q, t)
        ok = (dist <= RECOMMENDED_RECALL_MAX_DISTANCE) if should_pass else (dist > RECOMMENDED_RECALL_MAX_DISTANCE)
        fails.append(not ok)
        worst = dist if worst is None else (max(worst, dist) if should_pass else min(worst, dist))
        print(f"  {'OK ' if ok else 'FAIL'} distance={dist:.4f}  {query!r} <-> {a!r}")
    return worst


async def main() -> None:
    embedder = LocalEmbedder()
    print(f"Threshold RECOMMENDED_RECALL_MAX_DISTANCE = {RECOMMENDED_RECALL_MAX_DISTANCE}\n")
    fails: list[bool] = []
    max_rel = await _check(embedder, "=== Requetes PERTINENTES vs faits (distance <= seuil) ===", RELEVANT_FACT_PAIRS, _search_text, True, fails)
    min_off = await _check(embedder, "\n=== Requetes HORS-SUJET vs faits (distance > seuil) ===", OFF_TOPIC_FACT_PAIRS, _search_text, False, fails)
    max_rel_ep = await _check(embedder, "\n=== Requetes PERTINENTES vs episodes (distance <= seuil) ===", RELEVANT_EPISODE_PAIRS, episode_text, True, fails)
    min_off_ep = await _check(embedder, "\n=== Requetes HORS-SUJET vs episodes (distance > seuil) ===", OFF_TOPIC_EPISODE_PAIRS, episode_text, False, fails)

    n_fails = sum(fails)
    print(f"\nMarge faits: pertinent max={max_rel:.4f} / hors-sujet min={min_off:.4f}")
    print(f"Marge episodes: pertinent max={max_rel_ep:.4f} / hors-sujet min={min_off_ep:.4f}")
    print(f"{'TOUT PASSE' if n_fails == 0 else f'{n_fails} ECHEC(S)'} — seuil {'valide' if n_fails == 0 else 'a recalibrer'} empiriquement (milieu de marge conseille).")


if __name__ == "__main__":
    asyncio.run(main())
