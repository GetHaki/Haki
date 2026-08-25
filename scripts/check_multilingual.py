"""Preuve multilingue (sprint 6) — contre la vraie stack.

Prérequis : API lancée avec `HAKI_LLM_PROVIDER=openai` (extraction via
OpenRouter, cle dans .env) et l'embedder local par defaut
(multilingual-e5-large, 1024 dims) :

    uv run alembic upgrade head
    HAKI_LLM_PROVIDER=openai uv run uvicorn app.main:app --port 8100
    uv run python scripts/check_multilingual.py

Scénario : 3 evenements captures dans 3 langues (francais, anglais,
espagnol) pour UN sujet -> consolidation -> requetes context CROISEES
(capture EN, question FR ; capture FR, question ES ; capture ES, question
EN). Le script verifie que le bon fait remonte a chaque fois et imprime un
tableau PASS/FAIL par paire de langues. Exit 0 si tout passe.

Note documentee : les predicate/value extraits peuvent rester en anglais
technique (ex. predicate "invoice_language", value {"language": "fr"}) —
c'est voulu : la LANGUE D'ENTREE ne contraint pas la langue du schema de
faits ; seule compte la fidelite semantique.

Auth : si le serveur exige une cle (HAKI_AUTH_REQUIRED=true, defaut), le
script utilise HAKI_API_KEY si definie, sinon tente le bootstrap documente
(premiere cle libre quand la table api_keys est vide).
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone

import httpx

BASE_URL = os.environ.get("HAKI_API_URL", "http://localhost:8100")
PROJECT_ID = "prj_multilang"
ORG_ID = "org_multilang"

# (langue, contenu capture) — preferences distinctes, un seul sujet.
CAPTURES = [
    ("fr", "Je préfère recevoir mes factures en français."),
    ("en", "My support plan is the premium tier."),
    ("es", "Prefiero que las reuniones sean por la mañana."),
]

# (langue capture, langue question, question, mots-cles attendus)
CHECKS = [
    ("en", "fr", "quel est mon forfait support ?", ["premium"]),
    ("fr", "es", "¿en qué idioma debo recibir las facturas?", ["français", "french", "francés", '"fr"']),
    ("es", "en", "when does the user prefer meetings?", ["morning", "mañana", "matin"]),
]


def make_client() -> httpx.Client:
    """Client HTTP, avec cle API si le serveur en exige une."""
    api_key = os.environ.get("HAKI_API_KEY")
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)
    if api_key:
        client.headers["Authorization"] = f"Bearer {api_key}"
        return client
    # Auth exigee ? Le bootstrap documente cree la premiere cle librement.
    probe = client.post(
        "/v1/context",
        json={"project_id": PROJECT_ID, "subject_id": "probe", "query": "probe"},
    )
    if probe.status_code == 401:
        created = client.post(
            "/v1/keys",
            json={"org_id": ORG_ID, "project_id": PROJECT_ID, "label": "multilingual check"},
        )
        if created.status_code != 201:
            print(
                "FAIL: le serveur exige une cle API et le bootstrap est ferme. "
                "Definir HAKI_API_KEY (voir docs/SECURITY.md)."
            )
            sys.exit(1)
        key = created.json()["key"]
        client.headers["Authorization"] = f"Bearer {key}"
        print(f"cle API bootstrap creee pour {PROJECT_ID} ({key[:8]}…)")
    return client


def capture(client: httpx.Client, subject_id: str, lang: str, content: str) -> None:
    response = client.post(
        "/v1/capture",
        json={
            "idempotency_key": f"multilang-{uuid.uuid4()}",
            "events": [
                {
                    "org_id": ORG_ID,
                    "project_id": PROJECT_ID,
                    "subject_type": "user",
                    "subject_id": subject_id,
                    "kind": "conversation.message",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "payload": {"role": "user", "content": content},
                }
            ],
        }
    )
    response.raise_for_status()
    print(f"  capture [{lang}] « {content} » -> 202")


def main() -> int:
    print(f"Haki multilingual check — {BASE_URL}")
    client = make_client()
    subject_id = f"usr_poly_{uuid.uuid4().hex[:8]}"
    print(f"sujet: {subject_id} | projet: {PROJECT_ID}")

    print("\n== 1. Capture (3 langues) ==")
    for lang, content in CAPTURES:
        capture(client, subject_id, lang, content)

    print("\n== 2. Consolidation (extraction OpenRouter) ==")
    result = client.post("/v1/consolidate")
    result.raise_for_status()
    print(f"  jobs traites: {result.json()['processed']}")

    print("\n== 3. Requetes croisees ==")
    rows: list[tuple[str, str, bool, str]] = []
    for captured_lang, query_lang, query, keywords in CHECKS:
        response = client.post(
            "/v1/context",
            json={
                "project_id": PROJECT_ID,
                "subject_id": subject_id,
                "query": query,
                "purpose": "multilingual-check",
            },
        )
        response.raise_for_status()
        facts = response.json()["packet"]["facts"]
        rendered = json.dumps(facts, ensure_ascii=False).lower()
        hit = any(keyword.lower() in rendered for keyword in keywords)
        served = ", ".join(
            f"{f['predicate']}={json.dumps(f['value'], ensure_ascii=False)}"
            for f in facts
        ) or "(aucun fait)"
        rows.append((captured_lang, query_lang, hit, served))

    print("\n== Resultat ==")
    print(f"{'capture':<8} {'question':<9} {'verdict':<7} faits servis")
    print("-" * 100)
    all_pass = True
    for captured_lang, query_lang, hit, served in rows:
        verdict = "PASS" if hit else "FAIL"
        all_pass = all_pass and hit
        print(f"{captured_lang:<8} {query_lang:<9} {verdict:<7} {served}")
    print("-" * 100)
    print("OK — toutes les paires passent" if all_pass else "ECHEC — voir lignes FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
