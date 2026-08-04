"""Live verification of the sprint-10 predicate-stability fix (5K scenario).

Against the REAL stack (API + OpenRouter extraction): capture "personal
best 27:12" then "new personal best 25:50" in natural language, consolidate,
and check exactly ONE active fact remains (25:50, old one superseded) and
/v1/context serves the new value.

Usage: uv run python scripts/verify_5k_supersession.py
Requires the API on :8000 with HAKI_LLM_PROVIDER=openai and HAKI_ADMIN_KEY,
and HAKI_EVAL_ADMIN_KEY set to the same admin key.
"""

import asyncio
import json

from eval.haki_client import HakiClient, cleanup_project

PROJECT = "prj_verify_5k"
ORG = "org_eval"


async def main() -> int:
    haki = HakiClient("http://localhost:8000")
    assert await haki.health(), "API down"
    await cleanup_project(PROJECT)  # idempotent re-run
    key = await haki.create_project_key(ORG, PROJECT, label="verify 5k")

    events = [
        {
            "org_id": ORG,
            "project_id": PROJECT,
            "subject_type": "user",
            "subject_id": "usr_runner",
            "kind": "conversation.message",
            "occurred_at": "2023-05-23T10:00:00Z",
            "payload": {
                "messages": [
                    {"role": "user", "content": "I ran the charity 5K yesterday in 27:12 — that's my personal best!"},
                    {"role": "assistant", "content": "Great job!"},
                ]
            },
            "idempotency_key": "5k-evt-1",
        },
        {
            "org_id": ORG,
            "project_id": PROJECT,
            "subject_type": "user",
            "subject_id": "usr_runner",
            "kind": "conversation.message",
            "occurred_at": "2023-05-30T10:00:00Z",
            "payload": {
                "messages": [
                    {"role": "user", "content": "I did it! New personal best at the 5K today: 25:50!"},
                    {"role": "assistant", "content": "Congrats!"},
                ]
            },
            "idempotency_key": "5k-evt-2",
        },
    ]
    await haki.capture(key, [events[0]])
    await haki.consolidate_until_idle(key, PROJECT)
    await haki.capture(key, [events[1]])
    await haki.consolidate_until_idle(key, PROJECT)

    facts = await haki._client.get(
        "/v1/facts",
        params={"project_id": PROJECT, "subject_id": "usr_runner"},
        headers=haki._auth(key),
    )
    facts.raise_for_status()
    all_facts = facts.json()["facts"]
    print(json.dumps(
        [{"predicate": f["predicate"], "value": f["value"], "status": f["status"]} for f in all_facts],
        indent=2, ensure_ascii=False,
    ))
    active = [f for f in all_facts if f["status"] == "active"]
    superseded = [f for f in all_facts if f["status"] == "superseded"]

    body, _ = await haki.context(key, PROJECT, "usr_runner", "What was my personal best time in the charity 5K run?", 900)
    print("packet facts:", json.dumps(
        [{"predicate": f["predicate"], "value": f["value"]} for f in body["packet"]["facts"]],
        ensure_ascii=False,
    ))
    # Les FAITS servis ne doivent contenir que la valeur à jour. Les épisodes
    # (historique source daté) peuvent montrer l'ancienne valeur — c'est leur
    # rôle de provenance ; le tie-breaking temporel est côté prompt réponse.
    served_facts = json.dumps(body["packet"]["facts"], ensure_ascii=False)

    await haki.close()
    ok = (
        len(active) == 1
        and "25:50" in json.dumps(active[0]["value"])
        and len(superseded) == 1
        and "25:50" in served_facts
        and "27:12" not in served_facts
    )
    print("VERDICT:", "OK — un seul fait actif (25:50), ancien superseded, context sert 25:50" if ok else "ECHEC")
    await cleanup_project(PROJECT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
