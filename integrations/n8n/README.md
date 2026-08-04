# Haki × n8n

Deux façons de donner une mémoire long-terme à un agent n8n, sans Qdrant/Supabase/Data Table à configurer.

La chaîne imposée (PRD, Flow 2) :

```text
Chat Trigger / Webhook → Haki Context → AI Agent → Haki Capture → Respond
```

**Haki Context toujours avant l'agent, Haki Capture toujours après.** Le `subject_id` est obligatoire et ne peut jamais être vide ni `default` — une mémoire durable sans identité stable est une mémoire dangereuse.

## Option 1 — Template natif (V1 beta, marche partout)

[`haki-persistent-support-agent.json`](./haki-persistent-support-agent.json) : workflow importable utilisant uniquement le nœud **HTTP Request** natif — aucune installation, y compris sur les instances qui ne peuvent pas installer de nœuds communautaires.

1. n8n → *Import from file* → le JSON.
2. Configurer les **3 seules choses** (détaillées dans les sticky notes du canvas) :
   - **Credential Haki (header)** — Header Auth `Authorization: Bearer <clé>` sur les deux nœuds HTTP (en dev local sans `HAKI_API_KEY`, passer l'authentification sur *None*) ;
   - **Credential LLM** — sur *OpenAI Chat Model* (base URL OpenRouter pré-remplie, modifiable) ;
   - **Mapping du subject** — le template lit `{{ $json.body.subject_id }}` ; adapter à sa source (sessionId, email, ID Telegram/WhatsApp…), jamais une constante partagée.
3. Activer le workflow, puis :

```bash
curl -X POST http://localhost:5678/webhook/haki-support-agent \
  -H 'content-type: application/json' \
  -d '{"subject_id": "usr_123", "message": "je préfère que mes réponses soient en français"}'
```

Le nœud **IF « Subject valide ? »** rejette (HTTP 400) tout appel sans `subject_id` stable.

## Option 2 — Package de nœuds communautaires (V1.1)

[`n8n-nodes-haki/`](./n8n-nodes-haki/) : nœuds visuels **Haki Context** / **Haki Capture** + credential **Haki API** (base URL + clé optionnelle). Validation du subject intégrée (erreur d'exécution lisible si vide/`default`), sortie `context_text` prête à injecter, idempotence dérivée du run/thread, `wait_consolidation` pour une mémoire rappelable immédiatement.

Installation et détails : voir le [README du package](./n8n-nodes-haki/README.md). n8n Cloud exige un nœud vérifié — soumission Creator Portal prévue post-beta.

## Vérification de bout en bout

- Harness Node hors n8n (`n8n-nodes-haki/test/`) : 7/7 contre la vraie API.
- Exécution réelle dans n8n Docker (`n8nio/n8n` 2.32.7, package monté, workflows importés par l'API REST, appels webhook, provider LLM OpenRouter) : préférence « français » capturée au premier message et **rappelée au second** — via le template natif (réponse LLM réelle : « Ta langue préférée est le français. ») comme via les nœuds communautaires ; rejet HTTP 400 sans subject ; événements `conversation.turn` visibles dans `/v1/timeline`. Le workflow de test des nœuds communautaires est versionné : [`haki-e2e-test-workflow.json`](./haki-e2e-test-workflow.json).

## Limites honnêtes

Le builder peut casser la chaîne (supprimer Haki Capture, brancher l'agent ailleurs) : n8n ne permet pas d'imposer le passage. La mesure de couverture (appels Context vs Capture observés) arrivera dans la console Haki — la règle est aujourd'hui garantie par le template et les validations des nœuds, pas par interception.
