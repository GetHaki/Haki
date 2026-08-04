# n8n-nodes-haki

Nœuds n8n communautaires pour [Haki](https://github.com/BigOD2307/Haki) — la mémoire long-terme des agents IA.

- **Haki Context** — récupère le ContextPacket du sujet (faits vérifiés, warnings, `trace_id`). Toujours **avant** l'AI Agent.
- **Haki Capture** — enregistre le tour user/assistant. Toujours **après** l'AI Agent.

Aucun Qdrant, Supabase ou Data Table à configurer : les deux nœuds parlent à l'API Haki (`POST /v1/context`, `POST /v1/capture`, `POST /v1/consolidate`).

## Installation (n8n self-hosted)

### Via l'interface n8n

Settings → Community nodes → Install → `n8n-nodes-haki`.

### Via npm (self-hosted)

```bash
mkdir -p ~/.n8n/nodes
cd ~/.n8n/nodes
npm install n8n-nodes-haki
```

Puis redémarrer n8n. En Docker :

```bash
docker run -it --rm -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  --add-host host.docker.internal:host-gateway \
  n8nio/n8n
# puis dans le conteneur :
mkdir -p /home/node/.n8n/nodes && cd /home/node/.n8n/nodes && npm install n8n-nodes-haki
# redémarrer le conteneur pour charger les nœuds
```

Depuis les sources de ce dépôt :

```bash
cd integrations/n8n/n8n-nodes-haki
npm install && npm run build && npm pack
cd ~/.n8n/nodes && npm install <chemin>/n8n-nodes-haki-0.1.0.tgz
```

> **n8n Cloud** : les nœuds communautaires exigent un nœud **vérifié**. La soumission passe par le Creator Portal n8n (avec provenance GitHub Actions) — prévue post-beta, pas encore faite. En attendant, n8n Cloud : utiliser le [template natif HTTP Request](../haki-persistent-support-agent.json).

## Configuration

Une seule credential, **Haki API** :

| Champ | Défaut | Note |
|---|---|---|
| Base URL | `http://localhost:8100` | n8n en Docker + API sur l'hôte : `http://host.docker.internal:8100` |
| API Key | *(vide)* | Optionnelle en dev local ; obligatoire si `HAKI_API_KEY` est configurée côté serveur |

## Haki Context

| Champ | Requis | Défaut | Note |
|---|---|---|---|
| Project ID | oui | — | Projet Haki (jamais choisi par le modèle) |
| Subject ID | oui | — | Identifiant stable ; **erreur si vide ou `default`** |
| Query | oui | — | Le message courant (reranking des faits) |
| Budget Tokens | non | `900` | Budget du ContextPacket |
| Purpose | non | — | Type de tâche, consigné dans la trace |

Sortie : `context_text` (bloc `<haki_memory>` prêt à injecter dans le system prompt), `packet` (JSON complet : faits + warnings), `trace_id`, `token_count`.

## Haki Capture

| Champ | Requis | Défaut | Note |
|---|---|---|---|
| Project ID | oui | — | Même projet que Haki Context |
| Subject ID | oui | — | Reprendre `{{ $('Haki Context').item.json.subject_id }}` |
| User Message | oui | — | Message du tour |
| Assistant Message | oui | — | Réponse de l'agent |
| Thread ID / Run ID | non | — | Alimentent la clé d'idempotence |
| Wait Consolidation | non | `false` | Si actif : appelle aussi `POST /v1/consolidate` (mémoire rappelable immédiatement — dev/démo) |

**Idempotence** : la clé est dérivée de `run_id` (ou `thread_id`) + hash SHA-256 du contenu (`n8n-turn-<run|thread>-<hash16>`). Rejouer la même exécution ne duplique jamais le tour.

## Développement

```bash
npm install
npm run build          # tsc → dist/
npm test               # build + harness node:test contre la vraie API (HAKI_BASE_URL, défaut http://localhost:8100)
```

Le harness `test/run-tests.mjs` exécute les `execute()` des deux nœuds compilés avec un mock minimal d'`IExecuteFunctions`, contre une API Haki réelle : packet retourné, erreurs `NodeOperationError` sur subject invalide, capture visible dans `/v1/timeline`, idempotence au replay, consolidation synchrone.
