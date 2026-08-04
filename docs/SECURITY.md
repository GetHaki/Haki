# Haki — Sécurité V1 (sprint 6)

Auth par clés API par projet, Policy Engine déterministe, Row-Level
Security PostgreSQL, API feedback, résolution de conflits, preuve
multilingue. Ce document fait autorité sur le modèle de sécurité V1 ;
le README racine couvre l'installation générale.

## Modèle d'auth : clés API par projet

- Une clé (`hk_<hex>`) est liée à **un** `org_id` + **un** `project_id`.
  Seul le **hash sha256** est stocké (table `api_keys`, migration 0006) ;
  la clé en clair n'est retournée **qu'à la création**. Le `prefix`
  (8 premiers caractères) sert aux affichages masqués.
- `HAKI_AUTH_REQUIRED=true` (**défaut**) : tout endpoint `/v1/*` (sauf
  gestion des clés) et `/gateway/v1/*` exige
  `Authorization: Bearer hk_...`. Clé absente,
  invalide ou révoquée → **401 `unauthorized`**.
- **Liaison de scope** : si le body ou la query contient un `project_id`
  différent de celui de la clé → **403 `forbidden_scope`**, message
  générique, aucun indice sur l'existence d'autres projets. Vérifié sur
  capture (project_id de chaque événement), context, feedback, forget,
  resolve (body) et timeline, inspect, conflicts (query).
- **`/gateway/v1/*` (sprint 7)** : même middleware, même clé. Le body
  chat-completions ne porte pas de `project_id` : le scope mémoire est
  **celui de la clé**, sans exception. La clé Haki n'est jamais transmise
  au provider LLM — l'appel upstream utilise uniquement les credentials
  `HAKI_LLM_*` côté serveur.
- `HAKI_AUTH_REQUIRED=false` = **mode dev ouvert**, documenté, jamais en
  production : warning explicite loggué au démarrage
  (`haki.main`). C'est le mode utilisé par la suite de tests historique
  (`tests/conftest.py` le force, les tests d'auth activent le mode
  requis via la fixture `auth_required`).

### Gestion des clés

| Endpoint | Description |
|---|---|
| `POST /v1/keys` | Création. Réponse `201` avec la clé en clair (unique affichage). |
| `GET /v1/keys` | Liste **masquée** (prefix, jamais la clé ni le hash). |
| `DELETE /v1/keys/{id}` | Révocation (`revoked_at`). Effet immédiat : 401 aux appels suivants. |

Règles d'accès (V1, volontairement simple) :

- **`HAKI_ADMIN_KEY` défini** → mode admin : toute la gestion des clés
  exige `Authorization: Bearer <HAKI_ADMIN_KEY>`.
- **Non défini** → **bootstrap documenté** : la **première** création est
  libre (table `api_keys` vide). Ensuite, une clé valide gère les clés de
  **son propre projet** (création liée à son scope, liste et révocation
  bornées — une clé d'un autre projet renvoie le même 404 `key_not_found`
  qu'un id inconnu, sans fuite).

CLI : `haki keys create --project-id P --org-id O [--label L] [--save]`,
`haki keys list`, `haki keys revoke <id>`. `haki connect --api-url URL
--api-key KEY` stocke la clé dans `~/.haki/config.json` ; `haki verify`
tente un bootstrap de clé si aucune n'est configurée. SDK :
`create_key`, `list_keys`, `revoke_key` (sync + async).

## Policy Engine V1 (`app/policy/`)

Module **déterministe** (pas de LLM), appelé AVANT l'action par capture,
context, forget et par le middleware d'auth. Trois règles en V1 (pas de
règles custom utilisateur — sprint ultérieur) :

1. **Scope présent** — `subject_id` non vide sur chaque événement capturé
   (`missing_scope`, cohérent avec le Ledger).
2. **Clé ↔ projet** — la liaison de scope ci-dessus (403 `forbidden_scope`).
3. **`purpose` recommandé sur context** — warning `missing_purpose` dans
   le packet (et la trace persistée), **pas** une erreur en V1.

Chaque décision deny/warn est journalisée en ligne JSON structurée
(`haki.policy`, `policy_decision {...}`) ; chaque oubli est audité
(US 42). Erreurs typées : `unauthorized`, `forbidden_scope`,
`missing_scope`, jamais de révélation cross-projet.

## Row-Level Security (migration 0006)

RLS activé + `FORCE ROW LEVEL SECURITY` sur `events`, `facts`,
`context_traces`, `conflict_sets`, policy `haki_project_isolation` :

```sql
NULLIF(current_setting('haki.project_id', true), '') IS NULL
OR project_id = current_setting('haki.project_id', true)
```

- La dependency `get_session` pose `SELECT set_config('haki.project_id',
  :pid, true)` (SET LOCAL, portée transaction) depuis la clé résolue par
  l'auth. **Garantie PRD** : une requête qui oublie le filtre
  `project_id` dans le code ne voit que les lignes du projet de la clé —
  prouvée par `tests/test_rls.py` (SELECT sans `.where`, INSERT
  cross-projet rejeté par le WITH CHECK).
- **Mode dev ouvert** (décision documentée) : pas de SET, GUC NULL →
  policy permissive. C'est aussi le mode du worker interne et du serveur
  MCP (project **et** subject fixés par config serveur — aucun des deux
  outils `haki_*` n'accepte de `subject_id` en paramètre, le modèle ne
  choisit jamais le scope).
- **`NULLIF(..., '')` est indispensable** : après un `SET LOCAL` annulé
  en fin de transaction, Postgres laisse le GUC custom à `''` (pas NULL),
  et les connexions mutualisées (pool) le réutilisent — sans ça, toute
  connexion ayant servi une requête authentifiée cachait TOUTES les
  lignes (bug trouvé en démo live, test de régression
  `test_rls_empty_string_setting_is_permissive`).
- **Deux rôles** (décision documentée) : les migrations tournent avec le
  rôle propriétaire `haki` (DDL) ; le **runtime** utilise `haki_app`
  (créé par la migration, mot de passe `haki` — credential de dev local à
  remplacer en déploiement), qui n'est NI superuser NI propriétaire. Un
  superuser contourne RLS même avec FORCE : sans ce rôle dédié la
  garantie serait factice. Config : `HAKI_DATABASE_URL` (runtime,
  `haki_app`) et `HAKI_MIGRATION_DATABASE_URL` (alembic, `haki`).
- `/v1/consolidate` reste un endpoint **dev/ops cross-projets** : il
  utilise une session sans contexte RLS (`get_session_ops`), documenté.

## `POST /v1/feedback`

Body `{project_id, trace_id? | fact_id? (exactement un), rating:
"useful"|"irrelevant"|"incorrect", comment?}`. Chaque observation est
stockée (`feedback`, migration 0006). `incorrect` sur un `fact_id` →
transition Ledger du fait vers **`disputed`** : le Context Assembler ne
le sert plus comme actif (filtre de statut). Un fait d'un autre projet →
même 404 `fact_not_found` qu'un id inconnu. Réponse `201 {status:
"recorded", feedback_id, fact_status?}`. SDK : `feedback(...)`.

## `POST /v1/conflicts/{conflict_id}/resolve`

Body `{project_id, keep_fact_id}`. Le fait gardé → `active` (depuis
candidate/disputed via le Ledger), les **autres** faits du set →
`superseded` avec `supersedes_id` pointant le gardé, le set → `resolved`
+ `resolved_at`. Ensuite `/v1/context` sert le fait gardé normalement
(plus de blocage `conflict_open`). Erreurs typées :
`conflict_not_found` (404, y compris mauvais projet — pas de fuite),
`conflict_already_resolved` (409), `fact_not_in_conflict` (422).
Note cycle de vie : `candidate → superseded` a été ajouté aux
transitions du Ledger (le perdant d'un conflit est typiquement encore
candidat). SDK : `resolve_conflict(...)`.

## Preuve multilingue

`scripts/check_multilingual.py` (serveur lancé avec
`HAKI_LLM_PROVIDER=openai`, embedder local
paraphrase-multilingual-MiniLM-L12-v2) : capture FR + EN + ES pour un
sujet, consolidation (extraction OpenRouter), requêtes croisées
(EN→FR, FR→ES, ES→EN), tableau PASS/FAIL — **3/3 PASS** au dernier run.
Les predicate/value extraits restent en anglais technique
(`invoice_language`, `{"language": "fr"}`) quelle que soit la langue
d'entrée : **voulu et documenté** — la langue d'entrée ne contraint pas
le schéma de faits, seule compte la fidélité sémantique. Le script
utilise `HAKI_API_KEY` si définie, sinon le bootstrap documenté.

## Ce que V1 ne fait pas (bornes honnêtes)

Pas de RBAC/rôles, pas d'OAuth, pas de BYOK, pas de règles policy
custom, pas de rate limiting — sprint entreprise ultérieur. L'auth MCP
reste le bearer de dev `HAKI_API_KEY` (sprint 4). n8n : credential
Header Auth déjà supportée, rien à changer.
