# Haki — harnais d'évaluation public (sprint 10)

Le premier harnais de benchmark **reproductible** pour mémoire long-terme
d'agents IA. Personne ne publie de leaderboard neutre : tous les scores du
marché sont auto-rapportés avec des modèles, juges et budgets différents.
Ici, tout est figé et vérifiable :

- **datasets épinglés** : LongMemEval_S et LoCoMo, sha256 vérifié avant
  chaque run (un fichier différent = run refusé) ;
- **protocole figé** : config versionnée (`eval/configs/`) — modèle de
  réponse, modèle juge, temperature 0, budget ContextPacket (900 tokens),
  budget full-context baseline, versions des prompts (`eval/prompts/`),
  prix utilisés pour le coût ;
- **baseline honnête** : pas de chiffres marketing concurrents — la
  référence est un full-context **ré-exécuté ici**, même modèle, même
  prompt de réponse, même juge ;
- **métriques que personne ne publie** : contradiction leakage (réponse
  basée sur un fait supersédé), abstention accuracy, tokens du paquet de
  contexte vs full-context, latence context p50/p95, coût estimé.

## Protocole

Pour chaque question, deux systèmes dans le MÊME protocole :

1. **haki** : ingestion des sessions d'historique comme événements
   (`subject_id` = question, `occurred_at` = dates du dataset) →
   consolidation (`POST /v1/consolidate`) → `POST /v1/context`
   (budget 900 tokens) → réponse LLM avec le packet injecté ;
2. **baseline** : tout l'historique (ou les sessions les plus récentes qui
   tiennent dans `baseline_max_context_tokens`) dans le prompt ;
3. **juge** : LLM-as-judge (prompt versionné, temperature 0) →
   correct / incorrect / abstained + drapeau « relies on outdated
   information » (contradiction leakage, questions knowledge-update).

Chaque run vit dans un projet dédié `prj_eval_<dataset>_<run_id>`, nettoyé
après le run (`--keep-data` pour conserver).

## Reproduire

```bash
# 0. Postgres + migrations (voir README racine), dataset téléchargé
uv run python -m eval.download eval/configs/longmemeval_s.json
uv run python -m eval.download eval/configs/locomo.json

# 1. Lancer l'API avec le vrai LLM (extraction) et une clé admin locale
HAKI_LLM_PROVIDER=openai HAKI_ADMIN_KEY=<local> uv run uvicorn app.main:app --port 8000

# 2. Sous-ensemble rapide (ce qui est publié dans eval/results/)
HAKI_EVAL_ADMIN_KEY=<local> uv run python -m eval.run \
  --config eval/configs/longmemeval_s.json --subset 15 \
  --types knowledge-update,temporal-reasoning
HAKI_EVAL_ADMIN_KEY=<local> uv run python -m eval.run \
  --config eval/configs/locomo.json --subset 12

# 3. Run complet (coût/temps : LongMemEval_S = 500 questions × ~40 sessions
#    d'extraction LLM — compter plusieurs heures et quelques USD)
HAKI_EVAL_ADMIN_KEY=<local> uv run python -m eval.run --config eval/configs/longmemeval_s.json
```

Les rapports (JSON complet par question + Markdown lisible) sont écrits
dans `eval/results/<dataset>_<run_id>.{json,md}`.

## Sélection déterministe

`--subset N` prend les N premières questions **dans l'ordre du dataset**
(après `--types`, filtre exact sur le type). Mêmes arguments ⇒ mêmes
questions, toujours. Note LoCoMo : les questions sont ordonnées par
conversation, un petit subset ne couvre donc que les premières
conversations.

## Limites assumées

- Le coût d'extraction Haki (consolidation côté serveur) n'est pas mesuré
  par l'API ; il est **estimé** (1 passe LLM par session : tokens de
  l'historique en entrée, ~5 % en sortie) et marqué comme tel.
- La baseline est tronquée aux sessions récentes si l'historique dépasse
  `baseline_max_context_tokens` (chars/4, documenté par question dans le
  JSON : `sessions_used`, `truncated`).
- Le juge est le même modèle que le répondeur (gpt-4o-mini) : c'est le
  choix figé de la config V1 ; changer de juge = nouvelle config.
