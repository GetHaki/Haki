# Haki — public evaluation harness (sprint 10)

The first **reproducible** benchmark harness for AI agent long-term
memory. Nobody publishes a neutral leaderboard in this space: every
number on the market is self-reported, with different models, judges and
budgets. Here, everything is pinned and verifiable:

- **pinned datasets**: LongMemEval_S and LoCoMo, sha256 verified before
  every run (a different file = the run is refused);
- **frozen protocol**: versioned config (`eval/configs/`) — reader model,
  judge model, temperature 0, ContextPacket budget, full-context baseline
  budget, prompt versions (`eval/prompts/`), prices used for cost;
- **honest baseline**: no competitor marketing numbers — the reference is
  a full-context pass **re-run here**, same model, same answer prompt,
  same judge;
- **metrics nobody else publishes**: contradiction leakage (an answer
  based on a superseded fact), abstention accuracy, context packet tokens
  vs. full-context, context latency p50/p95, estimated cost.

No pre-computed numbers are checked into this repository (see "Publishing
your own run" below for why) — the harness produces yours when you run
it.

## Protocol

For each question, two systems under the SAME protocol:

1. **haki**: history sessions ingested as events (`subject_id` = the
   question, `occurred_at` = the dataset's own dates) → consolidation
   (`POST /v1/consolidate`) → `POST /v1/context` (budget from the config)
   → LLM answer with the packet injected;
2. **baseline**: the entire history (or the most recent sessions that fit
   in `baseline_max_context_tokens`) in the prompt;
3. **judge**: LLM-as-judge (versioned prompt, temperature 0) → correct /
   incorrect / abstained + a "relies on outdated information" flag
   (contradiction leakage, knowledge-update questions).

Each run lives in its own dedicated project `prj_eval_<dataset>_<run_id>`,
cleaned up after the run (`--keep-data` to preserve it).

## Reproducing

```bash
# 0. Postgres + migrations (see the root README), dataset downloaded
uv run python -m eval.download eval/configs/longmemeval_s.json
uv run python -m eval.download eval/configs/locomo.json

# 1. Start the API with a real LLM (extraction) and a local admin key
HAKI_LLM_PROVIDER=openai HAKI_ADMIN_KEY=<local> uv run uvicorn app.main:app --port 8000

# 2. Quick subset (fast sanity check)
HAKI_EVAL_ADMIN_KEY=<local> uv run python -m eval.run \
  --config eval/configs/longmemeval_s.json --subset 15 \
  --types knowledge-update,temporal-reasoning
HAKI_EVAL_ADMIN_KEY=<local> uv run python -m eval.run \
  --config eval/configs/locomo.json --subset 12

# 3. Full run (cost/time: LongMemEval_S = 500 questions x ~40 LLM
#    extraction sessions -- expect several hours and a few USD)
HAKI_EVAL_ADMIN_KEY=<local> uv run python -m eval.run --config eval/configs/longmemeval_s.json
```

## Publishing your own run

Reports (full JSON per question + readable Markdown) are written to
`eval/results/<dataset>_<run_id>.{json,md}` — gitignored on purpose, so
this repository never carries a stale or cherry-picked number: run the
harness yourself, on the pinned dataset and frozen config, and the
numbers you get are yours to trust or challenge.

## Deterministic selection

`--subset N` takes the first N questions **in dataset order** (after
`--types`, an exact type filter). Same arguments => same questions,
always. LoCoMo note: questions are ordered by conversation, so a small
subset only covers the first few conversations.

## Known limitations

- Haki's extraction cost (server-side consolidation) is not measured by
  the API; it is **estimated** (1 LLM pass per session: history tokens in,
  ~5% out) and marked as such.
- The baseline is truncated to the most recent sessions if the history
  exceeds `baseline_max_context_tokens` (chars/4, documented per question
  in the JSON: `sessions_used`, `truncated`).
- The judge is the same model as the reader (gpt-4o-mini): the frozen
  choice for the V1 config; changing the judge means a new config.
