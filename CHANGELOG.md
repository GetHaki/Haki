# Changelog

All notable changes to Haki are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-08-17

### Added

- **Cross-encoder reranker** (opt-in, `HAKI_RERANK_ENABLED`): re-scores the
  top retrieval candidates for higher accuracy before packing the context.
  Measured effect on a real A/B test: **+33.4 points accuracy** on the
  cases it targets (`eval/`, same protocol before/after).
- **Temporal grounding**: a fact extracted from a relative time expression
  ("last week", "il y a trois jours") now carries an exact resolved ISO
  date range instead of losing that information at write time. Every
  rendered date also ships a precomputed, verified offset ("21 days
  before the question") so the reading model never has to do date
  arithmetic itself — an LLM given raw dates alone gets that arithmetic
  right only 13.5-16% of the time (Test-of-Time benchmark).
- **Pseudo-relevance feedback (PRF) expansion**: retrieval now also
  considers entity names that recur across the top-ranked candidates,
  closing gaps where the right memory shares no exact keyword with the
  query.
- Animated demo of `haki verify` in the README Quickstart.

### Changed

- Default context budget raised from 900 to 2000 tokens, backed by
  published accuracy-vs-budget curves (the gain flattens well before
  4000 tokens on a `gpt-4o-mini`-class reader; 900 was leaving real
  accuracy on the table for no latency benefit).
- README rewritten and fully translated to English; roadmap, test badge,
  and internal links corrected to match what is actually in this
  repository (a few pointed at private-only paths or stale claims).

### Fixed

- **Security** (external review): constant-time comparison for
  shared-secret checks, a missing rate limit on `/v1/context`, an
  unbounded request payload size on capture, and a missing row-level-
  security policy on `forget_receipts`. None were exploitable in
  practice at the time they were found, but all four are closed.
- **Security**: rate-limiting was declared on every relevant route but
  never actually activated in this repository's `app/main.py` (no
  middleware, no exception handler) — every rate-limited request was
  returning a non-standard error body instead of this API's usual
  `{"error": {...}}` shape. Wired in.
- **Extraction**: an out-of-enum `fact_kind`, `volatility`, or
  `memory_form` value from the extraction model — observed on a real
  `gpt-4o-mini` run — used to silently destroy the entire candidate fact
  instead of falling back to that field's own documented default. Found
  the same day it started biasing a real measurement.
- **Consolidation**: the automatic "conflict overflow" reclassification
  (3+ competing values under one identity, reclassified as independent
  occurrences) now flags the facts it activates as such instead of
  serving them as silently certain — a non-deterministic extraction
  could otherwise misclassify a genuinely scalar attribute this way.
- 6 additional correctness and robustness bugs found by an internal code
  review across the retrieval and consolidation code — see git history
  for the full list.
- `httpx` timeout raised from 60s to 180s on the extraction provider, for
  long conversations whose prompt (including `existing_facts`) grows
  large enough to occasionally exceed the old limit.
- The public benchmarks section no longer implies pre-published results
  live in this repository — the harness is real and reproducible, but no
  cherry-picked number is ever committed here; run it yourself.

## [0.1.3] - 2026-08-14

- SDK coverage for mechanism D (`as_of`, volatility that degrades
  instead of excluding a stale fact) and the calibration eval protocol.

## [0.1.2] - 2026-08-13

- SDK coverage for the temporal tie-break fix and contested-conflict
  serving introduced the same day.

## [0.1.1] - 2026-08-10

- Follow-up fix shortly after the initial release.

## [0.1.0] - 2026-08-10

- Initial public release: Python and TypeScript SDKs, `haki` CLI, MCP
  server, OpenAI-compatible gateway.
