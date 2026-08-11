# Contributing to Haki

Thanks for considering a contribution. This document covers everything
needed to set up a local environment, run the test suite, and submit a
change.

## Development setup

Requirements: [uv](https://docs.astral.sh/uv/) and Docker.

```bash
git clone https://github.com/GetHaki/Haki.git
cd Haki

docker compose up -d          # PostgreSQL 16 + pgvector, Redis 7
uv sync                       # installs Python 3.12 if needed
uv run alembic upgrade head
uv run uvicorn app.main:app --port 8100
```

In a second terminal, confirm everything works:

```bash
uv run haki connect --api-url http://localhost:8100
uv run haki verify
```

### TypeScript SDK

```bash
cd sdk/typescript
npm install
npm run build
npm test
```

## Running the tests

```bash
uv run pytest                 # full Python suite, against real Postgres (no mocks)
cd sdk/typescript && npm test # TypeScript SDK suite
```

Tests verify behavioral guarantees, not implementation details — see the
"Tests" section of [README.md](README.md) for the list of invariants every
PR is expected to preserve (no fact ever traverses a project/subject
boundary, a superseded fact is never served as active, a network retry
never creates a duplicate, etc.). If you change behavior covered by an
existing test, update the test in the same PR rather than deleting it.

## Code style

- Python: no enforced formatter is bundled yet — match the existing style
  in the file you're editing (type hints throughout, docstrings that
  explain *why* a non-obvious decision was made, not what the code
  literally does).
- TypeScript: `npx tsc --noEmit` and `npx eslint .` must both be clean
  before opening a PR.
- Comments should explain non-obvious constraints or the reasoning behind
  a design decision — not restate what the code already says.

## Submitting a change

1. Fork the repository and create a branch from `main`.
2. Make your change, with tests that exercise the new behavior against a
   real database (see `tests/conftest.py` for the fixtures already
   available).
3. Run the full test suite locally — a PR with failing tests will not be
   reviewed until it's green.
4. Open a pull request describing what changed and why. Link any related
   issue.

## Scope of this repository

This repository is the self-hostable core of Haki: the API, the SDKs, MCP
and n8n integrations, and the public benchmark harness. It does not include
the hosted Cloud console or billing integration, which live in a separate
private repository — pull requests touching billing or Cloud-specific
provisioning are out of scope here.

## Reporting bugs and proposing features

Use [GitHub Issues](https://github.com/GetHaki/Haki/issues) with the
provided templates. For security vulnerabilities, see
[SECURITY.md](SECURITY.md) instead of opening a public issue.

## Questions

Open a [discussion](https://github.com/GetHaki/Haki/discussions) or an
issue — there's no separate chat channel yet.
