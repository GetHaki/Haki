## What changed and why

<!-- Describe the change and the problem it solves. Link a related issue if one exists. -->

## How it was tested

<!-- `uv run pytest` output, or the specific tests added/updated. New behavior should ship with a new test. -->

## Checklist

- [ ] `uv run pytest` passes locally
- [ ] `npx tsc --noEmit` and `npx eslint .` pass (if touching `sdk/typescript/`)
- [ ] New behavior has a test that would fail without this change
- [ ] No secrets, API keys, or credentials in the diff
