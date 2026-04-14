# Create PR

Perform a full review, test, and PR creation workflow for the current branch. Execute each step in order and stop if a step fails after reasonable fix attempts.

## Step 1: Review changes

- Run `git diff master...HEAD` to see all changes on this branch relative to master.
- Read every changed file and check for:
  - Python syntax errors
  - Logical errors (wrong variable names, incorrect control flow, off-by-one)
  - Mathematical errors (wrong formulas, incorrect array indexing, transposed dimensions)
  - Import errors or missing dependencies
- If issues are found, fix them before proceeding.

## Step 2: Tests

- Review existing tests in `tests/` to understand coverage.
- If new modules, functions, or classes were added/changed, write unit tests for them following the existing patterns in `tests/conftest.py` and `tests/test_*.py`.
- Run: `pytest -m "not integration and not slow" --cov=polyswyft --cov-report=term-missing -v`
- If any tests fail, diagnose and fix the code (or the test if the test is wrong). Re-run until green.

## Step 3: Update documentation

- Read `CLAUDE.md`. If the code changes affect the algorithm, package structure, or dev workflow, update it. Keep it short.
- Read `README.md`. If the public API, installation steps, or usage changed, update it. Do not rewrite sections that are unaffected.
- Do not create new documentation files unless explicitly asked.

## Step 4: Commit

- Run `git status` and `git log --oneline -5` to see current state and recent commit style.
- Stage all relevant changed files (do not stage `.env`, credentials, or large binary files).
- Write a conventional commit message (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Note: version bumping happens automatically on merge to master via `.github/workflows/version-bump.yml`. Do NOT manually bump versions. Just use the correct commit prefix:
  - `feat:` triggers minor bump
  - `fix:` triggers patch bump
  - `feat!:` or `BREAKING CHANGE` in body triggers major bump
- Commit the changes.

## Step 5: Create PR

- Push the current branch to origin with `-u`.
- Create a PR targeting `master` using `gh pr create` with:
  - A concise title (under 70 characters)
  - A body containing:
    - `## Summary` -- bullet points describing what changed and why
    - `## Test plan` -- checklist of how to verify the changes (e.g. run tests, check lint, manual verification steps)
- Print the PR URL when done.
