# Contributing

## Environment setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repository-url>
cd ai-password-attack-detection-system
uv sync --all-groups
cp .env.example .env
uv run pre-commit install
```

Verify the setup:

```bash
uv run password-attack-detector doctor
```

## Development workflow

### Branches

- Use a descriptive branch name: `feat/add-brute-force-rule`, `fix/config-path-default`
- Base all branches on `main`

### Commit messages

Follow conventional commits:

```
feat: add brute-force detection rule
fix: correct default path resolution on Windows
docs: update configuration table in README
test: add edge-case coverage for empty YAML config
refactor: extract common validator into utils
```

### Before opening a pull request

Run the full verification suite and confirm every check passes:

```bash
bash scripts/verify.sh
```

This runs: lockfile check, ruff lint, ruff format check, mypy strict, pytest (≥90% coverage), and build.

## Testing

```bash
uv run pytest                          # all tests with coverage
uv run pytest tests/unit/              # unit tests only
uv run pytest tests/integration/       # integration tests only
uv run pytest -v --no-cov             # verbose, no threshold enforcement
```

Coverage threshold is 90%. New code must be accompanied by meaningful tests.

Feature-layer tests share two helpers rather than rebuilding fixtures:

- `tests/features/factories.py` — `make_event`, a compact `make_stream` DSL,
  and `make_labels`. Short logical names (`"u1"`) hash to schema-valid
  pseudonyms deterministically.
- `tests/features/reference_engine.py` — a naive O(n^2) reimplementation of
  every windowed and sequence feature. The production engine is checked against
  it for **exact** equality; if you change how an aggregate is computed, change
  both and keep the equality exact.

Tests that assert the point-in-time contract (anchor exclusion, same-timestamp
exclusion, future-mutation invariance) are the phase's core guarantee. Do not
weaken them to make new code pass.

## Adding a detection rule

Rules are **statically registered reviewed Python**. Adding one means editing
code and the catalog together — there is no plugin path, no discovery, and no
configuration-supplied import.

1. Add a `RuleSpec` to `detection/catalog.py` with a stable `rule_id`
   (`PAD-XX-NNN`), a semantic `rule_version`, declared `RuleParameter`
   thresholds with bounds, `EvidenceDefinition` templates, and honest
   `limitations`.
2. Implement the rule against `BaseRule` / `BasePreparedRule` in
   `detection/rules/`, reading only catalogued features.
3. Register it in `ALL_RULES` in `detection/rules/__init__.py`.
4. Add a firing fixture to `FIRING_ROWS` in `tests/unit/detection/rules/
   test_rule_matrix.py` — the shared matrix then applies automatically.
5. Add rule-specific threshold, boundary, and false-positive-control tests, and
   a cross-rule discrimination test if the new rule could be confused with an
   existing one.
6. Regenerate the catalog documentation:
   `uv run password-attack-detector detection catalog --format markdown -o docs/rule-catalog.md`

Non-negotiable for any rule:

- **Never accept logic from YAML.** Configuration supplies values for declared
  parameters and selects registered rules — nothing else. No `eval`, `exec`,
  or dynamic import.
- **Keep published rule IDs and versions stable.** A test pins the current set,
  so any change is a deliberate, reviewed diff.
- **Return `insufficient_data`, not `not_fired`, for unobserved history.** A
  null feature means unobserved; reporting a clean negative makes a
  false-positive rate meaningless.
- **AND multiple conditions** and carry an explicit false-positive control.
  A single weak threshold is not a rule.
- **Never claim proof.** Evidence describes what was observed and what it is
  consistent with. Proof-asserting and probability-asserting language is
  rejected at import.
- **Never read a label, split, campaign, entity scope, raw event, or model
  output.** Only `detection/evaluation.py` may read ground truth, and only
  `detection/alerts.py` may read entity scope.

## Linting and type checking

```bash
uv run ruff check . --fix              # lint with auto-fix
uv run ruff format .                   # format
uv run mypy src tests                  # strict type checking
```

## Secure contribution guidelines

- Never commit plaintext passwords, hashes, API keys, or credentials
- Never commit a `.env` file (it is in `.gitignore`)
- Never add offensive tooling, credential-cracking utilities, or code that
  automates authentication attempts against real systems
- Never add dependency groups that contain attack tooling
- Use `pydantic.SecretStr` for any new configuration fields that hold secrets
- Authentication event datasets (introduced in Phase 2+) must use synthetic or
  properly anonymised data

See [SECURITY.md](SECURITY.md) for the vulnerability reporting policy.
