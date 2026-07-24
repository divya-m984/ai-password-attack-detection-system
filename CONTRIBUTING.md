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

This runs: lockfile check, ruff lint, ruff format check, mypy, pytest, and build.

## Testing

```bash
uv run pytest                          # all tests with coverage
uv run pytest tests/unit/              # unit tests only
uv run pytest tests/integration/       # integration tests only
uv run pytest -v --no-cov             # verbose, no threshold enforcement
```

Coverage threshold is 85%. New code must be accompanied by meaningful tests.

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
