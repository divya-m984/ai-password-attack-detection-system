# CLAUDE.md — Technical repository instructions

## Project overview

Password Attack Detector is a defensive authentication anomaly detection system
built in Python 3.12. It detects brute-force attacks, password spraying,
credential stuffing, and similar suspicious authentication patterns.

The system is **defensive only**. It never stores plaintext passwords, never
cracks credentials, and never automates authentication attempts.

## Architecture

```
src/password_attack_detector/
├── __init__.py          Package version and metadata
├── __main__.py          Module entry point (python -m password_attack_detector)
├── cli.py               Typer CLI (version, doctor, show-config)
├── config.py            Typed settings via pydantic-settings
├── exceptions.py        Project exception hierarchy
├── logging_config.py    Structured logging (structlog)
├── paths.py             Centralized pathlib-based path management
└── utils/__init__.py    Shared utilities (Phase 1: empty)
```

Phase 2+ will add: data engineering, rule-based detection, ML models, FastAPI
service, SOC dashboard, persistence, monitoring, MLOps, and deployment.

## Standard commands

```bash
# Install all dependencies
uv sync --all-groups

# Run all quality checks (mirrors CI)
bash scripts/verify.sh

# Tests
uv run pytest
uv run pytest -v --no-cov             # verbose, no coverage enforcement

# Linting and formatting
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
uv run ruff format --check .          # CI check (no writes)

# Type checking
uv run mypy src tests

# CLI
uv run password-attack-detector version
uv run password-attack-detector doctor
uv run password-attack-detector show-config
uv run python -m password_attack_detector --help

# Lockfile
uv lock --check

# Build
uv build
```

## Package layout

- **Distribution name**: `password-attack-detector`
- **Import name**: `password_attack_detector`
- **Source layout**: `src/password_attack_detector/`
- **Build backend**: hatchling
- **Dev dependencies**: PEP 735 `[dependency-groups] dev`

## Configuration system

Settings are loaded via `load_settings(environment=None)` in `config.py`.

Priority (highest to lowest):
1. Constructor kwargs
2. `PAD_*` environment variables
3. `.env` file
4. `configs/{environment}.yaml`
5. Field defaults

Supported environments: `development`, `testing`, `production`.
Environment variable prefix: `PAD_`.

The YAML data is injected via a closure-based `_BoundSettings` subclass
created inside `load_settings()`. There is no shared mutable class state.

## Path management

`paths.py` provides:
- `find_repo_root(start=None)` — discovers repo root by walking up to `pyproject.toml`; caches result
- `reset_repo_root_cache()` — used by `autouse` test fixture to prevent cross-test pollution
- `ensure_dir(path)` — the only function that creates directories; explicit calls only
- `get_data_dir()`, `get_artifacts_dir()`, `get_models_dir()`, `get_reports_dir()`, `get_configs_dir()`

No directories are created on import.

## Logging

`logging_config.py` (not `logging.py`) provides:
- `setup_logging(log_level, environment, *, force=False)` — configures structlog
- `reset_logging()` — clears configuration flag (tests only)
- `get_logger(name)` — returns a `FilteringBoundLogger`

Development: human-readable `ConsoleRenderer`. Production: `JSONRenderer`.

## Testing conventions

- All tests use `uv run pytest`
- Coverage threshold: 85% (currently ~96%)
- Use `tmp_path` for temporary file system operations
- Use `monkeypatch` for environment variable overrides
- Patch `find_repo_root` via `monkeypatch.setattr` to inject `fake_repo_root`
- The `autouse` fixtures in `conftest.py` reset path cache and logging state automatically

## Code conventions

- Python 3.12 only — use modern syntax (union types, `match`, etc.)
- `from __future__ import annotations` at the top of every module
- Ruff selects: `E W F I N UP B SIM C4 PTH RUF` — do not bypass with broad `noqa`
- mypy `strict = true` — no broad `type: ignore`; narrow ignores require explanatory comment
- No `pip` commands — always use `uv`
- `uv run` prefix for all Python tool invocations

## Security constraints

- Never store or log plaintext passwords, hashes, or credential lists
- Never add offensive exploitation functionality
- Never automate authentication attempts against real services
- Sensitive config fields must use `pydantic.SecretStr`
- `.env` is in `.gitignore` and must never be committed
- The `show-config` command redacts `SecretStr` fields

## Dependency rules

- Add runtime deps to `[project] dependencies` in `pyproject.toml`
- Add dev/test deps to `[dependency-groups] dev`
- Never use `pip install`; always `uv add` or edit `pyproject.toml` then `uv sync`
- Phase 1 excludes: NumPy, pandas, scikit-learn, FastAPI, Streamlit, MLflow, DVC,
  database drivers, Docker

## Before reporting success

Run the complete verification suite and confirm all checks pass:

```bash
bash scripts/verify.sh
```

Update tests and documentation when changing observable behavior.
