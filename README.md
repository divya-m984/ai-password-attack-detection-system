# Password Attack Detector

A defensive machine-learning system for detecting suspicious authentication
behavior — brute-force attacks, password spraying, credential stuffing,
distributed attacks, account-takeover indicators, impossible travel, bot-like
patterns, and unknown anomalies.

This system is designed exclusively for **detection and defense**. It never
stores plaintext passwords, never cracks credentials, and never automates
authentication attempts.

---

## Problem statement

Authentication logs contain signals that distinguish legitimate users from
attackers, but the volume and velocity of modern login traffic makes manual
analysis impossible. This project applies rule-based heuristics and machine
learning to continuously classify authentication events and alert on suspicious
patterns in near-real-time.

---

## Long-term architecture

```
Authentication logs
        │
        ▼
  Data engineering  ──►  Feature store
        │
        ▼
  Rule-based detection  ──►  Alert stream
        │
        ▼
  ML models (anomaly / classification)
        │
        ▼
  FastAPI detection service
        │
        ▼
  SOC dashboard (Streamlit)
        │
        ▼
  MLOps / monitoring / deployment
```

---

## Phase 1 status — Engineering foundation ✓

Phase 1 establishes the project skeleton:

- Typed configuration system (`pydantic-settings`, YAML per environment)
- Structured logging (`structlog` — human-readable in dev, JSON in production)
- Centralized path management (`pathlib`)
- Custom exception hierarchy
- Typer CLI (`version`, `doctor`, `show-config`)
- Ruff linting + formatting, mypy strict type-checking
- pytest test suite (≥85 % coverage)
- Pre-commit hooks
- GitHub Actions CI
- Repository documentation and security policy

No authentication data, ML models, APIs, or dashboards are included yet.

---

## Repository layout

```
.
├── configs/            Environment-specific YAML defaults
├── data/               Raw, interim, processed, and external datasets
├── artifacts/          Training artifacts and checkpoints
├── models/             Serialized model files
├── reports/            Generated analysis and metrics
├── scripts/            Developer utility scripts
├── src/
│   └── password_attack_detector/
│       ├── cli.py          Typer CLI entry point
│       ├── config.py       Typed settings (pydantic-settings)
│       ├── exceptions.py   Project exception hierarchy
│       ├── logging_config.py  Structured logging setup
│       └── paths.py        Centralized path management
└── tests/
    ├── unit/
    └── integration/
```

---

## Environment setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
cp .env.example .env          # edit as needed
uv run pre-commit install
```

---

## Configuration

Settings are loaded from (highest to lowest priority):

1. `PAD_*` environment variables
2. `.env` file
3. `configs/{environment}.yaml`
4. Field defaults

| Variable | Default | Description |
|---|---|---|
| `PAD_ENVIRONMENT` | `development` | `development`, `testing`, or `production` |
| `PAD_DEBUG` | `false` | Enable debug mode |
| `PAD_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `PAD_RANDOM_SEED` | `42` | Global random seed |
| `PAD_DATA_DIR` | `<repo>/data` | Data directory |
| `PAD_ARTIFACTS_DIR` | `<repo>/artifacts` | Artifacts directory |
| `PAD_MODELS_DIR` | `<repo>/models` | Models directory |
| `PAD_REPORTS_DIR` | `<repo>/reports` | Reports directory |

---

## CLI commands

```bash
# Show package version
uv run password-attack-detector version

# Run system health checks
uv run password-attack-detector doctor

# Display current configuration (secrets are redacted)
uv run password-attack-detector show-config

# Module invocation
uv run python -m password_attack_detector --help
```

---

## Tests

```bash
uv run pytest                                   # run all tests with coverage
uv run pytest tests/unit/                       # unit tests only
uv run pytest tests/integration/                # integration tests only
uv run pytest -v --no-cov                       # verbose, no coverage
```

---

## Linting and formatting

```bash
uv run ruff check .                # lint
uv run ruff check . --fix          # lint with auto-fix
uv run ruff format .               # format
uv run ruff format --check .       # format check (CI)
```

---

## Type checking

```bash
uv run mypy src tests
```

---

## Full verification

```bash
bash scripts/verify.sh
```

---

## Security boundaries

- Plaintext passwords are never collected, stored, or logged.
- Password hashes are never stored or processed.
- The system never automates login attempts.
- The system never generates credential lists.
- Sensitive configuration values use `SecretStr` and are redacted in CLI output.
- The `.env` file is excluded from version control.

See [SECURITY.md](SECURITY.md) for vulnerability reporting and disclosure policy.

---

## Planned phases

| Phase | Topic |
|---|---|
| 1 | Engineering foundation (current) |
| 2 | Data engineering and synthetic log generation |
| 3 | Rule-based detection (brute-force, spraying, stuffing) |
| 4 | Machine learning models |
| 5 | FastAPI detection service |
| 6 | SOC dashboard |
| 7 | Persistence and event storage |
| 8 | Monitoring and alerting |
| 9 | MLOps and experiment tracking |
| 10 | Deployment |

---

## Limitations

Phase 1 contains no real or synthetic authentication data, no detection logic,
no trained models, and no API or dashboard. All of those are introduced in
later phases.

---

## License

MIT — see [LICENSE](LICENSE).
