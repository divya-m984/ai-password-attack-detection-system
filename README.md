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

Phase 1 established the project skeleton: typed configuration, structured
logging, path management, exception hierarchy, Typer CLI, Ruff + mypy, pytest,
pre-commit hooks, and GitHub Actions CI.

---

## Phase 2 status — Data foundation ✓ (v0.2.0)

Phase 2 adds a complete data engineering layer:

- **Canonical authentication-event schema** (`AuthEvent`, Pydantic v2, strict
  validation, extra fields forbidden)
- **Strict ground-truth separation**: labels live in `labels.parquet` and are
  joined only by `event_id` — never merged into the canonical event table
- **Nine synthetic attack scenarios**: normal, brute-force, password spraying,
  credential stuffing, distributed brute-force, account-takeover indicator,
  impossible travel, bot activity, novel-anomaly holdout
- **Deterministic generation**: same config + seed → same content fingerprint
  across runs on the same `uv.lock` environment
- **CSV and JSONL ingestion** with HMAC-SHA256 pseudonymization of source
  identifiers
- **Prohibited-field rejection**: passwords, tokens, credentials, and GT columns
  are rejected at the header/key level before any row is read
- **Parquet serialization** with stable column ordering and UTC timestamps
- **Dataset validation** with schema, null, duplicate, and enum checks
- **Quality reporting** in JSON and Markdown (aggregate statistics only, no
  raw identifiers)
- **Manifest creation and verification**: SHA-256 checksums, content fingerprint,
  reproducibility metadata, 10 integrity checks
- **Data CLI** with 7 subcommands wired into the root CLI

Phase 2 does **not** implement: rolling features, rule-based detection,
ML training, FastAPI service, SOC dashboard, database persistence, MLflow,
DVC, or deployment.

---

## Data-layer architecture

```
configs/data/synthetic-*.yaml
        │
        ▼ SyntheticConfig
  generate_dataset()
        │
        ├── events.parquet      canonical AuthEvent rows
        ├── labels.parquet      GroundTruthLabel rows (separate)
        ├── events.jsonl        raw events as newline-delimited JSON
        ├── quality-report.json aggregate statistics
        ├── quality-report.md   Markdown quality report
        └── manifest.json       SHA-256 checksums + content fingerprint

Real data (CSV / JSONL)
        │
        ▼ CSVIngestionAdapter / JSONLIngestionAdapter
  scan_prohibited_keys()       reject passwords, tokens, GT columns
  PseudonymService.pseudonymize()  HMAC-SHA256 source identifiers
  AuthEvent.model_validate()   strict row validation
        │
        ├── events.parquet
        ├── events.jsonl
        └── manifest.json
```

---

## Repository layout

```
.
├── configs/
│   ├── data/
│   │   ├── synthetic-testing.yaml     small dataset for CI
│   │   └── synthetic-development.yaml larger dataset for local use
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
├── data/                   raw, interim, processed datasets (not tracked)
├── artifacts/              training artifacts (not tracked)
├── docs/
│   ├── data-contract.md
│   ├── data-dictionary.md
│   ├── ingestion.md
│   ├── privacy-model.md
│   ├── reproducibility.md
│   └── synthetic-generation.md
├── scripts/
│   └── verify.sh
├── src/
│   └── password_attack_detector/
│       ├── cli.py               root Typer CLI
│       ├── config.py            typed settings (pydantic-settings)
│       ├── exceptions.py        project exception hierarchy
│       ├── logging_config.py    structured logging (structlog)
│       ├── paths.py             centralized path management
│       └── data/
│           ├── cli.py           data command group (7 subcommands)
│           ├── enums.py         domain enumerations
│           ├── manifest.py      DatasetManifest + 10-check verification
│           ├── privacy.py       PseudonymService + prohibited-key scanner
│           ├── quality.py       QualityReport + JSON/Markdown renderer
│           ├── schemas.py       AuthEvent + GroundTruthLabel (Pydantic v2)
│           ├── serialization.py Parquet I/O + staged DatasetPublisher
│           ├── validation.py    DatasetValidator + ValidationResult
│           ├── ingestion/
│           │   ├── csv_adapter.py   CSV ingestion adapter
│           │   └── jsonl_adapter.py JSONL ingestion adapter
│           └── synthetic/
│               ├── config.py    SyntheticConfig (typed YAML model)
│               ├── campaigns.py attack campaign generators
│               ├── entities.py  entity generators
│               ├── generator.py generate_dataset() entry point
│               └── profiles.py  authentication behaviour profiles
└── tests/
    ├── integration/
    │   ├── test_cli.py
    │   └── test_data_cli.py     15 end-to-end CLI scenarios
    └── unit/data/
        ├── synthetic/
        ├── test_ingestion.py
        ├── test_manifest.py
        ├── test_privacy.py
        ├── test_quality.py
        ├── test_schemas.py
        ├── test_serialization.py
        └── test_validation.py
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
| `PAD_PSEUDONYMIZATION_KEY` | (unset) | HMAC key for ingestion; env-only, never YAML |

---

## CLI commands

### Phase 1 commands

```bash
uv run password-attack-detector version
uv run password-attack-detector doctor
uv run password-attack-detector show-config
uv run python -m password_attack_detector --help
```

### Phase 2 data commands

```bash
# Print canonical AuthEvent schema
uv run password-attack-detector data schema
uv run password-attack-detector data schema --format json

# Generate synthetic dataset
uv run password-attack-detector data generate \
  configs/data/synthetic-testing.yaml \
  --output-dir data/synthetic/test-run

# Validate canonical Parquet
uv run password-attack-detector data validate \
  data/synthetic/test-run/events.parquet

# Generate quality report (stdout JSON or write files)
uv run password-attack-detector data profile \
  data/synthetic/test-run/events.parquet

uv run password-attack-detector data profile \
  data/synthetic/test-run/events.parquet \
  --gt-path data/synthetic/test-run/labels.parquet \
  --output-dir data/synthetic/test-run/

# Create manifest for existing dataset
uv run password-attack-detector data manifest \
  data/synthetic/test-run/

# Verify manifest integrity
uv run password-attack-detector data verify-manifest \
  data/synthetic/test-run/

# Ingest CSV or JSONL (requires PAD_PSEUDONYMIZATION_KEY)
uv run password-attack-detector data ingest events.csv \
  --output-dir data/ingested/2024-01

uv run password-attack-detector data ingest events.jsonl \
  --output-dir data/ingested/2024-01 \
  --policy quarantine
```

### Generated artifact layout

```
output-dir/
  events.parquet        canonical events (no GT columns)
  labels.parquet        ground-truth labels (synthetic only)
  events.jsonl          raw events as newline-delimited JSON
  manifest.json         checksums, fingerprint, reproducibility info
```

---

## Tests and verification

```bash
# Run all tests with coverage
uv run pytest

# Verbose, no coverage
uv run pytest -v --no-cov

# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# Full verification (mirrors CI)
bash scripts/verify.sh
```

---

## Linting, formatting, and type checking

```bash
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
uv run ruff format --check .
uv run mypy src tests
```

---

## Privacy and security constraints

- No plaintext passwords, hashes, tokens, cookies, or real credentials are
  stored at any layer.
- Source identifiers are pseudonymized via HMAC-SHA256 before storage.
  **Pseudonymization reduces exposure but does not guarantee anonymity.**
- Prohibited sensitive field names are rejected at the ingestion header/key
  level before any values are read.
- Ground-truth labels are always stored separately from canonical events.
- The `PAD_PSEUDONYMIZATION_KEY` is never stored in YAML files, manifests,
  logs, or exception messages.
- Synthetic data never calls `PseudonymService` and does not require the key.

See [docs/privacy-model.md](docs/privacy-model.md) and
[SECURITY.md](SECURITY.md) for full details.

---

## Known limitations

- **Synthetic data is not evidence of real-world model performance.** It uses
  simplified attack simulations that do not capture real distributional properties.
- **Novel-anomaly holdout** (`supervised_training_eligible=False`) is not an
  ordinary supervised training class. It represents unknown attack types and
  must not be used as a labelled class during model training.
- **Reproducibility is bounded by the committed `uv.lock` environment.** A
  different library version may produce different output for the same seed.
- Phase 2 does not implement: rolling features, rule-based detection, ML
  training, FastAPI, SOC dashboard, database persistence, MLflow, DVC, or
  deployment. These are planned for future phases.

---

## Documentation

| File | Contents |
|---|---|
| [docs/data-contract.md](docs/data-contract.md) | Canonical event schema and prohibited fields |
| [docs/data-dictionary.md](docs/data-dictionary.md) | Column-level descriptions for all artifacts |
| [docs/privacy-model.md](docs/privacy-model.md) | Pseudonymization, key management, limitations |
| [docs/synthetic-generation.md](docs/synthetic-generation.md) | Nine scenarios, determinism, limitations |
| [docs/ingestion.md](docs/ingestion.md) | CSV/JSONL ingestion, field mapping, policies |
| [docs/reproducibility.md](docs/reproducibility.md) | Fingerprinting, environment pinning |

---

## Planned phases

| Phase | Topic |
|---|---|
| 1 | Engineering foundation ✓ |
| 2 | Data engineering and synthetic log generation ✓ |
| 3 | Rule-based detection (brute-force, spraying, stuffing) |
| 4 | Machine learning models |
| 5 | FastAPI detection service |
| 6 | SOC dashboard |
| 7 | Persistence and event storage |
| 8 | Monitoring and alerting |
| 9 | MLOps and experiment tracking |
| 10 | Deployment |

---

## License

MIT — see [LICENSE](LICENSE).
