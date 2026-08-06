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

Phase 2 does **not** implement: rule-based detection, ML training, FastAPI
service, SOC dashboard, database persistence, MLflow, DVC, or deployment.

---

## Phase 4 status — Rule-based detection ✓ (v0.4.0)

Phase 4 adds the decision layer: deterministic, explainable rule-based
detection over Phase 3 feature snapshots, correlation-aware risk scoring, and
alert construction with suppression. **No machine-learning model is
implemented in Phase 4**, and no real authentication traffic is generated.

- **Nine registered detection rules** across six families, statically
  registered and versioned in an executable catalog
- **Two-phase rule contract** — prepare once per run, evaluate per snapshot
- **Trusted-template evidence** with structured codes, comparators, and units
- **Bounded monotone signal strength**, floored so a fired rule always
  outscores nothing firing
- **Correlation-aware risk scoring** — group reduction then noisy-OR, so one
  behaviour restated three ways cannot inflate a score
- **Four severity levels** with three strictly ordered boundaries; `LOW` is an
  ordinary alert severity
- **Alert grouping, deduplication, cooldown, rate limiting, and escalation**
  with a validator-enforced accounting identity
- **Optional entity-scope table** for pseudonymous alert grouping, consumed
  only during alert construction
- **Artifact validation** with 32 stable `D0xx` codes and sanitized findings
- **Aggregate quality reporting** in JSON and Markdown
- **Ground-truth evaluation** kept strictly outside the detection path
- **Staged publication with rollback**, content fingerprints, and a manifest
  that reuses the shared verification framework
- **Detection CLI** with 6 subcommands wired into the root CLI

### The detection layer in one paragraph

Rules consume **Phase 3 point-in-time feature snapshots and nothing else** —
the detection engine never queries raw event history, and labels and split
assignments never enter it. Each enabled rule is prepared once, then evaluated
against every snapshot in `(anchor_event_time, anchor_event_id)` order. Fired
rules become weighted contributions, reduced within their correlation group and
combined across groups into a `risk_score` in `[0, 100]`. Assessments that clear
two configured gates are grouped into alerts, deduplicated within a window, and
suppressed by cooldown and a per-group rate limit — with every suppressed event
counted. All three tables are validated, fingerprinted, and published together
with a manifest promoted last.

### The nine rules

| Rule | Name | Family | Category | Default severity |
|---|---|---|---|---|
| `PAD-BF-001` | Concentrated brute-force indicator | `brute_force` | `brute_force` | high |
| `PAD-BF-002` | Successful authentication after failure burst | `brute_force` | `brute_force` | high |
| `PAD-DBF-001` | Distributed brute-force indicator | `brute_force` | `distributed_brute_force` | critical |
| `PAD-PS-001` | Password-spraying indicator | `spraying` | `password_spraying` | high |
| `PAD-CS-001` | Credential-stuffing indicator | `stuffing` | `credential_stuffing` | high |
| `PAD-ATO-001` | Account-takeover indicator | `account_compromise` | `account_takeover_indicator` | critical |
| `PAD-MFA-001` | MFA sequence anomaly indicator | `account_compromise` | `mfa_sequence_anomaly` | medium |
| `PAD-GEO-001` | Impossible-travel indicator | `location` | `impossible_travel_indicator` | high |
| `PAD-BOT-001` | Bot-like authentication indicator | `automation` | `bot_activity` | medium |

Every rule ANDs multiple conditions and carries an explicit false-positive
control — a concentration ceiling, an attempts-per-account ceiling, a
per-source volume ceiling, or a required supporting signal. Full declared
contracts: [docs/rule-catalog.md](docs/rule-catalog.md).

### Evidence, signal strength, and risk score

**None of these is a probability.**

- **Evidence** records what was observed and which configured condition it
  matched. It is an indicator, **not causal proof**. Messages come from frozen
  catalog templates; proof-asserting and probability-asserting language is
  rejected at import.
- **`signal_strength`** is a bounded ordinal magnitude in `(0, 1]` describing
  how far one rule's observations exceeded its thresholds. **Not a
  probability.**
- **`risk_score`** is a bounded ordinal magnitude in `[0, 100]` ordering
  findings by accumulated evidence. **Not a probability.**
- `PAD-ATO-001` and `PAD-GEO-001` produce **indicators**, never findings of
  compromise.

### Correlation-aware risk scoring

```
contribution_r = family_weight[family(r)] × signal_strength(r)
c_g            = max(contribution_r for r in group g)
combined       = 1 − Π over sorted g of (1 − c_g)
risk_score     = max(round(100 × combined, 4), min_fired_risk_score)
```

Zero fired rules yields exactly `0.0` — a module constant, so a zero always
means "nothing fired". Correlated rules cannot out-score the strongest of them;
an unrelated signal can never lower risk; the result is order-invariant and
deterministic. See [docs/risk-scoring.md](docs/risk-scoring.md).

### Severity and the LOW alert band

Four levels, three strictly ordered boundaries, all inclusive from below
(`medium: 40.0`, `high: 65.0`, `critical: 85.0`).

**`LOW` is a valid alert severity.** Two independent configured gates decide
whether an assessment becomes an alert: `risk_score >= min_alert_risk_score`
and `severity >= min_alert_severity` (default `low`). Nothing rejects an alert
for being `LOW`. An operator who wants `LOW` findings to stay diagnostic raises
`min_alert_severity` — a configuration decision, recorded as
`low_alert_reachable` in the quality report.

### Alert grouping, scope, and suppression

Grouping key: `(attack_category, correlation_group, scope_kind, scope_value)`.

- **`category_scoped`** — the fallback, using category, correlation group, and
  the configured time window
- **`entity_scoped`** — when an optional entity-scope table supplies a
  pseudonym for the group's declared dimension

`aggregate_risk_score` is the **arithmetic mean** of an alert's grouped
qualifying assessments; `peak_risk_score` is their **maximum**.

The **entity-scope table remains sensitive operational metadata**. It is opt-in,
**consumed only during alert construction**, and reaches exactly one column of
one artifact — never evidence, a report, a CLI summary, a manifest, or a
validation message. `DetectionEngine` and `RiskScorer` accept no scope argument.

Cooldown suppresses repeats; a more severe or higher-peaking finding bypasses
it; a per-group rate limit backstops that bypass. **Suppression retains
complete aggregate accounting** — a validator enforces
`qualifying == grouped + cooldown-suppressed + rate-limited`. See
[docs/alert-lifecycle.md](docs/alert-lifecycle.md).

### Validation, quality reporting, and evaluation

`DetectionValidator` returns findings rather than raising, with 32 stable
`D0xx` codes. **No message carries an event, detection, or alert identifier, a
scope value, an evidence value, a raw row, or an absolute path** — findings are
codes, column names, and counts.

The quality report is aggregate-only and carries its own definitions of what
each number means. It also records **where its numbers came from**. A report
written by `detection run` is a live-run report with every counter populated,
including measured zeros. A report rebuilt by `detection profile` from
published artifacts marks the counters those artifacts cannot supply — rule
evaluations that did not fire, disabled rules, suppression decisions, gate
rejections — as **unavailable** (`null` in JSON), never as zero. Zero would
assert those events did not happen; the tables simply do not record them. Ground-truth evaluation is a separate workflow: it is the
only component permitted to read labels, splits, or campaign metadata, it tunes
no threshold, it reports the novel-anomaly holdout separately, and **synthetic
evaluation does not demonstrate real-world effectiveness**.

### Manifests and verification

The detection manifest is a superset of the Phase 2 dataset manifest, so the
single shared `verify_dataset` implementation verifies detection directories
too — path containment, `..` rejection, symlink escape, and checksums are
inherited, never reimplemented. `verify-manifest` adds content fingerprints,
artifact roles, cross-table relationships, and configuration/catalog agreement.

Phase 4 does **not** implement: machine-learning models, model evaluation,
FastAPI, SOC dashboard, database persistence, MLflow, DVC, streaming detection,
or deployment.

---

## Phase 3 status — Feature engineering and behavioral baselines ✓ (v0.3.0)

Phase 3 turns validated telemetry into a model-ready feature layer. It trains
no model, makes no detection decision, and produces no risk score.

- **Point-in-time feature engine**: one snapshot per authentication event,
  with every historical aggregate computed strictly from `[t - window, t)`
- **Same-timestamp mutual exclusion**: simultaneous events are invisible to
  each other, enforced structurally rather than by comparison
- **~200 declared features** across user, source, user-source pair, device,
  session, sequence, geospatial, calendar, and baseline groups
- **Versioned feature catalog** — the single source of truth for the schema,
  the Arrow types, the writer, and the validator
- **Behavioral baselines** with explicit fit/transform separation, fitted only
  from permission-checked training events
- **Chronological campaign-aware splitting** with purge and embargo
- **Leakage auditor**: twelve named checks, four of them behavioural
- **Separate feature, label, and split tables** — ground truth never sits
  beside a model input
- **Feature validation and aggregate quality reporting**
- **Manifests and fingerprints** for configuration, catalog, features,
  baseline, and split
- **Features CLI** with 9 subcommands wired into the root CLI

Phase 3 does **not** implement: rule-based detection, model training, model
evaluation, feature importance, SHAP, anomaly-detection models, a model
registry, FastAPI, Streamlit, databases, MLflow, DVC, or deployment.

### Temporal semantics in one paragraph

Detection runs **after** an authentication event has completed and been
recorded, so the anchor event's own fields (outcome, method, MFA result,
client type, response time, country) are legitimate `current_*` context. Every
*historical* aggregate, by contrast, uses only events in the half-open interval
`[t - window, t)`: the anchor never enters its own history, and neither do
events sharing its exact timestamp. Ordering is `event_time` ascending then
`event_id` ascending, where the `event_id` tie-break governs output row order
only and never affects state. The consequence — adding or modifying any event
after time `t` cannot change any feature at or before `t` — is asserted by
tests and re-checked by the leakage auditor. See
[docs/temporal-semantics.md](docs/temporal-semantics.md).

---

## Feature-layer architecture

```
src/password_attack_detector/features/
├── config.py         Typed, versioned configuration and fingerprints
├── catalog.py        The feature catalog: single source of schema truth
├── temporal.py       Timestamp blocks, rolling accumulators, calendar
├── engine.py         The point-in-time feature engine
├── baselines.py      Behavioral baselines (fit / transform)
├── geospatial.py     Haversine and coarse-location features
├── splitting.py      Chronological, campaign-aware splitting
├── leakage.py        The twelve-check leakage auditor
├── validation.py     Feature dataset validation (F0xx codes)
├── quality.py        Aggregate quality reporting
├── serialization.py  Arrow schemas, three writers, staged publication
├── manifest.py       Reproducibility manifest
└── cli.py            The `features` command group
```

The engine keeps one append-only buffer per entity with one head index per
window, so it is O(n·k) for k windows rather than O(n²). Sums and
sums-of-squares accumulate as exact integers, which makes output bit-for-bit
reproducible and lets the tests compare against a naive reference
implementation with no tolerance at all.

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
│   ├── features/
│   │   ├── feature-testing.yaml       CI-sized feature configuration
│   │   └── feature-development.yaml   full window ladder, strict isolation
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
├── data/                   raw, interim, processed datasets (not tracked)
├── artifacts/              training artifacts (not tracked)
├── docs/
│   ├── behavioral-baselines.md
│   ├── data-contract.md
│   ├── data-dictionary.md
│   ├── dataset-splitting.md
│   ├── feature-catalog.md      generated from the catalog
│   ├── feature-contract.md
│   ├── ingestion.md
│   ├── leakage-prevention.md
│   ├── privacy-model.md
│   ├── reproducibility.md
│   ├── synthetic-generation.md
│   └── temporal-semantics.md
├── scripts/
│   └── verify.sh
├── src/
│   └── password_attack_detector/
│       ├── cli.py               root Typer CLI
│       ├── config.py            typed settings (pydantic-settings)
│       ├── exceptions.py        project exception hierarchy
│       ├── logging_config.py    structured logging (structlog)
│       ├── paths.py             centralized path management
│       ├── data/
│       │   ├── cli.py           data command group (7 subcommands)
│       │   ├── enums.py         domain enumerations
│       │   ├── manifest.py      DatasetManifest + 10-check verification
│       │   ├── privacy.py       PseudonymService + prohibited-key scanner
│       │   ├── quality.py       QualityReport + JSON/Markdown renderer
│       │   ├── schemas.py       AuthEvent + GroundTruthLabel (Pydantic v2)
│       │   ├── serialization.py Parquet I/O + staged DatasetPublisher
│       │   ├── validation.py    DatasetValidator + ValidationResult
│       │   ├── ingestion/
│       │   │   ├── csv_adapter.py   CSV ingestion adapter
│       │   │   └── jsonl_adapter.py JSONL ingestion adapter
│       │   └── synthetic/
│       │       ├── config.py    SyntheticConfig (typed YAML model)
│       │       ├── campaigns.py attack campaign generators
│       │       ├── entities.py  entity generators
│       │       ├── generator.py generate_dataset() entry point
│       │       └── profiles.py  authentication behaviour profiles
│       └── features/
│           ├── cli.py           features command group (9 subcommands)
│           ├── config.py        FeatureConfig + fingerprints
│           ├── catalog.py       the versioned feature catalog
│           ├── temporal.py      timestamp blocks, rolling accumulators
│           ├── engine.py        the point-in-time feature engine
│           ├── baselines.py     behavioral baselines (fit / transform)
│           ├── geospatial.py    haversine, coarse-location features
│           ├── splitting.py     chronological campaign-aware splitting
│           ├── leakage.py       the twelve-check leakage auditor
│           ├── validation.py    FeatureValidator (F0xx codes)
│           ├── quality.py       FeatureQualityReport + renderers
│           ├── serialization.py Arrow schemas + FeaturePublisher
│           └── manifest.py      FeatureDatasetManifest
└── tests/
    ├── features/
    │   ├── factories.py         deterministic event factories and a DSL
    │   └── reference_engine.py  naive O(n^2) oracle for the engine
    ├── integration/
    │   ├── test_cli.py
    │   ├── test_data_cli.py
    │   └── test_features_cli.py end-to-end feature pipeline
    └── unit/
        ├── data/
        └── features/
            ├── test_baselines.py
            ├── test_catalog.py
            ├── test_config.py
            ├── test_engine.py     invariance + reference equivalence
            ├── test_geospatial.py
            ├── test_leakage.py
            ├── test_quality.py
            ├── test_serialization.py
            ├── test_splitting.py
            ├── test_temporal.py
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

### Phase 3 feature commands

```bash
# Inspect the declared feature schema
uv run password-attack-detector features catalog \
  --config configs/features/feature-development.yaml

# Full pipeline: split, fit baseline, transform, audit, validate, publish
uv run password-attack-detector features build \
  data/interim/authentication_events.parquet \
  --labels data/interim/synthetic_ground_truth.parquet \
  --config configs/features/feature-development.yaml \
  --output-dir data/processed \
  --reports-dir reports

# Individual stages
uv run password-attack-detector features split EVENTS --labels LABELS -o DIR
uv run password-attack-detector features fit-baseline EVENTS -o DIR --split-table T
uv run password-attack-detector features transform EVENTS -o DIR --baseline DIR

# Gates and reports
uv run password-attack-detector features audit-leakage EVENTS --labels LABELS \
  --splits data/processed/feature_splits.parquet
uv run password-attack-detector features validate \
  data/processed/feature_snapshots.parquet
uv run password-attack-detector features profile EVENTS -o reports
uv run password-attack-detector features verify-manifest data/processed
```

Every command supports `--help`, returns non-zero on failure, refuses to
overwrite without `--force`, and prints no event identifier, pseudonym,
coordinate, or absolute path.

### Phase 4 detection commands

```bash
# Inspect the versioned rule catalog
uv run password-attack-detector detection catalog
uv run password-attack-detector detection catalog --format markdown -o docs/rule-catalog.md

# Full pipeline: detect, score, group, validate, publish
uv run password-attack-detector detection run \
  --features data/processed/feature_snapshots.parquet \
  --feature-manifest data/processed/feature_manifest.json \
  --config configs/detection/rules-development.yaml

# Optional pseudonymous alert grouping
uv run password-attack-detector detection run \
  --features data/processed/feature_snapshots.parquet \
  --config configs/detection/rules-development.yaml \
  --entity-scope data/processed/detection_entity_scope.parquet

# Validate published artifacts
uv run password-attack-detector detection validate \
  data/processed/rule_detections.parquet \
  --risk-assessments data/processed/risk_assessments.parquet \
  --alerts data/processed/security_alerts.parquet \
  --config configs/detection/rules-development.yaml

# Aggregate quality report (JSON and Markdown) rebuilt from published
# artifacts. Counters only a live run observes read "unavailable", not zero.
uv run password-attack-detector detection profile \
  data/processed/rule_detections.parquet \
  --risk-assessments data/processed/risk_assessments.parquet \
  --alerts data/processed/security_alerts.parquet

# Synthetic ground-truth evaluation -- the only workflow that reads labels
uv run password-attack-detector detection evaluate \
  --detections data/processed/rule_detections.parquet \
  --risk-assessments data/processed/risk_assessments.parquet \
  --alerts data/processed/security_alerts.parquet \
  --labels data/processed/feature_labels.parquet \
  --splits data/processed/feature_splits.parquet \
  --campaign-labels data/raw/labels.parquet

# Verify artifact integrity
uv run password-attack-detector detection verify-manifest data/processed
```

`detection run` has no `--labels` and no `--splits` option. The absence is the
enforcement: ground truth cannot reach the engine, the scorer, or the alert
builder. Every command refuses to overwrite without `--force` and returns
non-zero on any validation or publication failure.

### Generated artifact layout

Phase 2 dataset directory:

```
output-dir/
  events.parquet        canonical events (no GT columns)
  labels.parquet        ground-truth labels (synthetic only)
  events.jsonl          raw events as newline-delimited JSON
  manifest.json         checksums, fingerprint, reproducibility info
```

Phase 3 feature artifacts:

```
data/processed/
  feature_snapshots.parquet   model inputs only, no ground truth
  feature_labels.parquet      event_id, attack_class, malicious, eligibility
  feature_splits.parquet      event_id, split, exclusion_reason
  feature_manifest.json       checksums, fingerprints, validation, audit
  split_manifest.json         boundaries, purge, aggregate distributions

artifacts/baselines/<name>/
  baseline.json               metadata only, safe to read (0644)
  user_baselines.parquet      pseudonym-bearing (0600)
  source_baselines.parquet    pseudonym-bearing (0600)

reports/
  feature_quality.{json,md}   aggregate statistics only
  leakage_audit.{json,md}     twelve named checks
```

Phase 4 detection artifacts:

```
data/processed/
  rule_detections.parquet     fired rules only, with evidence and reason codes
  risk_assessments.parquet    one row per evaluated anchor, including zero-score
  security_alerts.parquet     grouped alerts; scope_value is the one protected column
  detection_manifest.json     checksums, content fingerprints, validation, roles
  detection_quality.{json,md} aggregate statistics only

reports/
  detection_quality.{json,md} aggregate statistics only
  rule_evaluation.{json,md}   synthetic ground-truth metrics
```

All generated content is git-ignored. `security_alerts.parquet` may carry
pseudonymous `scope_value` entries and requires protected storage for real
data.

### Measured throughput

Numbers from one local run on the Phase 2 development dataset (84,625 events,
168 hours, 201 features, `configs/features/feature-development.yaml`). They
describe **this machine on this dataset** and are recorded for capacity
planning, not as a performance claim:

| Stage | Wall clock |
|---|---|
| `features transform` (engine only) | ~49 s |
| `features build --skip-audit` | ~185 s |
| `features build` (with leakage audit) | ~387 s |

The audit roughly doubles a build because two of its checks are behavioural:
they recompute the whole feature table on a mutated stream and compare. That
cost is the point — it is what makes the timing contract verified rather than
asserted. For iteration on a large dataset, run `--skip-audit` and use
`features audit-leakage` separately as a gate.

Resulting split on that dataset: 42,309 train / 9,126 validation / 9,141 test /
3 novel-anomaly holdout / 24,046 purged (28.4%, the expected cost of a 24-hour
purge at two boundaries across 168 hours — see
[docs/dataset-splitting.md](docs/dataset-splitting.md)).

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
- Phases 1-3 do not implement: rule-based detection, ML training, model
  evaluation, FastAPI, SOC dashboard, database persistence, MLflow, DVC, or
  deployment. These are planned for future phases.
- Phase 3 computes features but trains no model. Nothing in this repository
  demonstrates detection effectiveness, and the synthetic scenarios exercise
  the feature set without establishing real-world behaviour.
- Baseline artifacts hold pseudonymous per-entity state. They are written only
  to git-ignored paths with restrictive permissions and must never be
  committed; real-data baselines require protected storage.
- **No ML model is implemented in Phase 4**, and **no real authentication
  traffic is generated**. Every decision comes from a reviewed rule with
  declared thresholds.
- **`signal_strength` and `risk_score` are not probabilities.** They are
  bounded ordinal magnitudes. **Evidence is not causal proof.**
- **Account-takeover and impossible-travel outputs are indicators.** Travel and
  a device replacement reproduce the first; a VPN or carrier gateway reproduces
  the second. Confirming either requires investigation this system does not do.
- **Synthetic evaluation does not prove real-world effectiveness.** Thresholds
  tuned on generated data reflect the generator's parameters.
- **The entity-scope table carries pseudonyms.** It is opt-in, consumed only
  during alert construction, and confined to one column of one artifact;
  real-data alert artifacts require protected storage.
- Alert suppression retains complete aggregate accounting, but a suppressed
  event is only guaranteed to have been *counted* — not to have been
  uninteresting. See [docs/detection-limitations.md](docs/detection-limitations.md).
- Detection quality is bounded by feature quality: a behaviour Phase 3 does not
  express is one no Phase 4 rule can detect.
- `detection profile` reconstructs a quality report from published artifacts
  and therefore cannot report engine-only or alert-builder-only counters. Those
  are reported as **unavailable**, not zero. Use the report written by
  `detection run` for a fully populated one.

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
| [docs/feature-contract.md](docs/feature-contract.md) | Feature tables, null semantics, prohibited columns |
| [docs/feature-catalog.md](docs/feature-catalog.md) | Generated: every declared feature |
| [docs/temporal-semantics.md](docs/temporal-semantics.md) | The point-in-time contract |
| [docs/behavioral-baselines.md](docs/behavioral-baselines.md) | Fit/transform separation, artifact privacy |
| [docs/leakage-prevention.md](docs/leakage-prevention.md) | The twelve leakage checks |
| [docs/dataset-splitting.md](docs/dataset-splitting.md) | Chronological, campaign-aware splits |
| [docs/rule-contract.md](docs/rule-contract.md) | What a rule may read, must return, and guarantees |
| [docs/rule-catalog.md](docs/rule-catalog.md) | Generated: every registered rule |
| [docs/risk-scoring.md](docs/risk-scoring.md) | Correlation-aware scoring and its proven properties |
| [docs/alert-lifecycle.md](docs/alert-lifecycle.md) | Grouping, scope, suppression, escalation |
| [docs/rule-evaluation.md](docs/rule-evaluation.md) | Metrics, split discipline, no threshold tuning |
| [docs/detection-limitations.md](docs/detection-limitations.md) | What Phase 4 does not do |

---

## Planned phases

| Phase | Topic |
|---|---|
| 1 | Engineering foundation ✓ |
| 2 | Data engineering and synthetic log generation ✓ |
| 3 | Feature engineering and behavioral baselines ✓ |
| 4 | Rule-based detection (brute-force, spraying, stuffing) ✓ |
| 5 | Machine learning models |
| 6 | FastAPI detection service |
| 7 | SOC dashboard |
| 8 | Persistence and event storage |
| 9 | Monitoring and alerting |
| 10 | MLOps and experiment tracking |
| 11 | Deployment |

---

## License

MIT — see [LICENSE](LICENSE).
