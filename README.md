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

## Phase 5 status — Machine-learning detection ✓ (v0.5.0)

Phase 5 adds a statistical detection layer over the Phase 3 feature snapshots,
kept **separate from the Phase 4 rule engine** rather than folded into it. The
separation is the point: it lets rule-only, model-only, and hybrid detection be
measured against each other on identical frozen splits, instead of one quietly
absorbing the other.

The layer is **offline and defensive**. Nothing in it serves a model, exposes an
endpoint, touches live authentication traffic, or handles a credential.

- **Executable model catalog** — five families, each declaring its own
  serializer, inference adapter, task support, and champion eligibility
- **Opt-in feature allowlist** — a catalog feature is not eligible until a
  reviewer admits it and accepts its declared leakage class
- **Train-only preprocessing and class weighting**, fitted on one split and
  carried everywhere else by fingerprint
- **Artifacts that are numbers, not objects** — JSON and arrays, no `pickle`,
  no `joblib`, no estimator reconstructed to score
- **Calibration on validation-A, operating point on validation-B** — and the
  word *probability* becomes available only after a calibrator is fitted and
  its calibration error measured
- **Append-only experiment ledger** with four record types and idempotent
  identical appends
- **Validation-only champion selection** — support-aware gates that report
  `inconclusive` rather than passing on an empty denominator, the mandatory
  M-000 comparison, and a frozen `champion.lock` carrying no metric at all
- **Batch prediction** under that frozen champion, with a label-free inference
  input, a sealed `PredictionManifest`, artifact validation, and an aggregate
  quality profile
- **A locked TEST evaluation read exactly once**, against a lineage frozen
  before it ran, with **exact** PR-AUC over distinct score levels
- **Validation-only fusion selection** over the whole declared candidate
  universe — `or_gate`, `and_gate`, `stacked` — with genuine out-of-fold
  meta-features, frozen before the reader that opens a test label can be called
- **Rule / model / hybrid comparison** on the same frozen split
- **Deterministic model attribution** — exact for logistic regression, the
  random forest, and the threshold baseline; typed unavailable otherwise
- **Frozen training reference profile and drift detection** against it, with
  feature drift and prediction drift kept apart
- **Generated governance** — [docs/model-card.md](docs/model-card.md) and
  [docs/phase5-acceptance.md](docs/phase5-acceptance.md), both derived from the
  contracts they describe
- **ML CLI** with 14 subcommands wired into the root CLI

### The label boundary

**Exactly one command opens the TEST ground truth: `ml evaluate`.** `ml predict`
has no `--labels` option, `ml validate` computes no accuracy, `ml profile`
reports the *distribution* of what a model said rather than whether it was
right, `ml explain` decomposes a model's own output, and `ml drift` compares two
populations. Across the whole project exactly two modules may open a
ground-truth table — `detection.evaluation` and `ml.dataset` — and an
import-graph test pins that set in both directions.

### What Phase 5 does not claim

Every figure this repository can produce was measured on **synthetic traffic
generated by this repository**. None of it is evidence about real
authentication systems. A frozen champion is a subject, not a result; structural
validity is not predictive quality; attribution is descriptive, not causal; and
drift is monitoring evidence, not model correctness. Nothing here retrains,
promotes, or rethresholds on any finding.

---

## Phase 6 status — Serving layer, analyst console, live replay, containers, deployment perimeter, Render adapter (Milestones 1–5B)

Phase 6 turns the finished engine into something runnable. Milestone 1 added the
HTTP serving layer; Milestone 2 added the SOC analyst console on top of it;
Milestone 3 added a safe synthetic live/replay demonstration that makes the
detector *watchable*; Milestone 4 packages the whole thing so one command starts
it; Milestone 5A prepares and verifies the perimeter a public deployment would
need. **Nothing is publicly deployed.** This project has no public URL, no
server, and no domain name.

```bash
docker compose up --build
```

| | |
| --- | --- |
| API | <http://localhost:8000> |
| Swagger | <http://localhost:8000/docs> |
| Dashboard | <http://localhost:8501> |

The API is an **adapter**. It computes no feature, re-derives no threshold,
re-weights no rule, and contains no second scoring implementation — every
quantity comes from the frozen Phase 3–5 code that owns it.

- **Application factory** (`create_app`) with a lifespan that resolves the
  runtime once; nothing loads at import
- **Fail-closed startup** — `build_runtime` never raises. A component that
  cannot be initialised is recorded with a stable reason code, readiness is
  false, and detection is refused. No silent fallback to a different model.
- **`GET /health`** — process liveness; reads nothing
- **`GET /ready`** — per-component readiness (feature contract, rule engine,
  model artifacts, ML champion, fusion) with sanitized reason codes, `503` when
  a required component is missing. The hybrid is required exactly when Phase 5
  froze one: a selected strategy that cannot be verified is `503`, while "nothing
  qualified on validation" stays `200` and is reported as a scientific outcome.
- **Serving bundle** (`deploy materialize`) — an offline, deterministic contract
  that makes a frozen `stacked` selection deployable. It reconstructs the fitted
  `StackedFusionState` from pre-TEST lineage only, recomputes its semantic
  fingerprint, refuses publication unless it equals the fingerprint Phase 5
  sealed, and publishes it with the complete model / preprocessor / calibrator /
  threshold / fusion lineage. Startup loads and verifies it; nothing fits at
  serving time.
- **Live inference is not a dataset split** — a scored request is
  `ServingScope.LIVE` (`live_serving`), never `MLSplit.TEST`. The frozen feature
  order, preprocessor, adapter, calibrator, and threshold are reused; the
  requirement to *be* a scientific split is not. One binary-decision
  implementation serves both paths.
- **`GET /version`** — package and every contract version; nothing host-specific
- **`POST /api/v1/detect`** and **`POST /api/v1/detect/batch`** — score a
  bounded, ordered **window** of authentication events for one anchor or many
- **`POST /api/v1/explain`** *(Milestone 2)* — decomposes the frozen model's
  decision for one anchor over the transformed columns it read. Uses Phase 5's
  own `local_contributions`, the scope-free primitive that takes no `MLSplit`, so
  a live row is attributed without claiming membership of any experimental
  population. Nothing is fitted, no operating point moves, and a family with no
  exact decomposition reports the attribution unavailable rather than an
  approximation.
- **`GET /api/v1/system/status`**, **`/api/v1/model/info`**, **`/api/v1/rules`**
  — public-safe layer, champion, and rule information
- **Three layers kept apart** — the rule verdict, the model verdict, and the
  frozen fusion verdict are separate typed objects. The rule layer's ordinal
  0–100 `risk_score` is never blended with the model's probability.
- **Stable error contract** — one envelope, sixteen codes, no traceback and no
  filesystem path in any response
- **Serving security** — strict schemas with `extra="forbid"`, a request-body
  ceiling enforced by the service itself, bounded event counts, no artifact
  path, model id, threshold, or fusion override reachable from a request, and no
  CORS policy until a concrete dashboard origin exists
- **Privacy** — credential material is refused under every spelling before any
  other validation; a supplied `source_ip` is pseudonymized on arrival and never
  returned; no entity pseudonym appears in any response
- **OpenAPI** — Swagger at `/docs`, tagged Health / Detection / System

### Serving package

```
src/password_attack_detector/api/
├── app.py            create_app(), lifespan, error handlers, body-size middleware
├── config.py         APISettings — locations and ceilings, never scientific identity
├── dependencies.py   how a route reaches the runtime; the readiness gate
├── errors.py         stable error codes and the single failure envelope
├── schemas.py        request and response contracts
├── services.py       the composition: events → features → rules → model → fusion
└── routes/
    ├── health.py     /health, /ready, /version
    ├── detection.py  /api/v1/detect, /api/v1/detect/batch
    ├── explain.py    /api/v1/explain
    ├── replay.py     /api/v1/demo/scenarios, /demo/runs, /demo/runs/{id}/timeline
    └── system.py     /api/v1/system/status, /model/info, /rules

src/password_attack_detector/replay/
├── enums.py          the closed vocabularies: scenario, state, pace
├── schemas.py        the replay wire contract, embedding AnchorDetection verbatim
├── scenarios.py      the reviewed, deterministic, credential-free catalog
├── store.py          bounded, process-local, non-persistent run storage
├── engine.py         the state machine and the pace; the detector is injected
└── service.py        the operations the API namespace is a shell over

src/password_attack_detector/deployment/
├── bundle.py         the sealed serving-bundle manifest; write and verify
├── materialize.py    reconstruct the frozen stacked state and check its digest
└── cli.py            deploy materialize, deploy inspect
```

### Why detection takes a window

Nearly every signal the rules and the model read is a windowed or sequence
quantity over an event's strictly-prior history. A single stateless event would
produce a snapshot whose history is empty — not "unknown", but *wrong*. So a
request carries an ordered batch of events plus the anchors it wants a verdict
for, and this service fabricates no history for a caller who supplies none.

### Running it locally

```bash
export PAD_API_ARTIFACT_ROOT=/absolute/path/to/artifacts
export PAD_API_ALLOWLIST_PATH=/absolute/path/to/allowlist.yaml
export PAD_API_FEATURE_CONFIG_PATH=/absolute/path/to/features.yaml
export PAD_API_ML_CONFIG_PATH=/absolute/path/to/configs/ml/model-development.yaml
export PAD_API_DETECTION_CONFIG_PATH=/absolute/path/to/rules.yaml

uv run uvicorn password_attack_detector.api.app:app --host 127.0.0.1 --port 8000
```

Before a champion has been frozen, the rule layer can be served alone with
`PAD_API_REQUIRE_ML_CHAMPION=false`. Then visit `/health`, `/ready`, and `/docs`.

If the frozen selection was `stacked`, publish the serving bundle once first —
`deploy materialize` against the pre-TEST artifacts, then `deploy inspect` to see
what is published. Details in [docs/api.md](docs/api.md) §2.

Full reference: **[docs/api.md](docs/api.md)**.

### Milestone 2 — the SOC analyst console

A dark, wide, analyst-oriented Streamlit console, and structurally **a client of
the serving API and nothing else**.

```
Browser → Streamlit console → DashboardAPIClient → FastAPI → Phase 3–5 engine
```

- **No detection capability is importable from it.** No dashboard module imports
  `ml`, `detection`, `features`, `deployment`, or `data`. There is no rule
  engine, preprocessor, model adapter, calibrator, threshold, fusion function, or
  serving-bundle reader in that process. A test walks every module's syntax tree
  and enforces it; the only project module shared is `exceptions`.
- **One door to the backend.** `api_client.py` is the only module importing
  `httpx`, with a bounded timeout, typed responses, no automatic retry on a
  detection, no redirect following, and no URL or traceback in any error it
  surfaces.
- **The wire contract is re-declared, not imported.** Importing `api.schemas`
  would have pulled the whole detection stack in transitively; a test asserts the
  client declares no field the service does not send.
- **No scientific setting exists.** `PROHIBITED_SETTING_NAMES` refuses a model
  id, threshold, fusion strategy, artifact root, or API key at import.
- **Ten views** — Overview, Detection Console, Live Replay, Authentication
  Events, Security Alerts, Attack Analytics, Rule vs ML vs Hybrid,
  Explainability, Drift Monitoring, System & Model.
- **Three safe synthetic templates** — normal activity, a brute-force-like
  failure burst, a spraying-like fan-out. Synthetic identities, RFC 5737
  documentation addresses, no credential material anywhere. Loading one fills the
  form; the analyst still presses submit, and the request still goes through the
  API.
- **Layers kept apart on screen** — rule, model, and hybrid get one column each.
  Nothing on the console blends the ordinal 0–100 `risk_score` with the model's
  probability, renames a decision score a probability, or re-derives a threshold.
- **Session-only history** — no alert store, no event database, no fabricated
  totals. With nothing submitted the pages say *"No detection activity in this
  dashboard session."* rather than showing a plausible number.
- **Offline is a designed state** — Streamlit still loads, the header shows API
  offline, pages show a fixed error state, submission is disabled, no traceback
  reaches the browser, no metadata is invented, and a retry control recovers when
  the service returns.
- **Credentials are refused before storage** — the session refuses a
  credential-shaped field name under the project's own normalisation, so one can
  never reach browser session state or the JSON preview, let alone the wire.

### Dashboard package

```
src/password_attack_detector/dashboard/
├── app.py            entrypoint: config, session, dispatch
├── config.py         DashboardSettings — location and presentation only
├── contracts.py      the wire shapes the console is prepared to read
├── api_client.py     the one door to the backend
├── state.py          what one browser session remembers
├── formatting.py     how values are rendered, and what they may be called
├── scenarios.py      safe synthetic templates and the console's vocabularies
├── theme.py          the stylesheet and the escaping HTML helpers
├── components/       header, status, metrics, alerts, charts, replay
└── views/            the ten views
```

`views/` rather than `pages/`: Streamlit treats a `pages/` directory beside the
entrypoint as an automatic multipage app, which would produce a second navigation
beside the real one.

### Local demo — two terminals

**Terminal 1 — the API:**

```bash
uv run uvicorn password_attack_detector.api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

**Terminal 2 — the console:**

```bash
uv run streamlit run \
  src/password_attack_detector/dashboard/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

| | |
| --- | --- |
| API | <http://127.0.0.1:8000> |
| Swagger | <http://127.0.0.1:8000/docs> |
| Dashboard | <http://127.0.0.1:8501> |

Configuration is four optional variables — `PAD_DASHBOARD_API_URL`,
`PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS`, `PAD_DASHBOARD_REFRESH_SECONDS`,
`PAD_DASHBOARD_PAGE_TITLE` — none of which can name a model, a threshold, or a
strategy.

Full reference: **[docs/dashboard.md](docs/dashboard.md)**.

### Milestone 3 — the live replay demonstration

A reviewed, deterministic synthetic scenario, emitted one event at a time into
the **existing** serving path, so an analyst can watch rules fire, the model
decide, and the frozen hybrid fuse the two, on a timeline, as it happens.

```
Scenario → replay engine → DetectionWindowRequest → detect_single() → timeline
                                                    (the same function
                                                     POST /api/v1/detect calls)
```

- **Nothing attacks anything.** No credential, no credential list, no external
  request, no login endpoint, no network scan. Every scenario is a fabrication —
  events that did not happen, involving entities that do not exist, replayed into
  a service the operator is running themselves. Import-time guards refuse a
  credential-shaped field name and any address outside the documentation ranges;
  an AST test refuses a network client anywhere in the package.
- **There is no second detection path.** The engine's detector is one call to
  `detect_single`, through the same request schema an HTTP body is validated by.
  A test reconstructs a replayed step's window, posts it to `POST /api/v1/detect`,
  and compares the two verdicts field by field.
- **Seven scenarios** — normal activity, brute force, password spraying,
  credential stuffing, account takeover, bot activity, and a mixed timeline. Each
  publishes the rules it fires, and an integration test asserts that list as an
  **equality** against the real frozen deployment.
- **Pace is presentation.** `instant`, `fast`, `normal`, `slow` map to bounded
  intervals and change wall-clock spacing only. The same scenario produces
  byte-identical verdicts at every pace — asserted against the real frozen
  stacked deployment, not a stub.
- **A strict state machine.** `created → running → completed | stopped | failed`,
  with every terminal state absorbing. A finished run does not restart; a second
  execution is a new run with its own identity. Content identity
  (`scenario_fingerprint`) and instance identity (`run_id`) are separate and
  documented.
- **Bounded, process-local, non-persistent storage.** Four active runs, 24
  retained, 256 records each, a 300-second run ceiling. Reaching a bound is a
  typed refusal (`429`) and **never** an eviction of a run somebody is watching.
  Restarting the API clears every run, and nothing presents it as a history.
- **Incremental polling.** `?after_sequence=` returns only what a client has not
  seen, bounded per page, with one `more_expected` flag that goes false when the
  run is terminal — which is when the console stops polling.
- **Replay is optional.** It is reported on `/api/v1/system/status` and as a
  non-required `/ready` component: a detection service does not become unready
  because a demonstration facility did not initialise.
- **A tenth console view** with start/stop/refresh, a live timeline, a cumulative
  layer-activity chart, and a run summary derived entirely from timeline records.
  Replay data appears on the Overview, Events, Alerts and Analytics views under
  its own heading or behind an explicit source selector — the two sources are
  never silently merged.

Two rules cannot be demonstrated, and this is stated rather than worked around:
`PAD-CS-001` and `PAD-ATO-001` gate on a fitted behavioural baseline that the
serving path does not load, so they report insufficient data on every live
request. No threshold was moved to make a demonstration look better.

Full reference: **[docs/live-replay.md](docs/live-replay.md)**.

### Milestone 4 — containers, and one command

Three services and one ordering: `prepare → api → dashboard`.

- **The image ships no trained model.** A fitted champion is an *output*, and an
  image carrying one would make the image the provenance of a scientific
  decision. Instead a one-shot `prepare` job runs the project's **real pipeline**
  — generate, build features, train, select, freeze, predict, detect, evaluate,
  materialize — into a named volume and exits. The API starts only after that job
  reports success, mounts the volume **read-only**, and verifies what it finds.
  Nothing is fitted at serving time, on the first run or any later one.
- **Deterministic across machines.** The container's champion scope key is
  byte-identical to a host run of the same script: same seeds, same tracked
  configurations, same commands, same answer.
- **The console holds nothing.** No volume, no artifact path, no `PAD_API_*`
  variable. It reaches the API at `http://api:8000` over the project network.
- **Hardened by default** — non-root (uid 10001), read-only root filesystems,
  all capabilities dropped, `no-new-privileges`, no host networking, no Docker
  socket, no bind mounts, and ports published to the host's loopback only.
- **No scientific control in the environment.** Every `PAD_API_*` variable in
  `compose.yaml` answers *where* or *how much*. Two tests enforce it, one of
  which refuses a variable name that merely *contains* `MODEL`, `THRESHOLD`,
  `FUSION`, `CHAMPION`, `SCORE`, `SECRET`, or `TOKEN`.

Full reference: **[docs/docker.md](docs/docker.md)**.

### Milestone 5A — the deployment perimeter

An **additive** overlay on the above. `docker compose up --build` is unchanged;
a server adds one file:

```bash
docker compose --env-file .env.deploy \
  -f compose.yaml -f compose.deploy.yaml up -d --build
```

```
internet → 80/443 → proxy (Caddy) → dashboard → api → frozen serving bundle
```

- **The application ports stop being published.** Not narrowed — removed.
  Compose *appends* sequences when it merges files, so an override cannot
  un-publish a port by restating a shorter list; `ports: !reset null` removes the
  key outright. A test reads the resolved `docker compose config` and asserts
  that only the proxy publishes anything, and that it publishes only 80 and 443.
- **The smallest useful public surface.** The default routing policy publishes
  the console and nothing else — the API is unreachable from the internet, and
  loses nothing by it, because the console's client runs server-side. A second
  reviewed policy adds Swagger and the read-only reports for a viva. Scoring and
  replay-control endpoints stay internal in **both**.
- **Two TLS modes.** `PAD_SITE_ADDRESS=:80` for an IP-only smoke test;
  a hostname for automatic HTTPS over ACME. No certificate or private key exists
  in this repository, and no hostname is invented in a tracked file.
- **Security headers audited, not assumed.** The CSP is `frame-ancestors 'none'`
  only: a `script-src` tight enough to be worth having stops Streamlit rendering,
  and one loose enough to work would have to permit `'unsafe-inline'` and
  `'unsafe-eval'` — a control that claims a protection it does not provide.
- **Bounded logs**, 10 MiB × 3 per container, and memory ceilings retuned from
  what Milestone 4 measured, for a 2 GiB machine.
- **A server bootstrap that refuses to run on a laptop**, never edits sshd, opens
  no port unless asked, and is honest that `docker` group membership is
  root-equivalent.
- **Verified locally, end to end.** The real topology was brought up with the
  proxy on an unprivileged loopback port and checked against fifteen points:
  8000 and 8501 refuse connections, the console and its websocket work through
  the proxy, all seven replay scenarios complete fused by the frozen `stacked`
  hybrid, the containers stay unprivileged and read-only, a restart preserves the
  champion and clears the replay history, and teardown leaves nothing.

Full reference: **[docs/deployment.md](docs/deployment.md)**.

### Milestone 5B — the Render free-tier adapter

A **second** deployment target, and it does not replace the first. The Compose
deployment above is untouched; this adapts the same application to a **Render
free web service**, where there is no persistent disk and no private network
between two free services.

```
internet → Render edge (TLS) → $PORT → caddy ─┬→ 127.0.0.1:8501  console
                                              └→ 127.0.0.1:8000  api → baked bundle
```

- **One service, not two.** Two free services would double instance-hour
  consumption against a shared allowance, give a viewer two independent cold
  starts, and — because free services get no private network — force the
  detection API to be *publicly exposed* so the console could call it. One
  container keeps the API on loopback, which is the same routing policy the VPS
  deployment enforces with a proxy.
- **The bundle is baked at build time, and that deviation is stated rather than
  glossed.** `Dockerfile` argues an image must not ship a trained model; Render
  Free removes the remedy that argument relies on, because there is no volume for
  a preparation job to write into and preparing at container start would mean
  fitting models on every cold start. So a discarded build stage runs the *same*
  tracked pipeline script over the *same* tracked configurations, verification
  fails the **build** rather than the deployment, and the result is copied
  root-owned into a runtime layer that cannot write it.
- **Proved identical, not asserted identical.** Every fingerprint the serving
  manifest carries — champion lock, model content, calibration, threshold,
  feature catalog, fusion selection, and the STACKED state
  `134f66ce…f96272a` — matches the Compose preparation's, and all three bundle
  payload files are byte-for-byte the same. A slow test runs both preparations
  and compares them.
- **A supervisor, not a shell.** `scripts/render_entrypoint.py` is PID 1,
  standard library only, and blocks in `select` on a signal self-pipe with no
  timeout — no busy loop, no systemd, no supervisord. It starts the proxy *last*,
  so nothing listens on the public port until both processes behind it are
  serving. Any child exiting stops the container; SIGTERM stops all three in
  reverse order and the API's lifespan completes.
- **`$PORT` is read, never guessed.** Missing, empty, non-numeric, zero and
  out-of-range values are all refused with exit code 2, and `10000` appears
  nowhere in the image, the supervisor, or the routing policy.
- **Measured under the real ceiling.** In a read-only container limited to
  512 MiB with no swap: **198.3 MiB peak** (38.7 %), **zero OOM events**, and a
  **92.2 s cold start at 0.1 CPU**. Every verdict was identical at 0.1 and
  0.5 CPU — throttling changes how long a demonstration takes, not what the
  detector decides.
- **Nothing persists, and nothing needs to.** The container runs with no volume
  and no bind mount, and a restart preserves the champion while clearing replay
  history — which the console already says it will.

Full reference: **[docs/render-deployment.md](docs/render-deployment.md)**.

### What Phase 6 does not claim yet

The service and the console run locally, in containers, have a verified public
perimeter, and have an adapter for a Render free service. **Nothing is deployed:
no server exists, and no Render service exists.** Neither is authenticated, and neither
is rate-limited: rate limiting was deliberately deferred rather than built on a
third-party proxy module that would replace a pinned official image with one this
project has to patch itself, and [docs/deployment.md](docs/deployment.md) §15
states that residual risk rather than implying it is covered. There is no
authentication anywhere in this system and none was added for a demonstration — a
half-built auth platform is a larger surface than the one it closes.

Of the nine rules, **four fire on live requests** (`PAD-BF-001`, `PAD-BF-002`,
`PAD-BOT-001`, `PAD-PS-001`), **three are live-serving but exercised by no replay
scenario** (`PAD-DBF-001`, `PAD-GEO-001`, `PAD-MFA-001` — they evaluate normally
and return clean negatives), and **two cannot fire at all** — see below.
[docs/deployment.md](docs/deployment.md) §16 gives the measured per-rule
outcomes over all 154 replay anchors.

Nothing is persisted: the console's history is one browser session, a replay
run lives in one API process's memory and is cleared by a restart, and there is
still no alert store, no event database, and no serving drift report. The replay
layer is synthetic only — there is no path for real traffic to enter one, and the
reviewed catalog is its whole input surface. A `stacked` deployment requires the
offline `deploy materialize` step to have been run against the frozen lineage;
in a container the `prepare` job does that, and outside one it is a command an
operator runs. Until it has, the hybrid is reported unavailable and readiness is
`503` rather than a strategy nobody selected being substituted.

**No figure the containerized demonstration reports is a performance claim.** Its
dataset is four hours of synthetic traffic, sized so the whole pipeline finishes
in about a minute; `configs/data/synthetic-ml-development.yaml` is the 30-day
configuration that exists for measurement, and it is deliberately not what a
container runs.

**`PAD-CS-001` and `PAD-ATO-001` remain undemonstrable, and this is a v0.6.0
release blocker rather than a fixed limitation.** Both gate on a fitted
behavioural baseline the serving path does not load. Milestone 4 audited adding
one to the serving bundle and **measured that it would not help**: a baseline
fitted from the deployment's own TRAIN split still leaves `user_in_baseline`
`False` and every `is_new_*_for_user` flag `None` for replay events, because the
catalog's identities are content-addressed pseudonyms no training population
contains. Doing it properly is also a bundle-schema change, not a packaging one.
No threshold was moved and no baseline was synthesised to make a demonstration
look better. See [docs/deployment.md](docs/deployment.md) §16,
[docs/docker.md](docs/docker.md) §14, [docs/api.md](docs/api.md) §12,
[docs/dashboard.md](docs/dashboard.md) §11, and
[docs/live-replay.md](docs/live-replay.md) §14.

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
├── Dockerfile              builder + runtime base + api/dashboard targets
├── Dockerfile.render       build -> prepare -> verify -> prune -> one runtime image
├── compose.yaml            prepare -> api -> dashboard  (local)
├── compose.deploy.yaml     + proxy, ports un-published  (server overlay)
├── render.yaml             Render blueprint: one free web service, nothing else
├── .dockerignore           exclude everything, re-admit what the build needs
├── .env.deploy.example     deployment template: hostname, ports, routing policy
├── deploy/
│   ├── caddy/
│   │   ├── Caddyfile           default policy: the console only
│   │   └── Caddyfile.api-docs  optional: + the API's read-only surface
│   └── render/
│       └── Caddyfile           single-container policy: console + /healthz
├── configs/
│   ├── data/
│   │   ├── synthetic-testing.yaml     small dataset for CI
│   │   ├── synthetic-demo.yaml        4h stream the container trains on
│   │   └── synthetic-development.yaml larger dataset for local use
│   ├── features/
│   │   ├── feature-testing.yaml       CI-sized feature configuration
│   │   ├── feature-demo.yaml          the container's feature contract
│   │   └── feature-development.yaml   full window ladder, strict isolation
│   ├── detection/
│   │   └── rules-demo.yaml            the container's rule configuration
│   ├── ml/
│   │   └── model-demo.yaml            the container's ML configuration
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
│   ├── deployment.md           the public perimeter, and how to stand one up
│   ├── render-deployment.md    the single-container Render free-tier adapter
│   ├── docker.md               the local containerized demonstration
│   ├── feature-catalog.md      generated from the catalog
│   ├── feature-contract.md
│   ├── ingestion.md
│   ├── leakage-prevention.md
│   ├── privacy-model.md
│   ├── reproducibility.md
│   ├── synthetic-generation.md
│   └── temporal-semantics.md
├── scripts/
│   ├── verify.sh
│   ├── prepare_demo_bundle.py  the offline pipeline the container runs once
│   ├── verify_serving_bundle.py verifies a prepared bundle; decides nothing
│   ├── render_entrypoint.py    PID 1 for the single-container deployment
│   ├── start_demo.sh           thin wrapper around `docker compose up`
│   ├── stop_demo.sh            thin wrapper around `docker compose down`
│   └── deploy/
│       └── bootstrap_server.sh SERVER ONLY — prepares a fresh Ubuntu host
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

### …or just run it

Docker Engine with Compose v2, and nothing else — no Python, no `uv`, no
database, no secret:

```bash
docker compose up --build     # or ./scripts/start_demo.sh
docker compose down           # or ./scripts/stop_demo.sh
```

The first run trains a champion before the API starts: a one-shot `prepare`
service runs the real pipeline offline and takes about a minute. Nothing is
fitted at serving time. See **[docs/docker.md](docs/docker.md)**.

On a server, add the deployment overlay — one reverse proxy becomes the only
public listener and both application ports stop being published:

```bash
cp .env.deploy.example .env.deploy      # hostname, publish spec, routing policy
docker compose --env-file .env.deploy \
  -f compose.yaml -f compose.deploy.yaml up -d --build
```

See **[docs/deployment.md](docs/deployment.md)** for the single-VPS perimeter and
**[docs/render-deployment.md](docs/render-deployment.md)** for the Render
free-tier adapter. Nothing is deployed today.

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

# Larger checks, deselected by default
uv run pytest -m slow --no-cov
```

The container suite lives behind the `slow` marker and skips itself when no
Docker daemon is reachable or the images have not been built:

```bash
docker compose build
uv run pytest -m slow tests/integration/test_docker_compose.py --no-cov
uv run pytest -m slow tests/integration/test_deployment_topology.py --no-cov
```

The first brings the local stack up only if it is not already up, and tears down
only what it started — running it will not destroy a stack somebody is
demonstrating from. The second brings up the **deployment** topology with the
proxy on an unprivileged loopback port, and removes everything it created,
volumes included; it skips itself if a `pad-demo` stack is already running,
because the service containers have fixed names.

The architecture and security claims that need no daemon are unit tests:
`tests/unit/deployment/test_container_contract.py` (95) and
`tests/unit/deployment/test_deployment_contract.py` (92) read the Dockerfile,
both Compose files, both Caddyfiles, `.dockerignore`, `.gitignore`, the
environment template and the bootstrap script, and assert what they promise.

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
- Phases 1-5 do not implement: a FastAPI service, a SOC dashboard, database
  persistence, MLflow, DVC, or deployment. These are planned for later phases.
- **Nothing is deployed.** Phase 6 Milestone 4 containerizes the system and
  Milestone 5A prepares and locally verifies the perimeter a public deployment
  would need. Neither performs one: this project has no public URL, no server,
  and no domain name. The deployment has **no authentication and no rate
  limiting**, and [docs/deployment.md](docs/deployment.md) §15 states that
  residual risk rather than implying it is covered.
- **Nothing in this repository demonstrates real-world detection
  effectiveness.** Every figure the locked evaluation produces was measured on
  traffic this repository generated, under a declared scenario configuration.
  It is evidence that the pipeline behaves as specified, not evidence about
  real authentication systems.
- Baseline artifacts hold pseudonymous per-entity state. They are written only
  to git-ignored paths with restrictive permissions and must never be
  committed; real-data baselines require protected storage.
- **No ML model takes part in Phase 4**, and **no real authentication traffic
  is generated anywhere.** Every Phase 4 decision comes from a reviewed rule
  with declared thresholds; the Phase 5 model is reported alongside the rule
  engine and never in place of it.
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
- **A frozen champion is a subject, not a result.** `champion.lock` says what an
  evaluation was permitted to run. It carries no metric and says nothing about
  how the model behaves.
- **Structural validity is not predictive quality.** `ml validate` establishes
  that a prediction artifact is internally consistent and attributable to the
  champion that produced it, never that flagging those rows was a good idea.
- **Calibration is internal.** A calibrator is calibrated against the frozen
  synthetic validation-A distribution. That is a property of that distribution,
  not of any real one.
- **Attribution is descriptive, not causal.** A contribution says how a fitted
  function decomposes over the columns it was handed. A family without an exact
  decomposition reports typed unavailable rather than an approximation.
- **Drift is monitoring evidence, not model correctness.** It is computed
  without labels, so it cannot say a model became wrong, and nothing retrains,
  promotes, or rethresholds on a finding.
- **The anomaly track is experimental** and the novel-anomaly holdout is a
  generalisation probe. Neither is part of the supervised result, and neither is
  combined with one.

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
| [docs/model-contract.md](docs/model-contract.md) | What an ML artifact must carry and guarantee |
| [docs/model-catalog.md](docs/model-catalog.md) | Generated: every declared model |
| [docs/champion-selection.md](docs/champion-selection.md) | Validation-only gates, ranking, and the freeze |
| [docs/experiment-ledger.md](docs/experiment-ledger.md) | The append-only run record |
| [docs/prediction-artifacts.md](docs/prediction-artifacts.md) | Batch inference outputs and their identity |
| [docs/test-evaluation.md](docs/test-evaluation.md) | The locked TEST protocol, fusion, comparison |
| [docs/explainability.md](docs/explainability.md) | Deterministic attribution and its limits |
| [docs/drift-monitoring.md](docs/drift-monitoring.md) | The frozen reference profile and drift semantics |
| [docs/model-card.md](docs/model-card.md) | Generated: purpose, scope, prohibited use, limitations |
| [docs/phase5-acceptance.md](docs/phase5-acceptance.md) | Generated: the final Phase 5 acceptance report |
| [docs/api.md](docs/api.md) | The HTTP serving layer: endpoints, constraints, privacy, limits |
| [docs/dashboard.md](docs/dashboard.md) | The analyst console: API boundary, views, session limits, offline behaviour |
| [docs/live-replay.md](docs/live-replay.md) | The synthetic replay demonstration: scenarios, determinism, run lifecycle, bounds |
| [docs/docker.md](docs/docker.md) | The local containerized demonstration: images, the preparation job, volumes, security model |
| [docs/deployment.md](docs/deployment.md) | The public perimeter: proxy, routing policy, TLS, firewall, update/rollback, per-rule availability |
| [docs/render-deployment.md](docs/render-deployment.md) | The Render free-tier adapter: one service, build-time bundle, supervision, `$PORT`, memory |

---

## Planned phases

| Phase | Topic |
|---|---|
| 1 | Engineering foundation ✓ |
| 2 | Data engineering and synthetic log generation ✓ |
| 3 | Feature engineering and behavioral baselines ✓ |
| 4 | Rule-based detection (brute-force, spraying, stuffing) ✓ |
| 5 | Machine-learning detection models ✓ |
| 6 | FastAPI detection service — Milestone 1 (serving foundation) ✓ |
| 7 | SOC dashboard |
| 8 | Persistence and event storage |
| 9 | Monitoring and alerting |
| 10 | MLOps and experiment tracking |
| 11 | Deployment |

---

## License

MIT — see [LICENSE](LICENSE).
