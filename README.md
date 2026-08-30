# AI-Powered Password Attack Detection System

[![Release](https://img.shields.io/badge/release-v0.6.0-1f6feb)](https://github.com/divya-m984/ai-password-attack-detection-system/releases/tag/v0.6.0)
[![CI](https://github.com/divya-m984/ai-password-attack-detection-system/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/divya-m984/ai-password-attack-detection-system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-3776ab)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A defensive security-analytics application that detects password-based attacks —
brute force, password spraying, credential stuffing, account takeover — in
authentication event streams. It produces three independent verdicts for every
event it is asked about: a deterministic **rule** verdict, a **machine-learning**
verdict, and a **frozen hybrid** verdict that fuses the two. The whole thing
ships as a deployable, interactive demonstration you can open in a browser.

> **Live demo — <https://pad-demo.onrender.com>**
>
> Hosted on Render Free. The service spins down when idle, so allow roughly
> **1–2 minutes** for it to wake after inactivity. Open it once before you need it.
>
> **The demonstration uses synthetic authentication data and does not collect
> real passwords or credentials.** No real login traffic, no real account, and no
> credential material is involved at any point.

**Current release: v0.6.0 — Deployable Security Analytics Application**

---

## Why this project

Password attacks do not share one signature. A brute-force burst against a single
account, a slow spray across a thousand accounts, and a stuffing run from a
rotating client pool all look different, and each looks different again from
ordinary login traffic on a bad Monday morning.

Two approaches cover different ground, and neither covers all of it:

- **Rules** capture patterns a human analyst has already named. They are
  deterministic and explainable, but blind to anything nobody wrote down.
- **Machine learning** scores broader behaviour from the data, but its output is
  a number that needs calibration, a threshold, and an honest evaluation before
  it means anything.

This project builds both, keeps them structurally separate all the way to the
screen, and combines them with a fusion strategy chosen on validation data and
frozen before any test label was opened. The evaluation discipline is as much the
point as the detection: features are computed strictly from an event's prior
history, splits are chronological and campaign-aware, and exactly one command in
the entire project may open the test ground truth.

This is a demonstration architecture on synthetic data. It is not a claim that
this design outperforms anything in production.

---

## What it does

**Data and features**

- Deterministic synthetic authentication-event generation across nine scenarios —
  same config and seed, same content fingerprint
- Privacy-aware CSV/JSONL ingestion: HMAC-SHA256 pseudonymization, and
  header-level rejection of credential and ground-truth columns
- A point-in-time feature engine: 201 declared features, every historical
  aggregate computed strictly from `[t − window, t)`
- Chronological, campaign-aware splitting with purge and embargo, audited by
  twelve leakage checks

**Detection**

- Nine registered rules across six families, each with explicit false-positive
  controls
- Supervised ML detection with a known-malicious category classifier and a
  separate, experimental anomaly track
- Validation-only champion selection, frozen calibration and operating point
- **STACKED** hybrid fusion, selected on validation and frozen
- A locked TEST evaluation read exactly once, on an append-only experiment ledger
- Deterministic model serialization, deterministic attribution, and PSI drift
  against a frozen training reference

**Application**

- FastAPI serving layer, a pure adapter over frozen state
- Streamlit security-analytics console (an API client and nothing else)
- Deterministic synthetic live-replay scenarios
- Docker and Docker Compose, a hardened reverse-proxy deployment topology, and a
  single-container Render Free adapter
- A public deployment

---

## Live demo

**<https://pad-demo.onrender.com>**

What a visitor can do:

| | |
|---|---|
| **Overview** | System status, which components are loaded, what is frozen |
| **Live Replay** | Run any of seven synthetic attack scenarios, one event at a time |
| **Alerts** | Inspect what fired, with evidence and severity |
| **Analytics** | Aggregate view of a run |
| **Rule vs ML vs Hybrid** | The three verdicts side by side, never blended |
| **Explainability** | Per-feature attribution for the model's own output |
| **Drift Monitoring** | PSI against the frozen training reference profile |

Notes for anyone opening it:

- **Cold start.** Render Free spins the service down after inactivity; the first
  request pays roughly one to two minutes. No uptime or performance guarantee is
  offered.
- **Nothing persists.** Replay history lives in one API process's memory and in
  one browser session. A restart clears it. There is no alert database.
- **No real authentication traffic is processed.** Every event is fabricated by
  this repository, involving entities that do not exist.
- **The raw detection API is not public.** Only the console and a liveness check
  are reachable from the internet — see [API](#api).

A walkthrough suitable for a live presentation: **[docs/demo.md](docs/demo.md)**.

---

## Demo scenarios

Seven reviewed, deterministic, credential-free scenarios, replayed one event at a
time into the same serving path an HTTP request takes. Each publishes the rules
it expects to fire, and an integration test asserts that list as an **equality**
against the real frozen deployment.

| Scenario | What it is | Events | Span | Rules that fire |
|---|---|---|---|---|
| **Normal login activity** | One analyst, one device, twelve ordinary sign-ins | 12 | 295 s | *(none — the control)* |
| **Concentrated brute force** | Thirty failures against one account from one source | 30 | 290 s | `PAD-BF-001`, `PAD-BOT-001` |
| **Password spraying** | One source, one failed attempt each against 24 accounts | 24 | 115 s | `PAD-PS-001`, `PAD-BOT-001` |
| **Credential stuffing** | 18 accounts, 21 attempts, four rotating clients, three successes | 21 | 280 s | `PAD-PS-001` |
| **Account-takeover indicator** | Ordinary sign-ins, then a failure burst, then a success from a new country | 14 | 160 s | `PAD-BF-001`, `PAD-BF-002` |
| **Automated client activity** | A service identity authenticating on a fixed twelve-second cadence | 24 | 276 s | `PAD-BOT-001` |
| **Mixed attack timeline** | Spraying, then concentrated failures, then a takeover success | 29 | 200 s | `PAD-BF-001`, `PAD-BF-002`, `PAD-PS-001` |

Normal activity firing **nothing** is the expected outcome, not a failed run.

The credential-stuffing and account-takeover scenarios carry a published caveat:
the two rules named after them **cannot fire on a live request** in this
deployment, and the rules listed above are what the traffic actually trips. That
is a stated limitation, not a tuning oversight —
see [Limitations](#limitations).

**Nothing attacks anything.** Import-time guards refuse credential-shaped field
names and any address outside the RFC 5737 documentation ranges, and an AST test
refuses a network client anywhere in the replay package. Pace (`instant`, `fast`,
`normal`, `slow`) changes wall-clock spacing only — the same scenario produces
byte-identical verdicts at every pace.

Full reference: **[docs/live-replay.md](docs/live-replay.md)**.

---

## Architecture

### Detection pipeline

```
              Authentication events
                       │
                       ▼
        Canonical schema · validation · privacy
                       │
                       ▼
            Point-in-time feature engine
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Rule engine (9 rules)     ML champion (frozen)
          │                         │
          └──────►  STACKED  ◄──────┘
                    fusion
                       │
                       ▼
                Final verdict
                       │
                       ▼
          Risk score · severity · alert
```

The rule layer, the model layer, and the fusion layer are **separate typed
objects** all the way to the screen. The rule layer's ordinal 0–100 `risk_score`
is never blended with the model's probability, and no page renames one the other.

### Deployment

```
                    Browser
                       │
                       ▼
             Render HTTPS edge (TLS)
                       │
                       ▼
                     Caddy
           (the only public listener)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  Streamlit console            internal FastAPI
   127.0.0.1:8501               127.0.0.1:8000
        │                             │
        └──── server-side client ─────┘
                                      │
                                      ▼
                        frozen serving bundle
```

Two properties do most of the work here:

- **The dashboard is an API client only.** No detection capability is importable
  from it, and its HTTP client runs *server-side* — a browser never talks to the
  detection service. That is what lets the API stay on loopback without costing
  the demonstration anything.
- **Scientific state is frozen before serving.** The champion, its calibrator,
  its threshold, and the fusion state are sealed into a fingerprinted serving
  bundle that the API mounts read-only and verifies at startup. Nothing is fitted
  at serving time, ever.

---

## Detection approach

| Layer | Produces | Nature |
|---|---|---|
| **Rules** | fired rules, evidence, `signal_strength`, `risk_score` 0–100 | deterministic, explainable, reviewed thresholds |
| **ML** | a calibrated probability and a binary decision | fitted on TRAIN, calibrated on validation-A, thresholded on validation-B |
| **Hybrid** | one fused decision | **STACKED**, selected on validation only, frozen before any TEST label was opened |

The current binary champion family is **logistic regression** (`M-010`), selected
by the validation-only gates from the three champion-eligible entries in a
six-model catalog. The remaining three are a mandatory prior baseline that exists
to be beaten, a declared-but-ineligible boosting model, and an experimental
anomaly track kept out of the supervised result entirely.

**No fallback hybrid exists.** The fusion runtime refuses to construct unless the
executing strategy equals the selected one and its state passes a fingerprint
check. If the frozen selection cannot be verified, readiness returns `503` — a
strategy nobody selected is never substituted.

**None of the rule numbers is a probability.** `signal_strength` is a bounded
ordinal magnitude in `(0, 1]`; `risk_score` is a bounded ordinal magnitude in
`[0, 100]`, and zero fired rules yields exactly `0.0`. Only a *calibrated* model
score may be called a probability. Rule evidence records what was observed and
which configured condition matched — it is an indicator, not causal proof.

Risk scoring is correlation-aware: contributions are reduced within a family and
combined with a noisy-OR, so one behaviour restated three ways cannot inflate a
score, an unrelated signal can never lower risk, and the result is
order-invariant. See [docs/risk-scoring.md](docs/risk-scoring.md).

### The nine rules

| Rule | Name | Family |
|---|---|---|
| `PAD-BF-001` | Concentrated brute-force indicator | `brute_force` |
| `PAD-BF-002` | Successful authentication after failure burst | `brute_force` |
| `PAD-DBF-001` | Distributed brute-force indicator | `brute_force` |
| `PAD-PS-001` | Password-spraying indicator | `spraying` |
| `PAD-CS-001` | Credential-stuffing indicator | `stuffing` |
| `PAD-ATO-001` | Account-takeover indicator | `account_compromise` |
| `PAD-MFA-001` | MFA sequence anomaly indicator | `account_compromise` |
| `PAD-GEO-001` | Impossible-travel indicator | `location` |
| `PAD-BOT-001` | Bot-like authentication indicator | `automation` |

Measured over all 154 replay anchors on the deployed serving path: **four fire**
(`PAD-BF-001`, `PAD-BF-002`, `PAD-BOT-001`, `PAD-PS-001`), **three evaluate
normally and return clean negatives** because no built-in scenario carries the
shape they need (`PAD-DBF-001`, `PAD-GEO-001`, `PAD-MFA-001`), and **two cannot
fire on any live request** (`PAD-CS-001`, `PAD-ATO-001`). Per-rule outcomes with
counts: [docs/deployment.md](docs/deployment.md) §16.

---

## Scientific integrity

- **Point-in-time features.** Every historical aggregate is computed strictly
  from `[t − window, t)`, with same-timestamp mutual exclusion enforced
  structurally rather than by convention.
- **Chronological, campaign-aware splitting** into TRAIN / validation-A /
  validation-B / TEST, with purge and embargo, and a twelve-check leakage auditor.
- **TEST influences nothing.** Training, calibration, thresholding, champion
  selection and fusion selection all complete on validation data before a test
  label is opened. **Exactly one command opens the TEST ground truth:
  `ml evaluate`.** Across the whole project exactly two modules may read a
  ground-truth table, and an import-graph test pins that set in both directions.
- **Deterministic serialization.** Model artifacts are numbers, not objects — no
  `pickle`, no `joblib`, no estimator reconstructed to score.
- **Frozen serving state.** The bundle the API loads is fingerprinted end to end
  and verified on every startup.

<details>
<summary><b>Frozen identities of the deployed demonstration</b></summary>

Reproduced independently by the Compose preparation job and by the Render image's
build stage, and asserted equal by an integration test:

| | |
|---|---|
| Champion scope key | `d85a151c597b8b9eca0cd570b236aab91cb3cb13f021ee23576421fa1b9f5e90` |
| Serving-manifest fingerprint | `8341204696599ea08a3299bf249b214131b35f1073f46f5d25d41283b8799954` |
| STACKED state fingerprint | `134f66ce2b837c4b74c9ab9ff2703125b9aedefb210d18afe923a0662f96272a` |
| Selected fusion strategy | `stacked` |

The full fingerprint chain is in
[docs/render-deployment.md](docs/render-deployment.md) §5.

</details>

---

## Quick start

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), Git.

```bash
git clone https://github.com/divya-m984/ai-password-attack-detection-system.git
cd ai-password-attack-detection-system
uv sync --all-groups
```

Verify the installation:

```bash
uv run password-attack-detector version
uv run password-attack-detector doctor
uv run password-attack-detector show-config
uv run password-attack-detector --help
```

The CLI has four command groups — `data`, `features`, `detection`, `ml` — plus
`deploy materialize` / `deploy inspect`. Every command supports `--help`, returns
non-zero on failure, refuses to overwrite without `--force`, and prints no event
identifier, pseudonym, coordinate, or absolute path.

Running the API and console directly on the host requires a prepared artifact
root; the container demo below produces one for you, and
[docs/api.md](docs/api.md) §10 documents the manual path.

---

## Docker demo

The easiest way to see the whole system. Docker Engine with Compose v2, and
nothing else — no Python, no `uv`, no database, no secret:

```bash
./scripts/start_demo.sh      # or: docker compose up --build
```

| | |
|---|---|
| API | <http://localhost:8000> |
| Swagger | <http://localhost:8000/docs> |
| Dashboard | <http://localhost:8501> |

```bash
./scripts/stop_demo.sh       # or: docker compose down
```

Three services, one ordering: `prepare → api → dashboard`.

**The first run takes longer, on purpose.** No image ships a trained model — a
fitted champion is an *output*, and an image carrying one would make the image
the provenance of a scientific decision. A one-shot `prepare` job runs the
project's real pipeline offline — generate, build features, train, select,
freeze, predict, detect, evaluate, materialize — into a named volume in about a
minute, then exits. The API starts only after that job reports success, mounts
the volume **read-only**, and verifies what it finds.

Hardened by default: non-root (uid 10001), read-only root filesystems, all
capabilities dropped, `no-new-privileges`, no host networking, no Docker socket,
no bind mounts, and ports published to the host's loopback only.

Full reference: **[docs/docker.md](docs/docker.md)**.

---

## Dashboard

A Streamlit security-operations console, structurally a client of the API and
nothing else.

| Group | Views |
|---|---|
| **Primary** | Overview · Live Replay · Alerts · Analytics · Explainability · Drift Monitoring |
| **Advanced** | Detection Console · Authentication Events · Rule vs ML vs Hybrid · System & Model |
| **About** | About System |

It is configured by four optional variables — `PAD_DASHBOARD_API_URL`,
`PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS`, `PAD_DASHBOARD_REFRESH_SECONDS`,
`PAD_DASHBOARD_PAGE_TITLE` — none of which can name a model, a threshold, or a
strategy.

Full reference: **[docs/dashboard.md](docs/dashboard.md)**.

---

## API

FastAPI powers the serving layer: fifteen operations covering detection,
explanation, replay control, system information, and health. It is an *adapter* —
it computes no feature, re-derives no threshold, and contains no second scoring
implementation.

**Locally**, the whole surface is available, with Swagger at
<http://localhost:8000/docs>.

**On the public Render deployment, the raw detection and control APIs are
intentionally not exposed.**

| | Routes |
|---|---|
| **Public** | `/` and everything under it (the console, including its websocket) · `/healthz` — liveness only: a status string, the service name, and the package version |
| **Internal** (bound to `127.0.0.1` inside the container) | `/api/v1/detect`, `/detect/batch` · `/api/v1/explain` · `/api/v1/demo/*` replay control · `/api/v1/system/status`, `/model/info`, `/rules` · `/ready` · `/version` · `/docs`, `/openapi.json` |

Scoring costs CPU and there is no rate limiting, so scoring is not offered to
anonymous callers; `/ready` names which scientific component is unavailable and
why, which is operator diagnostics rather than an anonymous caller's business.

Because Streamlit answers `200` with its own single-page shell for *any*
unrecognised path, a status code proves nothing about this policy — every routing
assertion in the test suite inspects the response **body**.

Full reference: **[docs/api.md](docs/api.md)**.

---

## Security and privacy

- **No plaintext passwords, hashes, tokens, cookies, or real credentials** are
  stored, logged, or accepted at any layer.
- **No credential field is accepted** by any request schema. Schemas are strict
  (`extra="forbid"`), so an undeclared field is refused outright, and prohibited
  names are rejected at the ingestion header level before any value is read.
- **No attack target can be supplied.** There is no host, URL, endpoint, or
  address field anywhere in the request surface.
- **No model, threshold, or fusion strategy is reachable from a request.**
  Deployment settings can say *where* things are and *how much* is allowed, never
  *which* — both settings classes carry an import-time guard refusing such a
  field name.
- **No training, promotion, freeze, or upload route exists.** The complete
  fifteen-operation route inventory is pinned as an equality against the running
  application's own schema.
- **No arbitrary code execution**: no `pickle`, no `joblib`, and no `eval`,
  `exec`, or dynamic import in any artifact reader.
- **Containers run non-root** (uid 10001) with read-only root filesystems, all
  capabilities dropped, `no-new-privileges`, no host networking, and no Docker
  socket.
- **Frozen scientific state is read-only at runtime** — mounted read-only under
  Compose, copied root-owned into a layer the serving account cannot write under
  Render.
- Source identifiers are pseudonymized via HMAC-SHA256. **Pseudonymization
  reduces exposure but does not guarantee anonymity.** Ground-truth labels are
  always stored separately from canonical events.
- **Replay is synthetic and bounded**, and no request leaves the deployment.

These are the constraints this repository builds and tests. They are not a
production security certification, and **there is no authentication and no rate
limiting anywhere in this system** — see [Limitations](#limitations).

See [docs/privacy-model.md](docs/privacy-model.md) and
[SECURITY.md](SECURITY.md).

---

## Verification

```bash
bash scripts/verify.sh        # the full suite, mirroring CI
```

That runs `uv lock --check`, `ruff check`, `ruff format --check`,
`mypy src tests` under `strict`, `pytest`, and `uv build` (wheel + sdist).
`pre-commit` hooks are configured for the same checks.

Measured for the v0.6.0 release:

| | |
|---|---|
| `uv run pytest` | **6,822 passed, 4 skipped** |
| Coverage | **94.43 %** (gate: 90 %) |
| `uv run pytest -m slow --no-cov` | **163 passed**, 0 skipped |

The slow suite holds the container and deployment-topology checks; it skipped
nothing, which is the meaningful result. Architecture and security claims that
need no Docker daemon are ordinary unit tests that read the Dockerfiles, both
Compose files, all three Caddyfiles, `.dockerignore`, `.gitignore`, the
environment templates and the bootstrap script, and assert what they promise.

Full acceptance record: **[docs/phase6-acceptance.md](docs/phase6-acceptance.md)**.

---

## Project structure

```
src/password_attack_detector/
├── api/            the FastAPI serving layer
├── dashboard/      the Streamlit console (API client only)
├── data/           schema, privacy, ingestion, synthetic generation
├── deployment/     the sealed serving bundle: write, verify, materialize
├── detection/      the nine rules, risk scoring, alerts, evaluation
├── features/       the point-in-time engine, baselines, splitting, leakage
├── ml/             catalog, training, calibration, champion, fusion, drift
└── replay/         the synthetic live replay

configs/            data, features, detection, ml, and environment configs
deploy/             Caddy routing policies (VPS and Render)
docs/               the documentation index below
scripts/            verify.sh, the demo scripts, the offline pipeline
tests/              unit/ and integration/
```

Deployment files live at the repository root: `Dockerfile`, `Dockerfile.render`,
`compose.yaml`, `compose.deploy.yaml`, and `render.yaml`.

---

## Limitations

Read this before drawing any conclusion from a demonstration.

**The data is synthetic.** Every figure this repository can produce was measured
on traffic this repository generated, under a declared scenario configuration. It
is evidence that the pipeline behaves as specified, **not evidence about real
authentication systems**, and it does not demonstrate real-world detection
effectiveness. The generator does not reproduce the distributional properties of
real login traffic, and thresholds tuned on generated data reflect the
generator's parameters.

**This is not production authentication infrastructure.** It observes and scores
event records. It does not authenticate anyone, sit in a login path, block a
session, or integrate with an identity provider.

**Nothing persists.** There is no alert database, no event store, and no serving
drift report. A replay run lives in one API process's memory and the console's
history in one browser session; a restart clears both and preserves only the
frozen champion.

**Render Free constraints are real.** Cold starts of roughly one to two minutes
after inactivity, 512 MiB of memory, and a fraction of a CPU. Throttling changes
how long a demonstration takes, not what the detector decides — every replay
verdict measured at 0.1 CPU was identical to the same verdict at 0.5 CPU.

**There is no public raw detection API**, by design. Scoring, explanation, replay
control, readiness and Swagger are internal; a viewer interacts with the console.

**There is no authentication and no rate limiting anywhere in this system**, and
none was added for a demonstration — a half-built auth platform is a larger
surface than the one it closes. That residual risk is stated in
[docs/deployment.md](docs/deployment.md) §15 rather than implied to be covered.

**`PAD-CS-001` and `PAD-ATO-001` cannot fire on a live request.** Both gate on a
fitted behavioural baseline that the serving path does not load, so
`user_in_baseline` is always false and every novelty flag is null. Adding a
baseline was audited and *measured* not to help: the replay catalog's identities
are UUIDv5 pseudonyms derived from the scenario itself, so no training population
contains them, whatever dataset a deployment trains on. Making those rules fire
would mean either loosening a frozen rule or coupling a content-addressed catalog
to one dataset. **v0.6.0 ships with this stated, not fixed** — no threshold was
moved and no baseline was synthesised to make a demonstration look better.

**On the detection layers themselves:** `signal_strength` and `risk_score` are
not probabilities; evidence is not causal proof; account-takeover and
impossible-travel outputs are *indicators* that travel, a new device, or a VPN
can reproduce; detection quality is bounded by feature quality; calibration is
internal to the synthetic validation distribution; attribution is descriptive,
not causal; drift is monitoring evidence, not model correctness; and the anomaly
track is experimental throughout. Nothing retrains, promotes, or rethresholds on
a finding.

**Reproducibility is bounded by the committed `uv.lock` environment.** A
different library version may produce different output for the same seed.

See [docs/detection-limitations.md](docs/detection-limitations.md) and
[docs/model-card.md](docs/model-card.md).

---

## Documentation

**Start here**

| File | Contents |
|---|---|
| [docs/demo.md](docs/demo.md) | The presentation walkthrough: what to click, in what order, what to say |
| [docs/phase6-acceptance.md](docs/phase6-acceptance.md) | The acceptance record and the v0.6.0 release decision |

**The application**

| File | Contents |
|---|---|
| [docs/api.md](docs/api.md) | Endpoints, constraints, privacy, limits |
| [docs/dashboard.md](docs/dashboard.md) | The console: API boundary, views, session limits, offline behaviour |
| [docs/live-replay.md](docs/live-replay.md) | Scenarios, determinism, run lifecycle, bounds |
| [docs/docker.md](docs/docker.md) | Local containers: images, preparation job, volumes, security |
| [docs/deployment.md](docs/deployment.md) | The VPS perimeter: proxy, routing, TLS, firewall, per-rule availability |
| [docs/render-deployment.md](docs/render-deployment.md) | The Render adapter: one service, build-time bundle, supervision, memory |

**How it works**

| File | Contents |
|---|---|
| [docs/data-contract.md](docs/data-contract.md) | Canonical event schema and prohibited fields |
| [docs/privacy-model.md](docs/privacy-model.md) | Pseudonymization, key management, limitations |
| [docs/synthetic-generation.md](docs/synthetic-generation.md) | Nine scenarios, determinism, limitations |
| [docs/temporal-semantics.md](docs/temporal-semantics.md) | The point-in-time contract |
| [docs/leakage-prevention.md](docs/leakage-prevention.md) | The twelve leakage checks |
| [docs/dataset-splitting.md](docs/dataset-splitting.md) | Chronological, campaign-aware splits |
| [docs/rule-catalog.md](docs/rule-catalog.md) | Generated: every registered rule |
| [docs/risk-scoring.md](docs/risk-scoring.md) | Correlation-aware scoring and its proven properties |
| [docs/champion-selection.md](docs/champion-selection.md) | Validation-only gates, ranking, and the freeze |
| [docs/test-evaluation.md](docs/test-evaluation.md) | The locked TEST protocol, fusion, comparison |
| [docs/explainability.md](docs/explainability.md) | Deterministic attribution and its limits |
| [docs/drift-monitoring.md](docs/drift-monitoring.md) | The frozen reference profile and drift semantics |
| [docs/model-card.md](docs/model-card.md) | Generated: purpose, scope, prohibited use, limitations |
| [docs/reproducibility.md](docs/reproducibility.md) | Fingerprinting and environment pinning |

The remaining contracts and generated catalogs are in [docs/](docs/).

---

## Release

**Current release: v0.6.0 — Deployable Security Analytics Application**
· [GitHub Release](https://github.com/divya-m984/ai-password-attack-detection-system/releases/tag/v0.6.0)

| Release | Theme |
|---|---|
| 0.1.0 | Engineering foundation — configuration, logging, paths, CLI, CI |
| 0.2.0 | Data foundation — canonical schema, synthetic generation, ingestion, manifests |
| 0.3.0 | Feature engineering — the point-in-time engine, baselines, splitting, leakage audit |
| 0.4.0 | Rule-based detection — nine rules, risk scoring, alerts |
| 0.5.0 | Machine-learning detection — catalog, champion, calibration, fusion, drift |
| **0.6.0** | **Deployable security analytics application** — serving, console, replay, containers, deployment, public demo |

The version is declared in `pyproject.toml` and
`src/password_attack_detector/__init__.py`, pinned against each other by test,
and reported by `password-attack-detector version`, `GET /version`, `GET /health`,
and every container image tag.

Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
Security policy: see [SECURITY.md](SECURITY.md).

---

## License

MIT — see [LICENSE](LICENSE).
