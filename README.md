# Password Attack Detector

**A deployable defensive security-analytics application that detects
password-based attacks in authentication traffic — with rules, with machine
learning, and with a frozen hybrid of the two.**

The system reads authentication events, turns them into point-in-time features,
and produces three separate verdicts for every anchor event: a deterministic
rule verdict, a machine-learning verdict, and a frozen hybrid verdict that fuses
them. It ships with an HTTP serving layer, a security-operations console, a
synthetic live-replay demonstration, containers, a hardened deployment
topology, and a public demo.

It is **defensive only**. It never stores plaintext passwords, never cracks
credentials, never automates authentication attempts, and never sends a request
to any system it is not itself running.

---

## Live demo

**<https://pad-demo.onrender.com>**

- Hosted on **Render Free**. The service **spins down after inactivity**, so the
  first request after a quiet period triggers a **cold start**. Allow roughly
  **1–2 minutes** before a presentation, and open the URL once to wake it.
- The demonstration runs entirely on **synthetic authentication data** generated
  by this repository. No real login traffic, no real account, and no credential
  material is involved at any point.
- No uptime, availability, or performance guarantee is offered or implied.
- Only the console and a health check are public. The detection API, the replay
  controls, the explainability endpoint, readiness, and Swagger are bound to
  loopback inside the container and are **not reachable from the internet** —
  see [Public deployment](#public-deployment).

A walkthrough suitable for a live presentation is in **[docs/demo.md](docs/demo.md)**.

---

## Project status

| | |
|---|---|
| Core engineering (Phases 1–5) | **complete** |
| Deployable demo (Phase 6) | **complete** |
| Public demo | **available** at <https://pad-demo.onrender.com> |
| Current release | **0.6.0** — *Deployable security analytics application* |

Every phase of the engineering plan is delivered: data foundation, feature
engineering, rule-based detection, machine-learning detection, serving,
console, replay, containers, deployment perimeter, and a public deployment.
What remains open is documented under [Limitations](#limitations) rather than
scheduled.

---

## Key capabilities

**Data and features**

- Canonical, strictly-validated authentication-event schema with ground truth
  held in a separate table and joined only by `event_id`
- Deterministic synthetic generation across nine scenarios — same config and
  seed produce the same content fingerprint
- CSV/JSONL ingestion with HMAC-SHA256 pseudonymization and header-level
  rejection of credential and ground-truth columns
- A point-in-time feature engine: ~200 declared features, every historical
  aggregate computed strictly from `[t - window, t)`, with same-timestamp mutual
  exclusion enforced structurally
- Chronological, campaign-aware splitting with purge and embargo, and a
  twelve-check leakage auditor — four of whose checks are behavioural

**Detection**

- Nine registered rules across six families, each ANDing multiple conditions and
  carrying an explicit false-positive control
- Correlation-aware risk scoring: group reduction then noisy-OR, so one
  behaviour restated three ways cannot inflate a score
- Alert grouping, deduplication, cooldown, rate limiting, and escalation with a
  validator-enforced accounting identity
- An executable model catalog of six models in six families, of which three are
  champion-eligible; the rest are a mandatory reference baseline that exists to
  be beaten, a declared-but-ineligible boosting model, and an experimental
  anomaly track kept out of the supervised result
- Validation-only champion selection, frozen calibration and operating point, an
  append-only experiment ledger, and a TEST evaluation read exactly once
- A frozen **STACKED** hybrid fusion selected on validation before any test
  label could be opened
- Deterministic, exact-or-unavailable model attribution, and PSI drift against a
  frozen training reference profile

**Application**

- A FastAPI serving layer that is an *adapter* — it computes no feature,
  re-derives no threshold, and contains no second scoring implementation
- A Streamlit SOC console that is structurally a client of that API and nothing
  else: no detection capability is importable from it
- A synthetic live replay that emits reviewed scenarios one event at a time into
  the same serving path an HTTP request takes
- Docker and Docker Compose, a hardened reverse-proxy deployment topology, and a
  single-container Render free-tier adapter
- A public deployment on Render Free

---

## Architecture

### The deployed application

```
                      Browser
                         │
                         ▼
              Render HTTPS edge (TLS)
                         │
                         ▼
                     Caddy  (the only public listener)
                         │
        ┌────────────────┴─────────────────┐
        │                                  │
        ▼                                  ▼
  Streamlit dashboard              internal FastAPI
  127.0.0.1:8501                   127.0.0.1:8000
        │                                  │
        └──────── server-side client ──────┘
                                           │
                                           ▼
                             frozen serving bundle
                                           │
                                           ▼
                        rules  +  ML  +  STACKED fusion
```

Three processes, one container, one supervisor. The console's HTTP client runs
**server-side**, so a browser never talks to the detection service; that is what
lets the API stay on loopback without costing the demonstration anything.

### The pipeline behind it

```
Authentication events
        │
        ▼
  Canonical schema + validation + privacy contracts
        │
        ▼
  Point-in-time feature engine  ──►  behavioural baselines (offline)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Rule engine (9 rules)          ML champion (frozen)
        │                              │
        └──────────► STACKED fusion ◄──┘
                          │
                          ▼
              Risk score, severity, alerts
```

The rule layer, the model layer and the fusion layer are **separate typed
objects** all the way to the screen. The rule layer's ordinal 0–100 `risk_score`
is never blended with the model's probability, and no page renames one the
other.

---

## Detection approach

### Three layers, kept apart

| Layer | Produces | Nature |
|---|---|---|
| **Rules** | fired rules, evidence, `signal_strength`, `risk_score` 0–100 | deterministic, explainable, reviewed thresholds |
| **ML** | a calibrated probability and a binary decision at a frozen threshold | fitted offline on TRAIN, calibrated on validation-A, thresholded on validation-B |
| **Hybrid** | one fused decision | **STACKED**, selected on validation only, frozen before any TEST label was opened |

**No fallback hybrid exists.** `FusionRuntime` refuses to construct unless the
executing strategy equals the selected one and a STACKED deployment carries a
loaded, fingerprint-verified state. If the frozen selection cannot be verified,
readiness is `503` — a strategy nobody selected is never substituted.

### The nine rules

| Rule | Name | Family | Default severity |
|---|---|---|---|
| `PAD-BF-001` | Concentrated brute-force indicator | `brute_force` | high |
| `PAD-BF-002` | Successful authentication after failure burst | `brute_force` | high |
| `PAD-DBF-001` | Distributed brute-force indicator | `brute_force` | critical |
| `PAD-PS-001` | Password-spraying indicator | `spraying` | high |
| `PAD-CS-001` | Credential-stuffing indicator | `stuffing` | high |
| `PAD-ATO-001` | Account-takeover indicator | `account_compromise` | critical |
| `PAD-MFA-001` | MFA sequence anomaly indicator | `account_compromise` | medium |
| `PAD-GEO-001` | Impossible-travel indicator | `location` | high |
| `PAD-BOT-001` | Bot-like authentication indicator | `automation` | medium |

Measured over 154 replay anchors on the serving path: **four fire**
(`PAD-BF-001`, `PAD-BF-002`, `PAD-BOT-001`, `PAD-PS-001`), **three evaluate
normally and return clean negatives because no scenario exercises them**
(`PAD-DBF-001`, `PAD-GEO-001`, `PAD-MFA-001`), and **two cannot fire on any live
request** (`PAD-CS-001`, `PAD-ATO-001` — see [Limitations](#limitations)).
Full per-rule outcomes: [docs/deployment.md](docs/deployment.md) §16.

### What the numbers are, and are not

**None of these is a probability.**

- **Evidence** records what was observed and which configured condition matched.
  It is an indicator, **not causal proof**. Messages come from frozen catalog
  templates; proof-asserting language is rejected at import.
- **`signal_strength`** is a bounded ordinal magnitude in `(0, 1]`.
- **`risk_score`** is a bounded ordinal magnitude in `[0, 100]`. Zero fired rules
  yields exactly `0.0`, so a zero always means "nothing fired".
- Only a **calibrated** model score may be called a probability, and the word
  becomes available only after a calibrator is fitted and its calibration error
  measured.

Correlation-aware scoring:

```
contribution_r = family_weight[family(r)] × signal_strength(r)
c_g            = max(contribution_r for r in group g)
combined       = 1 − Π over sorted g of (1 − c_g)
risk_score     = max(round(100 × combined, 4), min_fired_risk_score)
```

Correlated rules cannot out-score the strongest of them, an unrelated signal can
never lower risk, and the result is order-invariant.
See [docs/risk-scoring.md](docs/risk-scoring.md).

### Why detection takes a window

Nearly every signal the rules and the model read is a windowed or sequence
quantity over an event's strictly-prior history. A single stateless event would
produce a snapshot whose history is empty — not "unknown", but *wrong*. So a
request carries an **ordered batch of events plus the anchors it wants a verdict
for**, and the service fabricates no history for a caller who supplies none.

### The label boundary

**Exactly one command opens the TEST ground truth: `ml evaluate`.** `ml predict`
has no `--labels` option, `detection run` has no `--labels` and no `--splits`
option, `ml validate` computes no accuracy, `ml explain` decomposes a model's own
output, and `ml drift` compares two populations without labels. Across the whole
project exactly two modules may open a ground-truth table —
`detection.evaluation` and `ml.dataset` — and an import-graph test pins that set
in both directions.

---

## Demo scenarios

Seven reviewed, deterministic, credential-free scenarios are replayed one event
at a time into the same serving path an HTTP request takes. Each publishes the
rules it fires, and an integration test asserts that list as an **equality**
against the real frozen deployment.

| Scenario | Events | Span | Rules expected to fire | Minimum severity |
|---|---|---|---|---|
| Normal login activity | 12 | 295 s | *(none)* | — |
| Concentrated brute force | 30 | 290 s | `PAD-BF-001`, `PAD-BOT-001` | high |
| Password spraying | 24 | 115 s | `PAD-PS-001`, `PAD-BOT-001` | high |
| Credential stuffing | 21 | 280 s | `PAD-PS-001` | medium |
| Account-takeover indicator | 14 | 160 s | `PAD-BF-001`, `PAD-BF-002` | medium |
| Automated client activity | 24 | 276 s | `PAD-BOT-001` | low |
| Mixed attack timeline | 29 | 200 s | `PAD-BF-001`, `PAD-BF-002`, `PAD-PS-001` | medium |

Normal activity firing **nothing** is the expected outcome, not a failed run.
The credential-stuffing and account-takeover scenarios carry a published
limitation: the two rules named after them cannot fire on a live request, and
`PAD-PS-001` / `PAD-BF-00x` are what the traffic actually trips.

**Nothing attacks anything.** Every scenario is a fabrication — events that did
not happen, involving entities that do not exist, replayed into a service the
operator is running themselves. Import-time guards refuse a credential-shaped
field name and any address outside the RFC 5737 documentation ranges; an AST
test refuses a network client anywhere in the replay package.

Pace (`instant`, `fast`, `normal`, `slow`) changes wall-clock spacing only: the
same scenario produces byte-identical verdicts at every pace.

Full reference: **[docs/live-replay.md](docs/live-replay.md)**.

---

## Local quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
cp .env.example .env          # edit as needed
uv run pre-commit install
```

Check the installation:

```bash
uv run password-attack-detector version
uv run password-attack-detector doctor
uv run password-attack-detector show-config
```

### Running the service and console

The serving layer needs a prepared artifact root. The quickest honest way to get
one is the containerized preparation job below; to do it on the host, run
`scripts/prepare_demo_bundle.py`, which is the same script the container runs.

**Terminal 1 — the API:**

```bash
export PAD_API_ARTIFACT_ROOT=/absolute/path/to/artifacts
export PAD_API_ALLOWLIST_PATH=/absolute/path/to/allowlist.yaml
export PAD_API_FEATURE_CONFIG_PATH=configs/features/feature-demo.yaml
export PAD_API_ML_CONFIG_PATH=configs/ml/model-demo.yaml
export PAD_API_DETECTION_CONFIG_PATH=configs/detection/rules-demo.yaml

uv run uvicorn password_attack_detector.api.app:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — the console:**

```bash
uv run streamlit run \
  src/password_attack_detector/dashboard/app.py \
  --server.address 127.0.0.1 --server.port 8501
```

| | |
| --- | --- |
| API | <http://127.0.0.1:8000> |
| Swagger | <http://127.0.0.1:8000/docs> |
| Dashboard | <http://127.0.0.1:8501> |

Before a champion has been frozen, the rule layer can be served alone with
`PAD_API_REQUIRE_ML_CHAMPION=false`. If the frozen selection was `stacked`,
publish the serving bundle once first with `deploy materialize`, then inspect it
with `deploy inspect`. Details in [docs/api.md](docs/api.md) §2.

The console is configured by four optional variables —
`PAD_DASHBOARD_API_URL`, `PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS`,
`PAD_DASHBOARD_REFRESH_SECONDS`, `PAD_DASHBOARD_PAGE_TITLE` — none of which can
name a model, a threshold, or a strategy.

---

## Docker quick start

Docker Engine with Compose v2, and nothing else — no Python, no `uv`, no
database, no secret:

```bash
docker compose up --build     # or ./scripts/start_demo.sh
docker compose down           # or ./scripts/stop_demo.sh
```

| | |
| --- | --- |
| API | <http://localhost:8000> |
| Swagger | <http://localhost:8000/docs> |
| Dashboard | <http://localhost:8501> |

Three services and one ordering: `prepare → api → dashboard`.

- **The image ships no trained model.** A fitted champion is an *output*, and an
  image carrying one would make the image the provenance of a scientific
  decision. A one-shot `prepare` job runs the project's **real pipeline** —
  generate, build features, train, select, freeze, predict, detect, evaluate,
  materialize — into a named volume in about a minute and exits. The API starts
  only after that job reports success, mounts the volume **read-only**, and
  verifies what it finds. Nothing is fitted at serving time, ever.
- **Hardened by default** — non-root (uid 10001), read-only root filesystems,
  all capabilities dropped, `no-new-privileges`, no host networking, no Docker
  socket, no bind mounts, and ports published to the host's loopback only.
- **No scientific control in the environment.** Every `PAD_API_*` variable in
  `compose.yaml` answers *where* or *how much*. Two tests enforce it, one of
  which refuses a variable name that merely *contains* `MODEL`, `THRESHOLD`,
  `FUSION`, `CHAMPION`, `SCORE`, `SECRET`, or `TOKEN`.

Full reference: **[docs/docker.md](docs/docker.md)**.

---

## Public deployment

### The public demo

<https://pad-demo.onrender.com> runs the single-container Render adapter
(`Dockerfile.render` + `render.yaml` + `deploy/render/Caddyfile`).

**One service, not two.** Free Render services get no private network, so a
separate console service could only reach the API over the public internet —
which would force the detection API to be publicly exposed. One container keeps
the API on loopback, halves instance-hour burn, and gives a viewer one cold start
instead of two.

**The bundle is baked at build time, and that deviation is stated rather than
glossed.** Render Free has no persistent disk for a preparation job to write
into, and preparing at container start would mean fitting three model families
on 0.1 CPU on every cold start. So a **discarded build stage** runs the same
tracked pipeline script over the same tracked configurations, verification
**fails the build** rather than the deployment, and the result is copied
root-owned into a runtime layer that cannot write it. The bundle is then verified
three more times: after the prune step, by the entrypoint, and by the API at
startup.

### The public/private route contract

**Public — reachable from the internet:**

| Route | What it is |
|---|---|
| `/` and everything under it | the Streamlit console, including its `/_stcore/stream` websocket |
| `/healthz` | liveness only; rewritten to the API's `/health`. Body is a status string, the service name, and the package version — nothing else |

**Internal — bound to `127.0.0.1` inside the container, not proxied:**

| Route | Why it stays internal |
|---|---|
| `POST /api/v1/detect`, `/api/v1/detect/batch` | scoring; bounded is not free, and there is no rate limiting |
| `POST /api/v1/explain` | scoring-adjacent, same reason |
| `/api/v1/demo/*` | replay control — starting and stopping runs |
| `/api/v1/system/status`, `/api/v1/model/info`, `/api/v1/rules` | model and system information |
| `/ready` | names which scientific component is unavailable and why: operator diagnostics, not an anonymous caller's business |
| `/version` | build metadata |
| `/docs`, `/openapi.json` | Swagger / OpenAPI |

Streamlit answers `200` with its own single-page-application shell for **any**
unrecognised path, so a status code proves nothing about this policy. Every
routing assertion in the test suite inspects the **response body**.

### Self-hosting on a VPS

An **additive** overlay on the Compose stack. `docker compose up --build` is
unchanged; a server adds one file:

```bash
cp .env.deploy.example .env.deploy      # hostname, publish spec, routing policy
docker compose --env-file .env.deploy \
  -f compose.yaml -f compose.deploy.yaml up -d --build
```

```
internet → 80/443 → proxy (Caddy) → dashboard → api → frozen serving bundle
```

The application ports stop being published — not narrowed, removed
(`ports: !reset null`, because Compose *appends* sequences when it merges files).
Two TLS modes: `PAD_SITE_ADDRESS=:80` for an IP-only smoke test, or a hostname
for automatic HTTPS over ACME. No certificate, private key, or real hostname
exists in this repository.

Full references: **[docs/render-deployment.md](docs/render-deployment.md)** and
**[docs/deployment.md](docs/deployment.md)**.

---

## Testing and verification

```bash
# The full verification suite (mirrors CI)
bash scripts/verify.sh

# Individually
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest                          # unit + integration, coverage gate at 90%
uv build

# Larger checks, deselected by default
uv run pytest -m slow --no-cov

# Verbose, no coverage
uv run pytest -v --no-cov
```

The container suites live behind the `slow` marker and skip themselves when no
Docker daemon is reachable or the images have not been built:

```bash
docker compose build
uv run pytest -m slow tests/integration/test_docker_compose.py --no-cov
uv run pytest -m slow tests/integration/test_deployment_topology.py --no-cov

DOCKER_BUILDKIT=0 docker build -f Dockerfile.render -t pad-render:0.6.0 .
uv run pytest -m slow tests/integration/test_render_container.py --no-cov
```

The first brings the local stack up only if it is not already up and tears down
only what it started. The second brings up the **deployment** topology with the
proxy on an unprivileged loopback port and removes everything it created.

The architecture and security claims that need no daemon are ordinary unit
tests: `tests/unit/deployment/test_container_contract.py`,
`test_deployment_contract.py` and `test_render_contract.py` read the Dockerfiles,
both Compose files, all three Caddyfiles, `.dockerignore`, `.gitignore`, the
environment templates and the bootstrap script, and assert what they promise.

---

## Project structure

```
.
├── Dockerfile              builder + runtime base + api/dashboard targets
├── Dockerfile.render       build -> prepare -> verify -> prune -> one runtime image
├── compose.yaml            prepare -> api -> dashboard  (local)
├── compose.deploy.yaml     + proxy, ports un-published  (VPS overlay)
├── render.yaml             Render blueprint: one free web service
├── deploy/
│   ├── caddy/
│   │   ├── Caddyfile           VPS default policy: the console only
│   │   └── Caddyfile.api-docs  optional: + the API's read-only surface
│   └── render/
│       └── Caddyfile           single-container policy: console + /healthz
├── configs/
│   ├── data/       synthetic-{testing,demo,development,ml-development}.yaml
│   ├── features/   feature-{testing,demo,development,ml-development}.yaml
│   ├── detection/  rules-{testing,demo,development}.yaml
│   ├── ml/         model-{testing,demo,development}.yaml + two feature allowlists
│   └── {development,production,testing}.yaml
├── data/                   raw, interim, processed datasets (not tracked)
├── artifacts/              training artifacts (not tracked)
├── models/                 model artifacts (not tracked)
├── docs/                   the documentation index below
├── scripts/
│   ├── verify.sh                  the full verification suite
│   ├── prepare_demo_bundle.py     the offline pipeline the container runs once
│   ├── verify_serving_bundle.py   verifies a prepared bundle; decides nothing
│   ├── render_entrypoint.py       PID 1 for the single-container deployment
│   ├── generate_governance_docs.py regenerates the two generated documents
│   ├── start_demo.sh / stop_demo.sh
│   └── deploy/bootstrap_server.sh SERVER ONLY — prepares a fresh Ubuntu host
├── src/password_attack_detector/
│   ├── cli.py               root Typer CLI
│   ├── config.py            typed settings (pydantic-settings)
│   ├── exceptions.py        project exception hierarchy
│   ├── logging_config.py    structured logging (structlog)
│   ├── paths.py             centralized path management
│   ├── data/                schema, privacy, ingestion, synthetic generation
│   ├── features/            the point-in-time engine, baselines, splitting, leakage
│   ├── detection/           the nine rules, risk scoring, alerts, evaluation
│   ├── ml/                  catalog, training, calibration, champion, fusion, drift
│   ├── deployment/          the sealed serving bundle: write, verify, materialize
│   ├── api/                 the FastAPI serving layer
│   ├── replay/              the synthetic live replay
│   └── dashboard/           the Streamlit SOC console (API client only)
└── tests/
    ├── unit/                {api,dashboard,data,deployment,detection,features,ml,replay}
    └── integration/         end-to-end CLI, serving, replay, container and topology suites
```

### The three application packages

```
api/                              replay/                     dashboard/
├── app.py       factory, lifespan ├── enums.py    vocabularies ├── app.py        entrypoint
├── config.py    where, not which  ├── schemas.py  wire shapes  ├── config.py     presentation only
├── dependencies.py readiness gate ├── scenarios.py the catalog ├── contracts.py  re-declared wire
├── errors.py    22 stable codes   ├── store.py    bounded      ├── api_client.py the one door
├── schemas.py   request/response  ├── engine.py   state machine├── state.py      session memory
├── services.py  the composition   └── service.py  operations   ├── formatting.py rendering
└── routes/                                                     ├── theme.py      stylesheet
    health · detection · explain                                ├── components/
    replay · system                                             └── views/  eleven views
```

`views/` rather than `pages/`: Streamlit treats a `pages/` directory beside the
entrypoint as an automatic multipage app, which would build a second navigation
beside the real one.

### Console navigation

| Group | Views |
|---|---|
| **Primary** | Overview · Live Replay · Alerts · Analytics · Explainability · Drift Monitoring |
| **Advanced** | Detection Console · Authentication Events · Rule vs ML vs Hybrid · System & Model |
| **About** | About System |

Full reference: **[docs/dashboard.md](docs/dashboard.md)**.

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

The serving layer's `PAD_API_*` and the console's `PAD_DASHBOARD_*` variables can
only say **where** things are and **how much** is allowed — never **which**
model, threshold, or fusion strategy. Both settings classes carry an import-time
guard that refuses such a field name outright.

---

## CLI

Four command groups, wired into one root CLI. Every command supports `--help`,
returns non-zero on failure, refuses to overwrite without `--force`, and prints
no event identifier, pseudonym, coordinate, or absolute path.

```bash
uv run password-attack-detector --help

# Phase 1
uv run password-attack-detector version | doctor | show-config

# Data (7 subcommands)
uv run password-attack-detector data generate | ingest | validate \
  | profile | manifest | verify-manifest | schema

# Features (9 subcommands)
uv run password-attack-detector features catalog | split | fit-baseline \
  | transform | build | audit-leakage | validate | profile | verify-manifest

# Detection (6 subcommands)
uv run password-attack-detector detection catalog | run | validate \
  | profile | evaluate | verify-manifest

# Machine learning (14 subcommands)
uv run password-attack-detector ml catalog | audit-features | verify-manifest \
  | train | experiments | select | freeze-champion | predict | validate \
  | profile | evaluate | compare | explain | drift

# Deployment
uv run password-attack-detector deploy materialize | inspect
```

A worked end-to-end example, with every flag, is in
[docs/api.md](docs/api.md) §2 and in `scripts/prepare_demo_bundle.py`, which is
the pipeline the container actually runs.

---

## Security and privacy

- **No plaintext passwords, hashes, tokens, cookies, or real credentials** are
  stored, logged, or accepted at any layer. Credential material is refused under
  every spelling before any other validation runs, on the wire and again in the
  console's session state before a value could reach a browser preview.
- **No credential field is accepted** by any request schema. Schemas are strict
  (`extra="forbid"`), so a field the contract does not declare is refused
  outright.
- **No attack target can be supplied.** There is no host, URL, endpoint, or
  address field anywhere in the request surface, and an AST test refuses a
  network client in the replay package.
- **No filesystem path, model id, threshold, or fusion strategy is reachable
  from a request.** Deployment controls live in environment settings that are
  guarded at import against exactly those names.
- **No training, promotion, freeze, or upload route exists.** The complete route
  inventory — fifteen operations — is pinned as an equality against the running
  application's own schema, so adding one is a visible edit to that set.
  `api/services.py` separately carries an import-time guard refusing the *names*
  of the functions that would write frozen state.
- **No arbitrary code execution**: model artifacts are numbers, not objects — no
  `pickle`, no `joblib`, no estimator reconstructed to score, and no `eval`,
  `exec`, or dynamic import in any artifact reader.
- Source identifiers are pseudonymized via HMAC-SHA256 before storage.
  **Pseudonymization reduces exposure but does not guarantee anonymity.**
- Prohibited sensitive field names are rejected at the ingestion header/key level
  before any values are read.
- Ground-truth labels are always stored separately from canonical events.
- `PAD_PSEUDONYMIZATION_KEY` never appears in YAML, manifests, logs, or exception
  messages. Synthetic data never calls the pseudonym service and does not need it.
- Containers run **non-root** (uid 10001) with read-only root filesystems, all
  capabilities dropped, `no-new-privileges`, no host networking, and **no Docker
  socket**. The one capability granted anywhere is `NET_BIND_SERVICE`, on the VPS
  proxy only, because 80 and 443 are below 1024.
- The serving bundle is mounted read-only (Compose) or copied root-owned into a
  layer the serving account cannot write (Render). **Scientific state is
  read-only at runtime in both.**
- `/healthz` on the public deployment carries a status string, the service name,
  and the package version. Nothing about the host, the model, the artifacts, or
  which component is unhealthy.

See [docs/privacy-model.md](docs/privacy-model.md) and
[SECURITY.md](SECURITY.md) for full detail.

---

## Limitations

Read this section before drawing any conclusion from a demonstration.

### About the data and the evidence

- **The data is synthetic.** Every figure this repository can produce was
  measured on traffic this repository generated, under a declared scenario
  configuration. It is evidence that the pipeline behaves as specified, **not
  evidence about real authentication systems**, and it does not demonstrate
  real-world detection effectiveness.
- **This is not production authentication infrastructure.** It observes and
  scores event records. It does not authenticate anyone, sit in a login path,
  block a session, or integrate with an identity provider.
- **Simplified attack simulation.** The generator does not capture the
  distributional properties of real login traffic, and thresholds tuned on
  generated data reflect the generator's parameters.
- **Reproducibility is bounded by the committed `uv.lock` environment.** A
  different library version may produce different output for the same seed.
- **A frozen champion is a subject, not a result.** `champion.lock` says what an
  evaluation was permitted to run. It carries no metric.
- **Structural validity is not predictive quality**, **calibration is internal**
  to the synthetic validation distribution, **attribution is descriptive, not
  causal**, and **drift is monitoring evidence, not model correctness** — nothing
  retrains, promotes, or rethresholds on a finding.
- **The anomaly track is experimental** and the novel-anomaly holdout
  (`supervised_training_eligible=False`) is a generalisation probe, not a
  labelled training class.

### About the deployment

- **Render Free spins down after inactivity.** The first request after a quiet
  period pays a cold start of roughly one to two minutes. No uptime guarantee is
  offered.
- **Free-tier resources are constrained**: 512 MiB of memory and a fraction of a
  CPU. Throttling changes how long a demonstration takes, not what the detector
  decides — every replay verdict measured at 0.1 CPU was identical to the same
  verdict at 0.5 CPU.
- **Nothing persists.** The console's history is one browser session; a replay
  run lives in one API process's memory and is cleared by a restart. There is
  **no persistent alert database**, no event store, and no serving drift report.
  A restart preserves the champion and clears replay history, which the console
  says it will.
- **There is no public raw detection API.** Scoring, explanation, replay control,
  readiness and Swagger are internal. A viewer interacts with the console.
- **There is no authentication and no rate limiting anywhere in this system**,
  and none was added for a demonstration — a half-built auth platform is a larger
  surface than the one it closes. Rate limiting was deliberately deferred rather
  than built on a proxy module that would replace a pinned official image with
  one this project has to patch itself.
  [docs/deployment.md](docs/deployment.md) §15 states that residual risk rather
  than implying it is covered.
- **No real password or credential is ever collected**, by the demo or by
  anything else here. There is no field that could carry one.

### `PAD-CS-001` and `PAD-ATO-001` cannot fire on a live request

Both gate on a fitted **behavioural baseline** that the serving path does not
load. `api/services._feature_rows()` constructs the feature engine with no
baseline argument, so `user_in_baseline` and `source_in_baseline` are always
`False` and every `is_new_*_for_user` novelty flag is `None`. `PAD-CS-001`
therefore returns *insufficient data* (`ACCOUNT_ABSENT_FROM_BASELINE`) on every
live window, and `PAD-ATO-001` can never reach its `min_novel_context_count` of 2.

**Adding a baseline was audited and measured not to help.** A baseline fitted
from the demo deployment's own TRAIN split was loaded and the two scenarios run
through it: `user_in_baseline` stayed `False` and all five novelty flags stayed
`None`. The replay catalog's identities are UUIDv5 pseudonyms derived from the
scenario itself, so **no** training population contains them, whatever dataset a
deployment trains on. Making those rules fire would mean drawing scenario
identities from a deployment's own training population, coupling a
content-addressed catalog to one dataset; carrying a baseline in the bundle is
separately a schema change that bumps `BUNDLE_SCHEMA_VERSION` and extends the
fingerprint chain.

**No threshold was moved and no baseline was synthesised to make a demonstration
look better.** This is documented at
[docs/deployment.md](docs/deployment.md) §16, [docs/docker.md](docs/docker.md)
§14, [docs/api.md](docs/api.md) §12, [docs/dashboard.md](docs/dashboard.md) §11
and [docs/live-replay.md](docs/live-replay.md) §14.

### About the detection layers

- **`signal_strength` and `risk_score` are not probabilities.** They are bounded
  ordinal magnitudes, and **evidence is not causal proof**.
- **Account-takeover and impossible-travel outputs are indicators.** Travel and a
  device replacement reproduce the first; a VPN or carrier gateway reproduces the
  second. Confirming either requires investigation this system does not do.
- **Detection quality is bounded by feature quality**: a behaviour the feature
  layer does not express is one no rule can detect.
- **Alert suppression retains complete aggregate accounting**, but a suppressed
  event is only guaranteed to have been *counted* — not to have been
  uninteresting.
- `detection profile` rebuilds a quality report from published artifacts and so
  reports engine-only counters as **unavailable**, never as zero.
- **Baseline and alert artifacts hold pseudonymous per-entity state.** They are
  written only to git-ignored paths with restrictive permissions and must never
  be committed; real-data equivalents require protected storage.

See [docs/detection-limitations.md](docs/detection-limitations.md) and
[docs/model-card.md](docs/model-card.md).

---

## Documentation

**Start here**

| File | Contents |
|---|---|
| [docs/demo.md](docs/demo.md) | The presentation walkthrough: what to click, in what order, and what to say |
| [docs/phase6-acceptance.md](docs/phase6-acceptance.md) | The Phase 6 acceptance record and the v0.6.0 release decision |

**The application**

| File | Contents |
|---|---|
| [docs/api.md](docs/api.md) | The HTTP serving layer: endpoints, constraints, privacy, limits |
| [docs/dashboard.md](docs/dashboard.md) | The analyst console: API boundary, views, session limits, offline behaviour |
| [docs/live-replay.md](docs/live-replay.md) | The synthetic replay: scenarios, determinism, run lifecycle, bounds |
| [docs/docker.md](docs/docker.md) | The local containerized demonstration: images, preparation job, volumes, security |
| [docs/deployment.md](docs/deployment.md) | The VPS perimeter: proxy, routing policy, TLS, firewall, per-rule availability |
| [docs/render-deployment.md](docs/render-deployment.md) | The Render adapter: one service, build-time bundle, supervision, `$PORT`, memory |

**Data and features**

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

**Detection and models**

| File | Contents |
|---|---|
| [docs/rule-contract.md](docs/rule-contract.md) | What a rule may read, must return, and guarantees |
| [docs/rule-catalog.md](docs/rule-catalog.md) | Generated: every registered rule |
| [docs/risk-scoring.md](docs/risk-scoring.md) | Correlation-aware scoring and its proven properties |
| [docs/alert-lifecycle.md](docs/alert-lifecycle.md) | Grouping, scope, suppression, escalation |
| [docs/rule-evaluation.md](docs/rule-evaluation.md) | Metrics, split discipline, no threshold tuning |
| [docs/detection-limitations.md](docs/detection-limitations.md) | What the rule layer does not do |
| [docs/model-contract.md](docs/model-contract.md) | What an ML artifact must carry and guarantee |
| [docs/model-catalog.md](docs/model-catalog.md) | Generated: every declared model |
| [docs/champion-selection.md](docs/champion-selection.md) | Validation-only gates, ranking, and the freeze |
| [docs/experiment-ledger.md](docs/experiment-ledger.md) | The append-only run record |
| [docs/prediction-artifacts.md](docs/prediction-artifacts.md) | Batch inference outputs and their identity |
| [docs/test-evaluation.md](docs/test-evaluation.md) | The locked TEST protocol, fusion, comparison |
| [docs/explainability.md](docs/explainability.md) | Deterministic attribution and its limits |
| [docs/drift-monitoring.md](docs/drift-monitoring.md) | The frozen reference profile and drift semantics |
| [docs/model-card.md](docs/model-card.md) | Generated: purpose, scope, prohibited use, limitations |
| [docs/phase5-acceptance.md](docs/phase5-acceptance.md) | Generated: the Phase 5 acceptance report |

---

## Release

**Current version: 0.6.0 — *Deployable security analytics application***

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
and reported by `password-attack-detector version`, `GET /version`,
`GET /health`, and every container image tag.

---

## License

MIT — see [LICENSE](LICENSE).
