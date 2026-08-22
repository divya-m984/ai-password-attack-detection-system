# Live replay

A synthetic demonstration layer that makes the detection system *watchable*: it
takes a reviewed, deterministic scenario from its own catalog, emits it one event
at a time into the existing serving path, and records what the frozen rules, the
frozen model and the frozen hybrid said about each step.

Everything on this page is about **presentation**. Nothing in the replay layer
computes a feature, evaluates a rule, applies a threshold, calibrates a score, or
fuses two verdicts. It decides *when* a detection happens and records *what came
back*, and that is the whole of it.

---

## 1. Safety, first

This is a defensive demonstration facility. It never:

* attacks a real authentication service, or any service;
* sends a request to a login endpoint, an external host, or the public internet;
* generates, reads, stores, or transmits a password, a password list, a hash, a
  token, or any other credential;
* performs credential stuffing, brute forcing, or password cracking against
  anything;
* scans a network or connects to an arbitrary remote host;
* accepts a scenario, an event stream, a schedule, or an address from a caller.

Every scenario is a **fabrication**: a fixed list of authentication events
describing activity that did not happen, involving accounts, devices and sources
that do not exist, replayed into a service the operator is running themselves.

Three of those properties are enforced structurally rather than by review:

| Property | Where it is enforced |
|---|---|
| No scenario carries a credential-shaped field name | Import-time guard in `replay/scenarios.py`, using the project's own ingestion scanner |
| No literal address outside RFC 5737 / RFC 3849 appears | Import-time guard in `replay/scenarios.py` |
| No replay module imports a network client | AST test over every module in `tests/integration/test_replay_security.py` |
| The replay layer holds no detection capability | Import-time guard in `replay/service.py`, plus namespace assertions |

The built-in scenarios do not even use the documentation address ranges: every
event carries a pseudonymous `source_id`, so no address reaches the wire at all.

---

## 2. Architecture

```
Synthetic scenario (reviewed catalog, deterministic)
        ↓
Replay engine            state machine + bounded pace
        ↓
DetectionWindowRequest   the same schema an HTTP body is validated by
        ↓
detect_single()          the same function POST /api/v1/detect calls
        ↓  Phase 4 rules + frozen Phase 5 model + frozen Phase 5 fusion
Replay timeline record   embeds the serving layer's own AnchorDetection
        ↓
FastAPI /api/v1/demo/*
        ↓
Dashboard API client     the single client the console already had
        ↓
Live Replay view
```

The package:

```
src/password_attack_detector/replay/
    __init__.py     what the package is, and what it deliberately cannot do
    enums.py        the closed vocabularies: scenario, state, pace
    schemas.py      the wire contract
    scenarios.py    the reviewed catalog -- deterministic events, nothing else
    store.py        bounded, process-local, non-persistent run storage
    engine.py       the state machine and the pace; the detector is injected
    service.py      the operations the API namespace is a shell over
```

### There is no second detection path

The engine's detector is a `StepDetector` protocol — one awaitable taking a
window and an anchor. The binding lives in
`api/services.build_replay_detector()`, beside the `detect_single` it wraps, and
it does exactly this:

1. build a `DetectionWindowRequest` from the events emitted so far, with
   `anchor_selection="explicit"` naming the newest one;
2. dispatch `detect_single(runtime, request)` to a worker thread;
3. return `response.anchor`.

So a replayed step meets every check a client request meets — the credential-field
scan, the canonical event rules, the ordering and uniqueness contracts, the batch
ceiling, the readiness gate — because it *is* one.
`tests/integration/test_replay_detection.py` asserts the equality directly:
several steps' windows are reconstructed, posted to `POST /api/v1/detect`, and
compared field by field against what the replay recorded.

The detection call is dispatched off the event loop because it is CPU-bound.
`detect_single` builds a fresh `FeatureEngine` per call, so two concurrent runs
cannot see one another's history — the Milestone 1 request-isolation guarantee is
inherited rather than re-established.

---

## 3. The scenario catalog

Seven scenarios, each a shape of authentication behaviour the detection system
was built to reason about.

| Scenario | Events | Span | Rules it fires | Severity floor |
|---|---|---|---|---|
| `normal_activity` | 12 | 295 s | *(none)* | — |
| `brute_force` | 30 | 290 s | `PAD-BF-001`, `PAD-BOT-001` | `high` |
| `password_spraying` | 24 | 115 s | `PAD-BOT-001`, `PAD-PS-001` | `high` |
| `credential_stuffing` | 21 | 280 s | `PAD-PS-001` | `medium` |
| `account_takeover` | 14 | 160 s | `PAD-BF-001`, `PAD-BF-002` | `medium` |
| `bot_activity` | 24 | 276 s | `PAD-BOT-001` | `low` |
| `mixed_attack` | 29 | 200 s | `PAD-BF-001`, `PAD-BF-002`, `PAD-PS-001` | `medium` |

Every entry in the "rules it fires" column is asserted as an **equality** in
`tests/integration/test_replay_detection.py`, against the real frozen deployment.
Not a subset: a scenario cannot quietly acquire a rule nobody documented, and it
cannot claim one that never fires.

The severity column is a **floor**, published as `expected_severity_at_least` and
asserted as "at least this severe". A floor rather than an exact value because a
severity is a function of the rule configuration as well as of the events, and a
deployment that tuned a threshold should not fail a claim the catalog made about
someone else's. In practice `brute_force` reaches `critical` and
`normal_activity` stays at `low` with a zero risk score on every step, which
`test_normal_activity_produces_no_attack_demonstration` asserts exactly.

Each scenario publishes, through `GET /api/v1/demo/scenarios`:

* `scenario_id`, `name`, `description`, `purpose`
* `scenario_schema_version`, `revision`, `scenario_fingerprint`
* `event_count`, `duration_seconds`
* `expected_rule_ids`, `expected_rule_families`, `expected_severity_at_least`
* `demonstrates_ml`, `demonstrates_hybrid`
* `limitations`

The catalog publishes **no event bodies**. A client picks a scenario by name.

### Why every scenario fits inside five minutes

Every windowed rule condition in the catalog reads a window of five minutes or
more, so a scenario whose whole timeline fits in five minutes produces the same
counts under the repository's demo rule configuration (which points the rules at
5-minute windows for a CI-sized dataset) and under the rule catalog's own
defaults. A scenario spanning an hour would fire different rules under the two,
and the published expectations would be true of only one deployment.

### Two rules that cannot fire, and why

`PAD-CS-001` (credential stuffing) gates on `user_in_baseline`, and `PAD-ATO-001`
(account takeover) counts the `is_new_*_for_user` novelty flags. Both come from a
**fitted behavioural baseline**.

The serving path computes point-in-time features from the supplied window alone
and loads no baseline artifact, so both rules report *insufficient data* on every
live request — in every scenario, whatever the events look like. That is a
property of the serving path, not of these scenarios: it applies equally to a
window submitted by hand through `POST /api/v1/detect`.

The two affected scenarios therefore say so in their published `limitations`, and
their `expected_rule_ids` name the rules that do fire:

* `credential_stuffing` is reported by `PAD-PS-001`. Stuffing and spraying are
  close relatives at the rule layer — both are one source touching many accounts
  with a high failure share — and what separates them is exactly the
  baseline-derived "unfamiliar context for *this account*" signal. The scenario
  still demonstrates the contrast usefully: `PAD-BOT-001` fires on the spraying
  scenario and not on this one, because four rotating clients is not one tool.
* `account_takeover` is reported by `PAD-BF-001` (the burst) and `PAD-BF-002`
  (the success that followed it), which is the sequence an analyst actually
  triages.

**This was not addressed by loosening a rule or moving a threshold.** The Phase 4
configuration and the Phase 5 artifacts are frozen, and a demonstration is not a
reason to unfreeze one. `test_the_baseline_dependent_rules_never_fire_on_a_live_request`
records the finding as a test.

---

## 4. Determinism

A scenario's identity binds:

* the scenario schema version,
* the scenario identifier,
* the scenario revision,
* every event, canonically serialised.

`scenario_fingerprint` is the SHA-256 of that. Two processes that agree on it
replayed the same events, in the same order, with the same timestamps.

Nothing in a scenario is drawn from a clock or a random source. Event identifiers
are UUIDv5 values over a fixed namespace; event times are fixed offsets from a
fixed base instant (`2026-03-04T12:00:00Z`).

### The scientific timeline is not the wall clock

Two timestamps appear on every timeline record, and they are different things:

| Field | What it is |
|---|---|
| `source_event_time` | The scenario event's own fixed timestamp. Every point-in-time feature is computed against this. |
| `emitted_at` | When this process presented the step. Moves with the replay pace. **Nothing scientific reads it.** |

So:

> the same scenario + the same serving bundle + the same application version
> produces the same final detection outcomes whether it is replayed slowly,
> normally, quickly, or instantly.

This is asserted twice: in `tests/unit/replay/test_engine.py` against a stub
detector, and in `tests/integration/test_replay_detection.py` against the real
frozen deployment, where a run at `fast`, `normal` and `slow` is compared field
by field against the same run at `instant`.

---

## 5. The run state machine

```
CREATED ──▶ RUNNING ──▶ COMPLETED
   │           │
   │           ├──────▶ STOPPED
   │           └──────▶ FAILED
   ├──────────────────▶ STOPPED
   └──────────────────▶ FAILED
```

`COMPLETED`, `STOPPED` and `FAILED` are **absorbing**: they have no outgoing
edge. A finished run does not restart, and a second execution of the same
scenario is a *new run with its own identity*. `replay/enums.py` asserts the
absorbing property at import.

A self-transition is not an edge either. The idempotence a caller wants from
"stop a stopped run" is *the request succeeding without changing anything*, which
is a decision for the service layer rather than a hole in the graph.

Failure reason codes (stable; never a message, a path, or a stack frame):

| Code | Meaning |
|---|---|
| `replay_detection_failed` | A step could not be scored |
| `replay_deadline_exceeded` | The run outlived `max_run_seconds` |
| `replay_record_limit_reached` | The per-run record bound was reached |
| `replay_shutdown` | The process shut down while the run was active |

### Two identities, and the difference

| Identity | What it is | Scope |
|---|---|---|
| `scenario_fingerprint` | **Content** identity. The same on every machine for the same catalog entry. | Global |
| `run_id` | **Instance** identity. Distinguishes two simultaneous executions of one scenario. | This process only |

`run_id` is an opaque `run_<32 hex>` token. No host path, process identifier,
port, user, or secret contributes to it. It is random rather than a counter: a
counter would publish how many demonstrations this process has served.

---

## 6. Pace

A pace is a **word**, never a duration and never an expression.

| Pace | Delay between steps |
|---|---|
| `instant` | none |
| `fast` | 0.25 s |
| `normal` | 0.75 s |
| `slow` | 1.75 s |

Bounded at 2.0 s by `MAX_STEP_INTERVAL_SECONDS`, asserted at import. Accepting a
number of seconds would let one request occupy a run slot for as long as it
liked; accepting an expression would be worse.

The delay is an **interruptible wait on the run's stop event**, not a sleep — so
a stop at `slow` pace takes effect immediately rather than after up to two
seconds.

---

## 7. The run store

**Process-local. Demo-oriented. Non-persistent. Bounded.**

Restarting the API clears every run and every timeline. Two API processes behind
a load balancer would not see each other's runs. Nothing is written to disk, to a
database, or to a queue.

| Bound | Default | What happens at the bound |
|---|---|---|
| `max_active_runs` | 4 | `API019`, and **no existing run is stopped to make room** |
| `max_retained_runs` | 24 | The oldest *finished* run is evicted; an active run never is |
| `max_records_per_run` | 256 | The run fails with `replay_record_limit_reached` |
| `max_run_seconds` | 300 | The run fails with `replay_deadline_exceeded` |
| `max_timeline_page` | 100 | The page is clamped and `more_expected` stays true |

The asymmetry is the design: a store that silently killed the run you were
watching to serve a request you did not make would be worse than one that said
no.

### Concurrency

* One `asyncio.Lock` in the store, held only around mutations.
* One background task per run, one stop event per run.
* Two clients cannot start the same run twice — the transition happens *before*
  the task is scheduled, and the state machine has no `running → running` edge.
* Stop is idempotent and affects only the addressed run.
* A completed run cannot become running again.
* Reading a timeline takes a snapshot under the lock, so a page describes one
  moment rather than two.
* Every active run is cancelled and recorded as stopped in the application's
  lifespan shutdown.

---

## 8. Endpoints

All under `/api/v1/demo`, tagged **Demo** in Swagger.

### `GET /api/v1/demo/scenarios`

The reviewed catalog. Answered even where replay is switched off, so a client can
see what *would* be runnable.

### `POST /api/v1/demo/runs`

```json
{ "scenario_id": "brute_force", "pace": "normal" }
```

**Two fields, both closed vocabularies, `extra="forbid"`.** There is no field
here for an event list, a source address, an external target, a filesystem path,
a model identifier, a threshold, a fusion strategy, a schedule, or a credential —
and offering one is a refusal rather than a silently ignored key.

`201` with the run document: `run_id`, `scenario_id`, `scenario_name`,
`scenario_revision`, `scenario_fingerprint`, `pace`, `state`, `event_count`,
`emitted_count`, `created_at`, `started_at`, `finished_at`, `next_sequence`,
`more_expected`, `failure_reason`, `summary`.

Gated on a **ready** runtime: a replay against a runtime that cannot serve
detection would fail on its first step, and refusing up front with `API010` says
what is actually wrong.

### `GET /api/v1/demo/runs`

The runs this *process* is retaining, newest first, without their timelines,
alongside the bounds it is operating under — so a caller can see that an absent
run may have been evicted rather than never have existed.

### `GET /api/v1/demo/runs/{run_id}`

One run's state and its summary.

### `GET /api/v1/demo/runs/{run_id}/timeline`

```
?after_sequence=12&limit=50
```

Returns records with `sequence > after_sequence`, bounded by `limit`.

### `POST /api/v1/demo/runs/{run_id}/stop`

Stops the addressed run and no other. Idempotent. Records already emitted are
preserved. Nothing is killed at the process level.

The response is **honest**: `stop` awaits the run's task before answering, so by
the time a client sees the response the run is terminal and its timeline is
final. A client that polls immediately afterwards cannot see a record appear
after the stop it was told had happened.

---

## 9. The timeline cursor

A monotonically increasing `sequence`, from 1, one per scored step.

```
GET …/timeline                          → records 1..50, next_sequence=50, more_expected=true
GET …/timeline?after_sequence=50        → records 51..73, next_sequence=73, more_expected=false
GET …/timeline?after_sequence=73        → records [],     next_sequence=73, more_expected=false
```

`more_expected` is true when **either** this page was truncated **or** the run has
not reached a terminal state. That single boolean is everything a polling client
needs to decide whether to poll again; when it goes false, the client stops.

The cursor advances to the last record delivered and stays put on an empty page.
Advancing it past a gap would silently skip records that arrived between polls.

### The record contract

Each record carries `sequence`, `run_id`, `scenario_id`, `replay_state`,
`event_index`, `window_event_count`, `source_event_time`, `emitted_at`,
`authentication_outcome`, and `detection` — the serving layer's own
`AnchorDetection`, embedded **verbatim**.

Embedding rather than re-declaring is deliberate: the three layers stay in their
three separate objects, the rule layer's ordinal magnitude stays beside the
model's own score without either being combined with the other, and a field that
is public-safe in a detection response is public-safe here for the same reason.
Re-declaring those fields would create a second opinion about which of them may
be published.

A record never carries a credential, a secret, a raw feature vector, a model
coefficient, a tree array, an HMAC key, an artifact path, a server filesystem
path, or a private exception message.

`window_event_count` is how many events the detection window carried at that
step — every event emitted so far, which is the anchor's own strictly-prior
history plus the anchor. **No history is fabricated.**

---

## 10. Readiness and system status

Replay is an **optional demonstration facility**, not a detection layer.

`GET /api/v1/system/status` reports:

| Field | Meaning |
|---|---|
| `replay_enabled` | Whether this deployment serves the replay endpoints |
| `replay_available` | Whether a run can actually be started right now |
| `replay_required` | Whether readiness depends on it |
| `replay_unavailable_reason` | Stable code when it is enabled and not available |
| `replay_scenario_count` | Scenarios in the built-in catalog |
| `max_active_replay_runs` | Concurrent runs this deployment admits |

`GET /ready` gains a `replay` component with `required: false` by default.

**The decision, and why.** A detection service that refused to serve because its
optional demonstration history could not initialise would have its priorities
backwards. So `replay_required` defaults to `false`: replay failing leaves the
detector ready and every replay endpoint refusing with `API016`. A deployment
that exists *only* to demonstrate can set `PAD_API_REPLAY_REQUIRED=true` and get
the opposite behaviour.

`/health` is untouched. Liveness stays cheap and uninformative; whether an
optional subsystem came up is what a status document is for.

---

## 11. Error codes

Added without changing any existing code:

| Code | HTTP | Meaning |
|---|---|---|
| `API016` | 503 | Replay subsystem not available on this deployment |
| `API017` | 404 | No such scenario in the built-in catalog |
| `API018` | 404 | No such run in this process |
| `API019` | 429 | A replay bound is reached; nothing was started or extended |
| `API020` | 409 | The run cannot make that transition from the state it is in |
| `API021` | 422 | The timeline cursor is not a position a cursor can occupy |

Two of the status choices are deliberate:

* `API019` is **429**, not 503: the deployment is healthy and is refusing *this*
  request because a bound is reached, which is a different thing from the service
  being unable to serve.
* `API020` is **409**, not 422: the body is perfectly valid, and what refuses it
  is the state the addressed run happens to be in.

An unknown run, an evicted run, a run from before a restart, and a deployment
with replay switched off are four situations an operator would investigate
differently, so they are not collapsed into one 404.

---

## 12. The dashboard

A tenth navigation view, **Live Replay**, sitting directly under the Detection
Console — the two are the same act at two scales.

It shows: scenario, pace, status, progress, events emitted / total, active fusion
strategy, and current severity; a live timeline table; a cumulative layer-activity
chart; and a **Demo run summary** on completion.

Controls: **▶ Start replay**, **■ Stop replay**, **↻ Refresh**.

### Polling

While a run is active, the timeline block is a Streamlit fragment with a fixed
1.5-second interval. There is no `while` loop and no busy wait.

The interval is a constant tied to the *pace* vocabulary rather than to the
console's `refresh_seconds` health-refresh preference, because what it has to
keep up with is a replay step. The service's slowest pace puts 1.75 s between
steps.

The moment the service reports the run terminal, the fragment is not used at all
and the page stops calling the backend until somebody presses something.
`poll_interval()` is the whole policy, extracted so it can be tested without
Streamlit running, and it returns `None` for a terminal run.

Start and Stop are buttons **outside** the polling fragment. A run identifier is
written into session state the instant one is created and the Start control is
disabled while a run is attached and active, so a page that reruns — which
Streamlit does constantly — cannot start a second run behind the first. No POST
is ever sent on the console's own initiative.

### Where replay data appears elsewhere

| View | What it shows | How it is labelled |
|---|---|---|
| **Overview** | The attached run's state and summary | "Attached demo replay run" panel, only when one is attached |
| **Authentication Events** | The run's emitted steps | Separate "Server-side demo replay run" section |
| **Security Alerts** | The run's flagged steps | Separate section, never merged into the session table |
| **Attack Analytics** | Optionally the run's records | An explicit **Data source** selector, defaulting to the manual session |

**The two sources are never silently merged.** Manual submissions are what this
browser tab sent; a replay run is something happening on the server. They have
different origins and different lifetimes, so every replay-derived record carries
a `replay:` scenario prefix and every page that could show both makes the viewer
choose.

The console still cannot detect anything. It gained five typed methods on the
*same* API client it already had; `tests/integration/test_dashboard_security.py`
still asserts, by walking every module's syntax tree, that no dashboard module
imports the ML, detection, feature, deployment, or data packages, and that
`api_client` remains the only module in the package that imports `httpx`.

---

## 13. Local demo

### One command

```bash
docker compose up --build
```

Then open <http://localhost:8501>, go to **Live Replay**, and work through steps
5–11 below against `localhost` instead of `127.0.0.1`. The containerized
deployment serves `configs/detection/rules-demo.yaml`, which is the same rule
configuration the published `expected_rule_ids` were proved against — a unit test
asserts the two are equal, so what the catalog says about a scenario is true of
what the container computes. See [docker.md](docker.md).

### Or two terminals

**Terminal 1 — the API:**

```bash
uv run uvicorn password_attack_detector.api.app:app \
  --host 127.0.0.1 --port 8000
```

With a frozen deployment, set the artifact locations first — see
[api.md §10](api.md).

**Terminal 2 — the console:**

```bash
uv run streamlit run src/password_attack_detector/dashboard/app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Then:

1. `http://127.0.0.1:8000/health` → `200`
2. `http://127.0.0.1:8000/ready` → `200`
3. `http://127.0.0.1:8000/docs` → Swagger, with the **Demo** tag
4. `http://127.0.0.1:8501` → the console, header showing **API ONLINE** and
   **SYSTEM READY**
5. **Live Replay** → the catalog loads from the API
6. Choose `normal_activity`, pace `fast`, press **▶ Start replay** → the timeline
   fills, nothing is flagged, the run completes
7. Choose `brute_force`, pace `normal` → rule detections appear as the burst
   lengthens, the model decides on every step, the frozen **stacked** hybrid
   fuses them, severity climbs to `critical`
8. Choose `mixed_attack` → several kinds of finding on one timeline
9. Press **■ Stop replay** mid-run → the run ends and emits nothing further
10. Stop the API → the console shows **API OFFLINE** with no traceback
11. Restart the API → a new replay can be started

Or from the command line:

```bash
curl -s localhost:8000/api/v1/demo/scenarios | jq '.scenarios[].scenario_id'
RUN=$(curl -s -XPOST localhost:8000/api/v1/demo/runs \
        -H 'content-type: application/json' \
        -d '{"scenario_id":"brute_force","pace":"instant"}' | jq -r .run_id)
curl -s "localhost:8000/api/v1/demo/runs/$RUN" | jq .summary
curl -s "localhost:8000/api/v1/demo/runs/$RUN/timeline?after_sequence=25" \
  | jq '.records[] | {sequence, severity: .detection.severity, rules: .detection.rule.fired_rule_ids}'
```

Stop both servers afterwards.

---

## 14. Current limitations

* **No persistence.** Runs and timelines live in one process's memory. Restarting
  the API clears all of them. This is a demonstration facility, and nothing in
  the API documents, this page, or the console presents it as a history.
* **Process-local.** Two API processes behind a load balancer would not see each
  other's runs, and a run identifier means nothing outside the process that
  minted it.
* **Synthetic only.** There is no path for real traffic to enter a replay, and
  there should not be: the catalog is the whole input surface.
* **The scenarios are fixed.** Seven of them, reviewed, in source. There is no
  upload, no parameterisation beyond the pace, and no editor.
* **Two rules cannot be demonstrated, and loading a baseline would not change
  that.** `PAD-CS-001` and `PAD-ATO-001` need a fitted behavioural baseline the
  serving path does not load — see §3. Milestone 4 measured what would happen if
  it did: a baseline fitted from a deployment's own TRAIN split was loaded into a
  feature engine and these scenarios were run through it, and `user_in_baseline`
  came back `False` with every `is_new_*_for_user` flag `None`, exactly as
  before. The reason is in this layer's design rather than in the bundle's: a
  scenario's identities are content-addressed pseudonyms derived from the
  scenario itself, so that the same catalog means the same thing in every
  deployment — and no training population, anywhere, contains them. See
  [docker.md](docker.md) §14.
* **Rule expectations are configuration-dependent.** The published
  `expected_rule_ids` are proved against the repository's demo rule
  configuration. A deployment that changes a rule's windows or thresholds may see
  different rules fire; the fingerprint tells you the *scenario* is the same, not
  that the deployment is.
* **No authentication or rate limiting.** As with the rest of the API, this is a
  local demonstration service. The replay bounds limit resource use; they are not
  an access-control mechanism.
* **Containerised, not deployed.** Milestone 4 packages the demonstration so one
  command starts it on one machine, with both ports bound to that machine's
  loopback interface. Nothing is published or reachable from another host.

---

## Related documents

| Document | Contents |
|---|---|
| [api.md](api.md) | The serving contract this layer replays through |
| [dashboard.md](dashboard.md) | The analyst console the Live Replay view lives in |
| [docker.md](docker.md) | The containerized deployment this runs inside |
| [rule-catalog.md](rule-catalog.md) | The rules the scenarios exercise |
| [rule-contract.md](rule-contract.md) | What a rule may read, including the baseline signals |
| [behavioral-baselines.md](behavioral-baselines.md) | The fitted baseline the serving path does not carry |
| [temporal-semantics.md](temporal-semantics.md) | Why a window is required, and what point-in-time means |
| [privacy-model.md](privacy-model.md) | The pseudonymization contract the scenarios imitate the shape of |
| [detection-limitations.md](detection-limitations.md) | What the detection layer does not do |
