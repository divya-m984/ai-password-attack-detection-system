# The SOC analyst console

Phase 6 Milestone 2. A dark, wide, analyst-oriented Streamlit console for the
detection system — and, structurally, **a client of the serving API and nothing
else**.

---

## 1. Architecture, and the boundary that defines it

```
Browser
   ↓
Streamlit console            src/password_attack_detector/dashboard/
   ↓
DashboardAPIClient           the one door to the backend
   ↓  HTTP
FastAPI service              src/password_attack_detector/api/
   ↓
Phase 3–5 detection engine   features → rules → frozen champion → frozen fusion
```

The console renders what the detection system decided. It does not decide
anything, and four separate properties make that a fact rather than a
convention:

**No detection capability is importable from it.** No module under
`dashboard/` imports `password_attack_detector.ml`,
`password_attack_detector.detection`, `password_attack_detector.features`,
`password_attack_detector.deployment`, or `password_attack_detector.data`. There
is no rule engine, no preprocessor, no model adapter, no calibrator, no
threshold, no fusion function, and no serving-bundle reader in this process. A
test walks every module's syntax tree and enforces it — an AST walk rather than a
grep, because this package's docstrings name those modules constantly, explaining
why they are not imported.

The only project module the console shares is
`password_attack_detector.exceptions`, and a test asserts that the shared set is
exactly that.

**There is one door.** `dashboard/api_client.py` is the only module that imports
`httpx`, and a test asserts it over the whole package. Pages call the client;
they do not construct HTTP.

**The wire contract is re-declared, not imported.** `dashboard/contracts.py`
describes the API's published documents in its own Pydantic models. Importing
`api.schemas` would have been the obvious move and the wrong one: it would make
the console a second consumer of the *server's objects* instead of its
**published contract**, and it would pull the whole detection and ML stack in
transitively — so a console that is supposed to be unable to score anything would
have a scorer one import away.

The cost of re-declaring is drift, so a test holds the two side by side and
asserts that the client declares no field the service does not send.

**No scientific setting exists.** `DashboardSettings` answers *which service to
ask and how to render the answer*. `PROHIBITED_SETTING_NAMES` lists the settings
that would cross that line — a model id, a threshold, a fusion strategy, an
artifact root, an API key — and an import-time guard fails if one is ever
declared.

### What that buys

The console cannot produce a second detection path, cannot re-derive a
threshold, cannot substitute a fusion strategy, and cannot show a number the
service did not publish. When the API is down it shows nothing rather than
something plausible.

---

## 2. Package layout

```
src/password_attack_detector/dashboard/
├── app.py            entrypoint: config, session, dispatch
├── config.py         DashboardSettings — location and presentation only
├── contracts.py      the wire shapes the console is prepared to read
├── api_client.py     the one door to the backend
├── state.py          what one browser session remembers
├── formatting.py     how values are rendered, and what they may be called
├── scenarios.py      safe synthetic templates and the console's vocabularies
├── theme.py          the stylesheet and the three escaping HTML helpers
├── components/
│   ├── header.py     title block, connectivity badges, sidebar navigation
│   ├── status.py     connectivity probing and the error-display contract
│   ├── metrics.py    the compact cards
│   ├── alerts.py     the three-layer result view and the session table
│   ├── charts.py     session and replay charts
│   └── replay.py     where replay data appears on a page that is not Live Replay
└── views/
    ├── overview.py        Overview
    ├── detection.py       Detection Console
    ├── replay.py          Live Replay
    ├── events.py          Authentication Events
    ├── alerts.py          Security Alerts
    ├── analytics.py       Attack Analytics
    ├── comparison.py      Rule vs ML vs Hybrid
    ├── explainability.py  Explainability
    ├── drift.py           Drift Monitoring
    └── system.py          System & Model
```

### Why `views/` and not `pages/`

Streamlit treats a directory named `pages/` beside the entrypoint script as an
*automatic* multipage app: every module in it becomes a navigation entry, ordered
by filename, whether or not it was meant to be one. This console drives its own
navigation from a declared label list, so the magic directory would have produced
a second navigation beside the real one, listing the same views under filenames
and calling their render functions with no arguments.

Every view exposes one `render(client, status, session)` function. One signature
for all ten, so no view acquires its own way of reaching the backend; a test
asserts the signature and asserts that the dispatch table's keys are exactly the
navigation labels.

---

## 3. Configuration

Loaded from `PAD_DASHBOARD_*` environment variables, an untracked `.env`, or
constructor arguments.

| Variable | Default | Meaning |
| --- | --- | --- |
| `PAD_DASHBOARD_API_URL` | `http://127.0.0.1:8000` | Base URL of the serving API |
| `PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS` | `5.0` | Per-request deadline, ≤ 60 |
| `PAD_DASHBOARD_REFRESH_SECONDS` | `30` | Suggested manual re-read interval, 5–3600 |
| `PAD_DASHBOARD_PAGE_TITLE` | `AI-Powered Password Attack Detection System` | Header title |

`api_url` is a **location and only a location**. It can make the console talk to
a different deployment; it can never make a deployment report a different model,
threshold, or strategy — those come out of the deployment's own frozen artifacts.
It is validated at load rather than at first request, so a typo is a startup
error naming the field instead of a connection failure that reads like the API
being down. Only `http` and `https` are admitted.

There is deliberately no `api_key`, `token`, or `secret` field. The service
accepts no credential material, so a credential here would be one with nowhere to
go and a file to leak out of. Where the API location is rendered onto a page, any
userinfo is stripped first.

---

## 4. The API client

`DashboardAPIClient` wraps every endpoint the console reads:

| Method | Endpoint |
| --- | --- |
| `health()` | `GET /health` |
| `readiness()` | `GET /ready` |
| `version()` | `GET /version` |
| `system_status()` | `GET /api/v1/system/status` |
| `model_info()` | `GET /api/v1/model/info` |
| `rules()` | `GET /api/v1/rules` |
| `detect(events)` | `POST /api/v1/detect` |
| `detect_batch(events)` | `POST /api/v1/detect/batch` |
| `explain(events)` | `POST /api/v1/explain` |

**A failure is a value, not an exception.** Every method returns an `APIResult`
carrying either a parsed document or a `Problem` with a stable `ProblemKind`:

| Kind | Means |
| --- | --- |
| `offline` | No connection could be made |
| `timeout` | Connected, and the deadline passed |
| `refused` | The service answered and refused the request |
| `not_ready` | Running, but cannot serve detection |
| `server_error` | The service failed while handling the request |
| `malformed` | Something answered, and it was not this API |

Pages branch on the kind and render fixed text for it. That is what makes the
offline experience uniform across nine views, and it is what keeps a traceback
off the page — Streamlit renders an uncaught exception into the browser in full.

**Nothing is invented.** There is no cached last-good response standing in for a
live one, no default document for an endpoint that failed, and no zero
substituted for a count nobody returned.

**A detection is never retried.** `GET` is not retried either — the retry a
viewer wants is the one they asked for by pressing a button — but `POST` is worse
than merely unhelpful to repeat: a request that timed out may well have been
evaluated, and re-sending it would double an entry in the session's own history.

**A URL is never echoed back.** `httpx` puts the full request URL in its
exception messages, so the client discards the exception message entirely rather
than trying to scrub it. A scrubber has to be right every time; a discard has to
be right once.

**Redirects are not followed.** The configured base URL names the service; a
redirect would let whatever answers move the console somewhere the operator did
not configure.

A `503` from `/ready` is a **document**, not a problem: the body is the readiness
report, and the report is exactly what the page needs to show.

---

## 5. The ten views

| View | What it shows | Backend needed |
| --- | --- | --- |
| **Overview** | Posture cards, readiness, architecture, champion, rule summary, session activity, attached demo run | yes |
| **Detection Console** | Templates, event builder, window, JSON preview, submit, result | yes to submit |
| **Live Replay** | A server-side synthetic scenario, replayed step by step | yes |
| **Authentication Events** | The window composed in this session, and a demo run's emitted steps | no |
| **Security Alerts** | This session's detection results, and a demo run's flagged steps | no |
| **Attack Analytics** | Charts over this session's results, a demo run's, or both — behind a source selector | no |
| **Rule vs ML vs Hybrid** | The three layers, the architecture diagram, the active strategy | yes |
| **Explainability** | Per-anchor model attribution via `POST /api/v1/explain` | yes |
| **Drift Monitoring** | The drift contract and its thresholds | no |
| **System & Model** | `/version`, `/system/status`, `/model/info`, `/rules` | yes |

Navigation labels are stable: the documentation, the tests, and any demo script
refer to a view by its label.

### Overview

Behaves like a SOC landing page, with every operational value read live from the
API — API status, readiness, active layers, the frozen fusion strategy, the
enabled rule count, the champion family.

It deliberately shows **no global event or alert total**. There is no persistent
event store and no alert database yet, so any such figure would be invented.
Where a console would normally put "12,503 events today", this page puts the
session's own count, labelled as the session's own count, and with no session
activity it says:

> No detection activity in this dashboard session.

### Detection Console

The primary demonstration feature. An analyst composes a bounded
`DetectionWindowRequest` and submits it.

The form writes into a mapping that is posted **verbatim**. The console never
pre-validates a window against its own copy of the rules, never computes a
feature, and never predicts a verdict: the API's schema is the authority on what
a valid event is, and a second opinion here would eventually disagree with it.

Controls: event timestamp, outcome, authentication method, failure reason, MFA
outcome, client type, response time, application, country, pseudonymous user /
device / session labels, and a source identity that is either a pseudonymous id
or an address from the RFC 5737 documentation ranges. The field names and
vocabularies are the API's own.

Two modes are offered side by side — a **form view** (a table of the composed
events, with remove and clear) and a **JSON preview** showing the exact body that
will be posted.

Before submission the page states:

```
Events: N · Anchor mode: … · API: connected / not connected
```

Anchor mode selects `last` (one anchor) or `all` (every event). The deployment's
`max_batch_events` is read from `/api/v1/system/status` and a window over it is
flagged before the request is made.

### Live Replay

The tenth view, added in Milestone 3, and the one page that calls the backend
repeatedly.

It starts a **server-side** replay run, polls its timeline, and stops it. The run
executes in the API process; this page holds a run identifier, a cursor, and
whatever records it has fetched. A browser reload loses the *view* and not the
run, and another tab polling the same identifier sees the same timeline — which
is exactly the difference from the manual session history two tabs apart.

Shown: scenario, pace, status, progress, events emitted / total, active fusion
strategy, current severity; the latest steps in colour; the full timeline as
`TIME | EVENT | RULE | RISK | ML | HYBRID | SEVERITY`; a cumulative
layer-activity chart; and a **Demo run summary** on completion, every figure of
which the service derived from that run's own timeline records.

Controls: **▶ Start replay**, **■ Stop replay**, **↻ Refresh**.

**Polling.** While a run is active the timeline block is a Streamlit fragment
with a fixed 1.5-second interval — no `while` loop, no busy wait. The interval is
a constant tied to the service's *pace* vocabulary rather than to
`refresh_seconds`, because what it has to keep up with is a replay step and not
an operator's idea of how often to re-read a status. The moment the service
reports the run terminal the fragment is not used at all and the page stops
calling the backend. `poll_interval()` is the whole policy and returns `None` for
a terminal run; it is a plain function so it can be tested without Streamlit
running.

**Nothing starts or stops on the console's initiative.** Start and Stop are
buttons *outside* the polling fragment; a run identifier is written into session
state the instant one is created; and Start is disabled while a run is attached
and active, so a page that reruns — which Streamlit does constantly — cannot
start a second run behind the first.

**The scenario catalog comes from the API.** So does the pace vocabulary's
meaning; the console restates the four words and a test pins them against the
service's own enumeration. There is no scenario editor and no JSON field on this
page.

### Where replay data appears elsewhere

| View | What it adds | Label |
| --- | --- | --- |
| **Overview** | The attached run's state and summary | "Attached demo replay run", only when one is attached |
| **Authentication Events** | The run's emitted steps | Separate "Server-side demo replay run" section |
| **Security Alerts** | The run's flagged steps | Separate section, never merged into the session table |
| **Attack Analytics** | Optionally the run's records | An explicit **Data source** selector, defaulting to the manual session |

**The two sources are never silently merged.** Manual submissions are what this
browser tab sent; a replay run is something happening on the server. They have
different origins and different lifetimes, so every replay-derived record carries
a `replay:` scenario prefix, every page that shows both says which is which, and
the analytics selector offers "this dashboard session", "active demo replay run",
or "both, labelled by source" — and defaults to the first.

Full contract: **[live-replay.md](live-replay.md)**.

### Safe synthetic scenarios

Three templates: **normal login activity**, a **brute-force-like failure burst**,
and a **password-spraying-like fan-out**.

These are documentation fixtures, not attacks. Each is a list of request bodies
describing authentication activity that did not happen, aimed at a service the
operator is running themselves. Nothing contacts a login endpoint, tries a
credential, or carries one. Identities are synthetic labels hashed into the
pseudonym shape the wire contract requires — this is *not* the project's keyed
pseudonymization, which protects real identifiers on the server; there is no real
identifier here to protect. Addresses come only from RFC 5737's documentation
ranges, and an import-time guard refuses any other.

Loading a template **fills the form and does nothing else.** The analyst still
presses submit, and the request still goes through the API like any other.
Nothing bypasses the service and nothing is scored locally.

Full replay automation is a later milestone. These are three fixed windows.

### The result view

Rule, model, and hybrid get one column each, at the same width, with their own
headings — because they are three separate verdicts produced by three separate
mechanisms, and the moment they share a cell somebody reads them as one number.

Four things the console will not do:

- **No arithmetic.** Nothing averages, weights, or blends the rule layer's
  ordinal magnitude with the model's probability. The fused verdict is the
  server's, produced by the frozen strategy.
- **No renaming.** A decision score is called a decision score. The label comes
  from the `score_kind` the service declared, never from which field happens to
  be populated. `risk_score` is rendered `72.4 / 100`, never with a percent sign.
- **No threshold arithmetic.** The frozen operating point is displayed as a
  number the service published. Nothing compares it to a score to re-derive a
  flag or infers where it sits from observed decisions.
- **No fabricated severity.** The severity shown is the Phase 4 ordinal the
  service returned for the anchor.

A **final security assessment** states, in plain language, what the system
decided and which layer decided it. Where the layers disagree, the disagreement
is stated rather than resolved — resolving it is the fusion strategy's job, it
has already been done on the server, and a second opinion rendered here would be
the console detecting.

### Explainability

Calls `POST /api/v1/explain`, which Milestone 2 added to the serving layer.

The endpoint decomposes the frozen model's decision for one anchor over the
transformed columns it read, using Phase 5's own `local_contributions` — the
scope-free primitive, which takes a verified model and a transformed matrix and
**no split argument**. That is what made the endpoint implementable without
touching Phase 5's semantics: attributing a live row needs no claim about which
experimental population it came from. Phase 5's `explain_predictions`, which does
take a `scope: MLSplit` and refuses TEST, is neither called nor importable from
the serving module — an import-time guard refuses the name.

Nothing is fitted, no threshold is read or moved, no calibrator is applied, and
no artifact is written. A champion family with no exact decomposition reports the
attribution unavailable with a reason rather than an approximation.

The page says three things carefully:

- the contributions sum to a **decision value** — a logit, a mean leaf score, or
  a threshold step — and never to the calibrated probability;
- the list is **ranked and truncated**, and the omitted count is shown beside it;
- a contribution is **not a cause**. The vocabulary stays flat: "contribution",
  never "driver", "importance", or "because".

The reported residual is `decision value − (baseline + sum of the full
decomposition)`, checked by the service against its declared tolerance before the
response is built. No feature *value* is disclosed — Phase 5 gates that behind a
reviewed configuration flag, and a live wire surface is not where it gets turned
on, so the response schema has no field for one at all.

### Drift Monitoring

**There is no serving drift endpoint**, so the page loads no report and shows no
figure. It documents the contract: PSI per transformed input against a reference
profile captured from the **training** partition, warning at `0.10`, alert at
`0.25`, and the current state — *No serving drift report loaded*.

The thresholds are the project's documented defaults, restated here because the
console imports no part of the ML layer; a test asserts them against
`DriftConfig`'s own field defaults, so a value changed there is a failing test
rather than a page quoting the old one.

The page will not derive a drift figure from the session's own windows. A handful
of hand-built demonstration requests is not a population, and a PSI over them
would measure the demo.

### Rule vs ML vs Hybrid

Explains and compares the three layers, with a static CSS architecture diagram —
no external diagram service, no image fetch, no rendering library. A console that
reached out to a third party to draw its own architecture would be sending a
description of the deployment somewhere the operator did not choose.

When `stacked` is active the page says the frozen stacked fusion state is being
served, and shows the loaded state's **fingerprint** so an operator can confirm
which stacker is live. The state's parameters are not published by the API and
are not reachable from the console.

The session flag counts are a description of what each mechanism did on this
session's windows. They are **not** an accuracy comparison and could not be —
these windows carry no labels, and the one labelled evaluation was locked in
Phase 5.

### System & Model

Renders `/version`, `/api/v1/system/status`, `/api/v1/model/info` and
`/api/v1/rules` as they arrive: contract versions, enabled layers, executing and
frozen strategy, the loaded stacked fingerprint, the champion's family and
identifiers and lineage, the frozen decision threshold, and the full public rule
catalog.

The threshold is shown because the API deliberately publishes it — an operating
point nobody can see is one nobody can audit. It is displayed as a number and
never as a control: there is no widget on this console that writes a threshold,
and no endpoint that would accept one.

What the API does not publish, the page cannot show: no coefficient, no tree
array, no HMAC key, no artifact path, no host path, no environment. Those
absences are structural — the response schemas forbid extra fields.

---

## 6. Session state

Everything the console remembers lives in one Streamlit session and ends with it:
the draft window, the detection history (bounded at 200), the latest full result,
the latest attribution, the selected scenario, and a submission counter.

A `DetectionRecord` is assembled from a response document. The console never
re-derives a verdict, re-thresholds a score, or combines the layers to produce a
flag of its own. The one number it computes is a sequence counter.

**The sequence numbers are local.** They count submissions in this tab; they are
not alert identifiers, they are not stable across a reload, and nothing on the
server knows about them. Clearing the history resets the numbering with it —
leaving the counter running would number the next detection `#57` in a list whose
first entry is `#57`.

**No request body is kept in the history.** Identifiers, addresses, and
application names stay in the draft the analyst is editing. A resend re-reads
that visible draft, so nothing can be re-submitted that cannot be inspected
first.

Streamlit's cache is not used for detection results. Correctness is preferable to
caching here, and a cached mutable result is the one thing a security console
must not show.

A refreshed or new browser session starts empty. Persistent server-side history
belongs to a later milestone.

---

## 7. Refresh behaviour

The refresh control is a **button**, not a timer. An auto-refreshing console is a
client that keeps calling a service nobody is looking at. `refresh_seconds` is
shown as guidance for how often a manual re-read is worth doing and is never used
to schedule one. No detection request is ever re-submitted automatically.

---

## 8. Offline behaviour

If the API is not running:

- Streamlit still loads and the navigation still works.
- The header shows **API offline**; the system badge reads *unknown*.
- Pages that need the backend show a fixed error state naming the problem kind,
  and render nothing further.
- **No Python traceback appears on the page.** `showErrorDetails = "none"` in
  `.streamlit/config.toml` is the backstop; the client's failure-as-a-value
  contract is what makes it unnecessary.
- No metadata is fabricated — no placeholder model family, no invented rule
  count, no default strategy.
- Detection submission is **disabled**, and the page says nothing is queued and
  nothing is scored locally.
- A **Retry connection** button sits in the sidebar.

When the API comes back, the next render is fully live. A console that cached its
first failure would need a restart to notice the service coming up, which during
a demonstration reads as the console being broken rather than the API having
been. An integration test drives exactly that sequence in one process.

Two views need no backend at all and work offline by design: **Drift Monitoring**
(it documents a contract) and the session-backed pages.

---

## 9. Privacy and security

**No credential is accepted, built, or stored.** No control on any page writes a
credential-shaped key — there is no password box, not even a disabled or ignored
one, and a test asserts that `type="password"` appears nowhere in the package.

Beyond that, `DashboardSession.add_event` **refuses** a credential-shaped field
name before storing it, under the same normalisation the project's ingestion
scanner uses (lowercased, separators stripped, camelCase split), over a superset
of the names that scanner refuses. The service would refuse such a *request*
anyway — but by then the value would already have been written into browser
session state and echoed back in the JSON preview. Refusing at the session's door
means the draft is provably free of credential material, and therefore so is
everything that renders it and everything that posts it. The refusal message
names the field *count* and reads no value.

**Nothing user-entered becomes markup.** `theme.py` is the only module that emits
HTML with `unsafe_allow_html`, its stylesheet is a module constant that
interpolates nothing, and every value that reaches a styled block goes through
`escape_text` — including the accent colours, which land in a `style` attribute.

**Nothing sensitive is displayed.** No raw traceback, no server filesystem path,
no secret, no environment variable, no pseudonymization key, and no internal
exception message. The client discards exception messages rather than forwarding
them, and error text is fixed per problem kind.

That extends to the console's *own* configuration failure, which is the one
internal error it does render. `str(ValidationError)` appends `input_value=...`,
and the value that most often breaks `api_url` is a URL somebody pasted a token
into — so `load_dashboard_settings` builds its message from the failing **field
names and validation rules** and drops the value entirely. A scrubber has to be
right every time; a discard has to be right once.

**No scientific override is reachable.** The configuration declares none, the
request envelope carries exactly two keys (`events`, `anchor_selection`), and the
API declares no field that would receive one.

`.streamlit/config.toml` is tracked and carries theme and layout only.
`.streamlit/secrets.toml` does not exist, must not be created, and is refused by
`.gitignore`.

---

## 10. Running it locally

Two terminals.

**Terminal 1 — the API:**

```bash
export PAD_API_ARTIFACT_ROOT=/absolute/path/to/artifacts
export PAD_API_ALLOWLIST_PATH=/absolute/path/to/allowlist.yaml
export PAD_API_FEATURE_CONFIG_PATH=/absolute/path/to/features.yaml
export PAD_API_ML_CONFIG_PATH=/absolute/path/to/configs/ml/model-testing.yaml
export PAD_API_DETECTION_CONFIG_PATH=/absolute/path/to/rules.yaml

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

Then:

| | |
| --- | --- |
| API | <http://127.0.0.1:8000> |
| Swagger | <http://127.0.0.1:8000/docs> |
| Dashboard | <http://127.0.0.1:8501> |

If the frozen selection was `stacked`, publish the serving bundle once first with
`deploy materialize` — see [api.md](api.md) §2. Before a champion has been
frozen, the rule layer can be served alone with
`PAD_API_REQUIRE_ML_CHAMPION=false`; the console then shows the model and hybrid
layers as unavailable, with the service's own reason codes.

The console works with the API absent — that path is worth seeing at least once,
because it is what a viewer gets when they open the console first.

### One command instead of two terminals

```bash
docker compose up --build
```

The console then runs in its own container and reaches the API at
`http://api:8000` over the project network — Docker's internal DNS, because
inside that network `127.0.0.1` would name the console's own container. Its
container is given **no volume, no artifact path, and no `PAD_API_*` variable at
all**: the only detection it can display is one the API performed, and the
deployment declines to hand it the means to be tempted otherwise. See
[docker.md](docker.md).

One override applies there and only there. This repository's
`.streamlit/config.toml` binds `127.0.0.1`, which is right for a laptop and wrong
inside a network namespace nothing else can enter, so the container image sets
`STREAMLIT_SERVER_ADDRESS=0.0.0.0`. The tracked default stays loopback — nobody
gets a console on every interface by running it the ordinary way — and the
published port is bound to the host's loopback regardless.

### Behind a public reverse proxy

`compose.deploy.yaml` un-publishes 8501 entirely and puts a reverse proxy in
front. **The console then becomes the whole public surface**: `/` routes to it
and, under the default routing policy, nothing routes to the API — which costs
the demonstration nothing, because this client runs server-side and reaches the
API across the Compose network either way.

Three things about Streamlit behind a proxy that were verified rather than
assumed:

* **The websocket survives it.** `/_stcore/stream` upgrades to `HTTP/1.1 101`
  through the proxy, with no extra configuration: Caddy forwards the `Upgrade`
  header and skips compression for it. Without that the page loads and then
  never updates, which is the failure mode worth naming.
* **Streamlit's own origin and XSRF checks stay satisfied**, because the proxy
  passes the original `Host` header through.
* **The Content-Security-Policy is deliberately narrow.** Streamlit's bundled
  client evaluates generated code and installs inline styles, so a `script-src`
  policy tight enough to be worth having stops the console rendering, and one
  loose enough to keep it working would have to permit `'unsafe-inline'` and
  `'unsafe-eval'` — a control that claims a protection it does not provide. The
  policy is `frame-ancestors 'none'` only, doubled by `X-Frame-Options: DENY`.
  [deployment.md](deployment.md) §8 records the audit.

The console gets no new configuration in that topology: no volume, no artifact
path, no `PAD_API_*` variable, and nothing that names a hostname.

---

## 11. Current limitations

**Session-only history.** There is no alert store, no event database, and no
server-side history. Every count, chart, and table on the session pages describes
this browser tab, and says so on the page. A reload starts empty.

**A replay run is not history either.** The Live Replay view follows a run that
lives in one API process's memory, bounded and cleared by a restart. It is a
demonstration somebody is watching, not a record anybody should later rely on,
and every page that shows it says so. A reload loses the console's view of the
run; the run itself carries on until it finishes or is stopped.

**No serving drift report.** Drift is computed offline by `ml drift`; nothing in
the serving layer publishes a report, and the Drift Monitoring page documents the
contract rather than showing a figure.

**Per-anchor attribution only.** `POST /api/v1/explain` answers about one anchor
of one window. There is no population-level attribution over live traffic, and
there should not be: the Phase 5 aggregate report is computed over a partition,
and live requests are not one.

**No authentication, no authorization, no multi-user state.** The console is a
demonstration client. It is containerised as of Milestone 4 and has a verified
public perimeter as of Milestone 5A, but it is **not deployed**, not
authenticated, and not rate-limited. Anyone who can reach it can use it. Locally
that means anyone on the machine, because both ports are published to loopback
and no further. Behind the deployment perimeter it would mean anyone on the
internet — which is why the API is not published there, why the session state is
per-browser-tab and holds nothing, and why the abuse bounds in
[deployment.md](deployment.md) §15 were audited before any of it was written
down. Deliberately no authentication platform was added for that milestone: a
half-built one on a demonstration is a larger surface than the one it closes.

**No CORS policy on the API.** The console talks to the service from Python, not
from the browser, so no browser origin needs to be allowed yet. That stays true
behind the proxy: the browser's only origin is the proxy's own.

---

## 12. Related documents

- **[api.md](api.md)** — the serving API this console consumes
- **[live-replay.md](live-replay.md)** — the synthetic replay demonstration the Live Replay view drives
- **[docker.md](docker.md)** — the containerized deployment this console runs inside
- **[deployment.md](deployment.md)** — the public perimeter this console sits behind on a server
- **[rule-catalog.md](rule-catalog.md)** — the rules the catalog page lists
- **[explainability.md](explainability.md)** — the Phase 5 attribution contract
- **[drift-monitoring.md](drift-monitoring.md)** — what the drift page documents
- **[risk-scoring.md](risk-scoring.md)** — why `risk_score` is an ordinal
- **[privacy-model.md](privacy-model.md)** — the pseudonymization contract
