# Demonstration walkthrough

A presentation checklist for showing the Password Attack Detector to an
audience — a supervisor, an examiner, a review panel, or a colleague.

Everything below runs against the public deployment:

**<https://pad-demo.onrender.com>**

It works identically against a local `docker compose up --build` stack at
<http://localhost:8501>, which is the safer choice if the venue's network is
uncertain.

---

## Before you start

| | |
|---|---|
| **Wake the service** | Render Free spins the service down after inactivity. Open the URL **1–2 minutes before** you present and leave the tab open. A cold start takes roughly that long; a woken service responds immediately. |
| **Have a fallback** | If the venue has no network, run `docker compose up --build` beforehand and present against `localhost`. Every step below is identical. |
| **Know the honest framing** | This is a *demonstration of a detection system on synthetic data*. It is not a benchmark, not a production authentication component, and not evidence about real login traffic. Say this once, early — it makes every later claim easier to defend. |
| **Expect a fresh session** | Replay history is held in the API process's memory and cleared on restart. If the service cold-started, there is no previous run to find, and that is correct behaviour. |

**Timing.** The full walkthrough is about 12–15 minutes. Steps 1–7 alone are a
solid 6-minute version; steps 11–14 are the ones to drop first if you are short.

---

## 1. Open the public dashboard

Go to **<https://pad-demo.onrender.com>**.

The header shows the product name, a one-line description, and a connection
indicator. A green indicator means the console reached the detection API; the
console is a *client* of that API and shows an explicit offline state rather than
a traceback if it cannot.

**Say:** this page is a Streamlit console. It holds no detection logic at all —
no rules, no model, no thresholds. Everything it shows came from an HTTP call to
a FastAPI service running beside it inside the same container.

---

## 2. Explain the Overview

The landing page, in order down the screen:

1. **Six status cards** — System, Detection, Fusion, Rules, Model, Demo — in
   plain language rather than identifiers.
2. **"Start a live demo"** — the primary call to action.
3. **What this system does** — one paragraph on three-layer detection.
4. **Recent activity** — or *"No activity in this browser session yet."*
5. **Advanced system details** — component table, architecture, model identity
   and rule catalog, inside an expander.

**Point at the Fusion card.** It should read `Stacked`. That is the frozen hybrid
strategy, selected on validation data before any test label was opened, and
verified by fingerprint at startup. If it could not be verified the service would
refuse to be ready rather than quietly substitute a different strategy.

**Point at "No activity yet."** There is no persistent event store and no alert
database, so the page shows an empty state rather than an invented total. That
absence is deliberate and worth naming out loud.

---

## 3. Open Live Replay

Click **Start a live demo**, or select **Live Replay** in the sidebar.

**Say what a replay is, before running one.** A replay takes a reviewed,
deterministic, fabricated scenario and emits it into the detection service **one
event at a time**, through the same code path an HTTP request from a real client
would take. Nothing is attacked. No credential exists anywhere in a scenario. The
events describe entities that do not exist.

The controls are **▶ Start replay**, **■ Stop replay**, **↻ Refresh**, plus a
scenario selector and a pace selector.

**Pace is presentation only.** `instant`, `fast`, `normal`, `slow` change how
long the run takes on the wall clock and nothing else — the same scenario
produces byte-identical verdicts at every pace. Use `fast` when presenting;
`normal` if you want the timeline to build visibly.

---

## 4. Run **Normal login activity**

Select **Normal login activity** and press **▶ Start replay**.

Twelve events over about five simulated minutes. Watch the timeline fill:

```
TIME | EVENT | RULE | RISK | ML | HYBRID | SEVERITY
```

**Expected outcome: the rule layer fires nothing.** Risk scores are `0.0`, which
is a module constant — a zero always means "nothing fired", never "we did not
look".

**This is the most important slide in the demonstration, and it is easy to skip.**
A detector that flags everything is useless, and showing an attack scenario
without first showing a clean one proves nothing about discrimination. Say so.

**Be honest about the ML column.** The machine-learning layer flags most replay
events, including benign ones, because the serving path has no behavioural
baseline (see step 15). What discriminates here is the **frozen stacked hybrid**,
not the model alone. That is reported rather than tuned away.

---

## 5. Run **Concentrated brute force**

Select **Concentrated brute force** and start it. Thirty events, about five
simulated minutes.

**Expected:** `PAD-BF-001` (concentrated brute-force indicator) and `PAD-BOT-001`
(bot-like authentication indicator) fire, and severity reaches at least **high**.

Watch the risk score climb as the failure burst accumulates. Two rules fire, not
one — the same behaviour is legitimately visible as both a brute-force
concentration and an automation signature.

---

## 6. Show the rule, ML and hybrid columns side by side

Stay on the completed brute-force run and point at the three verdict columns.

| Column | What it is | What it is **not** |
|---|---|---|
| **RULE** | fired rules, their evidence, and an ordinal `risk_score` in `[0, 100]` | not a probability |
| **ML** | a calibrated probability and a binary decision at a frozen threshold | not a rule outcome |
| **HYBRID** | one fused decision from the frozen **STACKED** strategy | not a fallback, not an average |

**Say the sentence that matters:** the ordinal risk score is never blended with
the model's probability, and no page renames one the other. They are separate
typed objects from the engine all the way to this screen.

**And the second one:** there is no fallback hybrid. If the frozen stacked state
could not be verified, the service would report the hybrid unavailable and return
`503` — a strategy nobody selected is never substituted for the one that was.

Open the **Demo run summary** at the bottom of a completed run. Every figure in
it — triggered rule counts, fusion strategies used, severity distribution — was
derived by the service from that run's own timeline records, not recomputed by
the console.

---

## 7. Run **Password spraying**

Select **Password spraying** and start it. Twenty-four events over about two
simulated minutes.

**Expected:** `PAD-PS-001` (password-spraying indicator) and `PAD-BOT-001` fire,
severity at least **high**.

**Contrast it with brute force out loud.** Brute force is *many attempts against
one account*; spraying is *one or two attempts against many accounts*, which is
specifically designed to stay under a per-account lockout threshold. Different
rule, different feature, same three-layer treatment.

If you have time, run **Mixed attack timeline** as well: 29 events in which
`PAD-BF-001`, `PAD-BF-002` and `PAD-PS-001` all fire, which shows several
detections interleaved on one timeline rather than one clean signal at a time.

---

## 8. Inspect Alerts

Open **Alerts** in the sidebar.

The flagged steps from the run you just completed appear under their own heading,
labelled as a server-side demo replay run. If you had also submitted a window by
hand on the Detection Console, those results would appear in a **separate**
section.

**The two sources are never silently merged.** A manual submission is what this
browser tab sent; a replay run is something happening on the server. They have
different origins and different lifetimes, so every replay-derived record carries
a `replay:` prefix and every page that can show both says which is which.

Note what is **not** here: there is no alert database. Reload the page in a new
session and this list is empty. That is stated on the page rather than papered
over.

---

## 9. Inspect Analytics

Open **Analytics**.

Charts over detection results: severity distribution, rule activity, and layer
agreement. Note the explicit **Data source** selector — "this dashboard session",
"active demo replay run", or "both, labelled by source" — which defaults to the
manual session rather than quietly combining them.

---

## 10. Explain Rule vs ML vs Hybrid

Open **Rule vs ML vs Hybrid** (under **Advanced**).

This page states the architecture directly: the three layers, the active fusion
strategy, and why they are kept apart. Use it to make the design argument.

**Say:** keeping the layers separate is the point of the design, not an
implementation detail. It is what allowed rule-only, model-only and hybrid
detection to be *measured against each other* on identical frozen splits, instead
of one quietly absorbing the other. The hybrid was then selected on validation
data alone and frozen before the test labels were opened.

---

## 11. Show Explainability

Open **Explainability**.

For a scored anchor, this decomposes the frozen model's decision over the
transformed columns it actually read, using the exact-decomposition primitive —
`value × coefficient` for a linear model, the decision path for a forest, the
step for a threshold baseline.

**Two things to say:**

- **Attribution is exact or typed unavailable, never approximate.** A model
  family with no exact decomposition reports the attribution unavailable rather
  than showing an approximation dressed as an explanation.
- **Attribution is descriptive, not causal.** It says how a fitted function
  decomposes over the columns it was handed. It does not say why an attacker did
  something.

---

## 12. Show Drift Monitoring

Open **Drift Monitoring**.

This shows the drift contract: PSI against a frozen training reference profile,
with the reviewed warn threshold at `0.10` and alert threshold at `0.25`, plus
null-rate and unknown-rate shifts reported as fields.

**Say:** drift is computed **without labels**, so it cannot say a model became
wrong — only that the population it is seeing has moved. And nothing here
retrains, promotes, or re-thresholds on a finding. The drift module imports no
training, selection, freeze, threshold or evaluation entry point, so there is no
call it could make.

---

## 13. Show System & Model

Open **System & Model** (under **Advanced**).

This reads `/version`, `/api/v1/system/status`, `/api/v1/model/info` and
`/api/v1/rules` from the service and displays them: package version, every
contract version, the champion's identity, the frozen fusion strategy and its
fingerprint, and the full rule catalog.

**Point at the stacked state fingerprint.** It is the fingerprint Phase 5 sealed
before the test evaluation ran. The serving bundle was reconstructed offline from
pre-test lineage only, its fingerprint recomputed, and publication refused unless
the two matched. What is running is provably the thing that was evaluated.

Also open **About System** to close: it summarises the project, its capabilities
and its limitations on one page, with no backend call required.

---

## 14. Explain the architecture

Draw or narrate this:

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

Three processes, one container, one supervisor process as PID 1.

**The public surface is two routes.** `/` serves the console; `/healthz` is a
liveness check whose body is a status string, the service name and the package
version. **Everything else is internal** — scoring, explanation, replay control,
system information, readiness and Swagger are all bound to `127.0.0.1` inside the
container and answer only the console process beside them.

**Why that costs nothing:** the console's HTTP client runs *server-side*. A
browser never talks to the detection API, so proxying it would add public attack
surface that no part of the demonstration uses.

If asked why one container rather than two services: Render's free tier gives no
private network between two free services, so a separate console service could
only reach the API over the public internet — which would force the detection API
to be publicly exposed. One container is what keeps it on loopback.

---

## 15. State the limitations

Do this explicitly rather than waiting to be asked. It is the strongest part of
the presentation.

1. **The data is synthetic.** Every event is fabricated by a seeded generator
   from a tracked configuration. Nothing here is evidence about real
   authentication systems, and no figure is a performance claim.
2. **This is not production authentication infrastructure.** It observes and
   scores event records. It does not authenticate anyone, sit in a login path,
   block a session, or integrate with an identity provider.
3. **Replay state is in-memory and ephemeral.** A restart clears every run.
4. **Render Free cold-starts** after inactivity, and offers no uptime guarantee.
5. **Free-tier resources are constrained** — 512 MiB and a fraction of a CPU.
   Throttling changes how long a replay takes, not what the detector decides:
   every verdict measured at 0.1 CPU was identical to the same verdict at 0.5.
6. **There is no persistent alert database**, no event store, and no serving
   drift report.
7. **No real password or credential is ever collected.** There is no request
   field that could carry one; credential-shaped field names are refused before
   any other validation runs.
8. **There is no public raw detection API**, and there is no authentication and
   no rate limiting anywhere in the system. That residual risk is documented
   rather than implied to be covered.
9. **Two rules cannot fire on any live request.** `PAD-CS-001` (credential
   stuffing) and `PAD-ATO-001` (account takeover) gate on a fitted behavioural
   baseline that the serving path does not load, so they report *insufficient
   data* on every live window.

**Point 9 deserves the honest version**, because it is the question a good
examiner will ask. Adding a baseline to the serving bundle was **audited and
measured not to help**: a baseline fitted from the deployment's own training
split was loaded and the two scenarios re-run, and the relevant flags came back
unchanged — the replay catalog's identities are content-addressed pseudonyms that
no training population contains, by construction. Making those rules fire would
mean either loosening a frozen rule, which would publish a detection nobody
validated, or coupling a content-addressed catalog to one dataset. **No threshold
was moved and no baseline was synthesised to make the demonstration look
better.** The credential-stuffing scenario trips `PAD-PS-001` instead, and its
catalog entry says so.

---

## Likely questions

**"Is this using real passwords?"**
No. There is no field in any request schema that could carry one. Credential
material is refused under every spelling before any other validation runs, and
again in the console's session state before a value could reach a browser
preview.

**"Could this attack a real system?"**
No. A replay is events emitted into a service the operator is running themselves.
Import-time guards refuse any address outside the RFC 5737 documentation ranges,
and a syntax-tree test refuses a network client anywhere in the replay package.

**"How do you know the model wasn't tuned on the test set?"**
Exactly one command in the entire project may open the test ground truth:
`ml evaluate`. `ml predict` has no `--labels` option, `detection run` has neither
`--labels` nor `--splits`. Across the whole codebase exactly two modules may open
a ground-truth table, and an import-graph test pins that set in both directions.

**"What does the risk score mean?"**
It is a bounded ordinal magnitude in `[0, 100]` that orders findings by
accumulated evidence. It is **not** a probability. Correlated rules are reduced
within their group before being combined, so one behaviour restated three ways
cannot inflate a score.

**"Is the deployment secure?"**
It is hardened and it is honest about what it is not. Non-root containers, a
read-only root filesystem, all capabilities dropped, no Docker socket, no host
networking, and a serving bundle that cannot be written at runtime. It has **no
authentication and no rate limiting**, which is documented as the main residual
risk rather than glossed over.

**"How long did a cold start take?"**
About 92 seconds at 0.1 CPU when measured locally under the free tier's limits,
with peak memory at 199.1 MiB — 38.9% of the 512 MiB ceiling — and zero OOM
events.

---

## Related documents

| Document | Contents |
|---|---|
| [render-deployment.md](render-deployment.md) | The deployment a viewer is looking at |
| [dashboard.md](dashboard.md) | Every view, in detail |
| [live-replay.md](live-replay.md) | The scenario catalog and the run lifecycle |
| [api.md](api.md) | The serving contract behind the console |
| [phase6-acceptance.md](phase6-acceptance.md) | What was verified, and how |
| [model-card.md](model-card.md) | Purpose, scope, prohibited use, limitations |
