# Render deployment

How the demonstration is deployed to a **Render free web service**, why it is
shaped as one container rather than two services, and what the free tier's
limits mean for what a viewer sees.

> **Nothing is deployed.** No Render service exists for this project, no Render
> account is referenced anywhere in this repository, and no Render credential is
> stored, requested, or used. `render.yaml` describes a deployment that has not
> been created. Everything measured below was measured locally, in a container
> constrained to the free tier's limits.

This is a **second** deployment target, not a replacement. The single-VPS Docker
Compose deployment described in [`deployment.md`](deployment.md) is unchanged and
remains the reference architecture; the two share the same application code, the
same tracked configurations, the same pinned interpreter, the same proxy, and —
as [§5](#5-the-fingerprint-proof) demonstrates — the same frozen serving bundle,
fingerprint for fingerprint.

---

## Contents

1. [Why one service](#1-why-one-service)
2. [Free-tier constraints](#2-free-tier-constraints)
3. [Architecture](#3-architecture)
4. [The build-time serving bundle](#4-the-build-time-serving-bundle)
5. [The fingerprint proof](#5-the-fingerprint-proof)
6. [Process supervision](#6-process-supervision)
7. [The port contract](#7-the-port-contract)
8. [Public and private routes](#8-public-and-private-routes)
9. [The health check](#9-the-health-check)
10. [The ephemeral filesystem](#10-the-ephemeral-filesystem)
11. [Memory](#11-memory)
12. [CPU and cold starts](#12-cpu-and-cold-starts)
13. [Deploying from GitHub](#13-deploying-from-github)
14. [Logs](#14-logs)
15. [Redeploying and rolling back](#15-redeploying-and-rolling-back)
16. [Rule availability, and the baseline limitation](#16-rule-availability-and-the-baseline-limitation)
17. [Limitations](#17-limitations)
18. [What was verified locally](#18-what-was-verified-locally)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. Why one service

The obvious shape — one free web service running the API, another running the
console — is wrong on Render's free tier, for four independent reasons. Any one
of them would be a reason to prefer a single service; together they make it the
only defensible design.

**Instance hours are shared.** Free instance hours are drawn from one monthly
allowance across all of an account's free services. Two services consume it
twice as fast, so the demonstration would be unavailable roughly twice as often
for the same allowance, in exchange for nothing a viewer can see.

**There is no private network.** Render's private networking is not available to
free services. A console running as a separate free service could only reach the
API over the public internet, which means the detection API would have to be
**publicly exposed** for the console to work at all. The single-service design is
therefore not merely cheaper — it is what allows the API to stay private, which
is the same routing policy the VPS deployment enforces with a reverse proxy.

**Cold starts would compound.** Free services spin down after inactivity and
each wakes independently. A viewer would face two cold starts, frequently
arriving at a console that had woken up in front of an API that had not, and the
console would render a "service unavailable" page for a service that was merely
still starting.

**The attack surface would double.** Two public hostnames, two TLS endpoints, two
sets of routes, to run a demonstration that needs one page.

So both processes run in one container, behind one proxy, on one public port.

---

## 2. Free-tier constraints

These are treated as hard deployment constraints, not as guidance:

| Constraint | Value | Consequence for this deployment |
|---|---|---|
| Memory | 512 MB | Measured peak 198.3 MiB — see [§11](#11-memory) |
| CPU | 0.1 | Cold start ~92 s — see [§12](#12-cpu-and-cold-starts) |
| Disk | none | The serving bundle is baked into the image |
| Filesystem | ephemeral | Nothing scientific is written at runtime |
| Inactivity | spins down | Documented; the console tolerates a waking API |
| Private networking | unavailable | One service, loopback between processes |
| Services used | one web service | `render.yaml` declares exactly one |

No disk, no database, no worker, no private service, no cron job, and no paid
plan is declared. A test asserts each of those absences.

---

## 3. Architecture

```
                          Internet
                              |
                              |  HTTPS  (Render's edge, Render's certificate)
                              v
                   +----------------------+
                   |      Render edge     |   TLS terminated here
                   +----------------------+
                              |  plain HTTP to $PORT
 =============================|=========================== one container ====
                              v
                   +----------------------+
                   | caddy  0.0.0.0:$PORT |   the only public listener
                   +----------------------+
                       |                |
          /healthz     |                |   everything else
                       v                v
          +--------------------+   +----------------------+
          | FastAPI            |   | Streamlit console    |
          | 127.0.0.1:8000     |<--| 127.0.0.1:8501       |
          +--------------------+   +----------------------+
                       |             server-side API client,
                       |             over loopback
                       v
          +----------------------------------------+
          |  /srv/state -- frozen serving bundle    |
          |  root-owned, read-only, never written   |
          +----------------------------------------+
```

Four processes, one of which is an init:

| Process | Bound to | Role |
|---|---|---|
| `render_entrypoint.py` | — | PID 1. Verifies, starts, supervises, stops. |
| `caddy` | `0.0.0.0:$PORT` | The public boundary and the entire routing policy. |
| `uvicorn` | `127.0.0.1:8000` | The detection service. |
| `streamlit` | `127.0.0.1:8501` | The analyst console. |

The console's API client runs **server-side**, inside this container, and calls
`http://127.0.0.1:8000`. A browser never talks to the detection service. The
public Render URL is never used for an internal call: doing so would route
internal traffic out through the edge and back, double every latency, and
require the API to be publicly proxied.

Files:

| File | Role |
|---|---|
| `render.yaml` | The blueprint. One free web service, Docker runtime. |
| `Dockerfile.render` | Multi-stage build: install → prepare → verify → prune → runtime. |
| `deploy/render/Caddyfile` | The public routing policy and response hardening. |
| `scripts/render_entrypoint.py` | The supervisor. Standard library only. |
| `scripts/verify_serving_bundle.py` | Bundle verification, at build time and at start. |

---

## 4. The build-time serving bundle

### The deviation, stated plainly

`Dockerfile` — the VPS image — says at length that a container image must not
ship a trained model, because an image carrying a frozen champion makes the
image the provenance of a scientific decision. `Dockerfile.render` does exactly
that. The objection was not wrong; the remedy it relies on is unavailable here.

On a VPS the remedy is a one-shot `prepare` job that runs the real pipeline into
a named volume, which the API then mounts read-only. The provenance is the job,
and the job is not the service. Render Free has nowhere for that volume to live.
The alternative — preparing at container start — would mean fitting three model
families on 0.1 CPU on **every cold start**, which is precisely the "train at
startup" this project refuses, and would make the served model a fresh decision
each time the service woke up.

So the preparation moves to build time, into a stage that is discarded, and the
guarantee is preserved by different means:

1. the `prepare` stage runs `scripts/prepare_demo_bundle.py` — the *same*
   tracked script `compose.yaml` runs, over the *same* tracked configurations,
   with no option anywhere in it that can name a model, a threshold, or a
   strategy;
2. `scripts/verify_serving_bundle.py` runs immediately afterwards and **fails
   the build** if the bundle does not verify, so an unverifiable bundle never
   becomes an image;
3. the prune step removes what serving does not read, and the verification is
   run **again** afterwards, so a prune that removed something needed fails the
   build rather than the deployment;
4. the runtime stage receives the result through `COPY --from=prepare`,
   root-owned, and serves as an unprivileged account that cannot write it;
5. the supervisor verifies it a third time before starting anything;
6. the API verifies it a fourth time, on its own terms, during startup.

What is given up is that the image is now *a* provenance record. What is kept is
that it is a faithful one, that producing it required the whole pipeline, and
that nothing at runtime can alter it.

### What the image carries

The prepared state root is pruned to what serving actually opens:

```
/srv/state/
├── artifacts/        kept whole
│   ├── champion/     the frozen champion lock
│   ├── ledger/       the experiment ledger the champion verifies against
│   ├── runs/         the immutable training runs
│   ├── selections/   the champion selection evidence
│   ├── predictions/  the published prediction manifests
│   ├── evaluations/  the locked TEST evaluation receipts
│   └── serving/      the materialized serving bundle
├── allowlist.yaml    the reviewed ML feature allowlist
└── prepared.json     the preparation receipt
```

Removed: `dataset/`, `processed/`, `detection/`, `reports/` — the pipeline's
inputs and its offline evidence, none of which the API opens. Their absence is
also what makes it impossible for the runtime to re-run anything: there is
nothing left to run it over.

`artifacts/` is kept **whole** rather than reduced to the bundle. The champion
loader verifies against the ledger and the run that produced it, and the fusion
layer reads the locked evaluation receipt; a "bundle only" image would serve a
champion whose provenance could no longer be checked. It costs about 1 MiB.

**No scientific fitting happens at runtime.** The image contains no training
data, no `prepare_demo_bundle.py`, and no writable scientific path.

---

## 5. The fingerprint proof

The claim is that moving preparation from a Compose job into a build stage
changed nothing scientific. It is proved by doing both and comparing, not by
argument.

`tests/integration/test_render_container.py::test_the_baked_bundle_is_the_compose_prepared_bundle`
runs `compose.yaml`'s preparation job from scratch, in its own Compose project,
and compares every fingerprint the serving manifest carries against the ones
inside the Render image. Measured, twice, on two independent preparation runs:

| Field | Value |
|---|---|
| champion scope key | `d85a151c597b8b9eca0cd570b236aab91cb3cb13f021ee23576421fa1b9f5e90` |
| selected fusion strategy | `stacked` |
| **manifest fingerprint** | `8341204696599ea08a3299bf249b214131b35f1073f46f5d25d41283b8799954` |
| **STACKED state fingerprint** | `134f66ce2b837c4b74c9ab9ff2703125b9aedefb210d18afe923a0662f96272a` |
| fusion selection fingerprint | `43d95c5897c75a29f01d499ed8994ece9a4e7cc72616b98a6bb8152e3a3cca33` |
| champion lock fingerprint | `4aff9457b7e959accd98f7d59f8f0c978751da4242b1df7e64e978b6f067a768` |
| model content fingerprint | `4b821ba1a0694643748c63692b9d8e00267d840320c42bac9b89d22c2e1e7ce0` |
| calibration state fingerprint | `c1d2f05fb4fad3e570c2c325313211ca4778a45a644ec4df8f3a08ebfab875cd` |
| binary threshold fingerprint | `a999591967a10d531e82b9eb6349a13155d62d7acd4d0b353413f8a94d6c677b` |
| feature catalog fingerprint | `aeb89e86ed60f0a4842ba8e8f271f7089508e0458ad2288a66002d523ae0520d` |
| rule configuration fingerprint | `82951f6953dea84c5f807bf0e5dc385d41e55d2500696362649c95d2455bc844` |

Every payload file is byte-identical as well, not merely equal in digest:

```
d55bbac2500399dfdc408e707ee9f64dc16942e815709ac9b7152244fb66f4a2  fusion_selection.json
50f033e5d9856f43e1222ef76d3111d5ac6ea47598bef7f5c85c599f99fed5bf  fusion_stacked_state.json
9c8bb028065f483d947a58cf75742efd753f00938ffa664b80f4c71d7ff0f4c1  serving_bundle.json
```

The running service reports the same STACKED fingerprint on
`/api/v1/system/status`, so the artifact that verifies on disk is the artifact
being served.

**Scope of the claim.** This proves that the Render build and the Compose
preparation produce the same bundle **on the same machine**. Reproducibility
across different machines is a separate property, established by the project's
own reproducibility suite (`tests/integration/test_phase5_reproducibility.py`)
and not re-proved here. For that reason the `prepare` stage is deliberately
given none of the runtime stage's environment — no thread-count limits, no
allocator tuning — so it runs under exactly the conditions the Compose
preparation runs under.

---

## 6. Process supervision

`scripts/render_entrypoint.py` is PID 1. It uses the standard library only: it
lives for the lifetime of the container, and importing the project package would
pull numpy, pandas and scikit-learn into a process whose entire job is to wait.

**Startup, in order, with the next step gated on the previous one:**

1. read and validate `PORT`; refuse to start if it is missing or invalid (exit 2);
2. verify the baked serving bundle, in a **subprocess** so its memory is returned
   to the operating system; refuse to start if it does not verify (exit 3);
3. start the API, and wait until `127.0.0.1:8000/health` answers 200;
4. start the console, and wait until `127.0.0.1:8501/_stcore/health` answers 200;
5. start the proxy, and wait until `/healthz` answers through it.

The proxy starts **last**, which is what makes Render's health check meaningful:
until both processes behind it are serving, nothing is listening on the public
port at all, so the check cannot pass against a half-started container.

**Steady state.** The supervisor blocks in `select.select` with no timeout on a
self-pipe fed by `signal.set_wakeup_fd`. It consumes no CPU until a child dies
or a signal arrives — which on a 0.1 CPU instance is not a micro-optimisation.
There is no `sleep` loop anywhere in the file.

**Shutdown.** On SIGTERM or SIGINT the children are stopped in reverse order —
proxy, then console, then API — so the proxy stops accepting before the console
it fronts goes away, and the console goes before the API it reads from. Each
child leads its own session and is signalled as a process group, so nothing it
spawned can outlive it. Survivors past a 25-second grace period are killed; the
grace period is longer than the API's own 20-second graceful-shutdown budget on
purpose, because killing at exactly its deadline would race it.

**Failure.** If any supervised process exits, for any reason, the supervisor
stops the other two and exits non-zero (exit 4). It never restarts a child.
Render restarts the container; a supervisor that quietly restarted one process
would leave the service reporting healthy while running a combination nobody
deployed.

**Orphans.** Reaping uses `waitpid(-1, WNOHANG)`, so grandchildren reparented to
PID 1 are collected too. A container that never restarts would otherwise
accumulate zombies.

Exit codes: `0` clean stop · `1` startup failed · `2` bad `PORT` · `3` bundle did
not verify · `4` a supervised process died.

---

## 7. The port contract

Render assigns the public port and passes it as `PORT`. The deployment reads it
and **never guesses it**:

- `resolve_public_port()` refuses a missing, empty, non-numeric, zero, negative
  or out-of-range value, with a message, and exits 2. Port 0 is refused with the
  rest: it means "any free port", which for a service whose entire contract is
  *this* port is a failure dressed as success.
- The Caddyfile listens on `:{$PORT}` with **no default**. A default would let a
  misconfigured container come up quietly on a port nobody asked for.
- There is no `EXPOSE`, no `ENV PORT`, and no `10000` anywhere in the image, the
  supervisor, or the routing policy. 10000 is merely what Render happens to use
  today; an image that baked it would pass every local test and be unreachable
  the day that changed.

Verified against arbitrary values — the local test suite runs the container on
19080 and 41573, and the manual verification also used 34567.

---

## 8. Public and private routes

The routing policy is the same one the VPS deployment applies, adapted to
loopback upstreams. **The console is public; the API is not.**

| Route | Goes to | Why |
|---|---|---|
| `/healthz` | API `/health` | Render's health check. Status and package version only. |
| everything else | Streamlit console | The demonstration itself. |

**Not publicly reachable:** `/api/v1/detect`, `/api/v1/explain`, the replay
controls (`/api/v1/demo/*`), the system and model endpoints, `/api/v1/rules`,
`/ready`, `/version`, `/docs`, `/redoc`, `/openapi.json`. They are served on
`127.0.0.1:8000` and reachable only from inside the container — in practice,
only by the console process.

`/ready` is deliberately private even though it is a health endpoint: it names
*which* scientific component is unavailable and why. That is diagnostic detail
for an operator reading logs, not for an anonymous caller. The console renders
the same information for a viewer, having asked for it over loopback.

The interactive schema browser is switched off in the image
(`PAD_API_DOCS_ENABLED=false`) as well as being unroutable. The smallest useful
public surface is the one that does not depend on a proxy rule to stay small.

> **A note on testing this.** Streamlit answers `200` with its own single-page
> shell for any path it does not recognise, so `GET /api/v1/detect` through the
> public port returns 200 whether the routing policy works or not. Every routing
> test in this project therefore inspects the response **body**, not its status
> code. Checking status codes here would prove nothing.

**Response hardening**, applied to every route, identical to the VPS deployment:
`X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`,
a `Permissions-Policy` denying every sensor and device, `X-Frame-Options: DENY`,
`Content-Security-Policy: frame-ancestors 'none'`, and a `Strict-Transport-Security`
that defaults to `max-age=0`.

The CSP is `frame-ancestors` only, and the audit behind that is in
[`deployment.md` §8](deployment.md); it is not repeated here. HSTS defaults to
`max-age=0` even though Render serves the deployment over HTTPS: a free service
lives at an `onrender.com` subdomain the operator does not own, and pinning a
`max-age` there is a promise about somebody else's name. An operator who has
attached a custom domain sets `PAD_HSTS`.

Caddy's admin API is off. Its `Via: 1.1 Caddy` response header remains — it is a
hop-by-hop header a proxy is specified to add, and the VPS deployment emits it
too; the `Server` header is removed.

**No authentication, no credentials, no rate limiting.** The service accepts no
credential material of any kind, so there is nothing to authenticate. Rate
limiting was deferred rather than added with a fragile dependency, exactly as in
[`deployment.md` §15](deployment.md) — and the surface it would protect is
smaller here, since scoring and replay control are not publicly reachable at all.

---

## 9. The health check

`render.yaml` sets `healthCheckPath: /healthz`. The proxy rewrites it to the
API's `/health`, which reads no artifact, no model and no filesystem:

```json
{"status": "ok", "service": "password-attack-detector", "version": "0.5.0"}
```

**What a 200 here proves.** The proxy is running and can reach the API, so two
of the three processes are demonstrably alive. The console is covered
*structurally* rather than by the response body: the proxy is not started until
the console answers its own health endpoint, and if the console exits afterwards
the supervisor stops the whole container. So a container that answers `/healthz`
is one in which all three processes are running — not because the endpoint asks
all three, but because no other state can persist.

Nothing scientific appears in the body. The package version is already shown on
the console's own page.

---

## 10. The ephemeral filesystem

Render's filesystem is writable but ephemeral, and there is no disk. This
deployment needs neither.

| State | Where it lives | Survives a restart |
|---|---|---|
| Champion, calibration, threshold, feature contract | image layer, read-only | yes — it is the image |
| Frozen fusion selection, fitted stacker | image layer, read-only | yes |
| Reviewed configurations | image layer, read-only | yes |
| Replay run history | API process memory | **no** |
| Streamlit session state | browser + process memory | no |
| Caddy's own storage | `/tmp` | no, and nothing there matters |

**Replay history disappearing is intended, not tolerated.** A run is a recording
of a synthetic scenario being scored, not an authentication record, and the
console states on the page that runs are process-local and are not retained
across a restart. On a free service that spins down after inactivity, this
happens routinely: a viewer who runs a replay, leaves for an hour and comes back
finds an empty run list and the same frozen champion.

**This system stores no real authentication records, anywhere, ever.** Every
event in the demonstration is fabricated by a seeded generator. There is nothing
to back up, and a backup procedure would imply a persistence that does not
exist.

The container was verified locally with a **read-only root filesystem** and two
`tmpfs` mounts (`/tmp`, `$HOME`), which is stronger than Render requires: it
proves that no runtime write is needed for model state at all. `docker inspect`
confirms the container has no volume and no bind mount of any kind.

---

## 11. Memory

Measured in one container under `--memory 512m --memory-swap 512m` (no swap),
read-only root, running the proxy, the API and the console together, after the
full demonstration — every scenario replayed, plus a detection and an
explanation call.

| Measurement | Value | Share of 512 MiB |
|---|---|---|
| Idle, all three processes serving | 177.6 MiB | 34.7 % |
| Peak sampled during the demonstration | 197.2 MiB | 38.5 % |
| **Peak recorded by the kernel (`memory.peak`)** | **198.3 MiB** | **38.7 %** |
| OOM events / OOM kills (`memory.events`) | 0 / 0 | — |
| Headroom below the limit | 313.7 MiB | 61.3 % |

Approximate per-process resident set at idle (shared pages counted once per
process, so these deliberately do not sum to the total):

| Process | RSS |
|---|---|
| `uvicorn` (API, champion loaded) | 166 MiB |
| `streamlit` (console) | 55 MiB |
| `caddy` (proxy) | 45 MiB |
| `render_entrypoint.py` (PID 1) | 29 MiB |

The margin is comfortable, and the `memory.events` counters record that the
cgroup limit was never even approached — not merely that nothing was killed.
No scientific functionality was reduced to fit: every model family is trained at
build time, the champion is the one selection chose, and the stacked hybrid runs
on every request.

Build-time preparation is **not** included above. It peaks around 340 MiB and
happens on Render's build machine, not on the instance.

---

## 12. CPU and cold starts

Free services get 0.1 CPU and spin down after a period of inactivity. Measured
locally with `--cpus 0.1`, which is as close as a laptop gets to the real thing:

| Measurement | 0.1 CPU | 0.5 CPU |
|---|---|---|
| Cold start to `/healthz` answering 200 | **92.2 s** | 16.8 s |
| `normal_activity` replay (12 events) | 2.1 s | 0.4 s |
| `brute_force` replay (30 events) | 9.6 s | 1.4 s |
| `mixed_attack` replay (29 events) | 8.4 s | 1.2 s |

Almost all of the cold start is importing the scientific stack and loading the
champion; it is dominated by CPU, not by disk.

**Every verdict was identical at both CPU budgets** — the same rules fired the
same number of times in every scenario, and every step was fused by `stacked`.
Throttling changes how long a demonstration takes and nothing about what the
detector decides.

**What a viewer should expect.** The first request after a period of inactivity
waits while Render wakes the service and the three processes start. Extrapolating
from the 0.1 CPU measurement above, that is likely to be a minute or two — an
estimate, since it has not been observed on a real instance. Afterwards the
console is responsive and a replay completes in a few seconds.

The console tolerates an API that is not up yet: it reports the service's status
rather than failing, and its request timeout is raised to 30 seconds for this
deployment. The supervisor allows the API up to 300 seconds to start before
giving up, which is deliberately far more than a throttled cold start needs — a
ceiling tuned to a laptop would turn a slow-but-working start into a crash loop.

Nothing was optimised except deployment mechanics. No threshold, model, feature
or replay semantic was touched, and the `prepare` stage is given no thread or
allocator tuning precisely so that what it fits cannot change.

---

## 13. Deploying from GitHub

None of this has been done. It is the procedure, not a record.

**Prerequisites:** the repository on GitHub, and a Render account. No API token,
no credential in the repository, nothing stored anywhere in this project.

1. **Push the branch** carrying `render.yaml` and `Dockerfile.render`.
2. **In Render, create a Blueprint instance** pointed at the repository. Render
   reads `render.yaml` and offers one free web service named `pad-demo`.
3. **Review what it offers.** It should be exactly one web service, `plan: free`,
   Docker runtime, health check `/healthz`, no disk, no database, no environment
   variables. If it offers anything else, stop: the blueprint in this repository
   declares none of it.
4. **Apply.** The first build runs the full pipeline in the `prepare` stage —
   generate, build features, train, select, freeze, predict, detect, evaluate,
   materialize — then verifies the bundle, prunes, and assembles the runtime
   image. The pipeline itself takes about 70 seconds on the verification machine;
   the whole build on Render's builder will be several minutes longer, and this
   is an estimate rather than a measurement.
5. **Watch the deploy log** for the supervisor's own lines:

   ```
   [entrypoint] public port 10000
   [entrypoint] verifying the baked serving bundle
   Serving bundle verified.
     fusion strategy        stacked
   [entrypoint] api is serving
   [entrypoint] dashboard is serving
   [entrypoint] proxy is serving
   [entrypoint] serving on 0.0.0.0:10000
   ```

6. **Open the service URL.** The analyst console loads. `/healthz` returns the
   liveness document. Nothing else is public.

`autoDeploy` is `false`: a push to the branch does not publish a new public
demonstration on its own. Deploys are triggered from the Render dashboard. This
is the same reasoning as the VPS deployment's refusal to pull from git at
container start — a deployment should be an act, not a side effect.

**No environment variables are set in Render.** Every setting this deployment
needs is baked into the image, which is deliberate: a blueprint variable can be
edited afterwards in a web dashboard by anyone with access to it, and an image
layer cannot. There is therefore no place in the Render configuration where a
model, a threshold, a calibration or a fusion strategy could be introduced.

---

## 14. Logs

Everything goes to stdout and stderr and is collected by Render:

- the supervisor's own lines, prefixed `[entrypoint]`;
- the API's structured JSON logs (`PAD_API_ENVIRONMENT=production` selects the
  JSON renderer);
- Caddy's JSON access log, one object per request;
- Streamlit's startup output.

No log rotation is configured, and none is needed: nothing is written to a file.
Render retains logs for the free tier's own window.

Logs carry no credential and no plaintext identifier. Source addresses are
pseudonymized on arrival and never returned; the API's error contract keeps
tracebacks and submitted values out of both responses and logs.

---

## 15. Redeploying and rolling back

**Redeploy:** trigger a deploy from the Render dashboard. A new image is built,
which means the pipeline runs again. Because the pipeline is deterministic and
seeded, the rebuilt bundle should carry the same fingerprints — the two
preparation runs measured in [§5](#5-the-fingerprint-proof) did. If a rebuild
ever produced a different champion, that is a fact to investigate rather than a
build to retry.

**Roll back:** Render keeps previous deploys and can roll back to one. Because
the serving bundle lives inside the image, a rollback restores the exact
scientific state that image was built with — there is no separate volume that
could be left behind at a newer version. This is one place where baking the
bundle is an advantage over the VPS arrangement rather than a compromise.

**Nothing to purge.** There is no volume and no persistent state, so the VPS
deployment's warning about `--purge` has no analogue here.

---

## 16. Rule availability, and the baseline limitation

The rule catalog registers nine rules and all nine are enabled in this
deployment. They are **not** all demonstrable, and this deployment does not
claim otherwise. The measured three-way split, taken from
[`deployment.md` §16](deployment.md) and unchanged by the platform:

| Rule | Status on the serving path |
|---|---|
| `PAD-BF-001` | fires — brute force, account takeover, mixed attack |
| `PAD-BF-002` | fires — account takeover, mixed attack |
| `PAD-BOT-001` | fires — brute force, password spraying, bot activity |
| `PAD-PS-001` | fires — password spraying, credential stuffing, mixed attack |
| `PAD-DBF-001` | live, evaluates normally, no scenario exercises it |
| `PAD-GEO-001` | live, needs coordinates the scenarios do not carry |
| `PAD-MFA-001` | live, minimum-history gate is never met by a scenario |
| **`PAD-CS-001`** | **unavailable** — requires a behavioural baseline |
| **`PAD-ATO-001`** | **unavailable** — requires a behavioural baseline |

**`PAD-CS-001` and `PAD-ATO-001` cannot fire on any live request**, on this
deployment or on the VPS one. Both need per-account behavioural baseline context
that the serving path does not have: every anchor that reaches their gate returns
`ACCOUNT_ABSENT_FROM_BASELINE`. This is a property of the serving architecture,
not of Render, and not of the thresholds.

They were **not** "fixed". Lowering a threshold or synthesising a baseline to
make a demonstration look complete would produce screenshots of a system nobody
validated. The rules stay enabled, return their honest reason code, and the
limitation is stated here, in `deployment.md`, and in
[`detection-limitations.md`](detection-limitations.md).

Note also that the `credential_stuffing` scenario triggers `PAD-PS-001` rather
than `PAD-CS-001`, and its catalog entry says so.

---

## 17. Limitations

- **Nothing is deployed.** No Render service exists.
- **All data is synthetic.** Every event a viewer sees is fabricated by a seeded
  generator from a tracked configuration. No real authentication record is
  processed, displayed, or stored.
- **The demonstration is not a benchmark.** Replay scenarios are illustrations of
  a frozen system's behaviour, not an evaluation of it. The evaluation is the
  locked TEST evaluation in [`test-evaluation.md`](test-evaluation.md).
- **Two rules cannot fire** — see [§16](#16-rule-availability-and-the-baseline-limitation).
- **The service spins down.** A viewer arriving after a quiet period waits for a
  cold start, and finds no replay history.
- **0.1 CPU is slow.** Replays that take under two seconds locally take up to ten
  seconds here. Verdicts are unaffected.
- **No authentication and no rate limiting.** The public surface is one console
  and one liveness endpoint; scoring and replay control are not publicly
  reachable. A public deployment is still a public deployment.
- **One region, one instance, no redundancy.** It is a demonstration.
- **Build time is an estimate.** The pipeline's own runtime was measured; the
  total build on Render's builder was not.

---

## 18. What was verified locally

All of the following was run against the image `Dockerfile.render` produces, in
a container limited to 512 MiB with no swap, a read-only root filesystem, no
volume, all capabilities dropped and `no-new-privileges`.

| # | Check | Result |
|---|---|---|
| 1 | Image builds; Caddyfile validates at build time | pass |
| 2 | Bundle verifies inside the image | pass, `stacked` |
| 3 | Every bundle fingerprint equals the Compose preparation's | pass, 10/10 |
| 4 | Bundle payload files byte-identical | pass, 3/3 |
| 5 | A second, independent Compose preparation agrees | pass |
| 6 | Container starts read-only under 512 MiB | pass |
| 7 | `/healthz` answers 200 with the liveness document | pass |
| 8 | Public root serves the console | pass |
| 9 | Only three listeners: `127.0.0.1:8000`, `127.0.0.1:8501`, `:$PORT` | pass |
| 10 | 8000 and 8501 refuse TCP from the host | pass |
| 11 | Every API path on the public port lands on the console, not the API | pass, 13/13 |
| 12 | Caddy admin port 2019 not listening | pass |
| 13 | Websocket upgrade through the proxy | `HTTP/1.1 101` |
| 14 | Websocket upgrade with a Render-shaped `Host`/`Origin`/`X-Forwarded-Proto` | `HTTP/1.1 101` |
| 15 | All six readiness components ready | pass |
| 16 | Served fusion strategy and STACKED fingerprint match the bundle | pass |
| 17 | All 7 scenarios complete, fused by `stacked`, 0 unavailable | pass |
| 18 | Each scenario triggers the rules its catalog entry claims | pass |
| 19 | Control scenario fires no rule | pass |
| 20 | Detection over a caller-supplied window | 200, `PAD-BF-001`, severity high |
| 21 | Explanation available; residual `-0.0` | pass |
| 22 | Read-only rootfs refuses writes to `/srv/state`, `/app`, `/etc/caddy` | pass |
| 23 | No volume and no bind mount | pass |
| 24 | Peak memory (`memory.peak`) | 198.3 MiB |
| 25 | OOM kills (`memory.events`) | 0 |
| 26 | Cold start at 0.1 CPU | 92.2 s |
| 27 | Verdicts identical at 0.1 and 0.5 CPU | pass |
| 28 | SIGTERM with a replay in flight: clean stop, reverse order | exit 0 |
| 29 | API lifespan completed on shutdown | pass |
| 30 | A killed child stops the container | exit 4 |
| 31 | Public port refuses connections after that | pass |
| 32 | Restart preserves the champion and clears replay history | pass |
| 33 | Missing / empty / non-numeric / zero / out-of-range `PORT` refused | exit 2 |
| 34 | Absent bundle refused before anything starts | exit 3 |
| 35 | Arbitrary ports (19080, 34567, 41573) | pass |
| 36 | No container or volume left behind | pass |

---

## 19. Troubleshooting

**The deploy log ends at `refusing to start: PORT is not set`.**
Render sets `PORT` for every web service. Seeing this means the image was run
somewhere that does not — pass `-e PORT=...` when running it locally.

**`refusing to start: the baked serving bundle did not verify`.**
The image is not serviceable. Restarting will not help, because nothing at
runtime produces the bundle. Rebuild. If a rebuild reproduces it, the failure is
in the preparation stage and the build log will say which pipeline step failed.

**The deploy times out waiting for the health check.**
The health check cannot pass until the API and the console are both serving, by
design. On 0.1 CPU a cold start takes around 90 seconds locally and may take
longer on a busy instance. Check the log for `api is serving` and
`dashboard is serving`; if the first never appears, the API is failing to start
and its own error is above it.

**The console loads but never updates.**
The `/_stcore/stream` websocket is not upgrading. Confirm the proxy is running
and that nothing between the browser and Render is stripping `Upgrade` headers.
This was verified locally including with a Render-shaped `Host`, `Origin` and
`X-Forwarded-Proto: https`, so TLS termination at Render's edge is not the
cause.

**The service was fine and is now slow to answer.**
It spun down after inactivity. The first request wakes it; see
[§12](#12-cpu-and-cold-starts).

**Replay history is empty and it was not before.**
The service restarted or was woken from a spun-down state. Replay history lives
in process memory and is not persisted; see [§10](#10-the-ephemeral-filesystem).

**A rule never fires no matter what is replayed.**
Check [§16](#16-rule-availability-and-the-baseline-limitation) first. Two rules
cannot fire on any live request, and three more have no scenario that exercises
them. None of these is a deployment fault, and none should be "fixed" by moving
a threshold.

---

## See also

- [`deployment.md`](deployment.md) — the single-VPS Docker Compose deployment
- [`docker.md`](docker.md) — the local containerized demonstration
- [`api.md`](api.md) — the serving API
- [`dashboard.md`](dashboard.md) — the analyst console
- [`live-replay.md`](live-replay.md) — the synthetic replay demonstration
- [`detection-limitations.md`](detection-limitations.md) — what the detector cannot see
