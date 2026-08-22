# The containerized demonstration

Everything in this document describes a **local** deployment. Nothing here is
published, nothing here is reachable from another machine by default, and none
of it is a production deployment of anything.

```bash
docker compose up --build
```

That is the whole workflow. The first run takes a few minutes to build and about
a minute more to train a champion; after that, starting is seconds.

| | |
| --- | --- |
| API | <http://localhost:8000> |
| Swagger | <http://localhost:8000/docs> |
| Dashboard | <http://localhost:8501> |

---

## 1. Architecture

```
        browser
           │
           ├──────────────────► dashboard  :8501     Streamlit, no artifacts
           │                        │
           │                        │  http://api:8000  (Docker DNS)
           │                        ▼
           └──────────────────► api        :8000     FastAPI, all scientific state
                                    │
                                    ▼
                            /srv/state  (read-only)
                                    ▲
                                    │  written once, then never again
                                 prepare                one-shot job, exits
```

Three services and one ordering that matters:

```
prepare  ──►  api  ──►  dashboard
```

**`prepare` is a job, not a service.** It runs the project's real pipeline into a
named volume and exits. `api` starts only after it reports success — Compose's
`service_completed_successfully`, not `service_started`, so the API can never
come up against a state root a failed or half-finished job left behind.

**All scientific state is in the API container.** The dashboard gets no volume,
no artifact path, and no scientific configuration; the only detection it can
display is one the API performed. That boundary is not new to containers — the
console has never been able to detect anything — but the deployment declines to
give it the means anyway.

---

## 2. Prerequisites

* Docker Engine with Compose v2 (`docker compose`, not `docker-compose`).
  Verified against Docker **29.7.2** and Compose **5.4.0**.
* About 1.5 GB of free disk. The two images share every layer but the last: they
  report 1.08 GB each and occupy 1.08 GB **together**, of which 57 kB is unique
  to one of them. The state volume is a few megabytes.
* No internet access is needed after the images are built. Nothing in the running
  system makes an outbound request.

Nothing else. There is no database to provision, no secret to supply, and no
account to create — the service accepts no credential material.

---

## 3. Images

One `Dockerfile`, one builder, and two runtime targets that differ only in the
process they start.

| Stage | What it is |
| --- | --- |
| `builder` | `uv sync --frozen --no-dev` into `/app/.venv`. Nothing from here reaches a runtime image except the virtual environment. |
| `runtime` | The venv, `configs/`, `.streamlit/`, the preparation script, and an unprivileged account. No source tree, no tests, no toolchain. |
| `api` | `runtime` + the uvicorn command. Also the image the `prepare` job runs in. |
| `dashboard` | `runtime` + the Streamlit command and its address overrides. |

### Base image

```
python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28
```

Debian 13 (trixie), CPython 3.12.13. **Pinned by digest as well as by tag.** The
tag says what it is to a human; the digest is what actually gets pulled, so a
rebuilt `python:3.12-slim` upstream cannot silently change what this deployment
was verified against. Bumping it is a deliberate edit with a rebuild behind it.

`uv` is pinned too (`0.12.5`). An installer that resolved differently between two
builds would make "deterministic install from the lockfile" a claim about the
lockfile only.

### Why the preparation job shares the API image

A preparation step that trained under a different resolution of scikit-learn
than the API loads would publish an artifact the API might refuse — and the
refusal would arrive at startup rather than at build time. Sharing the image
makes the two provably the same code.

### What is deliberately absent

No compiler and no build toolchain: every wheel this project needs publishes a
manylinux build for CPython 3.12 on x86_64, so nothing is compiled at image build
time either. No package-manager cache. No `curl` or `wget` — the health checks
use the interpreter the image already ships. No `.git`, no editor or local-tool
state, no virtualenv from the host, no tests, no test caches, no coverage output,
no presentations, and no `.env`. And no trained model: a model is an *output*,
and an image carrying one would make the image the provenance of a scientific
decision.

`pip` is present, because the Python base image ships it. It cannot install
anything: the root filesystem is read-only and the process is unprivileged.

### `.dockerignore`

Written as **exclude everything, then re-admit what the build needs**:

```
*
!pyproject.toml
!uv.lock
!README.md
!src
!configs
!.streamlit
!scripts/prepare_demo_bundle.py
```

…followed by re-exclusions for anything cached, generated, local, or secret that
could otherwise ride in on one of those directories. A denylist is only as good
as the last person to remember to extend it; this way a new top-level file is
excluded until somebody names it on purpose. A unit test asserts both halves —
that the private things stay out, and that everything the Dockerfile copies is
admitted.

---

## 4. The serving bundle, and how the container gets one

This is the part that decides whether a containerized demonstration is honest.

The API requires verified frozen scientific state. It must not train at startup,
must not select a champion at startup, must not reconstruct a stacked hybrid at
startup, must not open the TEST split at startup, and must not substitute a
different fusion strategy when the selected one is unavailable. It does none of
those things — an import-time guard in the serving module refuses to let a
fitting function into its namespace at all.

But `artifacts/` and `models/` are untracked, and a fresh clone has no champion.
So the state is produced **offline, once, by something that is not the service**:
the `prepare` job, running `scripts/prepare_demo_bundle.py`.

### What the job runs

The project's real CLI, one command at a time:

```
data generate        synthetic events, from a seeded tracked configuration
features build       point-in-time features, splits, the manifest
ml catalog           the reviewed feature allowlist, drafted from the catalog
ml train             every enabled family, published as immutable runs
ml select            a champion, chosen on validation evidence
ml freeze-champion   the champion, sealed
ml predict           TRAIN, VALIDATION and TEST scored under the frozen model
detection run        the Phase 4 rule engine over the same snapshots
ml evaluate          the locked TEST evaluation, which selects the hybrid
deploy materialize   the serving bundle the API loads read-only
```

There is no shortcut in that list and no stand-in for any stage. Every artifact
the API serves is the output of the command that normally produces it.

**The script decides nothing.** Which champion wins and which strategy is
selected are outcomes read from the commands' published evidence. It has no
option that names a model, a threshold, or a strategy — a test asserts that of
the source — and it reports what the pipeline chose:

```
  champion scope         d85a151c597b8b9eca0cd570b236aab91cb3cb13f021ee23576421fa1b9f5e90
  fusion selection       selected
  selected strategy      stacked
```

If a run ever selected something other than `stacked`, that would be a fact to
report rather than a failure to retry differently.

### The configurations it is built from

Four tracked, reviewed files. Nothing is generated except by the commands above.

| File | What it fixes |
| --- | --- |
| `configs/data/synthetic-demo.yaml` | Four hours of synthetic traffic, ~4 100 events, seed 606 |
| `configs/features/feature-demo.yaml` | 1m/5m window ladder, 35/20/45 split, 5m purge |
| `configs/detection/rules-demo.yaml` | Six rules pointed at the 5-minute ladder |
| `configs/ml/model-demo.yaml` | Demonstration support floors; three model families |

Two of these deserve their reasons stated here as well as in the files.

**Why 35/20/45 rather than 60/20/20.** The synthetic generator draws campaign
start offsets from the *first part* of a stream's span — 50% of it for bot
activity, 60% for spraying and stuffing, 70% for brute force, 80% for impossible
travel. The tail of any generated dataset is therefore benign by construction,
and a conventional 60/20/20 cut hands train nearly every attack and leaves
validation with one or two campaigns — which the ML eligibility audit correctly
refuses as insufficient validation support. Moving the boundaries earlier puts
both evaluation windows inside the attack-bearing region.

**Why `rules-demo.yaml` matches the test fixture exactly.** The live/replay
scenario catalog publishes, per scenario, which rules a run is expected to
trigger, and those expectations are asserted against the integration fixture's
rule configuration. A container serving a *different* rule configuration would
still be honest about what it computed and dishonest about what it documented.
A unit test asserts the two are equal, so they cannot drift.

### Idempotence

The job writes a receipt (`/srv/state/prepared.json`) last, and only on success.
A second `docker compose up` finds it and skips the pipeline. A receipt whose
bundle manifest is missing, whose schema version is unfamiliar, or which will not
parse is treated as absent — rebuilding is cheap, and serving a half-published
bundle is not.

### It is the same state a host run produces

The champion scope key from the container is byte-identical to the one a host run
of the same script produces. Same seeds, same configurations, same commands, same
answer.

### Sizing, and what these numbers are not

**No figure this deployment reports is a performance claim about this system.**
The dataset is four hours long. Its evaluation windows are far too small for a
per-scenario metric to mean anything, and `configs/ml/model-demo.yaml` says so at
the top. `configs/data/synthetic-ml-development.yaml` — 30 days — is the
configuration that exists for measurement, and it is deliberately not what a
container runs.

---

## 5. Volumes and read-only state

| Service | Mount | Mode |
| --- | --- | --- |
| `prepare` | `serving-state:/srv/state` | read-write, for the length of one job |
| `api` | `serving-state:/srv/state` | **read-only** |
| `dashboard` | — | none at all |

The champion, its preprocessor, its calibrator, its threshold, the frozen fusion
selection and the materialized stacked state are all under that mount, and the
serving process has no business writing to any of them. Replay state is
process-local memory and needs no filesystem at all.

All three root filesystems are read-only. The few paths a framework insists on
(`/tmp`, and `$HOME` for Streamlit) are tmpfs and vanish with the container.

**No bind mount is required for a normal `docker compose up`,** and no service
mounts a host path. There is no development bind-mount profile in this file; if
one is ever wanted it belongs in a separate compose file, because a source tree
mounted over `/app` would make the image's contents irrelevant to what runs.

---

## 6. Network

One project-scoped bridge network, `demo`. The dashboard reaches the API across
it by service name — `http://api:8000`, because inside the network `127.0.0.1`
would name the console's own container.

Published ports are bound to the **host's loopback interface**:

```
127.0.0.1:8000 -> api:8000
127.0.0.1:8501 -> dashboard:8501
```

The demonstration is reachable from the machine running it and from nowhere else.
Publishing it to the local network is a decision somebody should have to make on
purpose.

Neither service needs outbound internet at runtime. No replay scenario performs
any external networking — the scenarios are fixed synthetic event lists, they use
reserved documentation address space where an address appears at all, and an
architectural test asserts that no replay module imports a network client.

---

## 7. Health and readiness

| Check | What it answers |
| --- | --- |
| `GET /health` | Is this process serving? |
| `GET /ready` | Can it detect — feature contract, rules, artifacts, champion, fusion, replay? |
| `GET /_stcore/health` | Is Streamlit serving? |

**The container health check is `/health`, deliberately.** A container that
reported itself permanently unhealthy because a scientific component failed would
hide that failure behind an orchestration state, instead of surfacing the
service's own reason codes — which the console renders and which the startup log
states. Readiness is a service concept, and the service is the right place to
read it.

Startup ordering does not depend on the health check: it is handled by `prepare`
completing successfully before the API starts at all.

The checks use the interpreter the image already ships. Installing an HTTP client
to ask a Python process whether it is alive would add a binary, a CVE surface,
and a layer for nothing. Intervals are 15 s with a 30 s start period — a health
check every second is a load generator with a nice name.

---

## 8. Logs

```bash
docker compose logs -f api
docker compose logs -f dashboard
docker compose logs prepare
```

The API's startup line states what it loaded:

```
serving runtime initialised components={'feature_contract': 'ready',
  'rule_engine': 'ready', 'model_artifacts': 'ready', 'ml_champion': 'ready',
  'fusion': 'ready', 'replay': 'ready'} ready=True
```

What container logs never carry: credentials or secrets of any kind (there are
none to carry), a pseudonym-bearing table, a feature vector, model coefficients,
a full manifest, an environment dump, or a stack trace returned to a client. A
traceback in a log is a leak, and a verification check asserts the API's log has
none.

---

## 9. Stopping

```bash
docker compose down            # stop, keep the serving state
docker compose down -v         # stop, and discard the serving state too
```

or the wrappers, which are thin and contain no scientific logic:

```bash
./scripts/start_demo.sh
./scripts/stop_demo.sh [--purge]
```

`down` sends SIGTERM to PID 1. Both images use exec-form commands, so uvicorn and
Streamlit *are* PID 1 and receive it directly — a shell-form command would put
`/bin/sh` there, which does not forward signals, and the FastAPI lifespan
shutdown would never run. The lifespan is what cancels active replay tasks; the
API is given a 20-second graceful-shutdown window, and in practice uses a
fraction of it.

Stopping during an active replay is safe and leaves nothing behind: the run's
task is cancelled, the store is memory, and the state volume was never writable
from that process.

Keeping the volume makes the next start seconds rather than a minute. Discarding
it is equally safe — the next start rebuilds the identical state from the same
seeded configurations.

The restart policy is `on-failure:3`, not `unless-stopped`. A transient crash
recovers; a persistent one stops and shows itself rather than looping; and a
machine reboot does not silently start a demonstration nobody asked for.

---

## 10. Restart semantics

| | Survives an API restart? |
| --- | --- |
| Champion, calibrator, threshold, frozen selection, stacked state | **Yes** — read-only volume, byte-identical |
| System status, model info | **Yes** — identical field for field |
| Replay run history | **No** — process memory, and gone |

The second row is the one worth being explicit about. **Replay history is not
persistence and is not presented as any.** It lives in the API process's memory,
it is bounded, and it is discarded when that process restarts. The console states
this on the page; it does not hide the loss, and a run identifier from before a
restart returns a typed "run not found" rather than a fabricated document.

---

## 11. Security model

| Property | How |
| --- | --- |
| Runtime UID ≠ 0 | `USER pad:pad`, uid/gid 10001, `nologin` shell |
| No privileged containers | never declared; asserted against the running container |
| No Docker socket | never mounted; asserted absent from both containers |
| No host networking | project bridge only |
| No host filesystem mounts | one named volume, and the console gets none |
| No added capabilities | `cap_drop: [ALL]`, no `cap_add` |
| No privilege escalation | `no-new-privileges:true` |
| Read-only root filesystems | all three services, with explicit tmpfs |
| Serving state immutable to the API | mounted `:ro` |
| No secrets in images | none exist to bake; `.env` and `secrets.toml` excluded |
| No SSH keys, no `.git`, no local-tool state | excluded by an allowlist-shaped context; `/app` holds four entries and a test names them |

**The dashboard cannot change a scientific setting.** Its settings module refuses
to declare one — an import-time guard holds the prohibited names — and the
container is given no artifact path, no state volume, and no `PAD_API_*` variable
at all.

**The API still refuses every override, over a real socket.** Credential-shaped
fields, `model_id`, `decision_threshold`, `fusion_strategy`, `artifact_root`,
external target URLs, filesystem paths, unknown scenario identifiers and
arbitrary timing values are all refused with typed error codes, and no refusal
echoes back the value or names a path.

**Compose exposes no scientific control.** Every `PAD_API_*` variable in
`compose.yaml` answers *where* or *how much* — a directory, a port, a log level,
a batch ceiling, a facility switch. Two tests enforce it: one checks the declared
names against the settings modules' own prohibition lists, and a stricter one
refuses any variable name merely *containing* `MODEL`, `THRESHOLD`, `FUSION`,
`CHAMPION`, `CALIBRAT`, `SCORE`, `PASSWORD`, `SECRET`, `TOKEN`, or `API_KEY`.

---

## 12. Resources

Measured with `docker stats` on the verified local run, not estimated. The
working column is the highest of a roughly one-per-second sample taken while the
preparation pipeline ran and while replays were driven through the API, so read
it as an observed peak rather than a true maximum — a spike between two samples
would not appear.

| | Idle | Observed peak |
| --- | --- | --- |
| `api` | 105 MiB | 121 MiB (running replays) |
| `dashboard` | 51 MiB | 65 MiB |
| `prepare` | — | 340 MiB (feature build and three model fits) |

**Disk:** the two images share every layer but ~57 kB, so both together occupy
about 1.08 GB rather than twice that. The state volume is a few megabytes.

Declared ceilings: 2 GB for `prepare`, 1 GB each for `api` and `dashboard`. The
preparation ceiling is generous on purpose — a job killed by the OOM killer would
leave a half-published state root behind.

**Recommended for a comfortable demonstration:** 2 CPU cores and 4 GB of RAM free,
plus ~1.5 GB of disk. It will run in less; the preparation step is the only part
that is not nearly idle, and it lasts about a minute.

**Not a capacity claim.** These are observations of one four-hour synthetic
dataset on one machine. Nothing here has been load-tested, and no figure implies
a request rate this system can sustain.

---

## 13. Troubleshooting

**`prepare` exits non-zero.** Read `docker compose logs prepare`. It names the
stage and prints that command's output. The most common cause on a modified
configuration is the ML eligibility audit refusing insufficient validation
support — see §4 on why the split fractions are what they are.

**The API is healthy but `/ready` says otherwise.** That is the intended
behaviour, not a fault: the process is serving and something scientific did not
load. `curl -s localhost:8000/ready` names the component and the reason, and the
console shows the same thing on every page.

**The dashboard shows "offline".** The API is not answering on `http://api:8000`.
Check `docker compose ps`. The console handles this state deliberately — it shows
fixed text, invents nothing, and recovers on its own when the API returns.

**Port already in use.** Something else holds 8000 or 8501 on the host. Stop it,
or change the published ports in `compose.yaml` (the host side only — the
container side and the internal URL must stay as they are).

**A replay run vanished.** The API restarted. See §10.

**Rebuilding from scratch.** `docker compose down -v && docker compose up --build`.

---

## 14. Current limitations

**Local only.** Nothing here is deployed, published, or reachable from another
machine. There is no TLS, no authentication, and no rate limiting, because there
is no exposure to protect — the ports are bound to loopback.

**Demonstration scale.** See §4. The dataset is four hours long and no figure it
produces is a performance claim.

**Replay history is memory.** See §10.

**Two rules cannot fire on any live request, and this is a v0.6.0 blocker.**
`PAD-CS-001` (credential stuffing) and `PAD-ATO-001` (account takeover) gate on a
fitted behavioural baseline. The serving path builds its feature engine without
one, so `user_in_baseline` and `source_in_baseline` are `False` and every
`is_new_*_for_user` novelty flag is `None` on every live request — replay or
hand-submitted. Both rules report *insufficient data* rather than a verdict.

Milestone 4 audited whether the baseline could be added to the serving bundle on
the same terms as the stacked state — materialized offline, deterministic,
fingerprinted, loaded read-only, failing closed. **It was not done, for two
independent reasons, and neither is a matter of effort.**

*It would not achieve the goal.* A baseline was fitted from this deployment's own
TRAIN split and loaded into a feature engine, and the replay scenarios were run
through it. `user_in_baseline` came back `False` and all five `is_new_*_for_user`
flags came back `None` — unchanged. The replay catalog's identities are
content-addressed synthetic pseudonyms that no training population contains, by
design, so a baseline fitted on *any* dataset leaves those flags exactly where
they are. Making the rules fire would mean drawing scenario identities from the
deployment's own training population, which would couple a content-addressed
catalog to whichever dataset a deployment happened to train on.

*It is a scientific-contract change, not a containerization one.* The bundle
manifest has no field for a baseline; adding one bumps `BUNDLE_SCHEMA_VERSION`,
extends the fingerprint chain, gives `deploy materialize` a feature-layer input
it does not currently take, and adds a baseline loader to the serving path. That
is a milestone, and doing it inside a containerization milestone would be doing
it without review.

Nothing was loosened to work around this. No rule threshold was changed, no
`min_novel_context_count` was lowered, and no baseline was synthesised for a
demonstration. The two rules stay enabled, keep reporting insufficient data, and
say why — in the scenario catalog's `limitations`, in
[live-replay.md](live-replay.md) §3, in [api.md](api.md) §12, and here.

---

## Related documents

* [api.md](api.md) — the detection service, its startup, and the serving bundle
* [dashboard.md](dashboard.md) — the analyst console
* [live-replay.md](live-replay.md) — the synthetic replay demonstration
* [reproducibility.md](reproducibility.md) — what determinism means in this project
