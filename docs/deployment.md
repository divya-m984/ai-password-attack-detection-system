# Public deployment on a single Linux VPS

> **No VPS deployment of this project exists.** The project's public demo runs on
> Render (<https://pad-demo.onrender.com>, see
> [`render-deployment.md`](render-deployment.md)); this document describes the
> *other* prepared target, which has no server and no domain name behind it. The
> exact Compose topology below was brought up on a development machine, exercised
> end to end, and torn down. Every hostname here is `example.org`, which RFC 2606
> reserves precisely so that documents like this one cannot accidentally name
> somebody's machine.

`docs/docker.md` describes the local demonstration: `docker compose up --build`,
two ports on loopback, one command. This document describes what is added on top
of it to face the internet — and only what is added. The container images, the
preparation job, the read-only serving state and the frozen scientific decisions
are all unchanged, and deliberately so: a deployment that rebuilt the system
differently would be publishing something the local verification never ran.

**A single VPS is one of two prepared targets.**
[`render-deployment.md`](render-deployment.md) describes the other: a single
Render free web service running the same application as one container, adapted
for a platform with no persistent disk and no private network between services.
Neither replaces the other, and neither is deployed. Where the two overlap —
the routing policy, the security headers, the CSP audit, the per-rule
availability measurement — this document is the reference and that one links
back to it.

---

## 1. The architecture

```
                 internet
                    |
             80 / 443  (the only published ports)
                    |
        +-----------v-----------+
        |         proxy         |   caddy:2.10-alpine, pinned by digest
        |  security headers     |   the only public listener
        |  TLS (hostname mode)  |
        +-----------+-----------+
                    |  Docker DNS, project bridge network
       +------------+------------------------+
       |                                      |
+------v-------+                     +--------v---------+
|  dashboard   |  http://api:8000    |       api        |
|  Streamlit   +-------------------->+  FastAPI         |
|  no volume   |   server-side       |  detection       |
|  no artifact |   httpx client      |                  |
+--------------+                     +--------+---------+
                                              |  read-only
                                     +--------v---------+
                                     |  serving-state   |  named volume
                                     |  frozen bundle   |  written once by
                                     |  champion, rules |  the `prepare` job
                                     +------------------+
```

One virtual machine. Four containers, three of them long-lived. Two published
ports. No database, no message broker, no orchestrator, no second host.

**Why a reverse proxy at all, when Streamlit and FastAPI both speak HTTP?**
Three reasons, and none of them is "because that is how it is done":

1. It is the only thing that has to face the internet, so it is the only thing
   whose attack surface has to be reasoned about as a public one. Caddy is a
   single static Go binary with no interpreter and no plugin loaded here.
2. It terminates TLS and renews certificates without a certificate ever existing
   in this repository. See §6.
3. It puts the security headers on in one place, so the two applications behind
   it cannot disagree about them.

**Why the API is not published.** The dashboard's API client runs *server-side*,
inside the dashboard container. A browser never talks to the detection service —
it talks to Streamlit, and Streamlit talks to the API across the project network.
So publishing the API adds a public surface that no part of the demonstration
uses. §5 covers the routing policy and the one case for relaxing it.

---

## 2. Sizing

Measured on the local topology run, not estimated. Sampled with `docker stats`
at roughly 1 Hz while three concurrent replays and repeated page loads went
through the proxy:

| Container | Idle | Observed peak | Ceiling set |
|---|---|---|---|
| `api` | 110 MiB | 124 MiB | 768 MiB |
| `dashboard` | 49 MiB | 63 MiB | 384 MiB |
| `proxy` | 14.5 MiB | 22.4 MiB | 128 MiB |
| `prepare` (one-shot) | — | 340 MiB | 1 GiB |

**Recommended machine: 2 GiB RAM, 1–2 vCPU, 25 GB disk.** That is a
DigitalOcean `s-1vcpu-2gb` or the equivalent elsewhere. The long-lived ceilings
add up to 1280 MiB, leaving roughly 700 MiB for the kernel, journald, sshd and
the Docker daemon itself. `prepare` is excluded from that sum because it is the
only thing running while it runs — the API does not start until it has exited.

Disk: the two application images share all but ~57 kB of their layers and occupy
about 1.08 GB together; Caddy's image adds about 50 MB. Allow ~5 GB for images,
build cache and the serving-state volume, and the rest is headroom.

**1 GiB is not enough.** The build compiles nothing, but resolving and unpacking
the scientific stack, and then fitting three model families in the preparation
job, will meet the OOM killer. If you must use a 1 GiB machine, build the images
elsewhere and push them to a registry rather than building on the server.

**On swap.** A 2 GiB machine running this does not need swap. If you add some as
emergency protection against an unexpected spike, understand what it is: it stops
the OOM killer from choosing a victim, and it does so by making the process
slow instead of dead. It is not additional memory capacity, and a deployment
that routinely touches swap is a deployment that needs a larger machine.

**This is not a capacity claim.** Nothing here has been load-tested. These are
the numbers one demonstration produced on one machine.

---

## 3. Prerequisites

* A fresh Ubuntu 24.04-class server. Nothing here is Ubuntu-specific except the
  bootstrap script's package management; the deployment itself is plain Docker
  Compose.
* Docker Engine with the Compose **v2** plugin, **v2.24 or newer**. The
  deployment uses Compose's `!reset` merge tag, which arrived in 2.24, and
  `depends_on: service_completed_successfully`. `docker-compose` v1 will not
  work, and neither will an early v2. `docker compose version` tells you.
  (Verified against Docker 29.7.2 and Compose 5.4.0.)
* SSH access with a key. Not a password — see §8.
* Nothing else. No cloud provider CLI, no API token, no account credentials.
  This deployment creates no cloud resource and talks to no provider API.

### DigitalOcean-compatible, not DigitalOcean-specific

The instructions below were written against a DigitalOcean Droplet because that
is the likely target, but nothing in them is DigitalOcean. Any provider that
gives you an Ubuntu VM, a public IPv4 address and a firewall will do — Hetzner,
Linode, Vultr, EC2, or a machine under a desk. Where a step is provider-specific
it says so and gives the generic equivalent.

---

## 4. Bootstrapping the server

`scripts/deploy/bootstrap_server.sh` is **server-only**. It refuses to run
without `--confirm`, refuses to run on anything that is not Ubuntu, and refuses
to run on a machine with a graphical session — a developer who invokes it by
accident on a laptop gets told it is the wrong machine, not told to use sudo.

```bash
# on the server, as root
git clone https://github.com/<owner>/ai-password-attack-detection-system.git
cd ai-password-attack-detection-system
bash scripts/deploy/bootstrap_server.sh --confirm
```

What it does:

* installs Docker Engine, buildx and the Compose plugin from Docker's own apt
  repository, with the signing key fetched over HTTPS rather than embedded here;
* enables and starts the Docker service;
* creates an unprivileged `pad` account with **no password** (`--disabled-password`)
  and adds it to the `docker` group;
* copies root's `authorized_keys` to that account, so it is reachable by key
  immediately and password authentication never has to be enabled;
* **reads** the effective sshd configuration and warns if password
  authentication or root login is enabled.

What it deliberately does not do:

* it never edits `sshd_config` — a bootstrap that rewrites sshd on a machine you
  are connected to over SSH is a bootstrap that can lock you out of it;
* it never touches Git identity, and performs no Git operation at all;
* it opens no firewall port unless you pass `--with-ufw`, and then exactly 22,
  80 and 443, with SSH allowed first so enabling the firewall cannot end the
  session that enabled it;
* it clones nothing, builds nothing, and starts nothing.

---

## 5. Getting the source onto the server, at a known revision

Clone and check out a **reviewed** revision. Do not deploy a moving branch.

```bash
# as the pad user
git clone https://github.com/<owner>/ai-password-attack-detection-system.git
cd ai-password-attack-detection-system

# Phase 6 development: the working branch, explicitly
git checkout phase-6-deployment-demo

# For a release, prefer a tag:
# git checkout v0.6.0

# Record what you are about to deploy. Put this in your notes.
git rev-parse HEAD
```

`git rev-parse HEAD` is the deployment's identity. When something behaves
unexpectedly, the first question is which commit is running, and the only way to
answer it is to have written it down.

**Nothing pulls automatically.** No container runs `git` at startup, no service
fetches, and there is no webhook. Updating is §12, and it is a decision somebody
makes.

**No rsync of untracked state.** Everything the deployment needs is tracked: the
Dockerfile, both Compose files, the routing policies, the demonstration
configurations, and the preparation script. The one untracked file is
`.env.deploy`, which you create on the server from a tracked template.

---

## 6. Routing policy: what the internet can reach

Two reviewed policies live in `deploy/caddy/`. One variable selects between them;
neither can be edited from the environment, and neither accepts an upstream URL
from anywhere.

### Default — `deploy/caddy/Caddyfile`

| Public path | Goes to |
|---|---|
| everything | the Streamlit console |

That is the whole table. The API is not reachable from the internet at all. The
console still reaches it over the project network, so **detection, explanation,
replay and every dashboard view work in full** — the only thing missing is a way
for a stranger to call the service directly.

A consequence worth knowing before you meet it: `GET /health` through the proxy
returns the console's page, not a health document. Health-check the deployment
with `docker compose ps` over SSH, or select the policy below.

### Optional — `deploy/caddy/Caddyfile.api-docs`

Selected with `PAD_PROXY_CADDYFILE=./deploy/caddy/Caddyfile.api-docs`. Adds
these paths, and only these:

| Public path | What it is |
|---|---|
| `/docs`, `/redoc`, `/openapi.json` | the interactive documents and the schema |
| `/health`, `/ready`, `/version` | liveness, readiness, contract versions |
| `/api/v1/system/status` | which layers this deployment runs |
| `/api/v1/model/info` | the frozen champion's public identity |
| `/api/v1/rules` | the public rule catalog |
| `/api/v1/demo/scenarios` | the replay scenario catalog |

Every one is a GET that reads state the service already publishes. Use this for
a viva or a review, where the API contract is part of what is being shown.

### Never published, under either policy

`POST /api/v1/detect`, `/api/v1/detect/batch`, `/api/v1/explain`,
`POST /api/v1/demo/runs`, `POST /api/v1/demo/runs/{id}/stop`, and
`GET /api/v1/demo/runs`.

Scoring is *bounded* — 1 MiB body, 500 events, no path input, no credential
field, no scientific override — but bounded is not free: every request builds a
point-in-time feature window and runs a rule pass and a model pass. Until this
deployment has rate limiting (see §14), an unauthenticated public POST that costs
CPU is a denial-of-service surface on a 2 GiB machine, and the demonstration does
not need it. Swagger's "Try it out" button will therefore return 404 or 405 for
the scoring endpoints. That is the proxy refusing the route, not the service
failing.

Also never published, in either policy and by construction rather than by rule:
artifact files, the serving bundle, any filesystem path, the Docker API, and any
preparation or training endpoint — the service has none of the last two.

---

## 7. TLS and hostnames

Two documented modes. Both are the same configuration with a different
`PAD_SITE_ADDRESS`.

### Mode 1 — IP-only smoke test

```
PAD_SITE_ADDRESS=:80
```

Plain HTTP on the machine's address. No DNS record required, no certificate
issued, nothing to get wrong before you have seen the thing work once. Browse to
`http://<droplet-ip>/`.

Do the smoke test first. A DNS or ACME problem and an application problem look
identical from a browser, and this separates them.

### Mode 2 — hostname and automatic HTTPS

1. Point an `A` record at the machine's address and wait for it to resolve.
   `dig +short demo.example.org` from somewhere other than the server.
2. Set the hostname:

   ```
   PAD_SITE_ADDRESS=demo.example.org
   ```

3. Restart the proxy. Caddy obtains a certificate over ACME on first request,
   redirects HTTP to HTTPS, and renews automatically thereafter.

Certificates live in the `caddy-data` named volume. **No certificate and no
private key is ever stored in this repository, and none may be committed.** Port
80 must stay open even in mode 2: the ACME HTTP challenge uses it, and the
HTTP→HTTPS redirect lives there.

To receive expiry notices, uncomment and set `email` in the Caddyfile's global
block. It is deliberately not an environment variable — an address wired to one
would end up in a tracked example file.

### HSTS

```
PAD_HSTS=max-age=0            # default
PAD_HSTS=max-age=31536000     # after HTTPS is confirmed working
```

The default is not an omission. `max-age=0` instructs a browser to *forget* any
pin it holds for this origin, which is the honest instruction from an HTTP-only
deployment. Turn it on only once mode 2 serves reliably, and understand that it
is hard to undo: a browser that has seen a long `max-age` will refuse plain HTTP
to that hostname until it expires.

---

## 8. Security headers

Set once, in a shared Caddyfile snippet, so the two routing policies cannot
drift apart. Read off a real response during the local topology run:

```
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
X-Frame-Options: DENY
Content-Security-Policy: frame-ancestors 'none'
Permissions-Policy: accelerometer=(), autoplay=(), camera=(), display-capture=(),
                    encrypted-media=(), fullscreen=(self), geolocation=(),
                    gyroscope=(), magnetometer=(), microphone=(), midi=(),
                    payment=(), usb=()
Strict-Transport-Security: max-age=0
```

The `Server` banner is removed rather than replaced.

### Why the CSP is `frame-ancestors` only

This was audited before it was written, which is the part worth recording.

Streamlit's client is a bundled single-page application. It evaluates generated
code and installs inline styles at runtime, so a `script-src` or `style-src`
policy tight enough to be worth having stops the console from rendering at all.
A policy loose enough to keep it working would have to permit `'unsafe-inline'`
and `'unsafe-eval'` — which is a policy that claims a protection it does not
provide, and which is worse than none because it reads like a control in an
audit.

`frame-ancestors` is the directive that is both meaningful here and compatible:
it is the one part of a CSP that constrains embedding rather than execution.
It is set to `'none'`, duplicated by `X-Frame-Options: DENY` for browsers that
honour only the legacy header, and a test pins it so it cannot later be
"improved" into something that breaks the page.

The console was exercised through the proxy with these headers in force: the
document renders, `/_stcore/health` answers, and the `/_stcore/stream` websocket
upgrades to `HTTP/1.1 101`.

---

## 9. Cloud firewall

Configure this at the provider, not only on the host — a host firewall protects
nothing if the host is misconfigured, and a cloud firewall is enforced before
packets reach the machine.

**Inbound**

| Port | Source | Why |
|---|---|---|
| 22/tcp | your address, or your VPN's | administration |
| 80/tcp | anywhere | the site, and the ACME HTTP challenge |
| 443/tcp | anywhere | the site, in hostname mode |

Everything else denied. In particular **never** open:

* 8000 or 8501 — the deployment does not publish them, and opening them would
  publish nothing, but a rule that exists will one day meet a configuration that
  uses it;
* 2375 / 2376 — the Docker daemon. This deployment never listens on them;
* 2019 — Caddy's admin API. It is disabled here (`admin off`, verified: the
  proxy container listens on `:80` and nothing else), and it is not published;
* any database port. There is no database.

Restricting SSH to a single address is the right default. If your address moves,
use your provider's console or a bastion rather than opening 22 to the world.

**Outbound**, honestly:

* **Nothing is required at run time.** No replay scenario performs outbound
  networking, the detection path reads nothing it was not given at startup, and
  no container phones home — Streamlit's usage statistics are off in the image.
* **HTTPS to the ACME provider is required in hostname mode**, and only there.
  Blocking it means no certificate.
* **HTTPS is required at build time**, to fetch base images and Python wheels.
  If your policy forbids general egress, build elsewhere and push to a registry.

---

## 10. The deployment user

The containers already run unprivileged: uid 10001 inside, `cap_drop: ALL`,
`no-new-privileges`, and a read-only root filesystem. The host should not operate
permanently as root either.

* **Bootstrap needs root** — installing packages and creating an account does.
  That is one command, run once.
* **Everything after that is the `pad` account**: the checkout, the environment
  file, `docker compose`, reading logs.
* **SSH is key-only.** The account is created with no password. Do not add one.

### About the `docker` group

The `pad` account is in the `docker` group so it can run Compose without sudo.
**Be clear about what that is: it is root-equivalent.** A member of the `docker`
group can start a container that bind-mounts `/` and runs as root inside it,
which is unrestricted access to the host. It is a convenience, not a privilege
boundary.

What it *does* buy: routine operations do not run as root, so a mistake in a
Compose command is not automatically a mistake made as root, and the audit trail
names an account rather than `root`. The boundary that actually matters here is
the one around SSH access to the machine.

If you need a real boundary, the answer is rootless Docker or a separate
administrative host — not a differently-named account in the same group.

---

## 11. First deployment

```bash
# as pad, in the checkout
cp .env.deploy.example .env.deploy
$EDITOR .env.deploy          # start with PAD_SITE_ADDRESS=:80

docker compose --env-file .env.deploy \
  -f compose.yaml -f compose.deploy.yaml up -d --build
```

`.env.deploy` is untracked and refused by `.gitignore`. It carries a hostname, a
publish specification, a routing-policy selection and an HSTS max-age — no
secret, and nothing scientific. There is nowhere in it to name a model, a
threshold, a champion, a calibrator or a fusion strategy, and a test asserts that
of the merged Compose configuration rather than trusting the file.

### What happens on that first `up`

1. Both images build. This is the one figure here that is an **estimate rather
   than a measurement** — the local verification ran on a development machine,
   not a droplet. Budget generously: almost all of it is downloading and
   unpacking the scientific stack, so it scales with the machine's network and
   disk rather than its CPU. Watch it with `docker compose ... logs -f`.
2. `prepare` runs the real pipeline offline: generate the synthetic dataset,
   build point-in-time features, build the model catalog, train three families,
   select a champion on validation evidence, freeze it, predict, detect,
   evaluate, and materialize the serving bundle. **Roughly 90 seconds** on the
   verification machine; a smaller droplet will take longer. It then exits.
3. `api` starts only once `prepare` reports success, mounts the bundle
   **read-only**, verifies it, and serves.
4. `dashboard` starts once the API is healthy.
5. `proxy` starts once the console is healthy.

Nothing is fitted inside the serving process, on this run or any later one. The
API has no endpoint that trains, selects, promotes or recalibrates anything, and
the state it reads is on a read-only mount.

### Verify it

```bash
docker compose --env-file .env.deploy -f compose.yaml -f compose.deploy.yaml ps
# prepare  exited (0)
# api      running (healthy)
# dashboard running (healthy)
# proxy    running (healthy)

curl -sI http://<address>/ | head -1        # HTTP/1.1 200 OK
docker compose ... exec api python -c \
  "import json,urllib.request;print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/system/status'))['fusion_strategy'])"
# stacked
```

Then open `http://<address>/` and run a replay scenario from the Live Replay
view. If the scenarios complete and the hybrid column says `stacked`, the
deployment is serving the same frozen system the local verification ran.

### Subsequent starts

`prepare` finds a receipt in the volume and exits immediately — the second `up`
takes seconds rather than a minute and a half, and the state it reuses is
verified rather than assumed. To force a rebuild of the scientific state, remove
the volume (§13) and start again; it rebuilds identically from the same seeded
configurations.

---

## 12. Logs

```bash
DC="docker compose --env-file .env.deploy -f compose.yaml -f compose.deploy.yaml"

$DC logs                 # everything, interleaved
$DC logs -f api          # the detection service
$DC logs -f dashboard    # the console
$DC logs -f proxy        # the public boundary, as JSON access logs
$DC logs prepare         # what the preparation job decided, after the fact
```

Rotation is configured for every service: `json-file`, 10 MiB per file, three
files. That is a hard ceiling of 30 MiB per container — 120 MiB for the whole
deployment — against Docker's default of keeping every byte forever. Recent logs
stay readable; ancient ones stop being the reason a 25 GB disk filled up.

What is in them: structured JSON from the application, one line per request from
the proxy. What is not: no credential, no request body, no password, no token —
the service accepts none of those under any spelling — no absolute host path, and
no stack trace. Tracebacks are logged by exception *type* and never returned to a
client; the error contract exists for exactly that reason.

---

## 13. Update and rollback

Manual, reviewed, and in that order. There is no auto-deploy and there is not
going to be one in this milestone.

### Update

```bash
DC="docker compose --env-file .env.deploy -f compose.yaml -f compose.deploy.yaml"

git fetch --all --tags
git log --oneline HEAD..origin/main      # read what you are about to run
git checkout <reviewed-tag-or-commit>
git rev-parse HEAD                        # record it

$DC build                                 # build first; a failed build changes nothing
$DC up -d                                 # then swap
$DC ps
```

Building before `up` matters: a build that fails leaves the running deployment
untouched, because nothing has been stopped yet.

### Rollback

```bash
git checkout <the-previous-recorded-commit>
docker compose ... build
docker compose ... up -d
```

That is the whole procedure, and it works because the previous revision is
identified by a commit you wrote down at deploy time. If you did not write it
down, `git reflog` and the `org.opencontainers` labels on the old images are your
fallback — but write it down.

### The serving-state volume survives both

Neither `up`, nor `build`, nor `down`, nor a failed start removes it. Only
`down --volumes` (or `./scripts/stop_demo.sh --purge` locally) does, and that is
the point: a failed new build cannot destroy the state the previous one was
serving.

**When `--purge` is dangerous — and when it is not.** Locally it is cheap: the
state is deterministic and the next start rebuilds it byte-identically from the
same seeded configurations. On a server it is not dangerous either, but it is
*expensive*: it forces the full preparation pipeline to run again before the API
can start, so the deployment is down for a minute and a half rather than a few
seconds. It also discards Caddy's certificate store, which means a fresh ACME
issuance and a step toward that provider's rate limit. Purge when you have
changed a demonstration configuration and want the state rebuilt from it. Do not
purge as a reflex when something looks wrong; read the logs first.

---

## 14. What persists, and what does not

| Thing | Survives a restart? | Survives `down`? | Survives `down --volumes`? | Backup needed? |
|---|---|---|---|---|
| Frozen serving bundle, champion, rules | yes | yes | **no** — rebuilt | no, it is reproducible |
| Caddy certificates | yes | yes | **no** — re-issued | no |
| Replay run history | **no** | no | no | no, and it must not be |
| Images and build cache | yes | yes | yes | no |
| `.env.deploy` | yes | yes | yes | it is one file; keep a copy |

**Nothing here needs a backup.** The serving state is the deterministic output of
a seeded pipeline that runs from tracked configurations: `docker compose up`
regenerates it identically, and the local verification proved that across a full
volume purge — the champion scope key and the stacked-state fingerprint came back
byte-identical.

**Replay history is intentionally ephemeral.** It lives in the API process's
memory, is bounded (4 concurrent runs, 24 retained, 256 records each), and is
gone when that process restarts. The dashboard states this on the page rather
than hiding it. It is a demonstration facility, not a record.

**This system stores no authentication records.** Not persistently, not
transiently beyond one request's window, and not at all in any durable store.
There is no database in this deployment. Every event it has ever seen is either
synthetic — fabricated by the replay catalog or the generator — or supplied by a
caller for the duration of one request and then discarded. No plaintext password,
hash, or credential of any kind is ever accepted, stored, or logged.

---

## 15. Public-demonstration abuse bounds

Audited before exposure, because "it is only a demo" is how a demo becomes
somebody's compute.

**Enforced by the service itself**, and therefore true regardless of what proxy
sits in front of it:

| Bound | Value | Where |
|---|---|---|
| Request body | 1 MiB | `RequestSizeLimitMiddleware`, `max_request_bytes` |
| Events per request | 500 (hard ceiling 5 000) | `max_batch_events` |
| Concurrent replay runs | 4 | `ReplayLimits.max_active_runs` |
| Retained runs | 24, oldest finished evicted first | `max_retained_runs` |
| Records per run | 256 | `max_records_per_run` |
| Wall-clock per run | 300 s, then abandoned with a reason | `max_run_seconds` |
| Timeline page | 100 records | `max_timeline_page` |

**Enforced by the perimeter**: the proxy caps a request body at 1 MiB, matched to
the service's own ceiling rather than chosen independently.

**Structurally impossible**, which is stronger than bounded:

* *No arbitrary target.* A replay request accepts a scenario name from a fixed
  enumeration and a pace from a four-word vocabulary. There is no field for an
  event, an address, a URL, a schedule, or a filesystem path, and
  `extra="forbid"` means offering one is refused rather than ignored.
* *No arbitrary networking.* No scenario performs outbound networking.
* *No scientific override.* No request field and no environment variable can name
  a model, a threshold, a champion, a calibrator or a fusion strategy —
  `APISettings` and `DashboardSettings` refuse to declare such a field at import,
  and the deployment tests assert it of the merged Compose configuration.
* *No training endpoint, no upload, no path input.* The service has none.
* *No credentials.* Every request schema refuses a credential-shaped field by
  name, and the refusal reports no count — the number of prohibited fields a
  request carried is itself a fact about the credentials it carried.

### Rate limiting: deferred, and why

**There is none, and this is the deployment's main residual risk.** Caddy has no
rate limiter in its standard distribution; adding one means building a custom
Caddy image with a third-party module, which replaces a pinned official image
with one this project would have to maintain and patch itself. That is a worse
trade than the risk it addresses, given what is actually exposed.

What limits the exposure meanwhile:

* under the default routing policy the only public endpoint is the Streamlit
  console, and the expensive endpoints are not routed from the internet at all;
* replay is bounded to 4 concurrent runs *process-wide*, so a hostile visitor
  cannot start a fifth;
* the container memory ceilings mean a runaway is contained by Docker rather
  than by the host OOM killer choosing a victim.

A visitor can still reload the console repeatedly. Accept that for a
demonstration, or put the deployment behind a network you control. **Do not add
an authentication platform for this milestone** — a half-built auth system on a
demo is a larger surface than the one it closes. Revisit at final acceptance.

---

## 16. Rule availability in this deployment

A deployment document that claimed all nine rules were live would be wrong, so
here is the measured classification. Every number below comes from running the
serving-shaped detection path over all seven replay scenarios — 154 anchors —
under `configs/detection/rules-demo.yaml`.

### Fully live-serving — observed firing in the deployed container

| Rule | Fired | Scenarios |
|---|---|---|
| `PAD-BF-001` Concentrated brute force | 30 | brute force, account takeover, mixed |
| `PAD-BF-002` Success after failure burst | 2 | account takeover, mixed |
| `PAD-BOT-001` Bot-like authentication | 18 | bot activity, brute force, spraying |
| `PAD-PS-001` Password spraying | 14 | spraying, credential stuffing, mixed |

### Live-serving, but not demonstrated by the replay catalog

These evaluate normally on a live request and return clean negatives with
substantive reasons. They do not fire during a replay because no built-in
scenario carries the shape or the columns they need — not because the serving
path cannot compute them. They are exercised instead by the Phase 4 rule
evaluation over the full synthetic dataset.

| Rule | What is missing from the replay catalog |
|---|---|
| `PAD-DBF-001` Distributed brute force | No scenario has one account failing across many low-volume sources. Measured: 87 clean negatives, dominated by `BELOW_SOURCE_FANOUT_THRESHOLD` and `PER_SOURCE_VOLUME_TOO_HIGH`. |
| `PAD-GEO-001` Impossible travel | Scenario events carry a country code but no `coarse_latitude`/`coarse_longitude`, and the rule needs a derived distance. Measured: 114 clean negatives, then `GEO_STATUS_MISSING_CURRENT_LOCATION` ×29. The request schema *does* accept coordinates, so a caller-supplied window can fire this rule. |
| `PAD-MFA-001` MFA sequence anomaly | The minimum-history gate is never met: 154 anchors, all `MFA_INSUFFICIENT_ATTEMPT_HISTORY` or `MFA_INSUFFICIENT_OBSERVATIONS`. No scenario carries enough multi-factor history for a sequence to be called anomalous. |

### Unavailable — required baseline context is absent

| Rule | Measured |
|---|---|
| `PAD-CS-001` Credential stuffing | 154 anchors, **all** `ACCOUNT_ABSENT_FROM_BASELINE`. |
| `PAD-ATO-001` Account-takeover indicator | 40 anchors reached the gate, **all** `ACCOUNT_ABSENT_FROM_BASELINE`. The other 114 short-circuited on `ANCHOR_DID_NOT_SUCCEED` first. |

These two cannot fire on **any** live request through this deployment, replay or
otherwise. The serving path constructs its feature engine with no behavioural
baseline, so `user_in_baseline` and `source_in_baseline` are always false and
every `is_new_*_for_user` novelty flag is null.

**This has not been "fixed", and will not be by moving a threshold.** Milestone 4
audited adding a baseline to the serving bundle and *measured* that it would not
help: a baseline was fitted from the demonstration deployment's own TRAIN split,
loaded, handed to the feature engine, and both affected scenarios were run
through it. The flags were unchanged. The replay catalog's identities are UUIDv5
pseudonyms derived from the scenario itself, so no training population contains
them, whatever dataset a deployment trains on.

Making them fire would require either loosening a frozen Phase 4 rule — which
would publish a detection nobody validated — or drawing scenario identities from
a deployment's own training population, which would couple a content-addressed
catalog to one dataset. Separately, carrying a baseline in the bundle is a schema
change: it bumps `BUNDLE_SCHEMA_VERSION`, extends the fingerprint chain, gives
`deploy materialize` a feature-layer input, and adds a loader to the serving
path.

**v0.6.0 ships with this stated, not fixed.** It was carried as an open item into
the release milestone, which examined it and released on the reasoning above: the
two available remedies each change a frozen scientific contract, and a packaging
release must not do that. It is stated in the replay catalog's own published
`limitations` for the two affected scenarios, in `docs/live-replay.md` §3,
`docs/api.md` §12, `docs/docker.md` §14, the README's limitations section and
`docs/phase6-acceptance.md`, and it is pinned by
`test_the_baseline_dependent_rules_never_fire_on_a_live_request`.

---

## 17. Verification that has actually been run

The deployment topology below was brought up on a development machine with the
proxy published on `127.0.0.1:18080` — the same configuration, the same routing
policy, the same containers, on a port that needs no privilege — and every one of
these was checked:

| # | Verified |
|---|---|
| 1 | The deployment Compose configuration builds |
| 2 | The whole application is reachable through the proxy's single host port |
| 3 | `127.0.0.1:8000` refuses a connection |
| 4 | `127.0.0.1:8501` refuses a connection |
| 5 | The console's document and `/_stcore/health` are served through the proxy |
| 6 | The console reaches the API as `api:8000`; all six components ready |
| 7 | `/_stcore/stream` upgrades through the proxy: `HTTP/1.1 101` |
| 8 | `/health` returns the console's page, not an API document |
| 9 | `/docs` and `/openapi.json` are not published under the default policy |
| 10 | All seven scenarios replay to completion through the console's own client |
| 11 | Every step fused by the frozen `stacked` hybrid; no fallback anywhere |
| 12 | `api` and `dashboard` run as uid 10001 |
| 13 | Read-only root filesystems, `cap_drop: ALL`, `no-new-privileges`; the proxy holds `NET_BIND_SERVICE` and nothing else |
| 14 | An API restart preserves the champion field-for-field and clears the replay history |
| 15 | `down --volumes` leaves no container, volume, network, or listening port |

Graceful shutdown was checked separately with a replay in flight: `stop api`
completed in 0.4 s with a clean `Waiting for application shutdown → Application
shutdown complete → Finished server process`, not a ten-second kill. The proxy
container was confirmed to listen on `:80` and nothing else — Caddy's admin API
is off.

Automated equivalents live in `tests/unit/deployment/test_deployment_contract.py`
(92 tests, no Docker required) and `tests/integration/test_deployment_topology.py`
(45 tests, marked `slow`, needs a daemon).

---

## 18. Stopping and tearing down

```bash
DC="docker compose --env-file .env.deploy -f compose.yaml -f compose.deploy.yaml"

$DC stop                 # stop, keep everything
$DC down                 # remove containers and the network; keep the volumes
$DC down --volumes       # also discard the serving state and Caddy's certificates
```

To remove the deployment from a machine entirely: `down --volumes`, then
`docker image rm pad-demo-api:0.6.0 pad-demo-dashboard:0.6.0`, then delete the
checkout. Nothing of this project writes outside the checkout, the named volumes,
and Docker's own storage.

---

## 19. Troubleshooting

**The build is killed.** Not enough memory. See §2; build elsewhere and push, or
use a larger machine.

**`prepare` exits non-zero and the API never starts.** That is the design — the
API refuses to serve state a failed job may have half-written. Read
`docker compose logs prepare`: it reports which pipeline stage failed and why.

**502 from the proxy.** The console is not up yet or has stopped.
`docker compose ps`, then `logs dashboard`.

**The page loads but nothing updates.** The websocket is not getting through.
Check for an intermediate proxy or CDN between you and the machine that strips
`Upgrade` headers. Caddy forwards it correctly — this was verified — so an
intermediary is the usual cause.

**No certificate in hostname mode.** Three checks, in order: does the `A` record
resolve to this machine from *outside* it; is port 80 open in the cloud firewall
(the ACME challenge needs it, even for HTTPS); and does `docker compose logs
proxy` show an ACME error. Repeated failures can hit the provider's rate limit —
switch to `PAD_SITE_ADDRESS=:80`, fix the cause, then switch back.

**`/health` returns an HTML page.** Working as configured. The default routing
policy publishes only the console. See §6.

**Swagger's "Try it out" returns 404 or 405.** Working as configured. Scoring
endpoints are not published. See §6.

**The dashboard says the model is unavailable.** Read
`docker compose logs api` — startup logs each of the six components with a stable
reason code. The service reports a broken deployment rather than crash-looping,
which is what makes it diagnosable from a log.

**Disk filling up.** Not the application logs — they are capped at 120 MiB total.
Check `docker system df`; it is usually build cache. `docker builder prune`.

---

## Related documents

| Document | What it covers |
|---|---|
| [docker.md](docker.md) | The local containerized demonstration this builds on |
| [render-deployment.md](render-deployment.md) | The other prepared target: one container on a Render free web service |
| [api.md](api.md) | The serving contract, endpoints, limits, error codes |
| [dashboard.md](dashboard.md) | The analyst console and its views |
| [live-replay.md](live-replay.md) | The synthetic replay catalog and its limitations |
| [rule-catalog.md](rule-catalog.md) | All nine rules, their parameters and evidence |
| [detection-limitations.md](detection-limitations.md) | What the detection layer does not claim |
| [privacy-model.md](privacy-model.md) | Pseudonymization and what is never stored |
