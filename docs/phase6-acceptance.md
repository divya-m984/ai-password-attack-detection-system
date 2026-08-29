# Phase 6 acceptance report

**Prepared for v0.6.0 release.**

This document records what Phase 6 built, what was verified, how it was
verified, and what remains a stated limitation. Every figure below was measured
on this repository's own synthetic data or read from a running container. None
of it is evidence about real authentication systems.

This report is **hand-written and evidence-bearing**, unlike
[`phase5-acceptance.md`](phase5-acceptance.md), which is generated from
executable contracts and deliberately carries no run's fingerprints. Where a
claim here rests on a test, the test is named.

> **The v0.6.0 GitHub release does not exist yet.** This report is the input to
> that decision, not a record of it. Merging, tagging, and publishing are the
> repository owner's actions.

---

## Contents

1. [Scope](#1-scope)
2. [Serving](#2-serving)
3. [Dashboard](#3-dashboard)
4. [Replay](#4-replay)
5. [Docker](#5-docker)
6. [Deployment hardening](#6-deployment-hardening)
7. [The Render adapter](#7-the-render-adapter)
8. [Public deployment acceptance](#8-public-deployment-acceptance)
9. [Dashboard polish](#9-dashboard-polish)
10. [Scientific identity](#10-scientific-identity)
11. [Scenario acceptance](#11-scenario-acceptance)
12. [Security acceptance](#12-security-acceptance)
13. [Verification results](#13-verification-results)
14. [Known limitations](#14-known-limitations)
15. [Release decision](#15-release-decision)

---

## 1. Scope

Phase 6 turned a finished offline engine into a running application. It added no
detection capability and changed no scientific decision: every quantity the
application reports comes from the frozen Phase 3–5 code that owns it.

| Milestone | What it added |
|---|---|
| M1 | The FastAPI serving layer and the offline serving bundle that makes a frozen `stacked` selection deployable |
| M2 | The Streamlit SOC console, and the `POST /api/v1/explain` endpoint it needed |
| M3 | The synthetic live/replay demonstration |
| M4 | `Dockerfile`, `compose.yaml`, and the one-shot `prepare` job |
| M5A | The VPS deployment perimeter: `compose.deploy.yaml`, Caddy, the routing policy |
| M5B | The Render free-tier adapter: one container, a build-time bundle, a PID 1 supervisor |
| M5C | The dashboard polish pass: eleven grouped views, human-readable labels, friendly empty states |
| M6 | Release engineering: the version bump, the final documentation pass, and this report |

**What Phase 6 explicitly did not add**: persistence, an alert database, an event
store, authentication, rate limiting, a serving drift report, or any path for
real authentication traffic to enter the system.

---

## 2. Serving

`src/password_attack_detector/api/` is an **adapter**. It computes no feature,
re-derives no threshold, re-weights no rule, and contains no second scoring
implementation.

| Property | How it is held |
|---|---|
| Nothing loads at import | `create_app()` resolves the runtime in a lifespan |
| Startup fails closed | `build_runtime` never raises; an uninitialisable component is recorded with a stable reason code, readiness is false, detection is refused |
| No silent substitution | `FusionRuntime.__post_init__` refuses to construct unless the executing strategy equals the selected one and STACKED carries a loaded state |
| No writer of frozen state is reachable | `api/services.py` carries an import-time guard refusing eleven function *names* — trainer, freezer, selector, publisher, materializer |
| No scientific setting exists | `APISettings` carries `_assert_no_scientific_override_field`, pinned by `tests/unit/api/test_config.py` |
| The route surface is closed | `test_the_served_operations_are_exactly_the_declared_ones` pins all fifteen operations as an equality |

**Live inference is not a dataset split.** A scored request is
`ServingScope.LIVE`, never `MLSplit.TEST`. The frozen feature order,
preprocessor, adapter, calibrator and threshold are reused; the requirement to
*be* a scientific split is not. One binary-decision implementation
(`apply_frozen_binary_decision`) serves both paths.

---

## 3. Dashboard

`src/password_attack_detector/dashboard/` is structurally a client of the
serving API and nothing else.

| Property | Test |
|---|---|
| No dashboard module imports `ml`, `detection`, `features`, `deployment`, or `data` | `test_no_dashboard_module_imports_a_scientific_package` (AST, per module) |
| The only shared project module is `exceptions` | `test_the_package_imports_only_its_own_exception_type_from_the_project` |
| Exactly one module speaks HTTP | `test_the_api_client_is_the_only_module_that_imports_httpx` |
| No module spawns a process or reads the filesystem | `test_no_dashboard_module_spawns_a_process_or_reads_the_filesystem` |
| No rendered surface reads the backend address | `test_no_rendered_surface_reads_the_backend_address` (AST, added for this release) |
| No credential reaches session state | `test_a_credential_never_enters_a_session_and_so_never_reaches_a_page` |
| Nothing user-entered becomes markup | `test_a_hostile_value_is_escaped_by_every_html_helper` |

The wire contract is **re-declared** in `dashboard/contracts.py` rather than
imported from `api.schemas`: importing it would drag the whole detection stack in
transitively, making "the console cannot score" a convention instead of a fact.
The cost is drift, paid for by a test holding the two field sets side by side.

---

## 4. Replay

Seven reviewed, deterministic, credential-free scenarios, emitted one event at a
time into the **existing** serving path.

| Property | How it is held |
|---|---|
| There is no second detection path | The engine's detector is one call to `detect_single`, through the same request schema an HTTP body is validated by; a test reconstructs a replayed window, posts it to `/api/v1/detect`, and compares verdicts field by field |
| Nothing attacks anything | Import-time guards refuse a credential-shaped field name and any address outside the RFC 5737 documentation ranges; an AST test refuses a network client anywhere in the package |
| Published expectations are true | `expected_rule_ids` is asserted as an **equality** against the real frozen deployment |
| Pace is presentation only | The same scenario produces byte-identical verdicts at every pace, asserted against the real frozen stacked deployment |
| Storage is bounded and non-persistent | Four active runs, 24 retained, 256 records each, a 300-second ceiling; reaching a bound is a typed `429`, never an eviction of a run somebody is watching |
| Replay is optional | Reported as a non-required `/ready` component: a detection service does not become unready because a demonstration facility did not initialise |

---

## 5. Docker

Three services, one ordering: `prepare → api → dashboard`.

The image ships **no trained model**. A one-shot `prepare` job runs the real
pipeline — generate, build features, train, select, freeze, predict ×3, detect,
evaluate, materialize — into a named volume and exits. The API starts only after
that job reports success, mounts the volume read-only, and verifies what it
finds.

Measured for this release: the preparation pipeline completed in **113.7 s** in
the Render image's build stage and in **174.9 s** as the Compose `prepare`
service, and **both produced the champion scope key**
`d85a151c597b8b9eca0cd570b236aab91cb3cb13f021ee23576421fa1b9f5e90` — the same key
three earlier independent builds produced, including one after a full `down -v`
purge. Two independent preparations at the new version, agreeing exactly.

The non-Docker half of the container contract is 96 ordinary unit tests in
`tests/unit/deployment/test_container_contract.py`, which read the Dockerfile,
`compose.yaml`, `.dockerignore` and the demonstration configurations and assert
what they promise. A test added for this release,
`test_every_image_tag_is_the_package_version`, ties every Compose image tag to
`__version__` so a tag cannot drift from the release it names.

---

## 6. Deployment hardening

`compose.deploy.yaml` is an **additive overlay**, never a replacement:
`docker compose up --build` is unchanged, and a unit test asserts the base file
still publishes `127.0.0.1:8000` / `127.0.0.1:8501`.

- The application ports stop being published — not narrowed, **removed**.
  Compose *appends* sequences when it merges files, so an override cannot
  un-publish a port by restating a shorter list; `ports: !reset null` removes the
  key outright. A test reads the resolved `docker compose config` and asserts
  that only the proxy publishes anything, and only 80 and 443.
- The default routing policy publishes the console and nothing else. A second
  reviewed policy (`Caddyfile.api-docs`) adds the read-only surface. Scoring and
  replay control stay internal in **both**.
- `NET_BIND_SERVICE` is the one capability granted anywhere, and only because 80
  and 443 are below 1024. Everything else is `cap_drop: ALL` + `read_only` +
  `no-new-privileges`.
- **Rate limiting was deliberately deferred**, not forgotten: Caddy's standard
  distribution has none, and adding one means replacing a pinned official image
  with one this project would have to patch itself. `docs/deployment.md` §15
  states that residual risk rather than implying it is covered.

**No VPS deployment of this project exists.** The topology was brought up
locally with the proxy on `127.0.0.1:18080`, checked against fifteen points, and
torn down.

---

## 7. The Render adapter

A **second** target that does not replace the first. The one deviation — an
image that ships a trained model — is stated rather than glossed:

Render Free has no persistent disk for a `prepare` job to write into, and
preparing at container start would mean fitting three model families on 0.1 CPU
on every cold start. So preparation moved to a **discarded build stage** running
the same tracked script over the same tracked configurations; verification
**fails the build**; the result is `COPY --from=prepare`, root-owned; and the
bundle is verified three more times — after the prune step, by the entrypoint,
and by the API at startup.

**Why one service, not two.** Free services get no private network, so a separate
console service could only reach the API over the public internet — which would
force the detection API to be publicly exposed. One container keeps the API on
loopback.

---

## 8. Public-deployment acceptance

Verified against `pad-render:0.6.0`, rebuilt from this tree, run with
`--memory 512m --memory-swap 512m --read-only --cap-drop ALL
--security-opt no-new-privileges` and **no volume**.

**The live service was not touched.** Every measurement below is local. The
public demo at <https://pad-demo.onrender.com> is redeployed by the repository
owner from the Render dashboard.

| Check | Result |
|---|---|
| Image builds | yes; bundle verified twice during the build |
| `$PORT` remains arbitrary | served on 19081/19082/19083 in three runs; `10000` appears nowhere in the image, supervisor, or routing policy |
| Proxy executable present and runnable | `/usr/local/bin/caddy`, verified by the supervisor's own preflight before anything is forked |
| Caddy starts and owns the public port | `0.0.0.0:$PORT` is the only public listener |
| FastAPI loopback only | `127.0.0.1:8000` |
| Streamlit loopback only | `127.0.0.1:8501` |
| `GET /healthz` | `200`, `{"status","service","version"}` and nothing else; `version` is `0.6.0` |
| `GET /` (dashboard root) | `200`, Streamlit console shell |
| Websocket upgrade | `HTTP/1.1 101` locally **and** with an `Origin` of `https://pad-demo.onrender.com`, which is the case a local HTTP test would otherwise miss |
| Private API routes publicly inaccessible | all thirteen return the console shell, carrying no API payload marker |
| Runtime under 512 MiB | idle **191.9 MiB**; peak after a four-scenario pass **199.1 MiB = 38.9 %** |
| OOM kills | `memory.events`: `max 0 oom 0 oom_kill 0` — the limit was never approached |
| Serving bundle immutable | root-owned `0644`, not writable by uid 10001, append refused with `EACCES` |
| No persistent disk requirement | `Mounts` is empty; `render.yaml` declares no disk |
| Container survives the demonstration | `status=running restarts=0 oomkilled=false` |

Cold start to `/healthz` was **10–24 s on an unthrottled host**. The published
figure of **92.2 s** is the measurement taken at Render's 0.1 CPU and remains the
number to quote for the live service.

**Streamlit answers `200` with its own SPA shell for any unrecognised path**, so
a status code proves nothing about the routing policy. Every routing assertion —
in the test suite and in the measurements above — inspects the **response body**.

The full container suite, `tests/integration/test_render_container.py`,
**71 tests, all passing**, ran against this image.

### Public and private routes, as deployed

| Public | |
|---|---|
| `/` and below | the Streamlit console, including its `/_stcore/stream` websocket |
| `/healthz` | rewritten to the API's `/health`; status, service name, package version |

| Internal — `127.0.0.1` only, not proxied | |
|---|---|
| `/api/v1/detect`, `/api/v1/detect/batch` | scoring |
| `/api/v1/explain` | attribution |
| `/api/v1/demo/*` | replay control |
| `/api/v1/system/status`, `/api/v1/model/info`, `/api/v1/rules` | model and system information |
| `/ready` | names which component is unavailable and why — operator diagnostics |
| `/version` | build metadata |
| `/docs`, `/openapi.json` | Swagger / OpenAPI |

---

## 9. Dashboard polish

Confirmed present in the released console:

| Group | Views |
|---|---|
| **Primary** | Overview · Live Replay · Alerts · Analytics · Explainability · Drift Monitoring |
| **Advanced** | Detection Console · Authentication Events · Rule vs ML vs Hybrid · System & Model |
| **About** | About System |

| Property | Test |
|---|---|
| Every navigation label has a view, in order | `test_every_navigation_label_has_a_view` |
| The eleven labels are exactly these, in this order | `test_the_navigation_offers_the_eleven_declared_areas` |
| The groups partition the navigation | `test_the_navigation_groups_partition_the_navigation` |
| The sidebar group headings sit on the group boundaries | `test_the_sidebar_group_headings_sit_on_the_group_boundaries` |
| The product title appears exactly once | `test_the_overview_states_the_product_name_once` |
| No literal Markdown markers in raw-HTML prose | `test_the_overview_prose_renders_no_literal_markdown` and `test_the_overview_prose_carries_no_literal_markdown_markers` |
| No backend address on any rendered page | `test_no_rendered_surface_reads_the_backend_address` *(added for this release)* |
| The Live Replay call to action navigates to a real page | `test_every_navigation_target_a_view_can_request_is_a_real_page` *(added for this release)* |
| Friendly empty states, not invented totals | `test_the_session_pages_start_empty_and_say_so`, `test_the_overview_invents_no_global_event_total` |
| Every view survives the API being down, with no traceback | `test_every_view_survives_the_api_being_down`, `test_an_offline_page_shows_no_traceback` |

The Overview renders **no heading of its own**: the product name and its
one-line description are the global header's, and a second near-identical title
would say the same thing twice on one screen.

---

## 10. Scientific identity

**The v0.6.0 packaging change moved no scientific quantity.** Nothing was
retrained for the release. The Render image's build stage reran the same
deterministic preparation over the same tracked configurations, and the three
sealed fingerprints came back identical:

| Quantity | Value | Status |
|---|---|---|
| Champion scope key | `d85a151c597b8b9eca0cd570b236aab91cb3cb13f021ee23576421fa1b9f5e90` | unchanged |
| Serving-manifest fingerprint | `8341204696599ea08a3299bf249b214131b35f1073f46f5d25d41283b8799954` | unchanged |
| STACKED state fingerprint | `134f66ce2b837c4b74c9ab9ff2703125b9aedefb210d18afe923a0662f96272a` | unchanged |

Reported three times by the image's own verifier — after preparation, after the
prune step, and from inside the running container — and reported a fourth time
by the running service on `/api/v1/system/status`, which returns the same STACKED
fingerprint.

`test_the_baked_bundle_is_the_compose_prepared_bundle` additionally runs the
Compose `prepare` job from scratch in its own project and asserts the **whole
verified identity** of its bundle equals the image's — champion lock, model
content, calibration, threshold, feature contract, fusion selection, the stacked
state, and the manifest fingerprint that covers all of them. It compares the
verifier's structured output rather than diffing files; the byte-for-byte
`diff -r` over the three payload files was done by hand at 0.5.0 and is recorded
in `docs/render-deployment.md` §5.

**Why a rebuild is admissible as proof.** The deterministic preparation is a
function of tracked inputs — configurations, seeds, and the pinned `uv.lock`
environment — none of which the version bump touched. A rebuild that produced
the same three digests is therefore stronger evidence than an unchanged file
would have been: it demonstrates the identity is *reproducible*, not merely
*preserved*.

**The frozen selection is `stacked`, and the deployment executes `stacked`.**
`/api/v1/system/status` reports `fusion_strategy = frozen_fusion_strategy =
stacked` with `fusion_unavailable_reason = None`. There is no fallback: a
`FusionRuntime` whose executing strategy differed from the selected one is
unconstructible.

---

## 11. Scenario acceptance

Measured against `pad-render:0.6.0` under the free tier's limits, driving the
internal API against the real frozen serving bundle. Nothing was tuned for the
demonstration.

| | Normal activity | Brute force | Password spraying | Mixed attack |
|---|---|---|---|---|
| Final state | `completed` | `completed` | `completed` | `completed` |
| Timeline steps | 12 | 30 | 24 | 29 |
| Rule layer present | 12/12 | 30/30 | 24/24 | 29/29 |
| ML layer available | 12/12 | 30/30 | 24/24 | 29/29 |
| Hybrid layer available | 12/12 | 30/30 | 24/24 | 29/29 |
| Fusion strategies observed | `stacked` | `stacked` | `stacked` | `stacked` |
| Fallback strategy present | none | none | none | none |
| `fusion_unavailable_reason` | none | none | none | none |
| Rules fired | *(none)* | `PAD-BF-001`, `PAD-BOT-001` | `PAD-PS-001`, `PAD-BOT-001` | `PAD-BF-001`, `PAD-BF-002`, `PAD-PS-001` |
| Credential-shaped keys in run + timeline | none | none | none | none |

Per-rule anchor counts the service derived from each run's own timeline records:

| Scenario | Triggered rule counts | Severity distribution |
|---|---|---|
| `normal_activity` | `{}` | `{low: 12}` |
| `brute_force` | `{PAD-BF-001: 22, PAD-BOT-001: 10}` | `{low: 9, medium: 6, high: 5, critical: 10}` |
| `password_spraying` | `{PAD-PS-001: 9, PAD-BOT-001: 4}` | `{low: 15, medium: 5, high: 4}` |
| `mixed_attack` | `{PAD-BF-001: 5, PAD-BF-002: 1, PAD-PS-001: 1}` | `{low: 24, medium: 5}` |

**`normal_activity` firing nothing is the expected outcome**, not a failed
replay. `Severity` has no `none` member — the ladder starts at `low` — so a
window that fires nothing scores `0.0` and reports `low`.

The credential-key sweep walked **every key** of the run document and the whole
paged timeline, at every depth, matching against the project's own credential
vocabulary. Nothing matched.

All seven catalogued scenarios (the four above plus `credential_stuffing`,
`account_takeover`, `bot_activity`) are additionally driven by
`test_every_scenario_completes` and
`test_each_scenario_triggers_the_rules_its_catalogue_entry_claims` in the
container suite, which passed.

---

## 12. Security acceptance

Re-audited for the release. Each row names what holds the property, not merely
that it holds.

| Contract | How it is held |
|---|---|
| No password field accepted | Credential-shaped names refused before any other validation; `API013` with no value or count echoed |
| No credential field accepted | Schemas are `extra="forbid"`, so an undeclared key is refused outright; spellings outside the named list are refused as `API001` |
| No arbitrary external attack target | No host, URL, endpoint, or address field exists in the request surface; an AST test refuses a network client in the replay package |
| No arbitrary filesystem path input | `artifact_root`, `allowlist_path`, `feature_config_path` are all undeclared keys and refused; no URL parameter may contain `path` |
| No model override | `model_id`, `catalog_model_id`, `model_family`, `champion_scope_key` all refused; `/api/v1/model/info` is unchanged by an attempt |
| No threshold override | `decision_threshold`, `threshold`, `score_kind` refused |
| No fusion override | `fusion_strategy`, `strategy` refused |
| No artifact upload | Route inventory pinned as an equality — there is no upload operation |
| No training endpoint | Same pin, plus `test_no_operation_can_write_frozen_state` reading the verbs |
| No champion promotion endpoint | Same pin, plus the import-time guard refusing `freeze_champion`, `build_champion_lock`, `select_fusion_strategy` |
| No arbitrary code execution | Artifacts are numbers, not objects: no `pickle`, `dill`, or `joblib` import anywhere in the ML layer, and no `eval`, `exec`, or dynamic import in any artifact reader |
| No public Docker socket | No service mounts `/var/run/docker.sock`; asserted for both Compose files and measured on the running Render container |
| No host networking | `NetworkMode` is `bridge`; no `network_mode: host` in any Compose file |
| Non-root containers | uid 10001 in every image; measured `id -u` = 10001 inside the running container; `--shell /usr/sbin/nologin` |
| Read-only scientific state | Volume mounted `:ro` under Compose; root-owned `0644` in a read-only rootfs under Render, with an append attempt refused `EACCES` |
| Private control routes on Render | All thirteen internal routes return the console shell through the public port, with no API payload marker in any body |
| `/healthz` carries only safe health information | Body is exactly `{"status","service","version"}` — nothing about the host, the model, the artifacts, or which component is unhealthy |

**Three tests were added for this release**, each closing a contract that was
true but held up by review rather than by execution:

| Test | What it closes |
|---|---|
| `test_the_served_operations_are_exactly_the_declared_ones` | The route inventory was never pinned, so "there is no training or upload endpoint" was unenforced |
| `test_no_operation_can_write_frozen_state` | A second reading of the same set, by verb and by path vocabulary, so a route added *and* pasted into the pin is still caught |
| `test_no_rendered_surface_reads_the_backend_address` | A page printing the API URL would put an internal endpoint on a public page |

Two more were added for correctness rather than security:
`test_every_image_tag_is_the_package_version` and
`test_every_navigation_target_a_view_can_request_is_a_real_page`.

**No security theatre was added.** No dependency was introduced, no control was
added that claims a protection it does not provide, and the two absent controls —
authentication and rate limiting — are stated rather than partially implemented.

---

## 13. Verification results

Run against this tree at version 0.6.0.

| Check | Result |
|---|---|
| `uv lock --check` | pass — 89 packages resolved, lockfile up to date |
| `uv run ruff check .` | pass — all checks passed |
| `uv run ruff format --check .` | pass — 394 files already formatted |
| `uv run mypy src tests` | pass — 350 source files, no issues |
| `uv run pytest` | **6822 passed, 4 skipped**, 163 deselected (26 m 20 s) |
| Coverage | **94.43 %**, gate 90 % |
| `uv run pytest -m slow --no-cov` | **163 passed**, 0 skipped (12 m 57 s) |
| `uv run pre-commit run --all-files` | pass — all eight hooks |
| `bash scripts/verify.sh` | pass |
| `uv build` | sdist and wheel, both declaring `0.6.0` |

**The four skips are fixture-shape conditionals, not failures**: two in
`test_ml_profile_cli.py` ("this publication carries no category head"), one in
`test_ml_evaluate_cli.py` ("no category head was frozen for this
configuration"), and one in `test_ml_fusion_firewall.py` ("this fixture has no
novel_anomaly_holdout rows to perturb"). Each skips a test whose subject the
fixture does not contain.

**The slow suite skipped nothing**, which is the meaningful result: every
container-requiring test ran against a live Docker daemon rather than skipping
itself.

| Suite | Result |
|---|---|
| `tests/integration/test_render_container.py` | 71 passed |
| `tests/integration/test_docker_compose.py` | passed |
| `tests/integration/test_deployment_topology.py` | passed |

**Distributions inspected**, not merely built:

| Artifact | `Version:` metadata | Packaged `__version__` |
|---|---|---|
| `password_attack_detector-0.6.0.tar.gz` | `0.6.0` | `0.6.0` |
| `password_attack_detector-0.6.0-py3-none-any.whl` | `0.6.0` | `0.6.0` |

The wheel was additionally installed into a clean Python 3.12 virtual
environment: `import password_attack_detector` reports `0.6.0`, and the installed
`password-attack-detector version` console script prints `0.6.0`.

`docs/model-card.md` and `docs/phase5-acceptance.md` are **generated** by
`scripts/generate_governance_docs.py` and were regenerated for the version bump;
a test asserts the tracked copies are byte-identical to what the script produces.
`phase5-acceptance.md` therefore now reports `Package version | 0.6.0`: it is a
contract-derived report of the current package, not a frozen snapshot of the
0.5.0 release, and its requirement statuses are unchanged.

---

## 14. Known limitations

Stated in full in the README. Summarised here as the acceptance record.

**Data and evidence**

1. **The data is synthetic.** Every figure was measured on traffic this
   repository generated. It is evidence that the pipeline behaves as specified,
   not evidence about real authentication systems.
2. **This is not production authentication infrastructure.** It observes and
   scores event records; it does not authenticate anyone or sit in a login path.
3. **No figure the demonstration reports is a performance claim.** Its dataset is
   four hours of synthetic traffic sized so the pipeline finishes in about a
   minute; the evaluation windows are far too small for a per-scenario metric to
   mean anything.

**Deployment**

4. **Replay state is in-memory and ephemeral.** A restart clears every run.
5. **Render Free cold-starts** after inactivity — roughly 1–2 minutes. No uptime
   guarantee is offered.
6. **Free-tier resources are constrained** (512 MiB, a fraction of a CPU).
   Throttling changes how long a demonstration takes, not what the detector
   decides: every verdict at 0.1 CPU matched the same verdict at 0.5 CPU.
7. **No persistent alert database**, no event store, no serving drift report.
8. **No real password or credential is ever collected.** No request field could
   carry one.
9. **No public raw detection API.** Scoring, explanation, replay control,
   readiness and Swagger are internal.
10. **No authentication and no rate limiting anywhere**, documented as the main
    residual risk rather than implied to be covered.

**`PAD-CS-001` and `PAD-ATO-001` cannot fire on a live request**

Both gate on a fitted behavioural baseline the serving path does not load.
`api/services._feature_rows()` builds the feature engine with no baseline, so
`user_in_baseline` and `source_in_baseline` are always `False` and every
`is_new_*_for_user` flag is `None`.

This was carried as an open item into the release milestone and is **released as
a documented limitation, not fixed.** The reasoning:

- **Adding a baseline was audited and measured not to help.** A baseline fitted
  from the deployment's own TRAIN split was loaded and the two scenarios re-run;
  `user_in_baseline` stayed `False` and all five novelty flags stayed `None`. The
  replay catalog's identities are UUIDv5 pseudonyms derived from the scenario
  itself, so no training population contains them, whatever dataset a deployment
  trains on.
- The two remedies that remain each change a frozen scientific contract: either
  loosening a reviewed Phase 4 rule, which would publish a detection nobody
  validated, or drawing scenario identities from a deployment's own training
  population, which would couple a content-addressed catalog to one dataset.
  Carrying a baseline in the bundle is separately a schema change that bumps
  `BUNDLE_SCHEMA_VERSION` and extends the fingerprint chain.
- A packaging release must not do either.

**No threshold was moved and no baseline was synthesised.** The rules stay
enabled and return their honest reason code
(`ACCOUNT_ABSENT_FROM_BASELINE`); `test_the_baseline_dependent_rules_do_not_fire_and_are_not_pretended_to`
pins the behaviour, and the `credential_stuffing` scenario's catalog entry
records that it trips `PAD-PS-001` instead.

**Rule availability, measured over 154 replay anchors**

| Status | Rules |
|---|---|
| Fire on live requests | `PAD-BF-001`, `PAD-BF-002`, `PAD-BOT-001`, `PAD-PS-001` |
| Live-serving, exercised by no scenario (clean negatives) | `PAD-DBF-001`, `PAD-GEO-001`, `PAD-MFA-001` |
| Cannot fire — baseline | `PAD-CS-001`, `PAD-ATO-001` |

The middle category is the one most easily collapsed by mistake. Those three
evaluate normally and return clean negatives; nothing about the serving path
blocks them. Describing them as broken would be as wrong as calling all nine
live.

---

## 15. Release decision

**Prepared for v0.6.0 release** — *Deployable security analytics application*.

Everything Phase 6 set out to deliver is delivered and verified:

- the serving layer, the console, the replay, the containers, the deployment
  perimeter, the Render adapter, and a working public demonstration;
- the three frozen scientific identities are unchanged and, more strongly,
  reproducible from tracked inputs at the new version;
- all four required scenarios complete with the rule, ML and hybrid layers
  available on every step, fused by `stacked`, with no fallback and no credential
  data anywhere;
- the full verification suite passes, including the slow container suites;
- the security contracts hold, with three of them newly executable rather than
  merely reviewed;
- the limitations are documented honestly, including the one that was carried
  into this milestone unresolved and is released as stated rather than hidden.

**The release itself has not happened.** No branch was merged, no tag was
created, no GitHub release was published, and the live Render service was not
redeployed. Those are the repository owner's actions, and this report is the
input to them.

---

## Related documents

| Document | Contents |
|---|---|
| [demo.md](demo.md) | The presentation walkthrough |
| [render-deployment.md](render-deployment.md) | The live deployment, and what was measured |
| [deployment.md](deployment.md) | The VPS perimeter, and per-rule availability |
| [docker.md](docker.md) | The local containerized demonstration |
| [api.md](api.md) | The serving contract |
| [dashboard.md](dashboard.md) | The console |
| [live-replay.md](live-replay.md) | The scenario catalog and run lifecycle |
| [phase5-acceptance.md](phase5-acceptance.md) | Generated: the Phase 5 contract-derived acceptance report |
| [model-card.md](model-card.md) | Generated: purpose, scope, prohibited use, limitations |
