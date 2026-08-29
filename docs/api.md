# The detection API

The HTTP serving layer for the Password Attack Detector.

This document describes what the API is, what it deliberately is not, and every
constraint a client has to satisfy.

---

## 1. Architecture boundary

The API is an **adapter**. The detection system it serves — the feature
contract, the rule catalog, the champion model, its calibrator, its operating
point, and the hybrid fusion strategy — was built, validated, and frozen in
Phases 3–5, before this layer existed.

A request travels through parts that already existed:

```
HTTP request
    ↓  api/schemas.py            wire validation, credential refusal
canonical AuthEvent              data/schemas.py
    ↓  features/engine.py        point-in-time feature snapshots
feature rows
    ↓  detection/engine.py       every enabled rule, once per snapshot
    ↓  detection/scoring.py      correlation-aware event risk
rule verdict
    ↓  ml/dataset.py             assemble_serving_batch, scope live_serving
    ↓  ml/predictions.py         frozen preprocessor, adapter, calibrator, threshold
model verdict
    ↓  ml/fusion.py              the strategy validation selected before TEST
fused verdict
    ↓  api/schemas.py
HTTP response
```

`POST /api/v1/explain` branches off the same path after the serving batch is
assembled: the model's *own* frozen preprocessor produces the matrix, and
`ml/explain.py`'s `local_contributions` decomposes it. No step is duplicated and
no step is skipped.

The serving package contains **no** feature computation, **no** threshold
arithmetic, **no** rule weighting, and **no** second scoring implementation.
Where a quantity is needed, the frozen implementation that owns it is called;
where that implementation refuses, the refusal is reported rather than routed
around.

### The frozen-scientific-state rule

Artifact **location** may vary between deployments. Scientific **identity** may
not.

* `PAD_API_*` variables say where to find the champion, the allowlist, the
  feature configuration, and the rule configuration.
* No environment variable, request field, query parameter, or header can name a
  model, move a threshold, choose a fusion strategy, alter a rule threshold, or
  supply a feature value.
* `APISettings` declares no such field, and an import-time guard
  (`_assert_no_scientific_override_field`) fails the build if one is ever added.
  A unit test pins the same prohibition list.
* Every request schema sets `extra="forbid"`, so an attempt to smuggle one in
  under an invented key is a validation error rather than a silently ignored
  field.

The service **reads** frozen artifacts and writes none. There is no training,
promotion, freezing, upload, or reselection endpoint, and `services.py` carries
an import-time guard refusing the names of the functions that would perform one.
That guard names `materialize_serving_bundle`, `write_serving_bundle`,
`reconstruct_stacked_state` and `prepare_fusion_selection` alongside the training
entry points: the offline materializer of §2 is as unwelcome inside a request
as a trainer is.

A second guard refuses `MLSplit`, `predict_binary`, `assemble_inference_dataset`,
and `explain_predictions`. The last is not a writer — it is Phase 5's
population-level attribution entry point, and it takes a `scope: MLSplit`. Having
it reachable from a request handler is how a serving path acquires a split label
it has no business claiming.

### Live inference is not a dataset split

A row scored by this service is **live serving traffic**. It is not TRAIN, not
VALIDATION, not TEST, and not the novel-anomaly holdout, and the serving path
says so in the one field a reader trusts to be true:

```python
class ServingScope(StrEnum):
    LIVE = "live_serving"
```

`ServingScope` is a separate type from `MLSplit`, deliberately. Adding a
`SERVING` member to `MLSplit` would change an enum that feeds Phase 5's published
fingerprints — the split appears in dataset digests, in fold definitions, and in
the sealed documents those produce — so a serving convenience would have moved
frozen scientific identity. It also would have made "serving" answerable to
`FIT_ELIGIBLE_SPLITS`, which is a question about experiments.

What the serving path reuses instead is everything that is genuinely frozen:

* the reviewed feature allowlist and its resolved feature **order**;
* the fitted preprocessor;
* the model adapter and its calibrator;
* the frozen decision threshold;
* the frozen fusion strategy.

What it does *not* reuse is the requirement to be a split. `assemble_serving_batch`
builds a `ServingFrame`, which has no `split` attribute at all, and preprocessing
was narrowed to accept it: `TransformableFrame` (feature names, anchors, matrix)
is what a **fitted** preprocessor may be applied to, and `FeatureFrame` — that
plus a split — is what preprocessing may be **fitted** on. `transform` takes the
former, `fit_preprocessing` the latter. A live batch is therefore transformable
in the type system without claiming a split, and the narrowing is a typing change
only: no arithmetic moved and no Phase 5 fingerprint changed.

The binary decision itself is written **once**. `apply_frozen_binary_decision`
holds the whole of it — column selection, transform, score, threshold comparison
— and both `predict_binary` (the published-split path, unchanged) and
`predict_serving_binary` (the live path) call it. There is no second operating
point to drift, and a test asserts the two paths produce identical scores for
equivalent feature state.

Three import-time guards keep this true: `ServingScope` may share no value with
`MLSplit`; no serving type may carry a `split`-shaped field; and the serving
module's namespace may hold no `MLSplit`, `SplitRow`, `assemble_inference_dataset`
or `predict_binary`. A unit test additionally parses `api/services.py` with `ast`
and asserts no `MLSplit` name or split member is referenced anywhere in it.

---

## 2. The serving bundle

Phase 5 may select `stacked`, and a stacked hybrid is not a function of two
booleans: it needs its fitted meta-learner. Phase 5 seals that state's
*fingerprint* into the locked evaluation receipt's fusion selection but does not
publish the state itself, so the selected hybrid was unservable — the API
correctly refused to substitute a gate, which left the deployment honest and
useless.

The **serving bundle** closes that gap without touching anything frozen. It is
produced offline by an operator command, never by the service.

```bash
uv run password-attack-detector deploy materialize \
  --features   /abs/features.parquet \
  --labels     /abs/labels.parquet \
  --splits     /abs/splits.parquet \
  --allowlist  /abs/allowlist.yaml \
  --risk-assessments            /abs/risk.parquet \
  --validation-prediction       /abs/predictions/<validation-prediction-id> \
  --campaign-labels             /abs/campaigns.parquet \
  --validation-risk-assessments /abs/validation_risk.parquet \
  --config /abs/configs/ml/model-development.yaml \
  --feature-config /abs/features.yaml \
  --output-root /abs/artifacts

uv run password-attack-detector deploy inspect --output-root /abs/artifacts
```

### Artifact contract

```
<bundle root>/serving/<champion scope key>/
├── serving_bundle.json        the sealed manifest
├── fusion_selection.json      the frozen FusionSelection, as Phase 5 sealed it
└── fusion_stacked_state.json  the fitted StackedFusionState — stacked only
```

* **`serving_bundle.json`** is a `SealedModel`: its `manifest_fingerprint` is a
  field, recomputed on every construction and every deserialization, so a loaded
  manifest that does not re-derive its own digest is refused before it is read.
  It carries `bundle_schema_version` (`1.0.0`) and `files`, a sorted tuple of
  `(name, sha256)` pairs covering every other payload in the directory.
* **Lineage carried** — champion scope key, champion lock fingerprint, freeze
  record id, validation selection id, catalog model id, model family, content
  model id, model content fingerprint; preprocessor fingerprint, calibration
  method and state fingerprint, binary threshold fingerprint; feature catalog
  fingerprint, allowlist fingerprint, eligible feature list fingerprint, ML
  configuration fingerprint, serializer id and version, dependency contract
  fingerprint; selected fusion strategy, fusion selection fingerprint, fusion
  configuration fingerprint, rule configuration fingerprint, validation evidence
  fingerprint, stacked state fingerprint, out-of-fold fold-definition and
  evidence fingerprints, base-model recipe fingerprint; and the evaluation
  record id and fingerprint the selection came from.
* **Never carried** — no metric, no `f1`, no TEST figure, no
  `decision_threshold` value, no `fallback_strategy`, and no `override` of any
  kind. An import-time guard holds the prohibition list and a unit test pins it.
  The bundle *names* the frozen threshold artifact by fingerprint; it does not
  restate the number, because a restated number is a number that can disagree.
* **`selected_fusion_strategy` is never `None`.** A bundle exists because a
  hybrid was selected. "Nothing qualified" is an absence of a bundle, not a
  bundle saying nothing.
* **Deterministic serialization** — canonical JSON throughout: sorted keys,
  ASCII, no whitespace. Publication is transactional: payloads are staged in a
  sibling directory, read back and verified, promoted by a single atomic rename,
  and the manifest is written last. Re-materializing an unchanged lineage writes
  nothing and reports the bundle already published; a *different* manifest for an
  existing scope is a refusal, never an overwrite.

### How the stacked state is materialized

Nothing is re-decided. The materializer reproduces the pre-TEST decision from the
same inputs Phase 5 used, and then proves it reproduced it:

1. **Find the frozen evidence.** The locked TEST evaluation receipt
   (`evaluations/*/test_evaluation.json`) for this champion lock is loaded as a
   sealed record, so it verifies its own digest. An unreadable receipt stops the
   materialization — it may be the one that selected the hybrid.
2. **Walk the digest chain to the selection.** The receipt's
   `report_fingerprints["system_comparison.json"]` is compared against the
   SHA-256 of the report actually on disk; the report's `fusion` payload is then
   loaded as a `FusionSelection`, which verifies its own seal. A report edited
   after publication fails here.
3. **Check the selection describes this deployment** — that it names this
   champion and the rule configuration fingerprint being supplied.
4. **Reconstruct.** Only now is anything fitted, and it is fitted by
   `prepare_fusion_selection` — the *same* entry point `ml evaluate` used, which
   structurally refuses a TEST parameter. Out-of-fold stacking over TRAIN rows,
   the frozen campaign folds, the frozen validation predictions, the frozen
   fusion configuration. No new threshold, no reselection, no new fold scheme.
5. **Compare fingerprints, then publish.** The reconstructed
   `StackedFusionState` recomputes its own semantic fingerprint, which must equal
   the one Phase 5 sealed. Only on equality is the bundle written.

### The exact Phase-5 fingerprint checked

`FusionSelection.stacked_state_fingerprint`, read from the `fusion` payload of
the published `system_comparison.json`, whose SHA-256 is itself pinned by
`TestEvaluationRecord.report_fingerprints["system_comparison.json"]` in the
sealed locked-evaluation receipt. The reconstructed side is
`StackedFusionState.recomputed_fingerprint()` — recomputed from the state's own
semantic content, not copied from the frozen value.

Because both digests are semantic, **one changed upstream input is enough to
break the equality**: a different rule configuration is refused at step 3
(`rule_configuration_mismatch`), a report edited after publication at step 2,
and a changed fold count, feature set, model, or fusion configuration at step 5
(`stacked_state_fingerprint_mismatch`). Publication is refused; nothing is
written; the command exits `2` and names which digest moved.

### No TEST label reaches it

Asserted structurally *and* checked empirically:

* `deploy materialize` has no `--prediction`, `--prediction-id`, or
  `--holdout-prediction` option — only `--validation-prediction`, which is
  validated to be a VALIDATION-scope publication.
* `reconstruct_stacked_state` and `materialize_serving_bundle` take no
  `test`-shaped parameter, and an import-time guard asserts it.
* The materializer's namespace holds no TEST outcome reader, no
  `select_fusion_strategy`, no `select_binary_threshold`, and no
  `freeze_champion`.
* An integration test inverts the `malicious` label on **every TEST row** and
  re-materializes from scratch. The stacked state and the manifest come out
  byte-identical.

### Gates

`or_gate` and `and_gate` need no fitted artifact — they are functions of the two
booleans. The materializer still publishes a bundle for them, and that bundle
still binds the frozen selection: `selected_fusion_strategy` plus the fusion
selection fingerprint, with no stacked state file. At startup a gate bundle is
optional (the receipt-verified selection is enough to execute a boolean gate),
but a bundle that is *present and unverifiable* is a refusal, and a bundle whose
strategy disagrees with the receipt is `fusion_selection_conflict`.

---

## 3. Startup

Nothing loads at import. `create_app()` builds the FastAPI object and registers
routes; the runtime is assembled once in the lifespan, before the first request.

Startup, in order:

1. **Serving configuration** — `APISettings`, from `PAD_API_*`, `.env`, or
   explicit arguments.
2. **Feature contract** — the Phase 3 feature configuration is loaded and its
   executable catalog built. Everything else is defined against it, so it goes
   first.
3. **Rule layer** — the Phase 4 configuration is loaded and every enabled rule
   is prepared exactly once.
4. **Model artifacts** — the artifact root is inspected for a frozen champion
   lock. This is a structural check only.
5. **ML champion** — `FrozenChampion.load` runs the whole Phase 5 verification
   chain (freeze receipt, validation selection, training run, artifact bytes,
   manifest digest, preprocessor, calibrator, operating point, dependency
   ranges), and the resulting model is then checked against the feature contract
   this build computes.
6. **Fusion selection** — the locked TEST evaluation receipts under the artifact
   root are read to learn which hybrid strategy validation selected for *this*
   champion lock. Each receipt verifies its own seal.
7. **Serving bundle** — the bundle for this champion scope is loaded and
   verified: the manifest's own digest, every payload digest, the frozen
   selection's seal, the stacked state's seal, and that the bundle's lineage is
   this champion's. A stacked selection requires it; a gate does not.

Step 7 **loads and verifies only**. Nothing at startup fits a stacker, refits a
preprocessor, reselects a strategy, chooses a threshold, or writes a file. The
proof is not a promise:

* `services.py` carries an import-time guard whose forbidden-name list includes
  `fit_stacked_fusion`-side entry points — `materialize_serving_bundle`,
  `reconstruct_stacked_state`, `prepare_fusion_selection`, `write_serving_bundle`
  — so the module cannot even reference the code that would fit or publish.
* An integration test monkeypatches `fit_stacked_fusion`, `build_stacked_state`
  and `prepare_fusion_selection` to raise on call, then starts the app against a
  materialized stacked bundle. Startup completes and reports the stacked hybrid
  ready.
* A second test parses the module and asserts no import path from it reaches a
  fitter or a publisher.

`build_runtime` **never raises**. A stage that fails records its component as
unavailable with a stable reason code; readiness becomes false and every
detection route refuses with `API010`. The process still answers `/health`,
`/version` and `/ready`, so a misconfigured deployment is diagnosable from a
container log rather than a crash loop.

No component ever falls back to a different model, a different threshold, or a
different strategy.

### Runtime components

| Component | Required | Meaning |
|---|---|---|
| `feature_contract` | yes | The Phase 3 configuration loaded and its catalog built |
| `rule_engine` | yes | Every enabled rule prepared against that catalog |
| `model_artifacts` | when a champion is required | A frozen champion lock is present under the artifact root |
| `ml_champion` | when a champion is required | The lock verified end to end and bound to the feature contract |
| `fusion` | **when one was selected** | The frozen strategy can actually execute |
| `replay` | only when configured required | The optional synthetic demonstration subsystem is assembled |

### Readiness semantics for the hybrid

`fusion` is optional **only when there is genuinely no selected hybrid.** The
distinction is between an honest scientific negative and a runtime that cannot do
what its own lineage says it does.

| Frozen selection | Serving state | `fusion` | `required` | `/ready` |
|---|---|---|---|---|
| none (nothing qualified) | — | `disabled`, `no_fusion_selection` | `false` | `200` |
| `or_gate` / `and_gate` | no bundle published | `ready` | `true` | `200` |
| `or_gate` / `and_gate` | bundle present and verified | `ready` | `true` | `200` |
| `or_gate` / `and_gate` | bundle present, unverifiable | `unavailable` | `true` | `503` |
| `stacked` | bundle verified, state loaded | `ready` | `true` | `200` |
| `stacked` | no bundle / state missing | `unavailable` | `true` | `503` |
| `stacked` | fingerprint or lineage mismatch | `unavailable` | `true` | `503` |
| two receipts, different strategies | — | `unavailable` | `true` | `503` |
| a receipt does not verify | — | `unavailable` | `true` | `503` |

Read the first row and the last row together. When Phase 5 froze **no** eligible
hybrid, that is not a fault: the hybrid is not required, readiness is `200`, and
`/api/v1/system/status` says the hybrid is unavailable *by scientific outcome*
(`hybrid_required: false`, `fusion_unavailable_reason: "no_fusion_selection"`).
When Phase 5 **did** freeze one, the hybrid is a required runtime component, and a
deployment that cannot execute it is `503` — a complete hybrid-capable runtime is
never reported ready when its selected hybrid cannot run.

There is no fallback. This is structural rather than conventional: `FusionRuntime`
validates in `__post_init__` that the executing strategy equals the selected one,
that a stacked strategy has a loaded state, that an executable hybrid names no
unavailability reason, that an unavailable one does, and that any selection makes
the component required. A `stacked` selection with a missing state is therefore
*unconstructible* as a running gate — the substitution cannot be written, not
merely must not be.

---

## 4. Endpoints

| Method | Path | Tag | Purpose |
|---|---|---|---|
| `GET` | `/health` | Health | Process liveness |
| `GET` | `/ready` | Health | Whether detection can be served |
| `GET` | `/version` | Health | Package and contract versions |
| `POST` | `/api/v1/detect` | Detection | Score one anchor within a window |
| `POST` | `/api/v1/detect/batch` | Detection | Score many anchors within one window |
| `POST` | `/api/v1/explain` | Detection | Attribute one anchor's model decision |
| `GET` | `/api/v1/system/status` | System | Which layers this deployment runs |
| `GET` | `/api/v1/model/info` | System | Frozen champion identity and operating point |
| `GET` | `/api/v1/rules` | System | The public rule catalog |

### `GET /health`

Process liveness only. Performs no artifact check, touches no model, and reads
no file, so a probe hitting it every second costs one JSON serialisation.

```json
{ "status": "ok", "service": "password-attack-detector", "version": "0.6.0" }
```

### `GET /ready`

Aggregate readiness across every component. `200` when ready, `503` when not.
Each component carries its own state and, when it is not ready, a stable
lower-case reason code.

```json
{
  "status": "not_ready",
  "service": "password-attack-detector",
  "version": "0.6.0",
  "components": [
    { "component": "feature_contract", "state": "ready",       "reason": null,                  "required": true },
    { "component": "rule_engine",      "state": "ready",       "reason": null,                  "required": true },
    { "component": "model_artifacts",  "state": "ready",       "reason": null,                  "required": true },
    { "component": "ml_champion",      "state": "unavailable", "reason": "champion_verification_failed", "required": true },
    { "component": "fusion",           "state": "disabled",    "reason": "ml_champion_unavailable",      "required": false }
  ]
}
```

Readiness reads state resolved at startup. It does not re-verify artifacts per
request: a readiness probe that reloaded a model would be a denial of service
with a green tick on it.

**Reason codes** (stable):

| Code | Meaning |
|---|---|
| `feature_config_unreadable` | The feature configuration could not be loaded |
| `feature_catalog_unavailable` | The catalog could not be built from it |
| `feature_contract_unavailable` | A component needed the catalog and there was none |
| `detection_config_unreadable` | The rule configuration could not be loaded |
| `rule_engine_preparation_failed` | A rule could not be prepared |
| `artifact_root_not_configured` | No artifact root was configured |
| `artifact_root_not_found` | The configured artifact root does not exist |
| `no_champion_frozen` | The root holds no champion lock |
| `allowlist_not_configured` | No reviewed feature allowlist was configured |
| `allowlist_unusable` | The allowlist could not be loaded or resolved |
| `ml_config_unreadable` | The ML configuration could not be loaded |
| `champion_verification_failed` | The frozen champion did not verify |
| `feature_contract_mismatch` | This build's feature contract is not the champion's |
| `ml_champion_disabled` | The model layer is switched off by configuration |
| `ml_champion_unavailable` | A layer that needed the champion could not have it |
| `no_fusion_selection` | No locked evaluation selected a hybrid strategy |
| `ambiguous_fusion_selection` | Two receipts for one champion name different strategies |
| `evaluation_receipt_unreadable` | An evaluation receipt did not verify |
| `serving_bundle_not_published` | A `stacked` hybrid was selected and no bundle was materialized |
| `serving_bundle_unverifiable` | A bundle is present but did not verify against itself |
| `serving_bundle_lineage_mismatch` | The bundle describes a different champion lineage |
| `fusion_selection_conflict` | The bundle and the receipt name different strategies |
| `fusion_refused_evidence` | The frozen fusion refused a row's evidence |

`serving_bundle_unverifiable` covers a failed manifest seal, a payload whose
digest does not match, a tampered stacked state, and a stacked bundle whose state
file is absent — one code, because they are one finding: the published state is
not the state that was verified at publication.

No reason code carries a path, a message, or a stack trace.

### `GET /version`

Deterministic contract versions. Two machines running this build answer
identically; nothing host-specific appears.

```json
{
  "service": "password-attack-detector",
  "package_version": "0.6.0",
  "api_schema_version": "1.0.0",
  "event_schema_version": "1.0.0",
  "feature_schema_version": "1.0.0",
  "detection_schema_version": "1.0.0",
  "scoring_version": "1.0.0",
  "ml_schema_version": "1.0.0",
  "fusion_schema_version": "1.0.0"
}
```

### `GET /api/v1/system/status`

Which detection layers are actually loaded, the champion's model *family*, the
enabled and registered rule counts, and this deployment's `max_batch_events`.

The hybrid arm reports the frozen decision and the running one separately, so a
reader can tell "nothing qualified" from "the selected hybrid cannot run":

```json
{
  "hybrid_detection_enabled": true,
  "hybrid_required": true,
  "frozen_fusion_strategy": "stacked",
  "fusion_strategy": "stacked",
  "fusion_unavailable_reason": null,
  "stacked_state_fingerprint": "d586539…"
}
```

* `frozen_fusion_strategy` — what Phase 5 selected, `null` when nothing qualified.
* `hybrid_required` — whether the hybrid is a required runtime component.
* `fusion_strategy` — what is actually executing, `null` when nothing is.
* `stacked_state_fingerprint` — the fitted meta-learner's semantic digest, present
  only when a stacked hybrid is running. It is an identity, not a parameter: the
  coefficients themselves are never published.

The response model validates these against each other. An enabled hybrid must
name a strategy, that strategy must **be** the frozen one, and it must report no
unavailability reason; a disabled hybrid must execute nothing; the hybrid is
required exactly when one was frozen (or when the lineage is ambiguous, which
requires a hybrid it cannot name); and only a running stacked hybrid may carry a
state fingerprint. A document describing a substituted gate is not serialisable.

The same document also reports the optional **synthetic replay demonstration**:

```json
{
  "replay_enabled": true,
  "replay_available": true,
  "replay_required": false,
  "replay_unavailable_reason": null,
  "replay_scenario_count": 7,
  "max_active_replay_runs": 4
}
```

Replay is a demonstration facility, not a detection layer. `replay_required` is
`false` by default, so a deployment whose replay subsystem could not initialise
stays **ready** and refuses only the replay endpoints. `/health` is untouched:
liveness stays cheap, and whether an optional subsystem came up is what a status
document is for. Details in [live-replay.md](live-replay.md) §10.

### `GET /api/v1/model/info`

The champion's family, catalog identifier, content-derived model identifier,
task, score kind, calibration state, decision threshold, and freeze lineage
(freeze record, training run, validation selection, scope key).

The threshold is published for transparency. It is a frozen artifact and cannot
be changed through this API.

Never present: coefficients, tree arrays, feature values, feature names,
hyperparameters, artifact paths, entity pseudonyms.

### `GET /api/v1/rules`

Every registered rule with its identifier, version, display name, description,
family, attack category, default severity, deprecation flag, and whether this
deployment enabled it. No thresholds and no feature names.

### `POST /api/v1/detect` and `POST /api/v1/detect/batch`

See §5 and §6.

### `POST /api/v1/explain`

Takes the **same window schema** `/api/v1/detect` takes, and answers a narrower
question about it: which transformed columns moved the frozen model's decision
for the selected anchor, and by how much.

Separate from `/api/v1/detect` rather than folded into its response, for two
reasons. A detection verdict is what an alert is raised on and is wanted on every
call; an attribution is what an analyst opens afterwards for one row, and
attaching it to every verdict would put a per-column table behind every alert.
And the two can legitimately disagree about availability: a champion family with
no exact decomposition still produces a perfectly good verdict, so the
explanation reports itself unavailable while detection carries on.

```json
{
  "api_schema_version": "1.0.0",
  "anchor_event_id": "6b1d7a4e-9c02-5f31-88ad-4e7f0b3c9d15",
  "anchor_event_time": "2026-03-04T12:04:50Z",
  "available": true,
  "unavailable_reason": null,
  "method": "linear_logit_contribution",
  "model_family": "logistic_regression",
  "score_kind": "calibrated_probability",
  "decision_value": -1.482391,
  "baseline_value": -2.104772,
  "contributions": [
    { "transformed_feature": "failed_attempts_5m", "contribution": 0.914233 },
    { "transformed_feature": "seconds_since_prior_failure", "contribution": -0.291845 }
  ],
  "transformed_feature_count": 61,
  "omitted_contribution_count": 59,
  "reconstruction_residual": 0.0
}
```

#### Why this needs no scope

The decomposition is Phase 5's own
`ml.explain.local_contributions`, called unchanged. It takes a verified model and
a transformed matrix and **no split argument** — which is precisely what makes
attributing a live row possible without claiming the row belongs to an
experimental population.

Phase 5's population-level entry point, `explain_predictions`, *does* take a
`scope: MLSplit` and refuses TEST. It is neither called nor importable here: the
import-time guard in `api/services.py` refuses the name alongside `MLSplit`
itself.

#### What it decomposes, and what it does not

`decision_value` is the **decision function's own quantity** — the logit for a
linear head, the mean leaf score for the forest, the step for the threshold
baseline. It is not the calibrated probability, and no contribution sums toward
one. The probability and the frozen operating point are reported by
`/api/v1/detect`; this endpoint reports **no threshold and no verdict**, because
the contributions sum to a logit and the threshold sits on a probability.

`reconstruction_residual` is `decision_value − (baseline_value + sum of the FULL
decomposition)`. It is checked against Phase 5's declared tolerance *before* the
response is built, over every column — never over the truncated list below, which
would check nothing. A decomposition that does not add up reports
`explanation_not_reconstructible` rather than being published with a caveat.

The contributions are **ranked by magnitude and bounded** by
`min(explain.top_k_features, 25)`. `omitted_contribution_count` says how many
were left out, so the reported set is never mistaken for the whole decomposition,
and `transformed_feature_count` says how many the decomposition covered.

There is **no `transformed_value` field**, and there will not be one. Phase 5
gates value disclosure behind a reviewed configuration flag because a transformed
value can be a country code; a live wire surface is not where that flag gets
turned on.

#### Refusals

| Condition | Code |
|---|---|
| The request selects other than exactly one anchor | `API011` |
| The runtime is not ready | `API010` |
| No frozen champion is loaded | `API008` |
| Window, ordering, identity, credential, and size rules | as `/api/v1/detect` |

An explanation that cannot be produced for a *scientific* reason is not a
refusal: it is a `200` with `available: false` and one of
`explanation_disabled`, `explanation_method_unavailable`, or
`explanation_not_reconstructible`. The detection verdict is unaffected and
remains available.

Nothing here fits, calibrates, re-thresholds, or writes. An integration test
hashes every artifact byte under the deployment before and after a run of
explanations and asserts they are identical.

---

### `GET /api/v1/demo/scenarios`, `POST /api/v1/demo/runs`, and the run endpoints

The synthetic live/replay demonstration, under `/api/v1/demo`, tagged **Demo**.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/demo/scenarios` | The reviewed, built-in scenario catalog |
| `POST /api/v1/demo/runs` | Start one scenario at one pace |
| `GET /api/v1/demo/runs` | The runs this *process* retains, newest first |
| `GET /api/v1/demo/runs/{run_id}` | One run's state and summary |
| `GET /api/v1/demo/runs/{run_id}/timeline` | One bounded page after a cursor |
| `POST /api/v1/demo/runs/{run_id}/stop` | Stop that run, idempotently |

A replay emits a fixed synthetic scenario one event at a time into **this
service's own `detect_single`** — the same function `POST /api/v1/detect` calls,
through the same request schema, with the window being every event emitted so
far and the anchor being the newest. There is no second detection path, and a
test reconstructs a replayed step's window, posts it to `/api/v1/detect`, and
compares the two verdicts field by field.

The create-run body is **two fields**:

```json
{ "scenario_id": "brute_force", "pace": "normal" }
```

Both are closed vocabularies, and `extra="forbid"` means anything else is a
refusal. There is no field for an event list, an address, a URL, a filesystem
path, a schedule, a model, a threshold, a fusion strategy, or a credential.

Runs are **process-local, bounded, and not retained across a restart**. Nothing
here is persistence, and no document presents it as history.

Full contract, scenario catalog, determinism guarantees, state machine, bounds
and limitations: **[live-replay.md](live-replay.md)**.

---

## 5. Request constraints

### Why a window rather than an event

Almost every signal the rules and the model read is a **windowed or sequence
quantity over the anchor's own strictly-prior history**: failures for this user
in the last five minutes, distinct accounts from this source, seconds since the
previous attempt, dispersion of inter-arrival times.

A single stateless event produces a snapshot whose history is empty — and an
empty history is not "unknown", it is *wrong*. It would score the tenth failure
of a brute-force burst exactly like the first login of the day.

So the unit of work is a window: an ordered batch of events, plus a declaration
of which of them the caller wants a verdict for. The caller supplies the
history. **This service fabricates none.**

### The request

```json
{
  "api_schema_version": "1.0.0",
  "anchor_selection": "last",
  "anchor_event_ids": [],
  "events": [ /* AuthEventRequest, ... */ ]
}
```

| Field | Constraint |
|---|---|
| `api_schema_version` | Must be `"1.0.0"` |
| `events` | 1 … `max_batch_events`; non-decreasing `event_time`; no repeated `event_id` |
| `anchor_selection` | `last` \| `all` \| `explicit` — defaults to `last` on `/detect` and `all` on `/detect/batch` |
| `anchor_event_ids` | Required when `anchor_selection` is `explicit`, forbidden otherwise; every id must appear in `events` |

`/api/v1/detect` requires the selection to resolve to exactly one anchor.

### An event

Field names, enumerations, and semantics are the canonical ones from
[`docs/data-contract.md`](data-contract.md).

| Field | Constraint |
|---|---|
| `event_id` | UUID, unique within the window |
| `event_time` | Timezone-aware timestamp; normalised to UTC |
| `user_id` | `u:<32 hex>` |
| `source_id` | `s:<32 hex>` — supply this **or** `source_ip` |
| `source_ip` | Valid IPv4 or IPv6 — pseudonymized on arrival, never returned |
| `device_id` | `d:<32 hex>` |
| `session_id` | `sess:<32 hex>` |
| `application_id` | 1–128 characters |
| `authentication_method` | `AuthMethod` |
| `authentication_outcome` | `AuthOutcome` |
| `failure_reason` | `FailureReason` or null; consistency enforced by the canonical schema |
| `mfa_outcome` | `MFAOutcome` or null |
| `country_code` | ISO 3166-1 alpha-2 |
| `region_code` | 1–16 characters |
| `coarse_latitude` / `coarse_longitude` | Finite, in `[-90, 90]` / `[-180, 180]` |
| `user_agent_family` / `operating_system_family` | 1–128 characters |
| `client_type` | `ClientType` or null |
| `response_time_ms` | Integer in `[0, 30000]` |

The pseudonym domain prefixes are checked, which is stricter than the canonical
schema's generic pattern: a historical dataset has to be accepted as it was
recorded, a live request does not.

Extra fields are forbidden on the event and on the envelope.

### Bounds

| Bound | Default | Configurable to | Enforced by |
|---|---|---|---|
| Body size | 1 MiB | 1 KiB – 16 MiB | Serving middleware, before the body is parsed |
| Events per window (deployment) | 500 | 1 – 5 000 | The detection service, `413 API005` |
| Events per window (absolute) | 5 000 | — | The request schema, `413 API005` |

---

## 6. Response semantics

```json
{
  "api_schema_version": "1.0.0",
  "window": {
    "event_count": 30,
    "anchor_count": 1,
    "feature_schema_version": "1.0.0",
    "detection_schema_version": "1.0.0",
    "enabled_rule_count": 9,
    "evaluated_snapshot_count": 30
  },
  "anchor": {
    "anchor_event_id": "…",
    "anchor_event_time": "2026-03-04T12:04:00Z",
    "rule": {
      "flagged": true,
      "risk_score": 95.47,
      "severity": "critical",
      "primary_attack_category": "brute_force",
      "contributing_categories": ["bot_activity", "brute_force"],
      "fired_rule_ids": ["PAD-BF-001", "PAD-BOT-001"],
      "fired_rule_count": 2,
      "insufficient_data_count": 2,
      "scoring_version": "1.0.0",
      "evidence": [ /* sanitized behavioral evidence */ ]
    },
    "ml": {
      "available": true,
      "unavailable_reason": null,
      "flagged": true,
      "score_kind": "calibrated_probability",
      "decision_score": 0.983,
      "probability": 0.826,
      "decision_threshold": 0.126
    },
    "hybrid": {
      "available": true,
      "unavailable_reason": null,
      "flagged": true,
      "strategy": "stacked"
    },
    "severity": "critical"
  }
}
```

`/api/v1/detect/batch` returns `anchors` — an array of the same objects — in
canonical `(anchor_event_time, anchor_event_id)` order, so the response does not
depend on the order the anchors were requested in.

### The three layers are never blended

* `rule.risk_score` is a bounded **ordinal severity magnitude on 0–100**. It is
  not a probability and must never be described as one.
* `ml.probability` is a calibrated likelihood, and is present **only** when the
  frozen operating point was selected against a calibrated probability. An
  uncalibrated decision score is never relabelled a probability; its absence is
  a null rather than a copy of `decision_score`.
* `hybrid.flagged` is a boolean produced by the frozen strategy, which combines
  the two **decisions** — never the two numbers.

There is no combined, blended, fused, or overall score field anywhere in the
response, and a test asserts there never will be. Adding, averaging, or weighting
an ordinal magnitude against a probability produces a number whose units do not
exist, and it does so silently.

`severity` at the top level is the Phase 4 ordinal severity for the anchor. It is
derived from the rule layer alone: a probability is not a severity.

### Unavailable layers

An unavailable layer reports `available: false`, a stable
`unavailable_reason`, and `null` for every verdict field. A layer never reports
a silent negative in place of an absent verdict: those are different findings,
and conflating them would understate detection gaps.

### Determinism

Two identical requests produce byte-identical responses. No wall clock, no
random state, no process identity, and no dictionary insertion order reaches a
verdict. Two concurrent windows are computed by separate feature-engine
instances, so neither can see the other's history.

---

## 7. Privacy

* **No credential material is accepted, under any spelling.** Every request
  model runs the Phase 2 prohibited-key scanner over the raw keys before any
  other validation. `password`, `passwordHash`, `password_hash`, `secret`,
  `token`, `accessToken`, `authorization`, `cookie`, `privateKey` and their
  normalised variants are refused with `API013`. The scanner reads names only —
  a prohibited field's *value* is never read, copied, logged, or included in a
  message, and the refusal does not even report how many were offered.
* **Source addresses are pseudonymized on arrival.** A `source_ip` is converted
  to a source pseudonym by the project's keyed HMAC service
  (`PAD_PSEUDONYMIZATION_KEY`, environment or untracked `.env` only) and then
  dropped. The address is never stored on the canonical event, never logged, and
  never returned. A deployment with no key refuses such a request with `API014`
  rather than processing the address any other way.
* **No entity identity is returned.** Pseudonymous user, source, device, and
  session identifiers are inputs. The only identity in a response is the
  caller's own `anchor_event_id`, which is what lets a client join the verdict
  back to its own record.
* **Evidence is sanitized by contract.** `EvidenceItem` rejects any value shaped
  like a pseudonym or a UUID, and rejects proof-asserting language. Rules
  describe behaviour consistent with a pattern; they never confirm a compromise.
* **Logs carry no content.** Request logs record the path, the method, the error
  code, and a problem count. An internal exception contributes its *type* and
  nothing else — a project exception message can legitimately name a column, a
  fingerprint, or a directory, and a log record is not a private place.

---

## 8. Error contract

Every failure — validation, refusal, routing, and unforeseen — is rendered in one
envelope:

```json
{ "error": { "code": "API004", "message": "…", "detail": { "problem_count": 1 } } }
```

`detail` is optional and always aggregate: counts, limits, and field names only.
Never a value, a path, or a row.

| Code | HTTP | Meaning |
|---|---|---|
| `API001` | 422 | Malformed request or schema mismatch |
| `API002` | 422 | Not a valid canonical authentication event |
| `API003` | 422 | Duplicate event identity within the window |
| `API004` | 422 | Events are not in non-decreasing `event_time` order |
| `API005` | 413 | Window carries more events than accepted |
| `API006` | 503 | Feature contract failure |
| `API007` | 503 | Rule detection unavailable |
| `API008` | 503 | ML champion unavailable |
| `API009` | 503 | Fusion unavailable |
| `API010` | 503 | Runtime not ready |
| `API011` | 422 | Anchor selection cannot be answered |
| `API012` | 413 | Request body too large |
| `API013` | 422 | Credential material offered |
| `API014` | 422 | Source address supplied but no pseudonymization key |
| `API015` | 404 / 405 | No such route, or method not allowed |
| `API016` | 503 | Replay subsystem not available on this deployment |
| `API017` | 404 | No such scenario in the built-in replay catalog |
| `API018` | 404 | No such replay run in this process |
| `API019` | 429 | A replay bound is reached; nothing was started or extended |
| `API020` | 409 | The replay run cannot make that transition from its current state |
| `API021` | 422 | The replay timeline cursor is not a position a cursor can occupy |
| `API099` | 500 | Internal error |

A code is a contract: `API007` means the same thing in this release and in every
later one, and a client may branch on it. The message beside it is human-facing
prose that may be reworded, so nothing machine-readable is encoded only there.

**No Python traceback ever reaches a client.** Debug mode is explicitly off. The
catch-all handler logs the exception type and discards its message entirely
rather than trying to scrub it — a scrubber has to be right every time, a
discard has to be right once.

---

## 9. Runtime configuration

All variables use the `PAD_API_` prefix and may also be set in an untracked
`.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `PAD_API_HOST` | `127.0.0.1` | Bind address |
| `PAD_API_PORT` | `8000` | Bind port |
| `PAD_API_LOG_LEVEL` | `INFO` | Log level |
| `PAD_API_ENVIRONMENT` | `development` | `development` \| `testing` \| `production` |
| `PAD_API_DOCS_ENABLED` | `true` | Serve `/docs`, `/redoc`, `/openapi.json` |
| `PAD_API_ARTIFACT_ROOT` | — | Root holding `champion/`, `runs/`, `ledger/`, `evaluations/` |
| `PAD_API_ALLOWLIST_PATH` | — | The reviewed ML feature allowlist |
| `PAD_API_FEATURE_CONFIG_PATH` | — | The Phase 3 feature configuration |
| `PAD_API_ML_CONFIG_PATH` | — | The ML configuration the champion was produced under |
| `PAD_API_DETECTION_CONFIG_PATH` | — | The Phase 4 rule configuration |
| `PAD_API_SERVING_BUNDLE_ROOT` | the artifact root | Where `serving/<scope key>/` lives, if not beside the artifacts |
| `PAD_API_CHAMPION_SCOPE_KEY` | — | Which frozen scope to serve, when more than one is frozen |
| `PAD_API_MAX_BATCH_EVENTS` | `500` | Events per window (1 – 5 000) |
| `PAD_API_MAX_REQUEST_BYTES` | `1048576` | Body ceiling (1 KiB – 16 MiB) |
| `PAD_API_REQUIRE_ML_CHAMPION` | `true` | Whether a loadable champion is required for readiness |
| `PAD_API_REPLAY_ENABLED` | `true` | Whether the synthetic demonstration replay endpoints are served |
| `PAD_API_REPLAY_REQUIRED` | `false` | Whether readiness depends on the replay subsystem |

Configured paths must be **absolute** and free of `..` segments: a relative path
would resolve against whatever working directory the process happened to start
in. Every path defaults to *absent* rather than to a guessed location — a
serving process that silently found "some artifacts" under a default directory
would be serving a model nobody named.

`PAD_API_SERVING_BUNDLE_ROOT` is a **location and only a location**. There is no
setting beside it that names a strategy, a threshold, or a state: the bundle's own
frozen selection decides what runs, and the locked receipt has to agree with it.
Pointing it somewhere with no bundle can make a hybrid *absent*; it can never make
a different one.

`PAD_API_REQUIRE_ML_CHAMPION=false` switches the model layer **off entirely**,
which `/ready`, `/api/v1/system/status`, `/api/v1/model/info` and every detection
response then say out loud. It cannot select a different model, move a
threshold, or change a strategy, so it is a deployment control rather than a
scientific one.

---

## 10. Local run

```bash
uv sync --all-groups

export PAD_API_ARTIFACT_ROOT=/absolute/path/to/artifacts
export PAD_API_ALLOWLIST_PATH=/absolute/path/to/allowlist.yaml
export PAD_API_FEATURE_CONFIG_PATH=/absolute/path/to/features.yaml
export PAD_API_ML_CONFIG_PATH=/absolute/path/to/configs/ml/model-development.yaml
export PAD_API_DETECTION_CONFIG_PATH=/absolute/path/to/rules.yaml

uv run uvicorn password_attack_detector.api.app:app --host 127.0.0.1 --port 8000
```

If the frozen selection was `stacked`, materialize the serving bundle once, before
starting the service — see §2. `deploy inspect` will report what is published:

```bash
uv run password-attack-detector deploy inspect --output-root "$PAD_API_ARTIFACT_ROOT"
```

To run the rule layer alone, before a champion has been frozen:

```bash
PAD_API_REQUIRE_ML_CHAMPION=false \
  uv run uvicorn password_attack_detector.api.app:app --host 127.0.0.1 --port 8000
```

Then:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
curl -s http://127.0.0.1:8000/version
open http://127.0.0.1:8000/docs
```

`password_attack_detector.api.app:app` is the module-level application.
`create_app(settings=..., runtime=...)` is the factory; a supplied `runtime`
skips artifact resolution entirely, which is how the test suite drives the whole
HTTP surface — including deliberately broken runtimes — without a filesystem.

To drive the same service from the analyst console instead of `curl`, start it in
a second terminal — see [dashboard.md](dashboard.md) §10.

### In a container instead

```bash
docker compose up --build
```

One command, no Python toolchain, and no separate materialization step: a
one-shot `prepare` service runs the whole pipeline — including the `deploy
materialize` of §2 — into a named volume, and the API starts only after it
succeeds and mounts that volume **read-only**. The environment it is given is
exactly the table in §9, with absolute paths inside the container. Nothing is
fitted at startup there either. See [docker.md](docker.md).

### On a server, behind a reverse proxy

`compose.deploy.yaml` overlays the above: the API keeps port 8000 but stops
publishing it, and a reverse proxy becomes the only public listener. **Under the
default routing policy this service is not reachable from the internet at all** —
the dashboard's client runs server-side and reaches it as `http://api:8000`
across the Compose network, so every endpoint below still works while none of
them is public.

A second reviewed policy publishes the read-only surface — `/docs`,
`/openapi.json`, `/health`, `/ready`, `/version`, `/api/v1/system/status`,
`/api/v1/model/info`, `/api/v1/rules`, `/api/v1/demo/scenarios` — and no more.
The scoring endpoints (`/detect`, `/detect/batch`, `/explain`) and the replay
control endpoints stay internal in both policies, because scoring costs CPU and
this deployment has no rate limiting yet. Swagger's "Try it out" will return 404
or 405 for them through the proxy; that is the proxy refusing the route, not this
service failing. See [deployment.md](deployment.md) §6.

---

## 11. Swagger

The interactive documents are enabled by default for the demonstration:

* `/docs` — Swagger UI
* `/redoc` — ReDoc
* `/openapi.json` — the schema

Operations are grouped under three tags: **Health**, **Detection**, **System**.
Every route is tagged, every schema field carries a description where the field
name alone is not enough, and no description names an internal implementation
detail. Set `PAD_API_DOCS_ENABLED=false` to serve none of the three.

---

## 12. Current limitations

* **A stacked hybrid needs a materialized bundle.** The state is not derivable
  from the locked receipt alone, so a stacked deployment requires the offline
  `deploy materialize` step of §2 to have been run against the frozen lineage.
  Until it has, the deployment is `503` rather than silently serving a gate. The
  materialization needs the pre-TEST inputs — features, labels, splits,
  allowlist, rule risk assessments, campaign labels, and the frozen validation
  prediction — so it runs where those artifacts are, not on the serving host.
* **The hybrid needs a locked TEST evaluation.** A deployment that has frozen a
  champion but never run `ml evaluate` with a validation prediction reports
  `no_fusion_selection`. That is the honest state: nothing selected a strategy.
* **No persistence.** Every request is scored from the window it supplies. The
  service stores no event, no verdict, and no alert, and it has no history of
  its own — which is exactly why a caller has to supply the window. Persistence
  arrives in a later phase. The replay layer's run store is the one exception
  and is not an exception at all: it is bounded, in-memory, process-local, and
  cleared by a restart, which the documents and the console both say.
* **Replay is synthetic only.** `/api/v1/demo` replays a reviewed built-in
  catalog. There is no path for real traffic to enter one, no way to upload or
  parameterise a scenario beyond its pace, and no field on any replay request
  that names a host, a path, or a scientific parameter.
* **Two rules cannot be demonstrated on live requests. v0.6.0 ships with this
  stated, not fixed** — the release milestone examined it and concluded that
  every way of closing it changes a frozen scientific contract, which a packaging
  release must not do. `PAD-CS-001` and `PAD-ATO-001` gate on a fitted behavioural
  baseline, and the serving path computes point-in-time features from the
  supplied window alone with no baseline artifact loaded. Both report
  insufficient data on every request through this API, whether it arrives from a
  client or from a replay.

  Milestone 4 audited adding a baseline to the serving bundle on the same terms
  as the stacked state — materialized offline, deterministic, fingerprinted,
  loaded read-only, failing closed — and did not do it, for two independent
  reasons. **It would not help:** a baseline fitted from a deployment's own TRAIN
  split was loaded into a feature engine and the replay scenarios run through it,
  and `user_in_baseline` came back `False` with all five `is_new_*_for_user`
  flags `None`, unchanged, because the catalog's identities are content-addressed
  synthetic pseudonyms that no training population contains. **And it is a
  contract change, not a packaging one:** the bundle manifest has no field for a
  baseline, so adding one bumps `BUNDLE_SCHEMA_VERSION`, extends the fingerprint
  chain, gives `deploy materialize` a feature-layer input it does not take, and
  adds a loader to the serving path. Nothing was loosened in the meantime — no
  rule threshold moved and no baseline was synthesised. See
  [live-replay.md](live-replay.md) §3 and [docker.md](docker.md) §14.
  [deployment.md](deployment.md) §16 classifies **all nine** rules against
  measured per-rule outcomes: four fire on live requests, three are live-serving
  but exercised by no replay scenario, and these two cannot fire at all.
* **No alerting, grouping, or suppression.** The Phase 4 alert lifecycle
  (grouping, cooldown, rate limiting, escalation) is not exposed. The API
  returns event-level risk assessments, not `SecurityAlert` records.
* **No category head or anomaly probe.** `/api/v1/model/info` reports whether a
  category head was frozen, but no endpoint returns a category assignment or an
  experimental anomaly score.
* **Per-anchor attribution only.** `/api/v1/explain` decomposes one anchor's
  model decision. There is no population-level attribution over live traffic, and
  there should not be: Phase 5's aggregate report — permutation sensitivity,
  unused-column counts, the sealed quality manifest — is computed over a
  *partition*, and live requests are not one. The aggregate report stays with
  `ml explain`.
* **No drift endpoint.** Drift is computed offline by `ml drift` against a
  reference profile captured at training time; nothing in the serving layer
  publishes a drift report.
* **No authentication, authorisation, rate limiting, or CORS.** The service
  binds to `127.0.0.1` by default and is not hardened for exposure to an
  untrusted network. No cross-origin policy is installed: the Milestone 2 console
  calls this service from Python rather than from a browser, so no origin needs
  allowing yet, and a permissive default would be a decision nobody made.

  Milestone 5A's deployment perimeter does not change this — it *routes around*
  it. The reverse proxy sets security headers and caps a request body, and the
  default routing policy publishes none of this service's endpoints. Rate
  limiting is still absent and was deliberately deferred rather than built on a
  third-party proxy module; [deployment.md](deployment.md) §15 states the
  residual risk in full. There is still no authentication anywhere in this
  system.
* **Synthetic evaluation only.** Every published figure about this system
  describes generated authentication traffic. It is not evidence of real-world
  detection effectiveness.
* **This API is not publicly reachable, by design.** The project's public demo
  (<https://pad-demo.onrender.com>) exposes the Streamlit console and one
  liveness route, `/healthz`. Every endpoint documented here — `/api/v1/detect`,
  `/api/v1/detect/batch`, `/api/v1/explain`, `/api/v1/demo/*`, the system
  endpoints, `/ready`, `/version`, `/docs` and `/openapi.json` — is bound to
  `127.0.0.1` inside the deployment's container and answers only the console
  process running beside it. Scoring is not exposed to anonymous callers, because
  the deployment has no authentication and no rate limiting. See
  [render-deployment.md](render-deployment.md) for the routing policy and
  [deployment.md](deployment.md) for the VPS equivalent.

* **Nothing the containerized demonstration measures is a performance claim.**
  Its champion is trained on four hours of synthetic traffic, sized so the
  pipeline finishes in about a minute. The evaluation windows are far too small
  for a per-scenario metric to mean anything, and
  `configs/ml/model-demo.yaml` says so at the top.

---

## Related documents

| Document | Contents |
|---|---|
| [data-contract.md](data-contract.md) | The canonical event schema this API's requests convert into |
| [privacy-model.md](privacy-model.md) | Pseudonymization, key management, limitations |
| [temporal-semantics.md](temporal-semantics.md) | Why a window is required |
| [rule-contract.md](rule-contract.md) | What a rule may read and must return |
| [risk-scoring.md](risk-scoring.md) | What `risk_score` is, and is not |
| [model-contract.md](model-contract.md) | What a model artifact must carry and guarantee |
| [test-evaluation.md](test-evaluation.md) | The locked TEST protocol and fusion selection |
| [explainability.md](explainability.md) | The attribution contract `/api/v1/explain` reuses |
| [dashboard.md](dashboard.md) | The analyst console that consumes this API |
| [live-replay.md](live-replay.md) | The synthetic replay demonstration built on `/api/v1/detect` |
| [docker.md](docker.md) | The containerized deployment that runs this service |
| [deployment.md](deployment.md) | The public perimeter: proxy, TLS, firewall, routing policy |
| [detection-limitations.md](detection-limitations.md) | What the detection layer does not do |
