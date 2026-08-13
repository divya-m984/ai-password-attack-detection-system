# Phase 5 acceptance report

Requirement-by-requirement standing of the machine-learning detection layer. Every status below is **derived** -- from an executable contract in the source, or from an artifact that was supplied. There is no hand-written pass in this document, and a requirement nothing established is recorded as `inconclusive` rather than assumed.

`inconclusive` blocks acceptance. `not_applicable` does not, and marks a contract that genuinely does not apply -- an optional head nothing froze has no behaviour to accept.

**No performance figure appears here.** This report says whether the locked evaluation happened under its protocol, never how it came out.

A requirement that names a pipeline artifact is established by supplying one. The tracked copy of this document is generated without any, so those requirements read `inconclusive` here; the integration suite builds the same report against a real CI-sized pipeline run and asserts each of them resolves.

| Field | Value |
| --- | --- |
| Acceptance schema | 1.0.0 |
| Package version | 0.5.0 |
| Report fingerprint | `0d5af7b268677a8b18a067fab04146d704fa8c1ebe5064d289dc52e9237cc7ea` |
| Requirements | 29 |
| Pass | 19 |
| Fail | 0 |
| Inconclusive | 10 |
| Not applicable | 0 |
| Accepted | no |

## Requirements

| Id | Milestone | Requirement | Status | Evidence |
| --- | --- | --- | --- | --- |
| `P5-M1-CATALOG` | M1 | The model catalog is versioned and declares eligibility per entry | pass | the catalog declares 6 specification(s), each carrying an explicit champion-eligibility flag |
| `P5-M1-CONFIG-IDENTITY` | M1 | The configuration fingerprint excludes paths and output settings | pass | 4 field(s) are excluded from the semantic configuration digest, so the same configuration in two directories fingerprints identically |
| `P5-M1-PROBABILITY-VOCABULARY` | M1 | Only a calibrated score may be described as a probability | pass | exactly one score kind is admitted as a probability, and it is the calibrated one |
| `P5-M10-DRIFT-COMPARED` | M10 | A later population was compared against the frozen reference | inconclusive | no drift run was supplied |
| `P5-M10-DRIFT-REFERENCE-SOURCE` | M10 | The drift reference is the training population | pass | exactly one split may be a drift reference, and it is train |
| `P5-M10-DRIFT-THRESHOLDED-METRIC` | M10 | Every drift result is thresholded by a configured value | pass | one metric is reported, the reviewed configuration declares its warn and alert values, and insufficient support and absence are distinct statuses from no-drift |
| `P5-M10-EXPLANATION-EXACT-OR-UNAVAILABLE` | M10 | Attribution is exact or typed unavailable, never approximate | pass | three local methods reconstruct the model's own decision quantity, and the model-agnostic global measure is excluded from that set because it decomposes nothing |
| `P5-M10-EXPLANATION-PRODUCED` | M10 | Deterministic attribution was produced for the frozen champion | inconclusive | no explanation was supplied |
| `P5-M10-EXPLANATION-SCOPE` | M10 | The locked evaluation population is never explained | pass | attribution is admitted over training and validation rows and over nothing else |
| `P5-M10-MONITORING-CARRIES-NO-OUTCOME` | M10 | Explanation and drift artifacts carry no outcome quantity | pass | neither the explanation report nor the drift report declares an outcome-dependent field, so neither can be read as evaluation |
| `P5-M10-NO-AUTOMATIC-RETRAINING` | M10 | No drift finding triggers a fit, a promotion, or a threshold change | pass | the drift module imports no training, selection, freeze, threshold, or evaluation entry point, so there is no call it could make |
| `P5-M10-REFERENCE-PROFILE-CAPTURED` | M10 | A reference profile was captured from the training population | inconclusive | no reference profile was supplied |
| `P5-M10-VERSION-CONSISTENT` | M10 | The package declares one version everywhere | pass | the runtime package declares version 0.5.0 |
| `P5-M2-LABEL-READER-BOUNDARY` | M2 | Exactly two modules may open a ground-truth table | pass | the label-reader allowlist is exactly {detection.evaluation, ml.dataset} |
| `P5-M2-NO-TEST-PARTITION` | M2 | No fitted quantity may name test or the holdout as its source | pass | the validation-partition vocabulary has two members and neither is test or the novel-anomaly holdout |
| `P5-M3-TRAIN-ONLY-FITTING` | M3 | Preprocessing and class weighting are fitted on training rows only | pass | exactly one split is fit-eligible, and it is train |
| `P5-M4-NO-EXECUTABLE-ARTIFACT` | M4 | Model artifacts carry numbers, never a serialized object | pass | no module in this layer imports pickle, dill, or joblib, and no artifact reader calls eval, exec, or a dynamic import |
| `P5-M5-CALIBRATION-SOURCE` | M5 | Calibration and threshold selection read separate validation halves | pass | the configuration pins the calibration source and the threshold source to different validation partitions, each as a Literal with one admissible value |
| `P5-M6-LEDGER-RECORD-TYPES` | M6 | The experiment ledger declares its record types exhaustively | pass | the append-only ledger declares 4 record types |
| `P5-M6-TRAINING-RUNS-PUBLISHED` | M6 | Training runs are published to the append-only ledger | inconclusive | no ledger was supplied, so no run count was established |
| `P5-M7-CATEGORY-HEAD` | M7 | A category triage head is frozen only when it cleared its gates | inconclusive | no champion lock was supplied, so no head could be inspected |
| `P5-M7-CHAMPION-FROZEN` | M7 | A champion was selected on validation only and frozen | inconclusive | no champion lock was supplied |
| `P5-M7-LOCK-CARRIES-NO-METRIC` | M7 | The champion lock records identity, never performance | pass | the champion lock declares no metric-shaped field and no prohibited metadata field |
| `P5-M8-PREDICTION-CARRIES-NO-OUTCOME` | M8 | A prediction publication carries no outcome quantity | pass | neither the prediction manifest nor the aggregate quality report declares an outcome-dependent field |
| `P5-M8-PREDICTIONS-PUBLISHED` | M8 | Predictions were published under the frozen champion | inconclusive | no prediction manifest was supplied |
| `P5-M9-FUSION-CANDIDATE-UNIVERSE` | M9 | The declared fusion candidate universe is the whole vocabulary | pass | three fusion strategies are declared, and the stacked strategy is one of them |
| `P5-M9-FUSION-SELECTED-ON-VALIDATION` | M9 | Fusion was selected from the whole candidate universe, before test | inconclusive | no fusion selection was supplied |
| `P5-M9-NOVEL-HOLDOUT-EXPERIMENTAL` | M9 | The novel-anomaly holdout is evaluated on its own experimental track | inconclusive | no holdout population was supplied |
| `P5-M9-TEST-EVALUATED-ONCE` | M9 | The locked test evaluation ran under a lineage frozen before it | inconclusive | no test evaluation record was supplied |

## What acceptance does not mean

- Every requirement above concerns a **contract**, not an outcome. A fully accepted Phase 5 is one whose protocol held, not one whose detector works.
- Every figure this repository can produce was measured on synthetic traffic generated by this repository. None of it is evidence about real authentication systems.
- This is not a production system. It serves nothing, deploys nowhere, and handles no credential.
