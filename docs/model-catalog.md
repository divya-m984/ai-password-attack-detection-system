# Model Catalog

Generated from the model registry. Do not edit by hand; regenerate with `password-attack-detector ml catalog --format markdown`.

- Model catalog version: `1.0.0`
- Catalog fingerprint: `ba4eac6fdc4f41fcd3d27d150e3238c863df15b3fa5af7cc6648840b793b389e`
- Declared model families: 6

This catalog declares *what may be fitted*, not what was fitted or how well anything performed. It contains no measured result, and no performance figure is ever transcribed into this repository's prose.

## What catalog membership does not mean

- **Membership is not championship.** A family listed here may be fitted and evaluated. Becoming the champion additionally requires clearing every validation gate, and no model is promoted on the strength of appearing in this table.
- **Serializer and inference parity are required, and are proven later.** A family becomes promotable only once its canonical serializer and its inference adapter reproduce identical scores after a round trip. Until that parity is demonstrated, a family is evaluable but not promotable.
- **`M-021` is gated.** Histogram gradient boosting reads private estimator attributes, so it ships with `champion_eligible = false` until a compatibility test pins that layout across the bounded dependency range.
- **`M-030` is an experimental anomaly model.** It is unsupervised, anomaly-only, and can never be the supervised champion. Its output is an ordered outlier magnitude.
- **Probabilities require calibration.** Every family's native output is a decision score or a class score. It may be described as a probability only after a calibrator has been fitted and its calibration error measured.
- **Phase 5 is offline and defensive.** Nothing here serves a model, exposes an endpoint, touches live authentication traffic, or handles a credential.

## Eligibility summary

| Eligibility status | Families |
|--------|-------|
| `anomaly_only` | 1 |
| `champion_eligible` | 4 |
| `serializer_unproven` | 1 |

## Model index

| Model | Family | Tasks | Champion eligible | Experimental |
|--------|-------|-------|-------|-------|
| `M-000` | `prior_baseline` | `binary_malicious`, `attack_category` | yes | no |
| `M-001` | `single_feature_threshold` | `binary_malicious` | yes | no |
| `M-010` | `logistic_regression` | `binary_malicious`, `attack_category` | yes | no |
| `M-020` | `random_forest` | `binary_malicious`, `attack_category` | yes | no |
| `M-021` | `histogram_gradient_boosting` | `binary_malicious`, `attack_category` | no | no |
| `M-030` | `isolation_forest` | `anomaly` | no | yes |

## M-000 -- Class prior baseline

Emits the training-split positive rate for every row, ignoring features entirely. It is the reference a candidate must beat before any of its numbers are worth reading: a model that cannot out-rank the base rate has learned nothing from the features.

| Property | Value |
|--------|-------|
| Model version | `1.0.0` |
| Family | `prior_baseline` |
| Supported tasks | `binary_malicious`, `attack_category` |
| Native score kind | `decision_score` |
| Calibration compatible | yes |
| Calibration methods | `none`, `platt`, `isotonic` |
| Multiclass capable | yes |
| Champion eligible | yes |
| Eligibility status | `champion_eligible` |
| Experimental | no |
| Anomaly only | no |
| Serializer | `json_prior_v1` |
| Inference adapter | `prior_v1` |
| Requires scikit-learn | no |

**Determinism controls**

- closed-form fit
- no random number generator

**Limitations**

- Constant output. Ranking is undefined, so every ranking metric over it is reported as unavailable rather than as a tie.
- Exists to be beaten. Promoting it would mean no candidate cleared the gates.

## M-001 -- Single-feature threshold baseline

Ranks rows by one eligible feature chosen on the training split by separation. It shows how much of a candidate's advantage comes from the feature set rather than from the learning algorithm.

| Property | Value |
|--------|-------|
| Model version | `1.0.0` |
| Family | `single_feature_threshold` |
| Supported tasks | `binary_malicious` |
| Native score kind | `decision_score` |
| Calibration compatible | yes |
| Calibration methods | `none`, `platt`, `isotonic` |
| Multiclass capable | no |
| Champion eligible | yes |
| Eligibility status | `champion_eligible` |
| Experimental | no |
| Anomaly only | no |
| Serializer | `json_threshold_v1` |
| Inference adapter | `threshold_v1` |
| Requires scikit-learn | no |

**Determinism controls**

- deterministic feature ranking with a lexicographic tie-break
- no random number generator

**Hyperparameters**

| Name | Kind | Default | Minimum | Maximum | Choices | Tunable |
|--------|-------|-------|-------|-------|-------|-------|
| `selection_metric` | `string` | `roc_auc` | - | - | `roc_auc`, `average_precision` | yes |
| `min_non_null_fraction` | `float` | `0.5` | `0` | `1` | - | yes |

**Limitations**

- One feature cannot express an interaction, so it under-reports what the feature set holds.
- Its chosen feature is recorded in the model artifact; a different training split may choose a different one.

## M-010 -- Logistic regression

Regularised linear model over the standardised design matrix. Its coefficients are directly readable, which makes it the easiest candidate to audit and the natural reference for how much a non-linear family actually adds.

| Property | Value |
|--------|-------|
| Model version | `1.0.0` |
| Family | `logistic_regression` |
| Supported tasks | `binary_malicious`, `attack_category` |
| Native score kind | `decision_score` |
| Calibration compatible | yes |
| Calibration methods | `none`, `platt`, `isotonic` |
| Multiclass capable | yes |
| Champion eligible | yes |
| Eligibility status | `champion_eligible` |
| Experimental | no |
| Anomaly only | no |
| Serializer | `json_linear_v1` |
| Inference adapter | `linear_logit_v1` |
| Requires scikit-learn | yes |
| Estimator | `LogisticRegression` |
| Public estimator attributes | `coef_`, `intercept_`, `classes_` |

**Determinism controls**

- random_state fixed
- deterministic lbfgs solver
- fixed feature order recorded in the artifact

**Hyperparameters**

| Name | Kind | Default | Minimum | Maximum | Choices | Tunable |
|--------|-------|-------|-------|-------|-------|-------|
| `penalty` | `string` | `l2` | - | - | `l2` | no |
| `solver` | `string` | `lbfgs` | - | - | `lbfgs` | no |
| `c_inverse_regularization` | `float` | `1.0` | `0.0001` | `10000` | - | yes |
| `max_iter` | `int` | `1000` | `50` | `100000` | - | yes |
| `tol` | `float` | `0.0001` | `1e-10` | `0.1` | - | yes |
| `class_weight` | `string` | `balanced` | - | - | `balanced`, `none` | yes |
| `random_state` | `int` | `42` | `0` | `2147483647` | - | no |

**Limitations**

- Linear in the transformed space; it cannot represent an interaction the preprocessing did not already encode.
- Its raw output is an ordered decision score. It becomes a calibrated quantity only after a calibrator is fitted and its calibration error measured.

## M-020 -- Random forest

Bagged axis-aligned trees over the untransformed design matrix. Chosen as the ensemble candidate because its fitted structure is exposed through documented array attributes, so a canonical serializer can read it without depending on an internal layout.

| Property | Value |
|--------|-------|
| Model version | `1.0.0` |
| Family | `random_forest` |
| Supported tasks | `binary_malicious`, `attack_category` |
| Native score kind | `decision_score` |
| Calibration compatible | yes |
| Calibration methods | `none`, `platt`, `isotonic` |
| Multiclass capable | yes |
| Champion eligible | yes |
| Eligibility status | `champion_eligible` |
| Experimental | no |
| Anomaly only | no |
| Serializer | `json_tree_ensemble_v1` |
| Inference adapter | `tree_vote_v1` |
| Requires scikit-learn | yes |
| Estimator | `RandomForestClassifier` |
| Public estimator attributes | `estimators_`, `classes_`, `n_outputs_`, `n_features_in_` |

**Determinism controls**

- random_state fixed
- n_jobs pinned to 1
- OMP_NUM_THREADS=1 for bit-for-bit reproduction

**Hyperparameters**

| Name | Kind | Default | Minimum | Maximum | Choices | Tunable |
|--------|-------|-------|-------|-------|-------|-------|
| `n_estimators` | `int` | `300` | `1` | `2000` | - | yes |
| `max_depth` | `int` | `12` | `1` | `64` | - | yes |
| `min_samples_leaf` | `int` | `5` | `1` | `1000` | - | yes |
| `max_features` | `string` | `sqrt` | - | - | `sqrt`, `log2` | yes |
| `class_weight` | `string` | `balanced` | - | - | `balanced`, `none` | yes |
| `random_state` | `int` | `42` | `0` | `2147483647` | - | no |
| `n_jobs` | `int` | `1` | `1` | `1` | - | no |

**Limitations**

- Leaf-frequency output is poorly calibrated by construction; it needs a fitted calibrator before it can be read as a rate.
- Artifact size grows with tree count and depth, so both are bounded in the declaration above.

## M-021 -- Histogram gradient boosting

Boosted trees over binned features, typically the strongest tabular family available here. It is gated: its fitted structure is reachable only through private estimator attributes, so it may be evaluated but not promoted until a compatibility test settles whether a stable serializer can be built on them.

| Property | Value |
|--------|-------|
| Model version | `1.0.0` |
| Family | `histogram_gradient_boosting` |
| Supported tasks | `binary_malicious`, `attack_category` |
| Native score kind | `decision_score` |
| Calibration compatible | yes |
| Calibration methods | `none`, `platt`, `isotonic` |
| Multiclass capable | yes |
| Champion eligible | no |
| Eligibility status | `serializer_unproven` |
| Experimental | no |
| Anomaly only | no |
| Serializer | `json_histogram_ensemble_v1` |
| Inference adapter | `histogram_raw_v1` |
| Requires scikit-learn | yes |
| Estimator | `HistGradientBoostingClassifier` |
| Public estimator attributes | `classes_`, `n_iter_`, `n_features_in_` |
| Private estimator attributes | `_predictors`, `_baseline_prediction` |

**Determinism controls**

- random_state fixed
- early_stopping pinned off
- OMP_NUM_THREADS=1 for bit-for-bit reproduction

**Hyperparameters**

| Name | Kind | Default | Minimum | Maximum | Choices | Tunable |
|--------|-------|-------|-------|-------|-------|-------|
| `max_iter` | `int` | `200` | `1` | `2000` | - | yes |
| `learning_rate` | `float` | `0.1` | `0.0001` | `1` | - | yes |
| `max_leaf_nodes` | `int` | `31` | `2` | `255` | - | yes |
| `min_samples_leaf` | `int` | `20` | `1` | `1000` | - | yes |
| `l2_regularization` | `float` | `0.0` | `0` | `100` | - | yes |
| `early_stopping` | `bool` | `False` | - | - | - | no |
| `random_state` | `int` | `42` | `0` | `2147483647` | - | no |

**Limitations**

- Its fitted trees are reachable only through private attributes, whose layout is not part of the scikit-learn public contract. Promotion is blocked until a compatibility test pins that layout on the bounded version range.
- If that test cannot be made to hold, this family stays evaluable and the random forest remains the ensemble candidate.

## M-030 -- Isolation forest

Unsupervised outlier scorer fitted on benign training rows without reading a label. It exists to probe behaviour on the novel-anomaly holdout, where no supervised model has a class to predict and no rule was written.

| Property | Value |
|--------|-------|
| Model version | `1.0.0` |
| Family | `isolation_forest` |
| Supported tasks | `anomaly` |
| Native score kind | `anomaly_score` |
| Calibration compatible | no |
| Calibration methods | `none` |
| Multiclass capable | no |
| Champion eligible | no |
| Eligibility status | `anomaly_only` |
| Experimental | yes |
| Anomaly only | yes |
| Serializer | `json_isolation_forest_v1` |
| Inference adapter | `isolation_path_v1` |
| Requires scikit-learn | yes |
| Estimator | `IsolationForest` |
| Public estimator attributes | `estimators_`, `estimators_features_`, `max_samples_`, `offset_`, `n_features_in_` |

**Determinism controls**

- random_state fixed
- n_jobs pinned to 1
- max_samples declared as an absolute count

**Hyperparameters**

| Name | Kind | Default | Minimum | Maximum | Choices | Tunable |
|--------|-------|-------|-------|-------|-------|-------|
| `n_estimators` | `int` | `200` | `1` | `1000` | - | yes |
| `max_samples` | `int` | `256` | `16` | `100000` | - | yes |
| `contamination` | `float` | `0.01` | `1e-06` | `0.5` | - | yes |
| `random_state` | `int` | `42` | `0` | `2147483647` | - | no |
| `n_jobs` | `int` | `1` | `1` | `1` | - | no |

**Limitations**

- Its output is an ordered outlier magnitude. It is not a calibrated quantity and no report may present it as one.
- It never influences champion selection or threshold choice; its results are reported in a separate holdout section.
- Its path-length normalisation is reimplemented from the published formula and pinned by a parity test, because the scikit-learn helper is private.

## Known limitations

- Models consume Phase 3 point-in-time feature snapshots. Split assignments, campaign metadata, and ground-truth labels reach the training path only through the single module permitted to read them, and never as model inputs.
- A declared hyperparameter range bounds what configuration may request. It says nothing about which value is appropriate for a given dataset.
- Determinism controls make a fit reproducible on one locked dependency set. They do not make results comparable across different scikit-learn releases, which is why the dependency range is bounded on both sides.
- Synthetic data exercises every family declared here but demonstrates nothing about real-world detection effectiveness.
