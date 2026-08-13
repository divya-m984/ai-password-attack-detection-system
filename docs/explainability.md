# Explainability — deterministic model attribution

Milestone 10 answers one narrow question about the frozen champion — *which
transformed columns moved this model's decision, and by how much* — and refuses
every wider one.

---

## 1. What an explanation is, and what it is not

An attribution here says how a **fitted function decomposes over the columns it
was handed**. That is all it says.

It does **not** say:

- that the behaviour behind a column *caused* the outcome;
- that changing that behaviour would change an attacker's success;
- that the model was right about the row;
- that the ranking transfers to another model, another split, or real traffic.

The vocabulary is kept deliberately flat throughout the code and the artifacts:
`contribution`, never `importance`, `impact`, `driver`, or `because`.

---

## 2. Exact or unavailable

There is no approximate path, no surrogate model, and no sampling. A per-feature
number that cannot be reconstructed reads exactly like one that can once it is
in a table, so a family this build cannot decompose exactly reports
`ExplanationStatus.UNAVAILABLE` and a stable reason code instead.

Three champion-eligible families decompose exactly, each against the **published
artifact alone** — the arrays the Milestone 4 serializer wrote, read through the
same public shapes inference reads them through. Nothing imports scikit-learn,
and nothing reaches into a `Tree` object.

| Family | Method | Decomposes | Baseline |
|---|---|---|---|
| `logistic_regression` | `linear_logit_contribution` | the logit | the fitted intercept |
| `random_forest` | `tree_path_contribution` | the mean leaf score | the ensemble-mean root value |
| `single_feature_threshold` | `single_feature_step_contribution` | the 0/1 step | `0.0` |

### Linear

`contribution_j = transformed_value_j × coefficient_j`, with the intercept
recorded separately as `baseline_value`. The binary decision value scikit-learn
computes is `X @ coef_[0] + intercept_[0]`, which is exactly what the published
`coefficients` and `intercept` arrays hold, so the sum reconstructs the score
rather than approximating it.

### Tree ensemble

The decision-path decomposition. Walking root to leaf in one tree, each split is
credited with the change it makes to that node's stored positive-class share.
Telescoping over the path leaves `leaf − root`; averaging over trees leaves
`mean_leaf − mean_root`, and `mean_leaf` is exactly what `traverse_forest`
returns. The equality is arithmetic, not empirical.

The comparison is made at `float32` against a `float64` threshold — the
estimator's own contract, reproduced here for the same reason the scorer
reproduces it: a row sitting on a cut takes a different branch otherwise.

### Threshold baseline

The model reads one reviewed column and votes. That column is credited with the
whole vote and every other column with exactly zero — not as an approximation,
but because the fitted function is constant in each of them.

---

## 3. The reconstruction contract

Every exact method is checked before anything is published:

```
baseline_value + sum(contributions) == decision_value
```

within `RECONSTRUCTION_TOLERANCE = 1e-9`. The residual is *recorded* on each row
explanation and the worst one is recorded on the aggregate report, so a reader
can see it was checked rather than take the claim on trust. A decomposition that
disagrees with the model's own scorer by more than the tolerance is refused —
not published with a caveat.

The `decision_value` is taken from the model's **own scorer**, not recomputed by
the attribution code. That is what makes the residual evidence: two independent
paths agree, rather than one path agreeing with itself.

---

## 4. Raw score, calibration, and threshold are three separate things

The decomposition is of the **decision function**. It is not a decomposition of
the calibrated probability, and nothing here describes one as a sum of
contributions.

```
contributions  ──sum──►  decision value        (this is what is decomposed)
                              │
                              ▼
                    frozen calibrator          (a separate transformation)
                              │
                              ▼
                  calibrated probability
                              │
                              ▼
                 frozen decision threshold     (a separate decision)
```

The score kind the operating point was applied to is recorded on the report as
context and is never summed with a contribution.

---

## 5. The global measure

`permutation_score_sensitivity` is model-agnostic, label-free, and deterministic:
for each transformed column, the mean absolute change in the model's decision
score when that column is replaced by a permuted copy of itself, averaged over
`explain.permutation_repeats` repeats.

The permutation for repeat *r* of column *j* is drawn from a generator seeded by
`(seed, r, j)`, so two runs produce identical numbers and the order columns are
visited in cannot change any of them.

**It decomposes nothing** and is deliberately excluded from
`EXACT_EXPLANATION_METHODS`. A classical permutation *importance* is measured
against a metric and therefore needs labels; this one is measured against the
model's own output, which is why it is available at all. Correlated columns can
mask each other, so a small number is not evidence a column is unused.

---

## 6. What may be explained

`EXPLANATION_ELIGIBLE_SPLITS` is `{train, validation}`. The absence of `test` and
`novel_anomaly_holdout` is the enforcement, and it is checked twice: by the
`ml explain` command and again by the library, which refuses an ineligible scope
whatever the caller says.

A per-row artifact derived from the locked evaluation population has no place
beside the one evaluation that was permitted to read it.

---

## 7. Privacy

- A contribution names a **transformed column**, which is a reviewed engineered
  feature name the allowlist already admitted.
- It carries a **value** only when `explain.include_feature_values` is on
  (default `false`). A feature value can be a country code, and widening
  disclosure is a deliberate act.
- Row-level explanations carry `anchor_event_id` — the minimum join identity a
  row-level artifact needs to be attributable to the prediction it explains, and
  the same identity that prediction row already carries — and nothing else. No
  pseudonym, no campaign, no coordinate, no raw feature row, no label.
- The **aggregate report carries no anchor at all**. A structural guard at import
  refuses an `anchor_event_id` field on any schema in the module except
  `PredictionExplanation`, and refuses a prohibited metadata field name on all of
  them.
- `explain.max_local_explanations` bounds how many row explanations are emitted
  (default `0`). Which rows are emitted is decided **after** sorting by anchor,
  so the selection cannot depend on scoring order.

---

## 8. Artifacts

| File | Contents |
|---|---|
| `explanation_manifest.json` | `ExplanationManifest` — identity and every frozen lineage fingerprint |
| `explanation_report.json` | `ExplanationQualityReport` — sealed aggregate summary |
| `explanation_report.md` | the same, rendered |
| `local_explanations.json` | emitted row explanations, when the bound is non-zero |

All four are written under `<artifact-root>/explanations/<explanation_id>/` and
are gitignored.

`explanation_id` is derived from content alone — the champion lock, the
prediction manifest, the explain configuration, and the digests of what was
produced — so the same run in two directories agrees byte for byte. No path, no
hostname, no timestamp reaches it.

The manifest binds the champion **and** the publication, and
`build_explanation_manifest` refuses a publication produced by a different
champion: attributing one model's output to another model's coefficients would be
wrong in a way no digest would catch.

---

## 9. The command

```bash
uv run password-attack-detector ml explain \
    --features   processed/feature_snapshots.parquet \
    --splits     processed/feature_splits.parquet \
    --feature-manifest processed/feature_manifest.json \
    --allowlist  allowlist.yaml \
    --split      validation \
    --prediction <prediction-id> \
    --output-root artifacts/ml
```

There is no `--labels` option. The command:

- verifies the champion lock in full before anything is transformed;
- validates the prediction publication and refuses an invalid one;
- checks that the rows supplied are **exactly** the rows that publication was
  produced from, by comparing `inference_input_fingerprint`;
- transforms them with the champion's own frozen preprocessor;
- fits nothing, writes nothing a later command consumes as input to a fit, and
  cannot change a champion, a threshold, a fusion selection, or an evaluation
  record.

Terminal output is aggregate: column names, magnitudes, and identity. No anchor,
no feature vector, no coefficient, no pseudonym, no absolute path.

---

## 10. Configuration

```yaml
explain:
  enabled: true
  partition: validation_a
  top_k_features: 25
  permutation_repeats: 5
  max_local_explanations: 0
  include_feature_values: false
```

`enabled: false` makes the command refuse rather than run behind its own flag.
The explanation manifest binds `explain_fingerprint()` — the digest of this
block alone, not the whole configuration, because an attribution run does not
depend on the calibration method or the gate values and binding those would make
an explanation's identity move when something it never read changed.

---

## 11. Limitations

- Attribution describes **this** fitted model on **this** population. It does not
  transfer.
- The decomposition is of the raw decision quantity. No contribution is a share
  of a calibrated probability.
- The global measure reports output movement under scrambling. Correlated
  columns mask each other.
- No label is read, so nothing here says whether any decision was correct.
- The populations involved are synthetic. A column that matters in generated
  traffic need not matter in real traffic.
