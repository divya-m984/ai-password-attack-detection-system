# Drift monitoring — the frozen reference profile and what moved

Milestone 10 captures a description of the training population once, freezes it,
and later compares another population against it.

---

## 1. Drift is monitoring evidence, not model correctness

Everything measured here is computed **without a single label**. Nothing in this
contract can say a model got worse — only that the rows it is being shown, or the
decisions it is making about them, no longer look like the population it was
baselined on.

Those are different findings with different remedies, and one is not weak
evidence for the other. A feature distribution can move because an organisation
onboarded a new office. A flagged rate can move because an attack actually
happened.

---

## 2. The reference is TRAIN, and only TRAIN

`DRIFT_REFERENCE_ELIGIBLE_SPLITS` has exactly one member. The reviewed
configuration says the same thing as a string (`drift.reference_source: train`);
the frozenset says it where a split is a type.

- **Test** and the **novel-anomaly holdout** are absent for the reason they are
  absent everywhere else in this layer.
- **Validation** is absent for a narrower reason: it fitted the calibrator and
  chose the operating point, so a baseline drawn from it would be a baseline
  drawn from rows the frozen champion was already tuned against.

---

## 3. The profile is immutable

`MLReferenceProfile` is a sealed record: its digest is a field, recomputed on
every construction and every deserialization, and a profile whose content and
digest disagree is refused rather than repaired. It carries no path, no host, and
no timestamp, so two captures of the same reference population in two
directories produce identical bytes.

`ml drift` writes it once. Recapturing the same population is idempotent;
recapturing a **different** one at the same location is refused:

> A baseline is immutable once created; rebaselining a monitor onto its own
> newest input is how drift stops being detectable.

---

## 4. Incoming data never redefines the partition

Every bin edge, category, class, and expected share is fixed at capture time from
reference rows alone. `ml.drift` reads the profile and never writes to it — and
there is no field on a frozen profile that could be written to.

Incoming values are **assigned** to cells:

| Situation | Where it lands | Why |
|---|---|---|
| Value beyond the reference range | the open-ended outer cell | numeric partitions carry **interior** edges only, so the lowest and highest cells are unbounded |
| Missing value | `__null__` | present on every feature partition, including one whose reference rows were never null |
| Category the reference vocabulary lacks | `__unknown__` | inventing a cell would let the incoming population define the partition it is measured against |
| Reference-rare category, when the preprocessor buckets rares | `__other__` | mirrors the frozen encoder |

**Nothing is discarded for being unexpected.** An observation nobody counted is
the one worth counting.

### Degenerate reference partitions

| `partition_kind` | When | Cells |
|---|---|---|
| `quantile` | the ordinary numeric case | `drift.quantile_count` cells plus `__null__` |
| `constant` | every observed reference value is identical | `equals_<c>`, `__unknown__`, `__null__` |
| `all_null` | the reference never observed a value | `__null__`, `__unknown__` |
| `boolean` | a boolean feature | `false`, `true`, `__null__` |
| `vocabulary` | a categorical feature | the **preprocessor's** frozen categories, plus the buckets |

A constant reference still detects drift — a later population that stops carrying
that value moves into `__unknown__`. A quantile grid would have collapsed to
nothing.

Categorical vocabularies come from the **frozen preprocessor**, not from the
reference data. A category the preprocessor retained but the reference rows never
carried is still a cell with zero expected mass, because the model has a column
for it and a population that starts using it has drifted.

---

## 5. One thresholded measure

`DriftMetric` has exactly one member, and a test asserts it stays that way.

The population stability index over a frozen partition is the only measure the
reviewed configuration declares warn and alert values for. A second thresholded
family would be reported against thresholds nobody has chosen.

```
PSI = Σ over cells of (p_incoming − p_reference) × ln(p_incoming / p_reference)
```

Each share is floored at `PSI_PROPORTION_FLOOR = 1e-6` first, and the vectors are
deliberately **not** renormalised afterwards — renormalising would make a cell's
contribution depend on how many other cells happened to be empty. The floor keeps
a zero-expected cell finite and keeps it large, which is the honest signal: a
category the reference never saw, now appearing, is strong drift.

Because the constant does not depend on either population's size, two runs over
the same data are equal.

### Null rates and unknown rates

Both are reported, and both are **fields on the result** rather than separately
thresholded metrics — because the `__null__` and `__unknown__` cells are part of
the very partition the index is summed over. A shift in either moves the
thresholded number rather than sitting beside it.

`FeatureDriftResult` carries `reference_null_rate`, `incoming_null_rate`,
`null_rate_delta`, and — for categorical features only — the three unknown-rate
equivalents. A validator refuses a delta that is not the difference of the rates
it is reported with.

---

## 6. Statuses

| Status | Meaning |
|---|---|
| `no_drift` | measured, adequate support, below the warning threshold |
| `drift_warning` | measured at or above `psi_warn_threshold`, below `psi_alert_threshold` |
| `drift_detected` | measured at or above `psi_alert_threshold` |
| `inconclusive` | computable, but over too few rows on one side to be evidence |
| `unavailable` | not computable at all: the quantity does not exist on one side |

**`inconclusive` is not `no_drift`.** A comparison over too few rows has not
established stability. **`unavailable` is not zero.** A quantity that does not
exist has not been measured as absent. A monitor that reports stability when it
measured nothing is worse than no monitor.

A schema validator enforces the distinction structurally: a measured status must
carry an `observed_value` and the thresholds it was decided against, and a
refusal must carry neither. A number beside a refusal reads as a measurement.

The aggregate is worst-first, and an **empty** result set aggregates to
`unavailable` rather than `no_drift`: nothing was compared, so nothing was found
stable.

Every result also carries a stable `reason_code` a caller may branch on:
`psi_below_warn_threshold`, `psi_at_or_above_warn_threshold`,
`psi_at_or_above_alert_threshold`, `insufficient_reference_support`,
`insufficient_incoming_support`, `quantity_absent_from_incoming`,
`quantity_absent_from_reference`.

### Support

`drift.min_reference_rows` is applied to **both** sides. A comparison against
thirty incoming rows is no more conclusive than one drawn from thirty reference
rows, and the reason codes distinguish which side was short.

---

## 7. Feature drift and prediction drift stay apart

Two result types, two report sections, two aggregate statuses, and no combined
verdict. A single blended number would average a shift in the inputs against a
shift in the outputs and say neither.

Prediction quantities, each partitioned from the reference publication:

| Quantity | Partition |
|---|---|
| `flagged_malicious_rate` | `{not_flagged, flagged}` |
| `decision_score` | quantile cells |
| `calibrated_probability` | quantile cells, only when the champion is calibrated |
| `category_predicted_class` | the frozen head's class space plus `unknown` |
| `category_unknown_rate` | `{known, unknown}` |
| `anomaly_score` | quantile cells — reported under its **own** quantity so the experimental track can never be read as part of the supervised result |

A quantity the incoming publication carries and the reference does not is
reported `unavailable` against the reference, never measured against a partition
invented on the spot.

---

## 8. Nothing retrains

There is **no code path** from a finding here to a model, a threshold, a fusion
strategy, a champion lock, or an evaluation record. The `ml.drift` module imports
no training, selection, freeze, threshold, evaluation, or publication entry
point, and the Phase 5 acceptance report checks that by parsing the module's own
import list rather than reading its documentation. A monitor that could call
`fit` is a monitor that might.

There is no webhook, no scheduler, and no action automation anywhere in this
layer. A warning is a reason for a human to look. It is never an action.

`ml drift` exits non-zero when either aggregate status is a warning or an alert,
so the finding is visible to whatever ran the command — and that is the entire
extent of its effect.

---

## 9. Artifacts

| File | Contents |
|---|---|
| `<artifact-root>/reference/reference_profile.json` | the frozen `MLReferenceProfile` |
| `<reports>/ml_drift_report.json` | `DriftManifest` and `MLDriftReport` |
| `<reports>/ml_drift_report.md` | the same, rendered |
| `<reports>/ml_reference_profile.md` | the profile's identity and shape |

All are gitignored.

The rendered profile document carries the identity, the lineage, and the *shape*
of each partition — not its expected masses. A table of two hundred numeric cells
is not review material, and rendering it would put the reference distribution
into a document that gets pasted around. The masses stay in the JSON.

Aggregate reports carry no row, no anchor, no feature value, and no bin-level
mass. A structural guard at import refuses a prohibited metadata field name on
every schema in both modules.

`reference_profile_id` and `drift_run_id` are both derived from content alone, so
the same comparison in two directories produces the same bytes.

---

## 10. The command

```bash
uv run password-attack-detector ml drift \
    --features   processed/feature_snapshots.parquet \
    --splits     processed/feature_splits.parquet \
    --feature-manifest processed/feature_manifest.json \
    --allowlist  allowlist.yaml \
    --incoming-features later/feature_snapshots.parquet \
    --incoming-splits   later/feature_splits.parquet \
    --incoming-split    validation \
    --reference-prediction <train-prediction-id> \
    --incoming-prediction <incoming-prediction-id> \
    --output-root artifacts/ml
```

`--features` / `--splits` hold the **reference** population; its TRAIN rows are
what the profile is cut from. `--incoming-features` / `--incoming-splits` default
to the same tables, which is the ordinary case when monitoring a later split of
one snapshot.

The prediction options are optional and come as a pair: comparing an output
distribution against no baseline, or a baseline against nothing, measures
neither. Omit both for feature drift only.

There is no `--labels` option, no metric here needs one, and nothing about
accuracy is computable from what the command reads.

---

## 11. Configuration

```yaml
drift:
  enabled: true
  reference_source: train
  quantile_count: 20
  psi_warn_threshold: 0.10
  psi_alert_threshold: 0.25
  min_reference_rows: 500
```

A validator refuses a warn threshold at or above the alert threshold.
`enabled: false` makes the command refuse rather than run behind its own flag.

The reference profile binds `drift_fingerprint()` — the digest of this block
alone. The cell count and the thresholds decide what a comparison means, and
nothing else in the configuration does.

---

## 12. Limitations

- **Synthetic populations.** A high index says two generated populations differ.
  It says nothing about real authentication traffic.
- **Chronological splits differ by construction.** Comparing training rows
  against a later split of the same stream will show real distributional
  movement in time-derived features. That is the data being what it is, not the
  monitor misbehaving.
- **PSI is descriptive.** It has no significance test attached, and the warn and
  alert values are conventional rather than derived from this data.
- **No label is read**, so no statement about accuracy, precision, recall, or
  calibration error is computable from anything here.
