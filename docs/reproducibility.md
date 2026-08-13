# Reproducibility

## Scope

Reproducibility applies to **synthetic dataset generation** only. Ingested
real-data datasets are inherently dependent on the source files provided.

## What is reproducible

Given the same `SyntheticConfig` (including `seed`) and the same locked
dependency environment (`uv.lock`), `generate_dataset` always produces an
identical set of `AuthEvent` and `GroundTruthLabel` objects with the same
`content_fingerprint`.

## What is NOT reproducible across environments

- **Different `uv.lock`**: a library version upgrade (NumPy, pandas, etc.) can
  change internal RNG behaviour even for the same seed.
- **Different Python minor version**: `3.12.x` vs `3.13.x` may differ.
- **Parallel execution**: the generator uses a single seeded RNG
  (`np.random.default_rng(seed)`) and is single-threaded. Concurrent runs on
  different machines with the same seed are reproducible.

## Deterministic identifiers

All synthetic UUIDs are UUIDv5 (name-based) with fixed namespaces. The same
seed and configuration always produce the same UUIDs. UUIDv4 (random) is never
used for synthetic generation.

## Content fingerprint

`compute_events_fingerprint(events)` produces a SHA-256 hex digest that:

1. Sorts events by `str(event_id)` (order-independent)
2. Serialises each event to a stable dict using `CANONICAL_EVENT_COLUMNS` order,
   UTC ISO-8601 timestamps, and string enum values
3. Hashes the sorted, JSON-serialised list with SHA-256

The fingerprint is independent of Parquet encoding and row ordering. Two
datasets with the same events in different file formats will have the same
fingerprint.

## Reproducibility metadata

Each `manifest.json` includes a `reproducibility` section:

```json
{
  "python_version": "3.12.13",
  "numpy_version": "1.26.4",
  "pandas_version": "2.2.2",
  "pyarrow_version": "14.0.2",
  "uv_lock_sha256": "<sha256 of uv.lock>",
  "generator_version": "1.0.0",
  "seed": 42
}
```

To reproduce a dataset:

1. Ensure the `uv.lock` SHA-256 matches (install with `uv sync --locked`).
2. Use the same `SyntheticConfig` YAML (or reproduce it from `config_fingerprint`).
3. Run `password-attack-detector data generate`.
4. Compare `content_fingerprint` values; they must match.

## Config fingerprint

`SyntheticConfig.fingerprint()` computes a SHA-256 of the configuration fields
that affect output (seed, all counts, enabled scenarios, campaign parameters).
Fields that do not affect output (output paths, overwrite flags) are excluded.

The config fingerprint is stored in `manifest.json` as `config_fingerprint`.

## Feature-layer fingerprints (Phase 3+)

| Fingerprint | Covers | Excludes |
|---|---|---|
| Feature configuration | Windows, thresholds, policies, nested baseline/split/geospatial settings | Output directories, overwrite flags, absolute paths, timestamps |
| Feature catalog | Names, groups, entities, windows, aggregates, types, nullability, units, leakage classes, thresholds, ranges | `description` and `null_semantics` prose |
| Feature content | Every cell of the feature table, rows sorted by anchor identifier | Row order, Parquet physical layout |
| Baseline content | Fitted per-entity state, sets sorted, floats at fixed precision | `created_at` |
| Split configuration | Mode, fractions or boundaries, purge, embargo, policies | Nothing path-dependent |

Two properties are deliberate and tested:

- The **configuration** fingerprint is path-independent: the same semantic
  configuration stored in two different directories produces the same digest.
- The **catalog** fingerprint excludes prose, so correcting a typo in a feature
  description does not invalidate every artifact that recorded the digest.

Feature computation is bit-for-bit reproducible because sums and
sums-of-squares accumulate as exact integers rather than floats. Floating-point
addition is not associative, so a float accumulator would make results depend
on eviction order.

## Phase 4: detection fingerprints

| Fingerprint | Covers | Excludes |
|---|---|---|
| Detection configuration | Enabled rules, effective per-rule parameters, family weights, signal, scoring, severity thresholds, alerting policy | `output_dir`, `reports_dir`, `overwrite` |
| Rule catalog | Identifiers, versions, families, categories, feature templates, parameter declarations, evidence codes, minimum history | Descriptions and limitations (prose) |
| Detection content | Every published detection row, canonically encoded | Row order, Parquet layout, path, timestamps |
| Risk content | Every published assessment row | Same |
| Alert content | Every published alert row | Same |
| Report content | A report's semantic fields | Named excluded keys such as `created_at` |

Content fingerprints sort rows by their semantic key and hash canonical JSON
with floats rounded to nine decimals, so the digest depends on logical content
and nothing else — not the order rows arrived in, not pyarrow's physical
layout, not the directory written to, and not when it was written.
**A creation timestamp never enters a deterministic fingerprint.**

`dataset_id` for a detection set is UUIDv5 over the three content fingerprints:
the same content always identifies the same set, on any machine, on any day.
Never `uuid4`, never a wall clock, never machine identity.

Detection identifiers are `uuid5(anchor_event_id | rule_id | rule_version)`, so
a detection keeps its identity across configuration retunings. The
configuration fingerprint that produced a *score* is recorded on the
`RiskAssessment` instead, which is what keeps two executions distinguishable.

Alert identifiers are `uuid5` over the alerting version, the grouping key, and
the alert's own `first_seen` — deliberately excluding the contributing rule
set, so an alert that later absorbs one more rule keeps its identity.

Detection is deterministic end to end: rules are prepared once and iterated in
sorted catalog order, snapshots are evaluated in
`(anchor_event_time, anchor_event_id)` order, correlation groups are sorted
before the noisy-OR product fixes float multiplication order, and alerts are
grouped in one ordered pass. Re-running with the same inputs and configuration
produces byte-identical Parquet files.

## Phase 5: machine-learning fingerprints

| Fingerprint | Covers | Excludes |
|---|---|---|
| Eligible feature list | The reviewed allowlist restricted to the configured leakage classes and feature groups, in catalog order | Allowlist file layout, YAML key order |
| Preprocessor | Every fitted imputation, encoding, vocabulary, and scaling statistic | Where and when it was fitted |
| Model content | Hyperparameters, class order, transformed column order, and a digest over every fitted array | Estimator internals, library version, path |
| Calibration state | The fitted calibrator's parameters and declared method | Its source file, its run directory |
| Threshold selection | The chosen operating point, the objective, and the evidence it was chosen from | The directory the selection ran in |
| ML configuration | Every semantic setting, hyperparameters at their *effective* values | `output_dir`, `models_dir`, `reports_dir`, `overwrite` |
| Champion lock | Every fingerprint above, plus the serializer, adapter, and dependency contract | Any metric, any path, any host, any timestamp |
| Prediction content and manifest | The scored rows, the frozen lineage, and the inference input | Physical Parquet layout, publication directory |
| Test evaluation record | The frozen lineage, the population, the rule configuration, and the fusion selection | When it ran, where it ran |
| Explanation manifest and report | The champion, the publication, the method, the explain configuration, and the digests of what was produced | Paths, the terminal it printed to |
| Reference profile | The split, the population digest, every partition, the champion lineage, and the drift configuration | The root it was captured under |
| Drift manifest and report | The reference profile, the incoming population, and the report | Both directories, the wall clock |

Two properties hold across every entry. **Identity is semantic**: the same inputs
rebuilt in two directories on two days produce the same value. And **no fitted
quantity is derived from an evaluation split**: the whole lineage above is fixed
before the TEST labels are opened, which is what makes a TEST prediction safe to
publish and a TEST evaluation meaningful.

The audit is executable. `tests/integration/test_phase5_reproducibility.py`
rebuilds the entire pipeline — dataset, features, preprocessing, models,
calibration, thresholds, the ledger, selection, the freeze, predictions, the
locked evaluation, attribution, the reference profile, and drift — in two
independent temporary roots and compares every identity above. Positive controls
prove the lineage is live: changing one training event's value moves the
preprocessor, the model, and the lock. Negative controls prove the boundaries
hold: physical row order moves nothing, a pipeline that never scores TEST
produces the identical champion, and neither attribution nor monitoring changes a
byte of any frozen artifact.

## Known limitations

- Reproducibility is bounded by the committed `uv.lock` environment. The
  `uv_lock_sha256` field in the manifest records the exact environment used.
- The project does not implement DVC or MLflow for experiment tracking.
  Reproducibility relies on the locked Python environment, the committed YAML
  configurations, and the project's own append-only experiment ledger.
- Estimator *fitting* uses scikit-learn, so a fitted model is reproducible only
  within the reviewed dependency range the champion lock records. Everything
  after fitting — scoring, calibration, thresholds, attribution — is project
  code reading published arrays, and stays reproducible across that range.
