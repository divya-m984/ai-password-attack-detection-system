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

## Known limitations

- Reproducibility is bounded by the committed `uv.lock` environment. The
  `uv_lock_sha256` field in the manifest records the exact environment used.
- Phase 2 does not implement DVC or MLflow for experiment tracking. Reproducibility
  relies on the locked Python environment and committed YAML configurations.
