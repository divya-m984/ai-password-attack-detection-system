# Feature Contract

## Overview

A **feature snapshot** is one row describing one canonical authentication
event. The full column list is generated in `docs/feature-catalog.md`; this
document describes the contract those columns obey.

## Schema version

`feature_schema_version = "1.0.0"`

The version is stamped on every row and recorded in every manifest. Increment
it when the column set changes in a backwards-incompatible way.

## Three separate tables

| Table | Key | Contents |
|---|---|---|
| `feature_snapshots.parquet` | `anchor_event_id` | Model inputs only |
| `feature_labels.parquet` | `event_id` | `attack_class`, `malicious`, `supervised_training_eligible` |
| `feature_splits.parquet` | `event_id` | `split`, `exclusion_reason` |

They are kept apart deliberately. A ground-truth column sitting beside a model
input is one careless `SELECT *` away from a meaningless evaluation.

`campaign_id` appears in **none** of them. The splitter reads it internally to
enforce campaign isolation, and it is never published.

## What a snapshot contains

**Identity and timing**

- `feature_schema_version`
- `anchor_event_id` — the only identifier, and never a model input
- `anchor_event_time` — UTC

**Current-event context** — the anchor's own recorded fields, legitimate
because detection runs after the event completed:

`current_authentication_outcome`, `current_authentication_method`,
`current_mfa_outcome`, `current_client_type`, `current_response_time_ms`,
`current_country_code`, `current_has_device`, `current_has_location`

**Engineered features** — calendar, windowed history, sequence, geospatial, and
baseline-derived columns.

## What a snapshot never contains

- Ground truth: `malicious`, `scenario`, `attack_class`, `label`, `target`,
  `is_attack`, `attack_probability`, `risk_score`, `supervised_training_eligible`
- Campaign metadata: `campaign_id`, `campaign_stage`, `scenario_variant`
- Split assignment: `split`, `exclusion_reason`
- Model output: `model_probability`
- Raw coordinates, user/source/device/session pseudonyms

These are rejected at catalog build time, checked again by the validator
(`F006`), and checked a third time by the leakage auditor.

## Naming

Windowed features are named `{entity}_{measure}__{window}`, for example
`source_attempt_count__1m`, `user_failure_rate__5m`,
`source_unique_user_count__15m`, `user_unique_source_count__1h`.

Names describe **observations and deviations, never conclusions**. Vague names
(`attempts`, `score`, `suspicious`, `anomaly`) and verdict names
(`account_takeover`, `bot_detected`, `impossible_travel`) are rejected at
catalog build time.

## Null versus zero

The distinction is load-bearing. "No attempts were made" and "no failures out
of many attempts" are different observations, and encoding both as `0.0` would
destroy that.

| Category | Rule |
|---|---|
| Windowed counts, unique counts | **Always 0, never null.** Zero events in a window is a true, informative observation. |
| Rates | **Null** when the denominator is below `min_count_for_rate`. Never `0.0` for an empty denominator. |
| Means | Null when the window holds no qualifying observations |
| Standard deviations, CoV | Null when fewer than two observations. Never `0.0` for one. |
| `seconds_since_*` | Null when no qualifying prior event exists |
| `prior_consecutive_*`, `prior_failures_since_*` | **0, never null** — zero consecutive failures is literally true |
| `previous_*_outcome` | Null when no prior event exists |
| Baseline-derived | **Null** when the entity is absent from the fitted baseline |

Nullability in Parquet **is** the validity signal; `col IS NULL` already
answers the question. Companion columns exist only where null is genuinely
ambiguous — four in the whole feature set:

- `user_previous_success_geo__status`
- `implied_velocity__status`
- `user_in_baseline`
- `source_in_baseline`

## Storage

Tables are written directly in PyArrow against a schema derived from the
catalog. Nullability is a declared per-field property, column order comes from
`catalog.column_order()`, and timestamps preserve UTC. `NaN` is never used as a
missing-value marker: it cannot express a missing integer, and `NaN != NaN`
would break the exact-equality comparisons the engine's guarantees rest on.

## Fingerprints

| Fingerprint | Covers | Excludes |
|---|---|---|
| Configuration | Semantic feature behaviour | Output paths, overwrite flags, timestamps |
| Catalog | Names, types, nullability, units, leakage classes, thresholds | `description`, `null_semantics` prose |
| Feature content | Every cell, order-independent | Row order, Parquet physical layout |
| Baseline content | Fitted state | `created_at` |

The configuration fingerprint is path-independent by construction: the same
semantic configuration in two directories produces the same digest.

## Known limitations

- `current_has_device` is constant in practice: the canonical event schema
  makes `device_id` mandatory, so the column is always `true`. It is retained
  for forward compatibility with sources that omit device identity, and is
  reported as zero-variance in the quality report.
- The feature set is wide (~200 columns at the shipped development
  configuration). Narrowing it is a modelling decision for a later phase; the
  catalog carries the `group`, `window`, and `entity` metadata needed to do it.
- Synthetic data exercises every code path here but does not demonstrate that
  these features detect anything in the real world.
