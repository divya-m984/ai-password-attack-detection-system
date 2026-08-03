# Leakage Prevention

## Overview

Leakage is the failure mode that matters most here, because it is silent.
A leaking feature pipeline produces *better* metrics, not worse ones, so it
passes every test that only looks at results.

This phase defends against it in three layers: structural guarantees in the
engine, declared metadata in the catalog, and an explicit auditor that checks
both — partly by behaviour, not just by name.

## Layer 1 — structure

The engine cannot include the anchor or its simultaneous peers in a history
window, because it never decides inclusion by comparison. It processes
timestamp blocks and emits every row in a block before ingesting any of them.
See `docs/temporal-semantics.md`.

## Layer 2 — declared metadata

Every column carries a `leakage_class` in the catalog:

| Class | Meaning |
|---|---|
| `key` | Identity and timing; never a model input |
| `current_event_context` | The anchor's own recorded fields, legitimate because detection runs after the event completed |
| `prior_only` | Computed strictly from `[t - window, t)` |
| `baseline_derived` | Derived from a baseline fitted on an approved reference interval |

The auditor uses this field rather than pattern-matching column names, so the
classification cannot silently rot as features are added.

## Layer 3 — the twelve checks

| # | Check | How |
|---|---|---|
| 1 | `NO_GROUND_TRUTH_COLUMNS` | Column set against the prohibited set |
| 2 | `NO_CAMPAIGN_COLUMNS` | Column set |
| 3 | `NO_SPLIT_COLUMNS` | Column set |
| 4 | `NO_FUTURE_CONTRIBUTION` | **Behavioural** — appends a later event, asserts nothing earlier moved |
| 5 | `NO_SAME_TIMESTAMP_CONTRIBUTION` | **Behavioural** — adds a simultaneous peer, asserts rows at or before that instant are unchanged |
| 6 | `BASELINE_SOURCE_FINGERPRINT_MATCHES_TRAIN` | Recomputes the training fingerprint from the split table and compares |
| 7 | `BASELINE_NOT_FIT_ON_EVALUATION_DATA` | Compares consumed event count against the training split |
| 8 | `PURGE_INTERVAL_RESPECTED` | **Behavioural** — for each supervised event, checks nothing in its lookback belongs to another supervised split |
| 9 | `CAMPAIGN_GROUPS_ISOLATED` | No campaign spans two supervised splits |
| 10 | `HOLDOUT_EXCLUDED_FROM_SUPERVISED` | No ineligible event sits in a supervised split |
| 11 | `CURRENT_VS_PRIOR_FIELDS_DISTINGUISHED` | Every column's declared class matches its role |
| 12 | `JOIN_KEY_INTEGRITY` | Features, labels, and splits join one-to-one on `event_id` |

## Why some checks must be behavioural

Checks 1–3 and 11–12 inspect names and schemas. They would pass on a pipeline
whose arithmetic silently reached forward in time, because nothing about the
name `user_failure_count__5m` says which events it counted.

Checks 4, 5, and 8 therefore mutate the input and compare outputs. Check 6
recomputes a fingerprint from an independent source. These are the checks that
would actually catch a regression in the engine.

## Skipped is not passed

A check whose inputs were not supplied is recorded as **skipped**, and the
audit's overall status becomes `warning` rather than `pass`. An audit that
quietly degrades to a subset of its checks is worse than one that says what it
could not verify.

## Running the audit

The auditor is a first-class CLI command, not only a step inside `build`:

```bash
uv run password-attack-detector features audit-leakage \
  data/interim/authentication_events.parquet \
  --labels data/interim/synthetic_ground_truth.parquet \
  --splits data/processed/feature_splits.parquet \
  --baseline artifacts/baselines/development \
  --config configs/features/feature-development.yaml
```

It exits non-zero on failure, so it works as a build gate. `features build`
runs it automatically and fails the build if it does not pass; `--skip-audit`
exists as an escape hatch and prints a loud warning.

## Report privacy

`LeakageAuditResult` contains counts, column names, and check messages only.
No event identifier, pseudonym, coordinate, or raw row appears in it, so the
result is safe to embed in a manifest or a build log. Tests assert this with
pattern matching over the rendered JSON and Markdown.

## Known limitations

- The behavioural checks probe the configured feature set on the supplied
  events. They demonstrate the timing contract holds *for that data*; they are
  not a proof for all possible inputs.
- Baseline provenance is verified by fingerprint comparison. That detects a
  mismatched fit, but not a baseline fitted from data that never entered the
  split table at all.
- Check 8 compares against the actual assignments rather than re-deriving
  boundaries, so it catches a purge that was configured but not applied. It
  cannot tell you what purge interval you *should* have chosen.
- Passing this audit says nothing about detection effectiveness.
