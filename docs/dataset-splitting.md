# Dataset Splitting

## Overview

Splits are cut on **time**, grouped by **campaign**, and separated by a
**purge** interval. Each of those three choices exists to close a specific
leak.

## Chronological, never random

A random split would place an attack campaign's early events in training and
its later events in test. A model could then memorise the campaign rather than
learn the behaviour, and the test score would measure recall of something it
had already seen.

Splitting on time also matches how detection is actually deployed: a model
trained on the past is applied to the future.

## The five splits

| Split | Meaning |
|---|---|
| `train` | Supervised training; the only data a baseline may be fitted on |
| `validation` | Supervised tuning |
| `test` | Supervised evaluation |
| `novel_anomaly_holdout` | Evaluation-only pool for anomaly detection |
| `excluded` | Retained in the tables, not usable for training or evaluation |

The split assignment lives in its own table keyed by `event_id`. It is never a
model feature.

## Boundaries

Two modes:

- **`fractions`** (default) — boundaries are the event times at the cumulative
  fraction positions. Because assignment compares `event_time < boundary`, a
  boundary landing inside a group of simultaneous events still keeps that whole
  group on one side. Events with byte-identical histories are never split apart.
- **`boundaries`** — three explicit UTC timestamps, validated strictly
  increasing.

A dataset too short or too concentrated in time to yield three non-empty
intervals is rejected rather than silently producing an empty split.

## Novel-anomaly holdout

Any event whose scenario is in `holdout_scenarios`, **or** whose
`supervised_training_eligible` is false, is routed to
`novel_anomaly_holdout` before any grouping happens.

Routing first stops a holdout campaign from dragging co-located events through
the grouping logic. The holdout is an evaluation-only pool — time-slicing it
would only shrink it without improving isolation.

The holdout is never converted into a supervised class and never appears in
`train`, `validation`, or `test`. It remains available for a later
anomaly-detection evaluation.

## Grouping

| Activity | Default grouping |
|---|---|
| Attack scenarios | By `campaign_id` |
| Normal activity | `singleton` — each benign event is its own group |

Grouping benign traffic by campaign is available (`normal_grouping:
campaign_id`) but is not the default. The leak that campaign-awareness closes
comes from *coordinated* activity whose members share structure a model could
memorise; benign traffic has no such coordination. Grouping it would create
enormous groups straddling every boundary and, under the default exclude
policy, discard the entire benign class.

## Campaigns that straddle a boundary

Default policy: **`exclude`**. Every event of a straddling campaign is
excluded, with reason `campaign_crosses_split_boundary`. It is the only policy
with zero leakage risk and the only one that is trivially auditable.

Escape hatch: **`assign_by_first_event`**. The group takes the split of its
earliest member in canonical order — fully deterministic, since the
`(event_time, event_id)` ordering leaves no tie to break. Use it when exclusion
starves a scenario on a small dataset.

A "majority" policy is deliberately not implemented: ties would need an
arbitrary rule, and the leaked-tail problem would remain.

## Purge and embargo

Direction is the easy thing to get backwards.

Features look **backwards**, so contamination flows **forward across** a
boundary. An event just *after* `train_end` has a lookback window that reaches
into training data — it is the contaminated one.

- **Purge** excludes events on the **later** side: `[B, B + purge)`.
  Reason: `purged_after_boundary`.
- **Embargo** excludes events on the **earlier** side: `[B - embargo, B)`.
  Reason: `embargoed_before_boundary`.

Under `strict_isolation` (the default), `purge` must be at least the maximum
configured feature window. A shorter purge raises `ConfigurationError` at
configuration time, not at run time.

## Sizing a dataset against the purge

Strict isolation costs a fixed amount of wall-clock time at each boundary, and
that cost is independent of the fractions:

    excluded fraction  ~=  2 * purge / dataset duration

For the 168-hour Phase 2 development dataset with a 24h purge, that is roughly
28.6% — whatever fractions are chosen. The fractions only decide **who pays**.

Under the 70/15/15 default, the validation and test windows are 25 hours each,
so a 24h purge leaves about one usable hour in each: a technically valid split
that is useless for evaluation. The shipped development configuration uses
50/25/25 instead, giving each evaluation window 42 hours of which about 18
survive.

**Every evaluation window must be comfortably longer than the purge interval.**
If it is not, lengthen the dataset rather than shortening the purge — a purge
shorter than the longest feature window does not isolate anything, and under
`strict_isolation` it is rejected outright.

Note also that `max_excluded_fraction` has to exceed `2 * purge / duration`, or
every build raises regardless of how well the split is otherwise formed.

## Nothing is dropped silently

Every input event receives an assignment, and that is asserted rather than
assumed — a mismatch raises. Every exclusion carries a machine-readable reason,
and the counts per reason appear in the split manifest.

If the excluded fraction exceeds `max_excluded_fraction` (default 0.10) the
split **raises**. Badly placed boundaries should be loud, not quiet.

## The split manifest

Aggregate only — no event identifier ever appears. It records the schema
version, the label and configuration fingerprints, the three boundaries, purge
and embargo seconds, per-split row counts, per-split class and scenario
distributions, per-split campaign counts, exclusion counts by reason, the
holdout count, and the excluded fraction.

## Known limitations

- Fraction-derived boundaries depend on the whole dataset, so appending events
  moves them. Pin explicit boundaries when reproducibility across differently
  sized inputs matters.
- The purge interval is a blunt instrument: it excludes every event in the
  interval, not only those whose windows actually reach across. That is
  deliberate — the conservative version is the auditable one.
- Campaign isolation depends on `campaign_id` being meaningful. It comes from
  the synthetic generator; a real deployment would need an equivalent notion of
  coordinated activity, which this phase does not attempt to infer.
- A split that passes every check here still says nothing about whether a model
  trained on it will detect anything.
