# Behavioral Baselines

## Overview

A behavioral baseline summarises what "usual" looks like for a user or a
source. It is **fitted** once from an approved reference interval and then
**applied** unchanged to every event. Separating those two operations is what
keeps evaluation honest.

## Fit and transform are separate operations

`fit` rebuilds all state from an explicit set of events. `transform_one` reads
that state and mutates nothing — the fitted state is a frozen dataclass, so
purity is enforced by the type rather than by convention.

Validation and test events therefore cannot move a training baseline, and that
is a structural property, not a discipline.

## Fitting is permission-checked, not trusted

`fit` takes an explicit `permitted_event_ids` set and an `interval`, and
**raises** on any event outside either. It never silently skips.

That distinction matters: a baseline that quietly dropped disallowed events
would still be fitted, and the mistake the parameter exists to catch would go
unnoticed.

`fit` also records `fitted_source_fingerprint` — a content digest of exactly
what it consumed. The leakage auditor recomputes that digest independently from
the split table and compares. A baseline fitted on the wrong events is
detectable without consulting the baseline's own claims.

## Fitted state

**Per user**

- Known devices, sources, countries, applications, authentication methods
- Smoothed hour-of-day histogram (24 buckets, sums to 1)
- Historical success rate and event rate per hour
- Response-time mean and standard deviation
- Interarrival median and 90th percentile
- Location centroid and located-event count
- Which known-value sets were truncated

**Per source**

- Event count and targeted-user count
- Success rate and event rate per hour
- Distinct client types and user agents, plus their Shannon entropies
- Response-time mean and standard deviation

Known-value sets are capped at `known_set_max_size`, most-frequent-first.
Without a cap, one high-volume entity could produce a ragged Parquet cell with
tens of thousands of entries. Truncation is recorded, never silent.

## Derived features

`is_new_device_for_user`, `is_new_source_for_user`, `is_new_country_for_user`,
`is_new_application_for_user`, `is_new_auth_method_for_user`,
`login_hour_deviation`, `user_success_rate_deviation`, `user_event_rate_ratio`,
`response_time_zscore`, `source_user_fanout_ratio`, `source_event_rate_ratio`,
`distance_from_user_baseline_centroid_km`, plus the coverage flags
`user_in_baseline` and `source_in_baseline`.

Every name describes a **deviation or a novelty**, never a conclusion. Nothing
here asserts an attack.

## Cold entities are null, not novel

An entity absent from the fitted baseline receives **null** for every derived
column — never `is_new_device_for_user = true`.

Reporting an unknown user's device as "new" would conflate two very different
observations: "we have never seen this user" and "we know this user, and this
device is new for them". The first is a statement about coverage, the second
about behaviour. Collapsing them gives a model a clean signal for *which split
a row came from*, which is exactly the leak the split design exists to prevent.
The two coverage flags make the distinction explicit.

The ratio features additionally compare an observed windowed count against a
fitted rate. The window is `baseline.rate_reference_window`, validated to be
one of the configured windows and one that tracks cardinality.

## Artifact layout and privacy

Fitted state is keyed by pseudonymous identifiers and holds per-entity
known-value sets. It is **sensitive operational metadata**.

```
artifacts/baselines/<name>/
  baseline.json             0644   metadata only, zero pseudonyms
  user_baselines.parquet    0600   pseudonym-bearing
  source_baselines.parquet  0600   pseudonym-bearing
```

Reports and CLI output read **only** `baseline.json`. That is how "identifiers
never appear in reports" is guaranteed structurally rather than remembered:
the code that renders summaries has no access to the other files.

Restrictive permissions are set explicitly rather than relying on the process
umask. They are best-effort — some filesystems do not support them — and
failing a whole fit over that would be worse than the weaker protection.

Baseline artifacts live under `artifacts/`, which is git-ignored. **Never
commit generated baseline state.** Real-data baselines require protected
storage with access control and retention limits appropriate to the underlying
authentication logs.

## Determinism

The content fingerprint covers the *logical* state, not any serialised bytes,
so it does not vary with pyarrow version or compression settings:

- Frozen sets are sorted before hashing
- Floats are rendered at fixed precision
- Entities are iterated in sorted key order
- `created_at` is **excluded** — it lives only in `baseline.json`

Fitting the same events twice, in any order, produces the same fingerprint.

## Known limitations

- **In-sample optimism.** Transforming a training event uses a baseline that
  saw that same event, so training rows are mildly optimistic. This phase
  accepts that and records `baseline_in_sample_for_train: true` in the
  artifact. Leave-one-out baselines are a later modelling concern; the flag is
  there so a later phase cannot forget the caveat exists.
- A baseline summarises the interval it was fitted on. Applying it to events
  far outside that interval is possible but not meaningful, and nothing in this
  phase prevents it.
- Baselines are fitted from synthetic data here. Nothing about their behaviour
  on synthetic scenarios demonstrates real-world effectiveness.
