# Synthetic Data Generation

## Purpose

The synthetic generator produces labelled authentication-event datasets for
development, testing, and offline model evaluation. Synthetic data is **not**
representative of any real user population and must not be used as evidence of
real-world model performance.

## Design principles

- **Deterministic**: the same `SyntheticConfig` and seed always produce
  identical output across runs on the same `uv.lock` environment.
- **Ground-truth-separated**: canonical events carry no label columns. Labels
  are in a separate `labels.parquet` file joined by `event_id`.
- **Privacy-safe by default**: pseudonym-format identifiers are generated;
  `PseudonymService` is never called for synthetic data.
- **Configurable scale**: `events_per_hour`, `duration_hours`, `num_users`,
  `num_sources`, `num_devices`, `num_applications`, and per-scenario parameters
  are all configurable.

## Reproducibility guarantee

Reproducibility is bounded by the committed `uv.lock` environment. The same
seed on the same locked environment always produces the same fingerprint. A
different Python, NumPy, or library version may produce different output even
for the same seed.

## Nine scenario types

### 1. `normal`
Baseline legitimate authentication traffic. Mixed methods, realistic outcome
distributions, geographic diversity. `malicious=False`.

### 2. `brute_force`
High-frequency repeated failures against a single account from a single source.
Configured by `attempts_per_campaign` and `num_campaigns`.

### 3. `password_spraying`
Low-frequency failures spread across many accounts using a common password.
Configured by `passwords_per_round` and `num_campaigns`.

### 4. `credential_stuffing`
Automated testing of credential pairs from an external breach list. Configured
by `credentials_per_batch` and `num_campaigns`.

### 5. `distributed_brute_force`
Brute-force attack coordinated across multiple source IPs to evade per-IP rate
limits. Configured by `attempts_per_source`, `num_sources`, `num_campaigns`.

### 6. `account_takeover_indicator`
Post-compromise indicators: unusual location after successful login, privilege
escalation pattern. Configured by `num_campaigns`.

### 7. `impossible_travel`
Authentication successes from geographically distant locations within an
impossibly short time window. Configured by `num_campaigns`.

### 8. `bot_activity`
Machine-speed uniform-interval authentication bursts with bot-like user agents.
Configured by `events_per_campaign` and `num_campaigns`.

### 9. `novel_anomaly_holdout`
Reserved for evaluating model generalisation to unknown attack types.
`supervised_training_eligible=False` for all events in this scenario — these
events must not be used as supervised training examples. This is not an
ordinary supervised class; it represents attacks the model has not seen.

## Label schema

Each event has one `GroundTruthLabel`:

```
event_id                 UUID (FK to events.parquet)
campaign_id              str  (UUIDv5 — stable per campaign)
scenario                 ScenarioType enum
malicious                bool
supervised_training_eligible bool (False for novel_anomaly_holdout)
generator_version        str
scenario_variant         str | None
campaign_stage           str | None
```

## Configuration

See `configs/data/synthetic-testing.yaml` for the minimal testing configuration
and `configs/data/synthetic-development.yaml` for the larger development dataset.

The development-size configuration must not run in CI or `scripts/verify.sh`.

## Content fingerprint

`compute_events_fingerprint` sorts events by `str(event_id)` and hashes a
canonical JSON representation using the `CANONICAL_EVENT_COLUMNS` ordering. The
hash is independent of Parquet row ordering and byte encoding.

## Known limitations

- Synthetic data does not capture real-world distributional properties of
  legitimate traffic (session timing, geographic correlation, browser
  distribution).
- Attack scenarios are simplistic simulations. Real attacks have more variable
  timing, diversity, and evasion behaviour.
- No feature engineering, rolling windows, or time-series features are computed
  in Phase 2. These are planned for future phases.
- Reproducibility requires the same `uv.lock` environment. Different library
  versions may produce different output.
- The novel-anomaly holdout class should not be treated as a regular supervised
  training class.
