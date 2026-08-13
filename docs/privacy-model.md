# Privacy Model

## Overview

The Password Attack Detector is a **defensive monitoring system**, not a
surveillance tool. Its privacy controls are designed to reduce exposure of
personal identifiers while preserving the signals needed for anomaly detection.

## Limitations

> **Pseudonymization reduces exposure but does not guarantee anonymity.**
>
> HMAC-SHA256 pseudonyms are stable per domain within a single key. An
> adversary who obtains the key and a reference user identifier can recover the
> link between a pseudonym and the original value. Pseudonymization is not
> equivalent to anonymization.
>
> This system is intended for internal security operations, not for public
> data release.

## What is never stored

- Plaintext passwords, password hashes, or salts
- Authentication tokens, cookies, session secrets, or refresh tokens
- API keys or private keys in credential files
- Real IP addresses or hostnames (replaced by pseudonyms)
- Real usernames or account identifiers (replaced by pseudonyms)
- Raw device fingerprints (replaced by pseudonyms)

These restrictions are enforced by `scan_prohibited_keys` and
`PROHIBITED_GT_COLUMNS` before any data is written.

## Pseudonymization

Real-data ingestion pseudonymizes four identifier fields:

| Field | Domain | Pseudonym prefix |
|---|---|---|
| `user_id` | `user` | `u:` |
| `source_id` | `source` | `s:` |
| `device_id` | `device` | `d:` |
| `session_id` | `session` | `sess:` |

### Algorithm

Each pseudonym is computed as:

```
HMAC-SHA256(key, domain + ":" + original_value)
```

where `key` is derived from `PAD_PSEUDONYMIZATION_KEY`.

Properties:
- **Deterministic**: the same key, domain, and value always yield the same pseudonym.
- **Cross-domain isolation**: `u:alice` and `s:alice` produce different pseudonyms,
  preventing cross-domain linkage.
- **Key-dependent**: without the key, pseudonyms cannot be reversed.

### Key management

- The key must be set via `PAD_PSEUDONYMIZATION_KEY` in the environment or an
  untracked `.env` file.
- The key is never stored in YAML configuration files, manifests, logs, or
  exception messages.
- The `show-config` command redacts the key field.
- The key is excluded from `Settings.model_dump()` output.

## Prohibited field enforcement

Before any row is processed, `scan_prohibited_keys` recursively inspects all
keys in the source record (to arbitrary nesting depth). If any key matches a
prohibited name (after normalization), the **entire dataset** is rejected —
not just the offending row.

Ground-truth column names (`campaign_id`, `scenario`, `malicious`, etc.) are
also rejected from canonical event files to enforce the GT-separation contract.

## Synthetic data

Synthetic data uses randomly generated UUIDv5 pseudonym-format identifiers and
never calls `PseudonymService`. The pseudonymization key is not required and
must not be used for synthetic generation.

## Location data

Coordinates are coarsened to one decimal place (approximately 11 km resolution)
before storage. Raw GPS coordinates or precise location data are never stored.
User agent strings include only family names, not version numbers.

## Data minimization in reports

Quality reports (`quality-report.json`, `quality-report.md`) contain only:
- Aggregate counts and statistics
- Column names (no data values)
- Enum distribution counts (no identifier values)

No raw event values, pseudonyms, or identifier substrings appear in reports.

## Feature-layer handling (Phase 3+)

Feature snapshots carry no identifier other than `anchor_event_id`, and no
coordinates. Geospatial features are published as derived distances, elapsed
intervals, and a categorical availability status — never as latitude or
longitude.

Fitted behavioral baselines are the one Phase 3 artifact that holds
pseudonymous identifiers. They are split by privacy class:

| File | Mode | Contents |
|---|---|---|
| `baseline.json` | 0644 | Metadata and fingerprints only, zero pseudonyms |
| `user_baselines.parquet` | 0600 | Pseudonymous per-user state |
| `source_baselines.parquet` | 0600 | Pseudonymous per-source state |

Reports and CLI output read only `baseline.json`. The code that renders
summaries has no access to the other files, so "identifiers never appear in
reports" is a structural property rather than a convention.

Baseline artifacts live under git-ignored `artifacts/` and must never be
committed. Real-data baselines require protected storage.

## Ground-truth separation

Ground-truth labels (scenario, malicious flag, campaign ID) are stored in a
separate `labels.parquet` file and are never merged into the canonical event
table. This prevents label leakage into feature computation and keeps the
canonical event log privacy-safe for contexts where labels should not be
accessible.

## Phase 4: detection artifacts

Detection consumes feature snapshots, which carry **no entity identifiers at
all** — the only key columns are `feature_schema_version`, `anchor_event_id`,
and `anchor_event_time`. A detection rule therefore has no path to a username,
a user, source, device, or session pseudonym, an IP address, or a coordinate,
and evidence cannot carry one even by accident. The evidence schema rejects any
value shaped like a UUID or a pseudonym as a second line of defence.

### The one protected column

| Artifact | Column | Sensitivity |
|---|---|---|
| `security_alerts.parquet` | `scope_value` | Pseudonymous operational metadata |
| `detection_entity_scope.parquet` (input) | `user_scope`, `source_scope` | Pseudonymous operational metadata |

Everything else Phase 4 writes — detections, risk assessments, quality reports,
evaluation reports, the manifest, validation findings, CLI summaries — is
aggregate or schema metadata.

### Structural confinement

`DetectionEngine` and `RiskScorer` accept no scope argument and import no scope
reader, so entity scope is **consumed only during alert construction**. A
signature test and an import-graph test both enforce it, and a further test
asserts that six named modules declare no parameter containing "scope".

`EntityScopeRecord.__repr__` and `EntityScopeTable.__repr__` are both redacted,
because a repr reaches log lines and tracebacks where no one is checking.

### Sanitized failure paths

Detection validation findings, manifest verification messages, and every CLI
error report codes, column names, and counts. A scope-table failure reports how
many anchors mismatched, never which. The CLI renders paths relative to the
working directory and falls back to a bare file name, so an absolute path under
a personal home directory never reaches a terminal.

## Phase 5: prediction artifacts (Milestone 8)

A prediction row carries the minimum technical identity a later evaluation needs
in order to join it to an outcome, and nothing else:

| Artifact | Columns | Sensitivity |
|---|---|---|
| `binary_predictions.parquet` | `anchor_event_id`, `anchor_event_time` | Join key |
| `category_predictions.parquet` | `anchor_event_id`, `anchor_event_time` | Join key |
| `anomaly_scores.parquet` | `anchor_event_id`, `anchor_event_time` | Join key |

An anchor identifier is a Phase 3 join key, not an entity identifier: feature
snapshots carry no user, source, device, or session pseudonym, so a prediction
built from them has no path to one. `PROHIBITED_PREDICTION_COLUMNS` names every
category a prediction table may not carry — pseudonyms, campaign identity, raw
IPs, credentials, coordinates, the raw feature vector, and every spelling of
ground truth — and an import-time guard refuses a row schema that declares one.

### The aggregate artifacts render no identifier at all

The `PredictionManifest`, the `MLValidationResult`, the `MLQualityReport` in both
its JSON and Markdown renderings, and every line the `ml predict`, `ml validate`,
and `ml profile` commands print carry counts, declared names, stable check codes,
and fingerprints. The join keys stay in the Parquet the next milestone will join
on. Privacy sweeps run over all of them, including the failure paths: a
validation failure reports how many rows contradicted their own threshold, never
which.

### No ground truth is read

Milestone 8 opens no label table. `ml predict` has no `--labels` option, the
inference loader has no label parameter, and the published artifacts carry no
label, no label fingerprint, and no outcome-dependent number — so there is
nothing for a prediction artifact to disclose about which events were attacks.

## Phase 5: explanation and drift artifacts (Milestone 10)

Two capabilities that describe a model and a population, and neither reads a
label at all.

### Row-level and aggregate are different contracts

The distinction is the whole design, and it is enforced structurally rather than
by convention:

| Artifact | May carry | Enforcement |
|---|---|---|
| `PredictionExplanation` | `anchor_event_id` — the same join key the prediction row it explains already carries | The one schema in the module permitted to declare it |
| `ExplanationQualityReport` | column names, magnitudes, counts | An import-time guard refuses a row identity on it |
| `ExplanationManifest` | fingerprints and counts | Same guard |
| `MLReferenceProfile` | partition cells, expected shares, fingerprints | Same guard |
| `FeatureDriftResult`, `PredictionDriftResult`, `MLDriftReport`, `DriftManifest` | statuses, supports, indices, rates | Same guard |

**No aggregate artifact in either module carries an anchor identifier**, and the
guard that enforces it runs at import: a schema that declared one would fail the
build rather than a review.

### A feature *name* is disclosed; a feature *value* is not

An attribution names a transformed column, which is an engineered feature name a
reviewer already admitted to the allowlist. It carries the *value* behind that
column only when `explain.include_feature_values` is explicitly turned on, and
the default is off — because a feature value can be a country code, and widening
what a report discloses should be a deliberate act somebody wrote down.

`explain.max_local_explanations` bounds how many row explanations are emitted at
all, and defaults to zero. Which rows are emitted is decided after sorting by
anchor, so the selection cannot depend on scoring order.

### The reference profile discloses a distribution, and the document does not

A reference profile necessarily contains the training population's expected
distribution — that is what makes it a baseline. It stays in the JSON artifact,
which is gitignored. The rendered Markdown carries the profile's *identity, its
lineage, and the shape* of each partition: how many cells, of what kind, at what
null rate. A table of two hundred numeric cells is not review material, and
rendering it would put the training distribution into a document that gets
pasted around.

### A category value is not a credential

A reviewed categorical feature records the authentication *method*, and one of
its values is `password`. That is a method name, not a credential, and it
appears in a reference profile's frozen vocabulary exactly as `mfa_push` or
`sso` does. The privacy sweep matches prohibited *field names* in key form for
this reason: a check that refused the string would refuse a legitimate,
reviewed, non-sensitive category value.

### No ground truth is read, and nothing is written back

Neither command has a `--labels` option and neither library takes a label
parameter. Both are terminal: no later command consumes their output as input to
a fit, and integration tests hash the whole artifact root either side of a run to
prove that no model, lock, ledger record, selection, publication, or evaluation
receipt changes by a byte.
