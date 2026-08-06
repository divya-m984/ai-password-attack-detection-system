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
