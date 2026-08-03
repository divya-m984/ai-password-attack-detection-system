# Data Dictionary

## events.parquet

The canonical authentication-event table. One row per authentication attempt.

| Column | Parquet type | Description |
|---|---|---|
| `schema_version` | STRING | Schema version string, always `"1.0.0"` |
| `event_id` | STRING | UUID string identifying this event uniquely |
| `event_time` | TIMESTAMP(µs, UTC) | When the authentication attempt occurred |
| `user_id` | STRING | HMAC-SHA256 pseudonym of the source user identifier |
| `source_id` | STRING | HMAC-SHA256 pseudonym of the source IP or hostname |
| `device_id` | STRING | HMAC-SHA256 pseudonym of the device identifier |
| `session_id` | STRING | HMAC-SHA256 pseudonym of the session identifier |
| `application_id` | STRING | Application that received the authentication request |
| `authentication_method` | STRING | Authentication mechanism used (see AuthMethod enum) |
| `authentication_outcome` | STRING | Result of the attempt (see AuthOutcome enum) |
| `failure_reason` | STRING (nullable) | Reason code when outcome is failure or blocked |
| `mfa_outcome` | STRING (nullable) | Multi-factor authentication result |
| `country_code` | STRING (nullable) | ISO 3166-1 alpha-2 country code |
| `region_code` | STRING (nullable) | Region or state code |
| `coarse_latitude` | DOUBLE (nullable) | Latitude rounded to 1 decimal place |
| `coarse_longitude` | DOUBLE (nullable) | Longitude rounded to 1 decimal place |
| `user_agent_family` | STRING (nullable) | Browser or client family (no version numbers) |
| `operating_system_family` | STRING (nullable) | OS family (no version numbers) |
| `client_type` | STRING (nullable) | Type of client (see ClientType enum) |
| `response_time_ms` | INT64 (nullable) | Round-trip response time in milliseconds |

## labels.parquet

The ground-truth label table. One row per event, joined to events by `event_id`.
This table is produced only for synthetic data and is never merged into
`events.parquet`.

| Column | Parquet type | Description |
|---|---|---|
| `event_id` | STRING | Foreign key to `events.parquet.event_id` |
| `campaign_id` | STRING | Identifier of the attack campaign (or normal-traffic group) |
| `scenario` | STRING | Attack scenario type (see ScenarioType enum) |
| `malicious` | BOOLEAN | True when this event is part of an attack |
| `supervised_training_eligible` | BOOLEAN | False for novel-anomaly holdout data |
| `generator_version` | STRING | Synthetic generator version that produced this row |
| `scenario_variant` | STRING (nullable) | Scenario sub-type for disambiguation |
| `campaign_stage` | STRING (nullable) | Phase of the attack campaign |

## manifest.json

The dataset integrity manifest. Written last during publication so its presence
signals a complete, coherent dataset.

| Field | Type | Description |
|---|---|---|
| `manifest_version` | string | Manifest schema version (`"1.0.0"`) |
| `dataset_id` | string | UUIDv5 derived from content fingerprint |
| `schema_version` | string | AuthEvent schema version |
| `source_type` | string | `"synthetic"` or `"ingested"` |
| `row_count` | integer | Number of rows in `events.parquet` |
| `ground_truth_row_count` | integer\|null | Number of rows in `labels.parquet` |
| `earliest_event_time` | string\|null | ISO-8601 UTC timestamp of oldest event |
| `latest_event_time` | string\|null | ISO-8601 UTC timestamp of newest event |
| `artifacts` | array | One entry per artifact file with relative path and SHA-256 |
| `canonical_schema_fingerprint` | string | SHA-256 of `AuthEvent` field names |
| `content_fingerprint` | string | Order-independent SHA-256 of event content |
| `config_fingerprint` | string\|null | SHA-256 of synthetic config (null for ingested) |
| `validation_status` | string | `"valid"`, `"warning"`, or `"invalid"` |
| `created_at` | string | ISO-8601 UTC timestamp when manifest was written |
| `reproducibility` | object | Runtime environment versions and seed |

### manifest.json — reproducibility sub-object

| Field | Description |
|---|---|
| `python_version` | Python version used to generate the dataset |
| `numpy_version` | NumPy version |
| `pandas_version` | pandas version |
| `pyarrow_version` | PyArrow version |
| `uv_lock_sha256` | SHA-256 of `uv.lock` for environment pinning |
| `generator_version` | Synthetic generator version (null for ingested) |
| `seed` | Random seed (null for ingested) |

## quality-report.json

JSON quality profile produced by `generate_quality_report`. Contains aggregate
statistics only — no raw event values or identifiers.

Key fields: `row_count`, `schema_version`, `null_rates` (per-column null
fractions), `auth_method_distribution`, `auth_outcome_distribution`,
`duplicate_event_id_count`, `sensitive_fields_found`, `gt_leakage_columns_found`,
`validation_status`, `quality_warnings`.
