# Data Contract — Canonical Authentication Event

## Overview

The canonical authentication event is the central record type for the Password
Attack Detector. Every ingested or generated event must conform to this contract
before it can be stored, validated, or analysed.

## Schema version

`schema_version = "1.0.0"`

The version is stored in every Parquet file and every manifest. Increment it
when the field set changes in a backwards-incompatible way.

## Canonical event fields (AuthEvent)

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | `"1.0.0"` | yes | Literal version tag |
| `event_id` | UUID | yes | UUIDv5 (deterministic for synthetic, UUIDv4 otherwise) |
| `event_time` | datetime (UTC) | yes | Always normalised to UTC |
| `user_id` | str | yes | Prefixed pseudonym: `u:<hex>` |
| `source_id` | str | yes | Prefixed pseudonym: `s:<hex>` |
| `device_id` | str | yes | Prefixed pseudonym: `d:<hex>` |
| `session_id` | str | yes | Prefixed pseudonym: `sess:<hex>` |
| `application_id` | str | yes | Opaque application identifier |
| `authentication_method` | enum | yes | See AuthMethod below |
| `authentication_outcome` | enum | yes | See AuthOutcome below |
| `failure_reason` | enum\|null | conditional | Required when outcome is FAILURE or BLOCKED |
| `mfa_outcome` | enum\|null | no | See MFAOutcome below |
| `country_code` | str\|null | no | ISO 3166-1 alpha-2 |
| `region_code` | str\|null | no | Free-form region code |
| `coarse_latitude` | float\|null | no | Rounded to 1 decimal place |
| `coarse_longitude` | float\|null | no | Rounded to 1 decimal place |
| `user_agent_family` | str\|null | no | Browser/client family |
| `operating_system_family` | str\|null | no | OS family |
| `client_type` | enum\|null | no | See ClientType below |
| `response_time_ms` | int\|null | no | Must be ≥ 0 |

## Enum values

### AuthMethod
`password`, `mfa_totp`, `mfa_sms`, `mfa_email`, `sso`, `oauth2`,
`api_key`, `certificate`, `biometric`, `passkey`

### AuthOutcome
`success`, `failure`, `blocked`, `challenged`

### FailureReason
`invalid_credentials`, `account_locked`, `account_disabled`,
`account_not_found`, `mfa_failed`, `mfa_expired`, `token_expired`,
`ip_blocked`, `rate_limited`, `suspicious_activity`, `unknown`

### MFAOutcome
`passed`, `failed`, `bypassed`, `not_required`, `not_enrolled`

### ClientType
`web_browser`, `mobile_app`, `desktop_app`, `api_client`, `cli_tool`,
`bot`, `unknown`

## Prohibited fields

The following fields must **never** appear in `AuthEvent` or any Parquet file
stored by this system:

- Ground-truth columns: `campaign_id`, `scenario`, `malicious`,
  `supervised_training_eligible`, `generator_version`, `scenario_variant`,
  `campaign_stage`
- Sensitive credential fields (detected by `scan_prohibited_keys`):
  `password`, `passwd`, `secret`, `token`, `cookie`, `api_key_value`,
  `private_key`, `credential`, `auth_token`, `access_token`, `refresh_token`,
  `hash`, `salt`

## Ground-truth separation

Ground-truth labels are stored in a **separate** `labels.parquet` file and
joined to events only through `event_id`. Labels must never appear in the
canonical event table.

See `docs/synthetic-generation.md` for the nine scenario types and label schema.

## Parquet encoding

- Column order: `CANONICAL_EVENT_COLUMNS` (stable, defined in `serialization.py`)
- Timestamps: `pa.timestamp("us", tz="UTC")`
- Enums: stored as `pa.string()`
- UUIDs: stored as `pa.string()`
- Nulls: `nullable=True` for optional fields

## Content fingerprint

`compute_events_fingerprint(events)` returns a SHA-256 hex digest that:

- Is order-independent (events sorted by `str(event_id)` before hashing)
- Is independent of Parquet byte encoding
- Changes whenever any event field changes

The fingerprint is stored in `manifest.json` for dataset verification.
