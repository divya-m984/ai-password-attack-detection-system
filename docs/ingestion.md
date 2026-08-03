# Data Ingestion

## Overview

The ingestion layer reads real authentication-event data from CSV or JSONL files,
pseudonymizes source identifiers, validates each row against the canonical
`AuthEvent` schema, and writes canonical Parquet output.

## Requirements

- `PAD_PSEUDONYMIZATION_KEY` must be set in the environment or an untracked
  `.env` file before running any ingestion command.
- The input file must not contain prohibited sensitive fields (passwords, tokens,
  credentials).
- The source format must be UTF-8 encoded.

## Supported formats

### CSV

- Must have a header row.
- Header names are normalized: stripped, lowercased, hyphens and spaces replaced
  with underscores.
- Normalized header names that match canonical `AuthEvent` field names are
  mapped automatically.
- Non-canonical columns are silently ignored.
- Empty/null cell values (`""`, `"none"`, `"null"`, `"na"`, `"n/a"`) are
  converted to `None`.
- Use `--field-map SOURCE=CANONICAL` for headers that do not normalize correctly.

### JSONL

- One JSON object per line. Blank lines are skipped.
- Top-level and nested keys are inspected recursively for prohibited names
  before any values are read.
- Extra top-level keys are silently ignored.
- Nesting depth is limited to prevent abuse.

## Prohibited field enforcement

Before any row is processed, the complete header (CSV) or key set (JSONL) is
inspected against:

1. Sensitive credential field names (via `scan_prohibited_keys`): `password`,
   `passwd`, `secret`, `token`, `cookie`, `api_key_value`, `private_key`,
   `credential`, etc. — entire dataset is rejected.
2. Ground-truth leakage column names (`campaign_id`, `scenario`, `malicious`,
   etc.) — entire dataset is rejected.

Rejection means the entire file is refused with exit code 1 and nothing is
written.

## Invalid-row policies

`--policy fail` (default)
: Abort on the first invalid row. No events are written.

`--policy quarantine`
: Skip invalid rows, recording a privacy-safe `QuarantineEntry` for each.
Processing continues; only valid rows are written. If all rows are quarantined,
ingestion exits with code 1 (empty result).

`QuarantineEntry` fields never contain raw source values, identifiers, or
secrets. They record only: row number, stable error code, affected canonical
field name, and a sanitized message.

## Pseudonymization

Four identifier fields are pseudonymized before writing:

- `user_id` — domain `"user"`
- `source_id` — domain `"source"`
- `device_id` — domain `"device"`
- `session_id` — domain `"session"`

Each pseudonym is `HMAC-SHA256(key, domain + ":" + original_value)` encoded as
a prefixed hex string. The original value is never stored.

## Output

After successful ingestion, the output directory contains:

```
output_dir/
  events.parquet    canonical events in CANONICAL_EVENT_COLUMNS order
  events.jsonl      raw canonical events as newline-delimited JSON
  manifest.json     integrity manifest with SHA-256 checksums
```

The output is staged to a temporary directory and promoted atomically to prevent
partial writes.

## CLI usage

```bash
# Basic CSV ingestion
password-attack-detector data ingest events.csv \
  --output-dir data/ingested/2024-01

# JSONL with quarantine policy
password-attack-detector data ingest events.jsonl \
  --output-dir data/ingested/2024-01 \
  --policy quarantine

# Custom field mapping
password-attack-detector data ingest export.csv \
  --output-dir data/ingested/2024-01 \
  --field-map LoginUser=user_id \
  --field-map SourceIP=source_id

# Overwrite existing output
password-attack-detector data ingest events.csv \
  --output-dir data/ingested/2024-01 \
  --force
```

## Limitations

- Ingested data has no ground-truth labels. `labels.parquet` is not produced.
- Row-level validation uses the `AuthEvent` Pydantic model; invalid rows are
  quarantined or cause failure depending on policy.
- Time zone normalization: `event_time` must be timezone-aware (UTC or with
  offset). Naive datetimes are rejected.
- No deduplication: if the source file contains duplicate `event_id` values,
  they are written as-is and flagged by dataset validation.
