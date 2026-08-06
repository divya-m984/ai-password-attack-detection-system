# Security Policy

## Supported versions

This project is in active development. Only the latest commit on the `main`
branch is supported.

## Reporting a vulnerability

If you discover a security issue, please **do not** open a public GitHub issue.
Report it privately by emailing the repository maintainer (see the repository's
contact information).

Include:
- A clear description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested mitigations

You will receive an acknowledgement within 5 business days.

## Responsible disclosure

Please allow reasonable time for the issue to be assessed and patched before
any public disclosure. This project follows a coordinated disclosure approach.

## Prohibited content

The following must never appear in this repository, in any branch or commit:

- Plaintext passwords or authentication credentials
- Password hashes
- Private keys, API tokens, or secrets of any kind
- Real user authentication data (even anonymised unless vetted)
- Code that automates login attempts against real services
- Credential-cracking utilities or wordlists
- Offensive exploitation modules or payloads
- Supply-chain attacks or dependency confusion packages

Any configuration fields that hold sensitive values must use `pydantic.SecretStr`
and must be redacted in all CLI output and log files.

## Privacy principles for authentication datasets (Phase 2+)

Phase 2 introduces a data engineering layer with the following guarantees:

- No plaintext passwords, password hashes, authentication tokens, cookies,
  or real credentials are stored at any layer.
- Source identifiers (user, source IP, device, session) are pseudonymized via
  HMAC-SHA256 (`PseudonymService`) before any data is written to disk.
  **Pseudonymization reduces exposure but does not guarantee anonymity.**
  The pseudonymization key (`PAD_PSEUDONYMIZATION_KEY`) must be kept secret.
- Prohibited sensitive field names are detected and rejected at the ingestion
  header/key level before any values are read. The entire dataset is rejected,
  not just the offending rows.
- Ground-truth labels are stored separately from canonical events and are never
  merged into the authentication-event table. The same separation holds for
  feature snapshots: `feature_snapshots.parquet` contains model inputs only,
  with labels and split assignments in their own tables joined by `event_id`.
  Campaign metadata is never published in any table.
- Fitted behavioral baselines hold pseudonymous per-entity state and are
  therefore sensitive operational metadata. They are written only to
  git-ignored `artifacts/` paths, with the pseudonym-bearing Parquet tables at
  mode 0600 and a separate metadata-only `baseline.json` that reports and CLI
  output read instead. Baseline state must never be committed, and real-data
  baselines require protected storage with access control and retention limits
  appropriate to the underlying authentication logs.
- Feature validation findings and leakage audit results report column names,
  error codes, and counts only. They never include an event identifier, a
  pseudonym, a coordinate, or a raw row, so they are safe to place in a build
  log or embed in a manifest.
- Synthetic data uses randomly generated pseudonym-format identifiers and never
  calls `PseudonymService`.
- Quality reports and manifests contain only aggregate statistics and column
  names — never raw event values or pseudonym strings. This applies equally to
  the Phase 3 feature quality report, leakage audit, split manifest, and
  feature manifest.

### Phase 4 detection artifacts

- Detection artifacts are git-ignored and unaudited. They are generated output,
  not reviewed content.
- **`security_alerts.parquet`'s `scope_value` column is the single field in the
  whole detection layer that may carry a pseudonym.** It is populated only when
  an optional entity-scope table is supplied, is classified sensitive
  operational metadata, and is excluded from evidence, reason codes, every
  report, every CLI summary, the detection manifest, and every validation
  message. Real-data alert artifacts require protected storage with access
  control and retention limits appropriate to the underlying authentication
  logs.
- The optional `detection_entity_scope.parquet` input carries pseudonyms in two
  columns. It is consumed **only during alert construction**;
  `DetectionEngine` and `RiskScorer` accept no scope argument and import no
  reader for it, so the confinement is a type error rather than a convention.
  `EntityScopeRecord` and `EntityScopeTable` both redact their `__repr__`,
  because a repr reaches logs and tracebacks.
- Detection evidence carries no identifier by construction: a feature snapshot
  contains no username, entity identifier, IP address, or coordinate, and the
  evidence schema additionally rejects any value shaped like a UUID or a
  pseudonym. Evidence messages come from frozen catalog templates; no
  caller-supplied string reaches one.
- Detection validation findings report codes, column names, and counts only —
  never an event, detection, or alert identifier, a scope value, an evidence
  value, a raw row, or an absolute path. They are safe to place in a build log.
- Detection quality reports, evaluation reports, and the detection manifest
  contain aggregate statistics and fingerprints only. The manifest records the
  alert grouping **mode name and a boolean**, never a scope value.
- Detection performs no authentication, generates no authentication traffic,
  and takes no response or blocking action.

See `docs/privacy-model.md` for the full privacy model.

## Environment file policy

The `.env` file is excluded from version control via `.gitignore`. The
`.env.example` file committed to the repository must not contain real values —
only safe placeholder examples.
