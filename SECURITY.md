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
  merged into the authentication-event table.
- Synthetic data uses randomly generated pseudonym-format identifiers and never
  calls `PseudonymService`.
- Quality reports and manifests contain only aggregate statistics and column
  names — never raw event values or pseudonym strings.

See `docs/privacy-model.md` for the full privacy model.

## Environment file policy

The `.env` file is excluded from version control via `.gitignore`. The
`.env.example` file committed to the repository must not contain real values —
only safe placeholder examples.
