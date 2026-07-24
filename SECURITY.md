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

## Privacy principles for future authentication datasets

When authentication event data is introduced in later phases:

- Use only synthetic or properly anonymised datasets
- Never store usernames, IP addresses, or session identifiers in a form that
  allows re-identification of real individuals
- Document the provenance of any external dataset
- Apply data minimisation — collect only the fields required for detection

## Environment file policy

The `.env` file is excluded from version control via `.gitignore`. The
`.env.example` file committed to the repository must not contain real values —
only safe placeholder examples.
