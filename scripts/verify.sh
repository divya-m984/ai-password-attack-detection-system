#!/usr/bin/env bash
# scripts/verify.sh — run all quality checks locally.
# Mirrors the GitHub Actions CI pipeline so failures are caught before pushing.
#
# Usage:
#   bash scripts/verify.sh          # from any directory
#   ./scripts/verify.sh             # from the repo root

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
uv build
