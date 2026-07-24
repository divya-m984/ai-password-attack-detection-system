"""Shared pytest fixtures for the Password Attack Detector test suite."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_paths_cache() -> Generator[None, None, None]:
    """Clear the module-level repo-root cache before and after every test.

    This prevents cross-test pollution when individual tests call
    find_repo_root() with a custom *start* path that differs from the real
    repository root.
    """
    import password_attack_detector.paths as _paths

    _paths.reset_repo_root_cache()
    yield
    _paths.reset_repo_root_cache()


@pytest.fixture(autouse=True)
def reset_logging_state() -> Generator[None, None, None]:
    """Reset the structlog configuration guard before and after every test."""
    import password_attack_detector.logging_config as _lc

    _lc.reset_logging()
    yield
    _lc.reset_logging()


@pytest.fixture()
def fake_repo_root(tmp_path: Path) -> Path:
    """Return a temporary directory that looks like a minimal project root.

    The directory contains:
    - ``pyproject.toml`` (minimal, enough for find_repo_root to accept it)
    - ``configs/``
    - ``data/``
    - ``artifacts/``
    - ``models/``
    - ``reports/``
    """
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test-project"\n')
    for sub in ("configs", "data", "artifacts", "models", "reports"):
        (tmp_path / sub).mkdir()
    return tmp_path
