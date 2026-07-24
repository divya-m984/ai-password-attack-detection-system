"""Centralized path management for the Password Attack Detector.

All path helpers are pure computations — no directories are created on import.
Use :func:`ensure_dir` explicitly whenever a directory must exist on disk.
"""

from __future__ import annotations

from pathlib import Path

from password_attack_detector.exceptions import ConfigurationError

__all__ = [
    "ensure_dir",
    "find_repo_root",
    "get_artifacts_dir",
    "get_configs_dir",
    "get_data_dir",
    "get_models_dir",
    "get_reports_dir",
    "reset_repo_root_cache",
]

# Module-level cache — cleared by reset_repo_root_cache() in tests.
_repo_root: Path | None = None


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up the directory tree from *start* looking for ``pyproject.toml``.

    Parameters
    ----------
    start:
        Directory from which to begin the search.  Defaults to the directory
        containing this file.  Pass an explicit path in tests to avoid relying
        on the module-level cache with a real repo root.

    Returns
    -------
    Path
        The first ancestor directory that contains ``pyproject.toml``.

    Raises
    ------
    ConfigurationError
        If no ``pyproject.toml`` is found before reaching the filesystem root.
    """
    global _repo_root

    # If a start override is provided, always perform a fresh search.
    if start is not None:
        return _search_for_root(start)

    # Use the cached value when no override is given.
    if _repo_root is None:
        _repo_root = _search_for_root(Path(__file__).resolve().parent)
    return _repo_root


def _search_for_root(start: Path) -> Path:
    """Perform the actual upward directory walk."""
    current = start.resolve()
    while True:
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            raise ConfigurationError(
                "Could not find the repository root: no pyproject.toml found "
                f"in {start!r} or any of its parent directories."
            )
        current = parent


def reset_repo_root_cache() -> None:
    """Clear the cached repository root.

    Call this in test fixtures to prevent cross-test pollution when tests
    inject different repository roots via the *start* parameter.
    """
    global _repo_root
    _repo_root = None


def get_data_dir() -> Path:
    """Return the project data directory (``<repo>/data``)."""
    return find_repo_root() / "data"


def get_artifacts_dir() -> Path:
    """Return the project artifacts directory (``<repo>/artifacts``)."""
    return find_repo_root() / "artifacts"


def get_models_dir() -> Path:
    """Return the project models directory (``<repo>/models``)."""
    return find_repo_root() / "models"


def get_reports_dir() -> Path:
    """Return the project reports directory (``<repo>/reports``)."""
    return find_repo_root() / "reports"


def get_configs_dir() -> Path:
    """Return the project configs directory (``<repo>/configs``)."""
    return find_repo_root() / "configs"


def ensure_dir(path: Path) -> Path:
    """Create *path* (and any missing parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The same *path* that was passed in.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
