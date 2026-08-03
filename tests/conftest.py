"""Shared pytest fixtures for the Password Attack Detector test suite."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Final

import pytest

#: Variables that override a library's own terminal detection by forcing
#: colour on or off.  These are removed so that detection simply runs.
#:
#: ``FORCE_COLOR`` and ``NO_COLOR`` are the cross-tool conventions Rich reads
#: directly; ``CLICOLOR`` and ``CLICOLOR_FORCE`` are the older de-facto pair
#: other tools still honour; ``TTY_INTERACTIVE`` is Rich-specific.  Any one of
#: them can put ANSI escapes into a captured, non-terminal buffer.
COLOR_OVERRIDE_ENV_VARS: Final[tuple[str, ...]] = (
    "FORCE_COLOR",
    "NO_COLOR",
    "CLICOLOR",
    "CLICOLOR_FORCE",
    "TTY_INTERACTIVE",
)

#: Rich's explicit "this device is not a terminal" signal.  It is checked
#: before ``FORCE_COLOR`` and before any ``isatty()`` probe, which makes it the
#: one setting that holds even when the code under test is imported while real
#: stdout is still attached to a terminal (``pytest -s``).
TTY_COMPATIBLE_OFF: Final[tuple[str, str]] = ("TTY_COMPATIBLE", "0")

#: Terminal type advertised to the code under test.  ``dumb`` is the
#: conventional way to say "this device understands no escape sequences".
#: Applied per-test only -- see :func:`neutral_terminal_environment`.
NEUTRAL_TERM: Final[str] = "dumb"


def pytest_configure(config: pytest.Config) -> None:
    """Neutralise ambient colour forcing before any test module is imported.

    This has to happen here rather than in a fixture, for a specific reason.

    ``rich.Console`` resolves its colour system **once, in ``__init__``**, via
    ``_detect_color_system()``, and caches it; only ``is_terminal`` re-reads
    the environment afterwards.  The CLI modules build their ``Console``
    objects at import time, and pytest imports test modules -- and therefore
    the CLI -- during collection.  So a developer with ``FORCE_COLOR``
    exported has that value latched into ``Console._color_system`` before the
    first fixture ever runs, and no amount of per-test ``monkeypatch`` can
    undo it.  ``pytest_configure`` runs after conftest is loaded but before
    collection: the last moment still ahead of that import.

    Two things are done, and the split matters:

    * The forcing variables are **removed**, so the library's own detection
      runs normally.  Under ``CliRunner`` that means a non-terminal buffer and
      plain text -- exactly the code path a piped invocation takes.
    * ``TTY_COMPATIBLE=0`` is **set**, because removal alone is not enough
      under ``pytest -s``, where stdout is still a real terminal at import
      time and detection legitimately finds one.

    ``TERM`` is deliberately *not* touched here.  Forcing ``TERM=dumb``
    process-wide would also strip the colour from pytest's own reporter for
    every developer, and ``TTY_COMPATIBLE`` already covers the case.

    Production colour behaviour is untouched: a real terminal, or a user who
    exports ``FORCE_COLOR`` deliberately, still gets colour.
    """
    for name in COLOR_OVERRIDE_ENV_VARS:
        os.environ.pop(name, None)
    name, value = TTY_COMPATIBLE_OFF
    os.environ[name] = value


@pytest.fixture(autouse=True)
def neutral_terminal_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Keep the terminal environment neutral for the duration of each test.

    :func:`pytest_configure` handles the import-time latch described above.
    This fixture covers the remaining window: a ``Console`` constructed *while
    a test runs*, and any test that sets one of these variables itself and
    would otherwise leak it into the next test.

    ``TERM`` is normalised here rather than in ``pytest_configure`` because
    ``monkeypatch`` restores it between tests, so pytest's own reporter keeps
    its colour while the code under test never sees a capable terminal.

    It is deliberately autouse and repository-wide.  The failure it guards
    against is ambient and silent, and scoping it per-file would leave every
    future CLI test one forgotten decorator away from the same defect.
    """
    for name in COLOR_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(*TTY_COMPATIBLE_OFF)
    monkeypatch.setenv("TERM", NEUTRAL_TERM)
    yield


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
    # Phase 2 data subdirectories
    for data_sub in ("raw", "interim", "processed", "external"):
        (tmp_path / "data" / data_sub).mkdir()
    return tmp_path
