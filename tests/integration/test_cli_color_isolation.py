"""Regression tests for terminal-colour isolation in the test suite.

A developer with ``FORCE_COLOR`` exported in their shell once saw four CLI
tests fail: Rich wrote ANSI escapes into ``CliRunner``'s captured buffer, so
``json.loads(result.output)`` raised and literal substring assertions missed.
Nothing about the code under test was wrong.

The subtlety that made it hard to fix is worth stating, because it dictates
where the fix has to live.  ``rich.Console`` resolves its colour system **once,
in ``__init__``**, and caches it; only ``is_terminal`` re-reads the
environment.  The CLI builds its ``Console`` objects at import time, and pytest
imports test modules during collection -- so by the time any fixture runs, the
ambient ``FORCE_COLOR`` is already latched.  The scrub therefore happens in
``pytest_configure`` (see ``tests/conftest.py``), which is the last hook that
still runs ahead of collection.

These tests pin all three halves of that: the scrub happened, the suite really
does survive an inherited ``FORCE_COLOR``, and production colour behaviour was
not sacrificed to achieve it.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from tests.conftest import (
    COLOR_OVERRIDE_ENV_VARS,
    NEUTRAL_TERM,
    TTY_COMPATIBLE_OFF,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The file whose tests originally broke under an inherited ``FORCE_COLOR``.
_CLI_TEST_FILE = "tests/integration/test_data_cli.py"

#: A terminal type with real capabilities, for the tests that must observe
#: colour actually being produced.  ``TERM=dumb`` -- which the conftest sets --
#: short-circuits Rich's colour detection via ``is_dumb_terminal`` even when
#: ``force_terminal`` is true, so these tests have to opt back out of it.
_CAPABLE_TERM = "xterm-256color"


class TestEnvironmentIsScrubbed:
    """The scrub itself: no forcing variable survives into a test."""

    @pytest.mark.parametrize("name", COLOR_OVERRIDE_ENV_VARS)
    def test_forcing_variable_is_absent(self, name: str) -> None:
        assert name not in os.environ, (
            f"{name} leaked into the test environment; CLI output may carry "
            f"ANSI escapes and string assertions will be unreliable"
        )

    def test_term_is_neutral(self) -> None:
        assert os.environ.get("TERM") == NEUTRAL_TERM

    def test_tty_compatible_declares_not_a_terminal(self) -> None:
        # Set rather than removed: it is the only signal that survives being
        # imported while real stdout is still a terminal (``pytest -s``).
        name, value = TTY_COMPATIBLE_OFF
        assert os.environ.get(name) == value

    def test_the_removal_list_covers_the_known_offenders(self) -> None:
        # FORCE_COLOR caused the original defect; the rest are the other
        # conventions that can force colour on or off.
        for name in ("FORCE_COLOR", "NO_COLOR", "CLICOLOR", "CLICOLOR_FORCE"):
            assert name in COLOR_OVERRIDE_ENV_VARS

    def test_tty_compatible_is_not_also_in_the_removal_list(self) -> None:
        # Removing it would undo the setting above on every test.
        assert TTY_COMPATIBLE_OFF[0] not in COLOR_OVERRIDE_ENV_VARS


class TestConsoleEmitsPlainText:
    """A console built during a test writes no escape sequences."""

    def test_styled_output_carries_no_ansi(self) -> None:
        buffer = io.StringIO()
        Console(file=buffer).print("[red]styled[/red]")
        assert "\x1b[" not in buffer.getvalue()
        assert "styled" in buffer.getvalue()

    def test_cli_json_output_parses(self) -> None:
        import json

        from typer.testing import CliRunner

        from password_attack_detector.cli import app

        result = CliRunner().invoke(app, ["data", "schema", "--format", "json"])
        assert result.exit_code == 0
        # The assertion that originally failed: raises on any ANSI prefix.
        assert json.loads(result.output)["schema_version"] == "1.0.0"


class TestProductionColourIsPreserved:
    """The fix is scoped to the test environment, not to the CLI itself.

    A fix that simply disabled colour everywhere would make the tests pass and
    quietly degrade the tool. These assert the opposite: colour still works
    exactly when a user asks for it.
    """

    def test_force_color_still_produces_colour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(TTY_COMPATIBLE_OFF[0], raising=False)
        monkeypatch.setenv("FORCE_COLOR", "3")
        monkeypatch.setenv("TERM", _CAPABLE_TERM)
        buffer = io.StringIO()
        Console(file=buffer).print("[red]styled[/red]")
        assert "\x1b[" in buffer.getvalue(), (
            "FORCE_COLOR must still force colour; the test-suite fix must not "
            "change how the CLI behaves for a real user"
        )

    def test_a_real_terminal_still_produces_colour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(TTY_COMPATIBLE_OFF[0], raising=False)
        monkeypatch.setenv("TERM", _CAPABLE_TERM)
        buffer = io.StringIO()
        Console(file=buffer, force_terminal=True).print("[red]styled[/red]")
        assert "\x1b[" in buffer.getvalue()

    def test_no_color_still_suppresses_colour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(TTY_COMPATIBLE_OFF[0], raising=False)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TERM", _CAPABLE_TERM)
        buffer = io.StringIO()
        Console(file=buffer, force_terminal=True).print("[red]styled[/red]")
        assert "\x1b[" not in buffer.getvalue()


class TestSuiteSurvivesInheritedForceColor:
    """The end-to-end guarantee, proved the only way it can be.

    An in-process assertion cannot demonstrate this: the defect lived in what
    the environment looked like *at interpreter start*, before any fixture
    could run. So this runs a real pytest process with ``FORCE_COLOR=3``
    exported, exactly as an affected developer's shell would.
    """

    def test_cli_tests_pass_with_force_color_exported(self) -> None:
        environment = {**os.environ, "FORCE_COLOR": "3", "TERM": _CAPABLE_TERM}
        environment.pop(TTY_COMPATIBLE_OFF[0], None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                _CLI_TEST_FILE,
                "-q",
                "--no-cov",
                "-p",
                "no:cacheprovider",
            ],
            cwd=_REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "The CLI test suite must be stable when FORCE_COLOR is inherited "
            f"from the shell.\n--- stdout ---\n{result.stdout[-3000:]}"
        )

    def test_the_guard_would_catch_a_regression(self) -> None:
        # Confidence check on the test above: with the conftest scrub bypassed,
        # an inherited FORCE_COLOR really does corrupt captured CLI output.
        # If this ever stops holding, the regression test has gone toothless.
        script = (
            "import json, os, sys\n"
            "os.environ.pop('TTY_COMPATIBLE', None)\n"
            "os.environ['FORCE_COLOR'] = '3'\n"
            "os.environ['TERM'] = 'xterm-256color'\n"
            "from typer.testing import CliRunner\n"
            "from password_attack_detector.cli import app\n"
            "out = CliRunner().invoke(app, ['data','schema','--format','json']).output\n"
            "sys.exit(0 if '\\x1b[' in out else 1)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO_ROOT,
            env={**os.environ, "FORCE_COLOR": "3", "TERM": _CAPABLE_TERM},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "Expected an unscrubbed FORCE_COLOR to inject ANSI into captured "
            "CLI output; if it no longer does, the isolation fixture may be "
            "guarding a problem that has moved elsewhere"
        )
