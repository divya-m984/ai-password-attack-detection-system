"""Integration tests for the password-attack-detector CLI."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from password_attack_detector import __version__
from password_attack_detector.cli import app

runner = CliRunner()


class TestVersionCommand:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_prints_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert __version__ in result.output

    def test_output_contains_package_name(self) -> None:
        result = runner.invoke(app, ["version"])
        assert "password-attack-detector" in result.output


class TestHelpCommand:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_lists_all_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "version" in result.output
        assert "doctor" in result.output
        assert "show-config" in result.output


class TestDoctorCommand:
    def test_exits_zero_clean_state(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0

    def test_output_contains_pass(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        result = runner.invoke(app, ["doctor"])
        assert "PASS" in result.output

    def test_reports_python_version(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        result = runner.invoke(app, ["doctor"])
        assert "Python" in result.output

    def test_exits_one_when_data_dir_missing(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        # Remove the data directory so the check fails.
        import shutil

        shutil.rmtree(fake_repo_root / "data")
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestShowConfigCommand:
    def test_exits_zero(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        result = runner.invoke(app, ["show-config"])
        assert result.exit_code == 0

    def test_output_contains_environment_field(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        result = runner.invoke(app, ["show-config"])
        assert "environment" in result.output

    def test_output_contains_log_level(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        result = runner.invoke(app, ["show-config"])
        assert "log_level" in result.output

    def test_secret_str_values_are_redacted(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SecretStr field values must never appear in show-config output."""
        from pydantic import SecretStr

        from password_attack_detector.config import is_secret_field

        # Verify the helper correctly identifies SecretStr.
        secret = SecretStr("super-secret-xyz")
        assert is_secret_field(secret) is True
        assert is_secret_field("plain-string") is False
        assert is_secret_field(42) is False

    def test_show_config_does_not_expose_raw_secret_str(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """show-config must not print the raw value of any SecretStr field."""
        _patch_repo_root(monkeypatch, fake_repo_root)

        # Patch load_settings (accessed inside show_config via deferred import)
        # by patching the source module it imports from.
        from pydantic import SecretStr

        import password_attack_detector.config as cfg_module

        secret_value = "super-secret-value-xyz"

        class _FakeSettings:
            model_fields: ClassVar[dict[str, None]] = {
                "environment": None,
                "api_key": None,
            }
            environment = "development"
            api_key = SecretStr(secret_value)

        monkeypatch.setattr(
            cfg_module, "load_settings", lambda env=None: _FakeSettings()
        )

        result = runner.invoke(app, ["show-config"])

        assert secret_value not in result.output
        assert "REDACTED" in result.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_repo_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    import password_attack_detector.config as _cfg
    import password_attack_detector.paths as _paths

    monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: root)
    monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: root)
    monkeypatch.setattr(_cfg, "get_configs_dir", lambda: root / "configs")
