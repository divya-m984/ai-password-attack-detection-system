"""Unit tests for password_attack_detector.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from password_attack_detector.config import Settings, load_settings
from password_attack_detector.exceptions import ConfigurationError


class TestLoadSettingsDefaults:
    def test_default_app_name(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert s.app_name == "Password Attack Detector"

    def test_default_environment(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert s.environment == "development"

    def test_default_debug_is_false(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        # Empty configs dir — no YAML overrides, so debug defaults to False.
        s = load_settings("production")
        assert s.debug is False

    def test_default_log_level(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("production")
        assert s.log_level == "INFO"

    def test_default_random_seed(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert s.random_seed == 42


class TestPathDefaults:
    def test_data_dir_points_to_fake_root(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert s.data_dir == fake_repo_root / "data"

    def test_artifacts_dir_points_to_fake_root(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert s.artifacts_dir == fake_repo_root / "artifacts"

    def test_models_dir_points_to_fake_root(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert s.models_dir == fake_repo_root / "models"

    def test_reports_dir_points_to_fake_root(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert s.reports_dir == fake_repo_root / "reports"


class TestEnvVarOverrides:
    def test_log_level_override(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        monkeypatch.setenv("PAD_LOG_LEVEL", "DEBUG")
        s = load_settings("development")
        assert s.log_level == "DEBUG"

    def test_log_level_normalised_to_uppercase(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        monkeypatch.setenv("PAD_LOG_LEVEL", "warning")
        s = load_settings("development")
        assert s.log_level == "WARNING"

    def test_debug_override(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        monkeypatch.setenv("PAD_DEBUG", "true")
        s = load_settings("development")
        assert s.debug is True

    def test_env_var_wins_over_yaml(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variable must take priority over YAML value."""
        _patch_repo_root(monkeypatch, fake_repo_root)
        # Write a YAML that says DEBUG
        _write_yaml(
            fake_repo_root / "configs" / "development.yaml", {"log_level": "DEBUG"}
        )
        # But env var says ERROR
        monkeypatch.setenv("PAD_LOG_LEVEL", "ERROR")
        s = load_settings("development")
        assert s.log_level == "ERROR"


class TestYamlLoading:
    def test_yaml_sets_log_level(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        _write_yaml(
            fake_repo_root / "configs" / "testing.yaml",
            {"log_level": "WARNING", "environment": "testing"},
        )
        s = load_settings("testing")
        assert s.log_level == "WARNING"

    def test_yaml_sets_debug(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        _write_yaml(
            fake_repo_root / "configs" / "development.yaml",
            {"debug": True},
        )
        s = load_settings("development")
        assert s.debug is True

    def test_missing_yaml_falls_back_to_defaults(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no YAML file exists the defaults must apply without error."""
        _patch_repo_root(monkeypatch, fake_repo_root)
        # configs/ directory is empty — no YAML files written
        s = load_settings("production")
        assert s.log_level == "INFO"


class TestValidation:
    def test_invalid_environment_raises_configuration_error(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        with pytest.raises(ConfigurationError, match="Invalid environment"):
            load_settings("staging")

    def test_invalid_log_level_raises_validation_error(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        monkeypatch.setenv("PAD_LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            load_settings("development")

    def test_settings_is_instance_of_settings_class(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        s = load_settings("development")
        assert isinstance(s, Settings)

    def test_environment_from_pad_env_var(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        monkeypatch.setenv("PAD_ENVIRONMENT", "testing")
        s = load_settings()  # no explicit environment arg
        assert s.environment == "testing"

    def test_invalid_environment_via_env_var(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_repo_root(monkeypatch, fake_repo_root)
        monkeypatch.setenv("PAD_ENVIRONMENT", "bad_env")
        with pytest.raises(ConfigurationError):
            load_settings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_repo_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Monkeypatch find_repo_root and get_configs_dir to use *root*."""
    import password_attack_detector.config as _cfg
    import password_attack_detector.paths as _paths

    monkeypatch.setattr(_paths, "find_repo_root", lambda start=None: root)
    monkeypatch.setattr(_cfg, "find_repo_root", lambda start=None: root)
    monkeypatch.setattr(_cfg, "get_configs_dir", lambda: root / "configs")


def _write_yaml(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))
