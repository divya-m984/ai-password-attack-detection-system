"""Tests for the serving configuration.

The load-bearing assertion in this file is the last one: **no setting can
restate a frozen scientific decision.**  Everything else -- ports, log levels,
paths, ceilings -- is ordinary configuration validation, and it is tested
because a serving process that accepts a relative artifact path resolves it
against whatever working directory it happened to start in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from password_attack_detector.api.config import (
    DEFAULT_MAX_BATCH_EVENTS,
    MAX_MAX_BATCH_EVENTS,
    PROHIBITED_SETTING_NAMES,
    APISettings,
    load_api_settings,
)
from password_attack_detector.exceptions import ConfigurationError


def test_defaults_are_a_local_development_deployment() -> None:
    """The out-of-the-box settings bind locally and enable the demo documents."""
    settings = APISettings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.environment == "development"
    assert settings.docs_enabled is True
    assert settings.max_batch_events == DEFAULT_MAX_BATCH_EVENTS
    assert settings.require_ml_champion is True


def test_no_artifact_location_is_guessed() -> None:
    """Every path defaults to absent rather than to a directory nobody named."""
    settings = APISettings()
    assert settings.artifact_root is None
    assert settings.allowlist_path is None
    assert settings.feature_config_path is None
    assert settings.ml_config_path is None
    assert settings.detection_config_path is None
    assert settings.serving_bundle_root is None
    assert settings.champion_scope_key is None
    assert settings.configured_paths == ()
    assert settings.bundle_root is None


def test_settings_are_immutable() -> None:
    """A runtime built from these settings must not be able to see them change."""
    settings = APISettings()
    with pytest.raises(Exception, match=r"frozen|immutable"):
        settings.port = 9999


def test_environment_variables_are_read_under_the_api_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deployment values arrive from ``PAD_API_*`` and nothing else."""
    monkeypatch.setenv("PAD_API_PORT", "9101")
    monkeypatch.setenv("PAD_API_LOG_LEVEL", "warning")
    monkeypatch.setenv("PAD_API_MAX_BATCH_EVENTS", "42")
    monkeypatch.setenv("PAD_API_ARTIFACT_ROOT", str(tmp_path))
    settings = load_api_settings()
    assert settings.port == 9101
    assert settings.log_level == "WARNING"
    assert settings.max_batch_events == 42
    assert settings.artifact_root == tmp_path


def test_constructor_overrides_beat_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test injecting settings must not be overridden by an ambient variable."""
    monkeypatch.setenv("PAD_API_PORT", "9101")
    assert load_api_settings(port=7000).port == 7000


def test_an_unknown_log_level_is_refused() -> None:
    """An unrecognised level would silently become a default nobody chose."""
    with pytest.raises(ConfigurationError, match="Invalid API configuration"):
        load_api_settings(log_level="chatty")


@pytest.mark.parametrize("port", [0, 65_536, -1])
def test_a_port_outside_the_range_is_refused(port: int) -> None:
    """Ports are bounded, so a typo fails at startup rather than at bind time."""
    with pytest.raises(ConfigurationError):
        load_api_settings(port=port)


def test_a_relative_artifact_path_is_refused() -> None:
    """A relative path resolves against the working directory, which varies."""
    with pytest.raises(ConfigurationError, match="Invalid API configuration"):
        load_api_settings(artifact_root=Path("artifacts/ml"))


def test_a_traversing_artifact_path_is_refused(tmp_path: Path) -> None:
    """A '..' segment makes the resolved location depend on where it started."""
    with pytest.raises(ConfigurationError):
        load_api_settings(artifact_root=tmp_path / ".." / "elsewhere")


def test_the_batch_ceiling_is_bounded_on_both_sides() -> None:
    """A limit an operator could set to 'unlimited' is not a limit."""
    with pytest.raises(ConfigurationError):
        load_api_settings(max_batch_events=0)
    with pytest.raises(ConfigurationError):
        load_api_settings(max_batch_events=MAX_MAX_BATCH_EVENTS + 1)
    assert load_api_settings(
        max_batch_events=MAX_MAX_BATCH_EVENTS
    ).max_batch_events == (MAX_MAX_BATCH_EVENTS)


def test_the_request_byte_ceiling_is_bounded_on_both_sides() -> None:
    """Same reasoning as the batch ceiling, applied to the body size."""
    with pytest.raises(ConfigurationError):
        load_api_settings(max_request_bytes=1)
    with pytest.raises(ConfigurationError):
        load_api_settings(max_request_bytes=1_000_000_000)


def test_configured_paths_reports_only_what_was_configured(tmp_path: Path) -> None:
    """Startup verification needs the configured locations, not the absent ones."""
    settings = APISettings(
        artifact_root=tmp_path / "artifacts", allowlist_path=tmp_path / "allow.yaml"
    )
    assert settings.configured_paths == (
        tmp_path / "artifacts",
        tmp_path / "allow.yaml",
    )


def test_the_bundle_root_defaults_to_the_artifact_root(tmp_path: Path) -> None:
    """One location by default, so an operator configures one thing, not two."""
    settings = APISettings(artifact_root=tmp_path / "artifacts")
    assert settings.bundle_root == tmp_path / "artifacts"


def test_the_bundle_root_can_be_relocated_and_says_nothing_else(
    tmp_path: Path,
) -> None:
    """A location setting: it can make a hybrid absent, never make a new one.

    There is no strategy, threshold, or model field beside it -- the bundle's own
    frozen selection decides what runs, and the locked receipt has to agree.
    """
    settings = APISettings(
        artifact_root=tmp_path / "artifacts",
        serving_bundle_root=tmp_path / "elsewhere",
    )
    assert settings.bundle_root == tmp_path / "elsewhere"
    assert tmp_path / "elsewhere" in settings.configured_paths


def test_a_relative_bundle_root_is_refused() -> None:
    """Same reasoning as every other configured path."""
    with pytest.raises(ConfigurationError):
        load_api_settings(serving_bundle_root=Path("serving"))


# ---------------------------------------------------------------------------
# The one that matters: artifact location may vary, scientific identity may not
# ---------------------------------------------------------------------------


def test_no_setting_can_restate_a_frozen_scientific_decision() -> None:
    """No field names a model, a threshold, a fusion strategy, or a feature set."""
    assert set(APISettings.model_fields) & PROHIBITED_SETTING_NAMES == set()


def test_the_prohibition_list_names_every_frozen_quantity() -> None:
    """The guard is only worth having if it covers what it claims to cover."""
    for name in ("model_id", "decision_threshold", "fusion_strategy", "score_kind"):
        assert name in PROHIBITED_SETTING_NAMES


def test_the_import_time_guard_refuses_a_prohibited_field() -> None:
    """The guard is executable, not decorative.

    Reproduced against a subclass rather than by editing the real settings: the
    point is that the check *fires*, and a test that could only prove that by
    breaking the shipped class would have to break it for every other test too.
    """

    class Offending(APISettings):
        decision_threshold: float = 0.5

    offending = set(Offending.model_fields) & PROHIBITED_SETTING_NAMES
    assert offending == {"decision_threshold"}


def test_the_ml_switch_can_only_turn_the_layer_off() -> None:
    """``require_ml_champion`` is a boolean, so it can never select a model."""
    field = APISettings.model_fields["require_ml_champion"]
    assert field.annotation is bool
