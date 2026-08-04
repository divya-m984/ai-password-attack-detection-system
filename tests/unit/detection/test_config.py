"""Tests for the typed detection configuration."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import (
    DETECTION_FINGERPRINT_EXCLUDED_FIELDS,
    PROHIBITED_CONFIG_KEY_TOKENS,
    AlertingConfig,
    DetectionConfig,
    RuleSettings,
    ScoringConfig,
    SeverityThresholds,
    SignalConfig,
    load_detection_config,
)
from password_attack_detector.detection.enums import (
    CorrelationGroup,
    RuleFamily,
    ScopeKind,
    Severity,
)
from password_attack_detector.exceptions import (
    ConfigurationError,
    DetectionConfigurationError,
)

SHIPPED_CONFIGS = ("rules-testing.yaml", "rules-development.yaml")


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "detection"


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "detection.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Valid configuration
# ---------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
    config = DetectionConfig()
    assert config.enabled_rule_ids == RULE_CATALOG.rule_ids
    assert config.detection_schema_version == "1.0.0"
    assert config.insufficient_history_policy == "report"


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_shipped_configs_load(name: str) -> None:
    config = load_detection_config(_config_dir() / name)
    assert len(config.enabled_rule_ids) == len(RULE_CATALOG)
    assert set(config.enabled_rule_ids) == set(RULE_CATALOG.rule_ids)


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_shipped_configs_reach_the_low_alert_band(name: str) -> None:
    """Both shipped configurations must be able to emit a LOW alert.

    LOW is an ordinary alert severity. A shipped default that silently made it
    unreachable would be a behaviour change disguised as a threshold.
    """
    config = load_detection_config(_config_dir() / name)
    assert config.alerting.min_alert_severity is Severity.LOW
    assert config.low_alert_reachable is True


def test_empty_yaml_yields_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_detection_config(path).enabled_rule_ids == RULE_CATALOG.rule_ids


def test_parameters_for_merges_defaults_with_overrides() -> None:
    config = DetectionConfig(
        rules={"PAD-BF-001": RuleSettings(parameters={"min_pair_failures": 12})}
    )
    parameters = config.parameters_for("PAD-BF-001")
    assert parameters["min_pair_failures"] == 12
    # Untouched parameters keep their declared defaults.
    assert parameters["window"] == "5m"
    assert parameters["max_source_unique_users"] == 3


def test_severity_override_is_honoured() -> None:
    config = DetectionConfig(
        rules={"PAD-BOT-001": RuleSettings(severity=Severity.CRITICAL)}
    )
    assert config.severity_for_rule("PAD-BOT-001") is Severity.CRITICAL
    assert config.severity_for_rule("PAD-BF-001") is Severity.HIGH


def test_weight_for_known_and_unknown_family() -> None:
    config = DetectionConfig()
    assert config.weight_for(RuleFamily.BRUTE_FORCE) == pytest.approx(0.90)

    partial = DetectionConfig(
        enabled_rule_ids=("PAD-BOT-001",),
        family_weights={RuleFamily.AUTOMATION: 0.5},
    )
    with pytest.raises(DetectionConfigurationError, match="No configured weight"):
        partial.weight_for(RuleFamily.LOCATION)


def test_is_enabled() -> None:
    config = DetectionConfig(enabled_rule_ids=("PAD-BF-001",))
    assert config.is_enabled("PAD-BF-001")
    assert not config.is_enabled("PAD-GEO-001")


# ---------------------------------------------------------------------------
# Rejected configuration
# ---------------------------------------------------------------------------


def test_unknown_rule_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unregistered rule"):
        DetectionConfig(enabled_rule_ids=("PAD-BF-001", "PAD-XX-999"))


def test_duplicate_rule_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="repeats rule"):
        DetectionConfig(enabled_rule_ids=("PAD-BF-001", "PAD-BF-001"))


def test_empty_enabled_rule_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one rule"):
        DetectionConfig(enabled_rule_ids=())


def test_unknown_rule_key_in_overrides_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unregistered rule"):
        DetectionConfig(rules={"PAD-XX-999": RuleSettings()})


def test_undeclared_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError, match="does not declare parameter"):
        DetectionConfig(
            rules={"PAD-BF-001": RuleSettings(parameters={"nonexistent": 1})}
        )


@pytest.mark.parametrize(
    ("rule_id", "parameters", "match"),
    [
        ("PAD-BF-001", {"min_pair_failure_rate": 1.5}, "at most"),
        ("PAD-BF-001", {"min_pair_failures": 0}, "at least"),
        ("PAD-BF-001", {"min_pair_failures": 2.5}, "must be an integer"),
        ("PAD-BF-001", {"window": "not-a-duration"}, "Invalid duration"),
        ("PAD-GEO-001", {"zero_elapsed_policy": "ignore"}, "must be one of"),
        ("PAD-GEO-001", {"require_country_change": "yes"}, "must be a boolean"),
    ],
)
def test_invalid_thresholds_are_rejected(
    rule_id: str, parameters: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        DetectionConfig(rules={rule_id: RuleSettings(parameters=parameters)})


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.5])
def test_invalid_family_weight_is_rejected(weight: float) -> None:
    with pytest.raises(ValidationError, match=r"must be in \(0, 1\]"):
        DetectionConfig(family_weights={RuleFamily.BRUTE_FORCE: weight})


def test_missing_family_weight_for_enabled_rule_is_rejected() -> None:
    with pytest.raises(ValidationError, match="does not cover enabled family"):
        DetectionConfig(family_weights={RuleFamily.AUTOMATION: 0.5})


@pytest.mark.parametrize(
    "thresholds",
    [
        {"medium": 65.0, "high": 40.0, "critical": 85.0},
        {"medium": 40.0, "high": 40.0, "critical": 85.0},
        {"medium": 40.0, "high": 90.0, "critical": 85.0},
    ],
)
def test_unordered_severity_thresholds_are_rejected(
    thresholds: dict[str, float],
) -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        SeverityThresholds(**thresholds)


@pytest.mark.parametrize("value", [0.0, -1.0, 100.1])
def test_severity_threshold_bounds(value: float) -> None:
    with pytest.raises(ValidationError):
        SeverityThresholds(medium=value)


def test_alert_score_floor_below_fired_floor_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be below"):
        DetectionConfig(
            scoring=ScoringConfig(min_fired_risk_score=20.0),
            alerting=AlertingConfig(min_alert_risk_score=10.0),
        )


@pytest.mark.parametrize("value", [0.0, -5.0, 120.0])
def test_invalid_alert_score_floor_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        AlertingConfig(min_alert_risk_score=value)


@pytest.mark.parametrize("value", ["0s", timedelta(0), timedelta(seconds=-1)])
def test_non_positive_grouping_window_is_rejected(value: Any) -> None:
    with pytest.raises(ValidationError):
        AlertingConfig(grouping_window=value)


@pytest.mark.parametrize("value", ["0s", timedelta(0), timedelta(seconds=-1)])
def test_non_positive_cooldown_is_rejected(value: Any) -> None:
    with pytest.raises(ValidationError):
        AlertingConfig(cooldown=value)


def test_malformed_duration_is_rejected() -> None:
    malformed: Any = "soon"
    with pytest.raises(ValidationError, match="Invalid duration"):
        AlertingConfig(cooldown=malformed)


def test_invalid_suppression_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AlertingConfig(max_alerts_per_group_per_window=0)


def test_incomplete_scope_dimension_is_rejected() -> None:
    with pytest.raises(ValidationError, match="does not cover correlation group"):
        AlertingConfig(
            scope_dimension={CorrelationGroup.SOURCE_FANOUT: ScopeKind.SOURCE}
        )


def test_unknown_scope_dimension_key_is_rejected() -> None:
    unknown: Any = {"not_a_group": "user"}
    with pytest.raises(ValidationError):
        AlertingConfig(scope_dimension=unknown)


@pytest.mark.parametrize("value", [0.5, 1.0, 200.0])
def test_invalid_saturation_multiple_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        SignalConfig(saturation_multiple=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.2])
def test_invalid_min_signal_strength_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        SignalConfig(min_signal_strength=value)


@pytest.mark.parametrize("value", [0.0, -1.0, 101.0])
def test_invalid_min_fired_risk_score_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        ScoringConfig(min_fired_risk_score=value)


def test_incompatible_feature_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="incompatible with the feature schema"):
        DetectionConfig(required_feature_schema_version="9.9.9")


def test_incompatible_catalog_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="incompatible with the registered"):
        DetectionConfig(rule_catalog_version="9.9.9")


def test_unsafe_output_dir_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"'\.\.'"):
        DetectionConfig(output_dir=Path("../elsewhere"))


def test_unsafe_reports_dir_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"'\.\.'"):
        DetectionConfig(reports_dir=Path("../elsewhere"))


def test_extra_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {"unexpected_setting": True})
    with pytest.raises(ConfigurationError, match="Invalid detection configuration"):
        load_detection_config(path)


# ---------------------------------------------------------------------------
# Secrets and executable configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["api_key", "secret", "pseudonymization_key", "auth_token", "db-password"],
)
def test_secret_shaped_keys_are_rejected(tmp_path: Path, key: str) -> None:
    path = _write(tmp_path, {key: "anything"})
    with pytest.raises(ConfigurationError, match="looks like a secret"):
        load_detection_config(path)


def test_nested_secret_shaped_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {"alerting": {"private_key": "x"}})
    with pytest.raises(ConfigurationError, match="looks like a secret"):
        load_detection_config(path)


def test_secret_inside_a_list_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, {"rules": [{"credentials": "x"}]})
    with pytest.raises(ConfigurationError, match="looks like a secret"):
        load_detection_config(path)


def test_secret_rejection_never_echoes_the_value(tmp_path: Path) -> None:
    path = _write(tmp_path, {"api_key": "hunter2-do-not-echo"})
    with pytest.raises(ConfigurationError) as exc:
        load_detection_config(path)
    assert "hunter2-do-not-echo" not in str(exc.value)


def test_controlled_vocabulary_keys_are_not_flagged() -> None:
    """A correlation group named ``credential_...`` is not a secret.

    The scanner exempts keys drawn from the project's own registries; the
    alternative -- dropping ``credential`` from the token set -- would leave a
    real ``credentials:`` key unguarded.
    """
    assert "credential" in PROHIBITED_CONFIG_KEY_TOKENS
    config = load_detection_config(_config_dir() / "rules-development.yaml")
    assert (
        config.alerting.scope_dimension[
            CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET
        ]
        is ScopeKind.USER
    )


def test_yaml_python_object_tag_is_refused(tmp_path: Path) -> None:
    """``safe_load`` must refuse a constructor tag rather than execute it."""
    path = tmp_path / "evil.yaml"
    path.write_text("!!python/object/apply:os.system ['echo x']\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        load_detection_config(path)


def test_detection_package_uses_no_dynamic_execution() -> None:
    """No module in the detection package may evaluate configuration."""
    package = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "password_attack_detector"
        / "detection"
    )
    forbidden = ("eval(", "exec(", "importlib", "__import__", "pickle", "subprocess")
    offenders = [
        f"{source.name}:{token}"
        for source in sorted(package.rglob("*.py"))
        for token in forbidden
        if token in source.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Loading failures
# ---------------------------------------------------------------------------


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_detection_config(tmp_path / "absent.yaml")


def test_non_mapping_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
        load_detection_config(path)


def test_invalid_content_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, {"enabled_rule_ids": ["PAD-XX-999"]})
    with pytest.raises(ConfigurationError, match="Invalid detection configuration"):
        load_detection_config(path)


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------


def test_severity_boundaries_are_inclusive_from_below() -> None:
    thresholds = SeverityThresholds(medium=40.0, high=65.0, critical=85.0)
    assert thresholds.severity_for(0.0) is Severity.LOW
    assert thresholds.severity_for(39.9999) is Severity.LOW
    assert thresholds.severity_for(40.0) is Severity.MEDIUM
    assert thresholds.severity_for(64.9999) is Severity.MEDIUM
    assert thresholds.severity_for(65.0) is Severity.HIGH
    assert thresholds.severity_for(84.9999) is Severity.HIGH
    assert thresholds.severity_for(85.0) is Severity.CRITICAL
    assert thresholds.severity_for(100.0) is Severity.CRITICAL


@pytest.mark.parametrize("severity", list(Severity))
def test_every_severity_is_accepted_as_the_alert_floor(severity: Severity) -> None:
    """All four severities are valid alert floors, LOW included."""
    config = AlertingConfig(min_alert_severity=severity)
    assert config.min_alert_severity is severity


def test_low_alert_reachability_reflects_both_gates() -> None:
    reachable = DetectionConfig()
    assert reachable.low_alert_reachable is True

    by_severity = DetectionConfig(
        alerting=AlertingConfig(min_alert_severity=Severity.MEDIUM)
    )
    assert by_severity.low_alert_reachable is False

    by_score = DetectionConfig(
        alerting=AlertingConfig(min_alert_risk_score=40.0),
        severity_thresholds=SeverityThresholds(medium=40.0, high=65.0, critical=85.0),
    )
    assert by_score.low_alert_reachable is False


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_hex() -> None:
    config = DetectionConfig()
    fingerprint = config.fingerprint()
    assert len(fingerprint) == 64
    assert fingerprint == DetectionConfig().fingerprint()


def test_fingerprint_ignores_paths_and_overwrite() -> None:
    baseline = DetectionConfig().fingerprint()
    relocated = DetectionConfig(
        output_dir=Path("somewhere/else"),
        reports_dir=Path("another/place"),
        overwrite=True,
    )
    assert relocated.fingerprint() == baseline


def test_fingerprint_changes_with_semantic_settings() -> None:
    baseline = DetectionConfig().fingerprint()
    assert DetectionConfig(enabled_rule_ids=("PAD-BF-001",)).fingerprint() != baseline
    assert (
        DetectionConfig(
            rules={"PAD-BF-001": RuleSettings(parameters={"min_pair_failures": 99})}
        ).fingerprint()
        != baseline
    )
    assert (
        DetectionConfig(signal=SignalConfig(saturation_multiple=4.0)).fingerprint()
        != baseline
    )
    assert (
        DetectionConfig(
            alerting=AlertingConfig(min_alert_severity=Severity.HIGH)
        ).fingerprint()
        != baseline
    )


def test_fingerprint_is_independent_of_rule_declaration_order() -> None:
    forward = DetectionConfig(enabled_rule_ids=("PAD-BF-001", "PAD-PS-001"))
    reverse = DetectionConfig(enabled_rule_ids=("PAD-PS-001", "PAD-BF-001"))
    assert forward.fingerprint() == reverse.fingerprint()


def test_fingerprint_records_effective_not_only_overridden_parameters() -> None:
    """The digest pins the thresholds that ran, not just the ones written down."""
    data = DetectionConfig().fingerprint_data()
    parameters = data["rules"]["PAD-BF-001"]["parameters"]
    assert parameters["min_pair_failures"] == 8
    assert parameters["window"] == "5m"


def test_every_field_is_fingerprinted_or_explicitly_excluded() -> None:
    """A new field must carry a decision about its fingerprint status."""
    covered = set(DetectionConfig().fingerprint_data())
    declared = set(DetectionConfig.model_fields)
    undecided = declared - covered - DETECTION_FINGERPRINT_EXCLUDED_FIELDS
    assert undecided == set()


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_shipped_config_fingerprint_survives_a_reload(
    tmp_path: Path, name: str
) -> None:
    source = _config_dir() / name
    copy = tmp_path / "copied.yaml"
    copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_detection_config(source).fingerprint() == (
        load_detection_config(copy).fingerprint()
    )
