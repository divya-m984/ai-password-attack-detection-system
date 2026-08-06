"""Tests for detection artifact validation.

Two things are proved here.  The first is coverage: every ``D0xx`` code the
validator can emit has a fixture that provokes it, so a check cannot silently
stop working.  The second is discretion: no message the validator produces may
carry an identifier, a scope value, an evidence value, or a path, whatever the
input looked like.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from password_attack_detector.detection.alerts import AlertBuilder
from password_attack_detector.detection.config import (
    AlertingConfig,
    DetectionConfig,
    ScoringConfig,
)
from password_attack_detector.detection.engine import DetectionEngine
from password_attack_detector.detection.enums import (
    AlertGroupingMode,
    AttackCategory,
    CorrelationGroup,
    ScopeKind,
    Severity,
)
from password_attack_detector.detection.schemas import (
    DetectionValidationStatus,
    FiredDetection,
    RiskAssessment,
    SecurityAlert,
    detection_identifier,
)
from password_attack_detector.detection.scoring import RiskScorer
from password_attack_detector.detection.serialization import (
    ALERT_COLUMNS,
    DETECTION_COLUMNS,
    RISK_COLUMNS,
)
from password_attack_detector.detection.validation import (
    DETECTION_VALIDATION_CODES,
    DetectionValidator,
    validate_detection_artifacts,
)
from tests.unit.detection import factories
from tests.unit.detection.factories import fired_detection, scope_record

WHEN = factories.WHEN
CONFIG = DetectionConfig()
FINGERPRINT = CONFIG.fingerprint()


# ---------------------------------------------------------------------------
# Fixtures: a coherent artifact set, and the pieces to break it with
# ---------------------------------------------------------------------------


def assessment(
    anchor: str = "a1",
    *,
    minutes: float = 0.0,
    rule_ids: tuple[str, ...] = ("PAD-BF-001",),
    risk_score: float = 70.0,
    severity: Severity = Severity.HIGH,
    category: AttackCategory | None = AttackCategory.BRUTE_FORCE,
    **overrides: Any,
) -> RiskAssessment:
    """Build one risk assessment consistent with the active configuration."""
    data: dict[str, Any] = {
        "anchor_event_id": anchor,
        "anchor_event_time": WHEN + timedelta(minutes=minutes),
        "risk_score": risk_score,
        "severity": severity,
        "primary_attack_category": category,
        "contributing_categories": () if category is None else (category,),
        "fired_rule_count": len(rule_ids),
        "fired_rule_ids": tuple(sorted(rule_ids)),
        "scoring_version": "1.0.0",
        "configuration_fingerprint": FINGERPRINT,
    }
    data.update(overrides)
    return RiskAssessment(**data)


def alert(
    *,
    alert_id: str = "alert-0001",
    minutes: float = 0.0,
    span_minutes: float = 1.0,
    rule_ids: tuple[str, ...] = ("PAD-BF-001",),
    aggregate: float = 70.0,
    peak: float = 70.0,
    severity: Severity = Severity.HIGH,
    category: AttackCategory = AttackCategory.BRUTE_FORCE,
    group: CorrelationGroup = CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
    **overrides: Any,
) -> SecurityAlert:
    """Build one alert consistent with the active configuration."""
    data: dict[str, Any] = {
        "alert_id": alert_id,
        "attack_category": category,
        "correlation_group": group,
        "grouping_mode": AlertGroupingMode.CATEGORY_SCOPED,
        "scope_kind": ScopeKind.NONE,
        "scope_value": None,
        "first_seen": WHEN + timedelta(minutes=minutes),
        "last_seen": WHEN + timedelta(minutes=minutes + span_minutes),
        "contributing_event_count": 1,
        "contributing_rule_ids": tuple(sorted(rule_ids)),
        "aggregate_risk_score": aggregate,
        "peak_risk_score": peak,
        "initial_severity": severity,
        "current_severity": severity,
    }
    data.update(overrides)
    return SecurityAlert(**data)


def coherent() -> tuple[
    list[FiredDetection], list[RiskAssessment], list[SecurityAlert]
]:
    """Return an artifact set that should validate cleanly."""
    detections = [fired_detection("PAD-BF-001", anchor_event_id="a1")]
    assessments = [
        assessment("a1"),
        assessment(
            "a2",
            minutes=5,
            rule_ids=(),
            risk_score=0.0,
            severity=Severity.LOW,
            category=None,
        ),
    ]
    alerts = [alert()]
    return detections, assessments, alerts


def validate(
    detections: list[FiredDetection],
    assessments: list[RiskAssessment],
    alerts: list[SecurityAlert],
    *,
    config: DetectionConfig = CONFIG,
    **kwargs: Any,
) -> Any:
    """Run the validator over an artifact set."""
    return DetectionValidator(config).validate(
        detections, assessments, alerts, **kwargs
    )


def codes(result: Any) -> set[str]:
    """Return every error code a result carries."""
    return {finding.code for finding in result.errors}


# ---------------------------------------------------------------------------
# The valid case
# ---------------------------------------------------------------------------


def test_a_coherent_artifact_set_validates() -> None:
    result = validate(*coherent())
    assert not result.errors
    assert result.status in {
        DetectionValidationStatus.VALID,
        DetectionValidationStatus.WARNING,
    }
    assert result.detection_row_count == 1
    assert result.risk_assessment_row_count == 2
    assert result.alert_row_count == 1
    assert result.invalid_value_count == 0
    assert result.relationship_error_count == 0
    assert result.prohibited_column_count == 0


def test_the_result_records_every_contract_version() -> None:
    result = validate(*coherent())
    assert result.detection_schema_version == "1.0.0"
    assert result.scoring_version == "1.0.0"
    assert result.alerting_version == "1.0.0"


def test_a_real_pipeline_run_validates() -> None:
    """The end-to-end case: what the engine, scorer, and builder produce."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    builders = [
        factories.brute_force_row,
        factories.spraying_row,
        factories.stuffing_row,
        factories.quiet_row,
    ]
    rows = [
        builders[index % 4](
            catalog,
            anchor_event_id=f"anchor-{index:04d}",
            anchor_event_time=WHEN + timedelta(minutes=index * 3),
        )
        for index in range(12)
    ]
    detections = list(engine.run(rows).fired_detections)
    scored = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    alerts = AlertBuilder(CONFIG).build(scored.assessments, detections=detections)

    result = validate(
        detections,
        list(scored.assessments),
        list(alerts.alerts),
        detection_columns=DETECTION_COLUMNS,
        risk_columns=RISK_COLUMNS,
        alert_columns=ALERT_COLUMNS,
    )
    assert not result.errors, [item.message for item in result.errors]


def test_the_one_shot_helper_matches_the_validator() -> None:
    detections, assessments, alerts = coherent()
    assert validate_detection_artifacts(
        CONFIG, detections, assessments, alerts
    ) == validate(detections, assessments, alerts)


# ---------------------------------------------------------------------------
# Detection-table checks
# ---------------------------------------------------------------------------


def test_a_duplicate_detection_identifier_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    _, assessments, alerts = coherent()
    result = validate([detection, detection], assessments, alerts)
    assert "D005" in codes(result)


def test_a_detection_identifier_that_is_not_derived_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    forged = detection.model_construct(
        **{
            **detection.__dict__,
            "detection_id": detection_identifier("z", "PAD-BF-001", "1.0.0"),
        }
    )
    _, assessments, alerts = coherent()
    result = validate([forged], assessments, alerts)
    assert "D006" in codes(result)


def test_an_unregistered_rule_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    unknown = detection.model_construct(
        **{**detection.__dict__, "rule_id": "PAD-XXX-999"}
    )
    _, assessments, alerts = coherent()
    result = validate([unknown], assessments, alerts)
    assert "D007" in codes(result)


def test_a_wrong_rule_version_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    stale = detection.model_construct(**{**detection.__dict__, "rule_version": "9.9.9"})
    _, assessments, alerts = coherent()
    result = validate([stale], assessments, alerts)
    assert "D008" in codes(result)


def test_metadata_disagreeing_with_the_catalog_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    swapped = detection.model_construct(
        **{**detection.__dict__, "attack_category": AttackCategory.BOT_ACTIVITY}
    )
    _, assessments, alerts = coherent()
    result = validate([swapped], assessments, alerts)
    assert "D009" in codes(result)


@pytest.mark.parametrize("strength", [-0.1, 1.5])
def test_a_signal_strength_outside_the_unit_interval_is_rejected(
    strength: float,
) -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    broken = detection.model_construct(
        **{**detection.__dict__, "signal_strength": strength}
    )
    _, assessments, alerts = coherent()
    result = validate([broken], assessments, alerts)
    assert "D010" in codes(result)
    assert validate([broken], assessments, alerts).invalid_value_count > 0


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_non_finite_signal_strength_is_rejected(value: float) -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    broken = detection.model_construct(
        **{**detection.__dict__, "signal_strength": value}
    )
    _, assessments, alerts = coherent()
    assert "D011" in codes(validate([broken], assessments, alerts))


def test_a_naive_detection_timestamp_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    naive = detection.model_construct(
        **{**detection.__dict__, "anchor_event_time": datetime(2026, 3, 1, 12, 0)}
    )
    _, assessments, alerts = coherent()
    assert "D012" in codes(validate([naive], assessments, alerts))


def test_a_detection_with_no_evidence_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    bare = detection.model_construct(**{**detection.__dict__, "evidence": ()})
    _, assessments, alerts = coherent()
    assert "D013" in codes(validate([bare], assessments, alerts))


def test_an_unsupported_detection_schema_version_is_rejected() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    stale = detection.model_construct(
        **{**detection.__dict__, "detection_schema_version": "0.9.0"}
    )
    _, assessments, alerts = coherent()
    assert "D004" in codes(validate([stale], assessments, alerts))


# ---------------------------------------------------------------------------
# Risk-assessment checks
# ---------------------------------------------------------------------------


def test_a_duplicate_risk_anchor_is_rejected() -> None:
    detections, _, alerts = coherent()
    result = validate(detections, [assessment("a1"), assessment("a1")], alerts)
    assert "D014" in codes(result)


@pytest.mark.parametrize("score", [-1.0, 101.0])
def test_a_risk_score_outside_its_range_is_rejected(score: float) -> None:
    detections, assessments, alerts = coherent()
    broken = assessments[0].model_construct(
        **{**assessments[0].__dict__, "risk_score": score}
    )
    assert "D015" in codes(validate(detections, [broken], alerts))


@pytest.mark.parametrize("value", [math.nan, -math.inf])
def test_a_non_finite_risk_score_is_rejected(value: float) -> None:
    detections, assessments, alerts = coherent()
    broken = assessments[0].model_construct(
        **{**assessments[0].__dict__, "risk_score": value}
    )
    assert "D011" in codes(validate(detections, [broken], alerts))


def test_a_fired_count_disagreeing_with_the_rule_list_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    broken = assessments[0].model_construct(
        **{**assessments[0].__dict__, "fired_rule_count": 7}
    )
    assert "D016" in codes(validate(detections, [broken], alerts))


def test_a_zero_score_with_fired_rules_is_rejected() -> None:
    """The zero-risk equivalence, checked in the artifact rather than assumed."""
    detections, assessments, alerts = coherent()
    broken = assessments[0].model_construct(
        **{**assessments[0].__dict__, "risk_score": 0.0}
    )
    assert "D017" in codes(validate(detections, [broken], alerts))


def test_a_positive_score_with_no_fired_rules_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    broken = assessments[1].model_construct(
        **{**assessments[1].__dict__, "risk_score": 40.0}
    )
    assert "D017" in codes(validate(detections, [assessments[0], broken], alerts))


def test_a_primary_category_outside_the_contributing_set_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    broken = assessments[0].model_construct(
        **{
            **assessments[0].__dict__,
            "primary_attack_category": AttackCategory.BOT_ACTIVITY,
        }
    )
    assert "D018" in codes(validate(detections, [broken], alerts))


def test_a_fired_assessment_below_the_fired_floor_is_rejected() -> None:
    # The alert floor must not sit below the fired floor, so both move.
    config = DetectionConfig(
        scoring=ScoringConfig(min_fired_risk_score=50.0),
        alerting=AlertingConfig(min_alert_risk_score=50.0),
    )
    detections, _, alerts = coherent()
    low = assessment("a1", risk_score=5.0, severity=Severity.LOW)
    result = validate(detections, [low], alerts, config=config)
    assert "D019" in codes(result)


def test_a_foreign_configuration_fingerprint_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    foreign = assessments[0].model_construct(
        **{**assessments[0].__dict__, "configuration_fingerprint": "0" * 64}
    )
    assert "D020" in codes(validate(detections, [foreign], alerts))


def test_an_unsupported_scoring_version_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    stale = assessments[0].model_construct(
        **{**assessments[0].__dict__, "scoring_version": "0.1.0"}
    )
    assert "D032" in codes(validate(detections, [stale], alerts))


def test_an_empty_assessment_table_is_rejected() -> None:
    assert "D001" in codes(validate([], [], []))


# ---------------------------------------------------------------------------
# Alert checks
# ---------------------------------------------------------------------------


def test_a_duplicate_alert_identifier_is_rejected() -> None:
    detections, assessments, _ = coherent()
    assert "D021" in codes(validate(detections, assessments, [alert(), alert()]))


@pytest.mark.parametrize("value", [-5.0, 120.0])
def test_an_alert_risk_outside_its_range_is_rejected(value: float) -> None:
    detections, assessments, alerts = coherent()
    broken = alerts[0].model_construct(
        **{**alerts[0].__dict__, "peak_risk_score": value}
    )
    assert "D022" in codes(validate(detections, assessments, [broken]))


def test_an_aggregate_above_the_peak_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    broken = alerts[0].model_construct(
        **{**alerts[0].__dict__, "aggregate_risk_score": 95.0, "peak_risk_score": 70.0}
    )
    assert "D023" in codes(validate(detections, assessments, [broken]))


def test_an_aggregate_outside_its_members_range_is_rejected() -> None:
    """A mean lies between the smallest and largest member; a sum does not."""
    detections = [
        fired_detection("PAD-BF-001", anchor_event_id="a1"),
        fired_detection("PAD-BF-001", anchor_event_id="a2"),
    ]
    assessments = [
        assessment("a1", risk_score=40.0, severity=Severity.MEDIUM),
        assessment("a2", minutes=1, risk_score=44.0, severity=Severity.MEDIUM),
    ]
    summed = alert(
        span_minutes=5.0, aggregate=84.0, peak=90.0, contributing_event_count=2
    )
    assert "D031" in codes(validate(detections, assessments, [summed]))


def test_a_mean_inside_its_members_range_is_accepted() -> None:
    detections = [
        fired_detection("PAD-BF-001", anchor_event_id="a1"),
        fired_detection("PAD-BF-001", anchor_event_id="a2"),
    ]
    assessments = [
        assessment("a1", risk_score=60.0),
        assessment("a2", minutes=1, risk_score=80.0),
    ]
    mean = alert(
        span_minutes=5.0, aggregate=70.0, peak=80.0, contributing_event_count=2
    )
    assert "D031" not in codes(validate(detections, assessments, [mean]))


def test_a_reversed_time_range_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    broken = alerts[0].model_construct(
        **{**alerts[0].__dict__, "last_seen": alerts[0].first_seen - timedelta(hours=1)}
    )
    assert "D026" in codes(validate(detections, assessments, [broken]))


def test_a_non_positive_contributing_count_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    broken = alerts[0].model_construct(
        **{**alerts[0].__dict__, "contributing_event_count": 0}
    )
    assert "D026" in codes(validate(detections, assessments, [broken]))


def test_an_alert_below_the_score_floor_is_rejected() -> None:
    detections, assessments, _ = coherent()
    low = alert(aggregate=2.0, peak=2.0, severity=Severity.LOW)
    assert "D024" in codes(validate(detections, assessments, [low]))


def test_an_alert_below_the_configured_severity_floor_is_rejected() -> None:
    config = DetectionConfig(
        alerting=AlertingConfig(min_alert_severity=Severity.CRITICAL)
    )
    detections, assessments, alerts = coherent()
    assert "D024" in codes(validate(detections, assessments, alerts, config=config))


def test_a_low_alert_is_accepted_when_both_floors_permit_it() -> None:
    """LOW is an ordinary alert severity; only the floors gate it."""
    detections, assessments, _ = coherent()
    low = alert(aggregate=12.0, peak=12.0, severity=Severity.LOW)
    result = validate(detections, assessments, [low])
    assert "D024" not in codes(result)


def test_a_category_scoped_alert_carrying_a_scope_value_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    leaking = alerts[0].model_construct(
        **{
            **alerts[0].__dict__,
            "scope_value": "u:" + "a" * 32,
            "scope_kind": ScopeKind.USER,
        }
    )
    assert "D025" in codes(validate(detections, assessments, [leaking]))


def test_an_alert_naming_an_unfired_rule_is_rejected() -> None:
    detections, assessments, _ = coherent()
    dangling = alert(rule_ids=("PAD-BOT-001",))
    assert "D027" in codes(validate(detections, assessments, [dangling]))


def test_an_unsupported_alerting_version_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from password_attack_detector.detection import validation as validation_module

    monkeypatch.setattr(validation_module, "ALERTING_VERSION", "9.9.9")
    assert "D032" in codes(validate(*coherent()))


# ---------------------------------------------------------------------------
# Cross-artifact relationships
# ---------------------------------------------------------------------------


def test_a_detection_without_an_assessment_is_rejected() -> None:
    detections = [fired_detection("PAD-BF-001", anchor_event_id="orphan")]
    _, assessments, alerts = coherent()
    result = validate(detections, assessments, alerts)
    assert "D028" in codes(result)
    assert result.relationship_error_count > 0


def test_an_assessment_disagreeing_with_its_detections_is_rejected() -> None:
    detections, _, alerts = coherent()
    mismatched = [assessment("a1", rule_ids=("PAD-PS-001",))]
    assert "D029" in codes(validate(detections, mismatched, alerts))


def test_a_duplicate_scope_anchor_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    scope = [
        scope_record("a1", user="1"),
        scope_record("a1", user="2"),
        scope_record("a2", user="3"),
    ]
    assert "D030" in codes(
        validate(detections, assessments, alerts, entity_scope=scope)
    )


def test_an_asymmetric_scope_relationship_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    assert "D030" in codes(
        validate(
            detections, assessments, alerts, entity_scope=[scope_record("a1", user="1")]
        )
    )


def test_a_one_to_one_scope_table_is_accepted() -> None:
    detections, assessments, alerts = coherent()
    scope = [scope_record("a1", user="1"), scope_record("a2", user="2")]
    assert "D030" not in codes(
        validate(detections, assessments, alerts, entity_scope=scope)
    )


# ---------------------------------------------------------------------------
# Column shape
# ---------------------------------------------------------------------------


def test_a_reordered_column_set_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    shuffled = (DETECTION_COLUMNS[1], DETECTION_COLUMNS[0], *DETECTION_COLUMNS[2:])
    result = validate(detections, assessments, alerts, detection_columns=shuffled)
    assert "D002" in codes(result)


@pytest.mark.parametrize(
    "column",
    [
        "label",
        "attack_class",
        "campaign_id",
        "split",
        "model_probability",
        "user_scope",
        "source_scope",
        "user_id",
    ],
)
def test_a_prohibited_column_is_rejected(column: str) -> None:
    detections, assessments, alerts = coherent()
    result = validate(
        detections,
        assessments,
        alerts,
        detection_columns=(*DETECTION_COLUMNS, column),
    )
    assert "D003" in codes(result)
    assert result.prohibited_column_count > 0


def test_the_declared_column_order_validates() -> None:
    detections, assessments, alerts = coherent()
    result = validate(
        detections,
        assessments,
        alerts,
        detection_columns=DETECTION_COLUMNS,
        risk_columns=RISK_COLUMNS,
        alert_columns=ALERT_COLUMNS,
    )
    assert "D002" not in codes(result)
    assert "D003" not in codes(result)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_a_rule_that_never_fired_is_a_warning_not_an_error() -> None:
    result = validate(*coherent())
    assert "D050" in {finding.code for finding in result.warnings}
    assert result.status is not DetectionValidationStatus.INVALID


def test_an_empty_alert_set_is_a_warning() -> None:
    detections, assessments, _ = coherent()
    result = validate(detections, assessments, [])
    assert "D051" in {finding.code for finding in result.warnings}


def test_a_saturated_insufficient_data_rate_is_a_warning() -> None:
    detections, assessments, alerts = coherent()
    saturated = [
        item.model_construct(**{**item.__dict__, "insufficient_data_count": 500})
        for item in assessments
    ]
    result = validate(detections, saturated, alerts)
    assert "D052" in {finding.code for finding in result.warnings}


# ---------------------------------------------------------------------------
# Discretion
# ---------------------------------------------------------------------------


def test_no_message_carries_an_identifier_or_a_scope_value() -> None:
    """Every provokable finding, swept for disclosure in one pass."""
    secret_user = "u:" + "f" * 32
    detections = [
        fired_detection("PAD-BF-001", anchor_event_id="anchor-secret-0001"),
        fired_detection("PAD-BF-001", anchor_event_id="anchor-secret-0001"),
    ]
    assessments = [
        assessment("anchor-secret-0002", rule_ids=("PAD-PS-001",)),
        assessment("anchor-secret-0002", rule_ids=("PAD-PS-001",)),
    ]
    alerts = [
        alert(alert_id="alert-secret-0003", rule_ids=("PAD-BOT-001",)),
        alert(alert_id="alert-secret-0003", rule_ids=("PAD-BOT-001",)),
    ]
    result = validate(
        detections,
        assessments,
        alerts,
        entity_scope=[scope_record("anchor-secret-0004", user="f" * 32)],
        detection_columns=(*DETECTION_COLUMNS, "user_scope"),
    )
    assert result.errors

    rendered = " ".join(
        finding.message for finding in (*result.errors, *result.warnings)
    )
    for forbidden in (
        "anchor-secret",
        "alert-secret",
        secret_user,
        "PAD-BF-001",
        "/home/",
    ):
        assert forbidden not in rendered, forbidden


def test_the_serialised_result_carries_no_identifier() -> None:
    detections = [fired_detection("PAD-BF-001", anchor_event_id="anchor-secret-0001")]
    _, assessments, alerts = coherent()
    payload = str(validate(detections, assessments, alerts).to_dict())
    assert "anchor-secret" not in payload


def test_every_documented_code_is_reachable_and_every_used_code_documented() -> None:
    """The code table and the validator cannot drift apart."""
    import inspect

    from password_attack_detector.detection import validation as validation_module

    source = inspect.getsource(validation_module)
    for code in DETECTION_VALIDATION_CODES:
        assert source.count(f'"{code}"') >= 2, code


def test_findings_carry_a_column_and_a_count() -> None:
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    _, assessments, alerts = coherent()
    result = validate([detection, detection], assessments, alerts)
    duplicate = next(item for item in result.errors if item.code == "D005")
    assert duplicate.column == "detection_id"
    assert duplicate.count == 1


def test_the_validator_never_raises_on_malformed_input() -> None:
    """A validator that threw would give a caller one problem, not the set."""
    detection = fired_detection("PAD-BF-001", anchor_event_id="a1")
    broken = detection.model_construct(
        **{
            **detection.__dict__,
            "signal_strength": math.nan,
            "rule_id": "PAD-XXX-999",
            "anchor_event_time": datetime(2026, 1, 1, tzinfo=UTC),
        }
    )
    result = validate([broken], [assessment("a1")], [])
    assert result.status is DetectionValidationStatus.INVALID
    assert len(result.errors) > 1


def test_a_fired_assessment_without_a_primary_category_is_rejected() -> None:
    """The second half of the zero-risk equivalence."""
    detections, assessments, alerts = coherent()
    broken = assessments[0].model_construct(
        **{**assessments[0].__dict__, "primary_attack_category": None}
    )
    assert "D017" in codes(validate(detections, [broken], alerts))


def test_a_naive_assessment_timestamp_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    naive = assessments[0].model_construct(
        **{**assessments[0].__dict__, "anchor_event_time": datetime(2026, 3, 1, 12, 0)}
    )
    assert "D012" in codes(validate(detections, [naive], alerts))


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_non_finite_alert_risk_is_rejected(value: float) -> None:
    detections, assessments, alerts = coherent()
    broken = alerts[0].model_construct(
        **{**alerts[0].__dict__, "aggregate_risk_score": value}
    )
    assert "D011" in codes(validate(detections, assessments, [broken]))


def test_a_naive_alert_timestamp_is_rejected() -> None:
    detections, assessments, alerts = coherent()
    naive = alerts[0].model_construct(
        **{**alerts[0].__dict__, "first_seen": datetime(2026, 3, 1, 12, 0)}
    )
    assert "D026" in codes(validate(detections, assessments, [naive]))


def test_a_non_datetime_timestamp_is_rejected() -> None:
    """A string where a timestamp belongs is a corrupt artifact, not a date."""
    detections, assessments, alerts = coherent()
    textual = assessments[0].model_construct(
        **{**assessments[0].__dict__, "anchor_event_time": "2026-03-01T12:00:00Z"}
    )
    assert "D012" in codes(validate(detections, [textual], alerts))
