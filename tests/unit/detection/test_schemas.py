"""Tests for the detection artifact contracts."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from password_attack_detector.detection import schemas
from password_attack_detector.detection.enums import (
    AlertGroupingMode,
    AttackCategory,
    CorrelationGroup,
    EvidenceComparator,
    PrivacyClass,
    RuleFamily,
    RuleStatus,
    ScopeKind,
    Severity,
    SuppressionReason,
    severity_at_least,
)
from password_attack_detector.detection.schemas import (
    PROHIBITED_CLAIM_TERMS,
    PROHIBITED_FIELDS,
    AlertingStats,
    DetectionValidationFinding,
    DetectionValidationResult,
    DetectionValidationStatus,
    EntityScopeRecord,
    EvidenceItem,
    FiredDetection,
    RiskAssessment,
    RuleEvaluationResult,
    SecurityAlert,
    detection_identifier,
)

ANCHOR = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
WHEN = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _evidence(**overrides: Any) -> EvidenceItem:
    data: dict[str, Any] = {
        "evidence_code": "BF_PAIR_FAILURE_COUNT",
        "feature_name": "pair_failure_count__5m",
        "comparator": EvidenceComparator.GTE,
        "observed_value": 12,
        "threshold_value": 8,
        "unit": "count",
        "message": "Observed 12 failed attempts; this contributed to this detection.",
    }
    data.update(overrides)
    return EvidenceItem(**data)


def _result(**overrides: Any) -> RuleEvaluationResult:
    data: dict[str, Any] = {
        "rule_id": "PAD-BF-001",
        "rule_version": "1.0.0",
        "anchor_event_id": ANCHOR,
        "anchor_event_time": WHEN,
        "status": RuleStatus.FIRED,
        "rule_family": RuleFamily.BRUTE_FORCE,
        "attack_category": AttackCategory.BRUTE_FORCE,
        "correlation_group": CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
        "severity": Severity.HIGH,
        "signal_strength": 0.5,
        "evidence": (_evidence(),),
        "reason_codes": ("BF_PAIR_FAILURE_COUNT",),
    }
    data.update(overrides)
    return RuleEvaluationResult(**data)


def _alert(**overrides: Any) -> SecurityAlert:
    data: dict[str, Any] = {
        "alert_id": "9f1d2c3b-4a5e-4f60-8192-a3b4c5d6e7f8",
        "attack_category": AttackCategory.BRUTE_FORCE,
        "correlation_group": CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
        "grouping_mode": AlertGroupingMode.CATEGORY_SCOPED,
        "scope_kind": ScopeKind.NONE,
        "scope_value": None,
        "first_seen": WHEN,
        "last_seen": WHEN + timedelta(minutes=5),
        "contributing_event_count": 3,
        "contributing_rule_ids": ("PAD-BF-001",),
        "aggregate_risk_score": 40.0,
        "peak_risk_score": 55.0,
        "initial_severity": Severity.MEDIUM,
        "current_severity": Severity.MEDIUM,
    }
    data.update(overrides)
    return SecurityAlert(**data)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_severity_has_exactly_four_levels() -> None:
    """There is no INFORMATIONAL level and no informational alert path."""
    assert [str(level) for level in Severity] == ["low", "medium", "high", "critical"]


def test_severity_ordering() -> None:
    assert severity_at_least(Severity.HIGH, Severity.MEDIUM)
    assert severity_at_least(Severity.LOW, Severity.LOW)
    assert not severity_at_least(Severity.LOW, Severity.MEDIUM)


def test_no_attack_category_is_a_holdout_or_blocked_account() -> None:
    values = {str(category) for category in AttackCategory}
    assert "novel_anomaly_holdout" not in values
    assert not any("blocked" in value for value in values)


def test_enums_serialise_to_their_values() -> None:
    assert str(RuleStatus.INSUFFICIENT_DATA) == "insufficient_data"
    assert str(AlertGroupingMode.ENTITY_SCOPED) == "entity_scoped"
    assert str(SuppressionReason.COOLDOWN) == "cooldown"
    assert str(PrivacyClass.OPERATIONAL_METADATA) == "operational_metadata"


# ---------------------------------------------------------------------------
# Prohibited fields
# ---------------------------------------------------------------------------


def _detection_models() -> list[type[BaseModel]]:
    return [
        value
        for value in vars(schemas).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value is not BaseModel
    ]


def test_no_model_declares_a_ground_truth_split_or_model_output_field() -> None:
    offenders = [
        f"{model.__name__}.{name}"
        for model in _detection_models()
        for name in model.model_fields
        if name in PROHIBITED_FIELDS
    ]
    assert offenders == []


def test_every_model_forbids_extra_fields() -> None:
    permissive = [
        model.__name__
        for model in _detection_models()
        if model.model_config.get("extra") != "forbid"
    ]
    assert permissive == []


def test_every_model_is_frozen() -> None:
    mutable = [
        model.__name__
        for model in _detection_models()
        if not model.model_config.get("frozen")
    ]
    assert mutable == []


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------


def test_evidence_round_trips() -> None:
    item = _evidence()
    assert item.observed_value == 12
    assert item.comparator is EvidenceComparator.GTE


@pytest.mark.parametrize("code", ["bf_lower", "1_LEADING_DIGIT", "HAS SPACE", ""])
def test_invalid_evidence_code_is_rejected(code: str) -> None:
    with pytest.raises(ValidationError, match="evidence_code"):
        _evidence(evidence_code=code)


@pytest.mark.parametrize(
    "value",
    [
        "u:0123456789abcdef0123456789abcdef",
        "s:0123456789abcdef0123456789abcdef",
        "sess:0123456789abcdef0123456789abcdef",
        ANCHOR,
    ],
)
def test_identifier_shaped_evidence_values_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="looks like an identifier"):
        _evidence(observed_value=value)
    with pytest.raises(ValidationError, match="looks like an identifier"):
        _evidence(threshold_value=value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_evidence_values_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        _evidence(observed_value=value)


@pytest.mark.parametrize(
    "message",
    [
        "This proves the account was compromised.",
        "The pattern confirms credential theft.",
        "This guarantees an attack occurred.",
        "The probability of attack is high.",
    ],
)
def test_claim_asserting_messages_are_rejected(message: str) -> None:
    with pytest.raises(ValidationError, match="claim-asserting"):
        _evidence(message=message)


def test_empty_message_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        _evidence(message="   ")


@pytest.mark.parametrize(
    "message",
    [
        "Observed 12 failures; this contributed to this detection.",
        "A rate of 0.9 matched the configured rule condition.",
        "This is consistent with automated retries.",
        "This may indicate lockout pressure.",
    ],
)
def test_approved_wording_is_accepted(message: str) -> None:
    assert _evidence(message=message).message == message


def test_prohibited_claim_terms_cover_the_documented_register() -> None:
    assert {"proves", "confirms", "guarantees", "probability"} <= PROHIBITED_CLAIM_TERMS


# ---------------------------------------------------------------------------
# RuleEvaluationResult
# ---------------------------------------------------------------------------


def test_fired_result_requires_evidence_reasons_and_strength() -> None:
    with pytest.raises(ValidationError, match="at least one evidence item"):
        _result(evidence=())
    with pytest.raises(ValidationError, match="at least one reason code"):
        _result(reason_codes=())
    with pytest.raises(ValidationError, match="signal_strength > 0"):
        _result(signal_strength=0.0)


def test_fired_result_rejects_repeated_evidence_codes() -> None:
    with pytest.raises(ValidationError, match="repeat an evidence_code"):
        _result(evidence=(_evidence(), _evidence()))


@pytest.mark.parametrize(
    "status", [RuleStatus.NOT_FIRED, RuleStatus.INSUFFICIENT_DATA, RuleStatus.DISABLED]
)
def test_non_fired_results_carry_no_evidence(status: RuleStatus) -> None:
    with pytest.raises(ValidationError, match="must not carry evidence"):
        _result(status=status, signal_strength=0.0, reason_codes=("X",))


def test_not_fired_result_is_valid_without_reasons() -> None:
    result = _result(
        status=RuleStatus.NOT_FIRED,
        signal_strength=0.0,
        evidence=(),
        reason_codes=(),
    )
    assert not result.fired


def test_insufficient_data_must_name_a_reason() -> None:
    with pytest.raises(ValidationError, match="must name why"):
        _result(
            status=RuleStatus.INSUFFICIENT_DATA,
            signal_strength=0.0,
            evidence=(),
            reason_codes=(),
        )


def test_disabled_result_carries_no_reasons() -> None:
    with pytest.raises(ValidationError, match="must not carry reason codes"):
        _result(
            status=RuleStatus.DISABLED,
            signal_strength=0.0,
            evidence=(),
            reason_codes=("WHY",),
        )


def test_non_fired_result_must_score_zero() -> None:
    with pytest.raises(ValidationError, match="signal_strength == 0"):
        _result(status=RuleStatus.NOT_FIRED, signal_strength=0.4, evidence=())


@pytest.mark.parametrize("strength", [-0.01, 1.01])
def test_signal_strength_is_bounded(strength: float) -> None:
    with pytest.raises(ValidationError):
        _result(signal_strength=strength)


@pytest.mark.parametrize("rule_id", ["BF-001", "pad-bf-001", "PAD-BF-1", ""])
def test_invalid_rule_id_is_rejected(rule_id: str) -> None:
    with pytest.raises(ValidationError, match="rule_id"):
        _result(rule_id=rule_id)


def test_invalid_rule_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="semantic version"):
        _result(rule_version="1.0")


def test_empty_anchor_is_rejected() -> None:
    with pytest.raises(ValidationError, match="anchor_event_id"):
        _result(anchor_event_id="  ")


def test_naive_anchor_time_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _result(anchor_event_time=datetime(2026, 3, 1, 12, 0))


def test_anchor_time_is_normalised_to_utc() -> None:
    offset = datetime(2026, 3, 1, 14, 0, tzinfo=timezone_plus_two())
    result = _result(anchor_event_time=offset)
    assert result.anchor_event_time == WHEN
    assert result.anchor_event_time.tzinfo is UTC


def timezone_plus_two() -> Any:
    from datetime import timezone

    return timezone(timedelta(hours=2))


@pytest.mark.parametrize("codes", [("lower",), ("A", "A")])
def test_invalid_reason_codes_are_rejected(codes: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        _result(reason_codes=codes)


# ---------------------------------------------------------------------------
# FiredDetection
# ---------------------------------------------------------------------------


def test_detection_identifier_is_deterministic() -> None:
    first = detection_identifier(ANCHOR, "PAD-BF-001", "1.0.0")
    second = detection_identifier(ANCHOR, "PAD-BF-001", "1.0.0")
    assert first == second
    assert first != detection_identifier(ANCHOR, "PAD-PS-001", "1.0.0")
    assert first != detection_identifier(ANCHOR, "PAD-BF-001", "1.1.0")


def test_from_result_builds_a_keyed_detection() -> None:
    detection = FiredDetection.from_result(_result())
    assert detection.detection_id == detection_identifier(ANCHOR, "PAD-BF-001", "1.0.0")
    assert detection.detection_schema_version == "1.0.0"
    assert detection.evidence == _result().evidence


def test_from_result_refuses_a_non_fired_result() -> None:
    with pytest.raises(ValueError, match="only a fired result"):
        FiredDetection.from_result(
            _result(status=RuleStatus.NOT_FIRED, signal_strength=0.0, evidence=())
        )


def test_detection_id_must_be_the_derived_identifier() -> None:
    with pytest.raises(ValidationError, match="deterministic identifier"):
        FiredDetection(
            detection_id="00000000-0000-5000-8000-000000000000",
            anchor_event_id=ANCHOR,
            anchor_event_time=WHEN,
            rule_id="PAD-BF-001",
            rule_version="1.0.0",
            rule_family=RuleFamily.BRUTE_FORCE,
            attack_category=AttackCategory.BRUTE_FORCE,
            severity=Severity.HIGH,
            signal_strength=0.5,
            correlation_group=CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
            evidence=(_evidence(),),
            reason_codes=("BF_PAIR_FAILURE_COUNT",),
        )


# ---------------------------------------------------------------------------
# RiskAssessment
# ---------------------------------------------------------------------------


def _assessment(**overrides: Any) -> RiskAssessment:
    data: dict[str, Any] = {
        "anchor_event_id": ANCHOR,
        "anchor_event_time": WHEN,
        "risk_score": 55.0,
        "severity": Severity.MEDIUM,
        "primary_attack_category": AttackCategory.BRUTE_FORCE,
        "contributing_categories": (AttackCategory.BRUTE_FORCE,),
        "fired_rule_count": 1,
        "fired_rule_ids": ("PAD-BF-001",),
        "scoring_version": "1.0.0",
    }
    data.update(overrides)
    return RiskAssessment(**data)


def test_zero_fired_rules_means_zero_score_and_no_category() -> None:
    assessment = RiskAssessment(
        anchor_event_id=ANCHOR,
        anchor_event_time=WHEN,
        risk_score=0.0,
        severity=Severity.LOW,
        scoring_version="1.0.0",
    )
    assert assessment.fired_rule_count == 0
    assert assessment.primary_attack_category is None


def test_zero_fired_rules_with_a_positive_score_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"must score 0\.0"):
        RiskAssessment(
            anchor_event_id=ANCHOR,
            anchor_event_time=WHEN,
            risk_score=10.0,
            severity=Severity.LOW,
            scoring_version="1.0.0",
        )


def test_fired_rules_with_a_zero_score_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"must score above 0\.0"):
        _assessment(risk_score=0.0)


def test_fired_count_must_match_the_identifier_list() -> None:
    with pytest.raises(ValidationError, match="must match the number"):
        _assessment(fired_rule_count=2)


def test_fired_rule_ids_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValidationError, match="must be sorted"):
        _assessment(
            fired_rule_count=2,
            fired_rule_ids=("PAD-PS-001", "PAD-BF-001"),
            contributing_categories=(
                AttackCategory.BRUTE_FORCE,
                AttackCategory.PASSWORD_SPRAYING,
            ),
        )
    with pytest.raises(ValidationError, match="must not repeat"):
        _assessment(fired_rule_count=2, fired_rule_ids=("PAD-BF-001", "PAD-BF-001"))


def test_primary_category_must_be_contributing() -> None:
    with pytest.raises(ValidationError, match="must appear in contributing"):
        _assessment(contributing_categories=(AttackCategory.BOT_ACTIVITY,))


@pytest.mark.parametrize("score", [-0.1, 100.1])
def test_risk_score_is_bounded(score: float) -> None:
    with pytest.raises(ValidationError):
        _assessment(risk_score=score)


def test_risk_score_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        _assessment(risk_score=math.nan)


def test_invalid_scoring_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="semantic version"):
        _assessment(scoring_version="v1")


# ---------------------------------------------------------------------------
# SecurityAlert
# ---------------------------------------------------------------------------


def test_alert_round_trips() -> None:
    alert = _alert()
    assert alert.scope_kind is ScopeKind.NONE
    assert alert.scope_value is None


def test_low_is_a_valid_alert_severity() -> None:
    """LOW alerts are ordinary. Nothing in the schema forbids them."""
    alert = _alert(initial_severity=Severity.LOW, current_severity=Severity.LOW)
    assert alert.current_severity is Severity.LOW


def test_last_seen_must_not_precede_first_seen() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        _alert(last_seen=WHEN - timedelta(seconds=1))


def test_alert_needs_a_contributing_rule() -> None:
    with pytest.raises(ValidationError, match="at least one contributing rule"):
        _alert(contributing_rule_ids=())


def test_contributing_rules_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValidationError, match="must be sorted"):
        _alert(contributing_rule_ids=("PAD-PS-001", "PAD-BF-001"))
    with pytest.raises(ValidationError, match="must not repeat"):
        _alert(contributing_rule_ids=("PAD-BF-001", "PAD-BF-001"))


def test_peak_must_not_be_below_aggregate() -> None:
    with pytest.raises(ValidationError, match="must not be below"):
        _alert(aggregate_risk_score=60.0, peak_risk_score=50.0)


def test_scope_value_requires_entity_scoped_grouping() -> None:
    with pytest.raises(ValidationError, match="entity-scoped grouping"):
        _alert(scope_kind=ScopeKind.USER, scope_value="u:" + "a" * 32)


def test_scope_value_requires_a_scope_kind() -> None:
    with pytest.raises(ValidationError, match="user or source scope kind"):
        _alert(
            grouping_mode=AlertGroupingMode.ENTITY_SCOPED,
            scope_kind=ScopeKind.NONE,
            scope_value="u:" + "a" * 32,
        )


def test_scope_kind_requires_a_scope_value() -> None:
    with pytest.raises(ValidationError, match="requires a scope value"):
        _alert(
            grouping_mode=AlertGroupingMode.ENTITY_SCOPED,
            scope_kind=ScopeKind.USER,
            scope_value=None,
        )


def test_entity_scoped_alert_is_valid() -> None:
    alert = _alert(
        grouping_mode=AlertGroupingMode.ENTITY_SCOPED,
        scope_kind=ScopeKind.SOURCE,
        scope_value="s:" + "b" * 32,
    )
    assert alert.scope_privacy_class is PrivacyClass.OPERATIONAL_METADATA


def test_severity_may_not_fall_below_its_initial_value() -> None:
    with pytest.raises(ValidationError, match="must not fall below"):
        _alert(initial_severity=Severity.HIGH, current_severity=Severity.MEDIUM)


def test_escalation_requires_a_raised_severity() -> None:
    with pytest.raises(ValidationError, match="must record a raised severity"):
        _alert(escalation_count=1)
    escalated = _alert(
        initial_severity=Severity.MEDIUM,
        current_severity=Severity.HIGH,
        escalation_count=1,
    )
    assert escalated.escalation_count == 1


# ---------------------------------------------------------------------------
# AlertingStats
# ---------------------------------------------------------------------------


def test_alerting_stats_totals() -> None:
    stats = AlertingStats(
        grouping_mode=AlertGroupingMode.CATEGORY_SCOPED,
        alert_count=4,
        grouped_detection_count=11,
        suppressed_by_reason={
            SuppressionReason.COOLDOWN: 2,
            SuppressionReason.RATE_LIMIT: 1,
        },
    )
    assert stats.suppressed_total == 3
    assert stats.low_alert_reachable is True


def test_negative_suppression_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be negative"):
        AlertingStats(
            grouping_mode=AlertGroupingMode.CATEGORY_SCOPED,
            suppressed_by_reason={SuppressionReason.COOLDOWN: -1},
        )


# ---------------------------------------------------------------------------
# EntityScopeRecord
# ---------------------------------------------------------------------------


def test_entity_scope_record_carries_both_dimensions_independently() -> None:
    record = EntityScopeRecord(
        anchor_event_id=ANCHOR,
        user_scope="u:" + "a" * 32,
        source_scope=None,
    )
    assert record.user_scope is not None
    assert record.source_scope is None
    assert record.privacy_class is PrivacyClass.OPERATIONAL_METADATA


def test_entity_scope_record_repr_redacts_every_field() -> None:
    """A repr reaches logs and tracebacks; a scope value must not."""
    record = EntityScopeRecord(
        anchor_event_id=ANCHOR,
        user_scope="u:" + "a" * 32,
        source_scope="s:" + "b" * 32,
    )
    for rendered in (repr(record), str(record), f"{record}"):
        assert "u:" not in rendered
        assert "s:" not in rendered
        assert ANCHOR not in rendered
        assert "redacted" in rendered


def test_entity_scope_record_requires_an_anchor() -> None:
    with pytest.raises(ValidationError, match="anchor_event_id"):
        EntityScopeRecord(anchor_event_id="  ")


def test_engine_and_scorer_accept_no_entity_scope() -> None:
    """The confinement is structural: no scope parameter exists to pass.

    Asserted against the whole detection package so a later milestone cannot
    thread a scope argument into evaluation without this test failing.
    """
    import ast
    from pathlib import Path

    package = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "password_attack_detector"
        / "detection"
    )
    alerting_only = {"alerts.py", "cli.py", "schemas.py", "serialization.py"}
    offenders: list[str] = []
    for source in sorted(package.rglob("*.py")):
        if source.name in alerting_only:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            arguments = [
                argument.arg
                for argument in (
                    *node.args.args,
                    *node.args.kwonlyargs,
                    *node.args.posonlyargs,
                )
            ]
            offenders.extend(
                f"{source.name}:{node.name}({name})"
                for name in arguments
                if "scope" in name and "scope_dimension" not in name
            )
    assert offenders == []


# ---------------------------------------------------------------------------
# Validation result structures
# ---------------------------------------------------------------------------


def test_validation_finding_requires_a_dnnn_code() -> None:
    with pytest.raises(ValidationError, match="D<nnn>"):
        DetectionValidationFinding(code="F001", message="wrong prefix")


def test_validation_finding_rejects_an_identifier_in_the_message() -> None:
    with pytest.raises(ValidationError, match="looks like an identifier"):
        DetectionValidationFinding(code="D007", message=ANCHOR)


def test_validation_result_status_follows_the_findings() -> None:
    error = DetectionValidationFinding(code="D013", message="3 values out of range")
    with pytest.raises(ValidationError, match="must be invalid"):
        DetectionValidationResult(
            status=DetectionValidationStatus.VALID, errors=(error,)
        )
    with pytest.raises(ValidationError, match="must be a warning"):
        DetectionValidationResult(
            status=DetectionValidationStatus.VALID, warnings=(error,)
        )

    result = DetectionValidationResult(
        status=DetectionValidationStatus.INVALID, errors=(error,)
    )
    assert result.passed is False
    assert DetectionValidationResult(status=DetectionValidationStatus.VALID).passed


def test_published_detection_requires_evidence_and_reasons() -> None:
    base: dict[str, Any] = {
        "detection_id": detection_identifier(ANCHOR, "PAD-BF-001", "1.0.0"),
        "anchor_event_id": ANCHOR,
        "anchor_event_time": WHEN,
        "rule_id": "PAD-BF-001",
        "rule_version": "1.0.0",
        "rule_family": RuleFamily.BRUTE_FORCE,
        "attack_category": AttackCategory.BRUTE_FORCE,
        "severity": Severity.HIGH,
        "signal_strength": 0.5,
        "correlation_group": CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
        "evidence": (_evidence(),),
        "reason_codes": ("BF_PAIR_FAILURE_COUNT",),
    }
    with pytest.raises(ValidationError, match="must carry evidence"):
        FiredDetection(**{**base, "evidence": ()})
    with pytest.raises(ValidationError, match="must carry reason codes"):
        FiredDetection(**{**base, "reason_codes": ()})


def test_zero_fired_rules_rejects_a_primary_category() -> None:
    with pytest.raises(ValidationError, match="has no primary category"):
        RiskAssessment(
            anchor_event_id=ANCHOR,
            anchor_event_time=WHEN,
            risk_score=0.0,
            severity=Severity.LOW,
            primary_attack_category=AttackCategory.BRUTE_FORCE,
            contributing_categories=(AttackCategory.BRUTE_FORCE,),
            scoring_version="1.0.0",
        )


def test_zero_fired_rules_rejects_categories_and_evidence() -> None:
    with pytest.raises(ValidationError, match="carries no categories or"):
        RiskAssessment(
            anchor_event_id=ANCHOR,
            anchor_event_time=WHEN,
            risk_score=0.0,
            severity=Severity.LOW,
            contributing_categories=(AttackCategory.BRUTE_FORCE,),
            scoring_version="1.0.0",
        )


def test_fired_assessment_needs_a_primary_category() -> None:
    with pytest.raises(ValidationError, match="needs a primary category"):
        _assessment(primary_attack_category=None)


def test_contributing_categories_must_be_sorted() -> None:
    with pytest.raises(ValidationError, match="contributing_categories must be sorted"):
        _assessment(
            fired_rule_count=2,
            fired_rule_ids=("PAD-BF-001", "PAD-BOT-001"),
            primary_attack_category=AttackCategory.BRUTE_FORCE,
            contributing_categories=(
                AttackCategory.BRUTE_FORCE,
                AttackCategory.BOT_ACTIVITY,
            ),
        )
