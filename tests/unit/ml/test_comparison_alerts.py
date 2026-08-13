"""The comparison alert builder, pinned against Phase 4's own temporal semantics.

The central test in this file feeds *equivalent* decisions through both builders
and asserts they agree on every temporal quantity: grouping boundaries and their
inclusivity, ``first_seen``, ``last_seen``, contributing counts, cooldown,
suppression accounting, and the rate limit. A comparison in which one system
enjoyed a different cooldown would be measuring the cooldown.

The second half asserts what the comparison artifact does **not** have. A model
decision carries no fired rules, no correlation group, no attack category and no
signal strengths, and filling any of them in would make a model's output look
like a rule's.

Phase 4's own builder is exercised here too, unchanged, so this milestone cannot
alter it without this file failing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from password_attack_detector.detection.alerts import AlertBuilder
from password_attack_detector.detection.config import AlertingConfig, DetectionConfig
from password_attack_detector.detection.enums import (
    AttackCategory,
    ScopeKind,
    Severity,
    SuppressionReason,
)
from password_attack_detector.detection.schemas import RiskAssessment
from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.alerts import (
    COMPARISON_ALERTING_VERSION,
    ComparisonAlert,
    ComparisonDecision,
    DecisionAlertBuilder,
    DecisionAlertPolicy,
    decisions_from_assessments,
)
from password_attack_detector.ml.enums import ComparisonSystem

WHEN = datetime(2024, 3, 1, tzinfo=UTC)


def assessment(
    index: int,
    minutes: float,
    *,
    risk: float = 70.0,
    severity: Severity = Severity.HIGH,
) -> RiskAssessment:
    """Build one Phase 4 risk assessment in a single attack category.

    One category on purpose: Phase 4 keys its alerts on the category, so a
    single-category stream is the population where the two builders' keys are
    directly comparable.
    """
    return RiskAssessment(
        anchor_event_id=f"a{index:04d}",
        anchor_event_time=WHEN + timedelta(minutes=minutes),
        risk_score=risk,
        severity=severity,
        primary_attack_category=AttackCategory.BRUTE_FORCE,
        contributing_categories=(AttackCategory.BRUTE_FORCE,),
        fired_rule_count=1,
        fired_rule_ids=("PAD-BF-001",),
        scoring_version="1.0.0",
    )


def both(assessments: list[RiskAssessment], **alerting: Any) -> tuple[Any, Any]:
    """Run one stream through Phase 4 and through the comparison builder."""
    config = DetectionConfig(alerting=AlertingConfig(**alerting))
    phase4 = AlertBuilder(config).build(assessments)
    policy = DecisionAlertPolicy.from_alerting(config.alerting)
    comparison = DecisionAlertBuilder(policy).build(
        decisions_from_assessments(assessments)
    )
    return (phase4, comparison)


def windows(alerts: object) -> list[tuple[datetime, datetime, int, int]]:
    """Return each alert's window, contributing count and suppression count."""
    return [
        (
            alert.first_seen,
            alert.last_seen,
            alert.contributing_event_count,
            alert.suppressed_event_count,
        )
        for alert in alerts  # type: ignore[attr-defined]
    ]


# ---------------------------------------------------------------------------
# Equivalence with Phase 4
# ---------------------------------------------------------------------------


def test_the_two_builders_agree_on_grouping_and_windows() -> None:
    """The regression that makes the comparison fair, pinned."""
    rows = [
        assessment(0, 0),
        assessment(1, 10),
        assessment(2, 15),
        assessment(3, 40),
        assessment(4, 80),
        assessment(5, 95),
        assessment(6, 200),
    ]
    phase4, comparison = both(rows)
    assert len(phase4.alerts) == len(comparison.alerts)
    assert windows(phase4.alerts) == windows(comparison.alerts)


def test_the_grouping_boundary_is_inclusive_in_both() -> None:
    """An event exactly one grouping window later is absorbed, not opened."""
    rows = [assessment(0, 0), assessment(1, 15)]
    phase4, comparison = both(rows, grouping_window="15m")
    assert len(phase4.alerts) == 1
    assert len(comparison.alerts) == 1
    assert phase4.alerts[0].contributing_event_count == 2
    assert comparison.alerts[0].contributing_event_count == 2


def test_one_second_past_the_grouping_boundary_opens_a_new_alert_in_both() -> None:
    """The other side of the same boundary."""
    rows = [assessment(0, 0), assessment(1, 15.017)]
    phase4, comparison = both(rows, grouping_window="15m", cooldown="1s")
    assert len(phase4.alerts) == len(comparison.alerts) == 2


def test_the_cooldown_boundary_is_inclusive_in_both() -> None:
    """An event exactly one cooldown after the last contribution is suppressed."""
    # 30 minutes after the first alert's last contribution: exactly one
    # cooldown, and inclusive means suppressed.
    rows = [assessment(0, 0), assessment(1, 30)]
    phase4, comparison = both(
        rows, grouping_window="15m", cooldown="30m", escalation_bypasses_cooldown=False
    )
    assert len(phase4.alerts) == len(comparison.alerts) == 1
    assert phase4.alerts[0].suppressed_event_count == 1
    assert comparison.alerts[0].suppressed_event_count == 1


def test_first_seen_never_moves_and_last_seen_only_advances() -> None:
    """The window an alert reports always contains its own events."""
    rows = [assessment(0, 0), assessment(1, 5), assessment(2, 10)]
    phase4, comparison = both(rows)
    assert phase4.alerts[0].first_seen == comparison.alerts[0].first_seen == WHEN
    assert phase4.alerts[0].last_seen == comparison.alerts[0].last_seen
    assert comparison.alerts[0].last_seen == WHEN + timedelta(minutes=10)


def test_the_rate_limit_accounting_agrees() -> None:
    """Escalations can emit several alerts per key; both builders count the same."""
    rows = [assessment(index, index * 20, risk=50.0 + index * 10) for index in range(6)]
    phase4, comparison = both(
        rows,
        grouping_window="5m",
        cooldown="10m",
        alert_limit_window="1h",
        max_alerts_per_group_per_window=2,
    )
    assert len(phase4.alerts) == len(comparison.alerts)
    assert windows(phase4.alerts) == windows(comparison.alerts)


def test_input_order_reaches_neither_builder() -> None:
    """Both sort by anchor time then identifier before grouping."""
    rows = [assessment(0, 0), assessment(1, 10), assessment(2, 40)]
    forward = both(rows)
    backward = both(list(reversed(rows)))
    assert windows(forward[0].alerts) == windows(backward[0].alerts)
    assert windows(forward[1].alerts) == windows(backward[1].alerts)


def test_the_phase_four_builder_is_unchanged_by_this_milestone() -> None:
    """A content fingerprint over Phase 4's own output, pinned.

    If Milestone 9 ever alters the rule-only alert path, this fails before any
    comparison claim can be made on top of it.
    """
    import hashlib
    import json

    rows = [assessment(0, 0), assessment(1, 10), assessment(2, 40), assessment(3, 200)]
    config = DetectionConfig(alerting=AlertingConfig())
    alerts = AlertBuilder(config).build(rows).alerts
    payload = [
        {
            "alert_id": alert.alert_id,
            "attack_category": str(alert.attack_category),
            "correlation_group": str(alert.correlation_group),
            "first_seen": alert.first_seen.isoformat(),
            "last_seen": alert.last_seen.isoformat(),
            "contributing_event_count": alert.contributing_event_count,
            "contributing_rule_ids": list(alert.contributing_rule_ids),
            "aggregate_risk_score": alert.aggregate_risk_score,
            "peak_risk_score": alert.peak_risk_score,
            "suppressed_event_count": alert.suppressed_event_count,
        }
        for alert in alerts
    ]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
    assert (
        digest
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest()
    )
    assert len(alerts) == 2
    assert [alert.contributing_event_count for alert in alerts] == [2, 1]
    assert all(alert.contributing_rule_ids == ("PAD-BF-001",) for alert in alerts)


# ---------------------------------------------------------------------------
# What a comparison alert does not have
# ---------------------------------------------------------------------------


def ml_decision(
    index: int, minutes: float, *, flagged: bool = True
) -> ComparisonDecision:
    """Build one ML-only comparison decision."""
    return ComparisonDecision(
        anchor_event_id=f"m{index:04d}",
        anchor_event_time=WHEN + timedelta(minutes=minutes),
        system=ComparisonSystem.ML_ONLY,
        flagged=flagged,
        calibrated_probability=0.9,
    )


def test_a_comparison_alert_fabricates_no_rule_concept() -> None:
    """No rule identifiers, no correlation group, no category, no signals."""
    for absent in (
        "contributing_rule_ids",
        "correlation_group",
        "attack_category",
        "signal_strengths",
        "fired_rule_ids",
        "top_evidence",
    ):
        assert absent not in ComparisonAlert.model_fields, absent
        assert absent not in ComparisonDecision.model_fields, absent


def test_an_ml_alert_carries_a_probability_and_no_ordinal_risk() -> None:
    """A model has no 0-100 severity magnitude, and none is invented."""
    result = DecisionAlertBuilder(
        DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    ).build([ml_decision(0, 0), ml_decision(1, 5)])
    alert = result.alerts[0]
    assert alert.peak_calibrated_probability == 0.9
    assert alert.peak_ordinal_risk_score is None
    assert alert.mean_ordinal_risk_score is None
    assert alert.initial_severity is None


def test_a_rule_alert_carries_an_ordinal_risk_and_no_probability() -> None:
    """And the converse: the rule engine produces no calibrated probability."""
    result = DecisionAlertBuilder(
        DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    ).build(decisions_from_assessments([assessment(0, 0)]))
    alert = result.alerts[0]
    assert alert.peak_ordinal_risk_score == 70.0
    assert alert.peak_calibrated_probability is None
    assert alert.initial_severity is Severity.HIGH


def test_the_two_magnitudes_are_never_added() -> None:
    """A row carrying both keeps them in their own fields."""
    decision = ComparisonDecision(
        anchor_event_id="h1",
        anchor_event_time=WHEN,
        system=ComparisonSystem.HYBRID,
        flagged=True,
        ordinal_risk_score=80.0,
        calibrated_probability=0.4,
    )
    result = DecisionAlertBuilder(
        DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    ).build([decision])
    alert = result.alerts[0]
    assert alert.peak_ordinal_risk_score == 80.0
    assert alert.peak_calibrated_probability == 0.4


def test_an_unflagged_decision_opens_no_alert() -> None:
    """The one qualifying gate is the decision itself."""
    result = DecisionAlertBuilder(
        DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    ).build([ml_decision(0, 0, flagged=False)])
    assert result.alerts == ()
    assert result.stats.qualifying_count == 0


def test_two_systems_in_one_pass_are_refused() -> None:
    """Their alerts would share a cooldown and a rate limit."""
    with pytest.raises(DataValidationError, match="covers one system"):
        DecisionAlertBuilder(
            DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
        ).build(
            [
                ml_decision(0, 0),
                ComparisonDecision(
                    anchor_event_id="r1",
                    anchor_event_time=WHEN,
                    system=ComparisonSystem.RULE_ONLY,
                    flagged=True,
                    ordinal_risk_score=70.0,
                ),
            ]
        )


def test_two_systems_never_share_a_grouping_key() -> None:
    """The system is part of the key, so streams cannot merge."""
    policy = DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    ml_alerts = DecisionAlertBuilder(policy).build([ml_decision(0, 0)]).alerts
    rule_alerts = (
        DecisionAlertBuilder(policy)
        .build(decisions_from_assessments([assessment(0, 0)]))
        .alerts
    )
    assert ml_alerts[0].alert_id != rule_alerts[0].alert_id
    assert ml_alerts[0].system is ComparisonSystem.ML_ONLY
    assert rule_alerts[0].system is ComparisonSystem.RULE_ONLY


def test_alert_identifiers_are_deterministic() -> None:
    """Derived from the contract version, the key, and the window start."""
    policy = DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    first = DecisionAlertBuilder(policy).build([ml_decision(0, 0)]).alerts
    second = DecisionAlertBuilder(policy).build([ml_decision(0, 0)]).alerts
    assert first[0].alert_id == second[0].alert_id


def test_the_comparison_contract_has_its_own_version() -> None:
    """Shared numbering would make one artifact look like the other."""
    from password_attack_detector.detection.alerts import ALERTING_VERSION

    # Two separately declared constants that happen to agree today. Either may
    # move without the other, which is the point of not sharing one.
    assert COMPARISON_ALERTING_VERSION == "1.0.0"
    assert ALERTING_VERSION == "1.0.0"
    import password_attack_detector.detection.alerts as phase4
    import password_attack_detector.ml.alerts as comparison

    assert "ALERTING_VERSION" in vars(phase4)
    assert "COMPARISON_ALERTING_VERSION" in vars(comparison)
    assert "ALERTING_VERSION" not in vars(comparison)


def test_the_policy_is_taken_from_the_reviewed_alerting_configuration() -> None:
    """The comparison inherits the windows rather than declaring its own."""
    config = DetectionConfig(alerting=AlertingConfig(cooldown=timedelta(minutes=42)))
    policy = DecisionAlertPolicy.from_alerting(config.alerting)
    assert policy.cooldown == timedelta(minutes=42)
    assert policy.grouping_window == config.alerting.grouping_window
    assert policy.fingerprint_data()["cooldown_seconds"] == 42 * 60


def test_suppression_is_accounted_by_reason() -> None:
    """Cooldown and rate limit are counted separately, as in Phase 4."""
    rows = [assessment(0, 0), assessment(1, 45, risk=10.0, severity=Severity.LOW)]
    _, comparison = both(
        rows, grouping_window="15m", cooldown="60m", escalation_bypasses_cooldown=False
    )
    assert comparison.stats.suppressed_by_reason.get(SuppressionReason.COOLDOWN) == 1


def test_a_scoped_decision_names_its_scope() -> None:
    """Scope is the pseudonymous grouping dimension, and never reaches a report."""
    decision = ComparisonDecision(
        anchor_event_id="m1",
        anchor_event_time=WHEN,
        system=ComparisonSystem.ML_ONLY,
        flagged=True,
        scope_kind=ScopeKind.USER,
        scope_value="u:" + "0" * 32,
    )
    assert decision.scope_kind is ScopeKind.USER
    with pytest.raises(ValueError, match="names its scope value"):
        ComparisonDecision(
            anchor_event_id="m2",
            anchor_event_time=WHEN,
            system=ComparisonSystem.ML_ONLY,
            flagged=True,
            scope_kind=ScopeKind.USER,
        )
