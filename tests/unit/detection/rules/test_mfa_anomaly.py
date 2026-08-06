"""Gates, thresholds, and wording for PAD-MFA-001."""

from __future__ import annotations

import pytest

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.enums import CorrelationGroup, RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories
from tests.unit.detection.factories import mfa_row


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def rule(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-MFA-001"].prepare(DetectionConfig(), catalog)


def test_fires_at_exact_threshold_equality(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = mfa_row(
        catalog,
        user_attempt_count__15m=6,
        user_mfa_failure_count__15m=4,
        user_challenge_count__15m=0,
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.FIRED
    assert "MFA_PRIOR_FAILURES" in result.reason_codes
    assert "MFA_PRIOR_CHALLENGES" not in result.reason_codes


# ---------------------------------------------------------------------------
# The minimum-history gate runs first
# ---------------------------------------------------------------------------


def test_one_ordinary_challenge_is_insufficient_data_not_a_negative(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """The case the rule exists to *not* report."""
    result = rule.evaluate(
        mfa_row(
            catalog,
            user_attempt_count__15m=1,
            user_mfa_failure_count__15m=0,
            user_challenge_count__15m=1,
            current_mfa_outcome="passed",
        )
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("MFA_INSUFFICIENT_ATTEMPT_HISTORY",)
    assert result.evidence == ()


def test_below_the_attempt_history_minimum_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(mfa_row(catalog, user_attempt_count__15m=5))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("MFA_INSUFFICIENT_ATTEMPT_HISTORY",)


def test_below_the_observation_minimum_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        mfa_row(
            catalog,
            user_attempt_count__15m=20,
            user_mfa_failure_count__15m=1,
            user_challenge_count__15m=1,
        )
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("MFA_INSUFFICIENT_OBSERVATIONS",)


def test_the_history_gates_are_inclusive(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    at_gate = mfa_row(
        catalog,
        user_attempt_count__15m=6,
        user_mfa_failure_count__15m=2,
        user_challenge_count__15m=4,
    )
    assert rule.evaluate(at_gate).status is RuleStatus.FIRED


def test_an_unrecorded_mfa_outcome_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(mfa_row(catalog, current_mfa_outcome=None))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("MFA_OUTCOME_NOT_RECORDED",)


# ---------------------------------------------------------------------------
# Both conditions are required
# ---------------------------------------------------------------------------


def test_elevated_prior_activity_with_a_normal_outcome_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(mfa_row(catalog, current_mfa_outcome="passed"))
    assert result.status is RuleStatus.NOT_FIRED
    assert "ANCHOR_MFA_OUTCOME_NOT_ABNORMAL" in result.reason_codes


def test_an_abnormal_outcome_without_elevated_activity_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        mfa_row(
            catalog,
            user_attempt_count__15m=20,
            user_mfa_failure_count__15m=2,
            user_challenge_count__15m=2,
            current_mfa_outcome="failed",
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "PRIOR_MFA_ACTIVITY_NOT_ELEVATED" in result.reason_codes


@pytest.mark.parametrize("outcome", ["failed", "bypassed"])
def test_both_abnormal_outcomes_satisfy_the_condition(
    rule: PreparedRule, catalog: FeatureCatalog, outcome: str
) -> None:
    assert rule.evaluate(mfa_row(catalog, current_mfa_outcome=outcome)).status is (
        RuleStatus.FIRED
    )


@pytest.mark.parametrize("outcome", ["passed", "not_required", "not_enrolled"])
def test_an_ordinary_outcome_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog, outcome: str
) -> None:
    """``not_required`` and ``not_enrolled`` describe a policy, not an anomaly."""
    assert rule.evaluate(mfa_row(catalog, current_mfa_outcome=outcome)).status is (
        RuleStatus.NOT_FIRED
    )


def test_either_elevated_signal_alone_satisfies_the_condition(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    failures_only = rule.evaluate(
        mfa_row(catalog, user_mfa_failure_count__15m=4, user_challenge_count__15m=0)
    )
    challenges_only = rule.evaluate(
        mfa_row(catalog, user_mfa_failure_count__15m=0, user_challenge_count__15m=4)
    )
    assert failures_only.status is RuleStatus.FIRED
    assert challenges_only.status is RuleStatus.FIRED


def test_one_below_both_elevation_thresholds_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        mfa_row(catalog, user_mfa_failure_count__15m=3, user_challenge_count__15m=3)
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "PRIOR_MFA_ACTIVITY_NOT_ELEVATED" in result.reason_codes


def test_stronger_activity_never_lowers_signal_strength(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    weaker = rule.evaluate(
        mfa_row(
            catalog,
            user_attempt_count__15m=6,
            user_mfa_failure_count__15m=4,
            user_challenge_count__15m=4,
        )
    )
    stronger = rule.evaluate(
        mfa_row(
            catalog,
            user_attempt_count__15m=90,
            user_mfa_failure_count__15m=40,
            user_challenge_count__15m=40,
        )
    )
    assert stronger.signal_strength > weaker.signal_strength


def test_a_stricter_history_gate_is_configurable(catalog: FeatureCatalog) -> None:
    config = DetectionConfig(
        rules={
            "PAD-MFA-001": RuleSettings(
                parameters={"min_mfa_history_events": 50, "min_mfa_observations": 25}
            )
        }
    )
    rule = RULE_IMPLEMENTATIONS["PAD-MFA-001"].prepare(config, catalog)
    result = rule.evaluate(mfa_row(catalog))
    assert result.status is RuleStatus.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# Naming and correlation
# ---------------------------------------------------------------------------


def test_the_rule_shares_the_session_anomaly_group_with_the_takeover_rule() -> None:
    assert (
        RULE_CATALOG.get("PAD-MFA-001").correlation_group
        is RULE_CATALOG.get("PAD-ATO-001").correlation_group
        is CorrelationGroup.SESSION_ANOMALY
    )


def test_the_rule_is_named_a_sequence_anomaly_not_mfa_fatigue() -> None:
    spec = RULE_CATALOG.get("PAD-MFA-001")
    assert "sequence anomaly" in spec.display_name.lower()
    assert "fatigue" not in spec.display_name.lower()
    assert "fatigue" not in spec.description.lower()
    assert "defeated" not in spec.description.lower()


def test_evidence_never_asserts_a_control_was_defeated(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(mfa_row(catalog))
    for item in result.evidence:
        lowered = item.message.lower()
        assert "defeated" not in lowered
        assert "bypassed the control" not in lowered
