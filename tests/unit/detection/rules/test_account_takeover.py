"""Thresholds and discriminators for PAD-ATO-001."""

from __future__ import annotations

import math

import pytest

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories
from tests.unit.detection.factories import takeover_row

NOVELTY_COLUMNS = (
    "is_new_device_for_user",
    "is_new_source_for_user",
    "is_new_country_for_user",
    "is_new_application_for_user",
    "is_new_auth_method_for_user",
)


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def rule(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-ATO-001"].prepare(DetectionConfig(), catalog)


def test_fires_at_exact_threshold_equality(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Two novel contexts and one supporting signal, each exactly at its floor."""
    row = takeover_row(catalog, prior_failures_since_user_success=3)
    result = rule.evaluate(row)
    assert result.status is RuleStatus.FIRED
    assert result.signal_strength > 0.0


def test_novelty_alone_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Travel with a new laptop produces novelty and nothing else."""
    row = takeover_row(
        catalog,
        is_new_device_for_user=True,
        is_new_source_for_user=True,
        is_new_country_for_user=True,
        is_new_application_for_user=True,
        is_new_auth_method_for_user=True,
        prior_failures_since_user_success=0,
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.NOT_FIRED
    assert result.reason_codes == ("NO_SUPPORTING_BEHAVIOURAL_DEVIATION",)


def test_a_supporting_signal_alone_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = takeover_row(
        catalog,
        is_new_device_for_user=False,
        is_new_country_for_user=False,
        prior_failures_since_user_success=50,
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_NOVEL_CONTEXT_THRESHOLD" in result.reason_codes


def test_one_novel_context_below_the_threshold_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = takeover_row(catalog, is_new_country_for_user=False)
    result = rule.evaluate(row)
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_NOVEL_CONTEXT_THRESHOLD" in result.reason_codes


def test_an_ordinary_successful_login_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = factories.quiet_row(catalog, current_authentication_outcome="success")
    assert rule.evaluate(row).status is RuleStatus.NOT_FIRED


def test_a_failed_authentication_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        takeover_row(catalog, current_authentication_outcome="failure")
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert result.reason_codes == ("ANCHOR_DID_NOT_SUCCEED",)


def test_an_account_absent_from_the_baseline_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A cold account's first device is not a *new* device."""
    result = rule.evaluate(takeover_row(catalog, user_in_baseline=False))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("ACCOUNT_ABSENT_FROM_BASELINE",)
    assert result.evidence == ()


def test_a_null_novelty_flag_counts_as_not_novel(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    overrides = dict.fromkeys(NOVELTY_COLUMNS, None)
    result = rule.evaluate(takeover_row(catalog, **overrides))
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_NOVEL_CONTEXT_THRESHOLD" in result.reason_codes


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("prior_failures_since_user_success", 3, "ATO_PRIOR_FAILURES"),
        ("login_hour_deviation", 0.90, "ATO_LOGIN_HOUR_DEVIATION"),
        ("response_time_zscore", 3.0, "ATO_RESPONSE_TIME_DEVIATION"),
        ("user_event_rate_ratio", 3.0, "ATO_EVENT_RATE_DEVIATION"),
        ("current_mfa_outcome", "failed", "ATO_MFA_DEVIATION"),
        ("distance_from_user_baseline_centroid_km", 1000.0, "ATO_BASELINE_DISTANCE"),
    ],
)
def test_each_supporting_signal_can_satisfy_the_requirement_alone(
    rule: PreparedRule,
    catalog: FeatureCatalog,
    column: str,
    value: object,
    code: str,
) -> None:
    overrides = {"prior_failures_since_user_success": 0, column: value}
    result = rule.evaluate(takeover_row(catalog, **overrides))
    assert result.status is RuleStatus.FIRED
    assert code in result.reason_codes


def test_a_negative_response_time_deviation_counts(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A response far faster than baseline deviates as much as one far slower."""
    result = rule.evaluate(
        takeover_row(
            catalog, prior_failures_since_user_success=0, response_time_zscore=-4.0
        )
    )
    assert result.status is RuleStatus.FIRED
    item = next(
        evidence
        for evidence in result.evidence
        if evidence.evidence_code == "ATO_RESPONSE_TIME_DEVIATION"
    )
    assert item.observed_value == pytest.approx(4.0)


@pytest.mark.parametrize("outcome", ["passed", "not_required", "not_enrolled"])
def test_an_ordinary_mfa_outcome_is_not_a_supporting_signal(
    rule: PreparedRule, catalog: FeatureCatalog, outcome: str
) -> None:
    result = rule.evaluate(
        takeover_row(
            catalog, prior_failures_since_user_success=0, current_mfa_outcome=outcome
        )
    )
    assert result.status is RuleStatus.NOT_FIRED


def test_one_representable_value_below_a_supporting_threshold_does_not_count(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        takeover_row(
            catalog,
            prior_failures_since_user_success=0,
            login_hour_deviation=math.nextafter(0.90, 0.0),
        )
    )
    assert result.status is RuleStatus.NOT_FIRED


def test_null_optional_features_contribute_nothing(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A missing baseline dimension is not evidence of anything."""
    result = rule.evaluate(
        takeover_row(
            catalog,
            prior_failures_since_user_success=0,
            login_hour_deviation=None,
            response_time_zscore=None,
            user_event_rate_ratio=None,
            current_mfa_outcome=None,
            distance_from_user_baseline_centroid_km=None,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED


def test_more_supporting_signals_never_lower_signal_strength(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    one = rule.evaluate(takeover_row(catalog, prior_failures_since_user_success=3))
    several = rule.evaluate(
        takeover_row(
            catalog,
            prior_failures_since_user_success=30,
            login_hour_deviation=1.0,
            response_time_zscore=9.0,
            user_event_rate_ratio=12.0,
            current_mfa_outcome="failed",
            distance_from_user_baseline_centroid_km=9000.0,
        )
    )
    assert several.signal_strength > one.signal_strength


def test_more_novel_contexts_never_lower_signal_strength(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    two = rule.evaluate(takeover_row(catalog, prior_failures_since_user_success=3))
    five = rule.evaluate(
        takeover_row(
            catalog,
            prior_failures_since_user_success=3,
            **dict.fromkeys(NOVELTY_COLUMNS, True),
        )
    )
    assert five.signal_strength > two.signal_strength


def test_a_stricter_configuration_withholds_the_finding(
    catalog: FeatureCatalog,
) -> None:
    config = DetectionConfig(
        rules={
            "PAD-ATO-001": RuleSettings(
                parameters={"min_novel_context_count": 4, "min_supporting_signals": 3}
            )
        }
    )
    rule = RULE_IMPLEMENTATIONS["PAD-ATO-001"].prepare(config, catalog)
    result = rule.evaluate(takeover_row(catalog, prior_failures_since_user_success=3))
    assert result.status is RuleStatus.NOT_FIRED


def test_the_rule_is_named_and_described_as_an_indicator() -> None:
    spec = RULE_CATALOG.get("PAD-ATO-001")
    assert "indicator" in spec.display_name.lower()
    assert "indicator" in str(spec.attack_category)
    assert "confirmed" not in spec.description.lower()


def test_evidence_never_asserts_a_takeover_occurred(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(takeover_row(catalog))
    for item in result.evidence:
        lowered = item.message.lower()
        assert "was taken over" not in lowered
        assert "compromised" not in lowered
