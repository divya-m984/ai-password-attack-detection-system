"""Thresholds and discriminators for PAD-PS-001."""

from __future__ import annotations

import math

import pytest

from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories
from tests.unit.detection.factories import spraying_row


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def rule(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-PS-001"].prepare(DetectionConfig(), catalog)


def test_fires_at_exact_threshold_equality(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = spraying_row(
        catalog,
        source_unique_user_count__1h=15,
        source_failure_count__1h=15,
        source_failure_rate__1h=0.70,
        source_attempt_count__1h=45,  # exactly 3.0 attempts per account
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.FIRED
    assert result.signal_strength > 0.0


@pytest.mark.parametrize(
    ("column", "value"),
    [("source_unique_user_count__1h", 14), ("source_failure_count__1h", 14)],
)
def test_one_step_below_an_integer_threshold_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog, column: str, value: int
) -> None:
    overrides = {
        "source_unique_user_count__1h": 15,
        "source_failure_count__1h": 15,
        "source_attempt_count__1h": 40,
        column: value,
    }
    assert rule.evaluate(spraying_row(catalog, **overrides)).status is (
        RuleStatus.NOT_FIRED
    )


def test_one_representable_value_below_the_rate_threshold_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = spraying_row(catalog, source_failure_rate__1h=math.nextafter(0.70, 0.0))
    assert rule.evaluate(row).status is RuleStatus.NOT_FIRED


def test_single_account_brute_force_does_not_fire_spraying(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """The low attempts-per-account ceiling is the whole discriminator."""
    result = rule.evaluate(
        spraying_row(
            catalog,
            source_unique_user_count__1h=1,
            source_attempt_count__1h=200,
            source_failure_count__1h=198,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_ACCOUNT_FANOUT_THRESHOLD" in result.reason_codes


def test_high_attempts_per_account_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        spraying_row(
            catalog, source_unique_user_count__1h=20, source_attempt_count__1h=400
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "ATTEMPTS_PER_ACCOUNT_TOO_HIGH" in result.reason_codes


def test_the_attempts_per_account_ceiling_is_inclusive(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    at = spraying_row(
        catalog, source_unique_user_count__1h=20, source_attempt_count__1h=60
    )
    above = spraying_row(
        catalog, source_unique_user_count__1h=20, source_attempt_count__1h=61
    )
    assert rule.evaluate(at).status is RuleStatus.FIRED
    assert rule.evaluate(above).status is RuleStatus.NOT_FIRED


def test_broad_legitimate_activity_without_failures_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A busy shared gateway is fan-out without failure evidence."""
    result = rule.evaluate(
        factories.quiet_row(
            catalog,
            source_unique_user_count__1h=200,
            source_attempt_count__1h=400,
            source_failure_count__1h=3,
            source_failure_rate__1h=0.0075,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_SOURCE_FAILURE_COUNT" in result.reason_codes
    assert "BELOW_SOURCE_FAILURE_RATE" in result.reason_codes


def test_a_null_failure_rate_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(spraying_row(catalog, source_failure_rate__1h=None))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("INSUFFICIENT_HISTORY_SOURCE_FAILURE_RATE__1H",)


def test_no_observed_accounts_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """An undefined attempts-per-account ratio is unseen, not clean."""
    result = rule.evaluate(
        spraying_row(
            catalog, source_unique_user_count__1h=0, source_attempt_count__1h=0
        )
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("NO_TARGETED_ACCOUNTS_OBSERVED",)


def test_stronger_fanout_never_lowers_signal_strength(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    weaker = rule.evaluate(
        spraying_row(
            catalog,
            source_unique_user_count__1h=15,
            source_failure_count__1h=15,
            source_failure_rate__1h=0.70,
            source_attempt_count__1h=45,
        )
    )
    stronger = rule.evaluate(
        spraying_row(
            catalog,
            source_unique_user_count__1h=150,
            source_failure_count__1h=150,
            source_failure_rate__1h=1.0,
            source_attempt_count__1h=150,
        )
    )
    assert stronger.signal_strength > weaker.signal_strength


def test_the_cadence_component_is_disabled_by_default(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        spraying_row(catalog, source_mean_interarrival_seconds__1h=900.0)
    )
    assert "PS_SOURCE_CADENCE" not in result.reason_codes


def test_a_configured_cadence_floor_adds_supporting_evidence(
    catalog: FeatureCatalog,
) -> None:
    config = DetectionConfig(
        rules={
            "PAD-PS-001": RuleSettings(
                parameters={"min_mean_interarrival_seconds": 60.0}
            )
        }
    )
    rule = RULE_IMPLEMENTATIONS["PAD-PS-001"].prepare(config, catalog)

    slow = rule.evaluate(
        spraying_row(catalog, source_mean_interarrival_seconds__1h=900.0)
    )
    assert slow.status is RuleStatus.FIRED
    assert "PS_SOURCE_CADENCE" in slow.reason_codes

    # Below the floor, and null, both remove the contribution rather than the
    # detection: the component supports, it never gates.
    for value in (10.0, None):
        result = rule.evaluate(
            spraying_row(catalog, source_mean_interarrival_seconds__1h=value)
        )
        assert result.status is RuleStatus.FIRED
        assert "PS_SOURCE_CADENCE" not in result.reason_codes
