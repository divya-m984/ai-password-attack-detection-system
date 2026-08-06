"""Thresholds and discriminators for PAD-DBF-001."""

from __future__ import annotations

import math

import pytest

from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories
from tests.unit.detection.factories import distributed_row


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def rule(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-DBF-001"].prepare(DetectionConfig(), catalog)


def test_fires_at_exact_threshold_equality(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = distributed_row(
        catalog,
        user_unique_source_count__1h=8,
        user_failure_count__1h=20,
        user_failure_rate__1h=0.80,
        pair_attempt_count__1h=5,
        source_unique_user_count__1h=3,
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.FIRED
    assert result.signal_strength > 0.0


@pytest.mark.parametrize(
    ("column", "value"),
    [("user_unique_source_count__1h", 7), ("user_failure_count__1h", 19)],
)
def test_one_step_below_an_integer_threshold_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog, column: str, value: int
) -> None:
    overrides = {
        "user_unique_source_count__1h": 8,
        "user_failure_count__1h": 20,
        column: value,
    }
    assert rule.evaluate(distributed_row(catalog, **overrides)).status is (
        RuleStatus.NOT_FIRED
    )


def test_one_representable_value_below_the_rate_threshold_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = distributed_row(catalog, user_failure_rate__1h=math.nextafter(0.80, 0.0))
    assert rule.evaluate(row).status is RuleStatus.NOT_FIRED


def test_one_high_volume_source_does_not_fire_distributed(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """The per-source ceiling is the distribution discriminator."""
    result = rule.evaluate(distributed_row(catalog, pair_attempt_count__1h=60))
    assert result.status is RuleStatus.NOT_FIRED
    assert "PER_SOURCE_VOLUME_TOO_HIGH" in result.reason_codes


def test_the_per_source_ceiling_is_inclusive(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    assert (
        rule.evaluate(distributed_row(catalog, pair_attempt_count__1h=5)).status
        is RuleStatus.FIRED
    )
    assert (
        rule.evaluate(distributed_row(catalog, pair_attempt_count__1h=6)).status
        is RuleStatus.NOT_FIRED
    )


def test_a_spraying_source_does_not_fire_distributed_brute_force(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(distributed_row(catalog, source_unique_user_count__1h=50))
    assert result.status is RuleStatus.NOT_FIRED
    assert "SOURCE_TARGETS_TOO_MANY_ACCOUNTS" in result.reason_codes


def test_few_sources_do_not_fire(rule: PreparedRule, catalog: FeatureCatalog) -> None:
    result = rule.evaluate(distributed_row(catalog, user_unique_source_count__1h=2))
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_SOURCE_FANOUT_THRESHOLD" in result.reason_codes


def test_a_null_failure_rate_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(distributed_row(catalog, user_failure_rate__1h=None))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("INSUFFICIENT_HISTORY_USER_FAILURE_RATE__1H",)


def test_a_popular_account_with_low_failure_rate_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Many sources is ordinary for a mobile user behind carrier NAT."""
    result = rule.evaluate(
        factories.quiet_row(
            catalog,
            user_unique_source_count__1h=30,
            user_failure_count__1h=2,
            user_failure_rate__1h=0.02,
            pair_attempt_count__1h=1,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED


def test_stronger_evidence_never_lowers_signal_strength(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    weaker = rule.evaluate(
        distributed_row(
            catalog,
            user_unique_source_count__1h=8,
            user_failure_count__1h=20,
            user_failure_rate__1h=0.80,
            pair_attempt_count__1h=5,
        )
    )
    stronger = rule.evaluate(
        distributed_row(
            catalog,
            user_unique_source_count__1h=80,
            user_failure_count__1h=200,
            user_failure_rate__1h=1.0,
            pair_attempt_count__1h=1,
        )
    )
    assert stronger.signal_strength > weaker.signal_strength
