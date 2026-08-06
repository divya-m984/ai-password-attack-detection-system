"""Thresholds and discriminators for PAD-BOT-001."""

from __future__ import annotations

import math

import pytest

from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories
from tests.unit.detection.factories import bot_row


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def rule(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-BOT-001"].prepare(DetectionConfig(), catalog)


def test_fires_at_exact_threshold_equality(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = bot_row(
        catalog,
        source_attempt_count__15m=20,
        source_interarrival_coefficient_of_variation__15m=0.15,
        source_mean_interarrival_seconds__15m=120.0,
        source_unique_user_agent_count__1h=2,
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.FIRED
    # Every component ties, and the rule has no rate component and no enabled
    # optional support, so the strength is exactly the configured floor.
    assert result.signal_strength == pytest.approx(
        DetectionConfig().signal.min_signal_strength, abs=1e-12
    )


def test_a_short_sequence_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Two fast attempts have a dispersion figure, but not a meaningful one."""
    result = rule.evaluate(bot_row(catalog, source_attempt_count__15m=19))
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_ATTEMPT_VOLUME_FLOOR" in result.reason_codes


def test_irregular_high_volume_timing_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Human-like burstiness fails the regularity condition at any volume."""
    result = rule.evaluate(
        bot_row(
            catalog,
            source_attempt_count__15m=5000,
            source_interarrival_coefficient_of_variation__15m=1.8,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "TIMING_TOO_IRREGULAR" in result.reason_codes


def test_one_representable_value_above_the_dispersion_ceiling_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = bot_row(
        catalog,
        source_interarrival_coefficient_of_variation__15m=math.nextafter(
            0.15, math.inf
        ),
    )
    assert rule.evaluate(row).status is RuleStatus.NOT_FIRED


def test_a_long_mean_interval_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        bot_row(catalog, source_mean_interarrival_seconds__15m=3600.0)
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "MEAN_INTERVAL_TOO_LONG" in result.reason_codes


def test_varied_client_characteristics_do_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(bot_row(catalog, source_unique_user_agent_count__1h=12))
    assert result.status is RuleStatus.NOT_FIRED
    assert "CLIENT_CHARACTERISTICS_TOO_VARIED" in result.reason_codes


def test_a_null_coefficient_of_variation_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """The catalog nulls it below two observations: too little timing history."""
    result = rule.evaluate(
        bot_row(catalog, source_interarrival_coefficient_of_variation__15m=None)
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == (
        "INSUFFICIENT_HISTORY_SOURCE_INTERARRIVAL_COEFFICIENT_OF_VARIATION__15M",
    )


def test_a_null_mean_interval_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(bot_row(catalog, source_mean_interarrival_seconds__15m=None))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == (
        "INSUFFICIENT_HISTORY_SOURCE_MEAN_INTERARRIVAL_SECONDS__15M",
    )


def test_perfectly_regular_timing_scores_above_a_marginal_case(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    marginal = rule.evaluate(
        bot_row(
            catalog,
            source_attempt_count__15m=20,
            source_interarrival_coefficient_of_variation__15m=0.15,
            source_mean_interarrival_seconds__15m=120.0,
            source_unique_user_agent_count__1h=2,
        )
    )
    machine = rule.evaluate(
        bot_row(
            catalog,
            source_attempt_count__15m=400,
            source_interarrival_coefficient_of_variation__15m=0.0,
            source_mean_interarrival_seconds__15m=1.0,
            source_unique_user_agent_count__1h=1,
        )
    )
    assert machine.signal_strength > marginal.signal_strength


def test_the_fanout_component_is_disabled_by_default(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(bot_row(catalog, source_unique_user_count__1h=900))
    assert result.status is RuleStatus.FIRED
    assert "BOT_SOURCE_FANOUT" not in result.reason_codes


def test_a_configured_fanout_floor_adds_supporting_evidence(
    catalog: FeatureCatalog,
) -> None:
    config = DetectionConfig(
        rules={"PAD-BOT-001": RuleSettings(parameters={"min_unique_users": 10})}
    )
    rule = RULE_IMPLEMENTATIONS["PAD-BOT-001"].prepare(config, catalog)

    broad = rule.evaluate(bot_row(catalog, source_unique_user_count__1h=40))
    assert "BOT_SOURCE_FANOUT" in broad.reason_codes

    narrow = rule.evaluate(bot_row(catalog, source_unique_user_count__1h=2))
    assert narrow.status is RuleStatus.FIRED
    assert "BOT_SOURCE_FANOUT" not in narrow.reason_codes


def test_legitimate_interactive_activity_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        factories.quiet_row(
            catalog,
            source_attempt_count__15m=30,
            source_interarrival_coefficient_of_variation__15m=1.4,
            source_mean_interarrival_seconds__15m=45.0,
            source_unique_user_agent_count__1h=4,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
