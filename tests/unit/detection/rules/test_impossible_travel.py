"""Thresholds, status branches, and privacy for PAD-GEO-001."""

from __future__ import annotations

import math
import re

import pytest

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories
from tests.unit.detection.factories import impossible_travel_row


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def rule(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-GEO-001"].prepare(DetectionConfig(), catalog)


def test_fires_at_exact_threshold_equality(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = impossible_travel_row(
        catalog,
        distance_km_from_user_previous_success=500.0,
        implied_velocity_kmh_from_previous_success=900.0,
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.FIRED


@pytest.mark.parametrize(
    ("column", "threshold"),
    [
        ("distance_km_from_user_previous_success", 500.0),
        ("implied_velocity_kmh_from_previous_success", 900.0),
    ],
)
def test_one_representable_value_below_a_threshold_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog, column: str, threshold: float
) -> None:
    overrides = {
        "distance_km_from_user_previous_success": 500.0,
        "implied_velocity_kmh_from_previous_success": 900.0,
        column: math.nextafter(threshold, 0.0),
    }
    assert rule.evaluate(impossible_travel_row(catalog, **overrides)).status is (
        RuleStatus.NOT_FIRED
    )


def test_a_failed_authentication_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        impossible_travel_row(catalog, current_authentication_outcome="failure")
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert result.reason_codes == ("ANCHOR_DID_NOT_SUCCEED",)


@pytest.mark.parametrize(
    "status",
    ["no_prior_success", "missing_current_location", "missing_prior_location"],
)
def test_missing_location_is_insufficient_data_naming_the_status(
    rule: PreparedRule, catalog: FeatureCatalog, status: str
) -> None:
    """Each status is unseen history; the reason code says which."""
    result = rule.evaluate(
        impossible_travel_row(catalog, user_previous_success_geo__status=status)
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == (f"GEO_STATUS_{status.upper()}",)
    assert result.evidence == ()


def test_an_unavailable_velocity_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        impossible_travel_row(
            catalog,
            implied_velocity__status="unavailable",
            implied_velocity_kmh_from_previous_success=None,
        )
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("GEO_VELOCITY_UNAVAILABLE",)


def test_a_usable_status_with_a_null_velocity_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A status promising a value while the value is absent is inconsistent.

    The rule reports that it could not compare rather than trusting the status
    column over the data it describes.
    """
    result = rule.evaluate(
        impossible_travel_row(
            catalog,
            implied_velocity__status="ok",
            implied_velocity_kmh_from_previous_success=None,
        )
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("GEO_VELOCITY_UNAVAILABLE",)


def test_a_null_distance_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        impossible_travel_row(catalog, distance_km_from_user_previous_success=None)
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("GEO_DISTANCE_UNAVAILABLE",)


def test_an_identical_location_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        impossible_travel_row(
            catalog,
            distance_km_from_user_previous_success=0.0,
            implied_velocity_kmh_from_previous_success=0.0,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert result.reason_codes == ("BELOW_MINIMUM_DISTANCE",)


def test_a_capped_velocity_is_a_usable_magnitude(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A capped value is a floor on the true velocity, which is enough."""
    result = rule.evaluate(
        impossible_travel_row(
            catalog,
            implied_velocity__status="capped",
            implied_velocity_kmh_from_previous_success=5000.0,
        )
    )
    assert result.status is RuleStatus.FIRED
    item = next(
        evidence
        for evidence in result.evidence
        if evidence.evidence_code == "GEO_IMPLIED_VELOCITY"
    )
    assert item.observed_value == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# The zero-elapsed branch
# ---------------------------------------------------------------------------


def test_a_zero_elapsed_interval_fires_with_its_own_reason_code(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Undefined velocity is the strongest form of this shape, not a weaker one."""
    result = rule.evaluate(
        impossible_travel_row(
            catalog,
            implied_velocity__status="zero_elapsed",
            implied_velocity_kmh_from_previous_success=None,
        )
    )
    assert result.status is RuleStatus.FIRED
    assert "GEO_ZERO_ELAPSED_INTERVAL" in result.reason_codes
    assert "GEO_IMPLIED_VELOCITY" not in result.reason_codes
    assert result.signal_strength == pytest.approx(1.0)


def test_the_zero_elapsed_policy_is_configurable(catalog: FeatureCatalog) -> None:
    config = DetectionConfig(
        rules={
            "PAD-GEO-001": RuleSettings(
                parameters={"zero_elapsed_policy": "insufficient_data"}
            )
        }
    )
    rule = RULE_IMPLEMENTATIONS["PAD-GEO-001"].prepare(config, catalog)
    result = rule.evaluate(
        impossible_travel_row(
            catalog,
            implied_velocity__status="zero_elapsed",
            implied_velocity_kmh_from_previous_success=None,
        )
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("GEO_VELOCITY_ZERO_ELAPSED",)


def test_a_zero_elapsed_interval_still_respects_the_distance_floor(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        impossible_travel_row(
            catalog,
            implied_velocity__status="zero_elapsed",
            implied_velocity_kmh_from_previous_success=None,
            distance_km_from_user_previous_success=10.0,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED


# ---------------------------------------------------------------------------
# Country change
# ---------------------------------------------------------------------------


def test_a_country_change_is_reported_but_not_required(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    with_change = rule.evaluate(impossible_travel_row(catalog))
    assert "GEO_COUNTRY_CHANGE" in with_change.reason_codes

    without = rule.evaluate(
        impossible_travel_row(catalog, country_changed_since_previous_success=False)
    )
    assert without.status is RuleStatus.FIRED
    assert "GEO_COUNTRY_CHANGE" not in without.reason_codes


def test_a_required_country_change_gates_the_finding(catalog: FeatureCatalog) -> None:
    config = DetectionConfig(
        rules={"PAD-GEO-001": RuleSettings(parameters={"require_country_change": True})}
    )
    rule = RULE_IMPLEMENTATIONS["PAD-GEO-001"].prepare(config, catalog)

    assert rule.evaluate(impossible_travel_row(catalog)).status is RuleStatus.FIRED
    for value in (False, None):
        result = rule.evaluate(
            impossible_travel_row(catalog, country_changed_since_previous_success=value)
        )
        assert result.status is RuleStatus.NOT_FIRED
        assert result.reason_codes == ("NO_COUNTRY_CHANGE",)


# ---------------------------------------------------------------------------
# Privacy: no coordinate, and a deliberately coarse distance
# ---------------------------------------------------------------------------


def test_the_reported_distance_is_rounded_to_the_configured_multiple(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        impossible_travel_row(
            catalog, distance_km_from_user_previous_success=4237.1234567
        )
    )
    item = next(
        evidence
        for evidence in result.evidence
        if evidence.evidence_code == "GEO_DISTANCE"
    )
    assert item.observed_value == pytest.approx(4240.0)


def test_a_coarser_rounding_is_configurable(catalog: FeatureCatalog) -> None:
    config = DetectionConfig(
        rules={"PAD-GEO-001": RuleSettings(parameters={"distance_rounding_km": 500.0})}
    )
    rule = RULE_IMPLEMENTATIONS["PAD-GEO-001"].prepare(config, catalog)
    result = rule.evaluate(
        impossible_travel_row(
            catalog, distance_km_from_user_previous_success=4237.1234567
        )
    )
    item = next(
        evidence
        for evidence in result.evidence
        if evidence.evidence_code == "GEO_DISTANCE"
    )
    assert item.observed_value == pytest.approx(4000.0)


def test_the_rule_declares_no_coordinate_feature() -> None:
    spec = RULE_CATALOG.get("PAD-GEO-001")
    forbidden = ("latitude", "longitude", "lat", "lon", "coordinate", "geohash")
    for template in (*spec.required_features, *spec.optional_features):
        parts = re.split(r"[^a-z]+", template.lower())
        assert not set(parts) & set(forbidden), template


def test_no_coordinate_appears_in_evidence(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A decimal-degree pair would look like this; none can be produced."""
    coordinate = re.compile(r"-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}")
    result = rule.evaluate(impossible_travel_row(catalog))
    for item in result.evidence:
        assert not coordinate.search(item.model_dump_json())


def test_the_rule_is_described_as_an_indicator() -> None:
    spec = RULE_CATALOG.get("PAD-GEO-001")
    assert "indicator" in spec.display_name.lower()
    assert "approximation" in spec.description.lower()
