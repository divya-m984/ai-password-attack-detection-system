"""Tests for the rule framework: preparation, snapshot access, evidence, signal."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import pytest

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import (
    DetectionConfig,
    RuleSettings,
    SignalConfig,
)
from password_attack_detector.detection.enums import RuleStatus, Severity
from password_attack_detector.detection.rules.base import (
    BasePreparedRule,
    BaseRule,
    RulePreparation,
    SignalComponent,
    SnapshotView,
    build_evidence,
    clamp,
    insufficient_history_reason_code,
    safe_ratio,
    saturate,
    saturate_inverse,
    weighted_strength,
)
from password_attack_detector.exceptions import RuleEvaluationError
from password_attack_detector.features.catalog import build_catalog
from password_attack_detector.features.config import FeatureConfig

ANCHOR = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
WHEN = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

SIGNAL = SignalConfig()


# ---------------------------------------------------------------------------
# A concrete rule built on the framework, used to exercise it end to end.
# ---------------------------------------------------------------------------


class _PairFailurePreparedRule(BasePreparedRule):
    """Fires when the pair failure count reaches its configured threshold."""

    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> Any:
        failures = view.count("pair_failure_count__{window}")
        threshold = self.preparation.param_int("min_pair_failures")
        if failures < threshold:
            return self.not_fired(
                anchor_id, anchor_time, reason_codes=("BELOW_PAIR_FAILURE_THRESHOLD",)
            )
        component = SignalComponent(
            name="pair_failures",
            weight=1.0,
            value=saturate(
                float(failures),
                float(threshold),
                self.preparation.signal.saturation_multiple,
            ),
        )
        return self.fired(
            anchor_id,
            anchor_time,
            signal_strength=self.strength([component]),
            evidence=[
                self.evidence_for(
                    "BF_PAIR_FAILURE_COUNT", failures, threshold_value=threshold
                )
            ],
            reason_codes=("BF_PAIR_FAILURE_COUNT",),
        )


class _PairFailureRule(BaseRule):
    """A real rule over PAD-BF-001's registered specification."""

    def __init__(self) -> None:
        super().__init__(RULE_CATALOG.get("PAD-BF-001"))
        self.build_calls = 0

    def _build(self, preparation: RulePreparation) -> Any:
        self.build_calls += 1
        return _PairFailurePreparedRule(preparation)


@pytest.fixture()
def feature_catalog() -> Any:
    return build_catalog(FeatureConfig())


@pytest.fixture()
def prepared(feature_catalog: Any) -> _PairFailurePreparedRule:
    rule = _PairFailureRule()
    return rule.prepare(DetectionConfig(), feature_catalog)  # type: ignore[return-value]


def _row(**overrides: Any) -> dict[str, Any]:
    """Build a snapshot row carrying every column PAD-BF-001 resolves to."""
    row: dict[str, Any] = {
        "anchor_event_id": ANCHOR,
        "anchor_event_time": WHEN,
        "pair_failure_count__5m": 12,
        "pair_failure_rate__5m": 0.9,
        "user_failure_count__5m": 14,
        "prior_consecutive_user_failures": 9,
        "source_unique_user_count__5m": 1,
        "pair_mean_interarrival_seconds__5m": 4.0,
        "user_blocked_count__5m": 3,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# clamp / safe_ratio
# ---------------------------------------------------------------------------


def test_clamp_confines_to_bounds() -> None:
    assert clamp(-1.0) == 0.0
    assert clamp(2.0) == 1.0
    assert clamp(0.4) == pytest.approx(0.4)
    assert clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_clamp_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        clamp(value)


def test_safe_ratio_returns_none_for_a_zero_denominator() -> None:
    """A zero denominator is an ordinary state, not an exception."""
    assert safe_ratio(5.0, 0.0) is None
    assert safe_ratio(0.0, 0.0) is None
    assert safe_ratio(9.0, 3.0) == pytest.approx(3.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_safe_ratio_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        safe_ratio(value, 1.0)
    with pytest.raises(ValueError, match="must be finite"):
        safe_ratio(1.0, value)


# ---------------------------------------------------------------------------
# saturate
# ---------------------------------------------------------------------------


def test_saturate_is_zero_at_exact_threshold_equality() -> None:
    """Threshold equality is explicit, not an accident of rounding."""
    assert saturate(8.0, 8.0, 3.0) == 0.0


def test_saturate_reaches_one_at_the_saturation_multiple() -> None:
    assert saturate(24.0, 8.0, 3.0) == pytest.approx(1.0)
    assert saturate(100.0, 8.0, 3.0) == pytest.approx(1.0)


def test_saturate_is_zero_below_the_threshold() -> None:
    assert saturate(7.9999, 8.0, 3.0) == 0.0
    assert saturate(0.0, 8.0, 3.0) == 0.0


def test_saturate_rises_monotonically() -> None:
    values = [saturate(float(n), 8.0, 3.0) for n in range(8, 30)]
    assert all(later >= earlier for earlier, later in pairwise(values))
    assert values[0] < values[-1]


def test_saturate_is_bounded_over_a_wide_sweep() -> None:
    for observed in (0.0, 1e-9, 1.0, 1e6, 1e12):
        for threshold in (1e-9, 1.0, 100.0, 1e9):
            for multiple in (1.0001, 2.0, 3.0, 50.0):
                value = saturate(observed, threshold, multiple)
                assert 0.0 <= value <= 1.0
                assert not math.isnan(value)


def test_saturate_handles_a_non_positive_threshold_without_dividing() -> None:
    assert saturate(1.0, 0.0, 3.0) == 1.0
    assert saturate(0.0, 0.0, 3.0) == 0.0
    assert saturate(1.0, -5.0, 3.0) == 1.0


def test_saturate_degenerates_to_a_step_at_multiple_one() -> None:
    assert saturate(9.0, 8.0, 1.0) == 1.0
    assert saturate(8.0, 8.0, 1.0) == 0.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_saturate_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        saturate(value, 8.0, 3.0)
    with pytest.raises(ValueError, match="must be finite"):
        saturate(8.0, value, 3.0)
    with pytest.raises(ValueError, match="must be finite"):
        saturate(8.0, 8.0, value)


# ---------------------------------------------------------------------------
# saturate_inverse
# ---------------------------------------------------------------------------


def test_saturate_inverse_is_zero_at_exact_threshold_equality() -> None:
    assert saturate_inverse(60.0, 60.0, 3.0) == 0.0


def test_saturate_inverse_rises_as_the_observation_falls() -> None:
    values = [saturate_inverse(gap, 60.0, 3.0) for gap in (60.0, 40.0, 30.0, 20.0)]
    assert all(later >= earlier for earlier, later in pairwise(values))
    assert values[-1] == pytest.approx(1.0)


def test_saturate_inverse_survives_a_zero_observation() -> None:
    """A zero interval must saturate, not divide by zero."""
    assert saturate_inverse(0.0, 60.0, 3.0) == pytest.approx(1.0)
    assert saturate_inverse(-5.0, 60.0, 3.0) == pytest.approx(1.0)


def test_saturate_inverse_with_a_non_positive_threshold_contributes_nothing() -> None:
    assert saturate_inverse(5.0, 0.0, 3.0) == 0.0


def test_saturate_inverse_is_bounded_over_a_wide_sweep() -> None:
    for observed in (0.0, 1e-12, 1.0, 1e9):
        for threshold in (1e-9, 1.0, 1e6):
            value = saturate_inverse(observed, threshold, 3.0)
            assert 0.0 <= value <= 1.0
            assert not math.isnan(value)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_saturate_inverse_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        saturate_inverse(value, 60.0, 3.0)


# ---------------------------------------------------------------------------
# weighted_strength
# ---------------------------------------------------------------------------


def test_strength_floors_at_the_configured_minimum() -> None:
    """A rule sitting exactly on its thresholds still reports a positive value."""
    components = [SignalComponent(name="a", weight=1.0, value=0.0)]
    assert weighted_strength(components, signal=SIGNAL) == pytest.approx(
        SIGNAL.min_signal_strength
    )


def test_strength_with_no_components_is_the_floor() -> None:
    assert weighted_strength([], signal=SIGNAL) == pytest.approx(
        SIGNAL.min_signal_strength
    )


def test_strength_reaches_one_when_every_component_saturates() -> None:
    components = [
        SignalComponent(name="a", weight=1.0, value=1.0),
        SignalComponent(name="b", weight=2.0, value=1.0),
    ]
    assert weighted_strength(components, signal=SIGNAL) == pytest.approx(1.0)


def test_strength_is_monotone_in_each_component() -> None:
    weaker = [SignalComponent(name="a", weight=1.0, value=0.2)]
    stronger = [SignalComponent(name="a", weight=1.0, value=0.8)]
    assert weighted_strength(stronger, signal=SIGNAL) > weighted_strength(
        weaker, signal=SIGNAL
    )


def test_strength_stays_in_the_unit_interval() -> None:
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = weighted_strength(
            [SignalComponent(name="a", weight=1.0, value=value)], signal=SIGNAL
        )
        assert SIGNAL.min_signal_strength <= result <= 1.0


def test_strength_respects_component_weights() -> None:
    components = [
        SignalComponent(name="dominant", weight=9.0, value=1.0),
        SignalComponent(name="minor", weight=1.0, value=0.0),
    ]
    assert weighted_strength(components, signal=SIGNAL) > 0.9


def test_strength_is_deterministic() -> None:
    components = [
        SignalComponent(name="a", weight=1.0, value=0.3),
        SignalComponent(name="b", weight=2.0, value=0.7),
    ]
    assert weighted_strength(components, signal=SIGNAL) == weighted_strength(
        components, signal=SIGNAL
    )


@pytest.mark.parametrize(
    ("weight", "value", "match"),
    [
        (0.0, 0.5, "positive weight"),
        (-1.0, 0.5, "positive weight"),
        (1.0, 1.5, r"\[0, 1\]"),
        (1.0, -0.1, r"\[0, 1\]"),
        (1.0, math.nan, "must be finite"),
        (1.0, math.inf, "must be finite"),
    ],
)
def test_invalid_signal_components_are_rejected(
    weight: float, value: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SignalComponent(name="bad", weight=weight, value=value)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def test_preparation_resolves_templates_to_columns(feature_catalog: Any) -> None:
    rule = _PairFailureRule()
    prepared_rule = rule.prepare(DetectionConfig(), feature_catalog)
    features = prepared_rule.preparation.features
    assert features["pair_failure_count__{window}"] == "pair_failure_count__5m"
    assert features["source_unique_user_count__{cardinality_window}"] == (
        "source_unique_user_count__5m"
    )


def test_preparation_happens_once_regardless_of_row_count(
    feature_catalog: Any,
) -> None:
    """Template resolution and catalog lookups must not recur per snapshot."""
    rule = _PairFailureRule()
    prepared_rule = rule.prepare(DetectionConfig(), feature_catalog)
    assert rule.build_calls == 1

    for _ in range(50):
        prepared_rule.evaluate(_row())
    assert rule.build_calls == 1


def test_preparation_fails_for_a_missing_required_feature(
    feature_catalog: Any,
) -> None:
    config = DetectionConfig(
        rules={"PAD-BF-001": RuleSettings(parameters={"window": "2m"})}
    )
    with pytest.raises(RuleEvaluationError, match="does not declare"):
        _PairFailureRule().prepare(config, feature_catalog)


def test_preparation_carries_the_configured_severity(feature_catalog: Any) -> None:
    config = DetectionConfig(
        rules={"PAD-BF-001": RuleSettings(severity=Severity.CRITICAL)}
    )
    prepared_rule = _PairFailureRule().prepare(config, feature_catalog)
    assert prepared_rule.preparation.severity is Severity.CRITICAL


def test_preparation_typed_parameter_accessors(prepared: Any) -> None:
    preparation = prepared.preparation
    assert preparation.param_int("min_pair_failures") == 8
    assert preparation.param_float("min_pair_failure_rate") == pytest.approx(0.80)
    assert preparation.param_str("window") == "5m"

    with pytest.raises(RuleEvaluationError, match="not an integer"):
        preparation.param_int("min_pair_failure_rate")
    with pytest.raises(RuleEvaluationError, match="not a boolean"):
        preparation.param_bool("min_pair_failures")
    with pytest.raises(RuleEvaluationError, match="not a string"):
        preparation.param_str("min_pair_failures")
    with pytest.raises(RuleEvaluationError, match="not a number"):
        preparation.param_float("window")
    with pytest.raises(RuleEvaluationError, match="no prepared parameter"):
        preparation.param_int("nonexistent")


def test_preparation_rejects_an_undeclared_template(prepared: Any) -> None:
    with pytest.raises(RuleEvaluationError, match="did not declare"):
        prepared.preparation.feature("user_success_count__5m")


def test_preparation_param_float_accepts_an_integer_parameter(prepared: Any) -> None:
    assert prepared.preparation.param_float("min_pair_failures") == pytest.approx(8.0)


def test_geospatial_rule_prepares_a_boolean_and_choice_parameter(
    feature_catalog: Any,
) -> None:
    """Exercises the boolean and choice accessors on a real specification."""

    class _GeoPrepared(BasePreparedRule):
        def _evaluate(
            self, view: SnapshotView, anchor_id: str, anchor_time: datetime
        ) -> Any:
            return self.not_fired(anchor_id, anchor_time)

    class _GeoRule(BaseRule):
        def __init__(self) -> None:
            super().__init__(RULE_CATALOG.get("PAD-GEO-001"))

        def _build(self, preparation: RulePreparation) -> Any:
            return _GeoPrepared(preparation)

    prepared_rule = _GeoRule().prepare(DetectionConfig(), feature_catalog)
    assert prepared_rule.preparation.param_bool("require_country_change") is False
    assert prepared_rule.preparation.param_str("zero_elapsed_policy") == "fire"


# ---------------------------------------------------------------------------
# SnapshotView
# ---------------------------------------------------------------------------


def test_view_reads_typed_columns(prepared: Any) -> None:
    view = SnapshotView(_row(), prepared.preparation)
    assert view.count("pair_failure_count__{window}") == 12
    assert view.number("pair_failure_rate__{window}") == pytest.approx(0.9)
    assert view.raw("prior_consecutive_user_failures") == 9


def test_view_returns_none_for_a_null_nullable_column(prepared: Any) -> None:
    view = SnapshotView(_row(pair_failure_rate__5m=None), prepared.preparation)
    assert view.number("pair_failure_rate__{window}") is None


def test_view_rejects_a_missing_column(prepared: Any) -> None:
    row = _row()
    del row["pair_failure_count__5m"]
    view = SnapshotView(row, prepared.preparation)
    with pytest.raises(RuleEvaluationError, match="does not carry column"):
        view.count("pair_failure_count__{window}")


def test_view_rejects_a_null_count(prepared: Any) -> None:
    view = SnapshotView(_row(pair_failure_count__5m=None), prepared.preparation)
    with pytest.raises(RuleEvaluationError, match="declares it non-nullable"):
        view.count("pair_failure_count__{window}")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_view_rejects_a_non_finite_number(prepared: Any, value: float) -> None:
    """A non-finite feature would dominate every saturation curve it touched."""
    view = SnapshotView(_row(pair_failure_rate__5m=value), prepared.preparation)
    with pytest.raises(RuleEvaluationError, match="NaN or infinite"):
        view.number("pair_failure_rate__{window}")


def test_view_rejects_a_wrongly_typed_column(prepared: Any) -> None:
    view = SnapshotView(_row(pair_failure_rate__5m="high"), prepared.preparation)
    with pytest.raises(RuleEvaluationError, match="not numeric"):
        view.number("pair_failure_rate__{window}")

    view = SnapshotView(_row(pair_failure_count__5m=True), prepared.preparation)
    with pytest.raises(RuleEvaluationError, match="not an integer"):
        view.count("pair_failure_count__{window}")


def test_view_flag_and_text_accessors(feature_catalog: Any) -> None:
    class _Prepared(BasePreparedRule):
        def _evaluate(
            self, view: SnapshotView, anchor_id: str, anchor_time: datetime
        ) -> Any:
            return self.not_fired(anchor_id, anchor_time)

    class _Rule(BaseRule):
        def __init__(self) -> None:
            super().__init__(RULE_CATALOG.get("PAD-CS-001"))

        def _build(self, preparation: RulePreparation) -> Any:
            return _Prepared(preparation)

    prepared_rule = _Rule().prepare(DetectionConfig(), feature_catalog)
    row = {
        "user_in_baseline": True,
        "is_new_device_for_user": None,
        "is_new_country_for_user": "yes",
    }
    view = SnapshotView(row, prepared_rule.preparation)
    assert view.flag("user_in_baseline") is True
    assert view.flag("is_new_device_for_user") is None
    with pytest.raises(RuleEvaluationError, match="not a boolean"):
        view.flag("is_new_country_for_user")
    assert view.text("is_new_country_for_user") == "yes"
    with pytest.raises(RuleEvaluationError, match="not a string"):
        view.text("user_in_baseline")


def test_view_reads_by_key_so_row_order_cannot_matter(prepared: Any) -> None:
    forward = _row()
    reversed_row = dict(reversed(list(forward.items())))
    assert list(forward) != list(reversed_row)

    first = SnapshotView(forward, prepared.preparation)
    second = SnapshotView(reversed_row, prepared.preparation)
    assert first.count("pair_failure_count__{window}") == second.count(
        "pair_failure_count__{window}"
    )


def test_missing_history_is_detected(prepared: Any) -> None:
    assert SnapshotView(_row(), prepared.preparation).missing_history() == ()
    view = SnapshotView(_row(pair_failure_rate__5m=None), prepared.preparation)
    assert view.missing_history() == ("pair_failure_rate__{window}",)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_is_built_from_the_declared_definition(prepared: Any) -> None:
    item = build_evidence(
        prepared.preparation, "BF_PAIR_FAILURE_COUNT", 12, threshold_value=8
    )
    assert item.evidence_code == "BF_PAIR_FAILURE_COUNT"
    assert item.feature_name == "pair_failure_count__5m"
    assert item.observed_value == 12
    assert item.threshold_value == 8
    assert "12" in item.message
    assert "8" in item.message


def test_evidence_is_deterministic(prepared: Any) -> None:
    first = build_evidence(
        prepared.preparation, "BF_PAIR_FAILURE_COUNT", 12, threshold_value=8
    )
    second = build_evidence(
        prepared.preparation, "BF_PAIR_FAILURE_COUNT", 12, threshold_value=8
    )
    assert first == second


def test_evidence_carries_no_identifier(prepared: Any) -> None:
    item = build_evidence(
        prepared.preparation, "BF_PAIR_FAILURE_COUNT", 12, threshold_value=8
    )
    rendered = item.model_dump_json()
    assert ANCHOR not in rendered
    assert "u:" not in rendered
    assert "s:" not in rendered


def test_evidence_for_an_undeclared_code_is_rejected(prepared: Any) -> None:
    with pytest.raises(RuleEvaluationError, match="declares no evidence"):
        build_evidence(prepared.preparation, "NOT_DECLARED", 1)


def test_derived_evidence_names_no_column(feature_catalog: Any) -> None:
    class _Prepared(BasePreparedRule):
        def _evaluate(
            self, view: SnapshotView, anchor_id: str, anchor_time: datetime
        ) -> Any:
            return self.not_fired(anchor_id, anchor_time)

    class _Rule(BaseRule):
        def __init__(self) -> None:
            super().__init__(RULE_CATALOG.get("PAD-ATO-001"))

        def _build(self, preparation: RulePreparation) -> Any:
            return _Prepared(preparation)

    prepared_rule = _Rule().prepare(DetectionConfig(), feature_catalog)
    item = build_evidence(
        prepared_rule.preparation, "ATO_NOVEL_CONTEXT_COUNT", 3, threshold_value=2
    )
    assert item.feature_name == "derived"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(12, "12"), (12.0, "12"), (0.8500, "0.85"), (True, "true"), (False, "false")],
)
def test_evidence_values_render_deterministically(
    prepared: Any, value: Any, expected: str
) -> None:
    item = build_evidence(prepared.preparation, "BF_PAIR_FAILURE_COUNT", value)
    assert expected in item.message


def test_insufficient_history_reason_code_is_derived_from_the_column() -> None:
    assert (
        insufficient_history_reason_code("pair_failure_rate__5m")
        == "INSUFFICIENT_HISTORY_PAIR_FAILURE_RATE__5M"
    )
    assert (
        insufficient_history_reason_code("implied_velocity__status")
        == "INSUFFICIENT_HISTORY_IMPLIED_VELOCITY__STATUS"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_rule_fires_above_the_threshold(prepared: Any) -> None:
    result = prepared.evaluate(_row(pair_failure_count__5m=24))
    assert result.status is RuleStatus.FIRED
    assert result.rule_id == "PAD-BF-001"
    assert result.anchor_event_id == ANCHOR
    assert result.evidence
    assert result.signal_strength == pytest.approx(1.0)


def test_rule_does_not_fire_below_the_threshold(prepared: Any) -> None:
    result = prepared.evaluate(_row(pair_failure_count__5m=7))
    assert result.status is RuleStatus.NOT_FIRED
    assert result.evidence == ()
    assert result.signal_strength == 0.0
    assert result.reason_codes == ("BELOW_PAIR_FAILURE_THRESHOLD",)


def test_threshold_equality_fires_at_the_minimum_strength(prepared: Any) -> None:
    """Exactly at the threshold the rule fires, at the configured floor."""
    result = prepared.evaluate(_row(pair_failure_count__5m=8))
    assert result.status is RuleStatus.FIRED
    assert result.signal_strength == pytest.approx(SIGNAL.min_signal_strength)
    assert result.signal_strength > 0.0


def test_one_below_the_threshold_does_not_fire(prepared: Any) -> None:
    assert prepared.evaluate(_row(pair_failure_count__5m=7)).status is (
        RuleStatus.NOT_FIRED
    )


def test_stronger_evidence_never_lowers_signal_strength(prepared: Any) -> None:
    strengths = [
        prepared.evaluate(_row(pair_failure_count__5m=count)).signal_strength
        for count in range(8, 40)
    ]
    assert all(later >= earlier for earlier, later in pairwise(strengths))
    assert strengths[-1] > strengths[0]


def test_signal_strength_stays_in_the_unit_interval(prepared: Any) -> None:
    for count in (8, 9, 16, 24, 1000, 10**9):
        strength = prepared.evaluate(_row(pair_failure_count__5m=count)).signal_strength
        assert 0.0 < strength <= 1.0
        assert not math.isnan(strength)


def test_unavailable_history_returns_insufficient_data(prepared: Any) -> None:
    """Unseen history is never reported as a clean negative."""
    result = prepared.evaluate(_row(pair_failure_rate__5m=None))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("INSUFFICIENT_HISTORY_PAIR_FAILURE_RATE__5M",)
    assert result.evidence == ()
    assert result.signal_strength == 0.0


def test_history_gate_precedes_any_threshold_comparison(prepared: Any) -> None:
    """A missing history column wins even when the thresholds would fire."""
    result = prepared.evaluate(
        _row(pair_failure_count__5m=10_000, pair_failure_rate__5m=None)
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA


def test_evaluation_is_deterministic(prepared: Any) -> None:
    row = _row(pair_failure_count__5m=17)
    assert prepared.evaluate(row) == prepared.evaluate(row)


def test_row_order_does_not_change_the_result(prepared: Any) -> None:
    forward = _row(pair_failure_count__5m=17)
    reversed_row = dict(reversed(list(forward.items())))
    assert prepared.evaluate(forward) == prepared.evaluate(reversed_row)


def test_extra_columns_in_the_row_are_ignored(prepared: Any) -> None:
    """A rule reads only what it declared, whatever else the snapshot carries."""
    baseline = prepared.evaluate(_row(pair_failure_count__5m=17))
    noisy = prepared.evaluate(
        _row(
            pair_failure_count__5m=17,
            user_success_count__5m=999,
            hour_of_day=3,
            is_weekend=True,
        )
    )
    assert baseline == noisy


def test_result_carries_no_identifier_in_its_evidence(prepared: Any) -> None:
    result = prepared.evaluate(_row(pair_failure_count__5m=17))
    for item in result.evidence:
        rendered = item.model_dump_json()
        assert ANCHOR not in rendered
        assert "u:" not in rendered


def test_disabled_result_constructor(prepared: Any) -> None:
    result = prepared.disabled(ANCHOR, WHEN)
    assert result.status is RuleStatus.DISABLED
    assert result.reason_codes == ()
    assert result.signal_strength == 0.0


# ---------------------------------------------------------------------------
# Anchor handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", ["anchor_event_id", "anchor_event_time"])
def test_missing_anchor_column_is_rejected(prepared: Any, column: str) -> None:
    row = _row()
    del row[column]
    with pytest.raises(RuleEvaluationError, match="anchor column"):
        prepared.evaluate(row)


def test_wrongly_typed_anchor_is_rejected(prepared: Any) -> None:
    with pytest.raises(RuleEvaluationError, match="must be a string"):
        prepared.evaluate(_row(anchor_event_id=42))
    with pytest.raises(RuleEvaluationError, match="must be a datetime"):
        prepared.evaluate(_row(anchor_event_time="2026-03-01T12:00:00Z"))


def test_anchor_columns_cannot_be_declared_as_features(prepared: Any) -> None:
    """The anchor keys are the row's identity, not a rule input."""
    with pytest.raises(RuleEvaluationError, match="did not declare"):
        prepared.preparation.feature("anchor_event_id")


# ---------------------------------------------------------------------------
# Boundary: no labels, splits, campaigns, scope, or model output
# ---------------------------------------------------------------------------


def test_rule_ignores_label_split_and_model_columns(prepared: Any) -> None:
    """Adding prohibited columns to a snapshot cannot change any output."""
    baseline = prepared.evaluate(_row(pair_failure_count__5m=17))
    contaminated = prepared.evaluate(
        _row(
            pair_failure_count__5m=17,
            attack_class="brute_force",
            malicious=True,
            campaign_id="c-1",
            split="test",
            supervised_training_eligible=False,
            model_probability=0.99,
            user_scope="u:" + "a" * 32,
        )
    )
    assert baseline == contaminated


def test_mutating_a_label_column_cannot_change_a_result(prepared: Any) -> None:
    first = prepared.evaluate(_row(pair_failure_count__5m=17, malicious=True))
    second = prepared.evaluate(_row(pair_failure_count__5m=17, malicious=False))
    assert first == second


def test_prepared_rule_holds_no_reference_to_events_or_scope(prepared: Any) -> None:
    preparation = prepared.preparation
    assert set(preparation.features) <= set(
        preparation.spec.required_features + preparation.spec.optional_features
    )
    assert all("scope" not in name for name in preparation.features.values())


def test_view_text_returns_none_for_a_null_column(feature_catalog: Any) -> None:
    class _Prepared(BasePreparedRule):
        def _evaluate(
            self, view: SnapshotView, anchor_id: str, anchor_time: datetime
        ) -> Any:
            return self.not_fired(anchor_id, anchor_time)

    class _Rule(BaseRule):
        def __init__(self) -> None:
            super().__init__(RULE_CATALOG.get("PAD-MFA-001"))

        def _build(self, preparation: RulePreparation) -> Any:
            return _Prepared(preparation)

    prepared_rule = _Rule().prepare(DetectionConfig(), feature_catalog)
    view = SnapshotView({"current_mfa_outcome": None}, prepared_rule.preparation)
    assert view.text("current_mfa_outcome") is None


def test_evidence_renders_a_string_observation(feature_catalog: Any) -> None:
    class _Prepared(BasePreparedRule):
        def _evaluate(
            self, view: SnapshotView, anchor_id: str, anchor_time: datetime
        ) -> Any:
            return self.not_fired(anchor_id, anchor_time)

    class _Rule(BaseRule):
        def __init__(self) -> None:
            super().__init__(RULE_CATALOG.get("PAD-MFA-001"))

        def _build(self, preparation: RulePreparation) -> Any:
            return _Prepared(preparation)

    prepared_rule = _Rule().prepare(DetectionConfig(), feature_catalog)
    item = build_evidence(prepared_rule.preparation, "MFA_CURRENT_OUTCOME", "failed")
    assert "failed" in item.message
    assert item.observed_value == "failed"
