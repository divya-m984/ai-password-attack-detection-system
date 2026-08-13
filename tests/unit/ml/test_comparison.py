"""The fair comparison: one population, three systems, and no winner.

Most of this file is about the *population*, because that is where a comparison
actually goes wrong. A system quietly missing a row is measured on an easier
denominator than its comparators, and the resulting table looks entirely normal.
So the universe check is asserted from every direction: missing, extra,
duplicate, reordered, and mislabelled.

The rest asserts that a comparison describes and never selects. There is no
winner field, no ranking, and the row order is a declared constant rather than
anything derived from the numbers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.alerts import (
    ComparisonDecision,
    DecisionAlertPolicy,
)
from password_attack_detector.ml.comparison import (
    COMPARISON_SCHEMA_VERSION,
    SYSTEM_ORDER,
    AlertLevelSummary,
    SystemComparison,
    SystemEvaluation,
    build_comparison,
    comparison_config_fingerprint,
    evaluate_system,
    population_fingerprint,
    require_common_universe,
)
from password_attack_detector.ml.enums import ComparisonSystem, MetricStatus

WHEN = datetime(2024, 3, 1, tzinfo=UTC)
ANCHORS = [f"a{index:04d}" for index in range(8)]
TRUTH = [index % 2 == 0 for index in range(8)]


def decision(
    system: ComparisonSystem,
    index: int,
    *,
    flagged: bool = True,
    anchor: str | None = None,
) -> ComparisonDecision:
    """Build one decision for the shared population."""
    return ComparisonDecision(
        anchor_event_id=anchor or ANCHORS[index],
        anchor_event_time=WHEN + timedelta(minutes=index),
        system=system,
        flagged=flagged,
    )


def stream(
    system: ComparisonSystem, flags: list[bool] | None = None
) -> tuple[ComparisonDecision, ...]:
    """Build one system's decisions over the shared population."""
    chosen = flags if flags is not None else TRUTH
    return tuple(
        decision(system, index, flagged=chosen[index]) for index in range(len(ANCHORS))
    )


# ---------------------------------------------------------------------------
# The common universe
# ---------------------------------------------------------------------------


def test_matching_streams_are_accepted() -> None:
    """The positive case, so the refusals below are not vacuous."""
    require_common_universe(
        {
            ComparisonSystem.RULE_ONLY: stream(ComparisonSystem.RULE_ONLY),
            ComparisonSystem.ML_ONLY: stream(ComparisonSystem.ML_ONLY),
        },
        expected_anchors=ANCHORS,
    )


def test_a_missing_decision_is_refused() -> None:
    """A system evaluated on fewer rows is not a comparator."""
    short = stream(ComparisonSystem.ML_ONLY)[:-1]
    with pytest.raises(DataValidationError, match="missing 1 decision"):
        require_common_universe(
            {ComparisonSystem.ML_ONLY: short}, expected_anchors=ANCHORS
        )


def test_an_extra_decision_is_refused() -> None:
    """And a system evaluated on more."""
    extra = (
        *stream(ComparisonSystem.ML_ONLY),
        decision(ComparisonSystem.ML_ONLY, 0, anchor="zzz"),
    )
    with pytest.raises(DataValidationError, match="outside the comparison population"):
        require_common_universe(
            {ComparisonSystem.ML_ONLY: extra}, expected_anchors=ANCHORS
        )


def test_a_duplicate_decision_is_refused() -> None:
    """Each anchor contributes exactly one row to each system."""
    rows = stream(ComparisonSystem.ML_ONLY)
    with pytest.raises(DataValidationError, match="more than once"):
        require_common_universe(
            {ComparisonSystem.ML_ONLY: (*rows, rows[0])}, expected_anchors=ANCHORS
        )


def test_a_reordered_stream_is_refused() -> None:
    """The outcomes are aligned positionally; a reordered system is misscored."""
    rows = list(stream(ComparisonSystem.ML_ONLY))
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(DataValidationError, match="different order"):
        require_common_universe(
            {ComparisonSystem.ML_ONLY: tuple(rows)}, expected_anchors=ANCHORS
        )


def test_a_decision_filed_under_the_wrong_system_is_refused() -> None:
    """A stream must be the system it claims to be."""
    rows = stream(ComparisonSystem.ML_ONLY)
    with pytest.raises(DataValidationError, match="names a different system"):
        require_common_universe(
            {ComparisonSystem.RULE_ONLY: rows}, expected_anchors=ANCHORS
        )


def test_a_repeated_expected_anchor_is_refused() -> None:
    """The population itself must be a set."""
    with pytest.raises(DataValidationError, match="repeats an anchor"):
        require_common_universe({}, expected_anchors=[*ANCHORS, ANCHORS[0]])


def test_an_empty_population_is_refused() -> None:
    """A comparison over no rows is not a comparison with no findings."""
    with pytest.raises(DataValidationError, match="over no rows"):
        require_common_universe({}, expected_anchors=[])


# ---------------------------------------------------------------------------
# Evaluating one system
# ---------------------------------------------------------------------------


def test_a_system_without_a_score_reports_no_discrimination_metric() -> None:
    """The rule engine's ordinal magnitude is not a ranking score."""
    evaluation = evaluate_system(
        system=ComparisonSystem.RULE_ONLY,
        decision_source="phase4:abc",
        decisions=stream(ComparisonSystem.RULE_ONLY),
        malicious=TRUTH,
        score_unavailable_reason="ordinal_rule_risk_is_not_a_ranking_score",
    )
    assert evaluation.continuous_score_available is False
    assert evaluation.metrics.pr_auc is None
    assert evaluation.calibrated_probability_available is False
    assert evaluation.alert_level_available is False


def test_a_system_with_a_score_reports_one() -> None:
    """And the converse."""
    evaluation = evaluate_system(
        system=ComparisonSystem.ML_ONLY,
        decision_source="prediction:abc",
        decisions=stream(ComparisonSystem.ML_ONLY),
        malicious=TRUTH,
        scores=[0.9 if flag else 0.1 for flag in TRUTH],
        probabilities=[0.9 if flag else 0.1 for flag in TRUTH],
    )
    assert evaluation.continuous_score_available is True
    assert evaluation.metrics.pr_auc == 1.0
    assert evaluation.calibrated_probability_available is True
    assert evaluation.metrics.calibration is not None


def test_alert_level_metrics_are_produced_under_a_shared_policy() -> None:
    """Event-level and alert-level quantities are reported side by side."""
    policy = DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    evaluation = evaluate_system(
        system=ComparisonSystem.ML_ONLY,
        decision_source="prediction:abc",
        decisions=stream(ComparisonSystem.ML_ONLY),
        malicious=TRUTH,
        score_unavailable_reason="hand_built",
        alert_policy=policy,
    )
    assert evaluation.alerts is not None
    assert evaluation.alerts.alert_count >= 1
    assert evaluation.alerts.qualifying_decision_count == sum(TRUTH)
    assert evaluation.alerts.alerts_per_qualifying_decision is not None


def test_an_alert_ratio_over_nothing_is_unavailable() -> None:
    """An empty denominator is never a zero."""
    policy = DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    evaluation = evaluate_system(
        system=ComparisonSystem.ML_ONLY,
        decision_source="prediction:abc",
        decisions=stream(ComparisonSystem.ML_ONLY, [False] * len(ANCHORS)),
        malicious=TRUTH,
        score_unavailable_reason="hand_built",
        alert_policy=policy,
    )
    assert evaluation.alerts is not None
    assert evaluation.alerts.alert_count == 0
    assert evaluation.alerts.alerts_per_qualifying_decision is None


def test_an_availability_flag_that_contradicts_the_metrics_is_refused() -> None:
    """The flags describe what is actually present."""
    valid = evaluate_system(
        system=ComparisonSystem.ML_ONLY,
        decision_source="prediction:abc",
        decisions=stream(ComparisonSystem.ML_ONLY),
        malicious=TRUTH,
        score_unavailable_reason="hand_built",
    )
    with pytest.raises(ValueError, match="continuous score"):
        SystemEvaluation(
            system=valid.system,
            decision_source=valid.decision_source,
            metrics=valid.metrics,
            alerts=None,
            continuous_score_available=True,
            calibrated_probability_available=False,
            alert_level_available=False,
        )


def test_a_system_must_name_the_artifact_it_decided_from() -> None:
    """A reader can tell a published prediction from a recomputed one."""
    valid = evaluate_system(
        system=ComparisonSystem.ML_ONLY,
        decision_source="prediction:abc",
        decisions=stream(ComparisonSystem.ML_ONLY),
        malicious=TRUTH,
        score_unavailable_reason="hand_built",
    )
    with pytest.raises(ValueError, match="names the frozen artifact"):
        SystemEvaluation(
            system=valid.system,
            decision_source="   ",
            metrics=valid.metrics,
            alerts=None,
            continuous_score_available=False,
            calibrated_probability_available=False,
            alert_level_available=False,
        )


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def systems(include_hybrid: bool = False) -> list[SystemEvaluation]:
    """Return two or three evaluated systems over the shared population."""
    built = [
        evaluate_system(
            system=ComparisonSystem.RULE_ONLY,
            decision_source="phase4:abc",
            decisions=stream(ComparisonSystem.RULE_ONLY),
            malicious=TRUTH,
            score_unavailable_reason="ordinal",
        ),
        evaluate_system(
            system=ComparisonSystem.ML_ONLY,
            decision_source="prediction:abc",
            decisions=stream(ComparisonSystem.ML_ONLY),
            malicious=TRUTH,
            score_unavailable_reason="hand_built",
        ),
    ]
    if include_hybrid:
        built.append(
            evaluate_system(
                system=ComparisonSystem.HYBRID,
                decision_source="fusion:abc/or_gate",
                decisions=stream(ComparisonSystem.HYBRID),
                malicious=TRUTH,
                score_unavailable_reason="fused_boolean",
            )
        )
    return built


def test_a_comparison_reports_systems_in_the_declared_order() -> None:
    """Row order is a constant, so it can never be read as a ranking."""
    comparison = build_comparison(
        systems=list(reversed(systems(include_hybrid=True))),
        anchors=ANCHORS,
        malicious=TRUTH,
        alert_policy=None,
        hybrid_unavailable_reason=None,
    )
    assert [item.system for item in comparison.systems] == list(SYSTEM_ORDER)


def test_an_absent_hybrid_names_why_it_is_absent() -> None:
    """A missing system is explained rather than silently dropped."""
    comparison = build_comparison(
        systems=systems(),
        anchors=ANCHORS,
        malicious=TRUTH,
        alert_policy=None,
        hybrid_unavailable_reason="no_fusion_selection",
    )
    assert comparison.for_system(ComparisonSystem.HYBRID) is None
    assert comparison.hybrid_unavailable_reason == "no_fusion_selection"


def test_a_present_hybrid_names_no_absence_reason() -> None:
    """Both directions of the same rule."""
    with pytest.raises(ValueError, match="absent hybrid names why"):
        build_comparison(
            systems=systems(include_hybrid=True),
            anchors=ANCHORS,
            malicious=TRUTH,
            alert_policy=None,
            hybrid_unavailable_reason="no_fusion_selection",
        )


def test_a_system_measured_on_another_population_is_refused() -> None:
    """Two numbers from two populations are not comparable."""
    other = evaluate_system(
        system=ComparisonSystem.HYBRID,
        decision_source="fusion:abc",
        decisions=stream(ComparisonSystem.HYBRID)[:4],
        malicious=TRUTH[:4],
        score_unavailable_reason="fused_boolean",
    )
    with pytest.raises(ValueError, match="different number of rows"):
        build_comparison(
            systems=[*systems(), other],
            anchors=ANCHORS,
            malicious=TRUTH,
            alert_policy=None,
            hybrid_unavailable_reason=None,
        )


def test_a_comparison_declares_no_winner() -> None:
    """A descriptive final evaluation is one step from a deployment decision."""
    for absent in ("winner", "best_system", "selected_system", "ranking", "rank"):
        assert absent not in SystemComparison.model_fields, absent
        assert absent not in SystemEvaluation.model_fields, absent
        assert absent not in AlertLevelSummary.model_fields, absent


def test_the_comparison_is_sealed_and_reproducible() -> None:
    """Two comparisons over one population are the same record."""
    first = build_comparison(
        systems=systems(),
        anchors=ANCHORS,
        malicious=TRUTH,
        alert_policy=None,
        hybrid_unavailable_reason="none",
    )
    second = build_comparison(
        systems=systems(),
        anchors=ANCHORS,
        malicious=TRUTH,
        alert_policy=None,
        hybrid_unavailable_reason="none",
    )
    assert first.to_json() == second.to_json()


def test_thin_support_is_reported_on_the_comparison() -> None:
    """A comparison over four positive rows is not a measurement."""
    comparison = build_comparison(
        systems=systems(),
        anchors=ANCHORS,
        malicious=TRUTH,
        alert_policy=None,
        hybrid_unavailable_reason="none",
        min_positive_rows=100,
    )
    assert comparison.support_status is MetricStatus.INSUFFICIENT_SUPPORT


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_the_population_fingerprint_covers_the_rows_and_the_outcomes() -> None:
    """Two populations with the same rows and different labels differ."""
    first = population_fingerprint(ANCHORS, TRUTH)
    flipped = population_fingerprint(ANCHORS, [not flag for flag in TRUTH])
    assert first != flipped
    assert first == population_fingerprint(ANCHORS, TRUTH)


def test_the_population_fingerprint_publishes_no_anchor() -> None:
    """Anchors take part; a hash is what leaves."""
    digest = population_fingerprint(ANCHORS, TRUTH)
    assert len(digest) == 64
    for anchor in ANCHORS:
        assert anchor not in digest


def test_misaligned_population_inputs_are_refused() -> None:
    """A digest over mismatched rows would be stable and wrong."""
    with pytest.raises(DataValidationError, match="cannot be paired"):
        population_fingerprint(ANCHORS, TRUTH[:2])


def test_the_comparison_configuration_binds_the_alert_policy() -> None:
    """A comparison under different windows is a different comparison."""
    policy = DecisionAlertPolicy.from_alerting(DetectionConfig().alerting)
    assert comparison_config_fingerprint(
        alert_policy=None
    ) != comparison_config_fingerprint(alert_policy=policy)


def test_the_comparison_contract_version_is_pinned() -> None:
    """A change to what a comparison means is a visible edit."""
    assert COMPARISON_SCHEMA_VERSION == "1.0.0"
