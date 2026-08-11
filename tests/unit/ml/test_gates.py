"""Champion gates: three outcomes, and the third one is the point.

A gate that could only pass or fail would turn every thin validation half into a
clean bill of health. Most of this suite is therefore about ``inconclusive`` --
when it is reached, that it blocks exactly as a failure does, and that it is
never quietly converted into either of the other two.

Nothing here builds a dataset. Gates consume frozen artifacts, so the fixtures
are frozen artifacts: a threshold selection, a calibration report, a catalog
verdict.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.calibration import CalibrationReport
from password_attack_detector.ml.config import (
    CategoryConfig,
    ChampionGateConfig,
    MLConfig,
)
from password_attack_detector.ml.enums import (
    CalibrationEvaluationKind,
    GateStatus,
    MetricStatus,
    ScoreKind,
    SelectionStatus,
    ValidationPartition,
)
from password_attack_detector.ml.gates import (
    BINARY_GATE_IDS,
    CATEGORY_GATE_IDS,
    BinaryGateInputs,
    CategoryGateInputs,
    GateEvidence,
    RateEvidence,
    evaluate_binary_gates,
    evaluate_category_gates,
    gate_config_fingerprint,
    wilson_interval,
)
from password_attack_detector.ml.ranking import (
    DISCRIMINATION_SCORE_KIND,
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
    RankingEvidence,
    build_ranking_evidence,
)
from password_attack_detector.ml.thresholds import (
    CategoryAbstentionSelection,
    CategoryClassSupport,
    ThresholdCurvePoint,
    ThresholdSelection,
)
from tests.ml import runs as rx
from tests.ml import selection as sx


def config(**overrides: Any) -> MLConfig:
    """Return a CI-sized configuration with the gate thresholds in *overrides*.

    The configuration refuses a gate stricter than the search it judges, so a
    tightened gate drags the corresponding search setting along with it.  That
    coupling is the real contract; a fixture that dodged it by constructing the
    model unvalidated would be testing a configuration the loader cannot build.
    """
    max_fpr = overrides.pop("max_fpr", 0.5)
    min_recall = overrides.pop("min_recall", 0.1)
    max_ece = overrides.pop("max_ece", 0.5)
    base = rx.config()
    settings: dict[str, Any] = {
        "gates": ChampionGateConfig(
            min_pr_auc_gain_over_baseline=overrides.pop("min_gain", 0.0),
            max_false_positive_rate=max_fpr,
            min_detection_rate=min_recall,
            max_expected_calibration_error=max_ece,
        ),
        "thresholds": type(base.thresholds)(
            **{
                **base.thresholds.model_dump(),
                "max_false_positive_rate": max_fpr,
                "min_detection_rate": min_recall,
            }
        ),
        "calibration": type(base.calibration)(
            **{
                **base.calibration.model_dump(),
                "max_expected_calibration_error": max_ece,
            }
        ),
    }
    settings.update(overrides)
    return rx.config(**settings)


def threshold(
    *,
    status: SelectionStatus = SelectionStatus.SELECTED,
    benign: int = 200,
    benign_flagged: int = 4,
    malicious: int = 50,
    malicious_flagged: int = 40,
    truncated: bool = False,
    curve_points: int = 8,
) -> ThresholdSelection:
    """Return a frozen binary threshold selection with the given counts."""
    selected = status is SelectionStatus.SELECTED
    fpr = benign_flagged / benign if benign else None
    recall = malicious_flagged / malicious if malicious else None
    precision = (
        malicious_flagged / (malicious_flagged + benign_flagged)
        if (malicious_flagged + benign_flagged)
        else None
    )

    def point(index: int) -> ThresholdCurvePoint:
        """Return the confusion matrix of the *index*-th candidate threshold.

        A rate whose denominator is empty is ``None``, exactly as the curve
        point's own validator insists: no benign rows means no false-positive
        rate, not a false-positive rate of zero.
        """
        hits = malicious - index
        false_alarms = max(0, benign_flagged - index)
        flagged = hits + false_alarms
        return ThresholdCurvePoint(
            threshold=round(0.1 * (index + 1), 9),
            true_positives=hits,
            false_positives=false_alarms,
            true_negatives=benign - false_alarms,
            false_negatives=index,
            precision=round(hits / flagged, 9) if flagged else None,
            recall=round(hits / malicious, 9) if malicious else None,
            false_positive_rate=(round(false_alarms / benign, 9) if benign else None),
            f1=None,
        )

    curve = tuple(point(index) for index in range(curve_points))
    return ThresholdSelection.seal(
        status=status,
        objective=rx.config().thresholds.objective,
        score_kind=sx.ScoreKind.DECISION_SCORE,
        tie_break="lowest_threshold",
        selected_threshold=0.5 if selected else None,
        objective_value=recall if selected else None,
        row_count=benign + malicious,
        benign_row_count=benign,
        benign_flagged_count=benign_flagged if selected else None,
        malicious_row_count=malicious,
        malicious_flagged_count=malicious_flagged if selected else None,
        false_positive_rate=round(fpr, 9) if selected and fpr is not None else None,
        detection_rate=round(recall, 9) if selected and recall is not None else None,
        precision=(round(precision, 9) if selected and precision is not None else None),
        f1=None,
        support_status=MetricStatus.MEASURED
        if selected
        else MetricStatus.INSUFFICIENT_SUPPORT,
        failing_requirements=() if selected else ("min_validation_benign_rows",),
        max_false_positive_rate=0.5,
        min_detection_rate=0.1,
        min_validation_positive_rows=5,
        min_validation_benign_rows=10,
        distinct_score_count=curve_points,
        candidate_count=curve_points,
        candidates_truncated=truncated,
        curve=curve if selected else (),
        source_partition=ValidationPartition.VALIDATION_B,
        source_partition_fingerprint=sx.PARTITION_FINGERPRINT,
        model_id="model-under-test",
        model_content_fingerprint=sx.MODEL_FINGERPRINT,
        preprocessor_fingerprint=sx.PREPROCESSOR_FINGERPRINT,
        calibration_state_fingerprint=None,
        calibration_method=sx.CalibrationMethod.NONE,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        data_selected=selected,
    )


def quality_report(
    *,
    ece: float = 0.05,
    kind: CalibrationEvaluationKind = (
        CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION
    ),
    status: MetricStatus = MetricStatus.MEASURED,
) -> CalibrationReport:
    """Return a frozen calibration report of the given kind."""
    partition = (
        ValidationPartition.VALIDATION_B
        if kind is CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION
        else ValidationPartition.VALIDATION_A
    )
    measured = status is MetricStatus.MEASURED
    admissible = (
        kind is CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION
        and measured
        and ece <= 0.5
    )
    return CalibrationReport.seal(
        method=sx.CalibrationMethod.PLATT,
        evaluation_kind=kind,
        status=status,
        source_partition=partition,
        fit_source_partition=ValidationPartition.VALIDATION_A,
        admissible_as_champion_evidence=admissible,
        brier_score=0.1 if measured else None,
        expected_calibration_error=ece if measured else None,
        within_configured_error=(ece <= 0.5) if measured else None,
        max_expected_calibration_error=0.5,
        raw_score_brier=None,
        raw_score_brier_status=MetricStatus.UNAVAILABLE,
        raw_score_brier_unavailable_reason="raw_score_contract_not_supplied",
        raw_score_kind=None,
        bin_count=2,
        bins=(
            sx.ReliabilityBin(
                index=0,
                lower_edge=0.0,
                upper_edge=0.5,
                includes_upper_edge=False,
                row_count=125,
                positive_count=25,
                mean_predicted_probability=0.2,
                observed_positive_rate=0.2,
                status=MetricStatus.MEASURED,
            ),
            sx.ReliabilityBin(
                index=1,
                lower_edge=0.5,
                upper_edge=1.0,
                includes_upper_edge=True,
                row_count=125,
                positive_count=25,
                mean_predicted_probability=0.8,
                observed_positive_rate=0.8,
                status=MetricStatus.MEASURED,
            ),
        ),
        row_count=250,
        positive_count=50,
        negative_count=200,
        min_calibration_rows=10,
        min_reliability_bin_rows=1,
        failing_requirements=() if measured else ("min_calibration_rows",),
        validation_partition_fingerprint=sx.PARTITION_FINGERPRINT,
        model_content_fingerprint=sx.MODEL_FINGERPRINT,
        calibration_state_fingerprint=sx.OTHER_FINGERPRINT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )


def ranking(
    *,
    positives: int = 40,
    negatives: int = 40,
    perfect: bool = True,
    model_id: str = "model-under-test",
    score_kind: ScoreKind = ScoreKind.DECISION_SCORE,
) -> RankingEvidence:
    """Return exact ranking evidence with a known PR-AUC.

    Built through the real constructor from a real score sample, so a gate
    fixture cannot assert against a curve the ranking contract would refuse.
    """
    if perfect:
        scores = [1.0 - index / 200.0 for index in range(positives)]
        scores += [0.1 - index / 2000.0 for index in range(negatives)]
    else:
        scores = [0.5] * (positives + negatives)
    malicious = [True] * positives + [False] * negatives
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    sample = sx.binary_sample(
        [scores[index] for index in order],
        [malicious[index] for index in order],
        partition=ValidationPartition.VALIDATION_B,
        score_kind=score_kind,
        model_id=model_id,
    )
    evidence = build_ranking_evidence(
        sample, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )
    assert evidence is not None
    return evidence


def binary_inputs(**overrides: Any) -> BinaryGateInputs:
    """Return gate inputs for a candidate that clears everything by default."""
    fields: dict[str, Any] = {
        "catalog_model_id": "M-010",
        "threshold": threshold(),
        "calibration_quality": quality_report(),
        "calibration_required": True,
        "champion_eligible": True,
        "publishable": True,
        "artifact_verified": True,
        "ranking": ranking(),
        "baseline_ranking": ranking(perfect=False, model_id="reference-baseline"),
        "baseline_unavailable_reason": None,
        "lineage_matches": True,
        "lineage_mismatch_reason": None,
    }
    fields.update(overrides)
    return BinaryGateInputs(**fields)


def gate(results: tuple[GateEvidence, ...], gate_id: str) -> GateEvidence:
    """Return one gate's evidence by identifier."""
    return next(item for item in results if item.gate_id == gate_id)


# ---------------------------------------------------------------------------
# The gate set
# ---------------------------------------------------------------------------


def test_every_configured_mandatory_gate_is_reported() -> None:
    """No gate may silently disappear: a missing gate reads as a passed one."""
    results = evaluate_binary_gates(binary_inputs(), config=config())
    assert tuple(item.gate_id for item in results) == BINARY_GATE_IDS
    assert all(item.mandatory for item in results)


def test_every_gate_is_reported_even_when_nothing_can_be_measured() -> None:
    """A candidate with no artifacts still produces a full gate report."""
    results = evaluate_binary_gates(
        binary_inputs(threshold=None, calibration_quality=None, ranking=None),
        config=config(),
    )
    assert tuple(item.gate_id for item in results) == BINARY_GATE_IDS
    assert all(item.status is not GateStatus.PASS for item in results[:6])


def test_a_clean_candidate_passes_every_gate() -> None:
    """The positive path, so the negative ones below mean something."""
    results = evaluate_binary_gates(binary_inputs(), config=config())
    assert all(item.status is GateStatus.PASS for item in results), [
        (item.gate_id, item.reason)
        for item in results
        if item.status is not GateStatus.PASS
    ]


def test_a_mandatory_inconclusive_gate_blocks() -> None:
    """Inconclusive is not a soft pass. It stops promotion exactly as a fail does."""
    results = evaluate_binary_gates(
        binary_inputs(calibration_quality=None), config=config()
    )
    calibration = gate(results, "calibration_quality")
    assert calibration.status is GateStatus.INCONCLUSIVE
    assert calibration.blocking is True
    failed = gate(
        evaluate_binary_gates(binary_inputs(lineage_matches=False), config=config()),
        "lineage_compatibility",
    )
    assert failed.status is GateStatus.FAIL
    assert failed.blocking is True


def test_gate_identifiers_are_stable_and_unique() -> None:
    """Reason codes and gate names are compared; prose would be reworded."""
    assert len(set(BINARY_GATE_IDS)) == len(BINARY_GATE_IDS)
    assert len(set(CATEGORY_GATE_IDS)) == len(CATEGORY_GATE_IDS)
    for results in (
        evaluate_binary_gates(binary_inputs(), config=config()),
        evaluate_category_gates(category_inputs(), config=config()),
    ):
        for item in results:
            assert item.reason == item.reason.lower()
            assert " " not in item.reason


# ---------------------------------------------------------------------------
# Rates, numerators, denominators
# ---------------------------------------------------------------------------


def test_a_rate_carries_its_numerator_and_denominator() -> None:
    """A rate a reader cannot recompute is a rate a reader has to trust."""
    results = evaluate_binary_gates(binary_inputs(), config=config())
    fpr = gate(results, "false_positive_ceiling")
    assert fpr.rate is not None
    assert fpr.rate.numerator == 4
    assert fpr.rate.denominator == 200
    assert fpr.observed == pytest.approx(0.02, abs=1e-9)
    assert fpr.rate.value == pytest.approx(0.02, abs=1e-9)


def test_an_empty_denominator_is_unavailable_not_zero() -> None:
    """The specific misreading the whole rate contract exists to prevent."""
    evidence = RateEvidence.of(0, 0, confidence=0.95)
    assert evidence.value is None
    assert evidence.interval_lower is None
    assert evidence.interval_upper is None
    with pytest.raises(ValidationError, match="exists exactly when"):
        RateEvidence(
            numerator=0,
            denominator=0,
            value=0.0,
            interval_lower=None,
            interval_upper=None,
            confidence=0.95,
        )


def test_no_benign_rows_never_reaches_a_compliant_ceiling() -> None:
    """Nothing flagged out of nothing is not compliance, at either layer.

    Milestone 5 refuses to *seal* a selection over no benign rows at all, so the
    only artifact that can carry an empty denominator here is one that selected
    nothing -- and the gate reports an unavailable operating point rather than a
    held ceiling.
    """
    with pytest.raises(ValidationError):
        threshold(benign=0, benign_flagged=0)
    results = evaluate_binary_gates(
        binary_inputs(
            threshold=threshold(
                status=SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT,
                benign=0,
                benign_flagged=0,
            )
        ),
        config=config(),
    )
    fpr = gate(results, "false_positive_ceiling")
    assert fpr.status is GateStatus.INCONCLUSIVE
    assert fpr.reason == "operating_point_unavailable"
    assert fpr.observed is None
    assert fpr.blocking


# ---------------------------------------------------------------------------
# Wilson intervals
# ---------------------------------------------------------------------------


def test_the_wilson_interval_stays_inside_the_unit_interval() -> None:
    """Where the normal approximation runs past zero, Wilson does not."""
    interval = wilson_interval(0, 20, confidence=0.95)
    assert interval is not None
    lower, upper = interval
    assert lower == 0.0
    assert 0.0 < upper < 1.0


def test_the_wilson_interval_narrows_with_support() -> None:
    """The same rate known better is a narrower claim, and the interval says so."""
    thin = wilson_interval(1, 20, confidence=0.95)
    thick = wilson_interval(50, 1000, confidence=0.95)
    assert thin is not None and thick is not None
    assert (thin[1] - thin[0]) > (thick[1] - thick[0])


def test_the_wilson_interval_is_unavailable_for_an_empty_denominator() -> None:
    """An interval around a proportion of nothing is not a wide interval."""
    assert wilson_interval(0, 0) is None


def test_an_untabulated_confidence_level_is_refused() -> None:
    """A gate must not rest on an interpolated quantile nobody checked."""
    with pytest.raises(ValueError, match="tabulated"):
        wilson_interval(1, 10, confidence=0.975)


def test_gated_rates_publish_their_interval() -> None:
    """Uncertainty is published beside every rate a gate acted on."""
    results = evaluate_binary_gates(binary_inputs(), config=config())
    for gate_id in ("false_positive_ceiling", "detection_rate_floor"):
        evidence = gate(results, gate_id).rate
        assert evidence is not None
        assert evidence.interval_lower is not None
        assert evidence.interval_upper is not None
        assert evidence.interval_lower <= (evidence.value or 0.0)
        assert (evidence.value or 0.0) <= evidence.interval_upper


# ---------------------------------------------------------------------------
# The resolution rule
# ---------------------------------------------------------------------------


def test_a_ceiling_the_sample_cannot_resolve_is_inconclusive() -> None:
    """The rule this project exists to state.

    With fifty benign rows the smallest observable non-zero rate is 2%, so a 1%
    ceiling cannot be *held* by any threshold that fires. Appearing under it
    would mean flagging nothing, and the ceiling would not have been tested.
    """
    results = evaluate_binary_gates(
        binary_inputs(threshold=threshold(benign=50, benign_flagged=0)),
        config=config(max_fpr=0.01),
    )
    fpr = gate(results, "false_positive_ceiling")
    assert fpr.status is GateStatus.INCONCLUSIVE
    assert fpr.reason == "false_positive_rate_resolution"
    assert fpr.support_required == 100
    assert fpr.support_observed == 50
    # The rate is still published: the gate refuses to conclude, not to report.
    assert fpr.rate is not None
    assert fpr.rate.value == 0.0


def test_a_resolvable_ceiling_is_decided_on_the_point_estimate() -> None:
    """The configured criterion is a ceiling on the rate, and that is what is used.

    Requiring the interval's upper bound to clear the ceiling would be a
    stricter criterion than the one written down, and inventing an acceptance
    criterion during selection is what a predeclared gate set exists to prevent.
    """
    results = evaluate_binary_gates(
        binary_inputs(threshold=threshold(benign=1000, benign_flagged=5)),
        config=config(max_fpr=0.01),
    )
    fpr = gate(results, "false_positive_ceiling")
    assert fpr.status is GateStatus.PASS
    assert fpr.observed == pytest.approx(0.005, abs=1e-9)
    assert fpr.rate is not None
    # The interval genuinely straddles the ceiling; the gate still passes on the
    # point estimate, and publishes the width so a reader can see it.
    assert fpr.rate.interval_upper is not None
    assert fpr.rate.interval_upper > 0.005


def test_a_rate_above_the_ceiling_fails_rather_than_being_inconclusive() -> None:
    """A measured violation is a finding, not missing evidence."""
    results = evaluate_binary_gates(
        binary_inputs(threshold=threshold(benign=200, benign_flagged=80)),
        config=config(max_fpr=0.1),
    )
    fpr = gate(results, "false_positive_ceiling")
    assert fpr.status is GateStatus.FAIL
    assert fpr.reason == "above_ceiling"


def test_the_detection_floor_fails_when_measured_below() -> None:
    """The mirrored constraint, and a measured negative in the same way."""
    results = evaluate_binary_gates(
        binary_inputs(threshold=threshold(malicious=50, malicious_flagged=5)),
        config=config(min_recall=0.5),
    )
    recall = gate(results, "detection_rate_floor")
    assert recall.status is GateStatus.FAIL
    assert recall.observed == pytest.approx(0.1, abs=1e-9)
    assert recall.reason == "below_floor"


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------


def test_thin_class_support_is_inconclusive_not_failed() -> None:
    """Below the floor nothing downstream can mean anything, so nothing is claimed."""
    from password_attack_detector.ml.schemas import SupportRequirement

    settings = rx.config(
        support=SupportRequirement(
            min_train_positive_rows=1,
            min_validation_positive_rows=500,
            min_validation_benign_rows=500,
            min_rows_per_category=1,
        ),
        gates=ChampionGateConfig(
            min_pr_auc_gain_over_baseline=0.0,
            max_false_positive_rate=0.5,
            min_detection_rate=0.1,
            max_expected_calibration_error=0.5,
        ),
    )
    results = evaluate_binary_gates(binary_inputs(), config=settings)
    support = gate(results, "validation_support")
    assert support.status is GateStatus.INCONCLUSIVE
    assert support.reason == "min_validation_class_rows"
    assert support.support_required == 500
    assert support.support_observed == 50


def test_an_unselected_threshold_makes_the_operating_gate_report_why() -> None:
    """No feasible threshold is a measurement; insufficient support is not."""
    infeasible = gate(
        evaluate_binary_gates(
            binary_inputs(
                threshold=threshold(status=SelectionStatus.NO_FEASIBLE_THRESHOLD)
            ),
            config=config(),
        ),
        "operating_threshold",
    )
    assert infeasible.status is GateStatus.FAIL
    assert infeasible.reason == "no_feasible_threshold"

    unresolved = gate(
        evaluate_binary_gates(
            binary_inputs(
                threshold=threshold(
                    status=SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
                )
            ),
            config=config(),
        ),
        "operating_threshold",
    )
    assert unresolved.status is GateStatus.INCONCLUSIVE
    assert unresolved.reason == "insufficient_validation_support"


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def test_an_unavailable_baseline_makes_the_gain_gate_inconclusive() -> None:
    """The gate is never waived, and the baseline is never promoted in its place."""
    results = evaluate_binary_gates(
        binary_inputs(
            baseline_ranking=None,
            baseline_unavailable_reason="reference_baseline_run_missing",
        ),
        config=config(),
    )
    baseline = gate(results, "baseline_pr_auc_gain")
    assert baseline.status is GateStatus.INCONCLUSIVE
    assert baseline.reason == "reference_baseline_run_missing"
    assert baseline.blocking is True


def test_no_gain_over_the_baseline_fails() -> None:
    """A candidate qualifies by beating the comparator, not by matching it."""
    results = evaluate_binary_gates(
        binary_inputs(baseline_ranking=ranking(model_id="strong-reference")),
        config=config(min_gain=0.05),
    )
    baseline = gate(results, "baseline_pr_auc_gain")
    assert baseline.status is GateStatus.FAIL
    assert baseline.reason == "no_gain_over_baseline"
    assert baseline.observed == 0.0


def test_the_gain_gate_measures_the_metric_the_configuration_names() -> None:
    """PR-AUC, step-wise, on the declared scoring stage -- said in the verdict."""
    baseline = gate(
        evaluate_binary_gates(binary_inputs(), config=config()),
        "baseline_pr_auc_gain",
    )
    assert baseline.metric == f"{RANKING_METRIC_NAME}_gain"
    assert RANKING_METRIC_NAME in (baseline.constraint or "")
    assert PR_AUC_INTEGRATION in (baseline.constraint or "")
    assert str(DISCRIMINATION_SCORE_KIND) in (baseline.constraint or "")
    assert "exact" in (baseline.constraint or "")


def test_the_operating_threshold_grid_cannot_reach_the_gain_gate() -> None:
    """A bounded threshold search is threshold evidence, not ranking evidence.

    The candidate's frozen operating-point curve is replaced with a truncated
    one -- the state that used to turn the verdict into an approximation -- and
    the exact metric does not move, because it never came from that curve.
    """
    exhaustive = evaluate_binary_gates(binary_inputs(), config=config())
    bounded = evaluate_binary_gates(
        binary_inputs(threshold=threshold(truncated=True, curve_points=2)),
        config=config(),
    )
    measured = gate(exhaustive, "baseline_pr_auc_gain")
    unchanged = gate(bounded, "baseline_pr_auc_gain")
    assert unchanged.observed == measured.observed
    assert unchanged.status is measured.status
    assert unchanged.reason == "beats_reference_baseline"
    assert "sampled" not in (unchanged.reason or "")


def test_no_verdict_rests_on_approximate_evidence() -> None:
    """There is no reason code by which an approximation could pass this gate."""
    for truncated in (False, True):
        for result in evaluate_binary_gates(
            binary_inputs(threshold=threshold(truncated=truncated)), config=config()
        ):
            assert "sampled" not in (result.reason or "")
            assert "approx" not in (result.reason or "")


def test_a_candidate_without_exact_evidence_is_inconclusive() -> None:
    """Missing ranking evidence blocks; it never becomes a gain of zero."""
    results = evaluate_binary_gates(binary_inputs(ranking=None), config=config())
    baseline = gate(results, "baseline_pr_auc_gain")
    assert baseline.status is GateStatus.INCONCLUSIVE
    assert baseline.reason == "ranking_evidence_unavailable"
    assert baseline.observed is None
    assert baseline.blocking is True


def test_two_scoring_stages_can_never_reach_one_comparison() -> None:
    """A calibrated score and a raw one are two quantities, not two readings.

    Enforced where the evidence is built rather than where it is compared: there
    is no ranking record at another stage for a gate to be handed, so the gate
    has no mixed comparison to refuse.
    """
    calibrated = sx.binary_sample(
        [0.9, 0.8, 0.2, 0.1],
        [True, True, False, False],
        partition=ValidationPartition.VALIDATION_B,
        score_kind=ScoreKind.CALIBRATED_PROBABILITY,
        calibration_state_fingerprint=sx.OTHER_FINGERPRINT,
        calibration_method=sx.CalibrationMethod.PLATT,
    )
    with pytest.raises(ModelTrainingError, match="decision_score"):
        build_ranking_evidence(calibrated, ml_config_fingerprint=sx.CONFIG_FINGERPRINT)

    both = binary_inputs()
    assert both.ranking is not None and both.baseline_ranking is not None
    assert both.ranking.score_kind is both.baseline_ranking.score_kind
    assert both.ranking.score_kind is DISCRIMINATION_SCORE_KIND


def test_tampered_evidence_never_reaches_a_gate() -> None:
    """A record whose declared metric was edited does not survive validation."""
    tampered = ranking(perfect=False).model_copy(update={"pr_auc": 0.0})
    with pytest.raises(ValidationError):
        binary_inputs(baseline_ranking=tampered)


# ---------------------------------------------------------------------------
# Calibration evidence
# ---------------------------------------------------------------------------


def test_only_an_out_of_sample_report_satisfies_the_calibration_gate() -> None:
    """An in-sample diagnostic measures a calibrator on the rows that fitted it."""
    results = evaluate_binary_gates(
        binary_inputs(
            calibration_quality=quality_report(
                kind=CalibrationEvaluationKind.IN_SAMPLE_FIT_DIAGNOSTIC
            )
        ),
        config=config(),
    )
    calibration = gate(results, "calibration_quality")
    assert calibration.status is GateStatus.INCONCLUSIVE
    assert calibration.reason == "in_sample_diagnostic_is_not_evidence"


def test_missing_required_calibration_blocks_the_candidate() -> None:
    """A candidate cannot pass by omitting the calibrator its protocol requires."""
    results = evaluate_binary_gates(
        binary_inputs(calibration_quality=None), config=config()
    )
    calibration = gate(results, "calibration_quality")
    assert calibration.status is GateStatus.INCONCLUSIVE
    assert calibration.reason == "calibration_evidence_absent"


def test_calibration_is_not_applicable_to_a_reference_baseline() -> None:
    """Not applicable, and recorded as such rather than fabricated or failed."""
    results = evaluate_binary_gates(
        binary_inputs(calibration_required=False, calibration_quality=None),
        config=config(),
    )
    calibration = gate(results, "calibration_quality")
    assert calibration.status is GateStatus.PASS
    assert calibration.reason == "calibration_not_applicable"


def test_calibration_above_the_ceiling_fails() -> None:
    """A measured miscalibration is a finding."""
    results = evaluate_binary_gates(
        binary_inputs(calibration_quality=quality_report(ece=0.4)),
        config=config(max_ece=0.1),
    )
    calibration = gate(results, "calibration_quality")
    assert calibration.status is GateStatus.FAIL


def test_calibration_on_inadequate_support_is_inconclusive() -> None:
    """A small ECE over eighty rows is not evidence of calibration."""
    results = evaluate_binary_gates(
        binary_inputs(
            calibration_quality=quality_report(status=MetricStatus.INSUFFICIENT_SUPPORT)
        ),
        config=config(),
    )
    calibration = gate(results, "calibration_quality")
    assert calibration.status is GateStatus.INCONCLUSIVE
    assert calibration.reason == "calibration_support_inadequate"


def test_a_missing_raw_brier_comparator_is_not_a_calibration_failure() -> None:
    """Most decision scores carry no promise of living on ``[0, 1]``.

    The configured gate is a ceiling on calibration error, not a requirement
    that a raw comparator exist, so its absence is not held against a candidate.
    """
    report = quality_report()
    assert report.raw_score_brier is None
    results = evaluate_binary_gates(
        binary_inputs(calibration_quality=report), config=config()
    )
    assert gate(results, "calibration_quality").status is GateStatus.PASS


# ---------------------------------------------------------------------------
# Serializer and lineage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"champion_eligible": False}, "family_not_champion_eligible"),
        ({"publishable": False}, "serializer_unproven"),
        ({"artifact_verified": False}, "artifact_verification_failed"),
    ],
)
def test_the_serializer_gate_names_which_condition_failed(
    override: dict[str, Any], reason: str
) -> None:
    """A model that cannot be stored and reloaded exactly is never promoted."""
    results = evaluate_binary_gates(binary_inputs(**override), config=config())
    serializer = gate(results, "serializer_eligibility")
    assert serializer.status is GateStatus.FAIL
    assert serializer.reason == reason


def test_a_lineage_mismatch_fails_with_its_reason() -> None:
    """Two runs from different experiments were not comparing like with like."""
    results = evaluate_binary_gates(
        binary_inputs(
            lineage_matches=False,
            lineage_mismatch_reason="lineage_differs_from_reference_baseline",
        ),
        config=config(),
    )
    lineage = gate(results, "lineage_compatibility")
    assert lineage.status is GateStatus.FAIL
    assert lineage.reason == "lineage_differs_from_reference_baseline"


# ---------------------------------------------------------------------------
# Category gates
# ---------------------------------------------------------------------------


def abstention(
    *,
    status: SelectionStatus = SelectionStatus.SELECTED,
    data_selected: bool = True,
    precision: float | None = 0.9,
    floor: float = 0.8,
    rows: int = 120,
    per_class: int = 40,
) -> CategoryAbstentionSelection:
    """Return a frozen category abstention selection.

    ``floor`` is the criterion the head was *selected* under, which is not
    necessarily the one in force now; a selected head always clears its own
    floor, so a drifted configuration is the only way the precision gate can
    reach a measured failure.
    """
    selected = status is SelectionStatus.SELECTED and data_selected
    covered = rows if selected else None
    correct = int(rows * (precision or 0.0)) if selected else None
    return CategoryAbstentionSelection.seal(
        status=status,
        objective="max_coverage_at_min_precision",
        tie_break="lowest_threshold",
        abstain_label="unknown",
        min_category_score=0.35,
        data_selected=selected,
        coverage=1.0 if selected else None,
        known_category_precision=precision if selected else None,
        known_category_error=(1.0 - precision) if selected and precision else None,
        covered_count=covered,
        correct_count=correct,
        known_malicious_row_count=rows,
        class_order=sx.CATEGORY_ORDER,
        class_support=tuple(
            CategoryClassSupport(
                class_name=name,
                row_count=per_class,
                covered_count=per_class if selected else None,
                correct_count=int(per_class * (precision or 0.0)) if selected else None,
            )
            for name in sx.CATEGORY_ORDER
        ),
        min_known_category_precision=floor,
        min_rows_per_category=2,
        min_known_malicious_rows=5,
        support_status=MetricStatus.MEASURED,
        failing_requirements=() if selected else ("min_known_malicious_rows",),
        distinct_score_count=8,
        candidate_count=8,
        candidates_truncated=False,
        source_partition=ValidationPartition.VALIDATION_B,
        source_partition_fingerprint=sx.PARTITION_FINGERPRINT,
        category_model_id="category-model",
        category_model_content_fingerprint=sx.MODEL_FINGERPRINT,
        preprocessor_fingerprint=sx.PREPROCESSOR_FINGERPRINT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )


def category_inputs(**overrides: Any) -> CategoryGateInputs:
    """Return category gate inputs that clear everything by default."""
    fields: dict[str, Any] = {
        "catalog_model_id": "M-010",
        "abstention": abstention(),
        "champion_eligible": True,
        "publishable": True,
        "artifact_verified": True,
        "lineage_matches": True,
        "lineage_mismatch_reason": None,
    }
    fields.update(overrides)
    return CategoryGateInputs(**fields)


def test_every_category_gate_is_reported() -> None:
    """The same completeness rule, on the other track."""
    results = evaluate_category_gates(category_inputs(), config=config())
    assert tuple(item.gate_id for item in results) == CATEGORY_GATE_IDS
    assert all(item.status is GateStatus.PASS for item in results)


def test_a_fallback_abstention_threshold_does_not_satisfy_its_gate() -> None:
    """A predeclared constant is a usable threshold and not a measurement."""
    results = evaluate_category_gates(
        category_inputs(
            abstention=abstention(
                status=SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT,
                data_selected=False,
            )
        ),
        config=config(),
    )
    point = gate(results, "category_abstention_point")
    assert point.status is GateStatus.INCONCLUSIVE
    assert point.blocking is True


def test_thin_per_class_support_is_inconclusive() -> None:
    """Checked per class: an aggregate would hide the class nobody evaluated."""
    settings = rx.config(
        category=CategoryConfig(
            min_category_score=0.2,
            min_rows_per_category=500,
            min_known_category_precision=0.4,
            min_known_malicious_rows=5,
        )
    )
    results = evaluate_category_gates(category_inputs(), config=settings)
    support = gate(results, "category_class_support")
    assert support.status is GateStatus.INCONCLUSIVE
    assert support.reason == "min_rows_per_category"
    assert support.support_observed == 40
    assert support.support_required == 500


def test_category_precision_below_the_floor_fails() -> None:
    """A head selected under a looser floor is judged by the floor in force."""
    strict = CategoryConfig(
        min_category_score=0.2,
        min_rows_per_category=2,
        min_known_category_precision=0.9,
        min_known_malicious_rows=5,
        abstention_search_grid_size=64,
    )
    results = evaluate_category_gates(
        category_inputs(abstention=abstention(precision=0.5, floor=0.4)),
        config=config(category=strict),
    )
    precision = gate(results, "category_precision_floor")
    assert precision.status is GateStatus.FAIL
    assert precision.rate is not None
    assert precision.rate.denominator == 120


# ---------------------------------------------------------------------------
# The gate configuration fingerprint
# ---------------------------------------------------------------------------


def test_the_gate_configuration_fingerprint_covers_every_acceptance_criterion() -> None:
    """Two selections under different criteria are different selections."""
    baseline = gate_config_fingerprint(config())
    assert gate_config_fingerprint(config()) == baseline
    assert gate_config_fingerprint(config(max_fpr=0.2)) != baseline
    assert gate_config_fingerprint(config(min_recall=0.9)) != baseline
    assert gate_config_fingerprint(config(min_gain=0.5)) != baseline
    assert gate_config_fingerprint(config(max_ece=0.05)) != baseline


def test_the_ranking_policy_is_part_of_the_gate_fingerprint() -> None:
    """Changing how ties break changes which candidate wins, so it is identity."""
    from password_attack_detector.ml.config import ChampionSelectionConfig

    baseline = gate_config_fingerprint(config())
    reordered = gate_config_fingerprint(
        config(
            selection=ChampionSelectionConfig(
                tie_break_order=(
                    "max_baseline_pr_auc_gain",
                    "min_false_positive_rate",
                    "min_expected_calibration_error",
                    "catalog_model_id",
                )
            )
        )
    )
    assert reordered != baseline


def test_a_tie_break_chain_must_end_deterministically() -> None:
    """Without a final total key, order would depend on the filesystem."""
    from password_attack_detector.ml.config import ChampionSelectionConfig

    with pytest.raises(ValidationError, match="catalog_model_id"):
        ChampionSelectionConfig(tie_break_order=("min_false_positive_rate",))
    with pytest.raises(ValidationError, match="repeats"):
        ChampionSelectionConfig(
            tie_break_order=(
                "min_false_positive_rate",
                "min_false_positive_rate",
                "catalog_model_id",
            )
        )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_a_decided_gate_must_have_observed_something() -> None:
    """A verdict with nothing behind it is not a verdict."""
    with pytest.raises(ValidationError, match="without an observation"):
        GateEvidence(
            gate_id="probe",
            metric="thing",
            status=GateStatus.PASS,
            mandatory=True,
            reason="ok",
            constraint="something",
        )


def test_no_gate_reports_a_non_finite_number() -> None:
    """NaN is stable in a digest and meaningless in a report."""
    with pytest.raises(ValidationError, match="finite"):
        GateEvidence(
            gate_id="probe",
            metric="thing",
            status=GateStatus.FAIL,
            mandatory=True,
            reason="bad",
            observed=math.inf,
            constraint="something",
        )


def test_no_gate_evidence_carries_an_identifier() -> None:
    """Gates report counts, rates, and codes."""
    import json

    results = evaluate_binary_gates(binary_inputs(), config=config())
    rendered = json.dumps([item.model_dump(mode="json") for item in results])
    for banned in ("e0000", "campaign", "u:", "/home/", "anchor_event_id"):
        assert banned not in rendered
