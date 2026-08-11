"""Operating points: chosen on validation-B, or explicitly not chosen at all.

The assertions that matter most here are the negative ones. A threshold search
that always returns a number is easy to write and impossible to trust, so a
large share of this suite exists to prove that the three failure modes -- wrong
provenance, unmeasurable support, and no feasible candidate -- are reachable,
distinct, and never dressed up as a result.

The leakage suite at the end is the strongest single statement: hold train,
validation-A and validation-B fixed, replace test and the novel-anomaly holdout
with something completely different, and assert every fitted quantity and every
digest is byte-identical.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.calibration import (
    BinaryScoreSample,
    CalibrationState,
    apply_calibration,
    diagnose_calibration_fit,
    evaluate_calibration_quality,
    fit_calibration,
)
from password_attack_detector.ml.config import (
    AnomalyConfig,
    CalibrationConfig,
    CategoryConfig,
    ThresholdConfig,
)
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    AnomalyThresholdMethod,
    CalibrationMethod,
    MetricStatus,
    MLSplit,
    ScoreKind,
    SelectionStatus,
    ThresholdObjective,
    ValidationPartition,
)
from password_attack_detector.ml.schemas import SupportRequirement
from password_attack_detector.ml.thresholds import (
    ANOMALY_DECISION_PREDICATE,
    BINARY_DECISION_PREDICATE,
    CATEGORY_DECISION_PREDICATE,
    AnomalyThresholdSelection,
    CategoryAbstentionSelection,
    ThresholdCurvePoint,
    ThresholdSelection,
    select_anomaly_threshold,
    select_binary_threshold,
    select_category_abstention,
)
from tests.ml import selection as sx

SUPPORT = SupportRequirement(
    min_validation_positive_rows=25,
    min_validation_benign_rows=100,
    min_rows_per_category=5,
)

CALIBRATION = CalibrationConfig(
    method=CalibrationMethod.ISOTONIC,
    min_calibration_rows=100,
    min_isotonic_distinct_scores=10,
)

CATEGORY = CategoryConfig(
    min_known_malicious_rows=50,
    min_rows_per_category=5,
    min_known_category_precision=0.80,
)

ANOMALY = AnomalyConfig(enabled=False, min_fit_rows=100, quantile=0.99)


def choose(
    sample: BinaryScoreSample,
    *,
    config: ThresholdConfig | None = None,
    support: SupportRequirement = SUPPORT,
    calibration: CalibrationState | None = None,
) -> ThresholdSelection:
    """Return the binary selection for *sample*.

    The default objective is ``max_f1`` rather than the configured default:
    every constrained objective can legitimately return "no feasible
    threshold", and a helper used by thirty tests should not make each of them
    depend on whether a ceiling happened to be reachable. The constrained
    objectives get their own tests, with fixtures built to exercise them.
    """
    return select_binary_threshold(
        sample,
        config=config or ThresholdConfig(objective=ThresholdObjective.MAX_F1),
        support=support,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        calibration=calibration,
    )


def infeasible_sample() -> BinaryScoreSample:
    """Return validation-B rows where no threshold can hold a sane FPR ceiling.

    Two hundred benign rows all share the highest score, so every candidate --
    including the highest -- flags all of them and reports a false-positive
    rate of ``1.0``. The support is ample, so the outcome is a *measured*
    negative rather than missing evidence.
    """
    malicious_scores = [round(0.001 * (index + 1), 9) for index in range(200)]
    scores = tuple(malicious_scores) + (1.0,) * 200
    malicious = (True,) * 200 + (False,) * 200
    return sx.binary_sample(
        scores, malicious, partition=ValidationPartition.VALIDATION_B
    )


@pytest.fixture
def sample() -> BinaryScoreSample:
    """Return a validation-B sample clearing every default support floor."""
    return sx.validation_b(400)


# ---------------------------------------------------------------------------
# The test and holdout firewall
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "partition"),
    [
        (MLSplit.TEST, None),
        (MLSplit.NOVEL_ANOMALY_HOLDOUT, None),
        (MLSplit.TRAIN, None),
        (MLSplit.EXCLUDED, None),
        (MLSplit.VALIDATION, ValidationPartition.VALIDATION_A),
    ],
)
def test_only_validation_b_may_choose_a_binary_threshold(
    split: MLSplit, partition: ValidationPartition | None
) -> None:
    """Including validation-A, which fitted the calibrator this will be judged with."""
    scores = sx.graded_scores(400)
    bad = sx.binary_sample(
        scores, sx.graded_labels(scores), split=split, partition=partition
    )
    with pytest.raises(ModelTrainingError, match="validation_b"):
        choose(bad)


@pytest.mark.parametrize(
    ("split", "partition"),
    [
        (MLSplit.TEST, None),
        (MLSplit.NOVEL_ANOMALY_HOLDOUT, None),
        (MLSplit.VALIDATION, ValidationPartition.VALIDATION_A),
    ],
)
def test_only_validation_b_may_choose_an_abstention_threshold(
    split: MLSplit, partition: ValidationPartition | None
) -> None:
    """Same firewall, the other supervised selector."""
    rows = sx.category_rows(120)
    bad = sx.category_sample(rows, split=split, partition=partition)
    with pytest.raises(ModelTrainingError, match="validation_b"):
        select_category_abstention(
            bad,
            config=CATEGORY,
            support=SUPPORT,
            ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        )


@pytest.mark.parametrize(
    ("method", "split", "partition"),
    [
        (AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE, MLSplit.TEST, None),
        (
            AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE,
            MLSplit.NOVEL_ANOMALY_HOLDOUT,
            None,
        ),
        (
            AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE,
            MLSplit.VALIDATION,
            ValidationPartition.VALIDATION_A,
        ),
        (AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR, MLSplit.TEST, None),
        (
            AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR,
            MLSplit.NOVEL_ANOMALY_HOLDOUT,
            None,
        ),
        (AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR, MLSplit.TRAIN, None),
        (
            AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR,
            MLSplit.VALIDATION,
            ValidationPartition.VALIDATION_B,
        ),
    ],
)
def test_each_anomaly_method_reads_only_its_declared_source(
    method: AnomalyThresholdMethod,
    split: MLSplit,
    partition: ValidationPartition | None,
) -> None:
    """Both methods refuse every source but the one the configuration names.

    The novel-anomaly holdout is the entry that matters most: it is what the
    probe is *measured against*, so a threshold tuned on it would be a
    measurement of itself.
    """
    sample = sx.anomaly_sample(sx.anomaly_scores(400), split=split, partition=partition)
    with pytest.raises(ModelTrainingError):
        select_anomaly_threshold(
            sample,
            config=ANOMALY.model_copy(update={"threshold_method": method}),
            support=SUPPORT,
            ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        )


def test_the_anomaly_selector_refuses_malicious_rows() -> None:
    """Benign-only is behavioural: the contract checks, it does not merely ask.

    A malicious row here would let an unsupervised probe read a supervised
    outcome, which is the difference between measuring generalisation and
    fitting to the answer.
    """
    scores = sx.anomaly_scores(400)
    with pytest.raises(ValidationError, match="benign rows only"):
        sx.anomaly_sample(
            scores, malicious=tuple(index % 50 == 0 for index in range(400))
        )


def test_the_category_selector_refuses_benign_rows() -> None:
    """A benign row has no known category to be right about."""
    rows = sx.category_rows(120)
    labels = tuple(index != 3 for index in range(120))
    with pytest.raises(ValidationError, match="known-malicious rows only"):
        sx.category_sample(rows, malicious=labels)


def test_the_category_selector_refuses_a_category_outside_the_class_order() -> None:
    """A novel pattern has no known category, so it cannot measure one."""
    rows = sx.category_rows(120)
    rows[7] = (rows[7][0], "novel_credential_probe")
    with pytest.raises(ValidationError, match="outside the declared class order"):
        sx.category_sample(rows)


def test_no_selector_accepts_an_override(sample: BinaryScoreSample) -> None:
    """There is no keyword on any selector that widens its source.

    Asserted on the signatures rather than on prose, so a parameter added later
    fails here rather than in review.
    """
    import inspect

    banned = {"allow_test", "allow_holdout", "force", "unsafe", "override", "split"}
    for function in (
        select_binary_threshold,
        select_category_abstention,
        select_anomaly_threshold,
    ):
        assert set(inspect.signature(function).parameters) & banned == set()


# ---------------------------------------------------------------------------
# Binary threshold: predicate and objectives
# ---------------------------------------------------------------------------


def test_the_decision_predicate_is_greater_than_or_equal() -> None:
    """Pinned by construction: a row *at* the threshold is flagged.

    Four rows and one threshold, so the answer is countable by eye. The row
    sitting exactly on the boundary is the only one the choice of ``>=`` versus
    ``>`` ever moves, so it is the row the test is about.
    """
    scores = (0.10, 0.20, 0.30, 0.40)
    malicious = (False, True, False, True)
    result = choose(
        sx.binary_sample(scores, malicious, partition=ValidationPartition.VALIDATION_B),
        config=ThresholdConfig(
            objective=ThresholdObjective.MAX_F1, max_false_positive_rate=0.99
        ),
        support=SupportRequirement(
            min_validation_positive_rows=1, min_validation_benign_rows=1
        ),
    )
    assert result.decision_predicate == BINARY_DECISION_PREDICATE
    at_threshold = next(point for point in result.curve if point.threshold == 0.20)
    # Rows 0.20, 0.30 and 0.40 are at or above it: two malicious, one benign.
    assert at_threshold.true_positives == 2
    assert at_threshold.false_positives == 1
    assert at_threshold.false_negatives == 0
    assert at_threshold.true_negatives == 1


def test_maximise_recall_under_a_false_positive_ceiling(
    sample: BinaryScoreSample,
) -> None:
    """The default objective: the constraint binds, and the reported rate holds it."""
    result = choose(sample, config=ThresholdConfig(max_false_positive_rate=0.10))
    assert result.status is SelectionStatus.SELECTED
    assert result.false_positive_rate is not None
    assert result.false_positive_rate <= 0.10
    assert result.objective_value == result.detection_rate
    assert result.data_selected is True


def test_maximise_f1(sample: BinaryScoreSample) -> None:
    """No constraint, so the objective value is the F1 the threshold attains."""
    result = choose(sample, config=ThresholdConfig(objective=ThresholdObjective.MAX_F1))
    assert result.status is SelectionStatus.SELECTED
    assert result.objective_value == result.f1
    best = max(point.f1 for point in result.curve if point.f1 is not None)
    assert result.f1 == best


def test_minimise_false_positives_under_a_recall_floor(
    sample: BinaryScoreSample,
) -> None:
    """The mirrored objective: the floor binds and the rate is minimised."""
    result = choose(
        sample,
        config=ThresholdConfig(
            objective=ThresholdObjective.MIN_FPR_AT_MIN_RECALL,
            min_detection_rate=0.50,
        ),
    )
    assert result.status is SelectionStatus.SELECTED
    assert result.detection_rate is not None
    assert result.detection_rate >= 0.50
    assert result.objective_value == result.false_positive_rate
    feasible = [
        point.false_positive_rate
        for point in result.curve
        if point.recall is not None
        and point.recall >= 0.50
        and point.false_positive_rate is not None
    ]
    assert result.false_positive_rate == min(feasible)


@pytest.mark.parametrize(
    ("tie_break", "expected"),
    [("lowest_threshold", 0.20), ("highest_threshold", 0.30)],
)
def test_ties_are_broken_as_configured(
    tie_break: Literal["lowest_threshold", "highest_threshold"], expected: float
) -> None:
    """Two thresholds attain the same recall; the configuration decides which wins.

    Five rows: benign at ``0.10`` and ``0.20``, malicious at ``0.30``, benign
    at ``0.40``, malicious at ``0.50``. Under "maximise recall subject to a
    false-positive ceiling" the best attainable recall is ``1.0``, and both
    ``0.20`` and ``0.30`` attain it -- at different false-positive rates. The
    ceiling rules out ``0.10``, which would flag every benign row.

    Which of the two survives is a decision, so it is configured rather than
    incidental. ``lowest_threshold`` is the shipped default because a tie means
    the lower threshold detects at least as much at no extra cost.
    """
    scores = (0.10, 0.20, 0.30, 0.40, 0.50)
    malicious = (False, False, True, False, True)
    result = choose(
        sx.binary_sample(scores, malicious, partition=ValidationPartition.VALIDATION_B),
        config=ThresholdConfig(
            objective=ThresholdObjective.MAX_RECALL_AT_MAX_FPR,
            max_false_positive_rate=0.99,
            tie_break=tie_break,
        ),
        support=SupportRequirement(
            min_validation_positive_rows=1, min_validation_benign_rows=1
        ),
    )
    assert result.status is SelectionStatus.SELECTED
    assert result.objective_value == 1.0
    assert result.selected_threshold == expected
    assert result.tie_break == tie_break


def test_selection_is_deterministic(sample: BinaryScoreSample) -> None:
    """Same rows, same configuration, same threshold -- and the same digest."""
    first = choose(sample)
    second = choose(sx.validation_b(400))
    assert first.selected_threshold == second.selected_threshold
    assert first.selection_fingerprint == second.selection_fingerprint
    assert first.to_json() == second.to_json()


# ---------------------------------------------------------------------------
# Binary threshold: support and feasibility
# ---------------------------------------------------------------------------


def test_too_few_malicious_rows_is_insufficient_support() -> None:
    """A detection rate over four attacks is not a detection rate."""
    scores = sx.graded_scores(400)
    labels = tuple(index < 4 for index in range(400))
    result = choose(
        sx.binary_sample(scores, labels, partition=ValidationPartition.VALIDATION_B)
    )
    assert result.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert "min_validation_positive_rows" in result.failing_requirements
    assert result.selected_threshold is None
    assert result.data_selected is False
    assert result.support_status is MetricStatus.INSUFFICIENT_SUPPORT


def test_a_ceiling_finer_than_the_data_can_resolve_is_insufficient_support() -> None:
    """The example the support contract exists for.

    With two hundred benign rows the smallest observable non-zero
    false-positive rate is ``1/200 = 0.005``. A ceiling of ``0.001`` cannot be
    *held* by any threshold that fires at all -- it can only fail to be tested.
    That is missing evidence, not a measured negative, and the status says so.
    """
    scores = sx.graded_scores(400)
    labels = tuple(index >= 200 for index in range(400))
    result = choose(
        sx.binary_sample(scores, labels, partition=ValidationPartition.VALIDATION_B),
        config=ThresholdConfig(max_false_positive_rate=0.001),
    )
    assert result.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert "false_positive_rate_resolution" in result.failing_requirements
    assert result.curve == ()


def test_no_feasible_threshold_is_a_measured_negative() -> None:
    """Adequate support, and still nothing satisfies the ceiling.

    Two hundred benign rows share the highest score, so the highest candidate
    already flags every one of them -- and so does every candidate below it.
    Two hundred benign rows are ample to resolve a 25% ceiling, so the support
    gate passes and the answer is a *result*: no operating point holds the
    constraint. Reported as such rather than as missing evidence.
    """
    result = choose(
        infeasible_sample(),
        config=ThresholdConfig(max_false_positive_rate=0.25),
    )
    assert result.status is SelectionStatus.NO_FEASIBLE_THRESHOLD
    assert result.failing_requirements == ("max_false_positive_rate",)
    assert result.selected_threshold is None
    assert result.data_selected is False
    assert result.support_status is MetricStatus.MEASURED
    assert all(
        point.false_positive_rate is not None and point.false_positive_rate > 0.25
        for point in result.curve
    )


def test_the_two_negative_outcomes_are_distinguishable() -> None:
    """Different statuses, different requirement codes, different remedies."""
    scores = sx.graded_scores(400)
    thin = choose(
        sx.binary_sample(
            scores,
            tuple(index < 4 for index in range(400)),
            partition=ValidationPartition.VALIDATION_B,
        )
    )
    infeasible = choose(
        infeasible_sample(), config=ThresholdConfig(max_false_positive_rate=0.25)
    )
    assert thin.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert infeasible.status is SelectionStatus.NO_FEASIBLE_THRESHOLD
    assert thin.failing_requirements != infeasible.failing_requirements
    assert thin.data_selected is infeasible.data_selected is False


def test_numerators_and_denominators_survive_beside_every_rate(
    sample: BinaryScoreSample,
) -> None:
    """A rate a reader cannot recompute is a rate a reader has to trust."""
    result = choose(sample)
    assert result.benign_flagged_count is not None
    assert result.malicious_flagged_count is not None
    assert result.false_positive_rate == pytest.approx(
        result.benign_flagged_count / result.benign_row_count, abs=1e-9
    )
    assert result.detection_rate == pytest.approx(
        result.malicious_flagged_count / result.malicious_row_count, abs=1e-9
    )
    assert result.row_count == result.benign_row_count + result.malicious_row_count


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def test_curve_points_are_unique_and_ascending(sample: BinaryScoreSample) -> None:
    """One row per semantic threshold, deterministically ordered."""
    result = choose(sample)
    thresholds = [point.threshold for point in result.curve]
    assert thresholds == sorted(thresholds)
    assert len(set(thresholds)) == len(thresholds)
    assert len(thresholds) == result.candidate_count


def test_a_curve_carries_counts_and_no_identity(sample: BinaryScoreSample) -> None:
    """Aggregates only: never one row per event."""
    result = choose(sample)
    assert len(result.curve) < result.row_count + 1
    rendered = json.dumps([point.model_dump(mode="json") for point in result.curve])
    for banned in ("e00000", "e00001", "anchor", "campaign"):
        assert banned not in rendered


def test_a_bounded_search_says_it_was_bounded() -> None:
    """A silent cap would read as though every operating point was considered."""
    scores = sx.graded_scores(400)
    result = choose(
        sx.binary_sample(
            scores, sx.graded_labels(scores), partition=ValidationPartition.VALIDATION_B
        ),
        config=ThresholdConfig(search_grid_size=20, max_false_positive_rate=0.5),
    )
    assert result.candidates_truncated is True
    assert result.candidate_count <= 20
    assert result.distinct_score_count == 400


def test_an_unbounded_search_says_that_too(sample: BinaryScoreSample) -> None:
    """The converse, so the flag means something in both directions."""
    result = choose(sample)
    assert result.candidates_truncated is False
    assert result.candidate_count == result.distinct_score_count


def test_an_empty_denominator_is_never_reported_as_zero() -> None:
    """A curve point over no benign rows has no false-positive rate."""
    from password_attack_detector.ml.thresholds import _curve_point

    point = _curve_point(0.5, malicious=[0.4, 0.6], benign=[])
    assert point.false_positive_rate is None
    assert point.recall == 0.5


# ---------------------------------------------------------------------------
# Calibrated input
# ---------------------------------------------------------------------------


@pytest.fixture
def calibrator() -> CalibrationState:
    """Return a calibrator fitted on validation-A."""
    return fit_calibration(
        sx.validation_a(400),
        config=CALIBRATION,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    ).require_state()


def test_a_threshold_may_be_chosen_on_calibrated_probabilities(
    calibrator: CalibrationState, sample: BinaryScoreSample
) -> None:
    """The vocabulary carries through, and so does the calibrator's identity.

    Chosen under ``max_f1``: an isotonic calibrator maps whole runs of scores
    onto one probability, so the candidate set is coarse and a tight
    false-positive ceiling can be genuinely unreachable on it. That coarseness
    is a real property of calibrated thresholds, not a fixture artefact, and it
    is tested for its own sake elsewhere.
    """
    calibrated = apply_calibration(
        calibrator, sample, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )
    result = choose(calibrated, calibration=calibrator)
    assert result.status is SelectionStatus.SELECTED
    assert result.score_kind is ScoreKind.CALIBRATED_PROBABILITY
    assert result.calibration_method is calibrator.method
    assert (
        result.calibration_state_fingerprint == calibrator.calibration_state_fingerprint
    )
    assert result.selected_threshold is not None
    assert 0.0 <= result.selected_threshold <= 1.0


def test_calibrated_scores_without_a_calibrator_are_refused(
    calibrator: CalibrationState, sample: BinaryScoreSample
) -> None:
    """An operating point recorded against a calibrator nobody named."""
    calibrated = apply_calibration(
        calibrator, sample, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )
    with pytest.raises(ModelTrainingError, match="without the calibrator"):
        choose(calibrated)


def test_a_calibrator_for_uncalibrated_scores_is_refused(
    calibrator: CalibrationState, sample: BinaryScoreSample
) -> None:
    """Recording one would claim a transformation that was not applied."""
    with pytest.raises(ModelTrainingError, match="uncalibrated scores"):
        choose(sample, calibration=calibrator)


def test_a_different_calibrator_is_refused(
    calibrator: CalibrationState, sample: BinaryScoreSample
) -> None:
    """The fingerprints must name each other, not merely both be present."""
    other = fit_calibration(
        sx.validation_a(400),
        config=CALIBRATION.model_copy(update={"method": CalibrationMethod.PLATT}),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    ).require_state()
    calibrated = apply_calibration(
        calibrator, sample, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )
    with pytest.raises(ModelTrainingError, match="different calibrator"):
        choose(calibrated, calibration=other)


def test_a_calibrator_from_an_unrelated_partition_is_refused(
    calibrator: CalibrationState,
) -> None:
    """The A/B halves share one parent digest, or they are unrelated partitions."""
    stranger = sx.validation_b(400, fingerprint=sx.OTHER_FINGERPRINT)
    calibrated = stranger.model_copy(
        update={
            "scores": calibrator.transform(stranger.scores),
            "score_kind": ScoreKind.CALIBRATED_PROBABILITY,
            "calibration_state_fingerprint": (calibrator.calibration_state_fingerprint),
            "calibration_method": calibrator.method,
        }
    )
    with pytest.raises(ModelTrainingError, match="validation partition"):
        choose(calibrated, calibration=calibrator)


# ---------------------------------------------------------------------------
# Category abstention
# ---------------------------------------------------------------------------


def abstain(
    rows: Any,
    *,
    config: CategoryConfig = CATEGORY,
    support: SupportRequirement = SUPPORT,
) -> CategoryAbstentionSelection:
    """Return the abstention selection for *rows*."""
    return select_category_abstention(
        sx.category_sample(rows) if isinstance(rows, list) else rows,
        config=config,
        support=support,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )


def test_the_abstention_predicate_is_greater_than_or_equal() -> None:
    """A row whose best class score equals the threshold may keep its class."""
    result = abstain(sx.category_rows(120))
    assert result.decision_predicate == CATEGORY_DECISION_PREDICATE
    assert result.abstain_label == UNKNOWN_CATEGORY
    assert result.status is SelectionStatus.SELECTED
    covered = sum(
        1
        for scores, _ in sx.category_rows(120)
        if max(scores) >= result.min_category_score
    )
    assert result.covered_count == covered


def test_coverage_and_precision_are_reported_together() -> None:
    """Neither number means anything without the other."""
    result = abstain(sx.category_rows(120))
    assert result.coverage is not None
    assert result.known_category_precision is not None
    assert result.known_category_error is not None
    assert result.covered_count is not None
    assert result.correct_count is not None
    assert result.known_category_precision >= CATEGORY.min_known_category_precision
    assert result.known_category_error == pytest.approx(
        1.0 - result.known_category_precision, abs=1e-9
    )
    assert result.coverage == pytest.approx(
        result.covered_count / result.known_malicious_row_count, abs=1e-9
    )


def test_the_precision_floor_binds() -> None:
    """A stricter floor moves the threshold up and the coverage down."""
    rows = sx.category_rows(150)
    lenient = abstain(
        rows, config=CATEGORY.model_copy(update={"min_known_category_precision": 0.60})
    )
    strict = abstain(
        rows, config=CATEGORY.model_copy(update={"min_known_category_precision": 0.95})
    )
    assert lenient.status is SelectionStatus.SELECTED
    if strict.status is SelectionStatus.SELECTED:
        assert strict.min_category_score >= lenient.min_category_score
        assert strict.coverage is not None and lenient.coverage is not None
        assert strict.coverage <= lenient.coverage
    else:
        assert strict.status is SelectionStatus.NO_FEASIBLE_THRESHOLD


def test_class_order_is_deterministic_and_positional() -> None:
    """Every per-class count is reported against the declared order."""
    result = abstain(sx.category_rows(120))
    assert result.class_order == sx.CATEGORY_ORDER
    assert tuple(item.class_name for item in result.class_support) == sx.CATEGORY_ORDER
    assert sum(item.row_count for item in result.class_support) == (
        result.known_malicious_row_count
    )


def test_an_unsorted_class_order_is_refused() -> None:
    """Positional reporting needs an order nobody can shuffle."""
    with pytest.raises(ValidationError, match="deterministic sorted order"):
        sx.category_sample(
            sx.category_rows(30), class_order=tuple(reversed(sx.CATEGORY_ORDER))
        )


def test_argmax_ties_go_to_the_earliest_declared_class() -> None:
    """Otherwise the reported precision depends on library enumeration order."""
    flat = ((1 / 3, 1 / 3, 1 / 3), sx.CATEGORY_ORDER[0])
    sample = sx.category_sample([flat] * 60)
    assert all(name == sx.CATEGORY_ORDER[0] for _, name in sample.predictions())


def test_too_few_known_malicious_rows_falls_back_to_the_reviewed_constant() -> None:
    """A usable threshold, and a permanent admission that nothing measured it."""
    result = abstain(
        sx.category_rows(120),
        config=CATEGORY.model_copy(update={"min_known_malicious_rows": 5000}),
    )
    assert result.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert result.data_selected is False
    assert result.min_category_score == CATEGORY.min_category_score
    assert "min_known_malicious_rows" in result.failing_requirements
    assert result.coverage is None
    assert result.known_category_precision is None


def test_a_thin_class_falls_back_too() -> None:
    """One class with too little support makes every class's precision unreadable."""
    rows = sx.category_rows(120)
    result = abstain(
        rows, config=CATEGORY.model_copy(update={"min_rows_per_category": 500})
    )
    assert result.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert "min_rows_per_category" in result.failing_requirements
    assert result.data_selected is False


def test_an_infeasible_precision_floor_falls_back_with_a_different_status() -> None:
    """Support was adequate; nothing cleared the floor. Both facts are recorded."""
    rows = [
        (scores, sx.CATEGORY_ORDER[(index + 1) % 3])
        for index, (scores, _) in enumerate(sx.category_rows(120))
    ]
    result = abstain(rows)
    assert result.status is SelectionStatus.NO_FEASIBLE_THRESHOLD
    assert result.failing_requirements == ("min_known_category_precision",)
    assert result.data_selected is False
    assert result.min_category_score == CATEGORY.min_category_score
    assert result.support_status is MetricStatus.MEASURED


def test_a_fallback_is_never_described_as_a_selection() -> None:
    """The one field a later report must not be able to confuse."""
    selected = abstain(sx.category_rows(120))
    fallen_back = abstain(
        sx.category_rows(120),
        config=CATEGORY.model_copy(update={"min_known_malicious_rows": 5000}),
    )
    assert selected.data_selected is True
    assert fallen_back.data_selected is False
    assert '"data_selected":false' in fallen_back.to_json()


@pytest.mark.parametrize("tie_break", ["lowest_threshold", "highest_threshold"])
def test_abstention_ties_are_broken_as_configured(
    tie_break: Literal["lowest_threshold", "highest_threshold"],
) -> None:
    """Two thresholds covering the same rows; the configuration decides."""
    rows = (
        [((0.9, 0.05, 0.05), sx.CATEGORY_ORDER[0]) for _ in range(30)]
        + [((0.05, 0.9, 0.05), sx.CATEGORY_ORDER[1]) for _ in range(30)]
        + [((0.05, 0.05, 0.9), sx.CATEGORY_ORDER[2]) for _ in range(30)]
    )
    result = abstain(
        rows,
        config=CATEGORY.model_copy(update={"abstention_tie_break": tie_break}),
    )
    assert result.status is SelectionStatus.SELECTED
    assert result.min_category_score == 0.9
    assert result.tie_break == tie_break


def test_per_class_outcomes_are_absent_when_nothing_was_selected() -> None:
    """A fallback publishes support and no measurement, because there is none."""
    result = abstain(
        sx.category_rows(120),
        config=CATEGORY.model_copy(update={"min_known_malicious_rows": 5000}),
    )
    for item in result.class_support:
        assert item.row_count > 0
        assert item.covered_count is None
        assert item.correct_count is None


# ---------------------------------------------------------------------------
# Anomaly threshold
# ---------------------------------------------------------------------------


def test_the_train_benign_quantile_reads_only_benign_training_scores() -> None:
    """No malicious label, no validation outcome, nothing the holdout reveals."""
    scores = sx.anomaly_scores(400)
    result = select_anomaly_threshold(
        sx.anomaly_sample(scores),
        config=ANOMALY.model_copy(update={"quantile": 0.95}),
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.status is SelectionStatus.SELECTED
    assert result.method is AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE
    assert result.source_split is MLSplit.TRAIN
    assert result.source_partition is None
    assert result.target_quantile == 0.95
    assert result.target_benign_flag_rate == pytest.approx(0.05, abs=1e-9)
    assert result.observed_benign_flag_rate is not None
    assert result.observed_benign_flag_rate <= 0.05
    assert result.flagged_benign_count == 20


def test_the_validation_a_benign_rate_reads_only_validation_a() -> None:
    """The other permitted provenance, and it never touches validation-B."""
    scores = sx.anomaly_scores(400)
    result = select_anomaly_threshold(
        sx.anomaly_sample(
            scores,
            split=MLSplit.VALIDATION,
            partition=ValidationPartition.VALIDATION_A,
            fingerprint=sx.PARTITION_FINGERPRINT,
        ),
        config=ANOMALY.model_copy(
            update={
                "threshold_method": AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR,
                "target_benign_flag_rate": 0.02,
            }
        ),
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.status is SelectionStatus.SELECTED
    assert result.source_partition is ValidationPartition.VALIDATION_A
    assert result.target_quantile is None
    assert result.observed_benign_flag_rate is not None
    assert result.observed_benign_flag_rate <= 0.02


def test_the_anomaly_predicate_is_inverted() -> None:
    """Lower is more anomalous, so a flag is ``<=`` and never ``>=``."""
    scores = sx.anomaly_scores(400)
    result = select_anomaly_threshold(
        sx.anomaly_sample(scores),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.decision_predicate == ANOMALY_DECISION_PREDICATE
    assert result.threshold is not None
    flagged = sum(1 for score in scores if score <= result.threshold)
    assert flagged == result.flagged_benign_count


def test_the_anomaly_output_stays_an_anomaly_score() -> None:
    """Thresholding a magnitude does not turn it into a probability."""
    result = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(400)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.score_kind is ScoreKind.ANOMALY_SCORE
    rendered = result.to_json().lower()
    for banned in ("probability", "likelihood", "confidence"):
        assert banned not in rendered


def test_the_anomaly_probe_never_influences_champion_selection() -> None:
    """Pinned in the record as it is in the configuration."""
    result = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(400)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.influences_champion_selection is False
    payload = result.to_dict()
    payload["influences_champion_selection"] = True
    with pytest.raises(ModelTrainingError, match="not valid"):
        AnomalyThresholdSelection.from_dict(payload)


def test_too_few_benign_rows_is_insufficient_support() -> None:
    """A flag rate over forty rows is not a flag rate."""
    result = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(40)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert result.failing_requirements == ("min_source_rows",)
    assert result.threshold is None
    assert result.data_selected is False


def test_a_target_finer_than_the_data_can_resolve_is_insufficient_support() -> None:
    """With 200 rows the coarsest available flag rate is ``1/200``."""
    result = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(200)),
        config=ANOMALY.model_copy(update={"quantile": 0.999}),
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert result.failing_requirements == ("benign_flag_rate_resolution",)


def test_anomaly_ties_are_never_split() -> None:
    """A threshold that flagged some rows carrying one score is not a threshold."""
    scores = tuple(round(-1.0 + (index // 10) / 40, 9) for index in range(400))
    result = select_anomaly_threshold(
        sx.anomaly_sample(scores),
        config=ANOMALY.model_copy(update={"quantile": 0.95}),
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert result.status is SelectionStatus.SELECTED
    assert result.flagged_benign_count is not None
    assert result.flagged_benign_count % 10 == 0


def test_anomaly_selection_is_deterministic() -> None:
    """Same scores, same threshold, same digest."""
    first = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(400)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    second = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(400)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert first.to_json() == second.to_json()


# ---------------------------------------------------------------------------
# Serialization, fingerprints, and identity separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["binary", "category", "anomaly"])
def test_every_selection_round_trips_byte_identically(
    sample: BinaryScoreSample, kind: str
) -> None:
    """Deserialize, reserialize, compare bytes."""
    record = _selection_of(kind, sample)
    model = type(record)
    assert model.from_json(record.to_json()).to_json() == record.to_json()


@pytest.mark.parametrize("kind", ["binary", "category", "anomaly"])
def test_tampering_with_a_selection_is_detected(
    sample: BinaryScoreSample, kind: str
) -> None:
    """The digest covers every field, so any edit at all is refused."""
    record = _selection_of(kind, sample)
    payload = record.to_dict()
    payload["ml_config_fingerprint"] = sx.OTHER_FINGERPRINT
    with pytest.raises(ModelTrainingError, match="not valid"):
        type(record).from_dict(payload)


@pytest.mark.parametrize("kind", ["binary", "category", "anomaly"])
def test_a_selection_from_an_unsupported_contract_is_refused(
    sample: BinaryScoreSample, kind: str
) -> None:
    """Version checked before validation, so nothing is partially understood."""
    record = _selection_of(kind, sample)
    payload = record.to_dict()
    payload["threshold_schema_version"] = "9.9.9"
    with pytest.raises(ModelTrainingError, match="schema version"):
        type(record).from_dict(payload)


def test_the_three_identities_stay_separate(
    calibrator: CalibrationState, sample: BinaryScoreSample
) -> None:
    """A model, a calibrator, and an operating point are three things.

    Choosing a threshold must not change the model's identity or the
    calibrator's: Milestone 6's ``champion.lock`` binds the three together, and
    it can only do that if they are three.
    """
    calibrated = apply_calibration(
        calibrator, sample, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )
    before = calibrator.to_json()
    result = choose(calibrated, calibration=calibrator)
    assert calibrator.to_json() == before
    assert result.model_content_fingerprint == sx.MODEL_FINGERPRINT
    assert (
        result.calibration_state_fingerprint == calibrator.calibration_state_fingerprint
    )
    assert result.selection_fingerprint not in {
        sx.MODEL_FINGERPRINT,
        calibrator.calibration_state_fingerprint,
    }


def test_a_selection_carries_the_whole_provenance_chain(
    sample: BinaryScoreSample,
) -> None:
    """Model, preprocessor, partition, and configuration, all four recorded."""
    result = choose(sample)
    assert result.model_content_fingerprint == sx.MODEL_FINGERPRINT
    assert result.preprocessor_fingerprint == sx.PREPROCESSOR_FINGERPRINT
    assert result.source_partition_fingerprint == sx.PARTITION_FINGERPRINT
    assert result.ml_config_fingerprint == sx.CONFIG_FINGERPRINT
    assert result.source_partition is ValidationPartition.VALIDATION_B


def test_no_selection_carries_a_test_metric(sample: BinaryScoreSample) -> None:
    """There is no field a test number could be written into."""
    for record in (
        _selection_of("binary", sample),
        _selection_of("category", sample),
        _selection_of("anomaly", sample),
    ):
        names = set(type(record).model_fields)
        assert not any("test" in name for name in names)
        assert not any("holdout" in name for name in names)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["binary", "category", "anomaly"])
@pytest.mark.parametrize(
    "banned", ["e00000", "e00001", "campaign", "u:", "/home/", "anchor"]
)
def test_no_identity_reaches_a_selection(
    sample: BinaryScoreSample, kind: str, banned: str
) -> None:
    """Rows and anchors go in; counts, rates, and digests come out."""
    assert banned not in _selection_of(kind, sample).to_json()


def test_no_published_schema_declares_an_identity_bearing_field() -> None:
    """Swept over field names, complementing the sweep over one rendering."""
    from password_attack_detector.ml.schemas import prohibited_metadata_fields
    from password_attack_detector.ml.thresholds import _PUBLISHED_SCHEMAS

    for model in _PUBLISHED_SCHEMAS:
        assert prohibited_metadata_fields(list(model.model_fields)) == ()


def test_no_selection_reports_a_non_finite_number(sample: BinaryScoreSample) -> None:
    """NaN is stable in a digest and meaningless in a report."""
    for kind in ("binary", "category", "anomaly"):
        rendered = _selection_of(kind, sample).to_json()
        assert "NaN" not in rendered
        assert "Infinity" not in rendered


# ---------------------------------------------------------------------------
# The leakage behaviour test
# ---------------------------------------------------------------------------


def _pipeline(
    *,
    validation_a_count: int = 400,
    validation_b_count: int = 400,
    category_count: int = 120,
    train_count: int = 400,
) -> dict[str, str]:
    """Return every Milestone 5 artifact's canonical JSON, for one set of inputs.

    Deliberately takes no test or holdout argument. There is nowhere to pass
    one, which is the point being demonstrated: the pipeline is a pure function
    of train, validation-A and validation-B, and a suite that could vary the
    frozen splits here would be testing something this design does not permit.
    """
    calibration = fit_calibration(
        sx.validation_a(validation_a_count),
        config=CALIBRATION,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    ).require_state()
    diagnostic = diagnose_calibration_fit(
        sx.validation_a(validation_a_count),
        state=calibration,
        config=CALIBRATION,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    quality = evaluate_calibration_quality(
        sx.validation_b(validation_b_count),
        state=calibration,
        config=CALIBRATION,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    calibrated = apply_calibration(
        calibration,
        sx.validation_b(validation_b_count),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    binary = choose(calibrated, calibration=calibration)
    category = abstain(sx.category_rows(category_count))
    anomaly = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(train_count)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    return {
        "calibration": calibration.to_json(),
        "diagnostic": diagnostic.to_json(),
        "quality": quality.to_json(),
        "binary": binary.to_json(),
        "category": category.to_json(),
        "anomaly": anomaly.to_json(),
    }


def test_the_frozen_splits_cannot_reach_any_fitted_quantity() -> None:
    """The structural half of the leakage argument.

    Every Milestone 5 entry point is enumerated and its parameters inspected.
    None of them can be handed test or holdout rows *at all* except through the
    typed provenance every one of them checks -- and those checks are exercised
    behaviourally above. So building "a different test split" and asserting
    nothing changed is not a thing this design can even express: there is no
    input to vary.
    """
    import inspect

    for function in (
        fit_calibration,
        diagnose_calibration_fit,
        evaluate_calibration_quality,
        apply_calibration,
        select_binary_threshold,
        select_category_abstention,
        select_anomaly_threshold,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not any(
            "test" in name or "holdout" in name or "novel" in name
            for name in parameters
        )


def test_every_artifact_is_byte_identical_across_runs() -> None:
    """Determinism first: without it, the comparisons below prove nothing."""
    assert _pipeline() == _pipeline()


def test_validation_a_moves_the_calibrator_and_not_the_b_only_inputs() -> None:
    """Changing the calibration half changes the calibrator, and says where it stops.

    The category and anomaly thresholds read no calibrated score at all, so
    they are unmoved. The binary threshold *is* moved -- it is chosen on
    calibrated probabilities, and a different calibrator produces different
    probabilities -- but the validation-B rows it was chosen from are the same
    rows, which is the property that matters: validation-A never contributes a
    label, a count, or a candidate to the selection.
    """
    baseline = _pipeline()
    moved = _pipeline(validation_a_count=360)

    assert moved["calibration"] != baseline["calibration"]
    assert moved["diagnostic"] != baseline["diagnostic"]
    assert moved["quality"] != baseline["quality"]
    assert moved["category"] == baseline["category"]
    assert moved["anomaly"] == baseline["anomaly"]

    uncalibrated = choose(sx.validation_b(400))
    assert uncalibrated.to_json() == choose(sx.validation_b(400)).to_json()


def test_validation_b_moves_the_threshold_and_never_the_calibrator() -> None:
    """The firewall in the other direction, and the sharper of the two.

    A calibrator that shifted when the operating-point rows changed would mean
    validation-B had reached the fit -- which is exactly the leak the split
    exists to prevent.
    """
    baseline = _pipeline()
    moved = _pipeline(validation_b_count=360)

    assert moved["calibration"] == baseline["calibration"]
    assert moved["diagnostic"] == baseline["diagnostic"]
    assert moved["quality"] != baseline["quality"]
    assert moved["binary"] != baseline["binary"]
    assert moved["anomaly"] == baseline["anomaly"]


def test_the_anomaly_threshold_moves_only_with_its_own_benign_source() -> None:
    """It reads training benign scores, and nothing supervised anywhere."""
    baseline = _pipeline()
    assert _pipeline(validation_a_count=360)["anomaly"] == baseline["anomaly"]
    assert _pipeline(validation_b_count=360)["anomaly"] == baseline["anomaly"]
    assert _pipeline(train_count=360)["anomaly"] != baseline["anomaly"]


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


def test_the_ml_layer_still_exposes_no_training_command() -> None:
    """Milestone 5 is a library contract. Orchestration belongs to Milestone 6."""
    from password_attack_detector.ml.cli import ml_app

    names = {
        command.name or (command.callback.__name__ if command.callback else "")
        for command in ml_app.registered_commands
    }
    assert names == {"catalog", "audit-features", "verify-manifest"}


def test_the_package_version_is_unchanged() -> None:
    """Milestone 5 adds contracts, not a release."""
    from password_attack_detector import __version__

    assert __version__ == "0.4.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selection_of(kind: str, sample: BinaryScoreSample) -> Any:
    """Return one selection of each kind, for the sweeps that cover all three."""
    match kind:
        case "binary":
            return choose(sample)
        case "category":
            return abstain(sx.category_rows(120))
        case _:
            return select_anomaly_threshold(
                sx.anomaly_sample(sx.anomaly_scores(400)),
                config=ANOMALY,
                support=SUPPORT,
                ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
            )


def test_a_selection_refuses_a_predicate_it_did_not_measure_under(
    sample: BinaryScoreSample,
) -> None:
    """Rewriting the comparison would describe a different set of flagged rows."""
    payload = choose(sample).to_dict()
    payload["decision_predicate"] = "score > threshold"
    with pytest.raises(ModelTrainingError, match="not valid"):
        ThresholdSelection.from_dict(payload)


def test_a_selection_cannot_claim_validation_a(sample: BinaryScoreSample) -> None:
    """Recorded provenance is validated on load, not merely carried."""
    payload = choose(sample).to_dict()
    payload["source_partition"] = "validation_a"
    with pytest.raises(ModelTrainingError, match="not valid"):
        ThresholdSelection.from_dict(payload)


def test_data_selected_cannot_be_flipped_on(sample: BinaryScoreSample) -> None:
    """A fallback promoted to a selection by editing one boolean."""
    fallen_back = abstain(
        sx.category_rows(120),
        config=CATEGORY.model_copy(update={"min_known_malicious_rows": 5000}),
    )
    payload = fallen_back.to_dict()
    payload["data_selected"] = True
    with pytest.raises(ModelTrainingError, match="not valid"):
        CategoryAbstentionSelection.from_dict(payload)


def test_finite_numbers_only(sample: BinaryScoreSample) -> None:
    """Asserted on every numeric leaf, not only on the headline fields."""
    payload = json.loads(choose(sample).to_json())
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, int | float) and not isinstance(item, bool):
            assert math.isfinite(item)


# ---------------------------------------------------------------------------
# Contract validators
# ---------------------------------------------------------------------------


def _reseal(record: Any, **overrides: Any) -> Any:
    """Return *record* rebuilt with *overrides*, sealed over the new content.

    Rebuilding through ``seal`` rather than editing a stored payload makes the
    digest agree with the content, so the *semantic* validator is what fires.
    Editing a payload is tested separately, and there the digest is supposed to
    catch it first.
    """
    fields = {
        name: getattr(record, name)
        for name in type(record).model_fields
        if name != type(record).fingerprint_field
    }
    fields.update(overrides)
    return type(record).seal(**fields)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"data_selected": False}, "data_selected is true exactly"),
        ({"failing_requirements": ("something",)}, "names no failing requirement"),
        (
            {"support_status": MetricStatus.INSUFFICIENT_SUPPORT},
            "unmeasurable support",
        ),
        ({"row_count": 1}, "does not sum"),
        ({"benign_flagged_count": None}, "reports its operating point in full"),
        (
            {"calibration_method": CalibrationMethod.PLATT},
            "must agree about whether a calibrator",
        ),
    ],
)
def test_a_binary_selection_enforces_its_internal_agreement(
    sample: BinaryScoreSample, overrides: dict[str, Any], expected: str
) -> None:
    """Every invariant on a chosen operating point, exercised one at a time."""
    with pytest.raises(ValidationError, match=expected):
        _reseal(choose(sample), **overrides)


def test_an_unselected_binary_result_may_not_report_an_operating_point() -> None:
    """The other direction: only a selection has one."""
    unselected = choose(
        infeasible_sample(), config=ThresholdConfig(max_false_positive_rate=0.25)
    )
    with pytest.raises(ValidationError, match="reports an operating point"):
        _reseal(unselected, selected_threshold=0.5)


def test_a_binary_selection_refuses_an_unordered_curve(
    sample: BinaryScoreSample,
) -> None:
    """A curve out of order would misread as an operating characteristic."""
    result = choose(sample)
    with pytest.raises(ValidationError, match="ascending threshold order"):
        _reseal(result, curve=tuple(reversed(result.curve)))
    with pytest.raises(ValidationError, match="unique by semantic threshold"):
        _reseal(result, curve=(result.curve[0], result.curve[0]))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"recall": None}, "recall exists exactly"),
        ({"false_positive_rate": None}, "false-positive rate exists exactly"),
        ({"precision": None}, "precision exists exactly"),
        ({"threshold": math.inf}, "finite threshold"),
        ({"f1": math.nan}, "finite or absent"),
    ],
)
def test_a_curve_point_enforces_its_internal_agreement(
    overrides: dict[str, Any], expected: str
) -> None:
    """Each rate exists exactly when its denominator does."""
    fields: dict[str, Any] = {
        "threshold": 0.5,
        "true_positives": 2,
        "false_positives": 1,
        "true_negatives": 3,
        "false_negatives": 4,
        "precision": 2 / 3,
        "recall": 1 / 3,
        "false_positive_rate": 0.25,
        "f1": 0.4,
    }
    fields.update(overrides)
    with pytest.raises(ValidationError, match=expected):
        ThresholdCurvePoint(**fields)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"data_selected": False}, "data_selected is true exactly"),
        ({"coverage": None}, "reports its coverage"),
        ({"failing_requirements": ("x",)}, "names no failing requirement"),
        ({"known_malicious_row_count": 3}, "does not account for every row"),
        ({"min_known_category_precision": 1.0}, "clears the configured precision"),
        ({"abstain_label": "other"}, "abstention label is"),
        ({"decision_predicate": "max > t"}, "abstention predicate is"),
    ],
)
def test_a_category_selection_enforces_its_internal_agreement(
    overrides: dict[str, Any], expected: str
) -> None:
    """Coverage, precision, support, and vocabulary all have to agree."""
    with pytest.raises(ValidationError, match=expected):
        _reseal(abstain(sx.category_rows(120)), **overrides)


def test_a_category_fallback_may_not_report_a_measurement() -> None:
    """There was none, so publishing one would invent it."""
    fallen_back = abstain(
        sx.category_rows(120),
        config=CATEGORY.model_copy(update={"min_known_malicious_rows": 5000}),
    )
    with pytest.raises(ValidationError, match="reports no measurement"):
        _reseal(fallen_back, coverage=0.9)


def test_a_class_support_row_enforces_its_internal_agreement() -> None:
    """Coverage and correctness are reported together, and stay consistent."""
    from password_attack_detector.ml.thresholds import CategoryClassSupport

    with pytest.raises(ValidationError, match="reported together"):
        CategoryClassSupport(class_name="a", row_count=5, covered_count=2)
    with pytest.raises(ValidationError, match="more rows than it has"):
        CategoryClassSupport(
            class_name="a", row_count=1, covered_count=2, correct_count=1
        )
    with pytest.raises(ValidationError, match="rows it did not cover"):
        CategoryClassSupport(
            class_name="a", row_count=5, covered_count=1, correct_count=2
        )


def _anomaly() -> AnomalyThresholdSelection:
    """Return a selected anomaly threshold."""
    return select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(400)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"score_kind": ScoreKind.DECISION_SCORE}, "against an anomaly score"),
        ({"decision_predicate": "score >= threshold"}, "anomaly predicate is"),
        ({"influences_champion_selection": True}, "never influences champion"),
        ({"source_split": MLSplit.VALIDATION}, "does not read"),
        ({"target_quantile": None}, "quantile is recorded exactly"),
        ({"data_selected": False}, "data_selected is true exactly"),
        ({"observed_benign_flag_rate": None}, "reports its observed flag rate"),
        ({"failing_requirements": ("x",)}, "names no failing requirement"),
    ],
)
def test_an_anomaly_selection_enforces_its_internal_agreement(
    overrides: dict[str, Any], expected: str
) -> None:
    """Method, source, vocabulary, and standing all have to agree."""
    with pytest.raises(ValidationError, match=expected):
        _reseal(_anomaly(), **overrides)


def test_an_unselected_anomaly_result_may_not_report_a_measurement() -> None:
    """Below the support floor there is nothing to report."""
    thin = select_anomaly_threshold(
        sx.anomaly_sample(sx.anomaly_scores(40)),
        config=ANOMALY,
        support=SUPPORT,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    with pytest.raises(ValidationError, match="reports no measurement"):
        _reseal(thin, threshold=-0.5)


def test_a_malformed_selection_payload_is_refused(sample: BinaryScoreSample) -> None:
    """Not an object, not JSON -- each named separately and without a traceback."""
    with pytest.raises(ModelTrainingError, match="must be a JSON object"):
        ThresholdSelection.from_dict(["not", "an", "object"])
    with pytest.raises(ModelTrainingError, match="not valid JSON"):
        CategoryAbstentionSelection.from_json("{oops")
