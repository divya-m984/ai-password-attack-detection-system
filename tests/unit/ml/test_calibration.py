"""The calibration contract: validation-A only, and a probability only once earned.

Three properties carry most of the weight here.

**The firewall is behavioural.** Every refusal is demonstrated by handing a
selector rows whose typed provenance names the wrong source and watching it
raise -- not by grepping the module for the word "test". A structural check
would pass on a module that named the split correctly and read it anyway.

**Parity is against an independent implementation.** The isotonic fit is
project-owned, so comparing it with a live ``IsotonicRegression`` compares two
implementations rather than one implementation with itself. The comparison is
on the documented public attributes, ``X_thresholds_`` and ``y_thresholds_``.

**Vocabulary is checked in both directions.** A calibrated probability without a
calibrator is refused, and a calibrator that fitted nothing cannot produce one.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    POSITIVE_CLASS,
    BinaryScoreSample,
    CalibrationOutcome,
    CalibrationReport,
    CalibrationState,
    IsotonicParameters,
    PlattParameters,
    ReliabilityBin,
    ScoreSampleSource,
    _fit_isotonic,
    _fit_platt,
    apply_calibration,
    diagnose_calibration_fit,
    evaluate_calibration_quality,
    fit_calibration,
    require_chain,
    require_out_of_sample_evidence,
    sigmoid,
)
from password_attack_detector.ml.config import CalibrationConfig
from password_attack_detector.ml.enums import (
    PROBABILITY_SCORE_KINDS,
    CalibrationEvaluationKind,
    CalibrationMethod,
    CalibrationStatus,
    MetricStatus,
    MLSplit,
    ScoreKind,
    ValidationPartition,
)
from tests.ml import selection as sx

#: A configuration sized for the fixtures in this module: small enough that a
#: four-hundred-row sample clears every floor, strict enough that the floors
#: still bite when a test deliberately shrinks the sample.
CONFIG = CalibrationConfig(
    method=CalibrationMethod.ISOTONIC,
    min_calibration_rows=100,
    reliability_bin_count=10,
    min_reliability_bin_rows=20,
    min_isotonic_distinct_scores=10,
)

PLATT_CONFIG = CONFIG.model_copy(update={"method": CalibrationMethod.PLATT})


def fit(
    sample: BinaryScoreSample, config: CalibrationConfig = CONFIG
) -> CalibrationOutcome:
    """Return the calibration outcome for *sample* under *config*."""
    return fit_calibration(
        sample, config=config, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )


@pytest.fixture
def sample() -> BinaryScoreSample:
    """Return a validation-A sample that clears every configured floor."""
    return sx.validation_a(400)


@pytest.fixture
def state(sample: BinaryScoreSample) -> CalibrationState:
    """Return a fitted isotonic calibrator."""
    return fit(sample).require_state()


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
        (MLSplit.VALIDATION, ValidationPartition.VALIDATION_B),
    ],
)
@pytest.mark.parametrize(
    "method", [CalibrationMethod.PLATT, CalibrationMethod.ISOTONIC]
)
def test_only_validation_a_may_fit_a_calibrator(
    split: MLSplit, partition: ValidationPartition | None, method: CalibrationMethod
) -> None:
    """Both methods refuse every source but validation-A.

    Parameterised over the method as well as the source on purpose: a firewall
    implemented once and applied twice is one edit away from being applied
    once.
    """
    scores = sx.graded_scores(400)
    bad = sx.binary_sample(
        scores, sx.graded_labels(scores), split=split, partition=partition
    )
    config = CONFIG.model_copy(update={"method": method})
    with pytest.raises(ModelTrainingError, match="validation_a"):
        fit(bad, config)


def test_the_refusal_names_no_bypass() -> None:
    """There is no flag, keyword, or environment variable that widens the source.

    Asserted on the message a caller actually sees, because the message is
    where somebody looks for the escape hatch.
    """
    scores = sx.graded_scores(400)
    bad = sx.binary_sample(scores, sx.graded_labels(scores), split=MLSplit.TEST)
    with pytest.raises(ModelTrainingError) as failure:
        fit(bad)
    assert "no option that widens this" in str(failure.value)


def test_calibration_evaluation_also_refuses_test(state: CalibrationState) -> None:
    """Measuring a calibrator on test would be a test evaluation nobody recorded."""
    scores = sx.graded_scores(400)
    bad = sx.binary_sample(scores, sx.graded_labels(scores), split=MLSplit.TEST)
    with pytest.raises(ModelTrainingError, match="validation_a"):
        diagnose_calibration_fit(
            bad,
            state=state,
            config=CONFIG,
            ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        )


@pytest.mark.parametrize(
    "split", [MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT, MLSplit.TRAIN]
)
def test_applying_a_calibrator_refuses_non_validation_rows(
    state: CalibrationState, split: MLSplit
) -> None:
    """Inference on frozen splits belongs to a later, recorded evaluation."""
    scores = sx.graded_scores(50)
    bad = sx.binary_sample(scores, sx.graded_labels(scores), split=split)
    with pytest.raises(ModelTrainingError, match="validation rows only"):
        apply_calibration(state, bad, ml_config_fingerprint=sx.CONFIG_FINGERPRINT)


def test_a_source_may_still_describe_test_rows_honestly() -> None:
    """The firewall lives in the consumers, not in the vocabulary.

    A provenance object naming the test split is constructible, and must be:
    the alternative is a type system in which nothing can truthfully say where
    test rows came from, which moves the problem rather than solving it.
    """
    honest = ScoreSampleSource(
        split=MLSplit.TEST, partition=None, source_fingerprint=sx.PARTITION_FINGERPRINT
    )
    assert honest.describe == "test"


def test_a_non_validation_source_cannot_name_a_validation_half() -> None:
    """Only the validation split is partitioned, so only it may name a half."""
    with pytest.raises(ValidationError, match="carries no validation partition"):
        ScoreSampleSource(
            split=MLSplit.TEST,
            partition=ValidationPartition.VALIDATION_A,
            source_fingerprint=sx.PARTITION_FINGERPRINT,
        )


def test_a_validation_source_must_name_its_half() -> None:
    """An unlabelled validation source could be either half, and they differ."""
    with pytest.raises(ValidationError, match="must name which half"):
        ScoreSampleSource(
            split=MLSplit.VALIDATION,
            partition=None,
            source_fingerprint=sx.PARTITION_FINGERPRINT,
        )


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def test_a_calibrator_consumes_the_frozen_decision_score(
    sample: BinaryScoreSample,
) -> None:
    """Calibrating a calibrated score would measure the calibrator, not the model."""
    fitted = fit(sample).require_state()
    calibrated = apply_calibration(
        fitted, sample, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )
    with pytest.raises(ModelTrainingError, match="frozen decision score"):
        fit(calibrated)


@pytest.mark.parametrize("kind", [ScoreKind.CLASS_SCORE, ScoreKind.ANOMALY_SCORE])
def test_a_binary_sample_refuses_class_and_anomaly_scores(kind: ScoreKind) -> None:
    """An anomaly score is never calibrated as though it were supervised."""
    scores = sx.graded_scores(20)
    with pytest.raises(ValidationError):
        sx.binary_sample(scores, sx.graded_labels(scores), score_kind=kind)


def test_uncanonical_rows_are_refused_rather_than_sorted() -> None:
    """A stage that quietly re-sorted would hide who lost the order."""
    scores = sx.graded_scores(400)
    labels = sx.graded_labels(scores)
    shuffled = sx.anchors(400)
    reordered = (*shuffled[1:], shuffled[0])
    bad = BinaryScoreSample(
        source=sx.source(ValidationPartition.VALIDATION_A),
        anchors=reordered,
        scores=scores,
        malicious=labels,
        score_kind=ScoreKind.DECISION_SCORE,
        model_id="m",
        model_content_fingerprint=sx.MODEL_FINGERPRINT,
        preprocessor_fingerprint=sx.PREPROCESSOR_FINGERPRINT,
    )
    with pytest.raises(ModelTrainingError, match="canonical order"):
        fit(bad)


def test_a_non_finite_score_is_refused() -> None:
    """A model that emitted a NaN is a fault to fix, not a value to calibrate."""
    with pytest.raises(ValidationError, match="non-finite"):
        sx.binary_sample((0.1, math.nan, 0.3), (False, True, True))


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------


def test_too_few_rows_is_an_outcome_not_a_fit() -> None:
    """Below the configured floor there is a status, and no state at all."""
    scores = sx.graded_scores(40)
    outcome = fit(sx.binary_sample(scores, sx.graded_labels(scores)))
    assert outcome.status is CalibrationStatus.INSUFFICIENT_CALIBRATION_SUPPORT
    assert outcome.state is None
    assert "min_calibration_rows" in outcome.failing_requirements
    with pytest.raises(ModelTrainingError, match="no calibrator was fitted"):
        outcome.require_state()


@pytest.mark.parametrize("label", [True, False])
def test_a_single_class_never_fits_silently(label: bool) -> None:
    """One class carries no information about the other's rate."""
    scores = sx.graded_scores(400)
    outcome = fit(sx.binary_sample(scores, (label,) * 400))
    assert outcome.status is CalibrationStatus.INSUFFICIENT_CALIBRATION_SUPPORT
    assert "both_classes_required" in outcome.failing_requirements
    assert outcome.state is None


def test_isotonic_needs_enough_distinct_scores() -> None:
    """A monotone fit over four values is a lookup table for those four values."""
    scores = tuple(round(0.1 * (index % 4), 9) for index in range(400))
    labels = tuple(index % 3 == 0 for index in range(400))
    outcome = fit(sx.binary_sample(scores, labels))
    assert outcome.status is CalibrationStatus.INSUFFICIENT_CALIBRATION_SUPPORT
    assert "min_isotonic_distinct_scores" in outcome.failing_requirements


def test_platt_tolerates_the_support_isotonic_refuses() -> None:
    """Two parameters need far less resolution than a piecewise fit does."""
    scores = tuple(round(0.1 * (index % 4), 9) for index in range(400))
    labels = tuple(index % 3 == 0 for index in range(400))
    outcome = fit(sx.binary_sample(scores, labels), PLATT_CONFIG)
    assert outcome.status is CalibrationStatus.FITTED


def test_method_none_is_an_ordinary_outcome(sample: BinaryScoreSample) -> None:
    """A model may ship uncalibrated, and that is not a failure."""
    outcome = fit(sample, CONFIG.model_copy(update={"method": CalibrationMethod.NONE}))
    assert outcome.status is CalibrationStatus.NOT_CALIBRATED
    assert outcome.method is CalibrationMethod.NONE
    assert outcome.state is None
    assert outcome.failing_requirements == ()


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------


def test_platt_fits_two_finite_scalars(sample: BinaryScoreSample) -> None:
    """The whole authoritative state is ``a``, ``b``, and how it got there."""
    state = fit(sample, PLATT_CONFIG).require_state()
    assert state.method is CalibrationMethod.PLATT
    assert state.isotonic is None
    assert state.platt is not None
    assert math.isfinite(state.platt.a)
    assert math.isfinite(state.platt.b)
    assert state.platt.a > 0.0  # higher decision score, higher probability


def test_platt_is_deterministic(sample: BinaryScoreSample) -> None:
    """Same rows, same configuration, same two numbers -- and same digest."""
    first = fit(sample, PLATT_CONFIG).require_state()
    second = fit(sample, PLATT_CONFIG).require_state()
    assert first.platt == second.platt
    assert first.calibration_state_fingerprint == second.calibration_state_fingerprint
    assert first.to_json() == second.to_json()


def test_platt_reaches_an_optimum(sample: BinaryScoreSample) -> None:
    """Verified independently: the gradient of the smoothed loss is ~zero.

    Recomputed here from the stored parameters rather than trusting the number
    the solver reported about itself.
    """
    state = fit(sample, PLATT_CONFIG).require_state()
    assert state.platt is not None
    positives = state.positive_count
    negatives = state.negative_count
    high = (positives + 1.0) / (positives + 2.0)
    low = 1.0 / (negatives + 2.0)

    grad_a = 0.0
    grad_b = 0.0
    for score, label in zip(sample.scores, sample.malicious, strict=True):
        residual = sigmoid(state.platt.a * score + state.platt.b) - (
            high if label else low
        )
        grad_a += residual * score
        grad_b += residual
    assert abs(grad_a) < 1e-6
    assert abs(grad_b) < 1e-6


def test_platt_transform_matches_its_own_parameters(sample: BinaryScoreSample) -> None:
    """Project-owned inference reproduces the fit from the stored scalars alone."""
    state = fit(sample, PLATT_CONFIG).require_state()
    assert state.platt is not None
    expected = [
        sigmoid(state.platt.a * score + state.platt.b) for score in sample.scores
    ]
    assert state.transform(sample.scores) == tuple(expected)


def test_platt_is_stable_at_extreme_scores(sample: BinaryScoreSample) -> None:
    """A separated row is where the naive sigmoid returns NaN; this one does not."""
    state = fit(sample, PLATT_CONFIG).require_state()
    extremes = (-1e6, -1e3, 0.0, 1e3, 1e6)
    values = state.transform(extremes)
    assert all(math.isfinite(value) for value in values)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert list(values) == sorted(values)


def test_the_stable_sigmoid_never_overflows() -> None:
    """``exp(710)`` overflows float64; the branch is what stops it mattering."""
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-300)
    assert all(math.isfinite(sigmoid(value)) for value in (-1e308, 0.0, 1e308))


def test_platt_reports_convergence_failure_rather_than_a_number(
    sample: BinaryScoreSample,
) -> None:
    """A budget of one Newton step is not enough, and the outcome says so."""
    outcome = fit(sample, PLATT_CONFIG.model_copy(update={"platt_max_iter": 1}))
    assert outcome.status is CalibrationStatus.CONVERGENCE_FAILED
    assert outcome.failing_requirements == ("platt_convergence",)
    assert outcome.state is None


def test_convergence_failure_is_distinct_from_thin_support(
    sample: BinaryScoreSample,
) -> None:
    """Two different problems, two different statuses, two different fixes."""
    failed = fit(sample, PLATT_CONFIG.model_copy(update={"platt_max_iter": 1}))
    thin = fit(sx.validation_a(40), PLATT_CONFIG)
    assert failed.status is not thin.status


def test_platt_smoothing_keeps_a_separable_fit_finite() -> None:
    """Perfect separation has no unsmoothed maximum-likelihood estimate.

    Platt's smoothed targets are what make the optimum exist for every input,
    which matters because generated traffic is entirely capable of being
    perfectly separable.
    """
    scores = sx.graded_scores(400)
    separable = tuple(score >= 0.5 for score in scores)
    parameters = _fit_platt(
        scores,
        tuple(1.0 if label else 0.0 for label in separable),
        positives=sum(separable),
        negatives=len(separable) - sum(separable),
        max_iter=1000,
        tolerance=1e-8,
    )
    assert parameters is not None
    assert math.isfinite(parameters.a)
    assert math.isfinite(parameters.b)


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------


def test_isotonic_output_is_monotone_and_bounded(state: CalibrationState) -> None:
    """The defining property, asserted on the transform rather than the fit."""
    probe = tuple(round(index / 200, 9) for index in range(201))
    values = state.transform(probe)
    assert list(values) == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_isotonic_clamps_outside_its_fitted_domain(state: CalibrationState) -> None:
    """Extrapolating a step function would invent a relationship nobody fitted."""
    assert state.isotonic is not None
    low, high = state.isotonic.domain
    below = state.transform((low - 10.0, low - 1.0, low))
    above = state.transform((high, high + 1.0, high + 10.0))
    assert len(set(below)) == 1
    assert len(set(above)) == 1
    assert below[0] == state.isotonic.y_thresholds[0]
    assert above[0] == state.isotonic.y_thresholds[-1]


def test_isotonic_averages_duplicate_scores() -> None:
    """Forty rows at one score weigh forty, and their order does not matter."""
    scores = (0.1,) * 4 + (0.9,) * 4
    labels = (False, True, False, False, True, True, True, False)
    parameters = _fit_isotonic(scores, tuple(1.0 if v else 0.0 for v in labels))
    assert parameters is not None
    assert parameters.x_thresholds == (0.1, 0.9)
    assert parameters.y_thresholds == (0.25, 0.75)

    reordered = _fit_isotonic(
        tuple(reversed(scores)),
        tuple(1.0 if v else 0.0 for v in reversed(labels)),
    )
    assert reordered is not None
    assert reordered.y_thresholds == parameters.y_thresholds


def test_isotonic_needs_two_breakpoints() -> None:
    """One breakpoint is a constant, and a constant is not a calibration."""
    assert _fit_isotonic((0.5,) * 10, (1.0, 0.0) * 5) is None


def test_isotonic_parameters_reject_a_non_monotone_fit() -> None:
    """Enforced by the type, so no fitter can produce an invalid state."""
    with pytest.raises(ValidationError, match="non-decreasing"):
        IsotonicParameters(x_thresholds=(0.1, 0.2), y_thresholds=(0.9, 0.1))
    with pytest.raises(ValidationError, match="strictly increasing"):
        IsotonicParameters(x_thresholds=(0.1, 0.1), y_thresholds=(0.1, 0.9))
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        IsotonicParameters(x_thresholds=(0.1, 0.2), y_thresholds=(0.1, 1.9))


# ---------------------------------------------------------------------------
# scikit-learn compatibility
# ---------------------------------------------------------------------------


def test_the_project_isotonic_fit_reproduces_scikit_learns_attributes(
    sample: BinaryScoreSample,
) -> None:
    """Parity on the two documented public attributes of a fitted estimator.

    ``X_thresholds_`` must match **exactly**: they are observed scores, carried
    through unrounded on both sides, so anything but equality would mean the
    two fits disagree about where the breakpoints are. ``y_thresholds_`` are
    fitted values, quantized to the nine decimals every fitted number in this
    project is stored at, so they agree to that precision and no further.

    No private attribute is read. If a future release renames one of these two,
    this test fails and the compatibility bound gets revisited -- which is what
    a bound is for.
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    targets = [1.0 if label else 0.0 for label in sample.malicious]
    mine = _fit_isotonic(sample.scores, tuple(targets))
    assert mine is not None

    reference = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    )
    reference.fit(np.asarray(sample.scores), np.asarray(targets))

    assert np.array_equal(
        np.asarray(mine.x_thresholds), np.asarray(reference.X_thresholds_)
    )
    assert (
        np.max(
            np.abs(np.asarray(mine.y_thresholds) - np.asarray(reference.y_thresholds_))
        )
        <= 5e-10
    )


def test_project_inference_reproduces_scikit_learns_predictions(
    state: CalibrationState, sample: BinaryScoreSample
) -> None:
    """Including outside the fitted domain, where the clipping contract applies."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    targets = [1.0 if label else 0.0 for label in sample.malicious]
    reference = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    )
    reference.fit(np.asarray(sample.scores), np.asarray(targets))

    probe = np.linspace(-0.5, 1.5, 211)
    mine = np.asarray(state.transform(tuple(float(value) for value in probe)))
    assert np.max(np.abs(mine - reference.predict(probe))) <= 5e-10


def test_the_reviewed_scikit_learn_range_still_holds() -> None:
    """The attributes this parity rests on are documented ones on this release."""
    from sklearn.isotonic import IsotonicRegression

    from password_attack_detector.ml.dependencies import (
        installed_version,
        sklearn_compatible,
    )

    assert sklearn_compatible(installed_version("scikit-learn"))
    fitted = IsotonicRegression().fit([0.0, 1.0, 2.0], [0.0, 1.0, 1.0])
    assert hasattr(fitted, "X_thresholds_")
    assert hasattr(fitted, "y_thresholds_")


# ---------------------------------------------------------------------------
# The probability vocabulary transition
# ---------------------------------------------------------------------------


def test_an_uncalibrated_sample_may_not_name_a_calibrator() -> None:
    """The two halves of the vocabulary rule, checked in both directions."""
    scores = sx.graded_scores(20)
    labels = sx.graded_labels(scores)
    with pytest.raises(ValidationError, match="only a calibrated probability may"):
        sx.binary_sample(
            scores,
            labels,
            calibration_state_fingerprint=sx.MODEL_FINGERPRINT,
            calibration_method=CalibrationMethod.PLATT,
        )
    with pytest.raises(ValidationError, match="requires the calibrator"):
        sx.binary_sample(scores, labels, score_kind=ScoreKind.CALIBRATED_PROBABILITY)


def test_a_calibrated_sample_may_not_declare_method_none() -> None:
    """The method is what made the number a probability."""
    scores = sx.graded_scores(20)
    with pytest.raises(ValidationError, match="cannot declare calibration method"):
        sx.binary_sample(
            scores,
            sx.graded_labels(scores),
            score_kind=ScoreKind.CALIBRATED_PROBABILITY,
            calibration_state_fingerprint=sx.MODEL_FINGERPRINT,
            calibration_method=CalibrationMethod.NONE,
        )


def test_applying_a_calibrator_is_the_only_transition(
    state: CalibrationState, sample: BinaryScoreSample
) -> None:
    """The output carries the calibrator, the method, and the new score kind."""
    calibrated = apply_calibration(
        state, sample, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
    )
    assert calibrated.score_kind is ScoreKind.CALIBRATED_PROBABILITY
    assert calibrated.calibration_method is state.method
    assert (
        calibrated.calibration_state_fingerprint == state.calibration_state_fingerprint
    )
    assert calibrated.model_content_fingerprint == sample.model_content_fingerprint
    assert all(0.0 <= value <= 1.0 for value in calibrated.scores)
    assert calibrated.malicious == sample.malicious


def test_a_state_cannot_exist_for_method_none(state: CalibrationState) -> None:
    """The exact claim the contract exists to refuse."""
    payload = state.to_dict()
    payload["method"] = "none"
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationState.from_dict(payload)


def test_a_state_must_emit_a_calibrated_probability(state: CalibrationState) -> None:
    """Its output kind is not a free field."""
    payload = state.to_dict()
    payload["output_score_kind"] = "decision_score"
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationState.from_dict(payload)


def test_a_state_names_the_class_the_probability_is_of(state: CalibrationState) -> None:
    """A probability without a named event is not a probability of anything."""
    assert state.positive_class == POSITIVE_CLASS == "malicious"
    semantics = state.score_semantics
    assert semantics.score_kind is ScoreKind.CALIBRATED_PROBABILITY
    assert semantics.calibration_method is state.method
    assert "probability" in semantics.description.lower()


def test_a_state_cannot_claim_validation_b(state: CalibrationState) -> None:
    """Recorded provenance is validated, not merely carried."""
    payload = state.to_dict()
    payload["fit_source_partition"] = "validation_b"
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationState.from_dict(payload)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_brier_score_matches_a_hand_calculation() -> None:
    """Four rows, worked out by hand, compared against the reported number."""
    scores = (0.0, 0.25, 0.5, 0.75)
    labels = (False, False, True, True)
    parameters = IsotonicParameters(x_thresholds=(0.0, 0.75), y_thresholds=(0.0, 1.0))
    state = _state_with(isotonic=parameters)
    probabilities = state.transform(scores)
    expected = sum(
        (probability - (1.0 if label else 0.0)) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(scores)

    report = diagnose_calibration_fit(
        sx.binary_sample(scores, labels),
        state=state,
        config=CONFIG.model_copy(
            update={"min_calibration_rows": 1, "min_reliability_bin_rows": 1}
        ),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert report.brier_score == pytest.approx(expected, abs=1e-9)


def test_expected_calibration_error_matches_a_hand_calculation() -> None:
    """Two populated bins, weighted by their support, and nothing else counted.

    Eight rows. The calibrator maps the low score to ``0.25`` and the high one
    to ``0.75``, so with two bins the halves separate cleanly:

    * bin 0 -- four rows predicted ``0.25``, two of them malicious, so the
      observed rate is ``0.50`` and the gap is ``0.25``;
    * bin 1 -- four rows predicted ``0.75``, three of them malicious, so the
      observed rate is ``0.75`` and the gap is zero.

    ``ECE = (4/8) * 0.25 + (4/8) * 0.00 = 0.125``.
    """
    scores = (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    labels = (False, False, True, True, True, True, True, False)
    state = _state_with(
        isotonic=IsotonicParameters(x_thresholds=(0.0, 1.0), y_thresholds=(0.25, 0.75))
    )
    report = diagnose_calibration_fit(
        sx.binary_sample(scores, labels),
        state=state,
        config=CONFIG.model_copy(
            update={
                "min_calibration_rows": 1,
                "min_reliability_bin_rows": 1,
                "reliability_bin_count": 2,
            }
        ),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert [item.row_count for item in report.bins] == [4, 4]
    assert [item.positive_count for item in report.bins] == [2, 3]
    assert [item.mean_predicted_probability for item in report.bins] == [0.25, 0.75]
    assert [item.observed_positive_rate for item in report.bins] == [0.5, 0.75]
    assert report.expected_calibration_error == 0.125


def test_reliability_bins_are_fixed_by_configuration(state: CalibrationState) -> None:
    """Equal-width edges, so two runs over different data compare bin for bin."""
    report = _report(state, sx.validation_a(400))
    assert report.bin_count == 10
    assert [bin.index for bin in report.bins] == list(range(10))
    for index, item in enumerate(report.bins):
        assert item.lower_edge == pytest.approx(index / 10)
        assert item.upper_edge == pytest.approx((index + 1) / 10)
    assert [item.includes_upper_edge for item in report.bins][:-1] == [False] * 9
    assert report.bins[-1].includes_upper_edge is True


def test_an_empty_bin_says_so_rather_than_reporting_zero() -> None:
    """A fabricated zero-support point is a point on a curve nobody voted for."""
    scores = (0.05, 0.06, 0.95, 0.96)
    labels = (False, False, True, True)
    state = _state_with(
        isotonic=IsotonicParameters(x_thresholds=(0.05, 0.96), y_thresholds=(0.0, 1.0))
    )
    report = _report(state, sx.binary_sample(scores, labels), bins=10, min_rows=1)
    empty = [item for item in report.bins if item.row_count == 0]
    assert empty
    for item in empty:
        assert item.status is MetricStatus.UNAVAILABLE
        assert item.mean_predicted_probability is None
        assert item.observed_positive_rate is None
    assert sum(item.row_count for item in report.bins) == len(scores)


def test_a_thinly_populated_bin_is_insufficient_not_measured(
    state: CalibrationState,
) -> None:
    """Below the configured floor a bin keeps its counts and drops its standing."""
    report = _report(state, sx.validation_a(400), bins=10, min_rows=10_000)
    populated = report.populated_bins
    assert populated
    assert all(item.status is MetricStatus.INSUFFICIENT_SUPPORT for item in populated)
    assert all(item.row_count > 0 for item in populated)


def test_a_small_error_on_thin_support_is_not_a_pass() -> None:
    """The specific misreading the status exists to prevent."""
    scores = sx.graded_scores(30)
    labels = sx.graded_labels(scores)
    state = _state_with(
        isotonic=IsotonicParameters(x_thresholds=(0.0, 1.0), y_thresholds=(0.0, 1.0))
    )
    report = diagnose_calibration_fit(
        sx.binary_sample(scores, labels),
        state=state,
        config=CONFIG,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )
    assert report.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert "min_calibration_rows" in report.failing_requirements
    assert report.within_configured_error is None
    assert report.expected_calibration_error is not None


def test_a_measured_report_commits_to_a_verdict(state: CalibrationState) -> None:
    """With adequate support the ceiling comparison is made and recorded."""
    report = _report(state, sx.validation_a(400))
    assert report.status is MetricStatus.MEASURED
    assert report.within_configured_error is not None
    assert report.failing_requirements == ()
    assert report.expected_calibration_error is not None
    assert report.brier_score is not None


def test_no_metric_is_nan_or_infinite(state: CalibrationState) -> None:
    """Swept over the whole rendered report rather than the headline numbers."""
    rendered = json.dumps(_report(state, sx.validation_a(400)).to_dict())
    assert "NaN" not in rendered
    assert "Infinity" not in rendered
    for value in _numbers(json.loads(rendered)):
        assert math.isfinite(value)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "what"),
    [
        ("model_fingerprint", "model"),
        ("preprocessor_fingerprint", "preprocessor"),
    ],
)
def test_a_calibrator_from_another_model_is_refused(
    state: CalibrationState, argument: str, what: str
) -> None:
    """A fitted quantity and the data it is applied to are one chain, or neither."""
    other = sx.validation_a(400, **{argument: sx.OTHER_FINGERPRINT})
    with pytest.raises(ModelTrainingError, match=what):
        apply_calibration(state, other, ml_config_fingerprint=sx.CONFIG_FINGERPRINT)


def test_a_calibrator_from_another_partition_is_refused(
    state: CalibrationState,
) -> None:
    """Validation-A and validation-B carry one parent digest, or they are unrelated."""
    other = sx.validation_a(400, fingerprint=sx.OTHER_FINGERPRINT)
    with pytest.raises(ModelTrainingError, match="validation partition"):
        apply_calibration(state, other, ml_config_fingerprint=sx.CONFIG_FINGERPRINT)


def test_a_calibrator_from_another_configuration_is_refused(
    state: CalibrationState, sample: BinaryScoreSample
) -> None:
    """Reporting a calibrator under a configuration it was not fitted under."""
    with pytest.raises(ModelTrainingError, match="configuration"):
        apply_calibration(state, sample, ml_config_fingerprint=sx.OTHER_FINGERPRINT)


def test_the_chain_check_covers_every_link(state: CalibrationState) -> None:
    """Four links, and each of them alone is enough to refuse."""
    intact = {
        "model_content_fingerprint": sx.MODEL_FINGERPRINT,
        "preprocessor_fingerprint": sx.PREPROCESSOR_FINGERPRINT,
        "source_fingerprint": sx.PARTITION_FINGERPRINT,
        "ml_config_fingerprint": sx.CONFIG_FINGERPRINT,
    }
    require_chain(stage="probe", state=state, **intact)
    for link in intact:
        broken = {**intact, link: sx.OTHER_FINGERPRINT}
        with pytest.raises(ModelTrainingError):
            require_chain(stage="probe", state=state, **broken)


# ---------------------------------------------------------------------------
# Serialization and fingerprints
# ---------------------------------------------------------------------------


def test_state_serialization_is_byte_identical_across_a_round_trip(
    state: CalibrationState,
) -> None:
    """Deserialize, reserialize, compare bytes -- not fields."""
    assert CalibrationState.from_json(state.to_json()).to_json() == state.to_json()


def test_report_serialization_is_byte_identical_across_a_round_trip(
    state: CalibrationState,
) -> None:
    """The same guarantee for the report, including its empty bins."""
    report = _report(state, sx.validation_a(400))
    assert CalibrationReport.from_json(report.to_json()).to_json() == report.to_json()


def test_a_fingerprint_recomputes_on_load(state: CalibrationState) -> None:
    """Not carried, recomputed: the digest is checked at every construction."""
    reloaded = CalibrationState.from_json(state.to_json())
    assert reloaded.recomputed_fingerprint() == state.calibration_state_fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_count", 999),
        ("input_support_max", 42.0),
        ("model_id", "somebody-elses-model"),
        ("ml_config_fingerprint", sx.OTHER_FINGERPRINT),
    ],
)
def test_tampering_with_a_state_is_detected(
    state: CalibrationState, field: str, value: object
) -> None:
    """Every field is covered, because the digest is taken over all of them."""
    payload = state.to_dict()
    payload[field] = value
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationState.from_dict(payload)


def test_tampering_with_a_fitted_parameter_is_detected(
    state: CalibrationState,
) -> None:
    """Including the numbers a tampered calibrator would exist to change."""
    payload = state.to_dict()
    payload["isotonic"]["y_thresholds"][0] = 0.999
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationState.from_dict(payload)


def test_a_state_from_an_unsupported_contract_is_refused(
    state: CalibrationState,
) -> None:
    """Version first, validation second: an unknown contract is never guessed at."""
    payload = state.to_dict()
    payload["calibration_schema_version"] = "2.0.0"
    with pytest.raises(ModelTrainingError, match="schema version"):
        CalibrationState.from_dict(payload)
    assert CALIBRATION_SCHEMA_VERSION == "1.0.0"


def test_an_unknown_field_is_refused(state: CalibrationState) -> None:
    """A loose reader would drop it and then verify a digest over the remainder."""
    payload = state.to_dict()
    payload["shipped_by"] = "somebody"
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationState.from_dict(payload)


def test_malformed_json_is_refused_without_a_traceback() -> None:
    """The error names the failure kind and nothing about the payload."""
    with pytest.raises(ModelTrainingError, match="not valid JSON"):
        CalibrationState.from_json("{not json")


def test_a_state_fingerprint_is_path_and_time_independent(
    sample: BinaryScoreSample,
) -> None:
    """Nothing here reads a clock, a directory, or a machine name."""
    first = fit(sample).require_state()
    second = fit(sx.validation_a(400)).require_state()
    assert first.calibration_state_fingerprint == second.calibration_state_fingerprint


def test_a_different_calibration_changes_the_fingerprint(
    sample: BinaryScoreSample,
) -> None:
    """Two calibrators that behave differently are two different calibrators."""
    isotonic = fit(sample).require_state()
    platt = fit(sample, PLATT_CONFIG).require_state()
    assert isotonic.calibration_state_fingerprint != platt.calibration_state_fingerprint


def test_a_state_cannot_be_constructed_with_a_supplied_digest(
    state: CalibrationState,
) -> None:
    """The digest covers every other field, so a caller cannot choose it."""
    payload = state.to_dict()
    payload["calibration_state_fingerprint"] = "0" * 64
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationState.from_dict(payload)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("banned", ["e00000", "e00001", "campaign", "u:", "/home/"])
def test_no_identity_reaches_a_state_or_a_report(
    state: CalibrationState, banned: str
) -> None:
    """Anchors go in; counts and digests come out."""
    report = _report(state, sx.validation_a(400))
    for rendered in (state.to_json(), report.to_json()):
        assert banned not in rendered


def test_a_state_declares_no_identity_bearing_field(
    state: CalibrationState,
) -> None:
    """Swept over the declared field names, not only over one rendering."""
    from password_attack_detector.ml.schemas import prohibited_metadata_fields

    for model in (CalibrationState, CalibrationReport):
        assert prohibited_metadata_fields(list(model.model_fields)) == ()


def test_a_provenance_failure_message_carries_no_identity() -> None:
    """An error is output too, and it is read by more people than a report is."""
    scores = sx.graded_scores(400)
    bad = sx.binary_sample(scores, sx.graded_labels(scores), split=MLSplit.TEST)
    with pytest.raises(ModelTrainingError) as failure:
        fit(bad)
    message = str(failure.value)
    assert "e00000" not in message
    assert "/" not in message.replace("validation-A", "").replace("A/B", "")


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


def test_calibration_imports_no_label_reader_and_no_estimator() -> None:
    """The two boundaries this module sits inside, asserted on its syntax tree.

    The label-reader allowlist stays exactly ``{detection.evaluation,
    ml.dataset}`` -- checked in full elsewhere; this is the assertion that
    Milestone 5 did not widen it. The estimator boundary is why the calibration
    fits here are project-owned in the first place.
    """
    import ast
    from pathlib import Path

    import password_attack_detector.ml.calibration as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert "sklearn" not in roots
    assert "pyarrow" not in roots
    assert "pandas" not in roots
    assert "scipy" not in roots
    assert "password_attack_detector.ml.dataset" not in imported
    assert "password_attack_detector.detection.evaluation" not in imported


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_with(**overrides: Any) -> CalibrationState:
    """Return a hand-built calibrator for tests that need known parameters.

    ``overrides`` go through :meth:`SealedModel.seal`, so the digest is computed
    over the overridden content and the *semantic* validator is what fires --
    rather than the seal check masking it, which is what mutating a serialized
    payload would do.
    """
    isotonic = overrides.get("isotonic", LINE)
    shape = isotonic if isotonic is not None else LINE
    fields: dict[str, Any] = {
        "method": CalibrationMethod.ISOTONIC,
        "source_score_kind": ScoreKind.DECISION_SCORE,
        "output_score_kind": ScoreKind.CALIBRATED_PROBABILITY,
        "fit_source_partition": ValidationPartition.VALIDATION_A,
        "positive_class": POSITIVE_CLASS,
        "platt": None,
        "isotonic": isotonic,
        "input_support_min": shape.x_thresholds[0],
        "input_support_max": shape.x_thresholds[-1],
        "row_count": 4,
        "positive_count": 2,
        "negative_count": 2,
        "distinct_score_count": len(shape.x_thresholds),
        "validation_partition_fingerprint": sx.PARTITION_FINGERPRINT,
        "model_id": "model-under-test",
        "model_content_fingerprint": sx.MODEL_FINGERPRINT,
        "preprocessor_fingerprint": sx.PREPROCESSOR_FINGERPRINT,
        "ml_config_fingerprint": sx.CONFIG_FINGERPRINT,
    }
    fields.update(overrides)
    return CalibrationState.seal(**fields)


#: A well-formed isotonic parameter set, for tests about everything else.
LINE = IsotonicParameters(x_thresholds=(0.0, 1.0), y_thresholds=(0.0, 1.0))


def _reseal(record: Any, **overrides: Any) -> Any:
    """Return *record* rebuilt with *overrides*, sealed over the new content.

    Typed field values are read back off the model rather than out of its JSON
    rendering, so the rebuilt record carries enums and tuples rather than the
    strings and lists they serialise to.
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
        ({"calibration_schema_version": "2.0.0"}, "schema version"),
        ({"source_score_kind": ScoreKind.CALIBRATED_PROBABILITY}, "decision score"),
        ({"positive_class": "benign"}, "positive_class must be"),
        ({"isotonic": None}, "declares no fitted parameters"),
        (
            {
                "method": CalibrationMethod.PLATT,
                "platt": PlattParameters(
                    a=1.0, b=0.0, iterations=2, final_gradient_norm=0.0
                ),
            },
            "exactly one fitted parameter set",
        ),
        ({"negative_count": 3}, "does not sum"),
        ({"input_support_min": 5.0, "input_support_max": 5.0}, "degenerate"),
    ],
)
def test_a_state_enforces_its_internal_agreement(
    overrides: dict[str, Any], expected: str
) -> None:
    """Every invariant on a fitted calibrator, exercised one at a time.

    Built through ``seal`` rather than by editing a stored payload, so the
    digest agrees with the content and the semantic check is the one that
    fires. Editing a payload is tested separately, and there the digest is
    supposed to catch it first.
    """
    with pytest.raises(ValidationError, match=expected):
        _state_with(**overrides)


def test_a_calibrator_refuses_a_non_finite_score_at_inference() -> None:
    """There is no probability to map a NaN onto."""
    state = _state_with(isotonic=LINE)
    with pytest.raises(ModelTrainingError, match="non-finite score"):
        state.transform((0.5, math.inf))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"status": CalibrationStatus.FITTED, "state": None},
            "must carry the state",
        ),
        (
            {
                "status": CalibrationStatus.CONVERGENCE_FAILED,
                "method": CalibrationMethod.NONE,
            },
            "can only produce an uncalibrated",
        ),
        (
            {
                "status": CalibrationStatus.NOT_CALIBRATED,
                "method": CalibrationMethod.PLATT,
            },
            "must record calibration method",
        ),
    ],
)
def test_an_outcome_enforces_its_internal_agreement(
    overrides: dict[str, Any], expected: str
) -> None:
    """A state exists exactly when a calibrator was fitted, in both directions."""
    fields: dict[str, Any] = {
        "status": CalibrationStatus.NOT_CALIBRATED,
        "method": CalibrationMethod.NONE,
        "state": None,
        "row_count": 4,
        "positive_count": 2,
        "negative_count": 2,
        "distinct_score_count": 2,
    }
    fields.update(overrides)
    with pytest.raises(ValidationError, match=expected):
        CalibrationOutcome(**fields)


def test_a_fitted_outcome_carrying_a_failure_is_refused(
    state: CalibrationState,
) -> None:
    """A successful fit names no failing requirement."""
    with pytest.raises(ValidationError, match="names no failing requirement"):
        CalibrationOutcome(
            status=CalibrationStatus.FITTED,
            method=CalibrationMethod.ISOTONIC,
            state=state,
            failing_requirements=("something",),
            row_count=4,
            positive_count=2,
            negative_count=2,
            distinct_score_count=2,
        )


def test_an_unfitted_outcome_carrying_a_state_is_refused(
    state: CalibrationState,
) -> None:
    """Only a successful fit may hand back a calibrator."""
    with pytest.raises(ValidationError, match="carries a fitted state"):
        CalibrationOutcome(
            status=CalibrationStatus.CONVERGENCE_FAILED,
            method=CalibrationMethod.ISOTONIC,
            state=state,
            row_count=4,
            positive_count=2,
            negative_count=2,
            distinct_score_count=2,
        )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"positive_count": 5}, "more positives than rows"),
        (
            {"row_count": 0, "positive_count": 0, "status": MetricStatus.MEASURED},
            "unavailable, not a measurement of zero",
        ),
        ({"mean_predicted_probability": None}, "no mean predicted probability"),
        ({"observed_positive_rate": None}, "no observed positive rate"),
    ],
)
def test_a_reliability_bin_enforces_its_internal_agreement(
    overrides: dict[str, Any], expected: str
) -> None:
    """An empty bin reports nothing; a populated one reports both rates."""
    fields: dict[str, Any] = {
        "index": 0,
        "lower_edge": 0.0,
        "upper_edge": 0.5,
        "includes_upper_edge": False,
        "row_count": 4,
        "positive_count": 2,
        "mean_predicted_probability": 0.25,
        "observed_positive_rate": 0.5,
        "status": MetricStatus.MEASURED,
    }
    fields.update(overrides)
    with pytest.raises(ValidationError, match=expected):
        ReliabilityBin(**fields)


def test_a_report_refuses_bins_that_do_not_account_for_every_row(
    state: CalibrationState,
) -> None:
    """The bins are the whole sample, or the summary describes something else."""
    report = _report(state, sx.validation_a(400))
    with pytest.raises(ValidationError, match="do not account for every row"):
        _reseal(report, row_count=report.row_count + 1)


def test_a_report_refuses_an_uncalibrated_method(state: CalibrationState) -> None:
    """The absence of a calibrator is not a measurement of one."""
    report = _report(state, sx.validation_a(400))
    with pytest.raises(ValidationError, match="no calibration report"):
        _reseal(report, method=CalibrationMethod.NONE)


def test_a_malformed_report_payload_is_refused() -> None:
    """Not an object, wrong version, wrong shape -- each named separately."""
    with pytest.raises(ModelTrainingError, match="must be a JSON object"):
        CalibrationReport.from_dict(["not", "an", "object"])
    with pytest.raises(ModelTrainingError, match="not valid JSON"):
        CalibrationReport.from_json("{oops")
    with pytest.raises(ModelTrainingError, match="must be a JSON object"):
        CalibrationState.from_dict("a string")


def _report(
    state: CalibrationState,
    sample: BinaryScoreSample,
    *,
    bins: int = 10,
    min_rows: int = 20,
) -> CalibrationReport:
    """Return a calibration report over *sample*."""
    return diagnose_calibration_fit(
        sample,
        state=state,
        config=CONFIG.model_copy(
            update={"reliability_bin_count": bins, "min_reliability_bin_rows": min_rows}
        ),
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
    )


def _numbers(payload: Any) -> list[float]:
    """Return every numeric leaf of a decoded JSON payload."""
    if isinstance(payload, bool) or payload is None:
        return []
    if isinstance(payload, int | float):
        return [float(payload)]
    if isinstance(payload, dict):
        return [value for item in payload.values() for value in _numbers(item)]
    if isinstance(payload, list):
        return [value for item in payload for value in _numbers(item)]
    return []


def test_platt_parameters_reject_a_non_finite_scalar() -> None:
    """Enforced by the type, since a digest over NaN is stable and meaningless."""
    with pytest.raises(ValidationError):
        PlattParameters(a=math.inf, b=0.0, iterations=1, final_gradient_norm=0.0)


# ---------------------------------------------------------------------------
# Fit diagnostics versus out-of-sample calibration quality
# ---------------------------------------------------------------------------


def _quality(
    state: CalibrationState,
    sample: BinaryScoreSample | None = None,
    *,
    config: CalibrationConfig | None = None,
    raw: Any = None,
) -> CalibrationReport:
    """Return the out-of-sample calibration-quality report on validation-B."""
    return evaluate_calibration_quality(
        sample if sample is not None else sx.validation_b(400),
        state=state,
        config=config or CONFIG,
        ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
        raw_score_semantics=raw,
    )


def test_the_two_reports_name_what_they_are_evidence_of(
    state: CalibrationState, sample: BinaryScoreSample
) -> None:
    """The distinction is a field, not a convention a reader has to reconstruct."""
    diagnostic = _report(state, sample)
    quality = _quality(state)

    assert (
        diagnostic.evaluation_kind is CalibrationEvaluationKind.IN_SAMPLE_FIT_DIAGNOSTIC
    )
    assert diagnostic.source_partition is ValidationPartition.VALIDATION_A
    assert quality.evaluation_kind is CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION
    assert quality.source_partition is ValidationPartition.VALIDATION_B

    # Both record where the calibrator was *fitted*, which is validation-A in
    # either case. Fitting and measuring are different questions.
    assert diagnostic.fit_source_partition is ValidationPartition.VALIDATION_A
    assert quality.fit_source_partition is ValidationPartition.VALIDATION_A


def test_an_in_sample_diagnostic_is_never_champion_evidence(
    state: CalibrationState, sample: BinaryScoreSample
) -> None:
    """However good the number is. The calibrator was fitted on exactly these rows."""
    diagnostic = _report(state, sample)
    assert diagnostic.status is MetricStatus.MEASURED
    assert diagnostic.within_configured_error is True
    assert diagnostic.admissible_as_champion_evidence is False


def test_an_out_of_sample_report_may_be_champion_evidence(
    state: CalibrationState,
) -> None:
    """Admissible exactly when the error was measured and cleared the ceiling."""
    quality = _quality(state)
    assert quality.status is MetricStatus.MEASURED
    assert quality.admissible_as_champion_evidence is bool(
        quality.within_configured_error
    )
    if quality.within_configured_error:
        assert require_out_of_sample_evidence(quality, stage="probe") is quality


def test_an_out_of_sample_report_on_thin_support_is_not_admissible(
    state: CalibrationState,
) -> None:
    """Adequate rows are part of being evidence, not a separate courtesy."""
    quality = _quality(
        state,
        sx.validation_b(400),
        config=CONFIG.model_copy(update={"min_calibration_rows": 5000}),
    )
    assert quality.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert quality.admissible_as_champion_evidence is False
    assert "min_calibration_rows" in quality.failing_requirements
    with pytest.raises(ModelTrainingError, match="not admissible"):
        require_out_of_sample_evidence(quality, stage="probe")


def test_an_in_sample_report_cannot_masquerade_as_out_of_sample(
    state: CalibrationState, sample: BinaryScoreSample
) -> None:
    """The substitution a later champion gate must not be able to make by accident.

    Three independent refusals, because any one of them alone could be relaxed:
    the guard rejects the kind outright; the kind and the partition are tied to
    each other by a validator; and admissibility is pinned false for in-sample
    whatever else the record says.
    """
    diagnostic = _report(state, sample)

    with pytest.raises(ModelTrainingError, match="out-of-sample"):
        require_out_of_sample_evidence(diagnostic, stage="champion calibration gate")

    with pytest.raises(ValidationError, match="is measured on"):
        _reseal(
            diagnostic,
            evaluation_kind=CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION,
        )

    with pytest.raises(ValidationError, match="never champion calibration"):
        _reseal(diagnostic, admissible_as_champion_evidence=True)


def test_relabelling_a_stored_diagnostic_is_detected(
    state: CalibrationState, sample: BinaryScoreSample
) -> None:
    """Editing the serialized payload fails the digest before it fails anything else."""
    payload = _report(state, sample).to_dict()
    payload["evaluation_kind"] = "out_of_sample_validation"
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationReport.from_dict(payload)

    payload = _report(state, sample).to_dict()
    payload["admissible_as_champion_evidence"] = True
    with pytest.raises(ModelTrainingError, match="not valid"):
        CalibrationReport.from_dict(payload)


def test_no_report_can_name_test_or_holdout(state: CalibrationState) -> None:
    """There is no member to name them with.

    ``ValidationPartition`` has neither, so a calibration report cannot claim a
    test or holdout provenance even by being edited -- and the two evaluation
    functions refuse those splits behaviourally before it could come up.
    """
    assert {str(member) for member in ValidationPartition} == {
        "validation_a",
        "validation_b",
    }
    for split in (MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT):
        scores = sx.graded_scores(400)
        bad = sx.binary_sample(scores, sx.graded_labels(scores), split=split)
        with pytest.raises(ModelTrainingError, match="validation_b"):
            _quality(state, bad)


@pytest.mark.parametrize(
    ("split", "partition"),
    [
        (MLSplit.TEST, None),
        (MLSplit.NOVEL_ANOMALY_HOLDOUT, None),
        (MLSplit.TRAIN, None),
        (MLSplit.VALIDATION, ValidationPartition.VALIDATION_A),
    ],
)
def test_quality_evaluation_reads_validation_b_only(
    state: CalibrationState, split: MLSplit, partition: ValidationPartition | None
) -> None:
    """Including validation-A, whose rows fitted the calibrator being measured."""
    scores = sx.graded_scores(400)
    bad = sx.binary_sample(
        scores, sx.graded_labels(scores), split=split, partition=partition
    )
    with pytest.raises(ModelTrainingError, match="validation_b"):
        _quality(state, bad)


def test_a_quality_report_needs_the_same_parent_partition(
    state: CalibrationState,
) -> None:
    """A calibrator from one dataset cannot be measured on another's validation-B.

    Validation-A and validation-B carry the same parent digest only when they
    came from one campaign-disjoint partitioning, so this is the check that a
    calibrator and the rows judging it describe the same data.
    """
    stranger = sx.validation_b(400, fingerprint=sx.OTHER_FINGERPRINT)
    with pytest.raises(ModelTrainingError, match="validation partition"):
        _quality(state, stranger)


def test_evaluating_quality_leaves_the_calibrator_byte_identical(
    sample: BinaryScoreSample,
) -> None:
    """Fit on A, freeze, serialize, evaluate on B, compare bytes and digest.

    The calibrator is *applied* on validation-B, never refitted. Asserted on the
    serialized bytes rather than on field equality, because bytes are what a
    later milestone stores and compares.
    """
    state = fit(sample).require_state()
    before_json = state.to_json()
    before_digest = state.calibration_state_fingerprint

    quality = _quality(state)

    assert state.to_json() == before_json
    assert state.calibration_state_fingerprint == before_digest
    assert state.recomputed_fingerprint() == before_digest
    assert quality.calibration_state_fingerprint == before_digest
    # And the reloaded calibrator is still the same calibrator.
    assert CalibrationState.from_json(before_json).to_json() == before_json


def test_a_frozen_calibrator_is_unaffected_by_later_validation_a_rows(
    sample: BinaryScoreSample,
) -> None:
    """Once fitted, the calibrator and its validation-B report are settled.

    Changing validation-A *afterwards* changes nothing, because the calibrator
    is a value rather than a view onto the rows that produced it.
    """
    state = fit(sample).require_state()
    before_state = state.to_json()
    before_quality = _quality(state).to_json()

    # A radically different validation-A: different size, different labels.
    replacement = sx.validation_a(180)
    other = fit(replacement).require_state()
    assert other.to_json() != before_state

    assert state.to_json() == before_state
    assert _quality(state).to_json() == before_quality


def test_changing_validation_b_moves_the_quality_report_and_not_the_calibrator(
    sample: BinaryScoreSample,
) -> None:
    """The direction that matters: measurement rows must never reach the fit."""
    state = fit(sample).require_state()
    before_state = state.to_json()
    baseline = _quality(state).to_json()

    moved = _quality(state, sx.validation_b(360)).to_json()

    assert state.to_json() == before_state
    assert moved != baseline


# ---------------------------------------------------------------------------
# Raw-score Brier comparison
# ---------------------------------------------------------------------------


def _semantics(*, lower: float, upper: float) -> Any:
    """Return an uncalibrated score contract with the given declared bounds."""
    from password_attack_detector.ml.schemas import ScoreSemantics

    return ScoreSemantics(
        score_kind=ScoreKind.DECISION_SCORE,
        calibration_method=CalibrationMethod.NONE,
        lower_bound=lower,
        upper_bound=upper,
        description="Uncalibrated ordered magnitude for the malicious class.",
    )


def test_no_raw_comparator_without_a_declared_contract(
    state: CalibrationState,
) -> None:
    """Absent the model's score contract, nothing is known about its range."""
    quality = _quality(state)
    assert quality.raw_score_brier is None
    assert quality.raw_score_brier_status is MetricStatus.UNAVAILABLE
    assert (
        quality.raw_score_brier_unavailable_reason == "raw_score_contract_not_supplied"
    )
    assert quality.raw_score_kind is None
    assert quality.improves_on_raw_score is None


def test_an_unbounded_decision_score_produces_no_raw_comparator(
    state: CalibrationState,
) -> None:
    """A Brier score needs a forecast on ``[0, 1]``, and this is not one.

    The anomaly head's declared range is exactly this shape -- ``[-1, 0]`` --
    and a decision score is under no obligation to be narrower.
    """
    quality = _quality(state, raw=_semantics(lower=-1.0, upper=1.0))
    assert quality.raw_score_brier is None
    assert quality.raw_score_brier_status is MetricStatus.UNAVAILABLE
    assert quality.raw_score_brier_unavailable_reason == "raw_score_not_unit_bounded"
    assert quality.improves_on_raw_score is None


def test_a_unit_bounded_decision_score_may_be_compared(
    state: CalibrationState,
) -> None:
    """And is still called a decision score. Computing a Brier score renames nothing."""
    sample = sx.validation_b(400)
    quality = _quality(state, sample, raw=_semantics(lower=0.0, upper=1.0))

    assert quality.raw_score_brier is not None
    assert quality.raw_score_brier_status is MetricStatus.MEASURED
    assert quality.raw_score_brier_unavailable_reason is None
    assert quality.raw_score_kind is ScoreKind.DECISION_SCORE
    assert quality.improves_on_raw_score is (
        quality.brier_score is not None
        and quality.brier_score < quality.raw_score_brier
    )

    # Computed from the raw scores exactly as they arrived -- no clipping, no
    # rescaling, no transformation of any kind.
    from password_attack_detector.ml.calibration import brier_score as plain_brier

    assert quality.raw_score_brier == pytest.approx(
        plain_brier(sample.scores, sample.malicious), abs=1e-9
    )


def test_the_raw_comparator_is_never_called_a_probability(
    state: CalibrationState,
) -> None:
    """The uncalibrated field keeps its name, and the report keeps the distinction."""
    quality = _quality(state, raw=_semantics(lower=0.0, upper=1.0))
    assert quality.raw_score_kind is ScoreKind.DECISION_SCORE
    assert quality.raw_score_kind not in PROBABILITY_SCORE_KINDS

    rendered = quality.to_json()
    assert '"raw_score_kind":"decision_score"' in rendered
    assert "raw_score_probability" not in rendered
    with pytest.raises(ValidationError, match="does not make it a probability"):
        _reseal(quality, raw_score_kind=ScoreKind.CALIBRATED_PROBABILITY)


def test_an_out_of_range_observation_is_refused_rather_than_clipped(
    state: CalibrationState,
) -> None:
    """The contract said bounded and the scores disagree. Clipping would hide that.

    A clipped comparator would be the Brier score of a forecast the model never
    made, and it would flatter or damn calibration according to how far outside
    the interval the raw scores happened to fall.
    """
    scores = tuple(round(1.5 * value, 9) for value in sx.graded_scores(400))
    sample = sx.binary_sample(
        scores,
        sx.graded_labels(sx.graded_scores(400)),
        partition=ValidationPartition.VALIDATION_B,
    )
    quality = _quality(state, sample, raw=_semantics(lower=0.0, upper=1.0))
    assert quality.raw_score_brier is None
    assert (
        quality.raw_score_brier_unavailable_reason
        == "raw_score_outside_declared_bounds"
    )


def test_an_already_calibrated_contract_is_not_a_raw_comparator(
    state: CalibrationState,
) -> None:
    """Comparing calibrated output against itself would measure nothing."""
    from password_attack_detector.ml.schemas import ScoreSemantics

    calibrated = ScoreSemantics(
        score_kind=ScoreKind.CALIBRATED_PROBABILITY,
        calibration_method=CalibrationMethod.ISOTONIC,
        lower_bound=0.0,
        upper_bound=1.0,
        description="A calibrated probability of the malicious class.",
    )
    quality = _quality(state, raw=calibrated)
    assert quality.raw_score_brier is None
    assert (
        quality.raw_score_brier_unavailable_reason == "raw_score_is_already_calibrated"
    )


def test_the_measurement_primitive_is_provenance_free() -> None:
    """One implementation behind every report, and it knows nothing about splits.

    That is what lets a later milestone measure a frozen champion on the test
    split without a second Brier implementation drifting away from this one --
    and it is safe to expose because it takes numbers, not rows, and decides
    nothing about where they came from.
    """
    import inspect

    from password_attack_detector.ml.calibration import brier_score as plain_brier

    assert plain_brier((0.0, 1.0), (False, True)) == 0.0
    assert plain_brier((1.0, 0.0), (False, True)) == 1.0
    assert plain_brier((0.5, 0.5), (False, True)) == 0.25
    parameters = set(inspect.signature(plain_brier).parameters)
    assert parameters == {"probabilities", "labels"}


def test_test_and_holdout_rows_change_nothing_they_are_refused_by(
    sample: BinaryScoreSample,
) -> None:
    """Radically different test and holdout rows move no fitted quantity at all.

    Stated the only way this design permits it to be stated. There is no
    parameter through which a test or holdout row could reach a fit or a
    measurement, so "vary the test split and assert nothing changed" becomes:
    build those rows, offer them to every entry point, watch each one refuse,
    and confirm the calibrator and its authoritative validation-B report are
    byte-identical afterwards.
    """
    state = fit(sample).require_state()
    before_state = state.to_json()
    before_quality = _quality(state).to_json()

    for split in (MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT):
        # Deliberately unlike anything the calibrator saw: different size,
        # inverted labels, scores in a different part of the range.
        scores = tuple(round(0.4 + index / 1000, 9) for index in range(250))
        intruder = sx.binary_sample(
            scores,
            tuple(index % 3 == 0 for index in range(250)),
            split=split,
        )
        with pytest.raises(ModelTrainingError):
            fit(intruder)
        with pytest.raises(ModelTrainingError):
            diagnose_calibration_fit(
                intruder,
                state=state,
                config=CONFIG,
                ml_config_fingerprint=sx.CONFIG_FINGERPRINT,
            )
        with pytest.raises(ModelTrainingError):
            _quality(state, intruder)
        with pytest.raises(ModelTrainingError):
            apply_calibration(
                state, intruder, ml_config_fingerprint=sx.CONFIG_FINGERPRINT
            )

    assert state.to_json() == before_state
    assert _quality(state).to_json() == before_quality
