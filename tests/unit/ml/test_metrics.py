"""TEST metric arithmetic, hand-computed and checked against the definitions.

Every expected value in this file was worked out by hand from a small confusion
matrix, not read back from the implementation. A metric suite whose expectations
came from the code it tests would pass through any sign error the code contained.

The other half is null semantics: an empty denominator is ``unavailable``, and
the suite asserts that in every place a denominator can empty -- because a zero
there reads as a measured absence, which is a claim the data does not support.
"""

from __future__ import annotations

import pytest

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.enums import MetricStatus
from password_attack_detector.ml.metrics import (
    METRIC_DEFINITION_VERSION,
    RELIABILITY_BUCKET_COUNT,
    BinaryTestMetrics,
    CalibrationTestMetrics,
    ConfusionMatrix,
    Rate,
    ReliabilityBucket,
    binary_metrics,
    calibration_metrics,
    confusion_matrix,
    metric_definition_fingerprint,
    rate,
)
from password_attack_detector.ml.ranking import (
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
)

#: A hand-built population: four rows, one of each confusion cell.
FLAGS = (True, True, False, False)
TRUTH = (True, False, True, False)


# ---------------------------------------------------------------------------
# The confusion matrix
# ---------------------------------------------------------------------------


def test_the_confusion_matrix_counts_each_cell() -> None:
    """One row in each cell, counted by hand."""
    matrix = confusion_matrix(FLAGS, TRUTH)
    assert matrix.true_positives == 1
    assert matrix.false_positives == 1
    assert matrix.true_negatives == 1
    assert matrix.false_negatives == 1
    assert matrix.row_count == 4
    assert matrix.positive_count == 2
    assert matrix.negative_count == 2
    assert matrix.flagged_count == 2


def test_a_perfect_detector_has_no_off_diagonal() -> None:
    """Six rows, every decision right."""
    matrix = confusion_matrix(
        (True, True, True, False, False, False),
        (True, True, True, False, False, False),
    )
    assert (matrix.true_positives, matrix.true_negatives) == (3, 3)
    assert (matrix.false_positives, matrix.false_negatives) == (0, 0)


def test_misaligned_decisions_and_outcomes_are_refused() -> None:
    """A metric over misaligned rows is arithmetically valid and entirely wrong."""
    with pytest.raises(DataValidationError, match="cannot be scored against"):
        confusion_matrix((True, False), (True,))


# ---------------------------------------------------------------------------
# Rates, and the denominators behind them
# ---------------------------------------------------------------------------


def test_a_rate_publishes_the_counts_it_came_from() -> None:
    """A reader can recompute the value rather than trust it."""
    computed = rate(3, 4)
    assert (computed.numerator, computed.denominator) == (3, 4)
    assert computed.value == 0.75
    assert computed.status is MetricStatus.MEASURED


def test_an_empty_denominator_is_unavailable_not_zero() -> None:
    """ "None of no rows" is not a measured absence."""
    computed = rate(0, 0)
    assert computed.value is None
    assert computed.status is MetricStatus.UNAVAILABLE
    assert computed.lower_bound is None


def test_a_wilson_interval_is_published_when_asked_for() -> None:
    """Published, never applied: nothing in this project gates on a bound."""
    computed = rate(1, 10, confidence=0.95)
    assert computed.confidence == 0.95
    assert computed.lower_bound is not None and computed.upper_bound is not None
    assert computed.lower_bound < computed.value < computed.upper_bound  # type: ignore[operator]
    assert computed.lower_bound >= 0.0 and computed.upper_bound <= 1.0


def test_a_rate_that_disagrees_with_its_counts_is_refused() -> None:
    """The published value must be the quotient it claims to be."""
    with pytest.raises(ValueError, match="does not equal the counts"):
        Rate(numerator=1, denominator=4, value=0.5)


def test_a_value_without_a_denominator_is_refused() -> None:
    """Both directions of the same rule."""
    with pytest.raises(ValueError, match="exactly when its denominator"):
        Rate(numerator=0, denominator=0, value=0.0)
    with pytest.raises(ValueError, match="exactly when its denominator"):
        Rate(numerator=1, denominator=4, value=None)


def test_a_numerator_above_its_denominator_is_refused() -> None:
    """More successes than trials is not a rate."""
    with pytest.raises(ValueError, match="cannot exceed its denominator"):
        Rate(numerator=5, denominator=4, value=1.25)


def test_an_interval_on_an_undefined_rate_is_refused() -> None:
    """There is no interval around a proportion of nothing."""
    with pytest.raises(ValueError, match="undefined rate has no interval"):
        Rate(
            numerator=0,
            denominator=0,
            value=None,
            lower_bound=0.0,
            upper_bound=1.0,
            confidence=0.95,
        )


# ---------------------------------------------------------------------------
# The binary metrics, computed by hand
# ---------------------------------------------------------------------------


def test_every_binary_metric_matches_the_hand_computation() -> None:
    """TP=FP=TN=FN=1, so precision, recall, FPR, F1 and balanced accuracy are 0.5."""
    metrics = binary_metrics(
        flags=FLAGS, malicious=TRUTH, score_unavailable_reason="hand_built"
    )
    assert metrics.precision.value == 0.5
    assert metrics.recall.value == 0.5
    assert metrics.false_positive_rate.value == 0.5
    assert metrics.f1 == 0.5
    assert metrics.balanced_accuracy == 0.5
    assert metrics.accuracy.value == 0.5


def test_a_skewed_matrix_matches_the_hand_computation() -> None:
    """TP=3 FP=1 FN=2 TN=4: precision 3/4, recall 3/5, FPR 1/5, F1 2*.75*.6/1.35."""
    flags = (True, True, True, True, False, False, False, False, False, False)
    truth = (True, True, True, False, True, True, False, False, False, False)
    metrics = binary_metrics(
        flags=flags, malicious=truth, score_unavailable_reason="hand_built"
    )
    assert metrics.confusion.true_positives == 3
    assert metrics.confusion.false_positives == 1
    assert metrics.confusion.false_negatives == 2
    assert metrics.confusion.true_negatives == 4
    assert metrics.precision.value == 0.75
    assert metrics.recall.value == 0.6
    assert metrics.false_positive_rate.value == 0.2
    assert metrics.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35, abs=1e-9)
    assert metrics.balanced_accuracy == pytest.approx((0.6 + 0.8) / 2, abs=1e-9)
    assert metrics.accuracy.value == 0.7


def test_f1_is_unavailable_when_precision_and_recall_are_both_zero() -> None:
    """``2PR/(P+R)`` has an empty denominator there; zero would be a division
    nobody performed."""
    metrics = binary_metrics(
        flags=(True, False),
        malicious=(False, True),
        score_unavailable_reason="hand_built",
    )
    assert metrics.precision.value == 0.0
    assert metrics.recall.value == 0.0
    assert metrics.f1 is None


def test_metrics_over_one_class_report_unavailable_rates() -> None:
    """No benign row means no false-positive rate to report."""
    metrics = binary_metrics(
        flags=(True, True),
        malicious=(True, True),
        score_unavailable_reason="hand_built",
    )
    assert metrics.false_positive_rate.value is None
    assert metrics.balanced_accuracy is None
    assert metrics.recall.value == 1.0


def test_thin_support_is_reported_as_insufficient() -> None:
    """A metric over three positive rows is not a measurement."""
    metrics = binary_metrics(
        flags=FLAGS,
        malicious=TRUTH,
        score_unavailable_reason="hand_built",
        min_positive_rows=100,
        min_benign_rows=100,
    )
    assert metrics.support_status is MetricStatus.INSUFFICIENT_SUPPORT
    assert metrics.measured is False


# ---------------------------------------------------------------------------
# PR-AUC, under the Milestone 7 convention
# ---------------------------------------------------------------------------


def test_pr_auc_uses_the_milestone_seven_convention() -> None:
    """Scores 0.9+ 0.8- 0.7+ 0.6-: 1*0.5 + (2/3)*0.5, computed by hand."""
    metrics = binary_metrics(
        flags=FLAGS,
        malicious=(True, False, True, False),
        scores=(0.9, 0.8, 0.7, 0.6),
    )
    assert metrics.pr_auc == pytest.approx(1.0 * 0.5 + (2 / 3) * 0.5, abs=1e-9)
    assert metrics.distinct_score_count == 4
    assert metrics.discrimination_metric == RANKING_METRIC_NAME
    assert metrics.discrimination_integration == PR_AUC_INTEGRATION


def test_a_perfect_ranking_scores_one() -> None:
    """Every positive above every negative."""
    metrics = binary_metrics(
        flags=(True, True, False, False),
        malicious=(True, True, False, False),
        scores=(0.9, 0.8, 0.2, 0.1),
    )
    assert metrics.pr_auc == 1.0


def test_tied_scores_are_one_level() -> None:
    """Rows sharing a score enter the cumulative counts together."""
    metrics = binary_metrics(
        flags=(True, True, True, True),
        malicious=(True, False, True, False),
        scores=(0.5, 0.5, 0.5, 0.5),
    )
    assert metrics.distinct_score_count == 1
    # One level: precision is the prevalence, recall is one.
    assert metrics.pr_auc == 0.5


def test_reordering_tied_rows_changes_nothing() -> None:
    """Nothing consults row order inside a tie."""
    first = binary_metrics(
        flags=(True,) * 4,
        malicious=(True, False, True, False),
        scores=(0.5, 0.5, 0.4, 0.4),
    )
    second = binary_metrics(
        flags=(True,) * 4,
        malicious=(False, True, False, True),
        scores=(0.5, 0.5, 0.4, 0.4),
    )
    assert first.pr_auc == second.pr_auc


def test_a_system_without_a_score_gets_no_manufactured_pr_auc() -> None:
    """The rule engine's ordinal magnitude is not a discrimination score."""
    metrics = binary_metrics(
        flags=FLAGS,
        malicious=TRUTH,
        score_unavailable_reason="ordinal_rule_risk_is_not_a_ranking_score",
    )
    assert metrics.pr_auc is None
    assert metrics.pr_auc_unavailable_reason is not None
    assert "ordinal" in metrics.pr_auc_unavailable_reason
    assert metrics.distinct_score_count is None


def test_an_unexplained_absent_score_is_refused() -> None:
    """An unexplained absent metric is indistinguishable from a bug."""
    with pytest.raises(DataValidationError, match="must state why"):
        binary_metrics(flags=FLAGS, malicious=TRUTH)


def test_pr_auc_is_unavailable_over_a_single_class() -> None:
    """Precision-recall is undefined when one class is missing."""
    metrics = binary_metrics(
        flags=(True, True), malicious=(True, True), scores=(0.9, 0.8)
    )
    assert metrics.pr_auc is None
    assert metrics.pr_auc_unavailable_reason is not None
    assert "one_class_only" in metrics.pr_auc_unavailable_reason


def test_misaligned_scores_are_refused() -> None:
    """A ranking over the wrong rows would be a ranking of something else."""
    with pytest.raises(DataValidationError, match="cannot be ranked against"):
        binary_metrics(flags=FLAGS, malicious=TRUTH, scores=(0.9, 0.8))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def test_the_brier_score_matches_the_hand_computation() -> None:
    """Two rows: (0.9 vs 1) and (0.2 vs 0) give (0.01 + 0.04) / 2."""
    computed = calibration_metrics(probabilities=(0.9, 0.2), malicious=(True, False))
    assert computed.brier_score == pytest.approx((0.01 + 0.04) / 2, abs=1e-9)


def test_a_perfectly_calibrated_pair_has_no_error() -> None:
    """Predicting one for a positive and zero for a negative."""
    computed = calibration_metrics(probabilities=(1.0, 0.0), malicious=(True, False))
    assert computed.brier_score == 0.0
    assert computed.expected_calibration_error == 0.0


def test_an_empty_bucket_reports_nothing_rather_than_zero() -> None:
    """A bucket nothing landed in is not a bucket where nothing was malicious."""
    computed = calibration_metrics(probabilities=(0.05,), malicious=(False,))
    assert computed.bucket_count == RELIABILITY_BUCKET_COUNT
    populated = [bucket for bucket in computed.buckets if bucket.row_count]
    assert len(populated) == 1
    for bucket in computed.buckets:
        if bucket.row_count == 0:
            assert bucket.observed_rate is None
            assert bucket.mean_predicted is None


def test_the_buckets_account_for_every_row() -> None:
    """Including a probability of exactly one, which the last bucket closes on."""
    computed = calibration_metrics(
        probabilities=(0.0, 0.5, 1.0), malicious=(False, True, True)
    )
    assert sum(bucket.row_count for bucket in computed.buckets) == 3


def test_a_probability_outside_the_unit_interval_is_refused() -> None:
    """A value outside [0, 1] is not a probability."""
    with pytest.raises(DataValidationError, match="not a probability"):
        calibration_metrics(probabilities=(1.5,), malicious=(True,))


def test_calibration_metrics_are_absent_without_probabilities() -> None:
    """Brier and ECE are defined for probabilities and for nothing else."""
    metrics = binary_metrics(
        flags=FLAGS, malicious=TRUTH, score_unavailable_reason="hand_built"
    )
    assert metrics.calibration is None


def test_calibration_metrics_are_present_with_probabilities() -> None:
    """The positive case, so the absence above means something."""
    metrics = binary_metrics(
        flags=FLAGS,
        malicious=TRUTH,
        scores=(0.9, 0.8, 0.7, 0.6),
        probabilities=(0.9, 0.8, 0.7, 0.6),
    )
    assert metrics.calibration is not None
    assert metrics.calibration.row_count == 4


def test_an_empty_bucket_range_is_refused() -> None:
    """A bucket must span a range."""
    with pytest.raises(ValueError, match="lower bound must fall below"):
        ReliabilityBucket(
            lower=0.5, upper=0.5, row_count=0, mean_predicted=None, observed_rate=None
        )


def test_a_populated_bucket_reports_its_statistics() -> None:
    """Both directions of the emptiness rule."""
    with pytest.raises(ValueError, match="populated bucket reports"):
        ReliabilityBucket(
            lower=0.0, upper=0.1, row_count=3, mean_predicted=None, observed_rate=0.5
        )


# ---------------------------------------------------------------------------
# The sealed record
# ---------------------------------------------------------------------------


def test_the_metrics_record_is_sealed_and_reproducible() -> None:
    """Two computations over one population are the same record."""
    first = binary_metrics(
        flags=FLAGS, malicious=TRUTH, score_unavailable_reason="hand_built"
    )
    second = binary_metrics(
        flags=FLAGS, malicious=TRUTH, score_unavailable_reason="hand_built"
    )
    assert first.to_json() == second.to_json()
    assert first.metrics_fingerprint == first.recomputed_fingerprint()


def test_an_edited_metrics_record_is_refused() -> None:
    """The digest is a field, recomputed on every deserialization."""
    from password_attack_detector.exceptions import ModelTrainingError

    payload = binary_metrics(
        flags=FLAGS, malicious=TRUTH, score_unavailable_reason="hand_built"
    ).to_dict()
    payload["row_count"] = payload["row_count"] + 1
    with pytest.raises(ModelTrainingError, match="binary test metrics"):
        BinaryTestMetrics.from_dict(payload)


def test_a_matrix_that_disagrees_with_its_counts_is_refused() -> None:
    """The record cannot claim support the matrix does not show."""
    valid = binary_metrics(
        flags=FLAGS, malicious=TRUTH, score_unavailable_reason="hand_built"
    )
    with pytest.raises(ValueError, match="does not account for every row"):
        BinaryTestMetrics.seal(
            confusion=ConfusionMatrix(
                true_positives=1,
                false_positives=1,
                true_negatives=1,
                false_negatives=1,
            ),
            precision=valid.precision,
            recall=valid.recall,
            false_positive_rate=valid.false_positive_rate,
            f1=valid.f1,
            balanced_accuracy=valid.balanced_accuracy,
            accuracy=valid.accuracy,
            pr_auc=None,
            pr_auc_unavailable_reason="hand_built",
            distinct_score_count=None,
            calibration=None,
            row_count=99,
            positive_count=2,
            negative_count=2,
            flagged_count=2,
            support_status=MetricStatus.MEASURED,
        )


def test_a_present_pr_auc_names_no_absence_reason() -> None:
    """An absent metric names why, and a present one names nothing."""
    valid = binary_metrics(flags=FLAGS, malicious=TRUTH, scores=(0.9, 0.8, 0.7, 0.6))
    assert valid.pr_auc is not None
    assert valid.pr_auc_unavailable_reason is None


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------


def test_the_metric_definition_is_fingerprinted() -> None:
    """A change to the arithmetic is visible as a different evaluation."""
    assert metric_definition_fingerprint() == metric_definition_fingerprint()
    assert len(metric_definition_fingerprint()) == 64


def test_the_metric_contract_version_is_pinned() -> None:
    """A change to what these numbers mean is a visible edit."""
    assert METRIC_DEFINITION_VERSION == "1.0.0"


def test_no_metric_schema_names_an_identifier() -> None:
    """A metric record is an aggregate and is safe to publish."""
    forbidden = {"anchor_event_id", "event_id", "campaign_id", "user_id", "labels"}
    for model in (
        BinaryTestMetrics,
        ConfusionMatrix,
        Rate,
        ReliabilityBucket,
        CalibrationTestMetrics,
    ):
        assert not set(model.model_fields) & forbidden, model.__name__
