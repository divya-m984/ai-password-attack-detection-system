"""TEST metric definitions, computed once against frozen decisions.

Every quantity here is **calculated**. There is no literal metric anywhere in
this module, no threshold search, and no code path that reads a number and
writes a configuration value back: a metric that could feed a setting is a
metric measured on data that was used for fitting.

**A rate carries its own denominator.** :class:`Rate` publishes the numerator,
the denominator, and the value, so a reader can recompute it rather than trust
it -- and can see that a 100% detection rate over four positive rows is not the
claim a 100% detection rate over four thousand would be. When the denominator
is empty the value is ``None``: never ``0.0``, which reads as a measured
absence, and never ``1.0``, which reads as measured perfection. Both are claims
the data does not support.

**One PR-AUC definition in this project.** The exact, tie-grouped, step-wise
convention lives in :mod:`~password_attack_detector.ml.ranking` and Milestone 7
selects champions with it. This module calls that same code rather than
reimplementing it, so a TEST figure and the validation figure that promoted the
model are the same measurement applied to different rows.

**A system without a continuous score has no PR-AUC, and gets none.** The rule
engine emits an ordinal 0-100 severity magnitude, not a discrimination score,
and reinterpreting it as one would manufacture a ranking metric out of a
quantity that was never meant to rank. The field is ``None`` and the reason is
recorded.

**Calibration metrics require a calibrated probability.** Brier and ECE are
defined for probabilities; applying either to a raw decision score or to an
ordinal risk magnitude would produce a number with no meaning. Both are
``None`` unless the decisions carry a calibrated probability.

Nothing in this module reads a label table. It is handed decisions and outcomes
that a permitted reader already joined, and it computes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.calibration import SealedModel, quantize
from password_attack_detector.ml.enums import MetricStatus
from password_attack_detector.ml.gates import wilson_interval
from password_attack_detector.ml.ranking import (
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
    ScoreLevel,
    pr_auc,
    score_levels,
)
from password_attack_detector.ml.schemas import Sha256Hex

__all__ = [
    "METRIC_DEFINITION_VERSION",
    "RELIABILITY_BUCKET_COUNT",
    "BinaryTestMetrics",
    "CalibrationTestMetrics",
    "ConfusionMatrix",
    "Rate",
    "ReliabilityBucket",
    "binary_metrics",
    "calibration_metrics",
    "confusion_matrix",
    "metric_definition_fingerprint",
    "rate",
]

#: The metric-contract version.  Separate from every artifact schema: what a
#: TEST receipt *binds* can change without the arithmetic changing, and a change
#: to the arithmetic must be visible even when nothing else moved.
METRIC_DEFINITION_VERSION: Final[str] = "1.0.0"

#: How many equal-width buckets a reliability diagram uses.  Declared rather
#: than configurable at evaluation time: a bucket count chosen after seeing the
#: TEST probabilities would be a presentation fitted to the result.
RELIABILITY_BUCKET_COUNT: Final[int] = 10


class Rate(BaseModel):
    """One rate, published beside the counts it was computed from.

    ``value`` is ``None`` exactly when the denominator is empty. The interval,
    where present, is a Wilson score interval at the recorded confidence -- it
    is *published*, never applied: nothing in this project passes or fails on
    an interval bound.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence: float | None = None

    @model_validator(mode="after")
    def check_rate(self) -> Self:
        """The value exists exactly when it is defined, and equals the quotient."""
        if (self.value is None) != (self.denominator == 0):
            raise ValueError(
                "a rate has a value exactly when its denominator is non-empty; "
                "an empty denominator is unavailable, never zero"
            )
        if self.numerator > self.denominator:
            raise ValueError("a rate's numerator cannot exceed its denominator")
        if self.value is not None:
            if not math.isfinite(self.value):
                raise ValueError("a rate must be finite")
            if quantize(self.numerator / self.denominator) != self.value:
                raise ValueError("a rate does not equal the counts it publishes")
        bounds = (self.lower_bound, self.upper_bound, self.confidence)
        if any(item is None for item in bounds) and any(
            item is not None for item in bounds
        ):
            raise ValueError("an interval is published in full or not at all")
        if self.lower_bound is not None and self.value is None:
            raise ValueError("an undefined rate has no interval")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("an interval's lower bound exceeds its upper bound")
        return self

    @property
    def status(self) -> MetricStatus:
        """Return whether this rate is a measurement."""
        return MetricStatus.UNAVAILABLE if self.value is None else MetricStatus.MEASURED


def rate(numerator: int, denominator: int, *, confidence: float | None = None) -> Rate:
    """Return the rate *numerator* / *denominator*, unavailable when empty.

    Supplying *confidence* attaches a Wilson score interval. It is attached for
    publication only: no gate, threshold, or comparison in this project reads a
    bound.
    """
    if denominator <= 0:
        return Rate(numerator=numerator, denominator=denominator, value=None)
    value = quantize(numerator / denominator)
    if confidence is None:
        return Rate(numerator=numerator, denominator=denominator, value=value)
    interval = wilson_interval(numerator, denominator, confidence=confidence)
    if interval is None:  # pragma: no cover - guarded by the check above
        return Rate(numerator=numerator, denominator=denominator, value=value)
    lower, upper = interval
    return Rate(
        numerator=numerator,
        denominator=denominator,
        value=value,
        lower_bound=quantize(lower),
        upper_bound=quantize(upper),
        confidence=confidence,
    )


class ConfusionMatrix(BaseModel):
    """The four counts every binary metric below is derived from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)

    @property
    def row_count(self) -> int:
        """Return how many rows the matrix accounts for."""
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    @property
    def positive_count(self) -> int:
        """Return how many rows were actually malicious."""
        return self.true_positives + self.false_negatives

    @property
    def negative_count(self) -> int:
        """Return how many rows were actually benign."""
        return self.false_positives + self.true_negatives

    @property
    def flagged_count(self) -> int:
        """Return how many rows the system flagged."""
        return self.true_positives + self.false_positives


class ReliabilityBucket(BaseModel):
    """One equal-width probability bucket of a reliability diagram.

    ``observed_rate`` is ``None`` for a bucket nothing landed in. An empty
    bucket is not a bucket in which nothing was malicious.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    row_count: int = Field(ge=0)
    mean_predicted: float | None
    observed_rate: float | None

    @model_validator(mode="after")
    def check_bucket(self) -> Self:
        """An empty bucket reports nothing rather than reporting zero."""
        if self.lower >= self.upper:
            raise ValueError("a bucket's lower bound must fall below its upper bound")
        empty = self.row_count == 0
        if empty != (self.mean_predicted is None):
            raise ValueError("a populated bucket reports its mean prediction")
        if empty != (self.observed_rate is None):
            raise ValueError("a populated bucket reports its observed rate")
        return self


class CalibrationTestMetrics(BaseModel):
    """How well a calibrated probability matched the outcomes it predicted.

    Published, never applied. Nothing in Milestone 9 may read these numbers and
    refit a calibrator: the calibrator was frozen before the TEST labels were
    opened, and a calibration corrected against TEST would be a calibrator
    fitted on TEST.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    brier_score: float
    expected_calibration_error: float
    bucket_count: int = Field(ge=1)
    buckets: tuple[ReliabilityBucket, ...]
    row_count: int = Field(ge=1)

    @model_validator(mode="after")
    def check_metrics(self) -> Self:
        """The buckets partition the unit interval and account for every row."""
        if not math.isfinite(self.brier_score) or not math.isfinite(
            self.expected_calibration_error
        ):
            raise ValueError("calibration metrics must be finite")
        if len(self.buckets) != self.bucket_count:
            raise ValueError("the bucket count disagrees with the buckets")
        if sum(bucket.row_count for bucket in self.buckets) != self.row_count:
            raise ValueError("the buckets do not account for every row")
        return self


class BinaryTestMetrics(SealedModel):
    """One system's binary TEST performance, computed once and sealed.

    Sealed so a receipt that carries these numbers cannot be edited into
    different ones. Carries no anchor, no row, and no identifier: a metric
    record is an aggregate and is safe to publish.
    """

    fingerprint_field: ClassVar[str] = "metrics_fingerprint"
    schema_version_field: ClassVar[str] = "metric_definition_version"
    schema_version: ClassVar[str] = METRIC_DEFINITION_VERSION
    record_label: ClassVar[str] = "binary test metrics"

    metric_definition_version: str = METRIC_DEFINITION_VERSION

    confusion: ConfusionMatrix
    precision: Rate
    recall: Rate
    false_positive_rate: Rate
    f1: float | None
    balanced_accuracy: float | None
    accuracy: Rate

    #: Exact PR-AUC under the Milestone 7 convention, or ``None`` with a stated
    #: reason. A system with no continuous discrimination score does not get a
    #: manufactured one.
    pr_auc: float | None
    pr_auc_unavailable_reason: str | None
    discrimination_metric: str = RANKING_METRIC_NAME
    discrimination_integration: str = PR_AUC_INTEGRATION
    distinct_score_count: int | None = Field(default=None, ge=1)

    calibration: CalibrationTestMetrics | None = None

    row_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    support_status: MetricStatus

    metrics_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_metrics(self) -> Self:
        """Counts agree with the matrix, and every absence names its reason."""
        matrix = self.confusion
        if matrix.row_count != self.row_count:
            raise ValueError("the confusion matrix does not account for every row")
        if matrix.positive_count != self.positive_count:
            raise ValueError("the positive support disagrees with the matrix")
        if matrix.negative_count != self.negative_count:
            raise ValueError("the benign support disagrees with the matrix")
        if matrix.flagged_count != self.flagged_count:
            raise ValueError("the flagged count disagrees with the matrix")
        if (self.pr_auc is None) != (self.pr_auc_unavailable_reason is not None):
            raise ValueError(
                "an absent PR-AUC names why it is absent, and a present one "
                "names nothing"
            )
        if (self.pr_auc is None) != (self.distinct_score_count is None):
            raise ValueError(
                "a computed PR-AUC publishes the number of distinct score levels "
                "it was integrated over"
            )
        for name in ("f1", "balanced_accuracy", "pr_auc"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or absent")
        return self

    @property
    def measured(self) -> bool:
        """Return whether this system had support for its metrics to mean anything."""
        return self.support_status is MetricStatus.MEASURED


def confusion_matrix(
    flags: Sequence[bool], malicious: Sequence[bool]
) -> ConfusionMatrix:
    """Return the confusion matrix for aligned decisions and outcomes.

    Raises:
        DataValidationError: when the two sequences disagree in length. A metric
            computed over misaligned rows would be arithmetically valid and
            entirely wrong.
    """
    if len(flags) != len(malicious):
        raise DataValidationError(
            f"{len(flags)} decision(s) cannot be scored against "
            f"{len(malicious)} outcome(s)"
        )
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for flagged, actual in zip(flags, malicious, strict=True):
        if flagged and actual:
            true_positives += 1
        elif flagged:
            false_positives += 1
        elif actual:
            false_negatives += 1
        else:
            true_negatives += 1
    return ConfusionMatrix(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def _f1(precision: Rate, recall: Rate) -> float | None:
    """Return the harmonic mean of precision and recall, or ``None``.

    Undefined when either component is, and when both are defined and zero:
    ``2PR/(P+R)`` has an empty denominator there, and reporting ``0.0`` would
    be a division nobody performed.
    """
    if precision.value is None or recall.value is None:
        return None
    total = precision.value + recall.value
    if total <= 0.0:
        return None
    return quantize(2.0 * precision.value * recall.value / total)


def _balanced_accuracy(recall: Rate, false_positive_rate: Rate) -> float | None:
    """Return the mean of sensitivity and specificity, or ``None``.

    Needs both classes present. A "balanced" accuracy over one class is the
    accuracy on that class, and calling it balanced would be a claim about a
    balance that was not measured.
    """
    if recall.value is None or false_positive_rate.value is None:
        return None
    return quantize((recall.value + (1.0 - false_positive_rate.value)) / 2.0)


def binary_metrics(
    *,
    flags: Sequence[bool],
    malicious: Sequence[bool],
    scores: Sequence[float] | None = None,
    probabilities: Sequence[float] | None = None,
    score_unavailable_reason: str | None = None,
    min_positive_rows: int = 1,
    min_benign_rows: int = 1,
    confidence: float = 0.95,
) -> BinaryTestMetrics:
    """Return one system's binary TEST metrics.

    Args:
        flags: the frozen decisions, one per evaluated row.
        malicious: the outcomes, aligned with *flags*.
        scores: a continuous discrimination score per row, when the system has
            one. Omit for a system that does not, and give
            *score_unavailable_reason* instead.
        probabilities: calibrated probabilities, when the system emits them.
            Brier and ECE are computed only from these.
        score_unavailable_reason: why PR-AUC is unavailable, required when
            *scores* is omitted.
        min_positive_rows: support floor below which the metrics are reported
            as ``INSUFFICIENT_SUPPORT`` rather than as measurements.
        min_benign_rows: the same floor for the benign class.
        confidence: the confidence level of the published Wilson intervals.

    Raises:
        DataValidationError: on misaligned inputs, or on an omitted score with
            no stated reason.
    """
    matrix = confusion_matrix(flags, malicious)
    precision = rate(matrix.true_positives, matrix.flagged_count, confidence=confidence)
    recall = rate(matrix.true_positives, matrix.positive_count, confidence=confidence)
    false_positive_rate = rate(
        matrix.false_positives, matrix.negative_count, confidence=confidence
    )
    accuracy = rate(
        matrix.true_positives + matrix.true_negatives,
        matrix.row_count,
        confidence=confidence,
    )

    area, levels, reason = _discrimination(
        scores=scores,
        malicious=malicious,
        score_unavailable_reason=score_unavailable_reason,
    )
    support = (
        MetricStatus.MEASURED
        if matrix.positive_count >= min_positive_rows
        and matrix.negative_count >= min_benign_rows
        else MetricStatus.INSUFFICIENT_SUPPORT
    )
    return BinaryTestMetrics.seal(
        confusion=matrix,
        precision=precision,
        recall=recall,
        false_positive_rate=false_positive_rate,
        f1=_f1(precision, recall),
        balanced_accuracy=_balanced_accuracy(recall, false_positive_rate),
        accuracy=accuracy,
        pr_auc=area,
        pr_auc_unavailable_reason=reason,
        distinct_score_count=None if levels is None else len(levels),
        calibration=(
            None
            if probabilities is None
            else calibration_metrics(probabilities=probabilities, malicious=malicious)
        ),
        row_count=matrix.row_count,
        positive_count=matrix.positive_count,
        negative_count=matrix.negative_count,
        flagged_count=matrix.flagged_count,
        support_status=support,
    )


def _discrimination(
    *,
    scores: Sequence[float] | None,
    malicious: Sequence[bool],
    score_unavailable_reason: str | None,
) -> tuple[float | None, tuple[ScoreLevel, ...] | None, str | None]:
    """Return the exact PR-AUC and its curve, or the reason there is none."""
    if scores is None:
        if not score_unavailable_reason:
            raise DataValidationError(
                "a system with no continuous discrimination score must state why; "
                "an unexplained absent metric is indistinguishable from a bug"
            )
        return (None, None, score_unavailable_reason)
    if len(scores) != len(malicious):
        raise DataValidationError(
            f"{len(scores)} score(s) cannot be ranked against "
            f"{len(malicious)} outcome(s)"
        )
    levels = score_levels(scores, malicious)
    if levels is None:
        return (
            None,
            None,
            "one_class_only: precision-recall is undefined when the evaluated "
            "population carries a single class",
        )
    positive_total = sum(1 for flag in malicious if flag)
    return (pr_auc(levels, positive_count=positive_total), levels, None)


def calibration_metrics(
    *, probabilities: Sequence[float], malicious: Sequence[bool]
) -> CalibrationTestMetrics:
    """Return Brier, expected calibration error, and the reliability buckets.

    Defined only for calibrated probabilities. Handing this function a raw
    decision score or an ordinal risk magnitude would produce a number whose
    units are meaningless, so the caller is responsible for supplying only what
    a verified calibrator produced.

    Raises:
        DataValidationError: on misaligned inputs, an empty population, or a
            probability outside the unit interval.
    """
    if len(probabilities) != len(malicious):
        raise DataValidationError(
            f"{len(probabilities)} probability value(s) cannot be scored against "
            f"{len(malicious)} outcome(s)"
        )
    if not probabilities:
        raise DataValidationError("calibration metrics need at least one row")
    for value in probabilities:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DataValidationError(
                "a calibrated probability outside [0, 1] is not a probability"
            )

    brier = quantize(
        sum(
            (value - (1.0 if outcome else 0.0)) ** 2
            for value, outcome in zip(probabilities, malicious, strict=True)
        )
        / len(probabilities)
    )

    width = 1.0 / RELIABILITY_BUCKET_COUNT
    grouped: list[list[tuple[float, bool]]] = [
        [] for _ in range(RELIABILITY_BUCKET_COUNT)
    ]
    for value, outcome in zip(probabilities, malicious, strict=True):
        # The final bucket is closed at 1.0 so a probability of exactly one has
        # somewhere to go; every other bucket is half-open.
        index = min(int(value / width), RELIABILITY_BUCKET_COUNT - 1)
        grouped[index].append((value, outcome))

    buckets: list[ReliabilityBucket] = []
    error = 0.0
    for index, rows in enumerate(grouped):
        lower = quantize(index * width)
        upper = quantize((index + 1) * width)
        if not rows:
            buckets.append(
                ReliabilityBucket(
                    lower=lower,
                    upper=upper,
                    row_count=0,
                    mean_predicted=None,
                    observed_rate=None,
                )
            )
            continue
        mean_predicted = sum(value for value, _ in rows) / len(rows)
        observed = sum(1 for _, outcome in rows if outcome) / len(rows)
        error += (len(rows) / len(probabilities)) * abs(mean_predicted - observed)
        buckets.append(
            ReliabilityBucket(
                lower=lower,
                upper=upper,
                row_count=len(rows),
                mean_predicted=quantize(mean_predicted),
                observed_rate=quantize(observed),
            )
        )

    return CalibrationTestMetrics(
        brier_score=brier,
        expected_calibration_error=quantize(error),
        bucket_count=RELIABILITY_BUCKET_COUNT,
        buckets=tuple(buckets),
        row_count=len(probabilities),
    )


def metric_definition_fingerprint() -> str:
    """Return the digest of the metric contract this build computes.

    Bound into a TEST receipt so a later change to the arithmetic -- a different
    F1 convention, a different bucket count, a different PR-AUC integration --
    is visible as a different evaluation rather than as the same one with new
    numbers.
    """
    from password_attack_detector.ml.calibration import digest

    payload: dict[str, Any] = {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "discrimination_metric": RANKING_METRIC_NAME,
        "discrimination_integration": PR_AUC_INTEGRATION,
        "reliability_bucket_count": RELIABILITY_BUCKET_COUNT,
        "f1": "harmonic_mean_of_precision_and_recall",
        "balanced_accuracy": "mean_of_sensitivity_and_specificity",
        "empty_denominator": "unavailable",
        "interval": "wilson_score_published_not_applied",
    }
    return digest(payload)
