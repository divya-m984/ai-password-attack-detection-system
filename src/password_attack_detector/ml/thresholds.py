"""Choosing operating points on validation-B, and refusing to when nobody can.

Three thresholds are selected here, and each answers a different question:

============================  ==============  ===============================
threshold                     read from       predicate
============================  ==============  ===============================
binary decision               validation-B    ``score >= threshold``
category abstention           validation-B    ``max(class_score) >= threshold``
anomaly flag                  train **or**    ``anomaly_score <= threshold``
                              validation-A
============================  ==============  ===============================

**Validation-B, not validation-A.**  The calibrator was fitted on validation-A.
Choosing the operating point on the same rows would pick the threshold that
best suits a calibrator those rows already trained, and the resulting
false-positive rate would be a description of that coincidence.  Both halves
come from one campaign-disjoint partitioning, and the provenance chain is
checked rather than assumed.

**Nothing here fits anything.**  The model is frozen, the preprocessor is
frozen, and the calibrator -- if there is one -- is frozen.  This module counts
rows above candidate thresholds and picks one.  It opens no file, reads no
label table, and refits nothing.

**A threshold is not always selectable, and that is an answer.**  Three
outcomes are possible and only one of them is a success.  If validation-B
cannot resolve the configured constraint -- too few benign rows to measure a
1% false-positive ceiling, too few of a category to measure its precision --
the result is ``insufficient_validation_support``.  If the support is adequate
and *no* candidate satisfies the constraint, the result is
``no_feasible_threshold``.  Neither is answered with a best-available number,
because a best-available threshold under a constraint nobody met is a threshold
that quietly violates it.

**Thresholds are observed scores, stored exactly.**  A candidate is a value
some row actually produced, kept at full precision rather than rounded like a
fitted parameter.  Rounding a threshold moves the boundary of a step function,
which changes which rows it flags -- so the number stored is the number
measured, and the number measured is the number a later prediction will apply.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.calibration import (
    BinaryScoreSample,
    CalibrationState,
    ScoreSampleSource,
    SealedModel,
    optional_quantized,
    quantize,
    require_chain,
    require_partition,
    require_train_benign,
)
from password_attack_detector.ml.config import (
    AnomalyConfig,
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
    is_probability,
)
from password_attack_detector.ml.ordering import AnchoredRow, assert_canonical
from password_attack_detector.ml.schemas import (
    Sha256Hex,
    SupportRequirement,
    prohibited_metadata_fields,
)

__all__ = [
    "ANOMALY_DECISION_PREDICATE",
    "BINARY_DECISION_PREDICATE",
    "CATEGORY_DECISION_PREDICATE",
    "THRESHOLD_SCHEMA_VERSION",
    "AnomalyScoreSample",
    "AnomalyThresholdSelection",
    "CategoryAbstentionSelection",
    "CategoryClassSupport",
    "CategoryScoreSample",
    "ThresholdCurvePoint",
    "ThresholdSelection",
    "select_anomaly_threshold",
    "select_binary_threshold",
    "select_category_abstention",
]

#: The threshold-selection contract's own version, separate from the model
#: contract and from the calibration contract.  A model, its calibrator, and its
#: operating point are three identities that change independently.
THRESHOLD_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The binary predicate, written down once and used everywhere.
#:
#: ``>=`` rather than ``>``.  The difference is only ever one row wide, and it
#: is exactly the row sitting on the threshold -- so the choice is stated,
#: stored in every selection, and asserted by a test rather than left to
#: whichever comparison somebody typed at the call site.
BINARY_DECISION_PREDICATE: Final[str] = "score >= threshold"

#: The abstention predicate.  A row whose best class score *equals* the
#: threshold may be assigned that class; a row below it becomes ``unknown``.
CATEGORY_DECISION_PREDICATE: Final[str] = "max(class_score) >= min_category_score"

#: The anomaly predicate, deliberately the other way round.  An anomaly score
#: follows the scikit-learn convention where **lower is more anomalous**, so a
#: flag is ``<=`` and not ``>=``.  Naming it separately rather than reusing the
#: binary predicate is the point: two opposite comparisons sharing one name is
#: how a detector ends up flagging the calmest traffic it can find.
ANOMALY_DECISION_PREDICATE: Final[str] = "anomaly_score <= threshold"


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rate, or ``None`` when its denominator is empty.

    Never ``0.0`` for an empty denominator.  "No benign rows were flagged out
    of none" and "no benign rows were flagged out of nine thousand" are
    different facts, and only one of them is a false-positive rate.
    """
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    """Return the harmonic mean of *precision* and *recall*, where defined."""
    if precision is None or recall is None:
        return None
    total = precision + recall
    if total <= 0.0:
        return None
    return 2.0 * precision * recall / total


def _candidates(values: Sequence[float], *, grid_size: int) -> tuple[float, ...]:
    """Return the deterministic ascending candidate thresholds for *values*.

    Candidates are exactly the **observed** scores.  Two consequences worth
    stating:

    * every candidate flags at least one row, so a "flag nothing" threshold is
      never returned -- a detector that never fires is not an operating point,
      and admitting it would make ``no_feasible_threshold`` unreachable under
      any false-positive ceiling;
    * a fixed arithmetic grid is not used, because a grid point between two
      observed scores produces the same confusion matrix as the observed score
      above it while reporting a threshold no row ever justified.

    When the distinct scores outnumber the configured grid, an evenly spaced
    subset is taken -- including both extremes -- and the caller records that
    the search was bounded.  A silent cap would let a report read as though it
    had considered every operating point.
    """
    unique = sorted(set(values))
    if len(unique) <= grid_size:
        return tuple(unique)
    last = len(unique) - 1
    step = last / (grid_size - 1)
    picked = sorted({min(last, int(index * step)) for index in range(grid_size)})
    picked = sorted(set(picked) | {0, last})
    return tuple(unique[index] for index in picked)


class ThresholdCurvePoint(BaseModel):
    """One candidate threshold's confusion matrix and the rates it implies.

    Aggregate counts only: no event identifier, no row, and one entry per
    semantic threshold rather than per event.  A rate whose denominator is
    empty is ``None`` rather than zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    f1: float | None

    @model_validator(mode="after")
    def check_point(self) -> Self:
        """Every number is finite, and the rates agree with the counts."""
        if not math.isfinite(self.threshold):
            raise ValueError("a curve point must name a finite threshold")
        for name in ("precision", "recall", "false_positive_rate", "f1"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"curve point {name!r} must be finite or absent")
        positives = self.true_positives + self.false_negatives
        negatives = self.false_positives + self.true_negatives
        if (self.recall is None) != (positives == 0):
            raise ValueError("recall exists exactly when there are positive rows")
        if (self.false_positive_rate is None) != (negatives == 0):
            raise ValueError(
                "a false-positive rate exists exactly when there are benign rows"
            )
        flagged = self.true_positives + self.false_positives
        if (self.precision is None) != (flagged == 0):
            raise ValueError("precision exists exactly when something was flagged")
        return self


class ThresholdSelection(SealedModel):
    """The chosen binary operating point, or the reason there is none.

    Carries the numerators and the denominators beside every rate, so a reader
    can recompute the rate rather than trust it -- and can see that a 0%
    false-positive rate over eleven benign rows is not the same claim as a 0%
    false-positive rate over eleven thousand.

    Records no test metric of any kind.  A test evaluation is a separate,
    later, once-only record, never an amendment to an operating point chosen
    before the test split was opened.
    """

    fingerprint_field: ClassVar[str] = "selection_fingerprint"
    schema_version_field: ClassVar[str] = "threshold_schema_version"
    schema_version: ClassVar[str] = THRESHOLD_SCHEMA_VERSION
    record_label: ClassVar[str] = "threshold selection"

    threshold_schema_version: str = THRESHOLD_SCHEMA_VERSION
    status: SelectionStatus
    objective: ThresholdObjective
    score_kind: ScoreKind
    decision_predicate: str = BINARY_DECISION_PREDICATE
    tie_break: str

    selected_threshold: float | None
    objective_value: float | None

    row_count: int = Field(ge=0)
    benign_row_count: int = Field(ge=0)
    benign_flagged_count: int | None
    malicious_row_count: int = Field(ge=0)
    malicious_flagged_count: int | None
    false_positive_rate: float | None
    detection_rate: float | None
    precision: float | None
    f1: float | None

    support_status: MetricStatus
    failing_requirements: tuple[str, ...]
    max_false_positive_rate: float
    min_detection_rate: float
    min_validation_positive_rows: int = Field(ge=1)
    min_validation_benign_rows: int = Field(ge=1)

    distinct_score_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    candidates_truncated: bool
    curve: tuple[ThresholdCurvePoint, ...]

    source_partition: ValidationPartition
    source_partition_fingerprint: Sha256Hex
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    calibration_state_fingerprint: Sha256Hex | None
    calibration_method: CalibrationMethod
    ml_config_fingerprint: Sha256Hex
    data_selected: bool
    selection_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_selection(self) -> Self:
        """A selection is complete or absent; there is nothing in between."""
        if self.source_partition is not ValidationPartition.VALIDATION_B:
            raise ValueError(
                "a binary operating point is chosen on validation-B; "
                "validation-A fits the calibrator and must not also choose the "
                "threshold it will be judged at"
            )
        if self.decision_predicate != BINARY_DECISION_PREDICATE:
            raise ValueError(
                f"the binary predicate is {BINARY_DECISION_PREDICATE!r}; a "
                f"selection measured under a different comparison would flag a "
                f"different set of rows than the one it reports"
            )
        selected = self.status is SelectionStatus.SELECTED
        if self.data_selected != selected:
            raise ValueError(
                "data_selected is true exactly when a threshold was chosen from "
                "validation data"
            )
        required = (
            self.selected_threshold,
            self.objective_value,
            self.false_positive_rate,
            self.detection_rate,
            self.precision,
            self.benign_flagged_count,
            self.malicious_flagged_count,
        )
        if selected and any(value is None for value in required):
            raise ValueError(
                "a selected threshold reports its operating point in full, "
                "numerators and denominators included"
            )
        if not selected and any(value is not None for value in required):
            raise ValueError(
                f"status {str(self.status)!r} reports an operating point; only "
                f"a selected threshold has one"
            )
        if selected and self.failing_requirements:
            raise ValueError("a selected threshold names no failing requirement")
        if selected and self.support_status is not MetricStatus.MEASURED:
            raise ValueError(
                "a threshold selected from unmeasurable support is not one"
            )
        if self.row_count != self.benign_row_count + self.malicious_row_count:
            raise ValueError("class support does not sum to the row count")
        if is_probability(self.score_kind) != (
            self.calibration_state_fingerprint is not None
        ):
            raise ValueError(
                "a calibrated probability names the calibrator that produced "
                "it, and nothing else may"
            )
        if is_probability(self.score_kind) != (
            self.calibration_method is not CalibrationMethod.NONE
        ):
            raise ValueError(
                "the calibration method and the score kind must agree about "
                "whether a calibrator was applied"
            )
        for value in (self.selected_threshold, self.objective_value):
            if value is not None and not math.isfinite(value):
                raise ValueError("a selection must report finite numbers")
        thresholds = [point.threshold for point in self.curve]
        if thresholds != sorted(thresholds):
            raise ValueError("curve points must be given in ascending threshold order")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("curve points must be unique by semantic threshold")
        return self


def select_binary_threshold(
    sample: BinaryScoreSample,
    *,
    config: ThresholdConfig,
    support: SupportRequirement,
    ml_config_fingerprint: str,
    calibration: CalibrationState | None = None,
) -> ThresholdSelection:
    """Choose the binary decision threshold on validation-B.

    Args:
        sample: validation-B scores and binary labels, canonically ordered.
            Either the frozen decision score or a calibrated probability; the
            sample says which, and a calibrated one must name its calibrator.
        config: the objective, the constraint, and the tie-break.
        support: the minimum class support validation-B must carry.
        ml_config_fingerprint: the configuration this run is carried out under.
        calibration: the calibrator whose probabilities *sample* carries, when
            it carries any.  Supplied so the provenance chain can be checked
            rather than assumed, and refused when it does not match.

    Returns:
        A selection whose ``status`` is ``SELECTED``,
        ``INSUFFICIENT_VALIDATION_SUPPORT``, or ``NO_FEASIBLE_THRESHOLD``.  Only
        the first carries an operating point, and only the first sets
        ``data_selected``.

    Raises:
        ModelTrainingError: on rows from anywhere but validation-B, uncanonical
            rows, or a calibrator that does not belong to this provenance chain.
            These are contract violations, not data outcomes.
    """
    stage = "binary threshold selection"
    require_partition(
        sample.source, expected=ValidationPartition.VALIDATION_B, stage=stage
    )
    sample.require_canonical(stage=stage)
    _require_calibration_agreement(sample, calibration, stage=stage)
    if calibration is not None:
        _require_selection_chain(
            stage=stage,
            sample_model=sample.model_content_fingerprint,
            sample_preprocessor=sample.preprocessor_fingerprint,
            sample_source=sample.source.source_fingerprint,
            ml_config_fingerprint=ml_config_fingerprint,
            calibration=calibration,
        )

    malicious = sorted(
        score
        for score, label in zip(sample.scores, sample.malicious, strict=True)
        if label
    )
    benign = sorted(
        score
        for score, label in zip(sample.scores, sample.malicious, strict=True)
        if not label
    )
    failures = _binary_support_failures(
        positives=len(malicious), negatives=len(benign), config=config, support=support
    )

    provenance: dict[str, Any] = {
        "objective": config.objective,
        "score_kind": sample.score_kind,
        "tie_break": config.tie_break,
        "row_count": sample.row_count,
        "benign_row_count": len(benign),
        "malicious_row_count": len(malicious),
        "max_false_positive_rate": quantize(config.max_false_positive_rate),
        "min_detection_rate": quantize(config.min_detection_rate),
        "min_validation_positive_rows": support.min_validation_positive_rows,
        "min_validation_benign_rows": support.min_validation_benign_rows,
        "distinct_score_count": sample.distinct_score_count,
        "source_partition": ValidationPartition.VALIDATION_B,
        "source_partition_fingerprint": sample.source.source_fingerprint,
        "model_id": sample.model_id,
        "model_content_fingerprint": sample.model_content_fingerprint,
        "preprocessor_fingerprint": sample.preprocessor_fingerprint,
        "calibration_state_fingerprint": sample.calibration_state_fingerprint,
        "calibration_method": sample.calibration_method,
        "ml_config_fingerprint": ml_config_fingerprint,
    }

    def unselected(
        status: SelectionStatus,
        failing: tuple[str, ...],
        *,
        candidate_count: int,
        truncated: bool,
        curve: tuple[ThresholdCurvePoint, ...],
    ) -> ThresholdSelection:
        """Return a selection that chose nothing, and says why."""
        return ThresholdSelection.seal(
            status=status,
            selected_threshold=None,
            objective_value=None,
            benign_flagged_count=None,
            malicious_flagged_count=None,
            false_positive_rate=None,
            detection_rate=None,
            precision=None,
            f1=None,
            support_status=(
                MetricStatus.INSUFFICIENT_SUPPORT
                if status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
                else MetricStatus.MEASURED
            ),
            failing_requirements=failing,
            candidate_count=candidate_count,
            candidates_truncated=truncated,
            curve=curve,
            data_selected=False,
            **provenance,
        )

    if failures:
        # No curve is emitted here, deliberately. Per-threshold rates over
        # support the layer has just declared unusable are exactly the numbers
        # somebody would go on to plot.
        return unselected(
            SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT,
            tuple(failures),
            candidate_count=0,
            truncated=False,
            curve=(),
        )

    candidates = _candidates(sample.scores, grid_size=config.search_grid_size)
    truncated = len(candidates) < sample.distinct_score_count
    curve = tuple(
        _curve_point(threshold, malicious=malicious, benign=benign)
        for threshold in candidates
    )

    best = _best_candidate(curve, config=config)
    if best is None:
        return unselected(
            SelectionStatus.NO_FEASIBLE_THRESHOLD,
            (_infeasible_requirement(config),),
            candidate_count=len(candidates),
            truncated=truncated,
            curve=curve,
        )

    point, value = best
    return ThresholdSelection.seal(
        status=SelectionStatus.SELECTED,
        selected_threshold=point.threshold,
        objective_value=value,
        benign_flagged_count=point.false_positives,
        malicious_flagged_count=point.true_positives,
        false_positive_rate=point.false_positive_rate,
        detection_rate=point.recall,
        precision=point.precision,
        f1=point.f1,
        support_status=MetricStatus.MEASURED,
        failing_requirements=(),
        candidate_count=len(candidates),
        candidates_truncated=truncated,
        curve=curve,
        data_selected=True,
        **provenance,
    )


def _require_calibration_agreement(
    sample: BinaryScoreSample,
    calibration: CalibrationState | None,
    *,
    stage: str,
) -> None:
    """Raise unless the sample and the supplied calibrator describe each other.

    Four states, two of them legitimate: an uncalibrated sample with no
    calibrator, and a calibrated sample with the calibrator that produced it.
    The other two are refused, because each would let a threshold be chosen on
    numbers whose origin nobody could name.
    """
    calibrated = is_probability(sample.score_kind)
    if calibrated and calibration is None:
        raise ModelTrainingError(
            f"{stage} was handed calibrated probabilities without the "
            f"calibrator that produced them; the operating point would be "
            f"recorded against a calibrator nobody named"
        )
    if not calibrated and calibration is not None:
        raise ModelTrainingError(
            f"{stage} was handed a calibrator for uncalibrated scores; "
            f"recording one would claim a transformation that was not applied"
        )
    if (
        calibrated
        and calibration is not None
        and sample.calibration_state_fingerprint
        != calibration.calibration_state_fingerprint
    ):
        raise ModelTrainingError(
            f"{stage} was handed a different calibrator from the one that "
            f"produced these probabilities"
        )


def _require_selection_chain(
    *,
    stage: str,
    sample_model: str,
    sample_preprocessor: str,
    sample_source: str,
    ml_config_fingerprint: str,
    calibration: CalibrationState,
) -> None:
    """Raise unless the calibrator and this partition share one provenance chain.

    The validation-A / validation-B link is the interesting one: both halves
    come from a single campaign-disjoint partitioning and therefore carry the
    same parent digest, so a calibrator fitted on one dataset's validation-A
    cannot be combined with another dataset's validation-B.
    """
    require_chain(
        stage=stage,
        model_content_fingerprint=sample_model,
        preprocessor_fingerprint=sample_preprocessor,
        source_fingerprint=sample_source,
        ml_config_fingerprint=ml_config_fingerprint,
        state=calibration,
    )


def _binary_support_failures(
    *,
    positives: int,
    negatives: int,
    config: ThresholdConfig,
    support: SupportRequirement,
) -> list[str]:
    """Return the stable requirement codes validation-B fails.

    The resolution check is the one worth explaining.  With ``N`` benign rows
    the smallest non-zero false-positive rate observable is ``1 / N``.  If the
    configured ceiling sits below that, the only way to satisfy it is to flag
    no benign row at all -- so the ceiling has not been *held*, it has merely
    not been *tested*, and reporting a threshold under it would be reporting a
    constraint the data could never have checked.
    """
    failures: list[str] = []
    if positives < support.min_validation_positive_rows:
        failures.append("min_validation_positive_rows")
    if negatives < support.min_validation_benign_rows:
        failures.append("min_validation_benign_rows")
    if (
        config.objective is ThresholdObjective.MAX_RECALL_AT_MAX_FPR
        and negatives * config.max_false_positive_rate < 1.0
    ):
        failures.append("false_positive_rate_resolution")
    return failures


def _curve_point(
    threshold: float, *, malicious: Sequence[float], benign: Sequence[float]
) -> ThresholdCurvePoint:
    """Return the confusion matrix at *threshold* under ``score >= threshold``.

    Counted with a binary search over the sorted score vectors rather than a
    scan per candidate: the result is identical and the search stays linear in
    the number of candidates rather than quadratic in the number of rows.
    """
    true_positives = len(malicious) - bisect_left(malicious, threshold)
    false_positives = len(benign) - bisect_left(benign, threshold)
    false_negatives = len(malicious) - true_positives
    true_negatives = len(benign) - false_positives

    precision = _rate(true_positives, true_positives + false_positives)
    recall = _rate(true_positives, len(malicious))
    false_positive_rate = _rate(false_positives, len(benign))
    return ThresholdCurvePoint(
        threshold=threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=optional_quantized(precision),
        recall=optional_quantized(recall),
        false_positive_rate=optional_quantized(false_positive_rate),
        f1=optional_quantized(_f1(precision, recall)),
    )


def _best_candidate(
    curve: Sequence[ThresholdCurvePoint], *, config: ThresholdConfig
) -> tuple[ThresholdCurvePoint, float] | None:
    """Return the feasible candidate that best satisfies the objective.

    Candidates are visited in ascending threshold order, which is what makes
    the tie-break meaningful: ``lowest_threshold`` keeps the first candidate
    that attains the best objective value and ``highest_threshold`` keeps the
    last, and every comparison is made on the same quantized number the
    selection goes on to report.  Comparing on one number and reporting another
    is how a tie becomes irreproducible.
    """
    prefer_high = config.objective is not ThresholdObjective.MIN_FPR_AT_MIN_RECALL
    keep_last = config.tie_break == "highest_threshold"
    best: tuple[ThresholdCurvePoint, float] | None = None

    for point in curve:
        value = _objective_value(point, config=config)
        if value is None:
            continue
        if best is None:
            best = (point, value)
            continue
        incumbent = best[1]
        better = value > incumbent if prefer_high else value < incumbent
        if better or (value == incumbent and keep_last):
            best = (point, value)
    return best


def _objective_value(
    point: ThresholdCurvePoint, *, config: ThresholdConfig
) -> float | None:
    """Return the objective at *point*, or ``None`` when it is not feasible.

    Feasibility and objective are decided together on purpose.  A constraint
    evaluated separately from the quantity it constrains is a constraint
    somebody eventually forgets to apply.
    """
    match config.objective:
        case ThresholdObjective.MAX_RECALL_AT_MAX_FPR:
            if (
                point.false_positive_rate is None
                or point.false_positive_rate > config.max_false_positive_rate
                or point.recall is None
            ):
                return None
            return point.recall
        case ThresholdObjective.MAX_F1:
            return point.f1
        case ThresholdObjective.MIN_FPR_AT_MIN_RECALL:
            if (
                point.recall is None
                or point.recall < config.min_detection_rate
                or point.false_positive_rate is None
            ):
                return None
            return point.false_positive_rate


def _infeasible_requirement(config: ThresholdConfig) -> str:
    """Return the stable code naming the constraint no candidate satisfied."""
    match config.objective:
        case ThresholdObjective.MAX_RECALL_AT_MAX_FPR:
            return "max_false_positive_rate"
        case ThresholdObjective.MAX_F1:
            return "f1_undefined_at_every_candidate"
        case ThresholdObjective.MIN_FPR_AT_MIN_RECALL:
            return "min_detection_rate"


# ---------------------------------------------------------------------------
# Category abstention
# ---------------------------------------------------------------------------


class CategoryScoreSample(BaseModel):
    """Per-class scores for **known-malicious** validation-B rows.

    Three things it must not carry, each refused rather than filtered:

    * benign rows -- the abstention threshold trades coverage against
      precision *among attacks*, and benign rows are the binary head's
      business;
    * a row whose true category is outside the declared class order -- a novel
      pattern has no known category to be right or wrong about, and letting one
      in would let the holdout influence a threshold it exists to test;
    * a class order that is not the deterministic one, since every count below
      is reported positionally.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    source: ScoreSampleSource
    anchors: tuple[AnchoredRow, ...]
    class_order: tuple[str, ...]
    class_scores: tuple[tuple[float, ...], ...]
    true_category: tuple[str, ...]
    malicious: tuple[bool, ...]
    score_kind: ScoreKind
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_sample(self) -> Self:
        """Every parallel sequence describes the same known-malicious rows."""
        if self.score_kind is not ScoreKind.CLASS_SCORE:
            raise ValueError(
                f"the category head emits a class score, not {str(self.score_kind)!r}"
            )
        if len(self.class_order) < 2:
            raise ValueError("a category head needs at least two known classes")
        if len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order repeats a class")
        if tuple(sorted(self.class_order)) != self.class_order:
            raise ValueError(
                "class_order must be the deterministic sorted order; every "
                "count in a selection is reported positionally against it"
            )
        if UNKNOWN_CATEGORY in self.class_order:
            raise ValueError(
                f"{UNKNOWN_CATEGORY!r} is the abstention outcome, not a known "
                f"class the head may be fitted to predict"
            )
        if not self.class_scores:
            raise ValueError("a category sample must carry at least one row")
        if len(self.true_category) != len(self.class_scores):
            raise ValueError("categories and score rows disagree in length")
        if len(self.malicious) != len(self.class_scores):
            raise ValueError("labels and score rows disagree in length")
        if len(self.anchors) != len(self.class_scores):
            raise ValueError("anchors and score rows disagree in length")
        if not all(self.malicious):
            raise ValueError(
                "the abstention threshold is chosen on known-malicious rows "
                "only; a benign row has no known category to be right about"
            )
        for row in self.class_scores:
            if len(row) != len(self.class_order):
                raise ValueError("every score row must be as wide as the class order")
            for value in row:
                if not math.isfinite(value):
                    raise ValueError("a class score must be finite")
        unknown = sorted(set(self.true_category) - set(self.class_order))
        if unknown:
            raise ValueError(
                f"{len(unknown)} row(s) carry a category outside the declared "
                f"class order; a row with no known category cannot measure a "
                f"known-category precision"
            )
        return self

    def require_canonical(self, *, stage: str) -> None:
        """Assert the rows are canonically ordered, or raise."""
        assert_canonical(self.anchors, stage=stage)

    @property
    def row_count(self) -> int:
        """Return the number of known-malicious rows."""
        return len(self.class_scores)

    def predictions(self) -> tuple[tuple[float, str], ...]:
        """Return each row's best class score and the class that attained it.

        Ties go to the **earliest class in the declared order**.  An arbitrary
        tie-break would make the reported precision depend on which class
        happened to be enumerated first by whichever library produced the
        matrix, and the whole point of a declared order is that nothing else
        gets to decide.
        """
        resolved: list[tuple[float, str]] = []
        for row in self.class_scores:
            best_index = 0
            for index in range(1, len(row)):
                if row[index] > row[best_index]:
                    best_index = index
            resolved.append((row[best_index], self.class_order[best_index]))
        return tuple(resolved)


class CategoryClassSupport(BaseModel):
    """One known class's validation-B support, and its share of the outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_name: str
    row_count: int = Field(ge=0)
    #: Rows of this class the operating point covered, and how many of those it
    #: got right.  Both absent unless a threshold was actually selected: at a
    #: fallback threshold there is no measurement, only a reviewed constant.
    covered_count: int | None = None
    correct_count: int | None = None

    @model_validator(mode="after")
    def check_support(self) -> Self:
        """Coverage and correctness are present together, and are consistent."""
        if (self.covered_count is None) != (self.correct_count is None):
            raise ValueError("coverage and correctness are reported together")
        if self.covered_count is None or self.correct_count is None:
            return self
        if self.covered_count > self.row_count:
            raise ValueError("a class cannot cover more rows than it has")
        if self.correct_count > self.covered_count:
            raise ValueError("a class cannot be right about rows it did not cover")
        return self


class CategoryAbstentionSelection(SealedModel):
    """The abstention threshold, and whether validation-B chose it.

    ``data_selected`` is the field that matters here.  When validation-B cannot
    support a selection, this record still carries a usable threshold -- the
    reviewed conservative constant from configuration -- and marks it as *not*
    selected from data.  The two are permanently distinguishable, so a later
    report can never describe a predeclared fallback as a measurement.
    """

    fingerprint_field: ClassVar[str] = "selection_fingerprint"
    schema_version_field: ClassVar[str] = "threshold_schema_version"
    schema_version: ClassVar[str] = THRESHOLD_SCHEMA_VERSION
    record_label: ClassVar[str] = "category abstention selection"

    threshold_schema_version: str = THRESHOLD_SCHEMA_VERSION
    status: SelectionStatus
    objective: str
    decision_predicate: str = CATEGORY_DECISION_PREDICATE
    tie_break: str
    abstain_label: str

    min_category_score: float
    data_selected: bool

    coverage: float | None
    known_category_precision: float | None
    known_category_error: float | None
    covered_count: int | None
    correct_count: int | None

    known_malicious_row_count: int = Field(ge=0)
    class_order: tuple[str, ...]
    class_support: tuple[CategoryClassSupport, ...]

    min_known_category_precision: float
    min_rows_per_category: int = Field(ge=1)
    min_known_malicious_rows: int = Field(ge=1)
    support_status: MetricStatus
    failing_requirements: tuple[str, ...]

    distinct_score_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    candidates_truncated: bool

    source_partition: ValidationPartition
    source_partition_fingerprint: Sha256Hex
    category_model_id: str
    category_model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    selection_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_selection(self) -> Self:
        """A measured operating point exists exactly when one was selected."""
        if self.source_partition is not ValidationPartition.VALIDATION_B:
            raise ValueError("the abstention threshold is chosen on validation-B")
        if self.decision_predicate != CATEGORY_DECISION_PREDICATE:
            raise ValueError(
                f"the abstention predicate is {CATEGORY_DECISION_PREDICATE!r}"
            )
        if self.abstain_label != UNKNOWN_CATEGORY:
            raise ValueError(f"the abstention label is {UNKNOWN_CATEGORY!r}")
        if not math.isfinite(self.min_category_score):
            raise ValueError("the abstention threshold must be finite")
        selected = self.status is SelectionStatus.SELECTED
        if self.data_selected != selected:
            raise ValueError(
                "data_selected is true exactly when validation-B chose the "
                "threshold; a predeclared fallback is never a selection"
            )
        measured = (
            self.coverage,
            self.known_category_precision,
            self.known_category_error,
            self.covered_count,
            self.correct_count,
        )
        if selected and any(value is None for value in measured):
            raise ValueError("a selected threshold reports its coverage and precision")
        if not selected and any(value is not None for value in measured):
            raise ValueError(
                "a fallback threshold reports no measurement; there was none"
            )
        if selected and self.failing_requirements:
            raise ValueError("a selected threshold names no failing requirement")
        if tuple(item.class_name for item in self.class_support) != self.class_order:
            raise ValueError("class support must be given in class order")
        if sum(item.row_count for item in self.class_support) != (
            self.known_malicious_row_count
        ):
            raise ValueError("per-class support does not account for every row")
        if selected and (
            self.known_category_precision is None
            or self.known_category_precision < self.min_known_category_precision
        ):
            raise ValueError(
                "a selected threshold clears the configured precision floor; a "
                "threshold that does not is not feasible"
            )
        return self


def select_category_abstention(
    sample: CategoryScoreSample,
    *,
    config: CategoryConfig,
    support: SupportRequirement,
    ml_config_fingerprint: str,
) -> CategoryAbstentionSelection:
    """Choose the abstention threshold on known-malicious validation-B rows.

    The objective is the one the configuration declares: take the **widest
    coverage** whose known-category precision still clears the floor.  Coverage
    falls as the threshold rises, but precision does not rise monotonically
    with it, so every candidate is evaluated rather than the search stopping at
    the first feasible one.

    When validation-B cannot support the choice -- too few known-malicious rows
    overall, or a class with too little support for its precision to mean
    anything -- the reviewed constant from configuration is returned with
    ``data_selected: false``.  It is a usable threshold and a permanently
    visible admission that no measurement chose it.

    Raises:
        ModelTrainingError: on rows from anywhere but validation-B, or
            uncanonical rows.
    """
    stage = "category abstention selection"
    require_partition(
        sample.source, expected=ValidationPartition.VALIDATION_B, stage=stage
    )
    sample.require_canonical(stage=stage)

    predictions = sample.predictions()
    best_scores = [score for score, _ in predictions]
    counts = dict.fromkeys(sample.class_order, 0)
    for name in sample.true_category:
        counts[name] += 1

    failures: list[str] = []
    if sample.row_count < config.min_known_malicious_rows:
        failures.append("min_known_malicious_rows")
    thin = sorted(
        name
        for name in sample.class_order
        if counts[name]
        < max(config.min_rows_per_category, support.min_rows_per_category)
    )
    if thin:
        failures.append("min_rows_per_category")

    distinct = len(set(best_scores))
    provenance: dict[str, Any] = {
        "objective": config.abstention_objective,
        "tie_break": config.abstention_tie_break,
        "abstain_label": config.abstain_label,
        "known_malicious_row_count": sample.row_count,
        "class_order": sample.class_order,
        "min_known_category_precision": quantize(config.min_known_category_precision),
        "min_rows_per_category": max(
            config.min_rows_per_category, support.min_rows_per_category
        ),
        "min_known_malicious_rows": config.min_known_malicious_rows,
        "distinct_score_count": distinct,
        "source_partition": ValidationPartition.VALIDATION_B,
        "source_partition_fingerprint": sample.source.source_fingerprint,
        "category_model_id": sample.model_id,
        "category_model_content_fingerprint": sample.model_content_fingerprint,
        "preprocessor_fingerprint": sample.preprocessor_fingerprint,
        "ml_config_fingerprint": ml_config_fingerprint,
    }
    bare_support = tuple(
        CategoryClassSupport(class_name=name, row_count=counts[name])
        for name in sample.class_order
    )

    def fallback(
        status: SelectionStatus,
        failing: tuple[str, ...],
        *,
        candidate_count: int,
        truncated: bool,
    ) -> CategoryAbstentionSelection:
        """Return the reviewed constant, marked as not selected from data."""
        return CategoryAbstentionSelection.seal(
            status=status,
            min_category_score=quantize(config.min_category_score),
            data_selected=False,
            coverage=None,
            known_category_precision=None,
            known_category_error=None,
            covered_count=None,
            correct_count=None,
            class_support=bare_support,
            support_status=(
                MetricStatus.INSUFFICIENT_SUPPORT
                if status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
                else MetricStatus.MEASURED
            ),
            failing_requirements=failing,
            candidate_count=candidate_count,
            candidates_truncated=truncated,
            **provenance,
        )

    if failures:
        return fallback(
            SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT,
            tuple(failures),
            candidate_count=0,
            truncated=False,
        )

    candidates = _candidates(best_scores, grid_size=config.abstention_search_grid_size)
    truncated = len(candidates) < distinct
    best = _best_abstention(
        candidates, predictions=predictions, sample=sample, config=config
    )
    if best is None:
        return fallback(
            SelectionStatus.NO_FEASIBLE_THRESHOLD,
            ("min_known_category_precision",),
            candidate_count=len(candidates),
            truncated=truncated,
        )

    threshold, covered, correct = best
    per_class = _class_outcomes(
        threshold, predictions=predictions, sample=sample, counts=counts
    )
    precision = correct / covered
    return CategoryAbstentionSelection.seal(
        status=SelectionStatus.SELECTED,
        min_category_score=threshold,
        data_selected=True,
        coverage=quantize(covered / sample.row_count),
        known_category_precision=quantize(precision),
        known_category_error=quantize(1.0 - precision),
        covered_count=covered,
        correct_count=correct,
        class_support=per_class,
        support_status=MetricStatus.MEASURED,
        failing_requirements=(),
        candidate_count=len(candidates),
        candidates_truncated=truncated,
        **provenance,
    )


def _best_abstention(
    candidates: Sequence[float],
    *,
    predictions: Sequence[tuple[float, str]],
    sample: CategoryScoreSample,
    config: CategoryConfig,
) -> tuple[float, int, int] | None:
    """Return the widest-coverage feasible threshold, or ``None`` if none is.

    Candidates are visited in ascending order, so ``lowest_threshold`` keeps
    the first threshold attaining the best coverage and ``highest_threshold``
    keeps the last.  Coverage is compared as a count rather than a rate: the
    denominator is the same for every candidate, so the count is the same
    ordering without a division that could make two distinct coverages compare
    equal after rounding.
    """
    keep_last = config.abstention_tie_break == "highest_threshold"
    best: tuple[float, int, int] | None = None
    for threshold in candidates:
        covered = 0
        correct = 0
        for (score, predicted), truth in zip(
            predictions, sample.true_category, strict=True
        ):
            if score >= threshold:
                covered += 1
                if predicted == truth:
                    correct += 1
        if covered == 0:
            continue
        if correct / covered < config.min_known_category_precision:
            continue
        if best is None or covered > best[1] or (covered == best[1] and keep_last):
            best = (threshold, covered, correct)
    return best


def _class_outcomes(
    threshold: float,
    *,
    predictions: Sequence[tuple[float, str]],
    sample: CategoryScoreSample,
    counts: dict[str, int],
) -> tuple[CategoryClassSupport, ...]:
    """Return per-class coverage and correctness at *threshold*, in class order.

    Attributed by the row's **true** class rather than its predicted one, so
    the numbers answer "how much of this attack type does the head cover, and
    how often is it right" rather than "how pure is this prediction bucket".
    """
    covered = dict.fromkeys(sample.class_order, 0)
    correct = dict.fromkeys(sample.class_order, 0)
    for (score, predicted), truth in zip(
        predictions, sample.true_category, strict=True
    ):
        if score >= threshold:
            covered[truth] += 1
            if predicted == truth:
                correct[truth] += 1
    return tuple(
        CategoryClassSupport(
            class_name=name,
            row_count=counts[name],
            covered_count=covered[name],
            correct_count=correct[name],
        )
        for name in sample.class_order
    )


# ---------------------------------------------------------------------------
# Anomaly threshold
# ---------------------------------------------------------------------------


class AnomalyScoreSample(BaseModel):
    """Benign anomaly scores from one permitted source.

    Benign rows only, and the check is behavioural rather than a promise: a
    malicious row in this sample would turn an unsupervised probe into a weakly
    supervised one without anything saying so.  The probe is a generalisation
    measurement, and a measurement that quietly reads the answer is not one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    source: ScoreSampleSource
    anchors: tuple[AnchoredRow, ...]
    scores: tuple[float, ...]
    malicious: tuple[bool, ...]
    score_kind: ScoreKind
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_sample(self) -> Self:
        """The rows are benign, finite, aligned, and carry anomaly scores."""
        if self.score_kind is not ScoreKind.ANOMALY_SCORE:
            raise ValueError(
                f"the anomaly probe emits an anomaly score, not "
                f"{str(self.score_kind)!r}; and an anomaly score is never a "
                f"probability, before or after thresholding"
            )
        if not self.scores:
            raise ValueError("an anomaly sample must carry at least one row")
        if len(self.malicious) != len(self.scores):
            raise ValueError("labels and scores disagree in length")
        if len(self.anchors) != len(self.scores):
            raise ValueError("anchors and scores disagree in length")
        if any(self.malicious):
            raise ValueError(
                "the anomaly threshold is chosen from benign rows only; a "
                "malicious row here would make an unsupervised probe read a "
                "supervised outcome"
            )
        for score in self.scores:
            if not math.isfinite(score):
                raise ValueError("an anomaly score must be finite")
        return self

    def require_canonical(self, *, stage: str) -> None:
        """Assert the rows are canonically ordered, or raise."""
        assert_canonical(self.anchors, stage=stage)

    @property
    def row_count(self) -> int:
        """Return the number of benign rows."""
        return len(self.scores)


class AnomalyThresholdSelection(SealedModel):
    """The experimental probe's flag threshold, and where it came from.

    Never a probability, before or after thresholding, and never a champion
    input: ``influences_champion_selection`` is pinned false here as it is in
    the configuration, so an experimental signal cannot reach a promotion
    decision by way of an operating point.
    """

    fingerprint_field: ClassVar[str] = "selection_fingerprint"
    schema_version_field: ClassVar[str] = "threshold_schema_version"
    schema_version: ClassVar[str] = THRESHOLD_SCHEMA_VERSION
    record_label: ClassVar[str] = "anomaly threshold selection"

    threshold_schema_version: str = THRESHOLD_SCHEMA_VERSION
    status: SelectionStatus
    method: AnomalyThresholdMethod
    score_kind: ScoreKind
    decision_predicate: str = ANOMALY_DECISION_PREDICATE

    threshold: float | None
    target_benign_flag_rate: float
    target_quantile: float | None
    observed_benign_flag_rate: float | None
    benign_row_count: int = Field(ge=0)
    flagged_benign_count: int | None
    min_source_rows: int = Field(ge=1)

    support_status: MetricStatus
    failing_requirements: tuple[str, ...]
    data_selected: bool
    influences_champion_selection: bool = False

    source_split: MLSplit
    source_partition: ValidationPartition | None
    source_fingerprint: Sha256Hex
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    selection_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_selection(self) -> Self:
        """The method, the source, and the reported vocabulary must agree."""
        if self.score_kind is not ScoreKind.ANOMALY_SCORE:
            raise ValueError(
                "an anomaly threshold is chosen against an anomaly score; "
                "nothing here becomes a probability"
            )
        if self.decision_predicate != ANOMALY_DECISION_PREDICATE:
            raise ValueError(
                f"the anomaly predicate is {ANOMALY_DECISION_PREDICATE!r}; a "
                f"lower score is more anomalous, so the comparison is inverted "
                f"relative to the binary head"
            )
        if self.influences_champion_selection:
            raise ValueError(
                "the anomaly probe never influences champion selection; a "
                "measurement that can change what it measures is not one"
            )
        expected_source = {
            AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE: (MLSplit.TRAIN, None),
            AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR: (
                MLSplit.VALIDATION,
                ValidationPartition.VALIDATION_A,
            ),
        }[self.method]
        if (self.source_split, self.source_partition) != expected_source:
            raise ValueError(
                f"method {str(self.method)!r} does not read "
                f"{str(self.source_split)!r} rows"
            )
        if (self.target_quantile is not None) != (
            self.method is AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE
        ):
            raise ValueError(
                "a quantile is recorded exactly by the method that uses one"
            )
        selected = self.status is SelectionStatus.SELECTED
        if self.data_selected != selected:
            raise ValueError(
                "data_selected is true exactly when a threshold was chosen"
            )
        measured = (
            self.threshold,
            self.observed_benign_flag_rate,
            self.flagged_benign_count,
        )
        if selected and any(value is None for value in measured):
            raise ValueError("a selected threshold reports its observed flag rate")
        if not selected and any(value is not None for value in measured):
            raise ValueError("an unselected threshold reports no measurement")
        if selected and self.failing_requirements:
            raise ValueError("a selected threshold names no failing requirement")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("the anomaly threshold must be finite")
        return self


def select_anomaly_threshold(
    sample: AnomalyScoreSample,
    *,
    config: AnomalyConfig,
    support: SupportRequirement,
    ml_config_fingerprint: str,
) -> AnomalyThresholdSelection:
    """Choose the anomaly flag threshold from the configured benign source.

    Two provenance methods, and the configuration picks one:

    ``train_benign_quantile``
        Read benign **training** scores only.  No validation outcome, no
        malicious label, and nothing the holdout reveals.

    ``validation_a_benign_fpr``
        Read benign **validation-A** scores only, holding the benign flag rate
        at or under a configured target.  Validation-A rather than validation-B
        so an experimental signal never reaches the rows that choose the
        supervised operating point.

    Both resolve to the same rule -- the largest threshold whose benign flag
    rate stays at or under the target -- so the two differ in *where they
    read*, which is the only thing that should distinguish them.

    Raises:
        ModelTrainingError: if the rows did not come from the source the
            configured method reads, or are not canonically ordered.
    """
    stage = f"anomaly threshold selection [{config.threshold_method!s}]"
    if config.threshold_method is AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE:
        require_train_benign(sample.source, stage=stage)
        target = 1.0 - config.quantile
        quantile: float | None = config.quantile
        minimum_rows = config.min_fit_rows
    else:
        require_partition(
            sample.source, expected=ValidationPartition.VALIDATION_A, stage=stage
        )
        target = config.target_benign_flag_rate
        quantile = None
        minimum_rows = support.min_validation_benign_rows
    sample.require_canonical(stage=stage)

    provenance: dict[str, Any] = {
        "method": config.threshold_method,
        "score_kind": ScoreKind.ANOMALY_SCORE,
        "target_benign_flag_rate": quantize(target),
        "target_quantile": optional_quantized(quantile),
        "benign_row_count": sample.row_count,
        "min_source_rows": minimum_rows,
        "source_split": sample.source.split,
        "source_partition": sample.source.partition,
        "source_fingerprint": sample.source.source_fingerprint,
        "model_id": sample.model_id,
        "model_content_fingerprint": sample.model_content_fingerprint,
        "preprocessor_fingerprint": sample.preprocessor_fingerprint,
        "ml_config_fingerprint": ml_config_fingerprint,
    }

    def unselected(
        status: SelectionStatus, failing: tuple[str, ...]
    ) -> AnomalyThresholdSelection:
        """Return a selection that chose nothing, and says why."""
        return AnomalyThresholdSelection.seal(
            status=status,
            threshold=None,
            observed_benign_flag_rate=None,
            flagged_benign_count=None,
            support_status=(
                MetricStatus.INSUFFICIENT_SUPPORT
                if status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
                else MetricStatus.MEASURED
            ),
            failing_requirements=failing,
            data_selected=False,
            **provenance,
        )

    if sample.row_count < minimum_rows:
        return unselected(
            SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT, ("min_source_rows",)
        )

    chosen = _largest_threshold_within(sample.scores, target=target)
    if chosen is None:
        # Even the single most anomalous score -- or the whole group tied with
        # it -- already exceeds the target share. The threshold that would
        # satisfy the target flags nothing, and a probe that never fires is not
        # an operating point. The resolution, not the data, is the limit.
        return unselected(
            SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT,
            ("benign_flag_rate_resolution",),
        )

    threshold, flagged = chosen
    return AnomalyThresholdSelection.seal(
        status=SelectionStatus.SELECTED,
        threshold=threshold,
        observed_benign_flag_rate=quantize(flagged / sample.row_count),
        flagged_benign_count=flagged,
        support_status=MetricStatus.MEASURED,
        failing_requirements=(),
        data_selected=True,
        **provenance,
    )


def _largest_threshold_within(
    scores: Sequence[float], *, target: float
) -> tuple[float, int] | None:
    """Return the largest observed score whose flagged share stays within *target*.

    "Flagged" is ``anomaly_score <= threshold``, so the count at a candidate
    includes every row tied with it -- ties are never split, because a
    threshold that flagged some of the rows carrying one score and not others
    would not be a threshold.

    Returns ``None`` when even the smallest observed score already flags more
    than the target allows, which is a statement about resolution rather than
    about the data: with ``N`` rows the coarsest available flag rate is the
    size of the smallest tied group over ``N``.
    """
    ordered = sorted(scores)
    total = len(ordered)
    best: tuple[float, int] | None = None
    index = 0
    while index < total:
        value = ordered[index]
        while index + 1 < total and ordered[index + 1] == value:
            index += 1
        flagged = index + 1
        if flagged / total <= target:
            best = (value, flagged)
        else:
            break
        index += 1
    return best


# ---------------------------------------------------------------------------
# Privacy sweep
# ---------------------------------------------------------------------------

#: Every schema in this module that may be published.  The typed inputs are
#: deliberately absent: choosing an operating point is a supervised operation
#: and its input legitimately carries labels and anchors.  Nothing derived from
#: them may.
_PUBLISHED_SCHEMAS: Final[tuple[type[BaseModel], ...]] = (
    AnomalyThresholdSelection,
    CategoryAbstentionSelection,
    CategoryClassSupport,
    ThresholdCurvePoint,
    ThresholdSelection,
)


def _assert_no_prohibited_fields() -> None:
    """Fail at import if a published schema declares an identity-bearing field."""
    for model in _PUBLISHED_SCHEMAS:
        offending = prohibited_metadata_fields(list(model.model_fields))
        if offending:
            raise ValueError(
                f"{model.__name__} declares prohibited metadata field(s) "
                f"{list(offending)}"
            )


_assert_no_prohibited_fields()
