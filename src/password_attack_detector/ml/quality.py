"""The aggregate profile of a prediction publication, and nothing more.

What this report contains is **the shape of the output**: how many rows were
scored, how many were flagged, where the scores sat, how often the category head
abstained, and whether the artifact validated.  Every number in it is derivable
from the published prediction artifacts alone -- no training data, no labels, no
split assignments, and no access to whatever the model got right.

What it therefore does **not** contain is any measure of whether the predictions
were correct.  There is no accuracy here, no precision or recall against truth,
no F1, no false-positive rate against truth, no PR-AUC, no Brier score, and no
calibration error.  Those all require labels, this milestone never opens any,
and a report carrying a zero where one belongs would read exactly like a
measurement of zero.  A structural guard at import refuses a field named for one.

**Structural validity is not predictive quality.**  A publication can pass every
check in this report while the model behind it is useless: the checks establish
that the rows are internally consistent, correctly typed, canonically ordered,
and attributable to the frozen champion that produced them.  Whether flagging
those particular rows was a good idea is the question the next milestone asks,
once, against the test labels -- and it is a different question, not a stricter
version of this one.

**Unavailable is not zero.**  A quantity nothing could produce is rendered as
``null`` and named as unavailable: an uncalibrated champion has no probability
distribution, and reporting a mean of ``0.0`` for one would describe a model
that was certain every row was benign.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.ml.calibration import SealedModel, quantize
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    AuditStatus,
    MLSplit,
    ScoreKind,
    is_probability,
)
from password_attack_detector.ml.prediction_manifest import (
    PredictionManifest,
    PredictionScopeRole,
)
from password_attack_detector.ml.prediction_validation import MLValidationResult
from password_attack_detector.ml.predictions import (
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
)
from password_attack_detector.ml.schemas import Sha256Hex

__all__ = [
    "QUALITY_SCHEMA_VERSION",
    "REPORTED_QUANTILES",
    "AnomalyDistribution",
    "BinaryDistribution",
    "CategoryClassCount",
    "CategoryDistribution",
    "MLQualityReport",
    "QuantilePoint",
    "build_quality_report",
    "quality_report_to_markdown",
]

#: The quality-report contract's own version.
QUALITY_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The quantiles reported for a score distribution.  A fixed, declared set:
#: choosing them per publication would make two reports incomparable, and
#: choosing them from the data would be a summary shaped by what it summarises.
REPORTED_QUANTILES: Final[tuple[float, ...]] = (0.05, 0.25, 0.50, 0.75, 0.95)


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rate, or ``None`` when its denominator is empty.

    Never ``0.0`` for an empty denominator: "none of no rows" and "none of nine
    thousand rows" are different statements, and only one of them is evidence.
    """
    if denominator <= 0:
        return None
    return quantize(numerator / denominator)


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank quantile of *values*.

    Nearest-rank rather than interpolated: the reported value is one a row
    actually carried, so a reader comparing it against a threshold is comparing
    two numbers of the same kind.
    """
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return quantize(ordered[rank - 1])


class QuantilePoint(BaseModel):
    """One reported quantile of a score distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quantile: float = Field(gt=0.0, lt=1.0)
    value: float


class BinaryDistribution(BaseModel):
    """How the binary head's output was distributed across the scored rows.

    Counts and score statistics.  No outcome appears here, because no outcome
    was read: ``flagged_count`` is how many rows the frozen threshold flagged,
    not how many of them were attacks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_rows: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    flagged_rate: float | None
    unflagged_count: int = Field(ge=0)
    unflagged_rate: float | None

    #: The score kind the frozen threshold was applied to, and its value.
    score_kind: ScoreKind
    decision_threshold: float

    decision_score_available: bool
    decision_score_minimum: float | None
    decision_score_maximum: float | None
    decision_score_mean: float | None
    decision_score_quantiles: tuple[QuantilePoint, ...] | None

    #: ``False`` when the champion has no calibrator.  The three statistics
    #: below are then ``None`` -- unavailable, not zero.
    calibrated_probability_available: bool
    probability_minimum: float | None
    probability_maximum: float | None
    probability_mean: float | None
    probability_quantiles: tuple[QuantilePoint, ...] | None
    #: Rows carrying no probability.  Equal to ``total_rows`` when no calibrator
    #: exists, and zero when one does: a partially populated probability column
    #: would mean the calibrator failed on some rows and nothing said so.
    null_probability_count: int = Field(ge=0)

    @model_validator(mode="after")
    def check_distribution(self) -> Self:
        """Counts sum, and an unavailable statistic is absent rather than zero."""
        if self.flagged_count + self.unflagged_count != self.total_rows:
            raise ValueError("flagged and unflagged counts do not sum to the total")
        probability_fields = (
            self.probability_minimum,
            self.probability_maximum,
            self.probability_mean,
            self.probability_quantiles,
        )
        if not self.calibrated_probability_available and any(
            value is not None for value in probability_fields
        ):
            raise ValueError(
                "a publication with no calibrator reports no probability "
                "statistic; an absent distribution is not a distribution of zeros"
            )
        if self.calibrated_probability_available:
            if self.null_probability_count:
                raise ValueError(
                    "a calibrated publication carries a probability on every row"
                )
            if is_probability(self.score_kind) is False:
                raise ValueError(
                    "a calibrated publication was decided on its calibrated "
                    "probability; a raw-score decision beside a probability "
                    "column is two operating points"
                )
        elif self.null_probability_count != self.total_rows:
            raise ValueError(
                "an uncalibrated publication carries no probability on any row"
            )
        return self


class CategoryClassCount(BaseModel):
    """How many rows one declared class was assigned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_name: str
    predicted_count: int = Field(ge=0)


class CategoryDistribution(BaseModel):
    """How often the category head committed, and to what.

    **Every rate here is over the category-applicable population**, which is the
    set of rows the binary champion flagged -- not the whole prediction table.
    The category head was fitted on known-malicious rows only, so a benign row
    is not a row it declined to classify; it is a row it was never asked about.
    Dividing by the whole table would dilute an abstention rate with rows that
    never reached the head, and would make a mostly-benign dataset read as a
    triage model that constantly refuses to commit.

    Both populations are reported, so the distinction is legible rather than
    implied:

    * ``not_applicable_count`` -- scored rows the binary head did not flag.
      Nothing was asked of the category head about them.
    * ``unknown_count`` -- applicable rows whose best class score fell below the
      frozen abstention floor.  The head was asked and declined to commit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Every row in the publication, applicable or not. The denominator this
    #: distribution deliberately does *not* use.
    binary_row_count: int = Field(ge=0)
    #: The rows the binary champion flagged: the denominator for every rate
    #: below. Zero is an honest value and produces ``None`` rates rather than
    #: zero ones.
    applicable_row_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)

    known_count: int = Field(ge=0)
    known_rate: float | None
    unknown_count: int = Field(ge=0)
    unknown_rate: float | None
    min_category_score: float
    class_order: tuple[str, ...]
    class_counts: tuple[CategoryClassCount, ...]
    max_score_minimum: float | None
    max_score_maximum: float | None
    max_score_mean: float | None

    @model_validator(mode="after")
    def check_distribution(self) -> Self:
        """Both populations sum, and every declared class is accounted for."""
        if self.applicable_row_count + self.not_applicable_count != (
            self.binary_row_count
        ):
            raise ValueError(
                "applicable and not-applicable counts do not sum to the rows scored"
            )
        if self.known_count + self.unknown_count != self.applicable_row_count:
            raise ValueError(
                "known and unknown counts do not sum to the applicable rows; a row "
                "the binary head never flagged is not an abstention"
            )
        if tuple(item.class_name for item in self.class_counts) != self.class_order:
            raise ValueError("class counts must be given in the frozen class order")
        if UNKNOWN_CATEGORY in self.class_order:
            raise ValueError(
                f"{UNKNOWN_CATEGORY!r} is the abstention outcome, not a declared class"
            )
        if sum(item.predicted_count for item in self.class_counts) != self.known_count:
            raise ValueError("per-class counts do not account for every known row")
        return self


class AnomalyDistribution(BaseModel):
    """How the experimental probe's magnitudes were distributed.

    Permanently marked experimental.  Nothing here is comparable with a
    supervised score, and the flag count -- where a frozen threshold exists -- is
    a count of rows the probe found unusual, not a count of attacks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_rows: int = Field(ge=0)
    score_minimum: float | None
    score_maximum: float | None
    score_mean: float | None
    score_quantiles: tuple[QuantilePoint, ...] | None
    anomaly_threshold: float | None
    flagged_count: int | None = Field(default=None, ge=0)
    flagged_rate: float | None = None
    experimental: bool = True
    influences_champion_selection: bool = False

    @model_validator(mode="after")
    def check_distribution(self) -> Self:
        """A flag count exists exactly when a frozen threshold does."""
        if not self.experimental:
            raise ValueError("the anomaly probe is permanently experimental")
        if self.influences_champion_selection:
            raise ValueError("the anomaly probe never influences champion selection")
        if (self.anomaly_threshold is None) != (self.flagged_count is None):
            raise ValueError(
                "a flag count and the threshold that produced it are reported "
                "together, or neither is"
            )
        if (self.flagged_count is None) != (self.flagged_rate is None):
            raise ValueError("a flag count and its rate are reported together")
        return self


class MLQualityReport(SealedModel):
    """The aggregate profile of one prediction publication.

    Reconstructible from the published artifacts alone: given the prediction
    directory and nothing else, this report can be rebuilt byte for byte.  That
    is what makes it checkable rather than merely informative.
    """

    fingerprint_field: ClassVar[str] = "report_fingerprint"
    schema_version_field: ClassVar[str] = "quality_schema_version"
    schema_version: ClassVar[str] = QUALITY_SCHEMA_VERSION
    record_label: ClassVar[str] = "ML quality report"

    quality_schema_version: str = QUALITY_SCHEMA_VERSION
    prediction_id: str
    prediction_content_fingerprint: Sha256Hex
    scope: MLSplit
    scope_role: PredictionScopeRole

    binary: BinaryDistribution
    category: CategoryDistribution | None = None
    anomaly: AnomalyDistribution | None = None

    #: What validation found.  Carried so a profile can never be read as a
    #: clean bill of health for an artifact that failed its checks.
    validation_status: AuditStatus
    validation_failures: tuple[str, ...]
    validation_check_count: int = Field(ge=1)

    report_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_report(self) -> Self:
        """A passing validation names no failure, and a failing one names some."""
        if (self.validation_status is AuditStatus.PASS) != (
            not self.validation_failures
        ):
            raise ValueError(
                "a passing validation names no failing check, and a failing one "
                "names at least one"
            )
        return self


def build_quality_report(
    *,
    manifest: PredictionManifest,
    validation: MLValidationResult,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    anomaly: Sequence[AnomalyScore] | None,
) -> MLQualityReport:
    """Return the aggregate profile of one publication.

    Derived from the rows and the manifest, which is exactly what a later reader
    has.  Nothing is taken from the training context, the dataset, or the model
    beyond what the publication already records.
    """
    return MLQualityReport.seal(
        prediction_id=manifest.prediction_id,
        prediction_content_fingerprint=manifest.prediction_content_fingerprint,
        scope=manifest.scope,
        scope_role=manifest.scope_role,
        binary=_binary_distribution(binary),
        category=(
            None
            if category is None
            else _category_distribution(
                category,
                binary=binary,
                class_order=manifest.lineage.category_class_order or (),
                min_category_score=manifest.lineage.min_category_score or 0.0,
            )
        ),
        anomaly=None if anomaly is None else _anomaly_distribution(anomaly),
        validation_status=validation.status,
        validation_failures=validation.failures,
        validation_check_count=len(validation.checks),
    )


def _statistics(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None, tuple[QuantilePoint, ...] | None]:
    """Return the minimum, maximum, mean, and quantiles of *values*, or nulls."""
    if not values:
        return (None, None, None, None)
    quantiles = tuple(
        QuantilePoint(quantile=fraction, value=_quantile(values, fraction))
        for fraction in REPORTED_QUANTILES
    )
    return (
        quantize(min(values)),
        quantize(max(values)),
        quantize(sum(values) / len(values)),
        quantiles,
    )


def _binary_distribution(rows: Sequence[BinaryPrediction]) -> BinaryDistribution:
    """Return the binary head's aggregate profile."""
    flagged = sum(1 for row in rows if row.flagged_malicious)
    scores = [row.malicious_decision_score for row in rows]
    probabilities = [
        row.malicious_probability
        for row in rows
        if row.malicious_probability is not None
    ]
    calibrated = bool(rows) and len(probabilities) == len(rows)
    score_min, score_max, score_mean, score_quantiles = _statistics(scores)
    probability_stats = _statistics(probabilities) if calibrated else (None,) * 4
    return BinaryDistribution(
        total_rows=len(rows),
        flagged_count=flagged,
        flagged_rate=_rate(flagged, len(rows)),
        unflagged_count=len(rows) - flagged,
        unflagged_rate=_rate(len(rows) - flagged, len(rows)),
        score_kind=(rows[0].score_kind if rows else ScoreKind.DECISION_SCORE),
        decision_threshold=rows[0].decision_threshold if rows else 0.0,
        decision_score_available=bool(rows),
        decision_score_minimum=score_min,
        decision_score_maximum=score_max,
        decision_score_mean=score_mean,
        decision_score_quantiles=score_quantiles,
        calibrated_probability_available=calibrated,
        probability_minimum=probability_stats[0],
        probability_maximum=probability_stats[1],
        probability_mean=probability_stats[2],
        probability_quantiles=probability_stats[3],
        null_probability_count=len(rows) - len(probabilities),
    )


def _category_distribution(
    rows: Sequence[CategoryPrediction],
    *,
    binary: Sequence[BinaryPrediction],
    class_order: Sequence[str],
    min_category_score: float,
) -> CategoryDistribution:
    """Return the category head's aggregate profile.

    The denominator is the applicable population -- the rows the binary head
    flagged -- which is exactly the rows this table contains. *binary* is
    supplied so the not-applicable count can be reported alongside it rather
    than left for a reader to subtract.

    ``min_category_score`` comes from the frozen lineage rather than from the
    first row, so a publication in which nothing was applicable still reports
    the floor the head would have been held to.
    """
    counts = dict.fromkeys(class_order, 0)
    unknown = 0
    for row in rows:
        if row.predicted_scenario == UNKNOWN_CATEGORY:
            unknown += 1
        elif row.predicted_scenario in counts:
            counts[row.predicted_scenario] += 1
        else:
            raise ValueError(
                "a category row names a class outside the frozen class order"
            )
    applicable = len(rows)
    best = [row.max_category_score for row in rows]
    minimum, maximum, mean, _ = _statistics(best)
    return CategoryDistribution(
        binary_row_count=len(binary),
        applicable_row_count=applicable,
        not_applicable_count=len(binary) - applicable,
        known_count=applicable - unknown,
        known_rate=_rate(applicable - unknown, applicable),
        unknown_count=unknown,
        unknown_rate=_rate(unknown, applicable),
        min_category_score=min_category_score,
        class_order=tuple(class_order),
        class_counts=tuple(
            CategoryClassCount(class_name=name, predicted_count=counts[name])
            for name in class_order
        ),
        max_score_minimum=minimum,
        max_score_maximum=maximum,
        max_score_mean=mean,
    )


def _anomaly_distribution(rows: Sequence[AnomalyScore]) -> AnomalyDistribution:
    """Return the experimental probe's aggregate profile."""
    scores = [row.anomaly_score for row in rows]
    minimum, maximum, mean, quantiles = _statistics(scores)
    threshold = rows[0].anomaly_threshold if rows else None
    flagged = (
        None if threshold is None else sum(1 for row in rows if row.flagged_anomalous)
    )
    return AnomalyDistribution(
        total_rows=len(rows),
        score_minimum=minimum,
        score_maximum=maximum,
        score_mean=mean,
        score_quantiles=quantiles,
        anomaly_threshold=threshold,
        flagged_count=flagged,
        flagged_rate=None if flagged is None else _rate(flagged, len(rows)),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(value: float | int | None) -> str:
    """Render a number, or say it is unavailable rather than printing a zero."""
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:g}"


def quality_report_to_markdown(report: MLQualityReport) -> str:
    """Render *report* as deterministic Markdown.

    Every number comes from the report, in declaration order, with no wall
    clock, no path, and no identifier: two renderings of the same report are
    byte-identical.
    """
    binary = report.binary
    lines = [
        "# ML prediction profile",
        "",
        f"- Prediction: `{report.prediction_id}`",
        f"- Scope: `{report.scope}` ({report.scope_role.role})",
        f"- Content fingerprint: `{report.prediction_content_fingerprint[:16]}`",
        f"- Validation: **{report.validation_status}** over "
        f"{report.validation_check_count:,} check(s)",
    ]
    if report.validation_failures:
        lines.append(f"- Failing checks: {', '.join(report.validation_failures)}")
    lines += [
        "",
        "## Binary predictions",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Rows scored | {binary.total_rows:,} |",
        f"| Flagged | {binary.flagged_count:,} |",
        f"| Flagged rate | {_cell(binary.flagged_rate)} |",
        f"| Unflagged | {binary.unflagged_count:,} |",
        f"| Score kind | `{binary.score_kind}` |",
        f"| Decision threshold | {binary.decision_threshold:g} |",
        f"| Decision score range | {_cell(binary.decision_score_minimum)} .. "
        f"{_cell(binary.decision_score_maximum)} |",
        f"| Decision score mean | {_cell(binary.decision_score_mean)} |",
        f"| Calibrated probability | "
        f"{'available' if binary.calibrated_probability_available else 'unavailable'} |",
        f"| Probability range | {_cell(binary.probability_minimum)} .. "
        f"{_cell(binary.probability_maximum)} |",
        f"| Probability mean | {_cell(binary.probability_mean)} |",
        f"| Rows without a probability | {binary.null_probability_count:,} |",
    ]

    if report.category is None:
        lines += ["", "## Category predictions", "", "No category head was frozen."]
    else:
        category = report.category
        lines += [
            "",
            "## Category triage",
            "",
            "Category triage runs only on the rows the binary champion flagged. "
            "Every rate below is over that applicable population, never over "
            "the whole table: the head was fitted on known-malicious rows, so a "
            "row it was never asked about is **not applicable** rather than an "
            f"abstention. `{UNKNOWN_CATEGORY}` means the head *was* asked and "
            "declined to commit.",
            "",
            "| Quantity | Value |",
            "|---|---|",
            f"| Rows scored | {category.binary_row_count:,} |",
            f"| Not applicable (binary did not flag) | "
            f"{category.not_applicable_count:,} |",
            f"| Category-applicable | {category.applicable_row_count:,} |",
            f"| Known class assigned | {category.known_count:,} |",
            f"| Abstained (`{UNKNOWN_CATEGORY}`) | {category.unknown_count:,} |",
            f"| Abstention rate (of applicable) | {_cell(category.unknown_rate)} |",
            f"| Abstention threshold | {category.min_category_score:g} |",
            "",
            "| Class | Predicted rows |",
            "|---|---|",
        ]
        lines += [
            f"| `{item.class_name}` | {item.predicted_count:,} |"
            for item in category.class_counts
        ]
        if category.applicable_row_count == 0:
            lines += [
                "",
                "No row was routed to triage, so every rate above is "
                "unavailable rather than zero. A head that was never asked is "
                "not a head that abstained.",
            ]

    if report.anomaly is not None:
        anomaly = report.anomaly
        lines += [
            "",
            "## Experimental anomaly probe",
            "",
            "| Quantity | Value |",
            "|---|---|",
            f"| Rows scored | {anomaly.total_rows:,} |",
            f"| Anomaly score range | {_cell(anomaly.score_minimum)} .. "
            f"{_cell(anomaly.score_maximum)} |",
            f"| Anomaly score mean | {_cell(anomaly.score_mean)} |",
            f"| Threshold | {_cell(anomaly.anomaly_threshold)} |",
            f"| Flagged | {_cell(anomaly.flagged_count)} |",
            f"| Flagged rate | {_cell(anomaly.flagged_rate)} |",
            "",
            "The anomaly probe is experimental. Its output is an ordered "
            "magnitude, never a probability, and it influences no champion "
            "selection, threshold, or supervised decision.",
        ]

    lines += [
        "",
        "## What this report is not",
        "",
        "Every figure above describes the *distribution of the model's output* "
        "and the *structural integrity of the artifact*. None of it describes "
        "whether the predictions are correct.",
        "",
        "No label was read to produce this publication, so no accuracy, "
        "precision, recall, F1, false-positive rate, PR-AUC, Brier score, or "
        "calibration error is computable from it. Those require the test "
        "labels, which remain unopened, and they are the subject of a separate "
        "later evaluation.",
        "",
        "Structural validity is not predictive quality: a publication can pass "
        "every check here and still come from a model that flags the wrong "
        "rows.",
        "",
        "The predictions were produced on synthetic authentication traffic. "
        "Score distributions on generated data reflect the generator's "
        "assumptions, not a production population.",
        "",
    ]
    return "\n".join(lines)


def _assert_no_outcome_field() -> None:
    """Fail at import if a quality schema declares a field that needs labels."""
    forbidden = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "true_positive_rate",
        "false_negative_rate",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "pr_auc",
        "roc_auc",
        "average_precision",
        "brier_score",
        "expected_calibration_error",
        "confusion_matrix",
        "detection_rate",
        "test_metrics",
        "label_fingerprint",
        "anchor_event_id",
    }
    models: tuple[type[BaseModel], ...] = (
        MLQualityReport,
        BinaryDistribution,
        CategoryDistribution,
        CategoryClassCount,
        AnomalyDistribution,
        QuantilePoint,
    )
    for model in models:
        offending = sorted(set(model.model_fields) & forbidden)
        if offending:
            raise ValueError(
                f"{model.__name__} declares outcome-dependent field(s) "
                f"{offending}; those require labels this milestone never reads"
            )


_assert_no_outcome_field()


def _assert_quantiles_are_declared_once() -> None:
    """Fail at import if the reported quantile set is malformed."""
    if len(set(REPORTED_QUANTILES)) != len(REPORTED_QUANTILES):
        raise ValueError("a reported quantile is declared twice")
    if list(REPORTED_QUANTILES) != sorted(REPORTED_QUANTILES):
        raise ValueError("reported quantiles must be in ascending order")
    if not all(0.0 < value < 1.0 for value in REPORTED_QUANTILES):
        raise ValueError("a reported quantile lies outside (0, 1)")


_assert_quantiles_are_declared_once()


def quality_report_to_dict(report: MLQualityReport) -> dict[str, Any]:
    """Return the JSON-ready mapping the report serialises to."""
    return report.to_dict()
