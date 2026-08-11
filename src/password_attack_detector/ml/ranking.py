"""Exact validation-B ranking evidence, and the PR-AUC the champion gate uses.

Two things are kept apart here that a single "curve" would quietly merge.

**Operating-point search** (Milestone 5) walks a *bounded* candidate grid to
choose one decision threshold. Its curve exists to justify that threshold, and
when the distinct scores outnumber the configured grid it is a sampled curve —
correct for its purpose, and useless as a discrimination measurement.

**Ranking evidence** (this module) is the discrimination measurement. It is
built from **every semantically distinct score level** in the frozen
validation-B scores, never from the threshold grid, so the number it yields does
not move when somebody changes ``search_grid_size``. A mandatory champion gate
measured on a sampled curve would be a gate whose strictness depended on a
performance setting.

**The metric is PR-AUC, integrated step-wise.** That is the name the reviewed
gate configuration uses (``min_pr_auc_gain_over_baseline``), so it is the name
this module computes, publishes, and is tested against. The integration
convention is stated explicitly rather than left to a library default:

.. code-block:: text

    levels sorted by score, descending, one level per distinct score
    for level i:  TP_i = cumulative true positives at scores >= s_i
                  FP_i = cumulative false positives at scores >= s_i
                  precision_i = TP_i / (TP_i + FP_i)
                  recall_i    = TP_i / P
    PR-AUC = sum over i of  precision_i * (recall_i - recall_{i-1})

with ``recall_0 = 0``. The rectangle is right-continuous: the precision actually
achieved at a level is carried across the recall that level gains. There is no
trapezoidal interpolation and no interpolated precision, because linear
interpolation between two points of a precision-recall curve does not correspond
to any achievable operating point.

**Ties are one level.** Rows sharing a score are one group and enter the
cumulative counts together. Nothing here consults an anchor, a row index, or the
order the rows arrived in — breaking ties by any of those would manufacture
discrimination *inside* a set of rows the model scored identically, which is
precisely the number a ranking metric must not invent.

**One scoring stage for every model.** The evidence is built from the frozen
pre-threshold model score (``decision_score``): the quantity the model itself
produces, before any calibration map and before any threshold. Calibration is
measured by its own gate and a threshold by another; letting either redefine the
ranking metric would mean comparing a raw score for one model against a
calibrated one for the next.

No identifier, row, or anchor is published. A level is a score and four counts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.calibration import (
    BinaryScoreSample,
    SealedModel,
    quantize,
    require_partition,
)
from password_attack_detector.ml.enums import ScoreKind, ValidationPartition
from password_attack_detector.ml.ordering import assert_canonical
from password_attack_detector.ml.schemas import Sha256Hex

__all__ = [
    "DISCRIMINATION_SCORE_KIND",
    "PR_AUC_INTEGRATION",
    "RANKING_METRIC_NAME",
    "RANKING_SCHEMA_VERSION",
    "RankingEvidence",
    "ScoreLevel",
    "build_ranking_evidence",
    "pr_auc",
]

#: The ranking-evidence contract's own version.
RANKING_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The persisted metric name.  It matches the reviewed gate setting
#: ``min_pr_auc_gain_over_baseline`` and the arithmetic in :func:`pr_auc`, and
#: it is pinned by a test so a later refactor cannot swap in a differently
#: defined quantity under the same name.
RANKING_METRIC_NAME: Final[Literal["pr_auc"]] = "pr_auc"

#: The declared integration convention.  Step-wise, right-continuous, no
#: interpolation of any kind.
PR_AUC_INTEGRATION: Final[Literal["stepwise"]] = "stepwise"

#: The one scoring stage every model's discrimination is measured at.
DISCRIMINATION_SCORE_KIND: Final[ScoreKind] = ScoreKind.DECISION_SCORE


class ScoreLevel(BaseModel):
    """One distinct score, the rows that share it, and the curve point it makes.

    ``positive_count`` and ``negative_count`` are the rows *at* this score;
    the cumulative counts are over every row scoring at or above it. Both are
    published because the first is what makes tie handling checkable and the
    second is what the metric integrates.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    cumulative_true_positives: int = Field(ge=0)
    cumulative_false_positives: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_level(self) -> Self:
        """The published rates agree with the counts they came from."""
        if not math.isfinite(self.score):
            raise ValueError("a score level must name a finite score")
        if self.positive_count + self.negative_count < 1:
            raise ValueError("a score level accounts for at least one row")
        flagged = self.cumulative_true_positives + self.cumulative_false_positives
        if quantize(self.cumulative_true_positives / flagged) != self.precision:
            raise ValueError("a level's precision disagrees with its counts")
        return self


class RankingEvidence(SealedModel):
    """Exact validation-B discrimination evidence for one model.

    Independent of the operating-threshold grid by construction: it is built
    from the distinct score levels, and there is no field here through which a
    grid size could reach the metric.
    """

    fingerprint_field: ClassVar[str] = "evidence_fingerprint"
    schema_version_field: ClassVar[str] = "ranking_schema_version"
    schema_version: ClassVar[str] = RANKING_SCHEMA_VERSION
    record_label: ClassVar[str] = "validation ranking evidence"

    ranking_schema_version: str = RANKING_SCHEMA_VERSION
    #: The metric this record carries, named rather than implied.
    metric_name: Literal["pr_auc"] = RANKING_METRIC_NAME
    #: The integration convention behind that name.
    integration: Literal["stepwise"] = PR_AUC_INTEGRATION
    #: The scoring stage the metric was measured at, identical for every model
    #: a selection compares.
    score_kind: ScoreKind

    pr_auc: float = Field(ge=0.0, le=1.0)

    row_count: int = Field(ge=1)
    positive_count: int = Field(ge=1)
    negative_count: int = Field(ge=1)
    distinct_score_count: int = Field(ge=1)
    #: Every distinct score level, best first.  Complete: this is the exact
    #: curve, not a sample of it, and no configured bound may truncate it.
    levels: tuple[ScoreLevel, ...]

    source_partition: ValidationPartition
    source_partition_fingerprint: Sha256Hex
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    evidence_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_evidence(self) -> Self:
        """The curve is complete, ordered, tie-grouped, and recomputes its metric."""
        if self.source_partition is not ValidationPartition.VALIDATION_B:
            raise ValueError("ranking evidence is measured on validation-B")
        if self.score_kind is not DISCRIMINATION_SCORE_KIND:
            raise ValueError(
                f"discrimination is measured at {str(DISCRIMINATION_SCORE_KIND)!r} "
                f"for every model a selection compares"
            )
        if not self.levels:
            raise ValueError("ranking evidence carries at least one score level")
        if len(self.levels) != self.distinct_score_count:
            raise ValueError("the level count disagrees with the distinct scores")
        scores = [level.score for level in self.levels]
        if scores != sorted(scores, reverse=True):
            raise ValueError("score levels are ordered by score, descending")
        if len(set(scores)) != len(scores):
            raise ValueError(
                "a score appears in two levels; rows sharing a score are one "
                "group, and splitting them would invent an ordering inside a tie"
            )
        if self.positive_count + self.negative_count != self.row_count:
            raise ValueError("the class counts do not account for every row")
        if sum(level.positive_count for level in self.levels) != self.positive_count:
            raise ValueError("the levels do not account for every positive row")
        if sum(level.negative_count for level in self.levels) != self.negative_count:
            raise ValueError("the levels do not account for every benign row")

        positives = 0
        negatives = 0
        for level in self.levels:
            positives += level.positive_count
            negatives += level.negative_count
            if (
                level.cumulative_true_positives != positives
                or level.cumulative_false_positives != negatives
            ):
                raise ValueError("a level's cumulative counts are not cumulative")
            if quantize(positives / self.positive_count) != level.recall:
                raise ValueError("a level's recall disagrees with its counts")
        if pr_auc(self.levels, positive_count=self.positive_count) != self.pr_auc:
            raise ValueError(
                "the recorded PR-AUC does not recompute from the published "
                "levels; the metric and the curve behind it disagree"
            )
        return self


def pr_auc(levels: Sequence[ScoreLevel], *, positive_count: int) -> float:
    """Return the exact step-wise PR-AUC of an ordered, tie-grouped curve.

    Computed from the integer counts rather than from the published rates, so
    the quantisation applied for storage cannot accumulate into the metric.

    Args:
        levels: distinct score levels, best first.
        positive_count: total positive rows, the recall denominator.

    Returns:
        The area under the precision-recall curve under the right-continuous
        step convention documented at the top of this module.
    """
    if positive_count <= 0:
        raise ModelTrainingError("PR-AUC is undefined without a positive row")
    area = 0.0
    previous_recall = 0.0
    for level in levels:
        flagged = level.cumulative_true_positives + level.cumulative_false_positives
        precision = level.cumulative_true_positives / flagged
        recall = level.cumulative_true_positives / positive_count
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return quantize(area)


def build_ranking_evidence(
    sample: BinaryScoreSample, *, ml_config_fingerprint: str
) -> RankingEvidence | None:
    """Return exact ranking evidence for *sample*, or ``None`` when undefined.

    ``None`` means the metric has no value on these rows -- a partition with no
    positive row has no recall, and one with no benign row has no precision to
    lose. That is an absence of evidence, and the champion gate treats it as
    one: inconclusive rather than a pass on a number nobody could compute.

    Args:
        sample: frozen validation-B scores at the declared discrimination
            scoring stage.
        ml_config_fingerprint: the configuration the run was carried out under.

    Raises:
        ModelTrainingError: on rows from anywhere but validation-B, on
            uncanonical rows, or on a sample whose scores are not the declared
            discrimination stage. These are contract violations, not outcomes.
    """
    stage = "validation ranking evidence"
    require_partition(
        sample.source, expected=ValidationPartition.VALIDATION_B, stage=stage
    )
    assert_canonical(sample.anchors, stage=stage)
    if sample.score_kind is not DISCRIMINATION_SCORE_KIND:
        raise ModelTrainingError(
            f"{stage} is measured on {str(DISCRIMINATION_SCORE_KIND)!r}; it was "
            f"handed {str(sample.score_kind)!r}. Measuring one model's "
            f"discrimination after calibration and another's before it would "
            f"compare two different quantities"
        )

    positive_total = sum(1 for flag in sample.malicious if flag)
    negative_total = len(sample.malicious) - positive_total
    if positive_total == 0 or negative_total == 0:
        return None

    # Grouped by score, never by row. Two rows sharing a score are one level
    # whatever order they arrived in, which is what makes the metric invariant
    # to the row order and to every identifier the rows carry.
    grouped: dict[float, list[int]] = {}
    for score, malicious in zip(sample.scores, sample.malicious, strict=True):
        counts = grouped.setdefault(score, [0, 0])
        counts[0 if malicious else 1] += 1

    levels: list[ScoreLevel] = []
    positives = 0
    negatives = 0
    for score in sorted(grouped, reverse=True):
        at_level = grouped[score]
        positives += at_level[0]
        negatives += at_level[1]
        # Every distinct level is published, including those above the first
        # positive row. They contribute no recall and therefore no area, and
        # keeping them is what makes the published curve the complete one rather
        # than the part of it that happened to matter.
        levels.append(
            ScoreLevel(
                score=quantize(score),
                positive_count=at_level[0],
                negative_count=at_level[1],
                cumulative_true_positives=positives,
                cumulative_false_positives=negatives,
                precision=quantize(positives / (positives + negatives)),
                recall=quantize(positives / positive_total),
            )
        )

    return RankingEvidence.seal(
        score_kind=sample.score_kind,
        pr_auc=pr_auc(levels, positive_count=positive_total),
        row_count=len(sample.scores),
        positive_count=positive_total,
        negative_count=negative_total,
        distinct_score_count=len(levels),
        levels=tuple(levels),
        source_partition=ValidationPartition.VALIDATION_B,
        source_partition_fingerprint=sample.source.source_fingerprint,
        model_id=sample.model_id,
        model_content_fingerprint=sample.model_content_fingerprint,
        preprocessor_fingerprint=sample.preprocessor_fingerprint,
        ml_config_fingerprint=ml_config_fingerprint,
    )
