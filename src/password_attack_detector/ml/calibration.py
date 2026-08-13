"""Turning a decision score into a probability, and saying so only once it is one.

Milestone 4 was careful never to call anything a probability.  This module is
where the word becomes available -- and it becomes available under conditions,
not by renaming a column.

**Validation-A, and nothing else.**  A calibrator is fitted on the half of the
validation split that Milestone 2 set aside for it.  Validation-B chooses the
operating point; test and the novel-anomaly holdout are read once, after
everything is frozen.  Every entry point here takes typed provenance and refuses
anything else, and there is no flag that relaxes it.  The refusal is behavioural
rather than nominal: nothing inspects a filename, and nothing trusts a caller's
description of where rows came from beyond the typed object it hands over.

**What this module may see.**  Scores, binary labels, anchors for the ordering
assertion, and fingerprints.  It opens no Parquet file, reads no label table, no
split table, and no campaign table; it never refits the model whose scores it
consumes and never touches the fitted preprocessor.  Those inputs arrive as
typed objects assembled by a caller that is allowed to read them --
:mod:`password_attack_detector.ml.dataset` today, training orchestration later.

**Fitting is project-owned.**  Platt scaling is a two-parameter Newton fit
written here; isotonic regression is a weighted pool-adjacent-violators fit
written here.  Neither imports scikit-learn, for the same reason no module
outside ``ml/models`` does: an estimator object is not an artifact, and a
calibrator that could only be evaluated by reconstructing one would be unusable
by a release whose internals had moved.  The isotonic state reduces to the two
attributes scikit-learn documents -- ``X_thresholds_`` and ``y_thresholds_`` --
and a test asserts that the project's fit reproduces the audited release's
values exactly, which is a stronger statement than delegating to it would be.

**Calibrated against what.**  A fitted calibrator here is calibrated against the
frozen synthetic validation-A distribution under the declared protocol.  That is
an internal property.  It is not evidence that these probabilities are
well-calibrated on real authentication traffic, and no number this module
produces should be read that way.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any, ClassVar, Final, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.config import CalibrationConfig
from password_attack_detector.ml.enums import (
    CalibrationEvaluationKind,
    CalibrationMethod,
    CalibrationStatus,
    MetricStatus,
    MLSplit,
    ScoreKind,
    ValidationPartition,
    is_probability,
)
from password_attack_detector.ml.imbalance import BINARY_CLASS_ORDER
from password_attack_detector.ml.ordering import AnchoredRow, assert_canonical
from password_attack_detector.ml.schemas import (
    ScoreSemantics,
    Sha256Hex,
    prohibited_metadata_fields,
)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "POSITIVE_CLASS",
    "BinaryScoreSample",
    "CalibrationOutcome",
    "CalibrationReport",
    "CalibrationState",
    "IsotonicParameters",
    "PlattParameters",
    "ReliabilityBin",
    "ScoreSampleSource",
    "apply_calibration",
    "brier_score",
    "diagnose_calibration_fit",
    "evaluate_calibration_quality",
    "fit_calibration",
    "require_chain",
    "require_out_of_sample_evidence",
    "require_partition",
]

#: The calibration contract's own version, independent of the Milestone 4 model
#: contract.  A calibrator is a separate identity fitted *after* a model is
#: frozen, so the two versions move independently and neither invalidates the
#: other.
CALIBRATION_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The class a calibrated probability is the probability *of*.
#:
#: Stated rather than inferred from a label ordering.  "The probability" is
#: meaningless without naming the event, and a calibrator whose positive class
#: was implied by whichever order a caller happened to pass would silently
#: invert on the day somebody passed the other one.
POSITIVE_CLASS: Final[str] = BINARY_CLASS_ORDER[1]

#: Fitted scalars are stored at this precision, matching every other fingerprint
#: in the project.
_FLOAT_PRECISION: Final[int] = 9

#: The digest a sealed record carries before its real one has been computed.
#: Never valid, never stored, and only ever seen by :meth:`SealedModel.seal`.
_UNSEALED: Final[str] = "0" * 64


def canonical_json(payload: Any) -> str:
    """Return the one rendering this module treats as canonical."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    """Return the SHA-256 hex digest of *payload*'s canonical rendering."""
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def quantize(value: float) -> float:
    """Return *value* at the precision fitted scalars are stored in.

    Raises:
        ModelTrainingError: if the value is not finite.  ``NaN`` and infinity
            are refused here rather than allowed to reach a digest, where they
            would be stable and meaningless.
    """
    if not math.isfinite(value):
        raise ModelTrainingError(
            "a fitted calibration parameter must be finite; NaN and infinity "
            "are neither serialisable nor meaningful"
        )
    return float(f"{value:.{_FLOAT_PRECISION}f}")


def optional_quantized(value: float | None) -> float | None:
    """Return *value* quantized, passing ``None`` through unchanged.

    ``None`` means "not defined here" -- an empty denominator, a bin nothing
    landed in.  Rendering it as ``0.0`` would turn the absence of a measurement
    into a measurement of zero, which is the specific mistake this layer's
    support statuses exist to prevent.
    """
    return None if value is None else quantize(value)


def sigmoid(value: float) -> float:
    """Return the logistic function of *value*, evaluated without overflowing.

    The branch matters.  ``exp(710)`` overflows to infinity in float64, so the
    naive form returns ``NaN`` for a strongly separated row -- exactly the rows
    a detector cares most about.
    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponentiated = math.exp(value)
    return exponentiated / (1.0 + exponentiated)


_SealedT = TypeVar("_SealedT", bound="SealedModel")


class SealedModel(BaseModel):
    """A frozen record carrying the digest of its own semantic content.

    The digest is a *field*, not a method, so it survives serialization and a
    later reader can tell a tampered payload from an intact one without having
    to be handed the expected value separately.  It is recomputed on every
    construction -- including every deserialization -- and a record whose
    content and digest disagree is refused rather than repaired.

    :meth:`seal` is the only way to build one, because the digest covers every
    other field and therefore cannot be supplied by a caller.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Name of the field holding this record's own digest.
    fingerprint_field: ClassVar[str] = "fingerprint"
    #: Name of the field holding the contract version this record was written
    #: under, and the version this build implements.  Checked *before*
    #: validation, so a payload from a contract this build does not implement is
    #: refused rather than partially understood.
    schema_version_field: ClassVar[str] = "schema_version"
    schema_version: ClassVar[str] = ""
    #: What this record is called in an error message.
    record_label: ClassVar[str] = "sealed record"

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic content this record's digest is taken over."""
        payload = dict(self.model_dump(mode="json"))
        payload.pop(type(self).fingerprint_field, None)
        return payload

    def recomputed_fingerprint(self) -> str:
        """Return the digest this record's content derives."""
        return digest(self.fingerprint_data())

    @model_validator(mode="after")
    def check_seal(self) -> Self:
        """Refuse a record whose content and recorded digest disagree."""
        recorded = getattr(self, type(self).fingerprint_field)
        if recorded != self.recomputed_fingerprint():
            raise ValueError(
                f"{type(self).__name__} does not recompute its recorded "
                f"fingerprint; the content and the digest disagree, so the "
                f"payload is refused rather than trusted"
            )
        return self

    @classmethod
    def seal(cls: type[_SealedT], **fields: Any) -> _SealedT:
        """Return a record of *fields* carrying the digest they derive.

        The probe is built with :meth:`~pydantic.BaseModel.model_construct` so
        the placeholder digest never has to pass the seal validator; only the
        returned record is validated, and it is validated in full.
        """
        unsealed = dict(fields)
        unsealed[cls.fingerprint_field] = _UNSEALED
        probe = cls.model_construct(**unsealed)
        sealed = dict(fields)
        sealed[cls.fingerprint_field] = digest(probe.fingerprint_data())
        return cls(**sealed)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping this record serialises to."""
        return dict(self.model_dump(mode="json"))

    def to_json(self) -> str:
        """Return canonical JSON: sorted keys, ASCII, no incidental whitespace."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls: type[_SealedT], payload: Any) -> _SealedT:
        """Return the record *payload* describes, or raise.

        Strict in three ways, in order: the payload must be an object, it must
        declare the contract version this build implements, and it must satisfy
        every field validator including the seal.  A record read loosely would
        drop what it did not understand and then verify a digest over the
        remainder, which is worse than not verifying one at all.

        Raises:
            ModelTrainingError: on any of the three, naming which.
        """
        if not isinstance(payload, dict):
            raise ModelTrainingError(
                f"{cls.record_label} must be a JSON object, got "
                f"{type(payload).__name__}"
            )
        declared = payload.get(cls.schema_version_field)
        if declared != cls.schema_version:
            raise ModelTrainingError(
                f"{cls.record_label} declares schema version {declared!r}; this "
                f"build implements {cls.schema_version!r}"
            )
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise ModelTrainingError(
                f"{cls.record_label} is not valid ({type(exc).__name__})"
            ) from None

    @classmethod
    def from_json(cls: type[_SealedT], text: str) -> _SealedT:
        """Return the record *text* encodes, or raise."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelTrainingError(
                f"{cls.record_label} is not valid JSON ({type(exc).__name__})"
            ) from None
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# Typed provenance and typed inputs
# ---------------------------------------------------------------------------


class ScoreSampleSource(BaseModel):
    """Where a batch of scores came from, stated rather than inferred.

    Every Milestone 5 entry point takes one of these and checks it.  Provenance
    is never derived from a filename, a directory, or a column that happens to
    be present: those describe where bytes were stored, and this describes what
    the rows *are*.

    A source naming the test split or the novel-anomaly holdout is
    constructible, deliberately.  Refusing to build one would move the firewall
    into the type system, where a test could not demonstrate that the selectors
    themselves refuse -- and the selectors refusing is the property that
    matters.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: MLSplit
    #: Which half of the validation split, when the split is validation.
    partition: ValidationPartition | None = None
    #: For a validation source, the digest of the **parent** partitioning that
    #: produced both halves, so validation-A and validation-B carry the same
    #: value and a mismatch proves two unrelated partitions.  For a training
    #: source, the dataset's training-data fingerprint.
    source_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_source(self) -> Self:
        """A validation source names its half; nothing else may name one."""
        if self.split is MLSplit.VALIDATION and self.partition is None:
            raise ValueError(
                "a validation source must name which half it came from; "
                "validation-A fits calibrators and validation-B chooses "
                "operating points, and an unlabelled half could be either"
            )
        if self.split is not MLSplit.VALIDATION and self.partition is not None:
            raise ValueError(
                f"split {str(self.split)!r} carries no validation partition; "
                f"only the validation split is partitioned"
            )
        return self

    @property
    def describe(self) -> str:
        """Return a short, identity-free description for an error message."""
        if self.partition is None:
            return str(self.split)
        return str(self.partition)


class BinaryScoreSample(BaseModel):
    """Frozen binary scores and their labels, for one partition.

    An *input*, not a report: it legitimately carries ground truth, because
    fitting a calibrator and choosing a threshold are both supervised
    operations.  Nothing derived from it may carry a label, a row, or an
    identifier onward, and the privacy sweep at the bottom of this module
    enforces that on everything published.

    ``anchors`` are here for one purpose: asserting canonical row order.  They
    never enter a fitted state, a report, or a digest.
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
    #: Present exactly when the scores are calibrated probabilities.  The two
    #: are tied by a validator, so a batch cannot claim the calibrated
    #: vocabulary without naming the calibrator that produced it.
    calibration_state_fingerprint: Sha256Hex | None = None
    calibration_method: CalibrationMethod = CalibrationMethod.NONE

    @model_validator(mode="after")
    def check_sample(self) -> Self:
        """Every parallel sequence describes the same rows, and the kind agrees."""
        if not self.scores:
            raise ValueError("a score sample must carry at least one row")
        if len(self.malicious) != len(self.scores):
            raise ValueError("scores and labels disagree in length")
        if len(self.anchors) != len(self.scores):
            raise ValueError("anchors and scores disagree in length")
        if not self.model_id.strip():
            raise ValueError("a score sample must name the model that produced it")
        for score in self.scores:
            if not math.isfinite(score):
                raise ValueError(
                    "a score sample carries a non-finite value; a model that "
                    "emitted one is a fault to fix, not a value to calibrate"
                )
        if self.score_kind is ScoreKind.CLASS_SCORE:
            raise ValueError(
                "a per-class score is not a binary decision score; the category "
                "head has its own typed input"
            )
        if self.score_kind is ScoreKind.ANOMALY_SCORE:
            raise ValueError(
                "an anomaly score is an unsupervised magnitude and is never "
                "calibrated as though it were a supervised probability"
            )

        calibrated = is_probability(self.score_kind)
        if calibrated and self.calibration_state_fingerprint is None:
            raise ValueError(
                "a calibrated probability requires the calibrator that produced "
                "it; a calibrated field without a fitted calibrator is a claim "
                "nobody can check"
            )
        if not calibrated and self.calibration_state_fingerprint is not None:
            raise ValueError(
                f"score kind {str(self.score_kind)!r} names a calibrator; only a "
                f"calibrated probability may"
            )
        if calibrated and self.calibration_method is CalibrationMethod.NONE:
            raise ValueError(
                "a calibrated probability cannot declare calibration method "
                "'none'; the method is what made it one"
            )
        if not calibrated and self.calibration_method is not CalibrationMethod.NONE:
            raise ValueError(
                "an uncalibrated score must declare calibration method 'none'"
            )
        if calibrated and any(not 0.0 <= score <= 1.0 for score in self.scores):
            raise ValueError("a calibrated probability is bounded by [0, 1]")
        return self

    def require_canonical(self, *, stage: str) -> None:
        """Assert the rows are canonically ordered, or raise.

        Asserted rather than sorted, for the reason every other stage asserts
        it: a selector that quietly re-sorted would hide the fact that somebody
        handed it rows nobody had ordered.
        """
        assert_canonical(self.anchors, stage=stage)

    @property
    def row_count(self) -> int:
        """Return the number of rows."""
        return len(self.scores)

    @property
    def positive_count(self) -> int:
        """Return the number of malicious rows."""
        return sum(1 for value in self.malicious if value)

    @property
    def negative_count(self) -> int:
        """Return the number of benign rows."""
        return self.row_count - self.positive_count

    @property
    def distinct_score_count(self) -> int:
        """Return the number of distinct score values."""
        return len(set(self.scores))


# ---------------------------------------------------------------------------
# Provenance enforcement
# ---------------------------------------------------------------------------


def require_partition(
    source: ScoreSampleSource,
    *,
    expected: ValidationPartition,
    stage: str,
) -> None:
    """Raise unless *source* is exactly the validation half *stage* may read.

    The single enforcement point for the validation-A / validation-B firewall
    and, by construction, for the test and holdout firewall: neither split can
    satisfy an equality against a :class:`ValidationPartition`, and the enum has
    no member either could be spelled as.

    Raises:
        ModelTrainingError: naming the stage, what it may read, and what it was
            handed.  No path, no identifier, no row.
    """
    if source.split is not MLSplit.VALIDATION or source.partition is not expected:
        raise ModelTrainingError(
            f"{stage} reads {str(expected)!r} rows only; it was handed "
            f"{source.describe!r}. There is no option that widens this: the "
            f"test split and the novel-anomaly holdout are read once, after "
            f"every fitted quantity is frozen"
        )


def require_train_benign(source: ScoreSampleSource, *, stage: str) -> None:
    """Raise unless *source* is the training split.

    Used by the anomaly threshold's training-quantile provenance, which reads
    benign training scores and no validation outcome at all.

    Raises:
        ModelTrainingError: if the rows came from anywhere else.
    """
    if source.split is not MLSplit.TRAIN:
        raise ModelTrainingError(
            f"{stage} reads train rows only; it was handed {source.describe!r}"
        )


def require_chain(
    *,
    stage: str,
    model_content_fingerprint: str,
    preprocessor_fingerprint: str,
    source_fingerprint: str,
    ml_config_fingerprint: str,
    state: CalibrationState,
) -> None:
    """Raise unless a calibrator belongs to the same provenance chain.

    Four independent links, each checked.  A calibrator fitted for one model
    cannot be combined with another model's scores; a calibrator fitted on one
    dataset's validation partition cannot be combined with another dataset's;
    and a calibrator fitted under one configuration cannot be reported under
    another.  Any of the four disagreeing means two runs have been spliced
    together, and the splice is refused rather than reported.

    Raises:
        ModelTrainingError: naming which link failed, and nothing else.
    """
    links = (
        ("model", state.model_content_fingerprint, model_content_fingerprint),
        ("preprocessor", state.preprocessor_fingerprint, preprocessor_fingerprint),
        (
            "validation partition",
            state.validation_partition_fingerprint,
            source_fingerprint,
        ),
        ("configuration", state.ml_config_fingerprint, ml_config_fingerprint),
    )
    for what, expected, actual in links:
        if expected != actual:
            raise ModelTrainingError(
                f"{stage} was handed a calibrator from a different {what}; a "
                f"fitted quantity and the data it is applied to must belong to "
                f"one provenance chain"
            )


# ---------------------------------------------------------------------------
# Fitted calibration state
# ---------------------------------------------------------------------------


class PlattParameters(BaseModel):
    """The two fitted scalars of a one-dimensional logistic calibrator.

    ``sigmoid(a * score + b)``, and nothing else.  Inference needs these two
    numbers and no estimator, which is what makes a stored calibrator readable
    by a build whose libraries have moved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    a: float
    b: float
    #: Newton iterations the fit took.  Recorded because a fit that used its
    #: whole budget and a fit that converged in four steps are different events
    #: even when they land on the same parameters.
    iterations: int = Field(ge=1)
    #: The largest absolute gradient component at the fitted point.  A fitted
    #: calibrator whose gradient is not near zero is not at an optimum, and
    #: recording the number lets a reader check rather than trust.
    final_gradient_norm: float = Field(ge=0.0)

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        """Both scalars are finite and stored at serialised precision."""
        for name in ("a", "b", "final_gradient_norm"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Platt parameter {name!r} must be finite")
        return self


class IsotonicParameters(BaseModel):
    """The breakpoints of a monotone piecewise-linear calibrator.

    Named for the two attributes scikit-learn documents on a fitted
    ``IsotonicRegression`` -- ``X_thresholds_`` and ``y_thresholds_`` -- because
    those are the values a reader can check this state against.  No private
    attribute is read, stored, or depended on.

    ``x_thresholds`` are **observed scores** and are stored at full precision.
    Rounding them would move the breakpoints of a step function, which is a
    different function; ``y_thresholds`` are fitted values and are quantized
    like every other fitted number in this project.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        """Strictly increasing in x, non-decreasing in y, bounded to [0, 1]."""
        if len(self.x_thresholds) != len(self.y_thresholds):
            raise ValueError("x_thresholds and y_thresholds disagree in length")
        if len(self.x_thresholds) < 2:
            raise ValueError(
                "an isotonic calibrator needs at least two breakpoints; one "
                "breakpoint is a constant, and a constant is not a calibration"
            )
        for value in (*self.x_thresholds, *self.y_thresholds):
            if not math.isfinite(value):
                raise ValueError("an isotonic breakpoint must be finite")
        for previous, current in zip(
            self.x_thresholds, self.x_thresholds[1:], strict=False
        ):
            if not current > previous:
                raise ValueError(
                    "x_thresholds must be strictly increasing; two breakpoints "
                    "at the same score describe no interval to interpolate over"
                )
        for previous, current in zip(
            self.y_thresholds, self.y_thresholds[1:], strict=False
        ):
            if current < previous:
                raise ValueError("y_thresholds must be non-decreasing")
        if any(not 0.0 <= value <= 1.0 for value in self.y_thresholds):
            raise ValueError("y_thresholds must lie in [0, 1]")
        return self

    @property
    def domain(self) -> tuple[float, float]:
        """Return the fitted score domain, outside which output is clamped."""
        return (self.x_thresholds[0], self.x_thresholds[-1])


class CalibrationState(SealedModel):
    """A fitted calibrator: parameters, support, and provenance.

    Constructible only for a calibrator that actually fitted.  There is no
    state for :attr:`CalibrationMethod.NONE`, so nothing can hold a state
    object and describe an uncalibrated score as a probability on the strength
    of it.

    Carries no event identifier, no campaign identifier, no pseudonym, no path,
    and no fitting timestamp.  Two runs that fitted the same calibrator on the
    same rows under the same configuration produce byte-identical state, in any
    directory, in any year.
    """

    fingerprint_field: ClassVar[str] = "calibration_state_fingerprint"
    schema_version_field: ClassVar[str] = "calibration_schema_version"
    schema_version: ClassVar[str] = CALIBRATION_SCHEMA_VERSION
    record_label: ClassVar[str] = "calibration state"

    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    method: CalibrationMethod
    #: What was calibrated: the frozen Milestone 4 binary decision score.
    source_score_kind: ScoreKind
    #: What comes out.  The one place in this project where the word is earned.
    output_score_kind: ScoreKind
    #: Where the calibrator was **fitted**.  Named ``fit_source_partition``
    #: rather than ``source_partition`` so it cannot be confused with a
    #: report's ``source_partition``, which names where the calibrator was
    #: *measured*.  A calibrator fitted on validation-A and measured on
    #: validation-B is the intended arrangement, and two fields called the same
    #: thing would make that arrangement impossible to read.
    fit_source_partition: ValidationPartition
    positive_class: str

    platt: PlattParameters | None = None
    isotonic: IsotonicParameters | None = None

    #: The score domain the calibrator was fitted over.  Outside it, an
    #: isotonic calibrator clamps and a Platt calibrator extrapolates, and a
    #: reader can only tell which case a later score is in by knowing this.
    input_support_min: float
    input_support_max: float
    row_count: int = Field(ge=1)
    positive_count: int = Field(ge=1)
    negative_count: int = Field(ge=1)
    distinct_score_count: int = Field(ge=2)

    validation_partition_fingerprint: Sha256Hex
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    calibration_state_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_state(self) -> Self:
        """The method, the parameters, and the vocabulary must all agree."""
        if self.calibration_schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"calibration state declares schema version "
                f"{self.calibration_schema_version!r}; this build implements "
                f"{CALIBRATION_SCHEMA_VERSION!r}"
            )
        if self.method is CalibrationMethod.NONE:
            raise ValueError(
                "there is no fitted state for calibration method 'none'; a "
                "state claiming a calibrated probability without a calibrator "
                "is the exact claim this contract exists to refuse"
            )
        if not is_probability(self.output_score_kind):
            raise ValueError(
                f"a fitted calibrator emits a calibrated probability, not "
                f"{str(self.output_score_kind)!r}"
            )
        if self.source_score_kind is not ScoreKind.DECISION_SCORE:
            raise ValueError(
                f"a binary calibrator consumes a decision score, not "
                f"{str(self.source_score_kind)!r}"
            )
        if self.fit_source_partition is not ValidationPartition.VALIDATION_A:
            raise ValueError(
                "a calibrator is fitted on validation-A; validation-B chooses "
                "the operating point and assesses the fitted calibrator, and "
                "must never also fit it"
            )
        if self.positive_class != POSITIVE_CLASS:
            raise ValueError(
                f"positive_class must be {POSITIVE_CLASS!r}; a probability "
                f"without a named event is not a probability of anything"
            )
        present = {
            CalibrationMethod.PLATT: self.platt is not None,
            CalibrationMethod.ISOTONIC: self.isotonic is not None,
        }
        if not present[self.method]:
            raise ValueError(
                f"method {str(self.method)!r} declares no fitted parameters"
            )
        if sum(present.values()) != 1:
            raise ValueError(
                "a calibration state carries exactly one fitted parameter set"
            )
        if self.row_count != self.positive_count + self.negative_count:
            raise ValueError("class support does not sum to the row count")
        if self.input_support_min >= self.input_support_max:
            raise ValueError(
                "the fitted score domain is degenerate; a calibrator fitted "
                "over a single score value maps everything to one number"
            )
        return self

    @property
    def score_semantics(self) -> ScoreSemantics:
        """Return what this calibrator's output means, and what it may be called."""
        return ScoreSemantics(
            score_kind=self.output_score_kind,
            calibration_method=self.method,
            lower_bound=0.0,
            upper_bound=1.0,
            description=(
                f"Calibrated probability of the {self.positive_class} class, "
                f"fitted on validation-A under the "
                f"{self.method!s} method. Calibrated against the frozen "
                f"validation distribution it was fitted on, which is not a "
                f"claim about any other population."
            ),
        )

    def transform(self, scores: Sequence[float]) -> tuple[float, ...]:
        """Return the calibrated probability for each score in *scores*.

        Project-owned inference from the stored parameters alone: no estimator
        is reconstructed, and nothing is read from a library.

        Raises:
            ModelTrainingError: on a non-finite input score.
        """
        for score in scores:
            if not math.isfinite(float(score)):
                raise ModelTrainingError(
                    "a calibrator was handed a non-finite score; there is no "
                    "probability to map it onto"
                )
        if self.platt is not None:
            return tuple(
                sigmoid(self.platt.a * float(score) + self.platt.b) for score in scores
            )
        assert self.isotonic is not None  # guaranteed by check_state
        return tuple(_interpolate(float(score), self.isotonic) for score in scores)


def _interpolate(score: float, parameters: IsotonicParameters) -> float:
    """Return the piecewise-linear value of the isotonic fit at *score*.

    Clamped outside the fitted domain rather than extrapolated.  Extrapolating
    a monotone step function past its last breakpoint would invent a
    relationship nothing was fitted on, and clamping is what the audited
    scikit-learn release does under ``out_of_bounds="clip"``.
    """
    xs = parameters.x_thresholds
    ys = parameters.y_thresholds
    if score <= xs[0]:
        return ys[0]
    if score >= xs[-1]:
        return ys[-1]
    low, high = 0, len(xs) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if xs[middle] <= score:
            low = middle
        else:
            high = middle
    span = xs[high] - xs[low]
    weight = (score - xs[low]) / span
    return ys[low] + weight * (ys[high] - ys[low])


class CalibrationOutcome(BaseModel):
    """Whether a calibrator was fitted, and the counts behind the answer.

    Returned by :func:`fit_calibration` whatever happened, because a caller
    reporting "no calibrator" still needs the support that produced that
    conclusion.  A state object exists only for
    :attr:`CalibrationStatus.FITTED`, so a failed fit cannot be mistaken for a
    quiet one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CalibrationStatus
    method: CalibrationMethod
    state: CalibrationState | None = None
    failing_requirements: tuple[str, ...] = ()
    row_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    distinct_score_count: int = Field(ge=0)

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        """A state exists exactly when a calibrator was fitted."""
        fitted = self.status is CalibrationStatus.FITTED
        if fitted and self.state is None:
            raise ValueError("a fitted outcome must carry the state it fitted")
        if not fitted and self.state is not None:
            raise ValueError(
                f"status {str(self.status)!r} carries a fitted state; only a "
                f"successful fit may"
            )
        if fitted and self.failing_requirements:
            raise ValueError("a fitted outcome names no failing requirement")
        if self.status is CalibrationStatus.NOT_CALIBRATED and (
            self.method is not CalibrationMethod.NONE
        ):
            raise ValueError(
                "an uncalibrated outcome must record calibration method 'none'"
            )
        if self.status is not CalibrationStatus.NOT_CALIBRATED and (
            self.method is CalibrationMethod.NONE
        ):
            raise ValueError(
                "calibration method 'none' can only produce an uncalibrated outcome"
            )
        return self

    @property
    def fitted(self) -> bool:
        """Return whether a calibrator was fitted and validated."""
        return self.status is CalibrationStatus.FITTED

    def require_state(self) -> CalibrationState:
        """Return the fitted state, or raise if there is none.

        Raises:
            ModelTrainingError: naming the status and the failing requirements.
        """
        if self.state is None:
            raise ModelTrainingError(
                f"no calibrator was fitted [{self.status!s}]: "
                f"{list(self.failing_requirements)}"
            )
        return self.state


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_calibration(
    sample: BinaryScoreSample,
    *,
    config: CalibrationConfig,
    ml_config_fingerprint: str,
) -> CalibrationOutcome:
    """Fit the configured calibrator on validation-A, or explain why not.

    Args:
        sample: validation-A decision scores and their binary labels, in
            canonical row order.
        config: the calibration policy.  Its ``method`` is fixed by
            configuration and is never chosen by comparing candidates -- see
            the note on :class:`~password_attack_detector.ml.config.CalibrationConfig`.
        ml_config_fingerprint: the digest of the configuration this run is
            being carried out under, recorded so a calibrator cannot later be
            reported under a different one.

    Returns:
        An outcome carrying a fitted state, or a status naming why none was
        fitted.  Insufficient support and a failed solve are different answers
        and are reported as such.

    Raises:
        ModelTrainingError: if the rows did not come from validation-A, are not
            canonically ordered, or do not carry a raw decision score.  These
            are contract violations rather than data outcomes, and there is no
            status that softens them.
    """
    stage = "calibration fitting"
    require_partition(
        sample.source, expected=ValidationPartition.VALIDATION_A, stage=stage
    )
    sample.require_canonical(stage=stage)
    if sample.score_kind is not ScoreKind.DECISION_SCORE:
        raise ModelTrainingError(
            f"{stage} consumes the frozen decision score; it was handed "
            f"{str(sample.score_kind)!r}. Calibrating an already-calibrated "
            f"score would report a measurement of the calibrator, not the model"
        )

    counts = {
        "row_count": sample.row_count,
        "positive_count": sample.positive_count,
        "negative_count": sample.negative_count,
        "distinct_score_count": sample.distinct_score_count,
    }

    def outcome(
        status: CalibrationStatus,
        method: CalibrationMethod,
        *,
        state: CalibrationState | None = None,
        failing: tuple[str, ...] = (),
    ) -> CalibrationOutcome:
        """Return an outcome carrying this sample's support, whatever happened."""
        return CalibrationOutcome(
            status=status,
            method=method,
            state=state,
            failing_requirements=failing,
            row_count=sample.row_count,
            positive_count=sample.positive_count,
            negative_count=sample.negative_count,
            distinct_score_count=sample.distinct_score_count,
        )

    if config.method is CalibrationMethod.NONE:
        return outcome(CalibrationStatus.NOT_CALIBRATED, CalibrationMethod.NONE)

    failures = _support_failures(sample, config=config)
    if failures:
        return outcome(
            CalibrationStatus.INSUFFICIENT_CALIBRATION_SUPPORT,
            config.method,
            failing=tuple(failures),
        )

    targets = tuple(1.0 if value else 0.0 for value in sample.malicious)
    platt: PlattParameters | None = None
    isotonic: IsotonicParameters | None = None
    if config.method is CalibrationMethod.PLATT:
        platt = _fit_platt(
            sample.scores,
            targets,
            positives=sample.positive_count,
            negatives=sample.negative_count,
            max_iter=config.platt_max_iter,
            tolerance=config.platt_tolerance,
        )
        if platt is None:
            return outcome(
                CalibrationStatus.CONVERGENCE_FAILED,
                config.method,
                failing=("platt_convergence",),
            )
    else:
        isotonic = _fit_isotonic(sample.scores, targets)
        if isotonic is None:
            return outcome(
                CalibrationStatus.INSUFFICIENT_CALIBRATION_SUPPORT,
                config.method,
                failing=("isotonic_distinct_breakpoints",),
            )

    state = CalibrationState.seal(
        method=config.method,
        source_score_kind=ScoreKind.DECISION_SCORE,
        output_score_kind=ScoreKind.CALIBRATED_PROBABILITY,
        fit_source_partition=ValidationPartition.VALIDATION_A,
        positive_class=POSITIVE_CLASS,
        platt=platt,
        isotonic=isotonic,
        input_support_min=min(sample.scores),
        input_support_max=max(sample.scores),
        validation_partition_fingerprint=sample.source.source_fingerprint,
        model_id=sample.model_id,
        model_content_fingerprint=sample.model_content_fingerprint,
        preprocessor_fingerprint=sample.preprocessor_fingerprint,
        ml_config_fingerprint=ml_config_fingerprint,
        **counts,
    )
    return outcome(CalibrationStatus.FITTED, config.method, state=state)


def _support_failures(
    sample: BinaryScoreSample, *, config: CalibrationConfig
) -> list[str]:
    """Return the stable requirement codes validation-A fails.

    Checked before any solver runs.  Fitting first and judging afterwards would
    produce a calibrator nobody should use, which somebody would eventually use.
    """
    failures: list[str] = []
    if sample.row_count < config.min_calibration_rows:
        failures.append("min_calibration_rows")
    if sample.positive_count == 0 or sample.negative_count == 0:
        failures.append("both_classes_required")
    if sample.distinct_score_count < 2:
        failures.append("min_distinct_scores")
    if (
        config.method is CalibrationMethod.ISOTONIC
        and sample.distinct_score_count < config.min_isotonic_distinct_scores
    ):
        failures.append("min_isotonic_distinct_scores")
    return failures


def _fit_platt(
    scores: Sequence[float],
    targets: Sequence[float],
    *,
    positives: int,
    negatives: int,
    max_iter: int,
    tolerance: float,
) -> PlattParameters | None:
    """Return the fitted logistic scalars, or ``None`` if the solve failed.

    Newton-Raphson on the two-parameter log-loss, with the **smoothed targets**
    of Platt (1999): a positive row is fitted against ``(N+ + 1) / (N+ + 2)``
    rather than against ``1``.  The smoothing is not cosmetic.  Without it the
    maximum-likelihood estimate does not exist whenever the two classes are
    perfectly separated by the score -- ``a`` diverges -- and perfect separation
    is entirely possible on generated data.  With it, the optimum is finite for
    every input, and the price is that no calibrated probability ever reaches
    exactly zero or one, which is the correct behaviour for a fit over finitely
    many rows anyway.

    Deterministic throughout: a fixed start, a fixed step rule, a fixed
    convergence test, and no random state anywhere.
    """
    high_target = (positives + 1.0) / (positives + 2.0)
    low_target = 1.0 / (negatives + 2.0)
    smoothed = tuple(high_target if value > 0.5 else low_target for value in targets)

    a = 0.0
    b = math.log((positives + 1.0) / (negatives + 1.0))
    iterations = 0
    gradient_norm = math.inf

    for step in range(1, max_iter + 1):
        iterations = step
        grad_a = 0.0
        grad_b = 0.0
        hess_aa = 0.0
        hess_ab = 0.0
        hess_bb = 0.0
        for score, target in zip(scores, smoothed, strict=True):
            probability = sigmoid(a * score + b)
            residual = probability - target
            weight = probability * (1.0 - probability)
            grad_a += residual * score
            grad_b += residual
            hess_aa += weight * score * score
            hess_ab += weight * score
            hess_bb += weight

        gradient_norm = max(abs(grad_a), abs(grad_b))
        if gradient_norm <= tolerance:
            break

        determinant = hess_aa * hess_bb - hess_ab * hess_ab
        if not math.isfinite(determinant) or abs(determinant) <= 0.0:
            return None
        delta_a = (hess_bb * grad_a - hess_ab * grad_b) / determinant
        delta_b = (hess_aa * grad_b - hess_ab * grad_a) / determinant
        if not (math.isfinite(delta_a) and math.isfinite(delta_b)):
            return None

        # Deterministic step halving. A full Newton step can overshoot on a
        # nearly separated sample; halving until the loss actually decreases is
        # a fixed rule rather than a tuning knob, and it terminates.
        current = _platt_loss(scores, smoothed, a, b)
        scale = 1.0
        accepted = False
        for _ in range(_MAX_STEP_HALVINGS):
            candidate_a = a - scale * delta_a
            candidate_b = b - scale * delta_b
            if _platt_loss(scores, smoothed, candidate_a, candidate_b) < current:
                a, b = candidate_a, candidate_b
                accepted = True
                break
            scale /= 2.0
        if not accepted:
            # No downhill step exists at this point. With a finite optimum that
            # means the gradient is already numerically zero, which the test
            # above did not accept -- so the solve has stalled and says so.
            return None
    else:
        return None

    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return PlattParameters(
        a=quantize(a),
        b=quantize(b),
        iterations=iterations,
        final_gradient_norm=quantize(gradient_norm),
    )


#: Halvings tried before a Newton step is declared unusable.  ``2**-30`` is far
#: below any step that could still change a float64 parameter meaningfully.
_MAX_STEP_HALVINGS: Final[int] = 30


def _platt_loss(
    scores: Sequence[float], targets: Sequence[float], a: float, b: float
) -> float:
    """Return the smoothed-target log-loss at ``(a, b)``.

    Written in the numerically stable form: for a large positive ``z`` the
    naive ``log(1 + exp(z))`` overflows, and ``z + log1p(exp(-z))`` does not.
    """
    total = 0.0
    for score, target in zip(scores, targets, strict=True):
        z = a * score + b
        stable = z + math.log1p(math.exp(-z)) if z >= 0.0 else math.log1p(math.exp(z))
        total += stable - target * z
    return total


def _fit_isotonic(
    scores: Sequence[float], targets: Sequence[float]
) -> IsotonicParameters | None:
    """Return the fitted monotone breakpoints, or ``None`` if there are too few.

    The classical weighted pool-adjacent-violators algorithm, in the order the
    audited scikit-learn release applies it:

    1. sort by score, breaking ties by target;
    2. collapse duplicate scores into one point whose target is the mean of
       theirs and whose weight is their count -- so a score seen forty times
       weighs forty, and the same forty rows presented in any order give the
       same fit;
    3. pool adjacent violators until the targets are non-decreasing;
    4. clamp to ``[0, 1]``;
    5. drop interior breakpoints whose value equals both neighbours, since a
       point in the middle of a flat run changes no interpolation.

    Step 5 is why the stored arrays match ``X_thresholds_`` and
    ``y_thresholds_`` rather than being merely equivalent to them.
    """
    ordered = sorted(zip(scores, targets, strict=True))
    unique_x: list[float] = []
    sums: list[float] = []
    weights: list[float] = []
    for score, target in ordered:
        if unique_x and score == unique_x[-1]:
            sums[-1] += target
            weights[-1] += 1.0
        else:
            unique_x.append(score)
            sums.append(target)
            weights.append(1.0)

    if len(unique_x) < 2:
        return None

    means = [total / weight for total, weight in zip(sums, weights, strict=True)]
    pooled = _pool_adjacent_violators(means, weights)
    clamped = [min(1.0, max(0.0, value)) for value in pooled]

    keep = [True] * len(clamped)
    for index in range(1, len(clamped) - 1):
        if (
            clamped[index] == clamped[index - 1]
            and clamped[index] == clamped[index + 1]
        ):
            keep[index] = False
    kept_x = [x for x, flag in zip(unique_x, keep, strict=True) if flag]
    kept_y = [y for y, flag in zip(clamped, keep, strict=True) if flag]
    if len(kept_x) < 2:
        return None

    return IsotonicParameters(
        x_thresholds=tuple(kept_x),
        y_thresholds=tuple(quantize(value) for value in kept_y),
    )


def _pool_adjacent_violators(
    values: Sequence[float], weights: Sequence[float]
) -> list[float]:
    """Return the weighted isotonic (non-decreasing) projection of *values*.

    Blocks are merged left to right; each block carries its weighted mean and
    total weight, and a new point that would violate monotonicity is absorbed
    into the block before it until it does not.  The projection is unique, so
    any correct implementation lands on the same answer -- which is what makes
    comparing this one against scikit-learn a real check rather than a
    tautology.
    """
    block_means: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []
    for value, weight in zip(values, weights, strict=True):
        mean = value
        total = weight
        size = 1
        while block_means and block_means[-1] > mean:
            previous_mean = block_means.pop()
            previous_weight = block_weights.pop()
            size += block_sizes.pop()
            combined = previous_weight + total
            mean = (previous_mean * previous_weight + mean * total) / combined
            total = combined
        block_means.append(mean)
        block_weights.append(total)
        block_sizes.append(size)

    projected: list[float] = []
    for mean, size in zip(block_means, block_sizes, strict=True):
        projected.extend([mean] * size)
    return projected


# ---------------------------------------------------------------------------
# The probability vocabulary transition
# ---------------------------------------------------------------------------


def apply_calibration(
    state: CalibrationState,
    sample: BinaryScoreSample,
    *,
    ml_config_fingerprint: str,
) -> BinaryScoreSample:
    """Return *sample* with its scores mapped onto calibrated probabilities.

    The one place an uncalibrated decision score becomes a
    ``calibrated_probability``, and it is a construction rather than a rename:
    the returned sample carries the calibrator's fingerprint and its method, so
    every downstream consumer can see which calibrator produced the numbers it
    is reading.

    Applying a calibrator is inference, not fitting, so it is permitted on
    validation-B -- that is how the operating point is chosen on calibrated
    scores.  It is *not* permitted on test or holdout rows: reading those is a
    Milestone 6 evaluation with its own record, and doing it here would be an
    evaluation nobody recorded.

    Raises:
        ModelTrainingError: on a source outside the validation split, a sample
            that is already calibrated, or a broken provenance chain.
    """
    stage = "calibration application"
    if sample.source.split is not MLSplit.VALIDATION:
        raise ModelTrainingError(
            f"{stage} reads validation rows only; it was handed "
            f"{sample.source.describe!r}"
        )
    if sample.score_kind is not ScoreKind.DECISION_SCORE:
        raise ModelTrainingError(
            f"{stage} consumes the frozen decision score; it was handed "
            f"{str(sample.score_kind)!r}"
        )
    require_chain(
        stage=stage,
        model_content_fingerprint=sample.model_content_fingerprint,
        preprocessor_fingerprint=sample.preprocessor_fingerprint,
        source_fingerprint=sample.source.source_fingerprint,
        ml_config_fingerprint=ml_config_fingerprint,
        state=state,
    )
    # Constructed rather than copied with an update: a copy skips validation,
    # and the invariants tying the calibrated vocabulary to a named calibrator
    # are exactly the ones worth re-checking at the moment they start applying.
    return BinaryScoreSample(
        source=sample.source,
        anchors=sample.anchors,
        scores=state.transform(sample.scores),
        malicious=sample.malicious,
        score_kind=ScoreKind.CALIBRATED_PROBABILITY,
        model_id=sample.model_id,
        model_content_fingerprint=sample.model_content_fingerprint,
        preprocessor_fingerprint=sample.preprocessor_fingerprint,
        calibration_state_fingerprint=state.calibration_state_fingerprint,
        calibration_method=state.method,
    )


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


class ReliabilityBin(BaseModel):
    """One bin of a reliability summary: counts, and what they support.

    An empty bin is present and says it is empty.  Rendering it as a zero
    observed rate would put a point on a reliability curve that no row voted
    for, and a reader comparing the curve against the diagonal would be
    comparing against fabricated evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    lower_edge: float
    upper_edge: float
    #: Only the last bin is closed on the right, so ``1.0`` lands somewhere.
    includes_upper_edge: bool
    row_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    mean_predicted_probability: float | None
    observed_positive_rate: float | None
    status: MetricStatus

    @model_validator(mode="after")
    def check_bin(self) -> Self:
        """An empty bin reports nothing; a populated one reports both rates."""
        if self.positive_count > self.row_count:
            raise ValueError("a bin cannot hold more positives than rows")
        empty = self.row_count == 0
        if empty and self.status is not MetricStatus.UNAVAILABLE:
            raise ValueError(
                "a bin holding no rows is unavailable, not a measurement of zero"
            )
        if empty != (self.mean_predicted_probability is None):
            raise ValueError("an empty bin has no mean predicted probability")
        if empty != (self.observed_positive_rate is None):
            raise ValueError("an empty bin has no observed positive rate")
        if not empty and self.status is MetricStatus.UNAVAILABLE:
            raise ValueError("a populated bin is not unavailable")
        return self


class CalibrationReport(SealedModel):
    """Aggregate calibration quality, and -- crucially -- what it is evidence of.

    Counts, rates, and digests.  No row, no identifier, no path.

    Two things a reader must not have to infer.  ``evaluation_kind`` says
    whether the calibrator was measured on the rows that fitted it or on rows it
    had never seen, and ``admissible_as_champion_evidence`` says whether the
    result may be offered to a later model-selection gate.  An in-sample report
    is inspectable and useful and is **never** admissible; the field is pinned
    false for it by a validator rather than left to a caller's discipline.

    The status is not decoration either.  ``ECE`` computed over eighty rows can
    be small for no better reason than that eighty rows cannot resolve a
    miscalibration, and a report that returned only the number would let that
    read as evidence of calibration.
    """

    fingerprint_field: ClassVar[str] = "report_fingerprint"
    schema_version_field: ClassVar[str] = "calibration_schema_version"
    schema_version: ClassVar[str] = CALIBRATION_SCHEMA_VERSION
    record_label: ClassVar[str] = "calibration report"

    calibration_schema_version: str = CALIBRATION_SCHEMA_VERSION
    method: CalibrationMethod
    #: What this report is evidence of.  Tied to :attr:`source_partition` by a
    #: validator in both directions, so neither can be edited into agreeing
    #: with a claim the other does not support.
    evaluation_kind: CalibrationEvaluationKind
    status: MetricStatus
    #: Where the calibrator was **measured**.  Validation-A for a fit
    #: diagnostic, validation-B for authoritative quality.
    source_partition: ValidationPartition
    #: Where the calibrator was **fitted**, carried through from its state so a
    #: report stands alone as a record of the protocol that produced it.
    fit_source_partition: ValidationPartition
    #: Whether this report may be offered to a later champion calibration gate.
    #: False for every in-sample diagnostic, whatever it measured.
    admissible_as_champion_evidence: bool

    brier_score: float | None
    expected_calibration_error: float | None
    #: Whether the measured error clears the configured ceiling.  ``None``
    #: whenever the error itself is not a measurement, because a comparison
    #: against an unmeasured quantity is not a verdict.
    within_configured_error: bool | None
    max_expected_calibration_error: float

    #: The same Brier score computed on the **uncalibrated** model output, for
    #: comparison.  Available only when the model's declared score contract is
    #: bounded to ``[0, 1]``: a Brier score needs a forecast on that interval,
    #: and clipping an arbitrary decision score into it to manufacture a
    #: comparison would compare against a number nothing produced.
    raw_score_brier: float | None
    raw_score_brier_status: MetricStatus
    #: A stable code naming why no raw comparator exists, or ``None``.
    raw_score_brier_unavailable_reason: str | None
    #: The uncalibrated kind that was compared.  Still a ``decision_score``:
    #: computing a Brier score against it does not make it a probability, and
    #: nothing here renames it.
    raw_score_kind: ScoreKind | None

    bin_count: int = Field(ge=2)
    bins: tuple[ReliabilityBin, ...]
    row_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    min_calibration_rows: int = Field(ge=1)
    min_reliability_bin_rows: int = Field(ge=1)
    failing_requirements: tuple[str, ...]

    validation_partition_fingerprint: Sha256Hex
    model_content_fingerprint: Sha256Hex
    calibration_state_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    report_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_report(self) -> Self:
        """Bins are complete, the kind matches the partition, the verdict follows."""
        if self.calibration_schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"calibration report declares schema version "
                f"{self.calibration_schema_version!r}; this build implements "
                f"{CALIBRATION_SCHEMA_VERSION!r}"
            )
        if self.method is CalibrationMethod.NONE:
            raise ValueError(
                "there is no calibration report for an uncalibrated model; the "
                "absence of a calibrator is not a measurement of one"
            )
        if self.fit_source_partition is not ValidationPartition.VALIDATION_A:
            raise ValueError("a calibrator is fitted on validation-A")

        expected_partition = {
            CalibrationEvaluationKind.IN_SAMPLE_FIT_DIAGNOSTIC: (
                ValidationPartition.VALIDATION_A
            ),
            CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION: (
                ValidationPartition.VALIDATION_B
            ),
        }[self.evaluation_kind]
        if self.source_partition is not expected_partition:
            raise ValueError(
                f"an {str(self.evaluation_kind)!r} report is measured on "
                f"{str(expected_partition)!r}; a report measured elsewhere is "
                f"evidence of something else"
            )

        measured = self.status is MetricStatus.MEASURED
        in_sample = (
            self.evaluation_kind is CalibrationEvaluationKind.IN_SAMPLE_FIT_DIAGNOSTIC
        )
        if in_sample and self.admissible_as_champion_evidence:
            raise ValueError(
                "an in-sample fit diagnostic is never champion calibration-"
                "quality evidence: the calibrator was fitted on exactly these "
                "rows, so a good result here says nothing about generalisation"
            )
        if not in_sample and self.admissible_as_champion_evidence != (
            measured and bool(self.within_configured_error)
        ):
            raise ValueError(
                "an out-of-sample report is admissible exactly when its error "
                "was measured on adequate support and cleared the configured "
                "ceiling"
            )

        if len(self.bins) != self.bin_count:
            raise ValueError("the reliability summary is missing bins")
        if tuple(item.index for item in self.bins) != tuple(range(self.bin_count)):
            raise ValueError("reliability bins must be given in index order")
        if sum(item.row_count for item in self.bins) != self.row_count:
            raise ValueError("the bins do not account for every row")
        if self.row_count != self.positive_count + self.negative_count:
            raise ValueError("class support does not sum to the row count")
        if self.status is MetricStatus.UNAVAILABLE and (
            self.brier_score is not None or self.expected_calibration_error is not None
        ):
            raise ValueError(
                "an unavailable report carries no numbers; there was nothing to measure"
            )
        if measured and (
            self.brier_score is None or self.expected_calibration_error is None
        ):
            raise ValueError("a measured report carries both aggregate numbers")
        if measured != (self.within_configured_error is not None):
            raise ValueError(
                "the calibration verdict exists exactly when the error is a measurement"
            )
        if measured and self.failing_requirements:
            raise ValueError("a measured report names no failing requirement")

        raw_measured = self.raw_score_brier_status is MetricStatus.MEASURED
        if raw_measured != (self.raw_score_brier is not None):
            raise ValueError(
                "a raw-score Brier comparator exists exactly when it was measured"
            )
        if raw_measured != (self.raw_score_kind is not None):
            raise ValueError(
                "a raw comparator names the uncalibrated kind it compared against"
            )
        if raw_measured == (self.raw_score_brier_unavailable_reason is not None):
            raise ValueError(
                "an unavailable raw comparator names why, and a measured one "
                "has no reason to give"
            )
        if self.raw_score_kind is not None and is_probability(self.raw_score_kind):
            raise ValueError(
                "the raw comparator compares against an uncalibrated score; a "
                "Brier score computed on it does not make it a probability"
            )
        return self

    @property
    def populated_bins(self) -> tuple[ReliabilityBin, ...]:
        """Return the bins that any row landed in."""
        return tuple(item for item in self.bins if item.row_count > 0)

    @property
    def improves_on_raw_score(self) -> bool | None:
        """Return whether calibration lowered the Brier score, where comparable.

        ``None`` when no mathematically valid comparator exists, which is the
        common case and not a failure: most decision scores carry no promise of
        living on ``[0, 1]``, and a comparison against a clipped one would be a
        comparison against a number the model never produced.
        """
        if self.brier_score is None or self.raw_score_brier is None:
            return None
        return self.brier_score < self.raw_score_brier


def require_out_of_sample_evidence(
    report: CalibrationReport, *, stage: str
) -> CalibrationReport:
    """Return *report* if it may stand as calibration-quality evidence, else raise.

    The guard a later champion gate calls instead of reading two fields and
    hoping.  It is not the gate: it decides nothing about a model, applies no
    threshold, and expresses no preference.  It only refuses to let an
    in-sample diagnostic be passed off as an out-of-sample measurement.

    Raises:
        ModelTrainingError: if the report is an in-sample fit diagnostic, or if
            it is out-of-sample but its own contract already marks it
            inadmissible.
    """
    if report.evaluation_kind is not CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION:
        raise ModelTrainingError(
            f"{stage} requires an out-of-sample calibration measurement; it was "
            f"handed an {str(report.evaluation_kind)!r} report. A calibrator "
            f"measured on the rows that fitted it describes those rows"
        )
    if not report.admissible_as_champion_evidence:
        raise ModelTrainingError(
            f"{stage} was handed an out-of-sample report that is not admissible "
            f"[{report.status!s}]: {list(report.failing_requirements)}"
        )
    return report


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Return the mean squared difference between forecast and outcome.

    A split-agnostic numeric primitive.  It knows nothing about provenance,
    which is what lets it be the one implementation behind a validation-A
    diagnostic, a validation-B measurement, and -- once a champion is frozen
    and a Milestone 6 evaluation record exists to hold it -- a test one.
    """
    return sum(
        (probability - (1.0 if label else 0.0)) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(probabilities)


def _raw_score_comparator(
    scores: Sequence[float],
    labels: Sequence[bool],
    semantics: ScoreSemantics | None,
) -> tuple[float | None, MetricStatus, str | None, ScoreKind | None]:
    """Return the Brier score of the **uncalibrated** output, where that is valid.

    A Brier score is the mean squared error of a forecast on ``[0, 1]``.  A
    decision score is an ordered magnitude, and most of them carry no promise
    of living on that interval -- so for most models there is simply no
    comparator, and the report says so rather than inventing one.

    The one thing never done here is **clipping**.  Squeezing an arbitrary
    decision score into ``[0, 1]`` would produce a number, and that number would
    be the Brier score of a forecast the model never made.  A comparison
    against it would flatter or damn calibration according to how far outside
    the interval the raw scores happened to fall.

    Nothing here renames anything either.  When a comparator does exist, the
    field it was computed from is still a ``decision_score``, and the report
    records that kind alongside the number.
    """
    if semantics is None:
        return (None, MetricStatus.UNAVAILABLE, "raw_score_contract_not_supplied", None)
    if is_probability(semantics.score_kind):
        return (
            None,
            MetricStatus.UNAVAILABLE,
            "raw_score_is_already_calibrated",
            None,
        )
    if semantics.lower_bound < 0.0 or semantics.upper_bound > 1.0:
        return (None, MetricStatus.UNAVAILABLE, "raw_score_not_unit_bounded", None)
    if any(not 0.0 <= score <= 1.0 for score in scores):
        # The contract says bounded and the observations disagree. That is a
        # fault in the model that produced them, and the honest answer is no
        # comparator -- not a clipped one that would hide the disagreement.
        return (
            None,
            MetricStatus.UNAVAILABLE,
            "raw_score_outside_declared_bounds",
            None,
        )
    if not scores:
        return (None, MetricStatus.UNAVAILABLE, "no_rows", None)
    return (
        quantize(brier_score(scores, labels)),
        MetricStatus.MEASURED,
        None,
        semantics.score_kind,
    )


def diagnose_calibration_fit(
    sample: BinaryScoreSample,
    *,
    state: CalibrationState,
    config: CalibrationConfig,
    ml_config_fingerprint: str,
    raw_score_semantics: ScoreSemantics | None = None,
) -> CalibrationReport:
    """Measure a fitted calibrator on the validation-A rows that fitted it.

    A **fit diagnostic**, and labelled as one.  A calibrator asked to describe
    the rows it was shaped by will describe them well whether or not it
    generalises, so this number cannot tell anybody whether the calibration is
    good.  What it can do is expose a fit that went wrong -- a monotone map that
    collapsed to a constant, a Platt fit that landed somewhere absurd -- and
    that is worth having.

    The returned report carries
    :attr:`~password_attack_detector.ml.enums.CalibrationEvaluationKind.IN_SAMPLE_FIT_DIAGNOSTIC`
    and ``admissible_as_champion_evidence: false``, pinned by a validator.  For
    calibration quality that a later gate may act on, see
    :func:`evaluate_calibration_quality`.

    Raises:
        ModelTrainingError: on a source outside validation-A, uncanonical rows,
            an already-calibrated sample, or a broken provenance chain.
    """
    return _calibration_report(
        sample,
        state=state,
        config=config,
        ml_config_fingerprint=ml_config_fingerprint,
        raw_score_semantics=raw_score_semantics,
        kind=CalibrationEvaluationKind.IN_SAMPLE_FIT_DIAGNOSTIC,
        partition=ValidationPartition.VALIDATION_A,
        stage="calibration fit diagnosis",
    )


def evaluate_calibration_quality(
    sample: BinaryScoreSample,
    *,
    state: CalibrationState,
    config: CalibrationConfig,
    ml_config_fingerprint: str,
    raw_score_semantics: ScoreSemantics | None = None,
) -> CalibrationReport:
    """Measure a frozen calibrator on validation-B, which it has never seen.

    The authoritative calibration-quality measurement, and the only kind a
    later champion gate may act on.  The calibrator is **applied**, never
    refitted: it arrives frozen, nothing here can mutate it --
    :class:`CalibrationState` has no mutable field -- and a test asserts its
    bytes and its digest are identical before and after.

    Validation-B is already the half that chooses the operating point, so
    measuring calibration here spends no additional data. It does mean the same
    rows that measure calibration also select a threshold; that is a real
    limitation of a two-way split and is stated in the model contract rather
    than worked around by borrowing rows from somewhere they are not allowed to
    come from.

    Args:
        sample: validation-B **decision** scores; the calibrator is applied
            here rather than by the caller, so the numbers measured are
            provably this calibrator's.
        state: the frozen calibrator, fitted on validation-A.
        config: the bin count, the support floors, and the error ceiling.
        ml_config_fingerprint: the configuration this run is carried out under.
        raw_score_semantics: the model's declared score contract, supplied when
            a raw-score Brier comparator is wanted.  Omitted, there is no
            comparator and the report says why.

    Raises:
        ModelTrainingError: on a source outside validation-B, uncanonical rows,
            an already-calibrated sample, or a calibrator whose validation-A fit
            belongs to a different parent partition, model, preprocessor, or
            configuration.
    """
    return _calibration_report(
        sample,
        state=state,
        config=config,
        ml_config_fingerprint=ml_config_fingerprint,
        raw_score_semantics=raw_score_semantics,
        kind=CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION,
        partition=ValidationPartition.VALIDATION_B,
        stage="calibration quality evaluation",
    )


def _calibration_report(
    sample: BinaryScoreSample,
    *,
    state: CalibrationState,
    config: CalibrationConfig,
    ml_config_fingerprint: str,
    raw_score_semantics: ScoreSemantics | None,
    kind: CalibrationEvaluationKind,
    partition: ValidationPartition,
    stage: str,
) -> CalibrationReport:
    """Measure *state* on *sample* and label the result with what it is evidence of.

    The shared body of both public evaluations.  The measurement is identical;
    everything that differs between them -- which partition is admissible, what
    the result may be used for -- is provenance, and provenance is attached
    here rather than left to whichever caller assembled the numbers.
    """
    require_partition(sample.source, expected=partition, stage=stage)
    sample.require_canonical(stage=stage)
    if sample.score_kind is not ScoreKind.DECISION_SCORE:
        raise ModelTrainingError(
            f"{stage} consumes the frozen decision score and applies the "
            f"calibrator itself; it was handed {str(sample.score_kind)!r}"
        )
    # The parent-partition link is the one that matters here: validation-A and
    # validation-B carry the same digest only when they came from one
    # campaign-disjoint partitioning, so a calibrator fitted on an unrelated
    # dataset's validation-A cannot be measured on this validation-B.
    require_chain(
        stage=stage,
        model_content_fingerprint=sample.model_content_fingerprint,
        preprocessor_fingerprint=sample.preprocessor_fingerprint,
        source_fingerprint=sample.source.source_fingerprint,
        ml_config_fingerprint=ml_config_fingerprint,
        state=state,
    )

    probabilities = state.transform(sample.scores)
    labels = sample.malicious
    bins = _reliability_bins(
        probabilities,
        labels,
        bin_count=config.reliability_bin_count,
        min_bin_rows=config.min_reliability_bin_rows,
    )

    failures: list[str] = []
    if sample.row_count < config.min_calibration_rows:
        failures.append("min_calibration_rows")
    if sample.positive_count == 0 or sample.negative_count == 0:
        failures.append("both_classes_required")

    if sample.row_count == 0:
        status = MetricStatus.UNAVAILABLE
        brier: float | None = None
        ece: float | None = None
        verdict: bool | None = None
    else:
        brier = quantize(brier_score(probabilities, labels))
        ece = quantize(_expected_calibration_error(bins, sample.row_count))
        status = (
            MetricStatus.INSUFFICIENT_SUPPORT if failures else MetricStatus.MEASURED
        )
        verdict = (
            None
            if status is not MetricStatus.MEASURED
            else ece <= config.max_expected_calibration_error
        )

    raw_brier, raw_status, raw_reason, raw_kind = _raw_score_comparator(
        sample.scores, labels, raw_score_semantics
    )
    admissible = kind is CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION and bool(
        verdict
    )

    return CalibrationReport.seal(
        method=state.method,
        evaluation_kind=kind,
        status=status,
        source_partition=partition,
        fit_source_partition=state.fit_source_partition,
        admissible_as_champion_evidence=admissible,
        brier_score=brier,
        expected_calibration_error=ece,
        within_configured_error=verdict,
        max_expected_calibration_error=quantize(config.max_expected_calibration_error),
        raw_score_brier=raw_brier,
        raw_score_brier_status=raw_status,
        raw_score_brier_unavailable_reason=raw_reason,
        raw_score_kind=raw_kind,
        bin_count=config.reliability_bin_count,
        bins=bins,
        row_count=sample.row_count,
        positive_count=sample.positive_count,
        negative_count=sample.negative_count,
        min_calibration_rows=config.min_calibration_rows,
        min_reliability_bin_rows=config.min_reliability_bin_rows,
        failing_requirements=tuple(failures),
        validation_partition_fingerprint=sample.source.source_fingerprint,
        model_content_fingerprint=sample.model_content_fingerprint,
        calibration_state_fingerprint=state.calibration_state_fingerprint,
        ml_config_fingerprint=ml_config_fingerprint,
    )


def _reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    bin_count: int,
    min_bin_rows: int,
) -> tuple[ReliabilityBin, ...]:
    """Return equal-width reliability bins over ``[0, 1]``.

    Edges are ``i / bin_count``, fixed by configuration rather than derived
    from the data.  Quantile bins would move with the score distribution, so
    two runs over different data could not be compared bin for bin -- and
    comparison is the whole point of a reliability summary.

    Bin ``i`` covers ``[i/n, (i+1)/n)``; the last bin is closed on the right so
    a probability of exactly ``1.0`` lands somewhere.
    """
    totals = [0] * bin_count
    positives = [0] * bin_count
    predicted = [0.0] * bin_count
    for probability, label in zip(probabilities, labels, strict=True):
        index = min(int(probability * bin_count), bin_count - 1)
        totals[index] += 1
        predicted[index] += probability
        if label:
            positives[index] += 1

    bins: list[ReliabilityBin] = []
    for index in range(bin_count):
        rows = totals[index]
        if rows == 0:
            status = MetricStatus.UNAVAILABLE
        elif rows < min_bin_rows:
            status = MetricStatus.INSUFFICIENT_SUPPORT
        else:
            status = MetricStatus.MEASURED
        bins.append(
            ReliabilityBin(
                index=index,
                lower_edge=quantize(index / bin_count),
                upper_edge=quantize((index + 1) / bin_count),
                includes_upper_edge=index == bin_count - 1,
                row_count=rows,
                positive_count=positives[index],
                mean_predicted_probability=(
                    None if rows == 0 else quantize(predicted[index] / rows)
                ),
                observed_positive_rate=(
                    None if rows == 0 else quantize(positives[index] / rows)
                ),
                status=status,
            )
        )
    return tuple(bins)


def _expected_calibration_error(
    bins: Sequence[ReliabilityBin], row_count: int
) -> float:
    """Return the support-weighted mean gap between predicted and observed rates.

    Empty bins contribute nothing, because there is no gap to weigh.  That is
    the definition rather than a convenience: an ECE that counted an empty bin
    as a perfect one would improve every time the data got sparser.
    """
    if row_count == 0:
        return 0.0
    total = 0.0
    for item in bins:
        if item.row_count == 0:
            continue
        assert item.mean_predicted_probability is not None
        assert item.observed_positive_rate is not None
        gap = abs(item.mean_predicted_probability - item.observed_positive_rate)
        total += (item.row_count / row_count) * gap
    return total


# ---------------------------------------------------------------------------
# Privacy sweep
# ---------------------------------------------------------------------------

#: Every schema in this module that may be published.  The input samples are
#: deliberately absent: fitting a calibrator is a supervised operation and its
#: input legitimately carries labels and anchors.  Nothing derived from them
#: may, and this is where that is enforced.
_PUBLISHED_SCHEMAS: Final[tuple[type[BaseModel], ...]] = (
    CalibrationReport,
    CalibrationState,
    IsotonicParameters,
    PlattParameters,
    ReliabilityBin,
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
