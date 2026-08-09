"""The model-adapter contract: what a family must provide to be publishable.

An adapter has two halves, and the split is the whole design.

**Fitting** may use scikit-learn.  It consumes an already-prepared training
batch -- a transformed matrix, a target column, the frozen preprocessor, the
class weights -- and produces a :class:`FittedModel`: pure numbers, pure
metadata, no estimator object.

**Scoring** may not.  Every adapter reconstructs its scores from the canonical
artifact alone, in project code, with no estimator present.  That is what makes
the round-trip parity test meaningful: the two paths do not share an
implementation, so agreement between them is evidence rather than tautology.

**What an adapter may not see.**  No Parquet, no label file, no split file, no
campaign metadata, no feature manifest.  A batch arrives already assembled by
Milestone 2, already transformed by Milestone 3, already canonically ordered --
and the fit asserts that ordering rather than trusting it.

**The preprocessor is an input, never a workspace.**  It arrives frozen and is
carried through by fingerprint only; nothing here can modify it, because
:class:`~password_attack_detector.ml.preprocessing.FittedPreprocessor` has no
mutable field.

**Vocabulary.**  Nothing in this milestone produces a probability.  A binary
head emits a ``decision_score``, a category head emits ``class_score``, and the
anomaly head emits ``anomaly_score``.  Milestone 5 fits calibrators; until then
the word is not available, and :class:`ScoreSemantics` refuses prose that uses
it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol, Self, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import (
    FIT_ELIGIBLE_SPLITS,
    CalibrationMethod,
    MLSplit,
    MLTask,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.imbalance import ClassWeightState
from password_attack_detector.ml.npz import array_digest, normalize_array
from password_attack_detector.ml.ordering import AnchoredRow, assert_canonical
from password_attack_detector.ml.preprocessing import FittedPreprocessor
from password_attack_detector.ml.schemas import ScoreSemantics

__all__ = [
    "ANOMALY_SCORE_COLUMN",
    "FittedModel",
    "ModelAdapter",
    "TrainingBatch",
    "score_semantics_for",
]

#: The single column an anomaly adapter emits.  Named rather than borrowed from
#: the class order, because an unsupervised model has no classes.
ANOMALY_SCORE_COLUMN: Final[str] = "anomaly_score"

#: Fitted parameters are stored at this precision, matching every other
#: fingerprint in the project.
_FLOAT_PRECISION: Final[int] = 9


def score_semantics_for(task: MLTask) -> ScoreSemantics:
    """Return what a Milestone 4 model's output means for *task*.

    Never a probability.  A logistic sigmoid produces a number in ``[0, 1]``
    that looks exactly like one, and calling it a probability before a
    calibrator has been fitted *and measured* is the single easiest way for this
    layer to mislead somebody. Milestone 5 changes the score kind; nothing here
    does.
    """
    match task:
        case MLTask.BINARY_MALICIOUS:
            return ScoreSemantics(
                score_kind=ScoreKind.DECISION_SCORE,
                calibration_method=CalibrationMethod.NONE,
                lower_bound=0.0,
                upper_bound=1.0,
                description=(
                    "Uncalibrated ordered magnitude for the malicious class. "
                    "Useful for ranking; no calibrator has been fitted, so it "
                    "carries no distributional meaning."
                ),
            )
        case MLTask.ATTACK_CATEGORY:
            return ScoreSemantics(
                score_kind=ScoreKind.CLASS_SCORE,
                calibration_method=CalibrationMethod.NONE,
                lower_bound=0.0,
                upper_bound=1.0,
                description=(
                    "Uncalibrated per-class ordered magnitude for the triage "
                    "head. Comparable across classes within one row, and not "
                    "comparable to any measured rate."
                ),
            )
        case MLTask.ANOMALY:
            return ScoreSemantics(
                score_kind=ScoreKind.ANOMALY_SCORE,
                calibration_method=CalibrationMethod.NONE,
                lower_bound=-1.0,
                upper_bound=0.0,
                description=(
                    "Unsupervised outlier magnitude under the scikit-learn "
                    "convention, where a lower value is more anomalous."
                ),
            )


def quantize(value: float) -> float:
    """Return *value* at the precision fitted scalars are stored in."""
    if not math.isfinite(value):
        raise ModelTrainingError(
            "a fitted parameter must be finite; NaN and infinity are neither "
            "serialisable nor meaningful"
        )
    return float(f"{value:.{_FLOAT_PRECISION}f}")


class TrainingBatch(BaseModel):
    """Everything an adapter is given, and nothing it is not.

    Assembled by the caller from Milestone 2's dataset and Milestone 3's
    preprocessor.  The target column is a sequence of class *names* rather than
    encoded integers, so a class order is always explicit and an adapter can
    never silently adopt whichever integer encoding a library preferred.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    split: MLSplit
    anchors: tuple[AnchoredRow, ...]
    transformed_feature_names: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    targets: tuple[str, ...]
    class_order: tuple[str, ...]
    preprocessor: FittedPreprocessor
    class_weights: ClassWeightState | None = None
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def check_batch(self) -> Self:
        """Every parallel sequence describes the same rows, in the same order."""
        if self.split not in FIT_ELIGIBLE_SPLITS:
            raise ValueError(
                f"a model may be fitted on "
                f"{sorted(str(item) for item in FIT_ELIGIBLE_SPLITS)} only; this "
                f"batch is {str(self.split)!r}"
            )
        if not self.matrix:
            raise ValueError("a training batch must carry at least one row")
        widths = {len(row) for row in self.matrix}
        if widths != {len(self.transformed_feature_names)}:
            raise ValueError(
                "every row must be as wide as the transformed feature order"
            )
        if len(self.anchors) != len(self.matrix):
            raise ValueError("anchors and matrix rows disagree in length")
        if self.targets and len(self.targets) != len(self.matrix):
            raise ValueError("targets and matrix rows disagree in length")
        if self.transformed_feature_names != self.preprocessor.output_feature_names:
            raise ValueError(
                "the batch's transformed feature order disagrees with the "
                "preprocessor that produced it"
            )
        if self.class_order and len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order repeats a class")
        unknown = sorted(set(self.targets) - set(self.class_order))
        if unknown:
            raise ValueError(
                f"{len(unknown)} target value(s) are outside the declared class "
                f"order, including {unknown[:3]}"
            )
        return self

    @property
    def row_count(self) -> int:
        """Return the number of training rows."""
        return len(self.matrix)

    def require_canonical(self) -> None:
        """Assert the rows are in canonical order, or raise.

        Asserted rather than sorted. Milestone 2 canonicalises on the way out of
        dataset assembly; a fit that quietly re-sorted would hide the fact that
        somebody handed it rows nobody had ordered, and the next stage that
        forgot would be the one producing a wrong answer silently.
        """
        assert_canonical(self.anchors, stage="model fit")

    def design(self) -> np.ndarray[Any, Any]:
        """Return the design matrix as a contiguous little-endian float64 array."""
        return normalize_array("design", np.asarray(self.matrix, dtype=np.float64))

    def encoded_targets(self) -> np.ndarray[Any, Any]:
        """Return targets as class-order indices, so no library picks an order."""
        position = {name: index for index, name in enumerate(self.class_order)}
        return np.asarray([position[value] for value in self.targets], dtype=np.int64)

    def sample_weights(self) -> np.ndarray[Any, Any] | None:
        """Return one weight per row, or ``None`` when nothing is weighted.

        Derived from the Milestone 3 class-weight state rather than from an
        estimator's own ``class_weight='balanced'``: the number that reached the
        loss is then the number recorded in the manifest, and a reader can check
        one against the other.
        """
        if self.class_weights is None:
            return None
        mapping = self.class_weights.as_mapping()
        missing = sorted(set(self.class_order) - set(mapping))
        if missing:
            raise ModelTrainingError(
                f"the class-weight state does not cover class(es) {missing}"
            )
        return np.asarray([mapping[value] for value in self.targets], dtype=np.float64)


class FittedModel(BaseModel):
    """One fitted model's canonical content: numbers and metadata, no estimator.

    This is what gets serialised, fingerprinted, and scored from. There is no
    reference to a scikit-learn object anywhere in it, which is what allows
    inference to run without reconstructing one -- and what makes the artifact
    readable by a future release that no longer has the same internals.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    serializer_id: str
    serializer_version: int = Field(ge=1)
    inference_adapter_id: str
    catalog_model_id: str
    family: ModelFamily
    task: MLTask
    class_order: tuple[str, ...]
    raw_feature_names: tuple[str, ...]
    transformed_feature_names: tuple[str, ...]
    hyperparameters: Mapping[str, bool | int | float | str]
    parameters: Mapping[str, Any]
    arrays: Mapping[str, Any]
    score_semantics: ScoreSemantics
    train_row_count: int = Field(ge=1)
    preprocessor_fingerprint: str
    eligible_feature_list_fingerprint: str
    class_weight_fingerprint: str | None = None
    champion_eligible: bool
    experimental: bool = False

    @model_validator(mode="after")
    def check_content(self) -> Self:
        """The declared shapes agree, and every array is publishable."""
        if not self.transformed_feature_names:
            raise ValueError("a fitted model must declare its transformed features")
        if len(set(self.transformed_feature_names)) != len(
            self.transformed_feature_names
        ):
            raise ValueError("transformed_feature_names repeats a column")
        if self.task is MLTask.ANOMALY:
            if self.class_order:
                raise ValueError(
                    "an anomaly model has no classes; it emits an ordered "
                    "magnitude, not a class"
                )
        elif len(self.class_order) < 2:
            raise ValueError(
                f"task {str(self.task)!r} needs at least two classes, got "
                f"{len(self.class_order)}"
            )
        for name, array in self.arrays.items():
            normalize_array(name, array)
        return self

    @property
    def score_columns(self) -> tuple[str, ...]:
        """Return the columns :meth:`score` emits, in order."""
        if self.task is MLTask.ANOMALY:
            return (ANOMALY_SCORE_COLUMN,)
        return self.class_order

    def content_payload(self) -> dict[str, Any]:
        """Return the canonical semantic content this model is identified by.

        Everything that changes what the model *does*, and nothing that
        describes where it was written or when. The array digest stands in for
        the arrays themselves so the payload stays JSON while still covering
        every fitted number.
        """
        return {
            "array_digest": array_digest(self.arrays),
            "catalog_model_id": self.catalog_model_id,
            "class_order": list(self.class_order),
            "class_weight_fingerprint": self.class_weight_fingerprint,
            "eligible_feature_list_fingerprint": self.eligible_feature_list_fingerprint,
            "family": str(self.family),
            "hyperparameters": {
                key: self.hyperparameters[key] for key in sorted(self.hyperparameters)
            },
            "inference_adapter_id": self.inference_adapter_id,
            "parameters": _canonical(self.parameters),
            "preprocessor_fingerprint": self.preprocessor_fingerprint,
            "raw_feature_names": list(self.raw_feature_names),
            "score_kind": str(self.score_semantics.score_kind),
            "serializer_id": self.serializer_id,
            "serializer_version": self.serializer_version,
            "task": str(self.task),
            "transformed_feature_names": list(self.transformed_feature_names),
        }

    def content_fingerprint(self) -> str:
        """Return the SHA-256 digest that gives this model its identity.

        Semantic, not byte-level: the directory it is written to, the archive's
        compression level, and the moment of publication are all absent, so the
        same model published twice in two places is the same model.
        """
        canonical = json.dumps(
            self.content_payload(), sort_keys=True, ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def require_matrix(
        self, rows: Sequence[Sequence[float]], columns: Sequence[str]
    ) -> None:
        """Raise unless *columns* is exactly the fitted transformed order.

        Order, not membership: the right columns in the wrong order would score
        a different feature in every position and report nothing wrong.
        """
        if tuple(columns) != self.transformed_feature_names:
            missing = sorted(set(self.transformed_feature_names) - set(columns))
            extra = sorted(set(columns) - set(self.transformed_feature_names))
            if missing or extra:
                raise ModelTrainingError(
                    f"the matrix disagrees with the fitted feature contract: "
                    f"{len(missing)} missing, {len(extra)} unexpected"
                )
            raise ModelTrainingError(
                "the matrix carries the fitted features in a different order; "
                "the same columns in a different order encode a different model"
            )
        for row in rows:
            if len(row) != len(self.transformed_feature_names):
                raise ModelTrainingError(
                    f"a row carries {len(row)} value(s) for "
                    f"{len(self.transformed_feature_names)} column(s)"
                )
            for value in row:
                if not math.isfinite(float(value)):
                    raise ModelTrainingError(
                        "the matrix carries a non-finite value; preprocessing "
                        "guarantees a finite matrix, so this one did not come "
                        "from it"
                    )


@runtime_checkable
class ModelAdapter(Protocol):
    """What every model family implements.

    Registered in a closed in-code registry, never discovered. The identifiers
    are the registry keys the catalog declares, and a test asserts the two
    agree one-to-one in both directions.
    """

    @property
    def catalog_model_id(self) -> str: ...

    @property
    def family(self) -> ModelFamily: ...

    @property
    def serializer_id(self) -> str: ...

    @property
    def serializer_version(self) -> int: ...

    @property
    def inference_adapter_id(self) -> str: ...

    @property
    def supported_tasks(self) -> tuple[MLTask, ...]: ...

    @property
    def publishable(self) -> bool:
        """Whether a fitted model of this family may be written as an artifact."""
        ...

    def fit(self, batch: TrainingBatch, *, task: MLTask) -> FittedModel:
        """Fit on canonical training rows and return canonical content."""
        ...

    def score(
        self,
        model: FittedModel,
        rows: Sequence[Sequence[float]],
        columns: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Score *rows* from the artifact alone, using no estimator."""
        ...


def _canonical(payload: Any) -> Any:
    """Return *payload* with floats at fixed precision and no non-finite value."""
    if isinstance(payload, dict):
        return {key: _canonical(payload[key]) for key in sorted(payload)}
    if isinstance(payload, list | tuple):
        return [_canonical(item) for item in payload]
    if isinstance(payload, bool) or payload is None or isinstance(payload, int | str):
        return payload
    if isinstance(payload, float):
        return quantize(payload)
    return str(payload)
