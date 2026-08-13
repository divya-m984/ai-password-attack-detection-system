"""M-021: histogram gradient boosting, gated on a private-attribute dependency.

This family fits, scores, and round-trips exactly.  It is still **not**
champion-eligible, and the reason is not accuracy.

Serialising it requires reading ``_predictors`` and ``_baseline_prediction``.
Both are private.  Neither appears in the scikit-learn API reference, neither
carries a deprecation policy, and the structured dtype of ``predictor.nodes``
is an internal layout that a patch release is entitled to rearrange.  Every
other family in this catalog is serialised from documented attributes, which is
what lets the reviewed version range be a *review* gate rather than a hope.

So the dependency is made explicit instead of hidden:

* :data:`PRIVATE_ATTRIBUTES` names exactly what is read.
* :data:`REQUIRED_NODE_FIELDS` names the structured fields and their dtypes.
* :func:`probe_compatibility` checks all of it against the installed release and
  returns a structured verdict.
* The catalog records ``champion_eligible=False`` and
  ``eligibility_status=serializer_unproven``, and a test asserts the flag is
  still false.

A passing probe does **not** promote the family.  The gate is whether the
serializer contract rests on a documented interface, and it does not.  If a
future release breaks the probe, this adapter can be deleted without touching
anything else: nothing depends on it, and the registry records it as
unpublishable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.models.base import (
    FittedModel,
    TrainingBatch,
    score_semantics_for,
)

__all__ = [
    "PRIVATE_ATTRIBUTES",
    "PUBLIC_ATTRIBUTES",
    "REQUIRED_NODE_FIELDS",
    "CompatibilityProbe",
    "HistogramBoostingAdapter",
    "probe_compatibility",
]

#: Documented attributes this adapter reads.
PUBLIC_ATTRIBUTES: Final[tuple[str, ...]] = ("classes_", "n_iter_", "n_features_in_")

#: Undocumented attributes this adapter reads, and what each is needed for.
#:
#: This tuple is the whole reason the family is gated. It is asserted by a test
#: against the installed release, so a release that removes one fails loudly
#: here rather than producing a model that cannot be scored.
PRIVATE_ATTRIBUTES: Final[tuple[tuple[str, str], ...]] = (
    ("_predictors", "the per-iteration fitted trees, as TreePredictor objects"),
    ("_baseline_prediction", "the constant the boosted sum starts from"),
)

#: Structured fields of ``predictor.nodes`` the traversal depends on, with the
#: dtype kind each must have. Names *and* kinds: a field that stayed but changed
#: width would silently truncate a threshold.
REQUIRED_NODE_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("value", "f"),
    ("feature_idx", "i"),
    ("num_threshold", "f"),
    ("missing_go_to_left", "u"),
    ("left", "u"),
    ("right", "u"),
    ("is_leaf", "u"),
    ("is_categorical", "u"),
)


@dataclass(frozen=True, slots=True)
class CompatibilityProbe:
    """What an inspection of the installed release found.

    Structured rather than a bare boolean so a failure names the attribute or
    field that moved, which is the information somebody deciding whether to drop
    the family actually needs.
    """

    compatible: bool
    sklearn_version: str
    missing_attributes: tuple[str, ...]
    missing_node_fields: tuple[str, ...]
    unexpected_field_kinds: tuple[str, ...]
    notes: tuple[str, ...]

    def summary(self) -> str:
        """Return a one-line, identifier-free verdict."""
        if self.compatible:
            return (
                f"private layout matches on scikit-learn {self.sklearn_version}; "
                f"family remains champion-ineligible by policy"
            )
        return (
            f"private layout does not match on scikit-learn "
            f"{self.sklearn_version}: "
            f"{len(self.missing_attributes)} attribute(s), "
            f"{len(self.missing_node_fields)} node field(s), "
            f"{len(self.unexpected_field_kinds)} dtype mismatch(es)"
        )


def probe_compatibility() -> CompatibilityProbe:
    """Fit a tiny model and inspect the private state the serializer needs.

    A fit rather than a class inspection: ``_predictors`` does not exist until
    something has been fitted, so an import-time check would prove nothing.
    """
    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import HistGradientBoostingClassifier

    missing_attributes: list[str] = []
    missing_fields: list[str] = []
    wrong_kinds: list[str] = []
    notes: list[str] = []

    rng = np.random.default_rng(0)
    design = rng.normal(size=(40, 3))
    targets = (design[:, 0] > 0).astype(np.int64)
    estimator = HistGradientBoostingClassifier(
        max_iter=3, early_stopping=False, random_state=42
    )
    estimator.fit(design, targets)

    for name, _ in PRIVATE_ATTRIBUTES:
        if not hasattr(estimator, name):
            missing_attributes.append(name)
    if missing_attributes:
        return CompatibilityProbe(
            compatible=False,
            sklearn_version=str(sklearn_version),
            missing_attributes=tuple(missing_attributes),
            missing_node_fields=(),
            unexpected_field_kinds=(),
            notes=("the private attributes the serializer reads are absent",),
        )

    predictors = estimator._predictors
    first = predictors[0][0]
    nodes = np.asarray(first.nodes)
    fields = nodes.dtype.fields or {}
    for field, expected_kind in REQUIRED_NODE_FIELDS:
        if field not in fields:
            missing_fields.append(field)
            continue
        actual = nodes.dtype[field].kind
        if actual != expected_kind:
            wrong_kinds.append(f"{field}:{actual}")

    baseline = np.asarray(estimator._baseline_prediction)
    if baseline.size != 1:
        notes.append(
            "the baseline prediction is not a single value; only the binary "
            "head is serialisable by this adapter"
        )

    return CompatibilityProbe(
        compatible=not (missing_fields or wrong_kinds),
        sklearn_version=str(sklearn_version),
        missing_attributes=(),
        missing_node_fields=tuple(missing_fields),
        unexpected_field_kinds=tuple(wrong_kinds),
        notes=tuple(notes),
    )


class HistogramBoostingAdapter:
    """Fits and scores; never publishes, and never becomes champion.

    ``publishable`` is ``False``, so :mod:`password_attack_detector.ml.serialization`
    refuses to write an artifact for it. The adapter remains useful as an
    experimental comparator that can be evaluated in-process without any of its
    private-state dependency reaching a stored file.
    """

    catalog_model_id: Final[str] = "M-021"
    family: Final[ModelFamily] = ModelFamily.HISTOGRAM_GRADIENT_BOOSTING
    serializer_id: Final[str] = "json_histogram_ensemble_v1"
    serializer_version: Final[int] = 1
    inference_adapter_id: Final[str] = "histogram_raw_v1"
    supported_tasks: Final[tuple[MLTask, ...]] = (MLTask.BINARY_MALICIOUS,)
    publishable: Final[bool] = False
    champion_eligible: Final[bool] = False
    reference_baseline: Final[bool] = False

    def __init__(
        self,
        *,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        early_stopping: bool = False,
        random_state: int = 42,
    ) -> None:
        """Bind the declared hyperparameters."""
        if early_stopping:
            raise ModelTrainingError(
                "early stopping is pinned off; it splits a validation set out "
                "of the training rows, which would fit on data the split "
                "contract assigned elsewhere"
            )
        self._max_iter = int(max_iter)
        self._learning_rate = float(learning_rate)
        self._max_leaf_nodes = int(max_leaf_nodes)
        self._min_samples_leaf = int(min_samples_leaf)
        self._l2 = float(l2_regularization)
        self._random_state = int(random_state)

    def fit(self, batch: TrainingBatch, *, task: MLTask) -> FittedModel:
        """Fit, then flatten the private node arrays into canonical content."""
        if task not in self.supported_tasks:
            raise ModelTrainingError(
                f"{self.catalog_model_id} does not support task {str(task)!r}"
            )
        batch.require_canonical()
        probe = probe_compatibility()
        if not probe.compatible:
            raise ModelTrainingError(
                f"the histogram-boosting private layout is not the one this "
                f"adapter reads: {probe.summary()}"
            )
        from sklearn.ensemble import HistGradientBoostingClassifier

        design = batch.design()
        targets = batch.encoded_targets()
        if len(batch.class_order) != 2:
            raise ModelTrainingError(
                "this adapter serialises the binary head only; the multiclass "
                "head stores one predictor per class per iteration, which is a "
                "second private-layout dependency and is not taken on"
            )

        estimator = HistGradientBoostingClassifier(
            max_iter=self._max_iter,
            learning_rate=self._learning_rate,
            max_leaf_nodes=self._max_leaf_nodes,
            min_samples_leaf=self._min_samples_leaf,
            l2_regularization=self._l2,
            early_stopping=False,
            random_state=self._random_state,
        )
        estimator.fit(design, targets, sample_weight=batch.sample_weights())

        arrays = _flatten_predictors(estimator)
        baseline = float(np.asarray(estimator._baseline_prediction).ravel()[0])

        return FittedModel(
            serializer_id=self.serializer_id,
            serializer_version=self.serializer_version,
            inference_adapter_id=self.inference_adapter_id,
            catalog_model_id=self.catalog_model_id,
            family=self.family,
            task=task,
            class_order=batch.class_order,
            raw_feature_names=batch.preprocessor.raw_feature_names,
            transformed_feature_names=batch.transformed_feature_names,
            hyperparameters={
                "max_iter": self._max_iter,
                "learning_rate": self._learning_rate,
                "max_leaf_nodes": self._max_leaf_nodes,
                "min_samples_leaf": self._min_samples_leaf,
                "l2_regularization": self._l2,
                "early_stopping": False,
                "random_state": self._random_state,
            },
            parameters={
                "baseline_prediction": baseline,
                "iteration_count": len(estimator._predictors),
                "n_features_in": int(estimator.n_features_in_),
                "private_state_dependency": [name for name, _ in PRIVATE_ATTRIBUTES],
                "split_rule": "value <= num_threshold goes left",
            },
            arrays=arrays,
            score_semantics=score_semantics_for(task),
            train_row_count=batch.row_count,
            preprocessor_fingerprint=batch.preprocessor.fingerprint(),
            eligible_feature_list_fingerprint=(
                batch.preprocessor.eligible_feature_list_fingerprint
            ),
            class_weight_fingerprint=(
                None
                if batch.class_weights is None
                else batch.class_weights.fingerprint()
            ),
            champion_eligible=self.champion_eligible,
            experimental=True,
        )

    def score(
        self,
        model: FittedModel,
        rows: Sequence[Sequence[float]],
        columns: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Return the sigmoid of the boosted raw sum, from stored arrays alone."""
        model.require_matrix(rows, columns)
        design = np.asarray(rows, dtype=np.float64).reshape(
            len(rows), len(model.transformed_feature_names)
        )
        raw = _raw_predict(
            model.arrays, design, float(model.parameters["baseline_prediction"])
        )
        positive = 1.0 / (1.0 + np.exp(-raw))
        if not np.isfinite(positive).all():
            raise ModelTrainingError("the boosted score produced a non-finite value")
        return tuple((float(1.0 - value), float(value)) for value in positive)


def _flatten_predictors(estimator: Any) -> dict[str, Any]:
    """Return concatenated node arrays across every boosting iteration."""
    value: list[float] = []
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    is_leaf: list[int] = []
    missing_left: list[int] = []
    offsets: list[int] = []

    for iteration in estimator._predictors:
        if len(iteration) != 1:
            raise ModelTrainingError(
                "an iteration carries more than one predictor; only the binary "
                "head is serialised by this adapter"
            )
        nodes = np.asarray(iteration[0].nodes)
        if nodes["is_categorical"].any():
            raise ModelTrainingError(
                "the fitted trees contain categorical splits, whose thresholds "
                "are bitsets in a separate private array; preprocessing emits "
                "one-hot numeric columns, so this indicates the estimator was "
                "handed a matrix this layer did not produce"
            )
        offsets.append(len(value))
        value.extend(float(item) for item in nodes["value"])
        feature.extend(int(item) for item in nodes["feature_idx"])
        threshold.extend(float(item) for item in nodes["num_threshold"])
        left.extend(int(item) for item in nodes["left"])
        right.extend(int(item) for item in nodes["right"])
        is_leaf.extend(int(item) for item in nodes["is_leaf"])
        missing_left.extend(int(item) for item in nodes["missing_go_to_left"])

    offsets.append(len(value))
    return {
        "tree_offsets": np.asarray(offsets, dtype=np.int64),
        "node_value": np.asarray(value, dtype=np.float64),
        "split_feature": np.asarray(feature, dtype=np.int64),
        "split_threshold": np.asarray(threshold, dtype=np.float64),
        "children_left": np.asarray(left, dtype=np.int64),
        "children_right": np.asarray(right, dtype=np.int64),
        "is_leaf": np.asarray(is_leaf, dtype=np.uint8),
        "missing_go_to_left": np.asarray(missing_left, dtype=np.uint8),
    }


def _raw_predict(
    arrays: Any, design: np.ndarray[Any, Any], baseline: float
) -> np.ndarray[Any, Any]:
    """Return the boosted raw sum for every row."""
    offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
    value = np.asarray(arrays["node_value"], dtype=np.float64)
    feature = np.asarray(arrays["split_feature"], dtype=np.int64)
    threshold = np.asarray(arrays["split_threshold"], dtype=np.float64)
    left = np.asarray(arrays["children_left"], dtype=np.int64)
    right = np.asarray(arrays["children_right"], dtype=np.int64)
    is_leaf = np.asarray(arrays["is_leaf"], dtype=np.uint8)
    missing_left = np.asarray(arrays["missing_go_to_left"], dtype=np.uint8)

    if len(offsets) < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != len(value):
        raise ModelTrainingError(
            "the iteration offset table does not describe the stored node arrays"
        )
    out = np.full(len(design), baseline, dtype=np.float64)
    for iteration in range(len(offsets) - 1):
        start, stop = int(offsets[iteration]), int(offsets[iteration + 1])
        for row_index in range(len(design)):
            node = start
            steps = 0
            while not is_leaf[node]:
                steps += 1
                if steps > stop - start:
                    raise ModelTrainingError(
                        "traversal exceeded the tree's node count; the stored "
                        "children describe a cycle"
                    )
                column = int(feature[node])
                if not 0 <= column < design.shape[1]:
                    raise ModelTrainingError(
                        "a stored split names a feature outside the matrix"
                    )
                cell = design[row_index, column]
                if np.isnan(cell):
                    child = left[node] if missing_left[node] else right[node]
                else:
                    child = left[node] if cell <= threshold[node] else right[node]
                node = start + int(child)
                if not start <= node < stop:
                    raise ModelTrainingError(
                        "a stored child index falls outside its own tree"
                    )
            out[row_index] += value[node]
    return out
