"""M-030: isolation forest, experimental, unsupervised, never a probability.

Three properties are structural rather than conventional.

**It never sees a target.**  :meth:`IsolationForestAdapter.fit` takes a batch
and ignores its ``targets`` entirely -- the estimator is handed a design matrix
and nothing else.  Selecting *which* rows are benign is the caller's job and
happens upstream in Milestone 2, where labels are legitimately readable; by the
time a batch reaches here there is no target for this adapter to consult.

**It cannot be champion.**  ``champion_eligible`` is ``False`` on every model it
produces, the catalog records ``anomaly_only``, and the task enum member it
declares is not in ``SUPERVISED_TASKS``.

**It emits an anomaly score, and the word "probability" is unavailable.**  The
value is scikit-learn's ``score_samples`` convention: negative, with lower
meaning more anomalous.  :class:`ScoreSemantics` refuses prose using the word
for this score kind, so the vocabulary contract is enforced by the type rather
than by review.

The scoring path reimplements the published isolation-forest formula --
expected path length ``2H(n-1) - 2(n-1)/n`` with the harmonic number
approximated as ``ln(n-1) + gamma`` -- because scikit-learn's own
``_average_path_length`` is private.  That reimplementation is the whole reason
parity is asserted numerically rather than assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.models.base import (
    FittedModel,
    TrainingBatch,
    score_semantics_for,
)
from password_attack_detector.ml.models.forest import TREE_INPUT_DTYPE

__all__ = ["IsolationForestAdapter", "average_path_length"]

#: Estimator attributes the serializer reads. All documented.
PUBLIC_ATTRIBUTES: Final[tuple[str, ...]] = (
    "estimators_",
    "estimators_features_",
    "max_samples_",
    "offset_",
    "n_features_in_",
)

_LEAF: Final[int] = -1


def average_path_length(sizes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return the expected path length of an unsuccessful BST search.

    The published formula, written out:

    * ``n <= 1`` contributes nothing -- a single sample is already isolated;
    * ``n == 2`` contributes exactly ``1``;
    * otherwise ``2(ln(n-1) + gamma) - 2(n-1)/n``.

    The ``n == 2`` case is a special case in the source, not a limit of the
    general expression, and omitting it moves the score of every leaf holding
    two samples.
    """
    counts = np.asarray(sizes, dtype=np.float64)
    out = np.zeros_like(counts)
    general = counts > 2.0
    out[general] = (
        2.0 * (np.log(counts[general] - 1.0) + np.euler_gamma)
        - 2.0 * (counts[general] - 1.0) / counts[general]
    )
    out[counts == 2.0] = 1.0
    return out


class IsolationForestAdapter:
    """Fit with scikit-learn on unlabelled rows; score by traversing stored trees."""

    catalog_model_id: Final[str] = "M-030"
    family: Final[ModelFamily] = ModelFamily.ISOLATION_FOREST
    serializer_id: Final[str] = "json_isolation_forest_v1"
    serializer_version: Final[int] = 1
    inference_adapter_id: Final[str] = "isolation_path_v1"
    supported_tasks: Final[tuple[MLTask, ...]] = (MLTask.ANOMALY,)
    publishable: Final[bool] = True
    champion_eligible: Final[bool] = False
    reference_baseline: Final[bool] = False

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_samples: int = 256,
        contamination: float = 0.01,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        """Bind the declared hyperparameters."""
        if n_jobs != 1:
            raise ModelTrainingError("n_jobs is pinned to 1 for reproducibility")
        self._n_estimators = int(n_estimators)
        self._max_samples = int(max_samples)
        self._contamination = float(contamination)
        self._random_state = int(random_state)

    def fit(self, batch: TrainingBatch, *, task: MLTask) -> FittedModel:
        """Fit on the design matrix alone; the batch's targets are never read."""
        if task is not MLTask.ANOMALY:
            raise ModelTrainingError(
                f"{self.catalog_model_id} fits the anomaly task only, not {str(task)!r}"
            )
        batch.require_canonical()
        from sklearn.ensemble import IsolationForest

        design = batch.design()
        if len(design) < 2:
            raise ModelTrainingError(
                "an isolation forest needs at least two rows to isolate anything"
            )
        # No `y`, and no sample weight: an unsupervised fit that consulted a
        # label would be a supervised fit wearing the wrong name.
        estimator = IsolationForest(
            n_estimators=self._n_estimators,
            max_samples=min(self._max_samples, len(design)),
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=1,
            bootstrap=False,
        )
        estimator.fit(design)

        arrays = _flatten(estimator)
        return FittedModel(
            serializer_id=self.serializer_id,
            serializer_version=self.serializer_version,
            inference_adapter_id=self.inference_adapter_id,
            catalog_model_id=self.catalog_model_id,
            family=self.family,
            task=task,
            class_order=(),
            raw_feature_names=batch.preprocessor.raw_feature_names,
            transformed_feature_names=batch.transformed_feature_names,
            hyperparameters={
                "n_estimators": self._n_estimators,
                "max_samples": self._max_samples,
                "contamination": self._contamination,
                "random_state": self._random_state,
                "n_jobs": 1,
            },
            parameters={
                "max_samples_fitted": int(estimator.max_samples_),
                "tree_count": len(estimator.estimators_),
                "n_features_in": int(estimator.n_features_in_),
                "score_convention": "lower is more anomalous",
                "supervised": False,
            },
            arrays=arrays,
            score_semantics=score_semantics_for(task),
            train_row_count=batch.row_count,
            preprocessor_fingerprint=batch.preprocessor.fingerprint(),
            eligible_feature_list_fingerprint=(
                batch.preprocessor.eligible_feature_list_fingerprint
            ),
            class_weight_fingerprint=None,
            champion_eligible=self.champion_eligible,
            experimental=True,
        )

    def score(
        self,
        model: FittedModel,
        rows: Sequence[Sequence[float]],
        columns: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Return one anomaly score per row, matching ``score_samples``."""
        model.require_matrix(rows, columns)
        design = np.asarray(rows, dtype=np.float64).reshape(
            len(rows), len(model.transformed_feature_names)
        )
        scores = _score_samples(model.arrays, design)
        return tuple((float(value),) for value in scores)


def _flatten(estimator: Any) -> dict[str, Any]:
    """Return the concatenated node arrays and per-tree feature subsets.

    Isolation trees are fitted on a random subset of columns, so the feature
    subset is part of the model and is stored beside the nodes. The subsets are
    all the same width for a given fit, which is what lets them be rectangular.
    """
    left: list[int] = []
    right: list[int] = []
    feature: list[int] = []
    threshold: list[float] = []
    samples: list[int] = []
    offsets: list[int] = []
    subsets: list[list[int]] = []

    widths = {len(item) for item in estimator.estimators_features_}
    if len(widths) != 1:
        raise ModelTrainingError(
            "the fitted trees use feature subsets of differing widths, which "
            "this serializer's rectangular layout cannot represent"
        )

    for sub_estimator, features in zip(
        estimator.estimators_, estimator.estimators_features_, strict=True
    ):
        tree = sub_estimator.tree_
        offsets.append(len(left))
        subsets.append([int(value) for value in features])
        left.extend(int(value) for value in tree.children_left)
        right.extend(int(value) for value in tree.children_right)
        feature.extend(int(value) for value in tree.feature)
        threshold.extend(float(value) for value in tree.threshold)
        samples.extend(int(value) for value in tree.n_node_samples)

    offsets.append(len(left))
    return {
        "tree_offsets": np.asarray(offsets, dtype=np.int64),
        "children_left": np.asarray(left, dtype=np.int64),
        "children_right": np.asarray(right, dtype=np.int64),
        "split_feature": np.asarray(feature, dtype=np.int64),
        "split_threshold": np.asarray(threshold, dtype=np.float64),
        "node_sample_count": np.asarray(samples, dtype=np.int64),
        "feature_subset": np.asarray(subsets, dtype=np.int64),
        "max_samples": np.asarray([int(estimator.max_samples_)], dtype=np.int64),
        "offset": np.asarray([float(estimator.offset_)], dtype=np.float64),
    }


def _score_samples(arrays: Any, design: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return ``-2 ** (-mean_depth / expected_depth)`` for every row."""
    offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
    left = np.asarray(arrays["children_left"], dtype=np.int64)
    right = np.asarray(arrays["children_right"], dtype=np.int64)
    feature = np.asarray(arrays["split_feature"], dtype=np.int64)
    threshold = np.asarray(arrays["split_threshold"], dtype=np.float64)
    node_samples = np.asarray(arrays["node_sample_count"], dtype=np.int64)
    subsets = np.asarray(arrays["feature_subset"], dtype=np.int64)
    max_samples = int(np.asarray(arrays["max_samples"])[0])

    tree_count = len(offsets) - 1
    if tree_count < 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(left):
        raise ModelTrainingError(
            "the tree offset table does not describe the stored node arrays"
        )
    if len(subsets) != tree_count:
        raise ModelTrainingError("the stored feature subsets do not cover every tree")

    depths = np.zeros(len(design), dtype=np.float64)
    # Isolation trees are the same scikit-learn tree implementation the forest
    # uses, so they carry the same float32 comparison contract. See
    # ``models.forest`` for what a float64 comparison costs.
    compared = np.asarray(design, dtype=np.dtype(TREE_INPUT_DTYPE))
    for tree in range(tree_count):
        start, stop = int(offsets[tree]), int(offsets[tree + 1])
        columns = subsets[tree]
        if columns.size and (columns.min() < 0 or columns.max() >= design.shape[1]):
            raise ModelTrainingError(
                "a stored feature subset names a column outside the matrix"
            )
        view = compared[:, columns]
        for row_index in range(len(design)):
            node = start
            depth = 0
            steps = 0
            while left[node] != _LEAF:
                steps += 1
                if steps > stop - start:
                    raise ModelTrainingError(
                        "traversal exceeded the tree's node count; the stored "
                        "children describe a cycle"
                    )
                column = int(feature[node])
                child = (
                    left[node]
                    if view[row_index, column] <= threshold[node]
                    else right[node]
                )
                node = start + int(child)
                if not start <= node < stop:
                    raise ModelTrainingError(
                        "a stored child index falls outside its own tree"
                    )
                depth += 1
            depths[row_index] += depth + float(
                average_path_length(np.asarray([node_samples[node]]))[0]
            )

    denominator = tree_count * float(average_path_length(np.asarray([max_samples]))[0])
    if denominator <= 0.0:
        raise ModelTrainingError(
            "the expected path length is not positive; the stored sample count "
            "cannot be right"
        )
    return -(2.0 ** (-depths / denominator))
