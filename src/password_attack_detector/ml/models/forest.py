"""M-020: a random forest, serialised as flat node arrays and traversed here.

The forest is stored as five parallel arrays per tree, concatenated across the
ensemble with an offset table: left children, right children, split features,
split thresholds, and leaf class distributions.  Nothing about the scikit-learn
``Tree`` object survives into the artifact, so the model stays readable when
that object's layout changes -- and the traversal below is genuinely
independent code, which is what makes the parity test evidence.

**The split rule is pinned, and so is the precision it is applied at.**
scikit-learn sends a row left when ``value <= threshold`` and right otherwise.
A boundary row is therefore a *left* row, and getting that inequality backwards
would agree with the estimator on almost every input and disagree on exactly
the rows sitting on a cut.

Less obviously, scikit-learn's tree predictor casts the matrix to **float32**
before comparing, while thresholds are stored as float64 -- and a fitted
threshold is a midpoint computed in that same float32 space.  A float64
comparison therefore disagrees with the estimator whenever a value and a
threshold are equal at float32 and ordered at float64.  It is a rare row and a
tiny score difference, which is precisely why it has to be reproduced
deliberately: :data:`TREE_INPUT_DTYPE` is the estimator's contract, not an
optimisation, and dropping it moved parity from exact to ``1.5e-2`` on the first
matrix that contained a standardised column.

**Leaf values are already normalised.**  In the reviewed scikit-learn series a
classification tree's ``value`` rows sum to one, so a forest's score is the
plain mean of the leaf rows across trees -- no per-tree renormalisation, and no
division by a sample count.  The serializer asserts the property at fit time
rather than assuming it, because a release that changed it would otherwise
produce a model that scored wrongly and silently.
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

__all__ = ["RandomForestAdapter", "traverse_forest"]

#: Estimator attributes the serializer reads. All documented and stable across
#: the reviewed minor series.
PUBLIC_ATTRIBUTES: Final[tuple[str, ...]] = (
    "estimators_",
    "classes_",
    "n_outputs_",
    "n_features_in_",
)

#: scikit-learn's sentinel for "this node has no child", i.e. it is a leaf.
_LEAF: Final[int] = -1

#: The dtype scikit-learn's tree predictor casts its input to before comparing
#: against a stored threshold. Reproducing the cast is what makes traversal
#: exact rather than nearly exact; see the module docstring.
TREE_INPUT_DTYPE: Final[str] = "float32"

#: How far a normalised leaf row may drift from summing to one before the
#: serializer refuses it. Generous enough for float64 accumulation, tight
#: enough that an un-normalised row (summing to a sample count) cannot pass.
_LEAF_SUM_TOLERANCE: Final[float] = 1e-9


class RandomForestAdapter:
    """Fit with scikit-learn; traverse flat node arrays here."""

    catalog_model_id: Final[str] = "M-020"
    family: Final[ModelFamily] = ModelFamily.RANDOM_FOREST
    serializer_id: Final[str] = "json_tree_ensemble_v1"
    serializer_version: Final[int] = 1
    inference_adapter_id: Final[str] = "tree_vote_v1"
    supported_tasks: Final[tuple[MLTask, ...]] = (
        MLTask.BINARY_MALICIOUS,
        MLTask.ATTACK_CATEGORY,
    )
    publishable: Final[bool] = True
    champion_eligible: Final[bool] = True
    reference_baseline: Final[bool] = False

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        """Bind the declared hyperparameters."""
        if n_jobs != 1:
            raise ModelTrainingError(
                "n_jobs is pinned to 1; parallel tree building introduces a "
                "reduction order that varies between runs"
            )
        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth)
        self._min_samples_leaf = int(min_samples_leaf)
        self._max_features = max_features
        self._random_state = int(random_state)

    def fit(self, batch: TrainingBatch, *, task: MLTask) -> FittedModel:
        """Fit the forest and flatten every tree into canonical arrays."""
        if task not in self.supported_tasks:
            raise ModelTrainingError(
                f"{self.catalog_model_id} does not support task {str(task)!r}"
            )
        batch.require_canonical()
        from sklearn.ensemble import RandomForestClassifier

        design = batch.design()
        targets = batch.encoded_targets()
        if len(set(targets.tolist())) < 2:
            raise ModelTrainingError(
                "the training batch carries a single class; a forest fitted on "
                "it would vote unanimously for it on every row"
            )

        estimator = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            max_features=self._max_features,
            criterion="gini",
            random_state=self._random_state,
            n_jobs=1,
            bootstrap=True,
            class_weight=None,
        )
        estimator.fit(design, targets, sample_weight=batch.sample_weights())

        observed = [int(value) for value in np.asarray(estimator.classes_).tolist()]
        if observed != list(range(len(batch.class_order))):
            raise ModelTrainingError(
                "the forest observed a different class set from the declared "
                "class order"
            )
        if int(estimator.n_outputs_) != 1:
            raise ModelTrainingError(
                "a multi-output forest has no single class order to serialise"
            )

        arrays = _flatten(estimator.estimators_, len(batch.class_order))

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
                "n_estimators": self._n_estimators,
                "max_depth": self._max_depth,
                "min_samples_leaf": self._min_samples_leaf,
                "max_features": str(self._max_features),
                "criterion": "gini",
                "random_state": self._random_state,
                "n_jobs": 1,
            },
            parameters={
                "tree_count": len(estimator.estimators_),
                "n_features_in": int(estimator.n_features_in_),
                "split_rule": "value <= threshold goes left",
                "aggregation": "mean of normalised leaf rows across trees",
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
        )

    def score(
        self,
        model: FittedModel,
        rows: Sequence[Sequence[float]],
        columns: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Return the mean leaf distribution across every tree."""
        model.require_matrix(rows, columns)
        design = np.asarray(rows, dtype=np.float64).reshape(
            len(rows), len(model.transformed_feature_names)
        )
        return traverse_forest(model.arrays, design, len(model.class_order))


def _flatten(estimators: Sequence[Any], class_count: int) -> dict[str, Any]:
    """Return the concatenated node arrays for every tree in the ensemble.

    One flat array per field plus a per-tree offset table, rather than a nested
    structure: the archive stores rectangular arrays, and an offset table is
    both smaller and easier to validate than a ragged encoding.
    """
    left: list[int] = []
    right: list[int] = []
    feature: list[int] = []
    threshold: list[float] = []
    leaf_values: list[list[float]] = []
    offsets: list[int] = []

    for estimator in estimators:
        tree = estimator.tree_
        offsets.append(len(left))
        values = np.asarray(tree.value, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] != 1 or values.shape[2] != class_count:
            raise ModelTrainingError(
                f"tree leaf values have shape {values.shape}, which does not "
                f"describe {class_count} classes for a single output"
            )
        rows = values[:, 0, :]
        leaves = np.asarray(tree.children_left) == _LEAF
        sums = rows[leaves].sum(axis=1)
        if leaves.any() and not np.allclose(sums, 1.0, atol=_LEAF_SUM_TOLERANCE):
            raise ModelTrainingError(
                "tree leaf rows are not normalised distributions on this "
                "scikit-learn release; the mean-of-leaves aggregation this "
                "serializer records would be wrong, so the fit is refused "
                "rather than published"
            )
        left.extend(int(value) for value in tree.children_left)
        right.extend(int(value) for value in tree.children_right)
        feature.extend(int(value) for value in tree.feature)
        threshold.extend(float(value) for value in tree.threshold)
        leaf_values.extend([float(cell) for cell in row] for row in rows)

    offsets.append(len(left))
    return {
        "tree_offsets": np.asarray(offsets, dtype=np.int64),
        "children_left": np.asarray(left, dtype=np.int64),
        "children_right": np.asarray(right, dtype=np.int64),
        "split_feature": np.asarray(feature, dtype=np.int64),
        "split_threshold": np.asarray(threshold, dtype=np.float64),
        "leaf_value": np.asarray(leaf_values, dtype=np.float64),
    }


def traverse_forest(
    arrays: Any, design: np.ndarray[Any, Any], class_count: int
) -> tuple[tuple[float, ...], ...]:
    """Return the mean leaf distribution for every row of *design*.

    Written as an explicit walk rather than a vectorised gather so the split
    rule is visible in one place and can be read against the estimator's
    documented behaviour.

    Raises:
        ModelTrainingError: on a malformed ensemble -- a missing array, an
            offset table that does not describe the node arrays, a node index
            out of range, a split feature outside the matrix, or a cycle.
    """
    required = (
        "tree_offsets",
        "children_left",
        "children_right",
        "split_feature",
        "split_threshold",
        "leaf_value",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ModelTrainingError(
            f"the stored ensemble is missing array(s) {sorted(missing)}"
        )
    offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
    left = np.asarray(arrays["children_left"], dtype=np.int64)
    right = np.asarray(arrays["children_right"], dtype=np.int64)
    feature = np.asarray(arrays["split_feature"], dtype=np.int64)
    threshold = np.asarray(arrays["split_threshold"], dtype=np.float64)
    values = np.asarray(arrays["leaf_value"], dtype=np.float64)

    node_count = len(left)
    if not (len(right) == len(feature) == len(threshold) == node_count):
        raise ModelTrainingError("the stored node arrays disagree in length")
    if values.shape != (node_count, class_count):
        raise ModelTrainingError(
            f"stored leaf values have shape {values.shape}, expected "
            f"({node_count}, {class_count})"
        )
    if len(offsets) < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != node_count:
        raise ModelTrainingError(
            "the tree offset table does not describe the stored node arrays"
        )
    if not np.all(np.diff(offsets) > 0):
        raise ModelTrainingError("the tree offset table is not strictly increasing")

    tree_count = len(offsets) - 1
    totals = np.zeros((len(design), class_count), dtype=np.float64)
    # The estimator's own contract: compare at float32, against a float64
    # threshold. See the module docstring for what happens without it.
    compared = np.asarray(design, dtype=np.dtype(TREE_INPUT_DTYPE))
    for tree in range(tree_count):
        start, stop = int(offsets[tree]), int(offsets[tree + 1])
        for row_index in range(len(design)):
            node = start
            steps = 0
            while left[node] != _LEAF:
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
                child = (
                    left[node]
                    if compared[row_index, column] <= threshold[node]
                    else right[node]
                )
                node = start + int(child)
                if not start <= node < stop:
                    raise ModelTrainingError(
                        "a stored child index falls outside its own tree"
                    )
            totals[row_index] += values[node]
    scored = totals / tree_count
    return tuple(tuple(float(value) for value in row) for row in scored)
