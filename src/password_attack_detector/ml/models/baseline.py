"""The two reference models: a class prior, and one feature with a threshold.

Neither is meant to be good. They exist so that "the model detects attacks" has
something to mean: a learned model that cannot beat the training prevalence, or
cannot beat a single well-chosen column with a cut point on it, has not earned
the complexity it costs.

Neither uses scikit-learn at all, for either fitting or scoring. The prior is a
count; the threshold is a sorted scan. Both are therefore exactly reproducible
without any library at all, which makes them the right comparators for families
whose reproducibility depends on one.

Neither is ever promoted automatically. Milestone 7 selects a champion, and
until then a baseline is a number to beat.
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
    quantize,
    score_semantics_for,
)

__all__ = ["PriorBaselineAdapter", "SingleFeatureThresholdAdapter"]

#: Direction conventions for the threshold baseline, and what a tie means.
#:
#: ``above`` flags a row whose value is **strictly greater** than the cut;
#: ``below`` flags one **strictly less**. A value exactly on the cut is
#: therefore never flagged under either direction. The rule is stated rather
#: than inherited so a boundary row cannot change class because somebody
#: reversed an inequality while refactoring.
_DIRECTIONS: Final[tuple[str, ...]] = ("above", "below")


class PriorBaselineAdapter:
    """M-000: predict the training class prior, unconditionally.

    The prior is a ratio of counts, so it is exactly reproducible, closed-form,
    and free of any random number generator. Every row gets the same score,
    which makes this the honest floor: a model whose ranking cannot beat a
    constant has learned nothing about ordering.

    **It is the reference, never a candidate.** ``champion_eligible`` is
    ``False`` here and in the catalog, permanently. Selection asks whether a
    candidate beats this model by the configured margin, and nothing beats
    itself; admitting it to its own contest would also hand selection a
    fallback that always passes, when the honest outcome of every candidate
    failing is no champion at all. It stays fully fitted, fully publishable,
    and reported in every comparison -- only unpromotable.
    """

    catalog_model_id: Final[str] = "M-000"
    family: Final[ModelFamily] = ModelFamily.PRIOR_BASELINE
    serializer_id: Final[str] = "json_prior_v1"
    serializer_version: Final[int] = 1
    inference_adapter_id: Final[str] = "prior_v1"
    supported_tasks: Final[tuple[MLTask, ...]] = (
        MLTask.BINARY_MALICIOUS,
        MLTask.ATTACK_CATEGORY,
    )
    publishable: Final[bool] = True
    champion_eligible: Final[bool] = False
    reference_baseline: Final[bool] = True

    def fit(self, batch: TrainingBatch, *, task: MLTask) -> FittedModel:
        """Return the training class prior for every class, in class order.

        Weighted by the Milestone 3 class weights when they are supplied, so
        the comparator is measured under the same weighting the learned
        families are. An unweighted prior would flatter or penalise them
        depending only on which policy was configured.
        """
        _require_task(task, self.supported_tasks, self.catalog_model_id)
        batch.require_canonical()

        weights = batch.sample_weights()
        if weights is None:
            weights = np.ones(batch.row_count, dtype=np.float64)
        totals = np.zeros(len(batch.class_order), dtype=np.float64)
        position = {name: index for index, name in enumerate(batch.class_order)}
        for value, weight in zip(batch.targets, weights, strict=True):
            totals[position[value]] += weight
        total = float(totals.sum())
        if total <= 0.0:
            raise ModelTrainingError(
                "the training batch carries no weighted rows, so a class prior "
                "would divide by zero"
            )
        prior = np.asarray([quantize(value / total) for value in totals])

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
            hyperparameters={},
            parameters={"weighted": batch.class_weights is not None},
            arrays={"class_prior": prior},
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
        """Return the stored prior for every row.

        The matrix is still validated even though no value is read from it: a
        caller handing this baseline a differently shaped matrix has a bug, and
        the comparator silently accepting it would hide the bug in whichever
        family it was really aimed at.
        """
        model.require_matrix(rows, columns)
        prior = tuple(float(value) for value in model.arrays["class_prior"])
        if len(prior) != len(model.class_order):
            raise ModelTrainingError(
                "the stored prior does not cover the declared class order"
            )
        return tuple(prior for _ in rows)


class SingleFeatureThresholdAdapter:
    """M-001: one reviewed column, one cut point, one direction.

    The column is **configured, never discovered**. Scanning every feature for
    the best split would be a model-selection procedure run on training data and
    reported as a baseline, which flatters the baseline and understates whatever
    it is compared against.

    The threshold *is* derived from training rows: the cut that maximises
    balanced accuracy over the candidate grid, with ties broken toward the
    smaller cut and then toward ``above``. Both tie rules are arbitrary; being
    written down is what matters, because an undetermined tie makes the fitted
    model depend on floating-point noise.
    """

    catalog_model_id: Final[str] = "M-001"
    family: Final[ModelFamily] = ModelFamily.SINGLE_FEATURE_THRESHOLD
    serializer_id: Final[str] = "json_threshold_v1"
    serializer_version: Final[int] = 1
    inference_adapter_id: Final[str] = "threshold_v1"
    supported_tasks: Final[tuple[MLTask, ...]] = (MLTask.BINARY_MALICIOUS,)
    publishable: Final[bool] = True
    champion_eligible: Final[bool] = True
    reference_baseline: Final[bool] = False

    def __init__(self, *, feature: str, min_non_null_fraction: float = 0.5) -> None:
        """Bind this baseline to one reviewed transformed column.

        Args:
            feature: the transformed column to threshold. Must be present in
                the batch's transformed feature order.
            min_non_null_fraction: the share of training rows that must carry an
                observed -- not imputed -- value. A column that was mostly
                missing would produce a cut point describing the imputation
                constant rather than the data.
        """
        if not feature:
            raise ModelTrainingError("the threshold baseline needs a named feature")
        if not 0.0 < min_non_null_fraction <= 1.0:
            raise ModelTrainingError("min_non_null_fraction must lie in (0, 1]")
        self._feature = feature
        self._min_non_null_fraction = min_non_null_fraction

    @property
    def feature(self) -> str:
        """Return the reviewed column this baseline thresholds."""
        return self._feature

    def fit(self, batch: TrainingBatch, *, task: MLTask) -> FittedModel:
        """Return the best cut point on the configured column."""
        _require_task(task, self.supported_tasks, self.catalog_model_id)
        batch.require_canonical()
        if len(batch.class_order) != 2:
            raise ModelTrainingError(
                f"the threshold baseline is binary; it was given "
                f"{len(batch.class_order)} classes"
            )
        if self._feature not in batch.transformed_feature_names:
            raise ModelTrainingError(
                "the configured threshold feature is not one of the transformed "
                "columns; a baseline may only read a column the reviewed "
                "feature contract already admits"
            )

        index = batch.transformed_feature_names.index(self._feature)
        values = np.asarray([row[index] for row in batch.matrix], dtype=np.float64)
        positive = batch.encoded_targets() == 1

        indicator = f"{self._feature}__missing"
        if indicator in batch.transformed_feature_names:
            observed = (
                np.asarray(
                    [
                        row[batch.transformed_feature_names.index(indicator)]
                        for row in batch.matrix
                    ],
                    dtype=np.float64,
                )
                == 0.0
            )
            fraction = float(observed.mean())
            if fraction < self._min_non_null_fraction:
                raise ModelTrainingError(
                    f"only {fraction:.3f} of training rows carry an observed "
                    f"value for the configured column, below the required "
                    f"{self._min_non_null_fraction:.3f}; a cut point fitted here "
                    f"would describe the imputation constant"
                )

        cut, direction, score = _best_cut(values, positive)

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
                "min_non_null_fraction": float(self._min_non_null_fraction)
            },
            parameters={
                "feature": self._feature,
                "feature_index": index,
                "threshold": quantize(cut),
                "direction": direction,
                "tie_is_flagged": False,
                "train_balanced_accuracy": quantize(score),
            },
            arrays={"threshold": np.asarray([quantize(cut)], dtype=np.float64)},
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
        """Return a hard 0/1 vote per class, with the tie rule applied exactly."""
        model.require_matrix(rows, columns)
        index = int(model.parameters["feature_index"])
        cut = float(model.arrays["threshold"][0])
        direction = str(model.parameters["direction"])
        if direction not in _DIRECTIONS:
            raise ModelTrainingError(
                f"threshold direction must be one of {list(_DIRECTIONS)}"
            )
        scored: list[tuple[float, ...]] = []
        for row in rows:
            value = float(row[index])
            # Strict on both sides: a value exactly on the cut is not flagged,
            # whichever direction was fitted.
            flagged = value > cut if direction == "above" else value < cut
            scored.append((0.0, 1.0) if flagged else (1.0, 0.0))
        return tuple(scored)


def _best_cut(
    values: np.ndarray[Any, Any], positive: np.ndarray[Any, Any]
) -> tuple[float, str, float]:
    """Return the cut, direction, and balanced accuracy maximising the objective.

    The grid is the sorted distinct observed values plus a point below the
    minimum, so every achievable partition is reachable. Ties break toward the
    smaller cut and then toward ``above`` -- deterministic, and documented in
    the fitted parameters so a reader need not infer it.
    """
    if positive.sum() == 0 or positive.sum() == len(positive):
        raise ModelTrainingError(
            "the training batch carries only one class, so no cut point can "
            "separate anything"
        )
    grid = np.unique(values)
    candidates = np.concatenate([[float(grid[0]) - 1.0], grid])
    positives = float(positive.sum())
    negatives = float(len(positive) - positive.sum())

    best: tuple[float, str, float] | None = None
    for cut in candidates:
        for direction in _DIRECTIONS:
            flagged = values > cut if direction == "above" else values < cut
            recall = float((flagged & positive).sum()) / positives
            specificity = float((~flagged & ~positive).sum()) / negatives
            balanced = (recall + specificity) / 2.0
            if best is None or balanced > best[2]:
                best = (float(cut), direction, balanced)
    assert best is not None
    return best


def _require_task(task: MLTask, supported: tuple[MLTask, ...], model_id: str) -> None:
    """Raise unless *task* is one this family declares."""
    if task not in supported:
        raise ModelTrainingError(
            f"{model_id} does not support task {str(task)!r}; it declares "
            f"{sorted(str(item) for item in supported)}"
        )
