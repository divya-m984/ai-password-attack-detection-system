"""M-010: logistic regression, serialised as coefficients rather than as an object.

Fitting uses scikit-learn.  Everything after that does not: the artifact is a
coefficient matrix, an intercept vector, and a class order, and scoring is a dot
product written here.  A model published today therefore stays readable by a
release whose ``LogisticRegression`` internals have changed entirely -- and the
parity test compares two genuinely independent implementations rather than one
implementation against itself.

**The score math is pinned, not assumed.**  For the binary head the decision
value is ``X @ coef_[0] + intercept_[0]`` and the reported score is its logistic
sigmoid.  For the category head it is ``X @ coef_.T + intercept_`` followed by a
softmax with the row maximum subtracted -- which is what the resolved release
does, and a test asserts the agreement rather than trusting the description.

**It is not a probability.**  The sigmoid produces a number in ``[0, 1]`` that
looks exactly like one.  Milestone 5 fits a calibrator and measures its
calibration error; until that has happened this is a ``decision_score``, and
:class:`ScoreSemantics` will not let the prose say otherwise.

**One deprecation, handled deliberately.**  The catalog declares
``penalty='l2'``.  scikit-learn 1.8 deprecated the ``penalty`` argument and 1.10
removes it, so passing it would emit a warning today and break at the top of the
reviewed range tomorrow.  ``l2`` *is* the estimator's default, so the declared
value is honoured by not passing the argument -- and the adapter refuses any
other value rather than silently ignoring it.
"""

from __future__ import annotations

import math
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

__all__ = ["LogisticRegressionAdapter"]

#: Estimator attributes the serializer reads.  All three are documented and
#: stable across the reviewed minor series; a test in ``test_dependencies``
#: asserts they exist on the installed release with the expected shapes.
PUBLIC_ATTRIBUTES: Final[tuple[str, ...]] = ("coef_", "intercept_", "classes_")

#: The only penalty the catalog declares, and the estimator's own default.
_SUPPORTED_PENALTY: Final[str] = "l2"
_SUPPORTED_SOLVER: Final[str] = "lbfgs"


class LogisticRegressionAdapter:
    """Fit with scikit-learn; score from stored coefficients."""

    catalog_model_id: Final[str] = "M-010"
    family: Final[ModelFamily] = ModelFamily.LOGISTIC_REGRESSION
    serializer_id: Final[str] = "json_linear_v1"
    serializer_version: Final[int] = 1
    inference_adapter_id: Final[str] = "linear_logit_v1"
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
        penalty: str = _SUPPORTED_PENALTY,
        solver: str = _SUPPORTED_SOLVER,
        c_inverse_regularization: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        """Bind the declared hyperparameters, refusing anything unreviewed."""
        if penalty != _SUPPORTED_PENALTY:
            raise ModelTrainingError(
                f"penalty {penalty!r} is not the reviewed value "
                f"{_SUPPORTED_PENALTY!r}; a penalty this adapter cannot honour "
                f"is refused rather than ignored"
            )
        if solver != _SUPPORTED_SOLVER:
            raise ModelTrainingError(
                f"solver {solver!r} is not the reviewed value "
                f"{_SUPPORTED_SOLVER!r}; the solver is a determinism control, "
                f"not a preference"
            )
        self._penalty = penalty
        self._solver = solver
        self._c = float(c_inverse_regularization)
        self._max_iter = int(max_iter)
        self._tol = float(tol)
        self._random_state = int(random_state)

    def fit(self, batch: TrainingBatch, *, task: MLTask) -> FittedModel:
        """Fit the estimator and keep only its documented attributes."""
        if task not in self.supported_tasks:
            raise ModelTrainingError(
                f"{self.catalog_model_id} does not support task {str(task)!r}"
            )
        batch.require_canonical()
        from sklearn.linear_model import LogisticRegression

        design = batch.design()
        targets = batch.encoded_targets()
        if len(set(targets.tolist())) < 2:
            raise ModelTrainingError(
                "the training batch carries a single class; a logistic model "
                "fitted on it would have nothing to separate"
            )

        # `penalty` is deliberately not passed: see the module docstring. Every
        # other determinism control is explicit rather than defaulted, so a
        # library default changing cannot change a fitted model silently.
        estimator = LogisticRegression(
            solver=self._solver,
            C=self._c,
            max_iter=self._max_iter,
            tol=self._tol,
            random_state=self._random_state,
            fit_intercept=True,
            class_weight=None,
        )
        estimator.fit(design, targets, sample_weight=batch.sample_weights())

        coefficients = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)
        observed = [int(value) for value in np.asarray(estimator.classes_).tolist()]
        if observed != list(range(len(batch.class_order))):
            raise ModelTrainingError(
                "the estimator observed a different class set from the declared "
                "class order; the encoded targets and the order have diverged"
            )
        expected_rows = 1 if len(batch.class_order) == 2 else len(batch.class_order)
        if coefficients.shape != (expected_rows, len(batch.transformed_feature_names)):
            raise ModelTrainingError(
                f"fitted coefficients have shape {coefficients.shape}, expected "
                f"({expected_rows}, {len(batch.transformed_feature_names)})"
            )
        if intercept.shape != (expected_rows,):
            raise ModelTrainingError(
                f"fitted intercept has shape {intercept.shape}, expected "
                f"({expected_rows},)"
            )

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
                "penalty": self._penalty,
                "solver": self._solver,
                "c_inverse_regularization": self._c,
                "max_iter": self._max_iter,
                "tol": self._tol,
                "random_state": self._random_state,
            },
            parameters={
                "multiclass": expected_rows > 1,
                "n_features_in": int(estimator.n_features_in_),
                "link": "sigmoid" if expected_rows == 1 else "softmax",
            },
            arrays={"coefficients": coefficients, "intercept": intercept},
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
        """Return per-class scores from the stored coefficients alone."""
        model.require_matrix(rows, columns)
        coefficients = np.asarray(model.arrays["coefficients"], dtype=np.float64)
        intercept = np.asarray(model.arrays["intercept"], dtype=np.float64)
        if coefficients.shape[1] != len(model.transformed_feature_names):
            raise ModelTrainingError(
                "the stored coefficients do not cover the declared transformed features"
            )
        if intercept.shape != (coefficients.shape[0],):
            raise ModelTrainingError(
                "the stored intercept does not match the coefficient rows"
            )
        design = np.asarray(rows, dtype=np.float64).reshape(
            len(rows), len(model.transformed_feature_names)
        )
        logits = design @ coefficients.T + intercept

        if coefficients.shape[0] == 1:
            if len(model.class_order) != 2:
                raise ModelTrainingError(
                    "a single coefficient row describes a binary model, but the "
                    "declared class order is not binary"
                )
            positive = _sigmoid(logits[:, 0])
            return tuple((float(1.0 - value), float(value)) for value in positive)

        if coefficients.shape[0] != len(model.class_order):
            raise ModelTrainingError(
                "the stored coefficients do not cover the declared class order"
            )
        return tuple(tuple(float(value) for value in row) for row in _softmax(logits))


def _sigmoid(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return the logistic function, evaluated without overflowing.

    The branch matters: ``exp(710)`` overflows to infinity in float64, so the
    naive form silently returns NaN for a strongly separated row -- exactly the
    rows a detector cares most about.
    """
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponentiated = np.exp(values[~positive])
    out[~positive] = exponentiated / (1.0 + exponentiated)
    if not np.isfinite(out).all():
        raise ModelTrainingError("the linear score produced a non-finite value")
    return out


def _softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return the row-wise softmax with the row maximum subtracted.

    The subtraction is not an optimisation: it is what the resolved
    scikit-learn release does, and reproducing its score exactly means
    reproducing its arithmetic exactly. A test pins the agreement.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    totals = exponentiated.sum(axis=1, keepdims=True)
    if not np.all(totals > 0.0) or not math.isfinite(float(totals.sum())):
        raise ModelTrainingError("the class scores produced a non-finite total")
    normalized: np.ndarray[Any, Any] = exponentiated / totals
    return normalized
