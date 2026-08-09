"""M-010: coefficients, and the score math they reconstruct.

The parity tests are the point. Fitting goes through scikit-learn; scoring goes
through project code that never sees an estimator; and the two must agree to
within a float64 rounding step. Agreement between two independent
implementations is evidence, which is why the estimator is constructed here
rather than reached for through the adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import CalibrationMethod, MLTask, ScoreKind
from password_attack_detector.ml.models import LogisticRegressionAdapter
from password_attack_detector.ml.models.linear import PUBLIC_ATTRIBUTES
from tests.ml.models import prepare, publish

#: Every parity assertion in this module. Chosen because both paths do the same
#: float64 arithmetic in the same order, so a single rounding step is the whole
#: budget -- and a wider tolerance would stop catching a reordered expression.
PARITY_TOLERANCE = 1e-12


@pytest.fixture
def binary() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=180)


@pytest.fixture
def triage() -> Any:
    """Return a prepared three-class training batch."""
    return prepare(count=180, task=MLTask.ATTACK_CATEGORY)


def reference(batch: Any) -> Any:
    """Return an estimator fitted exactly as the adapter fits one."""
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        tol=1e-4,
        random_state=42,
        fit_intercept=True,
        class_weight=None,
    )
    estimator.fit(
        batch.batch.design(),
        batch.batch.encoded_targets(),
        sample_weight=batch.batch.sample_weights(),
    )
    return estimator


# ---------------------------------------------------------------------------
# Determinism controls
# ---------------------------------------------------------------------------


def test_every_determinism_control_is_recorded(binary: Any) -> None:
    """Solver, tolerance, iteration cap, and seed all reach the artifact."""
    fitted = LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    assert fitted.hyperparameters["solver"] == "lbfgs"
    assert fitted.hyperparameters["max_iter"] == 1000
    assert fitted.hyperparameters["tol"] == 1e-4
    assert fitted.hyperparameters["random_state"] == 42
    assert fitted.hyperparameters["penalty"] == "l2"


def test_an_unreviewed_solver_is_refused() -> None:
    """The solver is a determinism control, not a preference."""
    with pytest.raises(ModelTrainingError, match="solver"):
        LogisticRegressionAdapter(solver="liblinear")


def test_an_unreviewed_penalty_is_refused() -> None:
    """A penalty this adapter cannot honour is refused rather than ignored."""
    with pytest.raises(ModelTrainingError, match="penalty"):
        LogisticRegressionAdapter(penalty="l1")


def test_fitting_emits_no_deprecation_warning(binary: Any) -> None:
    """The declared ``penalty='l2'`` is honoured by not passing the argument.

    scikit-learn 1.8 deprecated the argument and 1.10 removes it. Passing it
    would warn today and break at the top of the reviewed range, and ``l2`` is
    already the estimator's default -- so the contract is met by omission.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)


def test_two_fits_of_one_batch_agree_exactly(binary: Any) -> None:
    """A fixed seed and a fixed solver make the artifact reproducible."""
    adapter = LogisticRegressionAdapter()
    first = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    second = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    assert first.content_fingerprint() == second.content_fingerprint()


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_the_binary_score_matches_the_estimator(binary: Any) -> None:
    """Reconstructed from stored coefficients, to within one rounding step."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    mine = np.asarray(
        adapter.score(fitted, binary.matrix.rows, binary.matrix.output_feature_names)
    )
    theirs = reference(binary).predict_proba(binary.batch.design())
    assert np.max(np.abs(mine - theirs)) <= PARITY_TOLERANCE


def test_the_binary_decision_value_matches_exactly(binary: Any) -> None:
    """The linear part is a dot product, and dot products are bit-reproducible."""
    fitted = LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    design = binary.batch.design()
    mine = design @ np.asarray(fitted.arrays["coefficients"])[0] + float(
        np.asarray(fitted.arrays["intercept"])[0]
    )
    assert np.array_equal(mine, reference(binary).decision_function(design))


def test_the_multiclass_score_matches_the_estimator(triage: Any) -> None:
    """The softmax convention is reproduced, not approximated."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(triage.batch, task=MLTask.ATTACK_CATEGORY)
    mine = np.asarray(
        adapter.score(fitted, triage.matrix.rows, triage.matrix.output_feature_names)
    )
    theirs = reference(triage).predict_proba(triage.batch.design())
    assert np.max(np.abs(mine - theirs)) <= PARITY_TOLERANCE


def test_the_multiclass_softmax_subtracts_the_row_maximum(triage: Any) -> None:
    """Which convention is used changes the last bits, so it is pinned."""
    fitted = LogisticRegressionAdapter().fit(triage.batch, task=MLTask.ATTACK_CATEGORY)
    logits = triage.batch.design() @ np.asarray(
        fitted.arrays["coefficients"]
    ).T + np.asarray(fitted.arrays["intercept"])
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = shifted / shifted.sum(axis=1, keepdims=True)
    assert np.array_equal(
        expected, reference(triage).predict_proba(triage.batch.design())
    )


def test_repeated_scoring_is_identical(binary: Any) -> None:
    """Scoring is a pure function of the artifact and the matrix."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    first = adapter.score(
        fitted, binary.matrix.rows, binary.matrix.output_feature_names
    )
    second = adapter.score(
        fitted, binary.matrix.rows, binary.matrix.output_feature_names
    )
    assert first == second


def test_a_reordered_batch_scores_the_same_rows_the_same_way(binary: Any) -> None:
    """Row order is a property of the batch, not of the model."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    forward = adapter.score(
        fitted, binary.matrix.rows, binary.matrix.output_feature_names
    )
    backward = adapter.score(
        fitted, tuple(reversed(binary.matrix.rows)), binary.matrix.output_feature_names
    )
    assert tuple(reversed(backward)) == forward


def test_an_extreme_value_does_not_overflow(binary: Any) -> None:
    """A strongly separated row is exactly the row a detector cares about."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    extreme = tuple(1e6 for _ in fitted.transformed_feature_names)
    scored = adapter.score(
        fitted,
        [extreme, tuple(-1e6 for _ in extreme)],
        fitted.transformed_feature_names,
    )
    assert all(np.isfinite(value) for row in scored for value in row)
    assert all(0.0 <= value <= 1.0 for row in scored for value in row)


# ---------------------------------------------------------------------------
# Shape and contract
# ---------------------------------------------------------------------------


def test_the_coefficient_shape_matches_the_feature_order(binary: Any) -> None:
    """One row for a binary model, one column per transformed feature."""
    fitted = LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    coefficients = np.asarray(fitted.arrays["coefficients"])
    assert coefficients.shape == (1, len(fitted.transformed_feature_names))
    assert np.asarray(fitted.arrays["intercept"]).shape == (1,)


def test_the_multiclass_coefficient_shape_matches_the_class_order(triage: Any) -> None:
    """One row per class once there are more than two."""
    fitted = LogisticRegressionAdapter().fit(triage.batch, task=MLTask.ATTACK_CATEGORY)
    coefficients = np.asarray(fitted.arrays["coefficients"])
    assert coefficients.shape == (3, len(fitted.transformed_feature_names))
    assert fitted.class_order == triage.batch.class_order


def test_a_stored_coefficient_of_the_wrong_width_is_refused(binary: Any) -> None:
    """A hand-edited artifact cannot score against a different feature set."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    narrowed = fitted.model_copy(
        update={
            "arrays": {
                **fitted.arrays,
                "coefficients": np.asarray(fitted.arrays["coefficients"])[:, :-1],
            }
        }
    )
    with pytest.raises(ModelTrainingError, match="do not cover"):
        adapter.score(narrowed, binary.matrix.rows, binary.matrix.output_feature_names)


def test_a_matrix_in_the_wrong_column_order_is_refused(binary: Any) -> None:
    """The right columns in the wrong order score the wrong feature silently."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    swapped = (
        fitted.transformed_feature_names[1],
        fitted.transformed_feature_names[0],
        *fitted.transformed_feature_names[2:],
    )
    with pytest.raises(ModelTrainingError, match="different order"):
        adapter.score(fitted, binary.matrix.rows, swapped)


def test_a_single_class_batch_is_refused() -> None:
    """A logistic model needs something to separate."""
    rows = [(-1.0, index, "success", "us", True, True) for index in range(40)]
    single = prepare(count=40, rows=rows, weighted=False)
    with pytest.raises(ModelTrainingError, match="single class"):
        LogisticRegressionAdapter().fit(single.batch, task=MLTask.BINARY_MALICIOUS)


def test_the_score_is_not_called_a_probability(binary: Any) -> None:
    """No calibrator has been fitted, so the word is unavailable."""
    fitted = LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    assert fitted.score_semantics.score_kind is ScoreKind.DECISION_SCORE
    assert fitted.score_semantics.calibration_method is CalibrationMethod.NONE
    assert "probability" not in fitted.score_semantics.description.lower()


def test_the_declared_public_attributes_are_the_ones_read() -> None:
    """The serializer's dependency on the estimator is written down."""
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(max_iter=50).fit([[0.0], [1.0]], [0, 1])
    for attribute in PUBLIC_ATTRIBUTES:
        assert hasattr(estimator, attribute), attribute


def test_the_artifact_round_trips(binary: Any, tmp_path: Path) -> None:
    """Published and reloaded, it scores identically to the fitted object."""
    from password_attack_detector.ml.inference import InferenceModel

    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    directory = publish(tmp_path / "linear", fitted, binary.preprocessor)
    loaded = InferenceModel.load(directory)
    before = adapter.score(
        fitted, binary.matrix.rows, binary.matrix.output_feature_names
    )
    assert (
        loaded.score(binary.matrix.rows, binary.matrix.output_feature_names) == before
    )
