"""M-030: unsupervised, experimental, and never a probability.

Three structural properties, each with its own test: it receives no target, it
cannot become champion, and its output is an anomaly score whose prose is
forbidden the word "probability" by the type that carries it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.catalog import MODEL_CATALOG
from password_attack_detector.ml.enums import (
    PROBABILITY_SCORE_KINDS,
    SUPERVISED_TASKS,
    MLTask,
    ModelEligibilityStatus,
    ScoreKind,
)
from password_attack_detector.ml.models import IsolationForestAdapter
from password_attack_detector.ml.models.anomaly import (
    PUBLIC_ATTRIBUTES,
    average_path_length,
)
from password_attack_detector.ml.models.base import ANOMALY_SCORE_COLUMN
from tests.ml.models import prepare, publish

#: The approved bound for this family. Its scoring path reimplements a formula
#: involving a logarithm and Euler's constant, so bit-equality is not
#: mathematically defensible; what is observed is a single float64 rounding
#: step, and that is what is required.
PARITY_TOLERANCE = 1e-9


@pytest.fixture
def unlabelled() -> Any:
    """Return a prepared batch for the anomaly task, carrying no targets."""
    return prepare(count=180, task=MLTask.ANOMALY)


def reference(batch: Any, **overrides: Any) -> Any:
    """Return an estimator fitted exactly as the adapter fits one."""
    from sklearn.ensemble import IsolationForest

    settings = {
        "n_estimators": 30,
        "max_samples": 64,
        "contamination": 0.01,
        "random_state": 42,
        "n_jobs": 1,
        "bootstrap": False,
    }
    settings.update(overrides)
    estimator = IsolationForest(**settings)
    estimator.fit(batch.batch.design())
    return estimator


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_the_score_matches_score_samples(unlabelled: Any) -> None:
    """The reimplemented path-length formula agrees with the estimator."""
    adapter = IsolationForestAdapter(n_estimators=30, max_samples=64)
    fitted = adapter.fit(unlabelled.batch, task=MLTask.ANOMALY)
    mine = np.asarray(
        adapter.score(
            fitted, unlabelled.matrix.rows, unlabelled.matrix.output_feature_names
        )
    ).ravel()
    theirs = reference(unlabelled).score_samples(unlabelled.batch.design())
    assert np.max(np.abs(mine - theirs)) <= PARITY_TOLERANCE


def test_the_average_path_length_special_cases_are_reproduced() -> None:
    """``n <= 1`` is zero and ``n == 2`` is exactly one, by definition.

    The ``n == 2`` value is a special case in the published algorithm rather
    than a limit of the general expression, and omitting it moves the score of
    every leaf holding two samples.
    """
    values = average_path_length(np.asarray([0, 1, 2, 3, 10, 256]))
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] == 1.0
    assert values[3] > 1.0
    assert values[5] > values[4]


def test_repeated_scoring_is_identical(unlabelled: Any) -> None:
    """Scoring is a pure function of the stored trees."""
    adapter = IsolationForestAdapter(n_estimators=10, max_samples=32)
    fitted = adapter.fit(unlabelled.batch, task=MLTask.ANOMALY)
    first = adapter.score(
        fitted, unlabelled.matrix.rows, unlabelled.matrix.output_feature_names
    )
    assert (
        adapter.score(
            fitted, unlabelled.matrix.rows, unlabelled.matrix.output_feature_names
        )
        == first
    )


def test_the_fit_is_deterministic(unlabelled: Any) -> None:
    """Same seed, same rows, same content."""
    adapter = IsolationForestAdapter(n_estimators=15, max_samples=32)
    first = adapter.fit(unlabelled.batch, task=MLTask.ANOMALY)
    second = adapter.fit(unlabelled.batch, task=MLTask.ANOMALY)
    assert first.content_fingerprint() == second.content_fingerprint()


# ---------------------------------------------------------------------------
# It never sees a target
# ---------------------------------------------------------------------------


def test_the_anomaly_batch_carries_no_targets(unlabelled: Any) -> None:
    """There is nothing for the adapter to consult, by construction."""
    assert unlabelled.batch.targets == ()
    assert unlabelled.batch.class_order == ()
    assert unlabelled.batch.class_weights is None


def test_a_fitted_anomaly_model_declares_no_classes(unlabelled: Any) -> None:
    """An unsupervised model has no classes, and the type refuses to pretend."""
    fitted = IsolationForestAdapter(n_estimators=10, max_samples=32).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    assert fitted.class_order == ()
    assert fitted.class_weight_fingerprint is None
    assert fitted.score_columns == (ANOMALY_SCORE_COLUMN,)


def test_targets_present_in_the_batch_do_not_reach_the_fit() -> None:
    """Handing it a labelled batch changes nothing about the fitted model.

    The anomaly task is unsupervised whatever the caller passed, so a batch that
    happens to carry a target column must produce the same model as one that
    does not.
    """
    labelled = prepare(count=140, task=MLTask.BINARY_MALICIOUS)
    unlabelled = prepare(count=140, task=MLTask.ANOMALY)
    adapter = IsolationForestAdapter(n_estimators=12, max_samples=32)
    with_targets = adapter.fit(labelled.batch, task=MLTask.ANOMALY)
    without = adapter.fit(unlabelled.batch, task=MLTask.ANOMALY)
    assert np.array_equal(
        np.asarray(with_targets.arrays["split_threshold"]),
        np.asarray(without.arrays["split_threshold"]),
    )


def test_a_supervised_task_is_refused(unlabelled: Any) -> None:
    """This family fits the anomaly task and nothing else."""
    labelled = prepare(count=120)
    with pytest.raises(ModelTrainingError, match="anomaly task only"):
        IsolationForestAdapter(n_estimators=5).fit(
            labelled.batch, task=MLTask.BINARY_MALICIOUS
        )


# ---------------------------------------------------------------------------
# It cannot be champion, and it is not a probability
# ---------------------------------------------------------------------------


def test_the_catalog_records_it_as_anomaly_only() -> None:
    """The reviewed entry, not merely the adapter, says so."""
    spec = next(item for item in MODEL_CATALOG.specs if item.model_id == "M-030")
    assert spec.champion_eligible is False
    assert spec.eligibility_status is ModelEligibilityStatus.ANOMALY_ONLY
    assert spec.anomaly_only is True
    assert spec.experimental is True


def test_a_fitted_model_is_never_champion_eligible(unlabelled: Any) -> None:
    """The flag travels with the model."""
    fitted = IsolationForestAdapter(n_estimators=8, max_samples=32).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    assert fitted.champion_eligible is False
    assert fitted.experimental is True


def test_the_anomaly_task_is_not_a_supervised_task() -> None:
    """Structural: it is absent from the supervised set, so it cannot compete."""
    assert MLTask.ANOMALY not in SUPERVISED_TASKS


def test_the_score_kind_is_never_a_probability(unlabelled: Any) -> None:
    """And the word is refused in the prose by the type that carries it."""
    fitted = IsolationForestAdapter(n_estimators=8, max_samples=32).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    assert fitted.score_semantics.score_kind is ScoreKind.ANOMALY_SCORE
    assert fitted.score_semantics.score_kind not in PROBABILITY_SCORE_KINDS
    lowered = fitted.score_semantics.description.lower()
    assert "probability" not in lowered
    assert "likelihood" not in lowered
    assert "confidence" not in lowered


def test_the_score_convention_is_recorded(unlabelled: Any) -> None:
    """Lower is more anomalous, and the artifact says which way round it is."""
    fitted = IsolationForestAdapter(n_estimators=8, max_samples=32).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    assert fitted.parameters["score_convention"] == "lower is more anomalous"
    assert fitted.parameters["supervised"] is False


def test_no_threshold_is_selected(unlabelled: Any) -> None:
    """Threshold provenance belongs to a later milestone, and nothing pre-empts it."""
    fitted = IsolationForestAdapter(n_estimators=8, max_samples=32).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    rendered = str(fitted.parameters) + str(fitted.hyperparameters)
    assert "anomaly_threshold" not in rendered
    assert "flagged" not in rendered


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_parallel_fitting_is_refused() -> None:
    """Reproducibility over speed, as everywhere else in this layer."""
    with pytest.raises(ModelTrainingError, match="n_jobs"):
        IsolationForestAdapter(n_jobs=2)


def test_the_declared_public_attributes_are_the_ones_read(unlabelled: Any) -> None:
    """No private attribute is read, which is why this family may be published."""
    estimator = reference(unlabelled, n_estimators=4, max_samples=16)
    for attribute in PUBLIC_ATTRIBUTES:
        assert hasattr(estimator, attribute), attribute


def test_a_single_row_batch_is_refused() -> None:
    """There is nothing to isolate one row from."""
    rows = [(0.5, 1, "success", "us", True, True)]
    tiny = prepare(count=1, rows=rows, task=MLTask.ANOMALY)
    with pytest.raises(ModelTrainingError, match="at least two rows"):
        IsolationForestAdapter(n_estimators=5, max_samples=2).fit(
            tiny.batch, task=MLTask.ANOMALY
        )


def test_the_artifact_round_trips(unlabelled: Any, tmp_path: Path) -> None:
    """Published and reloaded, it scores identically."""
    from password_attack_detector.ml.inference import InferenceModel

    adapter = IsolationForestAdapter(n_estimators=12, max_samples=32)
    fitted = adapter.fit(unlabelled.batch, task=MLTask.ANOMALY)
    directory = publish(tmp_path / "anomaly", fitted, unlabelled.preprocessor)
    loaded = InferenceModel.load(directory)
    before = adapter.score(
        fitted, unlabelled.matrix.rows, unlabelled.matrix.output_feature_names
    )
    after = loaded.score(unlabelled.matrix.rows, unlabelled.matrix.output_feature_names)
    assert after == before
    assert loaded.score_columns == (ANOMALY_SCORE_COLUMN,)


def test_loading_it_as_a_champion_is_refused(unlabelled: Any, tmp_path: Path) -> None:
    """A caller that requires a promotable model does not get this one."""
    from password_attack_detector.exceptions import ModelNotReadyError
    from password_attack_detector.ml.inference import InferenceModel

    adapter = IsolationForestAdapter(n_estimators=8, max_samples=32)
    fitted = adapter.fit(unlabelled.batch, task=MLTask.ANOMALY)
    directory = publish(tmp_path / "anomaly", fitted, unlabelled.preprocessor)
    with pytest.raises(ModelNotReadyError, match="champion-eligible"):
        InferenceModel.load(directory, require_champion_eligible=True)
