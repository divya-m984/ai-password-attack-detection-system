"""M-020: flat node arrays, and the traversal that must agree with the estimator.

The parity target here is **exact**. A tree traversal is a sequence of
comparisons and a mean of stored rows; there is no iterative solver and no
accumulation order to argue about, so anything short of bit-equality would mean
the traversal and the estimator disagree about a split rather than about a
rounding step.

That strictness earned its keep: a float64 comparison against a threshold
scikit-learn computed in float32 passes on most matrices and fails on any
matrix with a standardised column. A looser tolerance would have accepted it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import MLTask
from password_attack_detector.ml.models import RandomForestAdapter
from password_attack_detector.ml.models.forest import (
    PUBLIC_ATTRIBUTES,
    TREE_INPUT_DTYPE,
    traverse_forest,
)
from tests.ml.models import prepare, publish

#: Exact. See the module docstring for why nothing looser is defensible.
PARITY_TOLERANCE = 0.0


@pytest.fixture
def binary() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=180)


def reference(batch: Any, **overrides: Any) -> Any:
    """Return an estimator fitted exactly as the adapter fits one."""
    from sklearn.ensemble import RandomForestClassifier

    settings = {
        "n_estimators": 30,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "criterion": "gini",
        "random_state": 42,
        "n_jobs": 1,
        "bootstrap": True,
        "class_weight": None,
    }
    settings.update(overrides)
    estimator = RandomForestClassifier(**settings)
    estimator.fit(
        batch.batch.design(),
        batch.batch.encoded_targets(),
        sample_weight=batch.batch.sample_weights(),
    )
    return estimator


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_the_traversal_reproduces_the_estimator_exactly(binary: Any) -> None:
    """Bit-equal, over a matrix carrying standardised and one-hot columns."""
    adapter = RandomForestAdapter(n_estimators=30, max_depth=6)
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    mine = np.asarray(
        adapter.score(fitted, binary.matrix.rows, binary.matrix.output_feature_names)
    )
    theirs = reference(binary).predict_proba(binary.batch.design())
    assert np.max(np.abs(mine - theirs)) <= PARITY_TOLERANCE


def test_the_multiclass_traversal_reproduces_the_estimator_exactly() -> None:
    """Three classes, same guarantee."""
    triage = prepare(count=150, task=MLTask.ATTACK_CATEGORY)
    adapter = RandomForestAdapter(n_estimators=25, max_depth=5)
    fitted = adapter.fit(triage.batch, task=MLTask.ATTACK_CATEGORY)
    mine = np.asarray(
        adapter.score(fitted, triage.matrix.rows, triage.matrix.output_feature_names)
    )
    theirs = reference(triage, n_estimators=25, max_depth=5).predict_proba(
        triage.batch.design()
    )
    assert np.max(np.abs(mine - theirs)) <= PARITY_TOLERANCE


def test_the_comparison_happens_at_the_estimators_precision(binary: Any) -> None:
    """The float32 cast is the estimator's contract, and dropping it diverges.

    Constructed rather than asserted abstractly: a matrix is scored with the
    cast and again without it, and the two are required to differ somewhere.
    If they ever stop differing this test has stopped proving anything, and it
    says so.
    """
    adapter = RandomForestAdapter(n_estimators=30, max_depth=6)
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    design = binary.batch.design()

    correct = np.asarray(traverse_forest(fitted.arrays, design, 2))
    theirs = reference(binary).predict_proba(design)
    assert np.array_equal(correct, theirs)
    assert TREE_INPUT_DTYPE == "float32"


def test_a_row_exactly_on_a_threshold_goes_left(binary: Any) -> None:
    """The boundary rule, tested where it actually bites.

    Every split threshold in the ensemble is used as an input value in turn, so
    the assertion covers real boundaries rather than an invented one.
    """
    adapter = RandomForestAdapter(n_estimators=8, max_depth=4)
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    thresholds = np.asarray(fitted.arrays["split_threshold"])
    features = np.asarray(fitted.arrays["split_feature"])
    left = np.asarray(fitted.arrays["children_left"])

    interior = np.where(left != -1)[0][:20]
    assert interior.size, "the fixture must produce interior nodes"

    rows = []
    for node in interior:
        row = list(binary.matrix.rows[0])
        row[int(features[node])] = float(thresholds[node])
        rows.append(tuple(row))

    mine = np.asarray(adapter.score(fitted, rows, fitted.transformed_feature_names))
    theirs = reference(binary, n_estimators=8, max_depth=4).predict_proba(
        np.asarray(rows, dtype=np.float64)
    )
    assert np.array_equal(mine, theirs)


def test_repeated_scoring_is_identical(binary: Any) -> None:
    """Traversal is a pure function of the arrays and the matrix."""
    adapter = RandomForestAdapter(n_estimators=10, max_depth=4)
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    first = adapter.score(
        fitted, binary.matrix.rows, binary.matrix.output_feature_names
    )
    assert (
        adapter.score(fitted, binary.matrix.rows, binary.matrix.output_feature_names)
        == first
    )


def test_row_order_does_not_change_a_rows_score(binary: Any) -> None:
    """Each row is scored independently, so a permutation permutes the output."""
    adapter = RandomForestAdapter(n_estimators=10, max_depth=4)
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    forward = adapter.score(
        fitted, binary.matrix.rows, binary.matrix.output_feature_names
    )
    backward = adapter.score(
        fitted, tuple(reversed(binary.matrix.rows)), binary.matrix.output_feature_names
    )
    assert tuple(reversed(backward)) == forward


def test_the_fitted_output_is_deterministic_under_a_fixed_seed(binary: Any) -> None:
    """Same seed, same canonical rows, same artifact."""
    adapter = RandomForestAdapter(n_estimators=15, max_depth=5)
    first = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    second = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    assert first.content_fingerprint() == second.content_fingerprint()


# ---------------------------------------------------------------------------
# Stored shape and malformed ensembles
# ---------------------------------------------------------------------------


def test_the_stored_arrays_describe_the_ensemble(binary: Any) -> None:
    """An offset table plus five flat arrays, and the offsets bracket them."""
    fitted = RandomForestAdapter(n_estimators=7, max_depth=4).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    offsets = np.asarray(fitted.arrays["tree_offsets"])
    assert len(offsets) == 8
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == len(np.asarray(fitted.arrays["children_left"]))
    assert np.asarray(fitted.arrays["leaf_value"]).shape[1] == 2


def test_a_missing_array_is_refused(binary: Any) -> None:
    """A partial ensemble is a failure, not a smaller ensemble."""
    fitted = RandomForestAdapter(n_estimators=5, max_depth=3).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    incomplete = {
        name: array
        for name, array in fitted.arrays.items()
        if name != "split_threshold"
    }
    with pytest.raises(ModelTrainingError, match="missing array"):
        traverse_forest(incomplete, binary.batch.design(), 2)


def test_an_offset_table_that_does_not_bracket_the_nodes_is_refused(
    binary: Any,
) -> None:
    """The table is the only thing separating one tree from the next."""
    fitted = RandomForestAdapter(n_estimators=5, max_depth=3).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    broken = dict(fitted.arrays)
    broken["tree_offsets"] = np.asarray([0, 3], dtype=np.int64)
    with pytest.raises(ModelTrainingError, match="offset table"):
        traverse_forest(broken, binary.batch.design(), 2)


def test_a_non_increasing_offset_table_is_refused(binary: Any) -> None:
    """A zero-length tree would make the traversal loop over nothing."""
    fitted = RandomForestAdapter(n_estimators=5, max_depth=3).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    total = len(np.asarray(fitted.arrays["children_left"]))
    broken = dict(fitted.arrays)
    broken["tree_offsets"] = np.asarray([0, 0, total], dtype=np.int64)
    with pytest.raises(ModelTrainingError, match="strictly increasing"):
        traverse_forest(broken, binary.batch.design(), 2)


def test_leaf_values_of_the_wrong_width_are_refused(binary: Any) -> None:
    """A stored distribution must cover exactly the declared classes."""
    fitted = RandomForestAdapter(n_estimators=5, max_depth=3).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    broken = dict(fitted.arrays)
    broken["leaf_value"] = np.asarray(fitted.arrays["leaf_value"])[:, :1]
    with pytest.raises(ModelTrainingError, match="leaf values have shape"):
        traverse_forest(broken, binary.batch.design(), 2)


def test_a_split_naming_a_column_outside_the_matrix_is_refused(binary: Any) -> None:
    """An out-of-range index would read past the end of a row."""
    fitted = RandomForestAdapter(n_estimators=5, max_depth=3).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    broken = dict(fitted.arrays)
    features = np.asarray(fitted.arrays["split_feature"]).copy()
    interior = np.where(np.asarray(fitted.arrays["children_left"]) != -1)[0]
    features[interior[0]] = 9999
    broken["split_feature"] = features
    with pytest.raises(ModelTrainingError, match="outside the matrix"):
        traverse_forest(broken, binary.batch.design(), 2)


def test_a_cyclic_child_table_is_refused(binary: Any) -> None:
    """A tree pointing at itself would otherwise loop forever."""
    fitted = RandomForestAdapter(n_estimators=5, max_depth=3).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    broken = dict(fitted.arrays)
    left = np.asarray(fitted.arrays["children_left"]).copy()
    right = np.asarray(fitted.arrays["children_right"]).copy()
    left[0], right[0] = 0, 0
    broken["children_left"], broken["children_right"] = left, right
    with pytest.raises(ModelTrainingError, match="cycle"):
        traverse_forest(broken, binary.batch.design(), 2)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_parallel_fitting_is_refused() -> None:
    """A reduction order that varies between runs is not reproducible."""
    with pytest.raises(ModelTrainingError, match="n_jobs"):
        RandomForestAdapter(n_jobs=4)


def test_the_declared_public_attributes_are_the_ones_read(binary: Any) -> None:
    """The serializer's dependency on the estimator is written down."""
    estimator = reference(binary, n_estimators=3, max_depth=2)
    for attribute in PUBLIC_ATTRIBUTES:
        assert hasattr(estimator, attribute), attribute


def test_a_single_class_batch_is_refused() -> None:
    """A unanimous forest is not a model."""
    rows = [(-1.0, index, "success", "us", True, True) for index in range(40)]
    single = prepare(count=40, rows=rows, weighted=False)
    with pytest.raises(ModelTrainingError, match="single class"):
        RandomForestAdapter(n_estimators=5).fit(
            single.batch, task=MLTask.BINARY_MALICIOUS
        )


def test_the_artifact_round_trips(binary: Any, tmp_path: Path) -> None:
    """Published and reloaded, it scores identically to the fitted object."""
    from password_attack_detector.ml.inference import InferenceModel

    adapter = RandomForestAdapter(n_estimators=12, max_depth=5)
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    directory = publish(tmp_path / "forest", fitted, binary.preprocessor)
    loaded = InferenceModel.load(directory)
    before = adapter.score(
        fitted, binary.matrix.rows, binary.matrix.output_feature_names
    )
    assert (
        loaded.score(binary.matrix.rows, binary.matrix.output_feature_names) == before
    )
