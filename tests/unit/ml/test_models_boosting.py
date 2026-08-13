"""M-021: the gate, and why a passing probe does not open it.

Every test here exists to keep one claim honest: this family is **not**
champion-eligible, and the reason is the shape of its dependency rather than
the quality of its scores. It fits, it reproduces the estimator exactly, and it
still may not be stored or promoted, because serialising it means reading
attributes scikit-learn does not document.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from password_attack_detector.exceptions import (
    ModelSerializationError,
    ModelTrainingError,
)
from password_attack_detector.ml.catalog import MODEL_CATALOG
from password_attack_detector.ml.enums import (
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
)
from password_attack_detector.ml.models import (
    PUBLISHABLE_FAMILIES,
    HistogramBoostingAdapter,
    probe_compatibility,
)
from password_attack_detector.ml.models.boosting import (
    PRIVATE_ATTRIBUTES,
    PUBLIC_ATTRIBUTES,
    REQUIRED_NODE_FIELDS,
)
from tests.ml.models import prepare, publish

PARITY_TOLERANCE = 1e-12


@pytest.fixture
def binary() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=180)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_family_is_not_champion_eligible_in_the_catalog() -> None:
    """The reviewed catalog records the gate, and it stays closed."""
    spec = next(item for item in MODEL_CATALOG.specs if item.model_id == "M-021")
    assert spec.champion_eligible is False
    assert spec.eligibility_status is ModelEligibilityStatus.SERIALIZER_UNPROVEN


def test_a_fitted_model_is_not_champion_eligible(binary: Any) -> None:
    """The flag travels with the model, not only with the catalog entry."""
    fitted = HistogramBoostingAdapter(max_iter=8).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    assert fitted.champion_eligible is False
    assert fitted.experimental is True


def test_the_family_is_not_publishable() -> None:
    """Not storable, so the private dependency never reaches a file."""
    assert ModelFamily.HISTOGRAM_GRADIENT_BOOSTING not in PUBLISHABLE_FAMILIES
    assert HistogramBoostingAdapter.publishable is False


def test_publishing_is_refused(binary: Any, tmp_path: Any) -> None:
    """Attempting it fails with a reason, rather than writing something fragile."""
    fitted = HistogramBoostingAdapter(max_iter=8).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    with pytest.raises(ModelSerializationError, match="not publishable"):
        publish(tmp_path / "boost", fitted, binary.preprocessor)


def test_a_passing_probe_does_not_promote_the_family() -> None:
    """The two are independent, and this asserts they stay that way.

    The probe reports on the installed release. Promotion is a policy decision
    about whether a serializer contract may rest on undocumented internals, and
    the answer does not change when the internals happen to be present.
    """
    probe = probe_compatibility()
    assert probe.compatible is True
    assert HistogramBoostingAdapter.publishable is False
    assert "champion-ineligible" in probe.summary()


# ---------------------------------------------------------------------------
# The compatibility probe
# ---------------------------------------------------------------------------


def test_the_probe_names_the_installed_release() -> None:
    """A verdict without a version is not actionable."""
    import sklearn

    assert probe_compatibility().sklearn_version == sklearn.__version__


def test_the_private_attribute_dependency_is_written_down() -> None:
    """Exactly two, each with a stated purpose."""
    names = [name for name, _ in PRIVATE_ATTRIBUTES]
    assert names == ["_predictors", "_baseline_prediction"]
    assert all(reason for _, reason in PRIVATE_ATTRIBUTES)


def test_every_declared_private_attribute_exists_on_a_fitted_estimator() -> None:
    """The declaration is checked against the release, not merely asserted."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    design = rng.normal(size=(40, 3))
    estimator = HistGradientBoostingClassifier(
        max_iter=3, early_stopping=False, random_state=42
    ).fit(design, (design[:, 0] > 0).astype(int))
    for name, _ in PRIVATE_ATTRIBUTES:
        assert hasattr(estimator, name), name
    for attribute in PUBLIC_ATTRIBUTES:
        assert hasattr(estimator, attribute), attribute


def test_every_required_node_field_is_present_with_the_declared_kind() -> None:
    """Names *and* dtype kinds: a field that changed width would truncate."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    design = rng.normal(size=(40, 3))
    estimator = HistGradientBoostingClassifier(
        max_iter=3, early_stopping=False, random_state=42
    ).fit(design, (design[:, 0] > 0).astype(int))
    nodes = np.asarray(estimator._predictors[0][0].nodes)
    for field, kind in REQUIRED_NODE_FIELDS:
        assert field in (nodes.dtype.fields or {}), field
        assert nodes.dtype[field].kind == kind, field


def test_a_failing_probe_is_reported_structurally() -> None:
    """A verdict names what moved, which is what a drop decision needs."""
    probe = probe_compatibility()
    assert probe.missing_attributes == ()
    assert probe.missing_node_fields == ()
    assert probe.unexpected_field_kinds == ()
    assert "does not match" not in probe.summary()


# ---------------------------------------------------------------------------
# It still has to be correct
# ---------------------------------------------------------------------------


def test_the_traversal_reproduces_the_estimator(binary: Any) -> None:
    """Being ineligible is not being allowed to be wrong."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    adapter = HistogramBoostingAdapter(max_iter=20)
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    estimator = HistGradientBoostingClassifier(
        max_iter=20,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=42,
    )
    estimator.fit(
        binary.batch.design(),
        binary.batch.encoded_targets(),
        sample_weight=binary.batch.sample_weights(),
    )
    mine = np.asarray(
        adapter.score(fitted, binary.matrix.rows, binary.matrix.output_feature_names)
    )
    theirs = estimator.predict_proba(binary.batch.design())
    assert np.max(np.abs(mine - theirs)) <= PARITY_TOLERANCE


def test_early_stopping_is_pinned_off() -> None:
    """It would carve a validation set out of rows the split contract placed."""
    with pytest.raises(ModelTrainingError, match="early stopping"):
        HistogramBoostingAdapter(early_stopping=True)


def test_the_multiclass_head_is_refused() -> None:
    """A second private-layout dependency this adapter does not take on."""
    triage = prepare(count=120, task=MLTask.ATTACK_CATEGORY)
    with pytest.raises(ModelTrainingError, match="does not support"):
        HistogramBoostingAdapter(max_iter=5).fit(
            triage.batch, task=MLTask.ATTACK_CATEGORY
        )


def test_the_fitted_model_records_its_private_dependency(binary: Any) -> None:
    """Anybody reading the fitted content can see what it rests on."""
    fitted = HistogramBoostingAdapter(max_iter=5).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    assert fitted.parameters["private_state_dependency"] == [
        "_predictors",
        "_baseline_prediction",
    ]


def test_the_fit_is_deterministic(binary: Any) -> None:
    """Same seed, same rows, same content."""
    adapter = HistogramBoostingAdapter(max_iter=8)
    first = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    second = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    assert first.content_fingerprint() == second.content_fingerprint()
