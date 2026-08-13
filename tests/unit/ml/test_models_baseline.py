"""The two reference models: a class prior, and one column with a cut point.

Both exist to be beaten, so both must be exactly reproducible without any
library at all. Neither may be promoted automatically, and a test asserts the
absence of any code path that would.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import MLTask, ScoreKind
from password_attack_detector.ml.models import (
    PriorBaselineAdapter,
    SingleFeatureThresholdAdapter,
)
from password_attack_detector.ml.models.base import TrainingBatch
from tests.ml.models import prepare, publish


@pytest.fixture
def batch() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=120)


# ---------------------------------------------------------------------------
# M-000 prior baseline
# ---------------------------------------------------------------------------


def test_the_prior_is_the_weighted_training_share(batch: Any) -> None:
    """A ratio of weighted counts, computed by hand here."""
    adapter = PriorBaselineAdapter()
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    weights = batch.batch.sample_weights()
    totals = {"benign": 0.0, "malicious": 0.0}
    for target, weight in zip(batch.batch.targets, weights, strict=True):
        totals[target] += float(weight)
    total = sum(totals.values())
    prior = np.asarray(fitted.arrays["class_prior"])
    assert prior[0] == pytest.approx(totals["benign"] / total, abs=1e-9)
    assert prior[1] == pytest.approx(totals["malicious"] / total, abs=1e-9)


def test_the_prior_sums_to_one(batch: Any) -> None:
    """It is a distribution over the declared classes."""
    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    assert float(np.asarray(fitted.arrays["class_prior"]).sum()) == pytest.approx(1.0)


def test_every_row_receives_the_same_score(batch: Any) -> None:
    """A constant model is the honest floor for a ranking comparison."""
    adapter = PriorBaselineAdapter()
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    scored = adapter.score(fitted, batch.matrix.rows, batch.matrix.output_feature_names)
    assert len(set(scored)) == 1


def test_the_prior_is_deterministic(batch: Any) -> None:
    """Two fits over the same rows agree exactly, with no seed involved."""
    adapter = PriorBaselineAdapter()
    first = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    second = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    assert first.content_fingerprint() == second.content_fingerprint()


def test_an_unweighted_prior_records_that_it_was_unweighted() -> None:
    """The comparator says which weighting it was measured under."""
    unweighted = prepare(count=80, weighted=False)
    fitted = PriorBaselineAdapter().fit(unweighted.batch, task=MLTask.BINARY_MALICIOUS)
    assert fitted.parameters["weighted"] is False
    assert fitted.class_weight_fingerprint is None


def test_the_prior_supports_the_category_head() -> None:
    """Three classes, and the prior still sums to one."""
    triage = prepare(count=90, task=MLTask.ATTACK_CATEGORY)
    fitted = PriorBaselineAdapter().fit(triage.batch, task=MLTask.ATTACK_CATEGORY)
    assert len(fitted.class_order) == 3
    assert float(np.asarray(fitted.arrays["class_prior"]).sum()) == pytest.approx(1.0)
    assert fitted.score_semantics.score_kind is ScoreKind.CLASS_SCORE


def test_the_prior_refuses_the_anomaly_task(batch: Any) -> None:
    """A family scores the tasks its catalog entry declares, and no others."""
    with pytest.raises(ModelTrainingError, match="does not support"):
        PriorBaselineAdapter().fit(batch.batch, task=MLTask.ANOMALY)


# ---------------------------------------------------------------------------
# M-001 single-feature threshold
# ---------------------------------------------------------------------------


def test_the_threshold_column_is_configured_not_discovered(batch: Any) -> None:
    """Scanning for the best column would be model selection sold as a baseline."""
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    assert fitted.parameters["feature"] == "user_failure_rate"
    assert adapter.feature == "user_failure_rate"


def test_a_column_outside_the_feature_contract_is_refused(batch: Any) -> None:
    """A baseline may only read a column the reviewed contract already admits."""
    adapter = SingleFeatureThresholdAdapter(feature="something_nobody_admitted")
    with pytest.raises(ModelTrainingError, match="not one of the transformed"):
        adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)


def test_a_value_exactly_on_the_cut_is_not_flagged(batch: Any) -> None:
    """The tie rule is strict on both sides, and it is stated in the artifact."""
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    cut = float(fitted.arrays["threshold"][0])
    index = fitted.transformed_feature_names.index("user_failure_rate")

    on_the_cut = list(batch.matrix.rows[0])
    on_the_cut[index] = cut
    scored = adapter.score(
        fitted, [tuple(on_the_cut)], fitted.transformed_feature_names
    )
    assert scored == ((1.0, 0.0),)
    assert fitted.parameters["tie_is_flagged"] is False


def test_the_direction_is_recorded_and_honoured(batch: Any) -> None:
    """A row past the cut in the fitted direction is flagged; the other is not."""
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    cut = float(fitted.arrays["threshold"][0])
    index = fitted.transformed_feature_names.index("user_failure_rate")
    assert fitted.parameters["direction"] in {"above", "below"}

    def scored_at(value: float) -> tuple[float, ...]:
        row = list(batch.matrix.rows[0])
        row[index] = value
        return adapter.score(fitted, [tuple(row)], fitted.transformed_feature_names)[0]

    high, low = scored_at(cut + 10.0), scored_at(cut - 10.0)
    assert high != low


def test_a_direction_the_contract_does_not_declare_is_refused(batch: Any) -> None:
    """A hand-edited artifact cannot introduce a third comparison."""
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    tampered = fitted.model_copy(
        update={"parameters": {**fitted.parameters, "direction": "sideways"}}
    )
    with pytest.raises(ModelTrainingError, match="direction"):
        adapter.score(tampered, batch.matrix.rows[:1], fitted.transformed_feature_names)


def test_a_mostly_missing_column_is_refused() -> None:
    """A cut point fitted there would describe the imputation constant."""
    rows = [
        (
            None if index % 4 else float(index),
            index,
            "success",
            "us",
            True,
            True,
        )
        for index in range(80)
    ]
    mostly_missing = prepare(count=80, rows=rows, weighted=False)
    adapter = SingleFeatureThresholdAdapter(
        feature="user_failure_rate", min_non_null_fraction=0.9
    )
    with pytest.raises(ModelTrainingError, match="observed value"):
        adapter.fit(mostly_missing.batch, task=MLTask.BINARY_MALICIOUS)


def test_a_single_class_batch_is_refused() -> None:
    """No cut point separates anything when there is nothing to separate."""
    rows = [(-1.0, index, "success", "us", True, True) for index in range(40)]
    single = prepare(count=40, rows=rows, weighted=False)
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    with pytest.raises(ModelTrainingError, match="only one class"):
        adapter.fit(single.batch, task=MLTask.BINARY_MALICIOUS)


def test_the_threshold_is_deterministic(batch: Any) -> None:
    """Two fits agree, including the tie-break, so the artifact is stable."""
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    first = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    second = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    assert first.content_fingerprint() == second.content_fingerprint()


def test_the_threshold_baseline_is_binary_only(batch: Any) -> None:
    """One cut on one column cannot express three classes."""
    triage = prepare(count=90, task=MLTask.ATTACK_CATEGORY)
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    with pytest.raises(ModelTrainingError, match="does not support"):
        adapter.fit(triage.batch, task=MLTask.ATTACK_CATEGORY)


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_an_impossible_missing_fraction_is_refused(fraction: float) -> None:
    """The bound is a share, so it lives in (0, 1]."""
    with pytest.raises(ModelTrainingError, match="min_non_null_fraction"):
        SingleFeatureThresholdAdapter(feature="a", min_non_null_fraction=fraction)


def test_an_unnamed_feature_is_refused() -> None:
    """The column is what this baseline *is*; it cannot be omitted."""
    with pytest.raises(ModelTrainingError, match="named feature"):
        SingleFeatureThresholdAdapter(feature="")


# ---------------------------------------------------------------------------
# Both baselines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter",
    [
        PriorBaselineAdapter(),
        SingleFeatureThresholdAdapter(feature="user_failure_rate"),
    ],
    ids=["prior", "threshold"],
)
def test_a_baseline_uses_no_estimator_library(adapter: Any, batch: Any) -> None:
    """Both fit and score without importing scikit-learn at all.

    Asserted by removing the module from the import cache and blocking its
    import for the duration: the comparators for families whose reproducibility
    depends on a library must not depend on it themselves.
    """
    import builtins

    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "sklearn":
            raise AssertionError("a baseline reached for scikit-learn")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
        adapter.score(fitted, batch.matrix.rows, batch.matrix.output_feature_names)
    finally:
        builtins.__import__ = real_import


@pytest.mark.parametrize(
    "adapter",
    [
        PriorBaselineAdapter(),
        SingleFeatureThresholdAdapter(feature="user_failure_rate"),
    ],
    ids=["prior", "threshold"],
)
def test_a_baseline_round_trips_through_an_artifact(
    adapter: Any, batch: Any, tmp_path: Path
) -> None:
    """Published and reloaded, it scores identically."""
    from password_attack_detector.ml.inference import InferenceModel

    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    directory = publish(tmp_path / "baseline", fitted, batch.preprocessor)
    loaded = InferenceModel.load(directory)
    before = adapter.score(fitted, batch.matrix.rows, batch.matrix.output_feature_names)
    after = loaded.score(batch.matrix.rows, batch.matrix.output_feature_names)
    assert after == before


def test_a_baseline_is_never_marked_champion(batch: Any) -> None:
    """Champion eligibility is a catalog property; promotion is a later decision.

    Being eligible is not being selected. Nothing in this milestone selects
    anything, and the manifest says so.
    """
    from password_attack_detector.ml.manifest import CHAMPION_NOT_SELECTED

    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    directory = publish(
        Path(__import__("tempfile").mkdtemp()) / "m", fitted, batch.preprocessor
    )
    import json

    from password_attack_detector.ml.manifest import read_model_manifest

    manifest = read_model_manifest(
        json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest.champion_status == CHAMPION_NOT_SELECTED


def test_a_batch_from_a_non_training_split_is_refused(batch: Any) -> None:
    """Enforced by the batch type, so no adapter has to remember it."""
    from password_attack_detector.ml.enums import MLSplit

    with pytest.raises(ValueError, match="fitted on"):
        TrainingBatch(
            split=MLSplit.TEST,
            anchors=batch.batch.anchors,
            transformed_feature_names=batch.batch.transformed_feature_names,
            matrix=batch.batch.matrix,
            targets=batch.batch.targets,
            class_order=batch.batch.class_order,
            preprocessor=batch.preprocessor,
        )
