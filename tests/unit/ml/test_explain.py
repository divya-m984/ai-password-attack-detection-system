"""Unit tests for deterministic model attribution.

Two properties are swept across the file. **An explanation reconstructs the
model's own decision quantity or it is not published** -- every exact method is
checked against the scorer rather than against itself, and a decomposition that
does not add up is refused rather than caveated. And **an explanation discloses
nothing it was not asked to** -- feature values stay behind a configuration flag,
row identity stays off the aggregate report, and a family without an exact method
gets a typed refusal instead of an approximation nobody can check.

The models here are fitted, not fixtures: a decomposition tested against a
hand-built array would be testing the array.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelNotReadyError
from password_attack_detector.ml.enums import (
    EXACT_EXPLANATION_METHODS,
    EXPLANATION_ELIGIBLE_SPLITS,
    ExplanationMethod,
    ExplanationStatus,
    MLSplit,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.explain import (
    RECONSTRUCTION_TOLERANCE,
    ExplanationQualityReport,
    FeatureContribution,
    PredictionExplanation,
    explain_predictions,
    explanation_report_to_markdown,
    global_sensitivity,
    local_contributions,
    method_for_family,
)
from password_attack_detector.ml.models.base import FittedModel
from password_attack_detector.ml.schemas import PROHIBITED_METADATA_FIELDS
from tests.ml.models import prepare


class _Model:
    """The narrow surface :func:`explain_predictions` actually consumes.

    Deliberately not an :class:`InferenceModel`: attribution needs a fitted
    model and a scorer and nothing else, and constructing a verified artifact
    directory for every case here would test the loader rather than the
    decomposition.
    """

    def __init__(self, fitted: FittedModel, adapter: Any) -> None:
        self.fitted = fitted
        self._adapter = adapter

    def score(self, rows: Any, columns: Any) -> tuple[tuple[float, ...], ...]:
        """Return what the family's own adapter returns."""
        scored: tuple[tuple[float, ...], ...] = self._adapter.score(
            self.fitted, rows, columns
        )
        return scored


def _fit(adapter: Any) -> tuple[_Model, tuple[tuple[float, ...], ...]]:
    """Fit *adapter* on the shared batch and return it with its design matrix."""
    from password_attack_detector.ml.enums import MLTask

    batch = prepare().batch
    fitted = adapter.fit(batch, task=MLTask.BINARY_MALICIOUS)
    return _Model(fitted, adapter), tuple(tuple(row) for row in batch.matrix)


@pytest.fixture
def linear() -> tuple[_Model, tuple[tuple[float, ...], ...]]:
    """A fitted logistic regression and the matrix it was fitted on."""
    from password_attack_detector.ml.models.linear import LogisticRegressionAdapter

    return _fit(LogisticRegressionAdapter(max_iter=200))


@pytest.fixture
def forest() -> tuple[_Model, tuple[tuple[float, ...], ...]]:
    """A fitted random forest and the matrix it was fitted on."""
    from password_attack_detector.ml.models.forest import RandomForestAdapter

    return _fit(RandomForestAdapter(n_estimators=4, max_depth=3))


# ---------------------------------------------------------------------------
# The method registry
# ---------------------------------------------------------------------------


def test_every_champion_eligible_family_has_an_exact_method() -> None:
    """A family that can be champion is a family whose decisions get explained.

    Not a nice-to-have: a champion nobody can decompose would make ``ml explain``
    report unavailable on the only model anybody runs it against.
    """
    from password_attack_detector.ml.catalog import MODEL_CATALOG

    eligible = {spec.family for spec in MODEL_CATALOG.specs if spec.champion_eligible}
    assert eligible
    for family in eligible:
        assert method_for_family(family) is not None, family


def test_an_unregistered_family_selects_nothing() -> None:
    """The registry is closed; an unknown family falls through to no method."""
    assert method_for_family(ModelFamily.HISTOGRAM_GRADIENT_BOOSTING) is None
    assert method_for_family(ModelFamily.ISOLATION_FOREST) is None
    assert method_for_family(ModelFamily.PRIOR_BASELINE) is None


def test_the_global_method_is_not_in_the_exact_set() -> None:
    """Permutation sensitivity decomposes nothing and must never claim to."""
    assert (
        ExplanationMethod.PERMUTATION_SCORE_SENSITIVITY not in EXACT_EXPLANATION_METHODS
    )
    assert len(EXACT_EXPLANATION_METHODS) == 3


def test_the_evaluation_splits_are_not_explainable() -> None:
    """The absence of the member is the enforcement, exactly as elsewhere."""
    assert MLSplit.TEST not in EXPLANATION_ELIGIBLE_SPLITS
    assert MLSplit.NOVEL_ANOMALY_HOLDOUT not in EXPLANATION_ELIGIBLE_SPLITS
    assert {MLSplit.TRAIN, MLSplit.VALIDATION} == EXPLANATION_ELIGIBLE_SPLITS


# ---------------------------------------------------------------------------
# Exact reconstruction
# ---------------------------------------------------------------------------


def test_linear_contributions_reconstruct_the_logit(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """intercept + sum(value * coefficient) == the decision function.

    Checked against the adapter's own score rather than against a second
    implementation of the dot product, which would only prove the test agrees
    with itself.
    """
    model, matrix = linear
    method, contributions, baselines, decisions = local_contributions(model, matrix)
    assert method is ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION
    for row, baseline, decision in zip(
        contributions, baselines, decisions, strict=True
    ):
        assert abs(decision - (baseline + math.fsum(row))) <= RECONSTRUCTION_TOLERANCE


def test_the_linear_baseline_is_the_fitted_intercept(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """The intercept is recorded separately, not smeared across the columns."""
    model, matrix = linear
    _, _, baselines, _ = local_contributions(model, matrix)
    stored = float(np.asarray(model.fitted.arrays["intercept"], dtype=np.float64)[0])
    assert set(baselines) == {stored}


def test_forest_contributions_reconstruct_the_mean_leaf_score(
    forest: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """The decision-path decomposition telescopes to leaf minus root.

    Averaged over trees that is exactly what ``traverse_forest`` returns, so the
    equality is arithmetic rather than empirical -- and it is asserted against
    the traversal rather than derived from it.
    """
    model, matrix = forest
    method, contributions, baselines, decisions = local_contributions(model, matrix)
    assert method is ExplanationMethod.TREE_PATH_CONTRIBUTION
    for row, baseline, decision in zip(
        contributions, baselines, decisions, strict=True
    ):
        assert abs(decision - (baseline + math.fsum(row))) <= RECONSTRUCTION_TOLERANCE


def test_the_forest_baseline_is_the_ensemble_mean_root_value(
    forest: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """One baseline for the whole ensemble, and it is a property of the trees."""
    model, matrix = forest
    _, _, baselines, _ = local_contributions(model, matrix)
    arrays = model.fitted.arrays
    offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
    values = np.asarray(arrays["leaf_value"], dtype=np.float64)
    expected = float(
        np.mean([values[int(offsets[tree]), 1] for tree in range(len(offsets) - 1)])
    )
    assert set(baselines) == {expected}


def test_the_forest_credits_only_columns_it_actually_split_on(
    forest: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """A column no split names must carry exactly zero, not a small number."""
    model, matrix = forest
    _, contributions, _, _ = local_contributions(model, matrix)
    split_on = {
        int(value)
        for value in np.asarray(model.fitted.arrays["split_feature"], dtype=np.int64)
        if int(value) >= 0
    }
    for row in contributions:
        for index, value in enumerate(row):
            if index not in split_on:
                assert value == 0.0


def test_the_threshold_baseline_credits_one_column_and_zeroes_the_rest() -> None:
    """The model reads one column, so every other contribution is exactly zero.

    Not an approximation: the fitted function is constant in each of them.
    """
    from password_attack_detector.ml.enums import MLTask
    from password_attack_detector.ml.models.baseline import (
        SingleFeatureThresholdAdapter,
    )

    batch = prepare().batch
    column = batch.transformed_feature_names[0]
    adapter = SingleFeatureThresholdAdapter(feature=column, min_non_null_fraction=0.1)
    fitted = adapter.fit(batch, task=MLTask.BINARY_MALICIOUS)
    model = _Model(fitted, adapter)
    matrix = tuple(tuple(row) for row in batch.matrix)

    method, contributions, baselines, decisions = local_contributions(model, matrix)
    assert method is ExplanationMethod.SINGLE_FEATURE_STEP_CONTRIBUTION
    index = int(fitted.parameters["feature_index"])
    for row, baseline, decision in zip(
        contributions, baselines, decisions, strict=True
    ):
        assert baseline == 0.0
        assert all(
            value == 0.0 for position, value in enumerate(row) if position != index
        )
        assert abs(decision - (baseline + math.fsum(row))) <= RECONSTRUCTION_TOLERANCE


def test_an_unsupported_family_refuses_rather_than_approximating() -> None:
    """A family with no exact decomposition gets a refusal, not a guess."""
    from password_attack_detector.ml.enums import MLTask
    from password_attack_detector.ml.models.boosting import (
        HistogramBoostingAdapter,
    )

    batch = prepare().batch
    adapter = HistogramBoostingAdapter(max_iter=5)
    fitted = adapter.fit(batch, task=MLTask.BINARY_MALICIOUS)
    model = _Model(fitted, adapter)

    with pytest.raises(ModelNotReadyError, match="no exact local"):
        local_contributions(model, tuple(tuple(row) for row in batch.matrix))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_decomposition_is_deterministic(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Two runs over the same matrix must agree exactly."""
    model, matrix = linear
    assert local_contributions(model, matrix) == local_contributions(model, matrix)


def test_the_global_measure_is_deterministic(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Seeded per (seed, repeat, column), so the numbers cannot wander."""
    model, matrix = linear
    first = global_sensitivity(model, matrix, repeats=2, seed=42)
    second = global_sensitivity(model, matrix, repeats=2, seed=42)
    assert first == second


def test_the_global_measure_moves_with_the_seed(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """A different seed is a different permutation, and the report says so.

    The point is not that the numbers differ -- it is that the seed is a real
    input, recorded on the report, rather than a decoration.
    """
    model, matrix = linear
    assert global_sensitivity(model, matrix, repeats=2, seed=42) != (
        global_sensitivity(model, matrix, repeats=2, seed=7)
    )


def test_the_global_measure_is_non_negative(
    forest: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """It is a mean absolute change; a negative one would be a bug."""
    model, matrix = forest
    assert all(
        value >= 0.0 for value in global_sensitivity(model, matrix, repeats=2, seed=1)
    )


def test_a_zero_repeat_count_is_refused(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Averaging over nothing is not a measurement."""
    model, matrix = linear
    with pytest.raises(ModelNotReadyError, match="at least one repeat"):
        global_sensitivity(model, matrix, repeats=0, seed=1)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _explain(model: _Model, matrix: Any, **overrides: Any) -> Any:
    """Run the orchestration with CI-sized defaults."""
    arguments: dict[str, Any] = {
        "model": model,
        "matrix": matrix,
        "anchor_event_ids": tuple(
            f"anchor-{index:04d}" for index in range(len(matrix))
        ),
        "scope": MLSplit.VALIDATION,
        "score_kind": ScoreKind.DECISION_SCORE,
        "top_k_features": 5,
        "permutation_repeats": 1,
        "permutation_seed": 42,
        "max_local_explanations": 0,
        "include_feature_values": False,
    }
    arguments.update(overrides)
    return explain_predictions(**arguments)


def test_an_exact_run_reports_its_method_and_residual(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """The status, the method, and the residual are all recorded together."""
    model, matrix = linear
    report, explanations = _explain(model, matrix)
    assert report.status is ExplanationStatus.EXACT
    assert report.method is ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION
    assert report.unavailable_reason is None
    assert report.max_reconstruction_residual is not None
    assert report.max_reconstruction_residual <= RECONSTRUCTION_TOLERANCE
    assert explanations == ()


def test_an_unsupported_family_yields_a_typed_unavailable_with_a_reason() -> None:
    """Unavailable is a status with a code, not an exception and not a zero."""
    from password_attack_detector.ml.enums import MLTask
    from password_attack_detector.ml.models.boosting import (
        HistogramBoostingAdapter,
    )

    batch = prepare().batch
    adapter = HistogramBoostingAdapter(max_iter=5)
    model = _Model(adapter.fit(batch, task=MLTask.BINARY_MALICIOUS), adapter)

    report, explanations = _explain(model, tuple(tuple(r) for r in batch.matrix))
    assert report.status is ExplanationStatus.UNAVAILABLE
    assert report.method is None
    assert report.unavailable_reason == "unsupported_model_family"
    assert report.max_reconstruction_residual is None
    assert report.unused_column_count is None
    assert explanations == ()
    # The global measure is model-agnostic, so it is still reported.
    assert report.top_contributions


def test_an_ineligible_scope_is_refused_by_the_library(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Not only by the command: a caller cannot route around the boundary."""
    model, matrix = linear
    for scope in (MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT):
        with pytest.raises(ModelNotReadyError, match="refused"):
            _explain(model, matrix, scope=scope)


def test_mismatched_anchors_are_refused(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """A row explained under the wrong anchor is worse than none."""
    model, matrix = linear
    with pytest.raises(ModelNotReadyError, match="different row counts"):
        _explain(model, matrix, anchor_event_ids=("only-one",))


def test_the_summary_is_bounded_by_the_configured_top_k(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """A report is a summary, not a dump of every column."""
    model, matrix = linear
    report, _ = _explain(model, matrix, top_k_features=3)
    assert len(report.top_contributions) == 3
    assert report.transformed_feature_count > 3


def test_the_summary_orders_ties_by_column_name(
    forest: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Two equally insensitive columns must not depend on matrix order."""
    model, matrix = forest
    report, _ = _explain(model, matrix, top_k_features=50, permutation_repeats=1)
    rows = [
        (item.mean_absolute_score_change, item.transformed_feature)
        for item in report.top_contributions
    ]
    assert rows == sorted(rows, key=lambda pair: (-pair[0], pair[1]))


def test_the_report_fingerprint_is_stable_across_runs(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Same model, same rows, same configuration -- same sealed identity."""
    model, matrix = linear
    first, _ = _explain(model, matrix)
    second, _ = _explain(model, matrix)
    assert first.explanation_report_fingerprint == second.explanation_report_fingerprint


def test_the_report_fingerprint_moves_with_the_seed(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """The seed changes what was measured, so it must change the identity."""
    model, matrix = linear
    first, _ = _explain(model, matrix, permutation_seed=1)
    second, _ = _explain(model, matrix, permutation_seed=2)
    assert first.explanation_report_fingerprint != second.explanation_report_fingerprint


# ---------------------------------------------------------------------------
# Row explanations and privacy
# ---------------------------------------------------------------------------


def test_no_row_explanation_is_emitted_by_default(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """The bound defaults to zero; disclosure is a deliberate act."""
    model, matrix = linear
    _, explanations = _explain(model, matrix)
    assert explanations == ()


def test_row_explanations_are_bounded_and_chosen_by_anchor(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Which rows are emitted must not depend on scoring order."""
    model, matrix = linear
    _, explanations = _explain(model, matrix, max_local_explanations=3)
    assert len(explanations) == 3
    anchors = [item.anchor_event_id for item in explanations]
    assert anchors == sorted(anchors)


def test_a_row_explanation_hides_feature_values_by_default(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """A feature value can be a country code; names alone are the default."""
    model, matrix = linear
    _, explanations = _explain(model, matrix, max_local_explanations=1)
    assert all(item.transformed_value is None for item in explanations[0].contributions)


def test_feature_values_appear_only_when_explicitly_enabled(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Widening disclosure is a configuration decision somebody made."""
    model, matrix = linear
    _, explanations = _explain(
        model, matrix, max_local_explanations=1, include_feature_values=True
    )
    assert any(
        item.transformed_value is not None for item in explanations[0].contributions
    )


def test_no_schema_here_declares_a_prohibited_field() -> None:
    """The same structural guard the quality report carries."""
    for model in (
        FeatureContribution,
        ExplanationQualityReport,
        PredictionExplanation,
    ):
        offending = set(model.model_fields) & PROHIBITED_METADATA_FIELDS
        # ``anchor_event_id`` is the one join identity a row-level artifact is
        # contractually allowed, and only there.
        assert offending <= {"anchor_event_id"}, (model.__name__, offending)
    assert "anchor_event_id" in PredictionExplanation.model_fields
    assert "anchor_event_id" not in ExplanationQualityReport.model_fields


def test_the_aggregate_report_carries_no_row_identity(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """Not merely absent from the fields: absent from the rendered document."""
    model, matrix = linear
    report, _ = _explain(model, matrix)
    rendered = report.to_json()
    assert "anchor" not in rendered


# ---------------------------------------------------------------------------
# Schema refusals
# ---------------------------------------------------------------------------


def test_a_row_explanation_that_does_not_add_up_is_refused() -> None:
    """The reconstruction is a validator, not a convention."""
    with pytest.raises(ValueError, match="do not reconstruct"):
        PredictionExplanation(
            anchor_event_id="anchor-0001",
            method=ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION,
            decision_value=1.0,
            baseline_value=0.0,
            contributions=(
                FeatureContribution(transformed_feature="a", contribution=0.25),
            ),
            reconstruction_residual=0.0,
        )


def test_a_global_method_may_not_be_recorded_as_a_row_explanation() -> None:
    """Permutation sensitivity decomposes nothing and must not claim a row."""
    with pytest.raises(ValueError, match="decomposes nothing"):
        PredictionExplanation(
            anchor_event_id="anchor-0001",
            method=ExplanationMethod.PERMUTATION_SCORE_SENSITIVITY,
            decision_value=0.0,
            baseline_value=0.0,
            contributions=(),
            reconstruction_residual=0.0,
        )


def test_a_non_finite_contribution_is_refused() -> None:
    """NaN would be stable in a digest and meaningless in a report."""
    with pytest.raises(ValueError, match="finite"):
        FeatureContribution(transformed_feature="a", contribution=float("nan"))


def test_an_exact_report_without_a_residual_is_refused() -> None:
    """Claiming exactness without reporting what it reconstructed to is not a claim."""
    with pytest.raises(ValueError, match="reports the residual"):
        ExplanationQualityReport.seal(
            status=ExplanationStatus.EXACT,
            unavailable_reason=None,
            method=ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            scope=MLSplit.VALIDATION,
            score_kind=ScoreKind.DECISION_SCORE,
            explained_row_count=1,
            transformed_feature_count=1,
            top_contributions=(),
            unused_column_count=0,
            permutation_repeats=1,
            permutation_seed=0,
            max_reconstruction_residual=None,
        )


def test_an_unavailable_report_without_a_reason_is_refused() -> None:
    """A refusal that does not say why is not a refusal anybody can act on."""
    with pytest.raises(ValueError, match="states a reason"):
        ExplanationQualityReport.seal(
            status=ExplanationStatus.UNAVAILABLE,
            unavailable_reason=None,
            method=None,
            model_family=ModelFamily.ISOLATION_FOREST,
            scope=MLSplit.VALIDATION,
            score_kind=ScoreKind.DECISION_SCORE,
            explained_row_count=1,
            transformed_feature_count=1,
            top_contributions=(),
            unused_column_count=None,
            permutation_repeats=1,
            permutation_seed=0,
            max_reconstruction_residual=None,
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_rendered_report_states_that_attribution_is_not_causal(
    linear: tuple[_Model, tuple[tuple[float, ...], ...]],
) -> None:
    """A table of column names beside numbers reads as a causal ranking.

    The caveat is part of the document rather than a footnote, because the
    document is what gets pasted around.
    """
    model, matrix = linear

    class _Manifest:
        explanation_id = "e" * 64
        champion_lock_fingerprint = "a" * 64
        catalog_model_id = "M-010"
        prediction_id = "p" * 36
        prediction_manifest_fingerprint = "b" * 64
        preprocessor_fingerprint = "c" * 64
        local_explanation_count = 0
        include_feature_values = False

    report, _ = _explain(model, matrix)
    rendered = explanation_report_to_markdown(report, cast(Any, _Manifest()))
    lowered = rendered.lower()
    assert "descriptive, not causal" in lowered
    assert "no label was read" in lowered
    assert "calibrated probability" in lowered
    assert "anchor" not in lowered
