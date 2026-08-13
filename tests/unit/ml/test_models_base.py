"""The adapter contract itself: what a batch must be, and what the registry enforces.

Two boundaries live here rather than in any one family. A **training batch** is
where "canonical, train-only, aligned" is checked once for everybody, and the
**registry** is where the catalog and the implementations are held to each
other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.catalog import MODEL_CATALOG
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    MLSplit,
    MLTask,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.models import (
    MODEL_IMPLEMENTATIONS,
    PriorBaselineAdapter,
    assert_registry_matches_catalog,
    catalog_spec_for,
)
from password_attack_detector.ml.models.base import (
    ANOMALY_SCORE_COLUMN,
    FittedModel,
    TrainingBatch,
    quantize,
    score_semantics_for,
)
from tests.ml.models import prepare


@pytest.fixture
def batch() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=80)


def rebuilt(source: Any, **changes: Any) -> TrainingBatch:
    """Return a batch like *source* with *changes* applied."""
    fields = {
        "split": source.batch.split,
        "anchors": source.batch.anchors,
        "transformed_feature_names": source.batch.transformed_feature_names,
        "matrix": source.batch.matrix,
        "targets": source.batch.targets,
        "class_order": source.batch.class_order,
        "preprocessor": source.preprocessor,
        "class_weights": source.batch.class_weights,
    }
    fields.update(changes)
    return TrainingBatch(**fields)


# ---------------------------------------------------------------------------
# The batch contract
# ---------------------------------------------------------------------------


def test_a_non_training_split_is_refused(batch: Any) -> None:
    """One rule, enforced once, so no adapter has to remember it."""
    with pytest.raises(ValueError, match="fitted on"):
        rebuilt(batch, split=MLSplit.VALIDATION)


def test_an_empty_batch_is_refused(batch: Any) -> None:
    """A model fitted on nothing is not a model."""
    with pytest.raises(ValueError, match="at least one row"):
        rebuilt(batch, matrix=(), anchors=(), targets=())


def test_a_ragged_matrix_is_refused(batch: Any) -> None:
    """A short row would shift every value one column to the left."""
    ragged = (*batch.batch.matrix[:-1], batch.batch.matrix[-1][:-1])
    with pytest.raises(ValueError, match="as wide as"):
        rebuilt(batch, matrix=ragged)


def test_mismatched_anchors_are_refused(batch: Any) -> None:
    """Anchors and rows describe the same events, one for one."""
    with pytest.raises(ValueError, match="anchors and matrix"):
        rebuilt(batch, anchors=batch.batch.anchors[:-1])


def test_mismatched_targets_are_refused(batch: Any) -> None:
    """A target column that does not cover the rows would silently truncate."""
    with pytest.raises(ValueError, match="targets and matrix"):
        rebuilt(batch, targets=batch.batch.targets[:-1])


def test_a_feature_order_disagreeing_with_the_preprocessor_is_refused(
    batch: Any,
) -> None:
    """The batch must be the matrix the supplied preprocessor produced."""
    swapped = (
        batch.batch.transformed_feature_names[1],
        batch.batch.transformed_feature_names[0],
        *batch.batch.transformed_feature_names[2:],
    )
    with pytest.raises(ValueError, match="disagrees with the preprocessor"):
        rebuilt(batch, transformed_feature_names=swapped)


def test_a_repeated_class_in_the_order_is_refused(batch: Any) -> None:
    """One class cannot occupy two positions in a score vector."""
    with pytest.raises(ValueError, match="repeats a class"):
        rebuilt(batch, class_order=("benign", "benign"))


def test_a_target_outside_the_class_order_is_refused(batch: Any) -> None:
    """A class nobody declared cannot be encoded."""
    stray = ("suspicious", *batch.batch.targets[1:])
    with pytest.raises(ValueError, match="outside the declared class order"):
        rebuilt(batch, targets=stray)


def test_non_canonical_rows_are_refused_at_fit(batch: Any) -> None:
    """Asserted, never sorted: a fit that re-sorted would hide the real fault."""
    scrambled = rebuilt(batch, anchors=tuple(reversed(batch.batch.anchors)))
    with pytest.raises(ModelTrainingError, match="canonical"):
        PriorBaselineAdapter().fit(scrambled, task=MLTask.BINARY_MALICIOUS)


def test_targets_are_encoded_by_the_declared_order(batch: Any) -> None:
    """No library picks the integer encoding; the declared order does."""
    encoded = batch.batch.encoded_targets()
    position = {name: index for index, name in enumerate(batch.batch.class_order)}
    assert list(encoded) == [position[value] for value in batch.batch.targets]


def test_sample_weights_come_from_the_recorded_class_weight_state(
    batch: Any,
) -> None:
    """The number that reaches the loss is the number the manifest records."""
    weights = batch.batch.sample_weights()
    mapping = batch.batch.class_weights.as_mapping()
    assert list(weights) == [mapping[value] for value in batch.batch.targets]


def test_an_unweighted_batch_supplies_no_weights() -> None:
    """``None`` rather than a vector of ones, so the absence is visible."""
    unweighted = prepare(count=60, weighted=False)
    assert unweighted.batch.sample_weights() is None


def test_a_class_weight_state_missing_a_class_is_refused(batch: Any) -> None:
    """Weighting a class the state does not cover would silently default it."""
    from password_attack_detector.ml.config import ImbalanceConfig
    from password_attack_detector.ml.imbalance import compute_class_weights

    narrow = compute_class_weights(
        ["benign"] * 5 + ["other"] * 5,
        task=MLTask.BINARY_MALICIOUS,
        class_order=("benign", "other"),
        config=ImbalanceConfig(),
    )
    mismatched = rebuilt(batch, class_weights=narrow)
    with pytest.raises(ModelTrainingError, match="does not cover"):
        mismatched.sample_weights()


def test_the_design_matrix_is_normalised(batch: Any) -> None:
    """Contiguous, little-endian float64: one representation, always."""
    design = batch.batch.design()
    assert design.dtype == np.dtype("<f8")
    assert design.flags["C_CONTIGUOUS"]
    assert design.shape == (
        len(batch.batch.matrix),
        len(batch.batch.transformed_feature_names),
    )


# ---------------------------------------------------------------------------
# Score semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "kind"),
    [
        (MLTask.BINARY_MALICIOUS, ScoreKind.DECISION_SCORE),
        (MLTask.ATTACK_CATEGORY, ScoreKind.CLASS_SCORE),
        (MLTask.ANOMALY, ScoreKind.ANOMALY_SCORE),
    ],
)
def test_no_task_produces_a_probability(task: MLTask, kind: ScoreKind) -> None:
    """Milestone 4 fits no calibrator, so no output may be called one."""
    semantics = score_semantics_for(task)
    assert semantics.score_kind is kind
    assert semantics.calibration_method is CalibrationMethod.NONE
    lowered = semantics.description.lower()
    assert "probability" not in lowered
    assert "likelihood" not in lowered


# ---------------------------------------------------------------------------
# The fitted model contract
# ---------------------------------------------------------------------------


def test_a_supervised_model_needs_at_least_two_classes(batch: Any) -> None:
    """A one-class score vector says nothing."""
    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    payload = fitted.model_dump()
    payload["class_order"] = ("benign",)
    with pytest.raises(ValueError, match="at least two classes"):
        FittedModel(**payload)


def test_an_anomaly_model_may_not_declare_classes() -> None:
    """It emits an ordered magnitude, not a class."""
    unlabelled = prepare(count=60, task=MLTask.ANOMALY)
    from password_attack_detector.ml.models import IsolationForestAdapter

    fitted = IsolationForestAdapter(n_estimators=5, max_samples=16).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    payload = fitted.model_dump()
    payload["class_order"] = ("a", "b")
    with pytest.raises(ValueError, match="has no classes"):
        FittedModel(**payload)


def test_the_score_columns_follow_the_task(batch: Any) -> None:
    """Class order for a supervised head, one named column for the anomaly head."""
    supervised = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    assert supervised.score_columns == batch.batch.class_order

    unlabelled = prepare(count=60, task=MLTask.ANOMALY)
    from password_attack_detector.ml.models import IsolationForestAdapter

    unsupervised = IsolationForestAdapter(n_estimators=5, max_samples=16).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    assert unsupervised.score_columns == (ANOMALY_SCORE_COLUMN,)


def test_a_matrix_narrower_than_the_contract_is_refused(batch: Any) -> None:
    """Checked once on the base class, so every family inherits the rule."""
    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    with pytest.raises(ModelTrainingError, match="missing"):
        fitted.require_matrix(batch.matrix.rows, fitted.transformed_feature_names[:-1])


def test_a_row_of_the_wrong_width_is_refused(batch: Any) -> None:
    """The declared width is checked per row."""
    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    with pytest.raises(ModelTrainingError, match="value"):
        fitted.require_matrix(
            [batch.matrix.rows[0][:-1]], fitted.transformed_feature_names
        )


def test_a_non_finite_matrix_value_is_refused(batch: Any) -> None:
    """Preprocessing guarantees finiteness, so this matrix did not come from it."""
    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    bad = (float("nan"), *batch.matrix.rows[0][1:])
    with pytest.raises(ModelTrainingError, match="non-finite"):
        fitted.require_matrix([bad], fitted.transformed_feature_names)


def test_a_non_finite_parameter_cannot_be_quantized() -> None:
    """A fitted parameter is a number, and NaN is not one."""
    with pytest.raises(ModelTrainingError, match="finite"):
        quantize(float("inf"))


def test_the_content_payload_carries_no_path_or_timestamp(batch: Any) -> None:
    """Identity is semantic, so nothing environmental may enter it."""
    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    rendered = str(fitted.content_payload())
    assert "/home/" not in rendered
    assert "created_at" not in rendered
    assert "fitted_at" not in rendered


def test_the_fitted_model_declares_no_prohibited_field() -> None:
    """The schema carries no ground-truth-shaped field name."""
    from password_attack_detector.ml.schemas import prohibited_metadata_fields

    assert prohibited_metadata_fields(list(FittedModel.model_fields)) == ()


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_registry_check_runs_and_passes() -> None:
    """Called at import; asserted here so the check itself is exercised."""
    assert_registry_matches_catalog()


def test_a_family_missing_an_implementation_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog entry with no implementation advertises a model nobody can fit."""
    from password_attack_detector.ml import models as registry

    reduced = {
        family: adapter
        for family, adapter in registry.MODEL_IMPLEMENTATIONS.items()
        if family is not ModelFamily.RANDOM_FOREST
    }
    monkeypatch.setattr(registry, "MODEL_IMPLEMENTATIONS", reduced)
    with pytest.raises(ModelTrainingError, match="no implementation"):
        registry.assert_registry_matches_catalog()


def test_a_mismatched_serializer_id_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An artifact would otherwise claim a reader it was not written by."""
    from password_attack_detector.ml import models as registry

    class Wrong:
        """A stand-in whose serializer identifier disagrees with the catalog."""

        catalog_model_id = "M-000"
        serializer_id = "json_something_else_v1"
        inference_adapter_id = "prior_v1"
        supported_tasks = (MLTask.BINARY_MALICIOUS,)
        champion_eligible = False
        reference_baseline = True

    altered: dict[Any, Any] = dict(registry.MODEL_IMPLEMENTATIONS)
    altered[ModelFamily.PRIOR_BASELINE] = Wrong
    monkeypatch.setattr(registry, "MODEL_IMPLEMENTATIONS", altered)
    with pytest.raises(ModelTrainingError, match="serializer_id"):
        registry.assert_registry_matches_catalog()


def test_an_implementation_supporting_an_undeclared_task_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family may not be fitted for a task its declaration excludes."""
    from password_attack_detector.ml import models as registry

    class Overreaching:
        """A stand-in claiming a task its catalog entry does not declare."""

        catalog_model_id = "M-000"
        serializer_id = "json_prior_v1"
        inference_adapter_id = "prior_v1"
        supported_tasks = (MLTask.BINARY_MALICIOUS, MLTask.ANOMALY)
        champion_eligible = False
        reference_baseline = True

    altered: dict[Any, Any] = dict(registry.MODEL_IMPLEMENTATIONS)
    altered[ModelFamily.PRIOR_BASELINE] = Overreaching
    monkeypatch.setattr(registry, "MODEL_IMPLEMENTATIONS", altered)
    with pytest.raises(ModelTrainingError, match="does not declare"):
        registry.assert_registry_matches_catalog()


def test_an_implementation_supporting_no_task_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family that declares nothing cannot be dispatched to."""
    from password_attack_detector.ml import models as registry

    class Empty:
        """A stand-in that declares no task at all."""

        catalog_model_id = "M-000"
        serializer_id = "json_prior_v1"
        inference_adapter_id = "prior_v1"
        supported_tasks: tuple[MLTask, ...] = ()
        champion_eligible = False
        reference_baseline = True

    altered: dict[Any, Any] = dict(registry.MODEL_IMPLEMENTATIONS)
    altered[ModelFamily.PRIOR_BASELINE] = Empty
    monkeypatch.setattr(registry, "MODEL_IMPLEMENTATIONS", altered)
    with pytest.raises(ModelTrainingError, match="no task at all"):
        registry.assert_registry_matches_catalog()


def test_every_implemented_family_has_a_catalog_entry() -> None:
    """The other direction: an unreviewed family cannot ship."""
    for family in MODEL_IMPLEMENTATIONS:
        assert catalog_spec_for(family).family is family


def test_a_family_the_catalog_does_not_declare_is_refused() -> None:
    """Looking one up is an error rather than a silent ``None``."""
    from password_attack_detector.ml import models as registry

    class Fictional:
        pass

    with pytest.raises(ModelTrainingError, match="declares no family"):
        registry.catalog_spec_for(Fictional)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# M-000 is the reference, never a candidate
#
# The invariant is stated in three places -- the catalog entry, the adapter, and
# a fitted model -- and asserted in all three, because the failure this guards
# against is one of them being edited on its own and turning the comparator into
# an automatic fallback champion.
# ---------------------------------------------------------------------------


def test_the_prior_baseline_is_publishable_and_never_eligible() -> None:
    """The direct invariant, on the adapter."""
    assert PriorBaselineAdapter.publishable is True
    assert PriorBaselineAdapter.champion_eligible is False
    assert PriorBaselineAdapter.reference_baseline is True


def test_a_fitted_prior_baseline_is_never_eligible(batch: Any) -> None:
    """The flag travels into the fitted model, and into its artifact."""
    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    assert fitted.champion_eligible is False
    assert fitted.experimental is False


def test_the_prior_baseline_is_not_experimental() -> None:
    """Ineligible for promotion is not the same as unfinished.

    It is fully implemented, fully serialisable, and reported in every
    comparison. Marking it experimental would understate all of that and would
    also imply its serializer contract is unproven, which it is not.
    """
    spec = catalog_spec_for(ModelFamily.PRIOR_BASELINE)
    assert spec.experimental is False
    assert spec.deprecated is False
    assert spec.serializer_id and spec.inference_adapter_id
    assert ModelFamily.PRIOR_BASELINE in registry_publishable()


def registry_publishable() -> Any:
    """Return the publishable family set, imported lazily for readability."""
    from password_attack_detector.ml.models import PUBLISHABLE_FAMILIES

    return PUBLISHABLE_FAMILIES


def test_the_champion_eligible_set_is_exactly_the_approved_one() -> None:
    """Pinned, with each absence carrying its own reason.

    ``M-000`` is the reference every candidate must beat; ``M-021`` is gated on
    an undocumented estimator interface; ``M-030`` is unsupervised. Nothing
    else may join the set without this assertion being edited.
    """
    from password_attack_detector.ml.models import (
        CHAMPION_ELIGIBLE_MODEL_IDS,
        REFERENCE_BASELINE_MODEL_IDS,
    )

    assert set(CHAMPION_ELIGIBLE_MODEL_IDS) == {"M-001", "M-010", "M-020"}
    assert set(REFERENCE_BASELINE_MODEL_IDS) == {"M-000"}
    assert not CHAMPION_ELIGIBLE_MODEL_IDS & REFERENCE_BASELINE_MODEL_IDS


def test_every_family_declares_eligibility_consistently_with_the_catalog() -> None:
    """Adapter and catalog entry agree, per family, in both directions."""
    for spec in MODEL_CATALOG.specs:
        adapter: Any = MODEL_IMPLEMENTATIONS[spec.family]
        assert adapter.champion_eligible == spec.champion_eligible, spec.model_id
        assert adapter.reference_baseline == spec.reference_baseline, spec.model_id


def test_an_adapter_claiming_eligibility_the_catalog_denies_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No registry transformation can silently promote the reference baseline.

    This is the failure mode the whole correction exists to prevent: an adapter
    that marked its own fitted models promotable while the reviewed catalog
    entry said otherwise would make M-000 a fallback champion that always
    passes, and ``NO_ELIGIBLE_CHAMPION`` would stop being reachable.
    """
    from password_attack_detector.ml import models as registry

    class Promoted:
        """A stand-in that claims the eligibility its catalog entry denies."""

        catalog_model_id = "M-000"
        serializer_id = "json_prior_v1"
        inference_adapter_id = "prior_v1"
        supported_tasks = (MLTask.BINARY_MALICIOUS,)
        champion_eligible = True
        reference_baseline = False

    altered: dict[Any, Any] = dict(registry.MODEL_IMPLEMENTATIONS)
    altered[ModelFamily.PRIOR_BASELINE] = Promoted
    monkeypatch.setattr(registry, "MODEL_IMPLEMENTATIONS", altered)
    with pytest.raises(ModelTrainingError, match="champion_eligible"):
        registry.assert_registry_matches_catalog()


def test_a_reference_baseline_marked_eligible_fails_the_registry_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checked at the registry level too, not only by the spec validator.

    A catalog constructed outside the spec's own validation -- a stub, a
    fixture, a future loader -- still cannot slip a promotable baseline past
    the registry.
    """
    from password_attack_detector.ml import models as registry

    class Contradictory:
        """A catalog stand-in that is both the reference and a candidate."""

        model_id = "M-000"
        family = ModelFamily.PRIOR_BASELINE
        serializer_id = "json_prior_v1"
        inference_adapter_id = "prior_v1"
        supported_tasks = (MLTask.BINARY_MALICIOUS,)
        champion_eligible = True
        reference_baseline = True

    class Agreeing:
        """An adapter that agrees with the contradictory entry."""

        catalog_model_id = "M-000"
        serializer_id = "json_prior_v1"
        inference_adapter_id = "prior_v1"
        supported_tasks = (MLTask.BINARY_MALICIOUS,)
        champion_eligible = True
        reference_baseline = True

    class Stub:
        specs = (Contradictory(),)

    altered: dict[Any, Any] = {ModelFamily.PRIOR_BASELINE: Agreeing}
    monkeypatch.setattr(registry, "MODEL_CATALOG", Stub())
    monkeypatch.setattr(registry, "MODEL_IMPLEMENTATIONS", altered)
    with pytest.raises(ModelTrainingError, match="reference baseline"):
        registry.assert_registry_matches_catalog()


def test_loading_the_reference_baseline_as_a_champion_is_refused(
    batch: Any, tmp_path: Any
) -> None:
    """End to end: a caller requiring a promotable model does not get M-000."""
    from password_attack_detector.exceptions import ModelNotReadyError
    from password_attack_detector.ml.inference import InferenceModel
    from tests.ml.models import publish

    fitted = PriorBaselineAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    directory = publish(tmp_path / "prior", fitted, batch.preprocessor)

    # It loads perfectly well as an ordinary model -- it is fully publishable.
    InferenceModel.load(directory)

    with pytest.raises(ModelNotReadyError, match="champion-eligible"):
        InferenceModel.load(directory, require_champion_eligible=True)
