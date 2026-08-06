"""Tests for the executable model catalog.

The catalog is the layer's registry, its documentation source, and one of its
fingerprints.  Most of these tests protect *eligibility*: a model must not be
able to declare itself promotable while reading a private estimator attribute,
while being experimental, or while being unable to perform the binary task.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from password_attack_detector.exceptions import MLConfigurationError
from password_attack_detector.ml.catalog import (
    MODEL_CATALOG,
    MODEL_CATALOG_VERSION,
    ModelCatalog,
    ModelSpec,
    build_model_catalog,
    model_catalog_to_markdown,
)
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    HyperparameterKind,
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.schemas import HyperparameterSpec

EXPECTED_MODEL_IDS = ("M-000", "M-001", "M-010", "M-020", "M-021", "M-030")


def _minimal_spec(**overrides: object) -> ModelSpec:
    """Return a valid champion-eligible spec, with *overrides* applied."""
    base: dict[str, object] = {
        "model_id": "M-900",
        "model_version": "1.0.0",
        "family": ModelFamily.PRIOR_BASELINE,
        "display_name": "Test family",
        "description": "A family declared for testing.",
        "supported_tasks": (MLTask.BINARY_MALICIOUS,),
        "native_score_kind": ScoreKind.DECISION_SCORE,
        "calibration_compatible": True,
        "supported_calibration_methods": (
            CalibrationMethod.NONE,
            CalibrationMethod.PLATT,
        ),
        "multiclass_capable": False,
        "champion_eligible": True,
        "eligibility_status": ModelEligibilityStatus.CHAMPION_ELIGIBLE,
        "serializer_id": "json_test_v1",
        "inference_adapter_id": "test_v1",
        "determinism_controls": ("closed-form fit",),
        "requires_sklearn": False,
    }
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Catalog membership and identity
# ---------------------------------------------------------------------------


def test_the_catalog_declares_the_six_planned_families() -> None:
    """Model identifiers are stable once published."""
    assert MODEL_CATALOG.model_ids == EXPECTED_MODEL_IDS
    assert len(MODEL_CATALOG) == 6


def test_model_identifiers_are_unique() -> None:
    """A repeated identifier would alias two families."""
    assert len(set(MODEL_CATALOG.model_ids)) == len(MODEL_CATALOG.model_ids)


def test_model_families_are_unique() -> None:
    """One spec per family: configuration looks a family up by name."""
    families = [spec.family for spec in MODEL_CATALOG.specs]
    assert len(set(families)) == len(families)


def test_every_declared_family_enum_member_has_a_spec() -> None:
    """A family the enum names but the catalog omits could never be fitted."""
    assert set(MODEL_CATALOG.families) == set(ModelFamily)


def test_the_catalog_is_ordered_by_model_identifier() -> None:
    """Iteration order is a property of the data, not of import order."""
    assert list(MODEL_CATALOG.model_ids) == sorted(MODEL_CATALOG.model_ids)


def test_a_duplicate_identifier_is_rejected() -> None:
    """Two specs sharing an identifier must fail loudly at construction."""
    spec = _minimal_spec()
    with pytest.raises(MLConfigurationError, match="Duplicate model identifier"):
        ModelCatalog((spec, spec))


def test_a_duplicate_family_is_rejected() -> None:
    """Two specs sharing a family would make family lookup ambiguous."""
    first = _minimal_spec(model_id="M-900")
    second = _minimal_spec(model_id="M-901")
    with pytest.raises(MLConfigurationError, match="Duplicate model family"):
        ModelCatalog((first, second))


def test_looking_up_an_unknown_model_fails_loudly() -> None:
    """A typo must not silently return nothing."""
    with pytest.raises(MLConfigurationError, match="Unknown model identifier"):
        MODEL_CATALOG.get("M-999")


def test_membership_and_lookup_agree() -> None:
    """``has`` and ``get`` must describe the same registry."""
    for model_id in MODEL_CATALOG.model_ids:
        assert MODEL_CATALOG.has(model_id)
        assert model_id in MODEL_CATALOG
        assert MODEL_CATALOG.get(model_id).model_id == model_id
    assert not MODEL_CATALOG.has("M-999")


# ---------------------------------------------------------------------------
# Eligibility -- the rules that keep a model from promoting itself
# ---------------------------------------------------------------------------


def test_exactly_the_four_intended_families_are_champion_eligible() -> None:
    """The gated and anomaly families are excluded, by declaration."""
    eligible = {spec.model_id for spec in MODEL_CATALOG.champion_eligible_specs()}
    assert eligible == {"M-000", "M-001", "M-010", "M-020"}


def test_the_gated_boosting_family_is_not_champion_eligible() -> None:
    """M-021 ships gated until its serializer contract is proven."""
    spec = MODEL_CATALOG.get("M-021")
    assert not spec.champion_eligible
    assert spec.eligibility_status is ModelEligibilityStatus.SERIALIZER_UNPROVEN
    assert spec.private_estimator_attributes


def test_the_anomaly_family_is_experimental_and_never_champion() -> None:
    """M-030 is unsupervised; it has no supervised class to be champion of."""
    spec = MODEL_CATALOG.get("M-030")
    assert spec.anomaly_only
    assert spec.experimental
    assert not spec.champion_eligible
    assert spec.eligibility_status is ModelEligibilityStatus.ANOMALY_ONLY
    assert spec.supported_tasks == (MLTask.ANOMALY,)
    assert spec.native_score_kind is ScoreKind.ANOMALY_SCORE
    assert not spec.calibration_compatible


def test_no_family_reading_a_private_attribute_may_be_champion_eligible() -> None:
    """The invariant that gates M-021, checked against the whole catalog.

    Reading a private attribute is a bet on an internal layout. A model built
    on one cannot be promoted until a compatibility test settles that bet.
    """
    for spec in MODEL_CATALOG.specs:
        if spec.private_estimator_attributes:
            assert not spec.champion_eligible, spec.model_id


def test_declaring_a_private_attribute_and_championship_together_is_rejected() -> None:
    """The rule is enforced at declaration, not merely observed in the data."""
    with pytest.raises(ValidationError, match="private estimator attribute"):
        _minimal_spec(
            requires_sklearn=True,
            estimator_class_name="SomeEstimator",
            public_estimator_attributes=("classes_",),
            private_estimator_attributes=("_predictors",),
        )


def test_an_experimental_family_cannot_be_champion_eligible() -> None:
    """Experimental means "not for promotion", and the type system says so."""
    with pytest.raises(ValidationError, match="experimental"):
        _minimal_spec(experimental=True)


def test_a_deprecated_family_cannot_be_champion_eligible() -> None:
    """A family on the way out must not become the thing everything depends on."""
    with pytest.raises(ValidationError, match="deprecated"):
        _minimal_spec(deprecated=True)


def test_a_family_that_cannot_do_the_binary_task_cannot_be_champion() -> None:
    """The champion is the binary model compared against the rule engine."""
    with pytest.raises(ValidationError, match="without supporting the binary task"):
        _minimal_spec(
            supported_tasks=(MLTask.ATTACK_CATEGORY,),
            multiclass_capable=True,
        )


def test_eligibility_status_and_the_flag_must_agree_in_both_directions() -> None:
    """A status and a flag that disagree would make the catalog unreadable."""
    with pytest.raises(ValidationError, match="declares status"):
        _minimal_spec(eligibility_status=ModelEligibilityStatus.EXPERIMENTAL)
    with pytest.raises(ValidationError, match="without the champion_eligible flag"):
        _minimal_spec(
            champion_eligible=False,
            eligibility_status=ModelEligibilityStatus.CHAMPION_ELIGIBLE,
        )


def test_an_anomaly_only_family_cannot_declare_a_supervised_task() -> None:
    """Anomaly and supervised fitting have different contracts."""
    with pytest.raises(ValidationError, match="anomaly-only but declares"):
        _minimal_spec(
            champion_eligible=False,
            eligibility_status=ModelEligibilityStatus.ANOMALY_ONLY,
            anomaly_only=True,
            supported_tasks=(MLTask.BINARY_MALICIOUS,),
        )


def test_a_family_cannot_mix_the_anomaly_task_with_a_supervised_one() -> None:
    """One estimator cannot honour both fitting contracts at once."""
    with pytest.raises(ValidationError, match="mixes the anomaly task"):
        _minimal_spec(
            champion_eligible=False,
            eligibility_status=ModelEligibilityStatus.EXPERIMENTAL,
            experimental=True,
            supported_tasks=(MLTask.BINARY_MALICIOUS, MLTask.ANOMALY),
        )


# ---------------------------------------------------------------------------
# Score and calibration semantics
# ---------------------------------------------------------------------------


def test_no_family_claims_a_calibrated_probability_as_its_native_output() -> None:
    """Calibration is a separately fitted stage, never an estimator's default."""
    for spec in MODEL_CATALOG.specs:
        assert spec.native_score_kind is not ScoreKind.CALIBRATED_PROBABILITY


def test_declaring_a_calibrated_native_output_is_rejected() -> None:
    """A family claiming otherwise would let a raw number be read as a rate."""
    with pytest.raises(ValidationError, match="calibrated probability as its native"):
        _minimal_spec(native_score_kind=ScoreKind.CALIBRATED_PROBABILITY)


def test_a_non_calibratable_family_may_declare_only_none() -> None:
    """Offering a method a family cannot honour would be a false promise."""
    with pytest.raises(ValidationError, match="may declare only 'none'"):
        _minimal_spec(
            calibration_compatible=False,
            supported_calibration_methods=(CalibrationMethod.ISOTONIC,),
        )


def test_a_calibratable_family_must_offer_a_method() -> None:
    """Compatible-but-offering-nothing is a contradiction."""
    with pytest.raises(ValidationError, match="declares no calibration method"):
        _minimal_spec(supported_calibration_methods=(CalibrationMethod.NONE,))


def test_a_category_capable_family_must_be_multiclass_capable() -> None:
    """A binary estimator cannot serve the category head."""
    with pytest.raises(ValidationError, match="not multiclass capable"):
        _minimal_spec(
            supported_tasks=(MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY),
            multiclass_capable=False,
        )


# ---------------------------------------------------------------------------
# Declaration hygiene
# ---------------------------------------------------------------------------


def test_no_spec_names_an_importable_module_path() -> None:
    """Model logic is never loaded from data; a dotted name could look like it."""
    for spec in MODEL_CATALOG.specs:
        if spec.estimator_class_name is not None:
            assert "." not in spec.estimator_class_name
            assert "/" not in spec.estimator_class_name


def test_a_dotted_estimator_name_is_rejected() -> None:
    """The dot check is the guard against a data-driven import creeping in."""
    with pytest.raises(ValidationError, match="no module path"):
        _minimal_spec(
            requires_sklearn=True,
            public_estimator_attributes=("classes_",),
            estimator_class_name="sklearn.linear_model.LogisticRegression",
        )


def test_a_family_needing_no_estimator_declares_no_attributes() -> None:
    """A hand-written baseline has no third-party internals to read."""
    with pytest.raises(ValidationError, match="requires no third-party estimator"):
        _minimal_spec(public_estimator_attributes=("coef_",))


def test_a_family_using_an_estimator_must_declare_a_readable_attribute() -> None:
    """A serializer with nothing to read could not export the fitted model."""
    with pytest.raises(ValidationError, match="declares no public attribute"):
        _minimal_spec(requires_sklearn=True, estimator_class_name="SomeEstimator")


def test_every_family_declares_how_it_is_made_reproducible() -> None:
    """Determinism is a contract, so it must be stated per family."""
    for spec in MODEL_CATALOG.specs:
        assert spec.determinism_controls


def test_a_family_with_no_determinism_control_is_rejected() -> None:
    """Silence about reproducibility is not the same as having none to state."""
    with pytest.raises(ValidationError, match="determinism controls"):
        _minimal_spec(determinism_controls=())


@pytest.mark.parametrize("term", ["probability", "proves", "guaranteed"])
def test_a_claim_asserting_description_is_rejected(term: str) -> None:
    """A family describes a method; it never asserts what its output proves."""
    with pytest.raises(ValidationError, match="claim-asserting"):
        _minimal_spec(description=f"This family {term} that traffic is hostile.")


def test_no_shipped_description_asserts_a_claim() -> None:
    """The rule holds for the catalog as shipped, not only for new entries."""
    for spec in MODEL_CATALOG.specs:
        lowered = spec.description.lower()
        for term in ("probability", "likelihood", "proves", "guaranteed"):
            assert term not in lowered, spec.model_id


def test_adapter_identifiers_are_versioned() -> None:
    """A changed output shape gets a new identifier, not a new meaning."""
    for spec in MODEL_CATALOG.specs:
        assert spec.serializer_id.rsplit("_", 1)[-1].startswith("v")
        assert spec.inference_adapter_id.rsplit("_", 1)[-1].startswith("v")


def test_an_unversioned_adapter_identifier_is_rejected() -> None:
    """Without a version suffix a contract change would be invisible."""
    with pytest.raises(ValidationError, match="adapter identifier"):
        _minimal_spec(serializer_id="json_test")


def test_every_stochastic_family_pins_a_seed() -> None:
    """A family using an estimator must declare how its randomness is fixed."""
    for spec in MODEL_CATALOG.specs:
        if not spec.requires_sklearn:
            continue
        names = {parameter.name for parameter in spec.hyperparameters}
        assert "random_state" in names, spec.model_id


def test_thread_counts_are_pinned_to_one_where_declared() -> None:
    """Parallel reductions reorder floating-point sums and break reproduction."""
    for spec in MODEL_CATALOG.specs:
        for parameter in spec.hyperparameters:
            if parameter.name == "n_jobs":
                assert parameter.default == 1
                assert parameter.minimum == 1
                assert parameter.maximum == 1
                assert not parameter.tunable


def test_a_repeated_hyperparameter_name_is_rejected() -> None:
    """Two declarations of one name would make the effective value ambiguous."""
    duplicate = HyperparameterSpec(
        name="max_iter",
        kind=HyperparameterKind.INT,
        default=10,
        minimum=1,
        description="Iterations.",
    )
    with pytest.raises(ValidationError, match="repeats a hyperparameter name"):
        _minimal_spec(hyperparameters=(duplicate, duplicate))


# ---------------------------------------------------------------------------
# Hyperparameter resolution
# ---------------------------------------------------------------------------


def test_effective_hyperparameters_start_from_the_declared_defaults() -> None:
    """With no overrides, the effective values are the declared ones."""
    spec = MODEL_CATALOG.for_family(ModelFamily.RANDOM_FOREST)
    assert spec.effective_hyperparameters() == spec.default_hyperparameters()


def test_effective_hyperparameters_apply_and_validate_overrides() -> None:
    """An override inside its bounds is applied; outside, it is rejected."""
    spec = MODEL_CATALOG.for_family(ModelFamily.RANDOM_FOREST)
    effective = spec.effective_hyperparameters({"n_estimators": 50})
    assert effective["n_estimators"] == 50

    with pytest.raises(MLConfigurationError, match="at most"):
        spec.effective_hyperparameters({"n_estimators": 999_999})


def test_an_undeclared_override_is_rejected_rather_than_ignored() -> None:
    """There is no passthrough to the estimator; a typo must fail."""
    spec = MODEL_CATALOG.for_family(ModelFamily.LOGISTIC_REGRESSION)
    with pytest.raises(MLConfigurationError, match="declares no hyperparameter"):
        spec.effective_hyperparameters({"n_estimators": 10})


def test_supports_reports_declared_tasks() -> None:
    """The accessor agrees with the declaration."""
    spec = MODEL_CATALOG.for_family(ModelFamily.SINGLE_FEATURE_THRESHOLD)
    assert spec.supports(MLTask.BINARY_MALICIOUS)
    assert not spec.supports(MLTask.ATTACK_CATEGORY)


def test_specs_for_task_returns_catalog_order() -> None:
    """Reporting reads this; order must not depend on iteration accidents."""
    binary = MODEL_CATALOG.specs_for_task(MLTask.BINARY_MALICIOUS)
    assert [spec.model_id for spec in binary] == sorted(
        spec.model_id for spec in binary
    )
    assert MODEL_CATALOG.specs_for_task(MLTask.ANOMALY) == (MODEL_CATALOG.get("M-030"),)


# ---------------------------------------------------------------------------
# Fingerprint and rendering determinism
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_a_stable_lower_case_digest() -> None:
    """Two calls agree, and the shape is a SHA-256 hex digest."""
    first = MODEL_CATALOG.fingerprint()
    second = build_model_catalog().fingerprint()
    assert first == second
    assert len(first) == 64
    assert first == first.lower()
    assert set(first) <= set("0123456789abcdef")


def test_the_fingerprint_ignores_prose() -> None:
    """Fixing a typo must not invalidate every artifact recording the digest."""
    original = _minimal_spec()
    reworded = _minimal_spec(
        display_name="Renamed family",
        description="Entirely different wording for the same declaration.",
        limitations=("A newly documented caveat.",),
    )
    assert ModelCatalog((original,)).fingerprint() == (
        ModelCatalog((reworded,)).fingerprint()
    )


def test_the_fingerprint_tracks_behavioural_change() -> None:
    """A changed hyperparameter default must move the digest."""
    original = _minimal_spec()
    changed = _minimal_spec(
        hyperparameters=(
            HyperparameterSpec(
                name="max_iter",
                kind=HyperparameterKind.INT,
                default=10,
                minimum=1,
                description="Iterations.",
            ),
        )
    )
    assert ModelCatalog((original,)).fingerprint() != (
        ModelCatalog((changed,)).fingerprint()
    )


def test_the_fingerprint_tracks_eligibility_change() -> None:
    """Promoting a family is a contract change and must be visible."""
    eligible = _minimal_spec()
    gated = _minimal_spec(
        champion_eligible=False,
        eligibility_status=ModelEligibilityStatus.SERIALIZER_UNPROVEN,
    )
    assert ModelCatalog((eligible,)).fingerprint() != (
        ModelCatalog((gated,)).fingerprint()
    )


def test_the_markdown_is_deterministic() -> None:
    """The generated document must be byte-stable across renders."""
    assert model_catalog_to_markdown() == model_catalog_to_markdown()


def test_the_markdown_records_the_version_and_fingerprint() -> None:
    """A generated document must say what it was generated from."""
    rendered = model_catalog_to_markdown()
    assert MODEL_CATALOG_VERSION in rendered
    assert MODEL_CATALOG.fingerprint() in rendered


def test_the_markdown_states_what_membership_does_not_mean() -> None:
    """The document must not read as an endorsement of every family listed."""
    rendered = model_catalog_to_markdown().lower()
    assert "membership is not championship" in rendered
    assert "serializer and inference parity are required" in rendered
    assert "`m-021` is gated" in rendered
    assert "experimental anomaly model" in rendered
    assert "probabilities require calibration" in rendered
    assert "offline and defensive" in rendered


def test_the_markdown_names_every_model() -> None:
    """A family the document omits would be invisible to a reviewer."""
    rendered = model_catalog_to_markdown()
    for model_id in EXPECTED_MODEL_IDS:
        assert model_id in rendered


def test_the_markdown_carries_no_measured_result() -> None:
    """The catalog declares what may be fitted, never how well anything did."""
    rendered = model_catalog_to_markdown().lower()
    for term in ("accuracy of", "auc of", "f1 of", "precision of", "recall of"):
        assert term not in rendered
