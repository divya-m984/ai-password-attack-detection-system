"""Tests for the ML layer's frozen contracts.

Three properties carry most of the weight here: non-finite numbers are refused
rather than propagated, probability terminology is gated on calibration, and no
aggregate-metadata schema can carry identity -- neither in a field name nor in
a field value.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from password_attack_detector.ml import schemas as schemas_module
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    ExperimentRecordType,
    GateStatus,
    HyperparameterKind,
    MLTask,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.schemas import (
    ML_SCHEMA_VERSION,
    PROHIBITED_METADATA_FIELDS,
    ArtifactDeclaration,
    DependencyRequirement,
    ExperimentRecordIdentity,
    GateResult,
    HyperparameterSpec,
    ScoreSemantics,
    SupportRequirement,
    prohibited_metadata_fields,
    version_tuple,
)

DIGEST = "a" * 64
RUN_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"

ALL_SCHEMAS = (
    ArtifactDeclaration,
    DependencyRequirement,
    ExperimentRecordIdentity,
    GateResult,
    HyperparameterSpec,
    ScoreSemantics,
    SupportRequirement,
)


def _identity() -> ExperimentRecordIdentity:
    """Return a minimal valid ledger identity."""
    return ExperimentRecordIdentity(
        record_type=ExperimentRecordType.TRAINING_RUN,
        run_id=RUN_ID,
        model_catalog_version="1.0.0",
        required_feature_schema_version="1.0.0",
        task=MLTask.BINARY_MALICIOUS,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        seed=42,
        ml_config_fingerprint=DIGEST,
        model_catalog_fingerprint=DIGEST,
    )


# ---------------------------------------------------------------------------
# Shared model configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_SCHEMAS)
def test_every_schema_is_frozen_and_forbids_extra_fields(
    model: type[BaseModel],
) -> None:
    """A contract that accepts unknown keys is not a contract."""
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"


@pytest.mark.parametrize("model", ALL_SCHEMAS)
def test_no_schema_declares_a_prohibited_metadata_field(
    model: type[BaseModel],
) -> None:
    """Aggregate metadata carries no ground truth, split, or identity field."""
    assert prohibited_metadata_fields(list(model.model_fields)) == ()


def test_the_import_time_sweep_covers_every_declared_schema() -> None:
    """A schema added later cannot opt out of the check by being left off.

    The module sweeps a hand-written tuple at import. This asserts that tuple
    is exhaustive, so the sweep cannot silently stop covering the module.
    """
    declared = {
        obj
        for _, obj in inspect.getmembers(schemas_module, inspect.isclass)
        if issubclass(obj, BaseModel)
        and obj is not BaseModel
        and obj.__module__ == schemas_module.__name__
    }
    assert declared == set(schemas_module._DECLARED_SCHEMAS)
    assert declared == set(ALL_SCHEMAS)


def test_the_prohibited_field_helper_rejects_a_non_iterable() -> None:
    """Misuse fails loudly rather than silently reporting no offenders."""
    with pytest.raises(TypeError):
        prohibited_metadata_fields("split")


def test_the_prohibited_field_helper_finds_offenders() -> None:
    """The helper is the same rule the schemas enforce on themselves."""
    assert prohibited_metadata_fields(["gate_name", "campaign_id", "split"]) == (
        "campaign_id",
        "split",
    )


def test_the_prohibited_set_covers_the_project_wide_ground_truth_columns() -> None:
    """Whatever Phase 2 forbids in a feature table is forbidden here too."""
    from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS

    assert PROHIBITED_GT_COLUMNS <= PROHIBITED_METADATA_FIELDS


# ---------------------------------------------------------------------------
# version_tuple
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.9.0", (1, 9, 0)),
        ("1.10", (1, 10)),
        ("2", (2,)),
        ("1.9.0rc1", (1, 9, 0)),
        ("1.9.0.post1", (1, 9, 0)),
        ("3.12.13", (3, 12, 13)),
    ],
)
def test_version_tuple_parses_leading_numeric_components(
    value: str, expected: tuple[int, ...]
) -> None:
    """Non-numeric suffixes stop the parse, the conservative reading."""
    assert version_tuple(value) == expected


def test_version_ordering_places_ten_above_nine() -> None:
    """String comparison would get this backwards; tuple comparison does not."""
    assert version_tuple("1.10") > version_tuple("1.9.0")


def test_version_tuple_rejects_a_value_with_no_numeric_component() -> None:
    """An unparseable version must fail rather than compare as empty."""
    with pytest.raises(ValueError, match="numeric release component"):
        version_tuple("unknown")


# ---------------------------------------------------------------------------
# HyperparameterSpec
# ---------------------------------------------------------------------------


def test_a_declared_default_outside_its_own_bounds_fails_at_declaration() -> None:
    """A bad default must fail at import, not at the first run that omits it."""
    with pytest.raises(ValidationError, match="at most"):
        HyperparameterSpec(
            name="n_estimators",
            kind=HyperparameterKind.INT,
            default=500,
            minimum=1,
            maximum=100,
            description="Trees.",
        )


def test_a_float_hyperparameter_rejects_a_non_finite_value() -> None:
    """NaN would be stable in a fingerprint and meaningless everywhere else."""
    spec = HyperparameterSpec(
        name="learning_rate",
        kind=HyperparameterKind.FLOAT,
        default=0.1,
        minimum=0.0,
        maximum=1.0,
        description="Shrinkage.",
    )
    with pytest.raises(ValueError, match="finite"):
        spec.validate_value(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        spec.validate_value(float("inf"))


def test_a_bool_is_not_accepted_as_an_integer_hyperparameter() -> None:
    """``bool`` subclasses ``int`` in Python; the contract says otherwise."""
    spec = HyperparameterSpec(
        name="max_iter",
        kind=HyperparameterKind.INT,
        default=100,
        minimum=1,
        description="Iterations.",
    )
    with pytest.raises(ValueError, match="must be an integer"):
        spec.validate_value(True)


def test_a_string_hyperparameter_enforces_its_choice_set() -> None:
    """An undeclared choice is rejected rather than passed to the estimator."""
    spec = HyperparameterSpec(
        name="max_features",
        kind=HyperparameterKind.STRING,
        default="sqrt",
        allowed_values=("sqrt", "log2"),
        description="Features per split.",
    )
    assert spec.validate_value("log2") == "log2"
    with pytest.raises(ValueError, match="must be one of"):
        spec.validate_value("all")


def test_allowed_values_require_a_string_hyperparameter() -> None:
    """A choice set on a numeric hyperparameter is a declaration error."""
    with pytest.raises(ValidationError, match="not a string"):
        HyperparameterSpec(
            name="max_iter",
            kind=HyperparameterKind.INT,
            default=1,
            allowed_values=("a", "b"),
            description="Iterations.",
        )


def test_a_hyperparameter_name_must_be_lower_snake_case() -> None:
    """Names are YAML keys; casing is part of the configuration contract."""
    with pytest.raises(ValidationError, match="must match"):
        HyperparameterSpec(
            name="MaxIter",
            kind=HyperparameterKind.INT,
            default=1,
            description="Iterations.",
        )


def test_minimum_above_maximum_is_rejected() -> None:
    """An unsatisfiable range would accept nothing at all."""
    with pytest.raises(ValidationError, match="minimum above maximum"):
        HyperparameterSpec(
            name="depth",
            kind=HyperparameterKind.INT,
            default=5,
            minimum=10,
            maximum=1,
            description="Depth.",
        )


# ---------------------------------------------------------------------------
# DependencyRequirement
# ---------------------------------------------------------------------------


def test_a_dependency_range_is_bounded_on_both_sides() -> None:
    """An inverted or empty range is rejected at declaration."""
    with pytest.raises(ValidationError, match="must be below"):
        DependencyRequirement(
            distribution="scikit-learn",
            minimum_version="1.10",
            below_version="1.9.0",
            reason="inverted",
        )


def test_a_dependency_range_reports_membership_correctly() -> None:
    """The bound is half-open: inclusive below, exclusive above."""
    requirement = DependencyRequirement(
        distribution="scikit-learn",
        minimum_version="1.9.0",
        below_version="1.10",
        reason="Serializer contract stability.",
    )
    assert requirement.contains("1.9.0")
    assert requirement.contains("1.9.4")
    assert not requirement.contains("1.10.0")
    assert not requirement.contains("1.8.2")
    assert requirement.specifier == "scikit-learn>=1.9.0,<1.10"


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


def test_an_inconclusive_gate_records_no_observation() -> None:
    """Inconclusive means "could not measure", not "measured and unsure"."""
    gate = GateResult(
        gate_name="beats_baseline",
        status=GateStatus.INCONCLUSIVE,
        message="Validation split carried no positive rows.",
    )
    assert gate.observed is None
    assert not gate.passed


def test_a_decided_gate_must_record_what_it_observed() -> None:
    """A pass with no number behind it is not auditable."""
    with pytest.raises(ValidationError, match="without an observation"):
        GateResult(
            gate_name="beats_baseline",
            status=GateStatus.PASS,
            message="Cleared.",
        )


def test_an_inconclusive_gate_may_not_record_an_observation() -> None:
    """If a number was observed, the gate is decidable and must decide."""
    with pytest.raises(ValidationError, match="records an observation"):
        GateResult(
            gate_name="beats_baseline",
            status=GateStatus.INCONCLUSIVE,
            observed=0.5,
            message="Unsure.",
        )


def test_only_a_pass_counts_as_passed() -> None:
    """Inconclusive must never read as a pass."""
    for status in (GateStatus.FAIL, GateStatus.INCONCLUSIVE):
        gate = GateResult(
            gate_name="parity",
            status=status,
            observed=None if status is GateStatus.INCONCLUSIVE else 0.0,
            message="Checked.",
        )
        assert not gate.passed


def test_a_gate_rejects_a_non_finite_observation() -> None:
    """A gate comparing against NaN would silently never fire."""
    with pytest.raises(ValidationError, match="finite"):
        GateResult(
            gate_name="parity",
            status=GateStatus.PASS,
            observed=float("inf"),
            threshold=1.0,
            message="Checked.",
        )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            "Anchor 3f2504e0-4f89-41d3-9a0c-0305e82c3301 failed.",
            "identifier",
        ),
        ("Entity u:0123456789abcdef0123456789abcdef drifted.", "pseudonym"),
        ("Observed near 51.5074000, -0.1278000 today.", "coordinates"),
    ],
)
def test_a_gate_message_may_not_carry_identity(text: str, match: str) -> None:
    """A message is prose; a message containing identity is a disclosure."""
    with pytest.raises(ValidationError, match=match):
        GateResult(
            gate_name="parity",
            status=GateStatus.FAIL,
            observed=0.0,
            message=text,
        )


def test_a_gate_message_may_not_be_empty() -> None:
    """A failing gate that says nothing is not actionable."""
    with pytest.raises(ValidationError, match="must not be empty"):
        GateResult(
            gate_name="parity",
            status=GateStatus.FAIL,
            observed=0.0,
            message="   ",
        )


# ---------------------------------------------------------------------------
# ScoreSemantics
# ---------------------------------------------------------------------------


def test_a_calibrated_probability_requires_a_fitted_calibrator() -> None:
    """The word is earned by calibration, not by declaration."""
    with pytest.raises(ValidationError, match="requires a fitted calibration"):
        ScoreSemantics(
            score_kind=ScoreKind.CALIBRATED_PROBABILITY,
            calibration_method=CalibrationMethod.NONE,
            lower_bound=0.0,
            upper_bound=1.0,
            description="Calibrated estimate.",
        )


def test_an_uncalibrated_score_may_not_declare_a_calibration_method() -> None:
    """A method without a calibrated kind would imply a claim it cannot make."""
    with pytest.raises(ValidationError, match="cannot declare calibration method"):
        ScoreSemantics(
            score_kind=ScoreKind.DECISION_SCORE,
            calibration_method=CalibrationMethod.ISOTONIC,
            lower_bound=0.0,
            upper_bound=1.0,
            description="Ordered magnitude.",
        )


@pytest.mark.parametrize("term", ["probability", "likelihood", "confidence"])
def test_an_uncalibrated_description_may_not_use_the_word(term: str) -> None:
    """Phase 4 forbids the word in evidence; this layer earns it or drops it."""
    with pytest.raises(ValidationError, match=term):
        ScoreSemantics(
            score_kind=ScoreKind.ANOMALY_SCORE,
            calibration_method=CalibrationMethod.NONE,
            lower_bound=-1.0,
            upper_bound=1.0,
            description=f"An outlier {term} for the row.",
        )


def test_a_calibrated_description_may_use_the_word() -> None:
    """Where the claim is true, the vocabulary is permitted."""
    semantics = ScoreSemantics(
        score_kind=ScoreKind.CALIBRATED_PROBABILITY,
        calibration_method=CalibrationMethod.ISOTONIC,
        lower_bound=0.0,
        upper_bound=1.0,
        description="Calibrated probability that the anchor event is malicious.",
    )
    assert "probability" in semantics.description


def test_a_calibrated_probability_is_bounded_by_the_unit_interval() -> None:
    """A probability outside [0, 1] is not one."""
    with pytest.raises(ValidationError, match=r"bounded by \[0, 1\]"):
        ScoreSemantics(
            score_kind=ScoreKind.CALIBRATED_PROBABILITY,
            calibration_method=CalibrationMethod.PLATT,
            lower_bound=0.0,
            upper_bound=100.0,
            description="Calibrated estimate.",
        )


def test_score_bounds_must_be_ordered() -> None:
    """An inverted or degenerate range describes nothing."""
    with pytest.raises(ValidationError, match="lower_bound must be below"):
        ScoreSemantics(
            score_kind=ScoreKind.DECISION_SCORE,
            calibration_method=CalibrationMethod.NONE,
            lower_bound=1.0,
            upper_bound=1.0,
            description="Ordered magnitude.",
        )


def test_phase_four_risk_and_ml_probability_are_not_the_same_type() -> None:
    """The two quantities live on different scales and must stay apart.

    A Phase 4 risk score is an ordinal 0-100 magnitude; an ML probability is a
    calibrated unit-interval quantity. Nothing in this module declares a field
    that could hold either one, which is what keeps them from being averaged.
    """
    ordinal_names = {"risk_score", "signal_strength", "severity"}
    for model in ALL_SCHEMAS:
        assert not ordinal_names & set(model.model_fields), model.__name__


# ---------------------------------------------------------------------------
# ArtifactDeclaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../escape.json", "a/../../b.json", "back\\slash.json"]
)
def test_an_artifact_path_must_be_a_safe_relative_path(path: str) -> None:
    """A manifest must never disclose or reach outside its own directory."""
    with pytest.raises(ValidationError):
        ArtifactDeclaration(
            role="model_parameters",
            relative_path=path,
            media_type="application/json",
            required=True,
            description="Model parameters.",
        )


def test_an_artifact_media_type_must_be_one_this_layer_writes() -> None:
    """Declaring a format nothing produces would be an unenforceable promise."""
    with pytest.raises(ValidationError, match="media_type"):
        ArtifactDeclaration(
            role="model_parameters",
            relative_path="model.pkl",
            media_type="application/octet-stream",
            required=True,
            description="Pickle.",
        )


def test_a_valid_artifact_declaration_round_trips() -> None:
    """The happy path holds for each format the layer actually writes."""
    for media_type, name in (
        ("application/json", "model.json"),
        ("application/x-npz", "arrays.npz"),
        ("application/vnd.apache.parquet", "ml_predictions.parquet"),
    ):
        declaration = ArtifactDeclaration(
            role="model_parameters",
            relative_path=name,
            media_type=media_type,
            required=True,
            description="Canonical model parameters.",
        )
        assert declaration.relative_path == name


# ---------------------------------------------------------------------------
# ExperimentRecordIdentity
# ---------------------------------------------------------------------------


def test_a_ledger_identity_carries_no_metric_field() -> None:
    """Identity is semantic inputs only; results live on their own records.

    In particular there is no ``test_metrics`` field. A test evaluation is a
    separate immutable record written after the champion is frozen, never an
    amendment to a training run.
    """
    fields = set(ExperimentRecordIdentity.model_fields)
    for forbidden in (
        "test_metrics",
        "validation_metrics",
        "train_metrics",
        "metrics",
        "score",
    ):
        assert forbidden not in fields


def test_a_ledger_identity_carries_no_timestamp_or_path() -> None:
    """Identity excludes anything observational or machine-specific.

    Re-running identical semantics must produce an identical run identifier,
    which a creation timestamp or an output directory would break.
    """
    fields = set(ExperimentRecordIdentity.model_fields)
    for forbidden in (
        "created_at",
        "observed_at",
        "timestamp",
        "output_dir",
        "models_dir",
        "hostname",
        "machine",
    ):
        assert forbidden not in fields


def test_a_ledger_identity_validates_its_fingerprints() -> None:
    """A fingerprint field must hold a lower-case SHA-256 digest."""
    with pytest.raises(ValidationError):
        _identity().model_copy(update={"ml_config_fingerprint": "not-a-digest"})
        ExperimentRecordIdentity(
            record_type=ExperimentRecordType.TRAINING_RUN,
            run_id=RUN_ID,
            model_catalog_version="1.0.0",
            required_feature_schema_version="1.0.0",
            task=MLTask.BINARY_MALICIOUS,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            seed=42,
            ml_config_fingerprint="NOTADIGEST",
            model_catalog_fingerprint=DIGEST,
        )


def test_a_ledger_identity_rejects_an_upper_case_fingerprint() -> None:
    """Casing must be canonical or two identical digests would not compare."""
    with pytest.raises(ValidationError):
        ExperimentRecordIdentity(
            record_type=ExperimentRecordType.TRAINING_RUN,
            run_id=RUN_ID,
            model_catalog_version="1.0.0",
            required_feature_schema_version="1.0.0",
            task=MLTask.BINARY_MALICIOUS,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            seed=42,
            ml_config_fingerprint=DIGEST.upper(),
            model_catalog_fingerprint=DIGEST,
        )


def test_a_ledger_identity_rejects_an_upper_case_run_id() -> None:
    """Run identifiers are canonical lower-case UUIDs."""
    with pytest.raises(ValidationError, match="lower-case UUID"):
        ExperimentRecordIdentity(
            record_type=ExperimentRecordType.CHAMPION_FREEZE,
            run_id=RUN_ID.upper(),
            model_catalog_version="1.0.0",
            required_feature_schema_version="1.0.0",
            seed=1,
            ml_config_fingerprint=DIGEST,
            model_catalog_fingerprint=DIGEST,
        )


def test_a_training_run_record_must_name_its_task() -> None:
    """A fit with no declared task cannot be reproduced from its record."""
    with pytest.raises(ValidationError, match="must name a task"):
        ExperimentRecordIdentity(
            record_type=ExperimentRecordType.TRAINING_RUN,
            run_id=RUN_ID,
            model_catalog_version="1.0.0",
            required_feature_schema_version="1.0.0",
            seed=42,
            ml_config_fingerprint=DIGEST,
            model_catalog_fingerprint=DIGEST,
        )


def test_a_record_naming_a_task_must_name_the_family_that_performed_it() -> None:
    """A task without a family is an incomplete provenance record."""
    with pytest.raises(ValidationError, match="must also name a model family"):
        ExperimentRecordIdentity(
            record_type=ExperimentRecordType.VALIDATION_SELECTION,
            run_id=RUN_ID,
            model_catalog_version="1.0.0",
            required_feature_schema_version="1.0.0",
            task=MLTask.BINARY_MALICIOUS,
            seed=42,
            ml_config_fingerprint=DIGEST,
            model_catalog_fingerprint=DIGEST,
        )


def test_a_ledger_identity_is_frozen() -> None:
    """An immutable record cannot be amended after it is written."""
    identity = _identity()
    with pytest.raises(ValidationError):
        identity.ml_config_fingerprint = DIGEST.replace("a", "b")


def test_the_four_record_types_are_all_constructible() -> None:
    """Every planned ledger record shape can actually be expressed."""
    shapes: dict[ExperimentRecordType, dict[str, Any]] = {
        ExperimentRecordType.TRAINING_RUN: {
            "task": MLTask.BINARY_MALICIOUS,
            "model_family": ModelFamily.LOGISTIC_REGRESSION,
        },
        ExperimentRecordType.VALIDATION_SELECTION: {
            "task": MLTask.BINARY_MALICIOUS,
            "model_family": ModelFamily.LOGISTIC_REGRESSION,
        },
        ExperimentRecordType.CHAMPION_FREEZE: {},
        ExperimentRecordType.TEST_EVALUATION: {},
    }
    for record_type, extra in shapes.items():
        record = ExperimentRecordIdentity(
            record_type=record_type,
            run_id=RUN_ID,
            model_catalog_version="1.0.0",
            required_feature_schema_version="1.0.0",
            seed=42,
            ml_config_fingerprint=DIGEST,
            model_catalog_fingerprint=DIGEST,
            **extra,
        )
        assert record.record_type is record_type
        assert record.ml_schema_version == ML_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# SupportRequirement
# ---------------------------------------------------------------------------


def test_support_floors_must_be_positive() -> None:
    """A floor of zero would let an empty class read as measured."""
    with pytest.raises(ValidationError):
        SupportRequirement(min_train_positive_rows=0)


def test_support_fingerprint_data_covers_every_field() -> None:
    """A field left out of the digest would silently weaken it."""
    requirement = SupportRequirement()
    assert set(requirement.fingerprint_data()) == set(SupportRequirement.model_fields)
