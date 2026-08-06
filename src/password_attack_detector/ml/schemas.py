"""Versioned, frozen contracts for the machine-learning detection layer.

Every model here is ``extra="forbid", frozen=True``.  Field order is the
declaration order, which is the order the fingerprint helpers and the Markdown
renderer read, so two runs of the same code emit the same bytes.

Three rules are enforced structurally rather than by convention:

* **Non-finite numbers are rejected.**  ``NaN`` and infinity are refused at
  validation time rather than propagating into a fingerprint, a report, or a
  Parquet column.  ``NaN != NaN`` would break every exact-equality comparison
  the layer's determinism rests on.
* **Probability terminology is gated on calibration.**  A
  :class:`ScoreSemantics` may only describe itself as a probability when its
  score kind is :attr:`~password_attack_detector.ml.enums.ScoreKind.CALIBRATED_PROBABILITY`
  and a calibrator was actually fitted.  Phase 4 spent real effort keeping
  ``risk_score`` from being read as a likelihood; this layer must not undo it.
* **Aggregate metadata carries no identity.**  No schema in this module may
  declare a field for a label, split assignment, campaign identifier, event
  identifier, entity scope value, credential, or raw coordinate, and no free
  text field may contain something shaped like one.  Both halves are checked:
  the field names at import time by :func:`prohibited_metadata_fields`, and the
  values at validation time by the text validators.

Phase 4's ordinal ``risk_score`` and this layer's calibrated probability are
**separately typed and never combined arithmetically**.  Nothing in this module
declares a field that could hold either one.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from password_attack_detector.ml.enums import (
    CalibrationMethod,
    ExperimentRecordType,
    GateStatus,
    HyperparameterKind,
    MLTask,
    ModelFamily,
    ScoreKind,
    is_probability,
)

__all__ = [
    "ML_SCHEMA_VERSION",
    "PROHIBITED_METADATA_FIELDS",
    "ArtifactDeclaration",
    "DependencyRequirement",
    "ExperimentRecordIdentity",
    "GateResult",
    "HyperparameterSpec",
    "ScoreSemantics",
    "Sha256Hex",
    "SupportRequirement",
    "UnitInterval",
    "prohibited_metadata_fields",
]

#: Version of the ML artifact and contract schema.  Declared ``Final`` without
#: an explicit annotation so its type narrows to ``Literal["1.0.0"]``, letting
#: the ML configuration default its ``ml_schema_version`` field from it.
ML_SCHEMA_VERSION: Final = "1.0.0"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_VERSION_BOUND_RE = re.compile(r"^\d+(\.\d+)*$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DISTRIBUTION_RE = re.compile(r"^[a-z][a-z0-9._-]*$")
_RELATIVE_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

#: A value shaped like a UUID.  Anchor identifiers, detection identifiers, and
#: alert identifiers are all UUIDs, so none of them may appear in metadata.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

#: A value shaped like a Phase 3 entity pseudonym (``u:``/``s:``/``d:``/
#: ``sess:`` followed by 32 hex characters).
_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")

#: A value shaped like a decimal latitude/longitude pair.
_COORDINATE_RE = re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}")

#: Field names no aggregate-metadata schema in this layer may declare.
#:
#: Ground truth, split assignment, campaign metadata, per-event identity, and
#: entity scope are all things the ML layer either never sees or must not
#: republish.  Credentials are listed because a model contract has no use for
#: one, so accepting a field named for one could only ever be a mistake.
PROHIBITED_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        # ground truth
        "label",
        "labels",
        "target",
        "malicious",
        "attack_class",
        "attack_label",
        "is_attack",
        "scenario",
        "scenario_variant",
        "y_true",
        "y_pred",
        "supervised_training_eligible",
        # Phase 3 and Phase 4 output columns.  ``risk_score`` is an ordinal
        # 0-100 magnitude and must never share a schema with a calibrated
        # probability; ``attack_probability`` and ``model_probability`` are
        # names the feature contract already forbids, and repeating that
        # prohibition here keeps one vocabulary across the project.
        "risk_score",
        "attack_probability",
        "model_probability",
        # split assignment
        "split",
        "exclusion_reason",
        # campaign metadata
        "campaign",
        "campaign_id",
        "campaign_stage",
        # per-event identity
        "event_id",
        "anchor_event_id",
        "detection_id",
        "alert_id",
        # entity scope and pseudonyms
        "user_id",
        "source_id",
        "device_id",
        "session_id",
        "scope_value",
        "user_scope",
        "source_scope",
        # coordinates
        "latitude",
        "longitude",
        "coordinates",
        # credentials
        "password",
        "secret",
        "token",
        "credential",
        "api_key",
        "private_key",
    }
)

#: A float constrained to the closed unit interval.
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]

#: A lower-case SHA-256 hex digest.
Sha256Hex = Annotated[
    str, StringConstraints(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)
]


def prohibited_metadata_fields(field_names: object) -> tuple[str, ...]:
    """Return the declared *field_names* that aggregate metadata may not carry.

    Exposed as a function rather than buried in a validator so tests can sweep
    every schema in the layer with the same rule the schemas enforce on
    themselves.
    """
    if not isinstance(field_names, list | tuple | set | frozenset):
        raise TypeError("field_names must be an iterable of strings")
    return tuple(
        sorted(name for name in field_names if str(name) in PROHIBITED_METADATA_FIELDS)
    )


def _reject_non_finite(value: float, field_name: str) -> float:
    """Return *value* when it is finite, else raise.

    ``NaN`` and infinity are refused here rather than allowed to reach a
    fingerprint or a report.  A digest over ``NaN`` is stable but meaningless,
    and ``NaN != NaN`` silently breaks the exact-equality comparisons this
    layer's determinism guarantees rest on.
    """
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _reject_identity_shaped(value: str, field_name: str) -> str:
    """Return *value* when it carries nothing shaped like identity.

    Checks the *value*, complementing :func:`prohibited_metadata_fields`, which
    checks the *name*.  A field called ``message`` is legitimate; a message
    containing an anchor identifier is not.
    """
    if _UUID_RE.search(value):
        raise ValueError(f"{field_name} must not contain an identifier")
    if _PSEUDONYM_RE.search(value):
        raise ValueError(f"{field_name} must not contain an entity pseudonym")
    if _COORDINATE_RE.search(value):
        raise ValueError(f"{field_name} must not contain coordinates")
    return value


class HyperparameterSpec(BaseModel):
    """One declared, typed, bounded hyperparameter of a model family.

    Bounds live in the declaration rather than in a validator so configuration
    validation, the generated documentation, and the catalog fingerprint all
    read one source.  A configuration may only supply a value for a
    hyperparameter declared here; there is no passthrough to the estimator.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: HyperparameterKind
    default: bool | int | float | str
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str, ...] | None = None
    unit: str | None = None
    tunable: bool = True
    description: str

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        """Hyperparameter names are lower snake case, matching the YAML keys."""
        if not _NAME_RE.match(value):
            raise ValueError(
                f"hyperparameter name {value!r} must match {_NAME_RE.pattern!r}"
            )
        return value

    @field_validator("description")
    @classmethod
    def check_description(cls, value: str) -> str:
        """Descriptions are prose and must carry no identity."""
        return _reject_identity_shaped(value, "description")

    @model_validator(mode="after")
    def check_declaration(self) -> Self:
        """Validate the bounds, the choice set, and the declared default."""
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"hyperparameter {self.name!r} has minimum above maximum")
        if self.allowed_values is not None:
            if self.kind is not HyperparameterKind.STRING:
                raise ValueError(
                    f"hyperparameter {self.name!r} declares allowed_values but is "
                    f"not a string hyperparameter"
                )
            if not self.allowed_values:
                raise ValueError(
                    f"hyperparameter {self.name!r} has an empty choice set"
                )
            if len(set(self.allowed_values)) != len(self.allowed_values):
                raise ValueError(
                    f"hyperparameter {self.name!r} repeats an allowed value"
                )
        # The declared default must survive this hyperparameter's own
        # validation, so a bad default fails at import rather than at the first
        # run that happens to omit an override.
        self.validate_value(self.default)
        return self

    def validate_value(self, value: object) -> bool | int | float | str:
        """Return *value* coerced to this hyperparameter's declared type.

        Raises:
            ValueError: if the value has the wrong type, falls outside the
                declared bounds, is not finite, or is not a declared choice.
        """
        match self.kind:
            case HyperparameterKind.BOOL:
                if not isinstance(value, bool):
                    raise ValueError(f"hyperparameter {self.name!r} must be a boolean")
                return value
            case HyperparameterKind.INT:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"hyperparameter {self.name!r} must be an integer")
                return self._check_bounds(float(value), value)
            case HyperparameterKind.FLOAT:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError(f"hyperparameter {self.name!r} must be a number")
                coerced = _reject_non_finite(
                    float(value), f"hyperparameter {self.name!r}"
                )
                return self._check_bounds(coerced, coerced)
            case HyperparameterKind.STRING:
                if not isinstance(value, str):
                    raise ValueError(f"hyperparameter {self.name!r} must be a string")
                if self.allowed_values is not None and value not in self.allowed_values:
                    raise ValueError(
                        f"hyperparameter {self.name!r} must be one of "
                        f"{list(self.allowed_values)}"
                    )
                return value

    def _check_bounds(
        self, numeric: float, value: int | float
    ) -> bool | int | float | str:
        """Return *value* when *numeric* lies inside the declared bounds."""
        if self.minimum is not None and numeric < self.minimum:
            raise ValueError(
                f"hyperparameter {self.name!r} must be at least {self.minimum}"
            )
        if self.maximum is not None and numeric > self.maximum:
            raise ValueError(
                f"hyperparameter {self.name!r} must be at most {self.maximum}"
            )
        return value


class DependencyRequirement(BaseModel):
    """A bounded version range a model artifact was produced under.

    Both bounds are mandatory.  An open upper bound would let an unreviewed
    minor-series upgrade change a serializer contract without anything failing,
    which is exactly the failure this layer is built to prevent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution: str
    minimum_version: str
    below_version: str
    reason: str

    @field_validator("distribution")
    @classmethod
    def check_distribution(cls, value: str) -> str:
        """Distribution names are lower case, as published on the index."""
        if not _DISTRIBUTION_RE.match(value):
            raise ValueError(
                f"distribution {value!r} must match {_DISTRIBUTION_RE.pattern!r}"
            )
        return value

    @field_validator("minimum_version", "below_version")
    @classmethod
    def check_version(cls, value: str) -> str:
        """Bounds are dotted numeric releases such as ``1.9.0`` or ``1.10``."""
        if not _VERSION_BOUND_RE.match(value):
            raise ValueError(f"version bound {value!r} must be dotted numeric")
        return value

    @model_validator(mode="after")
    def check_range(self) -> Self:
        """The lower bound must lie strictly below the exclusive upper bound."""
        if version_tuple(self.minimum_version) >= version_tuple(self.below_version):
            raise ValueError(
                f"{self.distribution}: minimum_version must be below below_version"
            )
        return self

    @property
    def specifier(self) -> str:
        """Return the range as a PEP 508 specifier, for display and manifests."""
        return f"{self.distribution}>={self.minimum_version},<{self.below_version}"

    def contains(self, version: str) -> bool:
        """Return whether *version* lies inside this bounded range."""
        candidate = version_tuple(version)
        return (
            version_tuple(self.minimum_version)
            <= candidate
            < version_tuple(self.below_version)
        )


def version_tuple(version: str) -> tuple[int, ...]:
    """Return the leading numeric release components of *version*.

    A deliberately small parser.  ``packaging`` is a development dependency
    only, and adding it to the runtime set to compare two dotted numbers would
    be a poor trade.  Non-numeric suffixes (``1.9.0rc1``, ``1.9.0.post1``) stop
    the parse, which is the conservative reading for a compatibility bound.
    """
    parts: list[int] = []
    for component in version.split("."):
        digits = ""
        for character in component:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        raise ValueError(f"version {version!r} has no numeric release component")
    return tuple(parts)


class SupportRequirement(BaseModel):
    """Minimum class support before a fitted result is allowed to mean anything.

    A metric computed over three positive rows is not a measurement.  When
    support falls below these floors the layer reports
    :attr:`~password_attack_detector.ml.enums.ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT`
    rather than a number that reads like a result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_train_positive_rows: int = Field(default=50, ge=1)
    min_validation_positive_rows: int = Field(default=25, ge=1)
    min_validation_benign_rows: int = Field(default=100, ge=1)
    min_rows_per_category: int = Field(default=20, ge=1)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to a configuration digest."""
        return {
            "min_train_positive_rows": self.min_train_positive_rows,
            "min_validation_positive_rows": self.min_validation_positive_rows,
            "min_validation_benign_rows": self.min_validation_benign_rows,
            "min_rows_per_category": self.min_rows_per_category,
        }


class GateResult(BaseModel):
    """The outcome of one champion-selection gate.

    ``INCONCLUSIVE`` and "no observation" are the same state, and the validator
    enforces it in both directions.  A gate that could not measure anything
    must not read as a pass, and a gate that did measure something must commit
    to a verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_name: str
    status: GateStatus
    observed: float | None = None
    threshold: float | None = None
    message: str

    @field_validator("gate_name")
    @classmethod
    def check_gate_name(cls, value: str) -> str:
        """Gate names are lower snake case and stable across runs."""
        if not _NAME_RE.match(value):
            raise ValueError(f"gate_name {value!r} must match {_NAME_RE.pattern!r}")
        return value

    @field_validator("observed", "threshold")
    @classmethod
    def check_finite(cls, value: float | None) -> float | None:
        """Gate numbers are finite or absent; never ``NaN``, never infinity."""
        if value is None:
            return None
        return _reject_non_finite(value, "gate value")

    @field_validator("message")
    @classmethod
    def check_message(cls, value: str) -> str:
        """Gate messages state counts and thresholds, never identity."""
        if not value.strip():
            raise ValueError("message must not be empty")
        return _reject_identity_shaped(value, "message")

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """An inconclusive gate has no observation, and vice versa."""
        inconclusive = self.status is GateStatus.INCONCLUSIVE
        if inconclusive and self.observed is not None:
            raise ValueError(
                f"gate {self.gate_name!r} is inconclusive but records an observation"
            )
        if not inconclusive and self.observed is None:
            raise ValueError(
                f"gate {self.gate_name!r} reports {self.status} without an observation"
            )
        return self

    @property
    def passed(self) -> bool:
        """Return whether this gate passed.  Inconclusive is never a pass."""
        return self.status is GateStatus.PASS


class ScoreSemantics(BaseModel):
    """What a model's numeric output means, and what it may be called.

    The single contract that keeps "probability" honest.  A calibrated
    probability requires a fitted calibrator and unit-interval bounds; every
    other score kind is an ordered magnitude whose description may not use the
    word at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    score_kind: ScoreKind
    calibration_method: CalibrationMethod
    lower_bound: float
    upper_bound: float
    description: str

    @field_validator("lower_bound", "upper_bound")
    @classmethod
    def check_finite(cls, value: float) -> float:
        """Score bounds are finite."""
        return _reject_non_finite(value, "score bound")

    @field_validator("description")
    @classmethod
    def check_description_identity(cls, value: str) -> str:
        """Score descriptions are prose and must carry no identity."""
        return _reject_identity_shaped(value, "description")

    @model_validator(mode="after")
    def check_semantics(self) -> Self:
        """Tie the score kind, the calibration method, the bounds, and the prose."""
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be below upper_bound")

        calibrated = is_probability(self.score_kind)
        if calibrated and self.calibration_method is CalibrationMethod.NONE:
            raise ValueError(
                "a calibrated probability requires a fitted calibration method"
            )
        if not calibrated and self.calibration_method is not CalibrationMethod.NONE:
            raise ValueError(
                f"score kind {self.score_kind} cannot declare calibration method "
                f"{self.calibration_method}; only a calibrated probability may"
            )
        if calibrated and (self.lower_bound < 0.0 or self.upper_bound > 1.0):
            raise ValueError("a calibrated probability is bounded by [0, 1]")

        # Phase 4 forbids the word in evidence prose; this layer permits it in
        # exactly one place, and only where it is true.
        if not calibrated:
            lowered = self.description.lower()
            for term in ("probability", "likelihood", "confidence"):
                if term in lowered:
                    raise ValueError(
                        f"score kind {self.score_kind} is not calibrated; its "
                        f"description must not use the word {term!r}"
                    )
        return self


class ArtifactDeclaration(BaseModel):
    """One file a published model directory is declared to contain.

    Declarations are metadata: a role, a relative path, and a media type.  No
    declaration carries an absolute path, so a manifest can never disclose
    where on a machine it was written.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    relative_path: str
    media_type: str
    required: bool
    description: str

    @field_validator("role")
    @classmethod
    def check_role(cls, value: str) -> str:
        """Artifact roles are lower snake case and stable across versions."""
        if not _NAME_RE.match(value):
            raise ValueError(f"role {value!r} must match {_NAME_RE.pattern!r}")
        return value

    @field_validator("relative_path")
    @classmethod
    def check_relative_path(cls, value: str) -> str:
        """Reject absolute paths, traversal, and platform-specific separators."""
        if not _RELATIVE_PATH_RE.match(value):
            raise ValueError(f"relative_path {value!r} is not a safe relative path")
        if ".." in value.split("/"):
            raise ValueError("relative_path must not contain '..' components")
        return value

    @field_validator("media_type")
    @classmethod
    def check_media_type(cls, value: str) -> str:
        """Only the formats this layer actually writes may be declared."""
        permitted = {
            "application/json",
            "application/x-npz",
            "application/vnd.apache.parquet",
        }
        if value not in permitted:
            raise ValueError(f"media_type {value!r} must be one of {sorted(permitted)}")
        return value

    @field_validator("description")
    @classmethod
    def check_description(cls, value: str) -> str:
        """Artifact descriptions are prose and must carry no identity."""
        return _reject_identity_shaped(value, "description")


class ExperimentRecordIdentity(BaseModel):
    """The immutable identity of one experiment-ledger record.

    Identity is derived from *semantics only*: what was fitted, on which data,
    under which configuration, with which seed.  A creation timestamp, an
    output directory, and a machine name are all deliberately absent -- they
    are observational metadata carried on the ledger envelope that Milestone 6
    adds, never on identity.  Re-running identical semantics therefore produces
    an identical ``run_id``, which is what makes the ledger idempotent.

    There is no metrics field of any kind on this model, and in particular no
    ``test_metrics``.  A test evaluation is its own
    :attr:`~password_attack_detector.ml.enums.ExperimentRecordType.TEST_EVALUATION`
    record written after the champion is frozen; it is never an amendment to a
    training run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ml_schema_version: str = ML_SCHEMA_VERSION
    record_type: ExperimentRecordType
    run_id: str
    model_catalog_version: str
    required_feature_schema_version: str
    task: MLTask | None = None
    model_family: ModelFamily | None = None
    seed: int = Field(ge=0)
    ml_config_fingerprint: Sha256Hex
    model_catalog_fingerprint: Sha256Hex
    feature_catalog_fingerprint: Sha256Hex | None = None
    split_config_fingerprint: Sha256Hex | None = None
    label_fingerprint: Sha256Hex | None = None
    training_data_fingerprint: Sha256Hex | None = None

    @field_validator(
        "ml_schema_version",
        "model_catalog_version",
        "required_feature_schema_version",
    )
    @classmethod
    def check_semver(cls, value: str) -> str:
        """Contract versions are semantic versions."""
        if not _SEMVER_RE.match(value):
            raise ValueError(f"{value!r} must be a semantic version")
        return value

    @field_validator("run_id")
    @classmethod
    def check_run_id(cls, value: str) -> str:
        """Run identifiers are lower-case UUIDs derived from semantic inputs.

        A UUID here is a *derived* identifier, not an event identifier: it is
        ``uuid5`` over fingerprints and carries no per-row identity.
        """
        if not _UUID_RE.fullmatch(value) or value != value.lower():
            raise ValueError("run_id must be a lower-case UUID")
        return value

    @model_validator(mode="after")
    def check_record_shape(self) -> Self:
        """A record that names a task must name the family that performed it."""
        if self.task is not None and self.model_family is None:
            raise ValueError("a record naming a task must also name a model family")
        if self.record_type is ExperimentRecordType.TRAINING_RUN and self.task is None:
            raise ValueError("a training-run record must name a task")
        return self


#: Every schema declared in this module.  Swept at import against
#: :data:`PROHIBITED_METADATA_FIELDS`, and swept again by a test that asserts
#: this tuple is exhaustive -- so a schema added in a later milestone cannot
#: quietly opt out of the check by being left off the list.
_DECLARED_SCHEMAS: Final[tuple[type[BaseModel], ...]] = (
    ArtifactDeclaration,
    DependencyRequirement,
    ExperimentRecordIdentity,
    GateResult,
    HyperparameterSpec,
    ScoreSemantics,
    SupportRequirement,
)


def _assert_no_prohibited_fields() -> None:
    """Fail at import if any schema declares a prohibited metadata field."""
    for model in _DECLARED_SCHEMAS:
        offending = prohibited_metadata_fields(list(model.model_fields))
        if offending:
            raise ValueError(
                f"{model.__name__} declares prohibited metadata field(s) "
                f"{list(offending)}"
            )


_assert_no_prohibited_fields()
