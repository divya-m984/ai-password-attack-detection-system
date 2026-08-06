"""The versioned model catalog -- the single source of truth for model metadata.

Every model family is described by exactly one :class:`ModelSpec`.  The catalog
drives:

* configuration validation (which families and hyperparameters exist);
* champion eligibility (which families may be promoted at all);
* the generated Markdown documentation;
* the catalog fingerprint recorded in every model manifest and ledger record.

**Model logic is never loaded from configuration.**  A ``ModelSpec`` declares
*data*: identifiers, bounded hyperparameters, the score interface, the
estimator attributes a serializer will read, and the determinism controls a fit
must set.  YAML supplies values for declared hyperparameters and nothing else.
There is no import path, no callable reference, no module name, and no plugin
mechanism anywhere in this contract; ``estimator_class_name`` is descriptive
prose validated to contain no dots precisely so it can never be mistaken for
one.  The Python implementations that consume these specs are registered
statically in Milestone 4.

Three eligibility rules are enforced by the model rather than by convention:

* A spec that declares a **private** estimator attribute cannot be
  ``champion_eligible``.  Reading a private attribute is a bet on an internal
  layout, and that bet must be settled by a compatibility test before a model
  built on it can be promoted.  ``M-021`` is the live example.
* An **anomaly-only** spec is never champion-eligible and never supervised.
* A champion-eligible spec must support the binary task, declare both a
  serializer and an inference adapter, and be neither experimental nor
  anomaly-only.

Catalog membership alone makes nothing a champion.  Promotion additionally
requires proven round-trip parity between the serializer and the inference
adapter, which Milestone 4 establishes and Milestone 7 gates on.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from password_attack_detector.exceptions import MLConfigurationError
from password_attack_detector.ml.enums import (
    SUPERVISED_TASKS,
    CalibrationMethod,
    HyperparameterKind,
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
    ScoreKind,
    is_probability,
)
from password_attack_detector.ml.schemas import HyperparameterSpec

__all__ = [
    "MODEL_CATALOG",
    "MODEL_CATALOG_VERSION",
    "ModelCatalog",
    "ModelSpec",
    "build_model_catalog",
    "model_catalog_to_markdown",
]

#: Declared ``Final`` without an explicit annotation so its type narrows to
#: ``Literal["1.0.0"]``, letting the ML configuration default its
#: ``model_catalog_version`` field from this constant.
MODEL_CATALOG_VERSION: Final = "1.0.0"

_MODEL_ID_RE = re.compile(r"^M-\d{3}$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_ADAPTER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*_v\d+$")
_ATTRIBUTE_RE = re.compile(r"^_?[a-z][a-z0-9_]*_?$")
_CLASS_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")

#: Wording a model description may not use.  A model family describes a
#: *method*; it never asserts that its output proves anything.  ``probability``
#: is included for the same reason Phase 4 excludes it from evidence prose:
#: a raw estimator output is not one, and a description is exactly where that
#: distinction gets lost.
_PROHIBITED_DESCRIPTION_TERMS: frozenset[str] = frozenset(
    {
        "probability",
        "likelihood",
        "confidence",
        "proof",
        "proves",
        "proven",
        "confirms",
        "confirmed",
        "guarantee",
        "guarantees",
        "guaranteed",
        "definitive",
        "conclusive",
    }
)


class ModelSpec(BaseModel):
    """Complete, machine-readable description of one model family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_version: str
    family: ModelFamily
    display_name: str
    description: str
    supported_tasks: tuple[MLTask, ...]
    hyperparameters: tuple[HyperparameterSpec, ...] = ()
    native_score_kind: ScoreKind
    calibration_compatible: bool
    supported_calibration_methods: tuple[CalibrationMethod, ...]
    multiclass_capable: bool
    champion_eligible: bool
    eligibility_status: ModelEligibilityStatus
    serializer_id: str
    inference_adapter_id: str
    public_estimator_attributes: tuple[str, ...] = ()
    private_estimator_attributes: tuple[str, ...] = ()
    determinism_controls: tuple[str, ...]
    requires_sklearn: bool
    estimator_class_name: str | None = None
    experimental: bool = False
    anomaly_only: bool = False
    limitations: tuple[str, ...] = ()
    deprecated: bool = False

    # -- field-level validation ---------------------------------------------

    @field_validator("model_id")
    @classmethod
    def check_model_id(cls, value: str) -> str:
        """Model identifiers are stable once published."""
        if not _MODEL_ID_RE.match(value):
            raise ValueError(f"model_id {value!r} must match {_MODEL_ID_RE.pattern!r}")
        return value

    @field_validator("model_version")
    @classmethod
    def check_model_version(cls, value: str) -> str:
        """Model versions are semantic versions."""
        if not _SEMVER_RE.match(value):
            raise ValueError(f"model_version {value!r} must be a semantic version")
        return value

    @field_validator("serializer_id", "inference_adapter_id")
    @classmethod
    def check_adapter_id(cls, value: str) -> str:
        """Adapter identifiers are versioned registry keys, never import paths.

        The trailing ``_v<N>`` is load-bearing: a serializer whose output shape
        changes gets a new identifier rather than a silently different meaning
        for the old one.
        """
        if not _ADAPTER_ID_RE.match(value):
            raise ValueError(
                f"adapter identifier {value!r} must match {_ADAPTER_ID_RE.pattern!r}"
            )
        return value

    @field_validator("public_estimator_attributes", "private_estimator_attributes")
    @classmethod
    def check_attributes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Estimator attribute names are bare identifiers, never dotted paths."""
        for name in value:
            if not _ATTRIBUTE_RE.match(name):
                raise ValueError(
                    f"estimator attribute {name!r} must be a bare attribute name"
                )
        if len(set(value)) != len(value):
            raise ValueError("estimator attribute list repeats a name")
        return value

    @field_validator("estimator_class_name")
    @classmethod
    def check_class_name(cls, value: str | None) -> str | None:
        """Reject anything that could be read as an importable path.

        This field is documentation.  Nothing imports it, and the dot check
        makes sure nothing ever could without an obvious edit.
        """
        if value is None:
            return None
        if not _CLASS_NAME_RE.match(value):
            raise ValueError(
                f"estimator_class_name {value!r} must be a bare class name with "
                f"no module path"
            )
        return value

    @field_validator("description")
    @classmethod
    def check_description(cls, value: str) -> str:
        """Descriptions state what a family does, never what its output proves."""
        tokens = {token.strip(".,;:!?()[]'\"") for token in value.lower().split()}
        offending = sorted(tokens & _PROHIBITED_DESCRIPTION_TERMS)
        if offending:
            raise ValueError(
                f"model description uses claim-asserting term(s) {offending}; "
                f"describe the method and what its output orders"
            )
        return value

    @field_validator("supported_tasks", "supported_calibration_methods")
    @classmethod
    def check_non_empty_unique(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        """Declared task and calibration sets are non-empty and repeat nothing."""
        if not value:
            raise ValueError("declared set must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("declared set repeats a member")
        return value

    @field_validator("determinism_controls")
    @classmethod
    def check_determinism_controls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Every family must say how it is made reproducible."""
        if not value:
            raise ValueError("a model family must declare its determinism controls")
        if len(set(value)) != len(value):
            raise ValueError("determinism controls repeat an entry")
        return value

    # -- cross-field validation ---------------------------------------------

    @model_validator(mode="after")
    def check_declaration(self) -> Self:
        """Validate hyperparameters, eligibility, and score semantics together."""
        names = [parameter.name for parameter in self.hyperparameters]
        if len(set(names)) != len(names):
            raise ValueError(f"model {self.model_id} repeats a hyperparameter name")

        self._check_calibration()
        self._check_tasks()
        self._check_eligibility()
        self._check_estimator_attributes()
        return self

    def _check_calibration(self) -> None:
        """A family that cannot be calibrated may declare only ``NONE``."""
        methods = set(self.supported_calibration_methods)
        if self.calibration_compatible:
            if methods == {CalibrationMethod.NONE}:
                raise ValueError(
                    f"model {self.model_id} is calibration compatible but declares "
                    f"no calibration method"
                )
        elif methods != {CalibrationMethod.NONE}:
            raise ValueError(
                f"model {self.model_id} is not calibration compatible and may "
                f"declare only 'none'"
            )
        # A native output is never already a calibrated probability: calibration
        # is a separate fitted step, and a family claiming otherwise would let
        # an uncalibrated number be reported as one.
        if is_probability(self.native_score_kind):
            raise ValueError(
                f"model {self.model_id} declares a calibrated probability as its "
                f"native output; calibration is a separately fitted stage"
            )

    def _check_tasks(self) -> None:
        """Tie the declared tasks to multiclass capability and anomaly status."""
        tasks = set(self.supported_tasks)
        if MLTask.ATTACK_CATEGORY in tasks and not self.multiclass_capable:
            raise ValueError(
                f"model {self.model_id} declares the category task but is not "
                f"multiclass capable"
            )
        if self.anomaly_only and tasks != {MLTask.ANOMALY}:
            raise ValueError(
                f"model {self.model_id} is anomaly-only but declares supervised tasks"
            )
        if MLTask.ANOMALY in tasks and tasks & SUPERVISED_TASKS:
            raise ValueError(
                f"model {self.model_id} mixes the anomaly task with a supervised "
                f"task; they have different fitting contracts"
            )
        if MLTask.ANOMALY in tasks and self.native_score_kind is not (
            ScoreKind.ANOMALY_SCORE
        ):
            raise ValueError(
                f"model {self.model_id} performs the anomaly task but does not "
                f"emit an anomaly score"
            )

    def _check_eligibility(self) -> None:
        """Champion eligibility is a claim with prerequisites, not a label."""
        if self.champion_eligible:
            if self.anomaly_only:
                raise ValueError(
                    f"model {self.model_id} is anomaly-only and can never be the "
                    f"supervised champion"
                )
            if self.experimental:
                raise ValueError(
                    f"model {self.model_id} is experimental and cannot be "
                    f"champion eligible"
                )
            if self.deprecated:
                raise ValueError(
                    f"model {self.model_id} is deprecated and cannot be champion "
                    f"eligible"
                )
            if MLTask.BINARY_MALICIOUS not in self.supported_tasks:
                raise ValueError(
                    f"model {self.model_id} cannot be champion without supporting "
                    f"the binary task"
                )
            if self.private_estimator_attributes:
                raise ValueError(
                    f"model {self.model_id} reads private estimator attribute(s) "
                    f"{list(self.private_estimator_attributes)} and must not be "
                    f"champion eligible until a compatibility test proves the "
                    f"serializer contract"
                )
            if self.eligibility_status is not ModelEligibilityStatus.CHAMPION_ELIGIBLE:
                raise ValueError(
                    f"model {self.model_id} is champion eligible but declares "
                    f"status {self.eligibility_status}"
                )
        elif self.eligibility_status is ModelEligibilityStatus.CHAMPION_ELIGIBLE:
            raise ValueError(
                f"model {self.model_id} declares champion-eligible status without "
                f"the champion_eligible flag"
            )
        if self.anomaly_only and (
            self.eligibility_status is not ModelEligibilityStatus.ANOMALY_ONLY
        ):
            raise ValueError(
                f"model {self.model_id} is anomaly-only but declares status "
                f"{self.eligibility_status}"
            )

    def _check_estimator_attributes(self) -> None:
        """A family needing no third-party estimator declares no attributes."""
        if not self.requires_sklearn:
            if self.public_estimator_attributes or self.private_estimator_attributes:
                raise ValueError(
                    f"model {self.model_id} declares estimator attributes but "
                    f"requires no third-party estimator"
                )
            if self.estimator_class_name is not None:
                raise ValueError(
                    f"model {self.model_id} names an estimator class but requires "
                    f"no third-party estimator"
                )
        elif not self.public_estimator_attributes:
            raise ValueError(
                f"model {self.model_id} uses a third-party estimator but declares "
                f"no public attribute for its serializer to read"
            )
        overlap = sorted(
            set(self.public_estimator_attributes)
            & set(self.private_estimator_attributes)
        )
        if overlap:
            raise ValueError(
                f"model {self.model_id} declares {overlap} as both public and private"
            )

    # -- accessors ----------------------------------------------------------

    def hyperparameter(self, name: str) -> HyperparameterSpec:
        """Return the declared hyperparameter *name*.

        Raises:
            MLConfigurationError: if this family declares no such
                hyperparameter.  Configuration validation depends on this being
                a hard failure: an unrecognised key is a typo that would
                otherwise be silently ignored.
        """
        for parameter in self.hyperparameters:
            if parameter.name == name:
                return parameter
        raise MLConfigurationError(
            f"Model {self.model_id} declares no hyperparameter {name!r}"
        )

    def default_hyperparameters(self) -> dict[str, bool | int | float | str]:
        """Return every declared hyperparameter at its default value."""
        return {parameter.name: parameter.default for parameter in self.hyperparameters}

    def effective_hyperparameters(
        self, overrides: dict[str, object] | None = None
    ) -> dict[str, bool | int | float | str]:
        """Return the declared defaults with *overrides* validated and applied.

        Raises:
            MLConfigurationError: if an override names an undeclared
                hyperparameter or carries a value outside its declared bounds.
        """
        effective = self.default_hyperparameters()
        for name, value in (overrides or {}).items():
            parameter = self.hyperparameter(name)
            try:
                effective[name] = parameter.validate_value(value)
            except ValueError as exc:
                raise MLConfigurationError(f"Model {self.model_id}: {exc}") from None
        return effective

    def supports(self, task: MLTask) -> bool:
        """Return whether this family may be fitted for *task*."""
        return task in self.supported_tasks


#: Fields of :class:`ModelSpec` that participate in the catalog fingerprint.
#: ``description``, ``display_name``, and ``limitations`` are excluded on
#: purpose: correcting prose must not invalidate every artifact that recorded
#: the digest.  Everything that changes *behaviour* is included.
_FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "model_id",
    "model_version",
    "family",
    "supported_tasks",
    "native_score_kind",
    "calibration_compatible",
    "supported_calibration_methods",
    "multiclass_capable",
    "champion_eligible",
    "eligibility_status",
    "serializer_id",
    "inference_adapter_id",
    "public_estimator_attributes",
    "private_estimator_attributes",
    "determinism_controls",
    "requires_sklearn",
    "estimator_class_name",
    "experimental",
    "anomaly_only",
    "deprecated",
)


def _json_scalar(value: object) -> Any:
    """Render a spec field into a JSON-stable scalar."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, tuple | list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


class ModelCatalog:
    """An immutable, ordered collection of :class:`ModelSpec`.

    Specs are held sorted by ``model_id`` so iteration order is a property of
    the data rather than of import order -- the same contract ``RuleCatalog``
    follows.
    """

    __slots__ = ("_by_family", "_by_id", "_specs")

    def __init__(self, specs: tuple[ModelSpec, ...]) -> None:
        by_id: dict[str, ModelSpec] = {}
        by_family: dict[ModelFamily, ModelSpec] = {}
        for spec in specs:
            if spec.model_id in by_id:
                raise MLConfigurationError(
                    f"Duplicate model identifier in catalog: {spec.model_id!r}"
                )
            if spec.family in by_family:
                raise MLConfigurationError(
                    f"Duplicate model family in catalog: {str(spec.family)!r}"
                )
            by_id[spec.model_id] = spec
            by_family[spec.family] = spec
        self._specs = tuple(sorted(specs, key=lambda s: s.model_id))
        self._by_id = by_id
        self._by_family = by_family

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Any:
        return iter(self._specs)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._by_id

    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        """Return every model specification, ordered by model identifier."""
        return self._specs

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Return every model identifier, in catalog order."""
        return tuple(spec.model_id for spec in self._specs)

    @property
    def families(self) -> tuple[ModelFamily, ...]:
        """Return every declared family, in catalog order."""
        return tuple(spec.family for spec in self._specs)

    def get(self, model_id: str) -> ModelSpec:
        """Return the specification for *model_id*.

        Raises:
            MLConfigurationError: if no such model is registered.
        """
        try:
            return self._by_id[model_id]
        except KeyError:
            raise MLConfigurationError(
                f"Unknown model identifier: {model_id!r}"
            ) from None

    def for_family(self, family: ModelFamily) -> ModelSpec:
        """Return the specification for *family*.

        Raises:
            MLConfigurationError: if no such family is registered.
        """
        try:
            return self._by_family[family]
        except KeyError:
            raise MLConfigurationError(
                f"Unknown model family: {str(family)!r}"
            ) from None

    def has(self, model_id: str) -> bool:
        """Return whether *model_id* is registered."""
        return model_id in self._by_id

    def has_family(self, family: ModelFamily) -> bool:
        """Return whether *family* is registered."""
        return family in self._by_family

    def specs_for_task(self, task: MLTask) -> tuple[ModelSpec, ...]:
        """Return every model supporting *task*, in catalog order."""
        return tuple(spec for spec in self._specs if spec.supports(task))

    def champion_eligible_specs(self) -> tuple[ModelSpec, ...]:
        """Return every model that may currently be promoted, in catalog order.

        Membership here is necessary but not sufficient: promotion also
        requires proven serializer and inference-adapter parity, which this
        catalog cannot assert on its own.
        """
        return tuple(spec for spec in self._specs if spec.champion_eligible)

    def eligibility_counts(self) -> dict[str, int]:
        """Return the number of models per eligibility status, for reporting."""
        counts: dict[str, int] = {}
        for spec in self._specs:
            key = str(spec.eligibility_status)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def fingerprint(self) -> str:
        """Return a SHA-256 digest of the catalog's machine-readable contract.

        Covers identifiers, versions, families, tasks, score interfaces,
        calibration compatibility, eligibility, adapter identifiers, estimator
        attributes, determinism controls, and hyperparameter declarations.
        Excludes free prose, so fixing a typo in a description does not
        invalidate every artifact that recorded this fingerprint.
        """
        payload = {
            "model_catalog_version": MODEL_CATALOG_VERSION,
            "models": [
                {
                    **{
                        field: _json_scalar(getattr(spec, field))
                        for field in _FINGERPRINT_FIELDS
                    },
                    "hyperparameters": [
                        {
                            "name": parameter.name,
                            "kind": str(parameter.kind),
                            "default": _json_scalar(parameter.default),
                            "minimum": _json_scalar(parameter.minimum),
                            "maximum": _json_scalar(parameter.maximum),
                            "allowed_values": _json_scalar(parameter.allowed_values),
                            "unit": parameter.unit,
                            "tunable": parameter.tunable,
                        }
                        for parameter in spec.hyperparameters
                    ],
                }
                for spec in self._specs
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Shared hyperparameter declarations
# ---------------------------------------------------------------------------


def _random_state() -> HyperparameterSpec:
    """Return the seed hyperparameter every stochastic estimator must set."""
    return HyperparameterSpec(
        name="random_state",
        kind=HyperparameterKind.INT,
        default=42,
        minimum=0,
        maximum=2**31 - 1,
        tunable=False,
        description=(
            "Seed passed to the estimator. Fixed rather than tuned: it is a "
            "reproducibility control, not a modelling choice."
        ),
    )


def _n_jobs() -> HyperparameterSpec:
    """Return the thread-count hyperparameter, pinned to one."""
    return HyperparameterSpec(
        name="n_jobs",
        kind=HyperparameterKind.INT,
        default=1,
        minimum=1,
        maximum=1,
        tunable=False,
        description=(
            "Worker count, pinned to 1. Parallel tree fitting reorders "
            "floating-point reductions, which would break bit-for-bit "
            "reproduction across machines."
        ),
    )


def _class_weight() -> HyperparameterSpec:
    """Return the class-weight hyperparameter shared by supervised families."""
    return HyperparameterSpec(
        name="class_weight",
        kind=HyperparameterKind.STRING,
        default="balanced",
        allowed_values=("balanced", "none"),
        description=(
            "Per-class weighting. Weights are derived from training-split "
            "counts only; no resampling crosses a split boundary."
        ),
    )


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------


def _prior_baseline_spec() -> ModelSpec:
    """M-000: the mandatory reference every candidate must beat."""
    return ModelSpec(
        model_id="M-000",
        model_version="1.0.0",
        family=ModelFamily.PRIOR_BASELINE,
        display_name="Class prior baseline",
        description=(
            "Emits the training-split positive rate for every row, ignoring "
            "features entirely. It is the reference a candidate must beat "
            "before any of its numbers are worth reading: a model that cannot "
            "out-rank the base rate has learned nothing from the features."
        ),
        supported_tasks=(MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY),
        hyperparameters=(),
        native_score_kind=ScoreKind.DECISION_SCORE,
        calibration_compatible=True,
        supported_calibration_methods=(
            CalibrationMethod.NONE,
            CalibrationMethod.PLATT,
            CalibrationMethod.ISOTONIC,
        ),
        multiclass_capable=True,
        champion_eligible=True,
        eligibility_status=ModelEligibilityStatus.CHAMPION_ELIGIBLE,
        serializer_id="json_prior_v1",
        inference_adapter_id="prior_v1",
        determinism_controls=("closed-form fit", "no random number generator"),
        requires_sklearn=False,
        limitations=(
            "Constant output. Ranking is undefined, so every ranking metric "
            "over it is reported as unavailable rather than as a tie.",
            "Exists to be beaten. Promoting it would mean no candidate cleared "
            "the gates.",
        ),
    )


def _single_feature_threshold_spec() -> ModelSpec:
    """M-001: a one-feature cut, the simplest thing that can rank at all."""
    return ModelSpec(
        model_id="M-001",
        model_version="1.0.0",
        family=ModelFamily.SINGLE_FEATURE_THRESHOLD,
        display_name="Single-feature threshold baseline",
        description=(
            "Ranks rows by one eligible feature chosen on the training split "
            "by separation. It shows how much of a candidate's advantage comes "
            "from the feature set rather than from the learning algorithm."
        ),
        supported_tasks=(MLTask.BINARY_MALICIOUS,),
        hyperparameters=(
            HyperparameterSpec(
                name="selection_metric",
                kind=HyperparameterKind.STRING,
                default="roc_auc",
                allowed_values=("roc_auc", "average_precision"),
                description=(
                    "Training-split criterion used to pick the single feature. "
                    "Evaluated on the training split only."
                ),
            ),
            HyperparameterSpec(
                name="min_non_null_fraction",
                kind=HyperparameterKind.FLOAT,
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                description=(
                    "A feature must be observed on at least this fraction of "
                    "training rows to be selectable, so a mostly-null column "
                    "cannot win on a handful of observations."
                ),
            ),
        ),
        native_score_kind=ScoreKind.DECISION_SCORE,
        calibration_compatible=True,
        supported_calibration_methods=(
            CalibrationMethod.NONE,
            CalibrationMethod.PLATT,
            CalibrationMethod.ISOTONIC,
        ),
        multiclass_capable=False,
        champion_eligible=True,
        eligibility_status=ModelEligibilityStatus.CHAMPION_ELIGIBLE,
        serializer_id="json_threshold_v1",
        inference_adapter_id="threshold_v1",
        determinism_controls=(
            "deterministic feature ranking with a lexicographic tie-break",
            "no random number generator",
        ),
        requires_sklearn=False,
        limitations=(
            "One feature cannot express an interaction, so it under-reports "
            "what the feature set holds.",
            "Its chosen feature is recorded in the model artifact; a different "
            "training split may choose a different one.",
        ),
    )


def _logistic_regression_spec() -> ModelSpec:
    """M-010: the linear reference, and the most transparent real candidate."""
    return ModelSpec(
        model_id="M-010",
        model_version="1.0.0",
        family=ModelFamily.LOGISTIC_REGRESSION,
        display_name="Logistic regression",
        description=(
            "Regularised linear model over the standardised design matrix. "
            "Its coefficients are directly readable, which makes it the "
            "easiest candidate to audit and the natural reference for how much "
            "a non-linear family actually adds."
        ),
        supported_tasks=(MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY),
        hyperparameters=(
            HyperparameterSpec(
                name="penalty",
                kind=HyperparameterKind.STRING,
                default="l2",
                allowed_values=("l2",),
                tunable=False,
                description=(
                    "Regularisation term. Restricted to L2 because the lbfgs "
                    "solver is the only one this layer declares deterministic."
                ),
            ),
            HyperparameterSpec(
                name="solver",
                kind=HyperparameterKind.STRING,
                default="lbfgs",
                allowed_values=("lbfgs",),
                tunable=False,
                description=(
                    "Deterministic quasi-Newton solver. Stochastic solvers are "
                    "excluded: their results depend on iteration order."
                ),
            ),
            HyperparameterSpec(
                name="c_inverse_regularization",
                kind=HyperparameterKind.FLOAT,
                default=1.0,
                minimum=1e-4,
                maximum=1e4,
                description=(
                    "Inverse regularisation strength. Smaller values shrink "
                    "coefficients harder."
                ),
            ),
            HyperparameterSpec(
                name="max_iter",
                kind=HyperparameterKind.INT,
                default=1000,
                minimum=50,
                maximum=100_000,
                description=(
                    "Solver iteration ceiling. A fit that does not converge "
                    "within it is an error, never a silently truncated result."
                ),
            ),
            HyperparameterSpec(
                name="tol",
                kind=HyperparameterKind.FLOAT,
                default=1e-4,
                minimum=1e-10,
                maximum=1e-1,
                description="Convergence tolerance for the solver.",
            ),
            _class_weight(),
            _random_state(),
        ),
        native_score_kind=ScoreKind.DECISION_SCORE,
        calibration_compatible=True,
        supported_calibration_methods=(
            CalibrationMethod.NONE,
            CalibrationMethod.PLATT,
            CalibrationMethod.ISOTONIC,
        ),
        multiclass_capable=True,
        champion_eligible=True,
        eligibility_status=ModelEligibilityStatus.CHAMPION_ELIGIBLE,
        serializer_id="json_linear_v1",
        inference_adapter_id="linear_logit_v1",
        public_estimator_attributes=("coef_", "intercept_", "classes_"),
        determinism_controls=(
            "random_state fixed",
            "deterministic lbfgs solver",
            "fixed feature order recorded in the artifact",
        ),
        requires_sklearn=True,
        estimator_class_name="LogisticRegression",
        limitations=(
            "Linear in the transformed space; it cannot represent an "
            "interaction the preprocessing did not already encode.",
            "Its raw output is an ordered decision score. It becomes a "
            "calibrated quantity only after a calibrator is fitted and its "
            "calibration error measured.",
        ),
    )


def _random_forest_spec() -> ModelSpec:
    """M-020: the ensemble candidate whose serializer reads documented arrays."""
    return ModelSpec(
        model_id="M-020",
        model_version="1.0.0",
        family=ModelFamily.RANDOM_FOREST,
        display_name="Random forest",
        description=(
            "Bagged axis-aligned trees over the untransformed design matrix. "
            "Chosen as the ensemble candidate because its fitted structure is "
            "exposed through documented array attributes, so a canonical "
            "serializer can read it without depending on an internal layout."
        ),
        supported_tasks=(MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY),
        hyperparameters=(
            HyperparameterSpec(
                name="n_estimators",
                kind=HyperparameterKind.INT,
                default=300,
                minimum=1,
                maximum=2000,
                description="Number of trees in the forest.",
            ),
            HyperparameterSpec(
                name="max_depth",
                kind=HyperparameterKind.INT,
                default=12,
                minimum=1,
                maximum=64,
                description=(
                    "Depth ceiling. Bounded rather than unlimited so the "
                    "serialised artifact stays a reviewable size."
                ),
            ),
            HyperparameterSpec(
                name="min_samples_leaf",
                kind=HyperparameterKind.INT,
                default=5,
                minimum=1,
                maximum=1000,
                description="Minimum training rows required to form a leaf.",
            ),
            HyperparameterSpec(
                name="max_features",
                kind=HyperparameterKind.STRING,
                default="sqrt",
                allowed_values=("sqrt", "log2"),
                description=(
                    "Features considered per split. A fixed fraction is "
                    "excluded so the value cannot depend on the column count."
                ),
            ),
            _class_weight(),
            _random_state(),
            _n_jobs(),
        ),
        native_score_kind=ScoreKind.DECISION_SCORE,
        calibration_compatible=True,
        supported_calibration_methods=(
            CalibrationMethod.NONE,
            CalibrationMethod.PLATT,
            CalibrationMethod.ISOTONIC,
        ),
        multiclass_capable=True,
        champion_eligible=True,
        eligibility_status=ModelEligibilityStatus.CHAMPION_ELIGIBLE,
        serializer_id="json_tree_ensemble_v1",
        inference_adapter_id="tree_vote_v1",
        public_estimator_attributes=(
            "estimators_",
            "classes_",
            "n_outputs_",
            "n_features_in_",
        ),
        determinism_controls=(
            "random_state fixed",
            "n_jobs pinned to 1",
            "OMP_NUM_THREADS=1 for bit-for-bit reproduction",
        ),
        requires_sklearn=True,
        estimator_class_name="RandomForestClassifier",
        limitations=(
            "Leaf-frequency output is poorly calibrated by construction; it "
            "needs a fitted calibrator before it can be read as a rate.",
            "Artifact size grows with tree count and depth, so both are "
            "bounded in the declaration above.",
        ),
    )


def _histogram_gradient_boosting_spec() -> ModelSpec:
    """M-021: evaluable, deliberately not promotable until M4 proves the export."""
    return ModelSpec(
        model_id="M-021",
        model_version="1.0.0",
        family=ModelFamily.HISTOGRAM_GRADIENT_BOOSTING,
        display_name="Histogram gradient boosting",
        description=(
            "Boosted trees over binned features, typically the strongest "
            "tabular family available here. It is gated: its fitted structure "
            "is reachable only through private estimator attributes, so it may "
            "be evaluated but not promoted until a compatibility test settles "
            "whether a stable serializer can be built on them."
        ),
        supported_tasks=(MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY),
        hyperparameters=(
            HyperparameterSpec(
                name="max_iter",
                kind=HyperparameterKind.INT,
                default=200,
                minimum=1,
                maximum=2000,
                description="Number of boosting iterations.",
            ),
            HyperparameterSpec(
                name="learning_rate",
                kind=HyperparameterKind.FLOAT,
                default=0.1,
                minimum=1e-4,
                maximum=1.0,
                description="Shrinkage applied to each iteration's contribution.",
            ),
            HyperparameterSpec(
                name="max_leaf_nodes",
                kind=HyperparameterKind.INT,
                default=31,
                minimum=2,
                maximum=255,
                description="Leaf ceiling per tree.",
            ),
            HyperparameterSpec(
                name="min_samples_leaf",
                kind=HyperparameterKind.INT,
                default=20,
                minimum=1,
                maximum=1000,
                description="Minimum training rows required to form a leaf.",
            ),
            HyperparameterSpec(
                name="l2_regularization",
                kind=HyperparameterKind.FLOAT,
                default=0.0,
                minimum=0.0,
                maximum=100.0,
                description="L2 penalty applied to leaf values.",
            ),
            HyperparameterSpec(
                name="early_stopping",
                kind=HyperparameterKind.BOOL,
                default=False,
                tunable=False,
                description=(
                    "Pinned off. Internal early stopping would carve a "
                    "validation slice out of the training split without the "
                    "campaign-group isolation this project requires."
                ),
            ),
            _random_state(),
        ),
        native_score_kind=ScoreKind.DECISION_SCORE,
        calibration_compatible=True,
        supported_calibration_methods=(
            CalibrationMethod.NONE,
            CalibrationMethod.PLATT,
            CalibrationMethod.ISOTONIC,
        ),
        multiclass_capable=True,
        champion_eligible=False,
        eligibility_status=ModelEligibilityStatus.SERIALIZER_UNPROVEN,
        serializer_id="json_histogram_ensemble_v1",
        inference_adapter_id="histogram_raw_v1",
        public_estimator_attributes=("classes_", "n_iter_", "n_features_in_"),
        private_estimator_attributes=("_predictors", "_baseline_prediction"),
        determinism_controls=(
            "random_state fixed",
            "early_stopping pinned off",
            "OMP_NUM_THREADS=1 for bit-for-bit reproduction",
        ),
        requires_sklearn=True,
        estimator_class_name="HistGradientBoostingClassifier",
        limitations=(
            "Its fitted trees are reachable only through private attributes, "
            "whose layout is not part of the scikit-learn public contract. "
            "Promotion is blocked until a compatibility test pins that layout "
            "on the bounded version range.",
            "If that test cannot be made to hold, this family stays evaluable "
            "and the random forest remains the ensemble candidate.",
        ),
    )


def _isolation_forest_spec() -> ModelSpec:
    """M-030: the novel-holdout probe, never a supervised champion."""
    return ModelSpec(
        model_id="M-030",
        model_version="1.0.0",
        family=ModelFamily.ISOLATION_FOREST,
        display_name="Isolation forest",
        description=(
            "Unsupervised outlier scorer fitted on benign training rows "
            "without reading a label. It exists to probe behaviour on the "
            "novel-anomaly holdout, where no supervised model has a class to "
            "predict and no rule was written."
        ),
        supported_tasks=(MLTask.ANOMALY,),
        hyperparameters=(
            HyperparameterSpec(
                name="n_estimators",
                kind=HyperparameterKind.INT,
                default=200,
                minimum=1,
                maximum=1000,
                description="Number of isolation trees.",
            ),
            HyperparameterSpec(
                name="max_samples",
                kind=HyperparameterKind.INT,
                default=256,
                minimum=16,
                maximum=100_000,
                description=(
                    "Rows drawn per tree. Declared as a count rather than a "
                    "fraction so the value does not depend on dataset size."
                ),
            ),
            HyperparameterSpec(
                name="contamination",
                kind=HyperparameterKind.FLOAT,
                default=0.01,
                minimum=1e-6,
                maximum=0.5,
                description=(
                    "Assumed outlier share, used only to place the estimator's "
                    "own offset. This layer sets its threshold from a training "
                    "benign quantile instead."
                ),
            ),
            _random_state(),
            _n_jobs(),
        ),
        native_score_kind=ScoreKind.ANOMALY_SCORE,
        calibration_compatible=False,
        supported_calibration_methods=(CalibrationMethod.NONE,),
        multiclass_capable=False,
        champion_eligible=False,
        eligibility_status=ModelEligibilityStatus.ANOMALY_ONLY,
        serializer_id="json_isolation_forest_v1",
        inference_adapter_id="isolation_path_v1",
        public_estimator_attributes=(
            "estimators_",
            "estimators_features_",
            "max_samples_",
            "offset_",
            "n_features_in_",
        ),
        determinism_controls=(
            "random_state fixed",
            "n_jobs pinned to 1",
            "max_samples declared as an absolute count",
        ),
        requires_sklearn=True,
        estimator_class_name="IsolationForest",
        experimental=True,
        anomaly_only=True,
        limitations=(
            "Its output is an ordered outlier magnitude. It is not a "
            "calibrated quantity and no report may present it as one.",
            "It never influences champion selection or threshold choice; its "
            "results are reported in a separate holdout section.",
            "Its path-length normalisation is reimplemented from the published "
            "formula and pinned by a parity test, because the scikit-learn "
            "helper is private.",
        ),
    )


def build_model_catalog() -> ModelCatalog:
    """Return the model catalog assembled from its declared specifications.

    Raises:
        MLConfigurationError: if two specifications share an identifier or a
            family.
    """
    return ModelCatalog(
        (
            _prior_baseline_spec(),
            _single_feature_threshold_spec(),
            _logistic_regression_spec(),
            _random_forest_spec(),
            _histogram_gradient_boosting_spec(),
            _isolation_forest_spec(),
        )
    )


#: The catalog every consumer reads.  Built once at import; immutable.
MODEL_CATALOG: Final[ModelCatalog] = build_model_catalog()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def model_catalog_to_markdown(catalog: ModelCatalog = MODEL_CATALOG) -> str:
    """Render the model catalog as Markdown documentation.

    Derived entirely from catalog metadata, so the documentation cannot drift
    from the registry.  Contains no data, no identifiers, no paths, and no
    performance figure.
    """
    lines: list[str] = [
        "# Model Catalog",
        "",
        "Generated from the model registry. Do not edit by hand; regenerate "
        "with `password-attack-detector ml catalog --format markdown`.",
        "",
        f"- Model catalog version: `{MODEL_CATALOG_VERSION}`",
        f"- Catalog fingerprint: `{catalog.fingerprint()}`",
        f"- Declared model families: {len(catalog)}",
        "",
        "This catalog declares *what may be fitted*, not what was fitted or "
        "how well anything performed. It contains no measured result, and no "
        "performance figure is ever transcribed into this repository's prose.",
        "",
        "## What catalog membership does not mean",
        "",
        "- **Membership is not championship.** A family listed here may be "
        "fitted and evaluated. Becoming the champion additionally requires "
        "clearing every validation gate, and no model is promoted on the "
        "strength of appearing in this table.",
        "- **Serializer and inference parity are required, and are proven "
        "later.** A family becomes promotable only once its canonical "
        "serializer and its inference adapter reproduce identical scores after "
        "a round trip. Until that parity is demonstrated, a family is "
        "evaluable but not promotable.",
        "- **`M-021` is gated.** Histogram gradient boosting reads private "
        "estimator attributes, so it ships with `champion_eligible = false` "
        "until a compatibility test pins that layout across the bounded "
        "dependency range.",
        "- **`M-030` is an experimental anomaly model.** It is unsupervised, "
        "anomaly-only, and can never be the supervised champion. Its output is "
        "an ordered outlier magnitude.",
        "- **Probabilities require calibration.** Every family's native output "
        "is a decision score or a class score. It may be described as a "
        "probability only after a calibrator has been fitted and its "
        "calibration error measured.",
        "- **Phase 5 is offline and defensive.** Nothing here serves a model, "
        "exposes an endpoint, touches live authentication traffic, or handles "
        "a credential.",
        "",
        "## Eligibility summary",
        "",
        "| Eligibility status | Families |",
        "|--------|-------|",
    ]
    for status, count in sorted(catalog.eligibility_counts().items()):
        lines.append(f"| `{status}` | {count} |")

    lines.extend(
        [
            "",
            "## Model index",
            "",
            "| Model | Family | Tasks | Champion eligible | Experimental |",
            "|--------|-------|-------|-------|-------|",
        ]
    )
    for spec in catalog.specs:
        tasks = ", ".join(f"`{task}`" for task in spec.supported_tasks)
        lines.append(
            f"| `{spec.model_id}` | `{spec.family}` | {tasks} | "
            f"{_yes_no(spec.champion_eligible)} | {_yes_no(spec.experimental)} |"
        )

    for spec in catalog.specs:
        lines.extend(_model_markdown(spec))

    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Models consume Phase 3 point-in-time feature snapshots. Split "
            "assignments, campaign metadata, and ground-truth labels reach the "
            "training path only through the single module permitted to read "
            "them, and never as model inputs.",
            "- A declared hyperparameter range bounds what configuration may "
            "request. It says nothing about which value is appropriate for a "
            "given dataset.",
            "- Determinism controls make a fit reproducible on one locked "
            "dependency set. They do not make results comparable across "
            "different scikit-learn releases, which is why the dependency "
            "range is bounded on both sides.",
            "- Synthetic data exercises every family declared here but "
            "demonstrates nothing about real-world detection effectiveness.",
            "",
        ]
    )
    return "\n".join(lines)


def _yes_no(value: bool) -> str:
    """Render a boolean for a documentation table."""
    return "yes" if value else "no"


def _model_markdown(spec: ModelSpec) -> list[str]:
    """Render one model family's section."""
    lines: list[str] = [
        "",
        f"## {spec.model_id} -- {spec.display_name}",
        "",
        spec.description,
        "",
        "| Property | Value |",
        "|--------|-------|",
        f"| Model version | `{spec.model_version}` |",
        f"| Family | `{spec.family}` |",
        f"| Supported tasks | {', '.join(f'`{t}`' for t in spec.supported_tasks)} |",
        f"| Native score kind | `{spec.native_score_kind}` |",
        f"| Calibration compatible | {_yes_no(spec.calibration_compatible)} |",
        (
            "| Calibration methods | "
            f"{', '.join(f'`{m}`' for m in spec.supported_calibration_methods)} |"
        ),
        f"| Multiclass capable | {_yes_no(spec.multiclass_capable)} |",
        f"| Champion eligible | {_yes_no(spec.champion_eligible)} |",
        f"| Eligibility status | `{spec.eligibility_status}` |",
        f"| Experimental | {_yes_no(spec.experimental)} |",
        f"| Anomaly only | {_yes_no(spec.anomaly_only)} |",
        f"| Serializer | `{spec.serializer_id}` |",
        f"| Inference adapter | `{spec.inference_adapter_id}` |",
        f"| Requires scikit-learn | {_yes_no(spec.requires_sklearn)} |",
    ]
    if spec.estimator_class_name is not None:
        lines.append(f"| Estimator | `{spec.estimator_class_name}` |")
    if spec.public_estimator_attributes:
        joined = ", ".join(f"`{name}`" for name in spec.public_estimator_attributes)
        lines.append(f"| Public estimator attributes | {joined} |")
    if spec.private_estimator_attributes:
        joined = ", ".join(f"`{name}`" for name in spec.private_estimator_attributes)
        lines.append(f"| Private estimator attributes | {joined} |")

    lines.extend(["", "**Determinism controls**", ""])
    lines.extend(f"- {control}" for control in spec.determinism_controls)

    if spec.hyperparameters:
        lines.extend(
            [
                "",
                "**Hyperparameters**",
                "",
                "| Name | Kind | Default | Minimum | Maximum | Choices | Tunable |",
                "|--------|-------|-------|-------|-------|-------|-------|",
            ]
        )
        for parameter in spec.hyperparameters:
            choices = (
                ", ".join(f"`{value}`" for value in parameter.allowed_values)
                if parameter.allowed_values is not None
                else "-"
            )
            lines.append(
                f"| `{parameter.name}` | `{parameter.kind}` | "
                f"`{parameter.default}` | {_optional(parameter.minimum)} | "
                f"{_optional(parameter.maximum)} | {choices} | "
                f"{_yes_no(parameter.tunable)} |"
            )

    if spec.limitations:
        lines.extend(["", "**Limitations**", ""])
        lines.extend(f"- {limitation}" for limitation in spec.limitations)

    return lines


def _optional(value: float | None) -> str:
    """Render an optional numeric bound for a documentation table."""
    if value is None:
        return "-"
    if value == int(value) and abs(value) < 1e15:
        return f"`{int(value)}`"
    return f"`{value}`"
