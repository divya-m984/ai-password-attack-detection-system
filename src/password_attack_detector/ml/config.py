"""Typed, versioned configuration for the machine-learning detection layer.

``MLConfig`` is the single source of truth for one modelling run.  Its
``fingerprint()`` method produces a SHA-256 hex digest covering *semantic*
behaviour only.  Output directories, overwrite flags, absolute paths, creation
timestamps, and machine-specific values are excluded, so the same semantic
configuration stored in two different directories fingerprints identically --
the same contract ``FeatureConfig`` and ``DetectionConfig`` already follow.

Configuration is **data, never logic**.  YAML is read with ``yaml.safe_load``
and supplies values for hyperparameters a model family has already declared in
the catalog.  There is no expression evaluation, no callable reference, no
import path, and no ``eval``, ``exec``, or dynamic import anywhere in this
package.  A key that looks like a secret is rejected outright: modelling needs
no credential, so accepting one could only ever be a mistake.

Four disciplines are enforced by the *type system* rather than by a runtime
check, which is the strongest form available here:

* **Calibration reads validation-A; thresholds read validation-B.**  Both
  fields are ``Literal`` types over a single member of
  :class:`~password_attack_detector.ml.enums.ValidationPartition`, and that enum
  has no test member and no holdout member.  A configuration cannot name the
  test split as a source of a fitted quantity because there is no name to give.
* **Validation is partitioned on campaign-group boundaries, never at a row
  midpoint.**  ``ValidationPartitionConfig.strategy`` admits exactly one value.
  A midpoint cut would put a campaign's early events in the partition that
  fits the calibrator and its later events in the partition that picks the
  threshold, which is the same leak the Phase 3 splitter exists to close.
* **Missingness indicators cannot be switched off.**  Phase 3's null-versus-zero
  distinction is load-bearing; imputing without an indicator would destroy it.
* **No resampling, ever.**  Imbalance is handled by weights computed from
  training-split counts. ``ImbalanceConfig.resampling`` admits only ``"none"``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from password_attack_detector.exceptions import ConfigurationError, MLConfigurationError
from password_attack_detector.features.catalog import FeatureGroup, LeakageClass
from password_attack_detector.features.config import FEATURE_SCHEMA_VERSION
from password_attack_detector.ml.catalog import (
    MODEL_CATALOG,
    MODEL_CATALOG_VERSION,
    ModelCatalog,
)
from password_attack_detector.ml.dependencies import ML_DEPENDENCY_REQUIREMENTS
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    CalibrationMethod,
    FusionStrategy,
    MLTask,
    ModelFamily,
    ThresholdObjective,
    ValidationPartition,
)
from password_attack_detector.ml.schemas import (
    ML_SCHEMA_VERSION,
    DependencyRequirement,
    SupportRequirement,
)

__all__ = [
    "ML_FINGERPRINT_EXCLUDED_FIELDS",
    "PROHIBITED_CONFIG_KEY_TOKENS",
    "AnomalyConfig",
    "CalibrationConfig",
    "CategoryConfig",
    "ChampionGateConfig",
    "DriftConfig",
    "ExplainConfig",
    "FusionConfig",
    "ImbalanceConfig",
    "MLConfig",
    "PreprocessingConfig",
    "ThresholdConfig",
    "ValidationPartitionConfig",
    "load_ml_config",
]

#: Fields of :class:`MLConfig` deliberately excluded from its fingerprint.
#: These describe *where* output goes, not *what* modelling means.
ML_FINGERPRINT_EXCLUDED_FIELDS: Final[frozenset[str]] = frozenset(
    {"output_dir", "models_dir", "reports_dir", "overwrite"}
)

#: Tokens that must not appear in any ML configuration key.
#:
#: Declared here rather than imported from the detection layer on purpose: the
#: ML package does not depend on the rule engine in either direction, and a
#: shared constant would be the first thread of a coupling this architecture
#: deliberately avoids.  The two lists are identical because the reasoning is:
#: neither layer has any use for a credential.
PROHIBITED_CONFIG_KEY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "secret",
        "secrets",
        "password",
        "passwords",
        "passwd",
        "token",
        "tokens",
        "credential",
        "credentials",
        "key",
        "keys",
        "apikey",
        "privatekey",
        "salt",
        "pepper",
        "cookie",
        "cookies",
        "bearer",
        "hash",
        "hashes",
    }
)

#: Leakage classes a model input may be drawn from.  ``KEY`` is excluded: the
#: anchor identifier, its timestamp, and the schema version are join and
#: provenance columns, never features.
_ELIGIBLE_LEAKAGE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        str(LeakageClass.PRIOR_ONLY),
        str(LeakageClass.CURRENT_EVENT_CONTEXT),
        str(LeakageClass.BASELINE_DERIVED),
    }
)

#: Feature groups a configuration may name in an allowlist.  ``key`` is
#: excluded for the same reason its leakage class is.
_ELIGIBLE_FEATURE_GROUPS: Final[frozenset[str]] = frozenset(
    str(group) for group in FeatureGroup if group is not FeatureGroup.KEY
)


def _controlled_vocabulary() -> frozenset[str]:
    """Return every configuration key drawn from a validated vocabulary.

    ``family_hyperparameters`` is keyed by model family, and each family's
    entry is keyed by a hyperparameter the catalog already declares.  Those
    keys are validated against the catalog itself, so the secret scanner
    exempts them rather than rejecting a legitimate name.

    Building the exemption from the catalog keeps it correct as the catalog
    grows; a hand-written allowlist would drift.
    """
    vocabulary: set[str] = {str(family) for family in ModelFamily}
    vocabulary |= {str(task) for task in MLTask}
    for spec in MODEL_CATALOG.specs:
        vocabulary |= {parameter.name for parameter in spec.hyperparameters}
    return frozenset(vocabulary)


_CONTROLLED_VOCABULARY: Final[frozenset[str]] = _controlled_vocabulary()


def _check_safe_path(path: Path | None, field_name: str) -> Path | None:
    """Reject an output path containing parent-directory traversal."""
    if path is not None and any(part == ".." for part in path.parts):
        raise ValueError(f"{field_name} must not contain '..' path components")
    return path


def _reject_secret_keys(data: object, path: str = "") -> None:
    """Walk a raw configuration mapping and reject secret-shaped keys.

    Raises:
        ConfigurationError: naming the offending key path only.  The value is
            never read, echoed, or included in the message.
    """
    if isinstance(data, dict):
        for raw_key, value in data.items():
            key = str(raw_key)
            if key not in _CONTROLLED_VOCABULARY:
                tokens = set(key.lower().replace("-", "_").split("_"))
                offending = sorted(tokens & PROHIBITED_CONFIG_KEY_TOKENS)
                if offending:
                    where = f"{path}.{key}" if path else key
                    raise ConfigurationError(
                        f"ML configuration key {where!r} looks like a secret "
                        f"({offending}); model configuration must never carry "
                        f"credentials"
                    )
            _reject_secret_keys(value, f"{path}.{key}" if path else key)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            _reject_secret_keys(item, f"{path}[{index}]")


def _fingerprint_scalar(value: object) -> Any:
    """Render a configuration value into a JSON-stable scalar."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, tuple | list):
        return [_fingerprint_scalar(item) for item in value]
    return str(value)


class PreprocessingConfig(BaseModel):
    """How raw feature snapshots become a numeric design matrix.

    Every statistic -- an imputation constant, a category list, a scaling mean
    -- is fitted on the training split alone.  ``statistics_source`` admits one
    value so no configuration can widen it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    statistics_source: Literal["train"] = "train"
    numeric_imputation: Literal["median", "zero"] = "median"
    #: Pinned on.  Phase 3 distinguishes "no attempts were made" from "no
    #: failures out of many"; imputing a null without flagging it would erase
    #: that distinction and hand the model a fabricated observation.
    add_missing_indicators: Literal[True] = True
    categorical_encoding: Literal["one_hot"] = "one_hot"
    min_category_frequency: int = Field(default=20, ge=1)
    max_category_cardinality: int = Field(default=64, ge=2)
    rare_category_label: str = "__other"
    unknown_category_label: str = "__unknown"
    missing_category_label: str = "__missing"
    standardize_numeric_for_linear_models: bool = True
    include_leakage_classes: tuple[str, ...] = (
        str(LeakageClass.PRIOR_ONLY),
        str(LeakageClass.CURRENT_EVENT_CONTEXT),
        str(LeakageClass.BASELINE_DERIVED),
    )
    include_feature_groups: tuple[str, ...] | None = None

    @field_validator("include_leakage_classes")
    @classmethod
    def check_leakage_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Only eligible leakage classes may be named; ``key`` never is."""
        if not value:
            raise ValueError("include_leakage_classes must name at least one class")
        if len(set(value)) != len(value):
            raise ValueError("include_leakage_classes repeats a class")
        unknown = sorted(set(value) - _ELIGIBLE_LEAKAGE_CLASSES)
        if unknown:
            raise ValueError(
                f"include_leakage_classes names ineligible class(es) {unknown}; "
                f"eligible: {sorted(_ELIGIBLE_LEAKAGE_CLASSES)}"
            )
        return value

    @field_validator("include_feature_groups")
    @classmethod
    def check_feature_groups(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """An explicit group allowlist must name declared, eligible groups."""
        if value is None:
            return value
        if not value:
            raise ValueError(
                "include_feature_groups must be omitted for 'all groups' rather "
                "than set to an empty list"
            )
        if len(set(value)) != len(value):
            raise ValueError("include_feature_groups repeats a group")
        unknown = sorted(set(value) - _ELIGIBLE_FEATURE_GROUPS)
        if unknown:
            raise ValueError(
                f"include_feature_groups names ineligible group(s) {unknown}; "
                f"eligible: {sorted(_ELIGIBLE_FEATURE_GROUPS)}"
            )
        return value

    @field_validator(
        "rare_category_label", "unknown_category_label", "missing_category_label"
    )
    @classmethod
    def check_bucket_label(cls, value: str) -> str:
        """Bucket labels are double-underscore prefixed so they cannot collide.

        A real country code or client type can never begin with ``__``, so a
        synthetic bucket is always distinguishable from an observed value.
        """
        if not value.startswith("__") or len(value) < 3:
            raise ValueError(
                f"bucket label {value!r} must start with '__' so it cannot "
                f"collide with an observed category value"
            )
        return value

    @model_validator(mode="after")
    def check_labels_distinct(self) -> Self:
        """The three synthetic bucket labels must be distinguishable."""
        labels = {
            self.rare_category_label,
            self.unknown_category_label,
            self.missing_category_label,
        }
        if len(labels) != 3:
            raise ValueError("rare, unknown, and missing bucket labels must differ")
        if self.min_category_frequency > self.max_category_cardinality:
            raise ValueError(
                "min_category_frequency above max_category_cardinality would "
                "collapse every category into the rare bucket"
            )
        return self

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "statistics_source": self.statistics_source,
            "numeric_imputation": self.numeric_imputation,
            "add_missing_indicators": self.add_missing_indicators,
            "categorical_encoding": self.categorical_encoding,
            "min_category_frequency": self.min_category_frequency,
            "max_category_cardinality": self.max_category_cardinality,
            "rare_category_label": self.rare_category_label,
            "unknown_category_label": self.unknown_category_label,
            "missing_category_label": self.missing_category_label,
            "standardize_numeric_for_linear_models": (
                self.standardize_numeric_for_linear_models
            ),
            "include_leakage_classes": sorted(self.include_leakage_classes),
            "include_feature_groups": (
                sorted(self.include_feature_groups)
                if self.include_feature_groups is not None
                else None
            ),
        }


class ImbalanceConfig(BaseModel):
    """How class imbalance is handled.

    By weighting, computed from training-split counts.  ``resampling`` admits
    only ``"none"``: synthesising or dropping rows to balance a class would
    change the data a split contract already fixed, and any resampler that
    looked across a split boundary would leak.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_weight_policy: Literal["balanced", "none"] = "balanced"
    computed_from: Literal["train"] = "train"
    resampling: Literal["none"] = "none"

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "class_weight_policy": self.class_weight_policy,
            "computed_from": self.computed_from,
            "resampling": self.resampling,
        }


class ValidationPartitionConfig(BaseModel):
    """How the validation split is divided into a calibration and a threshold half.

    The strategy admits exactly one value: the boundary is placed at the
    earliest **campaign-group** boundary at or after the target fraction, never
    at a row midpoint.  A midpoint cut would put a campaign's early events in
    the partition that fits the calibrator and its later events in the
    partition that chooses the operating point -- the same memorisation leak
    the Phase 3 splitter closes between train, validation, and test.

    The grouping key comes from the Phase 2 label table.  ``campaign_id`` is
    deliberately absent from the published feature tables, so a run that
    partitions validation must be handed that table explicitly.  Milestone 2
    implements the partitioning; this declares its contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["campaign_group_boundary"] = "campaign_group_boundary"
    #: Named ``grouping_column`` rather than ``group_key``: the configuration
    #: loader rejects any key carrying a credential-shaped token, and ``key``
    #: is one of them.  The rename is cheaper than an exemption, and it is the
    #: more accurate name -- ``campaign_id`` is a column of the label table.
    grouping_column: Literal["campaign_id"] = "campaign_id"
    boundary_placement: Literal["first_group_boundary_at_or_after_target"] = (
        "first_group_boundary_at_or_after_target"
    )
    #: Benign traffic is not coordinated, so each benign row is its own group
    #: and is placed by event time -- the same reasoning behind the Phase 3
    #: splitter's ``normal_grouping: singleton`` default.
    ungrouped_row_policy: Literal["singleton_by_event_time"] = "singleton_by_event_time"
    target_partition_a_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    min_partition_rows: int = Field(default=200, ge=1)
    min_partition_positive_rows: int = Field(default=10, ge=1)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "strategy": self.strategy,
            "grouping_column": self.grouping_column,
            "boundary_placement": self.boundary_placement,
            "ungrouped_row_policy": self.ungrouped_row_policy,
            "target_partition_a_fraction": _fingerprint_scalar(
                self.target_partition_a_fraction
            ),
            "min_partition_rows": self.min_partition_rows,
            "min_partition_positive_rows": self.min_partition_positive_rows,
        }


class CalibrationConfig(BaseModel):
    """How a raw decision score becomes a calibrated probability.

    ``source_partition`` is a ``Literal`` over validation-A alone.  The enum it
    draws from has no test member, so no configuration can name the test split
    here -- the absence is the enforcement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: CalibrationMethod = CalibrationMethod.ISOTONIC
    source_partition: Literal[ValidationPartition.VALIDATION_A] = (
        ValidationPartition.VALIDATION_A
    )
    max_expected_calibration_error: float = Field(default=0.05, gt=0.0, le=1.0)
    reliability_bin_count: int = Field(default=10, ge=2, le=100)
    min_calibration_rows: int = Field(default=200, ge=1)
    #: Pinned on.  A model output may be described as a probability only after
    #: a calibrator has been fitted and its calibration error measured; there
    #: is no configuration that relaxes the vocabulary rule.
    require_calibration_for_probability_terminology: Literal[True] = True

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "method": str(self.method),
            "source_partition": str(self.source_partition),
            "max_expected_calibration_error": _fingerprint_scalar(
                self.max_expected_calibration_error
            ),
            "reliability_bin_count": self.reliability_bin_count,
            "min_calibration_rows": self.min_calibration_rows,
            "require_calibration_for_probability_terminology": (
                self.require_calibration_for_probability_terminology
            ),
        }


class ThresholdConfig(BaseModel):
    """How the binary decision threshold is chosen.

    ``source_partition`` is a ``Literal`` over validation-B alone, for the same
    reason :class:`CalibrationConfig` pins validation-A: the enum has no test
    member to name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: ThresholdObjective = ThresholdObjective.MAX_RECALL_AT_MAX_FPR
    source_partition: Literal[ValidationPartition.VALIDATION_B] = (
        ValidationPartition.VALIDATION_B
    )
    max_false_positive_rate: float = Field(default=0.01, gt=0.0, lt=1.0)
    min_detection_rate: float = Field(default=0.50, gt=0.0, le=1.0)
    search_grid_size: int = Field(default=1000, ge=2, le=100_000)
    #: With several thresholds tied on the objective, take the lower one: it
    #: detects at least as much, and the tie means it costs no more.
    tie_break: Literal["lowest_threshold", "highest_threshold"] = "lowest_threshold"

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "objective": str(self.objective),
            "source_partition": str(self.source_partition),
            "max_false_positive_rate": _fingerprint_scalar(
                self.max_false_positive_rate
            ),
            "min_detection_rate": _fingerprint_scalar(self.min_detection_rate),
            "search_grid_size": self.search_grid_size,
            "tie_break": self.tie_break,
        }


class CategoryConfig(BaseModel):
    """How the known-attack category head behaves.

    The class space is derived at runtime from the label schema, never written
    down here: hard-coding a class count would let a renamed or added scenario
    pass silently.  What *is* configured is the abstention policy, because
    refusing to guess is a decision the operator should be able to see.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    fit_on: Literal["train_known_malicious"] = "train_known_malicious"
    class_order: Literal["sorted_scenario_value"] = "sorted_scenario_value"
    min_category_score: float = Field(default=0.35, ge=0.0, lt=1.0)
    abstain_label: str = UNKNOWN_CATEGORY
    min_rows_per_category: int = Field(default=20, ge=1)
    report_conditional_metrics: bool = True
    report_cascade_metrics: bool = True

    @field_validator("abstain_label")
    @classmethod
    def check_abstain_label(cls, value: str) -> str:
        """The abstention label is the project-wide sentinel, not a free choice.

        A per-configuration label would let two runs disagree about what
        "no known class" is called, which would silently break every
        comparison that groups by predicted category.
        """
        if value != UNKNOWN_CATEGORY:
            raise ValueError(f"abstain_label must be {UNKNOWN_CATEGORY!r}")
        return value

    @model_validator(mode="after")
    def check_reporting(self) -> Self:
        """An enabled category head must report at least one metric view."""
        if self.enabled and not (
            self.report_conditional_metrics or self.report_cascade_metrics
        ):
            raise ValueError(
                "an enabled category head must report conditional metrics, "
                "cascade metrics, or both"
            )
        return self

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "enabled": self.enabled,
            "fit_on": self.fit_on,
            "class_order": self.class_order,
            "min_category_score": _fingerprint_scalar(self.min_category_score),
            "abstain_label": self.abstain_label,
            "min_rows_per_category": self.min_rows_per_category,
            "report_conditional_metrics": self.report_conditional_metrics,
            "report_cascade_metrics": self.report_cascade_metrics,
        }


class AnomalyConfig(BaseModel):
    """How the experimental anomaly probe is fitted and thresholded.

    Fitted on benign training rows without reading a label, and thresholded
    from a training-benign quantile rather than from anything the holdout
    reveals.  Its results never influence champion selection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    fit_on: Literal["train_benign"] = "train_benign"
    threshold_method: Literal["train_benign_quantile"] = "train_benign_quantile"
    quantile: float = Field(default=0.99, gt=0.0, lt=1.0)
    min_fit_rows: int = Field(default=500, ge=1)
    #: Pinned off.  The anomaly probe is a generalisation measurement, and a
    #: measurement that can change the thing it measures is not one.
    influences_champion_selection: Literal[False] = False

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "enabled": self.enabled,
            "fit_on": self.fit_on,
            "threshold_method": self.threshold_method,
            "quantile": _fingerprint_scalar(self.quantile),
            "min_fit_rows": self.min_fit_rows,
            "influences_champion_selection": self.influences_champion_selection,
        }


class ChampionGateConfig(BaseModel):
    """The gates a candidate must clear on validation data to be promotable.

    Every gate is evaluated on validation only.  Serializer parity and
    determinism are pinned on: a model that cannot be reproduced or reloaded
    exactly is not a model this project will publish, whatever it scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_family: ModelFamily = ModelFamily.PRIOR_BASELINE
    min_pr_auc_gain_over_baseline: float = Field(default=0.05, ge=0.0, le=1.0)
    max_false_positive_rate: float = Field(default=0.01, gt=0.0, lt=1.0)
    min_detection_rate: float = Field(default=0.50, gt=0.0, le=1.0)
    max_expected_calibration_error: float = Field(default=0.05, gt=0.0, le=1.0)
    require_serializer_parity: Literal[True] = True
    require_determinism: Literal[True] = True
    require_eligibility_audit_pass: Literal[True] = True
    parity_tolerance: float = Field(default=1e-12, gt=0.0, le=1e-3)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "baseline_family": str(self.baseline_family),
            "min_pr_auc_gain_over_baseline": _fingerprint_scalar(
                self.min_pr_auc_gain_over_baseline
            ),
            "max_false_positive_rate": _fingerprint_scalar(
                self.max_false_positive_rate
            ),
            "min_detection_rate": _fingerprint_scalar(self.min_detection_rate),
            "max_expected_calibration_error": _fingerprint_scalar(
                self.max_expected_calibration_error
            ),
            "require_serializer_parity": self.require_serializer_parity,
            "require_determinism": self.require_determinism,
            "require_eligibility_audit_pass": self.require_eligibility_audit_pass,
            "parity_tolerance": _fingerprint_scalar(self.parity_tolerance),
        }


class FusionConfig(BaseModel):
    """Which rule-and-model fusion strategies are built and compared.

    Every enabled strategy is evaluated; none is assumed to win.  The winning
    strategy is chosen on validation-B, alongside the ML threshold, and frozen
    before the test split is opened.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    strategies: tuple[FusionStrategy, ...] = (
        FusionStrategy.OR_GATE,
        FusionStrategy.AND_GATE,
        FusionStrategy.STACKED,
    )
    selection_partition: Literal[ValidationPartition.VALIDATION_B] = (
        ValidationPartition.VALIDATION_B
    )
    stacked_max_iter: int = Field(default=1000, ge=50, le=100_000)

    @field_validator("strategies")
    @classmethod
    def check_strategies(
        cls, value: tuple[FusionStrategy, ...]
    ) -> tuple[FusionStrategy, ...]:
        """At least one strategy, and no repeats."""
        if not value:
            raise ValueError("strategies must name at least one fusion strategy")
        if len(set(value)) != len(value):
            raise ValueError("strategies repeats a strategy")
        return value

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "enabled": self.enabled,
            "strategies": sorted(str(strategy) for strategy in self.strategies),
            "selection_partition": str(self.selection_partition),
            "stacked_max_iter": self.stacked_max_iter,
        }


class ExplainConfig(BaseModel):
    """How feature attribution is computed and how much of it is emitted.

    Attribution is computed on a validation partition, never on test.  Local
    explanations are bounded and carry feature names only by default: a feature
    *value* can be a country code, and a report is not the place to widen what
    is disclosed without saying so.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    partition: ValidationPartition = ValidationPartition.VALIDATION_A
    top_k_features: int = Field(default=25, ge=1, le=200)
    permutation_repeats: int = Field(default=5, ge=1, le=100)
    max_local_explanations: int = Field(default=0, ge=0, le=10_000)
    include_feature_values: bool = False

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "enabled": self.enabled,
            "partition": str(self.partition),
            "top_k_features": self.top_k_features,
            "permutation_repeats": self.permutation_repeats,
            "max_local_explanations": self.max_local_explanations,
            "include_feature_values": self.include_feature_values,
        }


class DriftConfig(BaseModel):
    """How a training reference profile is captured and later compared.

    A hook, not a monitor: it captures a distribution profile at training time
    and compares a later batch against it.  Nothing here schedules, alerts, or
    retrains.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    reference_source: Literal["train"] = "train"
    quantile_count: int = Field(default=20, ge=2, le=1000)
    psi_warn_threshold: float = Field(default=0.10, gt=0.0, le=10.0)
    psi_alert_threshold: float = Field(default=0.25, gt=0.0, le=10.0)
    min_reference_rows: int = Field(default=500, ge=1)

    @model_validator(mode="after")
    def check_thresholds_ordered(self) -> Self:
        """The alert threshold must sit strictly above the warning threshold."""
        if self.psi_warn_threshold >= self.psi_alert_threshold:
            raise ValueError(
                "psi_warn_threshold must be strictly below psi_alert_threshold"
            )
        return self

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields contributing to the config fingerprint."""
        return {
            "enabled": self.enabled,
            "reference_source": self.reference_source,
            "quantile_count": self.quantile_count,
            "psi_warn_threshold": _fingerprint_scalar(self.psi_warn_threshold),
            "psi_alert_threshold": _fingerprint_scalar(self.psi_alert_threshold),
            "min_reference_rows": self.min_reference_rows,
        }


class MLConfig(BaseModel):
    """Complete configuration for one machine-learning run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ml_schema_version: Literal["1.0.0"] = ML_SCHEMA_VERSION
    required_feature_schema_version: str = FEATURE_SCHEMA_VERSION
    model_catalog_version: Literal["1.0.0"] = MODEL_CATALOG_VERSION
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    enabled_model_families: tuple[ModelFamily, ...] = (
        ModelFamily.PRIOR_BASELINE,
        ModelFamily.SINGLE_FEATURE_THRESHOLD,
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.ISOLATION_FOREST,
    )
    family_hyperparameters: dict[ModelFamily, dict[str, bool | int | float | str]] = (
        Field(default_factory=dict)
    )

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    validation_partition: ValidationPartitionConfig = Field(
        default_factory=ValidationPartitionConfig
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    category: CategoryConfig = Field(default_factory=CategoryConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    support: SupportRequirement = Field(default_factory=SupportRequirement)
    gates: ChampionGateConfig = Field(default_factory=ChampionGateConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)

    dependency_requirements: tuple[DependencyRequirement, ...] = (
        ML_DEPENDENCY_REQUIREMENTS
    )

    output_dir: Path | None = None
    models_dir: Path | None = None
    reports_dir: Path | None = None
    overwrite: bool = False

    # -- validation ---------------------------------------------------------

    @field_validator("required_feature_schema_version")
    @classmethod
    def check_feature_schema_version(cls, value: str) -> str:
        """The consumed feature contract version is a semantic version."""
        parts = value.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(
                f"required_feature_schema_version {value!r} must be a semantic version"
            )
        return value

    @field_validator("output_dir", "models_dir", "reports_dir")
    @classmethod
    def check_paths(cls, value: Path | None) -> Path | None:
        """Output paths must not escape upwards."""
        return _check_safe_path(value, "output path")

    @field_validator("enabled_model_families")
    @classmethod
    def check_families(cls, value: tuple[ModelFamily, ...]) -> tuple[ModelFamily, ...]:
        """Families are declared, unique, and drawn from the catalog."""
        if not value:
            raise ValueError("enabled_model_families must name at least one family")
        if len(set(value)) != len(value):
            duplicates = sorted(
                {str(family) for family in value if list(value).count(family) > 1}
            )
            raise ValueError(f"enabled_model_families repeats {duplicates}")
        return value

    @field_validator("dependency_requirements")
    @classmethod
    def check_requirements(
        cls, value: tuple[DependencyRequirement, ...]
    ) -> tuple[DependencyRequirement, ...]:
        """Every declared requirement names a distinct distribution."""
        names = [requirement.distribution for requirement in value]
        if len(set(names)) != len(names):
            raise ValueError("dependency_requirements repeats a distribution")
        return value

    @model_validator(mode="after")
    def check_cross_field_constraints(self) -> Self:
        """Validate families, hyperparameters, and gate coherence together."""
        self._check_catalog_agreement(MODEL_CATALOG)
        self._check_task_coverage(MODEL_CATALOG)
        self._check_gate_coherence()
        return self

    def _check_catalog_agreement(self, catalog: ModelCatalog) -> None:
        """Every enabled family and every override must exist in the catalog."""
        for family in self.enabled_model_families:
            if not catalog.has_family(family):
                raise ValueError(f"unknown model family {str(family)!r}")

        enabled = set(self.enabled_model_families)
        for family, overrides in self.family_hyperparameters.items():
            if family not in enabled:
                raise ValueError(
                    f"family_hyperparameters configures {str(family)!r}, which is "
                    f"not enabled"
                )
            spec = catalog.for_family(family)
            try:
                spec.effective_hyperparameters(dict(overrides))
            except MLConfigurationError as exc:
                raise ValueError(str(exc)) from None

    def _check_task_coverage(self, catalog: ModelCatalog) -> None:
        """The enabled set must be able to perform the work it is configured for."""
        specs = [catalog.for_family(family) for family in self.enabled_model_families]

        if self.gates.baseline_family not in set(self.enabled_model_families):
            raise ValueError(
                f"gate baseline family {str(self.gates.baseline_family)!r} must be "
                f"enabled; every candidate is measured against it"
            )
        if not any(spec.supports(MLTask.BINARY_MALICIOUS) for spec in specs):
            raise ValueError(
                "no enabled family supports the binary task, which is the primary "
                "task of this layer"
            )
        if self.category.enabled and not any(
            spec.supports(MLTask.ATTACK_CATEGORY) for spec in specs
        ):
            raise ValueError(
                "the category head is enabled but no enabled family supports the "
                "category task"
            )
        if self.anomaly.enabled and not any(
            spec.supports(MLTask.ANOMALY) for spec in specs
        ):
            raise ValueError(
                "the anomaly probe is enabled but no enabled family supports the "
                "anomaly task"
            )
        if self.calibration.method is not CalibrationMethod.NONE:
            calibratable = [spec for spec in specs if spec.calibration_compatible]
            if not calibratable:
                raise ValueError(
                    f"calibration method {str(self.calibration.method)!r} is "
                    f"configured but no enabled family is calibration compatible"
                )

    def _check_gate_coherence(self) -> None:
        """Reject gate settings that no run could ever satisfy."""
        if self.gates.max_false_positive_rate < self.thresholds.max_false_positive_rate:
            raise ValueError(
                "the champion gate's false-positive ceiling is stricter than the "
                "ceiling the threshold search optimises against; no threshold "
                "chosen on validation could ever clear the gate"
            )
        # Both detection-rate floors are always positive, so the only
        # unsatisfiable combination is a gate demanding more than the search
        # was told to hold.  That only binds under the objective that treats
        # the floor as a constraint; the other two do not promise it at all.
        if (
            self.thresholds.objective is ThresholdObjective.MIN_FPR_AT_MIN_RECALL
            and self.gates.min_detection_rate > self.thresholds.min_detection_rate
        ):
            raise ValueError(
                "the champion gate requires a higher detection rate than the "
                "threshold search is configured to hold"
            )
        if self.calibration.method is CalibrationMethod.NONE:
            return
        if (
            self.gates.max_expected_calibration_error
            < self.calibration.max_expected_calibration_error
        ):
            raise ValueError(
                "the champion gate's calibration ceiling is stricter than the "
                "calibration stage's own ceiling"
            )

    # -- accessors ----------------------------------------------------------

    def hyperparameters_for(
        self, family: ModelFamily, catalog: ModelCatalog = MODEL_CATALOG
    ) -> dict[str, bool | int | float | str]:
        """Return the effective hyperparameters for *family*.

        Declared defaults with configured overrides applied and validated, so
        the returned mapping is what a fit would actually run with.

        Raises:
            MLConfigurationError: if *family* is not enabled.
        """
        if family not in set(self.enabled_model_families):
            raise MLConfigurationError(
                f"Model family {str(family)!r} is not enabled in this configuration"
            )
        overrides = self.family_hyperparameters.get(family, {})
        return catalog.for_family(family).effective_hyperparameters(dict(overrides))

    def requirement_for(self, distribution: str) -> DependencyRequirement | None:
        """Return the declared bounded range for *distribution*, if any."""
        for requirement in self.dependency_requirements:
            if requirement.distribution == distribution:
                return requirement
        return None

    # -- fingerprint --------------------------------------------------------

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint.

        Every field of this model must appear here except those listed in
        :data:`ML_FINGERPRINT_EXCLUDED_FIELDS`.  A test asserts that invariant,
        so a field added without a decision about its fingerprint status fails
        the build rather than silently weakening the digest.

        Hyperparameters are recorded at their *effective* values, so the digest
        pins what would actually be fitted rather than only the overrides that
        happened to be written down.
        """
        return {
            "ml_schema_version": self.ml_schema_version,
            "required_feature_schema_version": self.required_feature_schema_version,
            "model_catalog_version": self.model_catalog_version,
            "seed": self.seed,
            "enabled_model_families": sorted(
                str(family) for family in self.enabled_model_families
            ),
            "family_hyperparameters": {
                str(family): {
                    name: _fingerprint_scalar(value)
                    for name, value in sorted(self.hyperparameters_for(family).items())
                }
                for family in sorted(self.enabled_model_families, key=str)
            },
            "preprocessing": self.preprocessing.fingerprint_data(),
            "imbalance": self.imbalance.fingerprint_data(),
            "validation_partition": self.validation_partition.fingerprint_data(),
            "calibration": self.calibration.fingerprint_data(),
            "thresholds": self.thresholds.fingerprint_data(),
            "category": self.category.fingerprint_data(),
            "anomaly": self.anomaly.fingerprint_data(),
            "support": self.support.fingerprint_data(),
            "gates": self.gates.fingerprint_data(),
            "fusion": self.fusion.fingerprint_data(),
            "explain": self.explain.fingerprint_data(),
            "drift": self.drift.fingerprint_data(),
            "dependency_requirements": [
                {
                    "distribution": requirement.distribution,
                    "minimum_version": requirement.minimum_version,
                    "below_version": requirement.below_version,
                }
                for requirement in sorted(
                    self.dependency_requirements,
                    key=lambda item: item.distribution,
                )
            ],
        }

    def fingerprint(self) -> str:
        """Return a SHA-256 hex digest of the semantic ML configuration.

        Deliberately **excludes** ``output_dir``, ``models_dir``,
        ``reports_dir``, ``overwrite``, absolute paths, creation timestamps,
        and machine-specific values, so the same semantic configuration in two
        different directories produces the same fingerprint.
        """
        canonical = json.dumps(self.fingerprint_data(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_ml_config(path: Path) -> MLConfig:
    """Load and validate an :class:`MLConfig` from a YAML file.

    Uses ``yaml.safe_load``, which constructs only plain scalars, lists, and
    mappings.  No Python object, callable, or import path can be materialised
    from a configuration file.

    Raises:
        ConfigurationError: if the file is missing, unreadable, not a mapping,
            carries a secret-shaped key, or fails validation.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read ML configuration: {type(exc).__name__}"
        ) from None

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        raise ConfigurationError("ML configuration is not valid YAML") from None

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigurationError("ML configuration must be a YAML mapping")

    _reject_secret_keys(data)

    try:
        return MLConfig(**data)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Invalid ML configuration: {exc}") from None
