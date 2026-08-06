"""Tests for the typed ML configuration.

The configuration is where most of this layer's scientific discipline is
encoded, so most of these tests are about what a configuration *cannot* say:
it cannot name the test split as a source of a fitted quantity, cannot switch
off missingness indicators, cannot request resampling, cannot let the anomaly
probe influence selection, and cannot halve validation at a row midpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from password_attack_detector.exceptions import ConfigurationError, MLConfigurationError
from password_attack_detector.ml.config import (
    ML_FINGERPRINT_EXCLUDED_FIELDS,
    AnomalyConfig,
    CalibrationConfig,
    CategoryConfig,
    ChampionGateConfig,
    DriftConfig,
    FusionConfig,
    MLConfig,
    PreprocessingConfig,
    ThresholdConfig,
    ValidationPartitionConfig,
    load_ml_config,
)
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    CalibrationMethod,
    FusionStrategy,
    ModelFamily,
    ThresholdObjective,
    ValidationPartition,
)

SUB_CONFIGS = (
    AnomalyConfig,
    CalibrationConfig,
    CategoryConfig,
    DriftConfig,
    FusionConfig,
    PreprocessingConfig,
    ThresholdConfig,
    ValidationPartitionConfig,
)


def _repo_root() -> Path:
    """Return the repository root, located from this test file."""
    return Path(__file__).resolve().parents[3]


def _config_path(name: str) -> Path:
    """Return the path to a tracked ML configuration."""
    return _repo_root() / "configs" / "ml" / name


def _write(tmp_path: Path, body: str) -> Path:
    """Write *body* to a temporary YAML file and return its path."""
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Shared model configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", (MLConfig, *SUB_CONFIGS))
def test_every_config_model_is_frozen_and_forbids_extra_fields(
    model: type[BaseModel],
) -> None:
    """An unknown key is a typo, and a typo must fail rather than be ignored."""
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"


def test_an_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    """Silently ignoring a misspelled key would ship the wrong configuration."""
    path = _write(tmp_path, "seed: 1\nlerning_rate: 0.1\n")
    with pytest.raises(ConfigurationError, match="Invalid ML configuration"):
        load_ml_config(path)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_every_field_is_fingerprinted_or_explicitly_excluded() -> None:
    """A field added without a fingerprint decision must fail the build.

    This is the invariant that keeps the digest honest as the configuration
    grows: there is no third option between "covered" and "listed as excluded".
    """
    config = MLConfig()
    covered = set(config.fingerprint_data())
    declared = set(MLConfig.model_fields)
    assert covered | ML_FINGERPRINT_EXCLUDED_FIELDS == declared
    assert not covered & ML_FINGERPRINT_EXCLUDED_FIELDS


def test_the_excluded_set_contains_only_output_location_fields() -> None:
    """Exclusions describe where output goes, never what modelling means."""
    assert {
        "output_dir",
        "models_dir",
        "reports_dir",
        "overwrite",
    } == ML_FINGERPRINT_EXCLUDED_FIELDS


def test_the_fingerprint_is_a_stable_lower_case_digest() -> None:
    """Two identical configurations agree, and the shape is SHA-256 hex."""
    first = MLConfig().fingerprint()
    assert first == MLConfig().fingerprint()
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_the_fingerprint_is_path_independent() -> None:
    """The same semantic configuration in two directories must agree."""
    left = MLConfig(
        output_dir=Path("/tmp/left"),
        models_dir=Path("/tmp/left/models"),
        reports_dir=Path("/tmp/left/reports"),
        overwrite=True,
    )
    right = MLConfig(
        output_dir=Path("/var/right"),
        models_dir=Path("/var/right/models"),
        reports_dir=Path("/var/right/reports"),
        overwrite=False,
    )
    assert left.fingerprint() == right.fingerprint()


def test_the_fingerprint_tracks_a_semantic_change() -> None:
    """Anything that changes what would be fitted must move the digest."""
    baseline = MLConfig().fingerprint()
    assert MLConfig(seed=99).fingerprint() != baseline
    # The operating ceiling has to move on both the search and the gate: the
    # coherence check rejects a gate stricter than the search it grades.
    loosened = MLConfig(
        thresholds=ThresholdConfig(max_false_positive_rate=0.02),
        gates=ChampionGateConfig(max_false_positive_rate=0.02),
    )
    assert loosened.fingerprint() != baseline
    assert MLConfig(thresholds=ThresholdConfig(search_grid_size=500)).fingerprint() != (
        baseline
    )


def test_the_fingerprint_records_effective_hyperparameters() -> None:
    """The digest pins what would run, not only what was written down.

    Restating a declared default explicitly must not change the digest, and
    changing a value must.
    """
    implicit = MLConfig()
    explicit = MLConfig(
        family_hyperparameters={ModelFamily.RANDOM_FOREST: {"n_estimators": 300}}
    )
    changed = MLConfig(
        family_hyperparameters={ModelFamily.RANDOM_FOREST: {"n_estimators": 100}}
    )
    assert implicit.fingerprint() == explicit.fingerprint()
    assert changed.fingerprint() != implicit.fingerprint()


def test_the_fingerprint_covers_the_declared_dependency_range() -> None:
    """Widening the reviewed range is a semantic change to the run."""
    from password_attack_detector.ml.schemas import DependencyRequirement

    widened = MLConfig(
        dependency_requirements=(
            DependencyRequirement(
                distribution="scikit-learn",
                minimum_version="1.9.0",
                below_version="2.0",
                reason="widened for the test",
            ),
        )
    )
    assert widened.fingerprint() != MLConfig().fingerprint()


# ---------------------------------------------------------------------------
# Split discipline, enforced by types
# ---------------------------------------------------------------------------


def test_calibration_can_only_read_validation_a(tmp_path: Path) -> None:
    """The calibrator is fitted on validation-A; nothing else is expressible."""
    assert CalibrationConfig().source_partition is ValidationPartition.VALIDATION_A
    for forbidden in ("test", "train", "validation_b", "novel_anomaly_holdout"):
        path = _write(tmp_path, f"calibration:\n  source_partition: {forbidden}\n")
        with pytest.raises(ConfigurationError):
            load_ml_config(path)


def test_thresholds_can_only_read_validation_b(tmp_path: Path) -> None:
    """The operating point is chosen on validation-B and nowhere else."""
    assert ThresholdConfig().source_partition is ValidationPartition.VALIDATION_B
    for forbidden in ("test", "train", "validation_a", "novel_anomaly_holdout"):
        path = _write(tmp_path, f"thresholds:\n  source_partition: {forbidden}\n")
        with pytest.raises(ConfigurationError):
            load_ml_config(path)


def test_no_configured_source_field_can_name_test_or_holdout() -> None:
    """Sweep every fitted-quantity source across the whole configuration."""
    config = MLConfig()
    sources = (
        config.preprocessing.statistics_source,
        config.imbalance.computed_from,
        str(config.calibration.source_partition),
        str(config.thresholds.source_partition),
        config.category.fit_on,
        config.anomaly.fit_on,
        config.anomaly.threshold_method,
        str(config.fusion.selection_partition),
        str(config.explain.partition),
        config.drift.reference_source,
    )
    for source in sources:
        assert "test" not in source, source
        assert "holdout" not in source, source
        assert "novel" not in source, source


def test_explanation_can_only_read_a_validation_partition() -> None:
    """Attribution is computed on validation; the enum offers nothing else."""
    for partition in ValidationPartition:
        assert "test" not in partition.value


# ---------------------------------------------------------------------------
# Validation partitioning
# ---------------------------------------------------------------------------


def test_validation_is_partitioned_on_campaign_group_boundaries() -> None:
    """Not a row midpoint. A midpoint cut would split a campaign in half."""
    partition = ValidationPartitionConfig()
    assert partition.strategy == "campaign_group_boundary"
    assert partition.grouping_column == "campaign_id"
    assert partition.boundary_placement == "first_group_boundary_at_or_after_target"


def test_a_row_midpoint_partition_strategy_is_rejected(tmp_path: Path) -> None:
    """The only value the strategy admits is the campaign-aware one."""
    for forbidden in ("row_midpoint", "midpoint", "random", "stratified"):
        path = _write(tmp_path, f"validation_partition:\n  strategy: {forbidden}\n")
        with pytest.raises(ConfigurationError):
            load_ml_config(path)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_an_out_of_range_partition_fraction_is_rejected(fraction: float) -> None:
    """A fraction at or beyond either end would produce an empty partition."""
    with pytest.raises(ValidationError):
        ValidationPartitionConfig(target_partition_a_fraction=fraction)


@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_partition_floor_is_rejected(value: int) -> None:
    """A floor of zero would let an empty partition read as sufficient."""
    with pytest.raises(ValidationError):
        ValidationPartitionConfig(min_partition_rows=value)


# ---------------------------------------------------------------------------
# Preprocessing policy
# ---------------------------------------------------------------------------


def test_missingness_indicators_cannot_be_switched_off(tmp_path: Path) -> None:
    """Phase 3's null-versus-zero distinction is load-bearing.

    Imputing a null without flagging it hands the model a fabricated
    observation and erases a distinction the feature contract preserves
    deliberately.
    """
    assert PreprocessingConfig().add_missing_indicators is True
    path = _write(tmp_path, "preprocessing:\n  add_missing_indicators: false\n")
    with pytest.raises(ConfigurationError):
        load_ml_config(path)


def test_the_key_leakage_class_cannot_be_named_as_a_model_input() -> None:
    """The anchor identifier and its timestamp are joins, never features."""
    with pytest.raises(ValidationError, match="ineligible class"):
        PreprocessingConfig(include_leakage_classes=("prior_only", "key"))


def test_an_unknown_leakage_class_is_rejected() -> None:
    """A misspelled class would silently drop every feature it named."""
    with pytest.raises(ValidationError, match="ineligible class"):
        PreprocessingConfig(include_leakage_classes=("prior_onl",))


def test_the_key_feature_group_cannot_be_named() -> None:
    """The same reasoning as the leakage class, applied to groups."""
    with pytest.raises(ValidationError, match="ineligible group"):
        PreprocessingConfig(include_feature_groups=("user_history", "key"))


def test_an_empty_leakage_class_list_is_rejected() -> None:
    """A design matrix with no eligible column is not a configuration."""
    with pytest.raises(ValidationError, match="at least one class"):
        PreprocessingConfig(include_leakage_classes=())


def test_an_empty_group_allowlist_must_be_omitted_rather_than_emptied() -> None:
    """An empty list reads as "none"; omission is how "all" is expressed."""
    with pytest.raises(ValidationError, match="must be omitted"):
        PreprocessingConfig(include_feature_groups=())
    assert PreprocessingConfig().include_feature_groups is None


def test_a_repeated_leakage_class_is_rejected() -> None:
    """A repeat is a mistake, and would double-count in the fingerprint."""
    with pytest.raises(ValidationError, match="repeats a class"):
        PreprocessingConfig(include_leakage_classes=("prior_only", "prior_only"))


@pytest.mark.parametrize("label", ["other", "_other", "__"])
def test_a_bucket_label_must_be_double_underscore_prefixed(label: str) -> None:
    """A real country code cannot start with ``__``, so a bucket is distinct."""
    with pytest.raises(ValidationError, match="must start with"):
        PreprocessingConfig(rare_category_label=label)


def test_the_three_bucket_labels_must_differ() -> None:
    """Colliding labels would make rare, unknown, and missing indistinguishable."""
    with pytest.raises(ValidationError, match="must differ"):
        PreprocessingConfig(rare_category_label="__x", unknown_category_label="__x")


def test_a_frequency_floor_above_the_cardinality_ceiling_is_rejected() -> None:
    """It would collapse every observed category into the rare bucket."""
    with pytest.raises(ValidationError, match="collapse every category"):
        PreprocessingConfig(min_category_frequency=100, max_category_cardinality=8)


def test_resampling_is_never_available(tmp_path: Path) -> None:
    """Weights only. A resampler crossing a split boundary would leak."""
    from password_attack_detector.ml.config import ImbalanceConfig

    assert ImbalanceConfig().resampling == "none"
    for forbidden in ("smote", "random_over", "random_under", "adasyn"):
        path = _write(tmp_path, f"imbalance:\n  resampling: {forbidden}\n")
        with pytest.raises(ConfigurationError):
            load_ml_config(path)


# ---------------------------------------------------------------------------
# Category and anomaly policy
# ---------------------------------------------------------------------------


def test_the_abstention_label_is_the_project_sentinel() -> None:
    """A per-run label would break every comparison that groups by category."""
    assert CategoryConfig().abstain_label == UNKNOWN_CATEGORY
    with pytest.raises(ValidationError, match="abstain_label must be"):
        CategoryConfig(abstain_label="other")


def test_the_category_class_space_is_not_configured() -> None:
    """It is derived from the label schema, so a renamed scenario fails a test.

    Hard-coding a class count or a class list here would let a scenario added
    or renamed in ``ScenarioType`` pass silently.
    """
    fields = set(CategoryConfig.model_fields)
    for forbidden in ("classes", "class_names", "category_count", "num_classes"):
        assert forbidden not in fields


def test_an_enabled_category_head_must_report_something() -> None:
    """Fitting a head and reporting neither view would be pointless."""
    with pytest.raises(ValidationError, match="must report conditional"):
        CategoryConfig(
            enabled=True,
            report_conditional_metrics=False,
            report_cascade_metrics=False,
        )


def test_the_anomaly_probe_cannot_influence_selection(tmp_path: Path) -> None:
    """A measurement that can change what it measures is not a measurement."""
    assert AnomalyConfig().influences_champion_selection is False
    path = _write(tmp_path, "anomaly:\n  influences_champion_selection: true\n")
    with pytest.raises(ConfigurationError):
        load_ml_config(path)


def test_the_anomaly_probe_fits_on_benign_training_rows_only() -> None:
    """It reads no label, and the only expressible source says so."""
    assert AnomalyConfig().fit_on == "train_benign"
    assert AnomalyConfig().threshold_method == "train_benign_quantile"


# ---------------------------------------------------------------------------
# Family selection and coherence
# ---------------------------------------------------------------------------


def test_an_unknown_model_family_is_rejected(tmp_path: Path) -> None:
    """A family the catalog does not declare could never be fitted."""
    path = _write(tmp_path, "enabled_model_families:\n  - neural_network\n")
    with pytest.raises(ConfigurationError):
        load_ml_config(path)


def test_a_duplicate_model_family_is_rejected() -> None:
    """A repeat is a mistake and would be ambiguous in the fingerprint."""
    with pytest.raises(ValidationError, match="repeats"):
        MLConfig(
            enabled_model_families=(
                ModelFamily.PRIOR_BASELINE,
                ModelFamily.PRIOR_BASELINE,
            )
        )


def test_an_empty_family_list_is_rejected() -> None:
    """A run with no model is not a run."""
    with pytest.raises(ValidationError, match="at least one family"):
        MLConfig(enabled_model_families=())


def test_the_gate_baseline_family_must_be_enabled() -> None:
    """Every candidate is measured against it, so it must actually be fitted."""
    with pytest.raises(ValidationError, match="must be enabled"):
        MLConfig(enabled_model_families=(ModelFamily.LOGISTIC_REGRESSION,))


def test_a_run_must_be_able_to_perform_the_binary_task() -> None:
    """The binary task is the primary one and the basis of every comparison."""
    with pytest.raises(ValidationError, match="no enabled family supports the binary"):
        MLConfig(
            enabled_model_families=(ModelFamily.ISOLATION_FOREST,),
            gates={"baseline_family": ModelFamily.ISOLATION_FOREST},  # type: ignore[arg-type]
        )


def test_an_enabled_anomaly_probe_needs_an_anomaly_capable_family() -> None:
    """Configuring a stage nothing can perform must fail at load."""
    with pytest.raises(ValidationError, match="anomaly probe is enabled"):
        MLConfig(
            enabled_model_families=(
                ModelFamily.PRIOR_BASELINE,
                ModelFamily.LOGISTIC_REGRESSION,
            ),
            anomaly=AnomalyConfig(enabled=True),
        )


def test_an_enabled_category_head_needs_a_multiclass_family() -> None:
    """The same rule, applied to the category stage.

    The single-feature threshold is the only binary-only family, so it is the
    only enabled set that can reach this check. The prior baseline is
    multiclass capable, which is why the usual configurations never trip it.
    """
    with pytest.raises(ValidationError, match="category head is enabled"):
        MLConfig(
            enabled_model_families=(ModelFamily.SINGLE_FEATURE_THRESHOLD,),
            gates=ChampionGateConfig(
                baseline_family=ModelFamily.SINGLE_FEATURE_THRESHOLD
            ),
            category=CategoryConfig(enabled=True),
            anomaly=AnomalyConfig(enabled=False),
        )


def test_hyperparameters_for_an_unenabled_family_are_rejected() -> None:
    """Configuring a family that will not be fitted is a mistake worth failing."""
    with pytest.raises(ValidationError, match="not enabled"):
        MLConfig(
            enabled_model_families=(ModelFamily.PRIOR_BASELINE,),
            family_hyperparameters={ModelFamily.RANDOM_FOREST: {"n_estimators": 10}},
            category=CategoryConfig(enabled=False),
            anomaly=AnomalyConfig(enabled=False),
        )


def test_an_out_of_range_hyperparameter_is_rejected() -> None:
    """Bounds declared in the catalog are enforced at configuration load."""
    with pytest.raises(ValidationError, match="at most"):
        MLConfig(
            family_hyperparameters={
                ModelFamily.RANDOM_FOREST: {"n_estimators": 999_999}
            }
        )


def test_hyperparameters_for_reports_effective_values() -> None:
    """Declared defaults with configured overrides applied."""
    config = MLConfig(
        family_hyperparameters={ModelFamily.RANDOM_FOREST: {"max_depth": 4}}
    )
    effective = config.hyperparameters_for(ModelFamily.RANDOM_FOREST)
    assert effective["max_depth"] == 4
    assert effective["n_estimators"] == 300


def test_hyperparameters_for_an_unenabled_family_raises() -> None:
    """Asking about a family that will not be fitted is a caller error."""
    config = MLConfig(
        enabled_model_families=(
            ModelFamily.PRIOR_BASELINE,
            ModelFamily.LOGISTIC_REGRESSION,
        ),
        category=CategoryConfig(enabled=False),
        anomaly=AnomalyConfig(enabled=False),
    )
    with pytest.raises(MLConfigurationError, match="not enabled"):
        config.hyperparameters_for(ModelFamily.RANDOM_FOREST)


# ---------------------------------------------------------------------------
# Gate coherence and ordering
# ---------------------------------------------------------------------------


def test_a_gate_stricter_than_the_threshold_search_is_rejected() -> None:
    """No threshold chosen on validation could ever clear such a gate."""
    with pytest.raises(ValidationError, match="stricter than the ceiling"):
        MLConfig(
            thresholds=ThresholdConfig(max_false_positive_rate=0.10),
            gates={"max_false_positive_rate": 0.01},  # type: ignore[arg-type]
        )


def test_a_calibration_gate_stricter_than_the_calibration_stage_is_rejected() -> None:
    """The same unsatisfiable shape, applied to calibration error."""
    with pytest.raises(ValidationError, match="calibration ceiling is stricter"):
        MLConfig(
            calibration=CalibrationConfig(max_expected_calibration_error=0.20),
            gates={"max_expected_calibration_error": 0.05},  # type: ignore[arg-type]
        )


def test_unordered_drift_thresholds_are_rejected() -> None:
    """An alert threshold at or below the warning threshold inverts the ladder."""
    with pytest.raises(ValidationError, match="strictly below"):
        DriftConfig(psi_warn_threshold=0.30, psi_alert_threshold=0.25)


def test_serializer_parity_and_determinism_cannot_be_switched_off(
    tmp_path: Path,
) -> None:
    """A model that cannot be reloaded exactly is not one this project ships."""
    for field in ("require_serializer_parity", "require_determinism"):
        path = _write(tmp_path, f"gates:\n  {field}: false\n")
        with pytest.raises(ConfigurationError):
            load_ml_config(path)


def test_an_empty_fusion_strategy_list_is_rejected() -> None:
    """Enabling fusion with nothing to build is a contradiction."""
    with pytest.raises(ValidationError, match="at least one fusion strategy"):
        FusionConfig(strategies=())


def test_a_repeated_fusion_strategy_is_rejected() -> None:
    """A repeat would evaluate one strategy twice and report it twice."""
    with pytest.raises(ValidationError, match="repeats a strategy"):
        FusionConfig(strategies=(FusionStrategy.OR_GATE, FusionStrategy.OR_GATE))


@pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 2.0])
def test_an_out_of_range_false_positive_ceiling_is_rejected(value: float) -> None:
    """A ceiling of zero or one describes no achievable operating point."""
    with pytest.raises(ValidationError):
        ThresholdConfig(max_false_positive_rate=value)


@pytest.mark.parametrize("value", [-1, 0])
def test_a_non_positive_support_floor_is_rejected(value: int) -> None:
    """A floor of zero would let an empty class read as measured."""
    from password_attack_detector.ml.schemas import SupportRequirement

    with pytest.raises(ValidationError):
        SupportRequirement(min_validation_positive_rows=value)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_path_traversing_output_directory_is_rejected() -> None:
    """An output path must not escape the directory it was pointed at."""
    with pytest.raises(ValidationError, match=r"'\.\.'"):
        MLConfig(output_dir=Path("../escape"))


def test_a_secret_shaped_key_is_rejected_before_the_value_is_read(
    tmp_path: Path,
) -> None:
    """Modelling needs no credential, so a key shaped like one is a mistake."""
    path = _write(tmp_path, "api_key: hunter2\n")
    with pytest.raises(ConfigurationError, match="looks like a secret"):
        load_ml_config(path)


def test_a_secret_rejection_message_never_echoes_the_value(tmp_path: Path) -> None:
    """The scanner names the key path only; the value is never read."""
    path = _write(tmp_path, "nested:\n  access_token: super-secret-value\n")
    with pytest.raises(ConfigurationError) as excinfo:
        load_ml_config(path)
    assert "super-secret-value" not in str(excinfo.value)


def test_a_declared_hyperparameter_name_is_not_mistaken_for_a_secret(
    tmp_path: Path,
) -> None:
    """The controlled vocabulary exempts names the catalog already declares."""
    path = _write(
        tmp_path,
        "family_hyperparameters:\n  logistic_regression:\n    class_weight: balanced\n",
    )
    config = load_ml_config(path)
    assert (
        config.hyperparameters_for(ModelFamily.LOGISTIC_REGRESSION)["class_weight"]
        == "balanced"
    )


def test_a_missing_configuration_file_reports_no_path(tmp_path: Path) -> None:
    """An error message must not disclose a filesystem layout."""
    missing = tmp_path / "absent.yaml"
    with pytest.raises(ConfigurationError, match="Cannot read ML configuration"):
        load_ml_config(missing)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    """A parse failure must be a configuration error, not a traceback."""
    path = _write(tmp_path, "seed: [unclosed\n")
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        load_ml_config(path)


def test_a_non_mapping_configuration_is_rejected(tmp_path: Path) -> None:
    """A list at the top level is not a configuration."""
    path = _write(tmp_path, "- one\n- two\n")
    with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
        load_ml_config(path)


def test_an_empty_configuration_loads_the_declared_defaults(tmp_path: Path) -> None:
    """An empty file is a valid configuration: every field has a default."""
    path = _write(tmp_path, "")
    assert load_ml_config(path).fingerprint() == MLConfig().fingerprint()


def test_yaml_cannot_construct_a_python_object(tmp_path: Path) -> None:
    """Configuration is data. ``safe_load`` cannot materialise a callable."""
    path = _write(tmp_path, "seed: !!python/object/apply:os.system ['echo hi']\n")
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        load_ml_config(path)


# ---------------------------------------------------------------------------
# The tracked configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["model-development.yaml", "model-testing.yaml"])
def test_the_tracked_configurations_load(name: str) -> None:
    """Both shipped configurations must validate against the current contract."""
    config = load_ml_config(_config_path(name))
    assert len(config.fingerprint()) == 64
    assert ModelFamily.PRIOR_BASELINE in config.enabled_model_families


def test_the_development_configuration_enables_the_full_candidate_set() -> None:
    """Baselines, a linear candidate, an ensemble candidate, and the probe."""
    config = load_ml_config(_config_path("model-development.yaml"))
    assert set(config.enabled_model_families) == {
        ModelFamily.PRIOR_BASELINE,
        ModelFamily.SINGLE_FEATURE_THRESHOLD,
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.ISOLATION_FOREST,
    }


def test_the_gated_boosting_family_is_not_enabled_in_development() -> None:
    """Enabling it before its serializer is proven would only produce a reject."""
    config = load_ml_config(_config_path("model-development.yaml"))
    assert ModelFamily.HISTOGRAM_GRADIENT_BOOSTING not in config.enabled_model_families


def test_the_testing_configuration_is_ci_sized() -> None:
    """Small enough that a contract test never pays for a forest fit."""
    config = load_ml_config(_config_path("model-testing.yaml"))
    assert ModelFamily.RANDOM_FOREST not in config.enabled_model_families
    assert config.hyperparameters_for(ModelFamily.LOGISTIC_REGRESSION)["max_iter"] == 50
    assert config.explain.enabled is False
    assert config.drift.enabled is False
    assert config.anomaly.enabled is False


def test_the_testing_configuration_relaxes_no_discipline() -> None:
    """CI must not pass on a pipeline development would reject.

    Sizes differ between the two configurations; policy does not.
    """
    development = load_ml_config(_config_path("model-development.yaml"))
    testing = load_ml_config(_config_path("model-testing.yaml"))
    for config in (development, testing):
        assert config.preprocessing.add_missing_indicators is True
        assert config.preprocessing.statistics_source == "train"
        assert config.imbalance.resampling == "none"
        assert config.imbalance.computed_from == "train"
        assert config.calibration.source_partition is ValidationPartition.VALIDATION_A
        assert config.thresholds.source_partition is ValidationPartition.VALIDATION_B
        assert config.validation_partition.strategy == "campaign_group_boundary"
        assert config.anomaly.influences_champion_selection is False
        assert config.gates.require_serializer_parity is True
        assert config.gates.require_determinism is True
        assert config.gates.require_eligibility_audit_pass is True
        assert config.category.abstain_label == UNKNOWN_CATEGORY


def test_the_two_tracked_configurations_are_semantically_distinct() -> None:
    """They differ in sizing, so their fingerprints must differ."""
    development = load_ml_config(_config_path("model-development.yaml"))
    testing = load_ml_config(_config_path("model-testing.yaml"))
    assert development.fingerprint() != testing.fingerprint()


def test_the_tracked_configurations_declare_the_bounded_dependency_range() -> None:
    """Every run records the reviewed range it was configured under."""
    from password_attack_detector.ml.dependencies import SKLEARN_REQUIREMENT

    for name in ("model-development.yaml", "model-testing.yaml"):
        config = load_ml_config(_config_path(name))
        requirement = config.requirement_for("scikit-learn")
        assert requirement is not None
        assert requirement.specifier == SKLEARN_REQUIREMENT.specifier


def test_no_tracked_configuration_names_a_module_path() -> None:
    """Configuration is data: nothing in it may look like an import."""
    for name in ("model-development.yaml", "model-testing.yaml"):
        text = _config_path(name).read_text(encoding="utf-8")
        for forbidden in ("!!python", "sklearn.", "import ", "lambda", "eval("):
            assert forbidden not in text, (name, forbidden)


def test_the_development_configuration_uses_the_declared_objective() -> None:
    """The shipped objective is the false-positive-constrained one."""
    config = load_ml_config(_config_path("model-development.yaml"))
    assert config.thresholds.objective is ThresholdObjective.MAX_RECALL_AT_MAX_FPR
    assert config.calibration.method is CalibrationMethod.ISOTONIC


def test_the_development_configuration_builds_all_three_fusion_strategies() -> None:
    """None is assumed to win, so all three must actually be constructed."""
    config = load_ml_config(_config_path("model-development.yaml"))
    assert set(config.fusion.strategies) == set(FusionStrategy)


def test_no_model_is_trained_by_loading_a_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Milestone 1 fits nothing. Loading a configuration must stay inert.

    Guards against a future edit that eagerly constructs an estimator during
    validation, which would make configuration loading expensive and would put
    a fit somewhere no test is watching.
    """
    calls: list[str] = []

    def _record(*_args: Any, **_kwargs: Any) -> None:
        calls.append("fit")

    from sklearn.linear_model import LogisticRegression

    monkeypatch.setattr(LogisticRegression, "fit", _record)
    load_ml_config(_config_path("model-development.yaml"))
    assert calls == []
