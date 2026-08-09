"""Machine-learning detection layer.

A statistical detection layer over the Phase 3 point-in-time feature snapshots,
kept **separate from the Phase 4 rule engine** rather than folded into it.  The
separation is the point: it lets rule-only, model-only, and hybrid detection be
measured against each other on identical frozen splits, instead of one quietly
absorbing the other.

What this layer is, and is not:

* It is **offline and defensive**.  Nothing here serves a model, exposes an
  endpoint, touches live authentication traffic, or handles a credential.
* It **does not replace the rule engine**.  A rule is a reviewed, explainable
  decision with declared thresholds; a model is a fitted one.  Both are
  reported, and neither is assumed to win.
* Its scores are **not probabilities until calibrated**.  A raw estimator
  output is an ordered decision score.  The word "probability" applies only
  after a calibrator has been fitted and its calibration error measured -- see
  :class:`~password_attack_detector.ml.enums.ScoreKind`.
* Phase 4's ordinal ``risk_score`` and this layer's calibrated probability are
  **separately typed and never combined arithmetically**.

Milestone 1 established the foundation: the dependency policy, the typed
enumerations and contracts, the versioned configuration, and the executable
model catalog.  Milestone 2 added the data contract: the reviewed opt-in feature
allowlist, canonical row ordering, dataset assembly, campaign-disjoint
validation partitioning, and the eligibility audit.  Milestone 3 added
train-only preprocessing and class weighting.  Milestone 4 adds the model
adapters, the authoritative JSON-and-array artifact, the deterministic archive,
the manifest, and fail-closed loading.  Calibration, threshold selection,
training orchestration, the experiment ledger, champion selection, prediction
publication, fusion, evaluation, explainability, and drift arrive in later
milestones and are deliberately absent here.

``dataset`` is re-exported deliberately sparingly.  It is the one module in this
layer permitted to read ground truth, and keeping its label types out of the
package's public surface means a caller who wants them has to import the module
by name -- which is exactly the moment the import-graph test notices.
"""

from __future__ import annotations

from password_attack_detector.ml.catalog import (
    MODEL_CATALOG,
    MODEL_CATALOG_VERSION,
    ModelCatalog,
    ModelSpec,
    build_model_catalog,
    model_catalog_to_markdown,
)
from password_attack_detector.ml.config import (
    ML_FINGERPRINT_EXCLUDED_FIELDS,
    MLConfig,
    load_ml_config,
)
from password_attack_detector.ml.dependencies import (
    FORBIDDEN_DIRECT_IMPORTS,
    ML_DEPENDENCY_REQUIREMENTS,
    SKLEARN_REQUIREMENT,
    collect_dependency_versions,
    sklearn_compatible,
)
from password_attack_detector.ml.eligibility import (
    CHECK_NAMES,
    MLEligibilityAuditor,
    MLEligibilityAuditResult,
    ml_audit_result_to_markdown,
)
from password_attack_detector.ml.enums import (
    FIT_ELIGIBLE_SPLITS,
    UNKNOWN_CATEGORY,
    AuditCheckStatus,
    AuditStatus,
    CalibrationMethod,
    ChampionStatus,
    ExperimentRecordType,
    FeatureDecisionPoint,
    FusionStrategy,
    GateStatus,
    MLSplit,
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
    ScoreKind,
    ThresholdObjective,
    ValidationPartition,
    ValidationPartitionStatus,
    is_probability,
)
from password_attack_detector.ml.features import (
    ALLOWLIST_SCHEMA_VERSION,
    ML_OUTPUT_COLUMNS,
    EligibleFeatureList,
    FeatureAdmission,
    FeatureAllowlist,
    load_feature_allowlist,
    resolve_eligible_features,
)
from password_attack_detector.ml.imbalance import (
    BINARY_CLASS_ORDER,
    IMBALANCE_SCHEMA_VERSION,
    ClassSupport,
    ClassWeightState,
    compute_class_weights,
)
from password_attack_detector.ml.inference import InferenceModel, ModelCompatibility
from password_attack_detector.ml.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ModelManifest,
    VerificationOutcome,
    build_model_manifest,
    verify_model_artifact,
)
from password_attack_detector.ml.models import (
    MODEL_IMPLEMENTATIONS,
    PUBLISHABLE_FAMILIES,
    FittedModel,
    ModelAdapter,
    TrainingBatch,
    adapter_class_for,
)
from password_attack_detector.ml.npz import (
    array_digest,
    read_npz_bytes,
    write_npz_bytes,
)
from password_attack_detector.ml.ordering import (
    assert_canonical,
    canonicalize_rows,
    is_canonical,
)
from password_attack_detector.ml.partition import (
    ValidationPartitionResult,
    partition_validation,
)
from password_attack_detector.ml.preprocessing import (
    PREPROCESSING_SCHEMA_VERSION,
    BooleanEncoding,
    CategoricalEncoding,
    FeatureFrame,
    FittedPreprocessor,
    NumericImputation,
    ScalingStatistic,
    TransformedMatrix,
    fit_preprocessor,
)
from password_attack_detector.ml.schemas import (
    ML_SCHEMA_VERSION,
    ArtifactDeclaration,
    DependencyRequirement,
    ExperimentRecordIdentity,
    GateResult,
    HyperparameterSpec,
    ScoreSemantics,
    SupportRequirement,
)
from password_attack_detector.ml.serialization import (
    MODEL_ARTIFACT_FILES,
    ModelDocument,
    build_model_document,
    model_id_for,
    write_model_directory,
)

__all__ = [
    "ALLOWLIST_SCHEMA_VERSION",
    "BINARY_CLASS_ORDER",
    "CHECK_NAMES",
    "FIT_ELIGIBLE_SPLITS",
    "FORBIDDEN_DIRECT_IMPORTS",
    "IMBALANCE_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "ML_DEPENDENCY_REQUIREMENTS",
    "ML_FINGERPRINT_EXCLUDED_FIELDS",
    "ML_OUTPUT_COLUMNS",
    "ML_SCHEMA_VERSION",
    "MODEL_ARTIFACT_FILES",
    "MODEL_CATALOG",
    "MODEL_CATALOG_VERSION",
    "MODEL_IMPLEMENTATIONS",
    "PREPROCESSING_SCHEMA_VERSION",
    "PUBLISHABLE_FAMILIES",
    "SKLEARN_REQUIREMENT",
    "UNKNOWN_CATEGORY",
    "ArtifactDeclaration",
    "AuditCheckStatus",
    "AuditStatus",
    "BooleanEncoding",
    "CalibrationMethod",
    "CategoricalEncoding",
    "ChampionStatus",
    "ClassSupport",
    "ClassWeightState",
    "DependencyRequirement",
    "EligibleFeatureList",
    "ExperimentRecordIdentity",
    "ExperimentRecordType",
    "FeatureAdmission",
    "FeatureAllowlist",
    "FeatureDecisionPoint",
    "FeatureFrame",
    "FittedModel",
    "FittedPreprocessor",
    "FusionStrategy",
    "GateResult",
    "GateStatus",
    "HyperparameterSpec",
    "InferenceModel",
    "MLConfig",
    "MLEligibilityAuditResult",
    "MLEligibilityAuditor",
    "MLSplit",
    "MLTask",
    "ModelAdapter",
    "ModelCatalog",
    "ModelCompatibility",
    "ModelDocument",
    "ModelEligibilityStatus",
    "ModelFamily",
    "ModelManifest",
    "ModelSpec",
    "NumericImputation",
    "ScalingStatistic",
    "ScoreKind",
    "ScoreSemantics",
    "SupportRequirement",
    "ThresholdObjective",
    "TrainingBatch",
    "TransformedMatrix",
    "ValidationPartition",
    "ValidationPartitionResult",
    "ValidationPartitionStatus",
    "VerificationOutcome",
    "adapter_class_for",
    "array_digest",
    "assert_canonical",
    "build_model_catalog",
    "build_model_document",
    "build_model_manifest",
    "canonicalize_rows",
    "collect_dependency_versions",
    "compute_class_weights",
    "fit_preprocessor",
    "is_canonical",
    "is_probability",
    "load_feature_allowlist",
    "load_ml_config",
    "ml_audit_result_to_markdown",
    "model_catalog_to_markdown",
    "model_id_for",
    "partition_validation",
    "read_npz_bytes",
    "resolve_eligible_features",
    "sklearn_compatible",
    "verify_model_artifact",
    "write_model_directory",
    "write_npz_bytes",
]
