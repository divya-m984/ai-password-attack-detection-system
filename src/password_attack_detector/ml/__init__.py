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
train-only preprocessing and class weighting.  Milestone 4 added the model
adapters, the authoritative JSON-and-array artifact, the deterministic archive,
the manifest, and fail-closed loading.  Milestone 5 added calibration fitted on
validation-A, operating points chosen on validation-B, and the point at which
the word "probability" becomes available.  Milestone 6 adds the orchestration
that composes all of it -- candidate enumeration, the three training tracks,
staged run publication, and the append-only experiment ledger.  Milestone 7
adds validation-only champion selection: support-aware gates, the mandatory
M-000 comparison, deterministic ranking, and the frozen ``champion.lock`` a
later test evaluation will be permitted to read.  Milestone 8 adds batch
inference under that frozen champion: a label-free inference input, binary and
category prediction artifacts, an experimental anomaly artifact kept apart from
them, deterministic Parquet, the ``PredictionManifest`` that identifies a
publication, prediction validation, and the aggregate quality profile.

**Prediction is not evaluation.**  Milestone 8 may score the test split and
never opens a test label: everything that could be tuned was frozen before it
ran, so a prediction changes nothing and no outcome-dependent number is
computable from what it publishes.  Test evaluation, fusion, explainability, and
drift arrive in later milestones and are deliberately absent here.

``dataset`` is re-exported deliberately sparingly.  It is the one module in this
layer permitted to read ground truth, and keeping its label types out of the
package's public surface means a caller who wants them has to import the module
by name -- which is exactly the moment the import-graph test notices.
"""

from __future__ import annotations

from password_attack_detector.ml.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    POSITIVE_CLASS,
    BinaryScoreSample,
    CalibrationOutcome,
    CalibrationReport,
    CalibrationState,
    IsotonicParameters,
    PlattParameters,
    ReliabilityBin,
    ScoreSampleSource,
    apply_calibration,
    diagnose_calibration_fit,
    evaluate_calibration_quality,
    fit_calibration,
    require_out_of_sample_evidence,
)
from password_attack_detector.ml.catalog import (
    MODEL_CATALOG,
    MODEL_CATALOG_VERSION,
    ModelCatalog,
    ModelSpec,
    build_model_catalog,
    model_catalog_to_markdown,
)
from password_attack_detector.ml.champion import (
    CHAMPION_LOCK_FILE,
    FREEZE_SCHEMA_VERSION,
    ChampionLock,
    FreezePublication,
    FrozenCategoryHead,
    build_champion_lock,
    freeze_champion,
    scope_key_for,
)
from password_attack_detector.ml.config import (
    ML_FINGERPRINT_EXCLUDED_FIELDS,
    ChampionSelectionConfig,
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
    AnomalyThresholdMethod,
    AuditCheckStatus,
    AuditStatus,
    CalibrationEvaluationKind,
    CalibrationMethod,
    CalibrationStatus,
    ChampionStatus,
    ExperimentRecordType,
    FeatureDecisionPoint,
    FusionStrategy,
    GateStatus,
    MetricStatus,
    MLSplit,
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
    ScoreKind,
    SelectionStatus,
    ThresholdObjective,
    TrainingRunStatus,
    ValidationPartition,
    ValidationPartitionStatus,
    is_probability,
)
from password_attack_detector.ml.experiments import (
    RunPublication,
    RunSummary,
    build_training_run_record,
    publish_training_run,
    reconcile,
    summarize,
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
from password_attack_detector.ml.gates import (
    BINARY_GATE_IDS,
    CATEGORY_GATE_IDS,
    GATE_SCHEMA_VERSION,
    BinaryGateInputs,
    CategoryGateInputs,
    GateEvidence,
    RateEvidence,
    evaluate_binary_gates,
    evaluate_category_gates,
    gate_config_fingerprint,
    wilson_interval,
)
from password_attack_detector.ml.imbalance import (
    BINARY_CLASS_ORDER,
    IMBALANCE_SCHEMA_VERSION,
    ClassSupport,
    ClassWeightState,
    compute_class_weights,
)
from password_attack_detector.ml.inference import InferenceModel, ModelCompatibility
from password_attack_detector.ml.ledger import (
    LEDGER_SCHEMA_VERSION,
    CandidateSelectionResult,
    ChampionFreezeRecord,
    ExperimentLedger,
    LedgerAppendResult,
    TrainingRunRecord,
    ValidationSelectionRecord,
)
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
from password_attack_detector.ml.prediction_manifest import (
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    PREDICTIONS_DIR,
    PredictionFile,
    PredictionLineage,
    PredictionManifest,
    PredictionScopeRole,
    prediction_content_fingerprint,
    scope_role_for,
)
from password_attack_detector.ml.prediction_publisher import (
    PredictionPublication,
    publish_predictions,
)
from password_attack_detector.ml.prediction_serialization import (
    ANOMALY_PREDICTION_SCHEMA,
    BINARY_PREDICTION_SCHEMA,
    CATEGORY_PREDICTION_SCHEMA,
    read_anomaly_scores,
    read_binary_predictions,
    read_category_predictions,
    write_anomaly_scores,
    write_binary_predictions,
    write_category_predictions,
)
from password_attack_detector.ml.prediction_validation import (
    PUBLISHED_CHECKS,
    STAGED_CHECKS,
    VALIDATION_SCHEMA_VERSION,
    MLValidationResult,
    ValidationCheck,
    validate_publication,
    validate_staged_predictions,
)
from password_attack_detector.ml.predictions import (
    ANOMALY_PREDICTION_COLUMNS,
    BINARY_PREDICTION_COLUMNS,
    CATEGORY_PREDICTION_COLUMNS,
    PREDICTION_SCHEMA_VERSION,
    PROHIBITED_PREDICTION_COLUMNS,
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
    ExperimentalAnomalyRun,
    FrozenCategoryModel,
    FrozenChampion,
    predict_anomaly,
    predict_binary,
    predict_category,
    verify_inference_feature_contract,
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
from password_attack_detector.ml.quality import (
    QUALITY_SCHEMA_VERSION,
    AnomalyDistribution,
    BinaryDistribution,
    CategoryDistribution,
    MLQualityReport,
    build_quality_report,
    quality_report_to_markdown,
)
from password_attack_detector.ml.ranking import (
    DISCRIMINATION_SCORE_KIND,
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
    RANKING_SCHEMA_VERSION,
    RankingEvidence,
    ScoreLevel,
    build_ranking_evidence,
    pr_auc,
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
from password_attack_detector.ml.selection import (
    SELECTION_SCHEMA_VERSION,
    CandidateEvidence,
    SelectionOutcome,
    SelectionPublication,
    champion_candidate_model_ids,
    load_candidate_evidence,
    publish_selection,
    select_binary_champion,
    select_category_head,
)
from password_attack_detector.ml.serialization import (
    MODEL_ARTIFACT_FILES,
    ModelDocument,
    build_model_document,
    model_id_for,
    write_model_directory,
)
from password_attack_detector.ml.thresholds import (
    ANOMALY_DECISION_PREDICATE,
    BINARY_DECISION_PREDICATE,
    CATEGORY_DECISION_PREDICATE,
    THRESHOLD_SCHEMA_VERSION,
    AnomalyScoreSample,
    AnomalyThresholdSelection,
    CategoryAbstentionSelection,
    CategoryClassSupport,
    CategoryScoreSample,
    ThresholdCurvePoint,
    ThresholdSelection,
    select_anomaly_threshold,
    select_binary_threshold,
    select_category_abstention,
)
from password_attack_detector.ml.training import (
    CandidateSpec,
    TrainingContext,
    TrainingRunOutcome,
    enumerate_candidates,
    train_all,
    train_candidate,
)

__all__ = [
    "ALLOWLIST_SCHEMA_VERSION",
    "ANOMALY_DECISION_PREDICATE",
    "ANOMALY_PREDICTION_COLUMNS",
    "ANOMALY_PREDICTION_SCHEMA",
    "BINARY_CLASS_ORDER",
    "BINARY_DECISION_PREDICATE",
    "BINARY_GATE_IDS",
    "BINARY_PREDICTION_COLUMNS",
    "BINARY_PREDICTION_SCHEMA",
    "CALIBRATION_SCHEMA_VERSION",
    "CATEGORY_DECISION_PREDICATE",
    "CATEGORY_GATE_IDS",
    "CATEGORY_PREDICTION_COLUMNS",
    "CATEGORY_PREDICTION_SCHEMA",
    "CHAMPION_LOCK_FILE",
    "CHECK_NAMES",
    "DISCRIMINATION_SCORE_KIND",
    "FIT_ELIGIBLE_SPLITS",
    "FORBIDDEN_DIRECT_IMPORTS",
    "FREEZE_SCHEMA_VERSION",
    "GATE_SCHEMA_VERSION",
    "IMBALANCE_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "ML_DEPENDENCY_REQUIREMENTS",
    "ML_FINGERPRINT_EXCLUDED_FIELDS",
    "ML_OUTPUT_COLUMNS",
    "ML_SCHEMA_VERSION",
    "MODEL_ARTIFACT_FILES",
    "MODEL_CATALOG",
    "MODEL_CATALOG_VERSION",
    "MODEL_IMPLEMENTATIONS",
    "POSITIVE_CLASS",
    "PREDICTIONS_DIR",
    "PREDICTION_MANIFEST_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "PREPROCESSING_SCHEMA_VERSION",
    "PROHIBITED_PREDICTION_COLUMNS",
    "PR_AUC_INTEGRATION",
    "PUBLISHABLE_FAMILIES",
    "PUBLISHED_CHECKS",
    "QUALITY_SCHEMA_VERSION",
    "RANKING_METRIC_NAME",
    "RANKING_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "SKLEARN_REQUIREMENT",
    "STAGED_CHECKS",
    "THRESHOLD_SCHEMA_VERSION",
    "UNKNOWN_CATEGORY",
    "VALIDATION_SCHEMA_VERSION",
    "AnomalyDistribution",
    "AnomalyScore",
    "AnomalyScoreSample",
    "AnomalyThresholdMethod",
    "AnomalyThresholdSelection",
    "ArtifactDeclaration",
    "AuditCheckStatus",
    "AuditStatus",
    "BinaryDistribution",
    "BinaryGateInputs",
    "BinaryPrediction",
    "BinaryScoreSample",
    "BooleanEncoding",
    "CalibrationEvaluationKind",
    "CalibrationMethod",
    "CalibrationOutcome",
    "CalibrationReport",
    "CalibrationState",
    "CalibrationStatus",
    "CandidateEvidence",
    "CandidateSelectionResult",
    "CandidateSpec",
    "CategoricalEncoding",
    "CategoryAbstentionSelection",
    "CategoryClassSupport",
    "CategoryDistribution",
    "CategoryGateInputs",
    "CategoryPrediction",
    "CategoryScoreSample",
    "ChampionFreezeRecord",
    "ChampionLock",
    "ChampionSelectionConfig",
    "ChampionStatus",
    "ClassSupport",
    "ClassWeightState",
    "DependencyRequirement",
    "EligibleFeatureList",
    "ExperimentLedger",
    "ExperimentRecordIdentity",
    "ExperimentRecordType",
    "ExperimentalAnomalyRun",
    "FeatureAdmission",
    "FeatureAllowlist",
    "FeatureDecisionPoint",
    "FeatureFrame",
    "FittedModel",
    "FittedPreprocessor",
    "FreezePublication",
    "FrozenCategoryHead",
    "FrozenCategoryModel",
    "FrozenChampion",
    "FusionStrategy",
    "GateEvidence",
    "GateResult",
    "GateStatus",
    "HyperparameterSpec",
    "InferenceModel",
    "IsotonicParameters",
    "LedgerAppendResult",
    "MLConfig",
    "MLEligibilityAuditResult",
    "MLEligibilityAuditor",
    "MLQualityReport",
    "MLSplit",
    "MLTask",
    "MLValidationResult",
    "MetricStatus",
    "ModelAdapter",
    "ModelCatalog",
    "ModelCompatibility",
    "ModelDocument",
    "ModelEligibilityStatus",
    "ModelFamily",
    "ModelManifest",
    "ModelSpec",
    "NumericImputation",
    "PlattParameters",
    "PredictionFile",
    "PredictionLineage",
    "PredictionManifest",
    "PredictionPublication",
    "PredictionScopeRole",
    "RankingEvidence",
    "RateEvidence",
    "ReliabilityBin",
    "RunPublication",
    "RunSummary",
    "ScalingStatistic",
    "ScoreKind",
    "ScoreLevel",
    "ScoreSampleSource",
    "ScoreSemantics",
    "SelectionOutcome",
    "SelectionPublication",
    "SelectionStatus",
    "SupportRequirement",
    "ThresholdCurvePoint",
    "ThresholdObjective",
    "ThresholdSelection",
    "TrainingBatch",
    "TrainingContext",
    "TrainingRunOutcome",
    "TrainingRunRecord",
    "TrainingRunStatus",
    "TransformedMatrix",
    "ValidationCheck",
    "ValidationPartition",
    "ValidationPartitionResult",
    "ValidationPartitionStatus",
    "ValidationSelectionRecord",
    "VerificationOutcome",
    "adapter_class_for",
    "apply_calibration",
    "array_digest",
    "assert_canonical",
    "build_champion_lock",
    "build_model_catalog",
    "build_model_document",
    "build_model_manifest",
    "build_quality_report",
    "build_ranking_evidence",
    "build_training_run_record",
    "canonicalize_rows",
    "champion_candidate_model_ids",
    "collect_dependency_versions",
    "compute_class_weights",
    "diagnose_calibration_fit",
    "enumerate_candidates",
    "evaluate_binary_gates",
    "evaluate_calibration_quality",
    "evaluate_category_gates",
    "fit_calibration",
    "fit_preprocessor",
    "freeze_champion",
    "gate_config_fingerprint",
    "is_canonical",
    "is_probability",
    "load_candidate_evidence",
    "load_feature_allowlist",
    "load_ml_config",
    "ml_audit_result_to_markdown",
    "model_catalog_to_markdown",
    "model_id_for",
    "partition_validation",
    "pr_auc",
    "predict_anomaly",
    "predict_binary",
    "predict_category",
    "prediction_content_fingerprint",
    "publish_predictions",
    "publish_selection",
    "publish_training_run",
    "quality_report_to_markdown",
    "read_anomaly_scores",
    "read_binary_predictions",
    "read_category_predictions",
    "read_npz_bytes",
    "reconcile",
    "require_out_of_sample_evidence",
    "resolve_eligible_features",
    "scope_key_for",
    "scope_role_for",
    "select_anomaly_threshold",
    "select_binary_champion",
    "select_binary_threshold",
    "select_category_abstention",
    "select_category_head",
    "sklearn_compatible",
    "summarize",
    "train_all",
    "train_candidate",
    "validate_publication",
    "validate_staged_predictions",
    "verify_inference_feature_contract",
    "verify_model_artifact",
    "wilson_interval",
    "write_anomaly_scores",
    "write_binary_predictions",
    "write_category_predictions",
    "write_model_directory",
    "write_npz_bytes",
]
