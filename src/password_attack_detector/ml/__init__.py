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

Milestone 1 establishes the foundation: the dependency policy, the typed
enumerations and contracts, the versioned configuration, and the executable
model catalog.  Dataset assembly, preprocessing, fitting, calibration,
threshold selection, inference, fusion, and evaluation arrive in later
milestones and are deliberately absent here.
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
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    CalibrationMethod,
    ChampionStatus,
    ExperimentRecordType,
    FusionStrategy,
    GateStatus,
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
    ScoreKind,
    ThresholdObjective,
    ValidationPartition,
    is_probability,
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

__all__ = [
    "FORBIDDEN_DIRECT_IMPORTS",
    "ML_DEPENDENCY_REQUIREMENTS",
    "ML_FINGERPRINT_EXCLUDED_FIELDS",
    "ML_SCHEMA_VERSION",
    "MODEL_CATALOG",
    "MODEL_CATALOG_VERSION",
    "SKLEARN_REQUIREMENT",
    "UNKNOWN_CATEGORY",
    "ArtifactDeclaration",
    "CalibrationMethod",
    "ChampionStatus",
    "DependencyRequirement",
    "ExperimentRecordIdentity",
    "ExperimentRecordType",
    "FusionStrategy",
    "GateResult",
    "GateStatus",
    "HyperparameterSpec",
    "MLConfig",
    "MLTask",
    "ModelCatalog",
    "ModelEligibilityStatus",
    "ModelFamily",
    "ModelSpec",
    "ScoreKind",
    "ScoreSemantics",
    "SupportRequirement",
    "ThresholdObjective",
    "ValidationPartition",
    "build_model_catalog",
    "collect_dependency_versions",
    "is_probability",
    "load_ml_config",
    "model_catalog_to_markdown",
    "sklearn_compatible",
]
