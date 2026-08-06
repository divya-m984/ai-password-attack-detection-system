"""Rule-based authentication attack detection.

The detection layer consumes Phase 3 point-in-time feature snapshots and
produces explainable, deterministic findings.  It never reads ground-truth
labels, split assignments, campaign metadata, or the canonical event table --
Phase 3 is the point-in-time boundary, and evaluation against ground truth is a
separate workflow.

Nothing here trains, loads, or serves a machine-learning model.  ``risk_score``
and ``signal_strength`` are bounded ordinal magnitudes, not probabilities.
"""

from __future__ import annotations

from password_attack_detector.detection.catalog import (
    RULE_CATALOG,
    RULE_CATALOG_VERSION,
    RuleCatalog,
    RuleSpec,
    catalog_to_markdown,
)
from password_attack_detector.detection.config import (
    DetectionConfig,
    load_detection_config,
)
from password_attack_detector.detection.engine import (
    DetectionEngine,
    EngineResult,
    EngineStats,
    evaluate_snapshots,
)
from password_attack_detector.detection.enums import (
    AlertGroupingMode,
    AttackCategory,
    CorrelationGroup,
    RuleFamily,
    RuleStatus,
    ScopeKind,
    Severity,
)
from password_attack_detector.detection.schemas import (
    DETECTION_SCHEMA_VERSION,
    EvidenceItem,
    FiredDetection,
    RiskAssessment,
    RuleEvaluationResult,
    SecurityAlert,
)

__all__ = [
    "DETECTION_SCHEMA_VERSION",
    "RULE_CATALOG",
    "RULE_CATALOG_VERSION",
    "AlertGroupingMode",
    "AttackCategory",
    "CorrelationGroup",
    "DetectionConfig",
    "DetectionEngine",
    "EngineResult",
    "EngineStats",
    "EvidenceItem",
    "FiredDetection",
    "RiskAssessment",
    "RuleCatalog",
    "RuleEvaluationResult",
    "RuleFamily",
    "RuleSpec",
    "RuleStatus",
    "ScopeKind",
    "SecurityAlert",
    "Severity",
    "catalog_to_markdown",
    "evaluate_snapshots",
    "load_detection_config",
]
