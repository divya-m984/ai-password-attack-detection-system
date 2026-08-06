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

from password_attack_detector.detection.alerts import (
    ALERTING_VERSION,
    AlertBuilder,
    AlertingResult,
    EntityScopeTable,
    build_entity_scope_table,
)
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
    AlertingStats,
    EntityScopeRecord,
    EvidenceItem,
    FiredDetection,
    RiskAssessment,
    RuleEvaluationResult,
    SecurityAlert,
)
from password_attack_detector.detection.scoring import (
    SCORING_VERSION,
    RiskScorer,
    ScoringResult,
)

__all__ = [
    "ALERTING_VERSION",
    "DETECTION_SCHEMA_VERSION",
    "RULE_CATALOG",
    "RULE_CATALOG_VERSION",
    "SCORING_VERSION",
    "AlertBuilder",
    "AlertGroupingMode",
    "AlertingResult",
    "AlertingStats",
    "AttackCategory",
    "CorrelationGroup",
    "DetectionConfig",
    "DetectionEngine",
    "EngineResult",
    "EngineStats",
    "EntityScopeRecord",
    "EntityScopeTable",
    "EvidenceItem",
    "FiredDetection",
    "RiskAssessment",
    "RiskScorer",
    "RuleCatalog",
    "RuleEvaluationResult",
    "RuleFamily",
    "RuleSpec",
    "RuleStatus",
    "ScopeKind",
    "ScoringResult",
    "SecurityAlert",
    "Severity",
    "build_entity_scope_table",
    "catalog_to_markdown",
    "evaluate_snapshots",
    "load_detection_config",
]
