"""Domain enumerations for the rule-based detection layer.

Every enum uses :class:`~enum.StrEnum` so it serialises to its string value in
Pydantic models, Parquet columns, JSON reports, and fingerprints.  That keeps
the on-disk contract stable across a future member rename, exactly as
``data/enums.py`` does for the canonical event schema.

Two design points are load-bearing:

* :class:`Severity` has **four** members and no ``INFORMATIONAL`` level.
  ``LOW`` is an ordinary severity, not a suppressed one -- whether a ``LOW``
  assessment produces an alert is decided by two configured floors
  (``min_alert_risk_score`` and ``min_alert_severity``), never by the level
  itself.
* :class:`AttackCategory` deliberately omits the novel-anomaly holdout.  That
  scenario is an evaluation-only pool and must never become a rule-based
  category.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "PROHIBITED_CATEGORY_TERMS",
    "SEVERITY_ORDER",
    "AlertGroupingMode",
    "AttackCategory",
    "CorrelationGroup",
    "EvaluationScope",
    "EvidenceComparator",
    "ParameterKind",
    "PrivacyClass",
    "RuleFamily",
    "RuleStatus",
    "ScopeKind",
    "Severity",
    "SuppressionReason",
    "severity_at_least",
]


class RuleFamily(StrEnum):
    """Coarse grouping of rules, used for family weighting and reporting."""

    BRUTE_FORCE = "brute_force"
    SPRAYING = "spraying"
    STUFFING = "stuffing"
    ACCOUNT_COMPROMISE = "account_compromise"
    LOCATION = "location"
    AUTOMATION = "automation"


class AttackCategory(StrEnum):
    """The behaviour a fired rule describes.

    Names ending in ``_indicator`` or ``_anomaly`` say so on purpose: those
    rules report behaviour consistent with a pattern, not a confirmed
    compromise.  There is no ``novel_anomaly_holdout`` member -- that scenario
    is reserved for evaluation and is never produced by a rule.
    """

    BRUTE_FORCE = "brute_force"
    PASSWORD_SPRAYING = "password_spraying"
    CREDENTIAL_STUFFING = "credential_stuffing"
    DISTRIBUTED_BRUTE_FORCE = "distributed_brute_force"
    ACCOUNT_TAKEOVER_INDICATOR = "account_takeover_indicator"
    IMPOSSIBLE_TRAVEL_INDICATOR = "impossible_travel_indicator"
    BOT_ACTIVITY = "bot_activity"
    MFA_SEQUENCE_ANOMALY = "mfa_sequence_anomaly"


#: Terms that must not appear in an attack-category value.  ``blocked_account``
#: is listed because blocked-account activity is supporting evidence inside the
#: brute-force rules, never a category of its own.
PROHIBITED_CATEGORY_TERMS: frozenset[str] = frozenset(
    {"blocked_account", "novel_anomaly", "novel_anomaly_holdout"}
)


class CorrelationGroup(StrEnum):
    """Rules whose evidence describes the same underlying behaviour.

    Contributions are reduced *within* a group before groups are combined, so
    two rules that restate one behaviour cannot inflate a risk score.
    """

    CREDENTIAL_GUESSING_SINGLE_TARGET = "credential_guessing_single_target"
    SOURCE_FANOUT = "source_fanout"
    SESSION_ANOMALY = "session_anomaly"
    LOCATION_MOVEMENT = "location_movement"
    AUTOMATION_TIMING = "automation_timing"


class Severity(StrEnum):
    """Severity ladder.  Four levels, three configured score boundaries."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


#: Explicit rank for every severity.  Comparisons go through this mapping
#: rather than relying on enum declaration order, which is not a contract.
SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def severity_at_least(observed: Severity, floor: Severity) -> bool:
    """Return whether *observed* ranks at or above *floor*."""
    return SEVERITY_ORDER[observed] >= SEVERITY_ORDER[floor]


class RuleStatus(StrEnum):
    """Outcome of evaluating one rule against one feature snapshot.

    ``INSUFFICIENT_DATA`` is distinct from ``NOT_FIRED`` on purpose: a rule
    that could not see the history it needs has not observed a clean negative,
    and conflating the two would silently understate detection gaps.
    """

    FIRED = "fired"
    NOT_FIRED = "not_fired"
    INSUFFICIENT_DATA = "insufficient_data"
    DISABLED = "disabled"


class EvidenceComparator(StrEnum):
    """How an observed value was compared against a configured threshold."""

    GTE = "gte"
    GT = "gt"
    LTE = "lte"
    LT = "lt"
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class EvaluationScope(StrEnum):
    """What a rule evaluates over.

    Phase 4 has exactly one scope: a single Phase 3 feature snapshot.  A rule
    never reaches outside the anchor row it is handed.
    """

    ANCHOR_EVENT = "anchor_event"


class AlertGroupingMode(StrEnum):
    """Which grouping regime produced an alert set.

    Recorded on every alert so two runs under different regimes can never be
    mistaken for one another.
    """

    ENTITY_SCOPED = "entity_scoped"
    CATEGORY_SCOPED = "category_scoped"


class ScopeKind(StrEnum):
    """Which pseudonymous dimension an alert was grouped on."""

    USER = "user"
    SOURCE = "source"
    NONE = "none"


class SuppressionReason(StrEnum):
    """Why a qualifying detection produced no new alert.

    Every suppressed detection carries one of these so no alert can disappear
    without an aggregate accounting record.
    """

    COOLDOWN = "cooldown"
    RATE_LIMIT = "rate_limit"
    BELOW_SCORE_FLOOR = "below_score_floor"
    BELOW_SEVERITY_FLOOR = "below_severity_floor"


class PrivacyClass(StrEnum):
    """Sensitivity classification for a detection artifact field."""

    NON_SENSITIVE = "non_sensitive"
    OPERATIONAL_METADATA = "operational_metadata"


class ParameterKind(StrEnum):
    """Type of a declared rule threshold parameter.

    ``WINDOW`` is a duration *label* such as ``"5m"`` that must name one of the
    configured Phase 3 feature windows; it is validated as a duration string
    here and resolved against the feature catalog at rule-preparation time.
    """

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    WINDOW = "window"
