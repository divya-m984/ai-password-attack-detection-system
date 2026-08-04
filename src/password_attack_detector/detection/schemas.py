"""Typed, versioned contracts for every detection artifact.

Each model is frozen and forbids extra fields, so a stray key is a validation
error rather than a silently carried column.  Field declaration order is the
authoritative column order for the Parquet tables published in a later
milestone; nothing downstream may reorder it.

Three prohibitions are enforced here rather than left to convention:

* **No ground truth, split assignment, or model output.**  ``PROHIBITED_FIELDS``
  lists the names, and a test asserts no model in this module declares one.
* **No identifiers, coordinates, or secrets inside evidence.**
  :class:`EvidenceItem` rejects a value shaped like a pseudonym or a UUID.  The
  anchor identifier is a *key* on the result models and is deliberately not
  part of any evidence payload.
* **No causal claims.**  Evidence messages are rejected when they contain
  proof-asserting language.  Rules describe behaviour that is *consistent with*
  a pattern; they never confirm a compromise.

:class:`EntityScopeRecord` is the one model carrying pseudonymous operational
metadata.  Its ``__repr__`` redacts every field, so a scope value cannot reach
a log line, traceback, or CLI summary by accident.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from password_attack_detector.detection.enums import (
    SEVERITY_ORDER,
    AlertGroupingMode,
    AttackCategory,
    CorrelationGroup,
    EvidenceComparator,
    PrivacyClass,
    RuleFamily,
    RuleStatus,
    ScopeKind,
    Severity,
    SuppressionReason,
)

__all__ = [
    "DETECTION_SCHEMA_VERSION",
    "PROHIBITED_CLAIM_TERMS",
    "PROHIBITED_FIELDS",
    "AlertingStats",
    "DetectionValidationFinding",
    "DetectionValidationResult",
    "DetectionValidationStatus",
    "EntityScopeRecord",
    "EvidenceItem",
    "FiredDetection",
    "RiskAssessment",
    "RuleEvaluationResult",
    "SecurityAlert",
    "detection_identifier",
]

#: Declared ``Final`` without an explicit annotation so its type narrows to
#: ``Literal["1.0.0"]``.  That lets the models below default their
#: ``detection_schema_version`` field from this constant instead of repeating
#: the literal, keeping one source of truth for the version.
DETECTION_SCHEMA_VERSION: Final = "1.0.0"

#: Namespace for deterministic detection identifiers.  UUIDv5 over semantic
#: inputs only -- never uuid4, wall-clock time, a process id, or ``hash()``.
_NS_DETECTION = uuid5(NAMESPACE_URL, "password-attack-detector/detection")

#: Field names that must never appear on a detection artifact: ground truth,
#: campaign metadata, split assignment, and model output.
PROHIBITED_FIELDS: frozenset[str] = frozenset(
    {
        "label",
        "target",
        "attack_label",
        "attack_class",
        "scenario",
        "scenario_variant",
        "campaign_id",
        "campaign_stage",
        "malicious",
        "is_attack",
        "supervised_training_eligible",
        "split",
        "exclusion_reason",
        "model_probability",
        "attack_probability",
        "prediction",
        "predicted_class",
        "feature_importance",
    }
)

#: Language that asserts proof or causality.  Evidence describes what was
#: observed and what it is consistent with; it never confirms an attack.
PROHIBITED_CLAIM_TERMS: frozenset[str] = frozenset(
    {
        "proof",
        "proves",
        "proven",
        "confirm",
        "confirms",
        "confirmed",
        "guarantee",
        "guarantees",
        "guaranteed",
        "certainly",
        "definitely",
        "definitive",
        "undeniable",
        "conclusive",
        "probability",
        "likelihood",
    }
)

#: Pseudonymous identifiers minted by the Phase 2 pseudonym service and by the
#: synthetic generator.  Rejected anywhere inside an evidence payload.
_PSEUDONYM_RE = re.compile(r"^(u|s|d|sess):[0-9a-f]{32}$")

#: Canonical UUID text form, used for event and campaign identifiers.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

#: Evidence and reason codes are machine-readable constants.
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Rule identifiers are stable once published.
_RULE_ID_RE = re.compile(r"^PAD-[A-Z]{2,4}-\d{3}$")

#: Semantic version of a rule or of the scoring/alerting contract.
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

_REDACTED = "<redacted>"

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
RiskScore = Annotated[float, Field(ge=0.0, le=100.0)]


def _require_finite(value: float, field_name: str) -> float:
    """Return *value* when it is a finite float, else raise ``ValueError``."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{field_name} must be finite; got a NaN or infinity")
    return value


def _reject_identifier_text(value: object, field_name: str) -> None:
    """Reject a value that looks like a pseudonym or a UUID."""
    if not isinstance(value, str):
        return
    candidate = value.strip()
    if _PSEUDONYM_RE.match(candidate) or _UUID_RE.match(candidate):
        raise ValueError(
            f"{field_name} looks like an identifier; evidence carries only "
            f"sanitized behavioral values"
        )


def _reject_claim_language(text: str, field_name: str) -> None:
    """Reject proof-asserting or probability-asserting wording."""
    tokens = {token.strip(".,;:!?()[]'\"") for token in text.lower().split()}
    offending = sorted(tokens & PROHIBITED_CLAIM_TERMS)
    if offending:
        raise ValueError(
            f"{field_name} uses claim-asserting term(s) {offending}; describe "
            f"what was observed and what it is consistent with"
        )


def detection_identifier(anchor_event_id: str, rule_id: str, rule_version: str) -> str:
    """Return the deterministic identifier for one fired detection.

    Derived with UUIDv5 from the anchor, rule, and rule version alone, so the
    same detection recomputed on another machine, in another process, or on
    another day carries the same identifier.
    """
    return str(uuid5(_NS_DETECTION, f"{anchor_event_id}|{rule_id}|{rule_version}"))


class EvidenceItem(BaseModel):
    """One structured reason a rule condition matched.

    Observed values are sanitized behavioral quantities: counts, rates,
    durations, rounded distances, capped velocities, novelty booleans, and enum
    context values.  Identifiers, coordinates, credentials, and free text from
    any source outside this package are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_code: str
    feature_name: str
    comparator: EvidenceComparator
    observed_value: bool | int | float | str
    threshold_value: bool | int | float | str | None = None
    unit: str | None = None
    message: str

    @field_validator("evidence_code")
    @classmethod
    def check_code(cls, value: str) -> str:
        """Evidence codes are upper-case machine-readable constants."""
        if not _CODE_RE.match(value):
            raise ValueError(f"evidence_code {value!r} must match {_CODE_RE.pattern!r}")
        return value

    @field_validator("observed_value", "threshold_value")
    @classmethod
    def check_value(cls, value: object) -> object:
        """Reject non-finite numbers and identifier-shaped strings."""
        if isinstance(value, float):
            return _require_finite(value, "evidence value")
        _reject_identifier_text(value, "evidence value")
        return value

    @field_validator("message")
    @classmethod
    def check_message(cls, value: str) -> str:
        """Reject empty messages and proof-asserting language."""
        if not value.strip():
            raise ValueError("evidence message must not be empty")
        _reject_claim_language(value, "evidence message")
        _reject_identifier_text(value, "evidence message")
        return value


class RuleEvaluationResult(BaseModel):
    """Outcome of evaluating one rule against one feature snapshot.

    The status/payload consistency rules are enforced rather than documented:
    a fired result must carry evidence, and a non-fired or insufficient-data
    result must not, so no caller can publish an unexplained detection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    rule_version: str
    anchor_event_id: str
    anchor_event_time: AwareDatetime
    status: RuleStatus
    rule_family: RuleFamily
    attack_category: AttackCategory
    correlation_group: CorrelationGroup
    severity: Severity
    signal_strength: UnitInterval = 0.0
    evidence: tuple[EvidenceItem, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @field_validator("rule_id")
    @classmethod
    def check_rule_id(cls, value: str) -> str:
        """Rule identifiers follow the published ``PAD-XX-NNN`` form."""
        if not _RULE_ID_RE.match(value):
            raise ValueError(f"rule_id {value!r} must match {_RULE_ID_RE.pattern!r}")
        return value

    @field_validator("rule_version")
    @classmethod
    def check_rule_version(cls, value: str) -> str:
        """Rule versions are semantic versions."""
        if not _SEMVER_RE.match(value):
            raise ValueError(f"rule_version {value!r} must be a semantic version")
        return value

    @field_validator("anchor_event_id")
    @classmethod
    def check_anchor(cls, value: str) -> str:
        """The anchor identifier is a join key and must be present."""
        if not value.strip():
            raise ValueError("anchor_event_id must not be empty")
        return value

    @field_validator("anchor_event_time")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        """Normalise the anchor timestamp to UTC."""
        return value.astimezone(UTC)

    @field_validator("signal_strength")
    @classmethod
    def check_signal(cls, value: float) -> float:
        """Signal strength must be a finite number in ``[0, 1]``."""
        return _require_finite(value, "signal_strength")

    @field_validator("reason_codes")
    @classmethod
    def check_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reason codes are upper-case constants and are not repeated."""
        for code in value:
            if not _CODE_RE.match(code):
                raise ValueError(
                    f"reason code {code!r} must match {_CODE_RE.pattern!r}"
                )
        if len(set(value)) != len(value):
            raise ValueError("reason_codes must not repeat a code")
        return value

    @model_validator(mode="after")
    def check_status_consistency(self) -> Self:
        """Enforce the payload each status is allowed to carry."""
        if self.status is RuleStatus.FIRED:
            if not self.evidence:
                raise ValueError("a fired result must carry at least one evidence item")
            if not self.reason_codes:
                raise ValueError("a fired result must carry at least one reason code")
            if self.signal_strength <= 0.0:
                raise ValueError("a fired result must have signal_strength > 0")
            codes = {item.evidence_code for item in self.evidence}
            if len(codes) != len(self.evidence):
                raise ValueError("evidence must not repeat an evidence_code")
            return self

        if self.evidence:
            raise ValueError(f"a {self.status} result must not carry evidence")
        if self.signal_strength != 0.0:
            raise ValueError(f"a {self.status} result must have signal_strength == 0")
        if self.status is RuleStatus.INSUFFICIENT_DATA and not self.reason_codes:
            raise ValueError(
                "an insufficient_data result must name why the history was unavailable"
            )
        if self.status is RuleStatus.DISABLED and self.reason_codes:
            raise ValueError("a disabled result must not carry reason codes")
        return self

    @property
    def fired(self) -> bool:
        """Return whether this rule fired."""
        return self.status is RuleStatus.FIRED


class FiredDetection(BaseModel):
    """One fired detection, as published in the rule-detection table.

    Only fired rules are persisted here.  Non-fired, insufficient-data, and
    disabled outcomes stay in the aggregate diagnostics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_schema_version: Literal["1.0.0"] = DETECTION_SCHEMA_VERSION
    detection_id: str
    anchor_event_id: str
    anchor_event_time: AwareDatetime
    rule_id: str
    rule_version: str
    rule_family: RuleFamily
    attack_category: AttackCategory
    severity: Severity
    signal_strength: UnitInterval
    correlation_group: CorrelationGroup
    evidence: tuple[EvidenceItem, ...]
    reason_codes: tuple[str, ...]

    @field_validator("anchor_event_time")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        """Normalise the anchor timestamp to UTC."""
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def check_payload(self) -> Self:
        """A published detection must be explained and deterministically keyed."""
        if not self.evidence:
            raise ValueError("a published detection must carry evidence")
        if not self.reason_codes:
            raise ValueError("a published detection must carry reason codes")
        expected = detection_identifier(
            self.anchor_event_id, self.rule_id, self.rule_version
        )
        if self.detection_id != expected:
            raise ValueError(
                "detection_id is not the deterministic identifier for this "
                "anchor, rule, and rule version"
            )
        return self

    @classmethod
    def from_result(cls, result: RuleEvaluationResult) -> FiredDetection:
        """Build a publishable detection from a fired evaluation result.

        Raises:
            ValueError: if *result* did not fire.
        """
        if result.status is not RuleStatus.FIRED:
            raise ValueError(
                f"only a fired result can be published; got {result.status}"
            )
        return cls(
            detection_id=detection_identifier(
                result.anchor_event_id, result.rule_id, result.rule_version
            ),
            anchor_event_id=result.anchor_event_id,
            anchor_event_time=result.anchor_event_time,
            rule_id=result.rule_id,
            rule_version=result.rule_version,
            rule_family=result.rule_family,
            attack_category=result.attack_category,
            severity=result.severity,
            signal_strength=result.signal_strength,
            correlation_group=result.correlation_group,
            evidence=result.evidence,
            reason_codes=result.reason_codes,
        )


class RiskAssessment(BaseModel):
    """Event-level risk for one anchor.

    ``risk_score`` is a bounded ordinal severity magnitude on a 0-100 scale.
    It is **not** a probability and must never be described as one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_schema_version: Literal["1.0.0"] = DETECTION_SCHEMA_VERSION
    anchor_event_id: str
    anchor_event_time: AwareDatetime
    risk_score: RiskScore
    severity: Severity
    primary_attack_category: AttackCategory | None = None
    contributing_categories: tuple[AttackCategory, ...] = ()
    fired_rule_count: int = Field(default=0, ge=0)
    fired_rule_ids: tuple[str, ...] = ()
    top_evidence: tuple[EvidenceItem, ...] = ()
    insufficient_data_count: int = Field(default=0, ge=0)
    scoring_version: str

    @field_validator("anchor_event_time")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        """Normalise the anchor timestamp to UTC."""
        return value.astimezone(UTC)

    @field_validator("risk_score")
    @classmethod
    def check_score(cls, value: float) -> float:
        """Risk score must be a finite number in ``[0, 100]``."""
        return _require_finite(value, "risk_score")

    @field_validator("scoring_version")
    @classmethod
    def check_scoring_version(cls, value: str) -> str:
        """The scoring contract is versioned semantically."""
        if not _SEMVER_RE.match(value):
            raise ValueError(f"scoring_version {value!r} must be a semantic version")
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Tie the score, the fired-rule set, and the primary category together."""
        if len(set(self.fired_rule_ids)) != len(self.fired_rule_ids):
            raise ValueError("fired_rule_ids must not repeat a rule")
        if self.fired_rule_count != len(self.fired_rule_ids):
            raise ValueError("fired_rule_count must match the number of fired rules")
        if tuple(sorted(self.fired_rule_ids)) != self.fired_rule_ids:
            raise ValueError("fired_rule_ids must be sorted for deterministic output")

        if self.fired_rule_count == 0:
            if self.risk_score != 0.0:
                raise ValueError("an assessment with no fired rules must score 0.0")
            if self.primary_attack_category is not None:
                raise ValueError(
                    "an assessment with no fired rules has no primary category"
                )
            if self.contributing_categories or self.top_evidence:
                raise ValueError(
                    "an assessment with no fired rules carries no categories or "
                    "evidence"
                )
            return self

        if self.risk_score <= 0.0:
            raise ValueError("an assessment with a fired rule must score above 0.0")
        if self.primary_attack_category is None:
            raise ValueError("an assessment with a fired rule needs a primary category")
        if self.primary_attack_category not in self.contributing_categories:
            raise ValueError(
                "primary_attack_category must appear in contributing_categories"
            )
        if tuple(sorted(self.contributing_categories)) != self.contributing_categories:
            raise ValueError("contributing_categories must be sorted")
        return self


class SecurityAlert(BaseModel):
    """A grouped, deduplicated alert over one or more risk assessments.

    ``scope_value`` is the single field in the whole detection layer that may
    carry a pseudonym.  It is present only under entity-scoped grouping, is
    classified :attr:`PrivacyClass.OPERATIONAL_METADATA`, and is excluded from
    every report, CLI summary, manifest, and validation message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Sensitivity of ``scope_value``.  Read by the publisher and the docs.
    scope_privacy_class: PrivacyClass = PrivacyClass.OPERATIONAL_METADATA

    detection_schema_version: Literal["1.0.0"] = DETECTION_SCHEMA_VERSION
    alert_id: str
    attack_category: AttackCategory
    correlation_group: CorrelationGroup
    grouping_mode: AlertGroupingMode
    scope_kind: ScopeKind
    scope_value: str | None = None
    first_seen: AwareDatetime
    last_seen: AwareDatetime
    contributing_event_count: int = Field(ge=1)
    contributing_rule_ids: tuple[str, ...]
    aggregate_risk_score: RiskScore
    peak_risk_score: RiskScore
    initial_severity: Severity
    current_severity: Severity
    escalation_count: int = Field(default=0, ge=0)
    suppressed_event_count: int = Field(default=0, ge=0)

    @field_validator("first_seen", "last_seen")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        """Normalise alert timestamps to UTC."""
        return value.astimezone(UTC)

    @field_validator("aggregate_risk_score", "peak_risk_score")
    @classmethod
    def check_score(cls, value: float) -> float:
        """Alert risk scores must be finite."""
        return _require_finite(value, "alert risk score")

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Enforce the alert's temporal, scope, and severity invariants."""
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not precede first_seen")
        if not self.contributing_rule_ids:
            raise ValueError("an alert must name at least one contributing rule")
        if tuple(sorted(self.contributing_rule_ids)) != self.contributing_rule_ids:
            raise ValueError("contributing_rule_ids must be sorted")
        if len(set(self.contributing_rule_ids)) != len(self.contributing_rule_ids):
            raise ValueError("contributing_rule_ids must not repeat a rule")
        if self.peak_risk_score < self.aggregate_risk_score:
            raise ValueError("peak_risk_score must not be below aggregate_risk_score")

        if self.scope_value is not None:
            if self.grouping_mode is not AlertGroupingMode.ENTITY_SCOPED:
                raise ValueError(
                    "a scope value is only carried under entity-scoped grouping"
                )
            if self.scope_kind is ScopeKind.NONE:
                raise ValueError("a scope value requires a user or source scope kind")
        elif self.scope_kind is not ScopeKind.NONE:
            raise ValueError("a user or source scope kind requires a scope value")

        if (
            SEVERITY_ORDER[self.current_severity]
            < SEVERITY_ORDER[self.initial_severity]
        ):
            raise ValueError("current_severity must not fall below initial_severity")
        if self.escalation_count > 0 and self.current_severity == self.initial_severity:
            raise ValueError("an escalated alert must record a raised severity")
        return self


class AlertingStats(BaseModel):
    """Aggregate accounting for one alert-construction pass.

    Every qualifying detection is either grouped into an alert or recorded in
    one of the suppression tallies, so no alert can disappear silently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grouping_mode: AlertGroupingMode
    alert_count: int = Field(default=0, ge=0)
    grouped_detection_count: int = Field(default=0, ge=0)
    escalated_count: int = Field(default=0, ge=0)
    scope_missing_count: int = Field(default=0, ge=0)
    low_alert_reachable: bool = True
    suppressed_by_reason: dict[SuppressionReason, int] = Field(default_factory=dict)
    suppressed_by_category: dict[AttackCategory, int] = Field(default_factory=dict)

    @field_validator("suppressed_by_reason", "suppressed_by_category")
    @classmethod
    def check_counts(cls, value: dict[Any, int]) -> dict[Any, int]:
        """Suppression tallies are non-negative."""
        if any(count < 0 for count in value.values()):
            raise ValueError("suppression counts must not be negative")
        return value

    @property
    def suppressed_total(self) -> int:
        """Return the total number of suppressed detections."""
        return sum(self.suppressed_by_reason.values())


class EntityScopeRecord(BaseModel):
    """Optional pseudonymous grouping scope for one anchor event.

    Keyed one-to-one by ``anchor_event_id`` -- the same key the feature
    snapshot, detection, and risk-assessment tables use.  ``user_scope`` and
    ``source_scope`` are carried and consumed independently.

    This table is supplied only to alert construction.  The detection engine
    and the risk scorer accept no scope argument at all, so the restriction is
    a type error rather than a convention.

    ``__repr__`` redacts every field: these are pseudonymous operational
    identifiers, and a repr reaches logs and tracebacks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Sensitivity of this record.  Read by the publisher and the docs.
    privacy_class: PrivacyClass = PrivacyClass.OPERATIONAL_METADATA

    anchor_event_id: str
    user_scope: str | None = None
    source_scope: str | None = None

    @field_validator("anchor_event_id")
    @classmethod
    def check_anchor(cls, value: str) -> str:
        """The anchor identifier is the sole join key and must be present."""
        if not value.strip():
            raise ValueError("anchor_event_id must not be empty")
        return value

    def __repr__(self) -> str:
        """Return a redacted representation.

        Every field of this model is a pseudonymous identifier, so none of them
        may appear in a log line, a traceback, or a CLI summary.
        """
        return f"EntityScopeRecord({_REDACTED})"

    __str__ = __repr__


class DetectionValidationStatus(StrEnum):
    """Overall verdict on a set of detection artifacts."""

    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


class DetectionValidationFinding(BaseModel):
    """One validation finding.  Carries counts and column names, never rows.

    A validator that printed the offending row would turn every failed run log
    into a data disclosure, so findings are deliberately aggregate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    column: str | None = None
    count: int = Field(default=0, ge=0)

    @field_validator("code")
    @classmethod
    def check_code(cls, value: str) -> str:
        """Validation codes are ``D`` followed by three digits."""
        if not re.match(r"^D\d{3}$", value):
            raise ValueError(f"validation code {value!r} must match 'D<nnn>'")
        return value

    @field_validator("message")
    @classmethod
    def check_message(cls, value: str) -> str:
        """Findings must not name an identifier."""
        _reject_identifier_text(value, "validation message")
        return value


class DetectionValidationResult(BaseModel):
    """Structured outcome of validating a set of detection artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DetectionValidationStatus
    detection_schema_version: str | None = None
    detection_row_count: int = Field(default=0, ge=0)
    risk_assessment_row_count: int = Field(default=0, ge=0)
    alert_row_count: int = Field(default=0, ge=0)
    errors: tuple[DetectionValidationFinding, ...] = ()
    warnings: tuple[DetectionValidationFinding, ...] = ()

    @model_validator(mode="after")
    def check_status(self) -> Self:
        """The status must follow from the findings."""
        if self.errors and self.status is not DetectionValidationStatus.INVALID:
            raise ValueError("a result with errors must be invalid")
        if (
            not self.errors
            and self.warnings
            and self.status is not DetectionValidationStatus.WARNING
        ):
            raise ValueError("a result with only warnings must be a warning")
        return self

    @property
    def passed(self) -> bool:
        """Return whether validation found no errors."""
        return self.status is not DetectionValidationStatus.INVALID
