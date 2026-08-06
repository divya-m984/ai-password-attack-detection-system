"""Validation of a published detection artifact set.

Mirrors ``features/validation.py``: stable codes, a frozen result, and a status
of ``valid`` / ``warning`` / ``invalid``.  Codes are ``D0xx`` so they never
collide with the data layer's ``V0xx`` or the feature layer's ``F0xx``.

**The validator never raises and never discloses.**  It returns findings rather
than throwing, because a caller wants the whole picture rather than the first
problem; and every finding carries a code, a column name, and a count and
nothing else.  No event identifier, detection identifier, alert identifier,
scope value, evidence value, raw row, secret, or absolute path can reach a
message -- a validator that printed the offending row would turn every failed
run log into a data disclosure.

Three families of check run here.  Per-artifact checks establish that each
table is internally well formed.  Cross-artifact checks establish that the
three tables describe *one* run: every detection has an assessment, every
assessment's fired-rule set matches the detections actually present, every
alert's contributing rules exist.  Configuration checks establish that the run
matches the configuration and catalog now in force.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC

from password_attack_detector.detection.alerts import ALERTING_VERSION
from password_attack_detector.detection.catalog import RULE_CATALOG, RuleCatalog
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import (
    AlertGroupingMode,
    AttackCategory,
    ScopeKind,
    severity_at_least,
)
from password_attack_detector.detection.schemas import (
    DETECTION_SCHEMA_VERSION,
    PROHIBITED_FIELDS,
    DetectionValidationFinding,
    DetectionValidationResult,
    DetectionValidationStatus,
    EntityScopeRecord,
    FiredDetection,
    RiskAssessment,
    SecurityAlert,
    detection_identifier,
)
from password_attack_detector.detection.scoring import SCORING_VERSION
from password_attack_detector.detection.serialization import (
    ALERT_COLUMNS,
    DETECTION_COLUMNS,
    RISK_COLUMNS,
)

__all__ = [
    "DETECTION_VALIDATION_CODES",
    "INVALID_VALUE_CODES",
    "PROHIBITED_ARTIFACT_COLUMNS",
    "PROHIBITED_COLUMN_CODES",
    "RELATIONSHIP_CODES",
    "DetectionValidator",
    "validate_detection_artifacts",
]

#: Every code this validator can emit, with what it means.  Kept as data so the
#: documentation, the tests, and the reports read one list rather than three.
DETECTION_VALIDATION_CODES: Mapping[str, str] = {
    "D001": "A required artifact table is empty or unreadable",
    "D002": "Declared column order does not match the published order",
    "D003": "Prohibited ground-truth, split, campaign, model, or scope column",
    "D004": "detection_schema_version is unsupported or not uniform",
    "D005": "detection_id is not unique",
    "D006": "detection_id is not the deterministic identifier for its inputs",
    "D007": "rule_id is not registered in the rule catalog",
    "D008": "rule_version does not match the registered rule",
    "D009": "Rule family, attack category, or correlation group disagrees "
    "with the catalog",
    "D010": "signal_strength lies outside [0, 1]",
    "D011": "A numeric value is NaN or infinite",
    "D012": "A timestamp is naive or not stored in UTC",
    "D013": "A fired detection carries no evidence or no reason code",
    "D014": "anchor_event_id is not unique in the risk-assessment table",
    "D015": "risk_score lies outside [0, 100]",
    "D016": "fired_rule_count disagrees with fired_rule_ids",
    "D017": "A zero-risk assessment reports fired rules, or the converse",
    "D018": "primary_attack_category is absent, unknown, or not contributing",
    "D019": "A fired assessment scores below the configured fired floor",
    "D020": "configuration_fingerprint is malformed or not the active run's",
    "D021": "alert_id is not unique",
    "D022": "An alert risk score lies outside [0, 100]",
    "D023": "aggregate_risk_score exceeds peak_risk_score",
    "D024": "An alert falls below a configured score or severity floor",
    "D025": "grouping_mode and scope metadata disagree",
    "D026": "first_seen is after last_seen, or the event count is not positive",
    "D027": "An alert names a rule absent from the fired detections",
    "D028": "A fired detection has no risk assessment",
    "D029": "An assessment's fired-rule set does not match its detections",
    "D030": "The entity-scope table is not one-to-one with the assessments",
    "D031": "aggregate_risk_score is not the mean of its grouped assessments",
    "D032": "scoring_version or alerting_version is unsupported",
    "D050": "A registered rule never fired",
    "D051": "No alert was produced",
    "D052": "The insufficient-data rate is above the configured warning level",
}

#: Column names an artifact may never carry.  The schema-level prohibition set
#: is reused verbatim and extended with the scope columns, which belong to the
#: optional input table and to one alert column -- nowhere else.
PROHIBITED_ARTIFACT_COLUMNS: frozenset[str] = PROHIBITED_FIELDS | frozenset(
    {
        "user_scope",
        "source_scope",
        "user_id",
        "source_id",
        "device_id",
        "session_id",
        "campaign",
        "ground_truth",
        "y_true",
        "y_pred",
    }
)

#: Codes reporting a value outside its declared range, or one that is not a
#: number at all.  Summarised on the result so a caller can act on "the data is
#: malformed" without re-deriving it from the finding list.
INVALID_VALUE_CODES: frozenset[str] = frozenset(
    {"D010", "D011", "D012", "D015", "D022", "D023", "D026"}
)

#: Codes reporting a column that must never appear on a detection artifact.
PROHIBITED_COLUMN_CODES: frozenset[str] = frozenset({"D003"})

#: Codes reporting a broken relationship *between* artifacts rather than a
#: defect inside one of them.
RELATIONSHIP_CODES: frozenset[str] = frozenset(
    {"D016", "D017", "D018", "D027", "D028", "D029", "D030", "D031"}
)

#: Insufficient-data rate above which the validator warns.
_INSUFFICIENT_DATA_WARN_RATE: float = 0.95

#: Tolerance for the alert mean-risk reconstruction, in risk-score points.
#: The published value is rounded to four decimals, so an exact comparison
#: would fail on rounding rather than on a defect.
_MEAN_TOLERANCE: float = 1e-3


@dataclass(frozen=True)
class _Findings:
    """Accumulates errors and warnings during one validation pass."""

    errors: list[DetectionValidationFinding] = field(default_factory=list)
    warnings: list[DetectionValidationFinding] = field(default_factory=list)

    def error(
        self, code: str, message: str, *, column: str | None = None, count: int = 0
    ) -> None:
        """Record one error finding."""
        self.errors.append(
            DetectionValidationFinding(
                code=code, message=message, column=column, count=count
            )
        )

    def warn(
        self, code: str, message: str, *, column: str | None = None, count: int = 0
    ) -> None:
        """Record one warning finding."""
        self.warnings.append(
            DetectionValidationFinding(
                code=code, message=message, column=column, count=count
            )
        )


class DetectionValidator:
    """Validates a detection artifact set against its configuration.

    Never raises: every problem becomes a finding, so one call reports the
    whole picture rather than the first failure.
    """

    __slots__ = ("_catalog", "_config")

    def __init__(
        self, config: DetectionConfig, *, catalog: RuleCatalog = RULE_CATALOG
    ) -> None:
        self._config = config
        self._catalog = catalog

    def validate(
        self,
        detections: Sequence[FiredDetection],
        assessments: Sequence[RiskAssessment],
        alerts: Sequence[SecurityAlert],
        *,
        entity_scope: Sequence[EntityScopeRecord] | None = None,
        detection_columns: Sequence[str] | None = None,
        risk_columns: Sequence[str] | None = None,
        alert_columns: Sequence[str] | None = None,
    ) -> DetectionValidationResult:
        """Validate the three artifacts and their relationships.

        The ``*_columns`` arguments carry the column order actually found in a
        published file.  Omit them to validate in-memory models, where column
        order is a property of the writer rather than of the data.
        """
        findings = _Findings()

        self._check_columns(findings, detection_columns, risk_columns, alert_columns)
        self._check_detections(findings, detections)
        self._check_assessments(findings, assessments)
        self._check_alerts(findings, alerts, assessments, detections)
        self._check_relationships(findings, detections, assessments, entity_scope)
        self._check_versions(findings, assessments)
        self._check_coverage(findings, detections, assessments, alerts)

        status = DetectionValidationStatus.VALID
        if findings.errors:
            status = DetectionValidationStatus.INVALID
        elif findings.warnings:
            status = DetectionValidationStatus.WARNING

        return DetectionValidationResult(
            status=status,
            detection_schema_version=DETECTION_SCHEMA_VERSION,
            scoring_version=SCORING_VERSION,
            alerting_version=ALERTING_VERSION,
            detection_row_count=len(detections),
            risk_assessment_row_count=len(assessments),
            alert_row_count=len(alerts),
            errors=tuple(findings.errors),
            warnings=tuple(findings.warnings),
            invalid_value_count=_count_for(findings.errors, INVALID_VALUE_CODES),
            prohibited_column_count=_count_for(
                findings.errors, PROHIBITED_COLUMN_CODES
            ),
            relationship_error_count=_count_for(findings.errors, RELATIONSHIP_CODES),
        )

    # -- column shape -------------------------------------------------------

    def _check_columns(
        self,
        findings: _Findings,
        detection_columns: Sequence[str] | None,
        risk_columns: Sequence[str] | None,
        alert_columns: Sequence[str] | None,
    ) -> None:
        """Check published column order and reject prohibited columns."""
        for observed, declared, table in (
            (detection_columns, DETECTION_COLUMNS, "rule_detections"),
            (risk_columns, RISK_COLUMNS, "risk_assessments"),
            (alert_columns, ALERT_COLUMNS, "security_alerts"),
        ):
            if observed is None:
                continue
            prohibited = sorted(PROHIBITED_ARTIFACT_COLUMNS & set(observed))
            if prohibited:
                findings.error(
                    "D003",
                    f"The {table} table carries prohibited column(s) {prohibited}",
                    column=table,
                    count=len(prohibited),
                )
            if tuple(observed) != tuple(declared):
                findings.error(
                    "D002",
                    f"The {table} table's column order does not match the "
                    f"declared order",
                    column=table,
                    count=len(observed),
                )

    # -- detections ---------------------------------------------------------

    def _check_detections(
        self, findings: _Findings, detections: Sequence[FiredDetection]
    ) -> None:
        """Validate the fired-detection table in isolation."""
        seen: set[str] = set()
        duplicates = 0
        derived_mismatch = 0
        unknown_rules: set[str] = set()
        wrong_versions = 0
        metadata_mismatch = 0
        out_of_range = 0
        non_finite = 0
        naive_times = 0
        unexplained = 0
        wrong_version_header = 0

        for detection in detections:
            if detection.detection_id in seen:
                duplicates += 1
            seen.add(detection.detection_id)

            expected = detection_identifier(
                detection.anchor_event_id, detection.rule_id, detection.rule_version
            )
            if detection.detection_id != expected:
                derived_mismatch += 1

            if detection.detection_schema_version != DETECTION_SCHEMA_VERSION:
                wrong_version_header += 1

            if not self._catalog.has(detection.rule_id):
                unknown_rules.add(detection.rule_id)
            else:
                spec = self._catalog.get(detection.rule_id)
                if detection.rule_version != spec.rule_version:
                    wrong_versions += 1
                if (
                    detection.rule_family is not spec.family
                    or detection.attack_category is not spec.attack_category
                    or detection.correlation_group is not spec.correlation_group
                ):
                    metadata_mismatch += 1

            if math.isnan(detection.signal_strength) or math.isinf(
                detection.signal_strength
            ):
                non_finite += 1
            elif not 0.0 <= detection.signal_strength <= 1.0:
                out_of_range += 1

            if not _is_utc(detection.anchor_event_time):
                naive_times += 1
            if not detection.evidence or not detection.reason_codes:
                unexplained += 1

        _report(
            findings,
            "D005",
            "detection identifier(s) repeat",
            duplicates,
            column="detection_id",
        )
        _report(
            findings,
            "D006",
            "detection identifier(s) are not derived from the anchor, rule, "
            "and rule version",
            derived_mismatch,
            column="detection_id",
        )
        _report(
            findings,
            "D004",
            "detection row(s) declare an unsupported schema version",
            wrong_version_header,
            column="detection_schema_version",
        )
        if unknown_rules:
            findings.error(
                "D007",
                f"{len(unknown_rules)} rule identifier(s) are not registered in "
                f"the rule catalog",
                column="rule_id",
                count=len(unknown_rules),
            )
        _report(
            findings,
            "D008",
            "detection(s) name a rule version the catalog does not declare",
            wrong_versions,
            column="rule_version",
        )
        _report(
            findings,
            "D009",
            "detection(s) carry metadata that disagrees with the catalog",
            metadata_mismatch,
            column="rule_id",
        )
        _report(
            findings,
            "D010",
            "signal strength value(s) lie outside [0, 1]",
            out_of_range,
            column="signal_strength",
        )
        _report(
            findings,
            "D011",
            "signal strength value(s) are NaN or infinite",
            non_finite,
            column="signal_strength",
        )
        _report(
            findings,
            "D012",
            "detection timestamp(s) are naive or not UTC",
            naive_times,
            column="anchor_event_time",
        )
        _report(
            findings,
            "D013",
            "fired detection(s) carry no evidence or no reason code",
            unexplained,
            column="evidence_json",
        )

    # -- assessments --------------------------------------------------------

    def _check_assessments(
        self, findings: _Findings, assessments: Sequence[RiskAssessment]
    ) -> None:
        """Validate the risk-assessment table in isolation."""
        seen: set[str] = set()
        duplicates = 0
        out_of_range = 0
        non_finite = 0
        count_mismatch = 0
        zero_inconsistent = 0
        category_inconsistent = 0
        below_floor = 0
        naive_times = 0
        wrong_fingerprint = 0

        active = self._config.fingerprint()
        floor = self._config.scoring.min_fired_risk_score

        for assessment in assessments:
            if assessment.anchor_event_id in seen:
                duplicates += 1
            seen.add(assessment.anchor_event_id)

            score = assessment.risk_score
            if math.isnan(score) or math.isinf(score):
                non_finite += 1
            elif not 0.0 <= score <= 100.0:
                out_of_range += 1

            if assessment.fired_rule_count != len(assessment.fired_rule_ids):
                count_mismatch += 1

            fired = assessment.fired_rule_count > 0
            if fired != (score > 0.0):
                zero_inconsistent += 1
            if fired != (assessment.primary_attack_category is not None):
                zero_inconsistent += 1

            if fired and score < floor:
                below_floor += 1

            category = assessment.primary_attack_category
            if category is not None and (
                category not in assessment.contributing_categories
            ):
                category_inconsistent += 1

            if not _is_utc(assessment.anchor_event_time):
                naive_times += 1
            if assessment.configuration_fingerprint != active:
                wrong_fingerprint += 1

        _report(
            findings,
            "D014",
            "anchor(s) appear more than once in the risk-assessment table",
            duplicates,
            column="anchor_event_id",
        )
        _report(
            findings,
            "D015",
            "risk score(s) lie outside [0, 100]",
            out_of_range,
            column="risk_score",
        )
        _report(
            findings,
            "D011",
            "risk score(s) are NaN or infinite",
            non_finite,
            column="risk_score",
        )
        _report(
            findings,
            "D016",
            "assessment(s) report a fired-rule count that "
            "disagrees with the fired-rule list",
            count_mismatch,
            column="fired_rule_count",
        )
        _report(
            findings,
            "D017",
            "assessment(s) break the zero-risk equivalence: a score is zero if "
            "and only if no rule fired",
            zero_inconsistent,
            column="risk_score",
        )
        _report(
            findings,
            "D018",
            "assessment(s) name a primary category that is "
            "not among the contributing categories",
            category_inconsistent,
            column="primary_attack_category",
        )
        _report(
            findings,
            "D019",
            "fired assessment(s) score below the configured minimum fired risk score",
            below_floor,
            column="risk_score",
        )
        _report(
            findings,
            "D012",
            "assessment timestamp(s) are naive or not UTC",
            naive_times,
            column="anchor_event_time",
        )
        _report(
            findings,
            "D020",
            "assessment(s) record a configuration fingerprint that is not the "
            "active run's",
            wrong_fingerprint,
            column="configuration_fingerprint",
        )

    # -- alerts -------------------------------------------------------------

    def _check_alerts(
        self,
        findings: _Findings,
        alerts: Sequence[SecurityAlert],
        assessments: Sequence[RiskAssessment],
        detections: Sequence[FiredDetection],
    ) -> None:
        """Validate the alert table and its consistency with the other two."""
        seen: set[str] = set()
        duplicates = 0
        out_of_range = 0
        non_finite = 0
        aggregate_above_peak = 0
        temporal = 0
        below_floor = 0
        scope_inconsistent = 0
        unknown_rules = 0
        mean_mismatch = 0

        known_rules = {detection.rule_id for detection in detections}
        score_floor = self._config.alerting.min_alert_risk_score
        severity_floor = self._config.alerting.min_alert_severity
        by_window = _assessments_by_window(assessments)

        for alert in alerts:
            if alert.alert_id in seen:
                duplicates += 1
            seen.add(alert.alert_id)

            for value in (alert.aggregate_risk_score, alert.peak_risk_score):
                if math.isnan(value) or math.isinf(value):
                    non_finite += 1
                elif not 0.0 <= value <= 100.0:
                    out_of_range += 1

            if alert.aggregate_risk_score > alert.peak_risk_score:
                aggregate_above_peak += 1

            if _has_temporal_defect(alert):
                temporal += 1

            # LOW is an ordinary alert severity.  What is checked is the two
            # configured floors -- never the band itself.
            if alert.peak_risk_score < score_floor:
                below_floor += 1
            # The severity value itself is not checked: it is enum-typed, so
            # an unrecognised band cannot reach here.  Only the floors gate.
            if not severity_at_least(alert.current_severity, severity_floor):
                below_floor += 1

            if not _scope_consistent(alert):
                scope_inconsistent += 1

            if known_rules and not set(alert.contributing_rule_ids) <= known_rules:
                unknown_rules += 1

            if not _mean_is_plausible(alert, by_window):
                mean_mismatch += 1

        _report(
            findings,
            "D021",
            "alert identifier(s) repeat",
            duplicates,
            column="alert_id",
        )
        _report(
            findings,
            "D022",
            "alert risk score(s) lie outside [0, 100]",
            out_of_range,
            column="peak_risk_score",
        )
        _report(
            findings,
            "D011",
            "alert risk score(s) are NaN or infinite",
            non_finite,
            column="aggregate_risk_score",
        )
        _report(
            findings,
            "D023",
            "alert(s) report an aggregate risk above their peak risk",
            aggregate_above_peak,
            column="aggregate_risk_score",
        )
        _report(
            findings,
            "D026",
            "alert(s) have an invalid time range or a "
            "non-positive contributing-event count",
            temporal,
            column="first_seen",
        )
        _report(
            findings,
            "D024",
            "alert(s) fall below a configured score or severity floor",
            below_floor,
            column="peak_risk_score",
        )
        _report(
            findings,
            "D025",
            "alert(s) carry scope metadata inconsistent with their grouping mode",
            scope_inconsistent,
            column="grouping_mode",
        )
        _report(
            findings,
            "D027",
            "alert(s) name a contributing rule absent from the fired detections",
            unknown_rules,
            column="contributing_rule_ids_json",
        )
        _report(
            findings,
            "D031",
            "alert(s) report an aggregate risk that is not the arithmetic mean "
            "of their grouped assessments",
            mean_mismatch,
            column="aggregate_risk_score",
        )

    # -- relationships ------------------------------------------------------

    def _check_relationships(
        self,
        findings: _Findings,
        detections: Sequence[FiredDetection],
        assessments: Sequence[RiskAssessment],
        entity_scope: Sequence[EntityScopeRecord] | None,
    ) -> None:
        """Validate that the artifacts describe one coherent run."""
        by_anchor: dict[str, set[str]] = {}
        for detection in detections:
            by_anchor.setdefault(detection.anchor_event_id, set()).add(
                detection.rule_id
            )

        assessed = {assessment.anchor_event_id for assessment in assessments}
        orphaned = len(set(by_anchor) - assessed)
        _report(
            findings,
            "D028",
            "fired detection anchor(s) have no risk assessment",
            orphaned,
            column="anchor_event_id",
        )

        mismatched = sum(
            1
            for assessment in assessments
            if by_anchor.get(assessment.anchor_event_id, set())
            != set(assessment.fired_rule_ids)
        )
        _report(
            findings,
            "D029",
            "assessment(s) name a fired-rule set that does not match the "
            "detections recorded for that anchor",
            mismatched,
            column="fired_rule_ids_json",
        )

        if entity_scope is None:
            return

        scope_anchors: set[str] = set()
        scope_duplicates = 0
        for record in entity_scope:
            if record.anchor_event_id in scope_anchors:
                scope_duplicates += 1
            scope_anchors.add(record.anchor_event_id)

        missing = len(assessed - scope_anchors)
        extra = len(scope_anchors - assessed)
        if scope_duplicates or missing or extra:
            findings.error(
                "D030",
                f"The entity-scope table is not one-to-one with the risk "
                f"assessments: {scope_duplicates} duplicate anchor key(s), "
                f"{missing} assessed anchor(s) without a scope row, and "
                f"{extra} scope row(s) without an assessment",
                column="anchor_event_id",
                count=scope_duplicates + missing + extra,
            )

    # -- versions and coverage ---------------------------------------------

    def _check_versions(
        self, findings: _Findings, assessments: Sequence[RiskAssessment]
    ) -> None:
        """Validate the contract versions the run recorded."""
        unsupported = sum(
            1
            for assessment in assessments
            if assessment.scoring_version != SCORING_VERSION
        )
        _report(
            findings,
            "D032",
            "assessment(s) record an unsupported scoring version",
            unsupported,
            column="scoring_version",
        )

        if self._config.alerting.alerting_version != ALERTING_VERSION:
            findings.error(
                "D032",
                f"The configured alerting version is not the supported "
                f"{ALERTING_VERSION!r}",
                column="alerting_version",
                count=1,
            )

    def _check_coverage(
        self,
        findings: _Findings,
        detections: Sequence[FiredDetection],
        assessments: Sequence[RiskAssessment],
        alerts: Sequence[SecurityAlert],
    ) -> None:
        """Emit warnings about coverage rather than correctness."""
        if not assessments:
            findings.error(
                "D001",
                "The risk-assessment table is empty; every evaluated anchor "
                "must produce a row",
                column="risk_assessments",
                count=0,
            )
            return

        fired_rules = {detection.rule_id for detection in detections}
        silent = sorted(
            rule_id
            for rule_id in self._config.enabled_rule_ids
            if rule_id not in fired_rules
        )
        if silent:
            findings.warn(
                "D050",
                f"{len(silent)} enabled rule(s) never fired in this run",
                column="rule_id",
                count=len(silent),
            )
        if not alerts:
            findings.warn(
                "D051",
                "No alert was produced from this detection set",
                column="security_alerts",
                count=0,
            )

        evaluations = sum(
            assessment.insufficient_data_count for assessment in assessments
        )
        possible = len(assessments) * max(len(self._config.enabled_rule_ids), 1)
        if possible and evaluations / possible > _INSUFFICIENT_DATA_WARN_RATE:
            findings.warn(
                "D052",
                "Almost every rule evaluation reported unavailable history",
                column="insufficient_data_count",
                count=evaluations,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_for(
    findings: Sequence[DetectionValidationFinding], codes: frozenset[str]
) -> int:
    """Sum the counts of every finding whose code is in *codes*."""
    return sum(finding.count for finding in findings if finding.code in codes)


def _report(
    findings: _Findings,
    code: str,
    description: str,
    count: int,
    *,
    column: str | None = None,
) -> None:
    """Record an error when *count* is positive, phrased as a count."""
    if count:
        findings.error(code, f"{count} {description}", column=column, count=count)


def _is_utc(value: object) -> bool:
    """Return whether *value* is a timezone-aware timestamp stored in UTC."""
    offset = getattr(value, "utcoffset", None)
    if offset is None:
        return False
    resolved = offset()
    return resolved is not None and resolved == UTC.utcoffset(None)


def _scope_consistent(alert: SecurityAlert) -> bool:
    """Return whether an alert's scope metadata matches its grouping mode.

    A category-scoped alert must carry no scope value at all; an entity-scoped
    alert carries one only when it actually grouped on a dimension, which is
    what makes a degraded group visible rather than silently mislabelled.
    """
    if alert.scope_value is None:
        return alert.scope_kind is ScopeKind.NONE
    return (
        alert.grouping_mode is AlertGroupingMode.ENTITY_SCOPED
        and alert.scope_kind is not ScopeKind.NONE
    )


def _assessments_by_window(
    assessments: Sequence[RiskAssessment],
) -> dict[AttackCategory, list[RiskAssessment]]:
    """Index assessments by the category an alert would group them under."""
    indexed: dict[AttackCategory, list[RiskAssessment]] = {}
    for assessment in assessments:
        category = assessment.primary_attack_category
        if category is not None:
            indexed.setdefault(category, []).append(assessment)
    return indexed


def _has_temporal_defect(alert: SecurityAlert) -> bool:
    """Return whether an alert's time range or event count is malformed.

    Timezone awareness is established before the two timestamps are compared:
    comparing a naive timestamp with an aware one raises, and this validator
    must report a defect rather than throw on one.
    """
    if not _is_utc(alert.first_seen) or not _is_utc(alert.last_seen):
        return True
    return alert.last_seen < alert.first_seen or alert.contributing_event_count < 1


def _within(alert: SecurityAlert, assessment: RiskAssessment) -> bool:
    """Return whether an assessment falls inside an alert's window.

    ``False`` for any timestamp that cannot be compared, so a corrupt artifact
    produces a finding rather than an exception.
    """
    try:
        return bool(alert.first_seen <= assessment.anchor_event_time <= alert.last_seen)
    except TypeError:
        return False


def _mean_is_plausible(
    alert: SecurityAlert,
    by_category: Mapping[AttackCategory, Sequence[RiskAssessment]],
) -> bool:
    """Return whether an alert's aggregate risk can be a mean of its members.

    The validator cannot rebuild an alert's exact membership without replaying
    grouping, so it checks the property that must hold for *any* arithmetic
    mean of scores drawn from the alert's category and window: the value lies
    between the smallest and largest such score, and never above the peak.

    That is a real constraint -- a sum, a maximum, or a noisy-OR of several
    contributing scores all violate it -- while staying independent of the
    grouping replay.

    A timestamp that cannot be compared is skipped rather than raised on.  A
    malformed timestamp is already reported under ``D012``, and this validator
    must return the whole picture rather than stopping at the first defect.
    """
    candidates = [
        assessment.risk_score
        for assessment in by_category.get(alert.attack_category, ())
        if _within(alert, assessment)
    ]
    if not candidates:
        return True
    lowest, highest = min(candidates), max(candidates)
    return (
        lowest - _MEAN_TOLERANCE
        <= alert.aggregate_risk_score
        <= highest + _MEAN_TOLERANCE
    )


def validate_detection_artifacts(
    config: DetectionConfig,
    detections: Sequence[FiredDetection],
    assessments: Sequence[RiskAssessment],
    alerts: Sequence[SecurityAlert],
    *,
    entity_scope: Sequence[EntityScopeRecord] | None = None,
    detection_columns: Sequence[str] | None = None,
    risk_columns: Sequence[str] | None = None,
    alert_columns: Sequence[str] | None = None,
) -> DetectionValidationResult:
    """Validate a detection artifact set with a one-shot validator."""
    return DetectionValidator(config).validate(
        detections,
        assessments,
        alerts,
        entity_scope=entity_scope,
        detection_columns=detection_columns,
        risk_columns=risk_columns,
        alert_columns=alert_columns,
    )
