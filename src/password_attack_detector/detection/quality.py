"""Aggregate quality reporting for one detection run.

Follows the Phase 2 and Phase 3 pattern: a frozen dataclass, a JSON renderer,
and a Markdown renderer built from ``| Metric | Value |`` tables.

**Everything here is an aggregate.**  The report never contains an event
identifier, a detection identifier, an alert identifier, an entity-scope value,
a user, source, device or session pseudonym, a coordinate, an evidence row, a
secret, or an absolute path.  Counts and distributions only, so a quality
report stays safe to attach to a ticket or paste into a review.

The report also carries its own :data:`DEFINITIONS`, because every number in it
is easy to over-read.  A risk score is an ordinal magnitude, not a probability;
a signal strength is the same; an aggregate risk is a mean and a peak is a
maximum; and evidence records what was observed, never proof of what caused it.
Those sentences are rendered into both output formats rather than left to a
reader's assumption.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from password_attack_detector.detection.alerts import ALERTING_VERSION, AlertingResult
from password_attack_detector.detection.catalog import RULE_CATALOG, RuleCatalog
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.engine import EngineResult
from password_attack_detector.detection.enums import (
    AlertGroupingMode,
    RuleStatus,
    Severity,
)
from password_attack_detector.detection.schemas import (
    DETECTION_SCHEMA_VERSION,
    DetectionValidationResult,
    SecurityAlert,
)
from password_attack_detector.detection.scoring import SCORING_VERSION, ScoringResult

__all__ = [
    "DEFINITIONS",
    "DetectionQualityReport",
    "generate_detection_quality_report",
    "report_to_json",
    "report_to_markdown",
]

#: What every number in this report does and does not mean.  Rendered into
#: both output formats so a reader never has to assume.
DEFINITIONS: Final[Mapping[str, str]] = {
    "risk_score": (
        "A bounded ordinal magnitude on a 0-100 scale. It orders findings by "
        "how much configured evidence accumulated behind them. It is not a "
        "probability and must not be read as one."
    ),
    "signal_strength": (
        "A bounded ordinal magnitude in [0, 1] describing how far one rule's "
        "observations exceeded its configured thresholds. It is not a "
        "probability."
    ),
    "aggregate_risk_score": (
        "The arithmetic mean of the risk scores of the qualifying assessments "
        "grouped into an alert."
    ),
    "peak_risk_score": (
        "The maximum risk score among the assessments grouped into an alert."
    ),
    "evidence": (
        "A record of what was observed and which configured condition it "
        "matched. Evidence is an indicator; it is not causal proof that an "
        "attack occurred."
    ),
}


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    """Return a five-number summary, or nulls when there is nothing to sum.

    An empty input yields ``None`` for every statistic rather than zero: a
    minimum of zero over no observations would read as a real measurement.
    """
    if not values:
        return dict.fromkeys(("min", "median", "mean", "p90", "max"))
    ordered = sorted(values)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        index = min(count - 1, max(0, math.ceil(fraction * count) - 1))
        return ordered[index]

    return {
        "min": ordered[0],
        "median": percentile(0.5),
        "mean": math.fsum(ordered) / count,
        "p90": percentile(0.9),
        "max": ordered[-1],
    }


@dataclass(frozen=True)
class DetectionQualityReport:
    """Aggregate description of one detection run."""

    detection_schema_version: str
    scoring_version: str
    alerting_version: str

    input_snapshot_count: int | None
    evaluated_snapshot_count: int
    total_rule_evaluation_count: int
    fired_count: int
    not_fired_count: int
    insufficient_data_count: int
    disabled_rule_count: int

    per_rule_trigger_counts: dict[str, int]
    per_family_trigger_counts: dict[str, int]
    attack_category_distribution: dict[str, int]
    severity_distribution: dict[str, int]
    risk_score_summary: dict[str, float | None]
    zero_score_assessment_count: int

    alert_count: int
    grouping_mode: str
    grouping_mode_counts: dict[str, int]
    grouped_detection_count: int
    suppressed_by_cooldown_count: int
    suppressed_by_rate_limit_count: int
    suppressed_total: int
    escalated_count: int
    below_score_floor_count: int
    below_severity_floor_count: int
    scope_missing_count: int
    entity_scoped_count: int
    category_scoped_count: int
    average_events_per_alert: float | None
    alert_duration_seconds_summary: dict[str, float | None]
    aggregate_risk_summary: dict[str, float | None]
    peak_risk_summary: dict[str, float | None]

    min_alert_risk_score: float
    min_alert_severity: str
    low_alert_reachable: bool

    configuration_fingerprint: str
    rule_catalog_fingerprint: str
    detection_fingerprint: str
    risk_fingerprint: str
    alert_fingerprint: str
    source_feature_manifest_fingerprint: str | None

    validation_status: str | None
    validation_result: dict[str, Any] | None
    warning_summary: list[str]
    definitions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping."""
        return {
            "detection_schema_version": self.detection_schema_version,
            "scoring_version": self.scoring_version,
            "alerting_version": self.alerting_version,
            "input_snapshot_count": self.input_snapshot_count,
            "evaluated_snapshot_count": self.evaluated_snapshot_count,
            "total_rule_evaluation_count": self.total_rule_evaluation_count,
            "fired_count": self.fired_count,
            "not_fired_count": self.not_fired_count,
            "insufficient_data_count": self.insufficient_data_count,
            "disabled_rule_count": self.disabled_rule_count,
            "per_rule_trigger_counts": self.per_rule_trigger_counts,
            "per_family_trigger_counts": self.per_family_trigger_counts,
            "attack_category_distribution": self.attack_category_distribution,
            "severity_distribution": self.severity_distribution,
            "risk_score_summary": self.risk_score_summary,
            "zero_score_assessment_count": self.zero_score_assessment_count,
            "alert_count": self.alert_count,
            "grouping_mode": self.grouping_mode,
            "grouping_mode_counts": self.grouping_mode_counts,
            "grouped_detection_count": self.grouped_detection_count,
            "suppressed_by_cooldown_count": self.suppressed_by_cooldown_count,
            "suppressed_by_rate_limit_count": self.suppressed_by_rate_limit_count,
            "suppressed_total": self.suppressed_total,
            "escalated_count": self.escalated_count,
            "below_score_floor_count": self.below_score_floor_count,
            "below_severity_floor_count": self.below_severity_floor_count,
            "scope_missing_count": self.scope_missing_count,
            "entity_scoped_count": self.entity_scoped_count,
            "category_scoped_count": self.category_scoped_count,
            "average_events_per_alert": self.average_events_per_alert,
            "alert_duration_seconds_summary": self.alert_duration_seconds_summary,
            "aggregate_risk_summary": self.aggregate_risk_summary,
            "peak_risk_summary": self.peak_risk_summary,
            "min_alert_risk_score": self.min_alert_risk_score,
            "min_alert_severity": self.min_alert_severity,
            "low_alert_reachable": self.low_alert_reachable,
            "configuration_fingerprint": self.configuration_fingerprint,
            "rule_catalog_fingerprint": self.rule_catalog_fingerprint,
            "detection_fingerprint": self.detection_fingerprint,
            "risk_fingerprint": self.risk_fingerprint,
            "alert_fingerprint": self.alert_fingerprint,
            "source_feature_manifest_fingerprint": (
                self.source_feature_manifest_fingerprint
            ),
            "validation_status": self.validation_status,
            "validation_result": self.validation_result,
            "warning_summary": self.warning_summary,
            "definitions": self.definitions,
        }


def generate_detection_quality_report(
    *,
    engine_result: EngineResult,
    scoring_result: ScoringResult,
    alerting_result: AlertingResult,
    config: DetectionConfig,
    validation_result: DetectionValidationResult | None = None,
    input_snapshot_count: int | None = None,
    detection_fingerprint: str = "",
    risk_fingerprint: str = "",
    alert_fingerprint: str = "",
    source_feature_manifest_fingerprint: str | None = None,
    catalog: RuleCatalog = RULE_CATALOG,
) -> DetectionQualityReport:
    """Build the aggregate quality report for one run.

    Every input is already an aggregate or a typed model; nothing here reads a
    feature snapshot, a label, or a raw event, so there is no path by which a
    row could reach the output.
    """
    status_counts = engine_result.status_counts
    alerts = alerting_result.alerts
    stats = alerting_result.stats

    per_rule = {
        rule_id: counts.get(str(RuleStatus.FIRED), 0)
        for rule_id, counts in sorted(engine_result.per_rule_counts.items())
    }
    per_family: dict[str, int] = {}
    for rule_id, fired in per_rule.items():
        family = str(catalog.get(rule_id).family)
        per_family[family] = per_family.get(family, 0) + fired

    categories: dict[str, int] = {}
    for detection in engine_result.fired_detections:
        key = str(detection.attack_category)
        categories[key] = categories.get(key, 0) + 1

    scores = [item.risk_score for item in scoring_result.assessments]
    durations = [
        (alert.last_seen - alert.first_seen).total_seconds() for alert in alerts
    ]

    return DetectionQualityReport(
        detection_schema_version=DETECTION_SCHEMA_VERSION,
        scoring_version=SCORING_VERSION,
        alerting_version=ALERTING_VERSION,
        input_snapshot_count=input_snapshot_count,
        evaluated_snapshot_count=engine_result.evaluated_snapshot_count,
        total_rule_evaluation_count=engine_result.total_rule_evaluation_count,
        fired_count=status_counts.get(str(RuleStatus.FIRED), 0),
        not_fired_count=status_counts.get(str(RuleStatus.NOT_FIRED), 0),
        insufficient_data_count=engine_result.insufficient_data_count,
        disabled_rule_count=engine_result.disabled_rule_count,
        per_rule_trigger_counts=per_rule,
        per_family_trigger_counts=dict(sorted(per_family.items())),
        attack_category_distribution=dict(sorted(categories.items())),
        severity_distribution={
            str(severity): scoring_result.severity_counts.get(str(severity), 0)
            for severity in Severity
        },
        risk_score_summary=_summary(scores),
        zero_score_assessment_count=scoring_result.zero_score_count,
        alert_count=len(alerts),
        grouping_mode=str(stats.grouping_mode),
        grouping_mode_counts=_grouping_mode_counts(alerts),
        grouped_detection_count=stats.grouped_detection_count,
        suppressed_by_cooldown_count=stats.suppressed_by_cooldown_count,
        suppressed_by_rate_limit_count=stats.suppressed_by_rate_limit_count,
        suppressed_total=stats.suppressed_total,
        escalated_count=stats.escalated_count,
        below_score_floor_count=stats.below_score_floor_count,
        below_severity_floor_count=stats.below_severity_floor_count,
        scope_missing_count=stats.scope_missing_count,
        entity_scoped_count=stats.entity_scoped_count,
        category_scoped_count=stats.category_scoped_count,
        average_events_per_alert=(
            None
            if not alerts
            else math.fsum(alert.contributing_event_count for alert in alerts)
            / len(alerts)
        ),
        alert_duration_seconds_summary=_summary(durations),
        aggregate_risk_summary=_summary(
            [alert.aggregate_risk_score for alert in alerts]
        ),
        peak_risk_summary=_summary([alert.peak_risk_score for alert in alerts]),
        min_alert_risk_score=config.alerting.min_alert_risk_score,
        min_alert_severity=str(config.alerting.min_alert_severity),
        low_alert_reachable=config.low_alert_reachable,
        configuration_fingerprint=engine_result.configuration_fingerprint,
        rule_catalog_fingerprint=engine_result.rule_catalog_fingerprint,
        detection_fingerprint=detection_fingerprint,
        risk_fingerprint=risk_fingerprint,
        alert_fingerprint=alert_fingerprint,
        source_feature_manifest_fingerprint=source_feature_manifest_fingerprint,
        validation_status=(
            None if validation_result is None else str(validation_result.status)
        ),
        validation_result=(
            None if validation_result is None else validation_result.to_dict()
        ),
        warning_summary=(
            []
            if validation_result is None
            else [item.code for item in validation_result.warnings]
        ),
        definitions=dict(DEFINITIONS),
    )


def _grouping_mode_counts(alerts: Sequence[SecurityAlert]) -> dict[str, int]:
    """Count alerts by the regime that grouped each of them."""
    counts = {str(mode): 0 for mode in AlertGroupingMode}
    for alert in alerts:
        counts[str(alert.grouping_mode)] += 1
    return counts


def report_to_json(report: DetectionQualityReport, *, indent: int = 2) -> str:
    """Render the report as JSON."""
    return json.dumps(report.to_dict(), indent=indent, sort_keys=True, default=str)


def _fmt(value: float | int | None, *, digits: int = 2) -> str:
    """Render a number for a Markdown table, or ``n/a`` when it is absent."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, datetime):  # pragma: no cover - defensive
        return value.isoformat()
    return f"{value:,.{digits}f}"


def _append_counts(
    lines: list[str], title: str, label: str, counts: Mapping[str, int]
) -> None:
    """Append a two-column counts table."""
    if not counts:
        return
    lines.extend(["", f"## {title}", "", f"| {label} | Count |", "|--------|-------|"])
    for key, value in sorted(counts.items()):
        lines.append(f"| `{key}` | {value:,} |")


def _append_summary(
    lines: list[str], title: str, summary: Mapping[str, float | None]
) -> None:
    """Append a five-number summary table."""
    lines.extend(["", f"## {title}", "", "| Statistic | Value |", "|--------|-------|"])
    for key in ("min", "median", "mean", "p90", "max"):
        lines.append(f"| {key} | {_fmt(summary.get(key))} |")


def report_to_markdown(report: DetectionQualityReport) -> str:
    """Render the report as Markdown, in the Phase 2 and Phase 3 table style."""
    lines: list[str] = [
        "# Detection Quality Report",
        "",
        "Aggregate statistics only. This report contains no event rows, "
        "identifiers, pseudonyms, entity-scope values, coordinates, evidence "
        "rows, or absolute paths.",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Detection schema version | `{report.detection_schema_version}` |",
        f"| Scoring version | `{report.scoring_version}` |",
        f"| Alerting version | `{report.alerting_version}` |",
        f"| Input snapshots | {_fmt(report.input_snapshot_count)} |",
        f"| Evaluated snapshots | {_fmt(report.evaluated_snapshot_count)} |",
        f"| Rule evaluations | {_fmt(report.total_rule_evaluation_count)} |",
        f"| Fired | {_fmt(report.fired_count)} |",
        f"| Not fired | {_fmt(report.not_fired_count)} |",
        f"| Insufficient data | {_fmt(report.insufficient_data_count)} |",
        f"| Disabled rules | {_fmt(report.disabled_rule_count)} |",
        f"| Zero-score assessments | {_fmt(report.zero_score_assessment_count)} |",
        f"| Validation status | `{report.validation_status or 'n/a'}` |",
    ]

    _append_counts(lines, "Triggers by rule", "Rule", report.per_rule_trigger_counts)
    _append_counts(
        lines, "Triggers by family", "Family", report.per_family_trigger_counts
    )
    _append_counts(
        lines, "Attack categories", "Category", report.attack_category_distribution
    )
    _append_counts(
        lines, "Severity distribution", "Severity", report.severity_distribution
    )
    _append_summary(lines, "Risk score", report.risk_score_summary)

    lines.extend(
        [
            "",
            "## Alerting",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Alerts | {_fmt(report.alert_count)} |",
            f"| Grouping mode | `{report.grouping_mode}` |",
            f"| Grouped detections | {_fmt(report.grouped_detection_count)} |",
            f"| Suppressed by cooldown | {_fmt(report.suppressed_by_cooldown_count)} |",
            f"| Suppressed by rate limit | "
            f"{_fmt(report.suppressed_by_rate_limit_count)} |",
            f"| Suppressed total | {_fmt(report.suppressed_total)} |",
            f"| Escalated | {_fmt(report.escalated_count)} |",
            f"| Below score floor | {_fmt(report.below_score_floor_count)} |",
            f"| Below severity floor | {_fmt(report.below_severity_floor_count)} |",
            f"| Scope missing | {_fmt(report.scope_missing_count)} |",
            f"| Entity-scoped alerts | {_fmt(report.entity_scoped_count)} |",
            f"| Category-scoped alerts | {_fmt(report.category_scoped_count)} |",
            f"| Average events per alert | {_fmt(report.average_events_per_alert)} |",
            f"| Minimum alert risk score | {_fmt(report.min_alert_risk_score)} |",
            f"| Minimum alert severity | `{report.min_alert_severity}` |",
            f"| LOW alerts reachable | {report.low_alert_reachable} |",
        ]
    )

    _append_summary(
        lines, "Alert duration (seconds)", report.alert_duration_seconds_summary
    )
    _append_summary(lines, "Aggregate risk", report.aggregate_risk_summary)
    _append_summary(lines, "Peak risk", report.peak_risk_summary)

    lines.extend(
        [
            "",
            "## Fingerprints",
            "",
            "| Artifact | Fingerprint |",
            "|--------|-------|",
            f"| Configuration | `{report.configuration_fingerprint}` |",
            f"| Rule catalog | `{report.rule_catalog_fingerprint}` |",
            f"| Fired detections | `{report.detection_fingerprint}` |",
            f"| Risk assessments | `{report.risk_fingerprint}` |",
            f"| Security alerts | `{report.alert_fingerprint}` |",
            f"| Source feature manifest | "
            f"`{report.source_feature_manifest_fingerprint or 'n/a'}` |",
            "",
            "## What these numbers mean",
            "",
        ]
    )
    for term, meaning in sorted(report.definitions.items()):
        lines.append(f"- **`{term}`** -- {meaning}")

    if report.warning_summary:
        lines.extend(["", "## Validation warnings", ""])
        lines.extend(f"- `{code}`" for code in report.warning_summary)

    return "\n".join(lines) + "\n"
