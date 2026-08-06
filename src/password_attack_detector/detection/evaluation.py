"""Ground-truth evaluation of a detection run.

**This is the only module in the detection package permitted to read labels,
split assignments, or campaign metadata.**  ``DetectionEngine``, ``RiskScorer``,
and ``AlertBuilder`` take no such argument and import nothing from here, so the
separation is a type error rather than a convention -- and a test asserts the
import graph stays that way.

Everything is *calculated*.  No metric in this module is a literal, and there
is no threshold-search code path anywhere: nothing here reads a metric and
writes a configuration value back.  Detection thresholds are chosen from the
train and validation splits by a human and are never tuned against the test
split, because a threshold fitted to a test set turns that set's numbers into a
description of the fitting rather than of the detector.

**Null semantics are explicit.**  A metric over an empty denominator is
``None``, never zero and never one.  Zero would read as a measured absence and
one as a measured perfection; both are claims the data does not support.

**Every result is synthetic.**  These numbers describe generated traffic with
known ground truth.  They say nothing about real-world effectiveness, and
:data:`SYNTHETIC_CAVEAT` is rendered at the top of both report formats so no
reader can miss it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.detection.enums import AttackCategory
from password_attack_detector.detection.schemas import (
    FiredDetection,
    RiskAssessment,
    SecurityAlert,
)

__all__ = [
    "CATEGORY_TO_SCENARIO",
    "NOVEL_HOLDOUT_SPLIT",
    "SYNTHETIC_CAVEAT",
    "CampaignMetrics",
    "CampaignRecord",
    "CategoryMetrics",
    "EvaluationReport",
    "LabelRecord",
    "SplitMetrics",
    "SplitRecord",
    "evaluate_detection_run",
    "report_to_json",
    "report_to_markdown",
]

#: Rendered at the top of every evaluation report, in both formats.
SYNTHETIC_CAVEAT: Final[str] = (
    "These metrics describe synthetic authentication traffic with known "
    "ground truth. They measure how this rule set behaves on generated data "
    "and are not evidence of real-world detection effectiveness."
)

#: Which synthetic scenario each rule-based attack category is claiming.
#: Declared explicitly rather than inferred from a name match, so a renamed
#: enum member fails a test instead of silently scoring against the wrong
#: class.
CATEGORY_TO_SCENARIO: Final[Mapping[AttackCategory, ScenarioType]] = {
    AttackCategory.BRUTE_FORCE: ScenarioType.BRUTE_FORCE,
    AttackCategory.PASSWORD_SPRAYING: ScenarioType.PASSWORD_SPRAYING,
    AttackCategory.CREDENTIAL_STUFFING: ScenarioType.CREDENTIAL_STUFFING,
    AttackCategory.DISTRIBUTED_BRUTE_FORCE: ScenarioType.DISTRIBUTED_BRUTE_FORCE,
    AttackCategory.ACCOUNT_TAKEOVER_INDICATOR: (
        ScenarioType.ACCOUNT_TAKEOVER_INDICATOR
    ),
    AttackCategory.IMPOSSIBLE_TRAVEL_INDICATOR: ScenarioType.IMPOSSIBLE_TRAVEL,
    AttackCategory.BOT_ACTIVITY: ScenarioType.BOT_ACTIVITY,
    # No rule claims an MFA scenario: the synthetic generator has none, so this
    # category is scored only at the event level and never as a class.
}

#: The split whose events are evaluation-only.  Reported on its own and never
#: folded into the supervised category metrics: its whole purpose is to measure
#: behaviour on anomalies no rule was written for, and treating it as an
#: ordinary class would make that measurement meaningless.
NOVEL_HOLDOUT_SPLIT: Final[str] = "novel_anomaly_holdout"

#: Label used in the confusion matrix for "no category was assigned".
_NO_CATEGORY: Final[str] = "none"


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One ground-truth label, as published by Phase 3."""

    event_id: str
    attack_class: str
    malicious: bool
    supervised_training_eligible: bool = True


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """One split assignment, as published by Phase 3."""

    event_id: str
    split: str
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    """One event's campaign membership, from the Phase 2 label table.

    Carried separately because ``FEATURE_LABEL_COLUMNS`` deliberately omits
    ``campaign_id``: the splitter reads it internally for group isolation, but
    it is never published beside a model input.  Campaign metrics therefore
    require this extra input, and report ``unavailable`` without it rather than
    being fabricated.
    """

    event_id: str
    campaign_id: str


@dataclass(frozen=True, slots=True)
class CategoryMetrics:
    """Precision, recall, and F1 for one attack category."""

    category: str
    support: int
    predicted: int
    true_positives: int
    precision: float | None
    recall: float | None
    f1: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "category": self.category,
            "support": self.support,
            "predicted": self.predicted,
            "true_positives": self.true_positives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    """Every event-level and alert-level metric for one split."""

    split: str
    event_count: int
    benign_count: int
    malicious_count: int
    false_positive_rate: float | None
    detection_rate: float | None
    per_scenario_detection_rate: dict[str, float | None]
    category_metrics: dict[str, CategoryMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    confusion_matrix_classes: list[str]
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    weighted_precision: float | None
    weighted_recall: float | None
    weighted_f1: float | None
    alerts_per_thousand_events: float | None
    alert_precision: float | None
    alert_recall: float | None
    duplicate_reduction_ratio: float | None
    insufficient_data_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "split": self.split,
            "event_count": self.event_count,
            "benign_count": self.benign_count,
            "malicious_count": self.malicious_count,
            "false_positive_rate": self.false_positive_rate,
            "detection_rate": self.detection_rate,
            "per_scenario_detection_rate": self.per_scenario_detection_rate,
            "category_metrics": {
                key: value.to_dict()
                for key, value in sorted(self.category_metrics.items())
            },
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_classes": self.confusion_matrix_classes,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "alerts_per_thousand_events": self.alerts_per_thousand_events,
            "alert_precision": self.alert_precision,
            "alert_recall": self.alert_recall,
            "duplicate_reduction_ratio": self.duplicate_reduction_ratio,
            "insufficient_data_rate": self.insufficient_data_rate,
        }


@dataclass(frozen=True, slots=True)
class CampaignMetrics:
    """Campaign-level coverage and latency."""

    available: bool
    campaign_count: int
    detected_campaign_count: int
    coverage: float | None
    false_negative_count: int
    time_to_first_detection_seconds: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "available": self.available,
            "campaign_count": self.campaign_count,
            "detected_campaign_count": self.detected_campaign_count,
            "coverage": self.coverage,
            "false_negative_count": self.false_negative_count,
            "time_to_first_detection_seconds": self.time_to_first_detection_seconds,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """The complete evaluation of one detection run against ground truth."""

    synthetic_caveat: str
    overall: SplitMetrics
    by_split: dict[str, SplitMetrics]
    novel_holdout: SplitMetrics | None
    campaign_metrics: CampaignMetrics
    threshold_optimization_performed: bool
    configuration_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping."""
        return {
            "synthetic_caveat": self.synthetic_caveat,
            "overall": self.overall.to_dict(),
            "by_split": {
                key: value.to_dict() for key, value in sorted(self.by_split.items())
            },
            "novel_holdout": (
                None if self.novel_holdout is None else self.novel_holdout.to_dict()
            ),
            "campaign_metrics": self.campaign_metrics.to_dict(),
            "threshold_optimization_performed": self.threshold_optimization_performed,
            "configuration_fingerprint": self.configuration_fingerprint,
        }


# ---------------------------------------------------------------------------
# Arithmetic with explicit null semantics
# ---------------------------------------------------------------------------


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return a ratio, or ``None`` when the denominator is empty.

    ``None`` rather than zero: a rate over no observations is unmeasured, and
    reporting zero would claim a measurement nobody made.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    """Return the harmonic mean of precision and recall, or ``None``."""
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` over an empty sequence."""
    if not values:
        return None
    return math.fsum(values) / len(values)


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Return a latency distribution, with nulls when nothing was detected."""
    if not values:
        return dict.fromkeys(("min", "median", "mean", "p90", "max"))
    ordered = sorted(values)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        return ordered[min(count - 1, max(0, math.ceil(fraction * count) - 1))]

    return {
        "min": ordered[0],
        "median": percentile(0.5),
        "mean": math.fsum(ordered) / count,
        "p90": percentile(0.9),
        "max": ordered[-1],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_detection_run(
    *,
    detections: Sequence[FiredDetection],
    assessments: Sequence[RiskAssessment],
    alerts: Sequence[SecurityAlert],
    labels: Sequence[LabelRecord],
    splits: Sequence[SplitRecord],
    campaigns: Sequence[CampaignRecord] | None = None,
    configuration_fingerprint: str = "",
) -> EvaluationReport:
    """Evaluate a detection run against ground truth.

    Input row order cannot affect any result: every join is by identifier into
    a dictionary, and every reported collection is sorted before it is emitted.
    """
    label_by_event = {record.event_id: record for record in labels}
    split_by_event = {record.event_id: record.split for record in splits}
    detections_by_anchor: dict[str, list[FiredDetection]] = {}
    for detection in detections:
        detections_by_anchor.setdefault(detection.anchor_event_id, []).append(detection)

    scored = [
        assessment
        for assessment in assessments
        if assessment.anchor_event_id in label_by_event
    ]

    overall = _metrics_for(
        "overall", scored, label_by_event, detections_by_anchor, alerts, len(detections)
    )

    by_split: dict[str, SplitMetrics] = {}
    for split in sorted({split_by_event.get(a.anchor_event_id, "") for a in scored}):
        if not split or split == NOVEL_HOLDOUT_SPLIT:
            continue
        subset = [
            assessment
            for assessment in scored
            if split_by_event.get(assessment.anchor_event_id) == split
        ]
        by_split[split] = _metrics_for(
            split,
            subset,
            label_by_event,
            detections_by_anchor,
            alerts,
            sum(len(detections_by_anchor.get(a.anchor_event_id, [])) for a in subset),
        )

    holdout_subset = [
        assessment
        for assessment in scored
        if split_by_event.get(assessment.anchor_event_id) == NOVEL_HOLDOUT_SPLIT
    ]
    novel_holdout = (
        None
        if not holdout_subset
        else _metrics_for(
            NOVEL_HOLDOUT_SPLIT,
            holdout_subset,
            label_by_event,
            detections_by_anchor,
            alerts,
            sum(
                len(detections_by_anchor.get(a.anchor_event_id, []))
                for a in holdout_subset
            ),
        )
    )

    return EvaluationReport(
        synthetic_caveat=SYNTHETIC_CAVEAT,
        overall=overall,
        by_split=by_split,
        novel_holdout=novel_holdout,
        campaign_metrics=_campaign_metrics(
            scored, label_by_event, detections_by_anchor, campaigns
        ),
        # A literal ``False``, and the only place this concept appears: no code
        # path in this package reads a metric and writes a threshold.
        threshold_optimization_performed=False,
        configuration_fingerprint=configuration_fingerprint,
    )


def _metrics_for(
    split: str,
    assessments: Sequence[RiskAssessment],
    labels: Mapping[str, LabelRecord],
    detections_by_anchor: Mapping[str, Sequence[FiredDetection]],
    alerts: Sequence[SecurityAlert],
    detection_count: int,
) -> SplitMetrics:
    """Compute every metric for one subset of assessments."""
    benign = [a for a in assessments if not labels[a.anchor_event_id].malicious]
    malicious = [a for a in assessments if labels[a.anchor_event_id].malicious]

    false_positives = sum(1 for a in benign if a.fired_rule_count > 0)
    true_positives = sum(1 for a in malicious if a.fired_rule_count > 0)

    per_scenario: dict[str, float | None] = {}
    for scenario in sorted({labels[a.anchor_event_id].attack_class for a in malicious}):
        subset = [
            a for a in malicious if labels[a.anchor_event_id].attack_class == scenario
        ]
        per_scenario[scenario] = _ratio(
            sum(1 for a in subset if a.fired_rule_count > 0), len(subset)
        )

    category_metrics, matrix, classes = _category_metrics(assessments, labels)
    supports = {key: value.support for key, value in category_metrics.items()}
    total_support = sum(supports.values())

    alert_precision, alert_recall = _alert_metrics(
        alerts, assessments, labels, detections_by_anchor
    )

    return SplitMetrics(
        split=split,
        event_count=len(assessments),
        benign_count=len(benign),
        malicious_count=len(malicious),
        false_positive_rate=_ratio(false_positives, len(benign)),
        detection_rate=_ratio(true_positives, len(malicious)),
        per_scenario_detection_rate=per_scenario,
        category_metrics=category_metrics,
        confusion_matrix=matrix,
        confusion_matrix_classes=classes,
        macro_precision=_mean(
            [m.precision for m in category_metrics.values() if m.precision is not None]
        ),
        macro_recall=_mean(
            [m.recall for m in category_metrics.values() if m.recall is not None]
        ),
        macro_f1=_mean([m.f1 for m in category_metrics.values() if m.f1 is not None]),
        weighted_precision=_weighted(
            category_metrics, supports, total_support, "precision"
        ),
        weighted_recall=_weighted(category_metrics, supports, total_support, "recall"),
        weighted_f1=_weighted(category_metrics, supports, total_support, "f1"),
        alerts_per_thousand_events=(
            None if not assessments else 1000.0 * len(alerts) / len(assessments)
        ),
        alert_precision=alert_precision,
        alert_recall=alert_recall,
        duplicate_reduction_ratio=_ratio(detection_count, len(alerts)),
        insufficient_data_rate=_ratio(
            math.fsum(a.insufficient_data_count for a in assessments), len(assessments)
        ),
    )


def _weighted(
    metrics: Mapping[str, CategoryMetrics],
    supports: Mapping[str, int],
    total: int,
    field_name: str,
) -> float | None:
    """Return the support-weighted mean of one metric, or ``None``."""
    if total == 0:
        return None
    weighted = 0.0
    covered = 0
    for key, metric in metrics.items():
        value = getattr(metric, field_name)
        if value is None:
            continue
        weighted += value * supports[key]
        covered += supports[key]
    return None if covered == 0 else weighted / covered


def _category_metrics(
    assessments: Sequence[RiskAssessment], labels: Mapping[str, LabelRecord]
) -> tuple[dict[str, CategoryMetrics], dict[str, dict[str, int]], list[str]]:
    """Compute per-category precision, recall, F1, and the confusion matrix.

    Class order is deterministic: the scenario names a rule can claim, sorted,
    followed by ``none``.  Fixing the order here means a matrix rendered twice
    reads the same twice, and that two runs are directly comparable.
    """
    classes = sorted({str(value) for value in CATEGORY_TO_SCENARIO.values()})
    classes.append(_NO_CATEGORY)

    matrix: dict[str, dict[str, int]] = {
        actual: dict.fromkeys(classes, 0) for actual in classes
    }
    for assessment in assessments:
        actual_class = _actual_class(labels[assessment.anchor_event_id], classes)
        predicted_class = _predicted_class(assessment, classes)
        matrix[actual_class][predicted_class] += 1

    metrics: dict[str, CategoryMetrics] = {}
    for name in classes:
        if name == _NO_CATEGORY:
            continue
        true_positives = matrix[name][name]
        predicted_count = sum(matrix[row][name] for row in classes)
        support = sum(matrix[name].values())
        precision = _ratio(true_positives, predicted_count)
        recall = _ratio(true_positives, support)
        metrics[name] = CategoryMetrics(
            category=name,
            support=support,
            predicted=predicted_count,
            true_positives=true_positives,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
        )
    return metrics, matrix, classes


def _actual_class(label: LabelRecord, classes: Sequence[str]) -> str:
    """Return the ground-truth class for one event.

    A benign event, and a malicious event of a scenario no rule claims, both
    map to ``none`` -- the detector is not expected to assign them a category.
    """
    if not label.malicious:
        return _NO_CATEGORY
    return label.attack_class if label.attack_class in classes else _NO_CATEGORY


def _predicted_class(assessment: RiskAssessment, classes: Sequence[str]) -> str:
    """Return the class the detector assigned to one event."""
    category = assessment.primary_attack_category
    if category is None:
        return _NO_CATEGORY
    scenario = CATEGORY_TO_SCENARIO.get(category)
    if scenario is None:
        return _NO_CATEGORY
    return str(scenario) if str(scenario) in classes else _NO_CATEGORY


def _alert_metrics(
    alerts: Sequence[SecurityAlert],
    assessments: Sequence[RiskAssessment],
    labels: Mapping[str, LabelRecord],
    detections_by_anchor: Mapping[str, Sequence[FiredDetection]],
) -> tuple[float | None, float | None]:
    """Return alert precision and alert recall.

    An alert counts as a true positive when the majority of the assessments in
    its category and time window are malicious.  Exact membership would require
    replaying grouping; the window-and-category approximation is stated here
    rather than presented as exact.
    """
    if not alerts:
        return None, None

    in_scope = {a.anchor_event_id for a in assessments}
    correct = 0
    covered_malicious: set[str] = set()
    for alert in alerts:
        members = [
            assessment
            for assessment in assessments
            if assessment.primary_attack_category is not None
            and str(assessment.primary_attack_category) == str(alert.attack_category)
            and alert.first_seen <= assessment.anchor_event_time <= alert.last_seen
        ]
        if not members:
            continue
        malicious = [m for m in members if labels[m.anchor_event_id].malicious]
        if len(malicious) * 2 > len(members):
            correct += 1
        covered_malicious.update(m.anchor_event_id for m in malicious)

    detectable = {
        anchor
        for anchor in in_scope
        if labels[anchor].malicious and detections_by_anchor.get(anchor)
    }
    return (
        _ratio(correct, len(alerts)),
        _ratio(len(covered_malicious & detectable), len(detectable)),
    )


def _campaign_metrics(
    assessments: Sequence[RiskAssessment],
    labels: Mapping[str, LabelRecord],
    detections_by_anchor: Mapping[str, Sequence[FiredDetection]],
    campaigns: Sequence[CampaignRecord] | None,
) -> CampaignMetrics:
    """Compute campaign coverage and time to first detection.

    A campaign counts as detected once, however many of its events fired, so
    repeated detections cannot inflate coverage.  Without a campaign table the
    metrics are reported as unavailable rather than fabricated.
    """
    if not campaigns:
        return CampaignMetrics(
            available=False,
            campaign_count=0,
            detected_campaign_count=0,
            coverage=None,
            false_negative_count=0,
            time_to_first_detection_seconds=_latency_summary([]),
        )

    campaign_by_event = {record.event_id: record.campaign_id for record in campaigns}
    by_campaign: dict[str, list[RiskAssessment]] = {}
    for assessment in assessments:
        label = labels.get(assessment.anchor_event_id)
        campaign = campaign_by_event.get(assessment.anchor_event_id)
        if campaign is None or label is None or not label.malicious:
            continue
        by_campaign.setdefault(campaign, []).append(assessment)

    detected = 0
    latencies: list[float] = []
    for campaign in sorted(by_campaign):
        members = by_campaign[campaign]
        first_event = min(member.anchor_event_time for member in members)
        fired = [
            member
            for member in members
            if detections_by_anchor.get(member.anchor_event_id)
        ]
        if not fired:
            continue
        detected += 1
        first_detection = min(member.anchor_event_time for member in fired)
        latencies.append((first_detection - first_event).total_seconds())

    return CampaignMetrics(
        available=True,
        campaign_count=len(by_campaign),
        detected_campaign_count=detected,
        coverage=_ratio(detected, len(by_campaign)),
        false_negative_count=len(by_campaign) - detected,
        time_to_first_detection_seconds=_latency_summary(latencies),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def report_to_json(report: EvaluationReport, *, indent: int = 2) -> str:
    """Render the evaluation report as JSON."""
    return json.dumps(report.to_dict(), indent=indent, sort_keys=True, default=str)


def _fmt(value: float | int | None, *, digits: int = 4) -> str:
    """Render a metric, or ``unavailable`` when its denominator was empty."""
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _split_table(metrics: SplitMetrics) -> list[str]:
    """Render one split's headline metrics as a Markdown table."""
    return [
        "",
        f"### `{metrics.split}`",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Events | {_fmt(metrics.event_count)} |",
        f"| Benign | {_fmt(metrics.benign_count)} |",
        f"| Malicious | {_fmt(metrics.malicious_count)} |",
        f"| False-positive rate | {_fmt(metrics.false_positive_rate)} |",
        f"| Detection rate | {_fmt(metrics.detection_rate)} |",
        f"| Macro F1 | {_fmt(metrics.macro_f1)} |",
        f"| Weighted F1 | {_fmt(metrics.weighted_f1)} |",
        f"| Alerts per 1,000 events | {_fmt(metrics.alerts_per_thousand_events)} |",
        f"| Alert precision | {_fmt(metrics.alert_precision)} |",
        f"| Alert recall | {_fmt(metrics.alert_recall)} |",
        f"| Duplicate-reduction ratio | {_fmt(metrics.duplicate_reduction_ratio)} |",
        f"| Insufficient-data rate | {_fmt(metrics.insufficient_data_rate)} |",
    ]


def report_to_markdown(report: EvaluationReport) -> str:
    """Render the evaluation report as Markdown."""
    lines: list[str] = [
        "# Rule Evaluation Report",
        "",
        f"> {report.synthetic_caveat}",
        "",
        "Aggregate metrics only. This report contains no event rows, "
        "identifiers, pseudonyms, entity-scope values, or absolute paths.",
        "",
        "| Property | Value |",
        "|--------|-------|",
        f"| Threshold optimization performed | "
        f"{report.threshold_optimization_performed} |",
        f"| Configuration fingerprint | "
        f"`{report.configuration_fingerprint or 'n/a'}` |",
        "",
        "## Overall",
    ]
    lines.extend(_split_table(report.overall))

    if report.by_split:
        lines.extend(["", "## By split"])
        for name in sorted(report.by_split):
            lines.extend(_split_table(report.by_split[name]))

    lines.extend(["", "## Novel-anomaly holdout", ""])
    if report.novel_holdout is None:
        lines.append("No novel-anomaly holdout events were present in this evaluation.")
    else:
        lines.append(
            "Reported separately. These events are evaluation-only and are "
            "never folded into the supervised category metrics."
        )
        lines.extend(_split_table(report.novel_holdout))

    campaign = report.campaign_metrics
    lines.extend(["", "## Campaigns", "", "| Metric | Value |", "|--------|-------|"])
    if not campaign.available:
        lines.append("| Availability | unavailable (no campaign table supplied) |")
    else:
        lines.extend(
            [
                f"| Campaigns | {_fmt(campaign.campaign_count)} |",
                f"| Detected | {_fmt(campaign.detected_campaign_count)} |",
                f"| Coverage | {_fmt(campaign.coverage)} |",
                f"| Undetected | {_fmt(campaign.false_negative_count)} |",
                f"| Median time to first detection (s) | "
                f"{_fmt(campaign.time_to_first_detection_seconds['median'])} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Category metrics (overall)",
            "",
            "| Category | Support | Predicted | Precision | Recall | F1 |",
            "|--------|-------|-------|-------|-------|-------|",
        ]
    )
    for name in sorted(report.overall.category_metrics):
        metric = report.overall.category_metrics[name]
        lines.append(
            f"| `{name}` | {metric.support:,} | {metric.predicted:,} | "
            f"{_fmt(metric.precision)} | {_fmt(metric.recall)} | "
            f"{_fmt(metric.f1)} |"
        )

    return "\n".join(lines) + "\n"
