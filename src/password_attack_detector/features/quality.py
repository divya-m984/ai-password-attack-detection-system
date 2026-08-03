"""Aggregate quality reporting for a feature dataset.

Follows the Phase 2 pattern: a frozen dataclass, a JSON renderer, and a
Markdown renderer built from ``| Metric | Value |`` tables.

**Everything here is an aggregate.**  The report never contains an event row,
an event identifier, a user/source/device/session pseudonym, a coordinate, a
secret, or an absolute path.  There is no feature-importance calculation
either: nothing in this phase trains a model, so nothing can rank features.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_TIME,
    FeatureCatalog,
    FeatureDType,
    LeakageClass,
)
from password_attack_detector.features.engine import FeatureFrame
from password_attack_detector.features.splitting import SplitAssignment

__all__ = [
    "FeatureQualityReport",
    "generate_feature_quality_report",
    "report_to_json",
    "report_to_markdown",
]

#: Variance below which a numeric column is called near-constant.
_NEAR_ZERO_VARIANCE = 1e-12


@dataclass(frozen=True)
class FeatureQualityReport:
    """Aggregate description of a feature dataset."""

    feature_schema_version: str
    row_count: int
    feature_count: int
    earliest_anchor_time: str | None
    latest_anchor_time: str | None
    feature_group_counts: dict[str, int]
    leakage_class_counts: dict[str, int]
    null_rates: dict[str, float]
    zero_variance_features: list[str]
    near_zero_variance_features: list[str]
    numeric_ranges: dict[str, dict[str, float]]
    infinite_value_count: int
    baseline_coverage: dict[str, float]
    novelty_rates: dict[str, float]
    geospatial_status_rates: dict[str, dict[str, float]]
    split_counts: dict[str, int]
    class_distribution_by_split: dict[str, dict[str, int]]
    exclusion_counts: dict[str, int]
    feature_fingerprint: str
    config_fingerprint: str
    catalog_fingerprint: str
    validation_result: dict[str, Any] | None = None
    leakage_audit_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping."""
        return {
            "feature_schema_version": self.feature_schema_version,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "earliest_anchor_time": self.earliest_anchor_time,
            "latest_anchor_time": self.latest_anchor_time,
            "feature_group_counts": self.feature_group_counts,
            "leakage_class_counts": self.leakage_class_counts,
            "null_rates": self.null_rates,
            "zero_variance_features": self.zero_variance_features,
            "near_zero_variance_features": self.near_zero_variance_features,
            "numeric_ranges": self.numeric_ranges,
            "infinite_value_count": self.infinite_value_count,
            "baseline_coverage": self.baseline_coverage,
            "novelty_rates": self.novelty_rates,
            "geospatial_status_rates": self.geospatial_status_rates,
            "split_counts": self.split_counts,
            "class_distribution_by_split": self.class_distribution_by_split,
            "exclusion_counts": self.exclusion_counts,
            "feature_fingerprint": self.feature_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "catalog_fingerprint": self.catalog_fingerprint,
            "validation_result": self.validation_result,
            "leakage_audit_result": self.leakage_audit_result,
        }


def _numeric_summary(values: list[Any]) -> dict[str, float] | None:
    """Return min/max/mean for a numeric column, ignoring nulls and non-finite."""
    finite = [
        float(v)
        for v in values
        if v is not None
        and not (isinstance(v, float) and (math.isinf(v) or math.isnan(v)))
    ]
    if not finite:
        return None
    return {
        "min": min(finite),
        "max": max(finite),
        "mean": sum(finite) / len(finite),
    }


def generate_feature_quality_report(
    frame: FeatureFrame,
    *,
    feature_fingerprint: str,
    assignments: list[SplitAssignment] | None = None,
    class_distribution_by_split: dict[str, dict[str, int]] | None = None,
    exclusion_counts: dict[str, int] | None = None,
    validation_result: dict[str, Any] | None = None,
    leakage_audit_result: dict[str, Any] | None = None,
) -> FeatureQualityReport:
    """Profile *frame* into an aggregate quality report."""
    catalog: FeatureCatalog = frame.catalog
    rows = frame.rows
    row_count = len(rows)

    columns: dict[str, list[Any]] = {
        name: [row[name] for row in rows] for name in catalog.column_order()
    }

    null_rates = {
        name: (sum(1 for v in values if v is None) / row_count if row_count else 0.0)
        for name, values in columns.items()
    }

    zero_variance: list[str] = []
    near_zero_variance: list[str] = []
    numeric_ranges: dict[str, dict[str, float]] = {}
    infinite_total = 0

    for spec in catalog.specs:
        values = columns[spec.name]
        observed = [v for v in values if v is not None]

        if spec.group.value != "key" and len(observed) > 1:
            distinct = len(set(observed))
            if distinct == 1:
                zero_variance.append(spec.name)

        if spec.dtype in (FeatureDType.FLOAT64, FeatureDType.INT64):
            infinite_total += sum(
                1 for v in observed if isinstance(v, float) and math.isinf(v)
            )
            summary = _numeric_summary(values)
            if summary is not None:
                numeric_ranges[spec.name] = summary
                spread = summary["max"] - summary["min"]
                if (
                    spec.name not in zero_variance
                    and 0.0 < spread <= _NEAR_ZERO_VARIANCE
                ):
                    near_zero_variance.append(spec.name)

    baseline_coverage: dict[str, float] = {
        str(name): (
            sum(1 for v in columns[name] if v is True) / row_count if row_count else 0.0
        )
        for name in ("user_in_baseline", "source_in_baseline")
        if name in columns
    }

    novelty_rates = {
        spec.name: (
            sum(1 for v in columns[spec.name] if v is True)
            / max(1, sum(1 for v in columns[spec.name] if v is not None))
        )
        for spec in catalog.specs
        if spec.name.startswith("is_new_")
    }

    geospatial_status_rates: dict[str, dict[str, float]] = {}
    for name in ("user_previous_success_geo__status", "implied_velocity__status"):
        if name not in columns:
            continue
        tally: dict[str, int] = {}
        for value in columns[name]:
            observed_status = str(value)
            tally[observed_status] = tally.get(observed_status, 0) + 1
        geospatial_status_rates[name] = {
            status: count / row_count for status, count in sorted(tally.items())
        }

    split_counts: dict[str, int] = {}
    if assignments is not None:
        for assignment in assignments:
            key = str(assignment.split)
            split_counts[key] = split_counts.get(key, 0) + 1

    times = [row[ANCHOR_EVENT_TIME] for row in rows]
    earliest = min(times) if times else None
    latest = max(times) if times else None

    return FeatureQualityReport(
        feature_schema_version=(
            str(rows[0]["feature_schema_version"]) if rows else "unknown"
        ),
        row_count=row_count,
        feature_count=len(catalog),
        earliest_anchor_time=_iso(earliest),
        latest_anchor_time=_iso(latest),
        feature_group_counts=catalog.group_counts(),
        leakage_class_counts={
            str(cls): len(catalog.specs_for_leakage_class(cls)) for cls in LeakageClass
        },
        null_rates=null_rates,
        zero_variance_features=sorted(zero_variance),
        near_zero_variance_features=sorted(near_zero_variance),
        numeric_ranges=numeric_ranges,
        infinite_value_count=infinite_total,
        baseline_coverage=baseline_coverage,
        novelty_rates=novelty_rates,
        geospatial_status_rates=geospatial_status_rates,
        split_counts=split_counts,
        class_distribution_by_split=class_distribution_by_split or {},
        exclusion_counts=exclusion_counts or {},
        feature_fingerprint=feature_fingerprint,
        config_fingerprint=catalog.config_fingerprint,
        catalog_fingerprint=catalog.fingerprint(),
        validation_result=validation_result,
        leakage_audit_result=leakage_audit_result,
    )


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def report_to_json(report: FeatureQualityReport, *, indent: int = 2) -> str:
    """Render the report as JSON."""
    return json.dumps(report.to_dict(), indent=indent, sort_keys=True, default=str)


def report_to_markdown(report: FeatureQualityReport) -> str:
    """Render the report as Markdown, in the Phase 2 table style."""
    lines: list[str] = [
        "# Feature Quality Report",
        "",
        "Aggregate statistics only. This report contains no event rows, "
        "identifiers, pseudonyms, coordinates, or absolute paths.",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Feature schema version | `{report.feature_schema_version}` |",
        f"| Feature snapshots | {report.row_count:,} |",
        f"| Declared features | {report.feature_count:,} |",
        f"| Earliest anchor | {report.earliest_anchor_time or 'n/a'} |",
        f"| Latest anchor | {report.latest_anchor_time or 'n/a'} |",
        f"| Infinite values | {report.infinite_value_count} |",
        f"| Zero-variance features | {len(report.zero_variance_features)} |",
        f"| Near-zero-variance features | {len(report.near_zero_variance_features)} |",
        f"| Feature fingerprint | `{report.feature_fingerprint}` |",
        f"| Configuration fingerprint | `{report.config_fingerprint}` |",
        f"| Catalog fingerprint | `{report.catalog_fingerprint}` |",
    ]

    _append_counts(lines, "Features by group", "Group", report.feature_group_counts)
    _append_counts(
        lines, "Features by leakage class", "Class", report.leakage_class_counts
    )

    if report.split_counts:
        _append_counts(lines, "Split sizes", "Split", report.split_counts)

    if report.class_distribution_by_split:
        lines.extend(
            [
                "",
                "## Class distribution by split",
                "",
                "| Split | Malicious | Benign |",
                "|--------|-------|-------|",
            ]
        )
        for split, distribution in sorted(report.class_distribution_by_split.items()):
            lines.append(
                f"| `{split}` | {distribution.get('malicious', 0):,} | "
                f"{distribution.get('benign', 0):,} |"
            )

    if report.exclusion_counts:
        _append_counts(lines, "Exclusions by reason", "Reason", report.exclusion_counts)

    if report.baseline_coverage:
        lines.extend(
            ["", "## Baseline coverage", "", "| Metric | Rate |", "|--------|-------|"]
        )
        for name, rate in sorted(report.baseline_coverage.items()):
            lines.append(f"| `{name}` | {rate:.3f} |")

    if report.novelty_rates:
        lines.extend(
            ["", "## Novelty rates", "", "| Feature | Rate |", "|--------|-------|"]
        )
        for name, rate in sorted(report.novelty_rates.items()):
            lines.append(f"| `{name}` | {rate:.3f} |")

    for name, status_rates in sorted(report.geospatial_status_rates.items()):
        lines.extend(
            [
                "",
                f"## `{name}`",
                "",
                "| Status | Rate |",
                "|--------|-------|",
            ]
        )
        for status, rate in sorted(status_rates.items()):
            lines.append(f"| `{status}` | {rate:.3f} |")

    sparse = sorted(
        ((name, rate) for name, rate in report.null_rates.items() if rate > 0.0),
        key=lambda item: (-item[1], item[0]),
    )[:20]
    if sparse:
        lines.extend(
            [
                "",
                "## Highest null rates",
                "",
                "| Feature | Null rate |",
                "|--------|-------|",
            ]
        )
        for name, rate in sparse:
            lines.append(f"| `{name}` | {rate:.3f} |")

    if report.validation_result is not None:
        lines.extend(
            [
                "",
                "## Validation",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Status | `{report.validation_result.get('status')}` |",
                f"| Errors | {len(report.validation_result.get('errors', []))} |",
                f"| Warnings | {len(report.validation_result.get('warnings', []))} |",
            ]
        )

    if report.leakage_audit_result is not None:
        lines.extend(
            [
                "",
                "## Leakage audit",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Status | `{report.leakage_audit_result.get('status')}` |",
                f"| Errors | {len(report.leakage_audit_result.get('errors', []))} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- These are descriptive statistics about a feature table. They say "
            "nothing about detection effectiveness.",
            "- No feature importance is calculated: this phase trains no model.",
            "- Synthetic data exercises the feature set but does not demonstrate "
            "real-world behaviour.",
            "- Zero-variance columns are reported, not removed. Whether to drop "
            "them is a later modelling decision.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_counts(
    lines: list[str], heading: str, label: str, counts: dict[str, int]
) -> None:
    """Append a two-column count table under a heading."""
    lines.extend(
        ["", f"## {heading}", "", f"| {label} | Count |", "|--------|-------|"]
    )
    for key, value in sorted(counts.items()):
        lines.append(f"| `{key}` | {value:,} |")
