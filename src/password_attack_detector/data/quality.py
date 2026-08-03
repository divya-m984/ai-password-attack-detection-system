"""Data quality report generation for canonical authentication-event datasets.

Produces JSON and Markdown reports that summarise dataset statistics without
exposing any event rows, pseudonymous values, raw identifiers, file-system
paths, home directories, or secrets.

Public API
----------
- ``generate_quality_report``  -- generate a ``QualityReport`` from a Parquet file
- ``QualityReport``            -- all quality statistics in one frozen dataclass
- ``report_to_json``           -- serialise ``QualityReport`` to JSON string
- ``report_to_markdown``       -- render ``QualityReport`` as Markdown

Report sections
---------------
Row count, temporal range, event-rate summary, duplicate count, per-column
null rates, unique entity counts, auth-method / outcome / failure-reason /
MFA / country / client-type distributions, response-time summary, schema
validation result, sensitive-field scan, GT leakage scan, quality warnings.

Optional ground-truth section (only when *gt_path* is explicitly supplied):
scenario / malicious / supervised-training-eligibility / campaign-stage
distributions.

Privacy guarantees
------------------
- No event rows, no pseudonym values, no raw locations.
- Only aggregate counts and distributions are reported.
- File paths (absolute or relative) are never included.
- Secrets and pseudonymization keys are never included.
- Home directory paths are never included.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    CampaignStage,
    ClientType,
    FailureReason,
    MFAOutcome,
    ScenarioType,
)
from password_attack_detector.data.privacy import scan_prohibited_keys
from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS
from password_attack_detector.data.validation import DatasetValidator
from password_attack_detector.exceptions import DataValidationError

__all__ = [
    "QualityReport",
    "generate_quality_report",
    "report_to_json",
    "report_to_markdown",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_AUTH_METHODS: frozenset[str] = frozenset(m.value for m in AuthMethod)
_VALID_AUTH_OUTCOMES: frozenset[str] = frozenset(o.value for o in AuthOutcome)
_VALID_FAILURE_REASONS: frozenset[str] = frozenset(r.value for r in FailureReason)
_VALID_MFA_OUTCOMES: frozenset[str] = frozenset(o.value for o in MFAOutcome)
_VALID_CLIENT_TYPES: frozenset[str] = frozenset(t.value for t in ClientType)
_VALID_SCENARIOS: frozenset[str] = frozenset(s.value for s in ScenarioType)
_VALID_STAGES: frozenset[str] = frozenset(s.value for s in CampaignStage)

# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityReport:
    """Aggregated data quality statistics for one canonical dataset.

    All fields are privacy-safe aggregates; no row-level data is stored.
    """

    # ---- Dataset metadata ----
    schema_version: str | None
    row_count: int
    duplicate_event_id_count: int

    # ---- Temporal range ----
    earliest_event_time: datetime | None
    latest_event_time: datetime | None
    event_rate_per_hour: float | None

    # ---- Null rates per column ----
    null_rates: dict[str, float]

    # ---- Unique entity counts ----
    unique_user_count: int
    unique_source_count: int
    unique_device_count: int
    unique_application_count: int

    # ---- Distributions (column_value -> count) ----
    auth_method_distribution: dict[str, int]
    auth_outcome_distribution: dict[str, int]
    failure_reason_distribution: dict[str, int]
    mfa_outcome_distribution: dict[str, int]
    country_distribution: dict[str, int]
    client_type_distribution: dict[str, int]

    # ---- Response time summary ----
    response_time_min_ms: float | None
    response_time_max_ms: float | None
    response_time_mean_ms: float | None
    response_time_p50_ms: float | None
    response_time_p95_ms: float | None

    # ---- Validation ----
    validation_status: str
    validation_error_count: int
    validation_warning_count: int

    # ---- Sensitive-field scan ----
    sensitive_fields_found: list[str]
    gt_leakage_columns_found: list[str]

    # ---- Quality warnings ----
    quality_warnings: list[str]

    # ---- Optional ground-truth section (None when no GT file supplied) ----
    gt_row_count: int | None = None
    scenario_distribution: dict[str, int] | None = None
    malicious_count: int | None = None
    supervised_training_eligible_count: int | None = None
    campaign_stage_distribution: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def generate_quality_report(
    events_path: Path,
    *,
    gt_path: Path | None = None,
) -> QualityReport:
    """Generate a ``QualityReport`` from a canonical events Parquet file.

    Parameters
    ----------
    events_path:
        Path to the canonical events Parquet file.
    gt_path:
        Optional path to the ground-truth labels Parquet file.  When supplied,
        the report includes scenario, malicious, and eligibility distributions.

    Returns
    -------
    QualityReport
        All quality statistics as a frozen dataclass.

    Raises
    ------
    DataValidationError
        If the events file is empty or cannot be read.
    """
    try:
        events_table = pq.read_table(events_path)
        df: pd.DataFrame = events_table.to_pandas()
    except Exception as exc:
        raise DataValidationError(
            f"Cannot read events file ({type(exc).__name__})"
        ) from exc

    if len(df) == 0:
        raise DataValidationError("Events dataset is empty; cannot generate report")

    return _build_report(df, gt_path=gt_path)


def report_to_json(report: QualityReport, *, indent: int = 2) -> str:
    """Serialise *report* to a JSON string.

    Datetime values are rendered as ISO-8601 UTC strings.  All keys are
    snake_case.  The output never contains absolute paths, home directories,
    secrets, or raw identifiers.

    Parameters
    ----------
    report:
        Quality report to serialise.
    indent:
        JSON indentation level (default 2).
    """
    return json.dumps(_report_to_dict(report), indent=indent, default=_json_default)


def report_to_markdown(report: QualityReport) -> str:
    """Render *report* as a GitHub-flavoured Markdown string.

    The Markdown output contains the same information as the JSON report
    formatted as human-readable sections and tables.  No absolute paths,
    secrets, or raw identifiers are included.

    Parameters
    ----------
    report:
        Quality report to render.
    """
    lines: list[str] = []
    _append = lines.append

    _append("# Data Quality Report")
    _append("")

    # Dataset overview
    _append("## Dataset Overview")
    _append("")
    _append("| Metric | Value |")
    _append("|--------|-------|")
    _append(f"| Schema version | `{report.schema_version or 'mixed/unknown'}` |")
    _append(f"| Row count | {report.row_count:,} |")
    _append(f"| Duplicate event IDs | {report.duplicate_event_id_count:,} |")
    _append(f"| Unique users | {report.unique_user_count:,} |")
    _append(f"| Unique sources | {report.unique_source_count:,} |")
    _append(f"| Unique devices | {report.unique_device_count:,} |")
    _append(f"| Unique applications | {report.unique_application_count:,} |")
    _append("")

    # Temporal range
    _append("## Temporal Range")
    _append("")
    if report.earliest_event_time and report.latest_event_time:
        _append("| | Timestamp |")
        _append("|-|-----------|")
        _append(f"| Earliest | `{report.earliest_event_time.isoformat()}` |")
        _append(f"| Latest | `{report.latest_event_time.isoformat()}` |")
        if report.event_rate_per_hour is not None:
            _append(f"| Event rate | {report.event_rate_per_hour:.2f} events/hour |")
    else:
        _append("_No timestamp data available._")
    _append("")

    # Validation
    _append("## Validation")
    _append("")
    _append(f"**Status:** `{report.validation_status}`  ")
    _append(
        f"Errors: {report.validation_error_count} | "
        f"Warnings: {report.validation_warning_count}"
    )
    _append("")

    # Distributions
    _append_distribution_section(
        lines, "Auth Method Distribution", report.auth_method_distribution
    )
    _append_distribution_section(
        lines, "Auth Outcome Distribution", report.auth_outcome_distribution
    )
    _append_distribution_section(
        lines, "Failure Reason Distribution", report.failure_reason_distribution
    )
    _append_distribution_section(
        lines, "MFA Outcome Distribution", report.mfa_outcome_distribution
    )
    _append_distribution_section(
        lines, "Country Distribution (top 10)", report.country_distribution
    )
    _append_distribution_section(
        lines, "Client Type Distribution", report.client_type_distribution
    )

    # Response time
    _append("## Response Time (ms)")
    _append("")
    if report.response_time_min_ms is not None:
        _append("| Metric | Value |")
        _append("|--------|-------|")
        _append(f"| Min | {report.response_time_min_ms:.0f} |")
        _append(f"| Mean | {report.response_time_mean_ms:.0f} |")
        _append(f"| Median (p50) | {report.response_time_p50_ms:.0f} |")
        _append(f"| p95 | {report.response_time_p95_ms:.0f} |")
        _append(f"| Max | {report.response_time_max_ms:.0f} |")
    else:
        _append("_No response time data available._")
    _append("")

    # Null rates
    _append("## Null Rates")
    _append("")
    if report.null_rates:
        _append("| Column | Null Rate |")
        _append("|--------|-----------|")
        for col, rate in sorted(report.null_rates.items()):
            _append(f"| `{col}` | {rate:.1%} |")
    _append("")

    # Security scan
    _append("## Security Scan")
    _append("")
    if report.sensitive_fields_found:
        _append(
            f"**WARNING:** {len(report.sensitive_fields_found)} sensitive field(s) detected."
        )
    else:
        _append("No sensitive fields detected.")
    if report.gt_leakage_columns_found:
        _append(
            f"**WARNING:** {len(report.gt_leakage_columns_found)} "
            f"ground-truth leakage column(s) detected."
        )
    else:
        _append("No ground-truth leakage columns detected.")
    _append("")

    # Quality warnings
    if report.quality_warnings:
        _append("## Quality Warnings")
        _append("")
        for w in report.quality_warnings:
            _append(f"- {w}")
        _append("")

    # Optional ground-truth section
    if report.gt_row_count is not None:
        _append("## Ground Truth")
        _append("")
        _append("| Metric | Value |")
        _append("|--------|-------|")
        _append(f"| GT row count | {report.gt_row_count:,} |")
        _append(f"| Malicious events | {report.malicious_count:,} |")
        _append(
            f"| Training-eligible events | "
            f"{report.supervised_training_eligible_count:,} |"
        )
        _append("")
        if report.scenario_distribution:
            _append_distribution_section(
                lines, "Scenario Distribution", report.scenario_distribution
            )
        if report.campaign_stage_distribution:
            _append_distribution_section(
                lines, "Campaign Stage Distribution", report.campaign_stage_distribution
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_report(df: pd.DataFrame, *, gt_path: Path | None) -> QualityReport:
    """Build a ``QualityReport`` from an in-memory events DataFrame."""
    cols = {str(c) for c in df.columns}
    warnings: list[str] = []

    # Duplicate count
    duplicate_count = (
        int(df["event_id"].duplicated().sum()) if "event_id" in cols else 0
    )
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate event_id values detected")

    # Temporal range
    earliest: datetime | None = None
    latest: datetime | None = None
    event_rate: float | None = None
    if "event_time" in cols and pd.api.types.is_datetime64_any_dtype(df["event_time"]):
        et = df["event_time"]
        earliest = et.min().to_pydatetime()
        latest = et.max().to_pydatetime()
        duration_hours = (latest - earliest).total_seconds() / 3600
        event_rate = len(df) / duration_hours if duration_hours > 0 else float(len(df))

    # Null rates
    null_rates: dict[str, float] = {
        str(c): float(df[c].isna().mean()) for c in df.columns
    }

    # Unique entity counts
    unique_users = int(df["user_id"].nunique()) if "user_id" in cols else 0
    unique_sources = int(df["source_id"].nunique()) if "source_id" in cols else 0
    unique_devices = int(df["device_id"].nunique()) if "device_id" in cols else 0
    unique_apps = int(df["application_id"].nunique()) if "application_id" in cols else 0

    # Distributions
    auth_method_dist = _distribution(df, "authentication_method")
    auth_outcome_dist = _distribution(df, "authentication_outcome")
    failure_reason_dist = _distribution(df, "failure_reason")
    mfa_outcome_dist = _distribution(df, "mfa_outcome")
    country_dist = _top_n_distribution(df, "country_code", n=10)
    client_type_dist = _distribution(df, "client_type")

    # Response time summary
    rt_min = rt_max = rt_mean = rt_p50 = rt_p95 = None
    if "response_time_ms" in cols:
        rt_col = df["response_time_ms"].dropna()
        if len(rt_col) > 0:
            rt_min = float(rt_col.min())
            rt_max = float(rt_col.max())
            rt_mean = float(rt_col.mean())
            rt_p50 = float(rt_col.quantile(0.50))
            rt_p95 = float(rt_col.quantile(0.95))

    # Schema validation
    validator = DatasetValidator()
    val_result = validator.validate_dataframe(df)
    schema_ver = val_result.schema_version

    # Sensitive-field scan (column names only, not values)
    header_dict: dict[str, Any] = {str(c): "" for c in df.columns}
    sensitive_found = scan_prohibited_keys(header_dict)
    gt_leakage_found = sorted(PROHIBITED_GT_COLUMNS & cols)

    if sensitive_found:
        warnings.append(
            f"{len(sensitive_found)} sensitive field name(s) detected in columns"
        )
    if gt_leakage_found:
        warnings.append(
            f"{len(gt_leakage_found)} ground-truth leakage column(s) detected"
        )

    # Ground-truth section
    gt_row_count: int | None = None
    scenario_dist: dict[str, int] | None = None
    malicious_count: int | None = None
    eligible_count: int | None = None
    stage_dist: dict[str, int] | None = None

    if gt_path is not None:
        try:
            gt_table = pq.read_table(gt_path)
            gt_df: pd.DataFrame = gt_table.to_pandas()
            gt_row_count = len(gt_df)
            scenario_dist = _distribution(gt_df, "scenario")
            malicious_count = (
                int(gt_df["malicious"].sum()) if "malicious" in gt_df.columns else 0
            )
            eligible_count = (
                int(gt_df["supervised_training_eligible"].sum())
                if "supervised_training_eligible" in gt_df.columns
                else 0
            )
            stage_dist = _distribution(gt_df, "campaign_stage")
        except Exception as exc:
            warnings.append(
                f"Ground-truth file could not be read ({type(exc).__name__})"
            )

    return QualityReport(
        schema_version=schema_ver,
        row_count=len(df),
        duplicate_event_id_count=duplicate_count,
        earliest_event_time=earliest,
        latest_event_time=latest,
        event_rate_per_hour=event_rate,
        null_rates=null_rates,
        unique_user_count=unique_users,
        unique_source_count=unique_sources,
        unique_device_count=unique_devices,
        unique_application_count=unique_apps,
        auth_method_distribution=auth_method_dist,
        auth_outcome_distribution=auth_outcome_dist,
        failure_reason_distribution=failure_reason_dist,
        mfa_outcome_distribution=mfa_outcome_dist,
        country_distribution=country_dist,
        client_type_distribution=client_type_dist,
        response_time_min_ms=rt_min,
        response_time_max_ms=rt_max,
        response_time_mean_ms=rt_mean,
        response_time_p50_ms=rt_p50,
        response_time_p95_ms=rt_p95,
        validation_status=str(val_result.status),
        validation_error_count=len(val_result.errors),
        validation_warning_count=len(val_result.warnings),
        sensitive_fields_found=sorted(sensitive_found),
        gt_leakage_columns_found=gt_leakage_found,
        quality_warnings=warnings,
        gt_row_count=gt_row_count,
        scenario_distribution=scenario_dist,
        malicious_count=malicious_count,
        supervised_training_eligible_count=eligible_count,
        campaign_stage_distribution=stage_dist,
    )


def _distribution(df: pd.DataFrame, col: str) -> dict[str, int]:
    """Return a value-count dict for *col*, sorted by count descending."""
    if col not in df.columns:
        return {}
    counts = df[col].value_counts(dropna=True)
    return {str(k): int(v) for k, v in counts.items()}


def _top_n_distribution(df: pd.DataFrame, col: str, n: int) -> dict[str, int]:
    """Return the top *n* value counts for *col*."""
    if col not in df.columns:
        return {}
    counts = df[col].value_counts(dropna=True).head(n)
    return {str(k): int(v) for k, v in counts.items()}


def _report_to_dict(report: QualityReport) -> dict[str, Any]:
    """Convert report to a JSON-serialisable dict."""
    return {
        "schema_version": report.schema_version,
        "row_count": report.row_count,
        "duplicate_event_id_count": report.duplicate_event_id_count,
        "earliest_event_time": (
            report.earliest_event_time.isoformat()
            if report.earliest_event_time
            else None
        ),
        "latest_event_time": (
            report.latest_event_time.isoformat() if report.latest_event_time else None
        ),
        "event_rate_per_hour": report.event_rate_per_hour,
        "null_rates": report.null_rates,
        "unique_user_count": report.unique_user_count,
        "unique_source_count": report.unique_source_count,
        "unique_device_count": report.unique_device_count,
        "unique_application_count": report.unique_application_count,
        "auth_method_distribution": report.auth_method_distribution,
        "auth_outcome_distribution": report.auth_outcome_distribution,
        "failure_reason_distribution": report.failure_reason_distribution,
        "mfa_outcome_distribution": report.mfa_outcome_distribution,
        "country_distribution": report.country_distribution,
        "client_type_distribution": report.client_type_distribution,
        "response_time_min_ms": report.response_time_min_ms,
        "response_time_max_ms": report.response_time_max_ms,
        "response_time_mean_ms": report.response_time_mean_ms,
        "response_time_p50_ms": report.response_time_p50_ms,
        "response_time_p95_ms": report.response_time_p95_ms,
        "validation_status": report.validation_status,
        "validation_error_count": report.validation_error_count,
        "validation_warning_count": report.validation_warning_count,
        "sensitive_fields_found": report.sensitive_fields_found,
        "gt_leakage_columns_found": report.gt_leakage_columns_found,
        "quality_warnings": report.quality_warnings,
        "gt_row_count": report.gt_row_count,
        "scenario_distribution": report.scenario_distribution,
        "malicious_count": report.malicious_count,
        "supervised_training_eligible_count": report.supervised_training_eligible_count,
        "campaign_stage_distribution": report.campaign_stage_distribution,
    }


def _json_default(obj: object) -> object:
    """JSON fallback serialiser for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _append_distribution_section(
    lines: list[str], title: str, dist: dict[str, int]
) -> None:
    """Append a distribution table section to *lines*."""
    lines.append(f"## {title}")
    lines.append("")
    if dist:
        lines.append("| Value | Count |")
        lines.append("|-------|-------|")
        for val, count in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"| `{val}` | {count:,} |")
    else:
        lines.append("_No data._")
    lines.append("")
