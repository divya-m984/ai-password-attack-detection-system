"""Unit tests for password_attack_detector.data.quality."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from password_attack_detector.data.quality import (
    QualityReport,
    generate_quality_report,
    report_to_json,
    report_to_markdown,
)
from password_attack_detector.data.schemas import SCHEMA_VERSION
from password_attack_detector.data.serialization import (
    write_events_parquet,
    write_labels_parquet,
)
from password_attack_detector.data.synthetic.config import SyntheticConfig
from password_attack_detector.data.synthetic.generator import generate_dataset
from password_attack_detector.exceptions import DataValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = datetime(2024, 1, 1, tzinfo=UTC)


def _small_config(**kwargs: object) -> SyntheticConfig:
    defaults: dict[str, object] = {
        "seed": 42,
        "start_time": _START,
        "duration_hours": 2,
        "num_users": 3,
        "num_sources": 2,
        "num_devices": 4,
        "num_applications": 1,
        "events_per_hour": 5,
    }
    defaults.update(kwargs)
    return SyntheticConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def events_parquet(tmp_path: Path) -> Path:
    """Write a small synthetic events Parquet and return the path."""
    cfg = _small_config()
    result = generate_dataset(cfg)
    p = tmp_path / "events.parquet"
    write_events_parquet(result.events, p)
    return p


@pytest.fixture()
def labels_parquet(tmp_path: Path) -> Path:
    """Write small synthetic labels Parquet and return the path."""
    cfg = _small_config()
    result = generate_dataset(cfg)
    p = tmp_path / "labels.parquet"
    write_labels_parquet(result.labels, p)
    return p


@pytest.fixture()
def quality_report(events_parquet: Path) -> QualityReport:
    return generate_quality_report(events_parquet)


# ---------------------------------------------------------------------------
# generate_quality_report
# ---------------------------------------------------------------------------


class TestGenerateQualityReport:
    def test_returns_quality_report(self, events_parquet: Path) -> None:
        report = generate_quality_report(events_parquet)
        assert isinstance(report, QualityReport)

    def test_row_count_positive(self, quality_report: QualityReport) -> None:
        assert quality_report.row_count > 0

    def test_schema_version_detected(self, quality_report: QualityReport) -> None:
        assert quality_report.schema_version == SCHEMA_VERSION

    def test_empty_parquet_raises_data_validation_error(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"event_id": pa.array([], type=pa.string())})
        p = tmp_path / "empty.parquet"
        pq.write_table(table, p)
        with pytest.raises(DataValidationError):
            generate_quality_report(p)

    def test_nonexistent_file_raises_data_validation_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DataValidationError):
            generate_quality_report(tmp_path / "missing.parquet")

    def test_temporal_range_is_set(self, quality_report: QualityReport) -> None:
        assert quality_report.earliest_event_time is not None
        assert quality_report.latest_event_time is not None
        assert quality_report.earliest_event_time <= quality_report.latest_event_time

    def test_event_rate_positive(self, quality_report: QualityReport) -> None:
        assert quality_report.event_rate_per_hour is not None
        assert quality_report.event_rate_per_hour > 0

    def test_unique_user_count_positive(self, quality_report: QualityReport) -> None:
        assert quality_report.unique_user_count > 0

    def test_null_rates_populated(self, quality_report: QualityReport) -> None:
        assert len(quality_report.null_rates) > 0
        assert "event_time" in quality_report.null_rates

    def test_no_sensitive_fields_in_clean_data(
        self, quality_report: QualityReport
    ) -> None:
        assert quality_report.sensitive_fields_found == []

    def test_no_gt_leakage_in_clean_data(self, quality_report: QualityReport) -> None:
        assert quality_report.gt_leakage_columns_found == []

    def test_auth_method_distribution_populated(
        self, quality_report: QualityReport
    ) -> None:
        assert len(quality_report.auth_method_distribution) > 0

    def test_auth_outcome_distribution_populated(
        self, quality_report: QualityReport
    ) -> None:
        assert len(quality_report.auth_outcome_distribution) > 0

    def test_duplicate_count_zero_for_clean_data(
        self, quality_report: QualityReport
    ) -> None:
        assert quality_report.duplicate_event_id_count == 0

    def test_validation_status_set(self, quality_report: QualityReport) -> None:
        assert quality_report.validation_status in {"valid", "warning", "invalid"}


class TestGenerateQualityReportWithGT:
    def test_gt_section_none_when_no_gt_path(
        self, quality_report: QualityReport
    ) -> None:
        assert quality_report.gt_row_count is None
        assert quality_report.scenario_distribution is None

    def test_gt_section_populated_when_gt_path_supplied(
        self, events_parquet: Path, labels_parquet: Path
    ) -> None:
        report = generate_quality_report(events_parquet, gt_path=labels_parquet)
        assert report.gt_row_count is not None
        assert report.gt_row_count > 0
        assert report.scenario_distribution is not None

    def test_malicious_count_non_negative(
        self, events_parquet: Path, labels_parquet: Path
    ) -> None:
        report = generate_quality_report(events_parquet, gt_path=labels_parquet)
        assert report.malicious_count is not None
        assert report.malicious_count >= 0

    def test_supervised_training_eligible_count_non_negative(
        self, events_parquet: Path, labels_parquet: Path
    ) -> None:
        report = generate_quality_report(events_parquet, gt_path=labels_parquet)
        assert report.supervised_training_eligible_count is not None
        assert report.supervised_training_eligible_count >= 0

    def test_invalid_gt_path_adds_warning(
        self, events_parquet: Path, tmp_path: Path
    ) -> None:
        report = generate_quality_report(
            events_parquet, gt_path=tmp_path / "missing.parquet"
        )
        # Should add a warning but not raise
        assert any("Ground-truth" in w for w in report.quality_warnings)


class TestQualityReportContentsArePrivacySafe:
    def test_no_absolute_paths_in_report(self, quality_report: QualityReport) -> None:
        j = report_to_json(quality_report)
        # Home paths or root-absolute paths should not appear
        assert "/home/" not in j
        assert "/root/" not in j

    def test_no_raw_event_values_in_null_rates(
        self, quality_report: QualityReport
    ) -> None:
        # null_rates should only have column names, not data values
        for key in quality_report.null_rates:
            assert "u:" not in key  # user_id pseudonym prefix
            assert "s:" not in key
            assert "d:" not in key

    def test_distributions_contain_only_known_enum_values(
        self, quality_report: QualityReport
    ) -> None:
        for method in quality_report.auth_method_distribution:
            assert method in {
                "password",
                "mfa_totp",
                "mfa_sms",
                "mfa_email",
                "sso",
                "oauth2",
                "api_key",
                "certificate",
                "biometric",
                "passkey",
            }

    def test_report_is_frozen(self, quality_report: QualityReport) -> None:
        with pytest.raises((AttributeError, TypeError)):
            quality_report.row_count = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# report_to_json
# ---------------------------------------------------------------------------


class TestReportToJson:
    def test_returns_valid_json_string(self, quality_report: QualityReport) -> None:
        j = report_to_json(quality_report)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_json_contains_row_count(self, quality_report: QualityReport) -> None:
        parsed = json.loads(report_to_json(quality_report))
        assert "row_count" in parsed
        assert parsed["row_count"] == quality_report.row_count

    def test_json_contains_null_rates(self, quality_report: QualityReport) -> None:
        parsed = json.loads(report_to_json(quality_report))
        assert "null_rates" in parsed
        assert isinstance(parsed["null_rates"], dict)

    def test_json_contains_distributions(self, quality_report: QualityReport) -> None:
        parsed = json.loads(report_to_json(quality_report))
        assert "auth_method_distribution" in parsed
        assert "auth_outcome_distribution" in parsed

    def test_datetime_fields_serialised_as_iso_strings(
        self, quality_report: QualityReport
    ) -> None:
        parsed = json.loads(report_to_json(quality_report))
        if parsed["earliest_event_time"]:
            # Should be an ISO format string
            dt = datetime.fromisoformat(parsed["earliest_event_time"])
            assert dt.tzinfo is not None

    def test_gt_fields_null_when_no_gt(self, quality_report: QualityReport) -> None:
        parsed = json.loads(report_to_json(quality_report))
        assert parsed["gt_row_count"] is None
        assert parsed["scenario_distribution"] is None

    def test_indent_parameter_respected(self, quality_report: QualityReport) -> None:
        j0 = report_to_json(quality_report, indent=0)
        j2 = report_to_json(quality_report, indent=2)
        # Indented version is longer
        assert len(j2) > len(j0)

    def test_no_home_paths_in_json(self, quality_report: QualityReport) -> None:
        j = report_to_json(quality_report)
        assert "/home/" not in j
        assert "/root/" not in j


# ---------------------------------------------------------------------------
# report_to_markdown
# ---------------------------------------------------------------------------


class TestReportToMarkdown:
    def test_returns_string(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert isinstance(md, str)

    def test_starts_with_h1(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert md.startswith("# Data Quality Report")

    def test_contains_overview_section(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert "## Dataset Overview" in md

    def test_contains_temporal_range(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert "## Temporal Range" in md

    def test_contains_validation_section(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert "## Validation" in md

    def test_contains_null_rates_section(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert "## Null Rates" in md

    def test_ends_with_newline(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert md.endswith("\n")

    def test_no_home_paths_in_markdown(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert "/home/" not in md

    def test_gt_section_absent_when_no_gt(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert "## Ground Truth" not in md

    def test_gt_section_present_when_gt_supplied(
        self, events_parquet: Path, labels_parquet: Path
    ) -> None:
        report = generate_quality_report(events_parquet, gt_path=labels_parquet)
        md = report_to_markdown(report)
        assert "## Ground Truth" in md

    def test_contains_row_count(self, quality_report: QualityReport) -> None:
        md = report_to_markdown(quality_report)
        assert str(quality_report.row_count) in md

    def test_quality_warnings_section_when_warnings(self, tmp_path: Path) -> None:
        # Generate a dataset with duplicates to trigger a warning.
        cfg = _small_config()
        result = generate_dataset(cfg)
        events = list(result.events)
        # Introduce a duplicate event_id.
        if events:
            from password_attack_detector.data.schemas import AuthEvent

            dup = AuthEvent(
                **{**events[0].model_dump(), "event_id": events[0].event_id}
            )
            events.append(dup)
        p = tmp_path / "dup_events.parquet"
        write_events_parquet(events, p)
        report = generate_quality_report(p)
        md = report_to_markdown(report)
        if report.quality_warnings:
            assert "## Quality Warnings" in md
