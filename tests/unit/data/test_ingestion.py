"""Unit tests for password_attack_detector.data.ingestion adapters."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from password_attack_detector.data.ingestion import (
    CSVIngestionAdapter,
    EmptyInputDiagnostic,
    IngestionResult,
    InvalidRowPolicy,
    JSONLIngestionAdapter,
    QuarantineEntry,
)
from password_attack_detector.exceptions import IngestionError

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

_U = "u:" + "a" * 32
_S = "s:" + "b" * 32
_D = "d:" + "c" * 32
_SESS = "sess:" + "d" * 32

_CSV_HEADER = (
    "schema_version,event_id,event_time,user_id,source_id,device_id,"
    "session_id,application_id,authentication_method,authentication_outcome,"
    "failure_reason,mfa_outcome,country_code,region_code,coarse_latitude,"
    "coarse_longitude,user_agent_family,operating_system_family,client_type,"
    "response_time_ms"
)


def _csv_row(**overrides: str) -> str:
    """Return a single CSV data row string for a valid success event."""
    row: dict[str, str] = {
        "schema_version": "1.0.0",
        "event_id": str(uuid4()),
        "event_time": "2024-01-01T00:00:00+00:00",
        "user_id": _U,
        "source_id": _S,
        "device_id": _D,
        "session_id": _SESS,
        "application_id": "app-00",
        "authentication_method": "password",
        "authentication_outcome": "success",
        "failure_reason": "",
        "mfa_outcome": "",
        "country_code": "",
        "region_code": "",
        "coarse_latitude": "",
        "coarse_longitude": "",
        "user_agent_family": "",
        "operating_system_family": "",
        "client_type": "",
        "response_time_ms": "",
    }
    row.update(overrides)
    return ",".join(row[k] for k in row)


def _make_csv(tmp_path: Path, *rows: str, header: str = _CSV_HEADER) -> Path:
    """Write a CSV file with the given header and rows."""
    p = tmp_path / "events.csv"
    lines = [header, *list(rows)]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _jsonl_record(**overrides: object) -> dict[str, object]:
    """Return a valid JSONL record dict."""
    rec: dict[str, object] = {
        "schema_version": "1.0.0",
        "event_id": str(uuid4()),
        "event_time": "2024-01-01T00:00:00+00:00",
        "user_id": _U,
        "source_id": _S,
        "device_id": _D,
        "session_id": _SESS,
        "application_id": "app-00",
        "authentication_method": "password",
        "authentication_outcome": "success",
        "failure_reason": None,
        "mfa_outcome": None,
    }
    rec.update(overrides)
    return rec


def _make_jsonl(tmp_path: Path, *records: dict[str, object]) -> Path:
    """Write a JSONL file with the given records."""
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Base type tests
# ---------------------------------------------------------------------------


class TestBaseTypes:
    def test_quarantine_entry_frozen(self) -> None:
        q = QuarantineEntry(
            row_number=1, error_code="I001", field_name=None, sanitized_message="m"
        )
        with pytest.raises((AttributeError, TypeError)):
            q.row_number = 99  # type: ignore[misc]

    def test_ingestion_result_frozen(self) -> None:
        r = IngestionResult(
            accepted_count=0,
            quarantine_count=0,
            total_rows_read=0,
            events=(),
            quarantine=(),
            empty_diagnostic=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.accepted_count = 5  # type: ignore[misc]

    def test_empty_input_diagnostic_frozen(self) -> None:
        d = EmptyInputDiagnostic(reason_code="E001", message="empty")
        with pytest.raises((AttributeError, TypeError)):
            d.reason_code = "E002"  # type: ignore[misc]

    def test_invalid_row_policy_values(self) -> None:
        assert InvalidRowPolicy.FAIL.value == "fail"
        assert InvalidRowPolicy.QUARANTINE.value == "quarantine"


# ---------------------------------------------------------------------------
# CSV adapter tests
# ---------------------------------------------------------------------------


class TestCSVIngestionAdapterValid:
    def test_single_valid_row_accepted(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, _csv_row())
        result = CSVIngestionAdapter().ingest(p)
        assert result.accepted_count == 1
        assert result.quarantine_count == 0
        assert result.total_rows_read == 1
        assert len(result.events) == 1
        assert result.empty_diagnostic is None

    def test_multiple_valid_rows_accepted(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, _csv_row(), _csv_row(), _csv_row())
        result = CSVIngestionAdapter().ingest(p)
        assert result.accepted_count == 3

    def test_events_have_correct_schema_version(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, _csv_row())
        result = CSVIngestionAdapter().ingest(p)
        assert result.events[0].schema_version == "1.0.0"

    def test_failure_row_with_reason_accepted(self, tmp_path: Path) -> None:
        row = _csv_row(
            authentication_outcome="failure",
            failure_reason="invalid_credentials",
        )
        p = _make_csv(tmp_path, row)
        result = CSVIngestionAdapter().ingest(p)
        assert result.accepted_count == 1

    def test_empty_optional_fields_become_none(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, _csv_row(country_code="", response_time_ms=""))
        result = CSVIngestionAdapter().ingest(p)
        assert result.events[0].country_code is None
        assert result.events[0].response_time_ms is None

    def test_normalized_column_headers_mapped(self, tmp_path: Path) -> None:
        # Hyphenated header names should be normalized.
        header = _CSV_HEADER.replace("event_id", "event-id").replace(
            "user_id", "user-id"
        )
        p = _make_csv(tmp_path, _csv_row(), header=header)
        result = CSVIngestionAdapter().ingest(p)
        assert result.accepted_count == 1

    def test_extra_columns_ignored(self, tmp_path: Path) -> None:
        # Extra columns that are not in the canonical schema are ignored.
        header = _CSV_HEADER + ",extra_column"
        row = _csv_row() + ",some_extra_value"
        p = _make_csv(tmp_path, row, header=header)
        result = CSVIngestionAdapter().ingest(p)
        assert result.accepted_count == 1


class TestCSVIngestionAdapterProhibitedHeaders:
    def test_password_header_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, header=_CSV_HEADER + ",password")
        with pytest.raises(IngestionError):
            CSVIngestionAdapter().ingest(p)

    def test_token_header_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, header=_CSV_HEADER + ",token")
        with pytest.raises(IngestionError):
            CSVIngestionAdapter().ingest(p)

    def test_label_gt_column_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, header=_CSV_HEADER + ",label")
        with pytest.raises(IngestionError):
            CSVIngestionAdapter().ingest(p)

    def test_malicious_gt_column_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, header=_CSV_HEADER + ",malicious")
        with pytest.raises(IngestionError):
            CSVIngestionAdapter().ingest(p)

    def test_scenario_gt_column_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, header=_CSV_HEADER + ",scenario")
        with pytest.raises(IngestionError):
            CSVIngestionAdapter().ingest(p)


class TestCSVIngestionAdapterFailPolicy:
    def test_invalid_row_raises_ingestion_error(self, tmp_path: Path) -> None:
        bad_row = _csv_row(authentication_method="not_a_method")
        p = _make_csv(tmp_path, bad_row)
        with pytest.raises(IngestionError):
            CSVIngestionAdapter(policy=InvalidRowPolicy.FAIL).ingest(p)

    def test_invalid_row_after_valid_raises(self, tmp_path: Path) -> None:
        good = _csv_row()
        bad = _csv_row(authentication_method="bad")
        p = _make_csv(tmp_path, good, bad)
        with pytest.raises(IngestionError):
            CSVIngestionAdapter(policy=InvalidRowPolicy.FAIL).ingest(p)


class TestCSVIngestionAdapterQuarantinePolicy:
    def test_invalid_row_quarantined(self, tmp_path: Path) -> None:
        bad_row = _csv_row(authentication_method="not_a_method")
        p = _make_csv(tmp_path, bad_row)
        result = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.accepted_count == 0
        assert result.quarantine_count == 1
        assert len(result.quarantine) == 1

    def test_valid_rows_still_accepted_when_bad_row_quarantined(
        self, tmp_path: Path
    ) -> None:
        good = _csv_row()
        bad = _csv_row(authentication_method="bad")
        p = _make_csv(tmp_path, good, bad)
        result = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.accepted_count == 1
        assert result.quarantine_count == 1

    def test_quarantine_entry_has_row_number(self, tmp_path: Path) -> None:
        bad_row = _csv_row(authentication_method="bad")
        p = _make_csv(tmp_path, bad_row)
        result = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.quarantine[0].row_number == 1

    def test_quarantine_entry_has_error_code(self, tmp_path: Path) -> None:
        bad_row = _csv_row(authentication_method="bad")
        p = _make_csv(tmp_path, bad_row)
        result = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.quarantine[0].error_code == "I002"

    def test_quarantine_message_has_no_raw_values(self, tmp_path: Path) -> None:
        bad_row = _csv_row(authentication_method="s3cr3t_method")
        p = _make_csv(tmp_path, bad_row)
        result = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        msg = result.quarantine[0].sanitized_message
        assert "s3cr3t_method" not in msg

    def test_all_quarantined_sets_empty_diagnostic(self, tmp_path: Path) -> None:
        bad = _csv_row(authentication_method="bad")
        p = _make_csv(tmp_path, bad)
        result = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.empty_diagnostic is not None
        assert result.empty_diagnostic.reason_code == "E002"


class TestCSVIngestionAdapterEmpty:
    def test_no_data_rows_sets_empty_diagnostic(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path)  # header only
        result = CSVIngestionAdapter().ingest(p)
        assert result.accepted_count == 0
        assert result.empty_diagnostic is not None
        assert result.empty_diagnostic.reason_code == "E001"

    def test_nonexistent_file_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.csv"
        with pytest.raises(IngestionError):
            CSVIngestionAdapter().ingest(p)


class TestCSVIngestionAdapterCounts:
    def test_total_rows_read_matches_data_rows(self, tmp_path: Path) -> None:
        p = _make_csv(tmp_path, _csv_row(), _csv_row())
        result = CSVIngestionAdapter().ingest(p)
        assert result.total_rows_read == 2

    def test_accepted_plus_quarantine_equals_total(self, tmp_path: Path) -> None:
        good = _csv_row()
        bad = _csv_row(authentication_method="bad")
        p = _make_csv(tmp_path, good, bad)
        result = CSVIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.accepted_count + result.quarantine_count == result.total_rows_read


# ---------------------------------------------------------------------------
# JSONL adapter tests
# ---------------------------------------------------------------------------


class TestJSONLIngestionAdapterValid:
    def test_single_valid_record_accepted(self, tmp_path: Path) -> None:
        p = _make_jsonl(tmp_path, _jsonl_record())
        result = JSONLIngestionAdapter().ingest(p)
        assert result.accepted_count == 1
        assert result.quarantine_count == 0
        assert len(result.events) == 1

    def test_multiple_valid_records_accepted(self, tmp_path: Path) -> None:
        p = _make_jsonl(tmp_path, _jsonl_record(), _jsonl_record(), _jsonl_record())
        result = JSONLIngestionAdapter().ingest(p)
        assert result.accepted_count == 3

    def test_extra_keys_ignored(self, tmp_path: Path) -> None:
        rec = _jsonl_record()
        rec["extra_field"] = "ignored"
        p = _make_jsonl(tmp_path, rec)
        result = JSONLIngestionAdapter().ingest(p)
        assert result.accepted_count == 1

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        lines = [json.dumps(_jsonl_record()), "", json.dumps(_jsonl_record())]
        p.write_text("\n".join(lines), encoding="utf-8")
        result = JSONLIngestionAdapter().ingest(p)
        assert result.accepted_count == 2
        assert result.total_rows_read == 2

    def test_events_have_utc_timestamps(self, tmp_path: Path) -> None:
        p = _make_jsonl(tmp_path, _jsonl_record())
        result = JSONLIngestionAdapter().ingest(p)
        from datetime import UTC

        assert result.events[0].event_time.tzinfo == UTC


class TestJSONLIngestionAdapterProhibitedKeys:
    def test_password_key_rejects_entire_dataset(self, tmp_path: Path) -> None:
        bad_rec = _jsonl_record()
        bad_rec["password"] = "hunter2"
        p = _make_jsonl(tmp_path, _jsonl_record(), bad_rec)
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter().ingest(p)

    def test_token_key_rejects_entire_dataset(self, tmp_path: Path) -> None:
        bad_rec = _jsonl_record()
        bad_rec["token"] = "abc"
        p = _make_jsonl(tmp_path, bad_rec)
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter().ingest(p)

    def test_gt_label_key_rejects_entire_dataset(self, tmp_path: Path) -> None:
        bad_rec = _jsonl_record()
        bad_rec["label"] = "normal"
        p = _make_jsonl(tmp_path, _jsonl_record(), bad_rec)
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter().ingest(p)

    def test_malicious_key_rejects_entire_dataset(self, tmp_path: Path) -> None:
        bad_rec = _jsonl_record()
        bad_rec["malicious"] = False
        p = _make_jsonl(tmp_path, bad_rec)
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter().ingest(p)

    def test_nested_prohibited_key_rejects_dataset(self, tmp_path: Path) -> None:
        bad_rec = _jsonl_record()
        bad_rec["metadata"] = {"password": "secret"}
        p = _make_jsonl(tmp_path, bad_rec)
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter().ingest(p)

    def test_nesting_depth_exceeded_rejects_dataset(self, tmp_path: Path) -> None:
        # Build a deeply nested object exceeding MAX_NESTING_DEPTH.
        nested: dict[str, object] = {"leaf": "value"}
        for _ in range(7):  # exceeds MAX_NESTING_DEPTH=5
            nested = {"child": nested}
        rec = _jsonl_record()
        rec["deep"] = nested
        p = _make_jsonl(tmp_path, rec)
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter().ingest(p)


class TestJSONLIngestionAdapterFailPolicy:
    def test_parse_error_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text("{invalid json}\n", encoding="utf-8")
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter(policy=InvalidRowPolicy.FAIL).ingest(p)

    def test_invalid_record_raises_ingestion_error(self, tmp_path: Path) -> None:
        bad = _jsonl_record(authentication_method="not_valid")
        p = _make_jsonl(tmp_path, bad)
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter(policy=InvalidRowPolicy.FAIL).ingest(p)

    def test_non_object_json_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text('["a", "b"]\n', encoding="utf-8")
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter(policy=InvalidRowPolicy.FAIL).ingest(p)


class TestJSONLIngestionAdapterQuarantinePolicy:
    def test_parse_error_quarantined(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        lines = [json.dumps(_jsonl_record()), "{bad json}", json.dumps(_jsonl_record())]
        p.write_text("\n".join(lines), encoding="utf-8")
        result = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.accepted_count == 2
        assert result.quarantine_count == 1

    def test_invalid_record_quarantined(self, tmp_path: Path) -> None:
        bad = _jsonl_record(authentication_method="invalid")
        p = _make_jsonl(tmp_path, _jsonl_record(), bad)
        result = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.accepted_count == 1
        assert result.quarantine_count == 1

    def test_quarantine_entry_has_line_number(self, tmp_path: Path) -> None:
        bad = _jsonl_record(authentication_method="bad")
        p = _make_jsonl(tmp_path, _jsonl_record(), bad)
        result = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.quarantine[0].row_number == 2

    def test_quarantine_message_no_raw_values(self, tmp_path: Path) -> None:
        bad = _jsonl_record(authentication_method="s3cr3t_method")
        p = _make_jsonl(tmp_path, bad)
        result = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        msg = result.quarantine[0].sanitized_message
        assert "s3cr3t_method" not in msg

    def test_parse_error_entry_code_i005(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text("{not valid json}\n", encoding="utf-8")
        result = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.quarantine[0].error_code == "I005"

    def test_all_quarantined_sets_empty_diagnostic(self, tmp_path: Path) -> None:
        bad = _jsonl_record(authentication_method="bad")
        p = _make_jsonl(tmp_path, bad)
        result = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.empty_diagnostic is not None
        assert result.empty_diagnostic.reason_code == "E002"


class TestJSONLIngestionAdapterEmpty:
    def test_empty_file_sets_empty_diagnostic(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text("", encoding="utf-8")
        result = JSONLIngestionAdapter().ingest(p)
        assert result.accepted_count == 0
        assert result.empty_diagnostic is not None
        assert result.empty_diagnostic.reason_code == "E001"

    def test_blank_lines_only_sets_empty_diagnostic(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        p.write_text("\n\n\n", encoding="utf-8")
        result = JSONLIngestionAdapter().ingest(p)
        assert result.empty_diagnostic is not None

    def test_nonexistent_file_raises_ingestion_error(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.jsonl"
        with pytest.raises(IngestionError):
            JSONLIngestionAdapter().ingest(p)


class TestJSONLIngestionAdapterCounts:
    def test_total_rows_excludes_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        lines = [json.dumps(_jsonl_record()), "", json.dumps(_jsonl_record())]
        p.write_text("\n".join(lines), encoding="utf-8")
        result = JSONLIngestionAdapter().ingest(p)
        assert result.total_rows_read == 2

    def test_accepted_plus_quarantine_equals_total(self, tmp_path: Path) -> None:
        good = _jsonl_record()
        bad = _jsonl_record(authentication_method="bad")
        p = _make_jsonl(tmp_path, good, bad)
        result = JSONLIngestionAdapter(policy=InvalidRowPolicy.QUARANTINE).ingest(p)
        assert result.accepted_count + result.quarantine_count == result.total_rows_read


# ---------------------------------------------------------------------------
# Ingestion __init__ re-exports
# ---------------------------------------------------------------------------


class TestIngestionPackageExports:
    def test_all_public_names_importable(self) -> None:
        from password_attack_detector.data.ingestion import (  # noqa: F401
            CSVIngestionAdapter,
            EmptyInputDiagnostic,
            IngestionResult,
            InvalidRowPolicy,
            JSONLIngestionAdapter,
            QuarantineEntry,
        )
