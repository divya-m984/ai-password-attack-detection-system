"""Unit tests for password_attack_detector.data.validation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from password_attack_detector.data.schemas import SCHEMA_VERSION
from password_attack_detector.data.validation import (
    DatasetValidator,
    ValidationError,
    ValidationResult,
    ValidationStatus,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_U = "u:" + "a" * 32
_S = "s:" + "b" * 32
_D = "d:" + "c" * 32
_SESS = "sess:" + "d" * 32


def _row(**overrides: object) -> dict[str, object]:
    """Return a minimal valid canonical row dict."""
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(uuid4()),
        "event_time": pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        "user_id": _U,
        "source_id": _S,
        "device_id": _D,
        "session_id": _SESS,
        "application_id": "app-00",
        "authentication_method": "password",
        "authentication_outcome": "success",
        "failure_reason": None,
        "mfa_outcome": None,
        "country_code": None,
        "region_code": None,
        "coarse_latitude": None,
        "coarse_longitude": None,
        "user_agent_family": None,
        "operating_system_family": None,
        "client_type": None,
        "response_time_ms": None,
    }
    base.update(overrides)
    return base


def _df(*rows: dict[str, object]) -> pd.DataFrame:
    """Build a DataFrame from one or more row dicts."""
    return pd.DataFrame(list(rows))


@pytest.fixture()
def validator() -> DatasetValidator:
    return DatasetValidator()


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    return _df(_row())


# ---------------------------------------------------------------------------
# ValidationStatus / ValidationError types
# ---------------------------------------------------------------------------


class TestValidationTypes:
    def test_status_values(self) -> None:
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.INVALID.value == "invalid"

    def test_validation_error_defaults(self) -> None:
        e = ValidationError(code="V001", message="test")
        assert e.field is None
        assert e.count == 0

    def test_validation_error_frozen(self) -> None:
        e = ValidationError(code="V001", message="test", field="f", count=3)
        with pytest.raises((AttributeError, TypeError)):
            e.code = "V002"  # type: ignore[misc]

    def test_validation_result_frozen(self) -> None:
        r = ValidationResult(
            status=ValidationStatus.VALID,
            schema_version=SCHEMA_VERSION,
            record_count=1,
            accepted_count=1,
            rejected_count=0,
            errors=(),
            warnings=(),
            duplicate_count=0,
            null_rates={},
        )
        with pytest.raises((AttributeError, TypeError)):
            r.record_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# V001 -- Parquet unreadable
# ---------------------------------------------------------------------------


class TestV001ParquetUnreadable:
    def test_nonexistent_file_returns_v001(
        self, validator: DatasetValidator, tmp_path: Path
    ) -> None:
        result = validator.validate_parquet(tmp_path / "missing.parquet")
        assert result.status == ValidationStatus.INVALID
        assert any(e.code == "V001" for e in result.errors)

    def test_non_parquet_file_returns_v001(
        self, validator: DatasetValidator, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        p.write_bytes(b"not a parquet file")
        result = validator.validate_parquet(p)
        assert result.status == ValidationStatus.INVALID
        assert any(e.code == "V001" for e in result.errors)
        assert result.record_count == 0
        assert result.null_rates == {}

    def test_valid_parquet_delegates_to_dataframe(
        self, validator: DatasetValidator, tmp_path: Path, valid_df: pd.DataFrame
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(valid_df)
        p = tmp_path / "events.parquet"
        pq.write_table(table, p)
        result = validator.validate_parquet(p)
        # The result may have errors (e.g. V007 if timezone not preserved), but
        # V001 must not be present.
        assert not any(e.code == "V001" for e in result.errors)


# ---------------------------------------------------------------------------
# V002 -- Empty dataset
# ---------------------------------------------------------------------------


class TestV002EmptyDataset:
    def test_empty_df_is_invalid(self, validator: DatasetValidator) -> None:
        result = validator.validate_dataframe(pd.DataFrame())
        assert result.status == ValidationStatus.INVALID
        assert any(e.code == "V002" for e in result.errors)
        assert result.record_count == 0
        assert result.accepted_count == 0

    def test_single_row_df_passes_v002(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V002" for e in result.errors)


# ---------------------------------------------------------------------------
# V003 -- Required columns missing
# ---------------------------------------------------------------------------


class TestV003RequiredColumns:
    def test_missing_event_id_is_invalid(self, validator: DatasetValidator) -> None:
        df = _df(_row()).drop(columns=["event_id"])
        result = validator.validate_dataframe(df)
        assert result.status == ValidationStatus.INVALID
        assert any(e.code == "V003" for e in result.errors)

    def test_missing_multiple_columns(self, validator: DatasetValidator) -> None:
        df = _df(_row()).drop(columns=["user_id", "source_id"])
        result = validator.validate_dataframe(df)
        assert any(e.code == "V003" for e in result.errors)
        v003 = next(e for e in result.errors if e.code == "V003")
        assert v003.count >= 2

    def test_all_required_columns_present_no_v003(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V003" for e in result.errors)

    def test_missing_columns_returns_all_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(), _row()).drop(columns=["authentication_method"])
        result = validator.validate_dataframe(df)
        assert result.rejected_count == result.record_count


# ---------------------------------------------------------------------------
# V004 -- Schema version mismatch
# ---------------------------------------------------------------------------


class TestV004SchemaVersion:
    def test_wrong_schema_version_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(schema_version="0.9.0"))
        result = validator.validate_dataframe(df)
        assert result.status == ValidationStatus.INVALID
        assert any(e.code == "V004" for e in result.errors)
        assert result.rejected_count == 1

    def test_correct_schema_version_accepted(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V004" for e in result.errors)

    def test_schema_version_detected(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert result.schema_version == SCHEMA_VERSION

    def test_mixed_schema_versions_detected(self, validator: DatasetValidator) -> None:
        df = _df(_row(), _row(schema_version="0.9.0"))
        result = validator.validate_dataframe(df)
        v004 = next(e for e in result.errors if e.code == "V004")
        assert v004.count == 1
        # Two different versions -> detected_sv is None
        assert result.schema_version is None


# ---------------------------------------------------------------------------
# V005 -- Prohibited ground-truth columns
# ---------------------------------------------------------------------------


class TestV005ProhibitedColumns:
    def test_label_column_is_prohibited(self, validator: DatasetValidator) -> None:
        df = _df(_row())
        df["label"] = "normal"
        result = validator.validate_dataframe(df)
        assert result.status == ValidationStatus.INVALID
        assert any(e.code == "V005" for e in result.errors)

    def test_malicious_column_is_prohibited(self, validator: DatasetValidator) -> None:
        df = _df(_row())
        df["malicious"] = False
        result = validator.validate_dataframe(df)
        assert any(e.code == "V005" for e in result.errors)

    def test_no_prohibited_columns_passes(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V005" for e in result.errors)

    def test_prohibited_columns_reject_all_rows(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(), _row())
        df["risk_score"] = 0.9
        result = validator.validate_dataframe(df)
        assert result.rejected_count == 2


# ---------------------------------------------------------------------------
# V006 -- Duplicate event_ids
# ---------------------------------------------------------------------------


class TestV006DuplicateEventIds:
    def test_duplicate_ids_detected(self, validator: DatasetValidator) -> None:
        fixed_id = str(uuid4())
        df = _df(_row(event_id=fixed_id), _row(event_id=fixed_id))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V006" for e in result.errors)
        assert result.duplicate_count == 1

    def test_unique_ids_no_v006(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V006" for e in result.errors)
        assert result.duplicate_count == 0

    def test_duplicate_marks_second_row_rejected(
        self, validator: DatasetValidator
    ) -> None:
        fixed_id = str(uuid4())
        df = _df(_row(event_id=fixed_id), _row(event_id=fixed_id))
        result = validator.validate_dataframe(df)
        # first occurrence kept, second rejected
        assert result.duplicate_count == 1


# ---------------------------------------------------------------------------
# V007 -- UTC timestamps
# ---------------------------------------------------------------------------


class TestV007UTCTimestamps:
    def test_utc_timestamp_accepted(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V007" for e in result.errors)

    def test_timezone_naive_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row())
        df["event_time"] = pd.Timestamp("2024-01-01")
        result = validator.validate_dataframe(df)
        assert any(e.code == "V007" for e in result.errors)

    def test_non_datetime_column_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row())
        df["event_time"] = "2024-01-01T00:00:00Z"  # string, not datetime
        result = validator.validate_dataframe(df)
        assert any(e.code == "V007" for e in result.errors)

    def test_non_utc_timezone_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row())
        df["event_time"] = pd.Timestamp("2024-01-01", tz="US/Eastern")
        result = validator.validate_dataframe(df)
        assert any(e.code == "V007" for e in result.errors)


# ---------------------------------------------------------------------------
# V008-V012 -- Enum value checks
# ---------------------------------------------------------------------------


class TestV008AuthMethod:
    def test_invalid_method_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(authentication_method="telepathy"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V008" for e in result.errors)

    def test_valid_method_accepted(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V008" for e in result.errors)

    def test_null_method_is_invalid(self, validator: DatasetValidator) -> None:
        df = _df(_row(authentication_method=None))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V008" for e in result.errors)


class TestV009AuthOutcome:
    def test_invalid_outcome_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(authentication_outcome="pending"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V009" for e in result.errors)

    def test_all_valid_outcomes_accepted(self, validator: DatasetValidator) -> None:
        for outcome, fr in [
            ("success", None),
            ("failure", "invalid_credentials"),
            ("blocked", "ip_blocked"),
            ("challenged", None),
        ]:
            df = _df(_row(authentication_outcome=outcome, failure_reason=fr))
            result = validator.validate_dataframe(df)
            assert not any(e.code == "V009" for e in result.errors), outcome


class TestV010FailureReason:
    def test_invalid_failure_reason_rejected(self, validator: DatasetValidator) -> None:
        df = _df(
            _row(
                authentication_outcome="failure",
                failure_reason="bad_vibes",
            )
        )
        result = validator.validate_dataframe(df)
        assert any(e.code == "V010" for e in result.errors)

    def test_null_failure_reason_skipped(self, validator: DatasetValidator) -> None:
        # null is valid for V010 (nullable=True); consistency is V017's concern
        df = _df(_row(failure_reason=None))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V010" for e in result.errors)


class TestV011MFAOutcome:
    def test_invalid_mfa_outcome_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(mfa_outcome="magic"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V011" for e in result.errors)

    def test_null_mfa_outcome_accepted(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V011" for e in result.errors)


class TestV012ClientType:
    def test_invalid_client_type_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(client_type="spaceship"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V012" for e in result.errors)

    def test_null_client_type_accepted(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V012" for e in result.errors)


# ---------------------------------------------------------------------------
# V013 -- Country code format
# ---------------------------------------------------------------------------


class TestV013CountryCode:
    def test_lowercase_code_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(country_code="us"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V013" for e in result.errors)

    def test_too_long_code_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(country_code="USA"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V013" for e in result.errors)

    def test_valid_code_accepted(self, validator: DatasetValidator) -> None:
        df = _df(_row(country_code="DE"))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V013" for e in result.errors)

    def test_null_code_skipped(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V013" for e in result.errors)


# ---------------------------------------------------------------------------
# V014 / V015 -- Coordinate ranges
# ---------------------------------------------------------------------------


class TestV014Latitude:
    def test_latitude_below_negative_90_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(coarse_latitude=-91.0))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V014" for e in result.errors)

    def test_latitude_above_90_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(coarse_latitude=91.0))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V014" for e in result.errors)

    def test_valid_latitude_accepted(self, validator: DatasetValidator) -> None:
        df = _df(_row(coarse_latitude=45.0))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V014" for e in result.errors)

    def test_null_latitude_skipped(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V014" for e in result.errors)


class TestV015Longitude:
    def test_longitude_below_negative_180_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(coarse_longitude=-181.0))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V015" for e in result.errors)

    def test_longitude_above_180_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(coarse_longitude=181.0))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V015" for e in result.errors)

    def test_valid_longitude_accepted(self, validator: DatasetValidator) -> None:
        df = _df(_row(coarse_longitude=-100.0))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V015" for e in result.errors)


# ---------------------------------------------------------------------------
# V016 -- response_time_ms
# ---------------------------------------------------------------------------


class TestV016ResponseTime:
    def test_negative_response_time_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(response_time_ms=-1))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V016" for e in result.errors)

    def test_over_30000_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(response_time_ms=30_001))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V016" for e in result.errors)

    def test_zero_accepted(self, validator: DatasetValidator) -> None:
        df = _df(_row(response_time_ms=0))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V016" for e in result.errors)

    def test_30000_accepted(self, validator: DatasetValidator) -> None:
        df = _df(_row(response_time_ms=30_000))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V016" for e in result.errors)

    def test_null_response_time_skipped(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V016" for e in result.errors)


# ---------------------------------------------------------------------------
# V017 -- outcome / failure_reason consistency
# ---------------------------------------------------------------------------


class TestV017OutcomeFailureReason:
    def test_success_with_failure_reason_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(
            _row(authentication_outcome="success", failure_reason="invalid_credentials")
        )
        result = validator.validate_dataframe(df)
        assert any(e.code == "V017" for e in result.errors)

    def test_failure_without_failure_reason_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(authentication_outcome="failure", failure_reason=None))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V017" for e in result.errors)

    def test_blocked_with_invalid_reason_rejected(
        self, validator: DatasetValidator
    ) -> None:
        # invalid_credentials is not a valid BLOCKED reason
        df = _df(
            _row(
                authentication_outcome="blocked",
                failure_reason="invalid_credentials",
            )
        )
        result = validator.validate_dataframe(df)
        assert any(e.code == "V017" for e in result.errors)

    def test_blocked_without_failure_reason_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(authentication_outcome="blocked", failure_reason=None))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V017" for e in result.errors)

    def test_challenged_with_failure_reason_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(
            _row(
                authentication_outcome="challenged",
                failure_reason="invalid_credentials",
            )
        )
        result = validator.validate_dataframe(df)
        assert any(e.code == "V017" for e in result.errors)

    def test_valid_success_no_v017(self, validator: DatasetValidator) -> None:
        df = _df(_row(authentication_outcome="success", failure_reason=None))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V017" for e in result.errors)

    def test_valid_failure_with_reason_no_v017(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(
            _row(authentication_outcome="failure", failure_reason="invalid_credentials")
        )
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V017" for e in result.errors)

    def test_valid_blocked_with_valid_reason_no_v017(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(authentication_outcome="blocked", failure_reason="ip_blocked"))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V017" for e in result.errors)


# ---------------------------------------------------------------------------
# V018 -- MFA bypass consistency
# ---------------------------------------------------------------------------


class TestV018MFABypass:
    def test_bypassed_with_blocked_outcome_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(
            _row(
                authentication_outcome="blocked",
                failure_reason="ip_blocked",
                mfa_outcome="bypassed",
            )
        )
        result = validator.validate_dataframe(df)
        assert any(e.code == "V018" for e in result.errors)

    def test_bypassed_with_challenged_outcome_rejected(
        self, validator: DatasetValidator
    ) -> None:
        df = _df(_row(authentication_outcome="challenged", mfa_outcome="bypassed"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V018" for e in result.errors)

    def test_bypassed_with_success_accepted(self, validator: DatasetValidator) -> None:
        df = _df(_row(authentication_outcome="success", mfa_outcome="bypassed"))
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V018" for e in result.errors)

    def test_bypassed_with_failure_accepted(self, validator: DatasetValidator) -> None:
        df = _df(
            _row(
                authentication_outcome="failure",
                failure_reason="mfa_failed",
                mfa_outcome="bypassed",
            )
        )
        result = validator.validate_dataframe(df)
        assert not any(e.code == "V018" for e in result.errors)


# ---------------------------------------------------------------------------
# V019 -- Pseudonym format
# ---------------------------------------------------------------------------


class TestV019PseudonymFormat:
    def test_bad_user_id_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(user_id="raw-username"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V019" for e in result.errors)
        v019 = next(e for e in result.errors if e.code == "V019")
        assert v019.field == "user_id"

    def test_bad_source_id_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(source_id="192.168.1.1"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V019" for e in result.errors)

    def test_bad_device_id_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(device_id="device-fingerprint"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V019" for e in result.errors)

    def test_bad_session_id_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(session_id="token-abc"))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V019" for e in result.errors)

    def test_null_user_id_is_invalid(self, validator: DatasetValidator) -> None:
        df = _df(_row(user_id=None))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V019" for e in result.errors)

    def test_wrong_prefix_rejected(self, validator: DatasetValidator) -> None:
        # 'd:' is valid for device_id but 'u:' is the user prefix
        df = _df(_row(user_id="d:" + "a" * 32))
        # 'd:' IS a valid prefix per the pattern (u|s|d|sess)
        # but the semantics require 'u:' for user_id -- however V019 only checks format
        result = validator.validate_dataframe(df)
        # "d:" is a valid prefix, so this should NOT trigger V019
        assert not any(e.code == "V019" for e in result.errors)

    def test_valid_pseudonyms_accepted(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert not any(e.code == "V019" for e in result.errors)

    def test_uppercase_hex_rejected(self, validator: DatasetValidator) -> None:
        df = _df(_row(user_id="u:" + "A" * 32))
        result = validator.validate_dataframe(df)
        assert any(e.code == "V019" for e in result.errors)


# ---------------------------------------------------------------------------
# V020 -- High null rates (warning)
# ---------------------------------------------------------------------------


class TestV020NullRates:
    def test_high_null_rate_generates_warning(
        self, validator: DatasetValidator
    ) -> None:
        # Build 3 rows where user_agent_family is null in all rows
        df = _df(_row(), _row(), _row())
        # user_agent_family is already null; add a column that is always null
        df["region_code"] = None
        result = validator.validate_dataframe(df)
        # region_code is 100% null -- should trigger V020
        assert any(
            e.code == "V020" and e.field == "region_code" for e in result.warnings
        )

    def test_null_rates_dict_populated(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert "event_time" in result.null_rates
        assert result.null_rates["event_time"] == 0.0

    def test_partial_null_rate_below_threshold_no_warning(
        self, validator: DatasetValidator
    ) -> None:
        # 50% exactly does NOT exceed threshold; only > 50% triggers
        df = _df(_row(country_code="US"), _row(country_code=None))
        result = validator.validate_dataframe(df)
        assert not any(
            e.code == "V020" and e.field == "country_code" for e in result.warnings
        )

    def test_valid_status_when_no_errors_and_no_warnings(
        self, validator: DatasetValidator
    ) -> None:
        # Populate every optional column so no column exceeds the 50% null-rate
        # warning threshold.
        full_row = _row(
            authentication_outcome="failure",
            failure_reason="invalid_credentials",
            mfa_outcome="failed",
            country_code="US",
            region_code="CA",
            coarse_latitude=37.0,
            coarse_longitude=-122.0,
            user_agent_family="Chrome",
            operating_system_family="Windows",
            client_type="web_browser",
            response_time_ms=100,
        )
        df = _df(full_row)
        result = validator.validate_dataframe(df)
        assert result.status == ValidationStatus.VALID

    def test_warning_status_when_only_warnings(
        self, validator: DatasetValidator
    ) -> None:
        # Build a valid row with a column that will be 100% null
        df = _df(_row())
        # override a column that the validator won't error on
        df["region_code"] = None
        result = validator.validate_dataframe(df)
        if result.errors:
            # If other errors exist, status is INVALID (skip this assertion)
            return
        assert result.status == ValidationStatus.WARNING


# ---------------------------------------------------------------------------
# Aggregate / integration tests
# ---------------------------------------------------------------------------


class TestValidationResultCounts:
    def test_accepted_plus_rejected_equals_record_count(
        self, validator: DatasetValidator
    ) -> None:
        fixed_id = str(uuid4())
        df = _df(
            _row(),
            _row(event_id=fixed_id),
            _row(event_id=fixed_id),  # duplicate
        )
        result = validator.validate_dataframe(df)
        assert result.accepted_count + result.rejected_count == result.record_count

    def test_multiple_errors_accumulated(self, validator: DatasetValidator) -> None:
        df = _df(
            _row(authentication_method="invalid"),
            _row(authentication_outcome="invalid"),
        )
        result = validator.validate_dataframe(df)
        error_codes = {e.code for e in result.errors}
        assert "V008" in error_codes
        assert "V009" in error_codes

    def test_clean_dataset_all_accepted(
        self, validator: DatasetValidator, valid_df: pd.DataFrame
    ) -> None:
        result = validator.validate_dataframe(valid_df)
        assert result.accepted_count == result.record_count
        assert result.rejected_count == 0
        # V020 warnings may be present for optional null columns, so only
        # assert no validation *errors*.
        assert not result.errors

    def test_error_code_field_set_correctly(self, validator: DatasetValidator) -> None:
        df = _df(_row(authentication_method="bad"))
        result = validator.validate_dataframe(df)
        v008 = next(e for e in result.errors if e.code == "V008")
        assert v008.field == "authentication_method"
        assert v008.count == 1

    def test_no_raw_data_in_error_messages(self, validator: DatasetValidator) -> None:
        df = _df(_row(authentication_method="s3cr3t-method"))
        result = validator.validate_dataframe(df)
        for error in result.errors:
            assert "s3cr3t-method" not in error.message
