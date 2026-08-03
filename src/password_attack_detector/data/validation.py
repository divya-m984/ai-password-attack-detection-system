"""Dataset validation for canonical authentication-event Parquet files.

Performs 20 deterministic validation checks over canonical authentication-event
datasets, returning machine-readable ``ValidationResult`` objects with stable
error codes.  Error messages never include raw data values, identifiers, file
paths, or secrets -- only aggregate counts and canonical field names.

Public API
----------
- ``ValidationStatus``  -- VALID / WARNING / INVALID
- ``ValidationError``   -- single finding with stable code and human-readable message
- ``ValidationResult``  -- complete validation outcome
- ``DatasetValidator``  -- stateless validator (Parquet file or in-memory DataFrame)

Validation checks
-----------------
V001  Parquet file unreadable (validate_parquet only)
V002  Empty dataset
V003  Required columns missing
V004  schema_version mismatch
V005  Prohibited ground-truth columns present
V006  Duplicate event_id values
V007  Non-UTC or non-datetime event_time column
V008  Unrecognized authentication_method values
V009  Unrecognized authentication_outcome values
V010  Unrecognized failure_reason values
V011  Unrecognized mfa_outcome values
V012  Unrecognized client_type values
V013  Invalid country_code format (not ISO 3166-1 alpha-2)
V014  coarse_latitude outside [-90, 90]
V015  coarse_longitude outside [-180, 180]
V016  response_time_ms outside [0, 30000]
V017  outcome / failure_reason cross-field inconsistency
V018  MFA_BYPASSED with non-SUCCESS / non-FAILURE outcome
V019  Invalid pseudonymous identifier format
V020  High null rate (warning, threshold 50%)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    ClientType,
    FailureReason,
    MFAOutcome,
)
from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS, SCHEMA_VERSION

__all__ = [
    "DatasetValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationStatus",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PSEUDONYM_COLUMNS: tuple[str, ...] = (
    "user_id",
    "source_id",
    "device_id",
    "session_id",
)

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "schema_version",
        "event_id",
        "event_time",
        "user_id",
        "source_id",
        "device_id",
        "session_id",
        "application_id",
        "authentication_method",
        "authentication_outcome",
    }
)

_VALID_AUTH_METHODS: frozenset[str] = frozenset(m.value for m in AuthMethod)
_VALID_AUTH_OUTCOMES: frozenset[str] = frozenset(o.value for o in AuthOutcome)
_VALID_FAILURE_REASONS: frozenset[str] = frozenset(r.value for r in FailureReason)
_VALID_MFA_OUTCOMES: frozenset[str] = frozenset(o.value for o in MFAOutcome)
_VALID_CLIENT_TYPES: frozenset[str] = frozenset(t.value for t in ClientType)

# Failure reasons valid when authentication_outcome is BLOCKED.
_BLOCKED_VALID_REASONS: frozenset[str] = frozenset(
    {
        FailureReason.IP_BLOCKED,
        FailureReason.RATE_LIMITED,
        FailureReason.SUSPICIOUS_ACTIVITY,
        FailureReason.ACCOUNT_LOCKED,
        FailureReason.ACCOUNT_DISABLED,
        FailureReason.UNKNOWN,
    }
)

# Pseudonym format: prefix:[0-9a-f]{32}
_PSEUDO_PATTERN: str = r"^(u|s|d|sess):[0-9a-f]{32}$"

# Null rate threshold above which a V020 warning is issued.
_NULL_RATE_WARN_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ValidationStatus(StrEnum):
    """Overall outcome of a validation run."""

    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


@dataclass(frozen=True)
class ValidationError:
    """A single validation finding (error or warning).

    ``code`` is a stable machine-readable identifier (e.g. ``"V006"``).
    ``message`` is human-readable and never contains raw data values,
    identifiers, file paths, or secrets.
    """

    code: str
    message: str
    field: str | None = None
    count: int = 0


@dataclass(frozen=True)
class ValidationResult:
    """Complete result of a dataset validation run."""

    status: ValidationStatus
    schema_version: str | None
    record_count: int
    accepted_count: int
    rejected_count: int
    errors: tuple[ValidationError, ...]
    warnings: tuple[ValidationError, ...]
    duplicate_count: int
    null_rates: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _col_rows(df: pd.DataFrame, mask: pd.Series[bool]) -> list[int]:
    """Return sorted list of integer row indices where *mask* is True."""
    return [int(i) for i in df[mask].index.tolist()]


def _check_enum_column(
    df: pd.DataFrame,
    col_name: str,
    valid_values: frozenset[str],
    code: str,
    errors: list[ValidationError],
    rejected_rows: set[int],
    *,
    nullable: bool,
) -> None:
    """Append a ``ValidationError`` if *col_name* contains unrecognized values.

    When *nullable* is ``True``, null values are skipped; when ``False`` null
    values are counted as invalid (required field).
    """
    if col_name not in df.columns:
        return
    col = df[col_name]
    if nullable:
        invalid = col.notna() & ~col.isin(valid_values)
    else:
        invalid = ~col.isin(valid_values)
    bad_count = int(invalid.sum())
    if bad_count > 0:
        errors.append(
            ValidationError(
                code=code,
                message=f"{bad_count} records have unrecognized {col_name!r} values",
                field=col_name,
                count=bad_count,
            )
        )
        rejected_rows.update(_col_rows(df, invalid))


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class DatasetValidator:
    """Stateless validator for canonical authentication-event datasets.

    All findings are machine-readable (stable ``V0xx`` codes) and privacy-safe
    (no raw values, no identifiers, no file system paths, no secrets).

    Usage::

        validator = DatasetValidator()
        result = validator.validate_parquet(path)   # from Parquet file
        result = validator.validate_dataframe(df)   # from in-memory DataFrame
    """

    def validate_parquet(self, path: Path) -> ValidationResult:
        """Read a Parquet file and run all validation checks.

        Returns an ``INVALID`` result (``V001``) immediately if the file
        cannot be read.  Otherwise delegates to ``validate_dataframe``.

        Parameters
        ----------
        path:
            Path to the Parquet file to validate.
        """
        try:
            table = pq.read_table(path)
            df: pd.DataFrame = table.to_pandas()
        except Exception as exc:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                schema_version=None,
                record_count=0,
                accepted_count=0,
                rejected_count=0,
                errors=(
                    ValidationError(
                        code="V001",
                        message=(
                            f"Parquet file could not be read ({type(exc).__name__})"
                        ),
                    ),
                ),
                warnings=(),
                duplicate_count=0,
                null_rates={},
            )
        return self.validate_dataframe(df)

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Run all in-memory validation checks against *df*.

        Parameters
        ----------
        df:
            DataFrame to validate. Columns must include all required canonical
            fields; optional columns are checked when present.
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # V002 -- Empty dataset (fatal: nothing else can be checked).
        if len(df) == 0:
            errors.append(
                ValidationError(code="V002", message="Dataset contains no records")
            )
            return ValidationResult(
                status=ValidationStatus.INVALID,
                schema_version=None,
                record_count=0,
                accepted_count=0,
                rejected_count=0,
                errors=tuple(errors),
                warnings=(),
                duplicate_count=0,
                null_rates={},
            )

        record_count = len(df)
        cols: set[str] = {str(c) for c in df.columns}

        # V003 -- Required columns missing (fatal: row checks need these columns).
        missing_cols = _REQUIRED_COLUMNS - cols
        if missing_cols:
            errors.append(
                ValidationError(
                    code="V003",
                    message=f"Required columns missing: {sorted(missing_cols)}",
                    count=len(missing_cols),
                )
            )
            return ValidationResult(
                status=ValidationStatus.INVALID,
                schema_version=None,
                record_count=record_count,
                accepted_count=0,
                rejected_count=record_count,
                errors=tuple(errors),
                warnings=(),
                duplicate_count=0,
                null_rates={},
            )

        # All required columns exist past this point.
        rejected_rows: set[int] = set()
        duplicate_count = 0

        # Detect schema_version for result metadata.
        sv_col = df["schema_version"].astype(str)
        detected_sv: str | None = str(sv_col.iloc[0]) if sv_col.nunique() == 1 else None

        # V004 -- schema_version mismatch.
        bad_sv = sv_col != SCHEMA_VERSION
        if bad_sv.any():
            bad_count = int(bad_sv.sum())
            errors.append(
                ValidationError(
                    code="V004",
                    message=(
                        f"{bad_count} records have schema_version != {SCHEMA_VERSION!r}"
                    ),
                    field="schema_version",
                    count=bad_count,
                )
            )
            rejected_rows.update(_col_rows(df, bad_sv))

        # V005 -- Prohibited ground-truth columns present.
        prohibited_present = PROHIBITED_GT_COLUMNS & cols
        if prohibited_present:
            errors.append(
                ValidationError(
                    code="V005",
                    message=(
                        "Prohibited ground-truth columns present: "
                        f"{sorted(prohibited_present)}"
                    ),
                    count=len(prohibited_present),
                )
            )
            # All rows are tainted when prohibited columns exist.
            rejected_rows.update(range(record_count))

        # V006 -- Duplicate event_ids.
        dup_mask = df["event_id"].duplicated(keep="first")
        duplicate_count = int(dup_mask.sum())
        if duplicate_count > 0:
            errors.append(
                ValidationError(
                    code="V006",
                    message=f"{duplicate_count} duplicate event_id values",
                    field="event_id",
                    count=duplicate_count,
                )
            )
            rejected_rows.update(_col_rows(df, dup_mask))

        # V007 -- Non-UTC or non-datetime event_time.
        et_col = df["event_time"]
        if not pd.api.types.is_datetime64_any_dtype(et_col):
            errors.append(
                ValidationError(
                    code="V007",
                    message="event_time is not a datetime column; UTC required",
                    field="event_time",
                    count=record_count,
                )
            )
            rejected_rows.update(range(record_count))
        elif not isinstance(et_col.dtype, pd.DatetimeTZDtype):
            errors.append(
                ValidationError(
                    code="V007",
                    message="event_time is timezone-naive; UTC required",
                    field="event_time",
                    count=record_count,
                )
            )
            rejected_rows.update(range(record_count))
        elif str(et_col.dtype.tz) not in {"UTC", "UTC+00:00", "+00:00"}:
            errors.append(
                ValidationError(
                    code="V007",
                    message=f"event_time timezone is not UTC: {et_col.dtype.tz!r}",
                    field="event_time",
                    count=record_count,
                )
            )
            rejected_rows.update(range(record_count))

        # V008-V012 -- Enum value checks.
        _check_enum_column(
            df,
            "authentication_method",
            _VALID_AUTH_METHODS,
            "V008",
            errors,
            rejected_rows,
            nullable=False,
        )
        _check_enum_column(
            df,
            "authentication_outcome",
            _VALID_AUTH_OUTCOMES,
            "V009",
            errors,
            rejected_rows,
            nullable=False,
        )
        _check_enum_column(
            df,
            "failure_reason",
            _VALID_FAILURE_REASONS,
            "V010",
            errors,
            rejected_rows,
            nullable=True,
        )
        _check_enum_column(
            df,
            "mfa_outcome",
            _VALID_MFA_OUTCOMES,
            "V011",
            errors,
            rejected_rows,
            nullable=True,
        )
        _check_enum_column(
            df,
            "client_type",
            _VALID_CLIENT_TYPES,
            "V012",
            errors,
            rejected_rows,
            nullable=True,
        )

        # V013 -- Country code format (ISO 3166-1 alpha-2: exactly 2 uppercase letters).
        if "country_code" in cols:
            cc_col = df["country_code"]
            invalid_cc = cc_col.notna() & ~cc_col.str.match(r"^[A-Z]{2}$", na=True)
            bad_count = int(invalid_cc.sum())
            if bad_count > 0:
                errors.append(
                    ValidationError(
                        code="V013",
                        message=f"{bad_count} records have invalid country_code format",
                        field="country_code",
                        count=bad_count,
                    )
                )
                rejected_rows.update(_col_rows(df, invalid_cc))

        # V014 -- Latitude out of range.
        if "coarse_latitude" in cols:
            lat_col = df["coarse_latitude"]
            invalid_lat = lat_col.notna() & ((lat_col < -90.0) | (lat_col > 90.0))
            bad_count = int(invalid_lat.sum())
            if bad_count > 0:
                errors.append(
                    ValidationError(
                        code="V014",
                        message=(
                            f"{bad_count} records have coarse_latitude outside [-90, 90]"
                        ),
                        field="coarse_latitude",
                        count=bad_count,
                    )
                )
                rejected_rows.update(_col_rows(df, invalid_lat))

        # V015 -- Longitude out of range.
        if "coarse_longitude" in cols:
            lon_col = df["coarse_longitude"]
            invalid_lon = lon_col.notna() & ((lon_col < -180.0) | (lon_col > 180.0))
            bad_count = int(invalid_lon.sum())
            if bad_count > 0:
                errors.append(
                    ValidationError(
                        code="V015",
                        message=(
                            f"{bad_count} records have coarse_longitude outside "
                            f"[-180, 180]"
                        ),
                        field="coarse_longitude",
                        count=bad_count,
                    )
                )
                rejected_rows.update(_col_rows(df, invalid_lon))

        # V016 -- response_time_ms out of range.
        if "response_time_ms" in cols:
            rt_col = df["response_time_ms"]
            invalid_rt = rt_col.notna() & ((rt_col < 0) | (rt_col > 30_000))
            bad_count = int(invalid_rt.sum())
            if bad_count > 0:
                errors.append(
                    ValidationError(
                        code="V016",
                        message=(
                            f"{bad_count} records have response_time_ms outside "
                            f"[0, 30000]"
                        ),
                        field="response_time_ms",
                        count=bad_count,
                    )
                )
                rejected_rows.update(_col_rows(df, invalid_rt))

        # V017 -- outcome / failure_reason cross-field consistency.
        if "failure_reason" in cols:
            outcome_col = df["authentication_outcome"]
            fr_col = df["failure_reason"]

            success_bad = (outcome_col == AuthOutcome.SUCCESS) & fr_col.notna()
            failure_bad = (outcome_col == AuthOutcome.FAILURE) & fr_col.isna()
            blocked_null = (outcome_col == AuthOutcome.BLOCKED) & fr_col.isna()
            blocked_bad_fr = (
                (outcome_col == AuthOutcome.BLOCKED)
                & fr_col.notna()
                & ~fr_col.isin(_BLOCKED_VALID_REASONS)
            )
            challenged_bad = (outcome_col == AuthOutcome.CHALLENGED) & fr_col.notna()

            inconsistent = (
                success_bad
                | failure_bad
                | blocked_null
                | blocked_bad_fr
                | challenged_bad
            )
            bad_count = int(inconsistent.sum())
            if bad_count > 0:
                errors.append(
                    ValidationError(
                        code="V017",
                        message=(
                            f"{bad_count} records have outcome/failure_reason "
                            f"inconsistencies"
                        ),
                        count=bad_count,
                    )
                )
                rejected_rows.update(_col_rows(df, inconsistent))

        # V018 -- MFA_BYPASSED only valid with SUCCESS or FAILURE outcomes.
        if "mfa_outcome" in cols:
            outcome_col2 = df["authentication_outcome"]
            mfa_col = df["mfa_outcome"]

            bypassed = mfa_col == MFAOutcome.BYPASSED
            valid_for_bypass = outcome_col2.isin(
                {AuthOutcome.SUCCESS, AuthOutcome.FAILURE}
            )
            mfa_bad = bypassed & ~valid_for_bypass
            bad_count = int(mfa_bad.sum())
            if bad_count > 0:
                errors.append(
                    ValidationError(
                        code="V018",
                        message=(
                            f"{bad_count} records have MFA_BYPASSED with "
                            f"non-SUCCESS/FAILURE outcome"
                        ),
                        field="mfa_outcome",
                        count=bad_count,
                    )
                )
                rejected_rows.update(_col_rows(df, mfa_bad))

        # V019 -- Pseudonymous identifier format for required ID columns.
        for pseudo_col_name in _PSEUDONYM_COLUMNS:
            if pseudo_col_name not in cols:
                continue
            pseudo_col = df[pseudo_col_name]
            # na=False treats null as non-matching (null IDs are also invalid).
            valid_fmt = pseudo_col.str.match(_PSEUDO_PATTERN, na=False)
            invalid_pseudo = ~valid_fmt
            bad_count = int(invalid_pseudo.sum())
            if bad_count > 0:
                errors.append(
                    ValidationError(
                        code="V019",
                        message=(
                            f"{bad_count} records have invalid pseudonym format "
                            f"in {pseudo_col_name!r}"
                        ),
                        field=pseudo_col_name,
                        count=bad_count,
                    )
                )
                rejected_rows.update(_col_rows(df, invalid_pseudo))

        # V020 -- High null rates (warning, not error).
        null_rates: dict[str, float] = {}
        for col_name in df.columns:
            col_str = str(col_name)
            rate = float(df[col_name].isna().mean())
            null_rates[col_str] = rate
            if rate > _NULL_RATE_WARN_THRESHOLD:
                null_count = int(df[col_name].isna().sum())
                warnings.append(
                    ValidationError(
                        code="V020",
                        message=(
                            f"Column {col_str!r} has {rate:.1%} null rate "
                            f"(threshold {_NULL_RATE_WARN_THRESHOLD:.0%})"
                        ),
                        field=col_str,
                        count=null_count,
                    )
                )

        # Compute final counts and overall status.
        rejected_count = len(rejected_rows)
        accepted_count = record_count - rejected_count

        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        return ValidationResult(
            status=status,
            schema_version=detected_sv,
            record_count=record_count,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            errors=tuple(errors),
            warnings=tuple(warnings),
            duplicate_count=duplicate_count,
            null_rates=null_rates,
        )
