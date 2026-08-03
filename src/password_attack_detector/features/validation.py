"""Dataset-level validation for published feature tables.

Mirrors ``data/validation.py``: stable error codes, a frozen result object,
and a status of ``valid`` / ``warning`` / ``invalid``.  Codes are ``F0xx`` so
they never collide with the data layer's ``V0xx``.

**No validation message ever contains an event identifier, a pseudonym, or a
raw row.**  Findings report column names, codes, and counts.  A validator that
printed the offending row would turn every failed build log into a data
disclosure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from enum import StrEnum
from pathlib import Path
from typing import Any

from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
    FeatureDType,
)
from password_attack_detector.features.config import FEATURE_SCHEMA_VERSION
from password_attack_detector.features.serialization import (
    LABELS_FILE,
    SNAPSHOTS_FILE,
    SPLITS_FILE,
    feature_arrow_schema,
)

__all__ = [
    "FeatureValidationError",
    "FeatureValidationResult",
    "FeatureValidationStatus",
    "FeatureValidator",
]

#: Null rate above which a column is reported as suspiciously sparse.
_NULL_RATE_WARN_THRESHOLD: float = 0.95


class FeatureValidationStatus(StrEnum):
    """Overall verdict on a feature dataset."""

    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


@dataclass(frozen=True)
class FeatureValidationError:
    """One validation finding.  Carries no row-level data."""

    code: str
    message: str
    column: str | None = None
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-serialisable mapping."""
        return {
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "count": self.count,
        }


@dataclass(frozen=True)
class FeatureValidationResult:
    """Structured outcome of validating a feature dataset."""

    status: FeatureValidationStatus
    feature_schema_version: str | None
    row_count: int
    feature_count: int
    errors: tuple[FeatureValidationError, ...]
    warnings: tuple[FeatureValidationError, ...]
    duplicate_anchor_count: int
    null_rates: dict[str, float]

    @property
    def passed(self) -> bool:
        """Return whether validation found no errors."""
        return self.status is not FeatureValidationStatus.INVALID

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serialisable mapping."""
        return {
            "status": str(self.status),
            "feature_schema_version": self.feature_schema_version,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "duplicate_anchor_count": self.duplicate_anchor_count,
            "null_rates": self.null_rates,
        }


class FeatureValidator:
    """Validates a feature snapshot table against its catalog."""

    def __init__(self, catalog: FeatureCatalog) -> None:
        self._catalog = catalog

    def validate_parquet(
        self,
        path: Path,
        *,
        labels_path: Path | None = None,
        splits_path: Path | None = None,
    ) -> FeatureValidationResult:
        """Validate the feature table at *path*, optionally with its companions.

        Never raises: an unreadable file becomes an ``F001`` error, because a
        validator that raises cannot report the other findings alongside it.
        """
        import pyarrow.parquet as pq

        errors: list[FeatureValidationError] = []
        warnings: list[FeatureValidationError] = []

        try:
            table = pq.read_table(path)
        except Exception as exc:
            return self._fatal(
                "F001", f"Feature table is unreadable ({type(exc).__name__})"
            )

        if table.num_rows == 0:
            return self._fatal("F002", "Feature table is empty")

        columns = tuple(table.schema.names)
        expected = self._catalog.column_order()

        # F003 / F004 -- column set and order.
        missing = [name for name in expected if name not in columns]
        undeclared = [name for name in columns if name not in expected]
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        if missing or undeclared or duplicates:
            if missing:
                errors.append(
                    FeatureValidationError(
                        "F003",
                        f"{len(missing)} declared feature(s) are missing",
                        column=missing[0],
                        count=len(missing),
                    )
                )
            if undeclared:
                errors.append(
                    FeatureValidationError(
                        "F003",
                        f"{len(undeclared)} undeclared column(s) present",
                        column=undeclared[0],
                        count=len(undeclared),
                    )
                )
            if duplicates:
                errors.append(
                    FeatureValidationError(
                        "F003",
                        f"{len(duplicates)} duplicate column name(s)",
                        column=duplicates[0],
                        count=len(duplicates),
                    )
                )
            return self._result(table.num_rows, len(columns), errors, warnings, 0, {})

        if columns != expected:
            errors.append(
                FeatureValidationError(
                    "F004",
                    "Column order differs from the catalog's declared order",
                    count=1,
                )
            )

        data = table.to_pydict()

        # F005 -- schema version.
        versions = {v for v in data.get("feature_schema_version", []) if v is not None}
        schema_version = next(iter(versions), None) if len(versions) == 1 else None
        if versions != {FEATURE_SCHEMA_VERSION}:
            errors.append(
                FeatureValidationError(
                    "F005",
                    f"feature_schema_version is not uniformly "
                    f"{FEATURE_SCHEMA_VERSION!r}",
                    column="feature_schema_version",
                    count=len(versions),
                )
            )

        # F006 -- prohibited columns.
        prohibited = sorted(set(columns) & PROHIBITED_FEATURE_COLUMNS)
        if prohibited:
            errors.append(
                FeatureValidationError(
                    "F006",
                    f"Prohibited column(s) present: {prohibited}",
                    column=prohibited[0],
                    count=len(prohibited),
                )
            )

        # F007 -- duplicate anchors.
        anchors = data[ANCHOR_EVENT_ID]
        duplicate_count = len(anchors) - len(set(anchors))
        if duplicate_count:
            errors.append(
                FeatureValidationError(
                    "F007",
                    f"{duplicate_count} duplicate anchor identifier(s)",
                    column=ANCHOR_EVENT_ID,
                    count=duplicate_count,
                )
            )

        # F008 -- anchor times must be timezone-aware UTC.
        errors.extend(self._check_anchor_times(table, data))

        # F009 / F010 -- declared types and nullability.
        errors.extend(self._check_types(table))
        errors.extend(self._check_nullability(data))

        # F011-F017 -- numeric sanity.
        errors.extend(self._check_numeric_ranges(data))

        # F018 / F019 -- relationships to the label and split tables.
        errors.extend(self._check_relationships(set(anchors), labels_path, splits_path))

        # Warnings.
        null_rates = self._null_rates(data)
        warnings.extend(self._sparse_column_warnings(null_rates))
        warnings.extend(self._zero_variance_warnings(data))

        return self._result(
            table.num_rows,
            len(columns),
            errors,
            warnings,
            duplicate_count,
            null_rates,
            schema_version=schema_version,
        )

    # -- individual checks --------------------------------------------------

    def _check_anchor_times(
        self, table: Any, data: dict[str, list[Any]]
    ) -> list[FeatureValidationError]:
        field = table.schema.field(ANCHOR_EVENT_TIME)
        if getattr(field.type, "tz", None) != "UTC":
            return [
                FeatureValidationError(
                    "F008",
                    "anchor_event_time is not stored as a UTC timestamp",
                    column=ANCHOR_EVENT_TIME,
                    count=1,
                )
            ]

        naive = sum(
            1
            for value in data[ANCHOR_EVENT_TIME]
            if value is None
            or getattr(value, "tzinfo", None) is None
            or value.utcoffset() != UTC.utcoffset(None)
        )
        if naive:
            return [
                FeatureValidationError(
                    "F008",
                    f"{naive} anchor timestamp(s) are missing or not UTC",
                    column=ANCHOR_EVENT_TIME,
                    count=naive,
                )
            ]
        return []

    def _check_types(self, table: Any) -> list[FeatureValidationError]:
        declared = feature_arrow_schema(self._catalog)
        mismatched = [
            name
            for name in declared.names
            if table.schema.field(name).type != declared.field(name).type
        ]
        if not mismatched:
            return []
        return [
            FeatureValidationError(
                "F009",
                f"{len(mismatched)} column(s) have a type other than the "
                f"catalog declares",
                column=mismatched[0],
                count=len(mismatched),
            )
        ]

    def _check_nullability(
        self, data: dict[str, list[Any]]
    ) -> list[FeatureValidationError]:
        offenders = [
            spec.name
            for spec in self._catalog.specs
            if not spec.nullable
            and any(value is None for value in data.get(spec.name, []))
        ]
        if not offenders:
            return []
        return [
            FeatureValidationError(
                "F010",
                f"{len(offenders)} non-nullable column(s) contain nulls",
                column=offenders[0],
                count=len(offenders),
            )
        ]

    def _check_numeric_ranges(
        self, data: dict[str, list[Any]]
    ) -> list[FeatureValidationError]:
        errors: list[FeatureValidationError] = []
        infinite: list[str] = []
        nan_columns: list[str] = []
        negative_counts: list[str] = []
        out_of_range: list[str] = []
        negative_durations: list[str] = []

        for spec in self._catalog.specs:
            if spec.dtype not in (FeatureDType.FLOAT64, FeatureDType.INT64):
                continue
            values = [v for v in data.get(spec.name, []) if v is not None]
            if not values:
                continue

            if spec.dtype is FeatureDType.FLOAT64:
                if any(isinstance(v, float) and math.isinf(v) for v in values):
                    infinite.append(spec.name)
                if any(isinstance(v, float) and math.isnan(v) for v in values):
                    nan_columns.append(spec.name)

            finite = [
                v
                for v in values
                if not (isinstance(v, float) and (math.isinf(v) or math.isnan(v)))
            ]
            if not finite:
                continue

            if spec.is_count_like and min(finite) < 0:
                negative_counts.append(spec.name)

            if spec.unit == "seconds" and min(finite) < 0:
                negative_durations.append(spec.name)

            if spec.value_range is not None:
                low, high = spec.value_range
                if min(finite) < low or max(finite) > high:
                    out_of_range.append(spec.name)

        for code, columns, description in (
            ("F011", infinite, "contain infinite values"),
            ("F012", nan_columns, "contain NaN"),
            ("F013", negative_counts, "contain negative counts"),
            ("F016", negative_durations, "contain negative durations"),
            ("F017", out_of_range, "fall outside their declared range"),
        ):
            if columns:
                errors.append(
                    FeatureValidationError(
                        code,
                        f"{len(columns)} column(s) {description}",
                        column=columns[0],
                        count=len(columns),
                    )
                )
        return errors

    @staticmethod
    def _check_relationships(
        anchors: set[str],
        labels_path: Path | None,
        splits_path: Path | None,
    ) -> list[FeatureValidationError]:
        import pyarrow.parquet as pq

        errors: list[FeatureValidationError] = []
        for code, path, name in (
            ("F018", labels_path, LABELS_FILE),
            ("F019", splits_path, SPLITS_FILE),
        ):
            if path is None:
                continue
            try:
                companion = set(pq.read_table(path).to_pydict()["event_id"])
            except Exception as exc:
                errors.append(
                    FeatureValidationError(
                        code,
                        f"Cannot read {name} ({type(exc).__name__})",
                        count=1,
                    )
                )
                continue
            difference = anchors ^ companion
            if difference:
                errors.append(
                    FeatureValidationError(
                        code,
                        f"{len(difference)} identifier(s) differ between the "
                        f"feature table and {name}",
                        count=len(difference),
                    )
                )
        return errors

    def _null_rates(self, data: dict[str, list[Any]]) -> dict[str, float]:
        rates: dict[str, float] = {}
        for spec in self._catalog.specs:
            values = data.get(spec.name, [])
            if not values:
                continue
            rates[spec.name] = sum(1 for v in values if v is None) / len(values)
        return rates

    @staticmethod
    def _sparse_column_warnings(
        null_rates: dict[str, float],
    ) -> list[FeatureValidationError]:
        sparse = sorted(
            name
            for name, rate in null_rates.items()
            if rate > _NULL_RATE_WARN_THRESHOLD
        )
        if not sparse:
            return []
        return [
            FeatureValidationError(
                "F022",
                f"{len(sparse)} column(s) are more than "
                f"{_NULL_RATE_WARN_THRESHOLD:.0%} null",
                column=sparse[0],
                count=len(sparse),
            )
        ]

    def _zero_variance_warnings(
        self, data: dict[str, list[Any]]
    ) -> list[FeatureValidationError]:
        constant: list[str] = []
        for spec in self._catalog.specs:
            if spec.group.value == "key":
                continue
            values = [v for v in data.get(spec.name, []) if v is not None]
            if len(values) > 1 and len(set(values)) == 1:
                constant.append(spec.name)
        if not constant:
            return []
        return [
            FeatureValidationError(
                "F023",
                f"{len(constant)} column(s) have zero variance",
                column=constant[0],
                count=len(constant),
            )
        ]

    # -- assembly -----------------------------------------------------------

    def _fatal(self, code: str, message: str) -> FeatureValidationResult:
        return FeatureValidationResult(
            status=FeatureValidationStatus.INVALID,
            feature_schema_version=None,
            row_count=0,
            feature_count=0,
            errors=(FeatureValidationError(code, message, count=1),),
            warnings=(),
            duplicate_anchor_count=0,
            null_rates={},
        )

    @staticmethod
    def _result(
        row_count: int,
        feature_count: int,
        errors: Sequence[FeatureValidationError],
        warnings: Sequence[FeatureValidationError],
        duplicate_count: int,
        null_rates: dict[str, float],
        *,
        schema_version: str | None = None,
    ) -> FeatureValidationResult:
        if errors:
            status = FeatureValidationStatus.INVALID
        elif warnings:
            status = FeatureValidationStatus.WARNING
        else:
            status = FeatureValidationStatus.VALID
        return FeatureValidationResult(
            status=status,
            feature_schema_version=schema_version,
            row_count=row_count,
            feature_count=feature_count,
            errors=tuple(errors),
            warnings=tuple(warnings),
            duplicate_anchor_count=duplicate_count,
            null_rates=null_rates,
        )


def validate_feature_directory(
    directory: Path, catalog: FeatureCatalog
) -> FeatureValidationResult:
    """Validate the three tables in a published feature directory."""
    return FeatureValidator(catalog).validate_parquet(
        directory / SNAPSHOTS_FILE,
        labels_path=(
            directory / LABELS_FILE if (directory / LABELS_FILE).exists() else None
        ),
        splits_path=(
            directory / SPLITS_FILE if (directory / SPLITS_FILE).exists() else None
        ),
    )
