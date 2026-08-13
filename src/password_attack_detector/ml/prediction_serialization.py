"""Deterministic Parquet for prediction tables, with the schema pinned.

Two properties, and both are asserted rather than assumed.

**The schema is pinned, not inferred.**  Column names, order, Arrow types, and
nullability are declared here and handed to the writer.  Inferring them from the
rows would make the file's shape depend on the data -- an all-``null``
probability column would come out as ``null`` type rather than nullable
``float64``, and a reader would then have to guess whether the calibrator was
absent or the inference broke.

**The bytes are deterministic.**  Every writer setting that could vary is fixed:
the format version, the compression codec, the page and row-group sizes, and
statistics.  Nothing observational is written -- no path, no hostname, no
timestamp of our own -- so the same rows written twice, in two directories, at
two times, produce byte-identical files.

Reading is treated as parsing **untrusted input**.  A prediction publication may
have been copied from another machine or edited by hand, so the reader checks the
file's schema against the declared one before it looks at a value, refuses a
duplicate or unexpected column, bounds the row count it will materialise, and
constructs the typed row models -- whose validators then re-derive every stored
decision.  No Parquet metadata is executed, interpreted, or trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.enums import ScoreKind
from password_attack_detector.ml.predictions import (
    ANOMALY_PREDICTION_COLUMNS,
    BINARY_PREDICTION_COLUMNS,
    CATEGORY_PREDICTION_COLUMNS,
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
)

__all__ = [
    "ANOMALY_PREDICTION_SCHEMA",
    "BINARY_PREDICTION_SCHEMA",
    "CATEGORY_PREDICTION_SCHEMA",
    "MAX_PREDICTION_ROWS",
    "PARQUET_WRITER_SETTINGS",
    "read_anomaly_scores",
    "read_binary_predictions",
    "read_category_predictions",
    "write_anomaly_scores",
    "write_binary_predictions",
    "write_category_predictions",
]

#: The pinned Arrow schema for a binary prediction table.
#:
#: ``malicious_probability`` is the one nullable numeric column, and it is
#: nullable for a reason rather than for convenience: a champion with no
#: calibrator has no probability to record, and a zero there would be a
#: confident claim that the row is certainly benign.
BINARY_PREDICTION_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("anchor_event_id", pa.string(), nullable=False),
        pa.field("anchor_event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("malicious_decision_score", pa.float64(), nullable=False),
        pa.field("malicious_probability", pa.float64(), nullable=True),
        pa.field("decision_threshold", pa.float64(), nullable=False),
        pa.field("flagged_malicious", pa.bool_(), nullable=False),
    ]
)

#: The pinned Arrow schema for a category prediction table.
CATEGORY_PREDICTION_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("anchor_event_id", pa.string(), nullable=False),
        pa.field("anchor_event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("predicted_scenario", pa.string(), nullable=False),
        pa.field("category_scores_json", pa.string(), nullable=False),
        pa.field("max_category_score", pa.float64(), nullable=False),
        pa.field("min_category_score", pa.float64(), nullable=False),
    ]
)

#: The pinned Arrow schema for an experimental anomaly table.
#:
#: There is no probability column here at all.  A nullable one would be filled
#: in eventually.
ANOMALY_PREDICTION_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("anchor_event_id", pa.string(), nullable=False),
        pa.field("anchor_event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("anomaly_score", pa.float64(), nullable=False),
        pa.field("anomaly_threshold", pa.float64(), nullable=True),
        pa.field("flagged_anomalous", pa.bool_(), nullable=True),
    ]
)

#: Every writer setting that could otherwise vary between two runs.
#:
#: ``write_statistics`` is off deliberately.  Min/max statistics are derived
#: from the data and would be identical for identical data, but they are also a
#: place where a future writer default could change the bytes without changing
#: the rows -- and the semantic fingerprint, not the file, is what this layer
#: treats as authoritative.
PARQUET_WRITER_SETTINGS: Final[dict[str, Any]] = {
    "version": "2.6",
    "compression": "snappy",
    "write_statistics": False,
    "use_dictionary": False,
    "data_page_size": 1 << 20,
    "write_page_index": False,
    "store_schema": True,
    "coerce_timestamps": "us",
    "allow_truncated_timestamps": False,
}

#: The largest table this reader will materialise.
#:
#: A declared row count is a claim by an untrusted file, and honouring an
#: enormous one is how a validation command becomes an allocation bomb.  The
#: ceiling is far above any dataset this project produces.
MAX_PREDICTION_ROWS: Final[int] = 50_000_000

#: Row-group size, pinned so the same rows always land in the same groups.
_ROW_GROUP_SIZE: Final[int] = 100_000


def _write(table: pa.Table, path: Path) -> None:
    """Write *table* to *path* under the pinned settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        row_group_size=_ROW_GROUP_SIZE,
        **PARQUET_WRITER_SETTINGS,
    )


def _column(values: Sequence[Any], field: pa.Field) -> pa.Array:
    """Return *values* as an Arrow array of exactly the declared type."""
    return pa.array(list(values), type=field.type)


def _table(rows: Sequence[Any], schema: pa.Schema, columns: Sequence[str]) -> pa.Table:
    """Return an Arrow table of *rows* under *schema*, column by declared column."""
    data = [
        _column([getattr(row, name) for row in rows], schema.field(name))
        for name in columns
    ]
    return pa.Table.from_arrays(data, schema=schema)


def write_binary_predictions(rows: Sequence[BinaryPrediction], path: Path) -> None:
    """Write binary predictions to *path* in the declared column order."""
    _write(_table(rows, BINARY_PREDICTION_SCHEMA, BINARY_PREDICTION_COLUMNS), path)


def write_category_predictions(rows: Sequence[CategoryPrediction], path: Path) -> None:
    """Write category predictions to *path* in the declared column order."""
    _write(_table(rows, CATEGORY_PREDICTION_SCHEMA, CATEGORY_PREDICTION_COLUMNS), path)


def write_anomaly_scores(rows: Sequence[AnomalyScore], path: Path) -> None:
    """Write experimental anomaly scores to *path* in the declared column order."""
    _write(_table(rows, ANOMALY_PREDICTION_SCHEMA, ANOMALY_PREDICTION_COLUMNS), path)


def _read_table(path: Path, schema: pa.Schema, what: str) -> list[dict[str, Any]]:
    """Return *path*'s rows as mappings, or refuse.

    The schema is compared before any value is read, so a file whose columns
    were renamed, reordered, retyped, or duplicated is rejected as a whole rather
    than producing rows with plausible-looking wrong values.
    """
    if not path.is_file():
        raise DataValidationError(f"The {what} is not present")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise DataValidationError(
            f"The {what} is not readable Parquet ({type(exc).__name__})"
        ) from None
    stored = parquet.schema_arrow
    names = list(stored.names)
    if len(set(names)) != len(names):
        raise DataValidationError(f"The {what} declares a duplicate column")
    if names != list(schema.names):
        raise DataValidationError(
            f"The {what} declares columns {names}, not the "
            f"{len(schema.names)} declared column(s) in their declared order"
        )
    for field in schema:
        found = stored.field(field.name)
        if found.type != field.type or found.nullable != field.nullable:
            raise DataValidationError(
                f"The {what} column {field.name!r} is not the declared type"
            )
    declared = parquet.metadata.num_rows
    if declared > MAX_PREDICTION_ROWS:
        raise DataValidationError(
            f"The {what} declares {declared:,} rows, above the reader's ceiling"
        )
    try:
        table = parquet.read()
    except Exception as exc:
        raise DataValidationError(
            f"The {what} could not be read ({type(exc).__name__})"
        ) from None
    return [dict(record) for record in table.to_pylist()]


def _aware(value: Any) -> Any:
    """Return a timestamp as a UTC-aware datetime, refusing a naive one.

    Arrow round-trips a ``tz="UTC"`` column as an aware datetime, so a naive one
    here means the column was not the declared type -- which the schema check
    already refuses.  Normalising anyway keeps the row models' own timezone
    requirement from depending on a library convention.
    """
    if hasattr(value, "tzinfo") and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def read_binary_predictions(path: Path) -> tuple[BinaryPrediction, ...]:
    """Return the binary predictions stored at *path*, validated row by row.

    Each row is reconstructed through :class:`~password_attack_detector.ml.\
predictions.BinaryPrediction`, whose validator recomputes the stored decision
    from the stored score and threshold. A row contradicting its own frozen
    predicate is refused here rather than repaired.
    """
    records = _read_table(path, BINARY_PREDICTION_SCHEMA, "binary prediction table")
    return tuple(
        BinaryPrediction(
            anchor_event_id=str(record["anchor_event_id"]),
            anchor_event_time=_aware(record["anchor_event_time"]),
            score_kind=_score_kind(record["malicious_probability"]),
            malicious_decision_score=float(record["malicious_decision_score"]),
            malicious_probability=(
                None
                if record["malicious_probability"] is None
                else float(record["malicious_probability"])
            ),
            decision_threshold=float(record["decision_threshold"]),
            flagged_malicious=bool(record["flagged_malicious"]),
        )
        for record in records
    )


def _score_kind(probability: Any) -> ScoreKind:
    """Return the score kind a stored row was decided on.

    Derived from whether a probability is present, which is exactly the
    invariant the row schema enforces in the other direction: a calibrated row
    carries a probability and an uncalibrated one does not.  Storing the kind as
    a seventh column would let a file claim a kind its own columns contradict.
    """
    return (
        ScoreKind.DECISION_SCORE
        if probability is None
        else ScoreKind.CALIBRATED_PROBABILITY
    )


def read_category_predictions(path: Path) -> tuple[CategoryPrediction, ...]:
    """Return the category predictions stored at *path*, validated row by row."""
    records = _read_table(path, CATEGORY_PREDICTION_SCHEMA, "category prediction table")
    return tuple(
        CategoryPrediction(
            anchor_event_id=str(record["anchor_event_id"]),
            anchor_event_time=_aware(record["anchor_event_time"]),
            predicted_scenario=str(record["predicted_scenario"]),
            category_scores_json=str(record["category_scores_json"]),
            max_category_score=float(record["max_category_score"]),
            min_category_score=float(record["min_category_score"]),
        )
        for record in records
    )


def read_anomaly_scores(path: Path) -> tuple[AnomalyScore, ...]:
    """Return the anomaly scores stored at *path*, validated row by row."""
    records = _read_table(path, ANOMALY_PREDICTION_SCHEMA, "anomaly score table")
    return tuple(
        AnomalyScore(
            anchor_event_id=str(record["anchor_event_id"]),
            anchor_event_time=_aware(record["anchor_event_time"]),
            anomaly_score=float(record["anomaly_score"]),
            anomaly_threshold=(
                None
                if record["anomaly_threshold"] is None
                else float(record["anomaly_threshold"])
            ),
            flagged_anomalous=(
                None
                if record["flagged_anomalous"] is None
                else bool(record["flagged_anomalous"])
            ),
        )
        for record in records
    )


def _assert_schemas_match_the_declared_columns() -> None:
    """Fail at import if an Arrow schema and its column tuple have drifted apart."""
    for schema, columns in (
        (BINARY_PREDICTION_SCHEMA, BINARY_PREDICTION_COLUMNS),
        (CATEGORY_PREDICTION_SCHEMA, CATEGORY_PREDICTION_COLUMNS),
        (ANOMALY_PREDICTION_SCHEMA, ANOMALY_PREDICTION_COLUMNS),
    ):
        if tuple(schema.names) != tuple(columns):
            raise ValueError(
                "a pinned Arrow schema and its declared column order disagree"
            )


_assert_schemas_match_the_declared_columns()
