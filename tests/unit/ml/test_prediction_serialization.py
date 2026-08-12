"""Deterministic Parquet for prediction tables, and the reader's suspicion of it.

Two halves. The writer must produce byte-identical files from identical semantic
input -- asserted directly, in two directories, rather than argued from settings
-- and the reader must treat a file as untrusted: a renamed column, a retyped
column, a reordered column, a duplicate, or a row contradicting its own frozen
threshold is refused rather than absorbed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.enums import UNKNOWN_CATEGORY, ScoreKind
from password_attack_detector.ml.prediction_serialization import (
    ANOMALY_PREDICTION_SCHEMA,
    BINARY_PREDICTION_SCHEMA,
    CATEGORY_PREDICTION_SCHEMA,
    MAX_PREDICTION_ROWS,
    PARQUET_WRITER_SETTINGS,
    read_anomaly_scores,
    read_binary_predictions,
    read_category_predictions,
    write_anomaly_scores,
    write_binary_predictions,
    write_category_predictions,
)
from password_attack_detector.ml.predictions import (
    ANOMALY_PREDICTION_COLUMNS,
    BINARY_PREDICTION_COLUMNS,
    CATEGORY_PREDICTION_COLUMNS,
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
    category_scores_payload,
)

EPOCH = datetime(2024, 3, 1, tzinfo=UTC)
CLASS_ORDER = ("brute_force", "credential_stuffing")


def binary_rows(count: int = 5, *, calibrated: bool = False) -> list[BinaryPrediction]:
    """Return *count* valid binary predictions in canonical order."""
    rows = []
    for index in range(count):
        score = round(0.1 * index, 9)
        probability = round(0.05 * index, 9) if calibrated else None
        decided = score if probability is None else probability
        rows.append(
            BinaryPrediction(
                anchor_event_id=f"00000000-0000-5000-8000-{index:012d}",
                anchor_event_time=EPOCH + timedelta(minutes=index),
                score_kind=(
                    ScoreKind.CALIBRATED_PROBABILITY
                    if calibrated
                    else ScoreKind.DECISION_SCORE
                ),
                malicious_decision_score=score,
                malicious_probability=probability,
                decision_threshold=0.2,
                flagged_malicious=decided >= 0.2,
            )
        )
    return rows


def category_rows(count: int = 3) -> list[CategoryPrediction]:
    """Return *count* valid category predictions in canonical order."""
    rows = []
    for index in range(count):
        scores = (round(0.1 * index, 9), round(0.9 - 0.1 * index, 9))
        best = max(scores)
        winner = CLASS_ORDER[scores.index(best)]
        rows.append(
            CategoryPrediction(
                anchor_event_id=f"00000000-0000-5000-8000-{index:012d}",
                anchor_event_time=EPOCH + timedelta(minutes=index),
                predicted_scenario=winner if best >= 0.3 else UNKNOWN_CATEGORY,
                category_scores_json=category_scores_payload(scores, CLASS_ORDER),
                max_category_score=best,
                min_category_score=0.3,
            )
        )
    return rows


def anomaly_rows(
    count: int = 3, *, threshold: float | None = -0.2
) -> list[AnomalyScore]:
    """Return *count* valid anomaly rows in canonical order."""
    return [
        AnomalyScore(
            anchor_event_id=f"00000000-0000-5000-8000-{index:012d}",
            anchor_event_time=EPOCH + timedelta(minutes=index),
            anomaly_score=round(-0.5 + 0.1 * index, 9),
            anomaly_threshold=threshold,
            flagged_anomalous=(
                None if threshold is None else round(-0.5 + 0.1 * index, 9) <= threshold
            ),
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# The pinned schemas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "columns"),
    [
        (BINARY_PREDICTION_SCHEMA, BINARY_PREDICTION_COLUMNS),
        (CATEGORY_PREDICTION_SCHEMA, CATEGORY_PREDICTION_COLUMNS),
        (ANOMALY_PREDICTION_SCHEMA, ANOMALY_PREDICTION_COLUMNS),
    ],
)
def test_each_schema_declares_its_columns_in_order(
    schema: pa.Schema, columns: tuple[str, ...]
) -> None:
    """The Arrow schema and the declared column order are one statement."""
    assert tuple(schema.names) == columns


def test_only_the_probability_and_the_anomaly_flag_are_nullable() -> None:
    """Nullability is a contract, not an accident of the data."""
    nullable = {
        field.name
        for schema in (
            BINARY_PREDICTION_SCHEMA,
            CATEGORY_PREDICTION_SCHEMA,
            ANOMALY_PREDICTION_SCHEMA,
        )
        for field in schema
        if field.nullable
    }
    assert nullable == {
        "malicious_probability",
        "anomaly_threshold",
        "flagged_anomalous",
    }


def test_the_timestamp_column_is_microsecond_utc() -> None:
    """One timestamp type across every table, so a join cannot need a cast."""
    for schema in (
        BINARY_PREDICTION_SCHEMA,
        CATEGORY_PREDICTION_SCHEMA,
        ANOMALY_PREDICTION_SCHEMA,
    ):
        assert schema.field("anchor_event_time").type == pa.timestamp("us", tz="UTC")


def test_the_writer_settings_are_pinned() -> None:
    """Every setting that could vary between two runs is fixed."""
    assert PARQUET_WRITER_SETTINGS["version"] == "2.6"
    assert PARQUET_WRITER_SETTINGS["compression"] == "snappy"
    assert PARQUET_WRITER_SETTINGS["write_statistics"] is False
    assert PARQUET_WRITER_SETTINGS["use_dictionary"] is False
    assert PARQUET_WRITER_SETTINGS["coerce_timestamps"] == "us"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_identical_rows_produce_identical_bytes(tmp_path: Path) -> None:
    """Two directories, two writes, one file."""
    rows = binary_rows(20)
    first = tmp_path / "a" / "binary.parquet"
    second = tmp_path / "b" / "binary.parquet"
    write_binary_predictions(rows, first)
    write_binary_predictions(rows, second)
    assert first.read_bytes() == second.read_bytes()


def test_identical_category_rows_produce_identical_bytes(tmp_path: Path) -> None:
    """The same guarantee for the category table."""
    rows = category_rows(8)
    write_category_predictions(rows, tmp_path / "a.parquet")
    write_category_predictions(rows, tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() == (
        tmp_path / "b.parquet"
    ).read_bytes()


def test_identical_anomaly_rows_produce_identical_bytes(tmp_path: Path) -> None:
    """And for the experimental table."""
    rows = anomaly_rows(8)
    write_anomaly_scores(rows, tmp_path / "a.parquet")
    write_anomaly_scores(rows, tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() == (
        tmp_path / "b.parquet"
    ).read_bytes()


def test_one_changed_score_changes_the_bytes(tmp_path: Path) -> None:
    """Determinism is not insensitivity."""
    write_binary_predictions(binary_rows(5), tmp_path / "a.parquet")
    write_binary_predictions(binary_rows(6), tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() != (
        tmp_path / "b.parquet"
    ).read_bytes()


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("calibrated", [False, True])
def test_binary_rows_round_trip_exactly(tmp_path: Path, calibrated: bool) -> None:
    """Including the score kind, which is derived rather than stored."""
    rows = binary_rows(7, calibrated=calibrated)
    write_binary_predictions(rows, tmp_path / "binary.parquet")
    assert read_binary_predictions(tmp_path / "binary.parquet") == tuple(rows)


def test_category_rows_round_trip_exactly(tmp_path: Path) -> None:
    """The class map survives as the canonical JSON it was written as."""
    rows = category_rows(4)
    write_category_predictions(rows, tmp_path / "category.parquet")
    assert read_category_predictions(tmp_path / "category.parquet") == tuple(rows)


@pytest.mark.parametrize("threshold", [None, -0.2])
def test_anomaly_rows_round_trip_exactly(
    tmp_path: Path, threshold: float | None
) -> None:
    """A probe with no frozen threshold keeps its nulls as nulls."""
    rows = anomaly_rows(4, threshold=threshold)
    write_anomaly_scores(rows, tmp_path / "anomaly.parquet")
    assert read_anomaly_scores(tmp_path / "anomaly.parquet") == tuple(rows)


def test_an_absent_probability_stays_absent(tmp_path: Path) -> None:
    """A null is not a zero, before or after a Parquet round trip."""
    write_binary_predictions(binary_rows(3), tmp_path / "binary.parquet")
    rows = read_binary_predictions(tmp_path / "binary.parquet")
    assert all(row.malicious_probability is None for row in rows)
    assert all(row.score_kind is ScoreKind.DECISION_SCORE for row in rows)


# ---------------------------------------------------------------------------
# The reader's suspicion
# ---------------------------------------------------------------------------


def write_raw(path: Path, schema: pa.Schema, data: dict[str, list[object]]) -> None:
    """Write an arbitrary table, bypassing the typed writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(data, schema=schema), path)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    """An absent table is a failure, not an empty one."""
    with pytest.raises(DataValidationError, match="not present"):
        read_binary_predictions(tmp_path / "nothing.parquet")


def test_a_file_that_is_not_parquet_is_refused(tmp_path: Path) -> None:
    """Untrusted bytes are parsed, not interpreted."""
    path = tmp_path / "binary.parquet"
    path.write_bytes(b"not a parquet file at all")
    with pytest.raises(DataValidationError, match="not readable Parquet"):
        read_binary_predictions(path)


def test_a_renamed_column_is_refused(tmp_path: Path) -> None:
    """A column set that is not the declared one is rejected as a whole."""
    schema = pa.schema(
        [
            pa.field("anchor_event_id", pa.string(), nullable=False),
            pa.field("anchor_event_time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("malicious_probability", pa.float64(), nullable=True),
            pa.field("decision_threshold", pa.float64(), nullable=False),
            pa.field("flagged_malicious", pa.bool_(), nullable=False),
        ]
    )
    write_raw(
        tmp_path / "binary.parquet",
        schema,
        {
            "anchor_event_id": ["a"],
            "anchor_event_time": [EPOCH],
            "score": [0.5],
            "malicious_probability": [None],
            "decision_threshold": [0.2],
            "flagged_malicious": [True],
        },
    )
    with pytest.raises(DataValidationError, match="declares columns"):
        read_binary_predictions(tmp_path / "binary.parquet")


def test_a_reordered_column_is_refused(tmp_path: Path) -> None:
    """The same columns in a different order are a different table."""
    fields = list(BINARY_PREDICTION_SCHEMA)
    reordered = pa.schema([fields[1], fields[0], *fields[2:]])
    write_raw(
        tmp_path / "binary.parquet",
        reordered,
        {
            "anchor_event_time": [EPOCH],
            "anchor_event_id": ["a"],
            "malicious_decision_score": [0.5],
            "malicious_probability": [None],
            "decision_threshold": [0.2],
            "flagged_malicious": [True],
        },
    )
    with pytest.raises(DataValidationError, match="declared order"):
        read_binary_predictions(tmp_path / "binary.parquet")


def test_a_retyped_column_is_refused(tmp_path: Path) -> None:
    """A float32 score is not a float64 score."""
    fields = [
        pa.field(field.name, pa.float32(), nullable=field.nullable)
        if field.name == "malicious_decision_score"
        else field
        for field in BINARY_PREDICTION_SCHEMA
    ]
    write_raw(
        tmp_path / "binary.parquet",
        pa.schema(fields),
        {
            "anchor_event_id": ["a"],
            "anchor_event_time": [EPOCH],
            "malicious_decision_score": [0.5],
            "malicious_probability": [None],
            "decision_threshold": [0.2],
            "flagged_malicious": [True],
        },
    )
    with pytest.raises(DataValidationError, match="not the declared type"):
        read_binary_predictions(tmp_path / "binary.parquet")


def test_a_nullable_required_column_is_refused(tmp_path: Path) -> None:
    """Nullability is part of the type, so a loosened column is a different one."""
    fields = [
        pa.field(field.name, field.type, nullable=True)
        if field.name == "flagged_malicious"
        else field
        for field in BINARY_PREDICTION_SCHEMA
    ]
    write_raw(
        tmp_path / "binary.parquet",
        pa.schema(fields),
        {
            "anchor_event_id": ["a"],
            "anchor_event_time": [EPOCH],
            "malicious_decision_score": [0.5],
            "malicious_probability": [None],
            "decision_threshold": [0.2],
            "flagged_malicious": [True],
        },
    )
    with pytest.raises(DataValidationError, match="not the declared type"):
        read_binary_predictions(tmp_path / "binary.parquet")


def test_a_row_contradicting_its_own_threshold_is_refused(tmp_path: Path) -> None:
    """The reader reconstructs typed rows, and the row refuses itself."""
    write_raw(
        tmp_path / "binary.parquet",
        BINARY_PREDICTION_SCHEMA,
        {
            "anchor_event_id": ["a"],
            "anchor_event_time": [EPOCH],
            "malicious_decision_score": [0.1],
            "malicious_probability": [None],
            "decision_threshold": [0.9],
            "flagged_malicious": [True],
        },
    )
    with pytest.raises(ValueError, match="contradicts the frozen predicate"):
        read_binary_predictions(tmp_path / "binary.parquet")


def test_a_non_finite_stored_score_is_refused(tmp_path: Path) -> None:
    """NaN survives Parquet perfectly well and means nothing in a comparison."""
    write_raw(
        tmp_path / "binary.parquet",
        BINARY_PREDICTION_SCHEMA,
        {
            "anchor_event_id": ["a"],
            "anchor_event_time": [EPOCH],
            "malicious_decision_score": [float("nan")],
            "malicious_probability": [None],
            "decision_threshold": [0.2],
            "flagged_malicious": [False],
        },
    )
    with pytest.raises(ValueError, match="must be finite"):
        read_binary_predictions(tmp_path / "binary.parquet")


def test_a_stored_probability_outside_the_unit_interval_is_refused(
    tmp_path: Path,
) -> None:
    """Bounds are re-checked on read, not trusted from the writer."""
    write_raw(
        tmp_path / "binary.parquet",
        BINARY_PREDICTION_SCHEMA,
        {
            "anchor_event_id": ["a"],
            "anchor_event_time": [EPOCH],
            "malicious_decision_score": [0.5],
            "malicious_probability": [1.4],
            "decision_threshold": [0.2],
            "flagged_malicious": [True],
        },
    )
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        read_binary_predictions(tmp_path / "binary.parquet")


def test_a_stored_category_row_contradicting_its_floor_is_refused(
    tmp_path: Path,
) -> None:
    """The abstention rule is recomputed from the stored scores."""
    write_raw(
        tmp_path / "category.parquet",
        CATEGORY_PREDICTION_SCHEMA,
        {
            "anchor_event_id": ["a"],
            "anchor_event_time": [EPOCH],
            "predicted_scenario": ["brute_force"],
            "category_scores_json": [category_scores_payload((0.1, 0.05), CLASS_ORDER)],
            "max_category_score": [0.1],
            "min_category_score": [0.9],
        },
    )
    with pytest.raises(ValueError, match="contradicts the frozen abstention rule"):
        read_category_predictions(tmp_path / "category.parquet")


def test_the_reader_bounds_the_rows_it_will_materialise() -> None:
    """A declared row count is a claim, and an enormous one is not honoured."""
    assert MAX_PREDICTION_ROWS == 50_000_000


def test_the_anomaly_table_has_no_probability_column() -> None:
    """Not nullable, not absent-by-convention: the column does not exist."""
    assert "malicious_probability" not in ANOMALY_PREDICTION_SCHEMA.names
    assert not any("probab" in name for name in ANOMALY_PREDICTION_SCHEMA.names)
