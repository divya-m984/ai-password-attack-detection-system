"""Tests for feature dataset validation."""

from __future__ import annotations

import random
import re
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    FeatureCatalog,
    build_catalog,
)
from password_attack_detector.features.config import (
    BaselineConfig,
    FeatureConfig,
    SplitConfig,
)
from password_attack_detector.features.engine import FeatureEngine, FeatureFrame
from password_attack_detector.features.serialization import (
    LABELS_FILE,
    SNAPSHOTS_FILE,
    SPLITS_FILE,
    write_feature_labels,
    write_feature_snapshots,
    write_feature_splits,
)
from password_attack_detector.features.splitting import ChronologicalSplitter
from password_attack_detector.features.validation import (
    FeatureValidationStatus,
    FeatureValidator,
    validate_feature_directory,
)
from tests.features.factories import make_event, make_labels

_CONFIG = FeatureConfig(
    windows=("1m", "5m"),
    cardinality_windows=("5m",),
    dispersion_windows=("5m",),
    device_session_windows=("5m",),
    pair_windows=("5m",),
    baseline=BaselineConfig(rate_reference_window="5m"),
    split=SplitConfig(purge=timedelta(minutes=5), max_excluded_fraction=0.5),
)

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)


def _events(count: int = 40) -> list[AuthEvent]:
    rng = random.Random(2468)
    return [
        make_event(
            t=float(index) * 120.0,
            user=f"u{rng.randint(1, 3)}",
            source=f"s{rng.randint(1, 2)}",
            outcome=rng.choice(["success", "failure"]),
            response_time_ms=rng.choice([None, 50, 200]),
            country=rng.choice([None, "US", "GB"]),
            key=str(index),
        )
        for index in range(count)
    ]


def _frame(events: list[AuthEvent] | None = None) -> FeatureFrame:
    resolved = _events() if events is None else events
    return FeatureEngine(_CONFIG, build_catalog(_CONFIG)).run(resolved)


@pytest.fixture()
def catalog() -> FeatureCatalog:
    return build_catalog(_CONFIG)


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """A complete, valid three-table feature dataset."""
    events = _events()
    frame = _frame(events)
    labels = make_labels(events)
    assignments = ChronologicalSplitter(_CONFIG.split).split(events, labels).assignments

    write_feature_snapshots(frame, tmp_path / SNAPSHOTS_FILE)
    write_feature_labels(labels, tmp_path / LABELS_FILE)
    write_feature_splits(assignments, tmp_path / SPLITS_FILE)
    return tmp_path


# --- happy path ------------------------------------------------------------


class TestValidDataset:
    def test_a_well_formed_dataset_passes(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        result = validate_feature_directory(dataset, catalog)
        assert result.status is not FeatureValidationStatus.INVALID, [
            e.message for e in result.errors
        ]
        assert result.passed

    def test_reports_the_shape(self, dataset: Path, catalog: FeatureCatalog) -> None:
        result = validate_feature_directory(dataset, catalog)
        assert result.row_count == 40
        assert result.feature_count == len(catalog)

    def test_reports_the_schema_version(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        result = validate_feature_directory(dataset, catalog)
        assert result.feature_schema_version == "1.0.0"

    def test_reports_null_rates(self, dataset: Path, catalog: FeatureCatalog) -> None:
        result = validate_feature_directory(dataset, catalog)
        assert result.null_rates
        assert all(0.0 <= rate <= 1.0 for rate in result.null_rates.values())

    def test_serialises_to_a_mapping(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        import json

        payload = json.loads(
            json.dumps(validate_feature_directory(dataset, catalog).to_dict())
        )
        assert payload["row_count"] == 40


# --- fatal problems --------------------------------------------------------


class TestFatalProblems:
    def test_unreadable_file_is_f001(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = tmp_path / SNAPSHOTS_FILE
        path.write_bytes(b"not parquet")
        result = FeatureValidator(catalog).validate_parquet(path)
        assert result.status is FeatureValidationStatus.INVALID
        assert result.errors[0].code == "F001"

    def test_missing_file_is_f001(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        result = FeatureValidator(catalog).validate_parquet(tmp_path / "absent.parquet")
        assert result.errors[0].code == "F001"

    def test_empty_table_is_f002(self, tmp_path: Path, catalog: FeatureCatalog) -> None:
        path = tmp_path / SNAPSHOTS_FILE
        empty = FeatureFrame(rows=(), catalog=catalog, stats=_frame().stats)
        write_feature_snapshots(empty, path)
        result = FeatureValidator(catalog).validate_parquet(path)
        assert result.errors[0].code == "F002"


# --- schema problems -------------------------------------------------------


def _write_modified(
    path: Path, frame: FeatureFrame, *, drop: str | None = None, add: str | None = None
) -> None:
    """Write a snapshot table with a column removed or an extra one added."""
    columns = list(frame.catalog.column_order())
    data = {name: [row[name] for row in frame.rows] for name in columns}
    if drop is not None:
        del data[drop]
    if add is not None:
        data[add] = [1] * len(frame.rows)
    pq.write_table(pa.table(data), path)


class TestSchemaProblems:
    def test_missing_column_is_f003(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = tmp_path / SNAPSHOTS_FILE
        _write_modified(path, _frame(), drop="user_attempt_count__5m")
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F003" for e in result.errors)

    def test_undeclared_column_is_f003(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = tmp_path / SNAPSHOTS_FILE
        _write_modified(path, _frame(), add="surprise_column")
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F003" for e in result.errors)

    def test_prohibited_column_is_rejected(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = tmp_path / SNAPSHOTS_FILE
        _write_modified(path, _frame(), add="malicious")
        result = FeatureValidator(catalog).validate_parquet(path)
        assert result.status is FeatureValidationStatus.INVALID

    def test_reordered_columns_are_f004(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        frame = _frame()
        columns = list(frame.catalog.column_order())
        columns[3], columns[4] = columns[4], columns[3]
        data = {name: [row[name] for row in frame.rows] for name in columns}
        path = tmp_path / SNAPSHOTS_FILE
        pq.write_table(pa.table(data), path)
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F004" for e in result.errors)

    def test_wrong_schema_version_is_f005(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        frame = _frame()
        rows = tuple({**row, "feature_schema_version": "9.9.9"} for row in frame.rows)
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(
            FeatureFrame(rows=rows, catalog=catalog, stats=frame.stats), path
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F005" for e in result.errors)

    def test_duplicate_anchor_is_f007(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        frame = _frame()
        duplicated = (*frame.rows, frame.rows[0])
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(
            FeatureFrame(rows=duplicated, catalog=catalog, stats=frame.stats), path
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F007" for e in result.errors)
        assert result.duplicate_anchor_count == 1


# --- value problems --------------------------------------------------------


class TestValueProblems:
    def _write_with(
        self, tmp_path: Path, catalog: FeatureCatalog, column: str, value: object
    ) -> Path:
        frame = _frame()
        rows = tuple({**row, column: value} for row in frame.rows)
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(
            FeatureFrame(rows=rows, catalog=catalog, stats=frame.stats), path
        )
        return path

    def test_infinite_value_is_f011(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = self._write_with(
            tmp_path, catalog, "user_mean_response_time_ms__5m", float("inf")
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F011" for e in result.errors)

    def test_nan_value_is_f012(self, tmp_path: Path, catalog: FeatureCatalog) -> None:
        path = self._write_with(
            tmp_path, catalog, "user_mean_response_time_ms__5m", float("nan")
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F012" for e in result.errors)

    def test_negative_count_is_f013(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = self._write_with(tmp_path, catalog, "user_attempt_count__5m", -1)
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F013" for e in result.errors)

    def test_rate_outside_the_unit_interval_is_f017(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = self._write_with(tmp_path, catalog, "user_failure_rate__5m", 1.5)
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F017" for e in result.errors)

    def test_cyclical_value_outside_range_is_f017(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = self._write_with(tmp_path, catalog, "hour_sin", 3.0)
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F017" for e in result.errors)

    def test_negative_duration_is_f016(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        path = self._write_with(
            tmp_path, catalog, "seconds_since_user_previous_event", -5.0
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code in {"F016", "F017"} for e in result.errors)

    def test_null_in_a_non_nullable_column_is_f010(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        frame = _frame()
        columns = list(frame.catalog.column_order())
        data = {name: [row[name] for row in frame.rows] for name in columns}
        data["user_attempt_count__5m"] = [None, *data["user_attempt_count__5m"][1:]]
        path = tmp_path / SNAPSHOTS_FILE
        pq.write_table(pa.table(data), path)
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(e.code == "F010" for e in result.errors)


# --- relationships ---------------------------------------------------------


class TestRelationships:
    def test_matched_tables_pass(self, dataset: Path, catalog: FeatureCatalog) -> None:
        result = validate_feature_directory(dataset, catalog)
        assert not any(e.code in {"F018", "F019"} for e in result.errors)

    def test_label_mismatch_is_f018(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        events = _events()
        write_feature_labels(make_labels(events[:-5]), dataset / LABELS_FILE)
        result = validate_feature_directory(dataset, catalog)
        assert any(e.code == "F018" for e in result.errors)

    def test_split_mismatch_is_f019(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        events = _events()
        labels = make_labels(events)
        assignments = (
            ChronologicalSplitter(_CONFIG.split).split(events, labels).assignments
        )
        write_feature_splits(assignments[:-3], dataset / SPLITS_FILE)
        result = validate_feature_directory(dataset, catalog)
        assert any(e.code == "F019" for e in result.errors)

    def test_unreadable_companion_is_reported(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        (dataset / LABELS_FILE).write_bytes(b"garbage")
        result = validate_feature_directory(dataset, catalog)
        assert any(e.code == "F018" for e in result.errors)


# --- warnings --------------------------------------------------------------


class TestWarnings:
    def test_zero_variance_columns_warn_not_fail(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        result = validate_feature_directory(dataset, catalog)
        # current_has_device is constant because the canonical schema makes
        # device_id mandatory; that is a warning, not an error.
        assert result.status is not FeatureValidationStatus.INVALID
        assert any(w.code == "F023" for w in result.warnings)

    def test_a_very_sparse_column_warns(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        frame = _frame()
        rows = tuple({**row, "user_failure_rate__5m": None} for row in frame.rows)
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(
            FeatureFrame(rows=rows, catalog=catalog, stats=frame.stats), path
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        assert any(w.code == "F022" for w in result.warnings)

    def test_warnings_alone_do_not_invalidate(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        result = validate_feature_directory(dataset, catalog)
        assert result.passed


# --- privacy ---------------------------------------------------------------


class TestValidationPrivacy:
    def test_no_finding_contains_an_identifier(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        events = _events()
        write_feature_labels(make_labels(events[:-5]), dataset / LABELS_FILE)
        result = validate_feature_directory(dataset, catalog)
        text = " ".join(e.message for e in (*result.errors, *result.warnings))
        assert not _UUID_RE.search(text)
        assert not _PSEUDONYM_RE.search(text)

    def test_findings_report_columns_and_counts_only(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        frame = _frame()
        rows = tuple({**row, "user_attempt_count__5m": -1} for row in frame.rows)
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(
            FeatureFrame(rows=rows, catalog=catalog, stats=frame.stats), path
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        finding = next(e for e in result.errors if e.code == "F013")
        assert finding.column == "user_attempt_count__5m"
        assert finding.count >= 1
        assert not _UUID_RE.search(finding.message)

    def test_serialised_result_carries_no_identifiers(
        self, dataset: Path, catalog: FeatureCatalog
    ) -> None:
        import json

        text = json.dumps(validate_feature_directory(dataset, catalog).to_dict())
        assert not _UUID_RE.search(text)
        assert not _PSEUDONYM_RE.search(text)

    def test_anchor_column_name_is_reported_without_values(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        frame = _frame()
        duplicated = (*frame.rows, frame.rows[0])
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(
            FeatureFrame(rows=duplicated, catalog=catalog, stats=frame.stats), path
        )
        result = FeatureValidator(catalog).validate_parquet(path)
        finding = next(e for e in result.errors if e.code == "F007")
        assert finding.column == ANCHOR_EVENT_ID
        assert not _UUID_RE.search(finding.message)
