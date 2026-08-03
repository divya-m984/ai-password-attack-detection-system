"""Tests for feature serialization, staged publication, and the manifest."""

from __future__ import annotations

import json
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.exceptions import (
    DataValidationError,
    FeatureComputationError,
)
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    FeatureCatalog,
    FeatureDType,
    build_catalog,
)
from password_attack_detector.features.config import (
    BaselineConfig,
    FeatureConfig,
    SplitConfig,
)
from password_attack_detector.features.engine import FeatureEngine, FeatureFrame
from password_attack_detector.features.manifest import (
    FEATURE_MANIFEST_VERSION,
    build_feature_manifest,
    verify_feature_dataset,
)
from password_attack_detector.features.serialization import (
    LABELS_FILE,
    MANIFEST_FILE,
    PUBLISHED_FILES,
    SNAPSHOTS_FILE,
    SPLITS_FILE,
    FeaturePublisher,
    compute_features_fingerprint,
    feature_arrow_schema,
    labels_for_events,
    write_feature_labels,
    write_feature_snapshots,
    write_feature_splits,
)
from password_attack_detector.features.splitting import ChronologicalSplitter
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


def _events(count: int = 60) -> list[AuthEvent]:
    rng = random.Random(9001)
    return [
        make_event(
            t=float(index) * 120.0,
            user=f"u{rng.randint(1, 4)}",
            source=f"s{rng.randint(1, 3)}",
            outcome=rng.choice(["success", "failure"]),
            country=rng.choice([None, "US", "GB"]),
            response_time_ms=rng.choice([None, 40, 300]),
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


# --- Arrow schema ----------------------------------------------------------


class TestArrowSchema:
    def test_schema_covers_every_catalog_column_in_order(
        self, catalog: FeatureCatalog
    ) -> None:
        schema = feature_arrow_schema(catalog)
        assert tuple(schema.names) == catalog.column_order()

    def test_field_types_match_the_declared_dtypes(
        self, catalog: FeatureCatalog
    ) -> None:
        import pyarrow as pa

        expected = {
            FeatureDType.INT64: pa.int64(),
            FeatureDType.FLOAT64: pa.float64(),
            FeatureDType.BOOL: pa.bool_(),
            FeatureDType.STRING: pa.string(),
            FeatureDType.TIMESTAMP: pa.timestamp("us", tz="UTC"),
        }
        schema = feature_arrow_schema(catalog)
        for spec in catalog.specs:
            assert schema.field(spec.name).type == expected[spec.dtype], spec.name

    def test_nullability_matches_the_catalog(self, catalog: FeatureCatalog) -> None:
        schema = feature_arrow_schema(catalog)
        for spec in catalog.specs:
            assert schema.field(spec.name).nullable == spec.nullable, spec.name


# --- writing ---------------------------------------------------------------


class TestWriteSnapshots:
    def test_round_trips_through_parquet(self, tmp_path: Path) -> None:
        frame = _frame()
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(frame, path)
        table = pq.read_table(path)
        assert table.num_rows == len(frame.rows)

    def test_column_order_is_preserved_on_disk(self, tmp_path: Path) -> None:
        frame = _frame()
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(frame, path)
        assert tuple(pq.read_schema(path).names) == frame.catalog.column_order()

    def test_written_types_match_the_catalog(self, tmp_path: Path) -> None:
        frame = _frame()
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(frame, path)
        written = pq.read_schema(path)
        declared = feature_arrow_schema(frame.catalog)
        for name in declared.names:
            assert written.field(name).type == declared.field(name).type, name

    def test_utc_is_preserved(self, tmp_path: Path) -> None:
        frame = _frame()
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(frame, path)
        field = pq.read_schema(path).field(ANCHOR_EVENT_TIME)
        assert field.type.tz == "UTC"

    def test_values_survive_the_round_trip(self, tmp_path: Path) -> None:
        frame = _frame()
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(frame, path)
        restored = pq.read_table(path).to_pylist()
        for original, back in zip(frame.rows, restored, strict=True):
            assert back[ANCHOR_EVENT_ID] == original[ANCHOR_EVENT_ID]
            assert back["user_attempt_count__5m"] == original["user_attempt_count__5m"]

    def test_nulls_stay_null_not_zero(self, tmp_path: Path) -> None:
        frame = _frame([make_event(t=0.0)])
        path = tmp_path / SNAPSHOTS_FILE
        write_feature_snapshots(frame, path)
        row = pq.read_table(path).to_pylist()[0]
        assert row["user_failure_rate__5m"] is None
        assert row["user_attempt_count__5m"] == 0

    def test_rows_that_do_not_match_the_catalog_are_rejected(
        self, tmp_path: Path, catalog: FeatureCatalog
    ) -> None:
        broken = FeatureFrame(
            rows=({"anchor_event_id": "x"},),
            catalog=catalog,
            stats=_frame().stats,
        )
        with pytest.raises(FeatureComputationError, match="do not match the catalog"):
            write_feature_snapshots(broken, tmp_path / SNAPSHOTS_FILE)

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / SNAPSHOTS_FILE
        write_feature_snapshots(_frame(), path)
        assert path.exists()


class TestWriteLabels:
    def test_writes_only_the_four_supervised_columns(self, tmp_path: Path) -> None:
        events = _events(10)
        path = tmp_path / LABELS_FILE
        write_feature_labels(make_labels(events), path)
        assert tuple(pq.read_schema(path).names) == (
            "event_id",
            "attack_class",
            "malicious",
            "supervised_training_eligible",
        )

    def test_campaign_id_is_never_published(self, tmp_path: Path) -> None:
        # The splitter reads campaign_id internally for group isolation; it is
        # never written beside anything a model consumes.
        events = _events(10)
        path = tmp_path / LABELS_FILE
        write_feature_labels(make_labels(events), path)
        assert "campaign_id" not in pq.read_schema(path).names

    def test_label_values_round_trip(self, tmp_path: Path) -> None:
        from password_attack_detector.data.enums import ScenarioType

        events = _events(5)
        labels = make_labels(events, scenario=ScenarioType.BRUTE_FORCE)
        path = tmp_path / LABELS_FILE
        write_feature_labels(labels, path)
        rows = pq.read_table(path).to_pylist()
        assert all(row["attack_class"] == "brute_force" for row in rows)
        assert all(row["malicious"] for row in rows)


class TestWriteSplits:
    def test_writes_the_three_split_columns(self, tmp_path: Path) -> None:
        events = _events(60)
        labels = make_labels(events)
        result = ChronologicalSplitter(_CONFIG.split).split(events, labels)
        path = tmp_path / SPLITS_FILE
        write_feature_splits(result.assignments, path)
        assert tuple(pq.read_schema(path).names) == (
            "event_id",
            "split",
            "exclusion_reason",
        )

    def test_exclusion_reason_is_nullable(self, tmp_path: Path) -> None:
        events = _events(60)
        labels = make_labels(events)
        result = ChronologicalSplitter(_CONFIG.split).split(events, labels)
        path = tmp_path / SPLITS_FILE
        write_feature_splits(result.assignments, path)
        assert pq.read_schema(path).field("exclusion_reason").nullable

    def test_every_event_appears_once(self, tmp_path: Path) -> None:
        events = _events(60)
        labels = make_labels(events)
        result = ChronologicalSplitter(_CONFIG.split).split(events, labels)
        path = tmp_path / SPLITS_FILE
        write_feature_splits(result.assignments, path)
        rows = pq.read_table(path).to_pylist()
        assert len({row["event_id"] for row in rows}) == len(events)


# --- fingerprinting --------------------------------------------------------


class TestFeatureFingerprint:
    def test_is_hex_sha256(self) -> None:
        fingerprint = compute_features_fingerprint(_frame())
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")

    def test_is_stable_across_runs(self) -> None:
        events = _events()
        assert compute_features_fingerprint(
            _frame(events)
        ) == compute_features_fingerprint(_frame(events))

    def test_is_independent_of_input_row_order(self) -> None:
        events = _events()
        shuffled = list(events)
        random.Random(4).shuffle(shuffled)
        assert compute_features_fingerprint(
            _frame(events)
        ) == compute_features_fingerprint(_frame(shuffled))

    def test_reacts_to_a_content_change(self) -> None:
        base = _events(40)
        changed = [*base, make_event(t=99999.0, user="u_new", key="extra")]
        assert compute_features_fingerprint(
            _frame(base)
        ) != compute_features_fingerprint(_frame(changed))


class TestLabelsForEvents:
    def test_returns_labels_in_the_requested_order(self) -> None:
        events = _events(10)
        labels = make_labels(events)
        ids = [str(e.event_id) for e in reversed(events)]
        assert [str(label.event_id) for label in labels_for_events(ids, labels)] == ids

    def test_missing_labels_are_rejected(self) -> None:
        events = _events(10)
        labels = make_labels(events[:-2])
        ids = [str(e.event_id) for e in events]
        with pytest.raises(DataValidationError, match="no matching ground-truth"):
            labels_for_events(ids, labels)


# --- staged publication ----------------------------------------------------


def _publish(
    tmp_path: Path, *, overwrite: bool = False, events: list[AuthEvent] | None = None
) -> Any:
    resolved = _events() if events is None else events
    frame = _frame(resolved)
    labels = make_labels(resolved)
    assignments = (
        ChronologicalSplitter(_CONFIG.split).split(resolved, labels).assignments
    )
    ordered_labels = labels_for_events(
        [row[ANCHOR_EVENT_ID] for row in frame.rows], labels
    )
    publisher = FeaturePublisher(tmp_path / "out", overwrite=overwrite)
    return publisher.publish(
        frame, ordered_labels, assignments, {"manifest_version": "1.0.0"}
    )


class TestFeaturePublisher:
    def test_writes_every_artifact(self, tmp_path: Path) -> None:
        published = _publish(tmp_path)
        for name in (*PUBLISHED_FILES, MANIFEST_FILE):
            assert (published.directory / name).exists(), name

    def test_reports_the_published_shape(self, tmp_path: Path) -> None:
        published = _publish(tmp_path)
        assert published.row_count == 60
        assert published.feature_count > 50
        assert len(published.feature_fingerprint) == 64

    def test_refuses_to_publish_an_empty_dataset(self, tmp_path: Path) -> None:
        empty = FeatureFrame(
            rows=(), catalog=build_catalog(_CONFIG), stats=_frame().stats
        )
        publisher = FeaturePublisher(tmp_path / "out")
        with pytest.raises(DataValidationError, match="empty feature dataset"):
            publisher.publish(empty, [], [], {})

    def test_refuses_to_overwrite_by_default(self, tmp_path: Path) -> None:
        _publish(tmp_path)
        with pytest.raises(DataValidationError, match="already exist"):
            _publish(tmp_path)

    def test_overwrites_when_asked(self, tmp_path: Path) -> None:
        _publish(tmp_path)
        published = _publish(tmp_path, overwrite=True)
        assert published.manifest_path.exists()

    def test_no_staging_directories_are_left_behind(self, tmp_path: Path) -> None:
        _publish(tmp_path)
        leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("_pad_")]
        assert not leftovers

    def test_manifest_is_promoted_last(self, tmp_path: Path) -> None:
        # Its presence is the signal that everything else is already in place.
        published = _publish(tmp_path)
        manifest_mtime = published.manifest_path.stat().st_mtime_ns
        for name in PUBLISHED_FILES:
            assert (published.directory / name).stat().st_mtime_ns <= manifest_mtime

    def test_rollback_restores_the_previous_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = _publish(tmp_path)
        original = first.manifest_path.read_text(encoding="utf-8")

        import shutil as shutil_module

        from password_attack_detector.features import serialization

        calls = {"n": 0}
        real_move = shutil_module.move

        def flaky_move(src: str, dst: str) -> Any:
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("simulated failure mid-promotion")
            return real_move(src, dst)

        monkeypatch.setattr(serialization.shutil, "move", flaky_move)  # type: ignore[attr-defined]

        with pytest.raises(OSError, match="simulated failure"):
            _publish(tmp_path, overwrite=True)

        assert first.manifest_path.read_text(encoding="utf-8") == original
        for name in PUBLISHED_FILES:
            assert (first.directory / name).exists()

    def test_failed_publication_leaves_no_staging_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from password_attack_detector.features import serialization

        def always_fails(src: str, dst: str) -> Any:
            raise OSError("nope")

        monkeypatch.setattr(serialization.shutil, "move", always_fails)  # type: ignore[attr-defined]
        with pytest.raises(OSError):
            _publish(tmp_path)
        leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("_pad_")]
        assert not leftovers


# --- manifest --------------------------------------------------------------


class TestFeatureManifest:
    def _build(self, tmp_path: Path) -> Any:
        events = _events(40)
        frame = _frame(events)
        labels = make_labels(events)
        assignments = (
            ChronologicalSplitter(_CONFIG.split).split(events, labels).assignments
        )

        staging = tmp_path / "staged"
        staging.mkdir(parents=True, exist_ok=True)
        write_feature_snapshots(frame, staging / SNAPSHOTS_FILE)
        write_feature_labels(labels, staging / LABELS_FILE)
        write_feature_splits(assignments, staging / SPLITS_FILE)

        return build_feature_manifest(
            staging_dir=staging,
            catalog=frame.catalog,
            config=_CONFIG,
            feature_fingerprint=compute_features_fingerprint(frame),
            source_dataset_fingerprint="a" * 64,
            baseline_fingerprint=None,
            row_count=len(frame.rows),
            label_count=len(labels),
            earliest_event_time=events[0].event_time,
            latest_event_time=events[-1].event_time,
            validation_status="valid",
        )

    def test_records_the_manifest_version(self, tmp_path: Path) -> None:
        assert self._build(tmp_path).manifest_version == FEATURE_MANIFEST_VERSION

    def test_names_the_feature_table_as_primary(self, tmp_path: Path) -> None:
        assert self._build(tmp_path).primary_artifact == SNAPSHOTS_FILE

    def test_records_checksums_for_every_artifact(self, tmp_path: Path) -> None:
        manifest = self._build(tmp_path)
        assert len(manifest.artifacts) == len(PUBLISHED_FILES)
        for artifact in manifest.artifacts:
            assert len(artifact.sha256) == 64

    def test_artifact_paths_are_relative(self, tmp_path: Path) -> None:
        for artifact in self._build(tmp_path).artifacts:
            assert not Path(artifact.relative_path).is_absolute()
            assert ".." not in artifact.relative_path

    def test_records_the_fingerprints(self, tmp_path: Path) -> None:
        manifest = self._build(tmp_path)
        assert len(manifest.feature_catalog_fingerprint) == 64
        assert len(manifest.config_fingerprint or "") == 64
        assert len(manifest.split_config_fingerprint) == 64

    def test_records_dependency_versions(self, tmp_path: Path) -> None:
        payload = self._build(tmp_path).to_dict()
        assert payload["reproducibility"]["python_version"]
        assert payload["reproducibility"]["pyarrow_version"]

    def test_dataset_id_is_derived_from_content(self, tmp_path: Path) -> None:
        first = self._build(tmp_path)
        second = self._build(tmp_path / "again")
        assert first.dataset_id == second.dataset_id

    def test_serialises_to_json(self, tmp_path: Path) -> None:
        payload = json.loads(json.dumps(self._build(tmp_path).to_dict(), default=str))
        assert payload["source_type"] == "features"

    def test_contains_no_absolute_paths(self, tmp_path: Path) -> None:
        text = json.dumps(self._build(tmp_path).to_dict(), default=str)
        assert str(tmp_path) not in text
        assert "/home/" not in text

    def test_contains_no_identifiers(self, tmp_path: Path) -> None:
        import re

        text = json.dumps(self._build(tmp_path).to_dict(), default=str)
        assert not re.search(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b", text)


class TestVerifyFeatureDataset:
    def _published(self, tmp_path: Path) -> Path:
        events = _events(40)
        frame = _frame(events)
        labels = make_labels(events)
        assignments = (
            ChronologicalSplitter(_CONFIG.split).split(events, labels).assignments
        )

        staging = tmp_path / "staged"
        staging.mkdir()
        write_feature_snapshots(frame, staging / SNAPSHOTS_FILE)
        write_feature_labels(labels, staging / LABELS_FILE)
        write_feature_splits(assignments, staging / SPLITS_FILE)

        manifest = build_feature_manifest(
            staging_dir=staging,
            catalog=frame.catalog,
            config=_CONFIG,
            feature_fingerprint=compute_features_fingerprint(frame),
            source_dataset_fingerprint="b" * 64,
            baseline_fingerprint=None,
            row_count=len(frame.rows),
            label_count=len(labels),
            earliest_event_time=events[0].event_time,
            latest_event_time=events[-1].event_time,
            validation_status="valid",
        )

        out = tmp_path / "out"
        publisher = FeaturePublisher(out)
        ordered_labels = labels_for_events(
            [row[ANCHOR_EVENT_ID] for row in frame.rows], labels
        )
        publisher.publish(frame, ordered_labels, assignments, manifest.to_dict())
        return out

    def test_a_freshly_published_dataset_verifies(self, tmp_path: Path) -> None:
        result = verify_feature_dataset(self._published(tmp_path))
        assert result.passed, [c.message for c in result.checks if not c.passed]

    def test_row_count_is_checked_against_the_feature_table(
        self, tmp_path: Path
    ) -> None:
        result = verify_feature_dataset(self._published(tmp_path))
        check = next(c for c in result.checks if c.name == "ROW_COUNT_MATCHES")
        assert check.passed
        assert "skipped" not in check.message.lower()

    def test_a_modified_artifact_is_detected(self, tmp_path: Path) -> None:
        directory = self._published(tmp_path)
        (directory / SPLITS_FILE).write_bytes(b"corrupted")
        result = verify_feature_dataset(directory)
        assert not result.passed
        assert not next(c.passed for c in result.checks if c.name == "CHECKSUMS_MATCH")

    def test_a_missing_artifact_is_detected(self, tmp_path: Path) -> None:
        directory = self._published(tmp_path)
        (directory / LABELS_FILE).unlink()
        result = verify_feature_dataset(directory)
        assert not result.passed

    def test_a_directory_without_a_manifest_fails(self, tmp_path: Path) -> None:
        result = verify_feature_dataset(tmp_path)
        assert not result.passed
