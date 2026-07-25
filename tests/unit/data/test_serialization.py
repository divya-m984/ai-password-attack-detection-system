"""Unit tests for password_attack_detector.data.serialization."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pyarrow.parquet as pq
import pytest

from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    FailureReason,
    ScenarioType,
)
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.data.serialization import (
    CANONICAL_EVENT_COLUMNS,
    CANONICAL_GT_COLUMNS,
    DatasetPublisher,
    PublishedDataset,
    compute_events_fingerprint,
    write_events_jsonl,
    write_events_parquet,
    write_labels_parquet,
)
from password_attack_detector.data.synthetic.config import (
    BotActivityParams,
    BruteForceParams,
    CampaignParameters,
    CredentialStuffingParams,
    DistributedBruteForceParams,
    EnabledScenarios,
    ImpossibleTravelParams,
    NovelAnomalyParams,
    PasswordSprayingParams,
    SyntheticConfig,
)
from password_attack_detector.data.synthetic.generator import generate_dataset
from password_attack_detector.exceptions import DataValidationError

_START = datetime(2024, 1, 1, tzinfo=UTC)
_U = "u:" + "a" * 32
_S = "s:" + "b" * 32
_D = "d:" + "c" * 32
_SESS = "sess:" + "d" * 32


def _make_event(
    *,
    event_id: UUID | None = None,
    failure_reason: FailureReason | None = None,
    outcome: AuthOutcome = AuthOutcome.SUCCESS,
) -> AuthEvent:
    if event_id is None:
        import uuid as _uuid

        event_id = _uuid.uuid4()
    return AuthEvent(
        event_id=event_id,
        event_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        user_id=_U,
        source_id=_S,
        device_id=_D,
        session_id=_SESS,
        application_id="app-00",
        authentication_method=AuthMethod.PASSWORD,
        authentication_outcome=outcome,
        failure_reason=failure_reason,
    )


def _make_label(event: AuthEvent) -> GroundTruthLabel:
    return GroundTruthLabel(
        event_id=event.event_id,
        campaign_id="test-camp",
        scenario=ScenarioType.NORMAL,
        malicious=False,
        supervised_training_eligible=True,
        generator_version="1.0.0",
    )


def _make_tiny_result() -> Any:
    cp = CampaignParameters(
        brute_force=BruteForceParams(attempts_per_campaign=2, num_campaigns=1),
        password_spraying=PasswordSprayingParams(
            passwords_per_round=2, num_campaigns=1
        ),
        credential_stuffing=CredentialStuffingParams(
            credentials_per_batch=2, num_campaigns=1
        ),
        distributed_brute_force=DistributedBruteForceParams(
            attempts_per_source=2, num_sources=2, num_campaigns=1
        ),
        impossible_travel=ImpossibleTravelParams(num_campaigns=1),
        bot_activity=BotActivityParams(events_per_campaign=2, num_campaigns=1),
        novel_anomaly_holdout=NovelAnomalyParams(num_campaigns=1),
    )
    es = EnabledScenarios(
        normal=True,
        brute_force=False,
        password_spraying=False,
        credential_stuffing=False,
        distributed_brute_force=False,
        account_takeover_indicator=False,
        impossible_travel=False,
        bot_activity=False,
        novel_anomaly_holdout=False,
    )
    cfg = SyntheticConfig(
        seed=42,
        start_time=_START,
        duration_hours=1,
        num_users=3,
        num_sources=2,
        num_devices=4,
        num_applications=1,
        events_per_hour=3,
        campaign_parameters=cp,
        enabled_scenarios=es,
    )
    return generate_dataset(cfg)


class TestCanonicalColumns:
    def test_event_columns_include_required_fields(self) -> None:
        required = {"event_id", "event_time", "user_id", "authentication_outcome"}
        assert required <= set(CANONICAL_EVENT_COLUMNS)

    def test_gt_columns_include_required_fields(self) -> None:
        required = {"event_id", "campaign_id", "scenario", "malicious"}
        assert required <= set(CANONICAL_GT_COLUMNS)

    def test_event_columns_no_gt_fields(self) -> None:
        from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS

        for col in PROHIBITED_GT_COLUMNS:
            assert col not in CANONICAL_EVENT_COLUMNS

    def test_event_columns_match_autoevent_fields(self) -> None:
        event_fields = set(AuthEvent.model_fields.keys())
        for col in CANONICAL_EVENT_COLUMNS:
            assert col in event_fields, f"{col!r} not in AuthEvent fields"

    def test_gt_columns_match_groundtruthlabel_fields(self) -> None:
        gt_fields = set(GroundTruthLabel.model_fields.keys())
        for col in CANONICAL_GT_COLUMNS:
            assert col in gt_fields, f"{col!r} not in GroundTruthLabel fields"


class TestComputeEventsFingerprint:
    def test_returns_64_hex_chars(self) -> None:
        e = _make_event()
        fp = compute_events_fingerprint([e])
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_deterministic(self) -> None:
        e = _make_event()
        assert compute_events_fingerprint([e]) == compute_events_fingerprint([e])

    def test_order_independent(self) -> None:
        import uuid

        e1 = _make_event(event_id=uuid.UUID("00000000-0000-0000-0000-000000000001"))
        e2 = _make_event(event_id=uuid.UUID("00000000-0000-0000-0000-000000000002"))
        fp1 = compute_events_fingerprint([e1, e2])
        fp2 = compute_events_fingerprint([e2, e1])
        assert fp1 == fp2

    def test_different_events_different_fingerprint(self) -> None:
        import uuid

        e1 = _make_event(event_id=uuid.UUID("00000000-0000-0000-0000-000000000001"))
        e2 = _make_event(event_id=uuid.UUID("00000000-0000-0000-0000-000000000002"))
        assert compute_events_fingerprint([e1]) != compute_events_fingerprint([e2])

    def test_adding_event_changes_fingerprint(self) -> None:
        import uuid

        e1 = _make_event(event_id=uuid.UUID("00000000-0000-0000-0000-000000000001"))
        e2 = _make_event(event_id=uuid.UUID("00000000-0000-0000-0000-000000000002"))
        fp1 = compute_events_fingerprint([e1])
        fp2 = compute_events_fingerprint([e1, e2])
        assert fp1 != fp2


class TestWriteEventsParquet:
    def test_creates_file(self, tmp_path: Path) -> None:
        e = _make_event()
        out = tmp_path / "events.parquet"
        write_events_parquet([e], out)
        assert out.exists()

    def test_readable_with_pyarrow(self, tmp_path: Path) -> None:
        e = _make_event()
        out = tmp_path / "events.parquet"
        write_events_parquet([e], out)
        table = pq.read_table(out)
        assert table.num_rows == 1

    def test_column_order_matches_canonical(self, tmp_path: Path) -> None:
        e = _make_event()
        out = tmp_path / "events.parquet"
        write_events_parquet([e], out)
        table = pq.read_table(out)
        assert list(table.schema.names) == list(CANONICAL_EVENT_COLUMNS)

    def test_event_id_preserved(self, tmp_path: Path) -> None:
        import uuid

        eid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        e = _make_event(event_id=eid)
        out = tmp_path / "events.parquet"
        write_events_parquet([e], out)
        table = pq.read_table(out)
        assert str(table.column("event_id")[0].as_py()) == str(eid)

    def test_multiple_events(self, tmp_path: Path) -> None:
        events = [_make_event() for _ in range(5)]
        out = tmp_path / "events.parquet"
        write_events_parquet(events, out)
        table = pq.read_table(out)
        assert table.num_rows == 5

    def test_nullable_fields_stored_as_null(self, tmp_path: Path) -> None:
        e = _make_event()  # country_code is None
        out = tmp_path / "events.parquet"
        write_events_parquet([e], out)
        table = pq.read_table(out)
        assert table.column("country_code")[0].as_py() is None

    def test_failure_reason_stored(self, tmp_path: Path) -> None:
        e = _make_event(
            outcome=AuthOutcome.FAILURE,
            failure_reason=FailureReason.INVALID_CREDENTIALS,
        )
        out = tmp_path / "events.parquet"
        write_events_parquet([e], out)
        table = pq.read_table(out)
        assert table.column("failure_reason")[0].as_py() == "invalid_credentials"


class TestWriteLabelsParquet:
    def test_creates_file(self, tmp_path: Path) -> None:
        e = _make_event()
        label = _make_label(e)
        out = tmp_path / "labels.parquet"
        write_labels_parquet([label], out)
        assert out.exists()

    def test_readable_with_pyarrow(self, tmp_path: Path) -> None:
        e = _make_event()
        label = _make_label(e)
        out = tmp_path / "labels.parquet"
        write_labels_parquet([label], out)
        table = pq.read_table(out)
        assert table.num_rows == 1

    def test_column_order_matches_canonical(self, tmp_path: Path) -> None:
        e = _make_event()
        label = _make_label(e)
        out = tmp_path / "labels.parquet"
        write_labels_parquet([label], out)
        table = pq.read_table(out)
        assert list(table.schema.names) == list(CANONICAL_GT_COLUMNS)

    def test_event_id_matches(self, tmp_path: Path) -> None:
        import uuid

        eid = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        e = _make_event(event_id=eid)
        label = _make_label(e)
        out = tmp_path / "labels.parquet"
        write_labels_parquet([label], out)
        table = pq.read_table(out)
        assert str(table.column("event_id")[0].as_py()) == str(eid)

    def test_malicious_field_stored(self, tmp_path: Path) -> None:
        e = _make_event()
        label = _make_label(e)
        out = tmp_path / "labels.parquet"
        write_labels_parquet([label], out)
        table = pq.read_table(out)
        assert table.column("malicious")[0].as_py() is False


class TestWriteEventsJsonl:
    def test_creates_file(self, tmp_path: Path) -> None:
        e = _make_event()
        out = tmp_path / "events.jsonl"
        write_events_jsonl([e], out)
        assert out.exists()

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        events = [_make_event() for _ in range(3)]
        out = tmp_path / "events.jsonl"
        write_events_jsonl(events, out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_event_id_present_in_json(self, tmp_path: Path) -> None:
        import uuid

        eid = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        e = _make_event(event_id=eid)
        out = tmp_path / "events.jsonl"
        write_events_jsonl([e], out)
        line = json.loads(out.read_text().strip())
        assert "event_id" in line

    def test_json_columns_match_canonical_order(self, tmp_path: Path) -> None:
        e = _make_event()
        out = tmp_path / "events.jsonl"
        write_events_jsonl([e], out)
        line = json.loads(out.read_text().strip())
        assert list(line.keys()) == list(CANONICAL_EVENT_COLUMNS)

    def test_empty_events_produces_empty_file(self, tmp_path: Path) -> None:
        out = tmp_path / "events.jsonl"
        write_events_jsonl([], out)
        assert out.read_text() == ""


class TestDatasetPublisherHappyPath:
    def test_publish_creates_all_files(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        publisher = DatasetPublisher(tmp_path / "output")
        published = publisher.publish(result)
        assert published.events_path.exists()
        assert published.labels_path.exists()
        assert published.jsonl_path.exists()
        assert published.manifest_path.exists()

    def test_publish_returns_published_dataset(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        publisher = DatasetPublisher(tmp_path / "output")
        published = publisher.publish(result)
        assert isinstance(published, PublishedDataset)

    def test_published_event_count_matches(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        publisher = DatasetPublisher(tmp_path / "output")
        published = publisher.publish(result)
        assert published.num_events == len(result.events)

    def test_published_label_count_matches(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        publisher = DatasetPublisher(tmp_path / "output")
        published = publisher.publish(result)
        assert published.num_labels == len(result.labels)

    def test_content_fingerprint_matches(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        publisher = DatasetPublisher(tmp_path / "output")
        published = publisher.publish(result)
        assert published.content_fingerprint == compute_events_fingerprint(
            result.events
        )

    def test_manifest_is_valid_json(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        publisher = DatasetPublisher(tmp_path / "output")
        published = publisher.publish(result)
        manifest = json.loads(published.manifest_path.read_text())
        assert "content_fingerprint" in manifest
        assert "num_events" in manifest

    def test_staging_cleaned_after_success(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        publisher = DatasetPublisher(tmp_path / "output")
        publisher.publish(result)
        # No staging directories should remain
        staging_dirs = list(tmp_path.glob("_pad_stage_*"))
        assert staging_dirs == []

    def test_overwrite_true_replaces_files(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        out = tmp_path / "output"
        DatasetPublisher(out).publish(result)
        result2 = _make_tiny_result()
        DatasetPublisher(out, overwrite=True).publish(result2)
        assert (out / "events.parquet").exists()
        assert (out / "manifest.json").exists()

    def test_backup_cleaned_after_overwrite_success(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        out = tmp_path / "output"
        DatasetPublisher(out).publish(result)
        DatasetPublisher(out, overwrite=True).publish(_make_tiny_result())
        backup_dirs = list(tmp_path.glob("_pad_backup_*"))
        assert backup_dirs == []


class TestDatasetPublisherErrorCases:
    def test_empty_events_raises_before_any_output(self, tmp_path: Path) -> None:
        result = _make_tiny_result()

        # Create a result with empty events using object.__setattr__ on frozen dataclass
        import dataclasses

        empty_result = dataclasses.replace(result, events=(), labels=())

        out = tmp_path / "output"
        publisher = DatasetPublisher(out)

        with pytest.raises(DataValidationError, match="empty"):
            publisher.publish(empty_result)

        # No files should have been created
        assert not out.exists() or not any(out.iterdir())

    def test_overwrite_false_raises_when_exists(self, tmp_path: Path) -> None:
        result = _make_tiny_result()
        out = tmp_path / "output"
        DatasetPublisher(out).publish(result)

        with pytest.raises(DataValidationError, match="already exists"):
            DatasetPublisher(out, overwrite=False).publish(result)

    def test_failed_first_time_publish_leaves_no_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _make_tiny_result()
        out = tmp_path / "output"

        call_count = [0]
        original_move = shutil.move

        def failing_move(src: str, dst: str) -> None:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise OSError("Simulated move failure")
            original_move(src, dst)

        monkeypatch.setattr(shutil, "move", failing_move)

        with pytest.raises(OSError):
            DatasetPublisher(out).publish(result)

        manifest = out / "manifest.json"
        assert not manifest.exists(), (
            "Manifest must not exist after failed first-time publish"
        )

    def test_failed_first_time_publish_leaves_no_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _make_tiny_result()
        out = tmp_path / "output"

        call_count = [0]
        original_move = shutil.move

        def failing_move(src: str, dst: str) -> None:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise OSError("Simulated move failure")
            original_move(src, dst)

        monkeypatch.setattr(shutil, "move", failing_move)

        with pytest.raises(OSError):
            DatasetPublisher(out).publish(result)

        # No artifact files should persist (rollback removes partially promoted files)
        events_file = out / "events.parquet"
        assert not events_file.exists(), (
            "events.parquet must not exist after failed first-time publish"
        )

    def test_failed_overwrite_restores_original_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _make_tiny_result()
        out = tmp_path / "output"

        # First publish succeeds
        DatasetPublisher(out).publish(result)

        # Record original content
        original_events = (out / "events.parquet").read_bytes()

        call_count = [0]
        original_move = shutil.move

        def failing_move(src: str, dst: str) -> None:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise OSError("Simulated move failure during overwrite")
            original_move(src, dst)

        monkeypatch.setattr(shutil, "move", failing_move)

        with pytest.raises(OSError):
            DatasetPublisher(out, overwrite=True).publish(_make_tiny_result())

        # Original file must be restored (byte-for-byte)
        restored = (out / "events.parquet").read_bytes()
        assert restored == original_events

    def test_staging_cleaned_after_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _make_tiny_result()
        out = tmp_path / "output"

        call_count = [0]
        original_move = shutil.move

        def failing_move(src: str, dst: str) -> None:
            call_count[0] += 1
            if call_count[0] >= 1:
                raise OSError("Immediate failure")
            original_move(src, dst)

        monkeypatch.setattr(shutil, "move", failing_move)

        with pytest.raises(OSError):
            DatasetPublisher(out).publish(result)

        staging_dirs = list(tmp_path.glob("_pad_stage_*"))
        assert staging_dirs == [], "Staging directory must be cleaned after failure"
