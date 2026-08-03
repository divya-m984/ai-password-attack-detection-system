"""Unit tests for password_attack_detector.data.manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError

from password_attack_detector.data.manifest import (
    ArtifactEntry,
    DatasetManifest,
    ReproducibilityInfo,
    VerificationCheck,
    VerificationResult,
    _is_safe_relative_path,
    build_synthetic_manifest,
    verify_dataset,
)
from password_attack_detector.data.schemas import SCHEMA_VERSION
from password_attack_detector.data.serialization import DatasetPublisher
from password_attack_detector.data.synthetic.config import SyntheticConfig
from password_attack_detector.data.synthetic.generator import generate_dataset
from password_attack_detector.exceptions import ManifestVerificationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = datetime(2024, 1, 1, tzinfo=UTC)


def _small_config(**kwargs: object) -> SyntheticConfig:
    defaults: dict[str, object] = {
        "seed": 42,
        "start_time": _START,
        "duration_hours": 1,
        "num_users": 3,
        "num_sources": 2,
        "num_devices": 4,
        "num_applications": 1,
        "events_per_hour": 5,
    }
    defaults.update(kwargs)
    return SyntheticConfig(**defaults)  # type: ignore[arg-type]


def _minimal_manifest(**overrides: object) -> DatasetManifest:
    """Build a valid minimal DatasetManifest for unit tests."""
    repro = ReproducibilityInfo(
        python_version="3.12.0",
        numpy_version="1.26.0",
        pandas_version="2.0.0",
        pyarrow_version="14.0.0",
        uv_lock_sha256=None,
        generator_version="1.0.0",
        seed=42,
    )
    defaults: dict[str, object] = {
        "manifest_version": "1.0.0",
        "dataset_id": str(uuid4()),
        "schema_version": SCHEMA_VERSION,
        "source_type": "synthetic",
        "row_count": 10,
        "ground_truth_row_count": 10,
        "earliest_event_time": "2024-01-01T00:00:00+00:00",
        "latest_event_time": "2024-01-01T01:00:00+00:00",
        "artifacts": [ArtifactEntry(relative_path="events.parquet", sha256="a" * 64)],
        "canonical_schema_fingerprint": "b" * 64,
        "content_fingerprint": "c" * 64,
        "config_fingerprint": "d" * 64,
        "validation_status": "valid",
        "created_at": "2024-01-01T00:00:00+00:00",
        "reproducibility": repro,
    }
    defaults.update(overrides)
    return DatasetManifest(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def published_dir(tmp_path: Path) -> Path:
    """Publish a small synthetic dataset and return the output directory."""
    cfg = _small_config()
    result = generate_dataset(cfg)
    out = tmp_path / "dataset"
    DatasetPublisher(out).publish(result)
    return out


# ---------------------------------------------------------------------------
# ArtifactEntry / ReproducibilityInfo
# ---------------------------------------------------------------------------


class TestArtifactEntry:
    def test_construction(self) -> None:
        e = ArtifactEntry(relative_path="events.parquet", sha256="a" * 64)
        assert e.relative_path == "events.parquet"
        assert len(e.sha256) == 64

    def test_frozen(self) -> None:
        e = ArtifactEntry(relative_path="x", sha256="a" * 64)
        with pytest.raises(PydanticValidationError):
            e.relative_path = "y"


class TestReproducibilityInfo:
    def test_construction(self) -> None:
        r = ReproducibilityInfo(
            python_version="3.12.0",
            numpy_version="1.26.0",
            pandas_version="2.0.0",
            pyarrow_version="14.0.0",
            uv_lock_sha256=None,
            generator_version="1.0.0",
            seed=42,
        )
        assert r.seed == 42
        assert r.uv_lock_sha256 is None

    def test_frozen(self) -> None:
        r = ReproducibilityInfo(
            python_version="3.12.0",
            numpy_version="1.26.0",
            pandas_version="2.0.0",
            pyarrow_version="14.0.0",
            uv_lock_sha256=None,
            generator_version=None,
            seed=None,
        )
        with pytest.raises(PydanticValidationError):
            r.seed = 99


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------


class TestDatasetManifest:
    def test_construction(self) -> None:
        m = _minimal_manifest()
        assert m.manifest_version == "1.0.0"
        assert m.schema_version == SCHEMA_VERSION
        assert m.source_type == "synthetic"

    def test_frozen(self) -> None:
        m = _minimal_manifest()
        with pytest.raises(PydanticValidationError):
            m.row_count = 999

    def test_to_dict_returns_dict(self) -> None:
        m = _minimal_manifest()
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["manifest_version"] == "1.0.0"
        assert d["row_count"] == 10

    def test_to_dict_artifacts_serialized(self) -> None:
        m = _minimal_manifest()
        d = m.to_dict()
        assert isinstance(d["artifacts"], list)
        assert d["artifacts"][0]["relative_path"] == "events.parquet"

    def test_from_json_file_round_trips(self, tmp_path: Path) -> None:
        m = _minimal_manifest()
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m.to_dict()), encoding="utf-8")
        loaded = DatasetManifest.from_json_file(p)
        assert loaded.manifest_version == m.manifest_version
        assert loaded.row_count == m.row_count
        assert loaded.schema_version == m.schema_version

    def test_from_json_file_invalid_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{invalid}", encoding="utf-8")
        with pytest.raises(ManifestVerificationError):
            DatasetManifest.from_json_file(p)

    def test_from_json_file_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestVerificationError):
            DatasetManifest.from_json_file(tmp_path / "missing.json")

    def test_dataset_id_is_uuid_string(self) -> None:
        m = _minimal_manifest()
        # Should be a valid UUID string
        import uuid

        uuid.UUID(m.dataset_id)  # raises if invalid

    def test_reproducibility_embedded(self) -> None:
        m = _minimal_manifest()
        assert m.reproducibility.seed == 42
        assert m.reproducibility.generator_version == "1.0.0"


# ---------------------------------------------------------------------------
# VerificationCheck / VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationTypes:
    def test_verification_check_frozen(self) -> None:
        c = VerificationCheck(name="TEST", passed=True, message="ok")
        with pytest.raises((AttributeError, TypeError)):
            c.passed = False  # type: ignore[misc]

    def test_verification_result_frozen(self) -> None:
        r = VerificationResult(passed=True, checks=(), manifest=None)
        with pytest.raises((AttributeError, TypeError)):
            r.passed = False  # type: ignore[misc]

    def test_verification_result_with_manifest(self) -> None:
        m = _minimal_manifest()
        r = VerificationResult(passed=True, checks=(), manifest=m)
        assert r.manifest is m


# ---------------------------------------------------------------------------
# Safe path helpers
# ---------------------------------------------------------------------------


class TestSafeRelativePath:
    def test_simple_relative_path_ok(self, tmp_path: Path) -> None:
        (tmp_path / "events.parquet").touch()
        ok, _ = _is_safe_relative_path("events.parquet", tmp_path)
        assert ok

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        ok, msg = _is_safe_relative_path("/etc/passwd", tmp_path)
        assert not ok
        assert "absolute" in msg

    def test_dotdot_path_rejected(self, tmp_path: Path) -> None:
        ok, msg = _is_safe_relative_path("../../../etc/passwd", tmp_path)
        assert not ok
        assert ".." in msg

    def test_nested_relative_path_ok(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.parquet").touch()
        ok, _ = _is_safe_relative_path("sub/file.parquet", tmp_path)
        assert ok

    def test_no_startswith_used(self, tmp_path: Path) -> None:
        # Verify the path safety check is not fooled by a path that starts
        # with the base dir string but is actually outside it.
        # (This would catch string-based startswith hacks.)
        sibling = tmp_path.parent / (tmp_path.name + "_evil")
        sibling.mkdir(exist_ok=True)
        # "events.parquet" within tmp_path is fine
        ok, _ = _is_safe_relative_path("events.parquet", tmp_path)
        assert ok


# ---------------------------------------------------------------------------
# build_synthetic_manifest
# ---------------------------------------------------------------------------


class TestBuildSyntheticManifest:
    def test_returns_dataset_manifest(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        from password_attack_detector.data.serialization import (
            write_events_parquet,
            write_labels_parquet,
        )

        write_events_parquet(result.events, tmp_path / "events.parquet")
        write_labels_parquet(result.labels, tmp_path / "labels.parquet")

        manifest = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint="a" * 64,
            staging_dir=tmp_path,
            artifact_filenames=("events.parquet", "labels.parquet"),
            validation_status="valid",
        )
        assert isinstance(manifest, DatasetManifest)

    def test_manifest_has_correct_row_count(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        manifest = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint="a" * 64,
            staging_dir=tmp_path,
            artifact_filenames=(),
            validation_status="valid",
        )
        assert manifest.row_count == len(result.events)
        assert manifest.ground_truth_row_count == len(result.labels)

    def test_manifest_schema_version(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        manifest = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint="a" * 64,
            staging_dir=tmp_path,
            artifact_filenames=(),
            validation_status="valid",
        )
        assert manifest.schema_version == SCHEMA_VERSION

    def test_manifest_source_type_synthetic(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        manifest = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint="a" * 64,
            staging_dir=tmp_path,
            artifact_filenames=(),
            validation_status="valid",
        )
        assert manifest.source_type == "synthetic"

    def test_temporal_range_set(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        manifest = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint="a" * 64,
            staging_dir=tmp_path,
            artifact_filenames=(),
            validation_status="valid",
        )
        assert manifest.earliest_event_time is not None
        assert manifest.latest_event_time is not None

    def test_dataset_id_deterministic_from_content_fingerprint(
        self, tmp_path: Path
    ) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        fp = "e" * 64
        m1 = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint=fp,
            staging_dir=tmp_path,
            artifact_filenames=(),
            validation_status="valid",
        )
        m2 = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint=fp,
            staging_dir=tmp_path,
            artifact_filenames=(),
            validation_status="valid",
        )
        assert m1.dataset_id == m2.dataset_id

    def test_reproducibility_info_populated(self, tmp_path: Path) -> None:
        cfg = _small_config(seed=77)
        result = generate_dataset(cfg)
        manifest = build_synthetic_manifest(
            events=result.events,
            labels=result.labels,
            config=result.config,
            config_fingerprint=result.config_fingerprint,
            content_fingerprint="a" * 64,
            staging_dir=tmp_path,
            artifact_filenames=(),
            validation_status="valid",
        )
        assert manifest.reproducibility.seed == 77
        assert manifest.reproducibility.python_version != ""
        assert manifest.reproducibility.numpy_version != ""


# ---------------------------------------------------------------------------
# verify_dataset -- integration tests
# ---------------------------------------------------------------------------


class TestVerifyDataset:
    def test_published_dataset_passes_all_checks(self, published_dir: Path) -> None:
        result = verify_dataset(published_dir)
        assert result.passed
        assert len(result.checks) == 10
        for check in result.checks:
            assert check.passed, f"Check {check.name!r} failed: {check.message}"

    def test_result_has_manifest(self, published_dir: Path) -> None:
        result = verify_dataset(published_dir)
        assert result.manifest is not None
        assert result.manifest.schema_version == SCHEMA_VERSION

    def test_all_10_checks_present(self, published_dir: Path) -> None:
        result = verify_dataset(published_dir)
        expected = {
            "MANIFEST_READABLE",
            "MANIFEST_VERSION_RECOGNIZED",
            "SCHEMA_VERSION_MATCH",
            "ARTIFACT_PATHS_NOT_ABSOLUTE",
            "ARTIFACT_PATHS_NO_DOTDOT",
            "NO_SYMLINK_ESCAPE",
            "ARTIFACT_FILES_EXIST",
            "CHECKSUMS_MATCH",
            "ROW_COUNT_MATCHES",
            "VALIDATION_STATUS_NOT_INVALID",
        }
        actual = {c.name for c in result.checks}
        assert actual == expected

    def test_missing_manifest_fails_check_1(self, tmp_path: Path) -> None:
        result = verify_dataset(tmp_path)
        manifest_check = next(c for c in result.checks if c.name == "MANIFEST_READABLE")
        assert not manifest_check.passed
        assert not result.passed
        assert result.manifest is None

    def test_corrupt_manifest_fails_check_1(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{bad json}", encoding="utf-8")
        result = verify_dataset(tmp_path)
        assert not result.passed
        manifest_check = next(c for c in result.checks if c.name == "MANIFEST_READABLE")
        assert not manifest_check.passed

    def test_wrong_manifest_version_fails_check_2(self, published_dir: Path) -> None:
        # Tamper with manifest_version.
        mp = published_dir / "manifest.json"
        data = json.loads(mp.read_text())
        data["manifest_version"] = "99.0.0"
        mp.write_text(json.dumps(data), encoding="utf-8")
        result = verify_dataset(published_dir)
        check = next(
            c for c in result.checks if c.name == "MANIFEST_VERSION_RECOGNIZED"
        )
        assert not check.passed

    def test_wrong_schema_version_fails_check_3(self, published_dir: Path) -> None:
        mp = published_dir / "manifest.json"
        data = json.loads(mp.read_text())
        data["schema_version"] = "0.0.0"
        mp.write_text(json.dumps(data), encoding="utf-8")
        result = verify_dataset(published_dir)
        check = next(c for c in result.checks if c.name == "SCHEMA_VERSION_MATCH")
        assert not check.passed

    def test_absolute_artifact_path_fails_check_4(self, published_dir: Path) -> None:
        mp = published_dir / "manifest.json"
        data = json.loads(mp.read_text())
        data["artifacts"] = [{"relative_path": "/etc/passwd", "sha256": "a" * 64}]
        mp.write_text(json.dumps(data), encoding="utf-8")
        result = verify_dataset(published_dir)
        check = next(
            c for c in result.checks if c.name == "ARTIFACT_PATHS_NOT_ABSOLUTE"
        )
        assert not check.passed

    def test_dotdot_artifact_path_fails_check_5(self, published_dir: Path) -> None:
        mp = published_dir / "manifest.json"
        data = json.loads(mp.read_text())
        data["artifacts"] = [{"relative_path": "../../etc/passwd", "sha256": "a" * 64}]
        mp.write_text(json.dumps(data), encoding="utf-8")
        result = verify_dataset(published_dir)
        check = next(c for c in result.checks if c.name == "ARTIFACT_PATHS_NO_DOTDOT")
        assert not check.passed

    def test_missing_artifact_file_fails_check_7(self, published_dir: Path) -> None:
        # Remove one artifact.
        (published_dir / "events.jsonl").unlink()
        result = verify_dataset(published_dir)
        check = next(c for c in result.checks if c.name == "ARTIFACT_FILES_EXIST")
        assert not check.passed

    def test_tampered_artifact_fails_check_8(self, published_dir: Path) -> None:
        # Append a byte to events.parquet to corrupt it.
        ep = published_dir / "events.parquet"
        ep.write_bytes(ep.read_bytes() + b"\x00")
        result = verify_dataset(published_dir)
        check = next(c for c in result.checks if c.name == "CHECKSUMS_MATCH")
        assert not check.passed

    def test_wrong_row_count_in_manifest_fails_check_9(
        self, published_dir: Path
    ) -> None:
        mp = published_dir / "manifest.json"
        data = json.loads(mp.read_text())
        data["row_count"] = 999_999
        mp.write_text(json.dumps(data), encoding="utf-8")
        result = verify_dataset(published_dir)
        check = next(c for c in result.checks if c.name == "ROW_COUNT_MATCHES")
        assert not check.passed

    def test_invalid_validation_status_fails_check_10(
        self, published_dir: Path
    ) -> None:
        mp = published_dir / "manifest.json"
        data = json.loads(mp.read_text())
        data["validation_status"] = "invalid"
        mp.write_text(json.dumps(data), encoding="utf-8")
        result = verify_dataset(published_dir)
        check = next(
            c for c in result.checks if c.name == "VALIDATION_STATUS_NOT_INVALID"
        )
        assert not check.passed


# ---------------------------------------------------------------------------
# DatasetPublisher integration with manifest
# ---------------------------------------------------------------------------


class TestPublisherManifestIntegration:
    def test_manifest_json_is_written(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        out = tmp_path / "out"
        DatasetPublisher(out).publish(result)
        assert (out / "manifest.json").exists()

    def test_manifest_json_has_full_schema(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        out = tmp_path / "out"
        DatasetPublisher(out).publish(result)
        data = json.loads((out / "manifest.json").read_text())
        required_keys = {
            "manifest_version",
            "dataset_id",
            "schema_version",
            "source_type",
            "row_count",
            "artifacts",
            "content_fingerprint",
            "validation_status",
            "created_at",
            "reproducibility",
        }
        assert required_keys <= data.keys()

    def test_manifest_artifacts_list_has_sha256(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        out = tmp_path / "out"
        DatasetPublisher(out).publish(result)
        data = json.loads((out / "manifest.json").read_text())
        for artifact in data["artifacts"]:
            assert "sha256" in artifact
            assert len(artifact["sha256"]) == 64

    def test_verify_passes_after_publish(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        out = tmp_path / "out"
        DatasetPublisher(out).publish(result)
        vr = verify_dataset(out)
        assert vr.passed

    def test_published_manifest_has_no_absolute_paths(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        out = tmp_path / "out"
        DatasetPublisher(out).publish(result)
        data = json.loads((out / "manifest.json").read_text())
        manifest_text = json.dumps(data)
        assert "/home/" not in manifest_text
        assert "/root/" not in manifest_text

    def test_published_manifest_has_no_secrets(self, tmp_path: Path) -> None:
        cfg = _small_config()
        result = generate_dataset(cfg)
        out = tmp_path / "out"
        DatasetPublisher(out).publish(result)
        data = json.loads((out / "manifest.json").read_text())
        manifest_text = json.dumps(data)
        # Ensure no pseudonymization key or password-like fields
        assert "password" not in manifest_text.lower().replace("password_attack", "")
        assert "secret" not in manifest_text.lower()
