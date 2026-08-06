"""Tests for the detection manifest and its verification.

The manifest is a superset of the Phase 2 dataset manifest, so most of the
safety checking is inherited rather than reimplemented.  This module proves
two things: that the inherited checks really do fire on a detection directory
(path traversal, symlink escape, checksums), and that the detection-specific
checks catch what the shared verifier cannot see -- substituted artifacts,
broken cross-table relationships, and a stale configuration.

It also re-runs the Phase 2 and Phase 3 manifest regressions, because
generalising a shared verifier is exactly the change that quietly breaks its
existing callers.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.detection.alerts import (
    AlertBuilder,
    build_entity_scope_table,
)
from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.engine import DetectionEngine
from password_attack_detector.detection.manifest import (
    DETECTION_MANIFEST_VERSION,
    EXPECTED_ARTIFACT_ROLES,
    build_detection_manifest,
    verify_detection_dataset,
)
from password_attack_detector.detection.scoring import RiskScorer
from password_attack_detector.detection.serialization import (
    ALERTS_FILE,
    DETECTIONS_FILE,
    MANIFEST_FILE,
    RISK_FILE,
    DetectionPublisher,
    compute_alert_fingerprint,
    compute_detection_fingerprint,
    compute_risk_fingerprint,
    write_fired_detections,
    write_risk_assessments,
)
from password_attack_detector.detection.validation import DetectionValidator
from tests.unit.detection import factories

WHEN = factories.WHEN
CONFIG = DetectionConfig()

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_PSEUDONYM_RE = re.compile(r"(u|s|d|sess):[0-9a-f]{32}")


def build_artifacts(*, scope: bool = False) -> dict[str, Any]:
    """Run the pipeline once and return everything a manifest needs."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    builders = [
        factories.brute_force_row,
        factories.spraying_row,
        factories.stuffing_row,
        factories.quiet_row,
    ]
    rows = [
        builders[index % 4](
            catalog,
            anchor_event_id=f"anchor-{index:04d}",
            anchor_event_time=WHEN + timedelta(minutes=index * 3),
        )
        for index in range(12)
    ]
    detections = list(engine.run(rows).fired_detections)
    scored = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    table = (
        build_entity_scope_table(
            [
                factories.scope_record(row["anchor_event_id"], user=f"{index:032x}")
                for index, row in enumerate(rows)
            ]
        )
        if scope
        else None
    )
    alerting = AlertBuilder(CONFIG).build(
        scored.assessments, detections=detections, entity_scope=table
    )
    return {
        "detections": detections,
        "assessments": list(scored.assessments),
        "alerts": list(alerting.alerts),
        "grouping_mode": str(alerting.stats.grouping_mode),
        "scope_present": scope,
    }


def publish(directory: Path, artifacts: dict[str, Any], **overrides: Any) -> Any:
    """Stage, build a manifest, and publish an artifact set."""
    staging = directory.parent / f"{directory.name}_staging"
    staging.mkdir(parents=True, exist_ok=True)
    from password_attack_detector.detection.serialization import write_alerts

    write_fired_detections(artifacts["detections"], staging / DETECTIONS_FILE)
    write_risk_assessments(artifacts["assessments"], staging / RISK_FILE)
    write_alerts(artifacts["alerts"], staging / ALERTS_FILE)

    validation = DetectionValidator(CONFIG).validate(
        artifacts["detections"], artifacts["assessments"], artifacts["alerts"]
    )
    times = [item.anchor_event_time for item in artifacts["assessments"]]
    kwargs: dict[str, Any] = {
        "staging_dir": staging,
        "config": CONFIG,
        "detection_fingerprint": compute_detection_fingerprint(artifacts["detections"]),
        "risk_fingerprint": compute_risk_fingerprint(artifacts["assessments"]),
        "alert_fingerprint": compute_alert_fingerprint(artifacts["alerts"]),
        "detection_row_count": len(artifacts["detections"]),
        "risk_row_count": len(artifacts["assessments"]),
        "alert_row_count": len(artifacts["alerts"]),
        "earliest_event_time": min(times),
        "latest_event_time": max(times),
        "alert_grouping_mode": artifacts["grouping_mode"],
        "entity_scope_present": artifacts["scope_present"],
        "validation_status": str(validation.status),
        "validation_result": validation.to_dict(),
        "quality_report_status": "ok",
    }
    kwargs.update(overrides)
    manifest = build_detection_manifest(**kwargs)
    DetectionPublisher(directory, overwrite=True).publish(
        artifacts["detections"],
        artifacts["assessments"],
        artifacts["alerts"],
        manifest.to_dict(),
    )
    return manifest


@pytest.fixture(scope="module")
def artifacts() -> dict[str, Any]:
    return build_artifacts()


@pytest.fixture()
def published(tmp_path: Path, artifacts: dict[str, Any]) -> Path:
    directory = tmp_path / "detection"
    publish(directory, artifacts)
    return directory


def failed_checks(result: Any) -> set[str]:
    """Return the names of every check that did not pass."""
    return {check.name for check in result.checks if not check.passed}


# ---------------------------------------------------------------------------
# Manifest contents
# ---------------------------------------------------------------------------


def test_a_published_directory_verifies(published: Path) -> None:
    result = verify_detection_dataset(published, config=CONFIG)
    assert result.passed, failed_checks(result)


def test_the_manifest_records_every_required_field(published: Path) -> None:
    raw = json.loads((published / MANIFEST_FILE).read_text())
    for field in (
        "manifest_version",
        "detection_schema_version",
        "alerting_version",
        "scoring_version",
        "required_feature_schema_version",
        "rule_catalog_fingerprint",
        "detection_config_fingerprint",
        "detection_row_count",
        "risk_assessment_row_count",
        "alert_row_count",
        "earliest_event_time",
        "latest_event_time",
        "alert_grouping_mode",
        "entity_scope_present",
        "artifacts",
        "artifact_roles",
        "detection_content_fingerprint",
        "risk_content_fingerprint",
        "alert_content_fingerprint",
        "reproducibility",
        "validation_status",
        "quality_report_status",
        "evaluation_report_present",
        "evaluation_report_status",
        "created_at",
    ):
        assert field in raw, field
    assert raw["manifest_version"] == DETECTION_MANIFEST_VERSION


def test_every_artifact_entry_is_a_relative_path_with_a_checksum(
    published: Path,
) -> None:
    raw = json.loads((published / MANIFEST_FILE).read_text())
    for entry in raw["artifacts"]:
        assert not Path(entry["relative_path"]).is_absolute()
        assert ".." not in Path(entry["relative_path"]).parts
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])


def test_the_artifact_roles_name_every_expected_table(published: Path) -> None:
    raw = json.loads((published / MANIFEST_FILE).read_text())
    assert raw["artifact_roles"] == EXPECTED_ARTIFACT_ROLES


def test_the_dataset_identifier_is_derived_from_the_content(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    """Same content, same identifier -- on any machine, on any day."""
    first = publish(tmp_path / "one", artifacts)
    second = publish(tmp_path / "two", artifacts)
    assert first.dataset_id == second.dataset_id
    assert first.content_fingerprint == second.content_fingerprint


def test_the_creation_timestamp_stays_out_of_the_content_fields(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    first = publish(tmp_path / "one", artifacts)
    second = publish(tmp_path / "two", artifacts)
    assert first.content_fields() == second.content_fields()
    assert "created_at" not in first.content_fields()
    assert "reproducibility" not in first.content_fields()


def test_the_manifest_carries_no_identifier_or_scope_value(tmp_path: Path) -> None:
    """The privacy sweep, over a run with entity scope deliberately enabled."""
    scoped = build_artifacts(scope=True)
    assert any(alert.scope_value for alert in scoped["alerts"])
    publish(tmp_path / "detection", scoped)
    rendered = (tmp_path / "detection" / MANIFEST_FILE).read_text()

    assert not _PSEUDONYM_RE.search(rendered)
    assert "anchor-" not in rendered
    assert "/home/" not in rendered
    for alert in scoped["alerts"]:
        if alert.scope_value:
            assert alert.scope_value not in rendered
        assert alert.alert_id not in rendered


def test_the_manifest_records_the_grouping_mode_and_scope_presence(
    tmp_path: Path,
) -> None:
    scoped = build_artifacts(scope=True)
    publish(tmp_path / "detection", scoped)
    raw = json.loads((tmp_path / "detection" / MANIFEST_FILE).read_text())
    assert raw["alert_grouping_mode"] == "entity_scoped"
    assert raw["entity_scope_present"] is True


def test_the_reproducibility_block_names_no_machine(published: Path) -> None:
    raw = json.loads((published / MANIFEST_FILE).read_text())
    rendered = json.dumps(raw["reproducibility"])
    import getpass
    import socket

    assert socket.gethostname() not in rendered
    assert getpass.getuser() not in rendered


# ---------------------------------------------------------------------------
# Verification: substitution and tampering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", [DETECTIONS_FILE, RISK_FILE, ALERTS_FILE])
def test_a_modified_artifact_fails_verification(published: Path, filename: str) -> None:
    (published / filename).write_bytes(b"corrupted")
    result = verify_detection_dataset(published, config=CONFIG)
    assert not result.passed
    assert "CHECKSUMS_MATCH" in failed_checks(result)


def test_a_substituted_artifact_fails_the_content_fingerprint(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    """Same shape, different content: the checksum and the digest both object."""
    directory = tmp_path / "detection"
    publish(directory, artifacts)
    write_fired_detections(artifacts["detections"][:1], directory / DETECTIONS_FILE)
    result = verify_detection_dataset(directory, config=CONFIG)
    assert not result.passed
    assert "DETECTION_CONTENT_FINGERPRINT" in failed_checks(result)


def test_a_wrong_row_count_fails_verification(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    directory = tmp_path / "detection"
    publish(directory, artifacts, risk_row_count=999)
    result = verify_detection_dataset(directory, config=CONFIG)
    assert not result.passed
    assert {"ROW_COUNT_MATCHES", "RISK_CONTENT_FINGERPRINT"} & failed_checks(result)


@pytest.mark.parametrize("filename", [DETECTIONS_FILE, RISK_FILE, ALERTS_FILE])
def test_a_missing_artifact_fails_verification(published: Path, filename: str) -> None:
    (published / filename).unlink()
    result = verify_detection_dataset(published, config=CONFIG)
    assert not result.passed
    assert "ARTIFACT_FILES_EXIST" in failed_checks(result)


def test_a_missing_manifest_fails_verification(published: Path) -> None:
    (published / MANIFEST_FILE).unlink()
    result = verify_detection_dataset(published, config=CONFIG)
    assert not result.passed
    assert "MANIFEST_READABLE" in failed_checks(result)


def test_an_unreadable_manifest_fails_verification(published: Path) -> None:
    (published / MANIFEST_FILE).write_text("{ not json")
    assert not verify_detection_dataset(published, config=CONFIG).passed


# ---------------------------------------------------------------------------
# Verification: path safety, inherited from the shared verifier
# ---------------------------------------------------------------------------


def _rewrite_artifact_path(directory: Path, path_value: str) -> None:
    """Point the first artifact entry at *path_value*."""
    manifest_path = directory / MANIFEST_FILE
    raw = json.loads(manifest_path.read_text())
    raw["artifacts"][0]["relative_path"] = path_value
    manifest_path.write_text(json.dumps(raw, indent=2))


def test_a_traversal_path_fails_verification(published: Path) -> None:
    _rewrite_artifact_path(published, f"../{DETECTIONS_FILE}")
    result = verify_detection_dataset(published, config=CONFIG)
    assert not result.passed
    assert "ARTIFACT_PATHS_NO_DOTDOT" in failed_checks(result)


def test_an_absolute_path_fails_verification(published: Path) -> None:
    _rewrite_artifact_path(published, f"/tmp/{DETECTIONS_FILE}")
    result = verify_detection_dataset(published, config=CONFIG)
    assert not result.passed
    assert "ARTIFACT_PATHS_NOT_ABSOLUTE" in failed_checks(result)


def test_a_sibling_prefix_escape_fails_verification(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    """``detection-evil`` shares a prefix with ``detection`` but is outside it."""
    directory = tmp_path / "detection"
    publish(directory, artifacts)
    sibling = tmp_path / "detection-evil"
    sibling.mkdir()
    (sibling / DETECTIONS_FILE).write_bytes(b"elsewhere")
    _rewrite_artifact_path(directory, f"../detection-evil/{DETECTIONS_FILE}")
    assert not verify_detection_dataset(directory, config=CONFIG).passed


def test_a_symlink_escape_fails_verification(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    directory = tmp_path / "detection"
    publish(directory, artifacts)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"elsewhere")
    link = directory / "linked.parquet"
    link.symlink_to(outside)
    _rewrite_artifact_path(directory, "linked.parquet")
    result = verify_detection_dataset(directory, config=CONFIG)
    assert not result.passed
    assert "NO_SYMLINK_ESCAPE" in failed_checks(result)


def test_an_unexpected_artifact_role_fails_verification(published: Path) -> None:
    manifest_path = published / MANIFEST_FILE
    raw = json.loads(manifest_path.read_text())
    raw["artifact_roles"][DETECTIONS_FILE] = "something_else"
    manifest_path.write_text(json.dumps(raw, indent=2))
    result = verify_detection_dataset(published, config=CONFIG)
    assert not result.passed
    assert "ARTIFACT_ROLES_EXPECTED" in failed_checks(result)


def test_an_extra_artifact_entry_fails_the_role_check(published: Path) -> None:
    manifest_path = published / MANIFEST_FILE
    raw = json.loads(manifest_path.read_text())
    raw["artifacts"].append({"relative_path": "surprise.parquet", "sha256": "0" * 64})
    manifest_path.write_text(json.dumps(raw, indent=2))
    result = verify_detection_dataset(published, config=CONFIG)
    assert "ARTIFACT_ROLES_EXPECTED" in failed_checks(result)


# ---------------------------------------------------------------------------
# Verification: fingerprints and relationships
# ---------------------------------------------------------------------------


def test_a_stale_configuration_fingerprint_fails_verification(
    published: Path,
) -> None:
    retuned = DetectionConfig(
        rules={"PAD-BF-001": RuleSettings(parameters={"min_pair_failures": 3})}
    )
    result = verify_detection_dataset(published, config=retuned)
    assert not result.passed
    assert "CONFIG_FINGERPRINT_MATCH" in failed_checks(result)


def test_a_stale_catalog_fingerprint_fails_verification(published: Path) -> None:
    manifest_path = published / MANIFEST_FILE
    raw = json.loads(manifest_path.read_text())
    raw["rule_catalog_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(raw, indent=2))
    result = verify_detection_dataset(published, config=CONFIG)
    assert "CATALOG_FINGERPRINT_MATCH" in failed_checks(result)


def test_an_unsupported_contract_version_fails_verification(
    published: Path,
) -> None:
    manifest_path = published / MANIFEST_FILE
    raw = json.loads(manifest_path.read_text())
    raw["scoring_version"] = "9.9.9"
    manifest_path.write_text(json.dumps(raw, indent=2))
    result = verify_detection_dataset(published, config=CONFIG)
    assert "CONTRACT_VERSIONS_SUPPORTED" in failed_checks(result)


def test_a_broken_detection_relationship_fails_verification(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    """An assessment table missing an anchor the detections still reference."""
    directory = tmp_path / "detection"
    publish(directory, artifacts)
    trimmed = [
        item
        for item in artifacts["assessments"]
        if item.anchor_event_id != artifacts["detections"][0].anchor_event_id
    ]
    write_risk_assessments(trimmed, directory / RISK_FILE)
    result = verify_detection_dataset(directory)
    assert not result.passed
    assert "ARTIFACT_RELATIONSHIPS" in failed_checks(result)


def test_unexpected_scope_metadata_fails_verification(tmp_path: Path) -> None:
    """A scope value in a run the manifest says had no scope table."""
    scoped = build_artifacts(scope=True)
    directory = tmp_path / "detection"
    publish(directory, scoped, entity_scope_present=False)
    result = verify_detection_dataset(directory)
    assert not result.passed
    assert "ARTIFACT_RELATIONSHIPS" in failed_checks(result)


def test_verification_without_a_configuration_skips_the_config_checks(
    published: Path,
) -> None:
    result = verify_detection_dataset(published)
    assert result.passed
    assert "CONFIG_FINGERPRINT_MATCH" not in {check.name for check in result.checks}


def test_verification_never_raises_on_a_corrupt_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = verify_detection_dataset(empty, config=CONFIG)
    assert not result.passed
    assert result.manifest is None


def test_an_invalid_validation_status_fails_verification(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    directory = tmp_path / "detection"
    publish(directory, artifacts, validation_status="invalid")
    result = verify_detection_dataset(directory, config=CONFIG)
    assert not result.passed
    assert "VALIDATION_STATUS_NOT_INVALID" in failed_checks(result)


def test_no_verification_message_carries_an_identifier(
    tmp_path: Path,
) -> None:
    scoped = build_artifacts(scope=True)
    directory = tmp_path / "detection"
    publish(directory, scoped)
    (directory / DETECTIONS_FILE).write_bytes(b"corrupted")
    result = verify_detection_dataset(directory, config=CONFIG)

    rendered = " ".join(check.message for check in result.checks)
    assert not _PSEUDONYM_RE.search(rendered)
    assert "anchor-" not in rendered
    assert str(tmp_path) not in rendered


# ---------------------------------------------------------------------------
# Phase 2 and Phase 3 regressions
# ---------------------------------------------------------------------------


def test_a_plain_phase_two_manifest_still_verifies(tmp_path: Path) -> None:
    """Generalising a shared verifier is what quietly breaks its callers.

    A Phase 2-shaped manifest -- no detection fields at all -- must still pass
    the same ``verify_dataset`` the detection directory now uses.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from password_attack_detector.data.manifest import (
        ArtifactEntry,
        DatasetManifest,
        _get_reproducibility,
        _sha256_file,
        verify_dataset,
    )
    from password_attack_detector.data.schemas import SCHEMA_VERSION

    directory = tmp_path / "events"
    directory.mkdir()
    pq.write_table(pa.table({"event_id": ["e1", "e2"]}), directory / "events.parquet")

    manifest = DatasetManifest(
        manifest_version="1.0.0",
        dataset_id="00000000-0000-5000-8000-000000000000",
        schema_version=SCHEMA_VERSION,
        source_type="synthetic",
        row_count=2,
        ground_truth_row_count=None,
        earliest_event_time=None,
        latest_event_time=None,
        artifacts=[
            ArtifactEntry(
                relative_path="events.parquet",
                sha256=_sha256_file(directory / "events.parquet"),
            )
        ],
        canonical_schema_fingerprint="0" * 64,
        content_fingerprint="1" * 64,
        config_fingerprint=None,
        validation_status="valid",
        created_at="2026-03-01T12:00:00+00:00",
        reproducibility=_get_reproducibility(generator_version=None, seed=None),
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str)
    )
    result = verify_dataset(directory)
    assert result.passed, failed_checks(result)


def test_a_phase_three_feature_manifest_still_verifies(tmp_path: Path) -> None:
    """The Phase 3 superset must still pass the shared verifier unchanged."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from password_attack_detector.data.manifest import _sha256_file
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import FeatureConfig
    from password_attack_detector.features.manifest import (
        build_feature_manifest,
        verify_feature_dataset,
    )
    from password_attack_detector.features.serialization import (
        LABELS_FILE,
        SNAPSHOTS_FILE,
        SPLITS_FILE,
    )
    from password_attack_detector.features.serialization import (
        MANIFEST_FILE as FEATURE_MANIFEST_FILE,
    )

    directory = tmp_path / "features"
    directory.mkdir()
    for name in (SNAPSHOTS_FILE, LABELS_FILE, SPLITS_FILE):
        pq.write_table(pa.table({"event_id": ["e1", "e2"]}), directory / name)

    config = FeatureConfig()
    manifest = build_feature_manifest(
        staging_dir=directory,
        catalog=build_catalog(config),
        config=config,
        feature_fingerprint="2" * 64,
        source_dataset_fingerprint="3" * 64,
        baseline_fingerprint=None,
        row_count=2,
        label_count=2,
        earliest_event_time=None,
        latest_event_time=None,
        validation_status="valid",
    )
    (directory / FEATURE_MANIFEST_FILE).write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str)
    )
    assert _sha256_file(directory / SNAPSHOTS_FILE)
    result = verify_feature_dataset(directory)
    assert result.passed, failed_checks(result)


def test_the_detection_manifest_reuses_the_shared_safety_logic() -> None:
    """Path containment and checksums are inherited, never reimplemented."""
    import inspect

    from password_attack_detector.detection import manifest as detection_manifest

    source = inspect.getsource(detection_manifest)
    assert "verify_dataset" in source
    for reimplemented in ("hashlib.sha256", "is_symlink", "os.path.realpath"):
        assert reimplemented not in source
