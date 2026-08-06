"""Tests for detection serialization and staged publication.

The publication protocol is the part with teeth: a failure must leave the
directory exactly as it was, and the manifest must appear only once everything
else is in place.  Those are tested by making the write fail on purpose and
then inspecting what survived.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.detection.alerts import (
    AlertBuilder,
    build_entity_scope_table,
)
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.engine import DetectionEngine
from password_attack_detector.detection.enums import EvidenceComparator
from password_attack_detector.detection.schemas import EvidenceItem, FiredDetection
from password_attack_detector.detection.scoring import RiskScorer
from password_attack_detector.detection.serialization import (
    ALERT_COLUMNS,
    ALERTS_FILE,
    DETECTION_COLUMNS,
    DETECTIONS_FILE,
    FLOAT_PRECISION,
    MANIFEST_FILE,
    QUALITY_JSON_FILE,
    RISK_COLUMNS,
    RISK_FILE,
    DetectionPublisher,
    compute_alert_fingerprint,
    compute_detection_fingerprint,
    compute_report_fingerprint,
    compute_risk_fingerprint,
    decode_evidence,
    encode_evidence,
    encode_string_list,
    read_alerts,
    read_fired_detections,
    read_risk_assessments,
    read_table_columns,
    write_alerts,
    write_fired_detections,
    write_risk_assessments,
)
from password_attack_detector.exceptions import DataValidationError
from tests.unit.detection import factories

WHEN = factories.WHEN
CONFIG = DetectionConfig()


@pytest.fixture(scope="module")
def artifacts() -> dict[str, Any]:
    """Run the real pipeline once and reuse its output across the module."""
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
    alerting = AlertBuilder(CONFIG).build(scored.assessments, detections=detections)
    return {
        "detections": detections,
        "assessments": list(scored.assessments),
        "alerts": list(alerting.alerts),
    }


def manifest_stub() -> dict[str, Any]:
    """A minimal manifest payload; the real one is built in the manifest tests."""
    return {"manifest_version": "1.0.0", "source_type": "detection"}


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_detections_round_trip(tmp_path: Path, artifacts: dict[str, Any]) -> None:
    path = tmp_path / DETECTIONS_FILE
    write_fired_detections(artifacts["detections"], path)
    restored = read_fired_detections(path)

    assert len(restored) == len(artifacts["detections"])
    by_id = {item.detection_id: item for item in restored}
    for original in artifacts["detections"]:
        read_back = by_id[original.detection_id]
        assert read_back.anchor_event_id == original.anchor_event_id
        assert read_back.rule_id == original.rule_id
        assert read_back.signal_strength == original.signal_strength
        assert read_back.severity is original.severity
        assert read_back.reason_codes == original.reason_codes
        # Evidence is compared in its canonical form, which is the published
        # representation and the one the fingerprint is taken over.
        assert encode_evidence(read_back.evidence) == encode_evidence(original.evidence)


def test_assessments_round_trip(tmp_path: Path, artifacts: dict[str, Any]) -> None:
    path = tmp_path / RISK_FILE
    write_risk_assessments(artifacts["assessments"], path)
    restored = read_risk_assessments(path)
    by_anchor = {item.anchor_event_id: item for item in restored}

    for original in artifacts["assessments"]:
        read_back = by_anchor[original.anchor_event_id]
        assert read_back.risk_score == original.risk_score
        assert read_back.severity is original.severity
        assert read_back.fired_rule_ids == original.fired_rule_ids
        assert read_back.primary_attack_category == original.primary_attack_category
        assert read_back.configuration_fingerprint == original.configuration_fingerprint


def test_alerts_round_trip(tmp_path: Path, artifacts: dict[str, Any]) -> None:
    path = tmp_path / ALERTS_FILE
    write_alerts(artifacts["alerts"], path)
    restored = read_alerts(path)
    assert {item.alert_id for item in restored} == {
        item.alert_id for item in artifacts["alerts"]
    }
    for original, read_back in zip(
        sorted(artifacts["alerts"], key=lambda item: item.alert_id),
        sorted(restored, key=lambda item: item.alert_id),
        strict=True,
    ):
        assert read_back == original


def test_timestamps_stay_utc(tmp_path: Path, artifacts: dict[str, Any]) -> None:
    write_risk_assessments(artifacts["assessments"], tmp_path / RISK_FILE)
    for item in read_risk_assessments(tmp_path / RISK_FILE):
        assert item.anchor_event_time.tzinfo is not None
        assert item.anchor_event_time.utcoffset() == timedelta(0)


def test_a_scope_value_survives_the_alert_round_trip(tmp_path: Path) -> None:
    """The one column permitted to carry a pseudonym must persist exactly."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    rows = [factories.brute_force_row(catalog, anchor_event_id="a1")]
    scored = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    scope = build_entity_scope_table([factories.scope_record("a1", user="ab" * 16)])
    alerts = AlertBuilder(CONFIG).build(scored.assessments, entity_scope=scope).alerts

    write_alerts(list(alerts), tmp_path / ALERTS_FILE)
    restored = read_alerts(tmp_path / ALERTS_FILE)
    assert restored[0].scope_value == alerts[0].scope_value
    assert restored[0].scope_value is not None


# ---------------------------------------------------------------------------
# Column order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "columns", "writer", "key"),
    [
        (DETECTIONS_FILE, DETECTION_COLUMNS, write_fired_detections, "detections"),
        (RISK_FILE, RISK_COLUMNS, write_risk_assessments, "assessments"),
        (ALERTS_FILE, ALERT_COLUMNS, write_alerts, "alerts"),
    ],
)
def test_the_written_column_order_is_the_declared_order(
    tmp_path: Path,
    artifacts: dict[str, Any],
    filename: str,
    columns: tuple[str, ...],
    writer: Any,
    key: str,
) -> None:
    path = tmp_path / filename
    writer(artifacts[key], path)
    assert read_table_columns(path) == list(columns)


def test_no_published_column_is_a_prohibited_one() -> None:
    from password_attack_detector.detection.validation import (
        PROHIBITED_ARTIFACT_COLUMNS,
    )

    for columns in (DETECTION_COLUMNS, RISK_COLUMNS, ALERT_COLUMNS):
        assert not set(columns) & PROHIBITED_ARTIFACT_COLUMNS


# ---------------------------------------------------------------------------
# Evidence encoding
# ---------------------------------------------------------------------------


def evidence(observed: Any = 12, threshold: Any = 8) -> EvidenceItem:
    """Build one evidence item for encoding tests."""
    return EvidenceItem(
        evidence_code="BF_PAIR_FAILURE_COUNT",
        feature_name="pair_failure_count__5m",
        comparator=EvidenceComparator.GTE,
        observed_value=observed,
        threshold_value=threshold,
        unit="count",
        message="Observed 12, at or above the configured threshold of 8.",
    )


def test_evidence_encoding_has_stable_key_order() -> None:
    encoded = encode_evidence([evidence()])
    keys = list(json.loads(encoded)[0])
    assert keys == sorted(keys)


def test_evidence_encoding_preserves_list_order() -> None:
    first, second = evidence(observed=1), evidence(observed=2)
    payload = json.loads(encode_evidence([first, second]))
    assert [item["observed_value"] for item in payload] == [1, 2]


def test_evidence_encoding_is_deterministic() -> None:
    items = [evidence(observed=1.5), evidence(observed=2)]
    assert encode_evidence(items) == encode_evidence(list(items))


def test_evidence_floats_use_the_declared_precision() -> None:
    payload = json.loads(encode_evidence([evidence(observed=1.23456789012345)]))
    rendered = payload[0]["observed_value"]
    assert isinstance(rendered, str)
    assert len(rendered.split(".")[1]) == FLOAT_PRECISION


def test_evidence_round_trips_through_its_encoding() -> None:
    items = (evidence(observed=1.5), evidence(observed=True, threshold=None))
    assert encode_evidence(decode_evidence(encode_evidence(items))) == encode_evidence(
        items
    )


def test_encoded_evidence_carries_no_python_repr() -> None:
    encoded = encode_evidence([evidence(observed=1.5)])
    for marker in ("<", "object at 0x", "EvidenceItem("):
        assert marker not in encoded


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_evidence_value_is_refused(value: float) -> None:
    """The model rejects it first; the encoder refuses it as a second line."""
    with pytest.raises((ValueError, DataValidationError)):
        encode_evidence([evidence(observed=value)])


def test_malformed_evidence_json_is_refused() -> None:
    for payload in ("not json", '{"a": 1}', "[1, 2]", '[{"evidence_code": "X"}]'):
        with pytest.raises(DataValidationError):
            decode_evidence(payload)


def test_a_malformed_string_list_is_refused() -> None:
    from password_attack_detector.detection.serialization import decode_string_list

    for payload in ("not json", '{"a": 1}', "[1, 2]"):
        with pytest.raises(DataValidationError):
            decode_string_list(payload)


def test_string_lists_encode_compactly_and_in_order() -> None:
    assert encode_string_list(["b", "a"]) == '["b","a"]'


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "digest"),
    [
        ("detections", compute_detection_fingerprint),
        ("assessments", compute_risk_fingerprint),
        ("alerts", compute_alert_fingerprint),
    ],
)
def test_fingerprints_are_independent_of_row_order(
    artifacts: dict[str, Any], key: str, digest: Any
) -> None:
    rows = artifacts[key]
    assert digest(rows) == digest(list(reversed(rows)))


def test_fingerprints_are_independent_of_the_output_path(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    write_fired_detections(artifacts["detections"], first / DETECTIONS_FILE)
    write_fired_detections(artifacts["detections"], second / DETECTIONS_FILE)
    assert compute_detection_fingerprint(
        read_fired_detections(first / DETECTIONS_FILE)
    ) == compute_detection_fingerprint(read_fired_detections(second / DETECTIONS_FILE))


def test_a_fingerprint_survives_the_parquet_round_trip(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    """The digest describes logical content, not Parquet's physical layout."""
    write_fired_detections(artifacts["detections"], tmp_path / DETECTIONS_FILE)
    assert compute_detection_fingerprint(
        read_fired_detections(tmp_path / DETECTIONS_FILE)
    ) == compute_detection_fingerprint(artifacts["detections"])


def test_a_changed_value_changes_the_fingerprint(artifacts: dict[str, Any]) -> None:
    detections = artifacts["detections"]
    altered = [
        detections[0].model_construct(
            **{**detections[0].__dict__, "signal_strength": 0.123456}
        ),
        *detections[1:],
    ]
    assert compute_detection_fingerprint(altered) != compute_detection_fingerprint(
        detections
    )


def test_a_report_fingerprint_excludes_the_named_fields() -> None:
    """A creation timestamp must never reach a deterministic digest."""
    base = {"metric": 1, "created_at": "2026-01-01T00:00:00+00:00"}
    later = {"metric": 1, "created_at": "2027-06-06T12:00:00+00:00"}
    assert compute_report_fingerprint(
        base, excluded=("created_at",)
    ) == compute_report_fingerprint(later, excluded=("created_at",))
    assert compute_report_fingerprint(base) != compute_report_fingerprint(later)


def test_a_report_fingerprint_is_key_order_independent() -> None:
    assert compute_report_fingerprint({"a": 1, "b": 2}) == compute_report_fingerprint(
        {"b": 2, "a": 1}
    )


# ---------------------------------------------------------------------------
# Staged publication
# ---------------------------------------------------------------------------


def publish(
    directory: Path,
    artifacts: dict[str, Any],
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> Any:
    """Publish an artifact set into *directory*."""
    return DetectionPublisher(directory, overwrite=overwrite).publish(
        artifacts["detections"],
        artifacts["assessments"],
        artifacts["alerts"],
        manifest_stub(),
        **kwargs,
    )


def test_publication_writes_every_artifact(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    output = tmp_path / "detection"
    result = publish(output, artifacts)

    assert {path.name for path in output.iterdir()} == {
        DETECTIONS_FILE,
        RISK_FILE,
        ALERTS_FILE,
        MANIFEST_FILE,
    }
    assert result.detection_row_count == len(artifacts["detections"])
    assert result.risk_row_count == len(artifacts["assessments"])
    assert result.alert_row_count == len(artifacts["alerts"])


def test_reports_are_published_alongside_the_tables(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    output = tmp_path / "detection"
    publish(output, artifacts, reports={QUALITY_JSON_FILE: '{"ok": true}'})
    assert (output / QUALITY_JSON_FILE).read_text() == '{"ok": true}'


def test_publication_refuses_to_overwrite_without_permission(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    output = tmp_path / "detection"
    publish(output, artifacts)
    with pytest.raises(DataValidationError, match="already exist"):
        publish(output, artifacts)


def test_publication_replaces_when_overwrite_is_requested(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    output = tmp_path / "detection"
    publish(output, artifacts)
    result = publish(output, artifacts, overwrite=True)
    assert result.manifest_path.exists()


def test_publishing_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="no risk assessments"):
        DetectionPublisher(tmp_path / "detection").publish([], [], [], manifest_stub())


def test_the_manifest_is_promoted_last(
    tmp_path: Path, artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its presence is the signal that everything else is already in place."""
    import shutil

    order: list[str] = []
    real_move = shutil.move

    def recording_move(source: str, destination: str) -> Any:
        order.append(Path(destination).name)
        return real_move(source, destination)

    monkeypatch.setattr(
        "password_attack_detector.detection.serialization.shutil.move", recording_move
    )
    publish(tmp_path / "detection", artifacts)
    assert order[-1] == MANIFEST_FILE
    assert set(order[:-1]) == {DETECTIONS_FILE, RISK_FILE, ALERTS_FILE}


def test_a_failed_first_publication_leaves_nothing_behind(
    tmp_path: Path, artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "detection"
    monkeypatch.setattr(
        "password_attack_detector.detection.serialization.write_alerts",
        _explode,
    )
    with pytest.raises(RuntimeError, match="staged failure"):
        publish(output, artifacts)

    assert not (output / MANIFEST_FILE).exists()
    assert list(output.iterdir()) == []


def test_a_failed_overwrite_restores_every_previous_artifact(
    tmp_path: Path, artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte for byte, including the previous manifest."""
    output = tmp_path / "detection"
    publish(output, artifacts)
    before = {path.name: path.read_bytes() for path in sorted(output.iterdir())}

    monkeypatch.setattr(
        "password_attack_detector.detection.serialization.shutil.move", _explode
    )
    with pytest.raises(RuntimeError, match="staged failure"):
        publish(output, artifacts, overwrite=True)

    after = {path.name: path.read_bytes() for path in sorted(output.iterdir())}
    assert after == before


def test_staging_and_backup_directories_are_removed(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    output = tmp_path / "detection"
    publish(output, artifacts)
    publish(output, artifacts, overwrite=True)
    leftovers = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(("_pad_det_stage_", "_pad_det_backup_"))
    ]
    assert leftovers == []


def test_staging_is_removed_even_when_publication_fails(
    tmp_path: Path, artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "password_attack_detector.detection.serialization.write_alerts", _explode
    )
    with pytest.raises(RuntimeError):
        publish(tmp_path / "detection", artifacts)
    leftovers = [
        path for path in tmp_path.iterdir() if path.name.startswith("_pad_det_")
    ]
    assert leftovers == []


def test_published_output_is_byte_identical_across_runs(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    publish(first, artifacts)
    publish(second, artifacts)
    for name in (DETECTIONS_FILE, RISK_FILE, ALERTS_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_the_manifest_is_written_as_sorted_json(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    output = tmp_path / "detection"
    publish(output, artifacts)
    text = (output / MANIFEST_FILE).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == manifest_stub()


def test_every_published_path_stays_relative_to_the_output_directory(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    output = tmp_path / "detection"
    result = publish(output, artifacts)
    for path in (
        result.detections_path,
        result.risk_path,
        result.alerts_path,
        result.manifest_path,
    ):
        assert path.parent == output


def _explode(*args: Any, **kwargs: Any) -> Any:
    """Fail on purpose, to exercise the rollback path."""
    raise RuntimeError("staged failure")


def test_a_non_finite_value_is_refused_before_it_reaches_a_file(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    detections = artifacts["detections"]
    broken: FiredDetection = detections[0].model_construct(
        **{**detections[0].__dict__, "signal_strength": math.nan}
    )
    with pytest.raises(DataValidationError, match="NaN or infinite"):
        write_fired_detections([broken], tmp_path / DETECTIONS_FILE)


def test_a_naive_timestamp_is_normalised_on_write(
    tmp_path: Path, artifacts: dict[str, Any]
) -> None:
    """Every stored timestamp is UTC, whatever offset it arrived with."""
    assessments = artifacts["assessments"]
    shifted = assessments[0].model_construct(
        **{
            **assessments[0].__dict__,
            "anchor_event_time": datetime(2026, 3, 1, 14, 0, tzinfo=UTC).astimezone(),
        }
    )
    write_risk_assessments([shifted], tmp_path / RISK_FILE)
    (restored,) = read_risk_assessments(tmp_path / RISK_FILE)
    assert restored.anchor_event_time == datetime(2026, 3, 1, 14, 0, tzinfo=UTC)


def test_reading_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="Cannot read"):
        read_fired_detections(tmp_path / "absent.parquet")


def test_the_canonical_encoder_renders_an_unknown_type_as_text() -> None:
    """Nothing reaches a digest as a Python repr, whatever type it started as."""
    from uuid import UUID

    from password_attack_detector.detection.serialization import _canonical_value

    rendered = _canonical_value(UUID("3f2504e0-4f89-41d3-9a0c-0305e82c3301"))
    assert rendered == "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
    assert "UUID(" not in rendered


def test_a_failure_after_promotion_removes_what_was_promoted(
    tmp_path: Path, artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rollback path that unlinks already-promoted files."""
    import shutil

    output = tmp_path / "detection"
    real_move = shutil.move
    moved: list[str] = []

    def failing_move(source: str, destination: str) -> Any:
        if Path(destination).name == MANIFEST_FILE:
            raise RuntimeError("staged failure")
        moved.append(Path(destination).name)
        return real_move(source, destination)

    monkeypatch.setattr(
        "password_attack_detector.detection.serialization.shutil.move", failing_move
    )
    with pytest.raises(RuntimeError, match="staged failure"):
        publish(output, artifacts)

    assert moved, "the test must promote something before failing"
    assert list(output.iterdir()) == []
