"""The model manifest and the verification chain that reads it.

The manifest is the artifact's account of itself, and verification tests that
account against the bytes. Every test here breaks one thing and asserts the
stable code it produces -- a caller and a CLI both need the code rather than
the prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import ManifestVerificationError
from password_attack_detector.ml.enums import MLTask
from password_attack_detector.ml.manifest import (
    CHAMPION_NOT_SELECTED,
    MANIFEST_SCHEMA_VERSION,
    ModelManifest,
    read_model_manifest,
    verify_model_artifact,
)
from password_attack_detector.ml.models import LogisticRegressionAdapter
from password_attack_detector.ml.serialization import (
    ARRAYS_FILE,
    MANIFEST_FILE,
    MODEL_ARTIFACT_FILES,
    MODEL_FILE,
    PREPROCESSOR_FILE,
)
from tests.ml.models import prepare, publish


@pytest.fixture
def published(tmp_path: Path) -> Path:
    """Return a freshly published, valid model directory."""
    batch = prepare(count=140)
    fitted = LogisticRegressionAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    return publish(tmp_path / "model", fitted, batch.preprocessor)


def manifest_of(directory: Path) -> ModelManifest:
    """Return the manifest a directory carries."""
    return read_model_manifest(
        json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    )


def rewrite(directory: Path, **changes: Any) -> None:
    """Rewrite the manifest with *changes* applied to its raw payload."""
    payload = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    payload.update(changes)
    (directory / MANIFEST_FILE).write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# What the manifest records
# ---------------------------------------------------------------------------


def test_a_valid_artifact_verifies(published: Path) -> None:
    """The happy path, and it reports what it checked."""
    outcome = verify_model_artifact(published)
    assert outcome.passed is True
    assert outcome.error_code is None
    assert outcome.checks_run >= 10
    assert outcome.file_count == len(MODEL_ARTIFACT_FILES)


def test_the_manifest_covers_every_file_except_itself(published: Path) -> None:
    """It cannot digest itself, so it covers the other three."""
    manifest = manifest_of(published)
    covered = {entry.relative_path for entry in manifest.files}
    assert covered == {MODEL_FILE, ARRAYS_FILE, PREPROCESSOR_FILE}
    assert set(manifest.expected_files) == set(MODEL_ARTIFACT_FILES)


def test_the_manifest_records_the_dependency_versions(published: Path) -> None:
    """ "scikit-learn 1.9" is not enough to reproduce a model."""
    manifest = manifest_of(published)
    assert "scikit-learn" in manifest.dependency_versions
    assert "python" in manifest.dependency_versions
    assert "numpy" in manifest.dependency_versions
    assert manifest.scikit_learn_version is not None


def test_the_manifest_records_the_bounded_range(published: Path) -> None:
    """Both bounds, so a reader can decide whether its runtime qualifies."""
    manifest = manifest_of(published)
    assert manifest.scikit_learn_minimum_version == "1.9.0"
    assert manifest.scikit_learn_below_version == "1.10"


def test_the_manifest_records_every_upstream_fingerprint(published: Path) -> None:
    """Feature contract, preprocessing, transformed order, model catalog."""
    manifest = manifest_of(published)
    assert manifest.eligible_feature_list_fingerprint
    assert manifest.preprocessor_fingerprint
    assert manifest.transformed_feature_fingerprint
    assert manifest.model_catalog_fingerprint
    assert manifest.class_weight_fingerprint


def test_the_manifest_records_calibration_as_not_fitted(published: Path) -> None:
    """Stated rather than inferred from a missing calibrator file."""
    assert manifest_of(published).calibration_status == "not_fitted"


def test_the_manifest_records_champion_as_not_selected(published: Path) -> None:
    """Eligibility is not selection, and nothing here selects."""
    assert manifest_of(published).champion_status == CHAMPION_NOT_SELECTED


def test_the_manifest_carries_no_metric(published: Path) -> None:
    """A test evaluation is a separate frozen record, never a manifest field."""
    payload = json.loads((published / MANIFEST_FILE).read_text(encoding="utf-8"))
    for banned in (
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "metrics",
        "score",
        "f1",
    ):
        assert not any(banned in key for key in payload), banned


def test_the_manifest_carries_no_prohibited_content(published: Path) -> None:
    """A privacy sweep of the rendered manifest."""
    text = (published / MANIFEST_FILE).read_text(encoding="utf-8")
    for token in (
        "anchor_event_id",
        "campaign_id",
        "usr_",
        "src_",
        "password",
        "credential",
        "latitude",
        "/home/",
        str(published),
    ):
        assert token not in text, token


def test_the_manifest_declares_no_prohibited_field() -> None:
    """The schema carries no ground-truth-shaped field name."""
    from password_attack_detector.ml.schemas import prohibited_metadata_fields

    assert prohibited_metadata_fields(list(ModelManifest.model_fields)) == ()


def test_the_manifest_fingerprint_excludes_the_timestamp(published: Path) -> None:
    """Observational metadata is recorded and never fingerprinted."""
    manifest = manifest_of(published)
    assert "published_at" not in manifest.fingerprint_data()


# ---------------------------------------------------------------------------
# What verification refuses
# ---------------------------------------------------------------------------


def test_a_missing_directory_is_reported(tmp_path: Path) -> None:
    """A stable code rather than an exception, so a CLI can render it."""
    outcome = verify_model_artifact(tmp_path / "nothing")
    assert outcome.passed is False
    assert outcome.error_code == "MODEL_DIR_MISSING"


def test_a_missing_file_is_reported(published: Path) -> None:
    """Every declared artifact must be there."""
    (published / PREPROCESSOR_FILE).unlink()
    assert verify_model_artifact(published).error_code == "MODEL_FILE_MISSING"


def test_an_unexpected_file_is_reported(published: Path) -> None:
    """A loader that tolerates extra files is one somebody can put something in."""
    (published / "extra.json").write_text("{}", encoding="utf-8")
    assert verify_model_artifact(published).error_code == "MODEL_FILE_UNEXPECTED"


def test_a_symlinked_artifact_is_reported(published: Path, tmp_path: Path) -> None:
    """A member pointing anywhere on the filesystem is refused."""
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}", encoding="utf-8")
    target = published / PREPROCESSOR_FILE
    target.unlink()
    target.symlink_to(elsewhere)
    assert verify_model_artifact(published).error_code == "MODEL_FILE_SYMLINK"


def test_an_oversized_artifact_is_reported(
    published: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ceiling is applied before anything is parsed."""
    from password_attack_detector.ml import manifest as module

    monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 8)
    assert verify_model_artifact(published).error_code == "MODEL_FILE_TOO_LARGE"


def test_a_malformed_manifest_is_reported(published: Path) -> None:
    """Not valid JSON is not a manifest."""
    (published / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    outcome = verify_model_artifact(published)
    assert outcome.error_code == "MANIFEST_INVALID"
    assert outcome.error_detail == "JSONDecodeError"


def test_a_manifest_from_another_schema_version_is_reported(published: Path) -> None:
    """A contract this build does not implement is not interpreted optimistically."""
    rewrite(published, manifest_schema_version="9.9.9")
    assert verify_model_artifact(published).error_code == "MANIFEST_INVALID"


def test_a_manifest_with_an_unknown_field_is_reported(published: Path) -> None:
    """Strict on load, as everywhere else in this layer."""
    rewrite(published, surprise_field=True)
    assert verify_model_artifact(published).error_code == "MANIFEST_INVALID"


@pytest.mark.parametrize("name", [MODEL_FILE, ARRAYS_FILE, PREPROCESSOR_FILE])
def test_a_tampered_file_is_reported(published: Path, name: str) -> None:
    """One appended byte in any covered file is enough."""
    path = published / name
    path.write_bytes(path.read_bytes() + b"\n")
    assert verify_model_artifact(published).error_code in {
        "CHECKSUM_MISMATCH",
        "SIZE_MISMATCH",
    }


def test_a_serializer_mismatch_is_reported(published: Path) -> None:
    """The manifest and the document must describe one model."""
    rewrite(published, serializer_id="json_something_else_v1")
    outcome = verify_model_artifact(published)
    assert outcome.error_code == "MANIFEST_DOCUMENT_DISAGREEMENT"


def test_a_model_id_mismatch_is_reported(published: Path) -> None:
    """An artifact assembled from two models does not verify."""
    rewrite(published, model_id="00000000-0000-5000-8000-000000000000")
    assert (
        verify_model_artifact(published).error_code == "MANIFEST_DOCUMENT_DISAGREEMENT"
    )


def test_a_preprocessor_fingerprint_mismatch_is_reported(published: Path) -> None:
    """A model scored against a differently built matrix is a different model."""
    rewrite(published, preprocessor_fingerprint="b" * 64)
    assert (
        verify_model_artifact(published).error_code == "MANIFEST_DOCUMENT_DISAGREEMENT"
    )


def test_a_feature_fingerprint_mismatch_is_reported(published: Path) -> None:
    """The reviewed feature contract is part of a model's identity."""
    rewrite(published, eligible_feature_list_fingerprint="c" * 64)
    assert (
        verify_model_artifact(published).error_code == "MANIFEST_DOCUMENT_DISAGREEMENT"
    )


def test_a_train_row_count_mismatch_is_reported(published: Path) -> None:
    """Counts are covered too, so a manifest cannot overstate its evidence."""
    rewrite(published, train_row_count=999999)
    assert (
        verify_model_artifact(published).error_code == "MANIFEST_DOCUMENT_DISAGREEMENT"
    )


def test_a_manifest_covering_itself_is_refused(published: Path) -> None:
    """A self-referential digest cannot be satisfied."""
    payload = json.loads((published / MANIFEST_FILE).read_text(encoding="utf-8"))
    payload["files"].append(
        {"relative_path": MANIFEST_FILE, "sha256": "d" * 64, "size_bytes": 1}
    )
    (published / MANIFEST_FILE).write_text(json.dumps(payload), encoding="utf-8")
    assert verify_model_artifact(published).error_code == "MANIFEST_INVALID"


def test_a_manifest_naming_a_file_outside_the_contract_is_refused() -> None:
    """A manifest cannot introduce an artifact the contract does not know about."""
    from password_attack_detector.ml.manifest import ManifestFileEntry

    with pytest.raises(ValueError, match="not one of the declared"):
        ManifestFileEntry(relative_path="../escape.json", sha256="e" * 64, size_bytes=1)


def test_a_champion_eligible_experimental_manifest_is_refused(
    published: Path,
) -> None:
    """The two states are mutually exclusive by definition."""
    rewrite(published, champion_eligible=True, experimental=True)
    assert verify_model_artifact(published).error_code == "MANIFEST_INVALID"


def test_a_manifest_claiming_calibration_is_refused(published: Path) -> None:
    """No calibrator exists at this contract version."""
    rewrite(published, calibration_status="platt")
    assert verify_model_artifact(published).error_code == "MANIFEST_INVALID"


def test_a_swapped_archive_is_reported(published: Path, tmp_path: Path) -> None:
    """Replacing the archive and its digest still fails on the array declaration."""
    import hashlib

    import numpy as np

    from password_attack_detector.ml.npz import write_npz_bytes

    payload = write_npz_bytes(
        {"coefficients": np.zeros((1, 3)), "intercept": np.zeros(1)}
    )
    (published / ARRAYS_FILE).write_bytes(payload)
    manifest = json.loads((published / MANIFEST_FILE).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["relative_path"] == ARRAYS_FILE:
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["size_bytes"] = len(payload)
    (published / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_model_artifact(published).error_code in {
        "ARRAY_SHAPE_MISMATCH",
        "ARRAY_DIGEST_MISMATCH",
        "ARCHIVE_CONTENT_MISMATCH",
    }


def test_a_failing_outcome_carries_no_path(published: Path) -> None:
    """A verification failure is often the first thing pasted into a ticket."""
    (published / MANIFEST_FILE).write_text("{oops", encoding="utf-8")
    outcome = verify_model_artifact(published)
    rendered = f"{outcome.error_code} {outcome.error_detail}"
    assert str(published) not in rendered
    assert "/" not in rendered


def test_the_manifest_schema_version_is_declared(published: Path) -> None:
    """State says which contract it was written against."""
    assert manifest_of(published).manifest_schema_version == MANIFEST_SCHEMA_VERSION


@pytest.mark.parametrize("payload", [None, [], "text", 7])
def test_a_manifest_of_the_wrong_shape_is_refused(payload: Any) -> None:
    """Valid JSON of the wrong shape is still not a manifest."""
    with pytest.raises(ManifestVerificationError):
        read_model_manifest(payload)


def test_the_manifest_round_trips(published: Path) -> None:
    """Serialise, load, serialise: byte-identical."""
    text = manifest_of(published).to_json()
    assert read_model_manifest(json.loads(text)).to_json() == text
