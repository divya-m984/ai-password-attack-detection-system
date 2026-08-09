"""The authoritative artifact: canonical content, derived identity, staged writes.

Two claims are load-bearing. **Identity is semantic** -- the same content is the
same model wherever and whenever it was written. And **a partial directory is
impossible** -- publication builds beside its destination and moves the finished
thing into place, so a failure leaves what was already there untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelSerializationError
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.models import (
    LogisticRegressionAdapter,
    PriorBaselineAdapter,
)
from password_attack_detector.ml.serialization import (
    ARRAYS_FILE,
    MANIFEST_FILE,
    MODEL_ARTIFACT_FILES,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    ModelDocument,
    build_model_document,
    model_id_for,
    read_model_document,
    stage_model_directory,
)
from tests.ml.models import prepare, publish


@pytest.fixture
def binary() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=140)


@pytest.fixture
def fitted(binary: Any) -> Any:
    """Return a fitted logistic model."""
    return LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_model_id_is_derived_from_the_content(fitted: Any) -> None:
    """Derived, never assigned: the document's own validator recomputes it."""
    document = build_model_document(fitted)
    assert document.model_id == model_id_for(
        document.model_content_fingerprint,
        task=document.task,
        family=document.model_family,
    )


def test_identity_does_not_depend_on_the_directory(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """The same model in two places is the same model."""
    left = publish(tmp_path / "somewhere", fitted, binary.preprocessor)
    right = publish(tmp_path / "elsewhere", fitted, binary.preprocessor)
    first = json.loads((left / MODEL_FILE).read_text(encoding="utf-8"))
    second = json.loads((right / MODEL_FILE).read_text(encoding="utf-8"))
    assert first["model_id"] == second["model_id"]
    assert first["model_content_fingerprint"] == second["model_content_fingerprint"]
    assert (left / MODEL_FILE).read_bytes() == (right / MODEL_FILE).read_bytes()


def test_identity_does_not_depend_on_the_clock(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """Two publications a measurable interval apart agree byte for byte."""
    left = publish(tmp_path / "first", fitted, binary.preprocessor)
    time.sleep(1.1)
    right = publish(tmp_path / "second", fitted, binary.preprocessor)
    for name in (MODEL_FILE, ARRAYS_FILE, PREPROCESSOR_FILE):
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


def test_an_observational_timestamp_does_not_change_identity(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """``published_at`` is recorded and excluded from every fingerprint."""
    from password_attack_detector.ml.manifest import read_model_manifest

    left = publish(
        tmp_path / "a", fitted, binary.preprocessor, published_at="2026-01-01T00:00:00Z"
    )
    right = publish(
        tmp_path / "b", fitted, binary.preprocessor, published_at="2030-06-06T12:00:00Z"
    )
    first = read_model_manifest(
        json.loads((left / MANIFEST_FILE).read_text(encoding="utf-8"))
    )
    second = read_model_manifest(
        json.loads((right / MANIFEST_FILE).read_text(encoding="utf-8"))
    )
    assert first.published_at != second.published_at
    assert first.model_id == second.model_id
    assert first.fingerprint() == second.fingerprint()


def test_a_changed_coefficient_changes_the_identity(fitted: Any) -> None:
    """And it is not a constant that always agrees."""
    nudged = fitted.model_copy(
        update={
            "arrays": {
                **fitted.arrays,
                "intercept": np.asarray(fitted.arrays["intercept"]) + 1.0,
            }
        }
    )
    assert nudged.content_fingerprint() != fitted.content_fingerprint()
    assert (
        build_model_document(nudged).model_id != build_model_document(fitted).model_id
    )


def test_a_changed_feature_order_changes_the_identity(fitted: Any) -> None:
    """The same columns in a different arrangement are a different model."""
    swapped = fitted.model_copy(
        update={
            "transformed_feature_names": (
                fitted.transformed_feature_names[1],
                fitted.transformed_feature_names[0],
                *fitted.transformed_feature_names[2:],
            )
        }
    )
    assert swapped.content_fingerprint() != fitted.content_fingerprint()


def test_the_family_and_task_participate_in_the_identity() -> None:
    """Two families with identical numbers are still two models."""
    digest = "a" * 64
    assert model_id_for(
        digest, task=MLTask.BINARY_MALICIOUS, family=ModelFamily.LOGISTIC_REGRESSION
    ) != model_id_for(
        digest, task=MLTask.BINARY_MALICIOUS, family=ModelFamily.RANDOM_FOREST
    )
    assert model_id_for(
        digest, task=MLTask.BINARY_MALICIOUS, family=ModelFamily.RANDOM_FOREST
    ) != model_id_for(
        digest, task=MLTask.ATTACK_CATEGORY, family=ModelFamily.RANDOM_FOREST
    )


def test_a_malformed_fingerprint_is_refused() -> None:
    """An identifier derived from a non-digest would not be derived from anything."""
    with pytest.raises(ModelSerializationError, match="SHA-256"):
        model_id_for(
            "short", task=MLTask.BINARY_MALICIOUS, family=ModelFamily.PRIOR_BASELINE
        )


def test_an_assigned_model_id_is_refused(fitted: Any) -> None:
    """A document naming an identifier its content does not derive is invalid."""
    payload = build_model_document(fitted).to_dict()
    payload["model_id"] = "00000000-0000-5000-8000-000000000000"
    with pytest.raises(ModelSerializationError, match="not a valid model document"):
        read_model_document(payload)


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def test_the_document_declares_every_array(fitted: Any) -> None:
    """Name, dtype, shape, and a value digest, so the archive can be checked."""
    document = build_model_document(fitted)
    declared = {entry.name for entry in document.array_manifest}
    assert declared == set(fitted.arrays)
    for entry in document.array_manifest:
        array = fitted.arrays[entry.name]
        assert entry.dtype == str(array.dtype)
        assert entry.shape == tuple(int(extent) for extent in array.shape)


def test_the_document_is_canonical_json(fitted: Any) -> None:
    """Sorted keys and ASCII, so one document has one rendering."""
    payload = json.loads(build_model_document(fitted).to_json())
    assert list(payload) == sorted(payload)


def test_calibration_is_recorded_as_not_fitted(fitted: Any) -> None:
    """Absent by contract, and stated rather than left to be inferred."""
    assert build_model_document(fitted).calibration_status == "not_fitted"


def test_a_document_claiming_calibration_is_refused(fitted: Any) -> None:
    """No calibrator exists at this contract version, so none can be declared."""
    payload = build_model_document(fitted).to_dict()
    payload["calibration_status"] = "isotonic"
    with pytest.raises(ModelSerializationError, match="not a valid model document"):
        read_model_document(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(surprise_field=True),
        lambda p: p.pop("class_order"),
        lambda p: p.update(ml_schema_version="9.9.9"),
    ],
    ids=["unknown-field", "missing-field", "wrong-version"],
)
def test_a_malformed_document_is_refused(fitted: Any, mutate: Any) -> None:
    """Strict on load: a document read loosely would drop what it did not
    understand and then verify a fingerprint over the remainder."""
    payload = build_model_document(fitted).to_dict()
    mutate(payload)
    with pytest.raises(ModelSerializationError):
        read_model_document(payload)


@pytest.mark.parametrize("payload", [None, [], "text", 42])
def test_a_document_of_the_wrong_shape_is_refused(payload: Any) -> None:
    """Valid JSON of the wrong shape is still not a model document."""
    with pytest.raises(ModelSerializationError):
        read_model_document(payload)


def test_the_document_round_trips(fitted: Any) -> None:
    """Serialise, load, serialise: byte-identical."""
    text = build_model_document(fitted).to_json()
    assert read_model_document(json.loads(text)).to_json() == text


def test_a_document_repeating_an_array_is_refused(fitted: Any) -> None:
    """One name cannot describe two arrays."""
    payload = build_model_document(fitted).to_dict()
    payload["array_manifest"] = [payload["array_manifest"][0]] * 2
    with pytest.raises(ModelSerializationError, match="not a valid model document"):
        read_model_document(payload)


# ---------------------------------------------------------------------------
# Staging and publication
# ---------------------------------------------------------------------------


def test_staging_writes_the_content_files_but_not_the_manifest(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """The manifest covers these files, so it cannot be written beside them."""
    staged = tmp_path / "staged"
    staged.mkdir()
    digests = stage_model_directory(
        staged, fitted=fitted, preprocessor=binary.preprocessor
    )
    assert set(digests) == {MODEL_FILE, ARRAYS_FILE, PREPROCESSOR_FILE}
    assert not (staged / MANIFEST_FILE).exists()


def test_a_published_directory_holds_exactly_the_declared_files(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """A closed set, not a minimum."""
    directory = publish(tmp_path / "model", fitted, binary.preprocessor)
    assert {item.name for item in directory.iterdir()} == set(MODEL_ARTIFACT_FILES)


def test_the_manifest_is_written_last(fitted: Any, binary: Any, tmp_path: Path) -> None:
    """Its presence is the signal that everything it covers is already there.

    Asserted by failing the manifest builder: the destination must then not
    exist at all, rather than existing with three of four files.
    """
    from password_attack_detector.ml.serialization import write_model_directory

    def failing(digests: dict[str, str], sizes: dict[str, int]) -> Any:
        raise ModelSerializationError("the manifest could not be built")

    target = tmp_path / "model"
    with pytest.raises(ModelSerializationError):
        write_model_directory(
            target,
            fitted=fitted,
            preprocessor=binary.preprocessor,
            manifest_builder=failing,
        )
    assert not target.exists()


def test_a_failed_publication_leaves_no_staging_directory(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """Temporary state is removed whether or not the write succeeded."""
    from password_attack_detector.ml.serialization import write_model_directory

    def failing(digests: dict[str, str], sizes: dict[str, int]) -> Any:
        raise ModelSerializationError("no")

    with pytest.raises(ModelSerializationError):
        write_model_directory(
            tmp_path / "model",
            fitted=fitted,
            preprocessor=binary.preprocessor,
            manifest_builder=failing,
        )
    assert list(tmp_path.iterdir()) == []


def test_a_failed_overwrite_restores_the_previous_model(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """The destination is exactly what it was, byte for byte."""
    from password_attack_detector.ml.serialization import write_model_directory

    target = publish(tmp_path / "model", fitted, binary.preprocessor)
    before = {item.name: item.read_bytes() for item in sorted(target.iterdir())}

    def failing(digests: dict[str, str], sizes: dict[str, int]) -> Any:
        raise ModelSerializationError("no")

    other = PriorBaselineAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    with pytest.raises(ModelSerializationError):
        write_model_directory(
            target,
            fitted=other,
            preprocessor=binary.preprocessor,
            manifest_builder=failing,
            overwrite=True,
        )
    after = {item.name: item.read_bytes() for item in sorted(target.iterdir())}
    assert after == before


def test_overwriting_without_permission_is_refused(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """Another run may already have recorded a fingerprint for what is there."""
    publish(tmp_path / "model", fitted, binary.preprocessor)
    with pytest.raises(ModelSerializationError, match="already exists"):
        publish(tmp_path / "model", fitted, binary.preprocessor)


def test_overwriting_with_permission_replaces_the_model(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """An explicit decision, and it takes effect completely."""
    target = publish(tmp_path / "model", fitted, binary.preprocessor)
    other = PriorBaselineAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    publish(tmp_path / "model", other, binary.preprocessor, overwrite=True)
    payload = json.loads((target / MODEL_FILE).read_text(encoding="utf-8"))
    assert payload["model_family"] == "prior_baseline"
    assert {item.name for item in target.iterdir()} == set(MODEL_ARTIFACT_FILES)


def test_a_mismatched_preprocessor_is_refused(fitted: Any, tmp_path: Path) -> None:
    """Publishing them together would produce an artifact that disagrees with
    itself."""
    other = prepare(count=90, seed=99)
    with pytest.raises(ModelSerializationError, match="not the one this model"):
        publish(tmp_path / "model", fitted, other.preprocessor)


def test_an_unpublishable_family_is_refused(binary: Any, tmp_path: Path) -> None:
    """Its serializer would rest on undocumented internals."""
    from password_attack_detector.ml.models import HistogramBoostingAdapter

    boosted = HistogramBoostingAdapter(max_iter=5).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    with pytest.raises(ModelSerializationError, match="not publishable"):
        publish(tmp_path / "model", boosted, binary.preprocessor)


def test_the_staged_archive_is_verified_before_promotion(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """Round-tripped in staging, so a broken archive never reaches a destination."""
    directory = publish(tmp_path / "model", fitted, binary.preprocessor)
    from password_attack_detector.ml.npz import read_npz_bytes

    restored = read_npz_bytes((directory / ARRAYS_FILE).read_bytes())
    assert set(restored) == set(fitted.arrays)
    for name, array in fitted.arrays.items():
        assert np.array_equal(restored[name], np.asarray(array))


def test_tampering_changes_what_verification_reports(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """A single appended byte is enough."""
    from password_attack_detector.ml.manifest import verify_model_artifact

    directory = publish(tmp_path / "model", fitted, binary.preprocessor)
    assert verify_model_artifact(directory).passed is True
    path = directory / ARRAYS_FILE
    path.write_bytes(path.read_bytes() + b"\n")
    outcome = verify_model_artifact(directory)
    assert outcome.passed is False
    assert outcome.error_code == "CHECKSUM_MISMATCH"


def test_the_document_carries_no_prohibited_content(
    fitted: Any, binary: Any, tmp_path: Path
) -> None:
    """A privacy sweep of the rendered document."""
    directory = publish(tmp_path / "model", fitted, binary.preprocessor)
    text = (directory / MODEL_FILE).read_text(encoding="utf-8")
    for token in (
        "anchor_event_id",
        "campaign_id",
        "usr_",
        "src_",
        "password",
        "token",
        "credential",
        "latitude",
        "longitude",
        "/home/",
        str(tmp_path),
    ):
        assert token not in text, token


def test_the_document_declares_no_prohibited_field() -> None:
    """The schema itself carries no ground-truth-shaped field name."""
    from password_attack_detector.ml.schemas import prohibited_metadata_fields

    assert prohibited_metadata_fields(list(ModelDocument.model_fields)) == ()
