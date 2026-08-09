"""The authoritative model artifact: canonical JSON beside a deterministic archive.

A published model is four files and no code:

===================== ===============================================
``model.json``        contract, hyperparameters, orders, array manifest
``arrays.npz``        every fitted number, deterministically encoded
``preprocessor.json`` the Milestone 3 state the matrix was built by
``model_manifest.json`` integrity and provenance, **written last**
===================== ===============================================

**Pickle and joblib are not authoritative, here or anywhere.**  Unpickling
executes whatever the payload asks for, which makes a model file a code file,
and a model file is exactly the artifact most likely to be copied between
machines by somebody who did not produce it.  A pickle is also opaque to
review, unstable across library versions, and impossible to diff.  So the
artifact is JSON and arrays, and loading it constructs project types only.

**Identity is semantic.**  ``model_content_fingerprint`` is a SHA-256 over the
canonical content -- the numbers, the orders, the hyperparameters, the upstream
fingerprints -- and ``model_id`` is a UUIDv5 over that.  Neither reads the
directory name, the model alias, the output path, a file timestamp, an archive
timestamp, or the moment of publication.  The same model written twice in two
places is the same model, and a test asserts it.

**Writing is staged.**  The directory is built beside its destination,
validated there, and only then moved into place, with the manifest written last
so its presence means a complete artifact.  Any failure leaves the destination
exactly as it was.  This is the publication *primitive*; the orchestration that
decides when to call it belongs to the training milestone.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import (
    ModelSerializationError,
)
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.models.base import FittedModel
from password_attack_detector.ml.npz import (
    array_digest,
    read_npz_bytes,
    write_npz_bytes,
)
from password_attack_detector.ml.preprocessing import FittedPreprocessor
from password_attack_detector.ml.schemas import ML_SCHEMA_VERSION, ScoreSemantics

__all__ = [
    "ARRAYS_FILE",
    "MANIFEST_FILE",
    "MODEL_ARTIFACT_FILES",
    "MODEL_FILE",
    "PREPROCESSOR_FILE",
    "ArrayEntry",
    "ModelDocument",
    "build_model_document",
    "canonical_json",
    "model_id_for",
    "read_model_document",
    "stage_model_directory",
    "write_model_directory",
]

MODEL_FILE: Final[str] = "model.json"
ARRAYS_FILE: Final[str] = "arrays.npz"
PREPROCESSOR_FILE: Final[str] = "preprocessor.json"
MANIFEST_FILE: Final[str] = "model_manifest.json"

#: Exactly the files a model directory may contain, in write order.
#:
#: A closed set, not a minimum. An unexpected file in a model directory is
#: refused rather than ignored: a loader that tolerates extra files is a loader
#: somebody can put something in.
MODEL_ARTIFACT_FILES: Final[tuple[str, ...]] = (
    MODEL_FILE,
    ARRAYS_FILE,
    PREPROCESSOR_FILE,
    MANIFEST_FILE,
)

#: Namespace for derived model identifiers. Fixed, so the same content always
#: derives the same identifier -- on any machine, in any run, in any year.
_NS_MODEL: Final[uuid.UUID] = uuid.UUID("7d5f6d1e-0d5a-5e5b-9c2f-2a8b4f6c1d30")

#: A calibrator is deliberately absent at this milestone, and its absence is
#: *stated* rather than left to be inferred from a missing file.
CALIBRATION_NOT_FITTED: Final[str] = "not_fitted"


def canonical_json(payload: Any) -> str:
    """Return the one rendering this layer treats as canonical."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def model_id_for(content_fingerprint: str, *, task: MLTask, family: ModelFamily) -> str:
    """Return the derived identifier for a model with this content.

    Derived, never assigned. Two models with the same content are the same
    model however they were named on disk, and a model whose content changed by
    one coefficient is a different one however carefully the directory was
    reused.
    """
    if len(content_fingerprint) != 64:
        raise ModelSerializationError(
            "a model content fingerprint is a 64-character SHA-256 digest"
        )
    name = f"{content_fingerprint}|{task!s}|{family!s}"
    return str(uuid.uuid5(_NS_MODEL, name))


class ArrayEntry(BaseModel):
    """One array declared by ``model.json`` and stored in ``arrays.npz``.

    The declaration exists so the archive can be checked against what the model
    says it should hold *before* anything is read out of it: a name, a dtype, a
    shape, and a digest of the values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dtype: str
    shape: tuple[int, ...]
    digest: str

    @model_validator(mode="after")
    def check_entry(self) -> ArrayEntry:
        """Shapes are non-negative and the digest is a SHA-256."""
        if any(extent < 0 for extent in self.shape):
            raise ValueError(f"array {self.name!r} declares a negative extent")
        if len(self.digest) != 64:
            raise ValueError(f"array {self.name!r} declares a malformed digest")
        return self


class ModelDocument(BaseModel):
    """The typed content of ``model.json``.

    Everything a scoring adapter needs, plus everything a verifier needs to
    decide whether to trust it. No path, no timestamp, no host, and no training
    row: the fitted numbers live in the archive, and everything here is
    contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ml_schema_version: str = ML_SCHEMA_VERSION
    serializer_id: str
    serializer_version: int = Field(ge=1)
    inference_adapter_id: str
    catalog_model_id: str
    model_family: ModelFamily
    task: MLTask
    model_id: str
    model_content_fingerprint: str
    hyperparameters: dict[str, bool | int | float | str]
    parameters: dict[str, Any]
    class_order: tuple[str, ...]
    raw_feature_names: tuple[str, ...]
    transformed_feature_names: tuple[str, ...]
    transformed_feature_fingerprint: str
    eligible_feature_list_fingerprint: str
    preprocessor_fingerprint: str
    class_weight_fingerprint: str | None
    array_manifest: tuple[ArrayEntry, ...]
    score_semantics: ScoreSemantics
    train_row_count: int = Field(ge=1)
    calibration_status: str = CALIBRATION_NOT_FITTED
    champion_eligible: bool
    experimental: bool

    @model_validator(mode="after")
    def check_document(self) -> ModelDocument:
        """Internal agreement: orders unique, arrays named once, identity derived."""
        if len(set(self.transformed_feature_names)) != len(
            self.transformed_feature_names
        ):
            raise ValueError("transformed_feature_names repeats a column")
        if len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order repeats a class")
        names = [entry.name for entry in self.array_manifest]
        if len(set(names)) != len(names):
            raise ValueError("array_manifest declares an array more than once")
        if self.model_id != model_id_for(
            self.model_content_fingerprint, task=self.task, family=self.model_family
        ):
            raise ValueError(
                "model_id is not the identifier this content derives; an "
                "assigned identifier is not an identity"
            )
        if self.calibration_status != CALIBRATION_NOT_FITTED:
            raise ValueError(
                f"calibration_status must be {CALIBRATION_NOT_FITTED!r} at this "
                f"contract version; no calibrator has been fitted, and a "
                f"document claiming otherwise is not one this build wrote"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping this document serialises to."""
        return dict(self.model_dump(mode="json"))

    def to_json(self) -> str:
        """Return the canonical rendering."""
        return canonical_json(self.to_dict())


def transformed_feature_fingerprint(names: tuple[str, ...]) -> str:
    """Return a digest over the transformed column order.

    Order-sensitive on purpose: the same columns in a different arrangement are
    a different matrix, and a fingerprint that ignored order would call them
    interchangeable.
    """
    return hashlib.sha256(canonical_json(list(names)).encode()).hexdigest()


def build_model_document(fitted: FittedModel) -> ModelDocument:
    """Return the ``model.json`` content describing *fitted*."""
    fingerprint = fitted.content_fingerprint()
    entries = tuple(
        ArrayEntry(
            name=name,
            dtype=str(fitted.arrays[name].dtype),
            shape=tuple(int(extent) for extent in fitted.arrays[name].shape),
            digest=array_digest({name: fitted.arrays[name]}),
        )
        for name in sorted(fitted.arrays)
    )
    return ModelDocument(
        serializer_id=fitted.serializer_id,
        serializer_version=fitted.serializer_version,
        inference_adapter_id=fitted.inference_adapter_id,
        catalog_model_id=fitted.catalog_model_id,
        model_family=fitted.family,
        task=fitted.task,
        model_id=model_id_for(fingerprint, task=fitted.task, family=fitted.family),
        model_content_fingerprint=fingerprint,
        hyperparameters=dict(fitted.hyperparameters),
        parameters=dict(fitted.parameters),
        class_order=fitted.class_order,
        raw_feature_names=fitted.raw_feature_names,
        transformed_feature_names=fitted.transformed_feature_names,
        transformed_feature_fingerprint=transformed_feature_fingerprint(
            fitted.transformed_feature_names
        ),
        eligible_feature_list_fingerprint=fitted.eligible_feature_list_fingerprint,
        preprocessor_fingerprint=fitted.preprocessor_fingerprint,
        class_weight_fingerprint=fitted.class_weight_fingerprint,
        array_manifest=entries,
        score_semantics=fitted.score_semantics,
        train_row_count=fitted.train_row_count,
        champion_eligible=fitted.champion_eligible,
        experimental=fitted.experimental,
    )


def read_model_document(payload: Any) -> ModelDocument:
    """Return the document *payload* describes, or raise.

    Strict: an unknown field, a missing field, or an unsupported schema version
    all fail. A document read loosely would drop what it did not understand and
    then verify a fingerprint over the remainder.
    """
    if not isinstance(payload, dict):
        raise ModelSerializationError(
            f"model.json must be a JSON object, got {type(payload).__name__}"
        )
    version = payload.get("ml_schema_version")
    if version != ML_SCHEMA_VERSION:
        raise ModelSerializationError(
            f"model.json declares ML schema version {version!r}; this build "
            f"implements {ML_SCHEMA_VERSION!r}"
        )
    try:
        return ModelDocument.model_validate(payload)
    except Exception as exc:
        raise ModelSerializationError(
            f"model.json is not a valid model document ({type(exc).__name__})"
        ) from None


def stage_model_directory(
    directory: Path,
    *,
    fitted: FittedModel,
    preprocessor: FittedPreprocessor,
) -> dict[str, str]:
    """Write the three content files into *directory* and return their digests.

    The pure primitive: no promotion, no backup, no rollback, and deliberately
    **no manifest** -- the manifest covers these files, so it is written by the
    caller once these have been verified in place.

    Raises:
        ModelSerializationError: when the family may not be published, or when
            the preprocessor does not describe the matrix the model was fitted
            on.
    """
    from password_attack_detector.ml.models import PUBLISHABLE_FAMILIES

    if fitted.family not in PUBLISHABLE_FAMILIES:
        raise ModelSerializationError(
            f"family {str(fitted.family)!r} is not publishable: its serializer "
            f"would rest on undocumented estimator internals, so it may be "
            f"fitted and compared in process but never stored"
        )
    if preprocessor.fingerprint() != fitted.preprocessor_fingerprint:
        raise ModelSerializationError(
            "the supplied preprocessor is not the one this model was fitted "
            "against; publishing them together would produce an artifact whose "
            "own fingerprints disagree"
        )
    if preprocessor.output_feature_names != fitted.transformed_feature_names:
        raise ModelSerializationError(
            "the supplied preprocessor emits a different transformed column "
            "order from the one the model was fitted on"
        )

    document = build_model_document(fitted)
    payloads = {
        MODEL_FILE: document.to_json().encode(),
        ARRAYS_FILE: write_npz_bytes(fitted.arrays),
        PREPROCESSOR_FILE: preprocessor.to_json().encode(),
    }
    digests: dict[str, str] = {}
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def write_model_directory(
    target: Path,
    *,
    fitted: FittedModel,
    preprocessor: FittedPreprocessor,
    manifest_builder: Any,
    overwrite: bool = False,
) -> Path:
    """Publish a complete model directory at *target*, atomically.

    The sequence, and why each step is where it is:

    1. build everything in a temporary sibling, so a failure never touches the
       destination;
    2. write the three content files and digest them there;
    3. verify the staged archive reads back to the same arrays -- the artifact
       is checked before it is promoted, not after;
    4. write ``model_manifest.json`` **last**, so its presence is the signal
       that everything it covers is already there;
    5. back the destination up if one exists and overwrite was permitted;
    6. rename the staging directory into place;
    7. restore the backup on any failure, and remove the staging directory
       either way.

    A sibling directory rather than the system temporary area: ``rename`` is
    atomic only within a filesystem, and a cross-device move would degrade into
    a copy that can be interrupted halfway.

    Args:
        target: the model directory to create.
        fitted: the model to publish.
        preprocessor: the state its matrix was built by.
        manifest_builder: called with the staged digests and sizes, returning
            the manifest to write. Injected rather than imported so this module
            does not depend on the manifest module that depends on it.
        overwrite: permit replacing an existing directory.

    Raises:
        ModelSerializationError: when the target exists and overwrite was not
            permitted, or when staging or verification fails.
    """
    target = Path(target)
    if target.exists() and not overwrite:
        raise ModelSerializationError(
            "the model directory already exists; overwriting is an explicit "
            "decision, because a published model is something another run may "
            "already have recorded a fingerprint for"
        )

    staging = target.parent / f".staging-{target.name}"
    backup = target.parent / f".backup-{target.name}"
    for scratch in (staging, backup):
        if scratch.exists():
            shutil.rmtree(scratch)
    staging.mkdir(parents=True)

    promoted = False
    try:
        digests = stage_model_directory(
            staging, fitted=fitted, preprocessor=preprocessor
        )
        _verify_staged_archive(staging, fitted.arrays)
        sizes = {name: (staging / name).stat().st_size for name in digests}
        manifest = manifest_builder(digests, sizes)
        (staging / MANIFEST_FILE).write_text(manifest.to_json(), encoding="utf-8")

        unexpected = sorted(
            item.name
            for item in staging.iterdir()
            if item.name not in MODEL_ARTIFACT_FILES
        )
        if unexpected:
            raise ModelSerializationError(
                f"staging produced {len(unexpected)} unexpected file(s); a model "
                f"directory holds exactly the declared artifacts"
            )

        if target.exists():
            target.rename(backup)
        staging.rename(target)
        promoted = True
    finally:
        if not promoted and backup.exists():
            if target.exists():
                shutil.rmtree(target)
            backup.rename(target)
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists():
            shutil.rmtree(backup)
    return target


def _verify_staged_archive(directory: Path, arrays: Mapping[str, Any]) -> None:
    """Raise unless the staged archive reads back to exactly *arrays*."""
    import numpy as np

    restored = read_npz_bytes((directory / ARRAYS_FILE).read_bytes())
    if set(restored) != set(arrays):
        raise ModelSerializationError(
            "the staged archive does not hold the arrays the model declares"
        )
    for name, array in arrays.items():
        if not np.array_equal(restored[name], np.asarray(array)):
            raise ModelSerializationError(
                f"the staged archive does not round-trip array {name!r}; the "
                f"artifact is refused rather than published"
            )
