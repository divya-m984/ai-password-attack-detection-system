"""The model manifest: what a published model claims, and how to check it.

The manifest is the artifact's own account of itself -- what produced it, what
it was fitted against, and what every file in the directory should hash to.
:func:`verify_model_artifact` reads that account and tests it, in an order
chosen so that nothing expensive or trusting happens before something cheap and
sceptical has passed.

**What is deliberately absent.**

*No metric.*  Not a validation score, not a test score, not an accuracy.  A
manifest that carried one would make the artifact the natural place to look up
how good the model is, and the whole Phase 5 discipline is that a test
evaluation is a separate, frozen, one-per-scope record.

*No identity.*  No event identifier, no campaign identifier, no pseudonym, no
coordinate, no absolute path, and no training row.  Every field is a version,
a fingerprint, a count, or a declared name.

*No timestamp in the fingerprints.*  ``published_at`` exists because the Phase 2
manifest convention has one and a reader wants it, and it is excluded from
every fingerprint and from the model identity, so republishing an identical
model produces an identical identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ManifestVerificationError
from password_attack_detector.ml.dependencies import (
    SKLEARN_BELOW_VERSION,
    SKLEARN_MINIMUM_VERSION,
)
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.schemas import ML_SCHEMA_VERSION, Sha256Hex
from password_attack_detector.ml.serialization import (
    CALIBRATION_NOT_FITTED,
    MANIFEST_FILE,
    MODEL_ARTIFACT_FILES,
    canonical_json,
)

__all__ = [
    "CHAMPION_NOT_SELECTED",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_ARTIFACT_BYTES",
    "ManifestFileEntry",
    "ModelManifest",
    "VerificationOutcome",
    "build_model_manifest",
    "read_model_manifest",
    "verify_model_artifact",
]

#: This manifest contract's own version.
MANIFEST_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Champion selection happens in a later milestone; the field says so rather
#: than being absent, so a reader never has to infer it from a missing key.
CHAMPION_NOT_SELECTED: Final[str] = "not_selected"

#: Ceiling applied to each declared artifact before it is read.
MAX_ARTIFACT_BYTES: Final[int] = 512 * 1024 * 1024


class ManifestFileEntry(BaseModel):
    """One file the manifest covers, by relative name and digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def check_path(self) -> ManifestFileEntry:
        """Only the declared artifact names, and nothing path-shaped."""
        if self.relative_path not in MODEL_ARTIFACT_FILES:
            raise ValueError(
                f"{self.relative_path!r} is not one of the declared model "
                f"artifacts; a manifest cannot introduce a file the contract "
                f"does not know about"
            )
        return self


class ModelManifest(BaseModel):
    """Provenance and integrity for one published model directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    ml_schema_version: str = ML_SCHEMA_VERSION
    serializer_id: str
    serializer_version: int = Field(ge=1)
    inference_adapter_id: str

    model_id: str
    model_content_fingerprint: Sha256Hex
    catalog_model_id: str
    model_family: ModelFamily
    task: MLTask
    seed: int = Field(ge=0)

    required_feature_schema_version: str
    feature_catalog_fingerprint: Sha256Hex | None = None
    allowlist_fingerprint: Sha256Hex | None = None
    eligible_feature_list_fingerprint: Sha256Hex
    transformed_feature_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    class_weight_fingerprint: Sha256Hex | None = None
    split_config_fingerprint: Sha256Hex | None = None
    training_data_fingerprint: Sha256Hex | None = None
    ml_config_fingerprint: Sha256Hex | None = None
    model_catalog_fingerprint: Sha256Hex

    class_order: tuple[str, ...]
    raw_feature_count: int = Field(ge=1)
    transformed_feature_count: int = Field(ge=1)
    train_row_count: int = Field(ge=1)

    dependency_versions: dict[str, str]
    scikit_learn_version: str | None = None
    scikit_learn_minimum_version: str = SKLEARN_MINIMUM_VERSION
    scikit_learn_below_version: str = SKLEARN_BELOW_VERSION
    uv_lock_sha256: Sha256Hex | None = None

    calibration_status: str = CALIBRATION_NOT_FITTED
    champion_status: str = CHAMPION_NOT_SELECTED
    champion_eligible: bool
    experimental: bool

    expected_files: tuple[str, ...] = MODEL_ARTIFACT_FILES
    files: tuple[ManifestFileEntry, ...]
    #: Observational only. Excluded from every fingerprint and from the model
    #: identity, so republishing identical content republishes an identical
    #: model.
    published_at: str | None = None

    @model_validator(mode="after")
    def check_manifest(self) -> ModelManifest:
        """The manifest covers exactly the declared artifacts, once each."""
        if set(self.expected_files) != set(MODEL_ARTIFACT_FILES):
            raise ValueError(
                "expected_files must be exactly the declared model artifacts"
            )
        covered = [entry.relative_path for entry in self.files]
        if len(set(covered)) != len(covered):
            raise ValueError("the manifest covers a file more than once")
        # The manifest cannot digest itself, so it covers the other three.
        required = set(MODEL_ARTIFACT_FILES) - {MANIFEST_FILE}
        if set(covered) != required:
            raise ValueError(
                f"the manifest must cover exactly {sorted(required)}, and does "
                f"not cover itself"
            )
        if self.calibration_status != CALIBRATION_NOT_FITTED:
            raise ValueError(
                f"calibration_status must be {CALIBRATION_NOT_FITTED!r} at this "
                f"contract version"
            )
        if self.champion_eligible and self.experimental:
            raise ValueError(
                "a model cannot be both champion-eligible and experimental"
            )
        return self

    def digest_of(self, relative_path: str) -> str:
        """Return the recorded digest for *relative_path*, or raise."""
        for entry in self.files:
            if entry.relative_path == relative_path:
                return entry.sha256
        raise ManifestVerificationError(
            f"the manifest records no digest for {relative_path!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping this manifest serialises to."""
        return dict(self.model_dump(mode="json"))

    def to_json(self) -> str:
        """Return the canonical rendering."""
        return canonical_json(self.to_dict())

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic content, with the observational field removed."""
        payload = self.to_dict()
        payload.pop("published_at", None)
        return payload

    def fingerprint(self) -> str:
        """Return a digest over the manifest's semantic content."""
        return hashlib.sha256(
            canonical_json(self.fingerprint_data()).encode()
        ).hexdigest()


def build_model_manifest(
    *,
    document: Any,
    digests: dict[str, str],
    sizes: dict[str, int],
    seed: int,
    required_feature_schema_version: str,
    model_catalog_fingerprint: str,
    feature_catalog_fingerprint: str | None = None,
    allowlist_fingerprint: str | None = None,
    split_config_fingerprint: str | None = None,
    training_data_fingerprint: str | None = None,
    ml_config_fingerprint: str | None = None,
    uv_lock_sha256: str | None = None,
    published_at: str | None = None,
) -> ModelManifest:
    """Return the manifest describing a staged model directory.

    Every fingerprint the model itself already carries is copied from the
    document rather than passed in again: a manifest that could disagree with
    the document it covers would be a second source of truth, and
    :func:`verify_model_artifact` would then be checking two claims rather than
    one claim against the bytes.
    """
    from password_attack_detector.ml.dependencies import (
        collect_dependency_versions,
        installed_version,
    )

    covered = sorted(set(MODEL_ARTIFACT_FILES) - {MANIFEST_FILE})
    missing = [name for name in covered if name not in digests or name not in sizes]
    if missing:
        raise ManifestVerificationError(
            f"the staged directory did not produce a digest for {missing}"
        )
    return ModelManifest(
        serializer_id=document.serializer_id,
        serializer_version=document.serializer_version,
        inference_adapter_id=document.inference_adapter_id,
        model_id=document.model_id,
        model_content_fingerprint=document.model_content_fingerprint,
        catalog_model_id=document.catalog_model_id,
        model_family=document.model_family,
        task=document.task,
        seed=seed,
        required_feature_schema_version=required_feature_schema_version,
        feature_catalog_fingerprint=feature_catalog_fingerprint,
        allowlist_fingerprint=allowlist_fingerprint,
        eligible_feature_list_fingerprint=document.eligible_feature_list_fingerprint,
        transformed_feature_fingerprint=document.transformed_feature_fingerprint,
        preprocessor_fingerprint=document.preprocessor_fingerprint,
        class_weight_fingerprint=document.class_weight_fingerprint,
        split_config_fingerprint=split_config_fingerprint,
        training_data_fingerprint=training_data_fingerprint,
        ml_config_fingerprint=ml_config_fingerprint,
        model_catalog_fingerprint=model_catalog_fingerprint,
        class_order=document.class_order,
        raw_feature_count=len(document.raw_feature_names),
        transformed_feature_count=len(document.transformed_feature_names),
        train_row_count=document.train_row_count,
        dependency_versions=collect_dependency_versions(),
        scikit_learn_version=installed_version("scikit-learn"),
        uv_lock_sha256=uv_lock_sha256,
        champion_eligible=document.champion_eligible,
        experimental=document.experimental,
        files=tuple(
            ManifestFileEntry(
                relative_path=name, sha256=digests[name], size_bytes=sizes[name]
            )
            for name in covered
        ),
        published_at=published_at,
    )


def read_model_manifest(payload: Any) -> ModelManifest:
    """Return the manifest *payload* describes, or raise."""
    if not isinstance(payload, dict):
        raise ManifestVerificationError(
            f"a model manifest must be a JSON object, got {type(payload).__name__}"
        )
    version = payload.get("manifest_schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestVerificationError(
            f"the manifest declares schema version {version!r}; this build "
            f"implements {MANIFEST_SCHEMA_VERSION!r}"
        )
    try:
        return ModelManifest.model_validate(payload)
    except Exception as exc:
        raise ManifestVerificationError(
            f"the model manifest is not valid ({type(exc).__name__})"
        ) from None


class VerificationOutcome(BaseModel):
    """The result of checking one model directory.

    Aggregate and sanitised: a code, a count, and a declared name. No path, no
    identifier, and no fitted value -- a verification failure is often the first
    thing pasted into a ticket.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checks_run: int = Field(ge=0)
    model_id: str | None = None
    model_family: str | None = None
    task: str | None = None
    ml_schema_version: str | None = None
    manifest_schema_version: str | None = None
    serializer_id: str | None = None
    serializer_version: int | None = None
    file_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_detail: str | None = None


def verify_model_artifact(directory: Path) -> VerificationOutcome:
    """Verify a model directory's structure, integrity, and internal agreement.

    Structural verification only: it reads JSON and hashes bytes, and never
    constructs an estimator or an inference adapter. That is
    :mod:`password_attack_detector.ml.inference`'s job, and it calls this first.

    The order is fail-closed and cheapest-first. Each step assumes only what the
    previous ones established:

    1. the directory exists and is a directory;
    2. it contains exactly the declared files -- no extra, no missing;
    3. no member is a symbolic link;
    4. no member exceeds the size ceiling;
    5. the manifest parses under a supported schema version;
    6. every recorded digest matches the bytes on disk;
    7. ``model.json`` parses under a supported schema version;
    8. the document and the manifest agree on identity, family, task, orders,
       and every upstream fingerprint;
    9. the archive holds exactly the declared arrays, with the declared dtypes,
       shapes, and value digests;
    10. the model identity recomputes from the declared content.

    Never raises for an invalid artifact: it returns a failing outcome carrying
    a stable code, which is what a CLI and a caller both want.
    """
    checks = 0
    try:
        checks += 1
        directory = Path(directory)
        if not directory.is_dir():
            return _fail(
                "MODEL_DIR_MISSING", "the model directory does not exist", checks
            )

        checks += 1
        present = {item.name for item in directory.iterdir()}
        unexpected = sorted(present - set(MODEL_ARTIFACT_FILES))
        missing = sorted(set(MODEL_ARTIFACT_FILES) - present)
        if missing:
            return _fail(
                "MODEL_FILE_MISSING",
                f"{len(missing)} declared artifact(s) are absent",
                checks,
            )
        if unexpected:
            return _fail(
                "MODEL_FILE_UNEXPECTED",
                f"{len(unexpected)} file(s) are not declared model artifacts",
                checks,
            )

        checks += 1
        for name in MODEL_ARTIFACT_FILES:
            member = directory / name
            if member.is_symlink():
                return _fail(
                    "MODEL_FILE_SYMLINK",
                    "a declared artifact is a symbolic link, which could point "
                    "anywhere on the filesystem",
                    checks,
                )
            if not member.is_file():
                return _fail(
                    "MODEL_FILE_NOT_REGULAR",
                    "a declared artifact is not a regular file",
                    checks,
                )

        checks += 1
        for name in MODEL_ARTIFACT_FILES:
            if (directory / name).stat().st_size > MAX_ARTIFACT_BYTES:
                return _fail(
                    "MODEL_FILE_TOO_LARGE",
                    "a declared artifact exceeds the configured size ceiling",
                    checks,
                )

        checks += 1
        try:
            manifest = read_model_manifest(
                json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
            )
        except (
            ManifestVerificationError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            return _fail("MANIFEST_INVALID", _sanitize(exc), checks)

        checks += 1
        for entry in manifest.files:
            payload = (directory / entry.relative_path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry.sha256:
                return _fail(
                    "CHECKSUM_MISMATCH",
                    "a declared artifact does not match its recorded digest",
                    checks,
                    manifest,
                )
            if len(payload) != entry.size_bytes:
                return _fail(
                    "SIZE_MISMATCH",
                    "a declared artifact does not match its recorded size",
                    checks,
                    manifest,
                )

        checks += 1
        from password_attack_detector.ml.serialization import (
            MODEL_FILE,
            model_id_for,
            read_model_document,
        )

        try:
            document = read_model_document(
                json.loads((directory / MODEL_FILE).read_text(encoding="utf-8"))
            )
        except Exception as exc:
            return _fail("MODEL_DOCUMENT_INVALID", _sanitize(exc), checks, manifest)

        checks += 1
        disagreements = [
            name
            for name, left, right in (
                ("model_id", document.model_id, manifest.model_id),
                (
                    "model_content_fingerprint",
                    document.model_content_fingerprint,
                    manifest.model_content_fingerprint,
                ),
                ("model_family", document.model_family, manifest.model_family),
                ("task", document.task, manifest.task),
                ("serializer_id", document.serializer_id, manifest.serializer_id),
                (
                    "serializer_version",
                    document.serializer_version,
                    manifest.serializer_version,
                ),
                (
                    "inference_adapter_id",
                    document.inference_adapter_id,
                    manifest.inference_adapter_id,
                ),
                (
                    "catalog_model_id",
                    document.catalog_model_id,
                    manifest.catalog_model_id,
                ),
                ("class_order", document.class_order, manifest.class_order),
                (
                    "preprocessor_fingerprint",
                    document.preprocessor_fingerprint,
                    manifest.preprocessor_fingerprint,
                ),
                (
                    "eligible_feature_list_fingerprint",
                    document.eligible_feature_list_fingerprint,
                    manifest.eligible_feature_list_fingerprint,
                ),
                (
                    "transformed_feature_fingerprint",
                    document.transformed_feature_fingerprint,
                    manifest.transformed_feature_fingerprint,
                ),
                (
                    "class_weight_fingerprint",
                    document.class_weight_fingerprint,
                    manifest.class_weight_fingerprint,
                ),
                (
                    "transformed_feature_count",
                    len(document.transformed_feature_names),
                    manifest.transformed_feature_count,
                ),
                (
                    "raw_feature_count",
                    len(document.raw_feature_names),
                    manifest.raw_feature_count,
                ),
                ("train_row_count", document.train_row_count, manifest.train_row_count),
                (
                    "champion_eligible",
                    document.champion_eligible,
                    manifest.champion_eligible,
                ),
                ("experimental", document.experimental, manifest.experimental),
            )
            if left != right
        ]
        if disagreements:
            return _fail(
                "MANIFEST_DOCUMENT_DISAGREEMENT",
                f"{len(disagreements)} field(s) disagree between the manifest "
                f"and the model document",
                checks,
                manifest,
            )

        checks += 1
        from password_attack_detector.ml.npz import array_digest, read_npz_bytes
        from password_attack_detector.ml.serialization import ARRAYS_FILE

        try:
            arrays = read_npz_bytes((directory / ARRAYS_FILE).read_bytes())
        except Exception as exc:
            return _fail("ARCHIVE_INVALID", _sanitize(exc), checks, manifest)

        declared_arrays = {entry.name: entry for entry in document.array_manifest}
        if set(arrays) != set(declared_arrays):
            return _fail(
                "ARCHIVE_CONTENT_MISMATCH",
                "the archive does not hold exactly the declared arrays",
                checks,
                manifest,
            )
        for name, array_entry in declared_arrays.items():
            array = arrays[name]
            if str(array.dtype) != array_entry.dtype:
                return _fail(
                    "ARRAY_DTYPE_MISMATCH",
                    f"array {name!r} has a different dtype from its declaration",
                    checks,
                    manifest,
                )
            if tuple(int(extent) for extent in array.shape) != array_entry.shape:
                return _fail(
                    "ARRAY_SHAPE_MISMATCH",
                    f"array {name!r} has a different shape from its declaration",
                    checks,
                    manifest,
                )
            if array_digest({name: array}) != array_entry.digest:
                return _fail(
                    "ARRAY_DIGEST_MISMATCH",
                    f"array {name!r} does not match its declared value digest",
                    checks,
                    manifest,
                )

        checks += 1
        recomputed = model_id_for(
            document.model_content_fingerprint,
            task=document.task,
            family=document.model_family,
        )
        if recomputed != document.model_id:
            return _fail(
                "MODEL_ID_MISMATCH",
                "the recorded model identifier is not the one this content derives",
                checks,
                manifest,
            )

        return VerificationOutcome(
            passed=True,
            checks_run=checks,
            model_id=manifest.model_id,
            model_family=str(manifest.model_family),
            task=str(manifest.task),
            ml_schema_version=manifest.ml_schema_version,
            manifest_schema_version=manifest.manifest_schema_version,
            serializer_id=manifest.serializer_id,
            serializer_version=manifest.serializer_version,
            file_count=len(MODEL_ARTIFACT_FILES),
        )
    except OSError as exc:
        # The message could quote a path, so only the type survives.
        return _fail("MODEL_DIR_UNREADABLE", type(exc).__name__, checks)


def _fail(
    code: str, detail: str, checks: int, manifest: ModelManifest | None = None
) -> VerificationOutcome:
    """Return a failing outcome carrying a stable code and sanitised detail."""
    return VerificationOutcome(
        passed=False,
        checks_run=checks,
        error_code=code,
        error_detail=detail,
        model_id=None if manifest is None else manifest.model_id,
        model_family=None if manifest is None else str(manifest.model_family),
        task=None if manifest is None else str(manifest.task),
        ml_schema_version=None if manifest is None else manifest.ml_schema_version,
        manifest_schema_version=(
            None if manifest is None else manifest.manifest_schema_version
        ),
        serializer_id=None if manifest is None else manifest.serializer_id,
        serializer_version=None if manifest is None else manifest.serializer_version,
        file_count=0 if manifest is None else len(MODEL_ARTIFACT_FILES),
    )


def _sanitize(exc: Exception) -> str:
    """Return a detail string that cannot carry a path or a payload.

    A JSON decoder quotes the offending text and a filesystem error quotes the
    path. Neither belongs in output somebody will paste into a ticket, so only
    the exception type survives.
    """
    return type(exc).__name__
