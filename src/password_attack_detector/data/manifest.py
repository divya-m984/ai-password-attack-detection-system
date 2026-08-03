"""Dataset manifest model, verification, and factory for published datasets.

A ``DatasetManifest`` is a machine-readable JSON document that accompanies
every published dataset directory.  It captures enough information to:

- Verify file integrity (SHA-256 checksums of each artifact).
- Reproduce the dataset (seed, generator version, config fingerprint).
- Assess dataset quality without opening the event files.

Safe-path contract
------------------
All artifact entries use **relative** paths only.  The verification functions
enforce this using ``Path.is_relative_to()`` (never ``str.startswith``):

1. Reject absolute paths (``path.is_absolute()``).
2. Reject ``..`` components (``".." in path.parts``).
3. Resolve and call ``resolved.is_relative_to(base.resolve())`` to detect
   symlink escapes.

Privacy guarantees
------------------
The manifest never contains:
- Hostnames or system usernames.
- Absolute filesystem paths or home-directory references.
- Plaintext secrets or pseudonymization keys.
- Raw identifiers from authentication events.

Public API
----------
- ``DatasetManifest``       -- frozen Pydantic manifest model
- ``ArtifactEntry``         -- single artifact (relative path + SHA-256)
- ``ReproducibilityInfo``   -- runtime versions and seed
- ``VerificationCheck``     -- one named boolean check with a message
- ``VerificationResult``    -- aggregated result of 10 verification checks
- ``build_synthetic_manifest`` -- factory for synthetic-data publishes
- ``verify_dataset``        -- run all 10 checks on a published directory
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from password_attack_detector.data.schemas import SCHEMA_VERSION, AuthEvent
from password_attack_detector.exceptions import ManifestVerificationError

__all__ = [
    "ArtifactEntry",
    "DatasetManifest",
    "ReproducibilityInfo",
    "VerificationCheck",
    "VerificationResult",
    "build_directory_manifest",
    "build_synthetic_manifest",
    "verify_dataset",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MANIFEST_VERSION: str = "1.0.0"
_KNOWN_MANIFEST_VERSIONS: frozenset[str] = frozenset({"1.0.0"})

# Namespace UUID for deterministic dataset_id generation (UUIDv5).
_NS_DATASET = uuid.UUID("b2c3d4e5-f6a7-5b8c-9d0e-1f2a3b4c5d6e")

# ---------------------------------------------------------------------------
# Manifest models
# ---------------------------------------------------------------------------


class ArtifactEntry(BaseModel):
    """One published artifact file (relative path + SHA-256 checksum)."""

    model_config = ConfigDict(frozen=True)

    relative_path: str
    """Path to the artifact relative to the manifest file's directory.
    Must be relative, must not contain ``..`` components."""

    sha256: str
    """Lowercase hex SHA-256 checksum of the artifact file."""


class ReproducibilityInfo(BaseModel):
    """Runtime environment metadata for dataset reproduction."""

    model_config = ConfigDict(frozen=True)

    python_version: str
    numpy_version: str
    pandas_version: str
    pyarrow_version: str
    uv_lock_sha256: str | None
    generator_version: str | None
    seed: int | None


class DatasetManifest(BaseModel):
    """Machine-readable manifest for one published authentication-event dataset.

    Written as ``manifest.json`` in the dataset directory.  The manifest is
    **always promoted last** during publication so its presence indicates a
    complete, coherent dataset.
    """

    model_config = ConfigDict(frozen=True)

    manifest_version: str
    dataset_id: str
    schema_version: str
    source_type: str
    row_count: int
    ground_truth_row_count: int | None
    earliest_event_time: str | None
    latest_event_time: str | None
    artifacts: list[ArtifactEntry]
    canonical_schema_fingerprint: str
    content_fingerprint: str
    config_fingerprint: str | None
    validation_status: str
    created_at: str
    reproducibility: ReproducibilityInfo

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all manifest fields."""
        return self.model_dump()

    @classmethod
    def from_json_file(cls, path: Path) -> DatasetManifest:
        """Load and validate a manifest from *path*.

        Raises
        ------
        ManifestVerificationError
            When the file cannot be read or does not conform to the schema.
        """
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return cls.model_validate(data)
        except Exception as exc:
            raise ManifestVerificationError(
                f"Cannot load manifest ({type(exc).__name__})"
            ) from exc


# ---------------------------------------------------------------------------
# Verification types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationCheck:
    """Result of a single named verification check."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class VerificationResult:
    """Aggregated result of running all 10 dataset verification checks.

    ``passed`` is ``True`` only when every individual check passes.
    ``manifest`` is ``None`` when the manifest file could not be loaded.
    """

    passed: bool
    checks: tuple[VerificationCheck, ...]
    manifest: DatasetManifest | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_fingerprint() -> str:
    """Return a SHA-256 fingerprint of the canonical event schema field names.

    Computed from ``AuthEvent.model_fields`` key order so that the fingerprint
    is stable across restarts and does not require importing from serialization.
    """
    canonical = json.dumps(list(AuthEvent.model_fields.keys()), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _uv_lock_sha256() -> str | None:
    """Return the SHA-256 of ``uv.lock`` at the repo root, or ``None``."""
    try:
        from password_attack_detector.paths import find_repo_root

        lock_path = find_repo_root() / "uv.lock"
        if lock_path.exists():
            return _sha256_file(lock_path)
    except Exception:
        pass
    return None


def _get_reproducibility(
    *,
    generator_version: str | None,
    seed: int | None,
) -> ReproducibilityInfo:
    """Capture runtime environment versions for reproducibility metadata."""
    return ReproducibilityInfo(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        numpy_version=str(np.__version__),
        pandas_version=str(pd.__version__),
        pyarrow_version=str(pa.__version__),
        uv_lock_sha256=_uv_lock_sha256(),
        generator_version=generator_version,
        seed=seed,
    )


def _is_safe_relative_path(rel_path: str, base_dir: Path) -> tuple[bool, str]:
    """Check whether *rel_path* is safe to use as an artifact path.

    Returns ``(True, "ok")`` on success or ``(False, reason)`` on failure.
    Uses ``Path.is_relative_to()`` -- never ``str.startswith``.
    """
    p = Path(rel_path)

    if p.is_absolute():
        return False, "path is absolute"

    if ".." in p.parts:
        return False, "path contains '..' component"

    # Detect symlink escape: resolve both and check containment.
    resolved = (base_dir / p).resolve()
    base_resolved = base_dir.resolve()
    if not resolved.is_relative_to(base_resolved):
        return False, "path escapes base directory (symlink or traversal)"

    return True, "ok"


def _find_events_artifact(manifest: DatasetManifest) -> ArtifactEntry | None:
    """Return the first artifact entry whose path ends with 'events.parquet'."""
    for entry in manifest.artifacts:
        if Path(entry.relative_path).name == "events.parquet":
            return entry
    return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_directory_manifest(
    directory: Path,
    events: Any,  # Sequence[AuthEvent]
    *,
    source_type: str = "ingested",
    labels_count: int = 0,
    validation_status: str = "valid",
    content_fingerprint: str,
    config_fingerprint: str | None = None,
) -> DatasetManifest:
    """Build a ``DatasetManifest`` from artifact files in an existing directory.

    Used by the ``data manifest`` CLI command to create a manifest for an
    already-published (or recently ingested) dataset that has no manifest yet.

    Parameters
    ----------
    directory:
        Directory containing the artifact files.
    events:
        Sequence of ``AuthEvent`` objects already read from ``events.parquet``.
    source_type:
        ``"synthetic"`` or ``"ingested"`` (default ``"ingested"``).
    labels_count:
        Number of ground-truth label rows (0 when no labels are present).
    validation_status:
        Validation status string (e.g. ``"valid"``).
    content_fingerprint:
        SHA-256 hex fingerprint of the event content.
    config_fingerprint:
        Optional config fingerprint (``None`` for ingested data).
    """
    artifact_entries: list[ArtifactEntry] = []
    for fname in ("events.parquet", "labels.parquet", "events.jsonl"):
        fpath = directory / fname
        if fpath.exists():
            artifact_entries.append(
                ArtifactEntry(relative_path=fname, sha256=_sha256_file(fpath))
            )

    earliest: str | None = None
    latest: str | None = None
    if events:
        times = [e.event_time for e in events]
        earliest = min(times).isoformat()
        latest = max(times).isoformat()

    dataset_id = str(uuid.uuid5(_NS_DATASET, content_fingerprint))
    reproducibility = _get_reproducibility(generator_version=None, seed=None)

    return DatasetManifest(
        manifest_version=_MANIFEST_VERSION,
        dataset_id=dataset_id,
        schema_version=SCHEMA_VERSION,
        source_type=source_type,
        row_count=len(events),
        ground_truth_row_count=labels_count if labels_count > 0 else None,
        earliest_event_time=earliest,
        latest_event_time=latest,
        artifacts=artifact_entries,
        canonical_schema_fingerprint=_schema_fingerprint(),
        content_fingerprint=content_fingerprint,
        config_fingerprint=config_fingerprint,
        validation_status=validation_status,
        created_at=datetime.now(UTC).isoformat(),
        reproducibility=reproducibility,
    )


def build_synthetic_manifest(
    *,
    events: Any,  # Sequence[AuthEvent] -- avoid import cycle
    labels: Any,  # Sequence[GroundTruthLabel]
    config: Any,  # SyntheticConfig
    config_fingerprint: str,
    content_fingerprint: str,
    staging_dir: Path,
    artifact_filenames: tuple[str, ...],
    validation_status: str,
) -> DatasetManifest:
    """Build a ``DatasetManifest`` for a synthetic dataset publish.

    Parameters
    ----------
    events:
        Sequence of ``AuthEvent`` objects (used for temporal range).
    labels:
        Sequence of ``GroundTruthLabel`` objects.
    config:
        ``SyntheticConfig`` used to generate the dataset.
    config_fingerprint:
        SHA-256 hex fingerprint of the config (from ``SyntheticConfig.fingerprint()``).
    content_fingerprint:
        SHA-256 hex fingerprint of the canonical event content.
    staging_dir:
        Directory containing the staged (pre-promotion) artifact files.
    artifact_filenames:
        Filenames of staged artifacts (relative to *staging_dir*).
    validation_status:
        Validation status string (e.g. ``"valid"``, ``"warning"``).
    """
    # Artifact entries with checksums.
    artifact_entries: list[ArtifactEntry] = []
    for fname in artifact_filenames:
        fpath = staging_dir / fname
        if fpath.exists():
            artifact_entries.append(
                ArtifactEntry(relative_path=fname, sha256=_sha256_file(fpath))
            )

    # Temporal range from events.
    earliest: str | None = None
    latest: str | None = None
    if events:
        times = [e.event_time for e in events]
        earliest = min(times).isoformat()
        latest = max(times).isoformat()

    # Deterministic dataset_id based on content fingerprint.
    dataset_id = str(uuid.uuid5(_NS_DATASET, content_fingerprint))

    reproducibility = _get_reproducibility(
        generator_version=config.generator_version,
        seed=config.seed,
    )

    return DatasetManifest(
        manifest_version=_MANIFEST_VERSION,
        dataset_id=dataset_id,
        schema_version=SCHEMA_VERSION,
        source_type="synthetic",
        row_count=len(events),
        ground_truth_row_count=len(labels),
        earliest_event_time=earliest,
        latest_event_time=latest,
        artifacts=artifact_entries,
        canonical_schema_fingerprint=_schema_fingerprint(),
        content_fingerprint=content_fingerprint,
        config_fingerprint=config_fingerprint,
        validation_status=validation_status,
        created_at=datetime.now(UTC).isoformat(),
        reproducibility=reproducibility,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_dataset(directory: Path) -> VerificationResult:
    """Run 10 integrity and safety checks on a published dataset directory.

    Parameters
    ----------
    directory:
        Path to the published dataset directory containing ``manifest.json``
        and artifact files.

    Returns
    -------
    VerificationResult
        All check results.  ``passed`` is ``True`` only when every check passes.

    Notes
    -----
    This function never raises; all failures are captured as ``VerificationCheck``
    results with ``passed=False``.
    """
    checks: list[VerificationCheck] = []
    manifest_path = directory / "manifest.json"

    # -----------------------------------------------------------------------
    # Check 1: Manifest readable and parseable.
    # -----------------------------------------------------------------------
    try:
        manifest = DatasetManifest.from_json_file(manifest_path)
        checks.append(
            VerificationCheck(
                name="MANIFEST_READABLE",
                passed=True,
                message="Manifest loaded and validated successfully",
            )
        )
    except Exception as exc:
        checks.append(
            VerificationCheck(
                name="MANIFEST_READABLE",
                passed=False,
                message=f"Cannot read or parse manifest ({type(exc).__name__})",
            )
        )
        return VerificationResult(passed=False, checks=tuple(checks), manifest=None)

    # -----------------------------------------------------------------------
    # Check 2: Manifest version recognized.
    # -----------------------------------------------------------------------
    version_ok = manifest.manifest_version in _KNOWN_MANIFEST_VERSIONS
    checks.append(
        VerificationCheck(
            name="MANIFEST_VERSION_RECOGNIZED",
            passed=version_ok,
            message=(
                "Manifest version is recognized"
                if version_ok
                else f"Unknown manifest_version: {manifest.manifest_version!r}"
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Check 3: Schema version matches current SCHEMA_VERSION.
    # -----------------------------------------------------------------------
    sv_ok = manifest.schema_version == SCHEMA_VERSION
    checks.append(
        VerificationCheck(
            name="SCHEMA_VERSION_MATCH",
            passed=sv_ok,
            message=(
                f"schema_version matches current {SCHEMA_VERSION!r}"
                if sv_ok
                else f"schema_version {manifest.schema_version!r} != {SCHEMA_VERSION!r}"
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Check 4: No artifact path is absolute.
    # -----------------------------------------------------------------------
    absolute_paths = [
        e.relative_path
        for e in manifest.artifacts
        if Path(e.relative_path).is_absolute()
    ]
    paths_not_absolute = len(absolute_paths) == 0
    checks.append(
        VerificationCheck(
            name="ARTIFACT_PATHS_NOT_ABSOLUTE",
            passed=paths_not_absolute,
            message=(
                "All artifact paths are relative"
                if paths_not_absolute
                else f"{len(absolute_paths)} absolute artifact path(s) found"
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Check 5: No artifact path contains ".." components.
    # -----------------------------------------------------------------------
    dotdot_paths = [
        e.relative_path
        for e in manifest.artifacts
        if ".." in Path(e.relative_path).parts
    ]
    no_dotdot = len(dotdot_paths) == 0
    checks.append(
        VerificationCheck(
            name="ARTIFACT_PATHS_NO_DOTDOT",
            passed=no_dotdot,
            message=(
                "No artifact paths contain '..' components"
                if no_dotdot
                else f"{len(dotdot_paths)} artifact path(s) contain '..' components"
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Check 6: No symlink escape -- resolved paths stay within directory.
    # -----------------------------------------------------------------------
    escape_count = 0
    for entry in manifest.artifacts:
        safe, _ = _is_safe_relative_path(entry.relative_path, directory)
        if not safe:
            escape_count += 1
    no_escape = escape_count == 0
    checks.append(
        VerificationCheck(
            name="NO_SYMLINK_ESCAPE",
            passed=no_escape,
            message=(
                "All artifact paths resolve within the dataset directory"
                if no_escape
                else f"{escape_count} artifact path(s) escape the dataset directory"
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Check 7: All listed artifact files exist.
    # -----------------------------------------------------------------------
    missing_files = [
        entry.relative_path
        for entry in manifest.artifacts
        if not (directory / entry.relative_path).exists()
    ]
    files_exist = len(missing_files) == 0
    checks.append(
        VerificationCheck(
            name="ARTIFACT_FILES_EXIST",
            passed=files_exist,
            message=(
                f"All {len(manifest.artifacts)} artifact file(s) exist"
                if files_exist
                else f"{len(missing_files)} artifact file(s) missing"
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Check 8: SHA-256 checksums match.
    # -----------------------------------------------------------------------
    checksum_failures = 0
    if files_exist:  # only check if files exist
        for entry in manifest.artifacts:
            actual = _sha256_file(directory / entry.relative_path)
            if actual != entry.sha256:
                checksum_failures += 1
    checksums_ok = checksum_failures == 0 and files_exist
    checks.append(
        VerificationCheck(
            name="CHECKSUMS_MATCH",
            passed=checksums_ok,
            message=(
                "All artifact checksums match"
                if checksums_ok
                else (
                    f"{checksum_failures} checksum mismatch(es)"
                    if files_exist
                    else "Skipped (files missing)"
                )
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Check 9: Row count in manifest matches actual events Parquet row count.
    # -----------------------------------------------------------------------
    events_entry = _find_events_artifact(manifest)
    row_count_ok = True
    row_count_msg = "No events.parquet artifact; row count check skipped"
    if events_entry is not None and files_exist:
        events_path = directory / events_entry.relative_path
        try:
            import pyarrow.parquet as pq

            meta = pq.read_metadata(events_path)
            actual_rows = meta.num_rows
            row_count_ok = actual_rows == manifest.row_count
            row_count_msg = (
                f"Row count matches ({manifest.row_count:,})"
                if row_count_ok
                else f"Row count mismatch: manifest={manifest.row_count}, actual={actual_rows}"
            )
        except Exception as exc:
            row_count_ok = False
            row_count_msg = f"Cannot read events Parquet ({type(exc).__name__})"
    checks.append(
        VerificationCheck(
            name="ROW_COUNT_MATCHES",
            passed=row_count_ok,
            message=row_count_msg,
        )
    )

    # -----------------------------------------------------------------------
    # Check 10: validation_status is not "invalid".
    # -----------------------------------------------------------------------
    status_ok = manifest.validation_status != "invalid"
    checks.append(
        VerificationCheck(
            name="VALIDATION_STATUS_NOT_INVALID",
            passed=status_ok,
            message=(
                f"Validation status is {manifest.validation_status!r}"
                if status_ok
                else "Dataset failed validation (status=invalid)"
            ),
        )
    )

    all_passed = all(c.passed for c in checks)
    return VerificationResult(
        passed=all_passed, checks=tuple(checks), manifest=manifest
    )
