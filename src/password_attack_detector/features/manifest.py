"""Reproducibility manifest for a published feature dataset.

The manifest is a **superset of the Phase 2 dataset manifest**, so the single
``verify_dataset`` implementation in ``data/manifest.py`` verifies feature
directories too: path safety, symlink escape, checksums, and row count all
come from there rather than being reimplemented.  Reimplementing path-safety
logic is exactly the kind of duplication that eventually diverges, and this
particular logic is security-relevant.

Every path recorded here is relative to the dataset directory, and no
identifier, coordinate, or absolute path ever appears.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from password_attack_detector.data.manifest import (
    ArtifactEntry,
    VerificationResult,
    _get_reproducibility,
    _sha256_file,
    verify_dataset,
)
from password_attack_detector.features.catalog import FeatureCatalog
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
)
from password_attack_detector.features.serialization import (
    MANIFEST_FILE,
    PUBLISHED_FILES,
    SNAPSHOTS_FILE,
)

__all__ = [
    "FEATURE_MANIFEST_VERSION",
    "FeatureDatasetManifest",
    "build_feature_manifest",
    "verify_feature_dataset",
]

FEATURE_MANIFEST_VERSION: str = "1.0.0"

#: Namespace for deterministic feature-dataset identifiers.
_NS_FEATURE_DATASET = uuid5(NAMESPACE_URL, "password-attack-detector/features")


@dataclass(frozen=True)
class FeatureDatasetManifest:
    """Machine-readable description of one published feature dataset."""

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
    reproducibility: Any
    primary_artifact: str

    # Feature-specific additions.
    feature_schema_version: str
    feature_catalog_fingerprint: str
    feature_count: int
    source_dataset_fingerprint: str
    baseline_fingerprint: str | None
    split_config_fingerprint: str
    validation_result: dict[str, Any] | None
    leakage_audit_result: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping.

        The Phase 2 fields come first and keep their exact names, which is
        what lets ``verify_dataset`` read this file unchanged.
        """
        return {
            "manifest_version": self.manifest_version,
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "row_count": self.row_count,
            "ground_truth_row_count": self.ground_truth_row_count,
            "earliest_event_time": self.earliest_event_time,
            "latest_event_time": self.latest_event_time,
            "artifacts": [
                {"relative_path": a.relative_path, "sha256": a.sha256}
                for a in self.artifacts
            ],
            "canonical_schema_fingerprint": self.canonical_schema_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "validation_status": self.validation_status,
            "created_at": self.created_at,
            "reproducibility": (
                self.reproducibility.model_dump()
                if hasattr(self.reproducibility, "model_dump")
                else self.reproducibility
            ),
            "primary_artifact": self.primary_artifact,
            "feature_schema_version": self.feature_schema_version,
            "feature_catalog_fingerprint": self.feature_catalog_fingerprint,
            "feature_count": self.feature_count,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "baseline_fingerprint": self.baseline_fingerprint,
            "split_config_fingerprint": self.split_config_fingerprint,
            "validation_result": self.validation_result,
            "leakage_audit_result": self.leakage_audit_result,
        }


def build_feature_manifest(
    *,
    staging_dir: Path,
    catalog: FeatureCatalog,
    config: FeatureConfig,
    feature_fingerprint: str,
    source_dataset_fingerprint: str,
    baseline_fingerprint: str | None,
    row_count: int,
    label_count: int,
    earliest_event_time: datetime | None,
    latest_event_time: datetime | None,
    validation_status: str,
    validation_result: dict[str, Any] | None = None,
    leakage_audit_result: dict[str, Any] | None = None,
    artifact_names: Sequence[str] = PUBLISHED_FILES,
) -> FeatureDatasetManifest:
    """Build a manifest describing the artifacts staged in *staging_dir*.

    Checksums are taken from the staged files, before promotion, so the
    manifest describes exactly the bytes that are about to be published.
    """
    artifacts = [
        ArtifactEntry(relative_path=name, sha256=_sha256_file(staging_dir / name))
        for name in artifact_names
        if (staging_dir / name).exists()
    ]

    return FeatureDatasetManifest(
        manifest_version=FEATURE_MANIFEST_VERSION,
        dataset_id=str(uuid5(_NS_FEATURE_DATASET, feature_fingerprint)),
        schema_version=config.source_schema_version,
        source_type="features",
        row_count=row_count,
        ground_truth_row_count=label_count,
        earliest_event_time=(
            None
            if earliest_event_time is None
            else earliest_event_time.astimezone(UTC).isoformat()
        ),
        latest_event_time=(
            None
            if latest_event_time is None
            else latest_event_time.astimezone(UTC).isoformat()
        ),
        artifacts=artifacts,
        canonical_schema_fingerprint=catalog.fingerprint(),
        content_fingerprint=feature_fingerprint,
        config_fingerprint=config.fingerprint(),
        validation_status=validation_status,
        created_at=datetime.now(UTC).isoformat(),
        reproducibility=_get_reproducibility(generator_version=None, seed=None),
        primary_artifact=SNAPSHOTS_FILE,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_catalog_fingerprint=catalog.fingerprint(),
        feature_count=len(catalog),
        source_dataset_fingerprint=source_dataset_fingerprint,
        baseline_fingerprint=baseline_fingerprint,
        split_config_fingerprint=config.split.fingerprint(),
        validation_result=validation_result,
        leakage_audit_result=leakage_audit_result,
    )


def verify_feature_dataset(directory: Path) -> VerificationResult:
    """Verify a published feature dataset directory.

    Delegates to the Phase 2 verifier, which reads ``primary_artifact`` from
    the manifest and therefore checks the feature snapshot table's row count
    rather than assuming an events table.
    """
    return verify_dataset(directory, manifest_name=MANIFEST_FILE)
