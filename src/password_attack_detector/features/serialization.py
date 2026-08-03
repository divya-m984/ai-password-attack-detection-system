"""Parquet serialization for the three feature tables, with staged publication.

Tables are built **directly in PyArrow** against a schema derived from the
catalog, rather than round-tripping through pandas.  That choice buys three
things:

* nullability becomes a declared, machine-checkable property per field;
* column order comes from :meth:`FeatureCatalog.column_order` and cannot drift;
* there is no ``NaN``-versus-``pd.NA`` ambiguity to poison the exact-equality
  tests the engine's guarantees rest on.  ``NaN`` cannot express a missing
  integer, and ``NaN != NaN`` breaks frame comparison outright.

:class:`FeaturePublisher` follows the same staged-rollback protocol as
``DatasetPublisher`` in the data package: stage everything to a temporary
directory, back up what is there, promote the data files, and promote the
manifest **last** so its presence always signals a coherent dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from password_attack_detector.data.schemas import GroundTruthLabel
from password_attack_detector.exceptions import (
    DataValidationError,
    FeatureComputationError,
)
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    FeatureCatalog,
    FeatureDType,
)
from password_attack_detector.features.engine import FeatureFrame
from password_attack_detector.features.splitting import SplitAssignment

__all__ = [
    "FEATURE_LABEL_COLUMNS",
    "FEATURE_SPLIT_COLUMNS",
    "LABELS_FILE",
    "MANIFEST_FILE",
    "PUBLISHED_FILES",
    "SNAPSHOTS_FILE",
    "SPLITS_FILE",
    "FeaturePublisher",
    "PublishedFeatureDataset",
    "compute_features_fingerprint",
    "feature_arrow_schema",
    "labels_for_events",
    "read_feature_snapshots",
    "read_ground_truth_labels",
    "write_feature_labels",
    "write_feature_snapshots",
    "write_feature_splits",
]

SNAPSHOTS_FILE: str = "feature_snapshots.parquet"
LABELS_FILE: str = "feature_labels.parquet"
SPLITS_FILE: str = "feature_splits.parquet"
MANIFEST_FILE: str = "feature_manifest.json"

#: Data files promoted before the manifest, in a fixed order.
PUBLISHED_FILES: tuple[str, ...] = (SNAPSHOTS_FILE, LABELS_FILE, SPLITS_FILE)

#: The label table.  Ground truth lives here and nowhere near the features.
#: ``campaign_id`` is deliberately absent: the splitter reads it internally for
#: group isolation, but it is never published beside a model input.
FEATURE_LABEL_COLUMNS: tuple[str, ...] = (
    "event_id",
    "attack_class",
    "malicious",
    "supervised_training_eligible",
)

#: The split-assignment table.  Never joined into the feature matrix.
FEATURE_SPLIT_COLUMNS: tuple[str, ...] = ("event_id", "split", "exclusion_reason")


def _arrow_type(dtype: FeatureDType) -> Any:
    """Map a catalog dtype onto its Arrow type."""
    import pyarrow as pa

    if dtype is FeatureDType.INT64:
        return pa.int64()
    if dtype is FeatureDType.FLOAT64:
        return pa.float64()
    if dtype is FeatureDType.BOOL:
        return pa.bool_()
    if dtype is FeatureDType.TIMESTAMP:
        return pa.timestamp("us", tz="UTC")
    return pa.string()


def feature_arrow_schema(catalog: FeatureCatalog) -> Any:
    """Build the Arrow schema for the feature snapshot table.

    Derived entirely from the catalog, so the written file's types and
    nullability are the declared ones by construction.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field(spec.name, _arrow_type(spec.dtype), nullable=spec.nullable)
            for spec in catalog.specs
        ]
    )


def _labels_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("attack_class", pa.string(), nullable=False),
            pa.field("malicious", pa.bool_(), nullable=False),
            pa.field("supervised_training_eligible", pa.bool_(), nullable=False),
        ]
    )


def _splits_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("exclusion_reason", pa.string(), nullable=True),
        ]
    )


def write_feature_snapshots(frame: FeatureFrame, path: Path) -> None:
    """Write the feature snapshot table.

    Raises:
        FeatureComputationError: if a row's keys do not match the catalog.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = feature_arrow_schema(frame.catalog)
    columns = frame.catalog.column_order()

    if frame.rows:
        first = frame.rows[0]
        if tuple(first) != columns:
            missing = [name for name in columns if name not in first]
            extra = [name for name in first if name not in columns]
            raise FeatureComputationError(
                f"Feature rows do not match the catalog: {len(missing)} missing "
                f"and {len(extra)} undeclared column(s)"
            )

    data = {name: [row[name] for row in frame.rows] for name in columns}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data, schema=schema), path)


def write_feature_labels(labels: Sequence[GroundTruthLabel], path: Path) -> None:
    """Write the label table, keyed by ``event_id``.

    Only the four supervised-learning fields are published.  Campaign
    metadata stays out of every published table.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {
        "event_id": [str(label.event_id) for label in labels],
        "attack_class": [str(label.scenario) for label in labels],
        "malicious": [label.malicious for label in labels],
        "supervised_training_eligible": [
            label.supervised_training_eligible for label in labels
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data, schema=_labels_schema()), path)


def write_feature_splits(assignments: Sequence[SplitAssignment], path: Path) -> None:
    """Write the split-assignment table, keyed by ``event_id``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {
        "event_id": [a.event_id for a in assignments],
        "split": [str(a.split) for a in assignments],
        "exclusion_reason": [a.exclusion_reason for a in assignments],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data, schema=_splits_schema()), path)


def read_ground_truth_labels(path: Path) -> list[GroundTruthLabel]:
    """Read a Phase 2 ground-truth label table back into model objects.

    Raises:
        DataValidationError: if the file cannot be read or does not conform.
    """
    import pyarrow.parquet as pq

    try:
        records = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise DataValidationError(
            f"Cannot read ground-truth labels: {type(exc).__name__}"
        ) from None

    try:
        return [GroundTruthLabel(**record) for record in records]
    except Exception as exc:
        raise DataValidationError(
            f"Ground-truth labels do not conform to the schema: {type(exc).__name__}"
        ) from None


def read_feature_snapshots(path: Path) -> Any:
    """Read the feature snapshot table into a pandas DataFrame.

    Uses the pyarrow dtype backend so nulls stay ``pd.NA`` rather than
    collapsing into ``NaN``, which would erase the integer/float distinction
    the catalog declares.

    Raises:
        DataValidationError: if the file cannot be read.
    """
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise DataValidationError(
            f"Cannot read feature snapshots: {type(exc).__name__}"
        ) from None
    return table.to_pandas(types_mapper=None)


def compute_features_fingerprint(frame: FeatureFrame) -> str:
    """Return an order-independent digest of the feature table's content.

    Rows are sorted by anchor identifier and rendered through a canonical
    JSON encoding, so the digest depends on the *logical* content rather than
    on row order or on Parquet's physical layout (which varies with pyarrow
    version and compression settings).
    """
    columns = frame.catalog.column_order()
    rows = [
        {name: _canonical_value(row[name]) for name in columns} for row in frame.rows
    ]
    rows.sort(key=lambda row: str(row[ANCHOR_EVENT_ID]))
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _canonical_value(value: Any) -> Any:
    """Render one cell into a JSON-stable, order-independent representation."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return f"{round(value, 9):.9f}"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def labels_for_events(
    event_ids: Sequence[str], labels: Sequence[GroundTruthLabel]
) -> list[GroundTruthLabel]:
    """Return the labels for *event_ids*, in that order.

    Raises:
        DataValidationError: if any event has no label.
    """
    by_id = {str(label.event_id): label for label in labels}
    missing = [key for key in event_ids if key not in by_id]
    if missing:
        raise DataValidationError(
            f"{len(missing)} feature row(s) have no matching ground-truth label"
        )
    return [by_id[key] for key in event_ids]


@dataclass(frozen=True)
class PublishedFeatureDataset:
    """Paths and fingerprints of a published feature dataset."""

    directory: Path
    snapshots_path: Path
    labels_path: Path
    splits_path: Path
    manifest_path: Path
    feature_fingerprint: str
    row_count: int
    feature_count: int


class FeaturePublisher:
    """Publishes the three feature tables and a manifest atomically enough.

    Follows ``DatasetPublisher``'s protocol: stage, back up, promote data,
    promote the manifest last, and roll back on any failure.  The manifest
    going last is what makes a directory containing one meaningful -- its
    presence means every other artifact is already in place.
    """

    def __init__(self, output_dir: Path, *, overwrite: bool = False) -> None:
        self._output_dir = Path(output_dir)
        self._overwrite = overwrite

    def publish(
        self,
        frame: FeatureFrame,
        labels: Sequence[GroundTruthLabel],
        assignments: Sequence[SplitAssignment],
        manifest: dict[str, Any],
    ) -> PublishedFeatureDataset:
        """Write every artifact, or leave the directory as it was.

        Raises:
            DataValidationError: if the frame is empty, or if artifacts exist
                and ``overwrite`` was not requested.
        """
        if not frame.rows:
            raise DataValidationError("Refusing to publish an empty feature dataset")

        output = self._output_dir
        targets = [output / name for name in (*PUBLISHED_FILES, MANIFEST_FILE)]

        if not self._overwrite and any(path.exists() for path in targets):
            raise DataValidationError(
                f"Feature artifacts already exist in {output.name!r}; pass "
                f"--force to replace them"
            )

        output.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix="_pad_feat_stage_"))
        backup: Path | None = None
        promoted: list[Path] = []

        try:
            write_feature_snapshots(frame, staging / SNAPSHOTS_FILE)
            write_feature_labels(labels, staging / LABELS_FILE)
            write_feature_splits(assignments, staging / SPLITS_FILE)
            (staging / MANIFEST_FILE).write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

            existing = [path for path in targets if path.exists()]
            if existing:
                backup = Path(
                    tempfile.mkdtemp(dir=output.parent, prefix="_pad_feat_backup_")
                )
                for path in existing:
                    shutil.copy2(path, backup / path.name)

            for name in PUBLISHED_FILES:
                destination = output / name
                shutil.move(str(staging / name), str(destination))
                promoted.append(destination)

            # The manifest goes last on purpose: its presence is the signal
            # that every other artifact in this directory is complete.
            manifest_path = output / MANIFEST_FILE
            shutil.move(str(staging / MANIFEST_FILE), str(manifest_path))
            promoted.append(manifest_path)

        except Exception:
            for path in promoted:
                path.unlink(missing_ok=True)
            if backup is not None:
                for path in backup.iterdir():
                    shutil.copy2(path, output / path.name)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)

        return PublishedFeatureDataset(
            directory=output,
            snapshots_path=output / SNAPSHOTS_FILE,
            labels_path=output / LABELS_FILE,
            splits_path=output / SPLITS_FILE,
            manifest_path=output / MANIFEST_FILE,
            feature_fingerprint=compute_features_fingerprint(frame),
            row_count=len(frame.rows),
            feature_count=len(frame.catalog),
        )
