"""Parquet/JSONL serialization and staged-rollback dataset publishing.

Public API
----------
- ``CANONICAL_EVENT_COLUMNS``   — fixed column ordering for AuthEvent Parquet
- ``CANONICAL_GT_COLUMNS``      — fixed column ordering for GroundTruthLabel Parquet
- ``compute_events_fingerprint`` — order-independent SHA-256 content hash
- ``write_events_parquet``       — write canonical events to Parquet
- ``write_labels_parquet``       — write ground-truth labels to Parquet
- ``write_events_jsonl``         — write raw events as newline-delimited JSON
- ``PublishedDataset``           — result of a successful ``DatasetPublisher.publish()``
- ``DatasetPublisher``           — 10-step staged publication with rollback

Staged-publication protocol
----------------------------
1.  Validate: non-empty events (raises ``DataValidationError`` before any I/O).
2.  Compute content fingerprint.
3.  Create staging directory via ``tempfile.mkdtemp`` on the same filesystem
    as the output directory.
4.  Write artifacts to staging.
5.  Check overwrite policy; raise if output exists and ``overwrite=False``.
6.  Backup existing published files when overwriting.
7.  Promote staged artifacts to output (non-manifest files first).
8.  Promote manifest **last**.
9.  On failure: restore from backup (overwrite) or remove partial artifacts
    (first-time), then re-raise.
10. Clean staging and backup on all exit paths.

Content fingerprint
-------------------
``compute_events_fingerprint`` sorts events by ``str(event_id)`` and hashes
a stable JSON representation using ``CANONICAL_EVENT_COLUMNS`` order with
UTC ISO-8601 timestamps and enum string values.  The hash is independent of
row ordering and Parquet encoding.
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
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from password_attack_detector.data.manifest import build_synthetic_manifest
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.data.validation import DatasetValidator
from password_attack_detector.exceptions import DataValidationError

if TYPE_CHECKING:
    from password_attack_detector.data.synthetic.generator import GenerationResult

__all__ = [
    "CANONICAL_EVENT_COLUMNS",
    "CANONICAL_GT_COLUMNS",
    "DatasetPublisher",
    "PublishedDataset",
    "compute_events_fingerprint",
    "read_events_parquet",
    "write_events_jsonl",
    "write_events_parquet",
    "write_labels_parquet",
]

# ---------------------------------------------------------------------------
# Canonical column orderings
# ---------------------------------------------------------------------------

CANONICAL_EVENT_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "event_id",
    "event_time",
    "user_id",
    "source_id",
    "device_id",
    "session_id",
    "application_id",
    "authentication_method",
    "authentication_outcome",
    "failure_reason",
    "mfa_outcome",
    "country_code",
    "region_code",
    "coarse_latitude",
    "coarse_longitude",
    "user_agent_family",
    "operating_system_family",
    "client_type",
    "response_time_ms",
)

CANONICAL_GT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "campaign_id",
    "scenario",
    "malicious",
    "supervised_training_eligible",
    "generator_version",
    "scenario_variant",
    "campaign_stage",
)

# ---------------------------------------------------------------------------
# PyArrow schemas
# ---------------------------------------------------------------------------

_EVENT_PA_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("application_id", pa.string(), nullable=False),
        pa.field("authentication_method", pa.string(), nullable=False),
        pa.field("authentication_outcome", pa.string(), nullable=False),
        pa.field("failure_reason", pa.string(), nullable=True),
        pa.field("mfa_outcome", pa.string(), nullable=True),
        pa.field("country_code", pa.string(), nullable=True),
        pa.field("region_code", pa.string(), nullable=True),
        pa.field("coarse_latitude", pa.float64(), nullable=True),
        pa.field("coarse_longitude", pa.float64(), nullable=True),
        pa.field("user_agent_family", pa.string(), nullable=True),
        pa.field("operating_system_family", pa.string(), nullable=True),
        pa.field("client_type", pa.string(), nullable=True),
        pa.field("response_time_ms", pa.int64(), nullable=True),
    ]
)

_GT_PA_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("campaign_id", pa.string(), nullable=False),
        pa.field("scenario", pa.string(), nullable=False),
        pa.field("malicious", pa.bool_(), nullable=False),
        pa.field("supervised_training_eligible", pa.bool_(), nullable=False),
        pa.field("generator_version", pa.string(), nullable=False),
        pa.field("scenario_variant", pa.string(), nullable=True),
        pa.field("campaign_stage", pa.string(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Value serialization helpers
# ---------------------------------------------------------------------------


def _event_value(col: str, event: AuthEvent) -> Any:
    """Return a Parquet-compatible value for *col* from *event*."""
    val: Any = getattr(event, col)
    if isinstance(val, UUID):
        return str(val)
    if isinstance(val, datetime):
        return val  # PyArrow accepts tz-aware datetime directly
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val  # preserve numeric/bool types for typed PA columns
    return str(val)  # StrEnum serializes to its string value


def _gt_value(col: str, label: GroundTruthLabel) -> Any:
    val: Any = getattr(label, col)
    if isinstance(val, UUID):
        return str(val)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val)


def _event_to_fingerprint_row(event: AuthEvent) -> dict[str, Any]:
    """Return a stable JSON-serialisable dict for fingerprinting."""
    row: dict[str, Any] = {}
    for col in CANONICAL_EVENT_COLUMNS:
        val: Any = getattr(event, col)
        if isinstance(val, UUID):
            row[col] = str(val)
        elif isinstance(val, datetime):
            row[col] = val.isoformat()
        elif val is None:
            row[col] = None
        else:
            row[col] = str(val)
    return row


# ---------------------------------------------------------------------------
# Public serialization functions
# ---------------------------------------------------------------------------


def compute_events_fingerprint(events: Sequence[AuthEvent]) -> str:
    """Return a SHA-256 hex content fingerprint for *events*.

    The fingerprint is:
    - **deterministic**: the same set of events always yields the same hash.
    - **order-independent**: events are sorted by ``str(event.event_id)``
      before hashing.
    - **independent of Parquet bytes**: it is computed from the logical
      content, not from the serialised file.

    Parameters
    ----------
    events:
        Sequence of ``AuthEvent`` objects to fingerprint.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.
    """
    sorted_events = sorted(events, key=lambda e: str(e.event_id))
    rows = [_event_to_fingerprint_row(e) for e in sorted_events]
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_events_parquet(path: Path) -> list[AuthEvent]:
    """Read a canonical events Parquet file and return ``AuthEvent`` objects.

    Parameters
    ----------
    path:
        Path to a Parquet file written by :func:`write_events_parquet`.

    Returns
    -------
    list[AuthEvent]
        Validated events in row order.

    Raises
    ------
    DataValidationError
        When the file cannot be read or a row fails schema validation.
    """
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise DataValidationError(
            f"Cannot read events Parquet ({type(exc).__name__})"
        ) from exc

    events: list[AuthEvent] = []
    for row in table.to_pylist():
        try:
            events.append(AuthEvent.model_validate(row))
        except Exception as exc:
            raise DataValidationError(
                f"Row validation failed ({type(exc).__name__})"
            ) from exc
    return events


def write_events_parquet(events: Sequence[AuthEvent], path: Path) -> None:
    """Write canonical events to a Parquet file at *path*.

    Columns appear in ``CANONICAL_EVENT_COLUMNS`` order with UTC timezone
    preservation.  Enum values are stored as their string values.

    Parameters
    ----------
    events:
        Events to serialise.
    path:
        Destination file path (parent directory must exist).
    """
    data: dict[str, list[Any]] = {col: [] for col in CANONICAL_EVENT_COLUMNS}
    for event in events:
        for col in CANONICAL_EVENT_COLUMNS:
            data[col].append(_event_value(col, event))
    table = pa.Table.from_pydict(data, schema=_EVENT_PA_SCHEMA)
    pq.write_table(table, path)


def write_labels_parquet(labels: Sequence[GroundTruthLabel], path: Path) -> None:
    """Write ground-truth labels to a Parquet file at *path*.

    Columns appear in ``CANONICAL_GT_COLUMNS`` order.
    """
    data: dict[str, list[Any]] = {col: [] for col in CANONICAL_GT_COLUMNS}
    for label in labels:
        for col in CANONICAL_GT_COLUMNS:
            data[col].append(_gt_value(col, label))
    table = pa.Table.from_pydict(data, schema=_GT_PA_SCHEMA)
    pq.write_table(table, path)


def write_events_jsonl(events: Sequence[AuthEvent], path: Path) -> None:
    """Write events as newline-delimited JSON (JSONL) to *path*.

    Each line is a JSON object with columns in ``CANONICAL_EVENT_COLUMNS``
    order.  Timestamps are ISO-8601 UTC strings; UUIDs are hyphenated strings.
    """
    lines: list[str] = []
    for event in events:
        row: dict[str, Any] = {}
        for col in CANONICAL_EVENT_COLUMNS:
            val: Any = getattr(event, col)
            if isinstance(val, UUID):
                row[col] = str(val)
            elif isinstance(val, datetime):
                row[col] = val.isoformat()
            elif val is None:
                row[col] = None
            else:
                row[col] = str(val)
        lines.append(json.dumps(row, ensure_ascii=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# DatasetPublisher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedDataset:
    """Paths and metadata for a successfully published dataset.

    All file paths are under ``output_dir`` as passed to ``DatasetPublisher``.
    """

    events_path: Path
    labels_path: Path
    jsonl_path: Path
    manifest_path: Path
    content_fingerprint: str
    num_events: int
    num_labels: int


class DatasetPublisher:
    """Publish a ``GenerationResult`` to disk with staged-rollback safety.

    The publication protocol is:

    1. Validate non-empty events (raises before any I/O).
    2. Compute content fingerprint.
    3. Create staging directory (same filesystem as output).
    4. Write artifacts to staging.
    5. Check overwrite policy.
    6. Backup existing files when overwriting.
    7. Promote staged non-manifest artifacts.
    8. Promote manifest **last**.
    9. On failure: restore backup (overwrite) or clean partial (first-time).
    10. Clean staging and backup on all exit paths.

    Parameters
    ----------
    output_dir:
        Directory where published artifacts are written.
    overwrite:
        When ``True``, existing artifacts are backed up and replaced.
        When ``False`` (default), raises ``DataValidationError`` if output
        already exists.
    """

    _EVENTS_FILE = "events.parquet"
    _LABELS_FILE = "labels.parquet"
    _JSONL_FILE = "events.jsonl"
    _MANIFEST_FILE = "manifest.json"

    def __init__(self, output_dir: Path, *, overwrite: bool = False) -> None:
        self._output_dir = output_dir
        self._overwrite = overwrite

    def publish(self, result: GenerationResult) -> PublishedDataset:
        """Execute the staged publication protocol.

        Parameters
        ----------
        result:
            Completed generation result.

        Returns
        -------
        PublishedDataset
            Paths and metadata for the published dataset.

        Raises
        ------
        DataValidationError
            When events are empty or output exists and ``overwrite=False``.
        """
        events = result.events
        labels = result.labels

        # Step 1: Validate non-empty before any I/O.
        if not events:
            raise DataValidationError("Cannot publish an empty events collection")

        # Step 2: Compute content fingerprint.
        content_fp = compute_events_fingerprint(events)

        # Step 3: Resolve output paths and create output directory.
        out = self._output_dir
        out.mkdir(parents=True, exist_ok=True)

        events_path = out / self._EVENTS_FILE
        labels_path = out / self._LABELS_FILE
        jsonl_path = out / self._JSONL_FILE
        manifest_path = out / self._MANIFEST_FILE

        existing = [
            p
            for p in (events_path, labels_path, jsonl_path, manifest_path)
            if p.exists()
        ]

        # Step 5 (checked before staging): raise on overwrite=False.
        if existing and not self._overwrite:
            raise DataValidationError(
                f"Output already exists in {out!r}; set overwrite=True to replace"
            )

        # Step 3 (cont.): Create staging directory on the same filesystem.
        staging_dir = Path(tempfile.mkdtemp(dir=out.parent, prefix="_pad_stage_"))
        backup_dir: Path | None = None
        promoted: list[Path] = []

        try:
            # Step 4: Write all artifacts to staging.
            write_events_parquet(events, staging_dir / self._EVENTS_FILE)
            write_labels_parquet(labels, staging_dir / self._LABELS_FILE)
            write_events_jsonl(events, staging_dir / self._JSONL_FILE)

            manifest = _build_manifest(
                result=result,
                content_fingerprint=content_fp,
                staging_dir=staging_dir,
                filenames=(self._EVENTS_FILE, self._LABELS_FILE, self._JSONL_FILE),
            )
            (staging_dir / self._MANIFEST_FILE).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            # Step 6: Backup existing files when overwriting.
            if existing:
                backup_dir = Path(
                    tempfile.mkdtemp(dir=out.parent, prefix="_pad_backup_")
                )
                for f in existing:
                    shutil.copy2(f, backup_dir / f.name)

            # Steps 7-8: Promote staged artifacts; manifest is last.
            try:
                for fname in (self._EVENTS_FILE, self._LABELS_FILE, self._JSONL_FILE):
                    dest = out / fname
                    shutil.move(str(staging_dir / fname), str(dest))
                    promoted.append(dest)

                # Manifest promoted last.
                shutil.move(str(staging_dir / self._MANIFEST_FILE), str(manifest_path))
                promoted.append(manifest_path)

            except Exception:
                # Step 9: Rollback.
                if backup_dir is not None:
                    # Overwrite case: restore from backup.
                    for p in promoted:
                        p.unlink(missing_ok=True)
                    for src in backup_dir.iterdir():
                        shutil.copy2(src, out / src.name)
                else:
                    # First-time case: remove any partially promoted files.
                    for p in promoted:
                        p.unlink(missing_ok=True)
                raise

        finally:
            # Step 10: Clean staging and backup on all exit paths.
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            if backup_dir is not None and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

        return PublishedDataset(
            events_path=events_path,
            labels_path=labels_path,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            content_fingerprint=content_fp,
            num_events=len(events),
            num_labels=len(labels),
        )


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(
    *,
    result: GenerationResult,
    content_fingerprint: str,
    staging_dir: Path,
    filenames: tuple[str, ...],
) -> dict[str, Any]:
    """Build a ``DatasetManifest`` dict for the staged publish.

    Validates the staged events Parquet to capture validation_status, then
    delegates to ``build_synthetic_manifest`` for the full manifest model.
    """
    # Locate the staged events Parquet and validate it to capture status.
    events_fname = next(
        (f for f in filenames if Path(f).name == "events.parquet"), None
    )
    if events_fname is not None:
        validator = DatasetValidator()
        val_result = validator.validate_parquet(staging_dir / events_fname)
        validation_status = str(val_result.status)
    else:
        validation_status = "valid"

    manifest = build_synthetic_manifest(
        events=result.events,
        labels=result.labels,
        config=result.config,
        config_fingerprint=result.config_fingerprint,
        content_fingerprint=content_fingerprint,
        staging_dir=staging_dir,
        artifact_filenames=filenames,
        validation_status=validation_status,
    )
    return manifest.to_dict()
