"""Parquet serialization and staged publication for the detection artifacts.

Tables are built **directly in PyArrow** against schemas declared here rather
than round-tripping through pandas, for the same reasons Phase 3 does it:
nullability becomes a declared, machine-checkable property, column order comes
from a module-level tuple and cannot drift, and there is no ``NaN``-versus-
``pd.NA`` ambiguity to poison the exact-equality tests every determinism
guarantee rests on.

Evidence and rule-identifier lists are stored as **canonical JSON strings**
rather than Arrow list columns.  A JSON string has one byte representation for
one logical value, which keeps the content fingerprint, the validator, and the
round-trip test exact; an Arrow list's physical layout varies with pyarrow
version and compression settings and would leak into any digest taken over it.

:class:`DetectionPublisher` follows the ``FeaturePublisher`` protocol exactly:
stage to a temporary directory, back up what is already there, promote the data
files, promote the manifest **last**, and roll back on any failure.  This is
**staged publication with rollback** -- no cross-directory filesystem atomicity
is claimed or available, and the manifest going last is what makes its presence
meaningful: it appears only once every other artifact is in place.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from password_attack_detector.detection.enums import AttackCategory
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    FiredDetection,
    RiskAssessment,
    SecurityAlert,
)
from password_attack_detector.exceptions import DataValidationError

__all__ = [
    "ALERTS_FILE",
    "ALERT_COLUMNS",
    "DETECTIONS_FILE",
    "DETECTION_COLUMNS",
    "EVALUATION_JSON_FILE",
    "EVALUATION_MD_FILE",
    "FLOAT_PRECISION",
    "MANIFEST_FILE",
    "PUBLISHED_FILES",
    "QUALITY_JSON_FILE",
    "QUALITY_MD_FILE",
    "RISK_COLUMNS",
    "RISK_FILE",
    "SCOPE_COLUMNS",
    "SCOPE_FILE",
    "DetectionPublisher",
    "PublishedDetectionSet",
    "compute_alert_fingerprint",
    "compute_detection_fingerprint",
    "compute_report_fingerprint",
    "compute_risk_fingerprint",
    "decode_evidence",
    "encode_evidence",
    "encode_string_list",
    "read_alerts",
    "read_fired_detections",
    "read_risk_assessments",
    "require_finite",
    "write_alerts",
    "write_fired_detections",
    "write_risk_assessments",
]

DETECTIONS_FILE: str = "rule_detections.parquet"
RISK_FILE: str = "risk_assessments.parquet"
ALERTS_FILE: str = "security_alerts.parquet"
SCOPE_FILE: str = "detection_entity_scope.parquet"
MANIFEST_FILE: str = "detection_manifest.json"
QUALITY_JSON_FILE: str = "detection_quality.json"
QUALITY_MD_FILE: str = "detection_quality.md"
EVALUATION_JSON_FILE: str = "rule_evaluation.json"
EVALUATION_MD_FILE: str = "rule_evaluation.md"

#: Data files promoted before the manifest, in a fixed order.
PUBLISHED_FILES: tuple[str, ...] = (DETECTIONS_FILE, RISK_FILE, ALERTS_FILE)

#: Decimal places every float is rounded to before it enters a fingerprint or a
#: serialized evidence payload.  Fixed rather than left to ``repr`` so two
#: machines that disagree in the last bit of a float still agree on the digest.
FLOAT_PRECISION: int = 9

#: Authoritative column order for the fired-detection table.  Nothing
#: downstream may reorder it; the validator checks written files against it.
DETECTION_COLUMNS: tuple[str, ...] = (
    "detection_schema_version",
    "detection_id",
    "anchor_event_id",
    "anchor_event_time",
    "rule_id",
    "rule_version",
    "rule_family",
    "attack_category",
    "severity",
    "signal_strength",
    "correlation_group",
    "evidence_json",
    "reason_codes_json",
)

#: Authoritative column order for the risk-assessment table.  One row per
#: evaluated anchor, including anchors where nothing fired: evaluation needs
#: the benign denominator, and a table restricted to firings cannot supply it.
RISK_COLUMNS: tuple[str, ...] = (
    "detection_schema_version",
    "anchor_event_id",
    "anchor_event_time",
    "risk_score",
    "severity",
    "primary_attack_category",
    "contributing_categories_json",
    "fired_rule_count",
    "fired_rule_ids_json",
    "top_evidence_json",
    "insufficient_data_count",
    "scoring_version",
    "configuration_fingerprint",
)

#: Authoritative column order for the alert table.  ``scope_value`` is the one
#: column in the whole detection layer that may carry a pseudonym.
ALERT_COLUMNS: tuple[str, ...] = (
    "detection_schema_version",
    "alert_id",
    "attack_category",
    "correlation_group",
    "grouping_mode",
    "scope_kind",
    "scope_value",
    "first_seen",
    "last_seen",
    "contributing_event_count",
    "contributing_rule_ids_json",
    "aggregate_risk_score",
    "peak_risk_score",
    "initial_severity",
    "current_severity",
    "escalation_count",
    "suppressed_event_count",
)

#: Column order for the optional entity-scope input table.
SCOPE_COLUMNS: tuple[str, ...] = ("anchor_event_id", "user_scope", "source_scope")


# ---------------------------------------------------------------------------
# Canonical JSON encoding
# ---------------------------------------------------------------------------


def require_finite(value: float) -> float:
    """Return *value* unchanged when it is finite.

    Raises:
        DataValidationError: if the value is ``NaN`` or infinite.  Neither has
            a canonical JSON form, and substituting one would make a digest
            agree where the data does not.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise DataValidationError(
            "Detection artifacts must not contain NaN or infinite values"
        )
    return value


def _canonical_number(value: float) -> float:
    """Round a finite float to the declared serialization precision.

    Applied to **text** representations only -- canonical evidence JSON and
    content fingerprints.  Parquet float columns keep their full ``float64``
    value, which is lossless, so a published number never silently loses
    precision on the way to disk.  Rounding exists so two machines that
    disagree in a float's last bit still agree on a digest.
    """
    return round(require_finite(value), FLOAT_PRECISION)


def _canonical_value(value: Any) -> Any:
    """Render one cell into a JSON-stable, order-independent representation."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return f"{_canonical_number(value):.{FLOAT_PRECISION}f}"
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def encode_evidence(items: Sequence[EvidenceItem]) -> str:
    """Serialize evidence to canonical JSON.

    Keys are sorted, list order is the emission order the rule fixed, floats
    are rounded to :data:`FLOAT_PRECISION`, and no Python ``repr`` is used
    anywhere.  Every value came from a trusted evidence template and has
    already been rejected by :class:`EvidenceItem` if it looked like an
    identifier -- this function adds no new content, only a stable encoding.
    """
    payload = [
        {
            "evidence_code": item.evidence_code,
            "feature_name": item.feature_name,
            "comparator": str(item.comparator),
            "observed_value": _canonical_value(item.observed_value),
            "threshold_value": _canonical_value(item.threshold_value),
            "unit": item.unit,
            "message": item.message,
        }
        for item in items
    ]
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def decode_evidence(encoded: str) -> tuple[EvidenceItem, ...]:
    """Rebuild evidence items from canonical JSON.

    Raises:
        DataValidationError: if the payload is not a JSON array of objects, or
            an item fails the evidence contract.  A stored payload that no
            longer validates is a corrupt artifact, not something to accept.
    """
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            f"Evidence payload is not valid JSON: {type(exc).__name__}"
        ) from None
    if not isinstance(payload, list):
        raise DataValidationError("Evidence payload must be a JSON array")

    items: list[EvidenceItem] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise DataValidationError("Each evidence entry must be a JSON object")
        try:
            items.append(
                EvidenceItem(
                    evidence_code=entry["evidence_code"],
                    feature_name=entry["feature_name"],
                    comparator=entry["comparator"],
                    observed_value=_decode_scalar(entry["observed_value"]),
                    threshold_value=_decode_scalar(entry.get("threshold_value")),
                    unit=entry.get("unit"),
                    message=entry["message"],
                )
            )
        except (KeyError, ValueError) as exc:
            raise DataValidationError(
                f"Stored evidence does not satisfy the evidence contract: "
                f"{type(exc).__name__}"
            ) from None
    return tuple(items)


def _decode_scalar(value: Any) -> Any:
    """Turn a canonical scalar back into its Python value."""
    if isinstance(value, str) and _looks_numeric(value):
        return float(value)
    return value


def _looks_numeric(value: str) -> bool:
    """Return whether *value* is a fixed-precision number this module wrote."""
    candidate = value[1:] if value.startswith("-") else value
    whole, _, fraction = candidate.partition(".")
    return (
        bool(whole)
        and whole.isdigit()
        and len(fraction) == FLOAT_PRECISION
        and fraction.isdigit()
    )


def encode_string_list(values: Sequence[str]) -> str:
    """Serialize a list of identifiers or codes to canonical JSON."""
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def decode_string_list(encoded: str) -> tuple[str, ...]:
    """Rebuild a string list from canonical JSON.

    Raises:
        DataValidationError: if the payload is not a JSON array of strings.
    """
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError:
        raise DataValidationError("Stored list column is not valid JSON") from None
    if not isinstance(payload, list) or any(
        not isinstance(item, str) for item in payload
    ):
        raise DataValidationError("Stored list column must be a JSON array of strings")
    return tuple(payload)


# ---------------------------------------------------------------------------
# Arrow schemas
# ---------------------------------------------------------------------------


def _timestamp() -> Any:
    import pyarrow as pa

    return pa.timestamp("us", tz="UTC")


def detection_arrow_schema() -> Any:
    """Return the Arrow schema for the fired-detection table."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("detection_schema_version", pa.string(), nullable=False),
            pa.field("detection_id", pa.string(), nullable=False),
            pa.field("anchor_event_id", pa.string(), nullable=False),
            pa.field("anchor_event_time", _timestamp(), nullable=False),
            pa.field("rule_id", pa.string(), nullable=False),
            pa.field("rule_version", pa.string(), nullable=False),
            pa.field("rule_family", pa.string(), nullable=False),
            pa.field("attack_category", pa.string(), nullable=False),
            pa.field("severity", pa.string(), nullable=False),
            pa.field("signal_strength", pa.float64(), nullable=False),
            pa.field("correlation_group", pa.string(), nullable=False),
            pa.field("evidence_json", pa.string(), nullable=False),
            pa.field("reason_codes_json", pa.string(), nullable=False),
        ]
    )


def risk_arrow_schema() -> Any:
    """Return the Arrow schema for the risk-assessment table."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("detection_schema_version", pa.string(), nullable=False),
            pa.field("anchor_event_id", pa.string(), nullable=False),
            pa.field("anchor_event_time", _timestamp(), nullable=False),
            pa.field("risk_score", pa.float64(), nullable=False),
            pa.field("severity", pa.string(), nullable=False),
            pa.field("primary_attack_category", pa.string(), nullable=True),
            pa.field("contributing_categories_json", pa.string(), nullable=False),
            pa.field("fired_rule_count", pa.int64(), nullable=False),
            pa.field("fired_rule_ids_json", pa.string(), nullable=False),
            pa.field("top_evidence_json", pa.string(), nullable=False),
            pa.field("insufficient_data_count", pa.int64(), nullable=False),
            pa.field("scoring_version", pa.string(), nullable=False),
            pa.field("configuration_fingerprint", pa.string(), nullable=False),
        ]
    )


def alert_arrow_schema() -> Any:
    """Return the Arrow schema for the alert table."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("detection_schema_version", pa.string(), nullable=False),
            pa.field("alert_id", pa.string(), nullable=False),
            pa.field("attack_category", pa.string(), nullable=False),
            pa.field("correlation_group", pa.string(), nullable=False),
            pa.field("grouping_mode", pa.string(), nullable=False),
            pa.field("scope_kind", pa.string(), nullable=False),
            # The one nullable-by-design column carrying a pseudonym.
            pa.field("scope_value", pa.string(), nullable=True),
            pa.field("first_seen", _timestamp(), nullable=False),
            pa.field("last_seen", _timestamp(), nullable=False),
            pa.field("contributing_event_count", pa.int64(), nullable=False),
            pa.field("contributing_rule_ids_json", pa.string(), nullable=False),
            pa.field("aggregate_risk_score", pa.float64(), nullable=False),
            pa.field("peak_risk_score", pa.float64(), nullable=False),
            pa.field("initial_severity", pa.string(), nullable=False),
            pa.field("current_severity", pa.string(), nullable=False),
            pa.field("escalation_count", pa.int64(), nullable=False),
            pa.field("suppressed_event_count", pa.int64(), nullable=False),
        ]
    )


# ---------------------------------------------------------------------------
# Row rendering
# ---------------------------------------------------------------------------


def detection_row(detection: FiredDetection) -> dict[str, Any]:
    """Render one fired detection into its published column mapping."""
    return {
        "detection_schema_version": detection.detection_schema_version,
        "detection_id": detection.detection_id,
        "anchor_event_id": detection.anchor_event_id,
        "anchor_event_time": detection.anchor_event_time.astimezone(UTC),
        "rule_id": detection.rule_id,
        "rule_version": detection.rule_version,
        "rule_family": str(detection.rule_family),
        "attack_category": str(detection.attack_category),
        "severity": str(detection.severity),
        "signal_strength": require_finite(detection.signal_strength),
        "correlation_group": str(detection.correlation_group),
        "evidence_json": encode_evidence(detection.evidence),
        "reason_codes_json": encode_string_list(detection.reason_codes),
    }


def risk_row(assessment: RiskAssessment) -> dict[str, Any]:
    """Render one risk assessment into its published column mapping."""
    return {
        "detection_schema_version": assessment.detection_schema_version,
        "anchor_event_id": assessment.anchor_event_id,
        "anchor_event_time": assessment.anchor_event_time.astimezone(UTC),
        "risk_score": require_finite(assessment.risk_score),
        "severity": str(assessment.severity),
        "primary_attack_category": (
            None
            if assessment.primary_attack_category is None
            else str(assessment.primary_attack_category)
        ),
        "contributing_categories_json": encode_string_list(
            [str(item) for item in assessment.contributing_categories]
        ),
        "fired_rule_count": assessment.fired_rule_count,
        "fired_rule_ids_json": encode_string_list(assessment.fired_rule_ids),
        "top_evidence_json": encode_evidence(assessment.top_evidence),
        "insufficient_data_count": assessment.insufficient_data_count,
        "scoring_version": assessment.scoring_version,
        "configuration_fingerprint": assessment.configuration_fingerprint,
    }


def alert_row(alert: SecurityAlert) -> dict[str, Any]:
    """Render one alert into its published column mapping."""
    return {
        "detection_schema_version": alert.detection_schema_version,
        "alert_id": alert.alert_id,
        "attack_category": str(alert.attack_category),
        "correlation_group": str(alert.correlation_group),
        "grouping_mode": str(alert.grouping_mode),
        "scope_kind": str(alert.scope_kind),
        "scope_value": alert.scope_value,
        "first_seen": alert.first_seen.astimezone(UTC),
        "last_seen": alert.last_seen.astimezone(UTC),
        "contributing_event_count": alert.contributing_event_count,
        "contributing_rule_ids_json": encode_string_list(alert.contributing_rule_ids),
        "aggregate_risk_score": require_finite(alert.aggregate_risk_score),
        "peak_risk_score": require_finite(alert.peak_risk_score),
        "initial_severity": str(alert.initial_severity),
        "current_severity": str(alert.current_severity),
        "escalation_count": alert.escalation_count,
        "suppressed_event_count": alert.suppressed_event_count,
    }


# ---------------------------------------------------------------------------
# Writers and readers
# ---------------------------------------------------------------------------


def _write_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    schema: Any,
    path: Path,
) -> None:
    """Write *rows* to *path* as Parquet in the declared column order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {name: [row[name] for row in rows] for name in columns}
    table = pa.Table.from_pydict(data, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")


def write_fired_detections(detections: Sequence[FiredDetection], path: Path) -> None:
    """Write the fired-detection table, sorted for stable file bytes."""
    rows = sorted(
        (detection_row(item) for item in detections),
        key=lambda row: (row["anchor_event_id"], row["rule_id"]),
    )
    _write_table(rows, DETECTION_COLUMNS, detection_arrow_schema(), path)


def write_risk_assessments(assessments: Sequence[RiskAssessment], path: Path) -> None:
    """Write the risk-assessment table, sorted for stable file bytes."""
    rows = sorted(
        (risk_row(item) for item in assessments),
        key=lambda row: row["anchor_event_id"],
    )
    _write_table(rows, RISK_COLUMNS, risk_arrow_schema(), path)


def write_alerts(alerts: Sequence[SecurityAlert], path: Path) -> None:
    """Write the alert table, sorted for stable file bytes."""
    rows = sorted((alert_row(item) for item in alerts), key=lambda row: row["alert_id"])
    _write_table(rows, ALERT_COLUMNS, alert_arrow_schema(), path)


def _read_table(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a Parquet file into its column order and row mappings.

    Raises:
        DataValidationError: if the file cannot be read.
    """
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise DataValidationError(
            f"Cannot read detection artifact: {type(exc).__name__}"
        ) from None
    columns = list(table.column_names)
    data = table.to_pydict()
    count = table.num_rows
    return columns, [
        {name: data[name][index] for name in columns} for index in range(count)
    ]


def read_fired_detections(path: Path) -> tuple[FiredDetection, ...]:
    """Read the fired-detection table back into typed models.

    Raises:
        DataValidationError: if a row does not satisfy the detection contract.
    """
    _, rows = _read_table(path)
    return tuple(
        FiredDetection(
            detection_schema_version=row["detection_schema_version"],
            detection_id=row["detection_id"],
            anchor_event_id=row["anchor_event_id"],
            anchor_event_time=row["anchor_event_time"],
            rule_id=row["rule_id"],
            rule_version=row["rule_version"],
            rule_family=row["rule_family"],
            attack_category=row["attack_category"],
            severity=row["severity"],
            signal_strength=row["signal_strength"],
            correlation_group=row["correlation_group"],
            evidence=decode_evidence(row["evidence_json"]),
            reason_codes=decode_string_list(row["reason_codes_json"]),
        )
        for row in rows
    )


def read_risk_assessments(path: Path) -> tuple[RiskAssessment, ...]:
    """Read the risk-assessment table back into typed models."""
    _, rows = _read_table(path)
    return tuple(
        RiskAssessment(
            detection_schema_version=row["detection_schema_version"],
            anchor_event_id=row["anchor_event_id"],
            anchor_event_time=row["anchor_event_time"],
            risk_score=row["risk_score"],
            severity=row["severity"],
            primary_attack_category=row["primary_attack_category"],
            contributing_categories=tuple(
                AttackCategory(value)
                for value in decode_string_list(row["contributing_categories_json"])
            ),
            fired_rule_count=row["fired_rule_count"],
            fired_rule_ids=decode_string_list(row["fired_rule_ids_json"]),
            top_evidence=decode_evidence(row["top_evidence_json"]),
            insufficient_data_count=row["insufficient_data_count"],
            scoring_version=row["scoring_version"],
            configuration_fingerprint=row["configuration_fingerprint"],
        )
        for row in rows
    )


def read_alerts(path: Path) -> tuple[SecurityAlert, ...]:
    """Read the alert table back into typed models."""
    _, rows = _read_table(path)
    return tuple(
        SecurityAlert(
            detection_schema_version=row["detection_schema_version"],
            alert_id=row["alert_id"],
            attack_category=row["attack_category"],
            correlation_group=row["correlation_group"],
            grouping_mode=row["grouping_mode"],
            scope_kind=row["scope_kind"],
            scope_value=row["scope_value"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            contributing_event_count=row["contributing_event_count"],
            contributing_rule_ids=decode_string_list(row["contributing_rule_ids_json"]),
            aggregate_risk_score=row["aggregate_risk_score"],
            peak_risk_score=row["peak_risk_score"],
            initial_severity=row["initial_severity"],
            current_severity=row["current_severity"],
            escalation_count=row["escalation_count"],
            suppressed_event_count=row["suppressed_event_count"],
        )
        for row in rows
    )


def read_table_columns(path: Path) -> list[str]:
    """Return the column names of a written detection artifact, in file order."""
    columns, _ = _read_table(path)
    return columns


# ---------------------------------------------------------------------------
# Content fingerprints
# ---------------------------------------------------------------------------


def _fingerprint_rows(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    sort_key: Sequence[str],
) -> str:
    """Digest logical rows, independently of their order.

    Rows are sorted by their semantic key and rendered through the canonical
    encoding, so the digest depends on the content and nothing else -- not the
    order rows arrived in, not Parquet's physical layout, not the path the file
    was written to, and not when it was written.
    """
    rendered = [{name: _canonical_value(row[name]) for name in columns} for row in rows]
    rendered.sort(key=lambda row: tuple(str(row[name]) for name in sort_key))
    canonical = json.dumps(rendered, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_detection_fingerprint(detections: Sequence[FiredDetection]) -> str:
    """Return the order-independent content digest of the detection set."""
    return _fingerprint_rows(
        [detection_row(item) for item in detections],
        DETECTION_COLUMNS,
        ("anchor_event_id", "rule_id"),
    )


def compute_risk_fingerprint(assessments: Sequence[RiskAssessment]) -> str:
    """Return the order-independent content digest of the assessment set."""
    return _fingerprint_rows(
        [risk_row(item) for item in assessments],
        RISK_COLUMNS,
        ("anchor_event_id",),
    )


def compute_alert_fingerprint(alerts: Sequence[SecurityAlert]) -> str:
    """Return the order-independent content digest of the alert set."""
    return _fingerprint_rows(
        [alert_row(item) for item in alerts], ALERT_COLUMNS, ("alert_id",)
    )


def compute_report_fingerprint(
    report: Mapping[str, Any], *, excluded: Sequence[str] = ()
) -> str:
    """Return the digest of a report's semantic content.

    *excluded* names top-level keys that describe *when* or *where* a report
    was produced rather than what it says.  A creation timestamp must never
    enter a deterministic fingerprint, or two identical runs would disagree.
    """
    payload = {key: value for key, value in report.items() if key not in set(excluded)}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Staged publication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedDetectionSet:
    """Paths and fingerprints of a published detection artifact set."""

    directory: Path
    detections_path: Path
    risk_path: Path
    alerts_path: Path
    manifest_path: Path
    detection_fingerprint: str
    risk_fingerprint: str
    alert_fingerprint: str
    detection_row_count: int
    risk_row_count: int
    alert_row_count: int


class DetectionPublisher:
    """Publishes the three detection tables and a manifest, or nothing at all.

    **Staged publication with rollback.**  No cross-directory filesystem
    atomicity is claimed -- POSIX offers none, and pretending otherwise would
    be worse than documenting the limit.  What is guaranteed is that a failure
    leaves the directory as it was: a first-time publication removes whatever
    it promoted, and a failed overwrite restores every previous artifact and
    the previous manifest byte for byte.
    """

    def __init__(self, output_dir: Path, *, overwrite: bool = False) -> None:
        self._output_dir = Path(output_dir)
        self._overwrite = overwrite

    def publish(
        self,
        detections: Sequence[FiredDetection],
        assessments: Sequence[RiskAssessment],
        alerts: Sequence[SecurityAlert],
        manifest: Mapping[str, Any],
        *,
        reports: Mapping[str, str] | None = None,
    ) -> PublishedDetectionSet:
        """Write every artifact, or leave the directory exactly as it was.

        *reports* maps a file name to its rendered text; report files are
        staged and promoted alongside the tables and roll back with them.

        Raises:
            DataValidationError: if there are no assessments to publish, or if
                artifacts exist and ``overwrite`` was not requested.
        """
        if not assessments:
            raise DataValidationError(
                "Refusing to publish a detection set with no risk assessments"
            )

        output = self._output_dir
        report_names = tuple(sorted(reports or {}))
        data_names = (*PUBLISHED_FILES, *report_names)
        targets = [output / name for name in (*data_names, MANIFEST_FILE)]

        if not self._overwrite and any(path.exists() for path in targets):
            raise DataValidationError(
                f"Detection artifacts already exist in {output.name!r}; pass "
                f"--force to replace them"
            )

        output.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix="_pad_det_stage_"))
        backup: Path | None = None
        promoted: list[Path] = []

        try:
            write_fired_detections(detections, staging / DETECTIONS_FILE)
            write_risk_assessments(assessments, staging / RISK_FILE)
            write_alerts(alerts, staging / ALERTS_FILE)
            for name in report_names:
                (staging / name).write_text((reports or {})[name], encoding="utf-8")
            (staging / MANIFEST_FILE).write_text(
                json.dumps(dict(manifest), indent=2, sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )

            existing = [path for path in targets if path.exists()]
            if existing:
                backup = Path(
                    tempfile.mkdtemp(dir=output.parent, prefix="_pad_det_backup_")
                )
                for path in existing:
                    shutil.copy2(path, backup / path.name)

            for name in data_names:
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

        return PublishedDetectionSet(
            directory=output,
            detections_path=output / DETECTIONS_FILE,
            risk_path=output / RISK_FILE,
            alerts_path=output / ALERTS_FILE,
            manifest_path=output / MANIFEST_FILE,
            detection_fingerprint=compute_detection_fingerprint(detections),
            risk_fingerprint=compute_risk_fingerprint(assessments),
            alert_fingerprint=compute_alert_fingerprint(alerts),
            detection_row_count=len(detections),
            risk_row_count=len(assessments),
            alert_row_count=len(alerts),
        )
