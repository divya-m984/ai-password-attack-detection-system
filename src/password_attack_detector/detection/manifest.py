"""Reproducibility manifest for a published detection artifact set.

The manifest is a **superset of the Phase 2 dataset manifest**, exactly as the
Phase 3 feature manifest is, so the single ``verify_dataset`` implementation in
``data/manifest.py`` verifies detection directories too.  Path containment,
``..`` rejection, absolute-path rejection, symlink-escape detection, SHA-256
checksums, and the primary-table row count all come from there rather than
being reimplemented.  That duplication would eventually diverge, and this
particular logic is security-relevant.

:func:`verify_detection_dataset` adds what the shared verifier cannot know:
that the three tables describe one run, that their normalized content
fingerprints still match, and that the configuration and catalog behind them
are the ones now in force.

Every path recorded here is relative to the artifact directory.  No scope
value, entity identifier, event identifier, detection identifier, alert
identifier, secret, absolute path, hostname, or username ever appears -- and a
test sweeps the rendered manifest to prove it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from password_attack_detector.data.manifest import (
    ArtifactEntry,
    VerificationCheck,
    VerificationResult,
    _get_reproducibility,
    _sha256_file,
    verify_dataset,
)
from password_attack_detector.detection.alerts import ALERTING_VERSION
from password_attack_detector.detection.catalog import RULE_CATALOG, RuleCatalog
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.schemas import DETECTION_SCHEMA_VERSION
from password_attack_detector.detection.scoring import SCORING_VERSION
from password_attack_detector.detection.serialization import (
    ALERTS_FILE,
    DETECTIONS_FILE,
    MANIFEST_FILE,
    RISK_FILE,
    compute_alert_fingerprint,
    compute_detection_fingerprint,
    compute_risk_fingerprint,
    read_alerts,
    read_fired_detections,
    read_risk_assessments,
)

__all__ = [
    "DETECTION_MANIFEST_VERSION",
    "EXPECTED_ARTIFACT_ROLES",
    "DetectionManifest",
    "build_detection_manifest",
    "verify_detection_dataset",
]

DETECTION_MANIFEST_VERSION: str = "1.0.0"

#: Namespace for deterministic detection-set identifiers.
_NS_DETECTION_SET = uuid5(NAMESPACE_URL, "password-attack-detector/detection/dataset")

#: Files that must be listed, and what each one is.  An artifact appearing
#: under an unexpected name -- or one of these missing -- means the directory
#: is not the artifact set the manifest claims.
EXPECTED_ARTIFACT_ROLES: dict[str, str] = {
    DETECTIONS_FILE: "fired_detections",
    RISK_FILE: "risk_assessments",
    ALERTS_FILE: "security_alerts",
}

#: Fields describing *when* a manifest was produced rather than what it
#: contains.  Excluded from every deterministic fingerprint: a timestamp in a
#: content digest would make two identical runs disagree.
_NON_CONTENT_FIELDS: frozenset[str] = frozenset({"created_at", "reproducibility"})


@dataclass(frozen=True)
class DetectionManifest:
    """Machine-readable description of one published detection artifact set."""

    # Phase 2 fields, in their exact names and order, so ``verify_dataset``
    # reads this file unchanged.
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

    # Detection-specific additions.
    detection_schema_version: str
    alerting_version: str
    scoring_version: str
    required_feature_schema_version: str
    source_feature_manifest_fingerprint: str | None
    rule_catalog_fingerprint: str
    detection_config_fingerprint: str
    detection_row_count: int
    risk_assessment_row_count: int
    alert_row_count: int
    detection_content_fingerprint: str
    risk_content_fingerprint: str
    alert_content_fingerprint: str
    artifact_roles: dict[str, str]
    alert_grouping_mode: str
    entity_scope_present: bool
    validation_result: dict[str, Any] | None
    quality_report_status: str | None
    evaluation_report_present: bool
    evaluation_report_status: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping.

        The Phase 2 field names come first and keep their exact spelling,
        which is what lets the shared verifier read this file unchanged.
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
                {"relative_path": item.relative_path, "sha256": item.sha256}
                for item in self.artifacts
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
            "detection_schema_version": self.detection_schema_version,
            "alerting_version": self.alerting_version,
            "scoring_version": self.scoring_version,
            "required_feature_schema_version": self.required_feature_schema_version,
            "source_feature_manifest_fingerprint": (
                self.source_feature_manifest_fingerprint
            ),
            "rule_catalog_fingerprint": self.rule_catalog_fingerprint,
            "detection_config_fingerprint": self.detection_config_fingerprint,
            "detection_row_count": self.detection_row_count,
            "risk_assessment_row_count": self.risk_assessment_row_count,
            "alert_row_count": self.alert_row_count,
            "detection_content_fingerprint": self.detection_content_fingerprint,
            "risk_content_fingerprint": self.risk_content_fingerprint,
            "alert_content_fingerprint": self.alert_content_fingerprint,
            "artifact_roles": self.artifact_roles,
            "alert_grouping_mode": self.alert_grouping_mode,
            "entity_scope_present": self.entity_scope_present,
            "validation_result": self.validation_result,
            "quality_report_status": self.quality_report_status,
            "evaluation_report_present": self.evaluation_report_present,
            "evaluation_report_status": self.evaluation_report_status,
        }

    def content_fields(self) -> dict[str, Any]:
        """Return the manifest's semantic content, without the timestamps."""
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in _NON_CONTENT_FIELDS
        }


def build_detection_manifest(
    *,
    staging_dir: Path,
    config: DetectionConfig,
    detection_fingerprint: str,
    risk_fingerprint: str,
    alert_fingerprint: str,
    detection_row_count: int,
    risk_row_count: int,
    alert_row_count: int,
    earliest_event_time: datetime | None,
    latest_event_time: datetime | None,
    alert_grouping_mode: str,
    entity_scope_present: bool,
    validation_status: str,
    validation_result: dict[str, Any] | None = None,
    source_feature_manifest_fingerprint: str | None = None,
    quality_report_status: str | None = None,
    evaluation_report_present: bool = False,
    evaluation_report_status: str | None = None,
    artifact_names: Sequence[str] = (),
    catalog: RuleCatalog = RULE_CATALOG,
) -> DetectionManifest:
    """Build a manifest describing the artifacts staged in *staging_dir*.

    Checksums are taken from the staged files, before promotion, so the
    manifest describes exactly the bytes that are about to be published.

    ``dataset_id`` is derived with UUIDv5 from the three content fingerprints,
    so the same content always identifies the same set -- never ``uuid4``,
    never a wall clock, never machine identity.
    """
    names = tuple(artifact_names) or tuple(EXPECTED_ARTIFACT_ROLES)
    artifacts = [
        ArtifactEntry(relative_path=name, sha256=_sha256_file(staging_dir / name))
        for name in names
        if (staging_dir / name).exists()
    ]
    combined = f"{detection_fingerprint}|{risk_fingerprint}|{alert_fingerprint}"

    return DetectionManifest(
        manifest_version=DETECTION_MANIFEST_VERSION,
        dataset_id=str(uuid5(_NS_DETECTION_SET, combined)),
        schema_version=config.required_feature_schema_version,
        source_type="detection",
        # The Phase 2 ``row_count`` describes the primary table, which here is
        # the risk-assessment table: one row per evaluated anchor.
        row_count=risk_row_count,
        ground_truth_row_count=None,
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
        content_fingerprint=combined,
        config_fingerprint=config.fingerprint(),
        validation_status=validation_status,
        created_at=datetime.now(UTC).isoformat(),
        reproducibility=_get_reproducibility(generator_version=None, seed=None),
        primary_artifact=RISK_FILE,
        detection_schema_version=DETECTION_SCHEMA_VERSION,
        alerting_version=ALERTING_VERSION,
        scoring_version=SCORING_VERSION,
        required_feature_schema_version=config.required_feature_schema_version,
        source_feature_manifest_fingerprint=source_feature_manifest_fingerprint,
        rule_catalog_fingerprint=catalog.fingerprint(),
        detection_config_fingerprint=config.fingerprint(),
        detection_row_count=detection_row_count,
        risk_assessment_row_count=risk_row_count,
        alert_row_count=alert_row_count,
        detection_content_fingerprint=detection_fingerprint,
        risk_content_fingerprint=risk_fingerprint,
        alert_content_fingerprint=alert_fingerprint,
        artifact_roles={
            name: EXPECTED_ARTIFACT_ROLES[name]
            for name in names
            if name in EXPECTED_ARTIFACT_ROLES
        },
        alert_grouping_mode=alert_grouping_mode,
        entity_scope_present=entity_scope_present,
        validation_result=validation_result,
        quality_report_status=quality_report_status,
        evaluation_report_present=evaluation_report_present,
        evaluation_report_status=evaluation_report_status,
    )


def verify_detection_dataset(
    directory: Path, *, config: DetectionConfig | None = None
) -> VerificationResult:
    """Verify a published detection directory.

    Delegates the shared checks -- manifest readability, version, path safety,
    symlink escape, checksums, primary row count, validation status -- to the
    Phase 2 verifier, then adds the cross-artifact checks it cannot know
    about.  Supplying *config* additionally checks that the configuration and
    catalog behind the artifacts are the ones now in force.

    Never raises: every failure is a check with ``passed=False``.
    """
    base = verify_dataset(directory, manifest_name=MANIFEST_FILE)
    checks = list(base.checks)
    manifest_path = directory / MANIFEST_FILE

    if base.manifest is None:
        return base

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - unreachable once base passed
        checks.append(
            VerificationCheck(
                name="DETECTION_MANIFEST_READABLE",
                passed=False,
                message=f"Cannot read detection manifest ({type(exc).__name__})",
            )
        )
        return VerificationResult(
            passed=False, checks=tuple(checks), manifest=base.manifest
        )

    checks.append(_check_roles(raw))
    checks.extend(_check_content(directory, raw))
    checks.extend(_check_relationships(directory, raw))
    if config is not None:
        checks.extend(_check_configuration(raw, config))

    return VerificationResult(
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
        manifest=base.manifest,
    )


def _check_roles(raw: dict[str, Any]) -> VerificationCheck:
    """Check the manifest lists exactly the expected artifact roles."""
    roles = raw.get("artifact_roles") or {}
    listed = {entry["relative_path"] for entry in raw.get("artifacts", [])}
    expected = set(EXPECTED_ARTIFACT_ROLES)
    missing = sorted(expected - listed)
    unexpected = sorted(
        name
        for name in listed
        if name not in expected and not name.endswith((".json", ".md"))
    )
    wrong_role = sorted(
        name
        for name, role in roles.items()
        if EXPECTED_ARTIFACT_ROLES.get(name) != role
    )
    passed = not missing and not unexpected and not wrong_role
    return VerificationCheck(
        name="ARTIFACT_ROLES_EXPECTED",
        passed=passed,
        message=(
            "Artifact roles match the expected detection set"
            if passed
            else (
                f"{len(missing)} expected artifact(s) missing, "
                f"{len(unexpected)} unexpected artifact(s), "
                f"{len(wrong_role)} role mismatch(es)"
            )
        ),
    )


def _check_content(directory: Path, raw: dict[str, Any]) -> list[VerificationCheck]:
    """Recompute each table's normalized content fingerprint and row count.

    Each table is read and digested by its own typed pair, so a reader can
    never be paired with the wrong fingerprint function.
    """
    return [
        _check_one_table(
            "DETECTION_CONTENT_FINGERPRINT",
            directory / DETECTIONS_FILE,
            raw.get("detection_content_fingerprint"),
            raw.get("detection_row_count"),
            lambda path: _digest(
                read_fired_detections(path), compute_detection_fingerprint
            ),
        ),
        _check_one_table(
            "RISK_CONTENT_FINGERPRINT",
            directory / RISK_FILE,
            raw.get("risk_content_fingerprint"),
            raw.get("risk_assessment_row_count"),
            lambda path: _digest(read_risk_assessments(path), compute_risk_fingerprint),
        ),
        _check_one_table(
            "ALERT_CONTENT_FINGERPRINT",
            directory / ALERTS_FILE,
            raw.get("alert_content_fingerprint"),
            raw.get("alert_row_count"),
            lambda path: _digest(read_alerts(path), compute_alert_fingerprint),
        ),
    ]


def _digest[T](
    rows: Sequence[T], fingerprint: Callable[[Sequence[T]], str]
) -> tuple[int, str]:
    """Return one table's row count and normalized content fingerprint."""
    return len(rows), fingerprint(rows)


def _check_one_table(
    name: str,
    path: Path,
    expected_digest: object,
    expected_count: object,
    read: Callable[[Path], tuple[int, str]],
) -> VerificationCheck:
    """Verify one artifact table's row count and content fingerprint."""
    if not path.exists():
        return VerificationCheck(
            name=name, passed=False, message=f"{path.name} is missing"
        )
    try:
        count, digest = read(path)
    except Exception as exc:
        return VerificationCheck(
            name=name,
            passed=False,
            message=f"Cannot read {path.name} ({type(exc).__name__})",
        )
    passed = digest == expected_digest and count == expected_count
    return VerificationCheck(
        name=name,
        passed=passed,
        message=(
            f"{path.name} content fingerprint and row count match"
            if passed
            else f"{path.name} content fingerprint or row count differs"
        ),
    )


def _check_relationships(
    directory: Path, raw: dict[str, Any]
) -> list[VerificationCheck]:
    """Check the three tables still describe one coherent run."""
    paths = {
        name: directory / name for name in (DETECTIONS_FILE, RISK_FILE, ALERTS_FILE)
    }
    if not all(path.exists() for path in paths.values()):
        return [
            VerificationCheck(
                name="ARTIFACT_RELATIONSHIPS",
                passed=False,
                message="Skipped: one or more artifact tables are missing",
            )
        ]
    try:
        detections = read_fired_detections(paths[DETECTIONS_FILE])
        assessments = read_risk_assessments(paths[RISK_FILE])
        alerts = read_alerts(paths[ALERTS_FILE])
    except Exception as exc:
        return [
            VerificationCheck(
                name="ARTIFACT_RELATIONSHIPS",
                passed=False,
                message=f"Cannot read the artifact tables ({type(exc).__name__})",
            )
        ]

    assessed = {item.anchor_event_id for item in assessments}
    orphaned = len({item.anchor_event_id for item in detections} - assessed)
    fired_rules = {item.rule_id for item in detections}
    dangling = sum(
        1 for alert in alerts if not set(alert.contributing_rule_ids) <= fired_rules
    )
    scope_leak = sum(
        1
        for alert in alerts
        if alert.scope_value is not None and not raw.get("entity_scope_present")
    )
    passed = orphaned == 0 and dangling == 0 and scope_leak == 0
    return [
        VerificationCheck(
            name="ARTIFACT_RELATIONSHIPS",
            passed=passed,
            message=(
                "Detection, risk, and alert relationships are consistent"
                if passed
                else (
                    f"{orphaned} detection anchor(s) without an assessment, "
                    f"{dangling} alert(s) naming an unknown rule, "
                    f"{scope_leak} alert(s) carrying unexpected scope metadata"
                )
            ),
        )
    ]


def _check_configuration(
    raw: dict[str, Any], config: DetectionConfig
) -> list[VerificationCheck]:
    """Check the manifest's fingerprints against the active configuration."""
    checks: list[VerificationCheck] = []
    config_ok = raw.get("detection_config_fingerprint") == config.fingerprint()
    checks.append(
        VerificationCheck(
            name="CONFIG_FINGERPRINT_MATCH",
            passed=config_ok,
            message=(
                "Detection configuration fingerprint matches the active run"
                if config_ok
                else "Detection configuration fingerprint differs from the active run"
            ),
        )
    )
    catalog_ok = raw.get("rule_catalog_fingerprint") == RULE_CATALOG.fingerprint()
    checks.append(
        VerificationCheck(
            name="CATALOG_FINGERPRINT_MATCH",
            passed=catalog_ok,
            message=(
                "Rule catalog fingerprint matches the registered catalog"
                if catalog_ok
                else "Rule catalog fingerprint differs from the registered catalog"
            ),
        )
    )
    schema_ok = (
        raw.get("detection_schema_version") == DETECTION_SCHEMA_VERSION
        and raw.get("scoring_version") == SCORING_VERSION
        and raw.get("alerting_version") == ALERTING_VERSION
    )
    checks.append(
        VerificationCheck(
            name="CONTRACT_VERSIONS_SUPPORTED",
            passed=schema_ok,
            message=(
                "Detection, scoring, and alerting versions are supported"
                if schema_ok
                else "One or more contract versions are unsupported"
            ),
        )
    )
    return checks
