"""Validating a prediction publication, without opening a single label.

A published prediction directory is **untrusted data**.  It may have been copied
from another machine, edited by hand, truncated by a full disk, or written by
somebody who would like this process to believe something.  So every check below
reads bytes and recomputes, and none of them trusts the artifact's account of
itself.

**This is not evaluation, and it cannot become evaluation.**  Nothing here reads
a label, a campaign, or a split assignment; nothing here computes an accuracy, a
precision, a recall, or any other outcome-dependent number; and there is no
parameter through which ground truth could be supplied.  What validation
establishes is that the publication is *structurally sound and internally
consistent* -- that the rows are the rows the manifest describes, that each
stored decision follows from that row's own stored numbers under the frozen
predicate, and that the lineage the manifest names is coherent.  Whether the
predictions were any good is a different question, asked once, later, by a
milestone that is allowed to open the test labels.

**A skipped mandatory check is not a pass.**  Every check carries a stable code
and one of three statuses, and the aggregate is ``PASS`` only when every
mandatory check ran and passed.  The two check sets are declared as constants and
a result is refused unless it reports exactly its stage's set, so a check that
quietly stopped being emitted fails the result rather than improving it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.calibration import SealedModel
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    AuditCheckStatus,
    AuditStatus,
    MLSplit,
    is_probability,
)
from password_attack_detector.ml.ordering import is_canonical
from password_attack_detector.ml.prediction_manifest import (
    ANOMALY_PREDICTION_FILE,
    BINARY_PREDICTION_FILE,
    CATEGORY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    QUALITY_REPORT_JSON_FILE,
    QUALITY_REPORT_MD_FILE,
    VALIDATION_RESULT_FILE,
    PredictionManifest,
    prediction_content_fingerprint,
    scope_role_for,
)
from password_attack_detector.ml.prediction_serialization import (
    ANOMALY_PREDICTION_SCHEMA,
    BINARY_PREDICTION_SCHEMA,
    CATEGORY_PREDICTION_SCHEMA,
    read_anomaly_scores,
    read_binary_predictions,
    read_category_predictions,
)
from password_attack_detector.ml.predictions import (
    PROHIBITED_PREDICTION_COLUMNS,
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
    category_applicable_anchors,
)
from password_attack_detector.ml.schemas import Sha256Hex
from password_attack_detector.ml.thresholds import (
    assign_category,
    flagged_anomalous,
    flagged_malicious,
)

__all__ = [
    "PUBLISHED_CHECKS",
    "STAGED_CHECKS",
    "VALIDATION_SCHEMA_VERSION",
    "MLValidationResult",
    "ValidationCheck",
    "ValidationStage",
    "validate_publication",
    "validate_staged_predictions",
]

#: The validation contract's own version.
VALIDATION_SCHEMA_VERSION: Final[str] = "1.0.0"


class ValidationStage(BaseModel):
    """Which of the two validation scopes produced a result.

    ``staged`` runs inside publication, over the rows and tables, before a
    manifest exists to describe them.  ``published`` runs over a complete
    publication and additionally checks the manifest, the declared file set,
    every checksum, and the derived prediction identity.

    A stage is recorded rather than inferred so a staged result can never be
    read as evidence that a manifest was checked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str

    @field_validator("stage")
    @classmethod
    def check_stage(cls, value: str) -> str:
        """Two stages, and no third."""
        if value not in {"staged", "published"}:
            raise ValueError(f"unknown validation stage {value!r}")
        return value


#: Row and table checks.  Run at both stages: what the rows say has to hold
#: whether or not a manifest describing them exists yet.
STAGED_CHECKS: Final[tuple[str, ...]] = (
    "M010",  # binary table schema and column order
    "M011",  # binary row count is positive and readable
    "M012",  # anchors are unique
    "M013",  # rows are in canonical order
    "M014",  # every stored number is finite
    "M015",  # calibrated probabilities lie in [0, 1]
    "M016",  # a probability exists exactly when a calibrator produced it
    "M017",  # the stored binary decision follows from the stored numbers
    "M018",  # one frozen threshold across the table
    "M019",  # category class maps are complete, finite, and ordered
    "M020",  # the stored category assignment follows from the stored scores
    "M021",  # the category table covers exactly the binary-positive rows
    "M022",  # anomaly rows carry a magnitude and no probability
    "M023",  # the stored anomaly flag follows from the stored numbers
    "M024",  # no prohibited column appears in any table
)

#: Everything above, plus the checks that need a manifest to check against.
PUBLISHED_CHECKS: Final[tuple[str, ...]] = (
    "M001",  # the manifest is present and parses at this contract version
    "M002",  # the manifest recomputes its own seal
    "M003",  # the prediction identity is the one the manifest's content derives
    "M004",  # exactly the declared files are present, and nothing else
    "M005",  # no member is a symbolic link, and no path escapes the directory
    "M006",  # every declared digest matches the bytes on disk
    "M007",  # every declared byte size matches the bytes on disk
    "M008",  # the declared row counts match the tables
    "M009",  # the prediction content fingerprint recomputes from the rows
    *STAGED_CHECKS,
    "M025",  # the scope and its declared role agree
    "M026",  # the aggregate reports the manifest binds are the ones present
    "M027",  # the frozen threshold on the rows is the one the lineage names
    "M028",  # the frozen category lineage matches what the rows record
)

#: What each code means, in one line.  Kept beside the codes so a stable code
#: cannot drift away from the thing it names.
_CHECK_NAMES: Final[dict[str, str]] = {
    "M001": "manifest_present_and_parsable",
    "M002": "manifest_seal_recomputes",
    "M003": "prediction_identity_recomputes",
    "M004": "declared_file_set_exact",
    "M005": "path_policy",
    "M006": "file_checksums_match",
    "M007": "file_sizes_match",
    "M008": "declared_row_counts_match",
    "M009": "prediction_content_fingerprint_matches",
    "M010": "binary_table_schema",
    "M011": "binary_table_readable",
    "M012": "unique_anchors",
    "M013": "canonical_row_order",
    "M014": "finite_values",
    "M015": "probability_bounds",
    "M016": "probability_requires_calibrator",
    "M017": "binary_decision_consistency",
    "M018": "single_frozen_threshold",
    "M019": "category_class_map_complete",
    "M020": "category_abstention_consistency",
    "M021": "category_rows_are_binary_positive",
    "M022": "anomaly_score_vocabulary",
    "M023": "anomaly_flag_consistency",
    "M024": "no_prohibited_column",
    "M025": "scope_role_consistency",
    "M026": "bound_reports_present",
    "M027": "threshold_matches_lineage",
    "M028": "category_lineage_matches_rows",
}


class ValidationCheck(BaseModel):
    """The outcome of one named prediction-artifact check.

    ``detail`` states counts and declared names.  It never carries an anchor
    identifier, a pseudonym, a coordinate, or a path: a validation report is
    read by whoever is investigating a failure, and a report that leaked the
    identity of the failing rows would be the wrong artifact to hand them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    name: str
    status: AuditCheckStatus
    mandatory: bool
    detail: str

    @field_validator("code")
    @classmethod
    def check_code(cls, value: str) -> str:
        """Codes are the stable ``M0xx`` namespace reserved for this milestone."""
        if value not in _CHECK_NAMES:
            raise ValueError(f"unknown validation check code {value!r}")
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """A check reports the name its code was reserved for."""
        if self.name != _CHECK_NAMES[self.code]:
            raise ValueError(
                f"check {self.code} reports name {self.name!r}, not the name "
                f"that code names"
            )
        if not self.detail.strip():
            raise ValueError("a check states why it reached its status")
        return self

    @property
    def passed(self) -> bool:
        """Return whether this check passed.  Skipped is never a pass."""
        return self.status is AuditCheckStatus.PASS


class MLValidationResult(SealedModel):
    """What validating one prediction publication found.

    Sealed so a result stored beside the artifacts it describes cannot be edited
    into a cleaner one.  Carries no metric: there is nothing outcome-dependent
    here to report, and a field for one would eventually hold a number somebody
    computed against labels.
    """

    fingerprint_field: ClassVar[str] = "result_fingerprint"
    schema_version_field: ClassVar[str] = "validation_schema_version"
    schema_version: ClassVar[str] = VALIDATION_SCHEMA_VERSION
    record_label: ClassVar[str] = "prediction validation result"

    validation_schema_version: str = VALIDATION_SCHEMA_VERSION
    stage: ValidationStage
    status: AuditStatus
    #: The split the rows were scored for.  ``None`` only when the manifest that
    #: names it could not be read -- reporting a scope nobody found would be a
    #: guess dressed as a fact.
    scope: MLSplit | None

    checks: tuple[ValidationCheck, ...]

    binary_row_count: int = Field(ge=0)
    category_row_count: int | None = Field(default=None, ge=0)
    anomaly_row_count: int | None = Field(default=None, ge=0)

    result_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_result(self) -> Self:
        """A result reports exactly its stage's check set, and grades it correctly."""
        expected = STAGED_CHECKS if self.stage.stage == "staged" else PUBLISHED_CHECKS
        codes = tuple(item.code for item in self.checks)
        if codes != tuple(sorted(expected)):
            raise ValueError(
                f"a {self.stage.stage} result reports {len(codes)} check(s); "
                f"this stage declares {len(expected)}, and a result missing one "
                f"is not a result that ran it"
            )
        mandatory_passed = all(item.passed for item in self.checks if item.mandatory)
        if (self.status is AuditStatus.PASS) != mandatory_passed:
            raise ValueError(
                "the aggregate status must be PASS exactly when every mandatory "
                "check passed; a skipped mandatory check is not a passed one"
            )
        if self.scope is None and self.status is AuditStatus.PASS:
            raise ValueError(
                "a result that could not determine which split it validated is "
                "never a passing result"
            )
        return self

    @property
    def passed(self) -> bool:
        """Return whether the publication validated."""
        return self.status is AuditStatus.PASS

    @property
    def failures(self) -> tuple[str, ...]:
        """Return the codes of every mandatory check that did not pass."""
        return tuple(
            item.code for item in self.checks if item.mandatory and not item.passed
        )


# ---------------------------------------------------------------------------
# Running the checks
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Collects check outcomes, one per declared code, in code order."""

    outcomes: dict[str, ValidationCheck]

    def record(
        self,
        code: str,
        status: AuditCheckStatus,
        detail: str,
        *,
        mandatory: bool = True,
    ) -> None:
        """Record one check, refusing to record the same code twice."""
        if code in self.outcomes:
            raise ValueError(f"check {code} was recorded twice")
        self.outcomes[code] = ValidationCheck(
            code=code,
            name=_CHECK_NAMES[code],
            status=status,
            mandatory=mandatory,
            detail=detail,
        )

    def ok(self, code: str, detail: str) -> None:
        """Record a passing check."""
        self.record(code, AuditCheckStatus.PASS, detail)

    def fail(self, code: str, detail: str) -> None:
        """Record a failing check."""
        self.record(code, AuditCheckStatus.FAIL, detail)

    def skip(self, code: str, detail: str) -> None:
        """Record a check that could not run.  Never a pass."""
        self.record(code, AuditCheckStatus.SKIPPED, detail)

    def verdict(self, code: str, condition: bool, passed: str, failed: str) -> bool:
        """Record *condition* as a pass or a failure and return it."""
        if condition:
            self.ok(code, passed)
        else:
            self.fail(code, failed)
        return condition

    def fill(self, codes: Sequence[str], detail: str) -> None:
        """Record every unrecorded code in *codes* as skipped."""
        for code in codes:
            if code not in self.outcomes:
                self.skip(code, detail)

    def checks(self) -> tuple[ValidationCheck, ...]:
        """Return the recorded checks in code order."""
        return tuple(self.outcomes[code] for code in sorted(self.outcomes))


def _row_checks(
    recorder: _Recorder,
    *,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    anomaly: Sequence[AnomalyScore] | None,
) -> None:
    """Record every check that reads only the rows themselves."""
    recorder.verdict(
        "M011",
        bool(binary),
        f"{len(binary):,} binary prediction row(s) read",
        "the binary prediction table carries no rows",
    )

    anchors = [row.anchor_event_id for row in binary]
    recorder.verdict(
        "M012",
        len(set(anchors)) == len(anchors),
        f"{len(anchors):,} anchor(s), each appearing once",
        f"{len(anchors) - len(set(anchors)):,} duplicate anchor(s); each anchor "
        f"event contributes exactly one prediction",
    )
    recorder.verdict(
        "M013",
        is_canonical(binary),
        "rows are in canonical anchor order",
        "rows are not sorted by anchor event time then anchor identifier",
    )

    # Every stored number was already refused if non-finite by the row model, so
    # a table that read back at all has finite values. The check is recorded
    # rather than assumed: a reader that stopped constructing typed rows would
    # otherwise turn a mandatory check into an absent one.
    non_finite = sum(
        1
        for row in binary
        if not _all_finite(
            row.malicious_decision_score,
            row.decision_threshold,
            row.malicious_probability,
        )
    )
    recorder.verdict(
        "M014",
        non_finite == 0,
        f"{len(binary):,} row(s) carry finite scores and thresholds",
        f"{non_finite:,} row(s) carry a non-finite value",
    )

    out_of_range = sum(
        1
        for row in binary
        if row.malicious_probability is not None
        and not 0.0 <= row.malicious_probability <= 1.0
    )
    recorder.verdict(
        "M015",
        out_of_range == 0,
        "every calibrated probability lies in [0, 1]",
        f"{out_of_range:,} probability value(s) lie outside [0, 1]",
    )

    mismatched_kind = sum(
        1
        for row in binary
        if (row.malicious_probability is not None) != is_probability(row.score_kind)
    )
    recorder.verdict(
        "M016",
        mismatched_kind == 0,
        "a probability is present exactly where a calibrator produced one",
        f"{mismatched_kind:,} row(s) carry a probability without a calibrated "
        f"score kind, or the reverse",
    )

    contradicting = sum(
        1
        for row in binary
        if row.flagged_malicious
        != flagged_malicious(row.decided_score, threshold=row.decision_threshold)
    )
    recorder.verdict(
        "M017",
        contradicting == 0,
        f"{len(binary):,} decision(s) recompute from their own score and threshold",
        f"{contradicting:,} row(s) contradict the frozen predicate applied to "
        f"their own stored numbers",
    )

    thresholds = {row.decision_threshold for row in binary}
    kinds = {str(row.score_kind) for row in binary}
    recorder.verdict(
        "M018",
        len(thresholds) <= 1 and len(kinds) <= 1,
        "one frozen operating point across the table",
        f"the table records {len(thresholds):,} threshold(s) and "
        f"{len(kinds):,} score kind(s); a publication applies one frozen "
        f"operating point to every row",
    )

    _category_checks(recorder, binary=binary, category=category)
    _anomaly_checks(recorder, anomaly=anomaly)

    published_columns = set(BINARY_PREDICTION_SCHEMA.names)
    published_columns |= set(CATEGORY_PREDICTION_SCHEMA.names)
    published_columns |= set(ANOMALY_PREDICTION_SCHEMA.names)
    offending = sorted(published_columns & PROHIBITED_PREDICTION_COLUMNS)
    recorder.verdict(
        "M024",
        not offending,
        f"{len(published_columns):,} published column(s), none of them a "
        f"pseudonym, a campaign, a coordinate, a credential, or a label",
        f"published table(s) declare prohibited column(s) {offending}",
    )


def _category_checks(
    recorder: _Recorder,
    *,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
) -> None:
    """Record the category checks, or mark them not applicable.

    Absence of a category head is not a failure and is not a skip of a mandatory
    check: there is no category artifact to validate, so the checks are recorded
    as passing over nothing with the reason stated. A *declared* category
    artifact that then fails is a failure.
    """
    if category is None:
        recorder.ok("M019", "no category head was frozen; there is no class map")
        recorder.ok("M020", "no category head was frozen; nothing abstained")
        recorder.ok(
            "M021",
            "no category artifact was published; no row claims triage applicability",
        )
        return

    malformed = 0
    for row in category:
        try:
            scores = row.class_scores()
        except ValueError:
            malformed += 1
            continue
        if UNKNOWN_CATEGORY in scores or list(scores) != sorted(scores):
            malformed += 1
    recorder.verdict(
        "M019",
        malformed == 0,
        f"{len(category):,} class map(s) are complete, finite, and in class order",
        f"{malformed:,} class map(s) are malformed, unordered, or declare the "
        f"abstention label as a scored class",
    )

    contradicting = 0
    for row in category:
        try:
            scores = row.class_scores()
        except ValueError:
            contradicting += 1
            continue
        order = tuple(scores)
        assigned, _ = assign_category(
            [scores[name] for name in order],
            order,
            min_category_score=row.min_category_score,
        )
        if assigned != row.predicted_scenario:
            contradicting += 1
    recorder.verdict(
        "M020",
        contradicting == 0,
        f"{len(category):,} assignment(s) recompute from their own scores and floor",
        f"{contradicting:,} row(s) contradict the frozen abstention rule",
    )

    # Applicability, and it is checked in both directions. A category row for a
    # row the binary head did not flag is a triage result for an event nobody
    # routed to triage; a flagged row missing from the table is a triage result
    # that was asked for and lost. Neither is repaired here.
    applicable = list(category_applicable_anchors(binary))
    categorised = [row.anchor_event_id for row in category]
    surplus = len(set(categorised) - set(applicable))
    missing = len(set(applicable) - set(categorised))
    recorder.verdict(
        "M021",
        categorised == applicable,
        f"{len(categorised):,} categorised row(s), exactly the binary-positive "
        f"rows, in the same canonical order",
        f"{surplus:,} categorised row(s) were not flagged by the binary head and "
        f"{missing:,} flagged row(s) were not categorised; a category artifact "
        f"covers the binary-positive rows and only those",
    )


def _anomaly_checks(
    recorder: _Recorder, *, anomaly: Sequence[AnomalyScore] | None
) -> None:
    """Record the experimental anomaly checks, or mark them not applicable."""
    if anomaly is None:
        recorder.ok("M022", "no experimental anomaly artifact was published")
        recorder.ok("M023", "no experimental anomaly artifact was published")
        return

    vocabulary = sum(
        1
        for row in anomaly
        if not row.experimental or row.influences_champion_selection
    )
    recorder.verdict(
        "M022",
        vocabulary == 0 and "malicious_probability" not in ANOMALY_COLUMN_NAMES,
        f"{len(anomaly):,} anomaly row(s) carry a magnitude, are marked "
        f"experimental, and influence no selection",
        f"{vocabulary:,} anomaly row(s) claim non-experimental status or "
        f"influence over champion selection",
    )

    contradicting = sum(
        1
        for row in anomaly
        if row.anomaly_threshold is not None
        and row.flagged_anomalous
        != flagged_anomalous(row.anomaly_score, threshold=row.anomaly_threshold)
    )
    recorder.verdict(
        "M023",
        contradicting == 0,
        "every anomaly flag recomputes from its own score and threshold",
        f"{contradicting:,} anomaly row(s) contradict the inverted predicate",
    )


#: The anomaly table's column names, read once so the vocabulary check reads a
#: fact rather than a memory.
ANOMALY_COLUMN_NAMES: Final[tuple[str, ...]] = tuple(ANOMALY_PREDICTION_SCHEMA.names)


def _all_finite(*values: float | None) -> bool:
    """Return whether every supplied value is finite or absent."""
    return all(value is None or math.isfinite(value) for value in values)


def validate_staged_predictions(
    *,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    anomaly: Sequence[AnomalyScore] | None,
    scope: MLSplit,
    binary_schema_matches: bool,
) -> MLValidationResult:
    """Validate rows read back from staging, before a manifest describes them.

    Args:
        binary: the binary predictions as read back from the staged file.
        category: the category predictions, or ``None`` when no head was frozen.
        anomaly: the experimental anomaly scores, or ``None``.
        scope: the split the rows were scored for.
        binary_schema_matches: whether the staged table's Arrow schema is the
            declared one.  Passed in rather than re-derived because the reader
            has already compared it and a second comparison could disagree.
    """
    recorder = _Recorder(outcomes={})
    recorder.verdict(
        "M010",
        binary_schema_matches,
        "the staged table carries the declared Arrow schema and column order",
        "the staged table's schema is not the declared one",
    )
    _row_checks(recorder, binary=binary, category=category, anomaly=anomaly)
    checks = recorder.checks()
    return _sealed(
        stage="staged",
        scope=scope,
        checks=checks,
        binary=binary,
        category=category,
        anomaly=anomaly,
    )


def _sealed(
    *,
    stage: str,
    scope: MLSplit | None,
    checks: tuple[ValidationCheck, ...],
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    anomaly: Sequence[AnomalyScore] | None,
) -> MLValidationResult:
    """Return the sealed result for a completed set of checks."""
    passed = all(item.passed for item in checks if item.mandatory)
    return MLValidationResult.seal(
        stage=ValidationStage(stage=stage),
        status=AuditStatus.PASS if passed else AuditStatus.FAIL,
        scope=scope,
        checks=checks,
        binary_row_count=len(binary),
        category_row_count=None if category is None else len(category),
        anomaly_row_count=None if anomaly is None else len(anomaly),
    )


# ---------------------------------------------------------------------------
# Validating a complete publication
# ---------------------------------------------------------------------------

#: The complete set of file names a publication may contain.
_PERMITTED_FILES: Final[frozenset[str]] = frozenset(
    {
        BINARY_PREDICTION_FILE,
        CATEGORY_PREDICTION_FILE,
        ANOMALY_PREDICTION_FILE,
        VALIDATION_RESULT_FILE,
        QUALITY_REPORT_JSON_FILE,
        QUALITY_REPORT_MD_FILE,
        PREDICTION_MANIFEST_FILE,
    }
)

#: The largest file this validator will read into memory to digest it.
_MAX_FILE_BYTES: Final[int] = 1 << 30


def validate_publication(directory: Path) -> MLValidationResult:
    """Validate the complete prediction publication at *directory*.

    Every check in :data:`PUBLISHED_CHECKS` is attempted.  A failure early in
    the chain leaves the checks that depended on it recorded as **skipped**, and
    a skipped mandatory check is not a pass -- so a publication whose manifest
    will not parse fails rather than passing the handful of checks that did not
    need one.

    Raises:
        DataValidationError: the directory does not exist.  Everything else is
            reported as a failing check rather than as an exception, because a
            caller asked whether the publication is valid and "no" is an answer.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DataValidationError("The prediction publication directory is not present")

    recorder = _Recorder(outcomes={})
    manifest = _check_manifest(recorder, directory)
    if manifest is None:
        recorder.fill(
            PUBLISHED_CHECKS,
            "not evaluated: the manifest could not be read, and a check that "
            "did not run is not a check that passed",
        )
        return _sealed(
            stage="published",
            scope=None,
            checks=recorder.checks(),
            binary=(),
            category=None,
            anomaly=None,
        )

    _check_files(recorder, directory, manifest)
    binary, category, anomaly = _read_tables(recorder, directory, manifest)
    if binary is None:
        recorder.fill(
            PUBLISHED_CHECKS,
            "not evaluated: the prediction tables could not be read",
        )
        return _sealed(
            stage="published",
            scope=manifest.scope,
            checks=recorder.checks(),
            binary=(),
            category=None,
            anomaly=None,
        )

    _row_checks(recorder, binary=binary, category=category, anomaly=anomaly)
    _check_against_manifest(
        recorder,
        manifest,
        binary=binary,
        category=category,
        anomaly=anomaly,
    )
    recorder.fill(PUBLISHED_CHECKS, "not evaluated")
    return _sealed(
        stage="published",
        scope=manifest.scope,
        checks=recorder.checks(),
        binary=binary,
        category=category,
        anomaly=anomaly,
    )


def _check_manifest(recorder: _Recorder, directory: Path) -> PredictionManifest | None:
    """Record the manifest checks and return the manifest, or ``None``."""
    path = directory / PREDICTION_MANIFEST_FILE
    if not path.is_file():
        recorder.fail("M001", "the publication carries no prediction manifest")
        return None
    try:
        manifest = PredictionManifest.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        recorder.fail(
            "M001",
            f"the prediction manifest is not valid at this contract version "
            f"({type(exc).__name__})",
        )
        return None
    recorder.ok("M001", "the prediction manifest parses at this contract version")
    # The seal validator already refused a manifest whose content and digest
    # disagree, so reaching this point *is* the evidence. Recomputing it here
    # states the check rather than implying it.
    recorder.verdict(
        "M002",
        manifest.prediction_manifest_fingerprint == manifest.recomputed_fingerprint(),
        "the manifest recomputes its recorded fingerprint",
        "the manifest's content and its recorded digest disagree",
    )
    recorder.verdict(
        "M003",
        manifest.prediction_id == manifest.derived_prediction_id(),
        "the prediction identifier is the one the manifest's content derives",
        "the prediction identifier is not derived from this manifest's content",
    )
    recorder.verdict(
        "M025",
        manifest.scope_role == scope_role_for(manifest.scope),
        f"the {str(manifest.scope)!r} scope is published under its own role",
        "the declared scope role is not the one this scope carries",
    )
    return manifest


def _check_files(
    recorder: _Recorder, directory: Path, manifest: PredictionManifest
) -> None:
    """Record the file-set, path-policy, checksum, and size checks."""
    present = sorted(item.name for item in directory.iterdir())
    declared = sorted(item.relative_path for item in manifest.files)
    recorder.verdict(
        "M004",
        present == sorted({*declared, PREDICTION_MANIFEST_FILE}),
        f"{len(declared):,} declared file(s) present, and nothing else",
        f"the directory holds {len(present):,} file(s) against "
        f"{len(declared) + 1:,} declared",
    )

    unsafe = [
        item.name
        for item in directory.iterdir()
        if item.is_symlink() or not item.is_file() or item.name not in _PERMITTED_FILES
    ]
    recorder.verdict(
        "M005",
        not unsafe,
        "every member is a regular file with a permitted name",
        f"{len(unsafe):,} member(s) are symbolic links, directories, or carry a "
        f"name no publication may contain",
    )

    mismatched_digest = 0
    mismatched_size = 0
    for declaration in manifest.files:
        path = directory / declaration.relative_path
        if not path.is_file() or path.is_symlink():
            mismatched_digest += 1
            mismatched_size += 1
            continue
        size = path.stat().st_size
        if size != declaration.byte_size:
            mismatched_size += 1
        if size > _MAX_FILE_BYTES:
            mismatched_digest += 1
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != declaration.sha256:
            mismatched_digest += 1
    recorder.verdict(
        "M006",
        mismatched_digest == 0,
        f"{len(manifest.files):,} declared digest(s) match the bytes on disk",
        f"{mismatched_digest:,} declared digest(s) do not match the bytes on disk",
    )
    recorder.verdict(
        "M007",
        mismatched_size == 0,
        f"{len(manifest.files):,} declared size(s) match the bytes on disk",
        f"{mismatched_size:,} declared size(s) do not match the bytes on disk",
    )

    bound = {
        VALIDATION_RESULT_FILE: manifest.validation_result_fingerprint,
        QUALITY_REPORT_JSON_FILE: manifest.quality_report_fingerprint,
    }
    unbound = 0
    for name, fingerprint in bound.items():
        report = next(
            (item for item in manifest.files if item.logical_name == name), None
        )
        if (report is None) != (fingerprint is None) or (
            report is not None and report.sha256 != fingerprint
        ):
            unbound += 1
    recorder.verdict(
        "M026",
        unbound == 0,
        "the aggregate reports the manifest binds are the ones it declares",
        f"{unbound:,} bound report(s) are absent, unbound, or bound to a "
        f"different digest",
    )


def _read_tables(
    recorder: _Recorder, directory: Path, manifest: PredictionManifest
) -> tuple[
    tuple[BinaryPrediction, ...] | None,
    tuple[CategoryPrediction, ...] | None,
    tuple[AnomalyScore, ...] | None,
]:
    """Read every declared table, recording the schema check."""
    declared = {item.logical_name for item in manifest.files}
    try:
        binary = read_binary_predictions(directory / BINARY_PREDICTION_FILE)
    except (DataValidationError, ValueError) as exc:
        recorder.fail("M010", f"the binary prediction table is invalid ({exc})")
        return (None, None, None)

    category: tuple[CategoryPrediction, ...] | None = None
    anomaly: tuple[AnomalyScore, ...] | None = None
    try:
        if CATEGORY_PREDICTION_FILE in declared:
            category = read_category_predictions(directory / CATEGORY_PREDICTION_FILE)
        if ANOMALY_PREDICTION_FILE in declared:
            anomaly = read_anomaly_scores(directory / ANOMALY_PREDICTION_FILE)
    except (DataValidationError, ValueError) as exc:
        recorder.fail("M010", f"a declared prediction table is invalid ({exc})")
        return (None, None, None)

    recorder.ok(
        "M010",
        f"{len(declared & _TABLE_FILES):,} table(s) carry the declared Arrow "
        f"schema and column order",
    )
    return (binary, category, anomaly)


#: The row-oriented artifacts, named once.
_TABLE_FILES: Final[frozenset[str]] = frozenset(
    {BINARY_PREDICTION_FILE, CATEGORY_PREDICTION_FILE, ANOMALY_PREDICTION_FILE}
)


def _check_against_manifest(
    recorder: _Recorder,
    manifest: PredictionManifest,
    *,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    anomaly: Sequence[AnomalyScore] | None,
) -> None:
    """Record the checks that compare the rows with what the manifest claims."""
    counts_agree = (
        manifest.row_count == len(binary)
        and manifest.category_row_count == (None if category is None else len(category))
        and manifest.anomaly_row_count == (None if anomaly is None else len(anomaly))
    )
    recorder.verdict(
        "M008",
        counts_agree,
        f"{manifest.row_count:,} declared row(s) match the tables",
        "the manifest's declared row counts and the tables disagree",
    )

    recomputed = prediction_content_fingerprint(
        binary=binary, category=category, anomaly=anomaly
    )
    recorder.verdict(
        "M009",
        recomputed == manifest.prediction_content_fingerprint,
        "the published rows recompute the manifest's content fingerprint",
        "the published rows do not recompute the manifest's content "
        "fingerprint; the artifact and the manifest describe different "
        "predictions",
    )

    lineage = manifest.lineage
    threshold_agrees = all(
        row.decision_threshold == lineage.decision_threshold
        and row.score_kind is lineage.binary_score_kind
        for row in binary
    )
    recorder.verdict(
        "M027",
        threshold_agrees,
        "every row applies the operating point the lineage names",
        "at least one row applies a threshold or score kind the frozen lineage "
        "does not name",
    )

    if category is None:
        recorder.verdict(
            "M028",
            lineage.category_run_id is None,
            "no category head is named, and none was published",
            "the lineage names a frozen category head that was never published",
        )
        return
    expected_order = lineage.category_class_order
    floor = lineage.min_category_score
    mismatched = 0
    for row in category:
        try:
            scores = row.class_scores()
        except ValueError:
            mismatched += 1
            continue
        if tuple(scores) != expected_order or row.min_category_score != floor:
            mismatched += 1
    recorder.verdict(
        "M028",
        mismatched == 0 and lineage.category_run_id is not None,
        "every category row records the frozen class order and abstention floor",
        f"{mismatched:,} category row(s) record a class order or abstention "
        f"floor the frozen head does not name",
    )


def _assert_check_codes_are_declared_once() -> None:
    """Fail at import if a code is declared twice or names nothing."""
    for codes in (STAGED_CHECKS, PUBLISHED_CHECKS):
        if len(set(codes)) != len(codes):
            raise ValueError("a validation check code is declared twice")
        unknown = sorted(set(codes) - set(_CHECK_NAMES))
        if unknown:
            raise ValueError(f"validation check code(s) {unknown} name nothing")
    if not set(STAGED_CHECKS) < set(PUBLISHED_CHECKS):
        raise ValueError(
            "the published check set must strictly contain the staged one; a "
            "published publication is checked at least as thoroughly as a "
            "staged one"
        )
    if set(PUBLISHED_CHECKS) != set(_CHECK_NAMES):
        raise ValueError("a declared check name is not in any check set")


_assert_check_codes_are_declared_once()


def _assert_no_prohibited_column_is_publishable() -> None:
    """Fail at import if a pinned Arrow schema declares a forbidden column."""
    for schema in (
        BINARY_PREDICTION_SCHEMA,
        CATEGORY_PREDICTION_SCHEMA,
        ANOMALY_PREDICTION_SCHEMA,
    ):
        offending = sorted(set(schema.names) & PROHIBITED_PREDICTION_COLUMNS)
        if offending:
            raise ValueError(
                f"a published prediction schema declares prohibited column(s) "
                f"{offending}"
            )


_assert_no_prohibited_column_is_publishable()
