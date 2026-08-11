"""Publishing a training run: staged, verified, promoted, then indexed.

A run directory is a claim that a model was fitted under stated conditions and
that everything needed to reproduce or audit it is present. A half-written one
is a claim nobody can check, so a run is built somewhere else entirely and moved
into place only once it is complete.

The order, and why each step is where it is:

1.  build the whole run in a temporary **sibling** directory -- a sibling
    because ``rename`` is atomic only within a filesystem, and a cross-device
    move degrades into a copy that can be interrupted halfway;
2.  write only the artifacts this run's task actually has;
3.  verify the model artifact with the Milestone 4 verifier -- the same checker
    a loader would use, not a lighter one written for publication;
4.  re-read every typed artifact and confirm it parses and recomputes its own
    digest;
5.  build the training-run record and validate it;
6.  write ``training_run.json`` **last**, so its presence means everything it
    covers is already there;
7.  promote the directory atomically;
8.  append to the ledger, which is idempotent, **after** the run is on disk.

**Ledger last, deliberately.** The two orderings fail differently. Ledger-first
can leave the ledger asserting a run that does not exist, and the only repair
would be deleting an immutable record. Run-first can leave a complete, valid,
*unindexed* run -- which is recoverable by reading the run and appending the
record it already contains, without rewriting anything. One failure mode needs
history rewritten; the other needs a directory read. :func:`reconcile` performs
the second.

**An existing run is never overwritten.** A run identifier is derived from the
run's semantics, so a second publication of the same identifier is either the
same run again -- confirmed byte for byte and left alone -- or a contradiction,
which is refused. There is no overwrite flag, because the only thing it could
do is destroy the earlier evidence.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from password_attack_detector.exceptions import ExperimentPublicationError
from password_attack_detector.ml.calibration import CalibrationReport, CalibrationState
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    ExperimentRecordType,
    MLTask,
    ModelFamily,
    SelectionStatus,
    TrainingRunStatus,
)
from password_attack_detector.ml.ledger import (
    ExperimentLedger,
    LedgerAppendResult,
    TrainingRunRecord,
)
from password_attack_detector.ml.manifest import (
    build_model_manifest,
    verify_model_artifact,
)
from password_attack_detector.ml.schemas import ExperimentRecordIdentity
from password_attack_detector.ml.serialization import (
    build_model_document,
    model_id_for,
    stage_model_directory,
)
from password_attack_detector.ml.thresholds import (
    AnomalyThresholdSelection,
    CategoryAbstentionSelection,
    ThresholdSelection,
)
from password_attack_detector.ml.training import TrainingContext, TrainingRunOutcome

__all__ = [
    "CALIBRATION_DIR",
    "MODEL_DIR",
    "RUNS_DIR",
    "THRESHOLD_DIR",
    "TRAINING_RUN_FILE",
    "RunPublication",
    "RunSummary",
    "build_training_run_record",
    "publish_training_run",
    "reconcile",
    "summarize",
]

#: Where runs live under the artifact root.
RUNS_DIR: Final[str] = "runs"

#: Subdirectories of one run.  Created only when the run has something to put
#: in them: an empty ``calibration/`` beside an anomaly run would advertise a
#: calibrator that cannot exist.
MODEL_DIR: Final[str] = "model"
CALIBRATION_DIR: Final[str] = "calibration"
THRESHOLD_DIR: Final[str] = "thresholds"

#: The run receipt, written last.
TRAINING_RUN_FILE: Final[str] = "training_run.json"

CALIBRATION_STATE_FILE: Final[str] = "calibration_state.json"
CALIBRATION_DIAGNOSTIC_FILE: Final[str] = "calibration_fit_diagnostic.json"
CALIBRATION_QUALITY_FILE: Final[str] = "calibration_validation_report.json"
BINARY_THRESHOLD_FILE: Final[str] = "binary_threshold.json"
CATEGORY_ABSTENTION_FILE: Final[str] = "category_abstention.json"
ANOMALY_THRESHOLD_FILE: Final[str] = "anomaly_threshold.json"


def _sha256(payload: bytes) -> str:
    """Return the SHA-256 hex digest of *payload*."""
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# The training-run record
# ---------------------------------------------------------------------------


def build_training_run_record(
    outcome: TrainingRunOutcome,
    *,
    context: TrainingContext,
    artifact_digests: Mapping[str, str] | None = None,
) -> TrainingRunRecord:
    """Return the immutable record describing *outcome*.

    Identity binds the **whole** semantic lineage: what was configured, what
    was admitted, what was fitted, what it was fitted on, and what was selected
    from it. Two runs that differ in any of those are different runs and derive
    different identifiers; two runs that differ only in where they were written
    or when are the same run and derive the same one.

    Nothing observational takes part. No output directory, no checkout path, no
    publication time, no host, and no user -- a record whose identity moved when
    somebody ran it from a different directory would make the ledger useless
    for exactly the question it exists to answer.
    """
    spec = context.catalog.for_family(outcome.candidate.family)
    fitted = outcome.fitted
    validation_a = context.partition.partition_a_row_count
    validation_b = context.partition.partition_b_row_count
    lineage = context.readable_lineage(
        outcome.candidate.task,
        anomaly_reads_validation_a=(
            outcome.anomaly_threshold is not None
            and outcome.anomaly_threshold.source_partition is not None
        ),
    )

    identity = ExperimentRecordIdentity.derive(
        record_type=ExperimentRecordType.TRAINING_RUN,
        model_catalog_version=context.config.model_catalog_version,
        required_feature_schema_version=(
            context.config.required_feature_schema_version
        ),
        task=outcome.candidate.task,
        model_family=outcome.candidate.family,
        catalog_model_id=outcome.candidate.catalog_model_id,
        seed=context.config.seed,
        ml_config_fingerprint=context.config.fingerprint(),
        model_catalog_fingerprint=context.catalog.fingerprint(),
        feature_catalog_fingerprint=context.feature_catalog.fingerprint(),
        # The dataset's own label, split, and training-data digests cover
        # *every* row it holds -- test and novel-anomaly holdout included. A run
        # identity built from them would move whenever somebody added a test
        # row, which is the firewall leaking through the identifier instead of
        # through the fit. They are left unset here rather than published as
        # non-semantic decoration.
        #
        # In their place, three role-scoped digests over exactly the rows this
        # track may read. Three and not one: a reader who sees an identity move
        # should be able to tell a changed feature value from a changed label
        # from a row that entered or left a readable role, because those are
        # three different findings.
        split_config_fingerprint=None,
        label_fingerprint=None,
        training_data_fingerprint=None,
        readable_training_data_fingerprint=lineage.training_data,
        readable_label_fingerprint=lineage.labels,
        readable_split_fingerprint=lineage.split,
        allowlist_fingerprint=context.allowlist_fingerprint,
        eligible_feature_list_fingerprint=(
            context.dataset.eligible_feature_list_fingerprint
        ),
        preprocessor_fingerprint=(
            None if outcome.preprocessor is None else outcome.preprocessor.fingerprint()
        ),
        class_weight_fingerprint=(
            None
            if outcome.class_weights is None
            else outcome.class_weights.fingerprint()
        ),
        validation_partition_fingerprint=context.partition.fingerprint,
        candidate_fingerprint=outcome.candidate.candidate_fingerprint,
        model_content_fingerprint=(
            None if fitted is None else fitted.content_fingerprint()
        ),
        calibration_state_fingerprint=(
            None
            if outcome.calibration_state is None
            else outcome.calibration_state.calibration_state_fingerprint
        ),
        threshold_selection_fingerprint=(
            None
            if outcome.binary_threshold is None
            else outcome.binary_threshold.selection_fingerprint
        ),
        category_abstention_fingerprint=(
            None
            if outcome.category_abstention is None
            else outcome.category_abstention.selection_fingerprint
        ),
        anomaly_threshold_fingerprint=(
            None
            if outcome.anomaly_threshold is None
            else outcome.anomaly_threshold.selection_fingerprint
        ),
        serializer_id=None if fitted is None else fitted.serializer_id,
        serializer_version=None if fitted is None else fitted.serializer_version,
        dependency_contract_fingerprint=context.dependency_contract_fingerprint(),
    )

    return TrainingRunRecord.seal(
        record_type=ExperimentRecordType.TRAINING_RUN,
        identity=identity,
        status=outcome.status,
        failing_requirements=outcome.failing_requirements,
        task=outcome.candidate.task,
        model_family=outcome.candidate.family,
        catalog_model_id=outcome.candidate.catalog_model_id,
        champion_eligible=spec.champion_eligible,
        reference_baseline=spec.reference_baseline,
        experimental=spec.experimental,
        model_id=(
            None
            if fitted is None
            else model_id_for(
                fitted.content_fingerprint(), task=fitted.task, family=fitted.family
            )
        ),
        train_row_count=outcome.train_row_count,
        validation_a_row_count=validation_a,
        validation_b_row_count=validation_b,
        calibration_method=(
            CalibrationMethod.NONE
            if outcome.calibration_state is None
            else outcome.calibration_state.method
        ),
        calibration_status=(
            None if outcome.calibration is None else outcome.calibration.status
        ),
        calibration_quality_admissible=(
            None
            if outcome.calibration_quality is None
            else outcome.calibration_quality.admissible_as_champion_evidence
        ),
        binary_threshold_status=(
            None
            if outcome.binary_threshold is None
            else outcome.binary_threshold.status
        ),
        category_abstention_status=(
            None
            if outcome.category_abstention is None
            else outcome.category_abstention.status
        ),
        category_threshold_data_selected=(
            None
            if outcome.category_abstention is None
            else outcome.category_abstention.data_selected
        ),
        anomaly_threshold_status=(
            None
            if outcome.anomaly_threshold is None
            else outcome.anomaly_threshold.status
        ),
        artifact_digests=tuple(sorted((artifact_digests or {}).items())),
    )


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


class RunPublication(BaseModel):
    """What one publication did."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    status: TrainingRunStatus
    task: MLTask
    model_family: ModelFamily
    catalog_model_id: str
    #: True when this call wrote the run directory; false when an identical run
    #: was already published and was left exactly as it was.
    created: bool
    ledger: LedgerAppendResult


def publish_training_run(
    outcome: TrainingRunOutcome,
    *,
    context: TrainingContext,
    root: Path,
    ledger: ExperimentLedger,
) -> RunPublication:
    """Publish *outcome* under *root* and index it, transactionally.

    Args:
        outcome: what the candidate produced. A run with no fitted model is not
            published as a directory -- there is nothing to put in one -- but it
            is still recorded in the ledger, so a candidate that could not be
            trained is visible rather than absent.
        context: the frozen inputs the run was carried out under.
        root: the artifact root; runs are written under ``<root>/runs``.
        ledger: the append-only ledger to index into.

    Raises:
        ExperimentPublicationError: if staging, verification, or promotion
            fails, or if a different run is already published under this
            identifier. The destination and the ledger are untouched.
        LedgerConflictError: if the ledger already holds a different record for
            this identifier. The published run is left in place; the ledger is
            not rewritten.
    """
    record = build_training_run_record(outcome, context=context)
    if outcome.fitted is None:
        # Nothing to write, and still something to record: a candidate that
        # never produced a model is part of the run history.
        return RunPublication(
            run_id=record.run_id,
            status=record.status,
            task=record.task,
            model_family=record.model_family,
            catalog_model_id=record.catalog_model_id,
            created=False,
            ledger=ledger.append(record),
        )

    runs_root = Path(root) / RUNS_DIR
    target = runs_root / record.run_id
    staging = runs_root / f".staging-{record.run_id}"

    if target.exists():
        # The published run is the authority on its own content, digests
        # included. Appending a freshly built record here would offer the
        # ledger a record with no artifact digests and manufacture a conflict
        # out of a repeat publication.
        stored = _require_identical(target, outcome, context=context)
        return RunPublication(
            run_id=stored.run_id,
            status=stored.status,
            task=stored.task,
            model_family=stored.model_family,
            catalog_model_id=stored.catalog_model_id,
            created=False,
            ledger=ledger.append(stored),
        )

    runs_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    promoted = False
    try:
        digests = _stage_run(staging, outcome, context=context)
        final_record = build_training_run_record(
            outcome, context=context, artifact_digests=digests
        )
        (staging / TRAINING_RUN_FILE).write_text(
            final_record.to_json() + "\n", encoding="utf-8"
        )
        _verify_staged_run(staging, outcome)
        staging.rename(target)
        promoted = True
    except ExperimentPublicationError:
        raise
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the training run could not be staged ({type(exc).__name__}); the "
            f"destination and the ledger are unchanged"
        ) from None
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)

    # The ledger is appended only now: a record asserting a run that does not
    # exist would need an immutable record deleted to repair, and an unindexed
    # valid run needs only to be read.
    return RunPublication(
        run_id=final_record.run_id,
        status=final_record.status,
        task=final_record.task,
        model_family=final_record.model_family,
        catalog_model_id=final_record.catalog_model_id,
        created=True,
        ledger=ledger.append(final_record),
    )


def _stage_run(
    staging: Path, outcome: TrainingRunOutcome, *, context: TrainingContext
) -> dict[str, str]:
    """Write every applicable artifact into *staging* and return their digests.

    Only what the task means. A binary run has no abstention point to write, an
    anomaly run has no calibrator, and a category run has no binary threshold --
    and none of them gets an empty file standing in for one, because a reader
    finding an empty artifact has to guess whether it is absent or broken.
    """
    assert outcome.fitted is not None  # guaranteed by the caller
    assert outcome.preprocessor is not None  # fitted and preprocessor travel together
    fitted = outcome.fitted
    digests: dict[str, str] = {}

    model_directory = staging / MODEL_DIR
    model_directory.mkdir(parents=True)
    # The model's *own* task-specific preprocessing state. Staging refuses a
    # mismatch, so pairing a category head with the binary track's encoder
    # would fail here rather than publish a directory whose fingerprints
    # disagree with each other.
    staged = stage_model_directory(
        model_directory, fitted=fitted, preprocessor=outcome.preprocessor
    )
    sizes = {name: (model_directory / name).stat().st_size for name in staged}
    manifest = build_model_manifest(
        document=build_model_document(fitted),
        digests=staged,
        sizes=sizes,
        seed=context.config.seed,
        required_feature_schema_version=context.config.required_feature_schema_version,
        model_catalog_fingerprint=context.catalog.fingerprint(),
        feature_catalog_fingerprint=context.feature_catalog.fingerprint(),
        allowlist_fingerprint=context.allowlist_fingerprint,
        split_config_fingerprint=context.dataset.split_fingerprint,
        training_data_fingerprint=context.dataset.training_data_fingerprint,
        ml_config_fingerprint=context.config.fingerprint(),
        # No publication timestamp. It is observational, it is excluded from
        # every fingerprint anyway, and leaving it out is what makes two runs
        # of the same semantics byte-identical rather than merely equivalent.
        published_at=None,
    )
    from password_attack_detector.ml.serialization import MANIFEST_FILE

    (model_directory / MANIFEST_FILE).write_text(manifest.to_json(), encoding="utf-8")
    for name in sorted({*staged, MANIFEST_FILE}):
        digests[f"{MODEL_DIR}/{name}"] = _sha256((model_directory / name).read_bytes())

    calibration_artifacts: list[tuple[str, Any]] = []
    if outcome.calibration_state is not None:
        calibration_artifacts.append(
            (CALIBRATION_STATE_FILE, outcome.calibration_state)
        )
    if outcome.calibration_diagnostic is not None:
        calibration_artifacts.append(
            (CALIBRATION_DIAGNOSTIC_FILE, outcome.calibration_diagnostic)
        )
    if outcome.calibration_quality is not None:
        calibration_artifacts.append(
            (CALIBRATION_QUALITY_FILE, outcome.calibration_quality)
        )
    digests |= _write_records(
        staging / CALIBRATION_DIR, CALIBRATION_DIR, calibration_artifacts
    )

    threshold_artifacts: list[tuple[str, Any]] = []
    if outcome.binary_threshold is not None:
        threshold_artifacts.append((BINARY_THRESHOLD_FILE, outcome.binary_threshold))
    if outcome.category_abstention is not None:
        threshold_artifacts.append(
            (CATEGORY_ABSTENTION_FILE, outcome.category_abstention)
        )
    if outcome.anomaly_threshold is not None:
        threshold_artifacts.append((ANOMALY_THRESHOLD_FILE, outcome.anomaly_threshold))
    digests |= _write_records(
        staging / THRESHOLD_DIR, THRESHOLD_DIR, threshold_artifacts
    )
    return digests


def _write_records(
    directory: Path, prefix: str, records: Sequence[tuple[str, Any]]
) -> dict[str, str]:
    """Write each sealed record and return its digest, creating nothing if empty."""
    if not records:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, record in records:
        payload = record.to_json() + "\n"
        (directory / name).write_text(payload, encoding="utf-8")
        digests[f"{prefix}/{name}"] = _sha256(payload.encode())
    return digests


def _verify_staged_run(staging: Path, outcome: TrainingRunOutcome) -> None:
    """Raise unless the staged run is complete, verified, and self-consistent.

    The model directory goes through the Milestone 4 verifier -- the same one a
    loader uses, so publication cannot pass a check that loading would fail.
    Every typed artifact is then read back from the bytes on disk and revalidated,
    which is what proves the file is the record rather than something that
    merely looked like it in memory.
    """
    outcome_of = verify_model_artifact(staging / MODEL_DIR)
    if not outcome_of.passed:
        raise ExperimentPublicationError(
            f"the staged model artifact failed verification "
            f"[{outcome_of.error_code}]: {outcome_of.error_detail}"
        )

    expected: list[tuple[Path, Any]] = []
    if outcome.calibration_state is not None:
        expected.append(
            (staging / CALIBRATION_DIR / CALIBRATION_STATE_FILE, CalibrationState)
        )
    if outcome.calibration_diagnostic is not None:
        expected.append(
            (
                staging / CALIBRATION_DIR / CALIBRATION_DIAGNOSTIC_FILE,
                CalibrationReport,
            )
        )
    if outcome.calibration_quality is not None:
        expected.append(
            (staging / CALIBRATION_DIR / CALIBRATION_QUALITY_FILE, CalibrationReport)
        )
    if outcome.binary_threshold is not None:
        expected.append(
            (staging / THRESHOLD_DIR / BINARY_THRESHOLD_FILE, ThresholdSelection)
        )
    if outcome.category_abstention is not None:
        expected.append(
            (
                staging / THRESHOLD_DIR / CATEGORY_ABSTENTION_FILE,
                CategoryAbstentionSelection,
            )
        )
    if outcome.anomaly_threshold is not None:
        expected.append(
            (
                staging / THRESHOLD_DIR / ANOMALY_THRESHOLD_FILE,
                AnomalyThresholdSelection,
            )
        )
    for path, model in expected:
        if not path.is_file():
            raise ExperimentPublicationError(
                "the staged run is missing an artifact its task requires"
            )
        try:
            model.from_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ExperimentPublicationError(
                f"a staged run artifact does not read back as itself "
                f"({type(exc).__name__})"
            ) from None

    receipt = staging / TRAINING_RUN_FILE
    if not receipt.is_file():
        raise ExperimentPublicationError("the staged run has no training-run record")
    try:
        TrainingRunRecord.from_json(receipt.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the staged training-run record is not valid ({type(exc).__name__})"
        ) from None


def _require_identical(
    target: Path, outcome: TrainingRunOutcome, *, context: TrainingContext
) -> TrainingRunRecord:
    """Return the published record at *target*, or raise if it is a different run.

    Compared on the record's canonical bytes rather than on the identifier
    alone: two runs sharing an identifier and disagreeing about their content
    is the case worth catching, and it is the only case an identifier check
    would miss.
    """
    receipt = target / TRAINING_RUN_FILE
    if not receipt.is_file():
        raise ExperimentPublicationError(
            "a run directory already exists here without a training-run record; "
            "an incomplete run is never completed in place"
        )
    try:
        stored = TrainingRunRecord.from_json(receipt.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the published run's record is not readable ({type(exc).__name__})"
        ) from None
    fresh = build_training_run_record(
        outcome, context=context, artifact_digests=dict(stored.artifact_digests)
    )
    if stored.to_json() != fresh.to_json():
        raise ExperimentPublicationError(
            "a different run is already published under this identifier; a "
            "published run is evidence and is never overwritten"
        )
    return stored


def reconcile(*, root: Path, ledger: ExperimentLedger) -> tuple[str, ...]:
    """Index every published run the ledger does not yet hold.

    The deterministic recovery for the one failure this design permits: a run
    that was promoted and then not indexed. The record is read from the run
    directory it was published with -- not rebuilt, not inferred -- and appended.

    Nothing is rewritten. A run already indexed is skipped; a run whose stored
    record disagrees with the ledger raises rather than being reconciled, since
    that is a contradiction and not a gap.

    Returns:
        The identifiers newly appended, in ascending order.
    """
    runs_root = Path(root) / RUNS_DIR
    if not runs_root.is_dir():
        return ()
    appended: list[str] = []
    for directory in sorted(runs_root.iterdir()):
        receipt = directory / TRAINING_RUN_FILE
        if not directory.is_dir() or not receipt.is_file():
            continue
        record = TrainingRunRecord.from_json(receipt.read_text(encoding="utf-8"))
        if ledger.append(record).created:
            appended.append(record.run_id)
    return tuple(appended)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One row of the experiment listing: statuses and identities, no metrics.

    Deliberately carries no validation number. Milestone 6 records what was
    run, not how well it did, and a listing that showed a score would be a
    ranking whatever it was called.
    """

    run_id: str
    record_type: ExperimentRecordType
    task: MLTask
    catalog_model_id: str
    model_family: ModelFamily
    model_id: str | None
    status: TrainingRunStatus
    calibration_method: CalibrationMethod
    calibration_status: str
    threshold_status: str
    reference_baseline: bool
    experimental: bool
    champion_eligible: bool


def summarize(record: TrainingRunRecord) -> RunSummary:
    """Return the inspection view of *record*."""
    threshold: SelectionStatus | None = (
        record.binary_threshold_status
        or record.category_abstention_status
        or record.anomaly_threshold_status
    )
    return RunSummary(
        run_id=record.run_id,
        record_type=record.record_type,
        task=record.task,
        catalog_model_id=record.catalog_model_id,
        model_family=record.model_family,
        model_id=record.model_id,
        status=record.status,
        calibration_method=record.calibration_method,
        calibration_status=(
            "not_applicable"
            if record.calibration_status is None
            else str(record.calibration_status)
        ),
        threshold_status=("not_applicable" if threshold is None else str(threshold)),
        reference_baseline=record.reference_baseline,
        experimental=record.experimental,
        champion_eligible=record.champion_eligible,
    )
