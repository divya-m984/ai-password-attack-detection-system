"""The experiment ledger: append-only, one record per file, never amended.

A ledger whose entries can change is a log, and a log is not evidence. So the
only operation this module offers is **append**. There is no update, no patch,
no upsert, no delete, and no "mark superseded" -- not as a private helper, not
behind a flag. A record's meaning is fixed the moment it is written.

**One record per file, not a JSONL stream.** Appending a line to a shared file
is a torn write waiting to happen: a process interrupted mid-append leaves a
truncated final line, and every later reader has to decide what to do about it.
A record written to its own file, created with ``O_EXCL``, cannot be
half-appended -- either the file exists complete or it does not exist -- and the
exclusive creation *is* the collision check, performed by the filesystem rather
than by a read-then-write race.

The layout, under the ledger root::

    ledger.json                      the ledger's own contract version
    training_run/<record_id>.json    one immutable record
    validation_selection/...         written by a later milestone
    champion_freeze/...              written by a later milestone
    test_evaluation/...              written by a later milestone

Only the directory for a record type actually being written is created. An
empty directory would advertise a record kind this milestone does not produce.

**Idempotency and conflict are different answers.** A record identifier is
derived from the record's own semantic content, so appending the same run twice
offers the same identifier with the same bytes: that is idempotent and quietly
succeeds. Offering the *same* identifier with *different* bytes is two runs
claiming to be one run, and raises
:class:`~password_attack_detector.exceptions.LedgerConflictError`. Nothing is
overwritten and nothing is merged, because both would destroy the earlier
claim in order to record the later one.

**A training run never carries a test metric.** Not an empty one, not a null
one, not one to be filled in later. A test evaluation is its own record type
written after a champion is frozen, and a schema with somewhere to put it would
be a schema somebody eventually fills in early.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import (
    DataValidationError,
    LedgerConflictError,
)
from password_attack_detector.ml.calibration import SealedModel, canonical_json
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    CalibrationStatus,
    ChampionStatus,
    ExperimentRecordType,
    GateStatus,
    MLTask,
    ModelFamily,
    SelectionStatus,
    TrainingRunStatus,
)
from password_attack_detector.ml.gates import GateEvidence
from password_attack_detector.ml.schemas import (
    ExperimentRecordIdentity,
    Sha256Hex,
    prohibited_metadata_fields,
)

__all__ = [
    "LEDGER_FILE",
    "LEDGER_SCHEMA_VERSION",
    "RECORD_TYPE_DIRECTORIES",
    "CandidateSelectionResult",
    "ChampionFreezeRecord",
    "ExperimentLedger",
    "LedgerAppendResult",
    "LedgerRecord",
    "TrainingRunRecord",
    "ValidationSelectionRecord",
]

#: The ledger contract's own version, independent of every artifact schema.
LEDGER_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The file naming the ledger's contract version, written once at the root.
LEDGER_FILE: Final[str] = "ledger.json"

#: Where each record type lives.  A closed mapping: a record type with no entry
#: has nowhere to be written, which is how a future record kind stays absent
#: rather than landing somewhere improvised.
RECORD_TYPE_DIRECTORIES: Final[Mapping[ExperimentRecordType, str]] = {
    ExperimentRecordType.TRAINING_RUN: "training_run",
    ExperimentRecordType.VALIDATION_SELECTION: "validation_selection",
    ExperimentRecordType.CHAMPION_FREEZE: "champion_freeze",
    ExperimentRecordType.TEST_EVALUATION: "test_evaluation",
}

#: Field names **no** ledger record may declare, beyond the project-wide set.
#:
#: ``test_metrics`` heads the list and is the reason the list exists: a record
#: that could hold one would eventually hold one, and a selection or a freeze
#: must stay a record of what was known before the test split was opened.  The
#: observational names are here because identity must not move with the machine
#: that produced it.
PROHIBITED_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "test_metrics",
        "test_evaluation",
        "test_score",
        "holdout_metrics",
        "output_dir",
        "run_directory",
        "hostname",
        "username",
        "temp_dir",
    }
)

#: Field names a **training run** may not declare, on top of the set above.
#:
#: Scoped to training runs rather than to every record, because the distinction
#: is the point: Milestone 6 ranks and promotes nothing, so a training run has
#: nowhere to record a verdict about itself.  A *selection* record legitimately
#: holds a ranking and a champion status -- that is what it is for -- and
#: forbidding the names everywhere would have forced the one record that should
#: carry them to call them something else.
PROHIBITED_TRAINING_RUN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "champion",
        "is_champion",
        "champion_status",
        "rank",
        "ranking",
    }
)

#: The largest record this reader will parse.  A ledger record is a few
#: kilobytes of fingerprints and counts; anything vastly larger is not one, and
#: refusing to read it is cheaper than discovering that after allocating it.
MAX_RECORD_BYTES: Final[int] = 1 << 20


class TrainingRunRecord(SealedModel):
    """One immutable training-run record.

    Identity, status, and the fingerprints binding this run to the artifacts it
    produced. It **binds** those artifacts; it does not restate them. The model
    keeps the identity Milestone 4 gave it, the calibrator and the thresholds
    keep theirs, and this record names them -- so a later milestone reads the
    artifact it wants rather than a copy of it that could disagree.

    No metric of any kind is stored here. Support *counts* are, because a status
    without the support behind it is not checkable, and counts are aggregates.
    """

    fingerprint_field: ClassVar[str] = "record_fingerprint"
    schema_version_field: ClassVar[str] = "ledger_schema_version"
    schema_version: ClassVar[str] = LEDGER_SCHEMA_VERSION
    record_label: ClassVar[str] = "training-run record"

    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    record_type: ExperimentRecordType
    identity: ExperimentRecordIdentity
    status: TrainingRunStatus
    failing_requirements: tuple[str, ...] = ()

    task: MLTask
    model_family: ModelFamily
    catalog_model_id: str
    #: Whether the catalog considers this family promotable *in principle*.
    #: Copied from the reviewed catalog, never decided here -- Milestone 6
    #: promotes nothing, and this is a property of the family rather than a
    #: verdict on the run.
    champion_eligible: bool
    reference_baseline: bool
    experimental: bool

    model_id: str | None = None
    train_row_count: int = Field(ge=0)
    validation_a_row_count: int = Field(ge=0)
    validation_b_row_count: int = Field(ge=0)

    calibration_method: CalibrationMethod = CalibrationMethod.NONE
    calibration_status: CalibrationStatus | None = None
    calibration_quality_admissible: bool | None = None
    binary_threshold_status: SelectionStatus | None = None
    category_abstention_status: SelectionStatus | None = None
    category_threshold_data_selected: bool | None = None
    anomaly_threshold_status: SelectionStatus | None = None

    artifact_digests: tuple[tuple[str, Sha256Hex], ...] = ()
    record_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_record(self) -> Self:
        """Identity, task, and status must describe one coherent run."""
        if self.record_type is not ExperimentRecordType.TRAINING_RUN:
            raise ValueError(
                f"a training-run record has record type "
                f"{str(ExperimentRecordType.TRAINING_RUN)!r}, not "
                f"{str(self.record_type)!r}"
            )
        if self.identity.record_type is not ExperimentRecordType.TRAINING_RUN:
            raise ValueError("the identity describes a different record type")
        if self.identity.run_id != self.identity.derived_run_id():
            raise ValueError(
                "the run identifier is not the one this identity's content "
                "derives; an assigned identifier is not an identity"
            )
        if self.identity.task is not self.task:
            raise ValueError("the record and its identity name different tasks")
        if self.identity.model_family is not self.model_family:
            raise ValueError("the record and its identity name different families")
        if self.identity.catalog_model_id not in (None, self.catalog_model_id):
            raise ValueError(
                "the record and its identity name different catalog entries"
            )

        if self.status is TrainingRunStatus.COMPLETED and self.failing_requirements:
            raise ValueError("a completed run names no failing requirement")
        if self.status is TrainingRunStatus.COMPLETED and self.model_id is None:
            raise ValueError("a completed run names the model it published")

        if self.task is MLTask.ANOMALY:
            if self.calibration_method is not CalibrationMethod.NONE:
                raise ValueError(
                    "an anomaly run carries no calibrator; an unsupervised "
                    "magnitude is never a calibrated probability"
                )
            if self.calibration_status is not None:
                raise ValueError("an anomaly run has no calibration outcome")
            if self.binary_threshold_status is not None:
                raise ValueError("an anomaly run has no binary operating point")
            if self.category_abstention_status is not None:
                raise ValueError("an anomaly run has no category abstention point")
        if self.task is MLTask.BINARY_MALICIOUS and (
            self.category_abstention_status is not None
            or self.anomaly_threshold_status is not None
        ):
            raise ValueError(
                "a binary run carries neither a category abstention point nor "
                "an anomaly threshold; each task publishes only what it means"
            )
        if self.task is MLTask.ATTACK_CATEGORY and (
            self.binary_threshold_status is not None
            or self.anomaly_threshold_status is not None
        ):
            raise ValueError(
                "a category head never inherits the binary operating point"
            )
        if (self.category_threshold_data_selected is not None) != (
            self.category_abstention_status is not None
        ):
            raise ValueError(
                "the abstention provenance flag exists exactly when an "
                "abstention point does"
            )

        names = [name for name, _ in self.artifact_digests]
        if names != sorted(names):
            raise ValueError("artifact digests must be given in sorted path order")
        if len(set(names)) != len(names):
            raise ValueError("artifact digests name a file more than once")
        for name in names:
            if name.startswith("/") or ".." in name.split("/"):
                raise ValueError("an artifact digest names a relative path only")
        return self

    @property
    def run_id(self) -> str:
        """Return the derived run identifier."""
        return self.identity.run_id

    @property
    def record_id(self) -> str:
        """Return the derived record identifier, which is the run identifier."""
        return self.identity.run_id


class CandidateSelectionResult(BaseModel):
    """One candidate's verdict in a selection, gates included.

    Stored on the selection record rather than beside it. A gate result that
    lived in a separate report could be edited without the record noticing,
    and the record's whole value is that its digest covers the reasoning as
    well as the conclusion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    catalog_model_id: str
    model_family: ModelFamily
    model_id: str | None = None
    model_content_fingerprint: Sha256Hex | None = None
    status: ChampionStatus
    #: Stable codes for the mandatory gates this candidate did not clear, in
    #: gate order. Empty exactly when the candidate is eligible.
    blocking_gates: tuple[str, ...] = ()
    gates: tuple[GateEvidence, ...] = ()

    @model_validator(mode="after")
    def check_result(self) -> Self:
        """An eligible candidate cleared every mandatory gate, and says so."""
        if self.status is ChampionStatus.ELIGIBLE and self.blocking_gates:
            raise ValueError("an eligible candidate names no blocking gate")
        if self.status is ChampionStatus.NOT_ELIGIBLE and not self.blocking_gates:
            raise ValueError(
                "a candidate that is not eligible names the gate that stopped it"
            )
        if self.status not in {ChampionStatus.ELIGIBLE, ChampionStatus.NOT_ELIGIBLE}:
            raise ValueError(
                f"a candidate is eligible or not; {str(self.status)!r} is a "
                f"verdict about the selection as a whole"
            )
        names = [item.gate_id for item in self.gates]
        if len(set(names)) != len(names):
            raise ValueError("a gate is reported more than once")
        blocking = {
            item.gate_id
            for item in self.gates
            if item.mandatory and item.status is not GateStatus.PASS
        }
        if blocking != set(self.blocking_gates):
            raise ValueError(
                "blocking_gates must name exactly the mandatory gates that did "
                "not pass; a mandatory gate nobody could measure blocks exactly "
                "as a failed one does"
            )
        return self


class ValidationSelectionRecord(SealedModel):
    """One immutable record of a champion selection carried out on validation-B.

    The complete decision, not its conclusion. Every candidate's gate results
    are here, including the ones that failed and the ones nobody could measure,
    because a record that stored only the winner would be a record of an
    assertion rather than of a comparison.

    Carries no test metric and no holdout metric, and has nowhere to put one. A
    test evaluation is a separate record type written after a champion is
    frozen; a selection record that could later grow a test number would stop
    being a record of what was known at selection time.
    """

    fingerprint_field: ClassVar[str] = "record_fingerprint"
    schema_version_field: ClassVar[str] = "ledger_schema_version"
    schema_version: ClassVar[str] = LEDGER_SCHEMA_VERSION
    record_label: ClassVar[str] = "validation-selection record"

    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    record_type: ExperimentRecordType
    identity: ExperimentRecordIdentity
    selection_schema_version: str
    task: MLTask
    status: ChampionStatus

    #: The candidate universe, in the order it was evaluated. Ordered rather
    #: than a set: the sequence is part of what makes a selection reproducible,
    #: and a set would let two different evaluation orders claim to be one
    #: selection.
    candidate_run_ids: tuple[str, ...]
    candidate_model_fingerprints: tuple[Sha256Hex, ...]
    #: The mandatory comparator's run, when the task has one. Never a candidate.
    reference_run_id: str | None = None

    #: Every candidate's verdict, in candidate order.
    candidate_results: tuple[CandidateSelectionResult, ...]
    #: The eligible candidates in ranked order. Empty unless the status is
    #: ``eligible``: a ranking of candidates that failed their gates would be a
    #: league table of models nobody may use.
    ranking: tuple[str, ...] = ()
    ranking_rationale: tuple[str, ...] = ()

    selected_run_id: str | None = None
    selected_model_id: str | None = None
    selected_model_content_fingerprint: Sha256Hex | None = None

    #: The discrimination metric this selection's mandatory comparison was
    #: decided by, its integration convention, and the scoring stage it was
    #: measured at -- named in the record rather than left to whichever
    #: implementation happened to be linked. A later refactor that changed the
    #: arithmetic without changing the name would be visible here, and a record
    #: written under one definition can never be read as though it were the
    #: other. Present for the binary task, which has a reference comparison, and
    #: absent for the category head, which has none.
    discrimination_metric: str | None = None
    discrimination_integration: str | None = None
    discrimination_score_kind: str | None = None

    validation_partition_fingerprint: Sha256Hex
    readable_label_fingerprint: Sha256Hex | None = None
    readable_split_fingerprint: Sha256Hex | None = None
    readable_training_data_fingerprint: Sha256Hex | None = None
    ml_config_fingerprint: Sha256Hex
    model_catalog_fingerprint: Sha256Hex
    gate_config_fingerprint: Sha256Hex
    record_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_record(self) -> Self:
        """A selection names a winner exactly when it found one."""
        if self.record_type is not ExperimentRecordType.VALIDATION_SELECTION:
            raise ValueError(
                f"a validation-selection record has record type "
                f"{str(ExperimentRecordType.VALIDATION_SELECTION)!r}, not "
                f"{str(self.record_type)!r}"
            )
        if self.identity.record_type is not ExperimentRecordType.VALIDATION_SELECTION:
            raise ValueError("the identity describes a different record type")
        if self.identity.run_id != self.identity.derived_run_id():
            raise ValueError(
                "the selection identifier is not the one this identity's "
                "content derives; an assigned identifier is not an identity"
            )
        if self.identity.task is not self.task:
            raise ValueError("the record and its identity name different tasks")
        if len(self.candidate_run_ids) != len(self.candidate_results):
            raise ValueError("every candidate must carry a result")
        if tuple(item.run_id for item in self.candidate_results) != (
            self.candidate_run_ids
        ):
            raise ValueError("candidate results must be given in candidate order")
        if len(set(self.candidate_run_ids)) != len(self.candidate_run_ids):
            raise ValueError("a candidate appears more than once")

        eligible = self.status is ChampionStatus.ELIGIBLE
        chosen = (
            self.selected_run_id,
            self.selected_model_id,
            self.selected_model_content_fingerprint,
        )
        if eligible and any(value is None for value in chosen):
            raise ValueError("an eligible selection names the candidate it chose")
        if not eligible and any(value is not None for value in chosen):
            raise ValueError(
                f"status {str(self.status)!r} names a selected candidate; only "
                f"an eligible selection has one"
            )
        if not eligible and self.ranking:
            raise ValueError(
                "only an eligible selection publishes a ranking; ordering "
                "candidates that failed their gates would be a league table of "
                "models nobody may use"
            )
        if eligible:
            if not self.ranking:
                raise ValueError("an eligible selection publishes its ranking")
            if self.ranking[0] != self.selected_run_id:
                raise ValueError("the selected candidate must lead the ranking")
            unknown = set(self.ranking) - set(self.candidate_run_ids)
            if unknown:
                raise ValueError("the ranking names a candidate nobody evaluated")
        if self.reference_run_id is not None and (
            self.reference_run_id in self.candidate_run_ids
        ):
            raise ValueError(
                "the reference baseline is never a candidate; a model cannot "
                "qualify by beating itself"
            )
        if self.selected_run_id is not None and (
            self.selected_run_id == self.reference_run_id
        ):
            raise ValueError("the reference baseline is never selected as champion")
        declared = (
            self.discrimination_metric,
            self.discrimination_integration,
            self.discrimination_score_kind,
        )
        if self.task is MLTask.BINARY_MALICIOUS and any(
            value is None for value in declared
        ):
            raise ValueError(
                "a binary selection names the discrimination metric, its "
                "integration convention, and the scoring stage it was measured "
                "at; an unnamed metric is one a later reader has to guess"
            )
        if self.task is not MLTask.BINARY_MALICIOUS and any(
            value is not None for value in declared
        ):
            raise ValueError(
                "only the task with a reference comparison declares a "
                "discrimination metric"
            )
        return self

    @property
    def record_id(self) -> str:
        """Return the derived selection identifier."""
        return self.identity.run_id


class ChampionFreezeRecord(SealedModel):
    """The immutable ledger receipt for freezing a champion.

    The lock is the artifact; this is the record that it happened, and the two
    are separate on purpose. The lock is what a later evaluation loads; the
    receipt is what makes the freeze part of an auditable history that cannot
    be rewritten afterwards.

    Carries no test metric, and never will: a later ``test_evaluation`` record
    will *reference* this frozen state rather than amend it.
    """

    fingerprint_field: ClassVar[str] = "record_fingerprint"
    schema_version_field: ClassVar[str] = "ledger_schema_version"
    schema_version: ClassVar[str] = LEDGER_SCHEMA_VERSION
    record_label: ClassVar[str] = "champion-freeze record"

    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    record_type: ExperimentRecordType
    identity: ExperimentRecordIdentity
    freeze_schema_version: str
    #: The scope this freeze occupies. One frozen champion per scope, and a
    #: second incompatible freeze inside it is refused rather than absorbed.
    scope_key: Sha256Hex
    champion_lock_fingerprint: Sha256Hex

    validation_selection_id: str
    selected_run_id: str
    selected_model_id: str
    selected_model_content_fingerprint: Sha256Hex

    category_selection_id: str | None = None
    category_run_id: str | None = None
    category_model_id: str | None = None

    ml_config_fingerprint: Sha256Hex
    model_catalog_fingerprint: Sha256Hex
    record_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_record(self) -> Self:
        """A freeze receipt describes one frozen champion, coherently."""
        if self.record_type is not ExperimentRecordType.CHAMPION_FREEZE:
            raise ValueError(
                f"a champion-freeze record has record type "
                f"{str(ExperimentRecordType.CHAMPION_FREEZE)!r}, not "
                f"{str(self.record_type)!r}"
            )
        if self.identity.record_type is not ExperimentRecordType.CHAMPION_FREEZE:
            raise ValueError("the identity describes a different record type")
        if self.identity.run_id != self.identity.derived_run_id():
            raise ValueError(
                "the freeze identifier is not the one this identity's content "
                "derives; an assigned identifier is not an identity"
            )
        if (self.category_run_id is None) != (self.category_selection_id is None):
            raise ValueError(
                "a frozen category head names both the run it came from and the "
                "selection that chose it, or neither"
            )
        if (self.category_model_id is None) != (self.category_run_id is None):
            raise ValueError("a frozen category head names its model")
        return self

    @property
    def record_id(self) -> str:
        """Return the derived freeze identifier."""
        return self.identity.run_id


#: Every immutable record kind the ledger stores.
#:
#: A union rather than a shared base class with behaviour: the three records
#: have a sealing contract in common and nothing else, and a common ancestor
#: would invite a field that "every record has" -- which is how a training run
#: eventually grows somewhere to put a test metric.
type LedgerRecord = TrainingRunRecord | ValidationSelectionRecord | ChampionFreezeRecord


class LedgerAppendResult(BaseModel):
    """What an append did, distinguishing a new record from a repeated one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    record_type: ExperimentRecordType
    #: True when this call wrote the record; false when an identical record was
    #: already stored. Both are success, and the caller can tell them apart.
    created: bool


class ExperimentLedger:
    """An append-only, file-backed ledger rooted at one directory.

    Deliberately not a database. A ledger somebody can read with ``cat`` and
    diff with ``git diff`` is a ledger somebody will actually audit, and this
    project's whole argument rests on its artifacts being inspectable without
    the tool that wrote them.
    """

    def __init__(self, root: Path) -> None:
        """Bind the ledger to *root*, creating nothing until something is written."""
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Return the ledger root directory."""
        return self._root

    def directory_for(self, record_type: ExperimentRecordType) -> Path:
        """Return where records of *record_type* are stored.

        Raises:
            DataValidationError: for a record type with no declared directory.
        """
        name = RECORD_TYPE_DIRECTORIES.get(record_type)
        if name is None:
            raise DataValidationError(
                f"record type {str(record_type)!r} has no declared ledger location"
            )
        return self._root / name

    def initialize(self) -> None:
        """Write the ledger's contract file, idempotently.

        Called before the first append and safe to call again: the contents are
        fixed, so a second call either finds the same bytes or discovers that
        something else wrote a different ledger here.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._root / LEDGER_FILE
        payload = canonical_json(
            {
                "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                "record_types": sorted(str(item) for item in RECORD_TYPE_DIRECTORIES),
            }
        )
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if existing != payload:
                raise LedgerConflictError(
                    "the ledger root already holds a different ledger contract; "
                    "a ledger is never rewritten in place"
                )
            return
        _atomic_write(target, payload)

    def append(self, record: LedgerRecord) -> LedgerAppendResult:
        """Append *record*, or confirm an identical one is already stored.

        Three outcomes, and only the third is a failure:

        * the identifier is unused -- the record is written;
        * the identifier is used and the stored bytes are identical -- nothing
          is written and the call succeeds, because appending the same fact
          twice is not a change;
        * the identifier is used and the stored bytes differ -- raises, because
          the two records disagree about what one run was.

        The write itself is exclusive creation: two processes racing on the same
        new identifier cannot both succeed, and the loser falls through to the
        comparison rather than clobbering the winner.

        Raises:
            LedgerConflictError: on the third outcome.
        """
        self.initialize()
        directory = self.directory_for(record.record_type)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{record.record_id}.json"
        payload = record.to_json() + "\n"

        try:
            _exclusive_write(target, payload)
        except FileExistsError:
            existing = target.read_text(encoding="utf-8")
            if existing != payload:
                raise LedgerConflictError(
                    f"ledger record {record.record_id} already exists with "
                    f"different content; the ledger is append-only, so the "
                    f"stored record is neither overwritten nor merged"
                ) from None
            return LedgerAppendResult(
                record_id=record.record_id,
                record_type=record.record_type,
                created=False,
            )
        return LedgerAppendResult(
            record_id=record.record_id, record_type=record.record_type, created=True
        )

    def contains(self, record_id: str, *, record_type: ExperimentRecordType) -> bool:
        """Return whether a record with *record_id* is stored."""
        return (self.directory_for(record_type) / f"{record_id}.json").is_file()

    def read(self, record_id: str) -> TrainingRunRecord:
        """Return the stored training-run record with *record_id*.

        Raises:
            DataValidationError: if the record is absent, oversized, truncated,
                or does not satisfy its own contract.
        """
        return self._read(
            record_id, ExperimentRecordType.TRAINING_RUN, TrainingRunRecord
        )

    def read_selection(self, record_id: str) -> ValidationSelectionRecord:
        """Return the stored validation-selection record with *record_id*.

        Raises:
            DataValidationError: if the record is absent, oversized, truncated,
                or does not satisfy its own contract.
        """
        return self._read(
            record_id,
            ExperimentRecordType.VALIDATION_SELECTION,
            ValidationSelectionRecord,
        )

    def read_freeze(self, record_id: str) -> ChampionFreezeRecord:
        """Return the stored champion-freeze record with *record_id*."""
        return self._read(
            record_id, ExperimentRecordType.CHAMPION_FREEZE, ChampionFreezeRecord
        )

    def _read[RecordT: LedgerRecord](
        self,
        record_id: str,
        record_type: ExperimentRecordType,
        model: type[RecordT],
    ) -> RecordT:
        """Return one stored record of *record_type*, or raise."""
        target = self.directory_for(record_type) / f"{record_id}.json"
        if not target.is_file():
            raise DataValidationError(f"the ledger holds no such {model.record_label}")
        return _read_record(target, model)

    def training_runs(self) -> tuple[TrainingRunRecord, ...]:
        """Return every stored training-run record, in identifier order.

        Sorted by identifier rather than by modification time: two ledgers
        holding the same records enumerate them identically, whatever order the
        filesystem happens to return.

        Raises:
            DataValidationError: on the first malformed or truncated record.
                A ledger that skipped one would be reporting a smaller history
                as a complete one.
        """
        return tuple(
            _read_record(path, TrainingRunRecord) for path in self._record_paths()
        )

    def validation_selections(self) -> tuple[ValidationSelectionRecord, ...]:
        """Return every stored selection record, in identifier order."""
        return tuple(
            _read_record(path, ValidationSelectionRecord)
            for path in self._paths_for(ExperimentRecordType.VALIDATION_SELECTION)
        )

    def champion_freezes(self) -> tuple[ChampionFreezeRecord, ...]:
        """Return every stored champion-freeze record, in identifier order."""
        return tuple(
            _read_record(path, ChampionFreezeRecord)
            for path in self._paths_for(ExperimentRecordType.CHAMPION_FREEZE)
        )

    def iter_training_runs(self) -> Iterator[TrainingRunRecord]:
        """Yield each stored training-run record, in identifier order."""
        for path in self._record_paths():
            yield _read_record(path, TrainingRunRecord)

    def _record_paths(self) -> list[Path]:
        """Return the stored training-run record paths, deterministically ordered."""
        return self._paths_for(ExperimentRecordType.TRAINING_RUN)

    def _paths_for(self, record_type: ExperimentRecordType) -> list[Path]:
        """Return one record type's stored paths, deterministically ordered.

        Sorted by identifier rather than by modification time: two ledgers
        holding the same records enumerate them identically, whatever order the
        filesystem happens to return.
        """
        directory = self.directory_for(record_type)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.json"))


def _read_record[RecordT: LedgerRecord](path: Path, model: type[RecordT]) -> RecordT:
    """Return the record stored at *path*, or raise.

    A truncated file fails to parse; a file that parses but does not recompute
    its own digest fails the seal; either way the ledger refuses it rather than
    returning a partial record that looks whole.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DataValidationError(
            f"cannot read a ledger record ({type(exc).__name__})"
        ) from None
    if size > MAX_RECORD_BYTES:
        raise DataValidationError(
            "a ledger record is larger than any record this contract produces"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise DataValidationError(
            f"cannot read a ledger record ({type(exc).__name__})"
        ) from None
    try:
        record = model.from_json(text)
    except Exception as exc:
        raise DataValidationError(
            f"a ledger record is not valid ({type(exc).__name__})"
        ) from None
    if not isinstance(record, model):
        raise DataValidationError(
            "a ledger record is not the kind its location declares"
        )
    return record


def _exclusive_write(target: Path, payload: str) -> None:
    """Create *target* with *payload*, failing if it already exists.

    ``O_EXCL`` makes the existence check and the creation one operation, so the
    collision check cannot be won by a process that checked first and wrote
    second. The write is flushed and fsynced before the handle closes: a record
    that is visible must be complete, because a later reader has no way to tell
    a short file from a short record.
    """
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _atomic_write(target: Path, payload: str) -> None:
    """Write *payload* to *target* through a temporary sibling and a rename."""
    staging = target.with_name(f".staging-{target.name}")
    try:
        _exclusive_write(staging, payload)
    except FileExistsError:
        staging.unlink()
        _exclusive_write(staging, payload)
    staging.replace(target)


def _assert_no_prohibited_fields() -> None:
    """Fail at import if any ledger record declares a field it must not have."""
    for model in (
        TrainingRunRecord,
        ValidationSelectionRecord,
        ChampionFreezeRecord,
        CandidateSelectionResult,
    ):
        declared = set(model.model_fields)
        banned = PROHIBITED_RECORD_FIELDS | (
            PROHIBITED_TRAINING_RUN_FIELDS
            if model is TrainingRunRecord
            else frozenset()
        )
        offending = prohibited_metadata_fields(list(declared))
        forbidden = sorted(declared & banned)
        if offending or forbidden:
            raise ValueError(
                f"{model.__name__} declares prohibited field(s) "
                f"{sorted({*offending, *forbidden})}"
            )


_assert_no_prohibited_fields()
