"""Freezing a champion: the last decision made before the test split is opened.

A champion lock is a promise about *what will be evaluated*. Everything a later
test evaluation is permitted to run — the model, the preprocessing state it was
fitted against, the calibrator, the operating point, and the contracts they were
all produced under — is named here by fingerprint, once, before anyone has seen
a test row. That is the entire point: an evaluation whose subject could still
change afterwards would not be an evaluation of anything in particular.

**Freezing verifies again.** The selection already checked every gate; freezing
re-reads the artifacts, re-runs the Milestone 4 verifier, re-derives the
candidate universe from the ledger, and re-checks that every fingerprint in the
lock matches the artifact it names. Trusting the selection record would make the
lock a copy of a claim rather than a check of one, and the two would agree right
up until the run that mattered.

**There is no force.** Not a flag, not an environment variable, not a keyword.
Every refusal below is a state in which promoting a model would mean asserting
something nobody established, and an override would be a way to assert it
anyway.

**One champion per scope.** The scope is the *experiment*: the configuration,
the catalogs, the reviewed feature contract, the validation partition, the
readable lineage, and the acceptance criteria. A second freeze inside one scope
is either the same freeze again — confirmed byte for byte and left alone — or a
contradiction, which is refused. A materially different experiment is a
different scope and gets its own.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, model_validator

from password_attack_detector.exceptions import ExperimentPublicationError
from password_attack_detector.ml.calibration import SealedModel
from password_attack_detector.ml.catalog import MODEL_CATALOG, ModelCatalog
from password_attack_detector.ml.config import MLConfig
from password_attack_detector.ml.dependencies import (
    installed_version,
    sklearn_compatible,
)
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    ChampionStatus,
    ExperimentRecordType,
    MLTask,
    ModelFamily,
)
from password_attack_detector.ml.experiments import MODEL_DIR, RUNS_DIR
from password_attack_detector.ml.ledger import (
    ChampionFreezeRecord,
    ExperimentLedger,
    LedgerAppendResult,
    ValidationSelectionRecord,
)
from password_attack_detector.ml.manifest import verify_model_artifact
from password_attack_detector.ml.schemas import ExperimentRecordIdentity, Sha256Hex
from password_attack_detector.ml.selection import (
    CandidateEvidence,
    champion_candidate_model_ids,
)

__all__ = [
    "CHAMPION_DIR",
    "CHAMPION_LOCK_FILE",
    "FREEZE_SCHEMA_VERSION",
    "ChampionLock",
    "FreezePublication",
    "FrozenCategoryHead",
    "build_champion_lock",
    "freeze_champion",
    "reconcile_freezes",
    "scope_key_for",
]

#: The freeze contract's own version.  Part of the lock's identity: a change to
#: what freezing binds makes a different promise out of the same selection.
FREEZE_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Where locks are published under the artifact root.
CHAMPION_DIR: Final[str] = "champion"
CHAMPION_LOCK_FILE: Final[str] = "champion.lock"


def _digest(payload: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON rendering of *payload*."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def scope_key_for(record: ValidationSelectionRecord) -> str:
    """Return the freeze scope one selection belongs to.

    The **experiment**, not the selection. Two selections that differ only in
    which candidates were available are answers to the same question and must
    not both be frozen; two selections carried out under different
    configurations, catalogs, feature contracts, validation partitions, or
    acceptance criteria are different questions and each may have its own
    champion.

    Deliberately more than ``task | split | labels``: those three collide across
    materially different experiments, and a scope that collides is a scope in
    which one champion silently replaces another.
    """
    return _digest(
        {
            "freeze_schema_version": FREEZE_SCHEMA_VERSION,
            "task": str(record.task),
            "ml_config_fingerprint": record.ml_config_fingerprint,
            "model_catalog_fingerprint": record.model_catalog_fingerprint,
            "gate_config_fingerprint": record.gate_config_fingerprint,
            "validation_partition_fingerprint": (
                record.validation_partition_fingerprint
            ),
            "readable_training_data_fingerprint": (
                record.readable_training_data_fingerprint
            ),
            "readable_label_fingerprint": record.readable_label_fingerprint,
            "readable_split_fingerprint": record.readable_split_fingerprint,
            "feature_catalog_fingerprint": (
                record.identity.feature_catalog_fingerprint
            ),
            "allowlist_fingerprint": record.identity.allowlist_fingerprint,
        }
    )


class FrozenCategoryHead(BaseModel):
    """The known-malicious category head bound alongside a binary champion.

    Present only when a category selection actually found one. Its absence is
    represented by the field being ``None`` rather than by a placeholder head,
    because a binary champion existing is no reason to promote a category model
    nobody selected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_selection_id: str
    training_run_id: str
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    category_abstention_fingerprint: Sha256Hex
    class_order: tuple[str, ...]

    @model_validator(mode="after")
    def check_head(self) -> FrozenCategoryHead:
        """A frozen head names at least two known classes, in a stable order."""
        if len(self.class_order) < 2:
            raise ValueError("a category head distinguishes at least two known classes")
        if len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order repeats a class")
        if tuple(sorted(self.class_order)) != self.class_order:
            raise ValueError(
                "class_order must be the deterministic sorted order it was fitted under"
            )
        return self


class ChampionLock(SealedModel):
    """The frozen champion: every fingerprint a later evaluation may rely on.

    Carries no metric of any kind. Not a validation metric, and certainly not a
    test one: the lock says *which* model was chosen and under what contracts,
    and the selection record beside it says why. Keeping the two apart is what
    stops the lock from becoming a place where a number could later be revised.

    No path, no host, no timestamp. Two freezes of the same selection in two
    directories produce identical bytes.
    """

    fingerprint_field: ClassVar[str] = "lock_fingerprint"
    schema_version_field: ClassVar[str] = "freeze_schema_version"
    schema_version: ClassVar[str] = FREEZE_SCHEMA_VERSION
    record_label: ClassVar[str] = "champion lock"

    freeze_schema_version: str = FREEZE_SCHEMA_VERSION
    scope_key: Sha256Hex
    task: MLTask

    validation_selection_id: str
    validation_selection_fingerprint: Sha256Hex

    training_run_id: str
    catalog_model_id: str
    model_family: ModelFamily
    model_id: str
    model_content_fingerprint: Sha256Hex
    model_manifest_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    class_weight_fingerprint: Sha256Hex | None
    calibration_method: CalibrationMethod
    calibration_state_fingerprint: Sha256Hex | None
    binary_threshold_fingerprint: Sha256Hex

    feature_catalog_fingerprint: Sha256Hex
    allowlist_fingerprint: Sha256Hex
    eligible_feature_list_fingerprint: Sha256Hex
    validation_partition_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    model_catalog_fingerprint: Sha256Hex
    gate_config_fingerprint: Sha256Hex

    serializer_id: str
    serializer_version: int
    inference_adapter_id: str
    #: The reviewed dependency ranges this champion was produced under. A later
    #: evaluation checks the runtime against these before it loads anything:
    #: estimator internals move between minor series, and the arrays in this
    #: model were extracted from one of them.
    dependency_contract_fingerprint: Sha256Hex
    scikit_learn_version: str | None

    category_head: FrozenCategoryHead | None = None
    lock_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_lock(self) -> ChampionLock:
        """A lock describes one champion, coherently, with nothing missing."""
        if self.task is not MLTask.BINARY_MALICIOUS:
            raise ValueError(
                "a champion lock freezes the binary task; the category head is "
                "bound alongside it rather than in place of it"
            )
        if self.model_family is ModelFamily.PRIOR_BASELINE:
            raise ValueError(
                "the reference baseline is never frozen as champion; it is the "
                "comparator every candidate is measured against"
            )
        if self.calibration_method is CalibrationMethod.NONE:
            if self.calibration_state_fingerprint is not None:
                raise ValueError("an uncalibrated champion names no calibrator")
        elif self.calibration_state_fingerprint is None:
            raise ValueError(
                "a calibrated champion names the calibrator that produced its "
                "probabilities"
            )
        if self.serializer_version < 1:
            raise ValueError("a serializer version is a positive integer")
        return self


class FreezePublication(BaseModel):
    """What one freeze did."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_key: str
    record_id: str
    lock_fingerprint: str
    catalog_model_id: str
    #: True when this call wrote the lock; false when an identical one was
    #: already frozen and was left exactly as it was.
    created: bool
    ledger: LedgerAppendResult


# ---------------------------------------------------------------------------
# Building the lock
# ---------------------------------------------------------------------------


def _refuse(message: str) -> None:
    """Refuse a freeze, naming the state that made it impossible."""
    raise ExperimentPublicationError(message)


def build_champion_lock(
    selection: ValidationSelectionRecord,
    *,
    evidence: dict[str, CandidateEvidence],
    config: MLConfig,
    root: Path,
    category: ValidationSelectionRecord | None = None,
    catalog: ModelCatalog = MODEL_CATALOG,
) -> ChampionLock:
    """Return the lock for *selection*, or refuse and say exactly why.

    Every check below is a state in which freezing would assert something
    nobody established. None of them can be waived.

    Raises:
        ExperimentPublicationError: on any refusal.
    """
    if selection.status is not ChampionStatus.ELIGIBLE:
        _refuse(
            f"the selection outcome is {str(selection.status)!r}; a champion is "
            f"frozen only from a selection that found one, and neither a "
            f"measured negative nor an unresolved question becomes a promotion"
        )
    run_id = selection.selected_run_id
    if run_id is None:
        _refuse("the selection names no chosen candidate")
    assert run_id is not None  # narrowed by the refusal above

    result = next(
        (item for item in selection.candidate_results if item.run_id == run_id), None
    )
    if result is None:
        _refuse("the selection's chosen candidate is not among its own results")
    assert result is not None
    if result.status is not ChampionStatus.ELIGIBLE or result.blocking_gates:
        _refuse(
            f"the chosen candidate did not clear every mandatory gate "
            f"({list(result.blocking_gates)}); a gate nobody could measure "
            f"blocks promotion exactly as a failed one does"
        )

    candidate = evidence.get(run_id)
    if candidate is None:
        _refuse("the chosen candidate's published run could not be read back")
    assert candidate is not None

    eligible_ids = set(champion_candidate_model_ids(selection.task, catalog=catalog))
    if candidate.catalog_model_id not in eligible_ids:
        _refuse(
            f"{candidate.catalog_model_id} is not champion-eligible for "
            f"{str(selection.task)!r}; the reviewed catalog decides that, and a "
            f"selection cannot promote outside it"
        )
    spec = catalog.for_family(candidate.run.model_family)
    if spec.reference_baseline:
        _refuse(
            "the reference baseline is never frozen as champion; a model cannot "
            "qualify by beating itself"
        )
    if spec.experimental or spec.anomaly_only:
        _refuse("an experimental or anomaly-only family is never a supervised champion")

    # Re-verified rather than trusted. The selection checked these; a lock built
    # from a stale claim would be a copy of an assertion.
    if not verify_model_artifact(candidate.directory / MODEL_DIR).passed:
        _refuse("the chosen candidate's model artifact does not verify")
    if candidate.threshold is None:
        _refuse("the chosen candidate has no frozen operating point")
    assert candidate.threshold is not None

    calibration_required = config.calibration.method is not CalibrationMethod.NONE
    calibration_fingerprint = candidate.run.identity.calibration_state_fingerprint
    if calibration_required and not spec.reference_baseline:
        if calibration_fingerprint is None:
            _refuse(
                "the configured protocol requires a calibrator and the chosen "
                "candidate has none"
            )
        if candidate.calibration_quality is None:
            _refuse(
                "the chosen candidate has no out-of-sample calibration report; "
                "an in-sample diagnostic is not champion evidence"
            )

    manifest_fingerprint = _manifest_fingerprint(candidate.directory / MODEL_DIR)
    identity = candidate.run.identity
    for name, value in (
        ("model content fingerprint", identity.model_content_fingerprint),
        ("preprocessor fingerprint", identity.preprocessor_fingerprint),
        ("feature catalog fingerprint", identity.feature_catalog_fingerprint),
        ("allowlist fingerprint", identity.allowlist_fingerprint),
        (
            "eligible feature list fingerprint",
            identity.eligible_feature_list_fingerprint,
        ),
        (
            "validation partition fingerprint",
            identity.validation_partition_fingerprint,
        ),
        ("serializer id", identity.serializer_id),
        # Never substituted. A lock that filled this in from something else
        # would name a runtime contract nobody reviewed, and the check a later
        # evaluation performs against it would be a check against a fiction.
        (
            "dependency contract fingerprint",
            identity.dependency_contract_fingerprint,
        ),
    ):
        if value is None:
            _refuse(f"the chosen candidate's run records no {name}")
    if identity.serializer_version is None:
        _refuse("the chosen candidate's run records no serializer version")
    if candidate.run.model_id is None:
        _refuse("the chosen candidate's run records no model identifier")
    if identity.model_content_fingerprint != (
        selection.selected_model_content_fingerprint
    ):
        _refuse(
            "the selection record and the published run disagree about the "
            "chosen model's content"
        )
    if candidate.threshold.model_content_fingerprint != (
        identity.model_content_fingerprint
    ):
        _refuse(
            "the frozen operating point was selected for a different model than "
            "the run publishes"
        )

    resolved = installed_version("scikit-learn")
    if spec.requires_sklearn and not sklearn_compatible(resolved):
        _refuse(
            "the installed scikit-learn lies outside the reviewed range this "
            "champion's arrays were extracted under"
        )

    head = _frozen_category_head(category, evidence=evidence)

    from password_attack_detector.ml.models import adapter_class_for

    adapter = adapter_class_for(candidate.run.model_family)
    return ChampionLock.seal(
        scope_key=scope_key_for(selection),
        task=selection.task,
        validation_selection_id=selection.record_id,
        validation_selection_fingerprint=selection.record_fingerprint,
        training_run_id=run_id,
        catalog_model_id=candidate.catalog_model_id,
        model_family=candidate.run.model_family,
        model_id=candidate.run.model_id,
        model_content_fingerprint=identity.model_content_fingerprint,
        model_manifest_fingerprint=manifest_fingerprint,
        preprocessor_fingerprint=identity.preprocessor_fingerprint,
        class_weight_fingerprint=identity.class_weight_fingerprint,
        calibration_method=candidate.run.calibration_method,
        calibration_state_fingerprint=calibration_fingerprint,
        binary_threshold_fingerprint=candidate.threshold.selection_fingerprint,
        feature_catalog_fingerprint=identity.feature_catalog_fingerprint,
        allowlist_fingerprint=identity.allowlist_fingerprint,
        eligible_feature_list_fingerprint=(identity.eligible_feature_list_fingerprint),
        validation_partition_fingerprint=(identity.validation_partition_fingerprint),
        ml_config_fingerprint=selection.ml_config_fingerprint,
        model_catalog_fingerprint=selection.model_catalog_fingerprint,
        gate_config_fingerprint=selection.gate_config_fingerprint,
        serializer_id=identity.serializer_id,
        serializer_version=identity.serializer_version,
        inference_adapter_id=str(getattr(adapter, "inference_adapter_id", "")),
        dependency_contract_fingerprint=identity.dependency_contract_fingerprint,
        scikit_learn_version=resolved,
        category_head=head,
    )


def _frozen_category_head(
    category: ValidationSelectionRecord | None,
    *,
    evidence: dict[str, CandidateEvidence],
) -> FrozenCategoryHead | None:
    """Return the category head to bind, or ``None`` when none was selected.

    Absence is represented, not filled in. A category selection that found
    nothing leaves the field empty; a later reader can tell that apart from a
    freeze where the question was never asked by looking at whether a category
    selection record exists at all.
    """
    if category is None or category.status is not ChampionStatus.ELIGIBLE:
        return None
    run_id = category.selected_run_id
    if run_id is None:
        return None
    chosen = evidence.get(run_id)
    if chosen is None or chosen.category_abstention is None:
        _refuse(
            "the selected category head's published run or abstention point "
            "could not be read back"
        )
    assert chosen is not None
    assert chosen.category_abstention is not None
    identity = chosen.run.identity
    if identity.model_content_fingerprint is None or (
        identity.preprocessor_fingerprint is None
    ):
        _refuse("the selected category head's run records an incomplete lineage")
    assert identity.model_content_fingerprint is not None
    assert identity.preprocessor_fingerprint is not None
    if chosen.run.model_id is None:
        _refuse("the selected category head's run records no model identifier")
    assert chosen.run.model_id is not None
    return FrozenCategoryHead(
        validation_selection_id=category.record_id,
        training_run_id=run_id,
        model_id=chosen.run.model_id,
        model_content_fingerprint=identity.model_content_fingerprint,
        preprocessor_fingerprint=identity.preprocessor_fingerprint,
        category_abstention_fingerprint=(
            chosen.category_abstention.selection_fingerprint
        ),
        class_order=chosen.category_abstention.class_order,
    )


def _manifest_fingerprint(model_directory: Path) -> str:
    """Return the digest of a published model manifest's bytes."""
    manifest = model_directory / "model_manifest.json"
    if not manifest.is_file():
        _refuse("the chosen candidate's model directory has no manifest")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def freeze_champion(
    selection: ValidationSelectionRecord,
    *,
    evidence: dict[str, CandidateEvidence],
    config: MLConfig,
    root: Path,
    ledger: ExperimentLedger,
    category: ValidationSelectionRecord | None = None,
    catalog: ModelCatalog = MODEL_CATALOG,
) -> FreezePublication:
    """Freeze the champion *selection* chose, transactionally.

    The Milestone 6 ordering: stage, verify, promote atomically, and only then
    append the ledger receipt. A receipt asserting a lock that does not exist
    would need an immutable record deleted to repair; an unindexed valid lock
    needs only to be read.

    Raises:
        ExperimentPublicationError: on any refusal in
            :func:`build_champion_lock`, or when a *different* champion is
            already frozen in this scope. There is no override.
    """
    _require_frozen_candidate_universe(selection, ledger=ledger)
    lock = build_champion_lock(
        selection,
        evidence=evidence,
        config=config,
        root=root,
        category=category,
        catalog=catalog,
    )
    record = _build_freeze_record(lock, selection=selection)

    champion_root = Path(root) / CHAMPION_DIR
    target = champion_root / lock.scope_key
    staging = champion_root / f".staging-{lock.scope_key}"

    if target.exists():
        stored = _require_identical_lock(target, lock)
        return FreezePublication(
            scope_key=stored.scope_key,
            record_id=record.record_id,
            lock_fingerprint=stored.lock_fingerprint,
            catalog_model_id=stored.catalog_model_id,
            created=False,
            ledger=ledger.append(record),
        )

    champion_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    promoted = False
    try:
        (staging / CHAMPION_LOCK_FILE).write_text(
            lock.to_json() + "\n", encoding="utf-8"
        )
        reloaded = ChampionLock.from_json(
            (staging / CHAMPION_LOCK_FILE).read_text(encoding="utf-8")
        )
        if reloaded.to_json() != lock.to_json():
            raise ExperimentPublicationError(
                "the staged champion lock does not read back as itself"
            )
        staging.rename(target)
        promoted = True
    except ExperimentPublicationError:
        raise
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the champion lock could not be staged ({type(exc).__name__}); the "
            f"destination and the ledger are unchanged"
        ) from None
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)

    return FreezePublication(
        scope_key=lock.scope_key,
        record_id=record.record_id,
        lock_fingerprint=lock.lock_fingerprint,
        catalog_model_id=lock.catalog_model_id,
        created=True,
        ledger=ledger.append(record),
    )


def _require_frozen_candidate_universe(
    selection: ValidationSelectionRecord, *, ledger: ExperimentLedger
) -> None:
    """Raise unless every candidate the selection evaluated is still on record.

    The selection is immutable, so its candidate list cannot change -- but the
    *ledger* it was drawn from could have grown a run since. Freezing against a
    universe that no longer matches would freeze a comparison nobody could
    reproduce.
    """
    indexed = {record.record_id for record in ledger.training_runs()}
    missing = sorted(set(selection.candidate_run_ids) - indexed)
    if missing:
        _refuse(
            f"{len(missing)} candidate run(s) named by the selection are absent "
            f"from the ledger; the frozen candidate universe and the recorded "
            f"one disagree"
        )
    if selection.reference_run_id is not None and (
        selection.reference_run_id not in indexed
    ):
        _refuse("the selection's reference baseline run is absent from the ledger")


def _build_freeze_record(
    lock: ChampionLock, *, selection: ValidationSelectionRecord
) -> ChampionFreezeRecord:
    """Return the immutable ledger receipt for freezing *lock*.

    Derived from the lock alone, apart from the binary selection it names. The
    lock already carries the category selection's identifier inside the head it
    bound, so a receipt rebuilt during reconciliation is byte-identical to the
    one the original freeze appended rather than a near-copy missing a field.
    """
    head = lock.category_head
    identity = ExperimentRecordIdentity.derive(
        record_type=ExperimentRecordType.CHAMPION_FREEZE,
        model_catalog_version=selection.identity.model_catalog_version,
        required_feature_schema_version=(
            selection.identity.required_feature_schema_version
        ),
        task=lock.task,
        model_family=lock.model_family,
        catalog_model_id=lock.catalog_model_id,
        seed=selection.identity.seed,
        ml_config_fingerprint=lock.ml_config_fingerprint,
        model_catalog_fingerprint=lock.model_catalog_fingerprint,
        feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
        allowlist_fingerprint=lock.allowlist_fingerprint,
        eligible_feature_list_fingerprint=lock.eligible_feature_list_fingerprint,
        preprocessor_fingerprint=lock.preprocessor_fingerprint,
        class_weight_fingerprint=lock.class_weight_fingerprint,
        validation_partition_fingerprint=lock.validation_partition_fingerprint,
        model_content_fingerprint=lock.model_content_fingerprint,
        calibration_state_fingerprint=lock.calibration_state_fingerprint,
        threshold_selection_fingerprint=lock.binary_threshold_fingerprint,
        category_abstention_fingerprint=(
            None if head is None else head.category_abstention_fingerprint
        ),
        serializer_id=lock.serializer_id,
        serializer_version=lock.serializer_version,
        dependency_contract_fingerprint=lock.dependency_contract_fingerprint,
        # The lock's own digest and the selection it came from. Two freezes of
        # different champions cannot collide, and re-freezing the same one
        # derives the same identifier.
        candidate_fingerprint=_digest(
            {
                "lock_fingerprint": lock.lock_fingerprint,
                "validation_selection_id": selection.record_id,
                "scope_key": lock.scope_key,
            }
        ),
    )
    return ChampionFreezeRecord.seal(
        record_type=ExperimentRecordType.CHAMPION_FREEZE,
        identity=identity,
        freeze_schema_version=FREEZE_SCHEMA_VERSION,
        scope_key=lock.scope_key,
        champion_lock_fingerprint=lock.lock_fingerprint,
        validation_selection_id=selection.record_id,
        selected_run_id=lock.training_run_id,
        selected_model_id=lock.model_id,
        selected_model_content_fingerprint=lock.model_content_fingerprint,
        category_selection_id=None if head is None else head.validation_selection_id,
        category_run_id=None if head is None else head.training_run_id,
        category_model_id=None if head is None else head.model_id,
        ml_config_fingerprint=lock.ml_config_fingerprint,
        model_catalog_fingerprint=lock.model_catalog_fingerprint,
    )


def _require_identical_lock(target: Path, lock: ChampionLock) -> ChampionLock:
    """Return the frozen lock at *target*, or raise if it is a different one."""
    path = target / CHAMPION_LOCK_FILE
    if not path.is_file():
        _refuse(
            "a champion directory already exists in this scope without a lock; "
            "an incomplete freeze is never completed in place"
        )
    try:
        stored = ChampionLock.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the frozen champion lock is not readable ({type(exc).__name__})"
        ) from None
    if stored.to_json() != lock.to_json():
        _refuse(
            "a different champion is already frozen in this scope; a lock is "
            "the promise a later evaluation rests on and is never rewritten"
        )
    return stored


def reconcile_freezes(*, root: Path, ledger: ExperimentLedger) -> tuple[str, ...]:
    """Index every frozen champion the ledger does not yet hold.

    Reads each lock and re-derives the receipt it implies. Unlike a selection or
    a run -- whose records sit beside them -- a freeze receipt is derived from
    the lock plus the selection it names, so reconciliation reads both rather
    than rebuilding either.
    """
    champion_root = Path(root) / CHAMPION_DIR
    if not champion_root.is_dir():
        return ()
    appended: list[str] = []
    for directory in sorted(champion_root.iterdir()):
        path = directory / CHAMPION_LOCK_FILE
        if not directory.is_dir() or not path.is_file():
            continue
        lock = ChampionLock.from_json(path.read_text(encoding="utf-8"))
        selection = ledger.read_selection(lock.validation_selection_id)
        record = _build_freeze_record(lock, selection=selection)
        if ledger.append(record).created:
            appended.append(record.record_id)
    return tuple(appended)


def _assert_runs_dir_is_reachable() -> None:
    """Fail at import if the run layout constant this module reads went missing."""
    if not RUNS_DIR:
        raise ValueError("a champion lock is derived from published runs")


_assert_runs_dir_is_reachable()
