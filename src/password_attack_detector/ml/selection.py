"""Choosing a champion on validation-B, or saying plainly that nobody qualified.

Selection reads **published Milestone 6 run artifacts and nothing else**. It
opens no Parquet table, imports no label reader, and refits nothing: every
number it uses was measured, frozen, and digest-sealed before this module ran.
There is no parameter on any function here through which a test row or a
novel-anomaly holdout row could arrive, and no flag that would widen one.

**The mandatory comparator.** Every binary selection needs one M-000 reference
run from the same experiment lineage. M-000 is fitted, published, and
permanently unpromotable: a candidate qualifies by beating it, and nothing beats
itself. When the reference cannot be established -- missing, invalid, lineage
mismatched, no usable operating point -- the improvement gate is *inconclusive*
for every candidate and the selection fails closed. It is never waived, and the
baseline is never promoted in its own absence.

**Three outcomes, and only one is a winner.**

======================================  =====================================
``eligible``                            one candidate cleared every mandatory
                                        gate, and the ranking chose it
``no_eligible_champion``                every candidate was measured to fail
``insufficient_validation_support``     the question could not be resolved
======================================  =====================================

The precedence between the last two is deliberate. If any candidate's only
blockers are *inconclusive* gates, the selection is unresolved -- that candidate
might have qualified on data this run did not have -- so the answer is
``insufficient_validation_support``. Only when every candidate carries at least
one measured failure is the negative a real finding.

**Ranking never substitutes for a gate.** A candidate outside the gates is not
ranked last; it is not ranked. Ordering happens only among candidates that
already passed everything mandatory, under the objective and tie-break chain the
configuration declares in advance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from password_attack_detector.exceptions import (
    DataValidationError,
    ExperimentPublicationError,
)
from password_attack_detector.ml.calibration import CalibrationReport
from password_attack_detector.ml.catalog import MODEL_CATALOG, ModelCatalog
from password_attack_detector.ml.config import MLConfig
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    ChampionStatus,
    ExperimentRecordType,
    GateStatus,
    MLTask,
    ModelFamily,
    TrainingRunStatus,
)
from password_attack_detector.ml.experiments import (
    CALIBRATION_DIR,
    MODEL_DIR,
    RANKING_DIR,
    RUNS_DIR,
    THRESHOLD_DIR,
    TRAINING_RUN_FILE,
    VALIDATION_RANKING_FILE,
)
from password_attack_detector.ml.gates import (
    BINARY_GATE_IDS,
    CATEGORY_GATE_IDS,
    BinaryGateInputs,
    CategoryGateInputs,
    GateEvidence,
    evaluate_binary_gates,
    evaluate_category_gates,
    gate_config_fingerprint,
)
from password_attack_detector.ml.ledger import (
    CandidateSelectionResult,
    ExperimentLedger,
    LedgerAppendResult,
    TrainingRunRecord,
    ValidationSelectionRecord,
)
from password_attack_detector.ml.manifest import verify_model_artifact
from password_attack_detector.ml.ranking import (
    DISCRIMINATION_SCORE_KIND,
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
    RankingEvidence,
)
from password_attack_detector.ml.schemas import ExperimentRecordIdentity
from password_attack_detector.ml.thresholds import (
    CategoryAbstentionSelection,
    ThresholdSelection,
)

__all__ = [
    "SELECTION_DIR",
    "SELECTION_FILE",
    "SELECTION_SCHEMA_VERSION",
    "CandidateEvidence",
    "SelectionOutcome",
    "SelectionPublication",
    "champion_candidate_model_ids",
    "load_candidate_evidence",
    "publish_selection",
    "select_binary_champion",
    "select_category_head",
    "selection_report_markdown",
]

#: The selection contract's own version.  Part of every selection identity: a
#: change to how candidates are compared makes a different selection out of the
#: same candidates, and two of them must not collide.
SELECTION_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Where selections are published under the artifact root.
SELECTION_DIR: Final[str] = "selections"
SELECTION_FILE: Final[str] = "validation_selection.json"
GATES_JSON_FILE: Final[str] = "ml_gates.json"
GATES_MD_FILE: Final[str] = "ml_gates.md"

#: The calibration report that may serve as champion evidence.  The in-sample
#: fit diagnostic sits beside it in the same directory and is never read here.
_QUALITY_REPORT: Final[str] = "calibration_validation_report.json"


def _digest(payload: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON rendering of *payload*."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def champion_candidate_model_ids(
    task: MLTask, *, catalog: ModelCatalog = MODEL_CATALOG
) -> tuple[str, ...]:
    """Return the catalog entries eligible to become champion for *task*.

    Derived from the executable catalog, never written down twice. A hidden
    second list would be the thing that quietly disagreed with the reviewed one,
    and the disagreement would only surface as a promotion nobody expected.

    The exclusions fall out of the catalog rather than being special-cased here:
    M-000 is a reference baseline and is not champion-eligible, M-021's
    serializer is unproven so it is not champion-eligible either, and M-030
    supports only the anomaly task.
    """
    return tuple(
        sorted(
            spec.model_id
            for spec in catalog.specs
            if spec.champion_eligible and task in spec.supported_tasks
        )
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """One published run, read back and verified, ready to be judged.

    Everything here came off disk and was revalidated against its own digest.
    Nothing was recomputed from rows, because the rows are not available to this
    module and must not be.
    """

    run: TrainingRunRecord
    directory: Path
    threshold: ThresholdSelection | None
    category_abstention: CategoryAbstentionSelection | None
    calibration_quality: CalibrationReport | None
    #: Exact validation-B discrimination evidence, built by Milestone 6 from
    #: every distinct score level. Separate from ``threshold`` on purpose: the
    #: operating point comes off a bounded search and this does not.
    ranking: RankingEvidence | None
    artifact_verified: bool

    @property
    def run_id(self) -> str:
        """Return the run's derived identifier."""
        return self.run.record_id

    @property
    def catalog_model_id(self) -> str:
        """Return the catalog entry this run fitted."""
        return self.run.catalog_model_id

    @property
    def lineage(self) -> tuple[Any, ...]:
        """Return the scope facts every candidate in one selection must share.

        The experiment, not the candidate: configuration, catalogs, feature
        contract, the readable data this task was permitted to see, and the
        validation partition its evidence was measured on. Two runs that differ
        in any of these were not comparing like with like.
        """
        identity = self.run.identity
        return (
            identity.ml_config_fingerprint,
            identity.model_catalog_fingerprint,
            identity.feature_catalog_fingerprint,
            identity.allowlist_fingerprint,
            identity.eligible_feature_list_fingerprint,
            identity.validation_partition_fingerprint,
            identity.readable_training_data_fingerprint,
            identity.readable_label_fingerprint,
            identity.readable_split_fingerprint,
        )


def _read_sealed(path: Path, model: Any) -> Any:
    """Return the sealed artifact at *path*, or ``None`` when it is absent.

    Raises:
        DataValidationError: when the file exists and does not read back as
            itself. A present-but-invalid artifact is a fault, not an absence,
            and treating it as missing would let a corrupted threshold quietly
            become "this candidate has no operating point".
    """
    if not path.is_file():
        return None
    try:
        return model.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataValidationError(
            f"a published selection input is not valid ({type(exc).__name__})"
        ) from None


def load_candidate_evidence(
    root: Path, *, ledger: ExperimentLedger
) -> tuple[CandidateEvidence, ...]:
    """Return every published run under *root*, verified and in identifier order.

    Ordered by run identifier rather than by directory listing, so two machines
    enumerating the same runs produce the same sequence. Selection is not
    permitted to depend on what the filesystem returned first.

    A run the ledger does not hold is skipped: the ledger is the record of what
    was published, and a directory nobody indexed is not evidence. A run the
    ledger holds whose directory is absent is likewise skipped -- it produced no
    model, which is a status Milestone 6 records without writing a directory.
    """
    indexed = {record.record_id: record for record in ledger.training_runs()}
    runs_root = Path(root) / RUNS_DIR
    evidence: list[CandidateEvidence] = []
    for run_id in sorted(indexed):
        directory = runs_root / run_id
        if not (directory / TRAINING_RUN_FILE).is_file():
            continue
        stored = TrainingRunRecord.from_json(
            (directory / TRAINING_RUN_FILE).read_text(encoding="utf-8")
        )
        if stored.to_json() != indexed[run_id].to_json():
            raise DataValidationError(
                "a published run and its ledger record disagree; selection "
                "refuses evidence whose own history contradicts it"
            )
        evidence.append(
            CandidateEvidence(
                run=stored,
                directory=directory,
                threshold=_read_sealed(
                    directory / THRESHOLD_DIR / "binary_threshold.json",
                    ThresholdSelection,
                ),
                category_abstention=_read_sealed(
                    directory / THRESHOLD_DIR / "category_abstention.json",
                    CategoryAbstentionSelection,
                ),
                calibration_quality=_read_sealed(
                    directory / CALIBRATION_DIR / _QUALITY_REPORT,
                    CalibrationReport,
                ),
                ranking=_read_sealed(
                    directory / RANKING_DIR / VALIDATION_RANKING_FILE,
                    RankingEvidence,
                ),
                artifact_verified=verify_model_artifact(directory / MODEL_DIR).passed,
            )
        )
    return tuple(evidence)


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """What one selection decided, and everything it decided it from."""

    task: MLTask
    status: ChampionStatus
    record: ValidationSelectionRecord
    results: tuple[CandidateSelectionResult, ...]
    ranking: tuple[str, ...]
    rationale: tuple[str, ...]
    reference_run_id: str | None

    @property
    def eligible(self) -> bool:
        """Return whether a champion was found."""
        return self.status is ChampionStatus.ELIGIBLE

    @property
    def selected_run_id(self) -> str | None:
        """Return the chosen run, when there is one."""
        return self.record.selected_run_id


def _candidate_status(gates: Sequence[GateEvidence]) -> ChampionStatus:
    """Return whether a candidate cleared every mandatory gate."""
    blocking = [item for item in gates if item.blocking]
    return ChampionStatus.NOT_ELIGIBLE if blocking else ChampionStatus.ELIGIBLE


def _selection_status(
    results: Sequence[CandidateSelectionResult],
    gates_by_run: Mapping[str, tuple[GateEvidence, ...]],
) -> ChampionStatus:
    """Return the selection-level verdict, with unresolved dominating.

    The precedence, and why. A candidate blocked only by *inconclusive* gates
    might have qualified on data this run did not carry, so the selection
    question is open and the honest answer is that it could not be resolved.
    Only when every candidate carries at least one measured failure is a
    negative a finding rather than an absence of evidence.

    An empty candidate universe is a measured negative, not missing support:
    nothing was configured that could have been promoted.
    """
    if any(item.status is ChampionStatus.ELIGIBLE for item in results):
        return ChampionStatus.ELIGIBLE
    if not results:
        return ChampionStatus.NO_ELIGIBLE_CHAMPION
    for item in results:
        blocking = [gate for gate in gates_by_run[item.run_id] if gate.blocking]
        if blocking and all(
            gate.status is GateStatus.INCONCLUSIVE for gate in blocking
        ):
            return ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    return ChampionStatus.NO_ELIGIBLE_CHAMPION


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _ranking_key(
    evidence: CandidateEvidence, gates: Sequence[GateEvidence], config: MLConfig
) -> tuple[Any, ...]:
    """Return the sort key for an eligible candidate, best first.

    Every component is negated or not according to its direction, so a plain
    ascending sort produces the declared order and no comparator has to be read
    twice to know which way it points.
    """
    lookup = {item.gate_id: item for item in gates}
    detection = _observed(lookup, "detection_rate_floor")
    key: list[Any] = [-_or_zero(detection)]
    for criterion in config.selection.tie_break_order:
        match criterion:
            case "min_false_positive_rate":
                key.append(_or_one(_observed(lookup, "false_positive_ceiling")))
            case "max_baseline_pr_auc_gain":
                key.append(-_or_zero(_observed(lookup, "baseline_pr_auc_gain")))
            case "min_expected_calibration_error":
                key.append(_or_one(_observed(lookup, "calibration_quality")))
            case "catalog_model_id":
                key.append(evidence.catalog_model_id)
    return tuple(key)


def _observed(lookup: Mapping[str, GateEvidence], gate_id: str) -> float | None:
    """Return one gate's observed value, or ``None``."""
    gate = lookup.get(gate_id)
    return None if gate is None else gate.observed


def _or_zero(value: float | None) -> float:
    """Return *value*, treating an unmeasured quantity as the worst case.

    Only reachable for a candidate that passed every mandatory gate, so an
    absent value here means a gate passed without an observation -- which the
    gate contract forbids. The fallback exists so ranking is total rather than
    raising in a branch that should not occur.
    """
    return 0.0 if value is None else value


def _or_one(value: float | None) -> float:
    """Return *value*, treating an unmeasured minimised quantity as the worst case."""
    return 1.0 if value is None else value


def _rationale(
    ordered: Sequence[tuple[CandidateEvidence, tuple[Any, ...]]],
    config: MLConfig,
) -> tuple[str, ...]:
    """Return a stable, identity-free explanation of the ranking.

    Codes rather than prose, for the same reason gate reasons are: a sentence
    gets reworded and every assertion about it breaks.
    """
    lines = [
        f"objective={config.selection.ranking_objective}",
        f"tie_break={'>'.join(config.selection.tie_break_order)}",
    ]
    lines.extend(
        f"rank={position + 1} model={evidence.catalog_model_id}"
        for position, (evidence, _) in enumerate(ordered)
    )
    if len(ordered) > 1 and ordered[0][1][:-1] == ordered[1][1][:-1]:
        # The final key is the catalog identifier, so reaching it means every
        # semantic criterion tied. Recorded, because a selection decided by a
        # name is a weaker claim than one decided by a measurement.
        lines.append("tie_resolved_by=catalog_model_id")
    return tuple(lines)


# ---------------------------------------------------------------------------
# Binary selection
# ---------------------------------------------------------------------------


def _lineage_reason(
    candidate: CandidateEvidence, reference: CandidateEvidence | None
) -> tuple[bool, str | None]:
    """Return whether *candidate* shares the reference run's experiment lineage."""
    if reference is None:
        return (True, None)
    if candidate.lineage == reference.lineage:
        return (True, None)
    return (False, "lineage_differs_from_reference_baseline")


def _reference_evidence(
    evidence: Sequence[CandidateEvidence], *, config: MLConfig, task: MLTask
) -> tuple[CandidateEvidence | None, str | None]:
    """Return the mandatory reference run for *task*, or why there is none.

    Exactly one usable reference. Two published M-000 runs in one scope would
    make "the baseline" ambiguous, and a selection that silently picked the
    first would be comparing against whichever the filesystem returned.

    **The readiness contract.** A reference is usable when its published run
    verifies, it carries exact validation-B ranking evidence, that evidence
    declares the same metric and scoring stage every candidate is measured at,
    and its lineage matches the selection scope -- the last checked per
    candidate by the lineage gate rather than here.

    It does **not** require champion eligibility, a calibrator, or a feasible
    operating threshold. A prior-probability baseline scores every row alike, so
    under any false-positive ceiling worth configuring it has no feasible
    threshold and its run stops at ``threshold_unavailable``. That is the
    baseline behaving as a baseline; its discrimination is still exactly
    measurable, and it is still permanently unpromotable. Demanding an operating
    point would make the mandatory comparison unsatisfiable under every shipped
    configuration.

    None of that excuses a malformed comparator. An unverified artifact, missing
    evidence, or evidence measured under a different metric definition each
    leaves the gate with nothing to compare, and the gate then fails closed.
    """
    family = config.gates.baseline_family
    matches = [
        item
        for item in evidence
        if item.run.model_family is family and item.run.task is task
    ]
    if not matches:
        return (None, "reference_baseline_run_missing")
    if len(matches) > 1:
        return (None, "reference_baseline_ambiguous")
    reference = matches[0]
    if not reference.artifact_verified:
        return (None, "reference_baseline_artifact_unverified")
    if reference.ranking is None:
        return (None, "reference_ranking_evidence_unavailable")
    if reference.ranking.score_kind is not DISCRIMINATION_SCORE_KIND or (
        reference.ranking.metric_name != RANKING_METRIC_NAME
    ):
        return (None, "reference_metric_definition_mismatch")
    return (reference, None)


def select_binary_champion(
    evidence: Sequence[CandidateEvidence],
    *,
    config: MLConfig,
    catalog: ModelCatalog = MODEL_CATALOG,
) -> SelectionOutcome:
    """Choose the binary champion from published runs, on validation-B alone.

    Args:
        evidence: published runs, in any order. Sorted internally, so the
            caller's order cannot reach the answer.
        config: the gate thresholds, support floors, and ranking policy.
        catalog: the executable catalog the candidate universe is derived from.

    Returns:
        An outcome carrying every candidate's gate results whatever the verdict.
        A selection that recorded only its winner would be a record of an
        assertion rather than of a comparison.
    """
    task = MLTask.BINARY_MALICIOUS
    eligible_ids = set(champion_candidate_model_ids(task, catalog=catalog))
    reference, reference_reason = _reference_evidence(
        evidence, config=config, task=task
    )

    candidates = sorted(
        (
            item
            for item in evidence
            if item.run.task is task and item.catalog_model_id in eligible_ids
        ),
        key=lambda item: (item.catalog_model_id, item.run_id),
    )

    results: list[CandidateSelectionResult] = []
    gates_by_run: dict[str, tuple[GateEvidence, ...]] = {}
    for candidate in candidates:
        matches, mismatch = _lineage_reason(candidate, reference)
        spec = catalog.for_family(candidate.run.model_family)
        gates = evaluate_binary_gates(
            BinaryGateInputs(
                catalog_model_id=candidate.catalog_model_id,
                threshold=candidate.threshold,
                calibration_quality=candidate.calibration_quality,
                calibration_required=(
                    config.calibration.method is not CalibrationMethod.NONE
                    and not spec.reference_baseline
                ),
                champion_eligible=spec.champion_eligible,
                publishable=candidate.run.status
                is not (TrainingRunStatus.SERIALIZER_UNAVAILABLE),
                artifact_verified=candidate.artifact_verified,
                ranking=candidate.ranking,
                baseline_ranking=None if reference is None else reference.ranking,
                baseline_unavailable_reason=reference_reason,
                lineage_matches=matches,
                lineage_mismatch_reason=mismatch,
            ),
            config=config,
        )
        gates_by_run[candidate.run_id] = gates
        results.append(
            CandidateSelectionResult(
                run_id=candidate.run_id,
                catalog_model_id=candidate.catalog_model_id,
                model_family=candidate.run.model_family,
                model_id=candidate.run.model_id,
                model_content_fingerprint=(
                    candidate.run.identity.model_content_fingerprint
                ),
                status=_candidate_status(gates),
                blocking_gates=tuple(item.gate_id for item in gates if item.blocking),
                gates=gates,
            )
        )

    status = _selection_status(results, gates_by_run)
    ranking: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()
    selected: CandidateEvidence | None = None
    if status is ChampionStatus.ELIGIBLE:
        passing = [
            item
            for item in candidates
            if any(
                result.run_id == item.run_id
                and result.status is ChampionStatus.ELIGIBLE
                for result in results
            )
        ]
        ordered = sorted(
            (
                (item, _ranking_key(item, gates_by_run[item.run_id], config))
                for item in passing
            ),
            key=lambda pair: pair[1],
        )
        ranking = tuple(item.run_id for item, _ in ordered)
        rationale = _rationale(ordered, config)
        selected = ordered[0][0]

    record = _build_selection_record(
        task=task,
        status=status,
        results=results,
        ranking=ranking,
        rationale=rationale,
        candidates=candidates,
        reference=reference,
        selected=selected,
        config=config,
        catalog=catalog,
    )
    return SelectionOutcome(
        task=task,
        status=status,
        record=record,
        results=tuple(results),
        ranking=ranking,
        rationale=rationale,
        reference_run_id=None if reference is None else reference.run_id,
    )


# ---------------------------------------------------------------------------
# Category selection
# ---------------------------------------------------------------------------


def select_category_head(
    evidence: Sequence[CandidateEvidence],
    *,
    config: MLConfig,
    catalog: ModelCatalog = MODEL_CATALOG,
) -> SelectionOutcome:
    """Choose the known-malicious category head, separately from the binary one.

    A separate question with separate evidence. The category head answers
    "which attack" on known-malicious validation-B rows; it has no reference
    baseline to beat and no binary operating point to inherit, and a binary
    champion existing is no reason to invent one.
    """
    task = MLTask.ATTACK_CATEGORY
    eligible_ids = set(champion_candidate_model_ids(task, catalog=catalog))
    candidates = sorted(
        (
            item
            for item in evidence
            if item.run.task is task and item.catalog_model_id in eligible_ids
        ),
        key=lambda item: (item.catalog_model_id, item.run_id),
    )
    scope = candidates[0] if candidates else None

    results: list[CandidateSelectionResult] = []
    gates_by_run: dict[str, tuple[GateEvidence, ...]] = {}
    for candidate in candidates:
        matches = scope is None or candidate.lineage == scope.lineage
        spec = catalog.for_family(candidate.run.model_family)
        gates = evaluate_category_gates(
            CategoryGateInputs(
                catalog_model_id=candidate.catalog_model_id,
                abstention=candidate.category_abstention,
                champion_eligible=spec.champion_eligible,
                publishable=candidate.run.status
                is not (TrainingRunStatus.SERIALIZER_UNAVAILABLE),
                artifact_verified=candidate.artifact_verified,
                lineage_matches=matches,
                lineage_mismatch_reason=None if matches else "lineage_differs",
            ),
            config=config,
        )
        gates_by_run[candidate.run_id] = gates
        results.append(
            CandidateSelectionResult(
                run_id=candidate.run_id,
                catalog_model_id=candidate.catalog_model_id,
                model_family=candidate.run.model_family,
                model_id=candidate.run.model_id,
                model_content_fingerprint=(
                    candidate.run.identity.model_content_fingerprint
                ),
                status=_candidate_status(gates),
                blocking_gates=tuple(item.gate_id for item in gates if item.blocking),
                gates=gates,
            )
        )

    status = _selection_status(results, gates_by_run)
    ranking: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()
    selected: CandidateEvidence | None = None
    if status is ChampionStatus.ELIGIBLE:
        passing = [
            item
            for item in candidates
            if any(
                result.run_id == item.run_id
                and result.status is ChampionStatus.ELIGIBLE
                for result in results
            )
        ]
        ordered = sorted(
            (
                (
                    item,
                    (
                        -_or_zero(
                            item.category_abstention.coverage
                            if item.category_abstention is not None
                            else None
                        ),
                        -_or_zero(
                            item.category_abstention.known_category_precision
                            if item.category_abstention is not None
                            else None
                        ),
                        item.catalog_model_id,
                    ),
                )
                for item in passing
            ),
            key=lambda pair: pair[1],
        )
        ranking = tuple(item.run_id for item, _ in ordered)
        rationale = (
            "objective=max_coverage_at_min_precision",
            "tie_break=max_known_category_precision>catalog_model_id",
            *(
                f"rank={position + 1} model={item.catalog_model_id}"
                for position, (item, _) in enumerate(ordered)
            ),
        )
        selected = ordered[0][0]

    record = _build_selection_record(
        task=task,
        status=status,
        results=results,
        ranking=ranking,
        rationale=rationale,
        candidates=candidates,
        reference=None,
        selected=selected,
        config=config,
        catalog=catalog,
    )
    return SelectionOutcome(
        task=task,
        status=status,
        record=record,
        results=tuple(results),
        ranking=ranking,
        rationale=rationale,
        reference_run_id=None,
    )


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def _build_selection_record(
    *,
    task: MLTask,
    status: ChampionStatus,
    results: Sequence[CandidateSelectionResult],
    ranking: tuple[str, ...],
    rationale: tuple[str, ...],
    candidates: Sequence[CandidateEvidence],
    reference: CandidateEvidence | None,
    selected: CandidateEvidence | None,
    config: MLConfig,
    catalog: ModelCatalog,
) -> ValidationSelectionRecord:
    """Return the immutable record of one selection.

    Identity binds the whole selection *problem*, not just its subject. Two
    selections over different candidate universes, different operating points,
    different calibrators, or different acceptance criteria are different
    selections even when the task and the data are the same -- so all of that
    enters the digest and two of them cannot collide.
    """
    scope = candidates[0] if candidates else reference
    identity = ExperimentRecordIdentity.derive(
        record_type=ExperimentRecordType.VALIDATION_SELECTION,
        model_catalog_version=config.model_catalog_version,
        required_feature_schema_version=config.required_feature_schema_version,
        task=task,
        model_family=(None if selected is None else selected.run.model_family)
        or ModelFamily.PRIOR_BASELINE,
        catalog_model_id=None if selected is None else selected.catalog_model_id,
        seed=config.seed,
        ml_config_fingerprint=config.fingerprint(),
        model_catalog_fingerprint=catalog.fingerprint(),
        feature_catalog_fingerprint=(
            None if scope is None else scope.run.identity.feature_catalog_fingerprint
        ),
        allowlist_fingerprint=(
            None if scope is None else scope.run.identity.allowlist_fingerprint
        ),
        eligible_feature_list_fingerprint=(
            None
            if scope is None
            else scope.run.identity.eligible_feature_list_fingerprint
        ),
        validation_partition_fingerprint=(
            None
            if scope is None
            else scope.run.identity.validation_partition_fingerprint
        ),
        readable_training_data_fingerprint=(
            None
            if scope is None
            else scope.run.identity.readable_training_data_fingerprint
        ),
        readable_label_fingerprint=(
            None if scope is None else scope.run.identity.readable_label_fingerprint
        ),
        readable_split_fingerprint=(
            None if scope is None else scope.run.identity.readable_split_fingerprint
        ),
        # The whole selection problem, digested. The candidate universe, each
        # candidate's frozen operating point and calibrator, the comparator, and
        # the acceptance criteria: change any of them and this is a different
        # question with a different answer.
        candidate_fingerprint=_selection_problem_fingerprint(
            task=task,
            candidates=candidates,
            reference=reference,
            config=config,
        ),
        # Left empty deliberately. A dependency contract describes the runtime a
        # model's arrays were extracted under; a selection fits nothing and so
        # was produced under none. The candidates' contracts are recorded on
        # their own runs, where they are true.
        dependency_contract_fingerprint=None,
    )
    return ValidationSelectionRecord.seal(
        record_type=ExperimentRecordType.VALIDATION_SELECTION,
        identity=identity,
        selection_schema_version=SELECTION_SCHEMA_VERSION,
        task=task,
        status=status,
        candidate_run_ids=tuple(item.run_id for item in candidates),
        candidate_model_fingerprints=tuple(
            item.run.identity.model_content_fingerprint or ("0" * 64)
            for item in candidates
        ),
        reference_run_id=None if reference is None else reference.run_id,
        candidate_results=tuple(results),
        ranking=ranking,
        ranking_rationale=rationale,
        selected_run_id=None if selected is None else selected.run_id,
        selected_model_id=None if selected is None else selected.run.model_id,
        selected_model_content_fingerprint=(
            None
            if selected is None
            else selected.run.identity.model_content_fingerprint
        ),
        validation_partition_fingerprint=(
            scope.run.identity.validation_partition_fingerprint
            if scope is not None
            and scope.run.identity.validation_partition_fingerprint is not None
            else "0" * 64
        ),
        readable_label_fingerprint=(
            None if scope is None else scope.run.identity.readable_label_fingerprint
        ),
        readable_split_fingerprint=(
            None if scope is None else scope.run.identity.readable_split_fingerprint
        ),
        readable_training_data_fingerprint=(
            None
            if scope is None
            else scope.run.identity.readable_training_data_fingerprint
        ),
        discrimination_metric=(
            RANKING_METRIC_NAME if task is MLTask.BINARY_MALICIOUS else None
        ),
        discrimination_integration=(
            PR_AUC_INTEGRATION if task is MLTask.BINARY_MALICIOUS else None
        ),
        discrimination_score_kind=(
            str(DISCRIMINATION_SCORE_KIND) if task is MLTask.BINARY_MALICIOUS else None
        ),
        ml_config_fingerprint=config.fingerprint(),
        model_catalog_fingerprint=catalog.fingerprint(),
        gate_config_fingerprint=gate_config_fingerprint(config),
    )


def _selection_problem_fingerprint(
    *,
    task: MLTask,
    candidates: Sequence[CandidateEvidence],
    reference: CandidateEvidence | None,
    config: MLConfig,
) -> str:
    """Return the digest that distinguishes one selection problem from another.

    Deliberately more than ``task | split | labels``. Two selections over the
    same data and different candidates -- or the same candidates at different
    operating points -- are materially different questions, and an identity that
    collapsed them would let one selection's record answer for the other.
    """
    return _digest(
        {
            "selection_schema_version": SELECTION_SCHEMA_VERSION,
            "task": str(task),
            "gate_config_fingerprint": gate_config_fingerprint(config),
            "discrimination_metric": RANKING_METRIC_NAME,
            "discrimination_integration": PR_AUC_INTEGRATION,
            "reference_run_id": None if reference is None else reference.run_id,
            "reference_ranking_fingerprint": (
                None
                if reference is None or reference.ranking is None
                else reference.ranking.evidence_fingerprint
            ),
            "candidates": [
                {
                    "run_id": item.run_id,
                    "catalog_model_id": item.catalog_model_id,
                    "model_content_fingerprint": (
                        item.run.identity.model_content_fingerprint
                    ),
                    "threshold_fingerprint": (
                        None
                        if item.threshold is None
                        else item.threshold.selection_fingerprint
                    ),
                    "calibrator_fingerprint": (
                        item.run.identity.calibration_state_fingerprint
                    ),
                    "ranking_evidence_fingerprint": (
                        None
                        if item.ranking is None
                        else item.ranking.evidence_fingerprint
                    ),
                    "category_abstention_fingerprint": (
                        None
                        if item.category_abstention is None
                        else item.category_abstention.selection_fingerprint
                    ),
                }
                for item in candidates
            ],
        }
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def selection_report(outcome: SelectionOutcome) -> dict[str, Any]:
    """Return the JSON-ready aggregate report for one selection.

    Counts, rates, statuses, and stable codes. No row, no identifier, no path,
    no coefficient, and no test number -- there is nowhere in this structure one
    could go.
    """
    return {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "task": str(outcome.task),
        "status": str(outcome.status),
        "record_id": outcome.record.record_id,
        "reference_run_id": outcome.reference_run_id,
        "selected_run_id": outcome.selected_run_id,
        "ranking": list(outcome.ranking),
        "ranking_rationale": list(outcome.rationale),
        "candidates": [
            {
                "run_id": item.run_id,
                "catalog_model_id": item.catalog_model_id,
                "model_family": str(item.model_family),
                "model_id": item.model_id,
                "status": str(item.status),
                "blocking_gates": list(item.blocking_gates),
                "ranking_position": (
                    outcome.ranking.index(item.run_id) + 1
                    if item.run_id in outcome.ranking
                    else None
                ),
                "gates": [gate.model_dump(mode="json") for gate in item.gates],
            }
            for item in outcome.results
        ],
    }


def selection_report_markdown(outcome: SelectionOutcome) -> str:
    """Return a deterministic Markdown rendering of one selection."""
    lines = [
        "# Champion gate report",
        "",
        f"Task: `{outcome.task}`",
        f"Outcome: **{outcome.status}**",
        "",
        (
            "Validation-only. No test split and no novel-anomaly holdout was "
            "read, and no figure here describes performance on unseen data."
        ),
        "",
        "## Candidates",
        "",
        "| Model | Status | Blocking gates | Rank |",
        "|---|---|---|---|",
    ]
    for item in outcome.results:
        position = (
            outcome.ranking.index(item.run_id) + 1
            if item.run_id in outcome.ranking
            else None
        )
        lines.append(
            f"| {item.catalog_model_id} | {item.status} | "
            f"{', '.join(item.blocking_gates) or '-'} | "
            f"{position if position is not None else '-'} |"
        )
    lines += ["", "## Gates", ""]
    for item in outcome.results:
        lines += [f"### {item.catalog_model_id}", ""]
        lines += ["| Gate | Status | Observed | Required | Support | Reason |"]
        lines += ["|---|---|---|---|---|---|"]
        for gate in item.gates:
            observed = "unavailable" if gate.observed is None else f"{gate.observed:g}"
            required = "-" if gate.required is None else f"{gate.required:g}"
            support = (
                "-"
                if gate.support_observed is None
                else f"{gate.support_observed:,}"
                + (
                    ""
                    if gate.support_required is None
                    else f"/{gate.support_required:,}"
                )
            )
            lines.append(
                f"| {gate.gate_id} | {gate.status} | {observed} | {required} | "
                f"{support} | {gate.reason} |"
            )
        lines.append("")
    if outcome.eligible:
        lines += [
            "## Selected",
            "",
            f"Run `{outcome.selected_run_id}` cleared every mandatory gate.",
            "",
            "Selection is validation-only. Nothing here says how this model "
            "behaves on data it has not seen.",
            "",
        ]
    else:
        lines += [
            "## No champion",
            "",
            f"Outcome `{outcome.status}`. No candidate was promoted, and the "
            "reference baseline is not a fallback: it is the comparator every "
            "candidate is measured against and is never itself promoted.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


class SelectionPublication(BaseModel):
    """What one selection publication did."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    task: MLTask
    status: ChampionStatus
    created: bool
    ledger: LedgerAppendResult


def publish_selection(
    outcome: SelectionOutcome, *, root: Path, ledger: ExperimentLedger
) -> SelectionPublication:
    """Publish *outcome* under *root* and index it, transactionally.

    The Milestone 6 ordering, unchanged and for the same reason: stage, verify,
    promote atomically, and only then append to the ledger. A ledger asserting a
    selection that does not exist would need an immutable record deleted to
    repair; an unindexed valid selection needs only to be read.

    A negative outcome is published exactly like a positive one. A selection that
    found no champion is a finding, and a history that recorded only successes
    would be a history of successes rather than of what happened.

    Raises:
        ExperimentPublicationError: if staging or promotion fails, or if a
            different selection is already published under this identifier.
    """
    record = outcome.record
    selections_root = Path(root) / SELECTION_DIR
    target = selections_root / record.record_id
    staging = selections_root / f".staging-{record.record_id}"

    if target.exists():
        stored = _require_identical_selection(target, record)
        return SelectionPublication(
            record_id=stored.record_id,
            task=stored.task,
            status=stored.status,
            created=False,
            ledger=ledger.append(stored),
        )

    selections_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    promoted = False
    try:
        (staging / GATES_JSON_FILE).write_text(
            json.dumps(selection_report(outcome), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / GATES_MD_FILE).write_text(
            selection_report_markdown(outcome), encoding="utf-8"
        )
        # Written last: its presence means everything it covers is already there.
        (staging / SELECTION_FILE).write_text(record.to_json() + "\n", encoding="utf-8")
        reloaded = ValidationSelectionRecord.from_json(
            (staging / SELECTION_FILE).read_text(encoding="utf-8")
        )
        if reloaded.to_json() != record.to_json():
            raise ExperimentPublicationError(
                "the staged selection record does not read back as itself"
            )
        staging.rename(target)
        promoted = True
    except ExperimentPublicationError:
        raise
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the selection could not be staged ({type(exc).__name__}); the "
            f"destination and the ledger are unchanged"
        ) from None
    finally:
        if not promoted and staging.exists():
            import shutil

            shutil.rmtree(staging)

    return SelectionPublication(
        record_id=record.record_id,
        task=record.task,
        status=record.status,
        created=True,
        ledger=ledger.append(record),
    )


def _require_identical_selection(
    target: Path, record: ValidationSelectionRecord
) -> ValidationSelectionRecord:
    """Return the published selection at *target*, or raise if it differs."""
    receipt = target / SELECTION_FILE
    if not receipt.is_file():
        raise ExperimentPublicationError(
            "a selection directory already exists here without a selection "
            "record; an incomplete selection is never completed in place"
        )
    try:
        stored = ValidationSelectionRecord.from_json(
            receipt.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the published selection record is not readable ({type(exc).__name__})"
        ) from None
    if stored.to_json() != record.to_json():
        raise ExperimentPublicationError(
            "a different selection is already published under this identifier; "
            "a published selection is evidence and is never overwritten"
        )
    return stored


def reconcile_selections(*, root: Path, ledger: ExperimentLedger) -> tuple[str, ...]:
    """Index every published selection the ledger does not yet hold.

    The Milestone 6 recovery pattern, applied to selections: the record is read
    from the directory it was published with, never rebuilt, and appended.
    Nothing is rewritten and nothing is deleted.
    """
    selections_root = Path(root) / SELECTION_DIR
    if not selections_root.is_dir():
        return ()
    appended: list[str] = []
    for directory in sorted(selections_root.iterdir()):
        receipt = directory / SELECTION_FILE
        if not directory.is_dir() or not receipt.is_file():
            continue
        record = ValidationSelectionRecord.from_json(
            receipt.read_text(encoding="utf-8")
        )
        if ledger.append(record).created:
            appended.append(record.record_id)
    return tuple(appended)


def _assert_gate_sets_are_complete() -> None:
    """Fail at import if a declared gate identifier set is empty."""
    if not BINARY_GATE_IDS or not CATEGORY_GATE_IDS:
        raise ValueError("a selection with no mandatory gates promotes anything")


_assert_gate_sets_are_complete()
