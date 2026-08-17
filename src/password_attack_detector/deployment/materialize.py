"""Reconstructing the frozen stacked hybrid, offline, and refusing to guess.

Phase 5 fits the stacked meta-learner inside ``ml evaluate``, seals its
fingerprint into the frozen :class:`~password_attack_detector.ml.fusion.\
FusionSelection`, and publishes the selection in the locked evaluation.  What it
does not publish is the fitted state.  This module produces it again -- and then
proves that what it produced *is* what Phase 5 fitted, by recomputing the digest
and comparing it with the sealed one.

**The reconstruction reads only pre-TEST lineage.**  It is the same orchestration
the original fit ran, reached through the same entry point:
:func:`~password_attack_detector.ml.stacking.prepare_fusion_selection`, which has
no TEST parameter and refuses at import to grow one.  The out-of-fold
meta-features come from TRAIN rows scored by models refitted without them; the
selection evidence comes from validation-B.  The TEST ground-truth reader is
gated behind a proof object nothing here constructs, and mutating the TEST labels
leaves the reconstructed state byte-identical -- which is a test rather than a
claim.

**Nothing is re-decided.**  There is no reselection: the strategy is read from
the frozen receipt and the reconstruction is refused if the recomputed selection
is not the frozen one.  There is no new fusion threshold: the stacker's decision
threshold is a declared constant in the fusion contract.  There is no fallback:
a mismatch, a missing input, or an unreadable receipt each stop the publication
rather than substituting a strategy nobody selected.

**Nothing is fitted at serving time.**  This module is invoked by an operator
through ``password-attack-detector deploy materialize``, writes an artifact, and
exits.  The serving runtime imports :mod:`~password_attack_detector.deployment.\
bundle` to *load* that artifact and never imports this module at all; the API's
own import-time guard refuses a fitting function in its namespace.

The order of checks below is deliberate, cheapest and most decisive first:

1.  the locked evaluation receipt for this champion is found and sealed;
2.  its ``system_comparison.json`` digests to what the receipt recorded;
3.  the frozen ``FusionSelection`` inside it recomputes its own seal and its
    fingerprint is the one the receipt named;
4.  the selection selected a strategy, and named the champion and the rule
    configuration this run was given;
5.  only then is anything refitted -- and for a boolean gate, nothing is.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from password_attack_detector.deployment.bundle import (
    ServingBundleManifest,
    bundle_files,
    write_serving_bundle,
)
from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    DataValidationError,
    ManifestVerificationError,
    ModelNotReadyError,
    PasswordAttackDetectorError,
)
from password_attack_detector.features.catalog import FeatureCatalog
from password_attack_detector.ml.config import MLConfig
from password_attack_detector.ml.dataset import MLDataset
from password_attack_detector.ml.enums import (
    FusionStrategy,
    MLSplit,
    SelectionStatus,
    ValidationPartition,
)
from password_attack_detector.ml.features import EligibleFeatureList
from password_attack_detector.ml.fusion import (
    FusionSelection,
    MLEvidence,
    RuleEvidence,
    StackedFusionState,
)
from password_attack_detector.ml.ledger import TestEvaluationRecord
from password_attack_detector.ml.partition import partition_validation
from password_attack_detector.ml.predictions import BinaryPrediction, FrozenChampion
from password_attack_detector.ml.stacking import (
    FusionPreparation,
    prepare_fusion_selection,
)
from password_attack_detector.ml.test_evaluation import (
    EVALUATION_RECEIPT_FILE,
    EVALUATIONS_DIR,
    SYSTEM_COMPARISON_JSON,
)
from password_attack_detector.ml.training import TrainingContext

__all__ = [
    "FrozenFusionEvidence",
    "MaterializationOutcome",
    "build_bundle_manifest",
    "materialize_serving_bundle",
    "read_frozen_fusion",
    "reconstruct_stacked_state",
]

#: The half of the validation split a selection may be made on.  Read from the
#: enum rather than restated, and identical to what ``ml evaluate`` used.
_SELECTION_HALF: Final[ValidationPartition] = ValidationPartition.VALIDATION_B


@dataclass(frozen=True, slots=True)
class FrozenFusionEvidence:
    """The frozen hybrid selection, read out of a locked TEST evaluation.

    Read, verified, and never amended.  The record is the authority on *which*
    strategy is deployable; the selection is the authority on *which stacker*
    that strategy applies.
    """

    record: TestEvaluationRecord
    selection: FusionSelection
    record_id: str

    @property
    def strategy(self) -> FusionStrategy:
        """Return the frozen strategy.  Present by construction."""
        assert self.selection.selected_strategy is not None  # checked on read
        return self.selection.selected_strategy

    @property
    def stacked(self) -> bool:
        """Return whether this selection needs a fitted state to be servable."""
        return self.strategy is FusionStrategy.STACKED


@dataclass(frozen=True, slots=True)
class MaterializationOutcome:
    """What one materialization did, and what it refused to do."""

    directory: Path | None
    created: bool
    strategy: FusionStrategy | None
    #: The digest Phase 5 sealed for the selected stacker, when there is one.
    frozen_state_fingerprint: str | None
    #: The digest this run's reconstruction derived, when it got that far.
    reconstructed_state_fingerprint: str | None
    manifest: ServingBundleManifest | None
    #: A stable reason code when nothing was published.  ``None`` on success.
    refusal: str | None

    @property
    def published(self) -> bool:
        """Return whether a verified bundle now exists."""
        return self.refusal is None and self.directory is not None

    @property
    def fingerprints_agree(self) -> bool:
        """Return whether the reconstruction recomputed the frozen digest."""
        return (
            self.frozen_state_fingerprint is not None
            and self.frozen_state_fingerprint == self.reconstructed_state_fingerprint
        )


def read_frozen_fusion(
    root: Path, *, champion_lock_fingerprint: str
) -> FrozenFusionEvidence:
    """Return the frozen fusion selection for one champion, fully verified.

    Reads the locked TEST evaluations under *root*, keeps the ones belonging to
    this champion, and refuses anything ambiguous.  The selection document is
    reached through the receipt's own recorded report digest, so a
    ``system_comparison.json`` edited after publication is refused rather than
    parsed.

    Raises:
        ArtifactNotFoundError: no locked evaluation for this champion selected a
            hybrid strategy.
        ManifestVerificationError: a receipt or a selection did not verify, or two
            receipts for one champion name different strategies.
    """
    receipts_root = Path(root) / EVALUATIONS_DIR
    if not receipts_root.is_dir():
        raise ArtifactNotFoundError(
            "this artifact root holds no locked TEST evaluation, so nothing has "
            "selected a hybrid strategy to deploy"
        )

    found: list[FrozenFusionEvidence] = []
    for directory in sorted(receipts_root.iterdir()):
        receipt = directory / EVALUATION_RECEIPT_FILE
        if not directory.is_dir() or not receipt.is_file():
            continue
        try:
            record = TestEvaluationRecord.from_json(receipt.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ManifestVerificationError(
                f"a locked evaluation receipt is not readable "
                f"({type(exc).__name__}); a hybrid is never deployed from a "
                f"receipt that does not verify"
            ) from None
        if record.champion_lock_fingerprint != champion_lock_fingerprint:
            continue
        if record.selected_fusion_strategy is None:
            continue
        found.append(
            FrozenFusionEvidence(
                record=record,
                selection=_selection_from(directory, record),
                record_id=record.record_id,
            )
        )

    if not found:
        raise ArtifactNotFoundError(
            "no locked TEST evaluation for this champion selected a hybrid "
            "strategy; there is no frozen hybrid to make deployable"
        )
    strategies = {item.strategy for item in found}
    if len(strategies) > 1:
        raise ManifestVerificationError(
            "two locked evaluations for this champion name different hybrid "
            "strategies; that is not a tie to break, it is a lineage no verdict "
            "can be attributed to"
        )
    fingerprints = {item.selection.selection_fingerprint for item in found}
    if len(fingerprints) > 1:
        raise ManifestVerificationError(
            "two locked evaluations for this champion name different fusion "
            "selections; one of them is not the selection this deployment would "
            "be serving"
        )
    return found[0]


def _selection_from(directory: Path, record: TestEvaluationRecord) -> FusionSelection:
    """Return the frozen selection this evaluation published, or raise.

    The report is located by the receipt's own file index and checked against the
    digest the receipt recorded, so the chain from sealed receipt to fusion
    selection has no unverified link in it.
    """
    recorded = dict(record.report_fingerprints).get(SYSTEM_COMPARISON_JSON)
    if recorded is None:
        raise ManifestVerificationError(
            "the locked evaluation receipt records no system-comparison report, "
            "so the frozen fusion selection cannot be read from it"
        )
    path = directory / SYSTEM_COMPARISON_JSON
    if not path.is_file():
        raise ManifestVerificationError(
            "the locked evaluation is missing the report its receipt records"
        )
    body = path.read_text(encoding="utf-8")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != recorded:
        raise ManifestVerificationError(
            "the locked evaluation's system-comparison report does not digest to "
            "the value its sealed receipt records; the report was changed after "
            "publication and is refused"
        )
    try:
        payload = json.loads(body)
        selection = FusionSelection.from_dict(payload["fusion"])
    except Exception as exc:
        raise ManifestVerificationError(
            f"the frozen fusion selection is not readable ({type(exc).__name__})"
        ) from None
    if selection.selection_fingerprint != record.fusion_selection_fingerprint:
        raise ManifestVerificationError(
            "the published fusion selection is not the one the sealed receipt names"
        )
    if selection.status is not SelectionStatus.SELECTED:
        raise ManifestVerificationError(
            "the receipt names a strategy the selection did not select"
        )
    if selection.selected_strategy is not record.selected_fusion_strategy:
        raise ManifestVerificationError(
            "the receipt and the selection name different hybrid strategies"
        )
    return selection


def reconstruct_stacked_state(
    *,
    dataset: MLDataset,
    config: MLConfig,
    eligible: EligibleFeatureList,
    feature_catalog: FeatureCatalog,
    champion: FrozenChampion,
    rule: Mapping[str, RuleEvidence],
    rule_configuration_fingerprint: str,
    validation_predictions: Sequence[BinaryPrediction],
) -> FusionPreparation:
    """Re-run the pre-TEST fusion stage and return everything it produced.

    Deterministic and label-honest by construction: every input is TRAIN-side,
    validation-side, or frozen lineage, and the orchestration it delegates to has
    no TEST parameter.  Nothing here chooses anything -- the caller compares what
    comes back with what Phase 5 sealed and refuses on disagreement.

    Raises:
        DataValidationError: when the validation split cannot be partitioned, or
            when a validation-B row carries no published score or no frozen rule
            decision.  A reconstruction over a subset would be a reconstruction
            of a different population.
        ModelNotReadyError: when the training context cannot be prepared.
    """
    validation = dataset.for_split(MLSplit.VALIDATION)
    partition = partition_validation(
        validation,
        config=config.validation_partition,
        support=config.support,
        campaign_metadata_supplied=dataset.campaign_fingerprint is not None,
    )
    if not partition.usable:
        raise DataValidationError(
            "the validation split does not carry enough support to cut the "
            "selection half from; the original selection was made on a "
            "partition this input cannot reproduce"
        )

    try:
        context = TrainingContext.prepare(
            dataset,
            config=config,
            eligible=eligible,
            feature_catalog=feature_catalog,
            partition=partition,
            allowlist_fingerprint=champion.lock.allowlist_fingerprint,
        )
    except PasswordAttackDetectorError as exc:
        raise ModelNotReadyError(
            f"the training context the fold refits run on could not be prepared "
            f"({type(exc).__name__})"
        ) from None

    scored = {row.anchor_event_id: row for row in validation_predictions}
    anchors: list[str] = []
    times: list[datetime] = []
    malicious: list[bool] = []
    rule_rows: list[RuleEvidence] = []
    ml_rows: list[MLEvidence] = []
    for anchor, is_malicious in zip(
        validation.anchors, validation.malicious, strict=True
    ):
        identifier = anchor.anchor_event_id
        if partition.assignment.get(identifier) is not _SELECTION_HALF:
            continue
        row = scored.get(identifier)
        evidence = rule.get(identifier)
        if row is None or evidence is None:
            raise DataValidationError(
                "a validation-B row carries no published model score or no frozen "
                "rule decision; a reconstruction over a subset would reproduce a "
                "different population than the one that was selected on"
            )
        anchors.append(identifier)
        times.append(anchor.anchor_event_time)
        malicious.append(is_malicious)
        rule_rows.append(evidence)
        ml_rows.append(
            MLEvidence(
                flagged=row.flagged_malicious,
                decision_score=row.malicious_decision_score,
                calibrated_probability=row.malicious_probability,
                score_kind=row.score_kind,
            )
        )
    if not anchors:
        raise DataValidationError(
            "validation-B is empty under this partition; there is nothing the "
            "original selection could be reproduced from"
        )

    return prepare_fusion_selection(
        context=context,
        catalog_model_id=champion.lock.catalog_model_id,
        champion_lock_fingerprint=champion.lock.lock_fingerprint,
        champion_freeze_record_id=champion.freeze_record_id,
        rule_flags={anchor: item.flagged for anchor, item in rule.items()},
        rule_configuration_fingerprint=rule_configuration_fingerprint,
        validation_anchor_event_ids=tuple(anchors),
        validation_anchor_event_times=tuple(times),
        validation_malicious=tuple(malicious),
        validation_rule=tuple(rule_rows),
        validation_ml=tuple(ml_rows),
        fold_count=config.fusion.stacked_fold_count,
        min_detection_rate=config.gates.min_detection_rate,
        max_false_positive_rate=config.gates.max_false_positive_rate,
        min_validation_positive_rows=config.support.min_validation_positive_rows,
        min_validation_benign_rows=config.support.min_validation_benign_rows,
    )


def materialize_serving_bundle(
    *,
    root: Path,
    dataset: MLDataset,
    config: MLConfig,
    eligible: EligibleFeatureList,
    feature_catalog: FeatureCatalog,
    champion: FrozenChampion,
    rule: Mapping[str, RuleEvidence],
    rule_configuration_fingerprint: str,
    validation_predictions: Sequence[BinaryPrediction],
) -> MaterializationOutcome:
    """Make the frozen hybrid deployable, or refuse and say exactly why.

    For a boolean gate this publishes the frozen selection and nothing else: a
    gate needs no fitted artifact, and fitting one to have something to put in
    the bundle would be publishing a stacker nobody selected.  The bundle still
    binds the selection, so a deployment reading it is reading the frozen
    strategy rather than a configured one.

    For ``STACKED`` it additionally reconstructs the meta-learner and verifies
    it, in this order:

    * the reconstruction must reproduce the **frozen selection**, fingerprint for
      fingerprint -- otherwise the inputs are not the inputs Phase 5 saw;
    * the reconstructed state must recompute the **frozen stacked-state
      fingerprint** -- otherwise the stacker is not the selected stacker.

    Either disagreement is a refusal.  Nothing is written, no strategy is
    substituted, and the returned outcome carries both digests so an operator can
    see which upstream input moved.

    Raises:
        ArtifactNotFoundError: no locked evaluation selected a hybrid.
        ManifestVerificationError: the frozen lineage did not verify.
    """
    frozen = read_frozen_fusion(
        root, champion_lock_fingerprint=champion.lock.lock_fingerprint
    )
    selection = frozen.selection
    if selection.rule_configuration_fingerprint != rule_configuration_fingerprint:
        return MaterializationOutcome(
            directory=None,
            created=False,
            strategy=frozen.strategy,
            frozen_state_fingerprint=selection.stacked_state_fingerprint,
            reconstructed_state_fingerprint=None,
            manifest=None,
            refusal="rule_configuration_mismatch",
        )

    reconstructed: StackedFusionState | None = None
    if frozen.stacked:
        preparation = reconstruct_stacked_state(
            dataset=dataset,
            config=config,
            eligible=eligible,
            feature_catalog=feature_catalog,
            champion=champion,
            rule=rule,
            rule_configuration_fingerprint=rule_configuration_fingerprint,
            validation_predictions=validation_predictions,
        )
        rebuilt = preparation.selection
        reconstructed = preparation.stacked_state
        if reconstructed is None:
            return MaterializationOutcome(
                directory=None,
                created=False,
                strategy=frozen.strategy,
                frozen_state_fingerprint=selection.stacked_state_fingerprint,
                reconstructed_state_fingerprint=None,
                manifest=None,
                refusal="stacked_state_not_reconstructible",
            )
        if (
            rebuilt is None
            or rebuilt.selection_fingerprint != selection.selection_fingerprint
        ):
            return MaterializationOutcome(
                directory=None,
                created=False,
                strategy=frozen.strategy,
                frozen_state_fingerprint=selection.stacked_state_fingerprint,
                reconstructed_state_fingerprint=reconstructed.state_fingerprint,
                manifest=None,
                refusal="fusion_selection_mismatch",
            )
        if reconstructed.state_fingerprint != selection.stacked_state_fingerprint:
            return MaterializationOutcome(
                directory=None,
                created=False,
                strategy=frozen.strategy,
                frozen_state_fingerprint=selection.stacked_state_fingerprint,
                reconstructed_state_fingerprint=reconstructed.state_fingerprint,
                manifest=None,
                refusal="stacked_state_fingerprint_mismatch",
            )

    manifest = build_bundle_manifest(
        champion=champion, frozen=frozen, stacked_state=reconstructed
    )
    directory, created = write_serving_bundle(
        root=root,
        manifest=manifest,
        selection=selection,
        stacked_state=reconstructed,
    )
    return MaterializationOutcome(
        directory=directory,
        created=created,
        strategy=frozen.strategy,
        frozen_state_fingerprint=selection.stacked_state_fingerprint,
        reconstructed_state_fingerprint=(
            None if reconstructed is None else reconstructed.state_fingerprint
        ),
        manifest=manifest,
        refusal=None,
    )


def build_bundle_manifest(
    *,
    champion: FrozenChampion,
    frozen: FrozenFusionEvidence,
    stacked_state: StackedFusionState | None,
) -> ServingBundleManifest:
    """Return the sealed manifest describing this bundle.

    Every value is copied from frozen state: the champion lock for the decision
    pipeline, the frozen selection for the hybrid, the sealed receipt for the
    provenance of both.  Nothing is computed here except the file digests and the
    manifest's own seal.
    """
    lock = champion.lock
    selection = frozen.selection
    files = bundle_files(selection=selection, stacked_state=stacked_state)
    return ServingBundleManifest.seal(
        champion_scope_key=lock.scope_key,
        champion_lock_fingerprint=lock.lock_fingerprint,
        champion_freeze_record_id=champion.freeze_record_id,
        validation_selection_id=lock.validation_selection_id,
        catalog_model_id=lock.catalog_model_id,
        model_family=lock.model_family,
        model_id=lock.model_id,
        model_content_fingerprint=lock.model_content_fingerprint,
        preprocessor_fingerprint=lock.preprocessor_fingerprint,
        calibration_method=lock.calibration_method,
        calibration_state_fingerprint=lock.calibration_state_fingerprint,
        binary_threshold_fingerprint=lock.binary_threshold_fingerprint,
        feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
        allowlist_fingerprint=lock.allowlist_fingerprint,
        eligible_feature_list_fingerprint=lock.eligible_feature_list_fingerprint,
        ml_config_fingerprint=lock.ml_config_fingerprint,
        serializer_id=lock.serializer_id,
        serializer_version=lock.serializer_version,
        dependency_contract_fingerprint=lock.dependency_contract_fingerprint,
        selected_fusion_strategy=frozen.strategy,
        fusion_selection_fingerprint=selection.selection_fingerprint,
        fusion_config_fingerprint=selection.fusion_config_fingerprint,
        rule_configuration_fingerprint=selection.rule_configuration_fingerprint,
        validation_evidence_fingerprint=selection.validation_evidence_fingerprint,
        stacked_state_fingerprint=(
            None if stacked_state is None else stacked_state.state_fingerprint
        ),
        oof_fold_definition_fingerprint=selection.oof_fold_definition_fingerprint,
        oof_evidence_fingerprint=selection.oof_evidence_fingerprint,
        base_model_recipe_fingerprint=selection.base_model_recipe_fingerprint,
        evaluation_record_id=frozen.record_id,
        evaluation_record_fingerprint=frozen.record.record_fingerprint,
        files=tuple(
            (name, hashlib.sha256(body.encode("utf-8")).hexdigest())
            for name, body in sorted(files.items())
        ),
    )


def _assert_no_test_outcome_reader() -> None:
    """Fail at import if this module acquires a way to read a TEST label.

    The reconstruction is legitimate precisely because it reads what the original
    fit read: TRAIN rows out of fold, and validation-B evidence.  A TEST outcome
    reader here would make the materialized stacker a function of the labels it
    was supposed to be frozen before, and it would look exactly like one that was
    not.
    """
    import sys

    forbidden = {
        "TestOutcome",
        "evaluate_test",
        "publish_evaluation",
        "select_binary_threshold",
        "select_fusion_strategy",
        "freeze_champion",
    }
    offending = sorted(forbidden & set(vars(sys.modules[__name__])))
    if offending:
        raise ValueError(
            f"{__name__} imported {offending}; materialization reproduces a "
            f"frozen decision and neither reads a TEST label nor makes a new one"
        )


_assert_no_test_outcome_reader()


def _assert_reconstruction_has_no_test_parameter() -> None:
    """Fail at import if an entry point here grows a TEST-shaped argument."""
    import inspect

    forbidden = {"test", "test_labels", "test_split", "holdout", "novel_holdout"}
    for function in (reconstruct_stacked_state, materialize_serving_bundle):
        offending = sorted(set(inspect.signature(function).parameters) & forbidden)
        if offending:
            raise ValueError(
                f"{function.__name__} declares parameter(s) {offending}; a "
                f"reconstruction reads pre-TEST lineage and nothing else"
            )


_assert_reconstruction_has_no_test_parameter()
