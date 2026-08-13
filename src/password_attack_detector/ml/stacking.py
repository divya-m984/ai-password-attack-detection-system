"""Building the stacker out of fold, and choosing a hybrid on validation alone.

This module is the orchestration between three things that must not be allowed
to touch each other carelessly: the trainer (which refits base models per fold),
the frozen Phase 4 rule engine (which supplies the other meta-feature), and the
fusion contract (which combines and selects).

**The stacker is fitted out of fold, and nothing else will do.** The
meta-features come from
:func:`~password_attack_detector.ml.training.out_of_fold_binary_scores`: every
TRAIN row's ML score is produced by a model refitted without that row, with the
preprocessing and the class weights refitted alongside it. Passing the
already-fitted full-TRAIN champion's predictions in here instead would fit the
meta-learner on the base model's own training performance, and
:func:`build_stacked_state` refuses evidence that does not carry a fold lineage
precisely so that substitution cannot be made quietly.

**Selection reads validation-B and nothing else.** The frozen champion is
applied to validation-B, the frozen rule engine's decisions for the same anchors
are joined in, all three strategies are evaluated over that one population, and
one is chosen -- or none is, which is a recorded outcome rather than a reason to
default to OR_GATE.

**TEST is not a parameter anywhere in this module.** Neither is the novel
holdout. Both are absent from every signature below, which is the firewall
stated as an interface rather than as a promise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from password_attack_detector.exceptions import DataValidationError, ModelNotReadyError
from password_attack_detector.ml.calibration import digest, quantize
from password_attack_detector.ml.enums import FusionStrategy, MLSplit, MLTask
from password_attack_detector.ml.fusion import (
    FUSION_CANDIDATES,
    FusionDecision,
    FusionSelection,
    MetaRow,
    MLEvidence,
    RuleEvidence,
    StackedFusionState,
    fit_stacked_fusion,
    fuse,
    select_fusion_strategy,
)
from password_attack_detector.ml.training import (
    CandidateSpec,
    OutOfFoldScores,
    TrainingContext,
    enumerate_candidates,
    out_of_fold_binary_scores,
)

__all__ = [
    "FusionFreezeProof",
    "FusionPreparation",
    "StackedConstruction",
    "base_model_recipe_fingerprint",
    "build_stacked_state",
    "establish_fusion_selection",
    "fuse_population",
    "no_fusion_evidence_proof",
    "oof_evidence_fingerprint",
    "prepare_fusion_selection",
    "validation_evidence_fingerprint",
]


@dataclass(frozen=True, slots=True)
class StackedConstruction:
    """A fitted stacker, or the stated reason there is none.

    Both outcomes are ordinary. A dataset with too few campaigns to cut folds
    from cannot support an out-of-fold stacker, and saying so leaves OR_GATE and
    AND_GATE free to compete -- which is a better answer than a stacker fitted
    on evidence that was never out of sample.
    """

    state: StackedFusionState | None
    fold_definition_fingerprint: str | None
    unavailable_reason: str | None

    @property
    def available(self) -> bool:
        """Return whether a stacker was fitted."""
        return self.state is not None


def build_stacked_state(
    *,
    out_of_fold: OutOfFoldScores,
    rule_flags: Mapping[str, bool],
) -> StackedConstruction:
    """Fit the meta-learner from out-of-fold TRAIN evidence, or say why not.

    Args:
        out_of_fold: per-row TRAIN scores from models that did not see their own
            row. An :class:`OutOfFoldScores` whose folds could not be cut
            carries the reason and produces no stacker.
        rule_flags: the frozen Phase 4 decision for each of those same anchors.

    Raises:
        DataValidationError: when the rule evidence does not cover exactly the
            out-of-fold rows. A stacker fitted on a subset would be fitted on a
            different population than the one it claims.
    """
    if not out_of_fold.available:
        return StackedConstruction(
            state=None,
            fold_definition_fingerprint=None,
            unavailable_reason=(
                out_of_fold.unavailable_reason
                or "out_of_fold_unavailable: no fold-scored TRAIN rows"
            ),
        )
    missing = [
        anchor for anchor in out_of_fold.anchor_event_ids if anchor not in rule_flags
    ]
    if missing:
        raise DataValidationError(
            f"{len(missing)} out-of-fold TRAIN row(s) carry no frozen rule "
            f"decision; a stacker fitted without them would be fitted on a "
            f"different population than it names"
        )

    rows = [
        MetaRow(
            ml_score=score,
            rule_flagged=rule_flags[anchor],
            malicious=outcome,
        )
        for anchor, score, outcome in zip(
            out_of_fold.anchor_event_ids,
            out_of_fold.scores,
            out_of_fold.malicious,
            strict=True,
        )
    ]
    try:
        state = fit_stacked_fusion(rows, fold_count=out_of_fold.fold_count)
    except Exception as exc:
        return StackedConstruction(
            state=None,
            fold_definition_fingerprint=out_of_fold.fold_definition_fingerprint,
            unavailable_reason=f"stacked_fit_failed: {type(exc).__name__}",
        )
    return StackedConstruction(
        state=state,
        fold_definition_fingerprint=out_of_fold.fold_definition_fingerprint,
        unavailable_reason=None,
    )


def fuse_population(
    *,
    anchor_event_ids: Sequence[str],
    anchor_event_times: Sequence[datetime],
    rule: Sequence[RuleEvidence],
    ml: Sequence[MLEvidence],
    stacked: StackedFusionState | None,
) -> dict[FusionStrategy, tuple[FusionDecision, ...]]:
    """Return every strategy's decisions over one aligned population.

    All three strategies are applied to the *same* rows in the *same* order, so
    a later comparison between them is a comparison of the strategies rather
    than of the populations they happened to see.

    ``STACKED`` yields an empty tuple when no state was fitted, which the
    selection reads as an unavailable candidate rather than as a failing one.
    """
    if not (len(anchor_event_ids) == len(anchor_event_times) == len(rule) == len(ml)):
        raise DataValidationError(
            "the fusion inputs are not aligned; one system's evidence would be "
            "paired with another row's"
        )
    fused: dict[FusionStrategy, tuple[FusionDecision, ...]] = {}
    for strategy in FUSION_CANDIDATES:
        if strategy is FusionStrategy.STACKED and stacked is None:
            fused[strategy] = ()
            continue
        fused[strategy] = tuple(
            fuse(
                strategy,
                anchor_event_id=anchor,
                anchor_event_time=when,
                rule=rule_row,
                ml=ml_row,
                stacked=stacked,
            )
            for anchor, when, rule_row, ml_row in zip(
                anchor_event_ids, anchor_event_times, rule, ml, strict=True
            )
        )
    return fused


def validation_evidence_fingerprint(
    *,
    anchor_event_ids: Sequence[str],
    malicious: Sequence[bool],
    rule: Sequence[RuleEvidence],
    ml: Sequence[MLEvidence],
) -> str:
    """Return the digest of exactly the validation-B evidence a selection saw.

    Binds the population, the outcomes, and both systems' decisions, so a
    changed validation half or a changed rule configuration makes a different
    selection rather than the same one with different numbers. Anchor
    identifiers take part and none is published: what leaves is a hash.
    """
    return digest(
        {
            "split": str(MLSplit.VALIDATION),
            "rows": [
                {
                    "anchor_event_id": anchor,
                    "malicious": outcome,
                    "rule_flagged": rule_row.flagged,
                    "rule_ordinal_risk_score": rule_row.ordinal_risk_score,
                    "ml_flagged": ml_row.flagged,
                    "ml_decision_score": ml_row.decision_score,
                    "ml_calibrated_probability": ml_row.calibrated_probability,
                }
                for anchor, outcome, rule_row, ml_row in zip(
                    anchor_event_ids, malicious, rule, ml, strict=True
                )
            ],
        }
    )


def establish_fusion_selection(
    *,
    anchor_event_ids: Sequence[str],
    anchor_event_times: Sequence[datetime],
    malicious: Sequence[bool],
    rule: Sequence[RuleEvidence],
    ml: Sequence[MLEvidence],
    construction: StackedConstruction,
    rule_configuration_fingerprint: str,
    champion_lock_fingerprint: str,
    champion_freeze_record_id: str,
    oof_evidence: str | None = None,
    base_model_recipe: str | None = None,
    min_detection_rate: float,
    max_false_positive_rate: float,
    min_validation_positive_rows: int,
    min_validation_benign_rows: int,
) -> FusionSelection:
    """Choose and freeze a hybrid strategy from validation-B evidence.

    Every argument is validation-side or frozen-lineage. There is no TEST
    parameter and no holdout parameter, so a caller cannot supply either even by
    mistake -- which is the firewall this milestone's whole ordering rests on.

    A strategy that could not be constructed (typically STACKED, when the folds
    could not be cut) enters the comparison as an *unavailable* candidate: the
    remaining strategies still compete, and the reason is recorded on the
    selection rather than being silently absorbed.
    """
    decisions = fuse_population(
        anchor_event_ids=anchor_event_ids,
        anchor_event_times=anchor_event_times,
        rule=rule,
        ml=ml,
        stacked=construction.state,
    )
    unavailable: dict[FusionStrategy, str] = {}
    if construction.state is None:
        unavailable[FusionStrategy.STACKED] = (
            construction.unavailable_reason or "stacked_unavailable"
        )
        # The candidate stays in the comparison with no decisions at all: it was
        # never built, so it was measured on nothing, and every one of its rates
        # is unavailable rather than zero.

    return select_fusion_strategy(
        decisions=decisions,
        malicious=malicious,
        min_detection_rate=min_detection_rate,
        max_false_positive_rate=max_false_positive_rate,
        min_validation_positive_rows=min_validation_positive_rows,
        min_validation_benign_rows=min_validation_benign_rows,
        rule_configuration_fingerprint=rule_configuration_fingerprint,
        champion_lock_fingerprint=champion_lock_fingerprint,
        champion_freeze_record_id=champion_freeze_record_id,
        validation_evidence_fingerprint=validation_evidence_fingerprint(
            anchor_event_ids=anchor_event_ids,
            malicious=malicious,
            rule=rule,
            ml=ml,
        ),
        stacked_state_fingerprint=(
            None if construction.state is None else construction.state.state_fingerprint
        ),
        stacked_unavailable_reason=construction.unavailable_reason,
        oof_fold_definition_fingerprint=construction.fold_definition_fingerprint,
        oof_evidence_fingerprint=oof_evidence,
        base_model_recipe_fingerprint=base_model_recipe,
        unavailable_strategies=unavailable,
    )


def oof_evidence_fingerprint(out_of_fold: OutOfFoldScores) -> str:
    """Return the digest of the out-of-fold evidence a stacker was fitted on.

    Distinct from the fold *definition* fingerprint. Two runs can cut identical
    folds and still produce different scores -- a changed base recipe, a changed
    preprocessor, a changed TRAIN population -- and a stacker fitted on
    different numbers is a different stacker.
    """
    return digest(
        {
            "anchor_event_ids": list(out_of_fold.anchor_event_ids),
            "scores": [quantize(value) for value in out_of_fold.scores],
            "malicious": list(out_of_fold.malicious),
            "fold_assignments": list(out_of_fold.fold_assignments),
            "fold_count": out_of_fold.fold_count,
            "fold_definition_fingerprint": out_of_fold.fold_definition_fingerprint,
        }
    )


def base_model_recipe_fingerprint(candidate: CandidateSpec) -> str:
    """Return the digest of the base recipe the fold refits instantiate."""
    return digest(
        {
            "task": str(candidate.task),
            "family": str(candidate.family),
            "catalog_model_id": candidate.catalog_model_id,
            "candidate_fingerprint": candidate.candidate_fingerprint,
        }
    )


# ---------------------------------------------------------------------------
# The pre-TEST orchestration
# ---------------------------------------------------------------------------

#: Held by :class:`FusionFreezeProof` and by nothing else.  A proof cannot be
#: constructed without it, so a caller cannot manufacture one to get past the
#: gate on the TEST reader.
_FREEZE_TOKEN: Final[object] = object()


@dataclass(frozen=True, slots=True)
class FusionFreezeProof:
    """Evidence that fusion selection reached its final frozen state.

    Exists to make the ordering structural rather than documented. The TEST
    ground-truth reader takes one of these, and only
    :func:`prepare_fusion_selection` can produce one -- so a caller cannot open
    a TEST label before the hybrid arm has been decided, even by mistake.
    """

    token: object
    outcome: str
    declared_candidates: tuple[FusionStrategy, ...]
    candidate_status: Mapping[FusionStrategy, str]
    fusion_selection_fingerprint: str | None

    def __post_init__(self) -> None:
        """Refuse a proof that did not come from a completed preparation."""
        if self.token is not _FREEZE_TOKEN:
            raise ModelNotReadyError(
                "a fusion freeze proof may only be produced by "
                "prepare_fusion_selection; constructing one directly would "
                "defeat the ordering it exists to enforce"
            )
        if set(self.declared_candidates) != set(FUSION_CANDIDATES):
            raise ModelNotReadyError(
                "a fusion freeze proof must name the whole declared candidate "
                "universe; a strategy dropped from the record is a strategy "
                "nobody can see was not considered"
            )


@dataclass(frozen=True, slots=True)
class FusionPreparation:
    """Everything the pre-TEST fusion stage produced."""

    selection: FusionSelection | None
    stacked_state: StackedFusionState | None
    construction: StackedConstruction
    out_of_fold: OutOfFoldScores
    proof: FusionFreezeProof

    @property
    def stacked_available(self) -> bool:
        """Return whether a stacker was genuinely constructed."""
        return self.stacked_state is not None


def _champion_candidate(
    context: TrainingContext, *, catalog_model_id: str
) -> CandidateSpec | None:
    """Return the declared candidate matching the frozen champion's recipe.

    The fold refits instantiate the *champion's own family and hyperparameters*.
    A stacker whose meta-feature came from some other model would be a stacker
    over a base model nobody froze.
    """
    for candidate in enumerate_candidates(context.config, catalog=context.catalog):
        if (
            candidate.task is MLTask.BINARY_MALICIOUS
            and candidate.catalog_model_id == catalog_model_id
        ):
            return candidate
    return None


def prepare_fusion_selection(
    *,
    context: TrainingContext,
    catalog_model_id: str,
    champion_lock_fingerprint: str,
    champion_freeze_record_id: str,
    rule_flags: Mapping[str, bool],
    rule_configuration_fingerprint: str,
    validation_anchor_event_ids: Sequence[str],
    validation_anchor_event_times: Sequence[datetime],
    validation_malicious: Sequence[bool],
    validation_rule: Sequence[RuleEvidence],
    validation_ml: Sequence[MLEvidence],
    fold_count: int,
    min_detection_rate: float,
    max_false_positive_rate: float,
    min_validation_positive_rows: int,
    min_validation_benign_rows: int,
) -> FusionPreparation:
    """Construct every fusion candidate and freeze a selection, before TEST.

    This is the orchestration ``ml evaluate`` performs *before* it is able to
    open a TEST label. All three declared strategies take part:

    * ``OR_GATE`` and ``AND_GATE`` are pure functions of the validation-B
      evidence and are always constructible;
    * ``STACKED`` is fitted from genuine out-of-fold TRAIN refits -- each fold
      refits the preprocessor, the class weights, and a fresh instance of the
      *champion's own family and configuration*, then scores only the rows held
      out of that fold.

    Every input is TRAIN-side or validation-side. There is no TEST parameter and
    no holdout parameter, which is the property the whole milestone's ordering
    rests on.

    A STACKED that cannot be built carries a typed reason and the gates still
    compete. What is no longer possible is STACKED being unavailable because
    nobody wired the fitting in.
    """
    candidate = _champion_candidate(context, catalog_model_id=catalog_model_id)
    if candidate is None:
        construction = StackedConstruction(
            state=None,
            fold_definition_fingerprint=None,
            unavailable_reason=(
                f"base_recipe_unavailable: the frozen champion's model "
                f"{catalog_model_id!r} is not a candidate this configuration "
                f"declares, so no fold refit could reproduce it"
            ),
        )
        out_of_fold = OutOfFoldScores(
            unavailable_reason=construction.unavailable_reason
        )
    else:
        out_of_fold = out_of_fold_binary_scores(
            context, candidate, fold_count=fold_count
        )
        missing = [
            anchor
            for anchor in out_of_fold.anchor_event_ids
            if anchor not in rule_flags
        ]
        if out_of_fold.available and missing:
            construction = StackedConstruction(
                state=None,
                fold_definition_fingerprint=out_of_fold.fold_definition_fingerprint,
                unavailable_reason=(
                    f"rule_evidence_unavailable: {len(missing)} out-of-fold "
                    f"TRAIN row(s) carry no frozen rule decision, and a stacker "
                    f"fitted without them would be fitted on a different "
                    f"population than it names"
                ),
            )
        else:
            construction = build_stacked_state(
                out_of_fold=out_of_fold, rule_flags=rule_flags
            )

    selection = establish_fusion_selection(
        anchor_event_ids=validation_anchor_event_ids,
        anchor_event_times=validation_anchor_event_times,
        malicious=validation_malicious,
        rule=validation_rule,
        ml=validation_ml,
        construction=construction,
        rule_configuration_fingerprint=rule_configuration_fingerprint,
        champion_lock_fingerprint=champion_lock_fingerprint,
        champion_freeze_record_id=champion_freeze_record_id,
        oof_evidence=(
            oof_evidence_fingerprint(out_of_fold) if out_of_fold.available else None
        ),
        base_model_recipe=(
            None if candidate is None else base_model_recipe_fingerprint(candidate)
        ),
        min_detection_rate=min_detection_rate,
        max_false_positive_rate=max_false_positive_rate,
        min_validation_positive_rows=min_validation_positive_rows,
        min_validation_benign_rows=min_validation_benign_rows,
    )

    status: dict[FusionStrategy, str] = {}
    for evidence in selection.candidates:
        status[evidence.strategy] = (
            "eligible"
            if evidence.eligible
            else "; ".join(evidence.blocking_reasons) or "ineligible"
        )
    for strategy in FUSION_CANDIDATES:
        status.setdefault(strategy, "not_measured")
    if construction.unavailable_reason is not None:
        status[FusionStrategy.STACKED] = construction.unavailable_reason

    return FusionPreparation(
        selection=selection,
        stacked_state=construction.state,
        construction=construction,
        out_of_fold=out_of_fold,
        proof=FusionFreezeProof(
            token=_FREEZE_TOKEN,
            outcome=str(selection.status),
            declared_candidates=tuple(FUSION_CANDIDATES),
            candidate_status=status,
            fusion_selection_fingerprint=selection.selection_fingerprint,
        ),
    )


def no_fusion_evidence_proof() -> FusionFreezeProof:
    """Return the frozen outcome for a run that was supplied no fusion evidence.

    Not a bypass. It is the honest terminal state of the fusion stage when no
    validation prediction was published: every declared candidate is recorded
    as unmeasured, there is no selection, and the hybrid arm will say so. The
    TEST reader is gated on this exactly as it is gated on a real selection,
    because "no evidence" is a conclusion the fusion stage reached, not a step
    it skipped.
    """
    reason = "no_validation_evidence: no validation prediction was supplied"
    return FusionFreezeProof(
        token=_FREEZE_TOKEN,
        outcome="no_fusion_selection",
        declared_candidates=tuple(FUSION_CANDIDATES),
        candidate_status=dict.fromkeys(FUSION_CANDIDATES, reason),
        fusion_selection_fingerprint=None,
    )


def _assert_no_test_parameter() -> None:
    """Fail at import if any entry point here grows a TEST-shaped argument.

    A structural guard rather than a review note: fusion selection is
    validation-only, and the way that stops being true is one plausible-sounding
    keyword at a time.
    """
    import inspect

    forbidden = {"test", "test_labels", "test_split", "holdout", "novel_holdout"}
    for function in (
        build_stacked_state,
        fuse_population,
        establish_fusion_selection,
        validation_evidence_fingerprint,
        prepare_fusion_selection,
    ):
        offending = sorted(set(inspect.signature(function).parameters) & forbidden)
        if offending:
            raise ValueError(
                f"{function.__name__} declares parameter(s) {offending}; fusion "
                f"selection reads validation evidence and nothing else"
            )


_assert_no_test_parameter()
