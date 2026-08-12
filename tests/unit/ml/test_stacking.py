"""The pre-TEST fusion orchestration: what it builds, and what it refuses.

``prepare_fusion_selection`` is the one place where the trainer, the frozen rule
engine, and the fusion contract meet. Its job is to leave *no* candidate in an
unexamined state before a TEST label becomes readable: OR_GATE and AND_GATE are
constructed from validation-B evidence, STACKED is fitted from genuine
out-of-fold TRAIN refits, and a STACKED that cannot be built carries a typed
reason instead of quietly disappearing.

The tests here are mostly about the second half of that sentence. "STACKED was
unavailable" is easy to produce accidentally and impossible to notice, so each
legitimate cause has its own test asserting the *reason* rather than only the
absence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from password_attack_detector.exceptions import ModelNotReadyError
from password_attack_detector.ml import stacking
from password_attack_detector.ml.enums import FusionStrategy, MLTask
from password_attack_detector.ml.fusion import (
    FUSION_CANDIDATES,
    MLEvidence,
    RuleEvidence,
)
from password_attack_detector.ml.stacking import (
    FusionFreezeProof,
    base_model_recipe_fingerprint,
    no_fusion_evidence_proof,
    oof_evidence_fingerprint,
    prepare_fusion_selection,
)
from password_attack_detector.ml.training import (
    OutOfFoldScores,
    TrainingContext,
    enumerate_candidates,
)
from tests.ml import runs

WHEN = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def context() -> TrainingContext:
    """Return one prepared training context, shared across the module."""
    return runs.context()


@pytest.fixture(scope="module")
def champion_model_id(context: TrainingContext) -> str:
    """Return a binary candidate's catalog identifier, as a champion would name."""
    for candidate in enumerate_candidates(context.config, catalog=context.catalog):
        if candidate.task is MLTask.BINARY_MALICIOUS:
            return candidate.catalog_model_id
    raise AssertionError("the configuration declares no binary candidate")


def _validation(context: TrainingContext) -> dict[str, Any]:
    """Return validation-B-shaped evidence for the prepared context.

    Built from the context's own validation rows so the population is real; the
    scores are synthetic because what is under test here is the orchestration,
    not the champion.
    """
    from password_attack_detector.ml.enums import MLSplit

    rows = context.dataset.for_split(MLSplit.VALIDATION)
    anchors = [anchor.anchor_event_id for anchor in rows.anchors]
    return {
        "validation_anchor_event_ids": tuple(anchors),
        "validation_anchor_event_times": tuple(
            anchor.anchor_event_time for anchor in rows.anchors
        ),
        "validation_malicious": tuple(rows.malicious),
        "validation_rule": tuple(
            RuleEvidence(flagged=value, ordinal_risk_score=80.0 if value else 5.0)
            for value in rows.malicious
        ),
        "validation_ml": tuple(
            MLEvidence(flagged=value, decision_score=0.9 if value else 0.1)
            for value in rows.malicious
        ),
    }


def _train_anchors(context: TrainingContext) -> list[str]:
    """Return the anchors of the supervised TRAIN population."""
    population = context.supervised_train()
    return [anchor.anchor_event_id for anchor in population.frame.anchors]


def _prepare(
    context: TrainingContext,
    champion_model_id: str,
    *,
    rule_flags: dict[str, bool] | None = None,
    fold_count: int = 3,
    catalog_model_id: str | None = None,
) -> Any:
    """Run the orchestration over the prepared context."""
    flags = (
        rule_flags
        if rule_flags is not None
        else dict.fromkeys(_train_anchors(context), True)
    )
    return prepare_fusion_selection(
        context=context,
        catalog_model_id=catalog_model_id or champion_model_id,
        champion_lock_fingerprint="a" * 64,
        champion_freeze_record_id="freeze-1",
        rule_flags=flags,
        rule_configuration_fingerprint="b" * 64,
        fold_count=fold_count,
        min_detection_rate=0.0,
        max_false_positive_rate=1.0,
        min_validation_positive_rows=1,
        min_validation_benign_rows=1,
        **_validation(context),
    )


# ---------------------------------------------------------------------------
# The declared candidate universe
# ---------------------------------------------------------------------------


def test_every_declared_strategy_is_examined(
    context: TrainingContext, champion_model_id: str
) -> None:
    """All three, always. A candidate nobody looked at is the failure mode."""
    preparation = _prepare(context, champion_model_id)
    assert set(preparation.proof.declared_candidates) == set(FUSION_CANDIDATES)
    assert set(preparation.proof.candidate_status) >= set(FUSION_CANDIDATES)


def test_the_stacker_is_actually_fitted(
    context: TrainingContext, champion_model_id: str
) -> None:
    """Out-of-fold refits happen here, not somewhere a caller must remember."""
    preparation = _prepare(context, champion_model_id)
    assert preparation.stacked_available, preparation.construction.unavailable_reason
    assert preparation.out_of_fold.available
    assert preparation.out_of_fold.fold_count == 3


def test_every_eligible_train_row_is_scored_exactly_once(
    context: TrainingContext, champion_model_id: str
) -> None:
    """No duplicate, no omission, and one fold assignment each."""
    preparation = _prepare(context, champion_model_id)
    out_of_fold = preparation.out_of_fold
    anchors = out_of_fold.anchor_event_ids
    assert len(anchors) == len(set(anchors))
    assert set(anchors) == set(_train_anchors(context))
    assert len(out_of_fold.scores) == len(anchors)
    assert len(out_of_fold.fold_assignments) == len(anchors)


def test_the_orchestration_is_deterministic(
    context: TrainingContext, champion_model_id: str
) -> None:
    """Two runs over identical inputs agree on every fitted quantity."""
    first = _prepare(context, champion_model_id)
    second = _prepare(context, champion_model_id)
    assert first.out_of_fold == second.out_of_fold
    assert first.selection is not None and second.selection is not None
    assert (
        first.selection.selection_fingerprint == second.selection.selection_fingerprint
    )


# ---------------------------------------------------------------------------
# Legitimate reasons STACKED may be unavailable
# ---------------------------------------------------------------------------


def test_more_folds_than_campaigns_is_a_typed_reason(
    context: TrainingContext, champion_model_id: str
) -> None:
    """Folds are campaign-indivisible, so this is a data answer, not a bug."""
    preparation = _prepare(context, champion_model_id, fold_count=999)
    assert not preparation.stacked_available
    reason = preparation.construction.unavailable_reason
    assert reason and ":" in reason
    assert preparation.selection is not None


def test_missing_rule_evidence_is_a_typed_reason(
    context: TrainingContext, champion_model_id: str
) -> None:
    """A stacker fitted on a subset would be fitted on a different population."""
    preparation = _prepare(context, champion_model_id, rule_flags={})
    assert not preparation.stacked_available
    assert preparation.construction.unavailable_reason is not None
    assert preparation.construction.unavailable_reason.startswith(
        "rule_evidence_unavailable:"
    )


def test_an_unknown_base_recipe_is_a_typed_reason(context: TrainingContext) -> None:
    """A fold refit must reproduce the champion's own family and configuration."""
    preparation = _prepare(
        context, "unused", catalog_model_id="pad-not-a-declared-model"
    )
    assert not preparation.stacked_available
    assert preparation.construction.unavailable_reason is not None
    assert preparation.construction.unavailable_reason.startswith(
        "base_recipe_unavailable:"
    )
    assert preparation.selection is not None


def test_no_unavailable_reason_blames_the_orchestration(
    context: TrainingContext, champion_model_id: str
) -> None:
    """ "Nobody wired it in" stopped being an acceptable answer."""
    for preparation in (
        _prepare(context, champion_model_id, fold_count=999),
        _prepare(context, champion_model_id, rule_flags={}),
        _prepare(context, "unused", catalog_model_id="pad-not-a-declared-model"),
    ):
        reason = preparation.construction.unavailable_reason or ""
        for excuse in ("not supplied", "refits nothing", "not wired", "orchestration"):
            assert excuse not in reason, reason


def test_an_unavailable_stacker_does_not_remove_it_from_the_record(
    context: TrainingContext, champion_model_id: str
) -> None:
    """The candidate stays declared; only its availability changed."""
    preparation = _prepare(context, champion_model_id, fold_count=999)
    assert FusionStrategy.STACKED in preparation.proof.declared_candidates
    assert preparation.proof.candidate_status[FusionStrategy.STACKED]
    assert preparation.selection is not None
    assert preparation.selection.stacked_unavailable_reason


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_selection_binds_the_stacked_lineage(
    context: TrainingContext, champion_model_id: str
) -> None:
    """Fold definition, evidence, base recipe, and stacker state are all named."""
    selection = _prepare(context, champion_model_id).selection
    assert selection is not None
    assert selection.oof_fold_definition_fingerprint
    assert selection.oof_evidence_fingerprint
    assert selection.base_model_recipe_fingerprint
    assert selection.stacked_state_fingerprint


def test_changing_the_fold_count_changes_the_selection_identity(
    context: TrainingContext, champion_model_id: str
) -> None:
    """A different fold state is a different selection."""
    three = _prepare(context, champion_model_id, fold_count=3).selection
    four = _prepare(context, champion_model_id, fold_count=4).selection
    assert three is not None and four is not None
    assert three.selection_fingerprint != four.selection_fingerprint


def test_the_evidence_fingerprint_separates_scores_from_fold_boundaries() -> None:
    """Identical folds with different scores must not share an identity."""
    base = OutOfFoldScores(
        anchor_event_ids=("a", "b"),
        scores=(0.1, 0.9),
        malicious=(False, True),
        fold_assignments=(0, 1),
        fold_count=2,
        fold_definition_fingerprint="c" * 64,
    )
    moved = OutOfFoldScores(
        anchor_event_ids=base.anchor_event_ids,
        scores=(0.2, 0.9),
        malicious=base.malicious,
        fold_assignments=base.fold_assignments,
        fold_count=base.fold_count,
        fold_definition_fingerprint=base.fold_definition_fingerprint,
    )
    assert oof_evidence_fingerprint(base) != oof_evidence_fingerprint(moved)
    assert oof_evidence_fingerprint(base) == oof_evidence_fingerprint(base)


def test_the_base_recipe_fingerprint_follows_the_candidate(
    context: TrainingContext,
) -> None:
    """Two different families are two different base recipes."""
    binary = [
        candidate
        for candidate in enumerate_candidates(context.config, catalog=context.catalog)
        if candidate.task is MLTask.BINARY_MALICIOUS
    ]
    fingerprints = {base_model_recipe_fingerprint(item) for item in binary}
    assert len(fingerprints) == len(binary)


# ---------------------------------------------------------------------------
# The freeze proof
# ---------------------------------------------------------------------------


def test_a_preparation_yields_a_usable_proof(
    context: TrainingContext, champion_model_id: str
) -> None:
    """The only supported way to get one."""
    proof = _prepare(context, champion_model_id).proof
    assert isinstance(proof, FusionFreezeProof)
    assert proof.fusion_selection_fingerprint


def test_a_proof_cannot_be_forged() -> None:
    """Without the module-private token, construction refuses."""
    with pytest.raises(ModelNotReadyError, match="prepare_fusion_selection"):
        FusionFreezeProof(
            token="not-the-token",
            outcome="selected",
            declared_candidates=tuple(FUSION_CANDIDATES),
            candidate_status={},
            fusion_selection_fingerprint=None,
        )


def test_a_proof_must_carry_the_whole_candidate_universe() -> None:
    """A partial record would hide a strategy nobody examined."""
    with pytest.raises(ModelNotReadyError, match="whole declared candidate"):
        FusionFreezeProof(
            token=stacking._FREEZE_TOKEN,
            outcome="selected",
            declared_candidates=(FusionStrategy.OR_GATE, FusionStrategy.AND_GATE),
            candidate_status={},
            fusion_selection_fingerprint=None,
        )


def test_the_no_evidence_proof_is_a_conclusion_not_a_bypass() -> None:
    """It still names every candidate, and still carries a reason for each."""
    proof = no_fusion_evidence_proof()
    assert set(proof.declared_candidates) == set(FUSION_CANDIDATES)
    assert proof.fusion_selection_fingerprint is None
    for strategy in FUSION_CANDIDATES:
        assert proof.candidate_status[strategy].startswith("no_validation_evidence:")


def test_the_orchestration_declares_no_test_parameter() -> None:
    """The import-time guard, asserted rather than assumed."""
    import inspect

    parameters = set(inspect.signature(prepare_fusion_selection).parameters)
    forbidden = {"test", "test_labels", "test_split", "holdout", "novel_holdout"}
    assert not parameters & forbidden
    assert not any("test" in name.split("_") for name in parameters)
