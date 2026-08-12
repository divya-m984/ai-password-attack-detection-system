"""Fusion: the truth tables, the out-of-fold contract, and validation-only choice.

Three properties this file exists to hold down.

**The two scales never meet.** An ordinal 0-100 rule magnitude and a calibrated
probability are different quantities, and no strategy here adds, averages, or
weights one against the other. The gates combine booleans; the stacker consumes
the rule side as a *decision*.

**The stacker never sees a row it was fitted on.** The out-of-fold orchestration
refits the whole pipeline per fold, cuts folds only at campaign boundaries, and
scores each TRAIN row exactly once. The suite proves each of those separately,
and proves that a full-TRAIN in-sample prediction cannot be passed off as
out-of-fold evidence.

**Selection reads validation and nothing else.** There is no TEST parameter on
any entry point, and the suite asserts that structurally as well as behaviourally.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from password_attack_detector.exceptions import DataValidationError, ModelTrainingError
from password_attack_detector.ml.enums import (
    FusionStrategy,
    MetricStatus,
    ScoreKind,
    SelectionStatus,
)
from password_attack_detector.ml.fusion import (
    FUSION_CANDIDATES,
    STACKED_FEATURE_NAMES,
    FusionDecision,
    FusionSelection,
    MetaRow,
    MLEvidence,
    RuleEvidence,
    StackedFusionState,
    campaign_folds,
    fit_stacked_fusion,
    fuse,
    fusion_config_fingerprint,
    select_fusion_strategy,
)
from password_attack_detector.ml.stacking import (
    build_stacked_state,
    establish_fusion_selection,
    fuse_population,
    validation_evidence_fingerprint,
)
from password_attack_detector.ml.training import (
    enumerate_candidates,
    out_of_fold_binary_scores,
)
from tests.ml import runs as rx

EPOCH = datetime(2024, 3, 1, tzinfo=UTC)
LOCK = "b" * 64
RULES = "a" * 64
FREEZE = "00000000-0000-5000-8000-000000000001"


def rule(flagged: bool, score: float | None = None) -> RuleEvidence:
    """Return one rule verdict."""
    return RuleEvidence(flagged=flagged, ordinal_risk_score=score)


def ml(
    flagged: bool, score: float = 0.5, probability: float | None = None
) -> MLEvidence:
    """Return one model verdict."""
    return MLEvidence(
        flagged=flagged,
        decision_score=score,
        calibrated_probability=probability,
        score_kind=(
            ScoreKind.CALIBRATED_PROBABILITY
            if probability is not None
            else ScoreKind.DECISION_SCORE
        ),
    )


def decide(
    strategy: FusionStrategy, rule_flag: bool, ml_flag: bool, **kwargs: Any
) -> FusionDecision:
    """Apply *strategy* to one pair of verdicts."""
    return fuse(
        strategy,
        anchor_event_id="a1",
        anchor_event_time=EPOCH,
        rule=rule(rule_flag, 80.0 if rule_flag else 5.0),
        ml=ml(ml_flag, 0.9 if ml_flag else 0.1),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The truth tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_flag", "ml_flag", "expected"),
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_the_or_gate_truth_table(
    rule_flag: bool, ml_flag: bool, expected: bool
) -> None:
    """All four rows, stated rather than sampled."""
    assert decide(FusionStrategy.OR_GATE, rule_flag, ml_flag).fused_flagged is expected


@pytest.mark.parametrize(
    ("rule_flag", "ml_flag", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_the_and_gate_truth_table(
    rule_flag: bool, ml_flag: bool, expected: bool
) -> None:
    """The other four."""
    assert decide(FusionStrategy.AND_GATE, rule_flag, ml_flag).fused_flagged is expected


def test_a_fused_decision_keeps_both_systems_evidence_apart() -> None:
    """Separately typed fields, and no combined score anywhere."""
    decision = decide(FusionStrategy.OR_GATE, True, True)
    assert decision.ordinal_risk_score == 80.0
    assert decision.malicious_decision_score == 0.9
    assert decision.calibrated_probability is None
    for absent in ("combined_score", "blended_score", "fused_score", "total_score"):
        assert absent not in FusionDecision.model_fields, absent


def test_a_gate_reports_no_stacked_probability() -> None:
    """A boolean gate has no score of its own to report."""
    assert decide(FusionStrategy.OR_GATE, True, False).stacked_probability is None


def test_a_decision_contradicting_its_strategy_is_refused() -> None:
    """The verdict must follow from the strategy it names."""
    with pytest.raises(ValueError, match="contradicts"):
        FusionDecision(
            anchor_event_id="a1",
            anchor_event_time=EPOCH,
            strategy=FusionStrategy.AND_GATE,
            rule_flagged=True,
            ordinal_risk_score=80.0,
            ml_flagged=False,
            malicious_decision_score=0.9,
            calibrated_probability=None,
            stacked_probability=None,
            fused_flagged=True,
        )


def test_an_ordinal_score_outside_the_phase_four_scale_is_refused() -> None:
    """A 0-100 magnitude is not a probability, and not a free number either."""
    with pytest.raises(ValueError, match="0-100 scale"):
        RuleEvidence(flagged=True, ordinal_risk_score=140.0)


def test_a_probability_without_a_calibrated_score_kind_is_refused() -> None:
    """An uncalibrated score is not a probability under another name."""
    with pytest.raises(ValueError, match="calibrated score kind"):
        MLEvidence(
            flagged=True,
            decision_score=0.5,
            calibrated_probability=0.5,
            score_kind=ScoreKind.DECISION_SCORE,
        )


def test_stacking_without_a_fitted_state_is_refused() -> None:
    """There is no default stacker to fall back on."""
    with pytest.raises(ModelTrainingError, match="no default stacker"):
        decide(FusionStrategy.STACKED, True, True)


# ---------------------------------------------------------------------------
# The stacker
# ---------------------------------------------------------------------------


def meta(count: int = 8) -> list[MetaRow]:
    """Return separable out-of-fold meta-features."""
    return [MetaRow(ml_score=0.9, rule_flagged=True, malicious=True)] * count + [
        MetaRow(ml_score=0.1, rule_flagged=False, malicious=False)
    ] * count


def test_the_stacker_consumes_exactly_two_declared_features() -> None:
    """The rule side enters as a decision, never as a magnitude."""
    state = fit_stacked_fusion(meta(), fold_count=3)
    assert state.feature_names == STACKED_FEATURE_NAMES
    assert STACKED_FEATURE_NAMES == ("ml_score", "rule_flag")
    assert len(state.coefficients) == 2


def test_the_stacker_is_deterministic() -> None:
    """No seed, no shuffling: two fits over one set of rows are one state."""
    assert (
        fit_stacked_fusion(meta(), fold_count=3).to_json()
        == fit_stacked_fusion(meta(), fold_count=3).to_json()
    )


def test_the_stacker_separates_the_two_regimes() -> None:
    """A fit that learned nothing would make this suite vacuous."""
    state = fit_stacked_fusion(meta(), fold_count=3)
    assert state.probability(ml_score=0.9, rule_flagged=True) > 0.5
    assert state.probability(ml_score=0.1, rule_flagged=False) < 0.5


def test_a_single_class_stacker_is_refused() -> None:
    """A single-class fit has no decision to learn."""
    rows = [MetaRow(ml_score=0.9, rule_flagged=True, malicious=True)] * 4
    with pytest.raises(ModelTrainingError, match="both classes"):
        fit_stacked_fusion(rows, fold_count=2)


def test_a_state_claiming_an_in_sample_source_is_refused() -> None:
    """An in-sample stacker learns that its base model is always right."""
    valid = fit_stacked_fusion(meta(), fold_count=3)
    with pytest.raises(ValueError, match="out-of-fold"):
        StackedFusionState.seal(
            coefficients=valid.coefficients,
            intercept=valid.intercept,
            iterations=valid.iterations,
            learning_rate=valid.learning_rate,
            l2_penalty=valid.l2_penalty,
            meta_feature_source="full_train_in_sample",
            fold_count=valid.fold_count,
            train_row_count=valid.train_row_count,
            positive_count=valid.positive_count,
            negative_count=valid.negative_count,
        )


def test_a_state_declaring_other_features_is_refused() -> None:
    """A stacker fed a label or a raw feature is a second model, not a combiner."""
    valid = fit_stacked_fusion(meta(), fold_count=3)
    with pytest.raises(ValueError, match="consumes"):
        StackedFusionState.seal(
            feature_names=("ml_score", "campaign_id"),
            coefficients=valid.coefficients,
            intercept=valid.intercept,
            iterations=valid.iterations,
            learning_rate=valid.learning_rate,
            l2_penalty=valid.l2_penalty,
            fold_count=valid.fold_count,
            train_row_count=valid.train_row_count,
            positive_count=valid.positive_count,
            negative_count=valid.negative_count,
        )


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def test_a_campaign_never_crosses_a_fold_boundary() -> None:
    """Splitting a campaign would let a model learn it and be scored on the rest."""
    campaigns = ["c1", "c2", "c1", "c3", "c2", "c4", "c1"]
    folds = campaign_folds(campaigns, fold_count=3)
    by_campaign: dict[str, set[int]] = {}
    for campaign, fold in zip(campaigns, folds, strict=True):
        by_campaign.setdefault(campaign, set()).add(fold)
    assert all(len(folds_used) == 1 for folds_used in by_campaign.values())


def test_folds_are_deterministic_and_order_independent_for_campaigns() -> None:
    """Sorted then dealt: the same campaigns always land in the same folds."""
    first = campaign_folds(["c1", "c2", "c3", "c4"], fold_count=2)
    second = campaign_folds(["c1", "c2", "c3", "c4"], fold_count=2)
    assert first == second
    reordered = campaign_folds(["c4", "c3", "c2", "c1"], fold_count=2)
    assignment = dict(zip(["c1", "c2", "c3", "c4"], first, strict=True))
    reassignment = dict(zip(["c4", "c3", "c2", "c1"], reordered, strict=True))
    assert assignment == reassignment


def test_a_single_fold_is_refused() -> None:
    """One fold is the in-sample fit this contract exists to prevent."""
    with pytest.raises(DataValidationError, match="at least two folds"):
        campaign_folds(["c1", "c2"], fold_count=1)


def test_too_few_campaigns_for_the_fold_count_is_refused() -> None:
    """Spreading three campaigns over five folds would have to split one."""
    with pytest.raises(DataValidationError, match="cannot be spread"):
        campaign_folds(["c1", "c2", "c3"], fold_count=5)


# ---------------------------------------------------------------------------
# The out-of-fold orchestration, against the real trainer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def context() -> Any:
    """Return one prepared training context, shared by the OOF tests."""
    return rx.context()


def binary_candidate(context: Any) -> Any:
    """Return the logistic-regression binary candidate."""
    from password_attack_detector.ml.enums import MLTask, ModelFamily

    return next(
        item
        for item in enumerate_candidates(context.config, catalog=context.catalog)
        if item.task is MLTask.BINARY_MALICIOUS
        and item.family is ModelFamily.LOGISTIC_REGRESSION
    )


def test_every_train_row_is_scored_exactly_once_out_of_fold(context: Any) -> None:
    """One prediction per row, from a model that did not see it."""
    scores = out_of_fold_binary_scores(context, binary_candidate(context), fold_count=3)
    assert scores.available
    population = context.supervised_train()
    assert len(scores.anchor_event_ids) == population.row_count
    assert len(set(scores.anchor_event_ids)) == len(scores.anchor_event_ids)
    assert [anchor.anchor_event_id for anchor in population.frame.anchors] == list(
        scores.anchor_event_ids
    )


def test_out_of_fold_campaigns_never_cross_their_fold(context: Any) -> None:
    """The campaign boundary holds through the whole orchestration."""
    scores = out_of_fold_binary_scores(context, binary_candidate(context), fold_count=3)
    campaign_by_anchor = {
        anchor.anchor_event_id: anchor.campaign_id
        for anchor in context.supervised_train().frame.anchors
    }
    seen: dict[str, set[int]] = {}
    for anchor, fold in zip(
        scores.anchor_event_ids, scores.fold_assignments, strict=True
    ):
        campaign = campaign_by_anchor[anchor]
        if campaign:
            seen.setdefault(campaign, set()).add(fold)
    assert seen
    assert all(len(folds) == 1 for folds in seen.values())


def test_out_of_fold_scores_are_deterministic(context: Any) -> None:
    """No random state anywhere in the fold construction or the refits."""
    first = out_of_fold_binary_scores(context, binary_candidate(context), fold_count=3)
    second = out_of_fold_binary_scores(
        rx.context(), binary_candidate(context), fold_count=3
    )
    assert first.scores == second.scores
    assert first.fold_definition_fingerprint == second.fold_definition_fingerprint


def test_out_of_fold_scores_differ_from_the_in_sample_fit(context: Any) -> None:
    """The whole point: a model that saw the row scores it differently.

    A base model fitted on all of TRAIN is confident about its own rows. If the
    out-of-fold scores matched it, nothing would have been held out.
    """
    from password_attack_detector.ml.imbalance import BINARY_CLASS_ORDER
    from password_attack_detector.ml.training import _fit, _score

    candidate = binary_candidate(context)
    population = context.supervised_train()
    preprocessor = context.preprocessor_for(candidate.task)
    adapter, fitted = _fit(
        candidate,
        context=context,
        preprocessor=preprocessor,
        population=population,
        targets=population.binary_targets(),
        class_order=BINARY_CLASS_ORDER,
        weights=None,
    )
    index = fitted.class_order.index(BINARY_CLASS_ORDER[1])
    in_sample = [
        float(row[index]) for row in _score(preprocessor, fitted, adapter, population)
    ]
    out_of_fold = list(
        out_of_fold_binary_scores(context, candidate, fold_count=3).scores
    )
    assert len(in_sample) == len(out_of_fold)
    assert in_sample != out_of_fold


def test_an_impossible_fold_construction_is_unavailable_not_approximated(
    context: Any,
) -> None:
    """Too few campaigns is an honest absence, not a row-level fallback."""
    scores = out_of_fold_binary_scores(
        context, binary_candidate(context), fold_count=99
    )
    assert not scores.available
    assert scores.unavailable_reason is not None
    assert "fold_construction_failed" in scores.unavailable_reason
    assert scores.scores == ()


def test_an_unavailable_fold_construction_yields_no_stacker(context: Any) -> None:
    """And the reason travels with the absence."""
    scores = out_of_fold_binary_scores(
        context, binary_candidate(context), fold_count=99
    )
    built = build_stacked_state(out_of_fold=scores, rule_flags={})
    assert not built.available
    assert built.unavailable_reason is not None


def test_a_stacker_needs_rule_evidence_for_every_out_of_fold_row(
    context: Any,
) -> None:
    """A stacker fitted on a subset is fitted on a different population."""
    scores = out_of_fold_binary_scores(context, binary_candidate(context), fold_count=3)
    with pytest.raises(DataValidationError, match="no frozen rule decision"):
        build_stacked_state(out_of_fold=scores, rule_flags={})


def test_the_stacker_is_built_from_the_out_of_fold_scores(context: Any) -> None:
    """The positive case, so the refusals above are not vacuous."""
    scores = out_of_fold_binary_scores(context, binary_candidate(context), fold_count=3)
    flags = {
        anchor: index % 3 == 0 for index, anchor in enumerate(scores.anchor_event_ids)
    }
    built = build_stacked_state(out_of_fold=scores, rule_flags=flags)
    assert built.available
    assert built.state is not None
    assert built.state.fold_count == 3
    assert built.fold_definition_fingerprint == scores.fold_definition_fingerprint


# ---------------------------------------------------------------------------
# Validation-only selection
# ---------------------------------------------------------------------------


def population(count: int = 40) -> dict[str, Any]:
    """Return an aligned validation-B population."""
    return {
        "anchor_event_ids": [f"v{index:04d}" for index in range(count)],
        "anchor_event_times": [
            EPOCH + timedelta(minutes=index) for index in range(count)
        ],
        "malicious": [index % 4 == 0 for index in range(count)],
        "rule": [
            rule(index % 4 == 0, 80.0 if index % 4 == 0 else 5.0)
            for index in range(count)
        ],
        "ml": [
            ml(index % 4 == 0, 0.9 if index % 4 == 0 else 0.1) for index in range(count)
        ],
    }


def select(**overrides: Any) -> FusionSelection:
    """Run a validation-only selection over the standard population."""
    from password_attack_detector.ml.stacking import StackedConstruction

    rows = population()
    settings: dict[str, Any] = {
        **rows,
        "construction": StackedConstruction(
            state=fit_stacked_fusion(meta(), fold_count=3),
            fold_definition_fingerprint="d" * 64,
            unavailable_reason=None,
        ),
        "rule_configuration_fingerprint": RULES,
        "champion_lock_fingerprint": LOCK,
        "champion_freeze_record_id": FREEZE,
        "min_detection_rate": 0.5,
        "max_false_positive_rate": 0.5,
        "min_validation_positive_rows": 5,
        "min_validation_benign_rows": 5,
    }
    settings.update(overrides)
    return establish_fusion_selection(**settings)


def test_a_selection_chooses_from_every_declared_candidate() -> None:
    """All three compete over one identical population."""
    selection = select()
    assert selection.status is SelectionStatus.SELECTED
    assert {item.strategy for item in selection.candidates} == set(FUSION_CANDIDATES)
    assert selection.selected_strategy in set(FUSION_CANDIDATES)


def test_a_selection_binds_the_lineage_that_could_change_it() -> None:
    """Rule configuration, champion, validation evidence, and the fold cut."""
    selection = select()
    assert selection.rule_configuration_fingerprint == RULES
    assert selection.champion_lock_fingerprint == LOCK
    assert selection.champion_freeze_record_id == FREEZE
    assert selection.oof_fold_definition_fingerprint == "d" * 64
    assert selection.validation_evidence_fingerprint
    assert selection.fusion_config_fingerprint


def test_a_changed_rule_configuration_changes_the_selection_identity() -> None:
    """A hybrid chosen against different rules is a different hybrid."""
    assert (
        select().selection_fingerprint
        != select(rule_configuration_fingerprint="f" * 64).selection_fingerprint
    )


def test_a_changed_validation_population_changes_the_selection_identity() -> None:
    """The evidence a selection saw is part of what it is."""
    rows = population()
    flipped = dict(rows)
    flipped["malicious"] = [not value for value in rows["malicious"]]
    assert select().selection_fingerprint != select(**flipped).selection_fingerprint


def test_no_eligible_strategy_is_recorded_rather_than_defaulted() -> None:
    """Falling back to OR_GATE would publish a hybrid nobody selected."""
    selection = select(min_detection_rate=1.1, max_false_positive_rate=0.0)
    assert selection.status is not SelectionStatus.SELECTED
    assert selection.selected_strategy is None
    assert all(not item.eligible for item in selection.candidates)


def test_thin_validation_support_is_reported_as_such() -> None:
    """Distinct from a measured negative."""
    selection = select(min_validation_positive_rows=999)
    assert selection.status is SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert selection.selected_strategy is None
    assert selection.support_status is MetricStatus.INSUFFICIENT_SUPPORT


def test_an_unavailable_stacker_does_not_block_the_gates() -> None:
    """OR and AND still compete; the stacker's absence is recorded."""
    from password_attack_detector.ml.stacking import StackedConstruction

    selection = select(
        construction=StackedConstruction(
            state=None,
            fold_definition_fingerprint=None,
            unavailable_reason="fold_construction_failed: DataValidationError",
        )
    )
    assert selection.stacked_unavailable_reason is not None
    assert selection.stacked_state_fingerprint is None
    stacked = next(
        item for item in selection.candidates if item.strategy is FusionStrategy.STACKED
    )
    assert not stacked.eligible
    assert any(
        "fold_construction_failed" in reason for reason in stacked.blocking_reasons
    )
    assert selection.selected_strategy is not FusionStrategy.STACKED


def test_a_stacked_selection_names_the_state_it_applies() -> None:
    """A selected stacker without its parameters could not be reapplied."""
    with pytest.raises(ValueError, match="names the state it applies"):
        FusionSelection.seal(
            status=SelectionStatus.SELECTED,
            selected_strategy=FusionStrategy.STACKED,
            stacked_state_fingerprint=None,
            stacked_unavailable_reason=None,
            candidates=(),
            min_detection_rate=0.5,
            max_false_positive_rate=0.5,
            min_validation_positive_rows=1,
            min_validation_benign_rows=1,
            support_status=MetricStatus.MEASURED,
            validation_row_count=10,
            validation_positive_count=5,
            validation_benign_count=5,
            rule_configuration_fingerprint=RULES,
            champion_lock_fingerprint=LOCK,
            champion_freeze_record_id=FREEZE,
            validation_evidence_fingerprint="e" * 64,
            fusion_config_fingerprint="f" * 64,
        )


def test_the_selection_is_reproducible() -> None:
    """Two selections over one set of evidence are the same record."""
    assert select().to_json() == select().to_json()


# ---------------------------------------------------------------------------
# The firewall, stated as a signature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "function",
    [
        select_fusion_strategy,
        establish_fusion_selection,
        build_stacked_state,
        fuse_population,
        validation_evidence_fingerprint,
        fit_stacked_fusion,
        campaign_folds,
    ],
)
def test_no_fusion_entry_point_takes_a_test_argument(function: Any) -> None:
    """Fusion selection reads validation evidence, and there is nowhere else."""
    parameters = set(inspect.signature(function).parameters)
    for absent in ("test", "test_labels", "test_split", "holdout", "novel_holdout"):
        assert absent not in parameters, absent


def test_the_fusion_contract_is_fingerprinted() -> None:
    """A change to the objective or the stacker's settings is a visible edit."""
    first = fusion_config_fingerprint(
        min_detection_rate=0.5,
        max_false_positive_rate=0.1,
        min_validation_positive_rows=5,
        min_validation_benign_rows=5,
    )
    second = fusion_config_fingerprint(
        min_detection_rate=0.6,
        max_false_positive_rate=0.1,
        min_validation_positive_rows=5,
        min_validation_benign_rows=5,
    )
    assert first != second
    assert len(first) == 64


def test_a_partial_candidate_set_is_refused() -> None:
    """Comparing a subset would make the winner depend on who was entered."""
    with pytest.raises(DataValidationError, match="exactly the declared candidate set"):
        select_fusion_strategy(
            decisions={FusionStrategy.OR_GATE: ()},
            malicious=[],
            min_detection_rate=0.5,
            max_false_positive_rate=0.5,
            min_validation_positive_rows=1,
            min_validation_benign_rows=1,
            rule_configuration_fingerprint=RULES,
            champion_lock_fingerprint=LOCK,
            champion_freeze_record_id=FREEZE,
            validation_evidence_fingerprint="e" * 64,
        )
