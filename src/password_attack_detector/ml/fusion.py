"""Combining a rule verdict with a model decision, chosen on validation alone.

Three strategies, and none of them performs arithmetic across the two systems.
Phase 4's ``risk_score`` is an **ordinal severity magnitude on 0-100**; the ML
layer's output is a calibrated probability or a raw decision score. Adding,
averaging, or weighting one against the other would produce a number whose units
do not exist, and it would do so silently -- the result would look like a score
and behave like nonsense. So:

* :attr:`~password_attack_detector.ml.enums.FusionStrategy.OR_GATE` and
  :attr:`~password_attack_detector.ml.enums.FusionStrategy.AND_GATE` combine the
  two **booleans**;
* :attr:`~password_attack_detector.ml.enums.FusionStrategy.STACKED` consumes
  them as **separately named typed inputs** to a fitted meta-learner, where the
  rule side enters as a decision and never as a magnitude reinterpreted as a
  likelihood.

Every fused row keeps the rule evidence and the ML evidence in their own typed
fields. Nothing here collapses them into a single blended number, and there is
nowhere in :class:`FusionDecision` to put one.

**The stacker never sees a row it was fitted on.** A meta-learner trained on
in-sample base-model predictions learns that the base model is right, because on
its own training rows it usually is -- and the resulting stacker is confident in
exactly the region where it should be cautious. So the meta-features come from
**out-of-fold** TRAIN predictions: each row's ML score is produced by a base
model fitted without that row, and the folds are cut at campaign boundaries so a
campaign's other events cannot leak across the cut either.

**Selection is validation-only, and it is frozen before TEST is opened.** The
strategy, the stacker's parameters, and the fusion threshold are all chosen on
validation-B evidence. :class:`FusionSelection` seals that choice, and its
identity binds the candidate set, the rule configuration, the ML champion, the
validation evidence, and the decision semantics -- so changing any of them is a
different selection rather than the same one with different numbers. No TEST
value reaches any of it.

**No fallback.** When no strategy clears the declared gates the selection
records ``no_eligible_fusion`` and the comparison reports the hybrid as
unavailable. Quietly defaulting to OR_GATE would publish a hybrid nobody
selected, and it would look exactly like one that had been.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import DataValidationError, ModelTrainingError
from password_attack_detector.ml.calibration import (
    SealedModel,
    digest,
    quantize,
    sigmoid,
)
from password_attack_detector.ml.enums import (
    FusionStrategy,
    MetricStatus,
    ScoreKind,
    SelectionStatus,
    is_probability,
)
from password_attack_detector.ml.metrics import BinaryTestMetrics, binary_metrics
from password_attack_detector.ml.schemas import Sha256Hex

__all__ = [
    "FUSION_CANDIDATES",
    "FUSION_SCHEMA_VERSION",
    "STACKED_FEATURE_NAMES",
    "FusionCandidateEvidence",
    "FusionDecision",
    "FusionSelection",
    "MLEvidence",
    "MetaRow",
    "RuleEvidence",
    "StackedFusionState",
    "campaign_folds",
    "fit_stacked_fusion",
    "fuse",
    "fusion_config_fingerprint",
    "select_fusion_strategy",
]

#: The fusion contract's own version.
FUSION_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The strategies a selection compares, in the order it compares them.  Fixed:
#: an ordering chosen after seeing the evidence would be a tie-break fitted to
#: the result.
FUSION_CANDIDATES: Final[tuple[FusionStrategy, ...]] = (
    FusionStrategy.OR_GATE,
    FusionStrategy.AND_GATE,
    FusionStrategy.STACKED,
)

#: The stacker's meta-features, declared rather than discovered.
#:
#: Two, both frozen decision outputs. The rule side enters as its **decision**,
#: not as its ordinal magnitude: a 0-100 severity fed to a logistic model would
#: be treated as a continuous quantity on a scale it does not have. No label, no
#: campaign identifier, no anchor identifier, and no raw feature is admitted --
#: a stacker that saw any of those would be a second model fitted on the data
#: the first one was supposed to summarise.
STACKED_FEATURE_NAMES: Final[tuple[str, ...]] = ("ml_score", "rule_flag")

#: Deterministic optimiser settings for the meta-learner.  Written down rather
#: than tuned: a stacker whose iteration count was chosen from validation would
#: have one more fitted parameter than it declares.
_STACKED_ITERATIONS: Final[int] = 500
_STACKED_LEARNING_RATE: Final[float] = 0.1
_STACKED_L2: Final[float] = 0.01
_STACKED_DECISION_THRESHOLD: Final[float] = 0.5


# ---------------------------------------------------------------------------
# Typed evidence
# ---------------------------------------------------------------------------


class RuleEvidence(BaseModel):
    """What the frozen Phase 4 rule engine said about one anchor.

    ``ordinal_risk_score`` is a **0-100 severity magnitude**, carried for
    reporting and never for arithmetic against a probability. It is nullable
    because an anchor the engine assessed but did not flag may legitimately
    carry no score under the Phase 4 null semantics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    flagged: bool
    ordinal_risk_score: float | None = None

    @model_validator(mode="after")
    def check_evidence(self) -> Self:
        """The ordinal score stays inside the scale Phase 4 declares."""
        if self.ordinal_risk_score is None:
            return self
        if not math.isfinite(self.ordinal_risk_score):
            raise ValueError("an ordinal risk score must be finite")
        if not 0.0 <= self.ordinal_risk_score <= 100.0:
            raise ValueError(
                "an ordinal risk score lies on the Phase 4 0-100 scale; a value "
                "outside it is not that quantity"
            )
        return self


class MLEvidence(BaseModel):
    """What the frozen ML champion said about one anchor.

    ``calibrated_probability`` exists only where a verified calibrator produced
    one, exactly as on a published prediction row. The raw decision score is
    always present: it is what the model itself emits.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    flagged: bool
    decision_score: float
    calibrated_probability: float | None = None
    score_kind: ScoreKind = ScoreKind.DECISION_SCORE

    @model_validator(mode="after")
    def check_evidence(self) -> Self:
        """A probability is present exactly where a calibrator produced one."""
        if not math.isfinite(self.decision_score):
            raise ValueError("a decision score must be finite")
        if is_probability(self.score_kind) != (self.calibrated_probability is not None):
            raise ValueError(
                "a calibrated score kind carries a probability, and nothing else "
                "may; an uncalibrated score is not a probability under another name"
            )
        if self.calibrated_probability is not None and not (
            0.0 <= self.calibrated_probability <= 1.0
        ):
            raise ValueError("a calibrated probability lies in [0, 1]")
        return self

    @property
    def fusion_score(self) -> float:
        """Return the ML quantity the stacker consumes.

        The calibrated probability where one exists, and the raw decision score
        otherwise -- the same quantity the frozen threshold was selected
        against, so the stacker and the binary head read the same number.
        """
        if self.calibrated_probability is not None:
            return self.calibrated_probability
        return self.decision_score


class FusionDecision(BaseModel):
    """One fused decision, with both systems' evidence kept separately typed.

    There is no combined score field, and there will not be one. The two inputs
    live on different scales and the fused output is a boolean; a blended number
    would be a third quantity nobody defined.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fusion_schema_version: str = FUSION_SCHEMA_VERSION
    anchor_event_id: str
    anchor_event_time: datetime
    strategy: FusionStrategy

    rule_flagged: bool
    ordinal_risk_score: float | None
    ml_flagged: bool
    malicious_decision_score: float
    calibrated_probability: float | None

    #: Present only for STACKED: the meta-learner's own output. ``None`` for the
    #: boolean gates, which have no score of their own to report.
    stacked_probability: float | None
    fused_flagged: bool

    @model_validator(mode="after")
    def check_decision(self) -> Self:
        """The fused verdict must follow from the strategy it names."""
        if self.anchor_event_time.tzinfo is None:
            raise ValueError("anchor_event_time must be timezone-aware")
        stacked = self.strategy is FusionStrategy.STACKED
        if stacked != (self.stacked_probability is not None):
            raise ValueError(
                "a stacked decision reports the meta-learner's probability, and "
                "a boolean gate reports none"
            )
        if self.strategy is FusionStrategy.OR_GATE:
            expected = self.rule_flagged or self.ml_flagged
        elif self.strategy is FusionStrategy.AND_GATE:
            expected = self.rule_flagged and self.ml_flagged
        else:
            assert self.stacked_probability is not None  # narrowed above
            expected = self.stacked_probability >= _STACKED_DECISION_THRESHOLD
        if self.fused_flagged != expected:
            raise ValueError(
                f"the fused decision contradicts {str(self.strategy)!r} applied "
                f"to this row's own evidence"
            )
        return self


# ---------------------------------------------------------------------------
# The stacked meta-learner
# ---------------------------------------------------------------------------


class StackedFusionState(SealedModel):
    """A fitted two-feature meta-learner, serialised as its own parameters.

    Project-owned end to end: the fit below is a deterministic gradient descent
    over declared iterations, and inference is the logistic function applied to
    the stored coefficients. No estimator is pickled, no library is consulted at
    load time, and two fits over identical meta-features produce identical
    bytes.
    """

    fingerprint_field: ClassVar[str] = "state_fingerprint"
    schema_version_field: ClassVar[str] = "fusion_schema_version"
    schema_version: ClassVar[str] = FUSION_SCHEMA_VERSION
    record_label: ClassVar[str] = "stacked fusion state"

    fusion_schema_version: str = FUSION_SCHEMA_VERSION
    feature_names: tuple[str, ...] = STACKED_FEATURE_NAMES
    coefficients: tuple[float, ...]
    intercept: float
    decision_threshold: float = _STACKED_DECISION_THRESHOLD

    iterations: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    l2_penalty: float = Field(ge=0.0)

    #: How the meta-features were produced.  Pinned to the out-of-fold contract:
    #: a state claiming any other provenance would be a stacker fitted on rows
    #: its base model had already seen.
    meta_feature_source: str = "out_of_fold_train"
    fold_count: int = Field(ge=2)
    train_row_count: int = Field(ge=1)
    positive_count: int = Field(ge=1)
    negative_count: int = Field(ge=1)

    state_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_state(self) -> Self:
        """The parameters are finite, declared, and out-of-fold by construction."""
        if self.feature_names != STACKED_FEATURE_NAMES:
            raise ValueError(
                f"the stacker consumes {list(STACKED_FEATURE_NAMES)}; a state "
                f"declaring other inputs was fitted on something else"
            )
        if len(self.coefficients) != len(self.feature_names):
            raise ValueError("a coefficient is stored for each declared feature")
        for value in (*self.coefficients, self.intercept):
            if not math.isfinite(value):
                raise ValueError("a fitted parameter must be finite")
        if self.meta_feature_source != "out_of_fold_train":
            raise ValueError(
                "a stacked state is fitted on out-of-fold TRAIN predictions; an "
                "in-sample stacker learns that its base model is always right"
            )
        if self.train_row_count != self.positive_count + self.negative_count:
            raise ValueError("class support does not sum to the row count")
        return self

    def probability(self, *, ml_score: float, rule_flagged: bool) -> float:
        """Return the meta-learner's probability for one row."""
        features = (float(ml_score), 1.0 if rule_flagged else 0.0)
        total = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, features, strict=True)
        )
        return quantize(sigmoid(total))


@dataclass(frozen=True, slots=True)
class MetaRow:
    """One out-of-fold meta-feature row: what the base systems said, and the truth.

    ``ml_score`` must have come from a base model fitted **without** this row.
    Nothing in this module can verify that from the number alone, which is why
    :func:`campaign_folds` exists and why the fitted state records the fold
    count it was built under.
    """

    ml_score: float
    rule_flagged: bool
    malicious: bool


def campaign_folds(
    campaigns: Sequence[str | None], *, fold_count: int
) -> tuple[int, ...]:
    """Assign each row a fold, cutting only at campaign boundaries.

    A campaign is a coordinated burst: its events share a cause, and splitting
    one across a fold boundary would let the base model learn a campaign in
    training and be scored on the rest of it out of fold. Every event of one
    campaign therefore lands in the same fold.

    Rows with no campaign are assigned round-robin by position in the canonical
    order, so ordinary background traffic still spreads evenly. The assignment
    is a pure function of the campaign identifiers and the row order, with no
    randomness anywhere.

    Raises:
        DataValidationError: on fewer than two folds, or on a fold count larger
            than the number of distinct groups to spread across.
    """
    if fold_count < 2:
        raise DataValidationError(
            "out-of-fold meta-features need at least two folds; a single fold is "
            "the in-sample fit this contract exists to prevent"
        )
    named = sorted({item for item in campaigns if item})
    if len(named) < fold_count:
        raise DataValidationError(
            f"{len(named)} distinct campaign(s) cannot be spread across "
            f"{fold_count} fold(s) without splitting one"
        )
    # Sorted, then dealt in order: the same campaigns always land in the same
    # folds, on any machine, in any run.
    assigned = {name: index % fold_count for index, name in enumerate(named)}
    folds: list[int] = []
    unnamed = 0
    for item in campaigns:
        if item:
            folds.append(assigned[item])
            continue
        folds.append(unnamed % fold_count)
        unnamed += 1
    return tuple(folds)


def fit_stacked_fusion(
    rows: Sequence[MetaRow], *, fold_count: int
) -> StackedFusionState:
    """Fit the meta-learner on out-of-fold meta-features.

    Deterministic gradient descent from a zero initialisation over a declared
    number of iterations. No random seed, no early stopping, and no convergence
    criterion read from the data: every one of those would be a decision made
    from the rows, and the point of this fit is that it is reproducible from
    them.

    Raises:
        ModelTrainingError: when the rows carry a single class, which gives the
            logistic fit nothing to separate.
    """
    positives = sum(1 for row in rows if row.malicious)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
        raise ModelTrainingError(
            "a stacked fusion needs both classes among its out-of-fold rows; a "
            "single-class fit has no decision to learn"
        )

    features = [(float(row.ml_score), 1.0 if row.rule_flagged else 0.0) for row in rows]
    targets = [1.0 if row.malicious else 0.0 for row in rows]
    weights = [0.0, 0.0]
    intercept = 0.0
    count = float(len(rows))

    for _ in range(_STACKED_ITERATIONS):
        gradient = [0.0, 0.0]
        bias_gradient = 0.0
        for values, target in zip(features, targets, strict=True):
            prediction = sigmoid(
                intercept + weights[0] * values[0] + weights[1] * values[1]
            )
            error = prediction - target
            gradient[0] += error * values[0]
            gradient[1] += error * values[1]
            bias_gradient += error
        for index in range(2):
            weights[index] -= _STACKED_LEARNING_RATE * (
                gradient[index] / count + _STACKED_L2 * weights[index]
            )
        intercept -= _STACKED_LEARNING_RATE * (bias_gradient / count)

    return StackedFusionState.seal(
        coefficients=(quantize(weights[0]), quantize(weights[1])),
        intercept=quantize(intercept),
        iterations=_STACKED_ITERATIONS,
        learning_rate=_STACKED_LEARNING_RATE,
        l2_penalty=_STACKED_L2,
        fold_count=fold_count,
        train_row_count=len(rows),
        positive_count=positives,
        negative_count=negatives,
    )


# ---------------------------------------------------------------------------
# Applying a strategy
# ---------------------------------------------------------------------------


def fuse(
    strategy: FusionStrategy,
    *,
    anchor_event_id: str,
    anchor_event_time: datetime,
    rule: RuleEvidence,
    ml: MLEvidence,
    stacked: StackedFusionState | None = None,
) -> FusionDecision:
    """Return the fused decision for one anchor under *strategy*.

    Raises:
        ModelTrainingError: when STACKED is requested without a fitted state.
            There is no default stacker: an unfitted one would be a constant
            dressed as a model.
    """
    probability: float | None = None
    if strategy is FusionStrategy.OR_GATE:
        flagged = rule.flagged or ml.flagged
    elif strategy is FusionStrategy.AND_GATE:
        flagged = rule.flagged and ml.flagged
    else:
        if stacked is None:
            raise ModelTrainingError(
                "a stacked fusion needs its fitted meta-learner; there is no "
                "default stacker to fall back on"
            )
        probability = stacked.probability(
            ml_score=ml.fusion_score, rule_flagged=rule.flagged
        )
        flagged = probability >= stacked.decision_threshold
    return FusionDecision(
        anchor_event_id=anchor_event_id,
        anchor_event_time=anchor_event_time,
        strategy=strategy,
        rule_flagged=rule.flagged,
        ordinal_risk_score=rule.ordinal_risk_score,
        ml_flagged=ml.flagged,
        malicious_decision_score=ml.decision_score,
        calibrated_probability=ml.calibrated_probability,
        stacked_probability=probability,
        fused_flagged=flagged,
    )


# ---------------------------------------------------------------------------
# Validation-only selection
# ---------------------------------------------------------------------------


class FusionCandidateEvidence(BaseModel):
    """What one strategy achieved on validation-B, and whether it qualified."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: FusionStrategy
    metrics: BinaryTestMetrics
    eligible: bool
    blocking_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def check_candidate(self) -> Self:
        """An eligible candidate names no blocking reason, and vice versa."""
        if self.eligible != (not self.blocking_reasons):
            raise ValueError(
                "an eligible candidate names no blocking reason, and a blocked "
                "one names at least one"
            )
        return self


class FusionSelection(SealedModel):
    """The frozen hybrid strategy, chosen on validation-B and nothing else.

    Sealed and fingerprinted so a TEST evaluation can bind it. Its identity
    covers the candidate set, the rule configuration, the ML champion, the
    validation evidence it was chosen from, and the decision semantics -- so a
    changed rule configuration or a changed champion is a different selection
    rather than the same one with different numbers.

    **No TEST quantity is anywhere in it.** There is no field for one, and
    :func:`select_fusion_strategy` takes no TEST argument.
    """

    fingerprint_field: ClassVar[str] = "selection_fingerprint"
    schema_version_field: ClassVar[str] = "fusion_schema_version"
    schema_version: ClassVar[str] = FUSION_SCHEMA_VERSION
    record_label: ClassVar[str] = "fusion selection"

    fusion_schema_version: str = FUSION_SCHEMA_VERSION
    status: SelectionStatus
    selected_strategy: FusionStrategy | None
    stacked_state_fingerprint: Sha256Hex | None

    candidates: tuple[FusionCandidateEvidence, ...]
    considered_strategies: tuple[FusionStrategy, ...] = FUSION_CANDIDATES

    #: The objective the selection maximised, declared before the evidence was
    #: seen and recorded so a later reader can check what was optimised.
    objective: str = "max_f1_at_min_detection_rate"
    min_detection_rate: float
    max_false_positive_rate: float
    min_validation_positive_rows: int = Field(ge=1)
    min_validation_benign_rows: int = Field(ge=1)
    support_status: MetricStatus

    validation_row_count: int = Field(ge=0)
    validation_positive_count: int = Field(ge=0)
    validation_benign_count: int = Field(ge=0)

    #: Everything a change to which makes this a different selection.
    rule_configuration_fingerprint: Sha256Hex
    champion_lock_fingerprint: Sha256Hex
    champion_freeze_record_id: str
    validation_evidence_fingerprint: Sha256Hex
    fusion_config_fingerprint: Sha256Hex
    #: How the out-of-fold folds were cut, when STACKED took part.  ``None``
    #: when it did not: a selection that never considered stacking has no fold
    #: definition to name, and naming one would be a lineage nobody built.
    oof_fold_definition_fingerprint: Sha256Hex | None = None
    #: The out-of-fold *evidence* itself -- the scores, the outcomes, and the
    #: fold each row was held out of.  Distinct from the fold definition on
    #: purpose: two runs can cut identical folds and still produce different
    #: scores if the base recipe changed, and that must be a different
    #: selection.
    oof_evidence_fingerprint: Sha256Hex | None = None
    #: The champion family and hyperparameters the fold refits used as their
    #: base recipe.  A stacker fitted over a different base model is a
    #: different stacker even when every fold boundary matches.
    base_model_recipe_fingerprint: Sha256Hex | None = None
    #: Why STACKED could not be constructed, when it could not.  The gates may
    #: still be compared against each other -- an unavailable stacker is a
    #: missing candidate, not a broken selection.
    stacked_unavailable_reason: str | None = None

    selection_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_selection(self) -> Self:
        """A selection names a strategy exactly when it found one."""
        selected = self.status is SelectionStatus.SELECTED
        if selected != (self.selected_strategy is not None):
            raise ValueError(
                "a selected fusion names its strategy, and an unselected one "
                "names none; there is no default hybrid"
            )
        if (
            selected
            and self.selected_strategy is FusionStrategy.STACKED
            and (self.stacked_state_fingerprint is None)
        ):
            raise ValueError("a selected stacked fusion names the state it applies")
        if self.stacked_state_fingerprint is not None and not any(
            item.strategy is FusionStrategy.STACKED for item in self.candidates
        ):
            raise ValueError(
                "a stacked state is named by a selection that considered stacking"
            )
        if selected:
            chosen = next(
                (
                    item
                    for item in self.candidates
                    if item.strategy is self.selected_strategy
                ),
                None,
            )
            if chosen is None or not chosen.eligible:
                raise ValueError(
                    "the selected strategy is not among its own eligible candidates"
                )
            if self.support_status is not MetricStatus.MEASURED:
                raise ValueError(
                    "a fusion selected from unmeasurable validation support is "
                    "not a selection"
                )
        return self


def fusion_config_fingerprint(
    *,
    min_detection_rate: float,
    max_false_positive_rate: float,
    min_validation_positive_rows: int,
    min_validation_benign_rows: int,
) -> str:
    """Return the digest of the fusion-selection contract this build applies."""
    return digest(
        {
            "fusion_schema_version": FUSION_SCHEMA_VERSION,
            "candidates": [str(item) for item in FUSION_CANDIDATES],
            "objective": "max_f1_at_min_detection_rate",
            "stacked_features": list(STACKED_FEATURE_NAMES),
            "stacked_iterations": _STACKED_ITERATIONS,
            "stacked_learning_rate": _STACKED_LEARNING_RATE,
            "stacked_l2_penalty": _STACKED_L2,
            "stacked_decision_threshold": _STACKED_DECISION_THRESHOLD,
            "meta_feature_source": "out_of_fold_train",
            "min_detection_rate": quantize(min_detection_rate),
            "max_false_positive_rate": quantize(max_false_positive_rate),
            "min_validation_positive_rows": min_validation_positive_rows,
            "min_validation_benign_rows": min_validation_benign_rows,
        }
    )


def select_fusion_strategy(
    *,
    decisions: Mapping[FusionStrategy, Sequence[FusionDecision]],
    malicious: Sequence[bool],
    min_detection_rate: float,
    max_false_positive_rate: float,
    min_validation_positive_rows: int,
    min_validation_benign_rows: int,
    rule_configuration_fingerprint: str,
    champion_lock_fingerprint: str,
    champion_freeze_record_id: str,
    validation_evidence_fingerprint: str,
    stacked_state_fingerprint: str | None = None,
    stacked_unavailable_reason: str | None = None,
    oof_fold_definition_fingerprint: str | None = None,
    oof_evidence_fingerprint: str | None = None,
    base_model_recipe_fingerprint: str | None = None,
    unavailable_strategies: Mapping[FusionStrategy, str] | None = None,
) -> FusionSelection:
    """Choose a hybrid strategy from validation-B evidence, or record that none qualified.

    Every argument is validation-side. There is no TEST parameter, and adding
    one would be the change this signature exists to make visible.

    The objective is declared in advance: among the strategies that clear the
    detection floor and the false-positive ceiling, take the highest F1; ties go
    to the earlier strategy in :data:`FUSION_CANDIDATES`, which is a fixed order
    rather than one chosen from the evidence.

    Raises:
        DataValidationError: when a candidate's decisions do not align with the
            outcomes, or when the candidate set is not the declared one.
    """
    if set(decisions) != set(FUSION_CANDIDATES):
        raise DataValidationError(
            "a fusion selection compares exactly the declared candidate set; "
            "comparing a subset would make the winner depend on who was entered"
        )

    positives = sum(1 for flag in malicious if flag)
    negatives = len(malicious) - positives
    supported = (
        positives >= min_validation_positive_rows
        and negatives >= min_validation_benign_rows
    )

    blocked_by_construction = dict(unavailable_strategies or {})
    candidates: list[FusionCandidateEvidence] = []
    for strategy in FUSION_CANDIDATES:
        rows = decisions[strategy]
        unavailable = blocked_by_construction.get(strategy)
        if unavailable is None and len(rows) != len(malicious):
            raise DataValidationError(
                f"{str(strategy)!r} produced {len(rows)} decision(s) for "
                f"{len(malicious)} outcome(s)"
            )
        # A candidate nobody could construct is measured on nothing, and every
        # one of its rates is therefore unavailable. That is the honest record:
        # it was not evaluated and found wanting, it was never built.
        scored_outcomes = () if unavailable is not None else malicious
        metrics = binary_metrics(
            flags=[row.fused_flagged for row in rows] if unavailable is None else [],
            malicious=scored_outcomes,
            score_unavailable_reason=(
                "fused_boolean: a fused decision is a boolean, and the two "
                "systems behind it are on different scales"
            ),
            min_positive_rows=min_validation_positive_rows,
            min_benign_rows=min_validation_benign_rows,
        )
        blocking: list[str] = []
        if unavailable is not None:
            # A strategy nobody could construct is a missing candidate, not a
            # failed one. The others still compete; the reason is recorded so a
            # reader can tell "did not qualify" from "could not be built".
            blocking.append(unavailable)
        if not supported:
            blocking.append("insufficient_validation_support")
        if metrics.recall.value is None or metrics.recall.value < min_detection_rate:
            blocking.append("below_min_detection_rate")
        if (
            metrics.false_positive_rate.value is None
            or metrics.false_positive_rate.value > max_false_positive_rate
        ):
            blocking.append("above_max_false_positive_rate")
        candidates.append(
            FusionCandidateEvidence(
                strategy=strategy,
                metrics=metrics,
                eligible=not blocking,
                blocking_reasons=tuple(blocking),
            )
        )

    eligible = [item for item in candidates if item.eligible]
    if not supported:
        status = SelectionStatus.INSUFFICIENT_VALIDATION_SUPPORT
        chosen: FusionStrategy | None = None
    elif not eligible:
        # A measured negative. Falling back to OR_GATE here would publish a
        # hybrid nobody selected, and it would be indistinguishable from one
        # that had qualified.
        status = SelectionStatus.NO_FEASIBLE_THRESHOLD
        chosen = None
    else:
        status = SelectionStatus.SELECTED
        chosen = max(
            eligible,
            key=lambda item: (
                item.metrics.f1 if item.metrics.f1 is not None else -1.0,
                -FUSION_CANDIDATES.index(item.strategy),
            ),
        ).strategy

    return FusionSelection.seal(
        status=status,
        selected_strategy=chosen,
        stacked_state_fingerprint=stacked_state_fingerprint,
        candidates=tuple(candidates),
        min_detection_rate=quantize(min_detection_rate),
        max_false_positive_rate=quantize(max_false_positive_rate),
        min_validation_positive_rows=min_validation_positive_rows,
        min_validation_benign_rows=min_validation_benign_rows,
        support_status=(
            MetricStatus.MEASURED if supported else MetricStatus.INSUFFICIENT_SUPPORT
        ),
        validation_row_count=len(malicious),
        validation_positive_count=positives,
        validation_benign_count=negatives,
        rule_configuration_fingerprint=rule_configuration_fingerprint,
        champion_lock_fingerprint=champion_lock_fingerprint,
        champion_freeze_record_id=champion_freeze_record_id,
        validation_evidence_fingerprint=validation_evidence_fingerprint,
        oof_fold_definition_fingerprint=oof_fold_definition_fingerprint,
        oof_evidence_fingerprint=oof_evidence_fingerprint,
        base_model_recipe_fingerprint=base_model_recipe_fingerprint,
        stacked_unavailable_reason=stacked_unavailable_reason,
        fusion_config_fingerprint=fusion_config_fingerprint(
            min_detection_rate=min_detection_rate,
            max_false_positive_rate=max_false_positive_rate,
            min_validation_positive_rows=min_validation_positive_rows,
            min_validation_benign_rows=min_validation_benign_rows,
        ),
    )


def _assert_no_blended_score_field() -> None:
    """Fail at import if a fusion schema grows somewhere to blend the two scales."""
    forbidden = {
        "combined_score",
        "blended_score",
        "fused_score",
        "weighted_score",
        "risk_probability",
        "rule_probability",
        "total_score",
    }
    models: tuple[type[BaseModel], ...] = (
        FusionDecision,
        RuleEvidence,
        MLEvidence,
        StackedFusionState,
        FusionSelection,
    )
    for model in models:
        offending = sorted(set(model.model_fields) & forbidden)
        if offending:
            raise ValueError(
                f"{model.__name__} declares blended-score field(s) {offending}; "
                f"an ordinal magnitude and a probability do not add"
            )


_assert_no_blended_score_field()


def _assert_candidate_set_is_the_enum() -> None:
    """Fail at import if the declared candidates drift from the enum."""
    if set(FUSION_CANDIDATES) != set(FusionStrategy):
        raise ValueError(
            "the fusion candidate set must be every declared strategy; a "
            "silently omitted candidate is a comparison nobody can audit"
        )


_assert_candidate_set_is_the_enum()
