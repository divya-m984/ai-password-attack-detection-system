"""The locked TEST evaluation: verify everything, then open the labels once.

This is the first module in Phase 5 permitted to see a TEST outcome, and the
order in which it does so is the whole design. Every decision that could change
what the TEST predictions *are* -- the model, the preprocessing, the calibrator,
the operating point, the category head, the fusion strategy -- was frozen and is
re-verified **before** a single label is looked at. By the time the outcomes
arrive there is nothing left to tune, so seeing them cannot change anything.

The order, and why each step is where it is:

1.  load the champion lock and verify its whole chain
2.  confirm the champion-freeze receipt in the ledger
3.  confirm the selected training run
4.  read the published TEST prediction manifest
5.  bind the exact prediction identity
6.  verify the frozen Phase 4 rule configuration
7.  take the **validation-only** frozen fusion selection
8.  confirm that selection carries no TEST-derived input
9.  assemble the complete intended evaluation configuration
10. verify everything about the common TEST universe that can be checked
    without labels
11. **only now** accept the TEST outcomes
12. align them to the frozen predictions by anchor identity, internally
13. compute rule-only, ML-only, and hybrid metrics over that one population
14. compute the category evaluation
15. compute the experimental novel-holdout evaluation, separately
16. render the deterministic reports
17. build the receipt
18. validate the staged evaluation
19. stage, 20. promote atomically, 21. append the ledger receipt last

There is no preliminary pass, no debug report, and no "just to see" invocation:
:func:`evaluate_test` computes the metrics once, under a configuration that was
complete before it was called.

**This module reads no Parquet and imports no label reader.** The exact
allowlist stays ``{detection.evaluation, ml.dataset}``, unchanged by this
milestone. TEST outcomes arrive as :class:`TestOutcome` values that a permitted
reader already produced, so the composition root decides what may be read and
this module only decides what may be computed.

**Nothing here selects anything.** No champion is re-chosen, no threshold moves,
no fusion strategy is re-run, and no metric feeds back into a configuration. The
final comparison is descriptive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import (
    DataValidationError,
    ExperimentPublicationError,
    ModelNotReadyError,
)
from password_attack_detector.ml.alerts import (
    ComparisonDecision,
    DecisionAlertPolicy,
)
from password_attack_detector.ml.calibration import SealedModel, digest, quantize
from password_attack_detector.ml.comparison import (
    SystemComparison,
    build_comparison,
    comparison_config_fingerprint,
    evaluate_system,
    require_common_universe,
)
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    ComparisonSystem,
    ExperimentRecordType,
    FusionStrategy,
    MetricStatus,
    MLSplit,
    SelectionStatus,
    TestEvaluationStatus,
)
from password_attack_detector.ml.fusion import (
    FusionSelection,
    MLEvidence,
    RuleEvidence,
    StackedFusionState,
    fuse,
)
from password_attack_detector.ml.ledger import (
    ExperimentLedger,
    LedgerAppendResult,
    TestEvaluationRecord,
)
from password_attack_detector.ml.metrics import (
    METRIC_DEFINITION_VERSION,
    Rate,
    metric_definition_fingerprint,
    rate,
)
from password_attack_detector.ml.prediction_manifest import PredictionManifest
from password_attack_detector.ml.predictions import (
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
    FrozenChampion,
)
from password_attack_detector.ml.schemas import ExperimentRecordIdentity, Sha256Hex

__all__ = [
    "ANOMALY_HOLDOUT_JSON",
    "ANOMALY_HOLDOUT_MD",
    "CATEGORY_EVALUATION_JSON",
    "CATEGORY_EVALUATION_MD",
    "EVALUATIONS_DIR",
    "EVALUATION_RECEIPT_FILE",
    "EVALUATION_SCHEMA_VERSION",
    "ML_EVALUATION_JSON",
    "ML_EVALUATION_MD",
    "SYNTHETIC_TEST_CAVEAT",
    "SYSTEM_COMPARISON_JSON",
    "SYSTEM_COMPARISON_MD",
    "AnomalyHoldoutEvaluation",
    "CategoryTestEvaluation",
    "EvaluationPublication",
    "TestEvaluation",
    "TestOutcome",
    "evaluate_test",
    "publish_evaluation",
    "reconcile_evaluations",
    "test_label_fingerprint",
    "test_population_fingerprint",
]

#: The evaluation contract's own version.
EVALUATION_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Where published evaluations live under the artifact root.
EVALUATIONS_DIR: Final[str] = "evaluations"
EVALUATION_RECEIPT_FILE: Final[str] = "test_evaluation.json"

ML_EVALUATION_JSON: Final[str] = "ml_evaluation.json"
ML_EVALUATION_MD: Final[str] = "ml_evaluation.md"
SYSTEM_COMPARISON_JSON: Final[str] = "system_comparison.json"
SYSTEM_COMPARISON_MD: Final[str] = "system_comparison.md"
CATEGORY_EVALUATION_JSON: Final[str] = "category_evaluation.json"
CATEGORY_EVALUATION_MD: Final[str] = "category_evaluation.md"
ANOMALY_HOLDOUT_JSON: Final[str] = "anomaly_holdout.json"
ANOMALY_HOLDOUT_MD: Final[str] = "anomaly_holdout.md"

#: Rendered at the top of every report this module writes.
SYNTHETIC_TEST_CAVEAT: Final[str] = (
    "These figures describe synthetic authentication traffic with known ground "
    "truth. They measure how these systems behave on generated data and are not "
    "evidence of real-world detection effectiveness. No number here is a "
    "production claim."
)

#: The one reason a rule-only system has no discrimination metric.
_RULE_SCORE_UNAVAILABLE: Final[str] = (
    "ordinal_rule_risk_is_not_a_discrimination_score: Phase 4 emits a bounded "
    "0-100 severity magnitude, and reinterpreting it as a ranking score would "
    "manufacture a metric out of a quantity that was never meant to rank"
)

#: The one reason a fused boolean has none.
_FUSED_SCORE_UNAVAILABLE: Final[str] = (
    "fused_boolean: a fused decision is a boolean, and the two systems behind "
    "it live on different scales"
)


@dataclass(frozen=True, slots=True)
class TestOutcome:
    """One TEST row's ground truth, as a permitted reader already produced it.

    Deliberately a plain typed value rather than an import from a label reader.
    This module never opens a table: the exact reader allowlist stays
    ``{detection.evaluation, ml.dataset}``, and the composition root that is
    allowed to read decides what to hand over.
    """

    anchor_event_id: str
    malicious: bool
    known_category: str | None
    split: MLSplit


def test_label_fingerprint(outcomes: Sequence[TestOutcome]) -> str:
    """Return the digest of exactly the TEST outcomes this evaluation scored.

    Scoped to the evaluated population and nothing wider. A digest over the
    whole label table would move whenever somebody added a row this evaluation
    never saw -- including a novel-holdout row, which must not touch a
    supervised receipt's identity.

    Anchor identifiers take part so the digest is order-independent and exact;
    none of them is published. What leaves this function is a hash.
    """
    return digest(
        {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "rows": sorted(
                (
                    [
                        item.anchor_event_id,
                        item.malicious,
                        item.known_category,
                    ]
                    for item in outcomes
                ),
                key=lambda row: str(row[0]),
            ),
        }
    )


def test_population_fingerprint(outcomes: Sequence[TestOutcome]) -> str:
    """Return the digest of exactly which rows were evaluated, and under what split.

    Separate from the label digest on purpose: a row entering or leaving the
    evaluated population and a row's outcome changing are two different
    findings, and one digest for both would tell a reader that something moved
    without telling them what.
    """
    return digest(
        {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "eligibility": "test_split_membership_only",
            "rows": sorted(
                ([item.anchor_event_id, str(item.split)] for item in outcomes),
                key=lambda row: str(row[0]),
            ),
        }
    )


# ---------------------------------------------------------------------------
# Category evaluation
# ---------------------------------------------------------------------------


class CategoryClassMetrics(BaseModel):
    """One known class's TEST support and its per-class rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_name: str
    true_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    precision: Rate
    recall: Rate


class CategoryTestEvaluation(SealedModel):
    """How the frozen category head performed, with routing kept separate.

    Two questions, answered separately because they have different denominators:
    how many rows the binary head *routed* to triage, and how the head did on
    the rows it was asked about. A single "category accuracy" over the whole
    TEST split would silently divide by rows the head never saw.

    Three states stay apart throughout: ``not_applicable`` (the binary head did
    not flag the row, so nothing was asked), ``unknown`` (it was asked and the
    best class score fell below the frozen floor), and a known class.
    """

    fingerprint_field: ClassVar[str] = "evaluation_fingerprint"
    schema_version_field: ClassVar[str] = "evaluation_schema_version"
    schema_version: ClassVar[str] = EVALUATION_SCHEMA_VERSION
    record_label: ClassVar[str] = "category test evaluation"

    evaluation_schema_version: str = EVALUATION_SCHEMA_VERSION

    #: Routing, from the binary layer.
    scored_row_count: int = Field(ge=0)
    applicable_row_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)

    #: Among applicable rows.
    known_output_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    known_output_coverage: Rate
    abstention_rate: Rate
    #: Applicable rows whose ground truth carries no known malicious category --
    #: a benign row the binary head flagged. Counted separately: the head cannot
    #: be right or wrong about a class the row does not have.
    applicable_without_true_category: int = Field(ge=0)

    #: Correctness, over the rows that have both a known prediction and a known
    #: truth.
    scorable_count: int = Field(ge=0)
    accuracy_on_known_outputs: Rate
    #: Every applicable row with a known truth, counting an abstention as not
    #: correct. Named so nobody mistakes it for the figure above.
    applicable_set_accuracy: Rate

    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None

    class_order: tuple[str, ...]
    per_class: tuple[CategoryClassMetrics, ...]
    #: Rows counted by (true class or ``none``, predicted class or ``unknown``),
    #: in deterministic order.
    confusion: tuple[tuple[str, str, int], ...]

    min_category_score: float
    support_status: MetricStatus
    evaluation_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_evaluation(self) -> Self:
        """The three states partition the scored rows, and the classes are declared."""
        if self.applicable_row_count + self.not_applicable_count != (
            self.scored_row_count
        ):
            raise ValueError(
                "applicable and not-applicable counts do not sum to the rows "
                "scored; a row the binary head never flagged is not an abstention"
            )
        if self.known_output_count + self.unknown_count != self.applicable_row_count:
            raise ValueError(
                "known outputs and abstentions do not sum to the applicable rows"
            )
        if tuple(item.class_name for item in self.per_class) != self.class_order:
            raise ValueError("per-class metrics must be given in the frozen order")
        if UNKNOWN_CATEGORY in self.class_order:
            raise ValueError(
                f"{UNKNOWN_CATEGORY!r} is the abstention outcome, not a class"
            )
        keys = [(row[0], row[1]) for row in self.confusion]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError(
                "the confusion matrix is given once per cell, in deterministic order"
            )
        return self


def evaluate_category(
    *,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    outcomes: Mapping[str, TestOutcome],
    min_category_score: float,
    class_order: Sequence[str],
) -> CategoryTestEvaluation:
    """Evaluate the frozen category head over the rows it was actually asked about.

    Binary-negative rows are **not applicable** and are never counted as
    abstentions: the head was not asked about them, and counting them as
    refusals would inflate the abstention rate with rows it never saw.
    """
    declared = tuple(class_order)
    applicable = [row for row in binary if row.flagged_malicious]
    predicted_by_anchor = {row.anchor_event_id: row for row in (category or ())}

    known_output = 0
    unknown = 0
    without_truth = 0
    scorable = 0
    correct = 0
    per_class_true = dict.fromkeys(declared, 0)
    per_class_predicted = dict.fromkeys(declared, 0)
    per_class_correct = dict.fromkeys(declared, 0)
    confusion: dict[tuple[str, str], int] = {}

    for row in applicable:
        prediction = predicted_by_anchor.get(row.anchor_event_id)
        assigned = (
            UNKNOWN_CATEGORY if prediction is None else prediction.predicted_scenario
        )
        truth = outcomes[row.anchor_event_id].known_category
        if assigned == UNKNOWN_CATEGORY:
            unknown += 1
        else:
            known_output += 1
            per_class_predicted[assigned] = per_class_predicted.get(assigned, 0) + 1
        if truth is None:
            without_truth += 1
        else:
            per_class_true[truth] = per_class_true.get(truth, 0) + 1
            scorable += 1
            if assigned == truth:
                correct += 1
                per_class_correct[truth] = per_class_correct.get(truth, 0) + 1
        key = (truth or "none", assigned)
        confusion[key] = confusion.get(key, 0) + 1

    per_class = tuple(
        CategoryClassMetrics(
            class_name=name,
            true_count=per_class_true[name],
            predicted_count=per_class_predicted[name],
            correct_count=per_class_correct[name],
            precision=rate(per_class_correct[name], per_class_predicted[name]),
            recall=rate(per_class_correct[name], per_class_true[name]),
        )
        for name in declared
    )
    macro_precision = _macro([item.precision.value for item in per_class])
    macro_recall = _macro([item.recall.value for item in per_class])
    macro_f1 = _macro(
        [_harmonic(item.precision.value, item.recall.value) for item in per_class]
    )

    # Only rows with a known truth can be right or wrong, so both accuracies
    # divide by that population. The difference between them is the treatment of
    # an abstention: the first excludes it, the second counts it as not correct.
    scorable_known_output = sum(
        1
        for row in applicable
        if outcomes[row.anchor_event_id].known_category is not None
        and (
            predicted_by_anchor.get(row.anchor_event_id) is not None
            and predicted_by_anchor[row.anchor_event_id].predicted_scenario
            != UNKNOWN_CATEGORY
        )
    )
    return CategoryTestEvaluation.seal(
        scored_row_count=len(binary),
        applicable_row_count=len(applicable),
        not_applicable_count=len(binary) - len(applicable),
        known_output_count=known_output,
        unknown_count=unknown,
        known_output_coverage=rate(known_output, len(applicable)),
        abstention_rate=rate(unknown, len(applicable)),
        applicable_without_true_category=without_truth,
        scorable_count=scorable,
        accuracy_on_known_outputs=rate(correct, scorable_known_output),
        applicable_set_accuracy=rate(correct, scorable),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        class_order=declared,
        per_class=per_class,
        confusion=tuple(
            (truth, assigned, count)
            for (truth, assigned), count in sorted(confusion.items())
        ),
        min_category_score=quantize(min_category_score),
        support_status=(
            MetricStatus.MEASURED if applicable else MetricStatus.INSUFFICIENT_SUPPORT
        ),
    )


def _macro(values: Sequence[float | None]) -> float | None:
    """Return the mean of the defined values, or ``None`` when none is defined.

    A class with no support contributes nothing rather than a zero: averaging a
    zero in would report a macro figure dragged down by a class that was never
    present.
    """
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return quantize(sum(defined) / len(defined))


def _harmonic(precision: float | None, recall: float | None) -> float | None:
    """Return the harmonic mean of two rates, or ``None``."""
    if precision is None or recall is None:
        return None
    total = precision + recall
    if total <= 0.0:
        return None
    return quantize(2.0 * precision * recall / total)


# ---------------------------------------------------------------------------
# The experimental novel-anomaly holdout
# ---------------------------------------------------------------------------


class AnomalyHoldoutEvaluation(SealedModel):
    """The experimental generalisation probe, reported entirely separately.

    Permanently marked. Nothing here is champion evidence, nothing here
    influenced the fusion selection, and nothing here may be tuned in response
    to what it says -- the threshold it applies was frozen from benign TRAIN or
    validation-A long before this ran.

    There is no probability field. Thresholding an unsupervised magnitude does
    not turn it into a likelihood.
    """

    fingerprint_field: ClassVar[str] = "evaluation_fingerprint"
    schema_version_field: ClassVar[str] = "evaluation_schema_version"
    schema_version: ClassVar[str] = EVALUATION_SCHEMA_VERSION
    record_label: ClassVar[str] = "anomaly holdout evaluation"

    evaluation_schema_version: str = EVALUATION_SCHEMA_VERSION
    scope: str = "novel_holdout"
    experimental: bool = True
    champion_evidence: bool = False

    row_count: int = Field(ge=0)
    anomaly_threshold: float | None
    flagged_count: int | None
    flagged_rate: Rate | None
    score_minimum: float | None
    score_maximum: float | None
    score_mean: float | None
    #: How often the probe flagged a row the holdout considers novel. Reported
    #: because the holdout's whole purpose is generalisation, and withheld as
    #: ``None`` where there is no support for it.
    novel_detection_rate: Rate | None
    support_status: MetricStatus
    evaluation_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_evaluation(self) -> Self:
        """The probe stays experimental, and a flag needs a threshold."""
        if not self.experimental or self.champion_evidence:
            raise ValueError(
                "the novel-holdout probe is permanently experimental and is "
                "never champion evidence"
            )
        if self.scope != "novel_holdout":
            raise ValueError("this evaluation covers the novel holdout only")
        if (self.anomaly_threshold is None) != (self.flagged_count is None):
            raise ValueError(
                "a flag count and the frozen threshold that produced it are "
                "reported together, or neither is"
            )
        return self


def evaluate_novel_holdout(
    *,
    scores: Sequence[AnomalyScore],
    outcomes: Mapping[str, TestOutcome],
) -> AnomalyHoldoutEvaluation:
    """Evaluate the experimental probe on the novel-anomaly holdout.

    Applies the already-frozen threshold and computes nothing that could be fed
    back into it. Every figure is descriptive.
    """
    values = [row.anomaly_score for row in scores]
    threshold = scores[0].anomaly_threshold if scores else None
    flagged = (
        None if threshold is None else sum(1 for row in scores if row.flagged_anomalous)
    )
    novel = [
        row
        for row in scores
        if outcomes.get(row.anchor_event_id) is not None
        and outcomes[row.anchor_event_id].malicious
    ]
    novel_flagged = sum(1 for row in novel if row.flagged_anomalous)
    return AnomalyHoldoutEvaluation.seal(
        row_count=len(scores),
        anomaly_threshold=threshold,
        flagged_count=flagged,
        flagged_rate=None if flagged is None else rate(flagged, len(scores)),
        score_minimum=quantize(min(values)) if values else None,
        score_maximum=quantize(max(values)) if values else None,
        score_mean=quantize(sum(values) / len(values)) if values else None,
        novel_detection_rate=(
            None if threshold is None else rate(novel_flagged, len(novel))
        ),
        support_status=(
            MetricStatus.MEASURED if scores else MetricStatus.INSUFFICIENT_SUPPORT
        ),
    )


# ---------------------------------------------------------------------------
# The locked evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestEvaluation:
    """Everything one locked evaluation produced, before it is published."""

    record: TestEvaluationRecord
    comparison: SystemComparison
    category: CategoryTestEvaluation | None
    anomaly: AnomalyHoldoutEvaluation | None
    reports: Mapping[str, str]

    @property
    def status(self) -> TestEvaluationStatus:
        """Return the evaluation's outcome."""
        return self.record.status


def _refuse(message: str) -> None:
    """Refuse an evaluation before any label is opened."""
    raise ModelNotReadyError(message)


def evaluate_test(
    *,
    champion: FrozenChampion,
    manifest: PredictionManifest,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    rule_decisions: Mapping[str, RuleEvidence],
    rule_configuration_fingerprint: str,
    fusion: FusionSelection | None,
    stacked_state: StackedFusionState | None,
    outcomes: Sequence[TestOutcome],
    alert_policy: DecisionAlertPolicy | None = None,
    anomaly: Sequence[AnomalyScore] | None = None,
    holdout_outcomes: Sequence[TestOutcome] | None = None,
    min_positive_rows: int = 1,
    min_benign_rows: int = 1,
) -> TestEvaluation:
    """Verify the frozen lineage, then evaluate once against TEST ground truth.

    The verification happens first and is not optional. *outcomes* is accepted
    as an argument rather than read here, so a caller that has not frozen its
    lineage cannot reach a metric by supplying labels earlier.

    Raises:
        ModelNotReadyError: on any frozen-lineage failure. Every one of these
            fires before the outcomes are used for anything.
        DataValidationError: when the systems do not cover one identical
            population.
    """
    # -- steps 1-5: the frozen champion and the published predictions ---------
    lock = champion.lock
    if manifest.scope is not MLSplit.TEST:
        _refuse(
            f"the supplied prediction publication covers {str(manifest.scope)!r}; "
            f"a TEST evaluation evaluates TEST predictions"
        )
    if manifest.lineage.champion_lock_fingerprint != lock.lock_fingerprint:
        _refuse(
            "the TEST predictions were produced by a different champion than the "
            "one supplied; the receipt would name a model that did not decide"
        )
    if manifest.lineage.champion_freeze_record_id != champion.freeze_record_id:
        _refuse("the predictions and the champion name different freeze receipts")
    if manifest.row_count != len(binary):
        _refuse("the supplied prediction rows do not match the manifest's count")
    if (manifest.category_row_count is None) != (category is None):
        _refuse(
            "the manifest and the supplied category rows disagree about whether a "
            "category artifact exists"
        )

    # -- steps 6-8: the rule comparator and the validation-only fusion --------
    if not rule_configuration_fingerprint:
        _refuse("a rule-only comparator names the frozen configuration it ran under")
    if fusion is not None:
        if fusion.champion_lock_fingerprint != lock.lock_fingerprint:
            _refuse("the frozen fusion selection was made against a different champion")
        if fusion.rule_configuration_fingerprint != rule_configuration_fingerprint:
            _refuse(
                "the frozen fusion selection was made against a different rule "
                "configuration"
            )
        if fusion.status is SelectionStatus.SELECTED and (
            fusion.selected_strategy is FusionStrategy.STACKED
        ):
            if stacked_state is None:
                _refuse("the selected stacked fusion has no frozen state to apply")
            elif stacked_state.state_fingerprint != fusion.stacked_state_fingerprint:
                _refuse(
                    "the supplied stacker is not the one the frozen selection chose"
                )

    # -- steps 9-10: the population, checked before any label is used ---------
    anchors = [row.anchor_event_id for row in binary]
    if len(set(anchors)) != len(anchors):
        raise DataValidationError(
            "the TEST prediction population repeats an anchor; each anchor "
            "contributes exactly one row to each system"
        )
    missing_rules = [anchor for anchor in anchors if anchor not in rule_decisions]
    if missing_rules:
        raise DataValidationError(
            f"the rule-only comparator is missing {len(missing_rules)} decision(s) "
            f"for the TEST population; a system evaluated on fewer rows than its "
            f"comparators is not a comparator"
        )

    # -- step 11: only now are the outcomes used -----------------------------
    outcome_by_anchor = {item.anchor_event_id: item for item in outcomes}
    missing_truth = [anchor for anchor in anchors if anchor not in outcome_by_anchor]
    if missing_truth:
        raise DataValidationError(
            f"{len(missing_truth)} evaluated row(s) carry no TEST outcome; the "
            f"population and the ground truth describe different rows"
        )
    non_test = [
        anchor
        for anchor in anchors
        if outcome_by_anchor[anchor].split is not MLSplit.TEST
    ]
    if non_test:
        raise DataValidationError(
            f"{len(non_test)} evaluated row(s) are not in the TEST split; a "
            f"supervised TEST evaluation covers TEST rows and only those"
        )

    # -- step 12: align, internally, by anchor identity ----------------------
    scoped = [outcome_by_anchor[anchor] for anchor in anchors]
    malicious = [item.malicious for item in scoped]
    times = [row.anchor_event_time for row in binary]

    rule_rows = [rule_decisions[anchor] for anchor in anchors]
    ml_rows = [
        MLEvidence(
            flagged=row.flagged_malicious,
            decision_score=row.malicious_decision_score,
            calibrated_probability=row.malicious_probability,
            score_kind=row.score_kind,
        )
        for row in binary
    ]

    # -- step 13: the three systems, over one identical population -----------
    decisions: dict[ComparisonSystem, tuple[ComparisonDecision, ...]] = {
        ComparisonSystem.RULE_ONLY: tuple(
            ComparisonDecision(
                anchor_event_id=anchor,
                anchor_event_time=when,
                system=ComparisonSystem.RULE_ONLY,
                flagged=evidence.flagged,
                ordinal_risk_score=evidence.ordinal_risk_score,
            )
            for anchor, when, evidence in zip(anchors, times, rule_rows, strict=True)
        ),
        ComparisonSystem.ML_ONLY: tuple(
            ComparisonDecision(
                anchor_event_id=anchor,
                anchor_event_time=when,
                system=ComparisonSystem.ML_ONLY,
                flagged=row.flagged_malicious,
                calibrated_probability=row.malicious_probability,
            )
            for anchor, when, row in zip(anchors, times, binary, strict=True)
        ),
    }

    hybrid_reason: str | None = None
    fused_rows: tuple[Any, ...] = ()
    if fusion is None:
        hybrid_reason = "no_fusion_selection: none was established before TEST"
    elif fusion.status is not SelectionStatus.SELECTED:
        hybrid_reason = (
            f"fusion_{fusion.status!s}: no hybrid strategy qualified on "
            f"validation-B, and none is substituted"
        )
    else:
        strategy = fusion.selected_strategy
        assert strategy is not None  # narrowed by the status check
        fused_rows = tuple(
            fuse(
                strategy,
                anchor_event_id=anchor,
                anchor_event_time=when,
                rule=rule_row,
                ml=ml_row,
                stacked=stacked_state,
            )
            for anchor, when, rule_row, ml_row in zip(
                anchors, times, rule_rows, ml_rows, strict=True
            )
        )
        decisions[ComparisonSystem.HYBRID] = tuple(
            ComparisonDecision(
                anchor_event_id=row.anchor_event_id,
                anchor_event_time=row.anchor_event_time,
                system=ComparisonSystem.HYBRID,
                flagged=row.fused_flagged,
                ordinal_risk_score=row.ordinal_risk_score,
                calibrated_probability=row.calibrated_probability,
            )
            for row in fused_rows
        )

    require_common_universe(decisions, expected_anchors=anchors)

    systems = [
        evaluate_system(
            system=ComparisonSystem.RULE_ONLY,
            decision_source=f"phase4_rule_configuration:{rule_configuration_fingerprint[:16]}",
            decisions=decisions[ComparisonSystem.RULE_ONLY],
            malicious=malicious,
            score_unavailable_reason=_RULE_SCORE_UNAVAILABLE,
            alert_policy=alert_policy,
            min_positive_rows=min_positive_rows,
            min_benign_rows=min_benign_rows,
        ),
        evaluate_system(
            system=ComparisonSystem.ML_ONLY,
            decision_source=f"prediction_publication:{manifest.prediction_id}",
            decisions=decisions[ComparisonSystem.ML_ONLY],
            malicious=malicious,
            scores=[row.malicious_decision_score for row in binary],
            probabilities=_probabilities(binary),
            alert_policy=alert_policy,
            min_positive_rows=min_positive_rows,
            min_benign_rows=min_benign_rows,
        ),
    ]
    if ComparisonSystem.HYBRID in decisions:
        assert fusion is not None and fusion.selected_strategy is not None
        systems.append(
            evaluate_system(
                system=ComparisonSystem.HYBRID,
                decision_source=(
                    f"fusion_selection:{fusion.selection_fingerprint[:16]}"
                    f"/{fusion.selected_strategy!s}"
                ),
                decisions=decisions[ComparisonSystem.HYBRID],
                malicious=malicious,
                score_unavailable_reason=_FUSED_SCORE_UNAVAILABLE,
                alert_policy=alert_policy,
                min_positive_rows=min_positive_rows,
                min_benign_rows=min_benign_rows,
            )
        )

    comparison = build_comparison(
        systems=systems,
        anchors=anchors,
        malicious=malicious,
        alert_policy=alert_policy,
        hybrid_unavailable_reason=hybrid_reason,
        min_positive_rows=min_positive_rows,
        min_benign_rows=min_benign_rows,
    )

    # -- step 14: the category head, on the rows it was routed ---------------
    category_evaluation = (
        None
        if champion.category is None
        else evaluate_category(
            binary=binary,
            category=category,
            outcomes=outcome_by_anchor,
            min_category_score=champion.category.min_category_score,
            class_order=champion.category.class_order,
        )
    )

    # -- step 15: the holdout, separately ------------------------------------
    holdout_evaluation = (
        None
        if anomaly is None
        else evaluate_novel_holdout(
            scores=anomaly,
            outcomes={item.anchor_event_id: item for item in (holdout_outcomes or ())},
        )
    )

    # -- steps 16-17: the reports and the receipt ----------------------------
    reports = _render_reports(
        comparison=comparison,
        category=category_evaluation,
        anomaly=holdout_evaluation,
        fusion=fusion,
        manifest=manifest,
    )
    status = (
        TestEvaluationStatus.COMPLETED
        if comparison.support_status is MetricStatus.MEASURED
        else TestEvaluationStatus.INSUFFICIENT_TEST_SUPPORT
    )
    record = _build_record(
        champion=champion,
        manifest=manifest,
        comparison=comparison,
        fusion=fusion,
        rule_configuration_fingerprint=rule_configuration_fingerprint,
        alert_policy=alert_policy,
        outcomes=scoped,
        status=status,
        reports=reports,
    )
    return TestEvaluation(
        record=record,
        comparison=comparison,
        category=category_evaluation,
        anomaly=holdout_evaluation,
        reports=reports,
    )


def _probabilities(binary: Sequence[BinaryPrediction]) -> list[float] | None:
    """Return the calibrated probabilities, or ``None`` when there are none.

    All or nothing: a partially populated column would mean the calibrator
    failed on some rows and nothing said so, and Brier over the remainder would
    be a metric on a population nobody named.
    """
    values = [row.malicious_probability for row in binary]
    if not values or any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _build_record(
    *,
    champion: FrozenChampion,
    manifest: PredictionManifest,
    comparison: SystemComparison,
    fusion: FusionSelection | None,
    rule_configuration_fingerprint: str,
    alert_policy: DecisionAlertPolicy | None,
    outcomes: Sequence[TestOutcome],
    status: TestEvaluationStatus,
    reports: Mapping[str, str],
) -> TestEvaluationRecord:
    """Return the immutable receipt for one locked evaluation.

    Identity is derived from the whole frozen evaluation state: the champion,
    the predictions, the scoped outcomes, the rule configuration, the fusion
    selection, the comparison configuration, the metric definitions, and the
    computed content. Nothing observational takes part.
    """
    lock = champion.lock
    head = champion.category
    label_digest = test_label_fingerprint(outcomes)
    population_digest = test_population_fingerprint(outcomes)
    comparison_config = comparison_config_fingerprint(alert_policy=alert_policy)
    metric_digest = metric_definition_fingerprint()
    report_digests = tuple(
        sorted(
            (name, hashlib.sha256(body.encode()).hexdigest())
            for name, body in reports.items()
        )
    )

    identity = ExperimentRecordIdentity.derive(
        record_type=ExperimentRecordType.TEST_EVALUATION,
        model_catalog_version=manifest.lineage.required_feature_schema_version,
        required_feature_schema_version=(
            manifest.lineage.required_feature_schema_version
        ),
        task=lock.task,
        model_family=lock.model_family,
        catalog_model_id=lock.catalog_model_id,
        seed=0,
        ml_config_fingerprint=lock.ml_config_fingerprint,
        model_catalog_fingerprint=lock.model_catalog_fingerprint,
        feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
        allowlist_fingerprint=lock.allowlist_fingerprint,
        eligible_feature_list_fingerprint=lock.eligible_feature_list_fingerprint,
        preprocessor_fingerprint=lock.preprocessor_fingerprint,
        validation_partition_fingerprint=lock.validation_partition_fingerprint,
        model_content_fingerprint=lock.model_content_fingerprint,
        calibration_state_fingerprint=lock.calibration_state_fingerprint,
        threshold_selection_fingerprint=lock.binary_threshold_fingerprint,
        category_abstention_fingerprint=(
            None if head is None else head.abstention.selection_fingerprint
        ),
        serializer_id=lock.serializer_id,
        serializer_version=lock.serializer_version,
        dependency_contract_fingerprint=lock.dependency_contract_fingerprint,
        # The whole frozen evaluation state, in one digest. A weak identity --
        # task, split, labels -- would collide across two evaluations that
        # differed in the rule configuration or the fusion strategy, and one
        # would silently be recorded as the other.
        candidate_fingerprint=digest(
            {
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "metric_definition_version": METRIC_DEFINITION_VERSION,
                "champion_freeze_record_id": champion.freeze_record_id,
                "champion_lock_fingerprint": lock.lock_fingerprint,
                "prediction_id": manifest.prediction_id,
                "prediction_manifest_fingerprint": (
                    manifest.prediction_manifest_fingerprint
                ),
                "prediction_content_fingerprint": (
                    manifest.prediction_content_fingerprint
                ),
                "test_label_fingerprint": label_digest,
                "evaluation_population_fingerprint": population_digest,
                "rule_configuration_fingerprint": rule_configuration_fingerprint,
                "fusion_selection_fingerprint": (
                    None if fusion is None else fusion.selection_fingerprint
                ),
                "comparison_configuration_fingerprint": comparison_config,
                "metric_definition_fingerprint": metric_digest,
                "comparison_fingerprint": comparison.comparison_fingerprint,
                "status": str(status),
            }
        ),
    )

    selected = (
        fusion.selected_strategy
        if fusion is not None and fusion.status is SelectionStatus.SELECTED
        else None
    )
    return TestEvaluationRecord.seal(
        record_type=ExperimentRecordType.TEST_EVALUATION,
        identity=identity,
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        status=status,
        champion_freeze_record_id=champion.freeze_record_id,
        champion_lock_fingerprint=lock.lock_fingerprint,
        champion_scope_key=lock.scope_key,
        validation_selection_id=lock.validation_selection_id,
        selected_run_id=lock.training_run_id,
        selected_model_id=lock.model_id,
        selected_model_content_fingerprint=lock.model_content_fingerprint,
        preprocessor_fingerprint=lock.preprocessor_fingerprint,
        calibration_state_fingerprint=lock.calibration_state_fingerprint,
        binary_threshold_fingerprint=lock.binary_threshold_fingerprint,
        category_run_id=None if head is None else head.run_id,
        category_model_content_fingerprint=(
            None if head is None else head.model.document.model_content_fingerprint
        ),
        category_abstention_fingerprint=(
            None if head is None else head.abstention.selection_fingerprint
        ),
        prediction_id=manifest.prediction_id,
        prediction_manifest_fingerprint=manifest.prediction_manifest_fingerprint,
        prediction_content_fingerprint=manifest.prediction_content_fingerprint,
        test_label_fingerprint=label_digest,
        evaluation_population_fingerprint=population_digest,
        rule_configuration_fingerprint=rule_configuration_fingerprint,
        fusion_selection_fingerprint=(
            None if selected is None or fusion is None else fusion.selection_fingerprint
        ),
        selected_fusion_strategy=selected,
        comparison_configuration_fingerprint=comparison_config,
        metric_definition_fingerprint=metric_digest,
        comparison_fingerprint=comparison.comparison_fingerprint,
        report_fingerprints=report_digests,
        row_count=comparison.row_count,
        positive_count=comparison.positive_count,
        negative_count=comparison.negative_count,
    )


def _render_reports(
    *,
    comparison: SystemComparison,
    category: CategoryTestEvaluation | None,
    anomaly: AnomalyHoldoutEvaluation | None,
    fusion: FusionSelection | None,
    manifest: PredictionManifest,
) -> dict[str, str]:
    """Return every report this evaluation publishes, keyed by file name.

    Deterministic: sorted JSON, no wall clock, no path, no identifier. Two
    evaluations of one frozen state render byte-identical documents.
    """
    reports: dict[str, str] = {}
    ml_only = comparison.for_system(ComparisonSystem.ML_ONLY)
    payload: dict[str, Any] = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "caveat": SYNTHETIC_TEST_CAVEAT,
        "scope": str(MLSplit.TEST),
        "prediction_id": manifest.prediction_id,
        "metrics": None if ml_only is None else ml_only.metrics.to_dict(),
        "support": {
            "row_count": comparison.row_count,
            "positive_count": comparison.positive_count,
            "negative_count": comparison.negative_count,
            "status": str(comparison.support_status),
        },
    }
    reports[ML_EVALUATION_JSON] = _json(payload)
    reports[ML_EVALUATION_MD] = _ml_markdown(comparison, manifest)

    reports[SYSTEM_COMPARISON_JSON] = _json(
        {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "caveat": SYNTHETIC_TEST_CAVEAT,
            "comparison": comparison.to_dict(),
            "hybrid_unavailable_reason": comparison.hybrid_unavailable_reason,
            "fusion": None if fusion is None else fusion.to_dict(),
        }
    )
    reports[SYSTEM_COMPARISON_MD] = _comparison_markdown(comparison, fusion)

    if category is not None:
        reports[CATEGORY_EVALUATION_JSON] = _json(
            {
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "caveat": SYNTHETIC_TEST_CAVEAT,
                "category": category.to_dict(),
            }
        )
        reports[CATEGORY_EVALUATION_MD] = _category_markdown(category)

    if anomaly is not None:
        reports[ANOMALY_HOLDOUT_JSON] = _json(
            {
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "caveat": SYNTHETIC_TEST_CAVEAT,
                "anomaly_holdout": anomaly.to_dict(),
            }
        )
        reports[ANOMALY_HOLDOUT_MD] = _anomaly_markdown(anomaly)
    return reports


def _json(payload: Any) -> str:
    """Return deterministic JSON: sorted keys, ASCII, trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _cell(value: float | None) -> str:
    """Render a number, or say it is unavailable rather than printing a zero."""
    return "unavailable" if value is None else f"{value:g}"


def _ml_markdown(comparison: SystemComparison, manifest: PredictionManifest) -> str:
    """Render the ML-only TEST evaluation."""
    system = comparison.for_system(ComparisonSystem.ML_ONLY)
    lines = [
        "# ML TEST evaluation",
        "",
        SYNTHETIC_TEST_CAVEAT,
        "",
        f"- Prediction: `{manifest.prediction_id}`",
        f"- Rows evaluated: {comparison.row_count:,}",
        f"- Positive support: {comparison.positive_count:,}",
        f"- Benign support: {comparison.negative_count:,}",
        "",
    ]
    if system is None:
        lines.append("No ML-only evaluation is available.")
        return "\n".join(lines) + "\n"
    metrics = system.metrics
    lines += [
        "| Metric | Value | Numerator | Denominator |",
        "|---|---|---|---|",
        f"| Precision | {_cell(metrics.precision.value)} | "
        f"{metrics.precision.numerator} | {metrics.precision.denominator} |",
        f"| Recall | {_cell(metrics.recall.value)} | "
        f"{metrics.recall.numerator} | {metrics.recall.denominator} |",
        f"| False-positive rate | {_cell(metrics.false_positive_rate.value)} | "
        f"{metrics.false_positive_rate.numerator} | "
        f"{metrics.false_positive_rate.denominator} |",
        f"| F1 | {_cell(metrics.f1)} | - | - |",
        f"| Balanced accuracy | {_cell(metrics.balanced_accuracy)} | - | - |",
        f"| PR-AUC (exact, stepwise) | {_cell(metrics.pr_auc)} | - | - |",
        "",
        f"Confusion: TP {metrics.confusion.true_positives:,}, "
        f"FP {metrics.confusion.false_positives:,}, "
        f"TN {metrics.confusion.true_negatives:,}, "
        f"FN {metrics.confusion.false_negatives:,}.",
        "",
    ]
    if metrics.calibration is not None:
        lines += [
            "## Calibration",
            "",
            f"- Brier score: {metrics.calibration.brier_score:g}",
            f"- Expected calibration error: "
            f"{metrics.calibration.expected_calibration_error:g}",
            "",
            "Measured, never applied: the calibrator was frozen before these "
            "labels were opened and is not corrected against them.",
            "",
        ]
    lines += [
        "## What this is not",
        "",
        "These figures describe one frozen champion on one synthetic TEST "
        "split. No selection, threshold, calibrator, or fusion strategy was "
        "changed in response to them, and none may be.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _comparison_markdown(
    comparison: SystemComparison, fusion: FusionSelection | None
) -> str:
    """Render the rule/ML/hybrid comparison."""
    lines = [
        "# System comparison on TEST",
        "",
        SYNTHETIC_TEST_CAVEAT,
        "",
        "Every system below decided the **same** rows, in the same order, "
        "against the same ground truth, under the same alert policy. Row order "
        "in this table is a declared order and is not a ranking.",
        "",
        "| System | Precision | Recall | FPR | F1 | PR-AUC | Alerts |",
        "|---|---|---|---|---|---|---|",
    ]
    for system in comparison.systems:
        metrics = system.metrics
        alerts = (
            "unavailable" if system.alerts is None else f"{system.alerts.alert_count:,}"
        )
        lines.append(
            f"| `{system.system}` | {_cell(metrics.precision.value)} | "
            f"{_cell(metrics.recall.value)} | "
            f"{_cell(metrics.false_positive_rate.value)} | {_cell(metrics.f1)} | "
            f"{_cell(metrics.pr_auc)} | {alerts} |"
        )
    lines += ["", "## Availability", ""]
    for system in comparison.systems:
        lines.append(
            f"- `{system.system}` — decided from {system.decision_source}; "
            f"continuous score "
            f"{'available' if system.continuous_score_available else 'unavailable'}, "
            f"calibrated probability "
            f"{'available' if system.calibrated_probability_available else 'unavailable'}."
        )
        if system.metrics.pr_auc_unavailable_reason:
            lines.append(
                f"  - PR-AUC unavailable: {system.metrics.pr_auc_unavailable_reason}"
            )
    if comparison.hybrid_unavailable_reason is not None:
        lines += [
            "",
            f"**Hybrid unavailable.** {comparison.hybrid_unavailable_reason}. No "
            f"hybrid is substituted: a fallback strategy would be a system "
            f"nobody selected, and it would be indistinguishable from one that "
            f"had qualified.",
        ]
    elif fusion is not None:
        lines += [
            "",
            f"**Hybrid strategy:** `{fusion.selected_strategy}`, selected on "
            f"validation-B under the objective `{fusion.objective}` before any "
            f"TEST label was opened.",
        ]
    lines += [
        "",
        "## What this comparison is not",
        "",
        "It is descriptive. It does not select a champion, choose a threshold, "
        "pick a fusion strategy, or change any deployment state. Every one of "
        "those decisions was frozen before these labels were read, and none of "
        "them may be revisited in the light of this table.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _category_markdown(category: CategoryTestEvaluation) -> str:
    """Render the category triage evaluation."""
    lines = [
        "# Category triage evaluation on TEST",
        "",
        SYNTHETIC_TEST_CAVEAT,
        "",
        "Category triage runs only on the rows the binary champion flagged. A "
        "row it cleared is **not applicable** — nothing was asked about it — "
        f"while `{UNKNOWN_CATEGORY}` means the head was asked and declined to "
        "commit. The two are never collapsed.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Rows scored | {category.scored_row_count:,} |",
        f"| Not applicable | {category.not_applicable_count:,} |",
        f"| Category-applicable | {category.applicable_row_count:,} |",
        f"| Known class emitted | {category.known_output_count:,} |",
        f"| Abstained (`{UNKNOWN_CATEGORY}`) | {category.unknown_count:,} |",
        f"| Abstention rate (of applicable) | "
        f"{_cell(category.abstention_rate.value)} |",
        f"| Applicable rows with no true category | "
        f"{category.applicable_without_true_category:,} |",
        f"| Accuracy on known outputs | "
        f"{_cell(category.accuracy_on_known_outputs.value)} |",
        f"| Applicable-set accuracy (abstention counted wrong) | "
        f"{_cell(category.applicable_set_accuracy.value)} |",
        f"| Macro precision | {_cell(category.macro_precision)} |",
        f"| Macro recall | {_cell(category.macro_recall)} |",
        f"| Macro F1 | {_cell(category.macro_f1)} |",
        f"| Abstention threshold | {category.min_category_score:g} |",
        "",
        "| Class | True | Predicted | Correct | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for item in category.per_class:
        lines.append(
            f"| `{item.class_name}` | {item.true_count:,} | "
            f"{item.predicted_count:,} | {item.correct_count:,} | "
            f"{_cell(item.precision.value)} | {_cell(item.recall.value)} |"
        )
    lines += ["", "| True | Predicted | Rows |", "|---|---|---|"]
    for truth, assigned, count in category.confusion:
        lines.append(f"| `{truth}` | `{assigned}` | {count:,} |")
    lines += [
        "",
        "Class order is the known-malicious scenario space, never the Phase 4 "
        "rule categories.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _anomaly_markdown(anomaly: AnomalyHoldoutEvaluation) -> str:
    """Render the experimental novel-holdout probe."""
    return (
        "\n".join(
            [
                "# Novel-anomaly holdout (experimental)",
                "",
                "**experimental — novel_holdout — not_champion_evidence**",
                "",
                SYNTHETIC_TEST_CAVEAT,
                "",
                "| Quantity | Value |",
                "|---|---|",
                f"| Rows scored | {anomaly.row_count:,} |",
                f"| Anomaly score range | {_cell(anomaly.score_minimum)} .. "
                f"{_cell(anomaly.score_maximum)} |",
                f"| Anomaly score mean | {_cell(anomaly.score_mean)} |",
                f"| Frozen threshold | {_cell(anomaly.anomaly_threshold)} |",
                "| Flagged | "
                + (
                    "unavailable"
                    if anomaly.flagged_count is None
                    else f"{anomaly.flagged_count:,}"
                )
                + " |",
                f"| Novel detection rate | "
                f"{_cell(None if anomaly.novel_detection_rate is None else anomaly.novel_detection_rate.value)} |",
                "",
                "This probe emits an **anomaly score**, never a probability. Its "
                "threshold was frozen from benign TRAIN or validation-A long before "
                "this evaluation, it influenced no champion and no fusion strategy, "
                "and nothing may be tuned in response to what it reports here.",
                "",
            ]
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


class EvaluationPublication(BaseModel):
    """What one evaluation publication did."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    status: TestEvaluationStatus
    row_count: int
    #: True when this call wrote the evaluation; false when an identical one was
    #: already published and was left exactly as it was.
    created: bool
    ledger: LedgerAppendResult


def _write(path: Path, payload: str) -> None:
    """Write *payload* to *path* and flush it to the device."""
    with path.open("wb") as handle:
        handle.write(payload.encode())
        handle.flush()
        os.fsync(handle.fileno())


def publish_evaluation(
    evaluation: TestEvaluation, *, root: Path, ledger: ExperimentLedger
) -> EvaluationPublication:
    """Publish *evaluation* under *root*, transactionally, then index it.

    Staged into a sibling directory, re-read, verified, promoted atomically, and
    only then appended to the ledger. Receipt last, deliberately: a ledger entry
    asserting an evaluation that does not exist would need an immutable record
    deleted to repair, while an unindexed valid evaluation needs only to be read.

    Raises:
        ExperimentPublicationError: when staging or promotion fails, or when a
            *different* evaluation is already published under this identity.
            Nothing is overwritten.
    """
    evaluations_root = Path(root) / EVALUATIONS_DIR
    target = evaluations_root / evaluation.record.record_id
    staging = evaluations_root / f".staging-{evaluation.record.record_id}"

    if target.exists():
        stored = _require_identical(target, evaluation.record)
        return EvaluationPublication(
            record_id=stored.record_id,
            status=stored.status,
            row_count=stored.row_count,
            created=False,
            ledger=ledger.append(stored),
        )

    evaluations_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    promoted = False
    try:
        for name, body in sorted(evaluation.reports.items()):
            _write(staging / name, body)
        _write(staging / EVALUATION_RECEIPT_FILE, evaluation.record.to_json() + "\n")
        _verify_staged(staging, evaluation)
        staging.rename(target)
        promoted = True
    except ExperimentPublicationError:
        raise
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the test evaluation could not be staged ({type(exc).__name__}); the "
            f"destination and the ledger are unchanged"
        ) from None
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)

    return EvaluationPublication(
        record_id=evaluation.record.record_id,
        status=evaluation.record.status,
        row_count=evaluation.record.row_count,
        created=True,
        ledger=ledger.append(evaluation.record),
    )


def _verify_staged(staging: Path, evaluation: TestEvaluation) -> None:
    """Raise unless the staged evaluation reads back as exactly itself."""
    for name, body in evaluation.reports.items():
        stored = (staging / name).read_text(encoding="utf-8")
        if stored != body:
            raise ExperimentPublicationError(
                "a staged evaluation report does not read back as itself"
            )
        recorded = dict(evaluation.record.report_fingerprints).get(name)
        if recorded != hashlib.sha256(stored.encode()).hexdigest():
            raise ExperimentPublicationError(
                "a staged report's digest is not the one the receipt records"
            )
    receipt = staging / EVALUATION_RECEIPT_FILE
    reloaded = TestEvaluationRecord.from_json(receipt.read_text(encoding="utf-8"))
    if reloaded.to_json() != evaluation.record.to_json():
        raise ExperimentPublicationError(
            "the staged test-evaluation receipt does not read back as itself"
        )


def _require_identical(
    target: Path, record: TestEvaluationRecord
) -> TestEvaluationRecord:
    """Return the published receipt at *target*, or refuse if it differs."""
    path = target / EVALUATION_RECEIPT_FILE
    if not path.is_file():
        raise ExperimentPublicationError(
            "an evaluation directory already exists here without a receipt; an "
            "incomplete evaluation is never completed in place"
        )
    try:
        stored = TestEvaluationRecord.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the published evaluation receipt is not readable ({type(exc).__name__})"
        ) from None
    if stored.to_json() != record.to_json():
        raise ExperimentPublicationError(
            "a different test evaluation is already published under this "
            "identity; a published evaluation is evidence and is never rewritten"
        )
    return stored


def reconcile_evaluations(*, root: Path, ledger: ExperimentLedger) -> tuple[str, ...]:
    """Index every published evaluation the ledger does not yet hold.

    The deterministic recovery for the one failure this ordering permits: an
    evaluation promoted and then not indexed. The receipt is read from the
    directory it was published with -- not rebuilt, not inferred -- and appended.
    """
    evaluations_root = Path(root) / EVALUATIONS_DIR
    if not evaluations_root.is_dir():
        return ()
    appended: list[str] = []
    for directory in sorted(evaluations_root.iterdir()):
        receipt = directory / EVALUATION_RECEIPT_FILE
        if not directory.is_dir() or not receipt.is_file():
            continue
        record = TestEvaluationRecord.from_json(receipt.read_text(encoding="utf-8"))
        if ledger.append(record).created:
            appended.append(record.record_id)
    return tuple(appended)


def _assert_no_label_reader_import() -> None:
    """Fail at import if this module ever acquires a ground-truth reader.

    The exact allowlist is ``{detection.evaluation, ml.dataset}`` and this
    milestone does not widen it: TEST outcomes arrive as :class:`TestOutcome`
    values that a permitted reader already produced.
    """
    import sys

    module = sys.modules[__name__]
    forbidden = {"LabelRecord", "SplitRecord", "CampaignRecord", "GroundTruthLabel"}
    offending = sorted(forbidden & set(vars(module)))
    if offending:
        raise ValueError(
            f"{__name__} imported label-bearing symbol(s) {offending}; TEST "
            f"outcomes reach this module as typed arguments, never as a read"
        )


_assert_no_label_reader_import()
