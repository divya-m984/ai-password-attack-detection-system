"""Comparing rule-only, ML-only, and hybrid over one identical TEST population.

A comparison is only a comparison if the three systems answered the *same*
question about the *same* rows. Most of this module is therefore about the
population rather than about the metrics: :func:`require_common_universe`
refuses a comparison in which one system is missing a decision, carries an extra
one, repeats an anchor, or disagrees about the order -- because every one of
those quietly changes a denominator, and a denominator that moves between two
columns of a table makes the table meaningless.

**Nothing here selects anything.** This module is handed frozen decisions and
outcomes and it computes. There is no ranking that feeds back into a champion,
no "winner" that changes a lock, and no code path from a TEST metric to a
configuration value. The final comparison is *descriptive*: it says what the
three systems did, and it changes nothing about what any of them is.

**Availability is a first-class outcome.** A system with no continuous
discrimination score has no PR-AUC and the report says so; a hybrid that no
validation-only selection chose is reported as unavailable rather than
substituted with a default. A cell that reads ``unavailable`` is a cell nobody
could fill, and it is never rendered as a zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.alerts import (
    COMPARISON_ALERTING_VERSION,
    ComparisonAlertingResult,
    ComparisonDecision,
    DecisionAlertBuilder,
    DecisionAlertPolicy,
)
from password_attack_detector.ml.calibration import SealedModel, digest
from password_attack_detector.ml.enums import ComparisonSystem, MetricStatus
from password_attack_detector.ml.metrics import (
    METRIC_DEFINITION_VERSION,
    BinaryTestMetrics,
    binary_metrics,
    metric_definition_fingerprint,
)
from password_attack_detector.ml.schemas import Sha256Hex

__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "SYSTEM_ORDER",
    "AlertLevelSummary",
    "SystemComparison",
    "SystemEvaluation",
    "build_comparison",
    "comparison_config_fingerprint",
    "evaluate_system",
    "population_fingerprint",
    "require_common_universe",
]

#: The comparison contract's own version.
COMPARISON_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The order systems are reported in.  Declared, so a table's row order can
#: never be mistaken for a ranking -- and so nobody can produce a ranking by
#: choosing an order after seeing the numbers.
SYSTEM_ORDER: Final[tuple[ComparisonSystem, ...]] = (
    ComparisonSystem.RULE_ONLY,
    ComparisonSystem.ML_ONLY,
    ComparisonSystem.HYBRID,
)


def require_common_universe(
    decisions: Mapping[ComparisonSystem, Sequence[ComparisonDecision]],
    *,
    expected_anchors: Sequence[str],
) -> None:
    """Raise unless every system decided exactly the expected rows, in order.

    Checked before a single metric is computed. A system silently missing a row
    would be measured on an easier population than its comparators, and the
    resulting table would compare three numbers that are not comparable.

    Raises:
        DataValidationError: on a duplicate anchor, a missing decision, an extra
            decision, or an ordering disagreement. Counts are reported; no
            identifier is.
    """
    if len(set(expected_anchors)) != len(expected_anchors):
        raise DataValidationError(
            f"the comparison population repeats an anchor "
            f"({len(expected_anchors) - len(set(expected_anchors))} duplicate(s)); "
            f"each anchor contributes exactly one row to each system"
        )
    if not expected_anchors:
        raise DataValidationError(
            "a comparison over no rows is not a comparison with no findings"
        )
    if not decisions:
        raise DataValidationError("a comparison needs at least one system")

    expected = list(expected_anchors)
    for system, rows in sorted(decisions.items(), key=lambda item: str(item[0])):
        anchors = [row.anchor_event_id for row in rows]
        if len(set(anchors)) != len(anchors):
            raise DataValidationError(
                f"{str(system)!r} decided {len(anchors) - len(set(anchors))} "
                f"anchor(s) more than once"
            )
        missing = len(set(expected) - set(anchors))
        extra = len(set(anchors) - set(expected))
        if missing or extra:
            raise DataValidationError(
                f"{str(system)!r} is missing {missing} decision(s) and carries "
                f"{extra} decision(s) outside the comparison population; a system "
                f"evaluated on a different set of rows is not a comparator"
            )
        if anchors != expected:
            raise DataValidationError(
                f"{str(system)!r} decided the comparison population in a "
                f"different order; the outcomes are aligned positionally and a "
                f"reordered system would be scored against the wrong rows"
            )
        if any(row.system is not system for row in rows):
            raise DataValidationError(
                f"a decision filed under {str(system)!r} names a different system"
            )


class AlertLevelSummary(BaseModel):
    """Alert-level quantities for one system, kept apart from event-level ones.

    An alert count is not a true positive count and the two are never mixed:
    event-level metrics answer "was this row's decision right", alert-level
    quantities answer "how much did an analyst have to look at". Both are
    reported; neither is derived from the other.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_alerting_version: str = COMPARISON_ALERTING_VERSION
    alert_count: int = Field(ge=0)
    qualifying_decision_count: int = Field(ge=0)
    grouped_decision_count: int = Field(ge=0)
    escalated_count: int = Field(ge=0)
    distinct_group_count: int = Field(ge=0)
    suppressed_count: int = Field(ge=0)
    #: Alerts per qualifying decision. ``None`` when nothing qualified: an
    #: undefined ratio, never a zero.
    alerts_per_qualifying_decision: float | None

    @model_validator(mode="after")
    def check_summary(self) -> Self:
        """The ratio exists exactly when it is defined."""
        if (self.alerts_per_qualifying_decision is None) != (
            self.qualifying_decision_count == 0
        ):
            raise ValueError(
                "the alert ratio exists exactly when something qualified; an "
                "empty denominator is unavailable, never zero"
            )
        return self


def _alert_summary(result: ComparisonAlertingResult) -> AlertLevelSummary:
    """Return the published alert-level view of one system's pass."""
    stats = result.stats
    qualifying = stats.qualifying_count
    return AlertLevelSummary(
        alert_count=stats.alert_count,
        qualifying_decision_count=qualifying,
        grouped_decision_count=stats.grouped_decision_count,
        escalated_count=stats.escalated_count,
        distinct_group_count=stats.distinct_group_count,
        suppressed_count=sum(stats.suppressed_by_reason.values()),
        alerts_per_qualifying_decision=(
            None if qualifying == 0 else round(stats.alert_count / qualifying, 9)
        ),
    )


class SystemEvaluation(BaseModel):
    """One system's TEST result: what it decided, and what it could be measured on.

    ``availability`` is as important as the metrics. A rule engine has no
    calibrated probability and no discrimination score, and saying so is more
    useful than a table of zeros in those columns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: ComparisonSystem
    decision_source: str
    metrics: BinaryTestMetrics
    alerts: AlertLevelSummary | None
    continuous_score_available: bool
    calibrated_probability_available: bool
    alert_level_available: bool

    @model_validator(mode="after")
    def check_evaluation(self) -> Self:
        """Availability flags describe what is actually present."""
        if self.continuous_score_available != (self.metrics.pr_auc is not None):
            raise ValueError(
                "a system with a continuous score reports a discrimination "
                "metric, and one without reports none"
            )
        if self.calibrated_probability_available != (
            self.metrics.calibration is not None
        ):
            raise ValueError(
                "calibration metrics are reported exactly where a calibrated "
                "probability exists"
            )
        if self.alert_level_available != (self.alerts is not None):
            raise ValueError("alert availability must match the alert summary")
        if not self.decision_source.strip():
            raise ValueError("a system names the frozen artifact it decided from")
        return self


def evaluate_system(
    *,
    system: ComparisonSystem,
    decision_source: str,
    decisions: Sequence[ComparisonDecision],
    malicious: Sequence[bool],
    scores: Sequence[float] | None = None,
    score_unavailable_reason: str | None = None,
    probabilities: Sequence[float] | None = None,
    alert_policy: DecisionAlertPolicy | None = None,
    min_positive_rows: int = 1,
    min_benign_rows: int = 1,
) -> SystemEvaluation:
    """Evaluate one system over the shared population.

    Args:
        system: which comparator this is.
        decision_source: the frozen artifact the decisions came from, named so a
            reader can tell a published prediction from a recomputed one.
        decisions: the system's decisions, aligned with *malicious*.
        malicious: the TEST outcomes, in the shared canonical order.
        scores: the system's continuous discrimination score, when it has one.
        score_unavailable_reason: why it does not, when it does not.
        probabilities: calibrated probabilities, when the system emits them.
        alert_policy: the shared alert policy. When omitted, no alert-level
            summary is produced -- for *any* system, so the comparison stays
            symmetric.
    """
    metrics = binary_metrics(
        flags=[row.flagged for row in decisions],
        malicious=malicious,
        scores=scores,
        probabilities=probabilities,
        score_unavailable_reason=score_unavailable_reason,
        min_positive_rows=min_positive_rows,
        min_benign_rows=min_benign_rows,
    )
    summary = (
        None
        if alert_policy is None
        else _alert_summary(DecisionAlertBuilder(alert_policy).build(decisions))
    )
    return SystemEvaluation(
        system=system,
        decision_source=decision_source,
        metrics=metrics,
        alerts=summary,
        continuous_score_available=scores is not None,
        calibrated_probability_available=probabilities is not None,
        alert_level_available=summary is not None,
    )


class SystemComparison(SealedModel):
    """The final descriptive comparison, sealed so a receipt can bind it.

    Carries no winner. Reporting one would invite a reader -- or a later
    milestone -- to treat a TEST result as a selection, and a selection made
    from TEST is the one thing this whole phase is arranged to prevent.
    """

    fingerprint_field: ClassVar[str] = "comparison_fingerprint"
    schema_version_field: ClassVar[str] = "comparison_schema_version"
    schema_version: ClassVar[str] = COMPARISON_SCHEMA_VERSION
    record_label: ClassVar[str] = "system comparison"

    comparison_schema_version: str = COMPARISON_SCHEMA_VERSION
    metric_definition_version: str = METRIC_DEFINITION_VERSION

    systems: tuple[SystemEvaluation, ...]
    #: Why the hybrid is absent, when it is. ``None`` when one was evaluated.
    hybrid_unavailable_reason: str | None

    row_count: int = Field(ge=1)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    support_status: MetricStatus

    population_fingerprint: Sha256Hex
    comparison_config_fingerprint: Sha256Hex
    comparison_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_comparison(self) -> Self:
        """Every system was measured on the same population, in declared order."""
        if not self.systems:
            raise ValueError("a comparison reports at least one system")
        order = [item.system for item in self.systems]
        if len(set(order)) != len(order):
            raise ValueError("a system appears twice in one comparison")
        if order != [item for item in SYSTEM_ORDER if item in set(order)]:
            raise ValueError(
                "systems are reported in the declared order; an order chosen "
                "after the numbers were seen would be a ranking"
            )
        for item in self.systems:
            if item.metrics.row_count != self.row_count:
                raise ValueError(
                    "a system was measured on a different number of rows than "
                    "the comparison population"
                )
            if item.metrics.positive_count != self.positive_count:
                raise ValueError(
                    "a system was measured against a different positive support"
                )
        present = {item.system for item in self.systems}
        if (ComparisonSystem.HYBRID in present) == (
            self.hybrid_unavailable_reason is not None
        ):
            raise ValueError(
                "an absent hybrid names why it is absent, and a present one "
                "names nothing"
            )
        if self.positive_count + self.negative_count != self.row_count:
            raise ValueError("class support does not sum to the row count")
        return self

    def for_system(self, system: ComparisonSystem) -> SystemEvaluation | None:
        """Return one system's evaluation, or ``None`` when it was not compared."""
        return next((item for item in self.systems if item.system is system), None)


def comparison_config_fingerprint(*, alert_policy: DecisionAlertPolicy | None) -> str:
    """Return the digest of the comparison contract this build applies.

    Binds the metric definitions and the shared alert policy, so a comparison
    carried out under different windows is a different comparison rather than
    the same one with different alert counts.
    """
    return digest(
        {
            "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
            "metric_definition_fingerprint": metric_definition_fingerprint(),
            "system_order": [str(item) for item in SYSTEM_ORDER],
            "alert_policy": (
                None if alert_policy is None else alert_policy.fingerprint_data()
            ),
        }
    )


def population_fingerprint(anchors: Sequence[str], malicious: Sequence[bool]) -> str:
    """Return the digest of exactly the population every system was scored on.

    Over the anchors and the outcomes together: two populations with the same
    rows and different labels are different populations, and a comparison
    carried out on one must not be mistaken for a comparison on the other.

    The anchors take part in the digest and appear in nothing that is published:
    what leaves this function is a hash.
    """
    if len(anchors) != len(malicious):
        raise DataValidationError(
            f"{len(anchors)} anchor(s) cannot be paired with "
            f"{len(malicious)} outcome(s)"
        )
    return digest(
        {
            "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
            "rows": [
                {"anchor_event_id": anchor, "malicious": outcome}
                for anchor, outcome in zip(anchors, malicious, strict=True)
            ],
        }
    )


def build_comparison(
    *,
    systems: Sequence[SystemEvaluation],
    anchors: Sequence[str],
    malicious: Sequence[bool],
    alert_policy: DecisionAlertPolicy | None,
    hybrid_unavailable_reason: str | None,
    min_positive_rows: int = 1,
    min_benign_rows: int = 1,
) -> SystemComparison:
    """Return the sealed descriptive comparison of every evaluated system."""
    positives = sum(1 for flag in malicious if flag)
    negatives = len(malicious) - positives
    ordered = tuple(
        item for declared in SYSTEM_ORDER for item in systems if item.system is declared
    )
    return SystemComparison.seal(
        systems=ordered,
        hybrid_unavailable_reason=hybrid_unavailable_reason,
        row_count=len(malicious),
        positive_count=positives,
        negative_count=negatives,
        support_status=(
            MetricStatus.MEASURED
            if positives >= min_positive_rows and negatives >= min_benign_rows
            else MetricStatus.INSUFFICIENT_SUPPORT
        ),
        population_fingerprint=population_fingerprint(anchors, malicious),
        comparison_config_fingerprint=comparison_config_fingerprint(
            alert_policy=alert_policy
        ),
    )


def _assert_no_winner_field() -> None:
    """Fail at import if the comparison grows somewhere to declare a winner.

    A descriptive final evaluation that names a winner is one step from a
    deployment decision made on TEST, and the step is short.
    """
    forbidden = {
        "winner",
        "best_system",
        "selected_system",
        "recommended_system",
        "ranking",
        "rank",
    }
    for model in (SystemComparison, SystemEvaluation, AlertLevelSummary):
        offending = sorted(set(model.model_fields) & forbidden)
        if offending:
            raise ValueError(
                f"{model.__name__} declares field(s) {offending}; a TEST "
                f"comparison describes, and never selects"
            )


_assert_no_winner_field()
