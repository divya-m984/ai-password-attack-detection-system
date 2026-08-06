"""Correlation-aware event risk scoring.

The scorer turns the rules that fired on one anchor into a single bounded
magnitude.  It reads nothing else: no feature snapshot, no label, no split, no
campaign, no entity scope, no canonical event, no model output.  Its whole
input is a set of fired detections, and this module imports no reader for
anything else, so the boundary is structural rather than a convention.

``risk_score`` is an **ordinal magnitude on a 0-100 scale, not a probability**.
Nothing here estimates how likely an attack is; the number orders findings by
how much configured evidence accumulated behind them.  No field, report label,
or docstring in this module calls it a probability, a likelihood, or a
confidence, and a test enforces that.

The procedure, in full::

    contribution_r = family_weight[family(r)] * signal_strength(r)   in (0, 1]
    c_g            = reduce(contribution_r for r in group g)         in (0, 1]
    combined       = 1 - product over sorted g of (1 - c_g)          in [0, 1)
    risk_score     = max(round(100 * combined), min_fired_risk_score)

Three properties follow from the shape and are each pinned by a test.  The
group reducer is idempotent under ``max``, so three rules describing one
behaviour cannot out-score the strongest of them -- that is what the
correlation groups are *for*.  Noisy-OR across groups is monotone
non-decreasing in every argument, so adding an unrelated signal can never lower
a score.  And group iteration is sorted before the product runs, so float
multiplication happens in a fixed order and two runs agree bit for bit.

With no rule fired the score is exactly ``0.0`` -- a module constant, not a
configurable floor -- so a zero can always be read as "nothing fired".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import (
    SEVERITY_ORDER,
    AttackCategory,
    CorrelationGroup,
    RuleFamily,
    RuleStatus,
    Severity,
)
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    FiredDetection,
    RiskAssessment,
    RuleEvaluationResult,
)

__all__ = [
    "SCORING_VERSION",
    "ZERO_RISK_SCORE",
    "RiskScorer",
    "RuleContribution",
    "ScoringResult",
    "ScoringStats",
]

#: Version of the scoring contract.  Recorded on every assessment.
SCORING_VERSION: Final = "1.0.0"

#: The score of an assessment with no fired rule.  Deliberately a constant and
#: not a configurable value: downstream validation, alerting, and evaluation
#: all rely on "zero means nothing fired" being exactly true.
ZERO_RISK_SCORE: Final[float] = 0.0

#: Anything the scorer can consume.  A ``RuleEvaluationResult`` carries the
#: full outcome set including non-firings, so passing the engine's diagnostic
#: stream lets the scorer derive the anchor inventory and the insufficient-data
#: tallies in the same pass.
ScorableResult = FiredDetection | RuleEvaluationResult


@dataclass(frozen=True, slots=True)
class RuleContribution:
    """One fired rule's weighted contribution to an anchor's risk.

    Kept as an explicit record rather than a bare float so the primary-category
    choice, the top-evidence selection, and the group reduction all read the
    same values, and so a test can inspect the arithmetic rather than infer it.
    """

    rule_id: str
    rule_family: RuleFamily
    attack_category: AttackCategory
    correlation_group: CorrelationGroup
    severity: Severity
    signal_strength: float
    weight: float
    contribution: float
    evidence: tuple[EvidenceItem, ...] = ()

    @property
    def rank_key(self) -> tuple[float, int, float, str]:
        """Return the deterministic ordering key, strongest first.

        Contribution decides; severity breaks a contribution tie; signal
        strength breaks a severity tie; the rule identifier breaks everything
        else.  The identifier is last and is always distinct within an anchor,
        so the order is total -- there is no case where two runs could disagree.
        """
        return (
            -self.contribution,
            -SEVERITY_ORDER[self.severity],
            -self.signal_strength,
            self.rule_id,
        )


@dataclass(frozen=True, slots=True)
class ScoringStats:
    """Structural counters for one scoring pass.

    Operation counts, never timings.  ``group_reductions`` is what proves the
    correlation reducer actually ran: a regression that scored every rule
    independently would leave it equal to the contribution count.
    """

    assessment_count: int = 0
    scored_detection_count: int = 0
    duplicate_detection_count: int = 0
    group_reductions: int = 0


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """Every assessment one scoring pass produced, plus aggregate counts."""

    assessments: tuple[RiskAssessment, ...] = ()
    scoring_version: str = SCORING_VERSION
    configuration_fingerprint: str = ""
    severity_counts: Mapping[str, int] = field(default_factory=dict)
    zero_score_count: int = 0
    stats: ScoringStats = ScoringStats()

    @property
    def scored_count(self) -> int:
        """Return how many anchors were assessed."""
        return len(self.assessments)


@dataclass(frozen=True, slots=True)
class _AnchorInput:
    """Everything the scorer collected for one anchor before scoring it."""

    anchor_event_time: datetime
    detections: dict[str, FiredDetection]
    insufficient_data_count: int = 0
    duplicate_count: int = 0


class RiskScorer:
    """Combines fired rules into a bounded, correlation-aware risk magnitude.

    Note the constructor and both scoring methods: there is no parameter for
    labels, splits, campaigns, entity scope, or events anywhere in the
    interface, and none can be added without changing a signature a test pins.
    """

    __slots__ = ("_config",)

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config

    # -- single anchor ------------------------------------------------------

    def score_anchor(
        self,
        anchor_event_id: str,
        anchor_event_time: datetime,
        detections: Sequence[FiredDetection],
        *,
        insufficient_data_count: int = 0,
    ) -> RiskAssessment:
        """Score one anchor from the rules that fired on it.

        A repeated ``rule_id`` is collapsed to its strongest occurrence rather
        than rejected: a caller concatenating two batches should get a stable
        answer, not an exception, and the published assessment must carry each
        rule once.

        Raises:
            DetectionConfigurationError: if a fired rule's family carries no
                configured weight.
        """
        collapsed = self._collapse(detections)
        return self._assess(
            anchor_event_id,
            _AnchorInput(
                anchor_event_time=anchor_event_time,
                detections=collapsed,
                insufficient_data_count=insufficient_data_count,
            ),
        )

    # -- batch --------------------------------------------------------------

    def score(
        self,
        results: Iterable[ScorableResult],
        *,
        evaluated_anchors: Mapping[str, datetime] | None = None,
    ) -> ScoringResult:
        """Score every anchor present in *results*.

        Accepts fired detections, full rule-evaluation results, or a mix.  When
        given evaluation results the scorer also learns which anchors were
        evaluated but produced nothing, and how much history was unavailable --
        so the benign denominator downstream evaluation needs comes from the
        same pass rather than from a second scan.

        *evaluated_anchors* extends the inventory with anchors that produced no
        result at all.  Supplying it is what makes the output "one row per
        evaluated anchor" rather than "one row per anchor something happened
        on"; without it, an anchor no rule reached is simply absent.

        The output is sorted by ``(anchor_event_time, anchor_event_id)``, so
        the order the results arrived in cannot survive into the result.
        """
        collected = self._collect(results, evaluated_anchors)

        assessments = tuple(
            self._assess(anchor_id, collected[anchor_id])
            for anchor_id in sorted(
                collected,
                key=lambda anchor: (collected[anchor].anchor_event_time, anchor),
            )
        )

        severity_counts = {str(severity): 0 for severity in Severity}
        for assessment in assessments:
            severity_counts[str(assessment.severity)] += 1

        scored = sum(len(entry.detections) for entry in collected.values())
        duplicates = sum(entry.duplicate_count for entry in collected.values())
        reductions = sum(
            len({d.correlation_group for d in entry.detections.values()})
            for entry in collected.values()
        )

        return ScoringResult(
            assessments=assessments,
            scoring_version=SCORING_VERSION,
            configuration_fingerprint=self._config.fingerprint(),
            severity_counts=severity_counts,
            zero_score_count=sum(
                1 for item in assessments if item.fired_rule_count == 0
            ),
            stats=ScoringStats(
                assessment_count=len(assessments),
                scored_detection_count=scored,
                duplicate_detection_count=duplicates,
                group_reductions=reductions,
            ),
        )

    # -- collection ---------------------------------------------------------

    def _collect(
        self,
        results: Iterable[ScorableResult],
        evaluated_anchors: Mapping[str, datetime] | None,
    ) -> dict[str, _AnchorInput]:
        """Group the inputs by anchor, collapsing repeats and counting gaps."""
        by_anchor: dict[str, dict[str, FiredDetection]] = {}
        times: dict[str, datetime] = {}
        insufficient: dict[str, int] = {}
        duplicates: dict[str, int] = {}

        for item in results:
            anchor = item.anchor_event_id
            times.setdefault(anchor, item.anchor_event_time)
            by_anchor.setdefault(anchor, {})
            insufficient.setdefault(anchor, 0)
            duplicates.setdefault(anchor, 0)

            if isinstance(item, RuleEvaluationResult):
                if item.status is RuleStatus.INSUFFICIENT_DATA:
                    insufficient[anchor] += 1
                if item.status is not RuleStatus.FIRED:
                    continue
                detection = FiredDetection.from_result(item)
            else:
                detection = item

            existing = by_anchor[anchor].get(detection.rule_id)
            if existing is not None:
                duplicates[anchor] += 1
                if detection.signal_strength <= existing.signal_strength:
                    continue
            by_anchor[anchor][detection.rule_id] = detection

        if evaluated_anchors is not None:
            for anchor, when in evaluated_anchors.items():
                times.setdefault(anchor, when)
                by_anchor.setdefault(anchor, {})
                insufficient.setdefault(anchor, 0)
                duplicates.setdefault(anchor, 0)

        return {
            anchor: _AnchorInput(
                anchor_event_time=times[anchor],
                detections=detections,
                insufficient_data_count=insufficient[anchor],
                duplicate_count=duplicates[anchor],
            )
            for anchor, detections in by_anchor.items()
        }

    @staticmethod
    def _collapse(
        detections: Sequence[FiredDetection],
    ) -> dict[str, FiredDetection]:
        """Index *detections* by rule, keeping the strongest of any repeat."""
        collapsed: dict[str, FiredDetection] = {}
        for detection in detections:
            existing = collapsed.get(detection.rule_id)
            if existing is None or detection.signal_strength > existing.signal_strength:
                collapsed[detection.rule_id] = detection
        return collapsed

    # -- scoring ------------------------------------------------------------

    def _assess(self, anchor_event_id: str, entry: _AnchorInput) -> RiskAssessment:
        """Build the assessment for one anchor."""
        if not entry.detections:
            # The only path to a zero score, which is what lets a reader treat
            # zero as "nothing fired" rather than "scored very low".
            return RiskAssessment(
                anchor_event_id=anchor_event_id,
                anchor_event_time=entry.anchor_event_time,
                risk_score=ZERO_RISK_SCORE,
                severity=self._config.severity_thresholds.severity_for(ZERO_RISK_SCORE),
                insufficient_data_count=entry.insufficient_data_count,
                scoring_version=SCORING_VERSION,
                configuration_fingerprint=self._config.fingerprint(),
            )

        contributions = self.contributions(entry.detections.values())
        risk_score = self.combine(contributions)
        ranked = sorted(contributions, key=lambda item: item.rank_key)

        return RiskAssessment(
            anchor_event_id=anchor_event_id,
            anchor_event_time=entry.anchor_event_time,
            risk_score=risk_score,
            severity=self._config.severity_thresholds.severity_for(risk_score),
            primary_attack_category=ranked[0].attack_category,
            contributing_categories=tuple(
                sorted({item.attack_category for item in contributions})
            ),
            fired_rule_count=len(contributions),
            fired_rule_ids=tuple(sorted(item.rule_id for item in contributions)),
            top_evidence=self._top_evidence(ranked),
            insufficient_data_count=entry.insufficient_data_count,
            scoring_version=SCORING_VERSION,
            configuration_fingerprint=self._config.fingerprint(),
        )

    def contributions(
        self, detections: Iterable[FiredDetection]
    ) -> tuple[RuleContribution, ...]:
        """Return the weighted contribution of every fired rule.

        Raises:
            DetectionConfigurationError: if a rule's family carries no
                configured weight.  That cannot happen for an enabled rule --
                coverage is validated at load -- so it signals a detection
                produced by a rule this configuration disabled.
        """
        built = [
            RuleContribution(
                rule_id=detection.rule_id,
                rule_family=detection.rule_family,
                attack_category=detection.attack_category,
                correlation_group=detection.correlation_group,
                severity=detection.severity,
                signal_strength=detection.signal_strength,
                weight=self._config.weight_for(detection.rule_family),
                contribution=(
                    self._config.weight_for(detection.rule_family)
                    * detection.signal_strength
                ),
                evidence=detection.evidence,
            )
            for detection in detections
        ]
        # Sorted here rather than at every use site, so every consumer of this
        # sequence sees one canonical order.
        return tuple(sorted(built, key=lambda item: item.rank_key))

    def combine(self, contributions: Sequence[RuleContribution]) -> float:
        """Reduce within correlation groups, then combine across them.

        Returns a risk score in ``[0, 100]``.  The result is independent of the
        order *contributions* arrives in: groups are reduced by a commutative
        operator and iterated in sorted order before the product runs.
        """
        if not contributions:
            return ZERO_RISK_SCORE

        grouped: dict[CorrelationGroup, list[float]] = {}
        for item in contributions:
            grouped.setdefault(item.correlation_group, []).append(item.contribution)

        reducer = self._config.scoring.correlation_reducer
        residual = 1.0
        for group in sorted(grouped, key=str):
            values = grouped[group]
            reduced = max(values) if reducer == "max" else _bounded_sum(sorted(values))
            residual *= 1.0 - reduced

        combined = _clamp_unit(1.0 - residual)
        scaled = round(100.0 * combined, self._config.scoring.round_decimals)
        floored = max(scaled, self._config.scoring.min_fired_risk_score)
        return min(floored, 100.0)

    def _top_evidence(
        self, ranked: Sequence[RuleContribution]
    ) -> tuple[EvidenceItem, ...]:
        """Return the strongest evidence items, in contribution order.

        Taken from the ranked contributions rather than pooled and re-sorted,
        because an evidence item carries no magnitude of its own -- its weight
        is entirely the weight of the rule that emitted it.
        """
        limit = self._config.scoring.top_evidence_count
        collected: list[EvidenceItem] = []
        for item in ranked:
            for evidence in item.evidence:
                if len(collected) >= limit:
                    return tuple(collected)
                collected.append(evidence)
        return tuple(collected)


def _bounded_sum(values: Sequence[float]) -> float:
    """Combine *values* within a group so breadth counts but stays bounded.

    The alternative to ``max``: three distinct rules in one group score above
    the strongest of them, but never above 1.0.  Values are multiplied in the
    order given, so the caller sorts first and the float arithmetic is fixed.
    """
    residual = 1.0
    for value in values:
        residual *= 1.0 - value
    return _clamp_unit(1.0 - residual)


def _clamp_unit(value: float) -> float:
    """Confine *value* to ``[0, 1]``, refusing a non-finite input.

    Raises:
        ValueError: if *value* is ``NaN`` or infinite.  Every input to the
            scorer is bounded by validation, so a non-finite value here is a
            defect rather than something to silently pin to a bound.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError("risk combination produced a NaN or infinite value")
    return max(0.0, min(1.0, value))
