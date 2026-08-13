"""Alerts for a system comparison, built without borrowing rule-only concepts.

Phase 4's :class:`~password_attack_detector.detection.alerts.AlertBuilder` groups
*rule* findings. Its grouping key is an attack category and a correlation group,
its alerts name the rules that fired, and its qualifying gates read a rule risk
score and a rule severity. None of that exists for a model decision, and the
tempting shortcut -- pass ML decisions through the Phase 4 builder with plausible
values filled in -- would publish alerts naming rules that never fired, in
categories nobody assigned, under a severity nothing computed.

So the comparison gets its own builder, and the split is deliberate:

* :class:`ComparisonDecision` is the one input shape all three systems produce.
  It carries a flag, and it carries whatever evidence its system actually has --
  an ordinal rule magnitude, a calibrated probability, a rule severity -- as
  **separately typed nullable fields**. A system that has none of a thing leaves
  it absent rather than filling it in.
* :class:`ComparisonAlert` has no ``contributing_rule_ids``, no
  ``correlation_group``, no ``attack_category``, and no signal strengths. Those
  are rule-only concepts and this artifact does not have them to give.

**The temporal semantics are Phase 4's, exactly.** Ordering, the inclusive
grouping window, the inclusive cooldown, the escalation bypass, the rate-limit
window and its accounting, ``first_seen`` never moving, ``last_seen`` only
advancing, superseded alerts still being emitted -- all reproduced, because a
comparison in which one system enjoyed a different cooldown would be measuring
the cooldown. A regression fixture feeds equivalent Phase 4 decisions through
this builder and asserts the grouping, the counts, and the window boundaries
match what Phase 4 produced, and the Phase 4 builder itself is untouched.

**The qualifying gate is the decision, and only the decision.** Phase 4
additionally suppresses below a risk floor and a severity floor; a model has
neither, so applying them to one system and not another would drop rows from one
comparator's alert stream. Every system's qualifying rule here is the same: the
system flagged the row. Callers who want Phase 4's floors apply them upstream,
to every system or to none.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.detection.config import AlertingConfig
from password_attack_detector.detection.enums import (
    SEVERITY_ORDER,
    ScopeKind,
    Severity,
    SuppressionReason,
)
from password_attack_detector.detection.schemas import RiskAssessment
from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.enums import ComparisonSystem

__all__ = [
    "COMPARISON_ALERTING_VERSION",
    "ComparisonAlert",
    "ComparisonAlertingResult",
    "ComparisonAlertingStats",
    "ComparisonDecision",
    "DecisionAlertBuilder",
    "DecisionAlertPolicy",
    "decisions_from_assessments",
]

#: The comparison alerting contract's own version.  Separate from Phase 4's
#: ``ALERTING_VERSION``: the two produce different artifacts and must be able to
#: change independently, and a shared version would make one look like the other.
COMPARISON_ALERTING_VERSION: Final[str] = "1.0.0"

#: Namespace for derived comparison-alert identifiers.  Distinct from the Phase 4
#: alert namespace, so a comparison alert can never collide with a security
#: alert however similar their grouping keys.
_NS_COMPARISON_ALERT: Final = uuid5(
    NAMESPACE_URL, "password-attack-detector/ml/comparison-alert"
)


class ComparisonDecision(BaseModel):
    """One system's decision about one anchor, with only the evidence it has.

    The three comparators fill different subsets of these fields, and that is
    the point: an absent ``ordinal_risk_score`` means the system does not
    produce one, not that it produced zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_event_id: str
    anchor_event_time: datetime
    system: ComparisonSystem
    flagged: bool

    #: Phase 4's 0-100 ordinal severity magnitude. Never combined arithmetically
    #: with a probability, and never rendered as one.
    ordinal_risk_score: float | None = None
    #: Phase 4's severity ladder, where the deciding system has one.
    severity: Severity | None = None
    #: The ML layer's calibrated probability, where a verified calibrator
    #: produced one.
    calibrated_probability: float | None = None
    #: The pseudonymous grouping dimension, when the caller supplied a scope
    #: table. Never rendered into a report.
    scope_kind: ScopeKind = ScopeKind.NONE
    scope_value: str | None = None

    @model_validator(mode="after")
    def check_decision(self) -> Self:
        """The evidence stays on its declared scale, and scope is coherent."""
        if self.anchor_event_time.tzinfo is None:
            raise ValueError("anchor_event_time must be timezone-aware")
        if self.ordinal_risk_score is not None:
            if not math.isfinite(self.ordinal_risk_score):
                raise ValueError("an ordinal risk score must be finite")
            if not 0.0 <= self.ordinal_risk_score <= 100.0:
                raise ValueError("an ordinal risk score lies on the 0-100 scale")
        if self.calibrated_probability is not None and not (
            0.0 <= self.calibrated_probability <= 1.0
        ):
            raise ValueError("a calibrated probability lies in [0, 1]")
        if (self.scope_kind is ScopeKind.NONE) != (self.scope_value is None):
            raise ValueError(
                "a scoped decision names its scope value, and an unscoped one "
                "names none"
            )
        return self

    @property
    def magnitude(self) -> float | None:
        """Return the quantity materiality is judged on, or ``None``.

        The ordinal risk where the system has one, else the calibrated
        probability. The two are never compared against each other -- a given
        system supplies one of them for every one of its rows, so a comparison
        inside one alert stream stays on one scale.
        """
        if self.ordinal_risk_score is not None:
            return self.ordinal_risk_score
        return self.calibrated_probability


@dataclass(frozen=True, slots=True)
class DecisionAlertPolicy:
    """The temporal policy every compared system is held to.

    One policy object, applied to all three streams. Built from the Phase 4
    alerting configuration by :meth:`from_alerting`, so the comparison inherits
    the reviewed windows rather than declaring its own -- and so a fixture can
    prove the two builders agree.
    """

    grouping_window: timedelta
    cooldown: timedelta
    alert_limit_window: timedelta
    max_alerts_per_group_per_window: int
    escalation_bypasses_cooldown: bool

    @classmethod
    def from_alerting(cls, alerting: AlertingConfig) -> DecisionAlertPolicy:
        """Return the policy an :class:`AlertingConfig` declares."""
        return cls(
            grouping_window=alerting.grouping_window,
            cooldown=alerting.cooldown,
            alert_limit_window=alerting.alert_limit_window,
            max_alerts_per_group_per_window=alerting.max_alerts_per_group_per_window,
            escalation_bypasses_cooldown=alerting.escalation_bypasses_cooldown,
        )

    def fingerprint_data(self) -> dict[str, object]:
        """Return the semantic fields identifying this policy."""
        return {
            "comparison_alerting_version": COMPARISON_ALERTING_VERSION,
            "grouping_window_seconds": self.grouping_window.total_seconds(),
            "cooldown_seconds": self.cooldown.total_seconds(),
            "alert_limit_window_seconds": self.alert_limit_window.total_seconds(),
            "max_alerts_per_group_per_window": (self.max_alerts_per_group_per_window),
            "escalation_bypasses_cooldown": self.escalation_bypasses_cooldown,
        }


class ComparisonAlert(BaseModel):
    """One grouped alert from one compared system.

    Deliberately narrower than a Phase 4 ``SecurityAlert``. What is missing is
    missing because this artifact does not have it: no contributing rule
    identifiers, no correlation group, no attack category, no signal strengths.
    Fabricating any of them would make a model's output look like a rule's.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_alerting_version: str = COMPARISON_ALERTING_VERSION
    alert_id: str
    system: ComparisonSystem
    scope_kind: ScopeKind
    scope_value: str | None
    first_seen: datetime
    last_seen: datetime
    contributing_event_count: int = Field(ge=1)
    suppressed_event_count: int = Field(ge=0)

    #: Both magnitudes are nullable and separately typed. They are never added,
    #: averaged together, or rendered as one number.
    peak_ordinal_risk_score: float | None
    mean_ordinal_risk_score: float | None
    peak_calibrated_probability: float | None
    mean_calibrated_probability: float | None
    initial_severity: Severity | None
    current_severity: Severity | None
    escalation_count: int = Field(ge=0)

    @model_validator(mode="after")
    def check_alert(self) -> Self:
        """The window contains its own events, and the means stay under the peaks."""
        if self.last_seen < self.first_seen:
            raise ValueError("an alert's window ends before it begins")
        for peak, mean, name in (
            (
                self.peak_ordinal_risk_score,
                self.mean_ordinal_risk_score,
                "ordinal risk",
            ),
            (
                self.peak_calibrated_probability,
                self.mean_calibrated_probability,
                "calibrated probability",
            ),
        ):
            if (peak is None) != (mean is None):
                raise ValueError(f"the {name} peak and mean are reported together")
            if peak is not None and mean is not None and mean > peak:
                raise ValueError(f"the mean {name} exceeds its peak")
        if (self.initial_severity is None) != (self.current_severity is None):
            raise ValueError("severities are reported together or not at all")
        return self


@dataclass(frozen=True, slots=True)
class ComparisonAlertingStats:
    """Aggregate accounting for one comparison alerting pass.

    Counts only. No grouping key, no scope value, and no anchor identifier: a
    comparison report must stay safe to share.
    """

    system: ComparisonSystem
    decision_count: int = 0
    qualifying_count: int = 0
    alert_count: int = 0
    grouped_decision_count: int = 0
    escalated_count: int = 0
    distinct_group_count: int = 0
    suppressed_by_reason: Mapping[SuppressionReason, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComparisonAlertingResult:
    """Every alert one system produced, plus its accounting."""

    alerts: tuple[ComparisonAlert, ...]
    stats: ComparisonAlertingStats

    @property
    def alert_count(self) -> int:
        """Return how many alerts were emitted."""
        return len(self.alerts)


@dataclass(frozen=True, slots=True)
class _GroupKey:
    """What makes two decisions the same comparison alert.

    The system is part of the key so two comparators' alerts can never be merged
    into one; the scope is the same pseudonymous dimension Phase 4 groups on.
    There is no attack category here, because a model decision does not have one.
    """

    system: ComparisonSystem
    scope_kind: ScopeKind
    scope_value: str | None


@dataclass(slots=True)
class _OpenAlert:
    """Mutable state for one alert while it is still absorbing decisions."""

    key: _GroupKey
    first_seen: datetime
    last_seen: datetime
    contributing_event_count: int
    ordinal_sum: float
    ordinal_count: int
    ordinal_peak: float | None
    probability_sum: float
    probability_count: int
    probability_peak: float | None
    initial_severity: Severity | None
    current_severity: Severity | None
    peak_magnitude: float | None
    escalation_count: int = 0
    suppressed_event_count: int = 0

    def absorb(self, decision: ComparisonDecision) -> bool:
        """Fold *decision* into this alert; return whether severity rose.

        ``first_seen`` never moves and ``last_seen`` only advances, exactly as
        in Phase 4, so the window an alert reports always contains every
        decision that contributed to it.
        """
        self.contributing_event_count += 1
        self.last_seen = max(self.last_seen, decision.anchor_event_time)
        if decision.ordinal_risk_score is not None:
            self.ordinal_sum += decision.ordinal_risk_score
            self.ordinal_count += 1
            self.ordinal_peak = (
                decision.ordinal_risk_score
                if self.ordinal_peak is None
                else max(self.ordinal_peak, decision.ordinal_risk_score)
            )
        if decision.calibrated_probability is not None:
            self.probability_sum += decision.calibrated_probability
            self.probability_count += 1
            self.probability_peak = (
                decision.calibrated_probability
                if self.probability_peak is None
                else max(self.probability_peak, decision.calibrated_probability)
            )
        magnitude = decision.magnitude
        if magnitude is not None:
            self.peak_magnitude = (
                magnitude
                if self.peak_magnitude is None
                else max(self.peak_magnitude, magnitude)
            )
        if decision.severity is not None and (
            self.current_severity is None
            or SEVERITY_ORDER[decision.severity] > SEVERITY_ORDER[self.current_severity]
        ):
            self.current_severity = decision.severity
            self.escalation_count += 1
            return True
        return False

    def finalize(self) -> ComparisonAlert:
        """Build the published alert."""
        ordinal_mean = (
            None
            if self.ordinal_count == 0
            else min(self.ordinal_sum / self.ordinal_count, self.ordinal_peak or 0.0)
        )
        probability_mean = (
            None
            if self.probability_count == 0
            else min(
                self.probability_sum / self.probability_count,
                self.probability_peak or 0.0,
            )
        )
        return ComparisonAlert(
            alert_id=_alert_identifier(self.key, self.first_seen),
            system=self.key.system,
            scope_kind=self.key.scope_kind,
            scope_value=self.key.scope_value,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            contributing_event_count=self.contributing_event_count,
            suppressed_event_count=self.suppressed_event_count,
            peak_ordinal_risk_score=self.ordinal_peak,
            mean_ordinal_risk_score=(
                None if ordinal_mean is None else round(ordinal_mean, 4)
            ),
            peak_calibrated_probability=self.probability_peak,
            mean_calibrated_probability=(
                None if probability_mean is None else round(probability_mean, 9)
            ),
            initial_severity=self.initial_severity,
            current_severity=self.current_severity,
            escalation_count=self.escalation_count,
        )


def _alert_identifier(key: _GroupKey, first_seen: datetime) -> str:
    """Derive the deterministic identifier for one comparison alert.

    Built from the contract version, the grouping key, and the alert's own
    window start -- never from ``uuid4``, the wall clock, a process identifier,
    ``hash()``, or ambient random state. Two runs over the same decisions
    produce the same identifiers on any machine.
    """
    payload = {
        "comparison_alerting_version": COMPARISON_ALERTING_VERSION,
        "system": str(key.system),
        "scope_kind": str(key.scope_kind),
        "scope_value": key.scope_value,
        "first_seen": first_seen.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return str(uuid5(_NS_COMPARISON_ALERT, canonical))


class DecisionAlertBuilder:
    """Groups comparison decisions into alerts under one shared policy.

    One ordered pass with dictionary indexes, mirroring the Phase 4 builder's
    shape as well as its semantics: cost is linear in the number of decisions
    and no step compares a decision against the set of open alerts.
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: DecisionAlertPolicy) -> None:
        self._policy = policy

    def build(
        self, decisions: Sequence[ComparisonDecision]
    ) -> ComparisonAlertingResult:
        """Group *decisions* into alerts.

        Raises:
            DataValidationError: when the decisions come from more than one
                system. One pass builds one system's stream; mixing two would
                make the rate limit and the cooldown shared between them.
        """
        systems = {decision.system for decision in decisions}
        if len(systems) > 1:
            raise DataValidationError(
                f"a comparison alerting pass covers one system; {len(systems)} "
                f"were supplied, and their alerts would share a cooldown"
            )
        return _Pass(self._policy).run(decisions)


class _Pass:
    """One ordered grouping pass over a single system's decisions."""

    __slots__ = (
        "_closed",
        "_early",
        "_escalated",
        "_grouped",
        "_open",
        "_policy",
        "_qualifying",
        "_suppressed",
        "_window",
    )

    def __init__(self, policy: DecisionAlertPolicy) -> None:
        self._policy = policy
        self._open: dict[_GroupKey, _OpenAlert] = {}
        self._closed: dict[_GroupKey, _OpenAlert] = {}
        self._early: list[_OpenAlert] = []
        self._window: dict[_GroupKey, tuple[datetime, int]] = {}
        self._suppressed: dict[SuppressionReason, int] = {}
        self._qualifying = 0
        self._grouped = 0
        self._escalated = 0

    def run(self, decisions: Sequence[ComparisonDecision]) -> ComparisonAlertingResult:
        """Group every decision, then close whatever is still open."""
        # Sorted here rather than trusted from the caller, exactly as Phase 4
        # does, so the order the decisions arrived in cannot reach the output.
        ordered = sorted(
            decisions,
            key=lambda item: (item.anchor_event_time, item.anchor_event_id),
        )
        for decision in ordered:
            self._consume(decision)

        emitted = [alert.finalize() for alert in self._early]
        emitted.extend(alert.finalize() for alert in self._closed.values())
        emitted.extend(alert.finalize() for alert in self._open.values())
        alerts = tuple(
            sorted(
                emitted,
                key=lambda alert: (
                    alert.first_seen,
                    str(alert.scope_kind),
                    alert.scope_value or "",
                    alert.alert_id,
                ),
            )
        )
        system = ordered[0].system if ordered else ComparisonSystem.ML_ONLY
        return ComparisonAlertingResult(
            alerts=alerts,
            stats=ComparisonAlertingStats(
                system=system,
                decision_count=len(decisions),
                qualifying_count=self._qualifying,
                alert_count=len(alerts),
                grouped_decision_count=self._grouped,
                escalated_count=self._escalated,
                distinct_group_count=len(self._window),
                suppressed_by_reason=dict(self._suppressed),
            ),
        )

    def _consume(self, decision: ComparisonDecision) -> None:
        """Group or suppress one decision.

        The one qualifying gate is the decision itself. Phase 4's risk and
        severity floors are deliberately not applied here: a model has neither,
        and applying them to one comparator only would drop rows from one
        system's stream and not another's.
        """
        if not decision.flagged:
            return
        self._qualifying += 1
        key = _GroupKey(decision.system, decision.scope_kind, decision.scope_value)

        open_alert = self._open.get(key)
        if open_alert is not None:
            if (
                decision.anchor_event_time - open_alert.last_seen
                <= self._policy.grouping_window
            ):
                # Inside the window, inclusive at the boundary: absorb.
                if open_alert.absorb(decision):
                    self._escalated += 1
                self._grouped += 1
                return
            self._close(key, open_alert)

        self._open_or_suppress(key, decision)

    def _open_or_suppress(self, key: _GroupKey, decision: ComparisonDecision) -> None:
        """Open a new alert for *key*, unless cooldown or the limit forbids it."""
        closed = self._closed.get(key)
        escalating = False
        if closed is not None:
            elapsed = decision.anchor_event_time - closed.last_seen
            if elapsed <= self._policy.cooldown:
                # Cooldown is inclusive at the boundary, as in Phase 4.
                escalating = self._is_material(decision, closed)
                if not (escalating and self._policy.escalation_bypasses_cooldown):
                    closed.suppressed_event_count += 1
                    self._suppress(SuppressionReason.COOLDOWN)
                    return

        if not self._claim_window_slot(key, decision.anchor_event_time):
            if closed is not None:
                closed.suppressed_event_count += 1
            self._suppress(SuppressionReason.RATE_LIMIT)
            return

        if escalating:
            self._escalated += 1
        self._open[key] = _OpenAlert(
            key=key,
            first_seen=decision.anchor_event_time,
            last_seen=decision.anchor_event_time,
            contributing_event_count=1,
            ordinal_sum=decision.ordinal_risk_score or 0.0,
            ordinal_count=1 if decision.ordinal_risk_score is not None else 0,
            ordinal_peak=decision.ordinal_risk_score,
            probability_sum=decision.calibrated_probability or 0.0,
            probability_count=1 if decision.calibrated_probability is not None else 0,
            probability_peak=decision.calibrated_probability,
            initial_severity=decision.severity,
            current_severity=decision.severity,
            peak_magnitude=decision.magnitude,
        )
        self._grouped += 1

    def _is_material(self, decision: ComparisonDecision, closed: _OpenAlert) -> bool:
        """Return whether *decision* carries evidence the closed alert did not.

        Phase 4's rule, generalised to a system that may have no severity:
        a higher severity where both have one, or a magnitude above anything the
        previous alert saw. A system with neither is never material, which is
        the conservative reading -- it suppresses a repeat rather than emitting
        one on the strength of a quantity nobody measured.
        """
        if (
            decision.severity is not None
            and closed.current_severity is not None
            and SEVERITY_ORDER[decision.severity]
            > SEVERITY_ORDER[closed.current_severity]
        ):
            return True
        magnitude = decision.magnitude
        return (
            magnitude is not None
            and closed.peak_magnitude is not None
            and magnitude > closed.peak_magnitude
        )

    def _claim_window_slot(self, key: _GroupKey, when: datetime) -> bool:
        """Return whether *key* may open another alert in its current window."""
        entry = self._window.get(key)
        if entry is None or when - entry[0] > self._policy.alert_limit_window:
            self._window[key] = (when, 1)
            return True
        start, count = entry
        if count >= self._policy.max_alerts_per_group_per_window:
            return False
        self._window[key] = (start, count + 1)
        return True

    def _close(self, key: _GroupKey, alert: _OpenAlert) -> None:
        """Move an open alert into the closed registry, starting its cooldown."""
        previous = self._closed.get(key)
        self._closed[key] = alert
        if previous is not None:
            # A key can close more than once in a run. The earlier alert is
            # already final and is emitted as its own row rather than merged.
            self._early.append(previous)
        del self._open[key]

    def _suppress(self, reason: SuppressionReason) -> None:
        """Record one suppressed decision against a reason."""
        self._suppressed[reason] = self._suppressed.get(reason, 0) + 1


def decisions_from_assessments(
    assessments: Iterable[RiskAssessment],
    *,
    system: ComparisonSystem = ComparisonSystem.RULE_ONLY,
    scope: Mapping[str, tuple[ScopeKind, str]] | None = None,
) -> tuple[ComparisonDecision, ...]:
    """Convert Phase 4 risk assessments into comparison decisions.

    Used by the rule-only comparator and by the equivalence fixture. Nothing is
    invented: the flag is "at least one rule fired", the ordinal score and the
    severity come straight off the assessment, and no probability is attached
    because the rule engine produces none.
    """
    rows: list[ComparisonDecision] = []
    for assessment in assessments:
        anchor = assessment.anchor_event_id
        located = None if scope is None else scope.get(anchor)
        kind, value = located if located is not None else (ScopeKind.NONE, None)
        rows.append(
            ComparisonDecision(
                anchor_event_id=anchor,
                anchor_event_time=assessment.anchor_event_time,
                system=system,
                flagged=assessment.fired_rule_count > 0,
                ordinal_risk_score=float(assessment.risk_score),
                severity=assessment.severity,
                scope_kind=kind,
                scope_value=value,
            )
        )
    return tuple(rows)


def _assert_no_rule_only_field() -> None:
    """Fail at import if a comparison alert grows a rule-only concept."""
    forbidden = {
        "contributing_rule_ids",
        "correlation_group",
        "attack_category",
        "signal_strengths",
        "fired_rule_ids",
        "top_evidence",
    }
    for model in (ComparisonAlert, ComparisonDecision):
        offending = sorted(set(model.model_fields) & forbidden)
        if offending:
            raise ValueError(
                f"{model.__name__} declares rule-only field(s) {offending}; a "
                f"model decision has none of them to report"
            )


_assert_no_rule_only_field()
