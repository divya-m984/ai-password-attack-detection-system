"""Alert construction: grouping, deduplication, cooldown, and escalation.

An alert is a claim that something is worth an analyst's attention.  Emitting
one per fired detection would make that claim thousands of times about the same
behaviour, so this module groups detections and suppresses repeats.  The whole
design tension is there: **suppression must reduce duplicate noise without ever
hiding materially new risk**, and every choice below is an answer to that.

Three things make suppression safe to trust.  A suppressed detection is always
counted -- :class:`AlertingStats` carries an accounting identity that a schema
validator enforces, so an alert cannot vanish without leaving a tally behind.
A higher severity bypasses cooldown rather than being swallowed by it.  And the
grouping key includes the attack category and correlation group, so an
unrelated finding is never suppressed by an open alert about something else.

**Entity scope is confined to this module.**  It is the one input in the whole
detection layer that carries pseudonymous identifiers.  ``DetectionEngine`` and
``RiskScorer`` take no scope argument and import nothing from here, so the
restriction is a type error rather than a convention.  A scope value reaches
exactly one field of one artifact -- ``SecurityAlert.scope_value`` -- and never
an evidence item, a report, a CLI summary, a manifest, a statistic, or an error
message.  Every failure raised here names counts and column names only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final
from uuid import NAMESPACE_URL, uuid5

from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import (
    SEVERITY_ORDER,
    AlertGroupingMode,
    AttackCategory,
    CorrelationGroup,
    ScopeKind,
    Severity,
    SuppressionReason,
    severity_at_least,
)
from password_attack_detector.detection.schemas import (
    AlertingStats,
    EntityScopeRecord,
    FiredDetection,
    RiskAssessment,
    SecurityAlert,
)
from password_attack_detector.exceptions import (
    DataValidationError,
    DetectionConfigurationError,
)

__all__ = [
    "ALERTING_VERSION",
    "AlertBuilder",
    "AlertingOperationStats",
    "AlertingResult",
    "EntityScopeTable",
    "build_entity_scope_table",
]

#: Version of the alerting contract.  Participates in every alert identifier,
#: so a change to the grouping semantics yields a new identifier set rather
#: than silently reusing the previous run's.
ALERTING_VERSION: Final = "1.0.0"

#: Namespace for deterministic alert identifiers.
_NS_ALERT: Final = uuid5(NAMESPACE_URL, "password-attack-detector/detection/alert")


# ---------------------------------------------------------------------------
# Entity scope
# ---------------------------------------------------------------------------


class EntityScopeTable:
    """An optional, validated one-to-one anchor-to-scope mapping.

    Built through :meth:`build` so a duplicate anchor is a hard failure rather
    than a last-wins merge.  A silent merge would attribute one event's alerts
    to whichever row happened to be read second, which is exactly the kind of
    quiet mis-grouping an analyst has no way to notice.

    ``__repr__`` is redacted: every value in here is a pseudonym, and a repr
    reaches logs and tracebacks.
    """

    __slots__ = ("_records",)

    def __init__(self, records: Mapping[str, EntityScopeRecord]) -> None:
        self._records = dict(records)

    @classmethod
    def build(cls, records: Iterable[EntityScopeRecord]) -> EntityScopeTable:
        """Build a table, rejecting a repeated anchor.

        Raises:
            DataValidationError: if an anchor appears more than once.  The
                message reports the count only; no anchor identifier and no
                scope value appears in it.
        """
        indexed: dict[str, EntityScopeRecord] = {}
        duplicates = 0
        for record in records:
            if record.anchor_event_id in indexed:
                duplicates += 1
            indexed[record.anchor_event_id] = record
        if duplicates:
            raise DataValidationError(
                f"The entity-scope table contains {duplicates} duplicate "
                f"anchor key(s); it must be keyed one-to-one by anchor"
            )
        return cls(indexed)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def anchor_count(self) -> int:
        """Return how many anchors the table covers."""
        return len(self._records)

    def scope_for(self, anchor_event_id: str, kind: ScopeKind) -> str | None:
        """Return the scope value for *anchor_event_id* on dimension *kind*.

        ``None`` when the anchor is absent, the dimension is
        :attr:`ScopeKind.NONE`, or that dimension was not recorded -- the three
        cases the caller handles identically by degrading to category grouping.
        """
        record = self._records.get(anchor_event_id)
        if record is None:
            return None
        match kind:
            case ScopeKind.USER:
                return record.user_scope
            case ScopeKind.SOURCE:
                return record.source_scope
            case ScopeKind.NONE:
                return None

    def validate_against(self, anchor_event_ids: Iterable[str]) -> None:
        """Assert the table and the assessment anchors match one to one.

        The check is symmetric on purpose.  A scope row with no assessment
        means the table was built from a different run; an assessment with no
        scope row means the table is incomplete.  Either way the grouping that
        follows would be quietly wrong, so both abort the run.

        Raises:
            DataValidationError: on any mismatch, reporting counts only.
        """
        assessment_anchors = set(anchor_event_ids)
        table_anchors = set(self._records)
        missing = len(assessment_anchors - table_anchors)
        extra = len(table_anchors - assessment_anchors)
        if missing or extra:
            raise DataValidationError(
                f"The entity-scope table does not match the risk assessments: "
                f"{missing} assessed anchor(s) have no scope row and {extra} "
                f"scope row(s) have no assessment"
            )

    def __repr__(self) -> str:
        """Return a redacted representation naming only the row count."""
        return f"EntityScopeTable(anchor_count={len(self._records)})"

    __str__ = __repr__


def build_entity_scope_table(
    records: Iterable[EntityScopeRecord],
) -> EntityScopeTable:
    """Build a validated :class:`EntityScopeTable` from *records*."""
    return EntityScopeTable.build(records)


# ---------------------------------------------------------------------------
# Grouping state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _GroupKey:
    """What makes two detections the same alert.

    The category and correlation group are part of the key so a different kind
    of finding is never absorbed into, or suppressed by, an open alert about
    something else.
    """

    attack_category: AttackCategory
    correlation_group: CorrelationGroup
    scope_kind: ScopeKind
    scope_value: str | None


@dataclass(slots=True)
class _OpenAlert:
    """Mutable state for one alert while it is still absorbing detections.

    State is O(1) per alert: a running risk sum rather than a list of scores, a
    set of rule identifiers bounded by the registry size.  Nothing here grows
    with the number of events the alert covers.
    """

    key: _GroupKey
    grouping_mode: AlertGroupingMode
    first_seen: datetime
    last_seen: datetime
    contributing_event_count: int
    rule_ids: set[str]
    risk_sum: float
    peak_risk_score: float
    initial_severity: Severity
    current_severity: Severity
    escalation_count: int = 0
    suppressed_event_count: int = 0

    def absorb(self, assessment: RiskAssessment, rule_ids: Iterable[str]) -> bool:
        """Fold *assessment* into this alert; return whether severity rose.

        ``first_seen`` never moves and ``last_seen`` only advances, so the
        window an alert reports always contains every event that contributed
        to it.
        """
        self.contributing_event_count += 1
        self.rule_ids.update(rule_ids)
        self.risk_sum += assessment.risk_score
        self.peak_risk_score = max(self.peak_risk_score, assessment.risk_score)
        self.last_seen = max(self.last_seen, assessment.anchor_event_time)

        if SEVERITY_ORDER[assessment.severity] > SEVERITY_ORDER[self.current_severity]:
            self.current_severity = assessment.severity
            self.escalation_count += 1
            return True
        return False

    def finalize(self) -> SecurityAlert:
        """Build the published alert.

        ``aggregate_risk_score`` is the mean contributing score and
        ``peak_risk_score`` the maximum, so the pair says "how bad typically"
        and "how bad at worst".  The mean is bounded above by the maximum,
        which is the invariant :class:`SecurityAlert` enforces.
        """
        aggregate = min(
            self.risk_sum / self.contributing_event_count, self.peak_risk_score
        )
        return SecurityAlert(
            alert_id=_alert_identifier(self.key, self.first_seen),
            attack_category=self.key.attack_category,
            correlation_group=self.key.correlation_group,
            grouping_mode=self.grouping_mode,
            scope_kind=self.key.scope_kind,
            scope_value=self.key.scope_value,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            contributing_event_count=self.contributing_event_count,
            contributing_rule_ids=tuple(sorted(self.rule_ids)),
            aggregate_risk_score=round(aggregate, 4),
            peak_risk_score=self.peak_risk_score,
            initial_severity=self.initial_severity,
            current_severity=self.current_severity,
            escalation_count=self.escalation_count,
            suppressed_event_count=self.suppressed_event_count,
        )


def _alert_identifier(key: _GroupKey, first_seen: datetime) -> str:
    """Derive the deterministic identifier for one alert.

    Built from the alerting contract version, the grouping key, and the
    alert's own time-bucket start.  Two runs over the same data therefore
    produce the same identifiers on any machine, on any day.

    Deliberately *excludes* the contributing rule set: an alert that later
    absorbs one more rule is the same alert, and its identity should not change
    underneath a reader who already recorded it.

    Never derived from ``uuid4``, wall-clock time, a process identifier,
    ``hash()``, machine identity, or ambient random state.
    """
    payload = {
        "alerting_version": ALERTING_VERSION,
        "attack_category": str(key.attack_category),
        "correlation_group": str(key.correlation_group),
        "scope_kind": str(key.scope_kind),
        "scope_value": key.scope_value,
        "first_seen": first_seen.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return str(uuid5(_NS_ALERT, canonical))


@dataclass(frozen=True, slots=True)
class AlertingOperationStats:
    """Structural counters for one alert-construction pass.

    Operation counts, never timings.  ``key_lookups`` growing in step with
    ``assessments_scanned`` is what proves grouping stayed a single indexed
    pass rather than a scan of the open alerts per assessment.
    """

    assessments_scanned: int = 0
    key_lookups: int = 0
    alerts_opened: int = 0
    alerts_closed: int = 0
    distinct_group_count: int = 0
    max_open_groups: int = 0


@dataclass(frozen=True, slots=True)
class AlertingResult:
    """Every alert one pass produced, plus its accounting."""

    alerts: tuple[SecurityAlert, ...] = ()
    stats: AlertingStats = field(
        default_factory=lambda: AlertingStats(
            grouping_mode=AlertGroupingMode.CATEGORY_SCOPED
        )
    )
    operations: AlertingOperationStats = AlertingOperationStats()

    @property
    def alert_count(self) -> int:
        """Return how many alerts were emitted."""
        return len(self.alerts)


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class AlertBuilder:
    """Groups risk assessments into deduplicated, suppressed alerts.

    One ordered pass with dictionary indexes: cost is linear in the number of
    assessments, and no step compares an assessment against the set of open
    alerts.
    """

    __slots__ = ("_config",)

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config

    def build(
        self,
        assessments: Sequence[RiskAssessment],
        *,
        detections: Sequence[FiredDetection] | None = None,
        entity_scope: EntityScopeTable | None = None,
    ) -> AlertingResult:
        """Group *assessments* into alerts.

        *detections* is optional and supplies contributing-rule metadata; when
        given, it is cross-checked against each assessment's own fired-rule set
        so a caller pairing mismatched inputs fails here rather than publishing
        an alert naming rules that never fired.

        *entity_scope* is the only place a pseudonym enters the detection
        layer.  When absent, grouping falls back to category, correlation
        group, and the configured time window.

        Raises:
            DataValidationError: if the scope table is inconsistent with the
                assessments, or -- under ``strict_scope`` -- if a required
                scope dimension is missing.  Messages carry counts only.
            DetectionConfigurationError: if a correlation group declares no
                scope dimension.
        """
        rule_ids_by_anchor = _index_rule_ids(detections)
        if rule_ids_by_anchor is not None:
            _check_rule_agreement(assessments, rule_ids_by_anchor)

        if entity_scope is not None:
            entity_scope.validate_against(
                assessment.anchor_event_id for assessment in assessments
            )

        run_mode = (
            AlertGroupingMode.CATEGORY_SCOPED
            if entity_scope is None
            else AlertGroupingMode.ENTITY_SCOPED
        )
        return _Pass(self._config, entity_scope, run_mode).run(assessments)


def _index_rule_ids(
    detections: Sequence[FiredDetection] | None,
) -> dict[str, set[str]] | None:
    """Index fired rule identifiers by anchor, or ``None`` when not supplied."""
    if detections is None:
        return None
    indexed: dict[str, set[str]] = {}
    for detection in detections:
        indexed.setdefault(detection.anchor_event_id, set()).add(detection.rule_id)
    return indexed


def _check_rule_agreement(
    assessments: Sequence[RiskAssessment], rule_ids_by_anchor: Mapping[str, set[str]]
) -> None:
    """Assert each assessment names exactly the rules its detections record.

    Raises:
        DataValidationError: on any disagreement, reporting the count only.
    """
    mismatched = sum(
        1
        for assessment in assessments
        if rule_ids_by_anchor.get(assessment.anchor_event_id, set())
        != set(assessment.fired_rule_ids)
    )
    if mismatched:
        raise DataValidationError(
            f"{mismatched} risk assessment(s) name a fired-rule set that does "
            f"not match the supplied detections"
        )


class _Pass:
    """One ordered grouping pass.  Holds the mutable state ``build`` needs."""

    __slots__ = (
        "_alerts_closed",
        "_alerts_opened",
        "_closed",
        "_config",
        "_early",
        "_escalated",
        "_grouped",
        "_key_lookups",
        "_max_open",
        "_open",
        "_qualifying",
        "_run_mode",
        "_scope",
        "_scope_missing",
        "_suppressed_by_category",
        "_suppressed_by_reason",
        "_window",
    )

    def __init__(
        self,
        config: DetectionConfig,
        scope: EntityScopeTable | None,
        run_mode: AlertGroupingMode,
    ) -> None:
        self._config = config
        self._scope = scope
        self._run_mode = run_mode

        self._open: dict[_GroupKey, _OpenAlert] = {}
        # The most recent closed alert per key governs cooldown and collects
        # suppression counts.  Anything it superseded is already final and
        # waits in ``_early`` for emission.
        self._closed: dict[_GroupKey, _OpenAlert] = {}
        self._early: list[_OpenAlert] = []
        # Per key: the start of the current rate-limiting window and how many
        # alerts it has already emitted.  Two integers per key, so state stays
        # bounded by the number of distinct groups rather than by event count.
        self._window: dict[_GroupKey, tuple[datetime, int]] = {}

        self._suppressed_by_reason: dict[SuppressionReason, int] = {}
        self._suppressed_by_category: dict[AttackCategory, int] = {}
        self._qualifying = 0
        self._grouped = 0
        self._escalated = 0
        self._scope_missing = 0
        self._key_lookups = 0
        self._alerts_opened = 0
        self._alerts_closed = 0
        self._max_open = 0

    def run(self, assessments: Sequence[RiskAssessment]) -> AlertingResult:
        """Group every assessment, then close whatever is still open."""
        # Sorted here rather than trusted from the caller, so the order the
        # assessments arrived in cannot reach the output.
        ordered = sorted(
            assessments,
            key=lambda item: (item.anchor_event_time, item.anchor_event_id),
        )
        for assessment in ordered:
            self._consume(assessment)

        emitted = [alert.finalize() for alert in self._early]
        emitted.extend(alert.finalize() for alert in self._closed.values())
        for alert in self._open.values():
            self._alerts_closed += 1
            emitted.append(alert.finalize())

        alerts = tuple(
            sorted(
                emitted,
                key=lambda alert: (
                    alert.first_seen,
                    str(alert.attack_category),
                    str(alert.correlation_group),
                    alert.alert_id,
                ),
            )
        )
        return AlertingResult(
            alerts=alerts,
            stats=self._stats(alerts, len(assessments)),
            operations=AlertingOperationStats(
                assessments_scanned=len(ordered),
                key_lookups=self._key_lookups,
                alerts_opened=self._alerts_opened,
                alerts_closed=self._alerts_closed,
                distinct_group_count=len(self._window),
                max_open_groups=self._max_open,
            ),
        )

    # -- per assessment -----------------------------------------------------

    def _consume(self, assessment: RiskAssessment) -> None:
        """Apply the gates, then group or suppress one assessment."""
        if assessment.fired_rule_count == 0:
            # Not a qualifying detection at all: nothing fired, so there is
            # nothing to alert about and nothing to suppress.
            return
        if assessment.risk_score < self._config.alerting.min_alert_risk_score:
            self._suppress(SuppressionReason.BELOW_SCORE_FLOOR, assessment)
            return
        if not severity_at_least(
            assessment.severity, self._config.alerting.min_alert_severity
        ):
            self._suppress(SuppressionReason.BELOW_SEVERITY_FLOOR, assessment)
            return

        self._qualifying += 1
        key, mode = self._key_for(assessment)
        self._key_lookups += 1

        open_alert = self._open.get(key)
        if open_alert is not None:
            if (
                assessment.anchor_event_time - open_alert.last_seen
                <= self._config.alerting.grouping_window
            ):
                # Inside the window, inclusive at the boundary: absorb.
                if open_alert.absorb(assessment, assessment.fired_rule_ids):
                    self._escalated += 1
                self._grouped += 1
                return
            self._close(key, open_alert)

        self._open_or_suppress(key, mode, assessment)

    def _open_or_suppress(
        self, key: _GroupKey, mode: AlertGroupingMode, assessment: RiskAssessment
    ) -> None:
        """Open a new alert for *key*, unless cooldown or the limit forbids it."""
        closed = self._closed.get(key)
        escalating = False
        if closed is not None:
            elapsed = assessment.anchor_event_time - closed.last_seen
            if elapsed <= self._config.alerting.cooldown:
                # Cooldown is inclusive at the boundary: an event exactly one
                # cooldown after the previous alert's last contribution is
                # still suppressed.
                escalating = self._is_material(assessment, closed)
                if not (
                    escalating and self._config.alerting.escalation_bypasses_cooldown
                ):
                    closed.suppressed_event_count += 1
                    self._suppress(SuppressionReason.COOLDOWN, assessment)
                    return

        if not self._claim_window_slot(key, assessment.anchor_event_time):
            if closed is not None:
                closed.suppressed_event_count += 1
            self._suppress(SuppressionReason.RATE_LIMIT, assessment)
            return

        if escalating:
            self._escalated += 1
        self._open[key] = _OpenAlert(
            key=key,
            grouping_mode=mode,
            first_seen=assessment.anchor_event_time,
            last_seen=assessment.anchor_event_time,
            contributing_event_count=1,
            rule_ids=set(assessment.fired_rule_ids),
            risk_sum=assessment.risk_score,
            peak_risk_score=assessment.risk_score,
            initial_severity=assessment.severity,
            current_severity=assessment.severity,
        )
        self._grouped += 1
        self._alerts_opened += 1
        self._max_open = max(self._max_open, len(self._open))

    def _is_material(self, assessment: RiskAssessment, closed: _OpenAlert) -> bool:
        """Return whether *assessment* carries risk the closed alert did not.

        Cooldown exists to suppress *repeats*.  A finding that is more severe,
        or that peaks above anything the previous alert saw, is not a repeat --
        suppressing it would be the failure mode this whole module is built to
        avoid.
        """
        return (
            SEVERITY_ORDER[assessment.severity]
            > SEVERITY_ORDER[closed.current_severity]
            or assessment.risk_score > closed.peak_risk_score
        )

    def _claim_window_slot(self, key: _GroupKey, when: datetime) -> bool:
        """Return whether *key* may open another alert in its current window.

        The horizon is ``alert_limit_window``, not ``grouping_window``.  Two
        alerts for one key are always at least one grouping window apart -- a
        closer assessment is absorbed instead of opening a new alert -- so a
        limit measured over the grouping window could never bite.  This limit
        exists to backstop the escalation bypass, which can legitimately emit
        several alerts for one key in quick succession.
        """
        limit = self._config.alerting.max_alerts_per_group_per_window
        entry = self._window.get(key)
        if entry is None or when - entry[0] > self._config.alerting.alert_limit_window:
            self._window[key] = (when, 1)
            return True
        start, count = entry
        if count >= limit:
            return False
        self._window[key] = (start, count + 1)
        return True

    def _close(self, key: _GroupKey, alert: _OpenAlert) -> None:
        """Move an open alert into the closed registry, starting its cooldown."""
        previous = self._closed.get(key)
        if previous is not None:
            # A key can close more than once in a run.  The earlier alert is
            # already final, so it is emitted as its own row rather than
            # merged; only the most recent one governs cooldown.
            self._closed[key] = alert
            self._emit_early(previous)
        else:
            self._closed[key] = alert
        del self._open[key]
        self._alerts_closed += 1

    def _emit_early(self, alert: _OpenAlert) -> None:
        """Retain a superseded alert so it still reaches the output."""
        self._early.append(alert)

    def _suppress(self, reason: SuppressionReason, assessment: RiskAssessment) -> None:
        """Record one suppressed assessment against a reason and a category."""
        self._suppressed_by_reason[reason] = (
            self._suppressed_by_reason.get(reason, 0) + 1
        )
        category = assessment.primary_attack_category
        if category is not None:
            self._suppressed_by_category[category] = (
                self._suppressed_by_category.get(category, 0) + 1
            )

    # -- scope --------------------------------------------------------------

    def _key_for(
        self, assessment: RiskAssessment
    ) -> tuple[_GroupKey, AlertGroupingMode]:
        """Return the grouping key and per-alert mode for *assessment*.

        Raises:
            DetectionConfigurationError: if the correlation group declares no
                scope dimension.
            DataValidationError: under ``strict_scope``, if the required
                dimension is absent for this anchor.
        """
        category, group = _classify(assessment)

        if self._scope is None:
            return (
                _GroupKey(category, group, ScopeKind.NONE, None),
                AlertGroupingMode.CATEGORY_SCOPED,
            )

        try:
            kind = self._config.alerting.scope_dimension[group]
        except KeyError:
            raise DetectionConfigurationError(
                f"Correlation group {group!r} declares no scope dimension"
            ) from None

        value = self._scope.scope_for(assessment.anchor_event_id, kind)
        if value is None:
            self._scope_missing += 1
            if self._config.alerting.strict_scope:
                raise DataValidationError(
                    "The entity-scope table is missing a required scope "
                    "dimension for at least one assessed anchor, and strict "
                    "scope mode is configured"
                )
            # Degrade this group only.  Losing one grouping key is a smaller
            # harm than losing the alert set, and the count makes it visible.
            return (
                _GroupKey(category, group, ScopeKind.NONE, None),
                AlertGroupingMode.CATEGORY_SCOPED,
            )
        return (
            _GroupKey(category, group, kind, value),
            AlertGroupingMode.ENTITY_SCOPED,
        )

    # -- output -------------------------------------------------------------

    def _stats(
        self, alerts: Sequence[SecurityAlert], assessment_count: int
    ) -> AlertingStats:
        """Build the published accounting for this pass.

        Carries counts only.  No grouping key, no scope value, and no anchor
        identifier reaches these statistics -- they are written to reports that
        must stay safe to share.
        """
        entity_scoped = sum(
            1
            for alert in alerts
            if alert.grouping_mode is AlertGroupingMode.ENTITY_SCOPED
        )
        return AlertingStats(
            grouping_mode=self._run_mode,
            assessment_count=assessment_count,
            qualifying_count=self._qualifying,
            alert_count=len(alerts),
            grouped_detection_count=self._grouped,
            escalated_count=self._escalated,
            scope_missing_count=self._scope_missing,
            entity_scoped_count=entity_scoped,
            category_scoped_count=len(alerts) - entity_scoped,
            low_alert_reachable=self._config.low_alert_reachable,
            suppressed_by_reason=dict(self._suppressed_by_reason),
            suppressed_by_category=dict(self._suppressed_by_category),
        )


def _classify(
    assessment: RiskAssessment,
) -> tuple[AttackCategory, CorrelationGroup]:
    """Return the category and correlation group an assessment alerts under.

    Raises:
        DataValidationError: if the assessment carries no primary category.
            A qualifying assessment always has one -- the schema enforces it --
            so reaching this means the caller built one by hand incorrectly.
    """
    category = assessment.primary_attack_category
    if category is None:  # pragma: no cover - enforced by RiskAssessment
        raise DataValidationError(
            "a qualifying risk assessment must carry a primary attack category"
        )
    return category, _CATEGORY_GROUP[category]


#: Which correlation group each attack category belongs to.  Derived from the
#: rule catalog at import so the mapping cannot drift from the registry, and
#: frozen so alert grouping is a declared contract rather than a lookup that
#: might miss.
def _build_category_group_map() -> dict[AttackCategory, CorrelationGroup]:
    """Map every registered attack category to its correlation group.

    Raises:
        DetectionConfigurationError: if two rules share a category but declare
            different correlation groups.  That would make an alert's group
            depend on which rule happened to rank first, which is exactly the
            kind of order dependence this layer must not have.
    """
    from password_attack_detector.detection.catalog import RULE_CATALOG

    mapping: dict[AttackCategory, CorrelationGroup] = {}
    for spec in RULE_CATALOG.specs:
        existing = mapping.get(spec.attack_category)
        if existing is not None and existing is not spec.correlation_group:
            raise DetectionConfigurationError(
                f"Attack category {spec.attack_category!r} maps to more than "
                f"one correlation group"
            )
        mapping[spec.attack_category] = spec.correlation_group
    return mapping


_CATEGORY_GROUP: Final[dict[AttackCategory, CorrelationGroup]] = (
    _build_category_group_map()
)
