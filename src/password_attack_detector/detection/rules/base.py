"""The rule framework: preparation, snapshot access, evidence, and signal maths.

A rule is two objects.  A :class:`Rule` holds the specification and knows how to
*prepare*; a :class:`PreparedRule` holds resolved column names and frozen
thresholds and knows how to *evaluate*.  The split is what keeps the hot path
cheap: template resolution, catalog lookups, prohibited-column checks, and
threshold validation happen once per run, and per-snapshot work is dictionary
reads and bounded arithmetic.

**A rule sees one feature snapshot and nothing else.**  :class:`SnapshotView`
can reach only the columns the rule declared and the preparation resolved, so a
rule cannot read a label, a split assignment, campaign metadata, an entity
scope, a model output, or any row other than its anchor.  There is no handle to
the canonical event table anywhere in this module.

Signal strength is built from bounded, monotone components:

* :func:`saturate` for "greater is worse" quantities;
* :func:`saturate_inverse` for "smaller is worse" quantities;
* :func:`weighted_strength` to combine them.

Every helper is total.  There is no division by zero, no ``NaN``, and no
infinity: degenerate inputs take a documented branch instead of producing one.
At exact threshold equality a component contributes ``0.0``, and
:func:`weighted_strength` floors the result at ``min_signal_strength``, which is
strictly positive -- so a rule that fires always reports a strictly positive
strength.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from password_attack_detector.detection.catalog import (
    RuleSpec,
    resolve_rule_features,
)
from password_attack_detector.detection.config import DetectionConfig, SignalConfig
from password_attack_detector.detection.enums import RuleStatus, Severity
from password_attack_detector.detection.schemas import (
    EvidenceItem,
    RuleEvaluationResult,
)
from password_attack_detector.exceptions import RuleEvaluationError
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    FeatureCatalog,
)

__all__ = [
    "BasePreparedRule",
    "BaseRule",
    "PreparedRule",
    "Rule",
    "RulePreparation",
    "SignalComponent",
    "SnapshotView",
    "build_evidence",
    "clamp",
    "insufficient_history_reason_code",
    "safe_ratio",
    "saturate",
    "saturate_inverse",
    "weighted_strength",
]

#: Floor used in place of a zero denominator in :func:`saturate_inverse`.  Small
#: enough not to distort a real interval, large enough that the ratio stays
#: finite for any representable input.
_EPSILON: float = 1e-9

#: Placeholder feature name for evidence derived from several columns rather
#: than read from one.
_DERIVED_FEATURE: str = "derived"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Return *value* confined to ``[lower, upper]``.

    Raises:
        ValueError: if *value* is ``NaN`` or infinite.  A non-finite input is a
            defect in the caller, not something to silently pin to a bound.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError("cannot clamp a NaN or infinite value")
    return max(lower, min(upper, value))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator``, or ``None`` when it is undefined.

    Returns ``None`` for a zero denominator rather than raising, because an
    undefined ratio is an ordinary state for a rule -- a source with no
    attempts has no attempts-per-account figure -- and the caller decides
    whether that means ``not_fired`` or ``insufficient_data``.

    Raises:
        ValueError: if either operand is ``NaN`` or infinite.
    """
    for name, value in (("numerator", numerator), ("denominator", denominator)):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be finite")
    if denominator == 0.0:
        return None
    return numerator / denominator


def saturate(observed: float, threshold: float, saturation_multiple: float) -> float:
    """Return a bounded "greater is worse" component in ``[0, 1]``.

    The component is ``0.0`` at exact threshold equality and reaches ``1.0`` at
    ``saturation_multiple`` times the threshold, rising linearly between.  That
    makes threshold equality explicit rather than an accident of rounding: a
    rule that only just crosses its threshold contributes nothing beyond the
    configured floor.

    Degenerate inputs take documented branches instead of dividing by zero: a
    non-positive threshold saturates for any positive observation, and a
    saturation multiple of one becomes a step.

    Raises:
        ValueError: if any argument is ``NaN`` or infinite.
    """
    _require_finite_args(
        observed=observed, threshold=threshold, saturation_multiple=saturation_multiple
    )
    if threshold <= 0.0:
        return 1.0 if observed > 0.0 else 0.0
    ratio = observed / threshold
    return _saturation_curve(ratio, saturation_multiple)


def saturate_inverse(
    observed: float, threshold: float, saturation_multiple: float
) -> float:
    """Return a bounded "smaller is worse" component in ``[0, 1]``.

    The mirror of :func:`saturate`, for quantities where a *lower* observation
    is more suspicious -- an interarrival gap, a coefficient of variation.  The
    component is ``0.0`` at exact threshold equality and reaches ``1.0`` when
    the observation is ``saturation_multiple`` times *below* the threshold.

    A zero or negative observation is floored at :data:`_EPSILON`, so the ratio
    is always finite and the function never divides by zero.

    Raises:
        ValueError: if any argument is ``NaN`` or infinite.
    """
    _require_finite_args(
        observed=observed, threshold=threshold, saturation_multiple=saturation_multiple
    )
    if threshold <= 0.0:
        return 0.0
    ratio = threshold / max(observed, _EPSILON)
    return _saturation_curve(ratio, saturation_multiple)


def _saturation_curve(ratio: float, saturation_multiple: float) -> float:
    """Map a threshold-relative *ratio* onto ``[0, 1]``."""
    if saturation_multiple <= 1.0:
        return 1.0 if ratio > 1.0 else 0.0
    return clamp((ratio - 1.0) / (saturation_multiple - 1.0))


def _require_finite_args(**values: float) -> None:
    """Raise when any keyword argument is ``NaN`` or infinite."""
    for name, value in values.items():
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be finite; got a NaN or infinity")


@dataclass(frozen=True, slots=True)
class SignalComponent:
    """One weighted contribution to a rule's signal strength."""

    name: str
    weight: float
    value: float

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise ValueError(
                f"signal component {self.name!r} must carry a positive weight"
            )
        if math.isnan(self.value) or math.isinf(self.value):
            raise ValueError(f"signal component {self.name!r} must be finite")
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"signal component {self.name!r} must lie in [0, 1]")


def weighted_strength(
    components: Sequence[SignalComponent], *, signal: SignalConfig
) -> float:
    """Combine *components* into a signal strength in ``(0, 1]``.

    The weighted mean of the components is rescaled onto
    ``[min_signal_strength, 1]``.  The floor is what guarantees a fired rule
    reports a strictly positive strength even when every component sits exactly
    at its threshold.

    Components are summed in the order given, so the result is bit-identical
    across runs.  With no components the floor is returned -- a rule that fired
    on a purely categorical condition still carries the minimum strength.

    Raises:
        ValueError: if the total weight is not positive.
    """
    if not components:
        return signal.min_signal_strength

    total_weight = math.fsum(component.weight for component in components)
    if total_weight <= 0.0:
        raise ValueError("signal components must carry a positive total weight")

    weighted = math.fsum(component.weight * component.value for component in components)
    raw = clamp(weighted / total_weight)
    floor = signal.min_signal_strength
    return clamp(floor + (1.0 - floor) * raw)


# ---------------------------------------------------------------------------
# Preparation and snapshot access
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RulePreparation:
    """Everything one rule needs, resolved once before any snapshot is read.

    Holds the concrete column names the rule's templates resolved to, the
    effective thresholds, the configured severity, and the shared signal
    normalization parameters.  Nothing here is recomputed per row.
    """

    spec: RuleSpec
    parameters: Mapping[str, Any]
    features: Mapping[str, str]
    severity: Severity
    signal: SignalConfig

    def feature(self, template: str) -> str:
        """Return the column name *template* resolved to.

        Raises:
            RuleEvaluationError: if the rule did not declare *template*.
        """
        try:
            return self.features[template]
        except KeyError:
            raise RuleEvaluationError(
                f"rule {self.spec.rule_id} did not declare feature template "
                f"{template!r}"
            ) from None

    def param_int(self, name: str) -> int:
        """Return integer parameter *name*."""
        value = self._param(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuleEvaluationError(
                f"rule {self.spec.rule_id} parameter {name!r} is not an integer"
            )
        return value

    def param_float(self, name: str) -> float:
        """Return numeric parameter *name* as a float."""
        value = self._param(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuleEvaluationError(
                f"rule {self.spec.rule_id} parameter {name!r} is not a number"
            )
        return float(value)

    def param_bool(self, name: str) -> bool:
        """Return boolean parameter *name*."""
        value = self._param(name)
        if not isinstance(value, bool):
            raise RuleEvaluationError(
                f"rule {self.spec.rule_id} parameter {name!r} is not a boolean"
            )
        return value

    def param_str(self, name: str) -> str:
        """Return string parameter *name*."""
        value = self._param(name)
        if not isinstance(value, str):
            raise RuleEvaluationError(
                f"rule {self.spec.rule_id} parameter {name!r} is not a string"
            )
        return value

    def _param(self, name: str) -> Any:
        try:
            return self.parameters[name]
        except KeyError:
            raise RuleEvaluationError(
                f"rule {self.spec.rule_id} has no prepared parameter {name!r}"
            ) from None


class SnapshotView:
    """Typed, bounded access to one Phase 3 feature snapshot.

    Columns are addressed by *template*, never by raw name, so a rule can only
    reach what it declared and the preparation resolved.  Reads are by key, so
    the order in which the snapshot mapping was built cannot change any result.
    """

    __slots__ = ("_preparation", "_row")

    def __init__(self, row: Mapping[str, Any], preparation: RulePreparation) -> None:
        self._row = row
        self._preparation = preparation

    def raw(self, template: str) -> Any:
        """Return the snapshot value for *template*.

        Raises:
            RuleEvaluationError: if the rule did not declare *template*, or the
                snapshot does not carry the resolved column.
        """
        name = self._preparation.feature(template)
        if name not in self._row:
            raise RuleEvaluationError(
                f"feature snapshot does not carry column {name!r}, required by "
                f"rule {self._preparation.spec.rule_id}"
            )
        return self._row[name]

    def number(self, template: str) -> float | None:
        """Return a numeric column as a float, or ``None`` when it is null.

        Raises:
            RuleEvaluationError: if the value is not numeric, or is ``NaN`` or
                infinite.  A non-finite feature is a corrupt snapshot, and
                silently treating it as a large number would let it dominate
                every saturation curve it touches.
        """
        value = self.raw(template)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuleEvaluationError(
                f"column for {template!r} is not numeric in this snapshot"
            )
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            raise RuleEvaluationError(
                f"column for {template!r} holds a NaN or infinite value"
            )
        return numeric

    def count(self, template: str) -> int:
        """Return a non-nullable count column.

        Raises:
            RuleEvaluationError: if the value is null or not an integer.  Count
                features are declared non-nullable by the feature catalog, so a
                null here means the snapshot does not match its schema.
        """
        value = self.raw(template)
        if value is None:
            raise RuleEvaluationError(
                f"count column for {template!r} is null; the feature catalog "
                f"declares it non-nullable"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuleEvaluationError(
                f"count column for {template!r} is not an integer"
            )
        return value

    def flag(self, template: str) -> bool | None:
        """Return a boolean column, or ``None`` when it is null."""
        value = self.raw(template)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise RuleEvaluationError(
                f"column for {template!r} is not a boolean in this snapshot"
            )
        return value

    def text(self, template: str) -> str | None:
        """Return a string column, or ``None`` when it is null."""
        value = self.raw(template)
        if value is None:
            return None
        if not isinstance(value, str):
            raise RuleEvaluationError(
                f"column for {template!r} is not a string in this snapshot"
            )
        return value

    def missing_history(self) -> tuple[str, ...]:
        """Return the minimum-history templates that are null on this snapshot.

        A non-empty result means the rule cannot reach a verdict: it has not
        observed a clean negative, it has not observed the history at all.
        """
        missing: list[str] = []
        for template in self._preparation.spec.min_history.required_non_null_features:
            if self.raw(template) is None:
                missing.append(template)
        return tuple(missing)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _format_value(value: bool | int | float | str) -> str:
    """Render a value for an evidence message, deterministically.

    Floats are rendered to at most four decimals with trailing zeros removed,
    so the same observation always produces the same message text and no
    platform-dependent repr reaches an artifact.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return value


def build_evidence(
    preparation: RulePreparation,
    evidence_code: str,
    observed_value: bool | int | float | str,
    *,
    threshold_value: bool | int | float | str | None = None,
) -> EvidenceItem:
    """Build one evidence item from the rule's declared definition.

    The message comes from the catalog's frozen template rendered with
    sanitized values.  No caller-supplied string reaches the message, and the
    template grammar is validated at catalog build time.

    Raises:
        RuleEvaluationError: if the rule declares no such evidence code.
    """
    definition = preparation.spec.evidence_definition(evidence_code)
    feature_name = (
        _DERIVED_FEATURE
        if definition.feature_template is None
        else preparation.feature(definition.feature_template)
    )
    message = definition.message_template.format(
        observed=_format_value(observed_value),
        threshold="" if threshold_value is None else _format_value(threshold_value),
        unit=definition.unit or "",
        feature=feature_name,
    )
    return EvidenceItem(
        evidence_code=evidence_code,
        feature_name=feature_name,
        comparator=definition.comparator,
        observed_value=observed_value,
        threshold_value=threshold_value,
        unit=definition.unit,
        message=message,
    )


def insufficient_history_reason_code(resolved_feature_name: str) -> str:
    """Return the deterministic reason code for unavailable history.

    Derived from the resolved column name so the reason names the feature that
    was unavailable without leaking anything beyond a schema column name.
    """
    normalized = "".join(
        character if character.isalnum() else "_"
        for character in resolved_feature_name.upper()
    )
    return f"INSUFFICIENT_HISTORY_{normalized}"


# ---------------------------------------------------------------------------
# Rule contract
# ---------------------------------------------------------------------------


@runtime_checkable
class PreparedRule(Protocol):
    """A rule with resolved columns and frozen thresholds, ready to evaluate."""

    preparation: RulePreparation

    def evaluate(self, row: Mapping[str, Any]) -> RuleEvaluationResult:
        """Evaluate one feature snapshot."""
        ...


@runtime_checkable
class Rule(Protocol):
    """A registered rule: a specification plus a one-time preparation step."""

    spec: RuleSpec

    def prepare(
        self, config: DetectionConfig, feature_catalog: FeatureCatalog
    ) -> PreparedRule:
        """Resolve columns and thresholds once, before any snapshot is read."""
        ...


class BaseRule(ABC):
    """Common preparation for every rule.

    Subclasses supply a :class:`RuleSpec` and build their prepared form; this
    class does the boundary work that must not be reimplemented per rule:
    resolving templates against the feature catalog, rejecting a prohibited or
    undeclared column, and freezing the effective thresholds.
    """

    def __init__(self, spec: RuleSpec) -> None:
        self.spec = spec

    def prepare(
        self, config: DetectionConfig, feature_catalog: FeatureCatalog
    ) -> PreparedRule:
        """Prepare this rule for a run.

        Called once per run.  Every catalog lookup, template resolution, and
        prohibited-column check happens here so evaluation stays arithmetic.

        Raises:
            RuleEvaluationError: if a declared feature does not exist in the
                configured feature catalog, is prohibited, or carries a leakage
                class a rule may not read.
        """
        parameters = config.parameters_for(self.spec.rule_id)
        features = resolve_rule_features(self.spec, parameters, feature_catalog)
        preparation = RulePreparation(
            spec=self.spec,
            parameters=parameters,
            features=features,
            severity=config.severity_for_rule(self.spec.rule_id),
            signal=config.signal,
        )
        return self._build(preparation)

    @abstractmethod
    def _build(self, preparation: RulePreparation) -> PreparedRule:
        """Return the prepared form of this rule."""


class BasePreparedRule(ABC):
    """Common evaluation scaffold for every prepared rule.

    Handles the parts of evaluation that must behave identically across rules:
    reading the anchor keys, applying the minimum-history gate before any
    threshold comparison, and assembling a well-formed result.  Subclasses
    implement :meth:`_evaluate`, which is reached only once the rule is known to
    have the history it declared.
    """

    def __init__(self, preparation: RulePreparation) -> None:
        self.preparation = preparation

    # -- entry point --------------------------------------------------------

    def evaluate(self, row: Mapping[str, Any]) -> RuleEvaluationResult:
        """Evaluate one feature snapshot.

        Raises:
            RuleEvaluationError: if the snapshot does not carry the anchor keys
                or a column this rule declared.
        """
        anchor_id, anchor_time = self._anchor(row)
        view = SnapshotView(row, self.preparation)

        missing = view.missing_history()
        if missing:
            return self.insufficient_data(
                anchor_id,
                anchor_time,
                reason_codes=tuple(
                    insufficient_history_reason_code(self.preparation.feature(template))
                    for template in missing
                ),
            )
        return self._evaluate(view, anchor_id, anchor_time)

    @abstractmethod
    def _evaluate(
        self, view: SnapshotView, anchor_id: str, anchor_time: datetime
    ) -> RuleEvaluationResult:
        """Evaluate a snapshot that satisfies the minimum-history requirement."""

    # -- result constructors ------------------------------------------------

    def fired(
        self,
        anchor_id: str,
        anchor_time: datetime,
        *,
        signal_strength: float,
        evidence: Sequence[EvidenceItem],
        reason_codes: Sequence[str],
    ) -> RuleEvaluationResult:
        """Build a fired result."""
        return self._result(
            anchor_id,
            anchor_time,
            status=RuleStatus.FIRED,
            signal_strength=signal_strength,
            evidence=tuple(evidence),
            reason_codes=tuple(reason_codes),
        )

    def not_fired(
        self,
        anchor_id: str,
        anchor_time: datetime,
        *,
        reason_codes: Sequence[str] = (),
    ) -> RuleEvaluationResult:
        """Build a not-fired result.

        Reason codes are optional here and name the condition that did not
        match, which is diagnostic rather than published.
        """
        return self._result(
            anchor_id,
            anchor_time,
            status=RuleStatus.NOT_FIRED,
            signal_strength=0.0,
            evidence=(),
            reason_codes=tuple(reason_codes),
        )

    def insufficient_data(
        self,
        anchor_id: str,
        anchor_time: datetime,
        *,
        reason_codes: Sequence[str],
    ) -> RuleEvaluationResult:
        """Build an insufficient-data result.

        Distinct from :meth:`not_fired` on purpose: the rule did not observe a
        clean negative, it did not observe the history at all.
        """
        return self._result(
            anchor_id,
            anchor_time,
            status=RuleStatus.INSUFFICIENT_DATA,
            signal_strength=0.0,
            evidence=(),
            reason_codes=tuple(reason_codes),
        )

    def disabled(self, anchor_id: str, anchor_time: datetime) -> RuleEvaluationResult:
        """Build a disabled result."""
        return self._result(
            anchor_id,
            anchor_time,
            status=RuleStatus.DISABLED,
            signal_strength=0.0,
            evidence=(),
            reason_codes=(),
        )

    def strength(self, components: Sequence[SignalComponent]) -> float:
        """Combine *components* using this rule's configured normalization."""
        return weighted_strength(components, signal=self.preparation.signal)

    def evidence_for(
        self,
        evidence_code: str,
        observed_value: bool | int | float | str,
        *,
        threshold_value: bool | int | float | str | None = None,
    ) -> EvidenceItem:
        """Build one declared evidence item for this rule."""
        return build_evidence(
            self.preparation,
            evidence_code,
            observed_value,
            threshold_value=threshold_value,
        )

    # -- internals ----------------------------------------------------------

    def _result(
        self,
        anchor_id: str,
        anchor_time: datetime,
        *,
        status: RuleStatus,
        signal_strength: float,
        evidence: tuple[EvidenceItem, ...],
        reason_codes: tuple[str, ...],
    ) -> RuleEvaluationResult:
        spec = self.preparation.spec
        return RuleEvaluationResult(
            rule_id=spec.rule_id,
            rule_version=spec.rule_version,
            anchor_event_id=anchor_id,
            anchor_event_time=anchor_time,
            status=status,
            rule_family=spec.family,
            attack_category=spec.attack_category,
            correlation_group=spec.correlation_group,
            severity=self.preparation.severity,
            signal_strength=signal_strength,
            evidence=evidence,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _anchor(row: Mapping[str, Any]) -> tuple[str, datetime]:
        """Return the anchor identifier and timestamp from *row*.

        The anchor keys are read directly rather than through
        :class:`SnapshotView`, because they are the row's identity rather than
        a rule input; no rule may declare them as a feature.
        """
        for column in (ANCHOR_EVENT_ID, ANCHOR_EVENT_TIME):
            if column not in row:
                raise RuleEvaluationError(
                    f"feature snapshot does not carry the anchor column {column!r}"
                )
        anchor_id = row[ANCHOR_EVENT_ID]
        anchor_time = row[ANCHOR_EVENT_TIME]
        if not isinstance(anchor_id, str):
            raise RuleEvaluationError("anchor_event_id must be a string")
        if not isinstance(anchor_time, datetime):
            raise RuleEvaluationError("anchor_event_time must be a datetime")
        return anchor_id, anchor_time
