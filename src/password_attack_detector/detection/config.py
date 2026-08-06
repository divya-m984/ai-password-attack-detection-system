"""Typed, versioned configuration for the rule-based detection layer.

``DetectionConfig`` is the single source of truth for one detection run.  Its
``fingerprint()`` method produces a SHA-256 hex digest covering *semantic*
detection behaviour only.  Output directories, overwrite flags, absolute paths,
creation timestamps, and machine-specific values are excluded, so the same
semantic configuration stored in two different directories fingerprints
identically -- the same contract ``FeatureConfig`` follows.

Configuration is **data, never logic**.  YAML is read with ``yaml.safe_load``
and supplies values for parameters a rule has already declared in the catalog.
There is no expression evaluation, no callable reference, no import path, and no
``eval``, ``exec``, or dynamic import anywhere in this package.  A key that
looks like a secret is rejected outright: detection needs no credential, so
accepting one could only ever be a mistake.

Two alert gates, both configured, decide whether an assessment becomes an
alert: ``min_alert_risk_score`` and ``min_alert_severity``.  ``LOW`` is an
ordinary severity and a valid alert severity -- nothing here forbids a ``LOW``
alert.  An operator who wants ``LOW`` findings to stay diagnostic raises
``min_alert_severity``; that is a configuration decision, not a rule of the
implementation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from password_attack_detector.detection.catalog import (
    RULE_CATALOG,
    RULE_CATALOG_VERSION,
    RuleCatalog,
)
from password_attack_detector.detection.enums import (
    CorrelationGroup,
    RuleFamily,
    ScopeKind,
    Severity,
)
from password_attack_detector.detection.schemas import DETECTION_SCHEMA_VERSION
from password_attack_detector.exceptions import (
    ConfigurationError,
    DetectionConfigurationError,
)
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    parse_duration,
)

__all__ = [
    "DETECTION_FINGERPRINT_EXCLUDED_FIELDS",
    "PROHIBITED_CONFIG_KEY_TOKENS",
    "AlertingConfig",
    "DetectionConfig",
    "RuleSettings",
    "ScoringConfig",
    "SeverityThresholds",
    "SignalConfig",
    "load_detection_config",
]

#: Fields of :class:`DetectionConfig` deliberately excluded from its
#: fingerprint.  These describe *where* output goes, not *what* detection means.
DETECTION_FINGERPRINT_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {"output_dir", "reports_dir", "overwrite"}
)

#: Tokens that must not appear in any detection configuration key.  Detection
#: consumes feature snapshots and thresholds; it has no use for a credential,
#: so a key shaped like one is a mistake worth failing on rather than ignoring.
#:
#: Matching is token-based (keys are split on ``_`` and ``-``) so a legitimate
#: key is never rejected for containing a substring.
PROHIBITED_CONFIG_KEY_TOKENS: frozenset[str] = frozenset(
    {
        "secret",
        "secrets",
        "password",
        "passwords",
        "passwd",
        "token",
        "tokens",
        "credential",
        "credentials",
        "key",
        "keys",
        "apikey",
        "privatekey",
        "salt",
        "pepper",
        "cookie",
        "cookies",
        "bearer",
        "hash",
        "hashes",
    }
)


def _controlled_vocabulary() -> frozenset[str]:
    """Return every configuration key drawn from a validated vocabulary.

    Some mappings are keyed by project vocabulary rather than by a fixed field
    name: ``rules`` by rule identifier, ``family_weights`` by rule family,
    ``scope_dimension`` by correlation group, and ``parameters`` by a name the
    rule already declares.  Those keys are validated against their own
    registries elsewhere, so the secret scanner exempts them from its token
    check rather than rejecting a legitimate term such as
    ``credential_guessing_single_target``.

    Building the exemption from the registries keeps it correct as they grow;
    a hand-written allowlist would drift.
    """
    vocabulary: set[str] = set(RULE_CATALOG.rule_ids)
    vocabulary |= {str(member) for member in CorrelationGroup}
    vocabulary |= {str(member) for member in RuleFamily}
    vocabulary |= {str(member) for member in Severity}
    vocabulary |= {str(member) for member in ScopeKind}
    for spec in RULE_CATALOG.specs:
        vocabulary |= {parameter.name for parameter in spec.parameters}
    return frozenset(vocabulary)


_CONTROLLED_VOCABULARY: frozenset[str] = _controlled_vocabulary()


def _coerce_duration(value: object) -> object:
    """Accept a duration string such as ``"15m"`` for a timedelta field.

    Reuses the Phase 3 duration grammar so one parser governs every duration in
    the project, and re-raises as ``ValueError`` because pydantic wraps only
    ``ValueError`` and ``AssertionError`` into a ``ValidationError``.
    """
    if isinstance(value, str):
        try:
            return parse_duration(value)
        except ConfigurationError as exc:
            raise ValueError(str(exc)) from None
    return value


def _check_safe_path(path: Path | None, field_name: str) -> Path | None:
    """Reject an output path containing parent-directory traversal."""
    if path is not None and any(part == ".." for part in path.parts):
        raise ValueError(f"{field_name} must not contain '..' path components")
    return path


def _reject_secret_keys(data: object, path: str = "") -> None:
    """Walk a raw configuration mapping and reject secret-shaped keys.

    Raises:
        ConfigurationError: naming the offending key path only.  The value is
            never read, echoed, or included in the message.
    """
    if isinstance(data, dict):
        for raw_key, value in data.items():
            key = str(raw_key)
            if key not in _CONTROLLED_VOCABULARY:
                tokens = set(key.lower().replace("-", "_").split("_"))
                offending = sorted(tokens & PROHIBITED_CONFIG_KEY_TOKENS)
                if offending:
                    where = f"{path}.{key}" if path else key
                    raise ConfigurationError(
                        f"Detection configuration key {where!r} looks like a "
                        f"secret ({offending}); detection configuration must "
                        f"never carry credentials"
                    )
            _reject_secret_keys(value, f"{path}.{key}" if path else key)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            _reject_secret_keys(item, f"{path}[{index}]")


class SignalConfig(BaseModel):
    """Normalization parameters shared by every rule's signal strength.

    ``saturation_multiple`` is the multiple of a threshold at which a component
    reaches its maximum.  ``min_signal_strength`` is the value a rule returns at
    exact threshold equality -- strictly positive, which is what guarantees a
    fired rule never scores as though nothing fired.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    saturation_multiple: float = Field(default=3.0, gt=1.0, le=100.0)
    min_signal_strength: float = Field(default=0.15, gt=0.0, lt=1.0)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint."""
        return {
            "saturation_multiple": self.saturation_multiple,
            "min_signal_strength": self.min_signal_strength,
        }


class ScoringConfig(BaseModel):
    """Correlation-aware risk aggregation policy.

    There is deliberately no configurable baseline score: an assessment with no
    fired rules scores exactly ``0.0``, so a zero can always be read as "nothing
    fired".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scoring_version: Literal["1.0.0"] = "1.0.0"
    correlation_reducer: Literal["max", "bounded_sum"] = "max"
    group_combiner: Literal["noisy_or"] = "noisy_or"
    min_fired_risk_score: float = Field(default=1.0, gt=0.0, le=100.0)
    round_decimals: int = Field(default=4, ge=0, le=12)
    top_evidence_count: int = Field(default=3, ge=1, le=20)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint."""
        return {
            "scoring_version": self.scoring_version,
            "correlation_reducer": self.correlation_reducer,
            "group_combiner": self.group_combiner,
            "min_fired_risk_score": self.min_fired_risk_score,
            "round_decimals": self.round_decimals,
            "top_evidence_count": self.top_evidence_count,
        }


class SeverityThresholds(BaseModel):
    """The three strictly ordered boundaries of the four-level severity ladder.

    Positive risk scores map across all four levels: below ``medium`` is ``LOW``
    -- an ordinary severity, not a suppressed one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    medium: float = Field(default=40.0, gt=0.0, le=100.0)
    high: float = Field(default=65.0, gt=0.0, le=100.0)
    critical: float = Field(default=85.0, gt=0.0, le=100.0)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        """Boundaries must be strictly increasing."""
        if not (self.medium < self.high < self.critical):
            raise ValueError(
                "severity thresholds must be strictly increasing "
                "(medium < high < critical)"
            )
        return self

    def severity_for(self, risk_score: float) -> Severity:
        """Return the severity band *risk_score* falls into.

        Boundaries are inclusive from below, so a score exactly equal to a
        boundary lands in the higher band.
        """
        if risk_score >= self.critical:
            return Severity.CRITICAL
        if risk_score >= self.high:
            return Severity.HIGH
        if risk_score >= self.medium:
            return Severity.MEDIUM
        return Severity.LOW

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint."""
        return {"medium": self.medium, "high": self.high, "critical": self.critical}


#: Default weight per rule family.  Account compromise ranks highest because a
#: successful authentication from novel context is the finding least tolerable
#: to miss; automation ranks lowest because legitimate service accounts share
#: its shape exactly.  A configuration that supplies ``family_weights`` replaces
#: this mapping wholesale, and the coverage validator then rejects any enabled
#: family the replacement left out.
_DEFAULT_FAMILY_WEIGHTS: dict[RuleFamily, float] = {
    RuleFamily.BRUTE_FORCE: 0.90,
    RuleFamily.SPRAYING: 0.85,
    RuleFamily.STUFFING: 0.85,
    RuleFamily.ACCOUNT_COMPROMISE: 0.95,
    RuleFamily.LOCATION: 0.80,
    RuleFamily.AUTOMATION: 0.70,
}

#: The scope dimension each correlation group groups on when an entity-scope
#: table is supplied.  Source-centric behaviour groups on the source; account-
#: centric behaviour groups on the account.
_DEFAULT_SCOPE_DIMENSION: dict[CorrelationGroup, ScopeKind] = {
    CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET: ScopeKind.USER,
    CorrelationGroup.SOURCE_FANOUT: ScopeKind.SOURCE,
    CorrelationGroup.SESSION_ANOMALY: ScopeKind.USER,
    CorrelationGroup.LOCATION_MOVEMENT: ScopeKind.USER,
    CorrelationGroup.AUTOMATION_TIMING: ScopeKind.SOURCE,
}


class AlertingConfig(BaseModel):
    """Alert grouping, deduplication, cooldown, and suppression policy.

    ``min_alert_risk_score`` and ``min_alert_severity`` are two independent
    gates.  All four severities are accepted for the second gate; ``low`` is the
    default, so a ``LOW`` assessment that reaches ``min_alert_risk_score``
    produces an ordinary ``LOW`` alert.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    alerting_version: Literal["1.0.0"] = "1.0.0"
    grouping_window: timedelta = timedelta(minutes=15)
    cooldown: timedelta = timedelta(minutes=30)
    max_alerts_per_group_per_window: int = Field(default=5, ge=1)
    #: Horizon the per-group alert limit counts over.
    #:
    #: Deliberately longer than ``grouping_window``.  Two alerts for one group
    #: are always at least one grouping window apart -- a closer assessment is
    #: absorbed into the open alert instead of opening a new one -- so a limit
    #: measured over the grouping window could never bite.  The limit exists to
    #: backstop the escalation bypass, which can legitimately emit several
    #: alerts for one group in quick succession, and that needs a horizon
    #: spanning more than one window to be meaningful.
    alert_limit_window: timedelta = timedelta(hours=1)
    min_alert_risk_score: float = Field(default=10.0, gt=0.0, le=100.0)
    min_alert_severity: Severity = Severity.LOW
    escalation_bypasses_cooldown: bool = True
    #: Whether a missing scope value is fatal.  Off by default: an anchor whose
    #: scope dimension is null degrades to category-scoped grouping and is
    #: counted, because losing one grouping key is a worse outcome than losing
    #: the whole alert set.  An operator who needs every alert attributable
    #: turns this on and gets a hard failure instead.
    strict_scope: bool = False
    scope_dimension: dict[CorrelationGroup, ScopeKind] = Field(
        default_factory=lambda: dict(_DEFAULT_SCOPE_DIMENSION)
    )

    @field_validator("grouping_window", "cooldown", "alert_limit_window", mode="before")
    @classmethod
    def coerce_duration(cls, value: object) -> object:
        """Accept duration strings such as ``"15m"`` for timedelta fields."""
        return _coerce_duration(value)

    @field_validator("grouping_window", "cooldown", "alert_limit_window", mode="after")
    @classmethod
    def check_positive(cls, value: timedelta) -> timedelta:
        """Every alerting interval must be strictly positive."""
        if value <= timedelta(0):
            raise ValueError("alerting intervals must be strictly positive")
        return value

    @model_validator(mode="after")
    def check_scope_dimension(self) -> Self:
        """Every correlation group must declare exactly one scope dimension."""
        declared = set(self.scope_dimension)
        known = set(CorrelationGroup)
        missing = sorted(str(group) for group in known - declared)
        if missing:
            raise ValueError(
                f"scope_dimension does not cover correlation group(s) {missing}"
            )
        return self

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint."""
        return {
            "alerting_version": self.alerting_version,
            "grouping_window_seconds": int(self.grouping_window.total_seconds()),
            "cooldown_seconds": int(self.cooldown.total_seconds()),
            "max_alerts_per_group_per_window": self.max_alerts_per_group_per_window,
            "alert_limit_window_seconds": int(self.alert_limit_window.total_seconds()),
            "min_alert_risk_score": self.min_alert_risk_score,
            "min_alert_severity": str(self.min_alert_severity),
            "escalation_bypasses_cooldown": self.escalation_bypasses_cooldown,
            "strict_scope": self.strict_scope,
            "scope_dimension": {
                str(group): str(kind)
                for group, kind in sorted(
                    self.scope_dimension.items(), key=lambda item: str(item[0])
                )
            },
        }


class RuleSettings(BaseModel):
    """Per-rule overrides: an optional severity and typed threshold values.

    ``parameters`` carries values only.  Each key must name a parameter the rule
    already declares in the catalog, and each value must satisfy that
    parameter's declared type and bounds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Severity | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class DetectionConfig(BaseModel):
    """Complete configuration for one detection run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_schema_version: Literal["1.0.0"] = DETECTION_SCHEMA_VERSION
    required_feature_schema_version: str = FEATURE_SCHEMA_VERSION
    # Deliberately a plain ``str`` rather than a ``Literal``: the cross-field
    # validator below rejects a mismatch with a message that names the
    # registered version, which is more useful than a bare type error.
    rule_catalog_version: str = RULE_CATALOG_VERSION

    enabled_rule_ids: tuple[str, ...] = Field(
        default_factory=lambda: RULE_CATALOG.rule_ids
    )
    rules: dict[str, RuleSettings] = Field(default_factory=dict)
    family_weights: dict[RuleFamily, float] = Field(
        default_factory=lambda: dict(_DEFAULT_FAMILY_WEIGHTS)
    )

    signal: SignalConfig = Field(default_factory=SignalConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    severity_thresholds: SeverityThresholds = Field(default_factory=SeverityThresholds)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)

    insufficient_history_policy: Literal["report", "suppress"] = "report"

    # Excluded from the fingerprint by construction -- see fingerprint().
    output_dir: Path | None = None
    reports_dir: Path | None = None
    overwrite: bool = False

    @field_validator("output_dir", "reports_dir", mode="after")
    @classmethod
    def check_paths(cls, value: Path | None) -> Path | None:
        """Reject output paths containing parent-directory traversal."""
        return _check_safe_path(value, "output path")

    @field_validator("enabled_rule_ids", mode="after")
    @classmethod
    def check_enabled(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an empty, duplicated, or unknown enabled-rule set."""
        if not value:
            raise ValueError("enabled_rule_ids must name at least one rule")
        duplicates = sorted({rule_id for rule_id in value if value.count(rule_id) > 1})
        if duplicates:
            raise ValueError(f"enabled_rule_ids repeats rule(s) {duplicates}")
        unknown = sorted(rule_id for rule_id in value if not RULE_CATALOG.has(rule_id))
        if unknown:
            raise ValueError(f"enabled_rule_ids names unregistered rule(s) {unknown}")
        return value

    @field_validator("family_weights", mode="after")
    @classmethod
    def check_weights(cls, value: dict[RuleFamily, float]) -> dict[RuleFamily, float]:
        """Family weights must lie in ``(0, 1]``."""
        for family, weight in value.items():
            if not 0.0 < weight <= 1.0:
                raise ValueError(
                    f"family weight for {family} must be in (0, 1]; got {weight}"
                )
        return value

    @model_validator(mode="after")
    def check_cross_field_constraints(self) -> Self:
        """Validate schema compatibility, rule overrides, weights, and floors."""
        if self.required_feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"required_feature_schema_version "
                f"{self.required_feature_schema_version!r} is incompatible with "
                f"the feature schema version {FEATURE_SCHEMA_VERSION!r}"
            )
        if self.rule_catalog_version != RULE_CATALOG_VERSION:
            raise ValueError(
                f"rule_catalog_version {self.rule_catalog_version!r} is "
                f"incompatible with the registered catalog version "
                f"{RULE_CATALOG_VERSION!r}"
            )

        unknown = sorted(
            rule_id for rule_id in self.rules if not RULE_CATALOG.has(rule_id)
        )
        if unknown:
            raise ValueError(f"rules names unregistered rule(s) {unknown}")

        # Validate every override against the rule's own declared parameter
        # contract, so a bad threshold fails at load rather than at the first
        # snapshot that reaches it.
        for rule_id, settings in self.rules.items():
            spec = RULE_CATALOG.get(rule_id)
            declared = {parameter.name for parameter in spec.parameters}
            surplus = sorted(set(settings.parameters) - declared)
            if surplus:
                raise ValueError(
                    f"rule {rule_id} does not declare parameter(s) {surplus}"
                )
            for name, value in settings.parameters.items():
                spec.parameter(name).validate_value(value)

        missing_weights = sorted(
            str(family)
            for family in self._enabled_families()
            if family not in self.family_weights
        )
        if missing_weights:
            raise ValueError(
                f"family_weights does not cover enabled family/families "
                f"{missing_weights}"
            )

        if self.alerting.min_alert_risk_score < self.scoring.min_fired_risk_score:
            raise ValueError(
                "alerting.min_alert_risk_score must not be below "
                "scoring.min_fired_risk_score; an alert floor beneath the fired "
                "floor cannot suppress anything"
            )
        return self

    def _enabled_families(self) -> set[RuleFamily]:
        """Return the families of every enabled rule."""
        return {RULE_CATALOG.get(rule_id).family for rule_id in self.enabled_rule_ids}

    # -- accessors ----------------------------------------------------------

    def is_enabled(self, rule_id: str) -> bool:
        """Return whether *rule_id* is enabled for this run."""
        return rule_id in self.enabled_rule_ids

    def parameters_for(
        self, rule_id: str, *, catalog: RuleCatalog = RULE_CATALOG
    ) -> dict[str, Any]:
        """Return the effective parameters for *rule_id*.

        Declared defaults, overlaid with any validated override.

        Raises:
            DetectionConfigurationError: if *rule_id* is not registered.
        """
        spec = catalog.get(rule_id)
        effective: dict[str, Any] = dict(spec.default_parameters())
        settings = self.rules.get(rule_id)
        if settings is not None:
            for name, value in settings.parameters.items():
                effective[name] = spec.parameter(name).validate_value(value)
        return effective

    def severity_for_rule(
        self, rule_id: str, *, catalog: RuleCatalog = RULE_CATALOG
    ) -> Severity:
        """Return the configured default severity for *rule_id*."""
        settings = self.rules.get(rule_id)
        if settings is not None and settings.severity is not None:
            return settings.severity
        return catalog.get(rule_id).default_severity

    def weight_for(self, family: RuleFamily) -> float:
        """Return the configured weight for *family*.

        Raises:
            DetectionConfigurationError: if the family carries no weight.  This
                cannot happen for an enabled rule -- coverage is validated at
                load -- so it signals a caller using a disabled family.
        """
        try:
            return self.family_weights[family]
        except KeyError:
            raise DetectionConfigurationError(
                f"No configured weight for rule family {family!r}"
            ) from None

    @property
    def low_alert_reachable(self) -> bool:
        """Return whether this configuration can ever emit a ``LOW`` alert.

        ``False`` when either gate excludes the ``LOW`` band: a severity floor
        above ``LOW``, or an alert score floor at or above the ``medium``
        boundary.  Recorded in the quality report so the choice is visible.
        """
        if self.alerting.min_alert_severity is not Severity.LOW:
            return False
        return self.alerting.min_alert_risk_score < self.severity_thresholds.medium

    # -- fingerprint --------------------------------------------------------

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint.

        Every field of this model must appear here except those listed in
        :data:`DETECTION_FINGERPRINT_EXCLUDED_FIELDS`.  A test asserts that
        invariant, so a field added without a decision about its fingerprint
        status fails the build rather than silently weakening the digest.

        Per-rule parameters are recorded at their *effective* values, so the
        digest pins the thresholds that actually ran rather than only the
        overrides that were written down.
        """
        return {
            "detection_schema_version": self.detection_schema_version,
            "required_feature_schema_version": self.required_feature_schema_version,
            "rule_catalog_version": self.rule_catalog_version,
            "enabled_rule_ids": sorted(self.enabled_rule_ids),
            "rules": {
                rule_id: {
                    "severity": str(self.severity_for_rule(rule_id)),
                    "parameters": {
                        name: _fingerprint_scalar(value)
                        for name, value in sorted(self.parameters_for(rule_id).items())
                    },
                }
                for rule_id in sorted(self.enabled_rule_ids)
            },
            "family_weights": {
                str(family): weight
                for family, weight in sorted(
                    self.family_weights.items(), key=lambda item: str(item[0])
                )
            },
            "signal": self.signal.fingerprint_data(),
            "scoring": self.scoring.fingerprint_data(),
            "severity_thresholds": self.severity_thresholds.fingerprint_data(),
            "alerting": self.alerting.fingerprint_data(),
            "insufficient_history_policy": self.insufficient_history_policy,
        }

    def fingerprint(self) -> str:
        """Return a SHA-256 hex digest of the semantic detection configuration.

        Deliberately **excludes** ``output_dir``, ``reports_dir``,
        ``overwrite``, absolute paths, creation timestamps, and machine-specific
        values, so the same semantic configuration in different directories
        produces the same fingerprint.
        """
        canonical = json.dumps(self.fingerprint_data(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _fingerprint_scalar(value: object) -> Any:
    """Render a parameter value into a JSON-stable scalar."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def load_detection_config(path: Path) -> DetectionConfig:
    """Load and validate a :class:`DetectionConfig` from a YAML file.

    Uses ``yaml.safe_load``, which constructs only plain scalars, lists, and
    mappings.  No Python object, callable, or import path can be materialised
    from a configuration file.

    Raises:
        ConfigurationError: if the file is missing, unreadable, not a mapping,
            carries a secret-shaped key, or fails validation.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read detection configuration: {type(exc).__name__}"
        ) from None

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        raise ConfigurationError("Detection configuration is not valid YAML") from None

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigurationError("Detection configuration must be a YAML mapping")

    _reject_secret_keys(data)

    try:
        return DetectionConfig(**data)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Invalid detection configuration: {exc}") from None
