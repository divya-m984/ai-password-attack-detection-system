"""The versioned rule catalog -- the single source of truth for rule metadata.

Every rule is described by exactly one :class:`RuleSpec`.  The catalog drives:

* configuration validation (which rule identifiers and parameters exist);
* rule preparation (which feature columns a rule resolves to);
* the generated Markdown documentation;
* the catalog fingerprint recorded in every detection manifest.

**Rule logic is never loaded from configuration.**  A ``RuleSpec`` declares
*data*: identifiers, thresholds with bounds, feature templates, and evidence
definitions.  YAML supplies values for declared parameters and nothing else --
there is no expression, no callable reference, and no import path anywhere in
this contract.  The Python implementations that consume these specs are
registered statically in :mod:`password_attack_detector.detection.rules`.

Feature names are declared as *templates* such as
``"pair_failure_count__{window}"``.  Placeholders name declared parameters, so
changing a window in configuration moves the resolved column without editing
code.  Resolution is checked against a real :class:`FeatureCatalog` at
preparation time, which is where a window that a feature is not emitted at
becomes a loud failure rather than a silent null.

The catalog and the registry share this module deliberately.  The registry *is*
the catalog's index; splitting them would create two files that must always be
edited together.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from password_attack_detector.detection.enums import (
    PROHIBITED_CATEGORY_TERMS,
    AttackCategory,
    CorrelationGroup,
    EvaluationScope,
    EvidenceComparator,
    ParameterKind,
    PrivacyClass,
    RuleFamily,
    Severity,
)
from password_attack_detector.exceptions import (
    ConfigurationError,
    DetectionConfigurationError,
    RuleEvaluationError,
)
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
    LeakageClass,
)
from password_attack_detector.features.config import parse_duration

__all__ = [
    "RULE_CATALOG",
    "RULE_CATALOG_VERSION",
    "EvidenceDefinition",
    "MinHistorySpec",
    "RuleCatalog",
    "RuleParameter",
    "RuleSpec",
    "build_rule_catalog",
    "catalog_to_markdown",
    "resolve_rule_features",
    "validate_catalog_against_features",
]

#: Declared ``Final`` without an explicit annotation so its type narrows to
#: ``Literal["1.0.0"]``, letting the detection configuration default its
#: ``rule_catalog_version`` field from this constant.
RULE_CATALOG_VERSION: Final = "1.0.0"

_RULE_ID_RE = re.compile(r"^PAD-[A-Z]{2,4}-\d{3}$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PARAMETER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
_BRACE_RE = re.compile(r"[{}]")

#: Placeholders an evidence message template may use.  Anything else is a
#: rejected template: message rendering must never reach an attribute, an
#: index, or a caller-supplied name.
_ALLOWED_MESSAGE_PLACEHOLDERS: frozenset[str] = frozenset(
    {"observed", "threshold", "unit", "feature"}
)

#: Leakage classes a rule is permitted to read.  ``KEY`` is excluded: the
#: anchor identifier and timestamp are passed to the rule separately and are
#: never rule inputs.
_PERMITTED_LEAKAGE_CLASSES: frozenset[LeakageClass] = frozenset(
    {
        LeakageClass.PRIOR_ONLY,
        LeakageClass.CURRENT_EVENT_CONTEXT,
        LeakageClass.BASELINE_DERIVED,
    }
)

#: Columns a rule may never declare, on top of the Phase 3 prohibition set.
_PROHIBITED_RULE_FEATURES: frozenset[str] = PROHIBITED_FEATURE_COLUMNS | frozenset(
    {ANCHOR_EVENT_ID, ANCHOR_EVENT_TIME, "feature_schema_version"}
)

#: Wording that asserts proof or causality.  Rejected in message templates.
_PROHIBITED_TEMPLATE_TERMS: frozenset[str] = frozenset(
    {
        "proof",
        "proves",
        "proven",
        "confirm",
        "confirms",
        "confirmed",
        "guarantee",
        "guarantees",
        "guaranteed",
        "certainly",
        "definitely",
        "definitive",
        "conclusive",
        "probability",
        "likelihood",
    }
)


class RuleParameter(BaseModel):
    """One declared, typed threshold parameter of a rule.

    Bounds live here rather than in a validator so that configuration
    validation, the generated documentation, and the catalog fingerprint all
    read the same declaration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: ParameterKind
    default: bool | int | float | str
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str, ...] | None = None
    unit: str | None = None
    description: str

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        """Parameter names are lower snake case, matching the YAML keys."""
        if not _PARAMETER_NAME_RE.match(value):
            raise ValueError(
                f"parameter name {value!r} must match {_PARAMETER_NAME_RE.pattern!r}"
            )
        return value

    @model_validator(mode="after")
    def check_declaration(self) -> Self:
        """Validate the bounds, the choice set, and the declared default."""
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"parameter {self.name!r} has minimum above maximum")
        if self.allowed_values is not None:
            if self.kind is not ParameterKind.STRING:
                raise ValueError(
                    f"parameter {self.name!r} declares allowed_values but is not "
                    f"a string parameter"
                )
            if not self.allowed_values:
                raise ValueError(f"parameter {self.name!r} has an empty choice set")
        # The default must survive the rule's own validation, so a bad default
        # fails at import rather than at the first run that omits an override.
        self.validate_value(self.default)
        return self

    def validate_value(self, value: object) -> bool | int | float | str:
        """Return *value* coerced to this parameter's declared type.

        Raises:
            ValueError: if the value has the wrong type, falls outside the
                declared bounds, or is not one of the declared choices.
        """
        match self.kind:
            case ParameterKind.BOOL:
                if not isinstance(value, bool):
                    raise ValueError(f"parameter {self.name!r} must be a boolean")
                return value
            case ParameterKind.INT:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"parameter {self.name!r} must be an integer")
                return self._check_bounds(float(value), value)
            case ParameterKind.FLOAT:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError(f"parameter {self.name!r} must be a number")
                coerced = float(value)
                if coerced != coerced or coerced in (float("inf"), float("-inf")):
                    raise ValueError(f"parameter {self.name!r} must be finite")
                return self._check_bounds(coerced, coerced)
            case ParameterKind.STRING:
                if not isinstance(value, str):
                    raise ValueError(f"parameter {self.name!r} must be a string")
                if self.allowed_values is not None and value not in self.allowed_values:
                    raise ValueError(
                        f"parameter {self.name!r} must be one of "
                        f"{list(self.allowed_values)}"
                    )
                return value
            case ParameterKind.WINDOW:
                if not isinstance(value, str):
                    raise ValueError(
                        f"parameter {self.name!r} must be a window label such as '5m'"
                    )
                # Reuses the Phase 3 duration parser so one grammar governs
                # every duration in the project.  It raises ConfigurationError,
                # which pydantic does not wrap, so it is re-raised as the
                # ValueError this method documents.
                try:
                    parse_duration(value)
                except ConfigurationError as exc:
                    raise ValueError(f"parameter {self.name!r}: {exc}") from None
                return value

    def _check_bounds(
        self, numeric: float, value: int | float
    ) -> bool | int | float | str:
        """Return *value* when *numeric* lies inside the declared bounds."""
        if self.minimum is not None and numeric < self.minimum:
            raise ValueError(f"parameter {self.name!r} must be at least {self.minimum}")
        if self.maximum is not None and numeric > self.maximum:
            raise ValueError(f"parameter {self.name!r} must be at most {self.maximum}")
        return value


class MinHistorySpec(BaseModel):
    """What history a rule needs before it can reach a verdict.

    When one of ``required_non_null_features`` is null on a snapshot, the rule
    returns ``insufficient_data``.  It never reports a clean negative for
    history it could not see.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_non_null_features: tuple[str, ...] = ()
    description: str = ""


class EvidenceDefinition(BaseModel):
    """A declared evidence item a rule may emit.

    The message template lives here so that code and documentation cannot
    drift, and so that no caller-supplied string can ever reach a rendered
    message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_code: str
    feature_template: str | None
    comparator: EvidenceComparator
    unit: str | None = None
    message_template: str
    description: str

    @field_validator("evidence_code")
    @classmethod
    def check_code(cls, value: str) -> str:
        """Evidence codes are upper-case machine-readable constants."""
        if not _CODE_RE.match(value):
            raise ValueError(f"evidence_code {value!r} must match {_CODE_RE.pattern!r}")
        return value

    @field_validator("message_template")
    @classmethod
    def check_template(cls, value: str) -> str:
        """Restrict the template to a fixed placeholder set and safe wording.

        Rendering uses ``str.format``, which can reach attributes and indices
        when a template is untrusted.  Restricting the grammar to a handful of
        bare names keeps that door closed even though every template in this
        module is a constant.
        """
        placeholders = set(_PLACEHOLDER_RE.findall(value))
        unknown = sorted(placeholders - _ALLOWED_MESSAGE_PLACEHOLDERS)
        if unknown:
            raise ValueError(
                f"message template uses unsupported placeholder(s) {unknown}; "
                f"allowed: {sorted(_ALLOWED_MESSAGE_PLACEHOLDERS)}"
            )
        if len(_BRACE_RE.findall(value)) != 2 * len(_PLACEHOLDER_RE.findall(value)):
            raise ValueError("message template contains an unbalanced or complex brace")

        tokens = {token.strip(".,;:!?()[]'\"") for token in value.lower().split()}
        offending = sorted(tokens & _PROHIBITED_TEMPLATE_TERMS)
        if offending:
            raise ValueError(
                f"message template uses claim-asserting term(s) {offending}; "
                f"describe what was observed and what it is consistent with"
            )
        return value


class RuleSpec(BaseModel):
    """Complete, machine-readable description of one detection rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    rule_version: str
    display_name: str
    family: RuleFamily
    attack_category: AttackCategory
    description: str
    default_severity: Severity
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...] = ()
    correlation_group: CorrelationGroup
    evaluation_scope: EvaluationScope = EvaluationScope.ANCHOR_EVENT
    parameters: tuple[RuleParameter, ...] = ()
    min_history: MinHistorySpec = MinHistorySpec()
    evidence: tuple[EvidenceDefinition, ...] = ()
    privacy_class: Literal[PrivacyClass.NON_SENSITIVE] = PrivacyClass.NON_SENSITIVE
    limitations: tuple[str, ...] = ()
    deprecated: bool = False

    @field_validator("rule_id")
    @classmethod
    def check_rule_id(cls, value: str) -> str:
        """Rule identifiers are stable once published."""
        if not _RULE_ID_RE.match(value):
            raise ValueError(f"rule_id {value!r} must match {_RULE_ID_RE.pattern!r}")
        return value

    @field_validator("rule_version")
    @classmethod
    def check_rule_version(cls, value: str) -> str:
        """Rule versions are semantic versions."""
        if not _SEMVER_RE.match(value):
            raise ValueError(f"rule_version {value!r} must be a semantic version")
        return value

    @field_validator("attack_category")
    @classmethod
    def check_category(cls, value: AttackCategory) -> AttackCategory:
        """Reject a category this project has decided not to produce."""
        if str(value) in PROHIBITED_CATEGORY_TERMS:
            raise ValueError(f"attack category {value!r} is not a rule-based category")
        return value

    @model_validator(mode="after")
    def check_declaration(self) -> Self:
        """Validate parameters, feature templates, evidence, and history."""
        if not self.required_features:
            raise ValueError(f"rule {self.rule_id} declares no required feature")

        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"rule {self.rule_id} repeats a parameter name")
        declared = set(names)

        overlap = sorted(set(self.required_features) & set(self.optional_features))
        if overlap:
            raise ValueError(
                f"rule {self.rule_id} declares {overlap} as both required and optional"
            )

        for template in (*self.required_features, *self.optional_features):
            self._check_feature_template(template, declared)

        missing_history = sorted(
            set(self.min_history.required_non_null_features)
            - set(self.required_features)
            - set(self.optional_features)
        )
        if missing_history:
            raise ValueError(
                f"rule {self.rule_id} requires history for undeclared feature(s) "
                f"{missing_history}"
            )

        codes = [item.evidence_code for item in self.evidence]
        if len(set(codes)) != len(codes):
            raise ValueError(f"rule {self.rule_id} repeats an evidence code")
        if not self.evidence:
            raise ValueError(f"rule {self.rule_id} declares no evidence")

        for item in self.evidence:
            if item.feature_template is None:
                continue
            if item.feature_template not in (
                *self.required_features,
                *self.optional_features,
            ):
                raise ValueError(
                    f"rule {self.rule_id} defines evidence {item.evidence_code} for "
                    f"undeclared feature template {item.feature_template!r}"
                )
        return self

    def _check_feature_template(self, template: str, declared: set[str]) -> None:
        """Reject an unresolvable, prohibited, or malformed feature template."""
        placeholders = set(_PLACEHOLDER_RE.findall(template))
        unknown = sorted(placeholders - declared)
        if unknown:
            raise ValueError(
                f"rule {self.rule_id} feature template {template!r} references "
                f"undeclared parameter(s) {unknown}"
            )
        if len(_BRACE_RE.findall(template)) != 2 * len(placeholders):
            raise ValueError(
                f"rule {self.rule_id} feature template {template!r} is malformed"
            )
        base = template.split("__", maxsplit=1)[0]
        if template in _PROHIBITED_RULE_FEATURES or base in _PROHIBITED_RULE_FEATURES:
            raise ValueError(
                f"rule {self.rule_id} declares prohibited feature {template!r}"
            )

    def parameter(self, name: str) -> RuleParameter:
        """Return the declared parameter named *name*.

        Raises:
            DetectionConfigurationError: if the rule declares no such parameter.
        """
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise DetectionConfigurationError(
            f"rule {self.rule_id} declares no parameter {name!r}"
        )

    def default_parameters(self) -> dict[str, bool | int | float | str]:
        """Return every declared parameter at its default value."""
        return {parameter.name: parameter.default for parameter in self.parameters}

    def evidence_definition(self, evidence_code: str) -> EvidenceDefinition:
        """Return the evidence definition for *evidence_code*.

        Raises:
            RuleEvaluationError: if the rule declares no such evidence.
        """
        for item in self.evidence:
            if item.evidence_code == evidence_code:
                return item
        raise RuleEvaluationError(
            f"rule {self.rule_id} declares no evidence {evidence_code!r}"
        )


#: Fields that participate in the catalog fingerprint.  ``description``,
#: ``limitations``, and every ``description`` nested inside a parameter or
#: evidence definition are excluded on purpose: correcting prose must not
#: invalidate every artifact that records this fingerprint.
_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "rule_id",
    "rule_version",
    "display_name",
    "family",
    "attack_category",
    "default_severity",
    "required_features",
    "optional_features",
    "correlation_group",
    "evaluation_scope",
    "privacy_class",
    "deprecated",
)


class RuleCatalog:
    """An immutable, ordered collection of :class:`RuleSpec`.

    Rules are held sorted by ``rule_id`` so iteration order is a property of
    the data rather than of import order.
    """

    __slots__ = ("_by_id", "_specs")

    def __init__(self, specs: tuple[RuleSpec, ...]) -> None:
        by_id: dict[str, RuleSpec] = {}
        for spec in specs:
            if spec.rule_id in by_id:
                raise DetectionConfigurationError(
                    f"Duplicate rule identifier in catalog: {spec.rule_id!r}"
                )
            by_id[spec.rule_id] = spec
        self._specs = tuple(sorted(specs, key=lambda s: s.rule_id))
        self._by_id = by_id

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Any:
        return iter(self._specs)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._by_id

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        """Return every rule specification, ordered by rule identifier."""
        return self._specs

    @property
    def rule_ids(self) -> tuple[str, ...]:
        """Return every rule identifier, in catalog order."""
        return tuple(spec.rule_id for spec in self._specs)

    def get(self, rule_id: str) -> RuleSpec:
        """Return the specification for *rule_id*.

        Raises:
            DetectionConfigurationError: if no such rule is registered.
        """
        try:
            return self._by_id[rule_id]
        except KeyError:
            raise DetectionConfigurationError(
                f"Unknown rule identifier: {rule_id!r}"
            ) from None

    def has(self, rule_id: str) -> bool:
        """Return whether *rule_id* is registered."""
        return rule_id in self._by_id

    def specs_for_family(self, family: RuleFamily) -> tuple[RuleSpec, ...]:
        """Return every rule in *family*, in catalog order."""
        return tuple(spec for spec in self._specs if spec.family is family)

    def specs_for_correlation_group(
        self, group: CorrelationGroup
    ) -> tuple[RuleSpec, ...]:
        """Return every rule in *group*, in catalog order."""
        return tuple(spec for spec in self._specs if spec.correlation_group is group)

    def families(self) -> tuple[RuleFamily, ...]:
        """Return every family represented in the catalog, sorted."""
        return tuple(sorted({spec.family for spec in self._specs}))

    def family_counts(self) -> dict[str, int]:
        """Return the number of rules per family, for reporting."""
        counts: dict[str, int] = {}
        for spec in self._specs:
            counts[str(spec.family)] = counts.get(str(spec.family), 0) + 1
        return counts

    def fingerprint(self) -> str:
        """Return a SHA-256 digest of the catalog's machine-readable contract.

        Covers identifiers, versions, families, categories, severities, feature
        templates, correlation groups, parameter declarations, evidence codes,
        and minimum-history requirements.  Excludes free prose, so fixing a typo
        in a description does not invalidate every artifact recording this
        fingerprint.
        """
        payload = {
            "rule_catalog_version": RULE_CATALOG_VERSION,
            "rules": [
                {
                    **{
                        field: _json_scalar(getattr(spec, field))
                        for field in _FINGERPRINT_FIELDS
                    },
                    "parameters": [
                        {
                            "name": parameter.name,
                            "kind": str(parameter.kind),
                            "default": _json_scalar(parameter.default),
                            "minimum": _json_scalar(parameter.minimum),
                            "maximum": _json_scalar(parameter.maximum),
                            "allowed_values": _json_scalar(parameter.allowed_values),
                            "unit": parameter.unit,
                        }
                        for parameter in spec.parameters
                    ],
                    "evidence": [
                        {
                            "evidence_code": item.evidence_code,
                            "feature_template": item.feature_template,
                            "comparator": str(item.comparator),
                            "unit": item.unit,
                        }
                        for item in spec.evidence
                    ],
                    "min_history": list(spec.min_history.required_non_null_features),
                }
                for spec in self._specs
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _json_scalar(value: object) -> Any:
    """Render a spec field into a JSON-stable scalar."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, tuple | list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def build_rule_catalog(specs: tuple[RuleSpec, ...]) -> RuleCatalog:
    """Build and self-validate a rule catalog from *specs*.

    Raises:
        DetectionConfigurationError: on a duplicate identifier, a duplicate
            display name, or an empty catalog.
    """
    if not specs:
        raise DetectionConfigurationError("The rule catalog must declare a rule")

    names = [spec.display_name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DetectionConfigurationError(
            f"Duplicate rule display name(s) in catalog: {duplicates}"
        )
    return RuleCatalog(specs)


def resolve_rule_features(
    spec: RuleSpec,
    parameters: dict[str, Any],
    feature_catalog: FeatureCatalog,
    *,
    include_optional: bool = True,
) -> dict[str, str]:
    """Resolve a rule's feature templates into concrete column names.

    Returns a mapping from template to resolved column name.  This is the one
    place where a rule's declared inputs meet the feature schema, so it is
    where every boundary is checked: existence, the Phase 3 prohibition set,
    and the leakage class.

    Raises:
        RuleEvaluationError: if a template references a column the feature
            catalog does not declare, a prohibited column, or a column whose
            leakage class a rule may not read.
    """
    templates = list(spec.required_features)
    if include_optional:
        templates.extend(spec.optional_features)

    resolved: dict[str, str] = {}
    for template in templates:
        try:
            name = template.format(**parameters)
        except KeyError as exc:
            raise RuleEvaluationError(
                f"rule {spec.rule_id} template {template!r} needs parameter {exc}"
            ) from None

        if name in _PROHIBITED_RULE_FEATURES:
            raise RuleEvaluationError(
                f"rule {spec.rule_id} resolves {template!r} to prohibited column "
                f"{name!r}"
            )
        if not feature_catalog.has(name):
            raise RuleEvaluationError(
                f"rule {spec.rule_id} requires feature {name!r}, which the "
                f"configured feature catalog does not declare"
            )
        feature = feature_catalog.get(name)
        if feature.leakage_class not in _PERMITTED_LEAKAGE_CLASSES:
            raise RuleEvaluationError(
                f"rule {spec.rule_id} may not read {name!r}: leakage class "
                f"{feature.leakage_class} is not a permitted rule input"
            )
        resolved[template] = name
    return resolved


def validate_catalog_against_features(
    catalog: RuleCatalog,
    feature_catalog: FeatureCatalog,
    *,
    parameters_by_rule: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Assert every rule in *catalog* resolves against *feature_catalog*.

    Uses each rule's declared defaults unless *parameters_by_rule* supplies an
    effective parameter mapping.

    Raises:
        RuleEvaluationError: on the first rule that cannot be resolved.
    """
    for spec in catalog.specs:
        parameters: dict[str, Any] = dict(spec.default_parameters())
        if parameters_by_rule is not None and spec.rule_id in parameters_by_rule:
            parameters = dict(parameters_by_rule[spec.rule_id])
        resolve_rule_features(spec, parameters, feature_catalog)


# ---------------------------------------------------------------------------
# Declaration helpers
#
# These keep the nine specifications below readable.  They add no behaviour.
# ---------------------------------------------------------------------------


def _p_int(
    name: str,
    default: int,
    description: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    unit: str = "count",
) -> RuleParameter:
    return RuleParameter(
        name=name,
        kind=ParameterKind.INT,
        default=default,
        minimum=minimum,
        maximum=maximum,
        unit=unit,
        description=description,
    )


def _p_float(
    name: str,
    default: float,
    description: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    unit: str = "ratio",
) -> RuleParameter:
    return RuleParameter(
        name=name,
        kind=ParameterKind.FLOAT,
        default=default,
        minimum=minimum,
        maximum=maximum,
        unit=unit,
        description=description,
    )


def _p_window(name: str, default: str, description: str) -> RuleParameter:
    return RuleParameter(
        name=name,
        kind=ParameterKind.WINDOW,
        default=default,
        unit="duration",
        description=description,
    )


def _p_bool(name: str, default: bool, description: str) -> RuleParameter:
    return RuleParameter(
        name=name,
        kind=ParameterKind.BOOL,
        default=default,
        description=description,
    )


def _p_choice(
    name: str, default: str, allowed: tuple[str, ...], description: str
) -> RuleParameter:
    return RuleParameter(
        name=name,
        kind=ParameterKind.STRING,
        default=default,
        allowed_values=allowed,
        description=description,
    )


def _e(
    code: str,
    feature: str | None,
    comparator: EvidenceComparator,
    message: str,
    description: str,
    *,
    unit: str | None = None,
) -> EvidenceDefinition:
    return EvidenceDefinition(
        evidence_code=code,
        feature_template=feature,
        comparator=comparator,
        unit=unit,
        message_template=message,
        description=description,
    )


_GTE = EvidenceComparator.GTE
_LTE = EvidenceComparator.LTE
_EQ = EvidenceComparator.EQ
_IN = EvidenceComparator.IN
_IS_TRUE = EvidenceComparator.IS_TRUE


# ---------------------------------------------------------------------------
# The nine registered rules
# ---------------------------------------------------------------------------

_PAD_BF_001 = RuleSpec(
    rule_id="PAD-BF-001",
    rule_version="1.0.0",
    display_name="Concentrated brute-force indicator",
    family=RuleFamily.BRUTE_FORCE,
    attack_category=AttackCategory.BRUTE_FORCE,
    description=(
        "Repeated failed authentication concentrated on one account from one "
        "source, at a cadence and failure share consistent with automated "
        "credential guessing. The source-fan-out ceiling keeps a password-"
        "spraying source out of this rule."
    ),
    default_severity=Severity.HIGH,
    correlation_group=CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
    required_features=(
        "pair_failure_count__{window}",
        "pair_failure_rate__{window}",
        "user_failure_count__{window}",
        "prior_consecutive_user_failures",
        "source_unique_user_count__{cardinality_window}",
    ),
    optional_features=(
        "pair_mean_interarrival_seconds__{window}",
        "user_blocked_count__{window}",
    ),
    parameters=(
        _p_window("window", "5m", "Window for the pair and user failure counts."),
        _p_window(
            "cardinality_window",
            "5m",
            "Window for the source's distinct-user count; must be a window at "
            "which cardinality features are emitted.",
        ),
        _p_int(
            "min_pair_failures", 8, "Failures for this user-source pair.", minimum=1
        ),
        _p_int("min_user_failures", 8, "Failures for this user.", minimum=1),
        _p_float(
            "min_pair_failure_rate",
            0.80,
            "Share of this pair's attempts that failed.",
            maximum=1.0,
        ),
        _p_int(
            "min_consecutive_failures",
            5,
            "Length of the unbroken failure run immediately before the anchor.",
            minimum=1,
        ),
        _p_int(
            "max_source_unique_users",
            3,
            "Ceiling on the source's distinct-user count; above this the "
            "behaviour is fan-out, not concentration.",
            minimum=1,
        ),
        _p_float(
            "max_mean_interarrival_seconds",
            60.0,
            "Cadence ceiling for the supporting timing component.",
            minimum=0.0,
            maximum=86400.0,
            unit="seconds",
        ),
        _p_int(
            "blocked_support_threshold",
            2,
            "Blocked-authentication count that adds supporting evidence. This "
            "is the only use of blocked-account activity; there is no "
            "standalone blocked-account rule.",
            minimum=1,
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=("pair_failure_rate__{window}",),
        description=(
            "The pair failure rate is null when the window holds no attempts "
            "for this user-source pair, which is unseen history rather than a "
            "clean negative."
        ),
    ),
    evidence=(
        _e(
            "BF_PAIR_FAILURE_COUNT",
            "pair_failure_count__{window}",
            _GTE,
            "Observed {observed} failed attempts for this user and source, at "
            "or above the configured threshold of {threshold}; this "
            "contributed to this detection.",
            "Failure volume concentrated on one user-source pair.",
            unit="count",
        ),
        _e(
            "BF_USER_FAILURE_COUNT",
            "user_failure_count__{window}",
            _GTE,
            "Observed {observed} failed attempts for this account, at or above "
            "the configured threshold of {threshold}; this contributed to this "
            "detection.",
            "Failure volume against the targeted account.",
            unit="count",
        ),
        _e(
            "BF_PAIR_FAILURE_RATE",
            "pair_failure_rate__{window}",
            _GTE,
            "A failure share of {observed} matched the configured rule "
            "condition of at least {threshold}.",
            "Share of the pair's attempts that failed.",
            unit="ratio",
        ),
        _e(
            "BF_CONSECUTIVE_USER_FAILURES",
            "prior_consecutive_user_failures",
            _GTE,
            "An unbroken run of {observed} prior failures reached the "
            "configured threshold of {threshold}; this is consistent with "
            "repeated credential guessing.",
            "Length of the immediately preceding failure run.",
            unit="count",
        ),
        _e(
            "BF_SOURCE_TARGET_CONCENTRATION",
            "source_unique_user_count__{cardinality_window}",
            _LTE,
            "The source touched {observed} distinct accounts, at or below the "
            "configured ceiling of {threshold}, so the activity is "
            "concentrated rather than fanned out.",
            "Concentration guard separating this rule from password spraying.",
            unit="count",
        ),
        _e(
            "BF_PAIR_INTERARRIVAL",
            "pair_mean_interarrival_seconds__{window}",
            _LTE,
            "A mean gap of {observed} seconds between this pair's attempts is "
            "consistent with automated retries.",
            "Supporting cadence evidence; contributes signal, never required.",
            unit="seconds",
        ),
        _e(
            "BF_BLOCKED_ACTIVITY",
            "user_blocked_count__{window}",
            _GTE,
            "This account accumulated {observed} blocked authentications, at "
            "or above {threshold}; this may indicate lockout pressure.",
            "Supporting blocked-account evidence; contributes signal, never required.",
            unit="count",
        ),
    ),
    limitations=(
        "A shared egress address used by many colleagues can concentrate "
        "unrelated failures onto one apparent source.",
        "A misconfigured client retrying a stale credential reproduces this "
        "shape without an attacker.",
    ),
)

_PAD_BF_002 = RuleSpec(
    rule_id="PAD-BF-002",
    rule_version="1.0.0",
    display_name="Successful authentication after failure burst",
    family=RuleFamily.BRUTE_FORCE,
    attack_category=AttackCategory.BRUTE_FORCE,
    description=(
        "A successful authentication immediately following a sustained run of "
        "failures for the same account or user-source pair. Apart from the "
        "current outcome, every condition reads prior-history sequence "
        "features only. Shares a correlation group with the other brute-force "
        "rules so one failure burst is never counted twice."
    ),
    default_severity=Severity.HIGH,
    correlation_group=CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
    required_features=(
        "current_authentication_outcome",
        "prior_failures_since_pair_success",
        "prior_failures_since_user_success",
        "previous_user_outcome",
        "seconds_since_user_previous_failure",
    ),
    parameters=(
        _p_int(
            "min_pair_failures_since_success",
            6,
            "Failures for this user-source pair since the pair last succeeded.",
            minimum=1,
        ),
        _p_int(
            "min_user_failures_since_success",
            8,
            "Failures for this account since it last succeeded.",
            minimum=1,
        ),
        _p_float(
            "max_seconds_since_previous_failure",
            300.0,
            "How recently the preceding failure must have occurred.",
            minimum=0.0,
            maximum=86400.0,
            unit="seconds",
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=(
            "previous_user_outcome",
            "seconds_since_user_previous_failure",
        ),
        description=(
            "Both are null for an account with no prior event or no prior "
            "failure, which is unseen history rather than a clean negative."
        ),
    ),
    evidence=(
        _e(
            "BF2_CURRENT_SUCCESS",
            "current_authentication_outcome",
            _EQ,
            "The anchor authentication succeeded, which matched the configured "
            "rule condition.",
            "The success condition that separates this rule from PAD-BF-001.",
        ),
        _e(
            "BF2_PAIR_FAILURE_BURST",
            "prior_failures_since_pair_success",
            _GTE,
            "This user and source accumulated {observed} failures since their "
            "last success, at or above the configured threshold of {threshold}.",
            "Failure burst for the user-source pair.",
            unit="count",
        ),
        _e(
            "BF2_USER_FAILURE_BURST",
            "prior_failures_since_user_success",
            _GTE,
            "This account accumulated {observed} failures since its last "
            "success, at or above the configured threshold of {threshold}.",
            "Failure burst for the account.",
            unit="count",
        ),
        _e(
            "BF2_PREVIOUS_OUTCOME",
            "previous_user_outcome",
            _IN,
            "The preceding event for this account was recorded as {observed}, "
            "which matched the configured rule condition.",
            "The immediately preceding outcome.",
        ),
        _e(
            "BF2_FAILURE_RECENCY",
            "seconds_since_user_previous_failure",
            _LTE,
            "The preceding failure was {observed} seconds earlier, within the "
            "configured window of {threshold}; this is consistent with a "
            "guessing run ending in a success.",
            "Recency of the preceding failure.",
            unit="seconds",
        ),
    ),
    limitations=(
        "A user who forgets a password, fails repeatedly, resets it, and then "
        "signs in successfully reproduces this shape exactly.",
        "The rule reports a sequence, not an assessment of who supplied the "
        "successful credential.",
    ),
)

_PAD_PS_001 = RuleSpec(
    rule_id="PAD-PS-001",
    rule_version="1.0.0",
    display_name="Password-spraying indicator",
    family=RuleFamily.SPRAYING,
    attack_category=AttackCategory.PASSWORD_SPRAYING,
    description=(
        "One source attempting a small number of authentications against many "
        "distinct accounts, with a high failure share. The low attempts-per-"
        "account ceiling is what separates spraying from single-account brute "
        "force."
    ),
    default_severity=Severity.HIGH,
    correlation_group=CorrelationGroup.SOURCE_FANOUT,
    required_features=(
        "source_unique_user_count__{cardinality_window}",
        "source_failure_count__{window}",
        "source_failure_rate__{window}",
        "source_attempt_count__{window}",
    ),
    optional_features=("source_mean_interarrival_seconds__{window}",),
    parameters=(
        _p_window("window", "1h", "Window for the source's counts and rate."),
        _p_window(
            "cardinality_window",
            "1h",
            "Window for the source's distinct-user count.",
        ),
        _p_int(
            "min_unique_users",
            15,
            "Distinct accounts the source touched.",
            minimum=2,
        ),
        _p_int("min_source_failures", 15, "Failures from this source.", minimum=1),
        _p_float(
            "min_source_failure_rate",
            0.70,
            "Share of this source's attempts that failed.",
            maximum=1.0,
        ),
        _p_float(
            "max_attempts_per_user",
            3.0,
            "Ceiling on attempts per targeted account; the low-and-slow guard.",
            minimum=1.0,
            unit="ratio",
        ),
        _p_float(
            "min_mean_interarrival_seconds",
            0.0,
            "Optional cadence floor; zero disables the component.",
            minimum=0.0,
            maximum=86400.0,
            unit="seconds",
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=("source_failure_rate__{window}",),
        description=(
            "The source failure rate is null when the window holds no attempts "
            "from this source."
        ),
    ),
    evidence=(
        _e(
            "PS_SOURCE_USER_FANOUT",
            "source_unique_user_count__{cardinality_window}",
            _GTE,
            "This source touched {observed} distinct accounts, at or above the "
            "configured threshold of {threshold}; this contributed to this "
            "detection.",
            "Account fan-out from a single source.",
            unit="count",
        ),
        _e(
            "PS_SOURCE_FAILURE_COUNT",
            "source_failure_count__{window}",
            _GTE,
            "This source produced {observed} failures, at or above the "
            "configured threshold of {threshold}.",
            "Failure volume from the source.",
            unit="count",
        ),
        _e(
            "PS_SOURCE_FAILURE_RATE",
            "source_failure_rate__{window}",
            _GTE,
            "A failure share of {observed} matched the configured rule "
            "condition of at least {threshold}.",
            "Share of the source's attempts that failed.",
            unit="ratio",
        ),
        _e(
            "PS_ATTEMPTS_PER_USER",
            "source_attempt_count__{window}",
            _LTE,
            "The source averaged {observed} attempts per targeted account, at "
            "or below the configured ceiling of {threshold}; this is "
            "consistent with spreading a few guesses across many accounts.",
            "Derived attempts-per-account ratio; the spraying discriminator.",
            unit="ratio",
        ),
        _e(
            "PS_SOURCE_CADENCE",
            "source_mean_interarrival_seconds__{window}",
            _GTE,
            "A mean gap of {observed} seconds between attempts is consistent "
            "with a deliberately slow cadence.",
            "Optional cadence evidence; contributes signal, never required.",
            unit="seconds",
        ),
    ),
    limitations=(
        "A shared corporate egress address legitimately serves many accounts "
        "and can exceed the fan-out threshold during an outage.",
        "A directory-service misconfiguration can fail many accounts at once "
        "without any attacker.",
    ),
)

_PAD_CS_001 = RuleSpec(
    rule_id="PAD-CS-001",
    rule_version="1.0.0",
    display_name="Credential-stuffing indicator",
    family=RuleFamily.STUFFING,
    attack_category=AttackCategory.CREDENTIAL_STUFFING,
    description=(
        "Broad account fan-out from one source with a mix of failures and "
        "occasional successes, varied client characteristics, and an "
        "unfamiliar device or country for the account. No credential value, "
        "password-reuse data, or credential list is read anywhere."
    ),
    default_severity=Severity.HIGH,
    correlation_group=CorrelationGroup.SOURCE_FANOUT,
    required_features=(
        "source_unique_user_count__{cardinality_window}",
        "source_success_count__{window}",
        "source_failure_count__{window}",
        "source_attempt_count__{window}",
        "source_unique_device_count__{cardinality_window}",
        "source_unique_user_agent_count__{cardinality_window}",
        "user_in_baseline",
        "is_new_device_for_user",
        "is_new_country_for_user",
    ),
    parameters=(
        _p_window("window", "1h", "Window for the source's outcome counts."),
        _p_window(
            "cardinality_window", "1h", "Window for the source's distinct counts."
        ),
        _p_int("min_unique_users", 10, "Distinct accounts touched.", minimum=2),
        _p_int(
            "min_successes",
            1,
            "Successes from this source; the mixed-outcome signature.",
            minimum=1,
        ),
        _p_int("min_failures", 10, "Failures from this source.", minimum=1),
        _p_int("min_unique_devices", 3, "Distinct devices seen.", minimum=1),
        _p_int("min_unique_user_agents", 3, "Distinct user-agent families.", minimum=1),
        _p_float(
            "max_attempts_per_user",
            4.0,
            "Ceiling on attempts per targeted account.",
            minimum=1.0,
            unit="ratio",
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=("user_in_baseline",),
        description=(
            "The unfamiliar-context condition needs a fitted baseline for the "
            "account. An account absent from the baseline yields "
            "insufficient_data, never a clean negative."
        ),
    ),
    evidence=(
        _e(
            "CS_SOURCE_USER_FANOUT",
            "source_unique_user_count__{cardinality_window}",
            _GTE,
            "This source touched {observed} distinct accounts, at or above the "
            "configured threshold of {threshold}.",
            "Account fan-out from a single source.",
            unit="count",
        ),
        _e(
            "CS_MIXED_OUTCOMES",
            "source_success_count__{window}",
            _GTE,
            "The source recorded {observed} successes alongside its failures, "
            "which matched the configured rule condition of at least "
            "{threshold}.",
            "The mixed-outcome signature separating stuffing from spraying.",
            unit="count",
        ),
        _e(
            "CS_CLIENT_DIVERSITY",
            "source_unique_device_count__{cardinality_window}",
            _GTE,
            "The source presented {observed} distinct client characteristics, "
            "at or above the configured threshold of {threshold}.",
            "Device and user-agent diversity from one source.",
            unit="count",
        ),
        _e(
            "CS_ATTEMPTS_PER_USER",
            "source_attempt_count__{window}",
            _LTE,
            "The source averaged {observed} attempts per targeted account, at "
            "or below the configured ceiling of {threshold}.",
            "Derived attempts-per-account ratio.",
            unit="ratio",
        ),
        _e(
            "CS_UNFAMILIAR_CONTEXT",
            "is_new_device_for_user",
            _IS_TRUE,
            "The anchor used a device or country not previously seen for this "
            "account, which is consistent with access from a reused "
            "credential.",
            "Unfamiliar-context requirement; needs a fitted baseline.",
        ),
    ),
    limitations=(
        "A NAT gateway or corporate proxy produces broad fan-out and varied "
        "client characteristics from a single apparent source.",
        "The rule observes behaviour only; it has no visibility into whether "
        "any credential was reused from another service.",
    ),
)

_PAD_DBF_001 = RuleSpec(
    rule_id="PAD-DBF-001",
    rule_version="1.0.0",
    display_name="Distributed brute-force indicator",
    family=RuleFamily.BRUTE_FORCE,
    attack_category=AttackCategory.DISTRIBUTED_BRUTE_FORCE,
    description=(
        "One account failing repeatedly across many distinct sources, each "
        "contributing few attempts. The per-source volume ceiling is what "
        "separates this from a single high-volume source, and the source "
        "fan-out ceiling keeps a spraying source out."
    ),
    default_severity=Severity.CRITICAL,
    correlation_group=CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
    required_features=(
        "user_unique_source_count__{cardinality_window}",
        "user_failure_count__{window}",
        "user_failure_rate__{window}",
        "pair_attempt_count__{window}",
        "source_unique_user_count__{cardinality_window}",
    ),
    parameters=(
        _p_window("window", "1h", "Window for the account's failure counts."),
        _p_window("cardinality_window", "1h", "Window for the distinct counts."),
        _p_int(
            "min_unique_sources",
            8,
            "Distinct sources targeting this account.",
            minimum=2,
        ),
        _p_int("min_user_failures", 20, "Failures against this account.", minimum=1),
        _p_float(
            "min_user_failure_rate",
            0.80,
            "Share of this account's attempts that failed.",
            maximum=1.0,
        ),
        _p_int(
            "max_pair_attempts",
            5,
            "Ceiling on attempts from any one source; the distribution guard.",
            minimum=1,
        ),
        _p_int(
            "max_source_unique_users",
            3,
            "Ceiling on the anchor source's distinct-user count, so a spraying "
            "source is not reported as distributed brute force.",
            minimum=1,
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=("user_failure_rate__{window}",),
        description=(
            "The account failure rate is null when the window holds no "
            "attempts for this account."
        ),
    ),
    evidence=(
        _e(
            "DBF_USER_SOURCE_FANOUT",
            "user_unique_source_count__{cardinality_window}",
            _GTE,
            "This account was reached from {observed} distinct sources, at or "
            "above the configured threshold of {threshold}; this contributed "
            "to this detection.",
            "Source fan-out against one account.",
            unit="count",
        ),
        _e(
            "DBF_USER_FAILURE_COUNT",
            "user_failure_count__{window}",
            _GTE,
            "This account accumulated {observed} failures, at or above the "
            "configured threshold of {threshold}.",
            "Aggregate failure volume against the account.",
            unit="count",
        ),
        _e(
            "DBF_USER_FAILURE_RATE",
            "user_failure_rate__{window}",
            _GTE,
            "A failure share of {observed} matched the configured rule "
            "condition of at least {threshold}.",
            "Share of the account's attempts that failed.",
            unit="ratio",
        ),
        _e(
            "DBF_LOW_PER_SOURCE_VOLUME",
            "pair_attempt_count__{window}",
            _LTE,
            "The contributing source made only {observed} attempts, at or "
            "below the configured ceiling of {threshold}, which is consistent "
            "with effort spread across many sources.",
            "Per-source volume ceiling; the distributed discriminator.",
            unit="count",
        ),
        _e(
            "DBF_SOURCE_NOT_FANNED_OUT",
            "source_unique_user_count__{cardinality_window}",
            _LTE,
            "The contributing source touched {observed} distinct accounts, at "
            "or below the configured ceiling of {threshold}, so it is not "
            "spraying.",
            "Spraying guard.",
            unit="count",
        ),
    ),
    limitations=(
        "A popular account behind a mobile carrier NAT can legitimately appear "
        "to be reached from many distinct sources.",
        "Coarse source identity means one attacker behind a proxy pool and "
        "many unrelated clients look alike.",
    ),
)

_PAD_ATO_001 = RuleSpec(
    rule_id="PAD-ATO-001",
    rule_version="1.0.0",
    display_name="Account-takeover indicator",
    family=RuleFamily.ACCOUNT_COMPROMISE,
    attack_category=AttackCategory.ACCOUNT_TAKEOVER_INDICATOR,
    description=(
        "A successful authentication from a combination of contexts new to "
        "the account, supported by at least one behavioural deviation. This is "
        "an indicator of behaviour consistent with account takeover, not a "
        "finding that an account was taken over."
    ),
    default_severity=Severity.CRITICAL,
    correlation_group=CorrelationGroup.SESSION_ANOMALY,
    required_features=(
        "current_authentication_outcome",
        "user_in_baseline",
        "is_new_device_for_user",
        "is_new_source_for_user",
        "is_new_country_for_user",
        "is_new_application_for_user",
        "is_new_auth_method_for_user",
    ),
    optional_features=(
        "prior_failures_since_user_success",
        "login_hour_deviation",
        "response_time_zscore",
        "user_event_rate_ratio",
        "current_mfa_outcome",
        "distance_from_user_baseline_centroid_km",
    ),
    parameters=(
        _p_int(
            "min_novel_context_count",
            2,
            "How many of the five novelty indicators must be true.",
            minimum=1,
            maximum=5,
        ),
        _p_int(
            "min_supporting_signals",
            1,
            "How many supporting behavioural deviations must be present.",
            minimum=1,
            maximum=6,
        ),
        _p_int(
            "min_prior_failures",
            3,
            "Failures since this account last succeeded.",
            minimum=1,
        ),
        _p_float(
            "min_login_hour_deviation",
            0.90,
            "How unusual the anchor hour must be for this account.",
            maximum=1.0,
        ),
        _p_float(
            "min_abs_response_time_zscore",
            3.0,
            "Absolute standardised response-time deviation.",
            unit="zscore",
        ),
        _p_float(
            "min_event_rate_ratio",
            3.0,
            "Recent event rate relative to the account's baseline rate.",
            minimum=1.0,
        ),
        _p_float(
            "min_baseline_distance_km",
            1000.0,
            "Distance from the account's baseline location centroid.",
            maximum=20100.0,
            unit="km",
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=("user_in_baseline",),
        description=(
            "Novelty is meaningless for an account absent from the fitted "
            "baseline: a cold account is not the same as a known account "
            "seeing a new device."
        ),
    ),
    evidence=(
        _e(
            "ATO_CURRENT_SUCCESS",
            "current_authentication_outcome",
            _EQ,
            "The anchor authentication succeeded, which matched the configured "
            "rule condition.",
            "The success precondition.",
        ),
        _e(
            "ATO_NOVEL_CONTEXT_COUNT",
            None,
            _GTE,
            "{observed} of the account's context indicators were new, at or "
            "above the configured threshold of {threshold}; this contributed "
            "to this detection.",
            "Count of novel contexts across device, source, country, "
            "application, and method.",
            unit="count",
        ),
        _e(
            "ATO_PRIOR_FAILURES",
            "prior_failures_since_user_success",
            _GTE,
            "The account recorded {observed} failures since its last success, "
            "at or above the configured threshold of {threshold}.",
            "Supporting signal: preceding failure pressure.",
            unit="count",
        ),
        _e(
            "ATO_LOGIN_HOUR_DEVIATION",
            "login_hour_deviation",
            _GTE,
            "The anchor hour scored {observed} on this account's hour "
            "deviation, at or above {threshold}; this is consistent with "
            "activity outside the account's usual pattern.",
            "Supporting signal: unusual hour for this account.",
            unit="ratio",
        ),
        _e(
            "ATO_RESPONSE_TIME_DEVIATION",
            "response_time_zscore",
            _GTE,
            "The response time deviated {observed} standard deviations from "
            "this account's baseline, at or above {threshold}.",
            "Supporting signal: response-time deviation.",
            unit="zscore",
        ),
        _e(
            "ATO_EVENT_RATE_DEVIATION",
            "user_event_rate_ratio",
            _GTE,
            "The account's recent event rate was {observed} times its "
            "baseline, at or above {threshold}.",
            "Supporting signal: event-rate deviation.",
            unit="ratio",
        ),
        _e(
            "ATO_MFA_DEVIATION",
            "current_mfa_outcome",
            _IN,
            "The multi-factor step was recorded as {observed}, which matched "
            "the configured rule condition.",
            "Supporting signal: multi-factor deviation.",
        ),
        _e(
            "ATO_BASELINE_DISTANCE",
            "distance_from_user_baseline_centroid_km",
            _GTE,
            "The anchor was {observed} km from this account's baseline "
            "location centroid, at or above {threshold}.",
            "Supporting signal: distance from the account's usual area.",
            unit="km",
        ),
    ),
    limitations=(
        "Travel, a device replacement, or a new office reproduces several "
        "novelty indicators at once for a legitimate user.",
        "The rule reports an indicator. Confirming account takeover requires "
        "investigation this system does not perform.",
    ),
)

_PAD_GEO_001 = RuleSpec(
    rule_id="PAD-GEO-001",
    rule_version="1.0.0",
    display_name="Impossible-travel indicator",
    family=RuleFamily.LOCATION,
    attack_category=AttackCategory.IMPOSSIBLE_TRAVEL_INDICATOR,
    description=(
        "A successful authentication whose distance from the account's "
        "previous located success implies a travel speed above a configured "
        "plausible maximum. Derived from coarse location data, which is an "
        "approximation; this is an indicator, not a finding of compromise. No "
        "coordinate is read or emitted -- the rule consumes derived distance, "
        "velocity, and status columns only."
    ),
    default_severity=Severity.HIGH,
    correlation_group=CorrelationGroup.LOCATION_MOVEMENT,
    required_features=(
        "current_authentication_outcome",
        "user_previous_success_geo__status",
        "implied_velocity__status",
        "distance_km_from_user_previous_success",
        "implied_velocity_kmh_from_previous_success",
    ),
    optional_features=(
        "country_changed_since_previous_success",
        "seconds_since_user_previous_success_with_location",
    ),
    parameters=(
        _p_float(
            "min_distance_km",
            500.0,
            "Minimum meaningful distance; below this, coarse-location error dominates.",
            minimum=1.0,
            maximum=20100.0,
            unit="km",
        ),
        _p_float(
            "min_velocity_kmh",
            900.0,
            "Implied travel speed above which the movement is implausible.",
            minimum=1.0,
            unit="kmh",
        ),
        _p_float(
            "distance_rounding_km",
            10.0,
            "Rounding applied to the distance reported in evidence, so no "
            "precise location is inferable.",
            minimum=1.0,
            unit="km",
        ),
        _p_bool(
            "require_country_change",
            False,
            "Whether a country change is also required.",
        ),
        _p_choice(
            "zero_elapsed_policy",
            "fire",
            ("fire", "insufficient_data"),
            "How to treat a zero elapsed interval, where implied velocity is "
            "undefined rather than ordinary travel.",
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=("user_previous_success_geo__status",),
        description=(
            "Any status other than 'ok' means the comparison could not be "
            "made: no prior located success, or a missing coordinate on either "
            "side. Each yields insufficient_data with the status as the reason."
        ),
    ),
    evidence=(
        _e(
            "GEO_CURRENT_SUCCESS",
            "current_authentication_outcome",
            _EQ,
            "The anchor authentication succeeded, which matched the configured "
            "rule condition.",
            "The success precondition.",
        ),
        _e(
            "GEO_DISTANCE",
            "distance_km_from_user_previous_success",
            _GTE,
            "The anchor was roughly {observed} km from this account's previous "
            "located success, at or above the configured threshold of "
            "{threshold}.",
            "Rounded great-circle distance from the previous located success.",
            unit="km",
        ),
        _e(
            "GEO_IMPLIED_VELOCITY",
            "implied_velocity_kmh_from_previous_success",
            _GTE,
            "Covering that distance in the elapsed time implies {observed} "
            "km/h, at or above the configured threshold of {threshold}; this "
            "may indicate that the two authentications did not come from the "
            "same traveller.",
            "Capped implied velocity between the two located successes.",
            unit="kmh",
        ),
        _e(
            "GEO_ZERO_ELAPSED_INTERVAL",
            "implied_velocity__status",
            _EQ,
            "The two located successes share a timestamp, so implied velocity "
            "is undefined rather than merely high; this matched the configured "
            "rule condition.",
            "Explicit zero-elapsed branch; never treated as ordinary travel.",
        ),
        _e(
            "GEO_COUNTRY_CHANGE",
            "country_changed_since_previous_success",
            _IS_TRUE,
            "The country differed from that of the previous located success, "
            "which matched the configured rule condition.",
            "Optional country-change requirement.",
        ),
    ),
    limitations=(
        "Coarse location is approximate; a VPN, a corporate egress point, or "
        "a mobile carrier gateway relocates an apparent origin by thousands of "
        "kilometres with no travel at all.",
        "The rule reports an indicator derived from approximate data, not "
        "evidence that an account was used by two people.",
    ),
)

_PAD_BOT_001 = RuleSpec(
    rule_id="PAD-BOT-001",
    rule_version="1.0.0",
    display_name="Bot-like authentication indicator",
    family=RuleFamily.AUTOMATION,
    attack_category=AttackCategory.BOT_ACTIVITY,
    description=(
        "Sustained authentication volume from one source at machine-like "
        "regular intervals with uniform client characteristics. The volume "
        "floor and the dispersion requirement together mean no short sequence "
        "and no single fast interval can fire this rule."
    ),
    default_severity=Severity.MEDIUM,
    correlation_group=CorrelationGroup.AUTOMATION_TIMING,
    required_features=(
        "source_attempt_count__{dispersion_window}",
        "source_interarrival_coefficient_of_variation__{dispersion_window}",
        "source_mean_interarrival_seconds__{dispersion_window}",
        "source_unique_user_agent_count__{cardinality_window}",
    ),
    optional_features=("source_unique_user_count__{cardinality_window}",),
    parameters=(
        _p_window(
            "dispersion_window",
            "15m",
            "Window for the timing statistics; must be a window at which "
            "dispersion features are emitted.",
        ),
        _p_window(
            "cardinality_window",
            "1h",
            "Window for the source's distinct client counts.",
        ),
        _p_int(
            "min_attempts",
            20,
            "Attempt floor; below this the timing statistics are not meaningful.",
            minimum=3,
        ),
        _p_float(
            "max_interarrival_cov",
            0.15,
            "Ceiling on the interarrival coefficient of variation; low values "
            "indicate machine-like regularity.",
            maximum=10.0,
        ),
        _p_float(
            "max_mean_interarrival_seconds",
            120.0,
            "Ceiling on the mean gap between attempts.",
            minimum=0.0,
            maximum=86400.0,
            unit="seconds",
        ),
        _p_int(
            "max_unique_user_agents",
            2,
            "Ceiling on distinct user-agent families; automation tends to "
            "repeat one client string.",
            minimum=1,
        ),
        _p_int(
            "min_unique_users",
            0,
            "Optional account fan-out floor; zero disables the component.",
            minimum=0,
        ),
    ),
    min_history=MinHistorySpec(
        required_non_null_features=(
            "source_interarrival_coefficient_of_variation__{dispersion_window}",
        ),
        description=(
            "The coefficient of variation is null below two in-window "
            "observations or at a zero mean, which is too little timing "
            "history to judge regularity."
        ),
    ),
    evidence=(
        _e(
            "BOT_ATTEMPT_VOLUME",
            "source_attempt_count__{dispersion_window}",
            _GTE,
            "This source made {observed} attempts, at or above the configured "
            "threshold of {threshold}; this contributed to this detection.",
            "Attempt volume floor.",
            unit="count",
        ),
        _e(
            "BOT_TIMING_REGULARITY",
            "source_interarrival_coefficient_of_variation__{dispersion_window}",
            _LTE,
            "Interarrival dispersion of {observed} is at or below the "
            "configured ceiling of {threshold}, which is consistent with "
            "machine-generated timing.",
            "Timing regularity; the automation discriminator.",
            unit="ratio",
        ),
        _e(
            "BOT_MEAN_INTERARRIVAL",
            "source_mean_interarrival_seconds__{dispersion_window}",
            _LTE,
            "A mean gap of {observed} seconds is at or below the configured "
            "ceiling of {threshold}.",
            "Mean interarrival interval.",
            unit="seconds",
        ),
        _e(
            "BOT_CLIENT_UNIFORMITY",
            "source_unique_user_agent_count__{cardinality_window}",
            _LTE,
            "The source presented {observed} distinct client families, at or "
            "below the configured ceiling of {threshold}.",
            "Repeated client characteristics.",
            unit="count",
        ),
        _e(
            "BOT_SOURCE_FANOUT",
            "source_unique_user_count__{cardinality_window}",
            _GTE,
            "The source touched {observed} distinct accounts, at or above {threshold}.",
            "Optional fan-out evidence; contributes signal, never required.",
            unit="count",
        ),
    ),
    limitations=(
        "Legitimate automation -- monitoring probes, service accounts, and "
        "scheduled integrations -- authenticates on exactly this rhythm.",
        "Regular timing describes a client, not an intent.",
    ),
)

_PAD_MFA_001 = RuleSpec(
    rule_id="PAD-MFA-001",
    rule_version="1.0.0",
    display_name="MFA sequence anomaly indicator",
    family=RuleFamily.ACCOUNT_COMPROMISE,
    attack_category=AttackCategory.MFA_SEQUENCE_ANOMALY,
    description=(
        "Elevated recent multi-factor failure or challenge activity for an "
        "account, combined with an abnormal multi-factor outcome on the anchor "
        "event. Both parts are required, and a minimum-history gate means a "
        "single ordinary challenge cannot fire the rule. Shares a correlation "
        "group with the account-takeover indicator so one compromise narrative "
        "is not counted twice."
    ),
    default_severity=Severity.MEDIUM,
    correlation_group=CorrelationGroup.SESSION_ANOMALY,
    required_features=(
        "user_attempt_count__{window}",
        "user_mfa_failure_count__{window}",
        "user_challenge_count__{window}",
        "current_mfa_outcome",
    ),
    parameters=(
        _p_window("window", "15m", "Window for the account's multi-factor counts."),
        _p_int(
            "min_mfa_history_events",
            6,
            "Attempts the account must have in the window before the rule will "
            "reach a verdict.",
            minimum=2,
        ),
        _p_int(
            "min_mfa_observations",
            3,
            "Combined challenge and multi-factor failure observations required "
            "before the rule will reach a verdict.",
            minimum=1,
        ),
        _p_int(
            "min_mfa_failures",
            4,
            "Multi-factor failures for this account.",
            minimum=1,
        ),
        _p_int(
            "min_challenges",
            4,
            "Challenged authentications for this account.",
            minimum=1,
        ),
    ),
    min_history=MinHistorySpec(
        description=(
            "Below min_mfa_history_events attempts, or below "
            "min_mfa_observations combined challenge and multi-factor failure "
            "observations, the rule returns insufficient_data. There is not "
            "enough multi-factor history to call a sequence anomalous."
        ),
    ),
    evidence=(
        _e(
            "MFA_HISTORY_SUFFICIENT",
            "user_attempt_count__{window}",
            _GTE,
            "The account had {observed} attempts in the window, at or above "
            "the configured minimum history of {threshold}.",
            "Minimum-history gate.",
            unit="count",
        ),
        _e(
            "MFA_PRIOR_FAILURES",
            "user_mfa_failure_count__{window}",
            _GTE,
            "The account recorded {observed} multi-factor failures, at or "
            "above the configured threshold of {threshold}; this contributed "
            "to this detection.",
            "Elevated prior multi-factor failure activity.",
            unit="count",
        ),
        _e(
            "MFA_PRIOR_CHALLENGES",
            "user_challenge_count__{window}",
            _GTE,
            "The account was challenged {observed} times, at or above the "
            "configured threshold of {threshold}.",
            "Elevated prior challenge activity.",
            unit="count",
        ),
        _e(
            "MFA_CURRENT_OUTCOME",
            "current_mfa_outcome",
            _IN,
            "The anchor's multi-factor step was recorded as {observed}, which "
            "matched the configured rule condition; the sequence is consistent "
            "with multi-factor pressure.",
            "Abnormal multi-factor outcome on the anchor; always required.",
        ),
    ),
    limitations=(
        "A user with a failing authenticator app, a clock-drifted token, or a "
        "newly enrolled device reproduces this sequence.",
        "The rule reports a sequence anomaly. It is not a finding that a "
        "multi-factor control was defeated.",
    ),
)


#: Every registered rule.  Static Python registration only: there is no
#: discovery, no plugin path, and no configuration-supplied logic.
_RULE_SPECS: tuple[RuleSpec, ...] = (
    _PAD_ATO_001,
    _PAD_BF_001,
    _PAD_BF_002,
    _PAD_BOT_001,
    _PAD_CS_001,
    _PAD_DBF_001,
    _PAD_GEO_001,
    _PAD_MFA_001,
    _PAD_PS_001,
)

#: The catalog every other module reads.  Built and self-validated at import.
RULE_CATALOG: RuleCatalog = build_rule_catalog(_RULE_SPECS)


def catalog_to_markdown(catalog: RuleCatalog = RULE_CATALOG) -> str:
    """Render the rule catalog as Markdown documentation.

    Derived entirely from catalog metadata, so the documentation cannot drift
    from the registry.  Contains no data, no identifiers, and no paths.
    """
    lines: list[str] = [
        "# Rule Catalog",
        "",
        "Generated from the rule registry. Do not edit by hand; regenerate "
        "with `password-attack-detector detection catalog --format markdown`.",
        "",
        f"- Rule catalog version: `{RULE_CATALOG_VERSION}`",
        f"- Catalog fingerprint: `{catalog.fingerprint()}`",
        f"- Registered rules: {len(catalog)}",
        "",
        "Rules describe behaviour observed in point-in-time feature snapshots. "
        "A fired rule reports that a configured condition matched; it is not "
        "evidence that an attack occurred.",
        "",
        "## Rules by family",
        "",
        "| Family | Rules |",
        "|--------|-------|",
    ]
    for family, count in sorted(catalog.family_counts().items()):
        lines.append(f"| `{family}` | {count} |")

    lines.extend(
        [
            "",
            "## Correlation groups",
            "",
            "Rules in one group describe the same underlying behaviour. Their "
            "contributions are reduced within the group before groups are "
            "combined, so restating one behaviour cannot inflate a risk score.",
            "",
            "| Correlation group | Rules |",
            "|--------|-------|",
        ]
    )
    for group in CorrelationGroup:
        members = catalog.specs_for_correlation_group(group)
        if not members:
            continue
        joined = ", ".join(f"`{spec.rule_id}`" for spec in members)
        lines.append(f"| `{group}` | {joined} |")

    lines.extend(["", "## Rule index", "", "| Rule | Version | Name | Severity |"])
    lines.append("|--------|-------|-------|-------|")
    for spec in catalog.specs:
        lines.append(
            f"| `{spec.rule_id}` | `{spec.rule_version}` | {spec.display_name} | "
            f"`{spec.default_severity}` |"
        )

    for spec in catalog.specs:
        lines.extend(_rule_markdown(spec))

    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Rules consume Phase 3 point-in-time feature snapshots. They "
            "never read ground-truth labels, split assignments, campaign "
            "metadata, or the canonical event table.",
            "- Signal strength is a bounded ordinal magnitude, not a "
            "statistical probability.",
            "- Evidence records what was observed and what it is consistent "
            "with. It is not causal proof.",
            "- Synthetic data exercises these rules but does not demonstrate "
            "real-world detection effectiveness.",
            "",
        ]
    )
    return "\n".join(lines)


def _rule_markdown(spec: RuleSpec) -> list[str]:
    """Render one rule's section."""
    lines: list[str] = [
        "",
        f"## {spec.rule_id} -- {spec.display_name}",
        "",
        spec.description,
        "",
        "| Property | Value |",
        "|--------|-------|",
        f"| Rule version | `{spec.rule_version}` |",
        f"| Family | `{spec.family}` |",
        f"| Attack category | `{spec.attack_category}` |",
        f"| Default severity | `{spec.default_severity}` |",
        f"| Correlation group | `{spec.correlation_group}` |",
        f"| Evaluation scope | `{spec.evaluation_scope}` |",
        f"| Privacy class | `{spec.privacy_class}` |",
        f"| Deprecated | {'yes' if spec.deprecated else 'no'} |",
        "",
        "### Required features",
        "",
    ]
    lines.extend(f"- `{template}`" for template in spec.required_features)

    if spec.optional_features:
        lines.extend(["", "### Optional features", ""])
        lines.extend(f"- `{template}`" for template in spec.optional_features)

    lines.extend(
        [
            "",
            "### Thresholds",
            "",
            "| Parameter | Kind | Default | Minimum | Maximum | Unit | Meaning |",
            "|--------|-------|-------|-------|-------|-------|-------|",
        ]
    )
    for parameter in spec.parameters:
        lines.append(
            f"| `{parameter.name}` | `{parameter.kind}` | `{parameter.default}` | "
            f"{'-' if parameter.minimum is None else parameter.minimum} | "
            f"{'-' if parameter.maximum is None else parameter.maximum} | "
            f"{parameter.unit or '-'} | {parameter.description} |"
        )

    lines.extend(["", "### Minimum history", ""])
    if spec.min_history.required_non_null_features:
        lines.append("Requires a non-null value for:")
        lines.append("")
        lines.extend(
            f"- `{template}`"
            for template in spec.min_history.required_non_null_features
        )
        lines.append("")
    lines.append(
        spec.min_history.description
        or "No additional history requirement beyond the declared features."
    )

    lines.extend(
        [
            "",
            "### Evidence",
            "",
            "| Code | Feature | Comparator | Unit | Meaning |",
            "|--------|-------|-------|-------|-------|",
        ]
    )
    for item in spec.evidence:
        lines.append(
            f"| `{item.evidence_code}` | "
            f"{'derived' if item.feature_template is None else f'`{item.feature_template}`'} | "
            f"`{item.comparator}` | {item.unit or '-'} | {item.description} |"
        )

    if spec.limitations:
        lines.extend(["", "### Expected limitations", ""])
        lines.extend(f"- {text}" for text in spec.limitations)
    return lines
