"""Tests for the rule catalog and the static registry."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from password_attack_detector.detection.catalog import (
    RULE_CATALOG,
    RULE_CATALOG_VERSION,
    EvidenceDefinition,
    MinHistorySpec,
    RuleParameter,
    RuleSpec,
    build_rule_catalog,
    catalog_to_markdown,
    resolve_rule_features,
    validate_catalog_against_features,
)
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import (
    AttackCategory,
    CorrelationGroup,
    EvaluationScope,
    EvidenceComparator,
    ParameterKind,
    PrivacyClass,
    RuleFamily,
    Severity,
)
from password_attack_detector.detection.rules import ALL_RULES, build_rule_index
from password_attack_detector.exceptions import (
    DetectionConfigurationError,
    RuleEvaluationError,
)
from password_attack_detector.features.catalog import build_catalog
from password_attack_detector.features.config import FeatureConfig

EXPECTED_RULE_IDS = (
    "PAD-ATO-001",
    "PAD-BF-001",
    "PAD-BF-002",
    "PAD-BOT-001",
    "PAD-CS-001",
    "PAD-DBF-001",
    "PAD-GEO-001",
    "PAD-MFA-001",
    "PAD-PS-001",
)


@pytest.fixture()
def feature_catalog() -> Any:
    return build_catalog(FeatureConfig())


def _minimal_spec(**overrides: Any) -> RuleSpec:
    data: dict[str, Any] = {
        "rule_id": "PAD-ZZ-999",
        "rule_version": "1.0.0",
        "display_name": "Test rule",
        "family": RuleFamily.BRUTE_FORCE,
        "attack_category": AttackCategory.BRUTE_FORCE,
        "description": "A rule declared only inside the test suite.",
        "default_severity": Severity.LOW,
        "correlation_group": CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
        "required_features": ("user_failure_count__{window}",),
        "parameters": (
            RuleParameter(
                name="window",
                kind=ParameterKind.WINDOW,
                default="5m",
                description="Window.",
            ),
        ),
        "evidence": (
            EvidenceDefinition(
                evidence_code="TEST_CODE",
                feature_template="user_failure_count__{window}",
                comparator=EvidenceComparator.GTE,
                message_template="Observed {observed}; this contributed to this detection.",
                description="Test evidence.",
            ),
        ),
    }
    data.update(overrides)
    return RuleSpec(**data)


# ---------------------------------------------------------------------------
# Registry composition
# ---------------------------------------------------------------------------


def test_the_nine_approved_rules_are_registered() -> None:
    assert RULE_CATALOG.rule_ids == EXPECTED_RULE_IDS
    assert len(RULE_CATALOG) == 9


def test_catalog_ordering_is_deterministic() -> None:
    """Order is a property of the data, not of import order."""
    assert list(RULE_CATALOG.rule_ids) == sorted(RULE_CATALOG.rule_ids)
    assert [spec.rule_id for spec in RULE_CATALOG] == list(EXPECTED_RULE_IDS)


def test_rule_ids_and_versions_are_stable() -> None:
    """Published identifiers and versions are a contract.

    A deliberate change must show up as a diff on this pinned set, not as a
    silent alteration of every artifact recording a rule version.
    """
    pinned = dict.fromkeys(EXPECTED_RULE_IDS, "1.0.0")
    actual = {spec.rule_id: spec.rule_version for spec in RULE_CATALOG}
    assert actual == pinned


def test_identifiers_are_unique() -> None:
    ids = [spec.rule_id for spec in RULE_CATALOG]
    assert len(set(ids)) == len(ids)


def test_display_names_are_unique() -> None:
    names = [spec.display_name for spec in RULE_CATALOG]
    assert len(set(names)) == len(names)


def test_every_rule_is_non_sensitive_and_current() -> None:
    for spec in RULE_CATALOG:
        assert spec.privacy_class is PrivacyClass.NON_SENSITIVE
        assert spec.evaluation_scope is EvaluationScope.ANCHOR_EVENT
        assert spec.deprecated is False


def test_every_rule_declares_evidence_and_limitations() -> None:
    for spec in RULE_CATALOG:
        assert spec.evidence, spec.rule_id
        assert spec.limitations, spec.rule_id


def test_correlation_groups_are_valid_members() -> None:
    for spec in RULE_CATALOG:
        assert spec.correlation_group in set(CorrelationGroup)


def test_brute_force_rules_share_a_correlation_group() -> None:
    """PAD-BF-001 and PAD-BF-002 must not double-count one failure burst."""
    group = CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET
    assert RULE_CATALOG.get("PAD-BF-001").correlation_group is group
    assert RULE_CATALOG.get("PAD-BF-002").correlation_group is group
    assert RULE_CATALOG.get("PAD-DBF-001").correlation_group is group


def test_mfa_and_account_takeover_share_a_correlation_group() -> None:
    group = CorrelationGroup.SESSION_ANOMALY
    assert RULE_CATALOG.get("PAD-ATO-001").correlation_group is group
    assert RULE_CATALOG.get("PAD-MFA-001").correlation_group is group


def test_bf_002_is_distinguished_by_the_current_success_condition() -> None:
    spec = RULE_CATALOG.get("PAD-BF-002")
    assert "current_authentication_outcome" in spec.required_features
    assert (
        "current_authentication_outcome"
        not in RULE_CATALOG.get("PAD-BF-001").required_features
    )


def test_bf_002_reads_only_prior_history_beyond_the_current_outcome(
    feature_catalog: Any,
) -> None:
    """Every other input must be a prior-history sequence feature."""
    from password_attack_detector.features.catalog import LeakageClass

    spec = RULE_CATALOG.get("PAD-BF-002")
    resolved = resolve_rule_features(spec, spec.default_parameters(), feature_catalog)
    for template, name in resolved.items():
        if template == "current_authentication_outcome":
            continue
        assert feature_catalog.get(name).leakage_class is LeakageClass.PRIOR_ONLY


def test_mfa_rule_requires_both_prior_activity_and_a_current_outcome() -> None:
    spec = RULE_CATALOG.get("PAD-MFA-001")
    assert "current_mfa_outcome" in spec.required_features
    assert "user_mfa_failure_count__{window}" in spec.required_features
    assert "user_challenge_count__{window}" in spec.required_features
    names = {parameter.name for parameter in spec.parameters}
    assert {"min_mfa_history_events", "min_mfa_observations"} <= names


def test_no_standalone_blocked_account_rule_exists() -> None:
    """Blocked-account activity is supporting evidence, never a rule."""
    for spec in RULE_CATALOG:
        assert "blocked" not in str(spec.attack_category)
        assert "blocked" not in str(spec.family)
        assert "blocked" not in spec.display_name.lower()

    supporting = RULE_CATALOG.get("PAD-BF-001")
    assert "user_blocked_count__{window}" in supporting.optional_features
    assert "user_blocked_count__{window}" not in supporting.required_features


def test_no_rule_produces_the_novel_anomaly_holdout() -> None:
    categories = {str(spec.attack_category) for spec in RULE_CATALOG}
    assert "novel_anomaly_holdout" not in categories


def test_family_and_group_lookups() -> None:
    brute_force = RULE_CATALOG.specs_for_family(RuleFamily.BRUTE_FORCE)
    assert {spec.rule_id for spec in brute_force} == {
        "PAD-BF-001",
        "PAD-BF-002",
        "PAD-DBF-001",
    }
    fanout = RULE_CATALOG.specs_for_correlation_group(CorrelationGroup.SOURCE_FANOUT)
    assert {spec.rule_id for spec in fanout} == {"PAD-PS-001", "PAD-CS-001"}
    assert RULE_CATALOG.family_counts()["brute_force"] == 3
    assert set(RULE_CATALOG.families()) == set(RuleFamily)


def test_membership_and_lookup_helpers() -> None:
    assert "PAD-BF-001" in RULE_CATALOG
    assert RULE_CATALOG.has("PAD-BF-001")
    assert not RULE_CATALOG.has("PAD-ZZ-999")
    with pytest.raises(DetectionConfigurationError, match="Unknown rule identifier"):
        RULE_CATALOG.get("PAD-ZZ-999")


def test_parameter_and_evidence_lookup_failures() -> None:
    spec = RULE_CATALOG.get("PAD-BF-001")
    assert spec.parameter("window").kind is ParameterKind.WINDOW
    with pytest.raises(DetectionConfigurationError, match="declares no parameter"):
        spec.parameter("nonexistent")
    with pytest.raises(RuleEvaluationError, match="declares no evidence"):
        spec.evidence_definition("NOT_A_CODE")


# ---------------------------------------------------------------------------
# Registry self-validation
# ---------------------------------------------------------------------------


def test_duplicate_rule_id_is_rejected() -> None:
    spec = _minimal_spec()
    with pytest.raises(DetectionConfigurationError, match="Duplicate rule identifier"):
        build_rule_catalog((spec, _minimal_spec(display_name="Other")))


def test_duplicate_display_name_is_rejected() -> None:
    with pytest.raises(DetectionConfigurationError, match="Duplicate rule display"):
        build_rule_catalog((_minimal_spec(), _minimal_spec(rule_id="PAD-ZZ-998")))


def test_empty_catalog_is_rejected() -> None:
    with pytest.raises(DetectionConfigurationError, match="must declare a rule"):
        build_rule_catalog(())


@pytest.mark.parametrize("rule_id", ["BF-001", "pad-bf-001", "PAD-BF-1", "PAD--001"])
def test_invalid_rule_id_is_rejected(rule_id: str) -> None:
    with pytest.raises(ValidationError, match="rule_id"):
        _minimal_spec(rule_id=rule_id)


def test_invalid_rule_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="semantic version"):
        _minimal_spec(rule_version="1.0")


def test_rule_without_required_features_is_rejected() -> None:
    with pytest.raises(ValidationError, match="declares no required feature"):
        _minimal_spec(required_features=())


def test_rule_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="declares no evidence"):
        _minimal_spec(evidence=())


def test_repeated_parameter_name_is_rejected() -> None:
    duplicate = RuleParameter(
        name="window",
        kind=ParameterKind.WINDOW,
        default="5m",
        description="Window.",
    )
    with pytest.raises(ValidationError, match="repeats a parameter name"):
        _minimal_spec(parameters=(duplicate, duplicate))


def test_feature_declared_both_required_and_optional_is_rejected() -> None:
    with pytest.raises(ValidationError, match="both required and optional"):
        _minimal_spec(optional_features=("user_failure_count__{window}",))


def test_template_referencing_an_undeclared_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError, match="undeclared parameter"):
        _minimal_spec(required_features=("user_failure_count__{missing_param}",))


@pytest.mark.parametrize(
    "template",
    [
        "attack_class",
        "malicious",
        "campaign_id",
        "split",
        "model_probability",
        "supervised_training_eligible",
        "anchor_event_id",
        "anchor_event_time",
        "feature_schema_version",
    ],
)
def test_prohibited_feature_templates_are_rejected(template: str) -> None:
    """Labels, campaign metadata, splits, model output, and keys are refused."""
    with pytest.raises(ValidationError, match="prohibited feature"):
        _minimal_spec(required_features=(template,))


def test_history_for_an_undeclared_feature_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires history for undeclared"):
        _minimal_spec(
            min_history=MinHistorySpec(required_non_null_features=("not_declared",))
        )


def test_evidence_for_an_undeclared_feature_is_rejected() -> None:
    stray = EvidenceDefinition(
        evidence_code="STRAY",
        feature_template="pair_failure_count__{window}",
        comparator=EvidenceComparator.GTE,
        message_template="Observed {observed}.",
        description="Stray.",
    )
    with pytest.raises(ValidationError, match="undeclared feature template"):
        _minimal_spec(evidence=(stray,))


def test_repeated_evidence_code_is_rejected() -> None:
    item = EvidenceDefinition(
        evidence_code="TEST_CODE",
        feature_template="user_failure_count__{window}",
        comparator=EvidenceComparator.GTE,
        message_template="Observed {observed}.",
        description="Duplicate.",
    )
    with pytest.raises(ValidationError, match="repeats an evidence code"):
        _minimal_spec(evidence=(item, item))


# ---------------------------------------------------------------------------
# Parameter declarations
# ---------------------------------------------------------------------------


def test_parameter_default_must_satisfy_its_own_bounds() -> None:
    with pytest.raises(ValidationError, match="at least"):
        RuleParameter(
            name="threshold",
            kind=ParameterKind.INT,
            default=0,
            minimum=1,
            description="Bad default.",
        )


def test_parameter_minimum_above_maximum_is_rejected() -> None:
    with pytest.raises(ValidationError, match="minimum above maximum"):
        RuleParameter(
            name="threshold",
            kind=ParameterKind.FLOAT,
            default=5.0,
            minimum=10.0,
            maximum=1.0,
            description="Inverted bounds.",
        )


@pytest.mark.parametrize("name", ["Window", "1window", "win-dow", ""])
def test_invalid_parameter_name_is_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match="parameter name"):
        RuleParameter(
            name=name,
            kind=ParameterKind.INT,
            default=1,
            description="Bad name.",
        )


def test_allowed_values_require_a_string_parameter() -> None:
    with pytest.raises(ValidationError, match="not a string parameter"):
        RuleParameter(
            name="mode",
            kind=ParameterKind.INT,
            default=1,
            allowed_values=("a",),
            description="Wrong kind.",
        )


def test_empty_choice_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="empty choice set"):
        RuleParameter(
            name="mode",
            kind=ParameterKind.STRING,
            default="a",
            allowed_values=(),
            description="Empty.",
        )


def test_float_parameter_rejects_non_finite() -> None:
    parameter = RuleParameter(
        name="ratio",
        kind=ParameterKind.FLOAT,
        default=1.0,
        description="Ratio.",
    )
    with pytest.raises(ValueError, match="must be finite"):
        parameter.validate_value(float("inf"))
    with pytest.raises(ValueError, match="must be finite"):
        parameter.validate_value(float("nan"))


def test_bool_parameter_rejects_an_integer() -> None:
    parameter = RuleParameter(
        name="flag",
        kind=ParameterKind.BOOL,
        default=False,
        description="Flag.",
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        parameter.validate_value(1)


def test_int_parameter_rejects_a_bool() -> None:
    parameter = RuleParameter(
        name="count",
        kind=ParameterKind.INT,
        default=1,
        description="Count.",
    )
    with pytest.raises(ValueError, match="must be an integer"):
        parameter.validate_value(True)


def test_float_parameter_accepts_an_integer() -> None:
    parameter = RuleParameter(
        name="ratio",
        kind=ParameterKind.FLOAT,
        default=1.0,
        description="Ratio.",
    )
    assert parameter.validate_value(2) == pytest.approx(2.0)


def test_window_parameter_rejects_a_non_duration() -> None:
    parameter = RuleParameter(
        name="window",
        kind=ParameterKind.WINDOW,
        default="5m",
        description="Window.",
    )
    with pytest.raises(ValueError, match="Invalid duration"):
        parameter.validate_value("5 minutes")
    with pytest.raises(ValueError, match="window label"):
        parameter.validate_value(5)


def test_string_parameter_rejects_a_non_string() -> None:
    parameter = RuleParameter(
        name="mode",
        kind=ParameterKind.STRING,
        default="a",
        allowed_values=("a", "b"),
        description="Mode.",
    )
    with pytest.raises(ValueError, match="must be a string"):
        parameter.validate_value(1)


# ---------------------------------------------------------------------------
# Evidence definitions
# ---------------------------------------------------------------------------


def _definition(**overrides: Any) -> EvidenceDefinition:
    data: dict[str, Any] = {
        "evidence_code": "CODE",
        "feature_template": None,
        "comparator": EvidenceComparator.GTE,
        "message_template": "Observed {observed}.",
        "description": "Item.",
    }
    data.update(overrides)
    return EvidenceDefinition(**data)


@pytest.mark.parametrize(
    "template",
    [
        "Observed {value}.",
        "Observed {observed.real}.",
        "Observed {0}.",
        "Observed {observed",
    ],
)
def test_unsupported_message_placeholders_are_rejected(template: str) -> None:
    with pytest.raises(ValidationError):
        _definition(message_template=template)


@pytest.mark.parametrize(
    "template",
    [
        "This proves an attack occurred.",
        "The evidence confirms compromise.",
        "This guarantees credential theft.",
    ],
)
def test_claim_asserting_message_templates_are_rejected(template: str) -> None:
    with pytest.raises(ValidationError, match="claim-asserting"):
        _definition(message_template=template)


def test_invalid_evidence_code_is_rejected() -> None:
    with pytest.raises(ValidationError, match="evidence_code"):
        _definition(evidence_code="lower_case")


def test_registered_messages_avoid_causal_language() -> None:
    banned = ("prove", "confirm", "guarantee", "definitely", "certainly")
    for spec in RULE_CATALOG:
        for item in spec.evidence:
            lowered = item.message_template.lower()
            assert not any(word in lowered for word in banned), item.evidence_code


# ---------------------------------------------------------------------------
# Resolution against the feature catalog
# ---------------------------------------------------------------------------


def test_every_registered_rule_resolves_against_the_feature_catalog(
    feature_catalog: Any,
) -> None:
    validate_catalog_against_features(RULE_CATALOG, feature_catalog)


def test_resolution_covers_required_and_optional_features(
    feature_catalog: Any,
) -> None:
    spec = RULE_CATALOG.get("PAD-BF-001")
    resolved = resolve_rule_features(spec, spec.default_parameters(), feature_catalog)
    assert resolved["pair_failure_count__{window}"] == "pair_failure_count__5m"
    assert "user_blocked_count__{window}" in resolved

    required_only = resolve_rule_features(
        spec, spec.default_parameters(), feature_catalog, include_optional=False
    )
    assert "user_blocked_count__{window}" not in required_only


def test_resolution_fails_for_a_window_the_catalog_does_not_emit(
    feature_catalog: Any,
) -> None:
    """A window that carries no such feature is a loud failure, not a null."""
    spec = RULE_CATALOG.get("PAD-BF-001")
    parameters = dict(spec.default_parameters())
    parameters["window"] = "2m"
    with pytest.raises(RuleEvaluationError, match="does not declare"):
        resolve_rule_features(spec, parameters, feature_catalog)


def test_resolution_fails_for_a_cardinality_window_without_unique_counts(
    feature_catalog: Any,
) -> None:
    spec = RULE_CATALOG.get("PAD-BF-001")
    parameters = dict(spec.default_parameters())
    parameters["cardinality_window"] = "1m"
    with pytest.raises(RuleEvaluationError, match="does not declare"):
        resolve_rule_features(spec, parameters, feature_catalog)


def test_resolution_fails_when_a_parameter_is_missing(feature_catalog: Any) -> None:
    spec = RULE_CATALOG.get("PAD-BF-001")
    with pytest.raises(RuleEvaluationError, match="needs parameter"):
        resolve_rule_features(spec, {}, feature_catalog)


def test_validate_catalog_accepts_effective_parameters(feature_catalog: Any) -> None:
    config = DetectionConfig()
    parameters = {
        rule_id: config.parameters_for(rule_id) for rule_id in config.enabled_rule_ids
    }
    validate_catalog_against_features(
        RULE_CATALOG, feature_catalog, parameters_by_rule=parameters
    )


def test_validate_catalog_surfaces_a_bad_override(feature_catalog: Any) -> None:
    with pytest.raises(RuleEvaluationError, match="does not declare"):
        validate_catalog_against_features(
            RULE_CATALOG,
            feature_catalog,
            parameters_by_rule={
                "PAD-BF-001": {
                    **RULE_CATALOG.get("PAD-BF-001").default_parameters(),
                    "window": "2m",
                }
            },
        )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_hex() -> None:
    fingerprint = RULE_CATALOG.fingerprint()
    assert len(fingerprint) == 64
    assert fingerprint == build_rule_catalog(RULE_CATALOG.specs).fingerprint()


def test_fingerprint_is_independent_of_declaration_order() -> None:
    forward = build_rule_catalog(RULE_CATALOG.specs)
    reverse = build_rule_catalog(tuple(reversed(RULE_CATALOG.specs)))
    assert forward.fingerprint() == reverse.fingerprint()


def test_fingerprint_ignores_prose_edits() -> None:
    """Fixing a typo must not invalidate every artifact recording the digest."""
    baseline = build_rule_catalog((_minimal_spec(),)).fingerprint()
    reworded = build_rule_catalog(
        (
            _minimal_spec(
                description="Completely different prose.",
                limitations=("A newly documented caveat.",),
            ),
        )
    ).fingerprint()
    assert reworded == baseline


def test_fingerprint_changes_with_a_semantic_edit() -> None:
    baseline = build_rule_catalog((_minimal_spec(),)).fingerprint()
    assert (
        build_rule_catalog(
            (_minimal_spec(default_severity=Severity.HIGH),)
        ).fingerprint()
        != baseline
    )
    assert (
        build_rule_catalog((_minimal_spec(rule_version="1.1.0"),)).fingerprint()
        != baseline
    )
    assert (
        build_rule_catalog(
            (
                _minimal_spec(
                    parameters=(
                        RuleParameter(
                            name="window",
                            kind=ParameterKind.WINDOW,
                            default="1h",
                            description="Window.",
                        ),
                    )
                ),
            )
        ).fingerprint()
        != baseline
    )


def test_catalog_version_is_pinned() -> None:
    assert RULE_CATALOG_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# Generated documentation
# ---------------------------------------------------------------------------


def test_markdown_is_deterministic() -> None:
    assert catalog_to_markdown() == catalog_to_markdown()


def test_markdown_documents_every_rule_from_catalog_metadata() -> None:
    rendered = catalog_to_markdown()
    for spec in RULE_CATALOG:
        assert f"## {spec.rule_id} -- {spec.display_name}" in rendered
        assert spec.description in rendered
        for parameter in spec.parameters:
            assert f"`{parameter.name}`" in rendered
        for item in spec.evidence:
            assert f"`{item.evidence_code}`" in rendered
        for limitation in spec.limitations:
            assert limitation in rendered


def test_markdown_records_the_fingerprint_and_version() -> None:
    rendered = catalog_to_markdown()
    assert RULE_CATALOG.fingerprint() in rendered
    assert RULE_CATALOG_VERSION in rendered


def test_markdown_states_the_boundary_and_the_limits() -> None:
    rendered = catalog_to_markdown().lower()
    assert "never read ground-truth labels" in rendered
    assert "not a statistical probability" in rendered
    assert "not causal proof" in rendered
    assert "does not demonstrate" in rendered


def test_markdown_carries_no_data_or_paths() -> None:
    rendered = catalog_to_markdown()
    assert "/home/" not in rendered
    assert "u:" not in rendered


def test_markdown_renders_a_rule_without_optional_features() -> None:
    """The optional-features section is omitted rather than left empty."""
    rendered = catalog_to_markdown(build_rule_catalog((_minimal_spec(),)))
    assert "### Optional features" not in rendered
    assert "### Required features" in rendered


def test_markdown_renders_a_rule_without_a_history_requirement() -> None:
    rendered = catalog_to_markdown(build_rule_catalog((_minimal_spec(),)))
    assert "No additional history requirement" in rendered


# ---------------------------------------------------------------------------
# Implementation registry
# ---------------------------------------------------------------------------


def test_every_catalogued_rule_is_registered_exactly_once() -> None:
    """Registration is a static tuple covering the catalog, with no duplicates."""
    identifiers = [rule.spec.rule_id for rule in ALL_RULES]
    assert sorted(identifiers) == list(RULE_CATALOG.rule_ids)


class _StubRule:
    def __init__(self, spec: RuleSpec) -> None:
        self.spec = spec

    def prepare(self, config: Any, feature_catalog: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def test_rule_index_accepts_catalog_backed_implementations() -> None:
    rules = [_StubRule(RULE_CATALOG.get("PAD-BF-001"))]
    index = build_rule_index(rules)
    assert set(index) == {"PAD-BF-001"}


def test_rule_index_rejects_a_duplicate_implementation() -> None:
    spec = RULE_CATALOG.get("PAD-BF-001")
    with pytest.raises(DetectionConfigurationError, match="Duplicate rule"):
        build_rule_index([_StubRule(spec), _StubRule(spec)])


def test_rule_index_rejects_an_unregistered_implementation() -> None:
    with pytest.raises(DetectionConfigurationError, match="no catalog entry"):
        build_rule_index([_StubRule(_minimal_spec())])


def test_rule_index_rejects_a_lookalike_specification() -> None:
    """An implementation must carry the catalog's own spec, not a copy.

    A copy would let rule metadata drift from the registry the fingerprint and
    the documentation are built from.
    """
    lookalike = RULE_CATALOG.get("PAD-BF-001").model_copy()
    with pytest.raises(DetectionConfigurationError, match="not the registered"):
        build_rule_index([_StubRule(lookalike)])


# ---------------------------------------------------------------------------
# Defensive guards
# ---------------------------------------------------------------------------


def test_float_parameter_rejects_a_non_number() -> None:
    parameter = RuleParameter(
        name="ratio",
        kind=ParameterKind.FLOAT,
        default=1.0,
        description="Ratio.",
    )
    with pytest.raises(ValueError, match="must be a number"):
        parameter.validate_value("high")


def test_malformed_feature_template_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is malformed"):
        _minimal_spec(required_features=("user_failure_count__{window",))


def test_prohibited_attack_category_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard fires if a prohibited category is ever added to the enum.

    No current member is prohibited, so the prohibition set is redirected at
    an existing member to prove the guard is wired rather than decorative.
    """
    from password_attack_detector.detection import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module, "PROHIBITED_CATEGORY_TERMS", frozenset({"bot_activity"})
    )
    with pytest.raises(ValidationError, match="not a rule-based category"):
        _minimal_spec(attack_category=AttackCategory.BOT_ACTIVITY)


def _string_column_spec(default: str) -> RuleSpec:
    """A spec whose feature name comes from a plain string parameter.

    Declaration-time checks inspect the template text, so a placeholder-only
    template passes them. This is what makes the resolution-time guards
    reachable.
    """
    return _minimal_spec(
        required_features=("{column}",),
        parameters=(
            RuleParameter(
                name="column",
                kind=ParameterKind.STRING,
                default=default,
                description="Column name.",
            ),
        ),
        evidence=(
            EvidenceDefinition(
                evidence_code="TEST_CODE",
                feature_template="{column}",
                comparator=EvidenceComparator.GTE,
                message_template="Observed {observed}.",
                description="Test evidence.",
            ),
        ),
    )


def test_resolution_rejects_a_prohibited_column(feature_catalog: Any) -> None:
    """A template that resolves onto a label or split column is refused."""
    spec = _string_column_spec("split")
    with pytest.raises(RuleEvaluationError, match="prohibited column"):
        resolve_rule_features(spec, {"column": "split"}, feature_catalog)

    with pytest.raises(RuleEvaluationError, match="prohibited column"):
        resolve_rule_features(spec, {"column": "malicious"}, feature_catalog)


def test_resolution_rejects_a_key_class_feature() -> None:
    """A rule may not read a key column even when it is not on the deny list."""
    from password_attack_detector.features.catalog import (
        FeatureCatalog,
        FeatureDType,
        FeatureGroup,
        FeatureSpec,
        LeakageClass,
    )
    from password_attack_detector.features.config import EntityKind

    key_only = FeatureCatalog(
        (
            FeatureSpec(
                name="internal_row_key",
                group=FeatureGroup.KEY,
                entity=EntityKind.NONE,
                dtype=FeatureDType.STRING,
                nullable=False,
                leakage_class=LeakageClass.KEY,
                null_semantics="Never null.",
                description="A key column that is not on the prohibition list.",
            ),
        ),
        config_fingerprint="test",
    )
    spec = _string_column_spec("internal_row_key")
    with pytest.raises(RuleEvaluationError, match="not a permitted rule input"):
        resolve_rule_features(spec, {"column": "internal_row_key"}, key_only)
