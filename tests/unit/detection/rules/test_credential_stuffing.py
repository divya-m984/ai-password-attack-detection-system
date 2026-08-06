"""Thresholds and discriminators for PAD-CS-001."""

from __future__ import annotations

import pytest

from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.detection.rules.base import PreparedRule
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories
from tests.unit.detection.factories import stuffing_row


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def rule(catalog: FeatureCatalog) -> PreparedRule:
    return RULE_IMPLEMENTATIONS["PAD-CS-001"].prepare(DetectionConfig(), catalog)


def test_fires_at_exact_threshold_equality(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    row = stuffing_row(
        catalog,
        source_unique_user_count__1h=10,
        source_success_count__1h=1,
        source_failure_count__1h=10,
        source_attempt_count__1h=40,  # exactly 4.0 attempts per account
        source_unique_device_count__1h=3,
        source_unique_user_agent_count__1h=3,
    )
    result = rule.evaluate(row)
    assert result.status is RuleStatus.FIRED
    assert result.signal_strength > 0.0


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("source_unique_user_count__1h", 9),
        ("source_failure_count__1h", 9),
    ],
)
def test_one_step_below_an_integer_threshold_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog, column: str, value: int
) -> None:
    overrides = {
        "source_unique_user_count__1h": 10,
        "source_failure_count__1h": 10,
        "source_attempt_count__1h": 30,
        column: value,
    }
    assert rule.evaluate(stuffing_row(catalog, **overrides)).status is (
        RuleStatus.NOT_FIRED
    )


def test_pure_failures_without_a_success_do_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """The mixed outcome is the signature; a pure-failure run is spraying."""
    result = rule.evaluate(stuffing_row(catalog, source_success_count__1h=0))
    assert result.status is RuleStatus.NOT_FIRED
    assert "OUTCOME_MIX_NOT_PRESENT" in result.reason_codes


def test_successes_without_failures_do_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(stuffing_row(catalog, source_failure_count__1h=0))
    assert result.status is RuleStatus.NOT_FIRED
    assert "OUTCOME_MIX_NOT_PRESENT" in result.reason_codes


def test_familiar_context_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """Broad legitimate volume without novelty must stay silent."""
    result = rule.evaluate(
        stuffing_row(
            catalog, is_new_device_for_user=False, is_new_country_for_user=False
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "CONTEXT_ALREADY_FAMILIAR_FOR_ACCOUNT" in result.reason_codes


@pytest.mark.parametrize(
    "column", ["is_new_device_for_user", "is_new_country_for_user"]
)
def test_either_novelty_alone_satisfies_the_context_condition(
    rule: PreparedRule, catalog: FeatureCatalog, column: str
) -> None:
    overrides = {
        "is_new_device_for_user": False,
        "is_new_country_for_user": False,
        column: True,
    }
    assert rule.evaluate(stuffing_row(catalog, **overrides)).status is RuleStatus.FIRED


def test_a_null_novelty_flag_counts_as_familiar(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        stuffing_row(catalog, is_new_device_for_user=None, is_new_country_for_user=None)
    )
    assert result.status is RuleStatus.NOT_FIRED


def test_an_account_absent_from_the_baseline_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """A cold account has no "unfamiliar" to measure against."""
    result = rule.evaluate(stuffing_row(catalog, user_in_baseline=False))
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("ACCOUNT_ABSENT_FROM_BASELINE",)
    assert result.evidence == ()


def test_either_kind_of_client_diversity_satisfies_the_condition(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    devices_only = rule.evaluate(
        stuffing_row(
            catalog,
            source_unique_device_count__1h=6,
            source_unique_user_agent_count__1h=1,
        )
    )
    agents_only = rule.evaluate(
        stuffing_row(
            catalog,
            source_unique_device_count__1h=1,
            source_unique_user_agent_count__1h=6,
        )
    )
    assert devices_only.status is RuleStatus.FIRED
    assert agents_only.status is RuleStatus.FIRED


def test_client_diversity_evidence_is_true_of_the_value_it_carries(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    """The reported count must satisfy the threshold reported alongside it."""
    for devices, agents in ((6, 1), (1, 6)):
        result = rule.evaluate(
            stuffing_row(
                catalog,
                source_unique_device_count__1h=devices,
                source_unique_user_agent_count__1h=agents,
            )
        )
        item = next(
            evidence
            for evidence in result.evidence
            if evidence.evidence_code == "CS_CLIENT_DIVERSITY"
        )
        assert isinstance(item.observed_value, int)
        assert isinstance(item.threshold_value, int)
        assert item.observed_value >= item.threshold_value


def test_uniform_clients_do_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        stuffing_row(
            catalog,
            source_unique_device_count__1h=1,
            source_unique_user_agent_count__1h=1,
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "BELOW_CLIENT_DIVERSITY_THRESHOLD" in result.reason_codes


def test_high_attempts_per_account_does_not_fire(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        stuffing_row(
            catalog, source_unique_user_count__1h=10, source_attempt_count__1h=500
        )
    )
    assert result.status is RuleStatus.NOT_FIRED
    assert "ATTEMPTS_PER_ACCOUNT_TOO_HIGH" in result.reason_codes


def test_no_observed_accounts_is_insufficient_data(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    result = rule.evaluate(
        stuffing_row(
            catalog, source_unique_user_count__1h=0, source_attempt_count__1h=0
        )
    )
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert result.reason_codes == ("NO_TARGETED_ACCOUNTS_OBSERVED",)


def test_two_unfamiliar_contexts_score_above_one(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    one = rule.evaluate(stuffing_row(catalog, is_new_country_for_user=False))
    two = rule.evaluate(stuffing_row(catalog))
    assert two.signal_strength > one.signal_strength


def test_stronger_evidence_never_lowers_signal_strength(
    rule: PreparedRule, catalog: FeatureCatalog
) -> None:
    weaker = rule.evaluate(
        stuffing_row(
            catalog,
            source_unique_user_count__1h=10,
            source_failure_count__1h=10,
            source_attempt_count__1h=40,
            source_unique_device_count__1h=3,
            source_unique_user_agent_count__1h=3,
            is_new_country_for_user=False,
        )
    )
    stronger = rule.evaluate(
        stuffing_row(
            catalog,
            source_unique_user_count__1h=100,
            source_failure_count__1h=100,
            source_attempt_count__1h=110,
            source_unique_device_count__1h=30,
            source_unique_user_agent_count__1h=30,
        )
    )
    assert stronger.signal_strength > weaker.signal_strength


def test_the_rule_reads_no_credential_or_reuse_feature() -> None:
    """Behavioural proxies only: no credential value reaches this rule."""
    spec = RULE_CATALOG.get("PAD-CS-001")
    forbidden = ("password", "credential", "secret", "hash", "token", "reuse")
    for template in (*spec.required_features, *spec.optional_features):
        assert not any(term in template.lower() for term in forbidden), template
