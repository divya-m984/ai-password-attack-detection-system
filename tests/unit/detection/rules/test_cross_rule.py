"""Cross-rule discrimination: each shape fires its own rule and not the others.

A rule that fires on a scenario it was not written for is a false positive that
no per-rule test can catch, because each per-rule test only ever looks at its
own rule.  These tests take one scenario snapshot and assert what the *whole*
registry does with it.

The discriminators being exercised here are the concentration ceiling in
PAD-BF-001, the attempts-per-account ceiling in PAD-PS-001, the per-source
volume ceiling in PAD-DBF-001, the success requirement in PAD-BF-002, the
unfamiliar-context requirement in PAD-CS-001, and the supporting-signal
requirement in PAD-ATO-001.
"""

from __future__ import annotations

from typing import Any

import pytest

from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


def fired_rules(row: dict[str, Any], catalog: FeatureCatalog) -> set[str]:
    """Return every rule that fires on *row*."""
    config = DetectionConfig()
    fired = set()
    for rule_id, rule in RULE_IMPLEMENTATIONS.items():
        if rule.prepare(config, catalog).evaluate(row).status is RuleStatus.FIRED:
            fired.add(rule_id)
    return fired


def status_of(rule_id: str, row: dict[str, Any], catalog: FeatureCatalog) -> RuleStatus:
    """Return *rule_id*'s status on *row*."""
    prepared = RULE_IMPLEMENTATIONS[rule_id].prepare(DetectionConfig(), catalog)
    return prepared.evaluate(row).status


# ---------------------------------------------------------------------------
# Guessing shapes are mutually exclusive
# ---------------------------------------------------------------------------


def test_concentrated_brute_force_does_not_trigger_spraying(
    catalog: FeatureCatalog,
) -> None:
    row = factories.brute_force_row(
        catalog,
        source_failure_rate__1h=0.98,
        source_failure_count__1h=200,
        source_attempt_count__1h=204,
        source_unique_user_count__1h=1,
    )
    assert status_of("PAD-BF-001", row, catalog) is RuleStatus.FIRED
    assert status_of("PAD-PS-001", row, catalog) is RuleStatus.NOT_FIRED
    assert status_of("PAD-CS-001", row, catalog) is not RuleStatus.FIRED


def test_spraying_does_not_trigger_concentrated_brute_force(
    catalog: FeatureCatalog,
) -> None:
    """One guess each against forty accounts: no pair accumulates failures."""
    row = factories.spraying_row(
        catalog,
        pair_failure_count__5m=1,
        pair_failure_rate__5m=1.0,
        user_failure_count__5m=1,
        prior_consecutive_user_failures=1,
        source_unique_user_count__5m=40,
    )
    assert status_of("PAD-PS-001", row, catalog) is RuleStatus.FIRED
    assert status_of("PAD-BF-001", row, catalog) is RuleStatus.NOT_FIRED


def test_a_high_volume_spraying_source_still_fails_the_concentration_guard(
    catalog: FeatureCatalog,
) -> None:
    """Even with pair failures high, broad fan-out keeps PAD-BF-001 silent."""
    row = factories.spraying_row(
        catalog,
        pair_failure_count__5m=50,
        pair_failure_rate__5m=1.0,
        user_failure_count__5m=50,
        prior_consecutive_user_failures=50,
        source_unique_user_count__5m=40,
    )
    result_codes = (
        RULE_IMPLEMENTATIONS["PAD-BF-001"]
        .prepare(DetectionConfig(), catalog)
        .evaluate(row)
        .reason_codes
    )
    assert "SOURCE_TARGETS_TOO_MANY_ACCOUNTS" in result_codes


def test_one_source_brute_force_does_not_trigger_distributed_brute_force(
    catalog: FeatureCatalog,
) -> None:
    row = factories.brute_force_row(
        catalog,
        user_unique_source_count__1h=1,
        user_failure_count__1h=200,
        user_failure_rate__1h=0.99,
        pair_attempt_count__1h=200,
        source_unique_user_count__1h=1,
    )
    assert status_of("PAD-BF-001", row, catalog) is RuleStatus.FIRED
    assert status_of("PAD-DBF-001", row, catalog) is RuleStatus.NOT_FIRED


def test_distributed_low_volume_sources_do_not_trigger_concentrated_brute_force(
    catalog: FeatureCatalog,
) -> None:
    row = factories.distributed_row(
        catalog,
        pair_failure_count__5m=2,
        pair_failure_rate__5m=1.0,
        user_failure_count__5m=2,
        prior_consecutive_user_failures=2,
        source_unique_user_count__5m=1,
    )
    assert status_of("PAD-DBF-001", row, catalog) is RuleStatus.FIRED
    assert status_of("PAD-BF-001", row, catalog) is RuleStatus.NOT_FIRED


def test_spraying_does_not_trigger_distributed_brute_force(
    catalog: FeatureCatalog,
) -> None:
    row = factories.spraying_row(
        catalog,
        user_unique_source_count__1h=20,
        user_failure_count__1h=60,
        user_failure_rate__1h=0.95,
        pair_attempt_count__1h=1,
    )
    assert status_of("PAD-PS-001", row, catalog) is RuleStatus.FIRED
    assert status_of("PAD-DBF-001", row, catalog) is RuleStatus.NOT_FIRED


# ---------------------------------------------------------------------------
# Success-anchored rules
# ---------------------------------------------------------------------------


def test_a_normal_successful_login_triggers_nothing(catalog: FeatureCatalog) -> None:
    row = factories.quiet_row(
        catalog,
        current_authentication_outcome="success",
        previous_user_outcome="success",
        seconds_since_user_previous_failure=None,
    )
    assert fired_rules(row, catalog) == set()


def test_a_failure_burst_and_its_closing_success_never_share_an_anchor(
    catalog: FeatureCatalog,
) -> None:
    """One anchor cannot be both a failure feeding BF-001 and BF-002's success."""
    failing = factories.brute_force_row(
        catalog,
        prior_failures_since_pair_success=20,
        prior_failures_since_user_success=20,
        previous_user_outcome="failure",
        seconds_since_user_previous_failure=5.0,
    )
    assert status_of("PAD-BF-001", failing, catalog) is RuleStatus.FIRED
    assert status_of("PAD-BF-002", failing, catalog) is RuleStatus.NOT_FIRED

    succeeding = factories.success_after_burst_row(
        catalog,
        pair_failure_count__5m=12,
        pair_failure_rate__5m=0.92,
        user_failure_count__5m=14,
        prior_consecutive_user_failures=9,
        source_unique_user_count__5m=1,
    )
    assert status_of("PAD-BF-002", succeeding, catalog) is RuleStatus.FIRED


def test_novelty_alone_triggers_neither_takeover_nor_impossible_travel(
    catalog: FeatureCatalog,
) -> None:
    row = factories.quiet_row(
        catalog,
        current_authentication_outcome="success",
        is_new_device_for_user=True,
        is_new_source_for_user=True,
        is_new_country_for_user=True,
        is_new_application_for_user=True,
        is_new_auth_method_for_user=True,
    )
    assert fired_rules(row, catalog) == set()


def test_impossible_travel_does_not_trigger_the_takeover_rule_on_novelty_alone(
    catalog: FeatureCatalog,
) -> None:
    row = factories.impossible_travel_row(
        catalog,
        user_in_baseline=True,
        is_new_country_for_user=True,
        is_new_source_for_user=True,
    )
    assert status_of("PAD-GEO-001", row, catalog) is RuleStatus.FIRED
    assert status_of("PAD-ATO-001", row, catalog) is RuleStatus.NOT_FIRED


def test_missing_location_triggers_no_geospatial_finding(
    catalog: FeatureCatalog,
) -> None:
    row = factories.takeover_row(catalog)
    assert status_of("PAD-GEO-001", row, catalog) is RuleStatus.INSUFFICIENT_DATA
    assert "PAD-GEO-001" not in fired_rules(row, catalog)


def test_one_mfa_challenge_triggers_no_mfa_finding(catalog: FeatureCatalog) -> None:
    row = factories.quiet_row(
        catalog,
        user_attempt_count__15m=1,
        user_challenge_count__15m=1,
        user_mfa_failure_count__15m=0,
        current_mfa_outcome="passed",
    )
    assert status_of("PAD-MFA-001", row, catalog) is RuleStatus.INSUFFICIENT_DATA
    assert fired_rules(row, catalog) == set()


def test_irregular_timing_triggers_no_automation_finding(
    catalog: FeatureCatalog,
) -> None:
    row = factories.bot_row(
        catalog, source_interarrival_coefficient_of_variation__15m=2.2
    )
    assert status_of("PAD-BOT-001", row, catalog) is RuleStatus.NOT_FIRED
    assert "PAD-BOT-001" not in fired_rules(row, catalog)


# ---------------------------------------------------------------------------
# Ground truth is unreachable from any rule
# ---------------------------------------------------------------------------


def test_novel_anomaly_ground_truth_is_never_read_by_any_rule(
    catalog: FeatureCatalog,
) -> None:
    """The holdout scenario must be invisible to the whole registry.

    Its whole purpose is to measure detection of behaviour no rule was written
    for.  A rule that could read the label would make that measurement
    meaningless.
    """
    for builder in (
        factories.quiet_row,
        factories.brute_force_row,
        factories.spraying_row,
        factories.takeover_row,
    ):
        row = builder(catalog)
        baseline = fired_rules(row, catalog)
        poisoned = {
            **row,
            "scenario": "novel_anomaly_holdout",
            "attack_class": "novel_anomaly",
            "malicious": True,
            "label": 1,
            "supervised_training_eligible": False,
        }
        assert fired_rules(poisoned, catalog) == baseline


def test_no_rule_declares_a_novel_anomaly_category() -> None:
    from password_attack_detector.detection.catalog import RULE_CATALOG

    for spec in RULE_CATALOG.specs:
        assert "novel" not in str(spec.attack_category)
        assert "holdout" not in str(spec.attack_category)


# ---------------------------------------------------------------------------
# Every scenario fires the rule it was written for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_id", "builder"),
    [
        ("PAD-ATO-001", factories.takeover_row),
        ("PAD-BF-001", factories.brute_force_row),
        ("PAD-BF-002", factories.success_after_burst_row),
        ("PAD-BOT-001", factories.bot_row),
        ("PAD-CS-001", factories.stuffing_row),
        ("PAD-DBF-001", factories.distributed_row),
        ("PAD-GEO-001", factories.impossible_travel_row),
        ("PAD-MFA-001", factories.mfa_row),
        ("PAD-PS-001", factories.spraying_row),
    ],
)
def test_each_scenario_fires_its_own_rule(
    rule_id: str, builder: Any, catalog: FeatureCatalog
) -> None:
    assert rule_id in fired_rules(builder(catalog), catalog)
