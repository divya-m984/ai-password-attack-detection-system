"""Tests for alert construction, grouping, suppression, and escalation.

Two things are being proved here.  The first is that grouping and suppression
are deterministic: same input, same alerts, whatever order the assessments
arrive in and whatever order the registry happens to iterate.  The second is
the one that matters operationally -- **suppression never hides materially new
risk, and never loses an event without leaving a count behind**.  The
accounting identity is asserted directly, and every boundary the lifecycle
turns on is pinned at exact equality and one microsecond either side.

Alongside those runs a continuous check that no pseudonymous scope value
escapes the single field permitted to carry one.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from password_attack_detector.detection import alerts as alerts_module
from password_attack_detector.detection import engine as engine_module
from password_attack_detector.detection import scoring as scoring_module
from password_attack_detector.detection.alerts import (
    ALERTING_VERSION,
    AlertBuilder,
    AlertingResult,
    EntityScopeTable,
    build_entity_scope_table,
)
from password_attack_detector.detection.config import (
    AlertingConfig,
    DetectionConfig,
    SeverityThresholds,
)
from password_attack_detector.detection.engine import DetectionEngine
from password_attack_detector.detection.enums import (
    AlertGroupingMode,
    AttackCategory,
    CorrelationGroup,
    ScopeKind,
    Severity,
)
from password_attack_detector.detection.schemas import (
    EntityScopeRecord,
    RiskAssessment,
)
from password_attack_detector.detection.scoring import RiskScorer
from tests.unit.detection import factories
from tests.unit.detection.factories import fired_detection, scope_record

WHEN = factories.WHEN

#: Category and group pairs, one per correlation group, used to build
#: assessments directly rather than through the whole rule pipeline.
BRUTE_FORCE = (AttackCategory.BRUTE_FORCE, "PAD-BF-001")
SPRAYING = (AttackCategory.PASSWORD_SPRAYING, "PAD-PS-001")
STUFFING = (AttackCategory.CREDENTIAL_STUFFING, "PAD-CS-001")
TAKEOVER = (AttackCategory.ACCOUNT_TAKEOVER_INDICATOR, "PAD-ATO-001")
TRAVEL = (AttackCategory.IMPOSSIBLE_TRAVEL_INDICATOR, "PAD-GEO-001")


def assessment(
    anchor_event_id: str,
    *,
    minutes: float = 0.0,
    micros: int = 0,
    scenario: tuple[AttackCategory, str] = BRUTE_FORCE,
    risk_score: float = 70.0,
    severity: Severity = Severity.HIGH,
    rule_ids: tuple[str, ...] | None = None,
) -> RiskAssessment:
    """Build one risk assessment directly.

    Bypasses the scorer on purpose: these tests are about what the builder does
    with a score, and deriving an exact score through the rule arithmetic would
    make them tests of the rules instead.
    """
    category, default_rule = scenario
    rules = tuple(sorted(rule_ids or (default_rule,)))
    return RiskAssessment(
        anchor_event_id=anchor_event_id,
        anchor_event_time=WHEN + timedelta(minutes=minutes, microseconds=micros),
        risk_score=risk_score,
        severity=severity,
        primary_attack_category=category,
        contributing_categories=(category,),
        fired_rule_count=len(rules),
        fired_rule_ids=rules,
        scoring_version="1.0.0",
    )


def build(assessments: list[RiskAssessment], **config_kwargs: object) -> AlertingResult:
    """Run the builder over *assessments* under a configured alerting policy."""
    config = DetectionConfig(alerting=AlertingConfig(**config_kwargs))  # type: ignore[arg-type]
    return AlertBuilder(config).build(assessments)


# ---------------------------------------------------------------------------
# The alert gates
# ---------------------------------------------------------------------------


def test_no_fired_rules_produces_no_alert() -> None:
    zero = RiskAssessment(
        anchor_event_id="a1",
        anchor_event_time=WHEN,
        risk_score=0.0,
        severity=Severity.LOW,
        scoring_version="1.0.0",
    )
    result = build([zero])
    assert result.alerts == ()
    assert result.stats.qualifying_count == 0
    assert result.stats.suppressed_total == 0


def test_a_score_below_the_floor_produces_no_alert_but_is_counted() -> None:
    result = build([assessment("a1", risk_score=5.0, severity=Severity.LOW)])
    assert result.alerts == ()
    assert result.stats.below_score_floor_count == 1
    assert result.stats.suppressed_by_category[AttackCategory.BRUTE_FORCE] == 1


def test_the_score_floor_is_inclusive() -> None:
    """Exactly at the floor is an alert; one representable step below is not."""
    floor = 10.0
    at = build([assessment("a1", risk_score=floor, severity=Severity.LOW)])
    assert at.alert_count == 1

    import math

    below = build(
        [assessment("a1", risk_score=math.nextafter(floor, 0.0), severity=Severity.LOW)]
    )
    assert below.alert_count == 0
    assert below.stats.below_score_floor_count == 1


def test_a_low_severity_assessment_becomes_a_low_alert() -> None:
    """LOW is a valid alert severity; nothing rejects it for being LOW."""
    result = build([assessment("a1", risk_score=12.0, severity=Severity.LOW)])
    assert result.alert_count == 1
    alert = result.alerts[0]
    assert alert.initial_severity is Severity.LOW
    assert alert.current_severity is Severity.LOW


@pytest.mark.parametrize(
    ("floor", "emitted"),
    [
        (Severity.LOW, True),
        (Severity.MEDIUM, False),
        (Severity.HIGH, False),
        (Severity.CRITICAL, False),
    ],
)
def test_every_min_alert_severity_setting_behaves_as_configured(
    floor: Severity, emitted: bool
) -> None:
    result = build(
        [assessment("a1", risk_score=12.0, severity=Severity.LOW)],
        min_alert_severity=floor,
    )
    assert (result.alert_count == 1) is emitted
    assert result.stats.below_severity_floor_count == (0 if emitted else 1)


def test_raising_the_severity_floor_leaves_the_assessment_untouched() -> None:
    """Withholding an alert is a reporting decision, not a data change."""
    source = assessment("a1", risk_score=12.0, severity=Severity.LOW)
    result = build([source], min_alert_severity=Severity.MEDIUM)
    assert result.alert_count == 0
    assert source.risk_score == 12.0
    assert source.severity is Severity.LOW


def test_a_higher_severity_clears_a_raised_floor() -> None:
    result = build(
        [assessment("a1", risk_score=70.0, severity=Severity.HIGH)],
        min_alert_severity=Severity.MEDIUM,
    )
    assert result.alert_count == 1


def test_no_code_path_rejects_an_alert_for_being_low_alone() -> None:
    """A LOW alert is withheld only by a configured floor, never by its band."""
    source = inspect.getsource(alerts_module)
    assert "Severity.LOW" not in source


# ---------------------------------------------------------------------------
# Grouping and deduplication
# ---------------------------------------------------------------------------


def test_repeated_detections_inside_the_window_become_one_alert() -> None:
    result = build(
        [
            assessment("a1", minutes=0),
            assessment("a2", minutes=5),
            assessment("a3", minutes=10),
        ]
    )
    assert result.alert_count == 1
    alert = result.alerts[0]
    assert alert.contributing_event_count == 3
    assert alert.first_seen == WHEN
    assert alert.last_seen == WHEN + timedelta(minutes=10)
    assert result.stats.grouped_detection_count == 3


def test_the_grouping_window_boundary_is_inclusive() -> None:
    """Exactly one window after the last contribution still groups.

    One microsecond later the alert closes instead.  What happens to that
    second assessment then depends on the cooldown, which is a separate
    boundary -- so this test looks at whether it was *absorbed*, not at how
    many alerts came out.
    """
    at = build([assessment("a1", minutes=0), assessment("a2", minutes=15)])
    assert at.alert_count == 1
    assert at.alerts[0].contributing_event_count == 2

    past = build([assessment("a1", minutes=0), assessment("a2", minutes=15, micros=1)])
    assert past.alerts[0].contributing_event_count == 1
    assert past.stats.grouped_detection_count == 1


def test_first_seen_never_moves_and_last_seen_only_advances() -> None:
    result = build(
        [
            assessment("a1", minutes=0),
            assessment("a2", minutes=3),
            assessment("a3", minutes=9),
        ]
    )
    alert = result.alerts[0]
    assert alert.first_seen == WHEN
    assert alert.last_seen == WHEN + timedelta(minutes=9)


def test_contributing_rule_ids_are_the_sorted_union() -> None:
    result = build(
        [
            assessment("a1", minutes=0, rule_ids=("PAD-BF-001",)),
            assessment("a2", minutes=1, rule_ids=("PAD-BF-002", "PAD-DBF-001")),
        ]
    )
    assert result.alerts[0].contributing_rule_ids == (
        "PAD-BF-001",
        "PAD-BF-002",
        "PAD-DBF-001",
    )


def test_peak_and_aggregate_risk_describe_the_group() -> None:
    result = build(
        [
            assessment("a1", minutes=0, risk_score=60.0),
            assessment("a2", minutes=1, risk_score=90.0),
        ]
    )
    alert = result.alerts[0]
    assert alert.peak_risk_score == 90.0
    assert alert.aggregate_risk_score == pytest.approx(75.0)
    assert alert.aggregate_risk_score <= alert.peak_risk_score


def test_a_different_attack_category_is_never_merged() -> None:
    result = build(
        [
            assessment("a1", minutes=0, scenario=BRUTE_FORCE),
            assessment("a2", minutes=1, scenario=SPRAYING),
        ]
    )
    assert result.alert_count == 2
    assert {alert.attack_category for alert in result.alerts} == {
        AttackCategory.BRUTE_FORCE,
        AttackCategory.PASSWORD_SPRAYING,
    }


def test_a_different_correlation_group_is_never_merged() -> None:
    result = build(
        [
            assessment("a1", minutes=0, scenario=BRUTE_FORCE),
            assessment("a2", minutes=1, scenario=TRAVEL),
        ]
    )
    assert result.alert_count == 2
    assert {alert.correlation_group for alert in result.alerts} == {
        CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET,
        CorrelationGroup.LOCATION_MOVEMENT,
    }


def test_categories_in_one_group_still_alert_separately() -> None:
    """Spraying and stuffing share a group but describe different behaviour."""
    result = build(
        [
            assessment("a1", minutes=0, scenario=SPRAYING),
            assessment("a2", minutes=1, scenario=STUFFING),
        ]
    )
    assert result.alert_count == 2
    assert {alert.correlation_group for alert in result.alerts} == {
        CorrelationGroup.SOURCE_FANOUT
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_alert_identifiers_are_stable_across_runs() -> None:
    inputs = [assessment("a1", minutes=0), assessment("a2", minutes=40)]
    first = build(inputs)
    second = build(inputs)
    assert [alert.alert_id for alert in first.alerts] == [
        alert.alert_id for alert in second.alerts
    ]
    assert first.alerts == second.alerts


def test_input_order_does_not_change_the_alert_set() -> None:
    inputs = [
        assessment("a1", minutes=0, scenario=BRUTE_FORCE),
        assessment("a2", minutes=4, scenario=SPRAYING),
        assessment("a3", minutes=9, scenario=BRUTE_FORCE),
        assessment("a4", minutes=50, scenario=TRAVEL),
    ]
    forward = build(inputs)
    backward = build(list(reversed(inputs)))
    assert forward.alerts == backward.alerts
    assert forward.stats == backward.stats


def test_alert_identifiers_are_unique_within_a_run() -> None:
    inputs = [
        assessment(f"a{index}", minutes=index * 40, scenario=scenario)
        for index, scenario in enumerate([BRUTE_FORCE, SPRAYING, TRAVEL, TAKEOVER] * 2)
    ]
    result = build(inputs, cooldown=timedelta(seconds=1))
    identifiers = [alert.alert_id for alert in result.alerts]
    assert len(set(identifiers)) == len(identifiers)


def test_the_alert_identifier_depends_on_its_semantic_inputs() -> None:
    baseline = build([assessment("a1", scenario=BRUTE_FORCE)]).alerts[0]
    other_category = build([assessment("a1", scenario=SPRAYING)]).alerts[0]
    later = build([assessment("a1", minutes=99, scenario=BRUTE_FORCE)]).alerts[0]
    assert baseline.alert_id != other_category.alert_id
    assert baseline.alert_id != later.alert_id


def test_alert_identifiers_use_no_ambient_state() -> None:
    """Checked against executable source only.

    The module's own prose names ``uuid4`` and ``hash()`` in order to rule them
    out, so a raw text scan would flag the very docstring that documents the
    guarantee.  Strings and comments are stripped first.
    """
    code = _executable_source(alerts_module)
    for forbidden in (
        "uuid4",
        "uuid1",
        "time.time",
        "datetime.now",
        "getpid",
        "random",
        "socket",
        "gethostname",
        "hash(",
    ):
        assert forbidden not in code, forbidden


def _executable_source(module: object) -> str:
    """Return a module's source with comments and string literals removed."""
    import io
    import tokenize

    source = inspect.getsource(module)  # type: ignore[arg-type]
    kept: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in {tokenize.COMMENT, tokenize.STRING}:
            continue
        kept.append(token.string)
    return " ".join(kept)


def test_alerts_are_emitted_in_a_deterministic_order() -> None:
    result = build(
        [
            assessment("a1", minutes=0, scenario=SPRAYING),
            assessment("a2", minutes=0, scenario=BRUTE_FORCE),
            assessment("a3", minutes=0, scenario=TRAVEL),
        ]
    )
    keys = [(alert.first_seen, str(alert.attack_category)) for alert in result.alerts]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_a_higher_severity_escalates_an_open_alert() -> None:
    result = build(
        [
            assessment("a1", minutes=0, risk_score=45.0, severity=Severity.MEDIUM),
            assessment("a2", minutes=2, risk_score=90.0, severity=Severity.CRITICAL),
        ]
    )
    alert = result.alerts[0]
    assert alert.initial_severity is Severity.MEDIUM
    assert alert.current_severity is Severity.CRITICAL
    assert alert.escalation_count == 1
    assert result.stats.escalated_count == 1


def test_greater_peak_risk_updates_an_alert_without_a_severity_change() -> None:
    result = build(
        [
            assessment("a1", minutes=0, risk_score=66.0, severity=Severity.HIGH),
            assessment("a2", minutes=2, risk_score=80.0, severity=Severity.HIGH),
        ]
    )
    alert = result.alerts[0]
    assert alert.peak_risk_score == 80.0
    assert alert.current_severity is Severity.HIGH
    assert alert.escalation_count == 0
    assert alert.contributing_event_count == 2


def test_a_lower_severity_never_reduces_an_alert() -> None:
    result = build(
        [
            assessment("a1", minutes=0, risk_score=90.0, severity=Severity.CRITICAL),
            assessment("a2", minutes=2, risk_score=45.0, severity=Severity.MEDIUM),
        ]
    )
    alert = result.alerts[0]
    assert alert.current_severity is Severity.CRITICAL
    assert alert.peak_risk_score == 90.0
    assert alert.escalation_count == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


def test_a_repeat_inside_the_cooldown_is_suppressed() -> None:
    result = build(
        [
            assessment("a1", minutes=0),
            # Past the grouping window, so the first alert closes; inside the
            # cooldown, so the repeat is suppressed rather than alerted.
            assessment("a2", minutes=20),
        ]
    )
    assert result.alert_count == 1
    assert result.stats.suppressed_by_cooldown_count == 1
    assert result.alerts[0].suppressed_event_count == 1


def test_the_cooldown_boundary_is_inclusive() -> None:
    """Exactly one cooldown after the last contribution is still suppressed."""
    at = build([assessment("a1", minutes=0), assessment("a2", minutes=30)])
    assert at.alert_count == 1
    assert at.stats.suppressed_by_cooldown_count == 1

    past = build([assessment("a1", minutes=0), assessment("a2", minutes=30, micros=1)])
    assert past.alert_count == 2
    assert past.stats.suppressed_by_cooldown_count == 0


def test_a_higher_severity_bypasses_the_cooldown() -> None:
    """Suppression must not hide a finding worse than the one it repeats."""
    result = build(
        [
            assessment("a1", minutes=0, risk_score=45.0, severity=Severity.MEDIUM),
            assessment("a2", minutes=20, risk_score=95.0, severity=Severity.CRITICAL),
        ]
    )
    assert result.alert_count == 2
    assert result.stats.suppressed_by_cooldown_count == 0
    assert result.stats.escalated_count == 1


def test_greater_peak_risk_bypasses_the_cooldown() -> None:
    result = build(
        [
            assessment("a1", minutes=0, risk_score=66.0, severity=Severity.HIGH),
            assessment("a2", minutes=20, risk_score=80.0, severity=Severity.HIGH),
        ]
    )
    assert result.alert_count == 2
    assert result.stats.escalated_count == 1


def test_an_equal_finding_inside_the_cooldown_stays_suppressed() -> None:
    result = build(
        [
            assessment("a1", minutes=0, risk_score=70.0),
            assessment("a2", minutes=20, risk_score=70.0),
        ]
    )
    assert result.alert_count == 1
    assert result.stats.suppressed_by_cooldown_count == 1


def test_disabling_the_escalation_bypass_suppresses_even_a_worse_finding() -> None:
    result = build(
        [
            assessment("a1", minutes=0, risk_score=45.0, severity=Severity.MEDIUM),
            assessment("a2", minutes=20, risk_score=95.0, severity=Severity.CRITICAL),
        ],
        escalation_bypasses_cooldown=False,
    )
    assert result.alert_count == 1
    assert result.stats.suppressed_by_cooldown_count == 1


def test_a_new_category_is_not_suppressed_by_an_unrelated_cooldown() -> None:
    result = build(
        [
            assessment("a1", minutes=0, scenario=BRUTE_FORCE),
            assessment("a2", minutes=20, scenario=SPRAYING),
        ]
    )
    assert result.alert_count == 2
    assert result.stats.suppressed_total == 0


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


#: An escalating stream: each assessment outranks the last, so every one
#: bypasses the cooldown.  This is the only path that can open many alerts for
#: one group in quick succession, and therefore the only thing the per-group
#: limit exists to backstop.
def _escalating(count: int, *, step_minutes: int = 20) -> list[RiskAssessment]:
    return [
        assessment(
            f"a{index}", minutes=index * step_minutes, risk_score=50.0 + index * 5
        )
        for index in range(count)
    ]


def test_an_escalating_stream_opens_an_alert_each_time_without_a_limit() -> None:
    """The behaviour the limit exists to bound, established first."""
    result = build(_escalating(5), max_alerts_per_group_per_window=99)
    assert result.alert_count == 5
    assert result.stats.escalated_count == 4


def test_the_per_group_limit_is_enforced() -> None:
    result = build(
        _escalating(5),
        max_alerts_per_group_per_window=2,
        alert_limit_window=timedelta(hours=4),
    )
    assert result.alert_count == 2
    assert result.stats.suppressed_by_rate_limit_count == 3


def test_the_limit_resets_once_its_window_passes() -> None:
    inputs = [
        *_escalating(2),
        assessment("late", minutes=600, risk_score=95.0, severity=Severity.CRITICAL),
    ]
    result = build(
        inputs,
        max_alerts_per_group_per_window=2,
        alert_limit_window=timedelta(hours=1),
    )
    assert result.alert_count == 3
    assert result.stats.suppressed_by_rate_limit_count == 0


def test_a_limit_of_one_still_emits_an_alert() -> None:
    result = build(
        _escalating(4),
        max_alerts_per_group_per_window=1,
        alert_limit_window=timedelta(hours=4),
    )
    assert result.alert_count == 1
    assert result.stats.suppressed_by_rate_limit_count == 3


def test_the_limit_horizon_is_configured_independently_of_grouping() -> None:
    """A grouping-window horizon would be vacuous, so it is its own field."""
    config = AlertingConfig()
    assert config.alert_limit_window > config.grouping_window


def test_rate_limited_events_are_still_accounted_for() -> None:
    result = build(
        _escalating(5),
        max_alerts_per_group_per_window=2,
        alert_limit_window=timedelta(hours=4),
    )
    stats = result.stats
    assert stats.qualifying_count == (
        stats.grouped_detection_count
        + stats.suppressed_by_cooldown_count
        + stats.suppressed_by_rate_limit_count
    )
    assert sum(alert.suppressed_event_count for alert in result.alerts) == 3


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_every_qualifying_assessment_is_grouped_or_suppressed() -> None:
    """The identity that makes suppression safe to reason about."""
    inputs = [
        assessment(f"a{index}", minutes=index * 7, risk_score=40.0 + index)
        for index in range(12)
    ]
    inputs.append(assessment("low", minutes=200, risk_score=2.0, severity=Severity.LOW))
    result = build(inputs, max_alerts_per_group_per_window=2)
    stats = result.stats

    assert stats.assessment_count == len(inputs)
    assert stats.qualifying_count == (
        stats.grouped_detection_count
        + stats.suppressed_by_cooldown_count
        + stats.suppressed_by_rate_limit_count
    )
    assert stats.below_score_floor_count == 1
    assert stats.qualifying_count + stats.below_score_floor_count == len(inputs)


def test_alert_counts_split_across_the_two_grouping_modes() -> None:
    result = build([assessment("a1")])
    assert (
        result.stats.entity_scoped_count + result.stats.category_scoped_count
        == result.stats.alert_count
    )


def test_statistics_expose_no_group_key_or_identifier() -> None:
    result = build([assessment("anchor-0001"), assessment("anchor-0002", minutes=90)])
    rendered = result.stats.model_dump_json()
    assert "anchor-" not in rendered
    assert "PAD-BF-001" not in rendered


def test_operation_counters_show_a_single_indexed_pass() -> None:
    inputs = [assessment(f"a{index}", minutes=index) for index in range(20)]
    result = build(inputs)
    assert result.operations.assessments_scanned == 20
    assert result.operations.key_lookups == 20
    assert result.operations.max_open_groups == 1


def test_grouping_cost_grows_linearly_with_input() -> None:
    """No quadratic scan: lookups track assessments, not their square."""
    small = build([assessment(f"a{index}", minutes=index) for index in range(10)])
    large = build([assessment(f"a{index}", minutes=index) for index in range(100)])
    assert small.operations.key_lookups == 10
    assert large.operations.key_lookups == 100


def test_open_group_state_is_bounded_by_distinct_groups() -> None:
    inputs = [
        assessment(f"a{index}", minutes=index, scenario=scenario)
        for index, scenario in enumerate([BRUTE_FORCE, SPRAYING, TRAVEL] * 10)
    ]
    result = build(inputs)
    assert result.operations.max_open_groups == 3
    assert result.operations.distinct_group_count == 3


# ---------------------------------------------------------------------------
# Entity scope
# ---------------------------------------------------------------------------


def scoped(
    assessments: list[RiskAssessment],
    records: Sequence[EntityScopeRecord],
    **config_kwargs: Any,
) -> AlertingResult:
    """Run the builder with a scope table."""
    config = DetectionConfig(alerting=AlertingConfig(**config_kwargs))
    return AlertBuilder(config).build(
        assessments, entity_scope=build_entity_scope_table(records)
    )


def test_no_scope_table_means_category_scoped_grouping() -> None:
    result = build([assessment("a1")])
    assert result.stats.grouping_mode is AlertGroupingMode.CATEGORY_SCOPED
    alert = result.alerts[0]
    assert alert.grouping_mode is AlertGroupingMode.CATEGORY_SCOPED
    assert alert.scope_kind is ScopeKind.NONE
    assert alert.scope_value is None
    assert result.stats.category_scoped_count == 1


def test_a_user_dimension_group_splits_on_the_user_scope() -> None:
    """Two accounts under brute force are two alerts, not one."""
    result = scoped(
        [
            assessment("a1", minutes=0, scenario=BRUTE_FORCE),
            assessment("a2", minutes=1, scenario=BRUTE_FORCE),
        ],
        [
            scope_record("a1", user="1", source="9"),
            scope_record("a2", user="2", source="9"),
        ],
    )
    assert result.alert_count == 2
    assert {alert.scope_kind for alert in result.alerts} == {ScopeKind.USER}
    assert result.stats.entity_scoped_count == 2


def test_a_source_dimension_group_splits_on_the_source_scope() -> None:
    result = scoped(
        [
            assessment("a1", minutes=0, scenario=SPRAYING),
            assessment("a2", minutes=1, scenario=SPRAYING),
        ],
        [
            scope_record("a1", user="1", source="7"),
            scope_record("a2", user="2", source="8"),
        ],
    )
    assert result.alert_count == 2
    assert {alert.scope_kind for alert in result.alerts} == {ScopeKind.SOURCE}


def test_one_source_serving_many_accounts_groups_into_one_spraying_alert() -> None:
    """The point of a source dimension: fan-out is one finding, not many."""
    result = scoped(
        [
            assessment(f"a{index}", minutes=index, scenario=SPRAYING)
            for index in range(5)
        ],
        [scope_record(f"a{index}", user=str(index), source="7") for index in range(5)],
    )
    assert result.alert_count == 1
    assert result.alerts[0].contributing_event_count == 5


def test_the_two_dimensions_are_consumed_independently() -> None:
    """A user-scoped group and a source-scoped group split differently."""
    result = scoped(
        [
            assessment("a1", minutes=0, scenario=BRUTE_FORCE),
            assessment("a2", minutes=1, scenario=BRUTE_FORCE),
            assessment("a3", minutes=2, scenario=SPRAYING),
            assessment("a4", minutes=3, scenario=SPRAYING),
        ],
        [
            scope_record("a1", user="1", source="7"),
            scope_record("a2", user="2", source="7"),
            scope_record("a3", user="1", source="7"),
            scope_record("a4", user="2", source="7"),
        ],
    )
    by_kind = {alert.scope_kind: alert for alert in result.alerts}
    # Brute force split into two on the user; spraying stayed one on the source.
    assert result.alert_count == 3
    assert by_kind[ScopeKind.SOURCE].contributing_event_count == 2


def test_a_missing_dimension_degrades_that_group_only() -> None:
    result = scoped(
        [
            assessment("a1", minutes=0, scenario=BRUTE_FORCE),
            assessment("a2", minutes=1, scenario=SPRAYING),
        ],
        [
            scope_record("a1", source="7"),  # no user scope
            scope_record("a2", user="1", source="7"),
        ],
    )
    assert result.stats.scope_missing_count == 1
    assert result.stats.grouping_mode is AlertGroupingMode.ENTITY_SCOPED
    modes = {alert.attack_category: alert.grouping_mode for alert in result.alerts}
    assert modes[AttackCategory.BRUTE_FORCE] is AlertGroupingMode.CATEGORY_SCOPED
    assert modes[AttackCategory.PASSWORD_SPRAYING] is AlertGroupingMode.ENTITY_SCOPED
    assert result.stats.entity_scoped_count == 1
    assert result.stats.category_scoped_count == 1


def test_a_missing_dimension_does_not_fail_the_run_by_default() -> None:
    result = scoped(
        [assessment("a1", scenario=BRUTE_FORCE)], [scope_record("a1", source="7")]
    )
    assert result.alert_count == 1
    assert result.stats.scope_missing_count == 1


def test_strict_scope_mode_fails_on_a_missing_dimension() -> None:
    from password_attack_detector.exceptions import DataValidationError

    with pytest.raises(DataValidationError, match="strict"):
        scoped(
            [assessment("a1", scenario=BRUTE_FORCE)],
            [scope_record("a1", source="7")],
            strict_scope=True,
        )


def test_a_duplicate_scope_anchor_is_a_hard_failure() -> None:
    from password_attack_detector.exceptions import DataValidationError

    with pytest.raises(DataValidationError, match="duplicate"):
        build_entity_scope_table(
            [scope_record("a1", user="1"), scope_record("a1", user="2")]
        )


def test_an_unmatched_scope_row_is_refused() -> None:
    from password_attack_detector.exceptions import DataValidationError

    with pytest.raises(DataValidationError, match="does not match"):
        scoped(
            [assessment("a1")],
            [scope_record("a1", user="1"), scope_record("a2", user="2")],
        )


def test_an_assessment_without_a_scope_row_is_refused() -> None:
    from password_attack_detector.exceptions import DataValidationError

    with pytest.raises(DataValidationError, match="does not match"):
        scoped(
            [assessment("a1"), assessment("a2", minutes=1)],
            [scope_record("a1", user="1")],
        )


def test_the_relationship_error_names_counts_only() -> None:
    from password_attack_detector.exceptions import DataValidationError

    with pytest.raises(DataValidationError) as caught:
        scoped([assessment("anchor-0001")], [scope_record("anchor-0002", user="1")])
    message = str(caught.value)
    assert "anchor-0001" not in message
    assert "anchor-0002" not in message
    assert "u:" not in message


def test_scoped_grouping_is_order_invariant() -> None:
    assessments = [
        assessment(f"a{index}", minutes=index, scenario=BRUTE_FORCE)
        for index in range(6)
    ]
    records = [scope_record(f"a{index}", user=str(index % 2)) for index in range(6)]
    forward = scoped(assessments, records)
    backward = scoped(list(reversed(assessments)), list(reversed(records)))
    assert forward.alerts == backward.alerts
    assert forward.stats == backward.stats


def test_scoped_alert_identifiers_are_stable_across_runs() -> None:
    assessments = [assessment("a1", scenario=BRUTE_FORCE)]
    records = [scope_record("a1", user="1")]
    assert [alert.alert_id for alert in scoped(assessments, records).alerts] == [
        alert.alert_id for alert in scoped(assessments, records).alerts
    ]


def test_two_scope_values_produce_different_alert_identifiers() -> None:
    first = scoped([assessment("a1")], [scope_record("a1", user="1")]).alerts[0]
    second = scoped([assessment("a1")], [scope_record("a1", user="2")]).alerts[0]
    assert first.alert_id != second.alert_id


# ---------------------------------------------------------------------------
# Scope confinement
# ---------------------------------------------------------------------------


def test_the_scope_value_reaches_exactly_one_field() -> None:
    result = scoped(
        [assessment("a1", scenario=BRUTE_FORCE)], [scope_record("a1", user="abc")]
    )
    alert = result.alerts[0]
    secret = alert.scope_value
    assert secret is not None

    for rendered in (
        result.stats.model_dump_json(),
        repr(result.stats),
        repr(result.operations),
        alert.model_dump_json(exclude={"scope_value"}),
    ):
        assert secret not in rendered


def test_the_scope_table_repr_is_redacted() -> None:
    table = build_entity_scope_table([scope_record("a1", user="abc", source="def")])
    assert "abc" not in repr(table)
    assert "def" not in repr(table)
    assert "anchor" not in repr(table).replace("anchor_count", "")
    assert isinstance(table, EntityScopeTable)
    assert table.anchor_count == 1


def test_the_scope_record_repr_is_redacted() -> None:
    record = scope_record("a1", user="abc", source="def")
    assert "abc" not in repr(record)
    assert "def" not in repr(record)
    assert "a1" not in repr(record)


@pytest.mark.parametrize("module", [engine_module, scoring_module])
def test_the_engine_and_scorer_never_import_the_scope_reader(module: object) -> None:
    source = inspect.getsource(module)  # type: ignore[arg-type]
    for forbidden in ("EntityScope", "detection.alerts", "scope_for", "AlertBuilder"):
        assert forbidden not in source


def test_the_engine_and_scorer_signatures_reject_a_scope_argument() -> None:
    for callable_object in (
        DetectionEngine.__init__,
        DetectionEngine.run,
        RiskScorer.__init__,
        RiskScorer.score,
        RiskScorer.score_anchor,
    ):
        parameters = inspect.signature(callable_object).parameters
        assert not any("scope" in name.lower() for name in parameters)


def test_only_the_builder_accepts_a_scope_table() -> None:
    parameters = inspect.signature(AlertBuilder.build).parameters
    assert "entity_scope" in parameters


# ---------------------------------------------------------------------------
# Contributing-rule metadata
# ---------------------------------------------------------------------------


def test_supplied_detections_are_cross_checked_against_the_assessments() -> None:
    from password_attack_detector.exceptions import DataValidationError

    builder = AlertBuilder(DetectionConfig())
    with pytest.raises(DataValidationError, match="does not match"):
        builder.build(
            [assessment("a1", rule_ids=("PAD-BF-001",))],
            detections=[fired_detection("PAD-PS-001", anchor_event_id="a1")],
        )


def test_matching_detections_are_accepted() -> None:
    builder = AlertBuilder(DetectionConfig())
    result = builder.build(
        [assessment("a1", rule_ids=("PAD-BF-001",))],
        detections=[fired_detection("PAD-BF-001", anchor_event_id="a1")],
    )
    assert result.alert_count == 1


# ---------------------------------------------------------------------------
# End to end over the real pipeline
# ---------------------------------------------------------------------------


def test_the_whole_pipeline_is_deterministic_and_accounted_for() -> None:
    catalog = factories.feature_catalog()
    config = DetectionConfig()
    engine = DetectionEngine(config, feature_catalog=catalog)
    builders = [
        factories.brute_force_row,
        factories.spraying_row,
        factories.stuffing_row,
        factories.quiet_row,
    ]
    rows = [
        builders[index % len(builders)](
            catalog,
            anchor_event_id=f"anchor-{index:04d}",
            anchor_event_time=WHEN + timedelta(minutes=index * 3),
        )
        for index in range(12)
    ]

    scored = RiskScorer(config).score(engine.run_diagnostic(rows))
    result = AlertBuilder(config).build(
        scored.assessments, detections=list(engine.run(rows).fired_detections)
    )

    assert result.alert_count > 0
    assert result.stats.qualifying_count == (
        result.stats.grouped_detection_count
        + result.stats.suppressed_by_cooldown_count
        + result.stats.suppressed_by_rate_limit_count
    )
    shuffled = AlertBuilder(config).build(list(reversed(scored.assessments)))
    assert shuffled.alerts == AlertBuilder(config).build(scored.assessments).alerts


def test_the_alerting_version_is_recorded_in_the_identifier_derivation() -> None:
    assert ALERTING_VERSION == "1.0.0"
    assert "ALERTING_VERSION" in inspect.getsource(alerts_module._alert_identifier)


def test_an_empty_input_produces_an_empty_result() -> None:
    result = build([])
    assert result.alerts == ()
    assert result.stats.assessment_count == 0
    assert result.alert_count == 0


def test_a_naive_timestamp_cannot_reach_an_alert() -> None:
    """Every alert timestamp is timezone-aware, as the schema requires."""
    result = build([assessment("a1")])
    for value in (result.alerts[0].first_seen, result.alerts[0].last_seen):
        assert value.tzinfo is not None
        assert value.utcoffset() == timedelta(0)


def test_severity_thresholds_do_not_affect_grouping() -> None:
    """Grouping keys are categorical; the severity ladder only gates."""
    config = DetectionConfig(severity_thresholds=SeverityThresholds(medium=5.0))
    result = AlertBuilder(config).build([assessment("a1"), assessment("a2", minutes=2)])
    assert result.alert_count == 1


def test_the_reference_timestamp_is_fixed_not_read_from_the_clock() -> None:
    """No assertion in this module depends on when it ran."""
    assert datetime(2026, 3, 1, 12, 0, tzinfo=UTC) == WHEN


# ---------------------------------------------------------------------------
# Scope table internals
# ---------------------------------------------------------------------------


def test_the_scope_table_reports_its_size() -> None:
    table = build_entity_scope_table(
        [scope_record("a1", user="1"), scope_record("a2", user="2")]
    )
    assert len(table) == 2
    assert table.anchor_count == 2


def test_an_absent_anchor_has_no_scope() -> None:
    table = build_entity_scope_table([scope_record("a1", user="1")])
    assert table.scope_for("missing", ScopeKind.USER) is None


def test_the_none_dimension_never_yields_a_scope_value() -> None:
    """Asking for no dimension returns nothing, not an arbitrary field."""
    table = build_entity_scope_table([scope_record("a1", user="1", source="2")])
    assert table.scope_for("a1", ScopeKind.NONE) is None
    assert table.scope_for("a1", ScopeKind.USER) is not None
    assert table.scope_for("a1", ScopeKind.SOURCE) is not None


def test_a_correlation_group_without_a_scope_dimension_is_refused() -> None:
    """The coverage validator prevents this; the guard proves it is wired."""
    from password_attack_detector.exceptions import DetectionConfigurationError

    config = DetectionConfig()
    crippled = config.alerting.model_copy(
        update={
            "scope_dimension": {
                group: kind
                for group, kind in config.alerting.scope_dimension.items()
                if group is not CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET
            }
        }
    )
    builder = AlertBuilder(config.model_copy(update={"alerting": crippled}))
    with pytest.raises(DetectionConfigurationError, match="no scope dimension"):
        builder.build(
            [assessment("a1", scenario=BRUTE_FORCE)],
            entity_scope=build_entity_scope_table([scope_record("a1", user="1")]),
        )


def test_a_category_spanning_two_correlation_groups_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alert's group must not depend on which rule ranked first."""
    from password_attack_detector.detection import catalog as catalog_module
    from password_attack_detector.exceptions import DetectionConfigurationError

    original = catalog_module.RULE_CATALOG
    conflicting = original.get("PAD-BF-002").model_copy(
        update={"correlation_group": CorrelationGroup.SOURCE_FANOUT}
    )
    monkeypatch.setattr(
        catalog_module,
        "RULE_CATALOG",
        catalog_module.RuleCatalog(
            tuple(
                conflicting if spec.rule_id == "PAD-BF-002" else spec
                for spec in original.specs
            )
        ),
    )
    with pytest.raises(DetectionConfigurationError, match="more than"):
        alerts_module._build_category_group_map()
