"""Tests for the deterministic detection engine.

The engine's value is entirely in its guarantees, so most of this module is
about what stays the same: the same inputs produce the same outputs, the input
ordering does not survive into a result, and no ground-truth column can reach a
rule.  The remainder covers the input validation that has to fail *before* any
rule sees a row, and the structural counters that prove the cost stayed linear.
"""

from __future__ import annotations

import inspect
import math
from datetime import timedelta
from typing import Any

import pytest

from password_attack_detector.detection import engine as engine_module
from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import DetectionConfig, RuleSettings
from password_attack_detector.detection.engine import (
    PROHIBITED_SNAPSHOT_COLUMNS,
    DetectionEngine,
    EngineResult,
    evaluate_snapshots,
)
from password_attack_detector.detection.enums import RuleStatus
from password_attack_detector.detection.rules import RULE_IMPLEMENTATIONS
from password_attack_detector.exceptions import (
    DataValidationError,
    DetectionConfigurationError,
)
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


@pytest.fixture()
def engine(catalog: FeatureCatalog) -> DetectionEngine:
    return DetectionEngine(DetectionConfig(), feature_catalog=catalog)


def dataset(catalog: FeatureCatalog, size: int = 6) -> list[dict[str, Any]]:
    """Build a small mixed dataset with distinct anchors and timestamps."""
    builders = [
        factories.brute_force_row,
        factories.spraying_row,
        factories.quiet_row,
        factories.mfa_row,
        factories.bot_row,
        factories.takeover_row,
    ]
    return [
        builders[index % len(builders)](
            catalog,
            anchor_event_id=f"anchor-{index:04d}",
            anchor_event_time=factories.WHEN + timedelta(minutes=index),
        )
        for index in range(size)
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_every_enabled_rule_is_prepared_exactly_once(engine: DetectionEngine) -> None:
    assert engine.enabled_rule_ids == RULE_CATALOG.rule_ids


def test_rules_are_evaluated_in_catalog_order(engine: DetectionEngine) -> None:
    """Catalog order is sorted by identifier, so it is a property of the data."""
    assert list(engine.enabled_rule_ids) == sorted(engine.enabled_rule_ids)


def test_disabling_a_rule_removes_it_from_preparation(
    catalog: FeatureCatalog,
) -> None:
    config = DetectionConfig(enabled_rule_ids=("PAD-BF-001", "PAD-PS-001"))
    engine = DetectionEngine(config, feature_catalog=catalog)
    assert engine.enabled_rule_ids == ("PAD-BF-001", "PAD-PS-001")


def test_an_enabled_rule_without_an_implementation_is_refused(
    catalog: FeatureCatalog,
) -> None:
    trimmed = {
        rule_id: rule
        for rule_id, rule in RULE_IMPLEMENTATIONS.items()
        if rule_id != "PAD-BF-001"
    }
    with pytest.raises(
        DetectionConfigurationError, match="no registered implementation"
    ):
        DetectionEngine(DetectionConfig(), feature_catalog=catalog, rules=trimmed)


def test_an_implementation_disagreeing_with_the_catalog_is_refused(
    catalog: FeatureCatalog,
) -> None:
    """A rule reading a column its catalog entry never declared must not run."""

    class _Divergent:
        def __init__(self) -> None:
            self.spec = RULE_CATALOG.get("PAD-BF-001")

        def prepare(self, config: DetectionConfig, feature_catalog: Any) -> Any:
            prepared = RULE_IMPLEMENTATIONS["PAD-BF-001"].prepare(
                config, feature_catalog
            )
            object.__setattr__(
                prepared.preparation,
                "features",
                {**prepared.preparation.features, "extra": "hour_of_day"},
            )
            return prepared

    rules = {**RULE_IMPLEMENTATIONS, "PAD-BF-001": _Divergent()}
    with pytest.raises(DetectionConfigurationError, match="does not match its catalog"):
        DetectionEngine(DetectionConfig(), feature_catalog=catalog, rules=rules)


def test_required_columns_are_the_union_of_what_the_rules_resolved(
    engine: DetectionEngine,
) -> None:
    assert engine.required_columns
    assert "anchor_event_id" not in engine.required_columns
    assert not engine.required_columns & PROHIBITED_SNAPSHOT_COLUMNS


# ---------------------------------------------------------------------------
# The interface accepts feature snapshots and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["scope", "entity", "label", "split", "campaign", "events", "truth", "model"],
)
def test_no_engine_entry_point_accepts_a_forbidden_argument(forbidden: str) -> None:
    """The boundary is a type error, not a convention."""
    for callable_object in (
        DetectionEngine.__init__,
        DetectionEngine.run,
        DetectionEngine.run_diagnostic,
        evaluate_snapshots,
    ):
        parameters = inspect.signature(callable_object).parameters
        assert not any(forbidden in name.lower() for name in parameters)


def test_the_engine_module_imports_no_label_or_scope_reader() -> None:
    source = inspect.getsource(engine_module)
    for forbidden in ("splitting", "evaluation", "entity_scope", "read_labels"):
        assert f"import {forbidden}" not in source
        assert f"from password_attack_detector.features.{forbidden}" not in source


def test_run_takes_exactly_one_positional_input() -> None:
    parameters = list(inspect.signature(DetectionEngine.run).parameters)
    assert parameters == ["self", "snapshots"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_input_produces_an_identical_result(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = dataset(catalog)
    assert engine.run(rows) == engine.run(rows)


def test_reordering_the_input_rows_changes_nothing(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = dataset(catalog)
    shuffled = [rows[index] for index in (4, 0, 5, 2, 1, 3)]
    assert engine.run(shuffled) == engine.run(rows)


def test_registry_ordering_does_not_affect_output_order(
    catalog: FeatureCatalog,
) -> None:
    reversed_registry = dict(reversed(list(RULE_IMPLEMENTATIONS.items())))
    forward = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    backward = DetectionEngine(
        DetectionConfig(), feature_catalog=catalog, rules=reversed_registry
    )
    rows = dataset(catalog)
    assert backward.enabled_rule_ids == forward.enabled_rule_ids
    assert backward.run(rows) == forward.run(rows)


def test_two_engines_built_from_one_configuration_agree(
    catalog: FeatureCatalog,
) -> None:
    rows = dataset(catalog)
    first = DetectionEngine(DetectionConfig(), feature_catalog=catalog).run(rows)
    second = DetectionEngine(DetectionConfig(), feature_catalog=catalog).run(rows)
    assert first == second


def test_appending_later_snapshots_does_not_alter_earlier_detections(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    """Each snapshot is evaluated alone; there is no cross-row state."""
    rows = dataset(catalog)
    baseline = engine.run(rows)
    future = factories.brute_force_row(
        catalog,
        anchor_event_id="anchor-9999",
        anchor_event_time=factories.WHEN + timedelta(days=30),
    )
    extended = engine.run([*rows, future])

    by_anchor = {
        detection.anchor_event_id: detection for detection in extended.fired_detections
    }
    for detection in baseline.fired_detections:
        assert by_anchor[detection.anchor_event_id] == detection


def test_the_engine_does_not_mutate_its_input(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = dataset(catalog)
    before = [dict(row) for row in rows]
    engine.run(rows)
    assert rows == before


def test_snapshots_sharing_a_timestamp_evaluate_in_anchor_order(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = [
        factories.brute_force_row(catalog, anchor_event_id=anchor)
        for anchor in ("anchor-b", "anchor-a", "anchor-c")
    ]
    result = engine.run(rows)
    anchors = [detection.anchor_event_id for detection in result.fired_detections]
    assert anchors == sorted(anchors)


def test_mutating_label_and_split_columns_has_no_effect(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    """Prohibited columns are rejected outright, so they cannot influence a run.

    This is the stronger guarantee: not "the value is ignored" but "a snapshot
    carrying it does not run at all".
    """
    rows = dataset(catalog)
    for column in ("label", "split", "campaign_id", "model_probability"):
        poisoned = [{**row, column: "x"} for row in rows]
        with pytest.raises(DataValidationError, match="prohibited column"):
            engine.run(poisoned)


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_evaluation_count_is_exactly_snapshots_times_enabled_rules(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = dataset(catalog)
    result = engine.run(rows)
    assert result.total_rule_evaluation_count == len(rows) * len(
        engine.enabled_rule_ids
    )
    assert result.evaluated_snapshot_count == len(rows)


def test_status_counts_account_for_every_evaluation(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    result = engine.run(dataset(catalog))
    assert sum(result.status_counts.values()) == result.total_rule_evaluation_count
    assert set(result.status_counts) == {str(status) for status in RuleStatus}


def test_per_rule_counts_sum_to_the_snapshot_count(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = dataset(catalog)
    result = engine.run(rows)
    assert set(result.per_rule_counts) == set(engine.enabled_rule_ids)
    for counts in result.per_rule_counts.values():
        assert sum(counts.values()) == len(rows)


def test_fired_detections_match_the_fired_status_count(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    result = engine.run(dataset(catalog))
    assert result.fired_count == result.status_counts[str(RuleStatus.FIRED)]
    assert result.fired_count > 0


def test_insufficient_data_is_reported_separately_from_not_fired(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    """Unobserved history must never be presented as a clean negative."""
    unobserved = factories.snapshot(catalog, anchor_event_id="anchor-cold")
    result = engine.run([unobserved])
    assert result.insufficient_data_count > 0
    assert (
        result.insufficient_data_count
        == result.status_counts[str(RuleStatus.INSUFFICIENT_DATA)]
    )
    assert result.status_counts[str(RuleStatus.FIRED)] == 0


def test_disabled_rules_are_counted_but_never_evaluated(
    catalog: FeatureCatalog,
) -> None:
    config = DetectionConfig(enabled_rule_ids=("PAD-BF-001",))
    engine = DetectionEngine(config, feature_catalog=catalog)
    rows = dataset(catalog)
    result = engine.run(rows)

    assert result.disabled_rule_count == len(RULE_CATALOG) - 1
    assert result.total_rule_evaluation_count == len(rows)
    assert result.status_counts[str(RuleStatus.DISABLED)] == 0
    assert set(result.per_rule_counts) == {"PAD-BF-001"}


def test_only_fired_detections_enter_the_detection_list(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    result = engine.run(dataset(catalog))
    assert all(detection.evidence for detection in result.fired_detections)
    assert all(detection.reason_codes for detection in result.fired_detections)
    assert all(detection.signal_strength > 0.0 for detection in result.fired_detections)


def test_the_diagnostic_view_returns_every_outcome(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = dataset(catalog)
    results = engine.run_diagnostic(rows)
    assert len(results) == len(rows) * len(engine.enabled_rule_ids)
    assert engine.run_diagnostic(rows) == results


def test_aggregate_diagnostics_contain_no_identifier(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    result = engine.run(dataset(catalog))
    rendered = repr((result.status_counts, result.per_rule_counts, result.stats))
    assert "anchor-" not in rendered
    assert factories.ANCHOR not in rendered


# ---------------------------------------------------------------------------
# Structural cost
# ---------------------------------------------------------------------------


def test_rule_preparation_count_is_independent_of_row_count(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    """The prepare-once guarantee, stated as an exact count."""
    small = engine.run(dataset(catalog, size=2))
    large = engine.run(dataset(catalog, size=6))
    assert small.stats.rule_preparations == len(engine.enabled_rule_ids)
    assert large.stats.rule_preparations == small.stats.rule_preparations


def test_validation_cost_grows_linearly_with_row_count(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    """Linear, not quadratic: no per-rule rescan of the dataset."""
    one = engine.run(dataset(catalog, size=1)).stats.validation_column_checks
    six = engine.run(dataset(catalog, size=6)).stats.validation_column_checks
    assert six == one * 6


def test_validation_reads_only_the_columns_the_rules_need(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    result = engine.run(dataset(catalog, size=1))
    assert result.stats.validation_column_checks == len(engine.required_columns)
    assert len(engine.required_columns) < len(catalog)


# ---------------------------------------------------------------------------
# Fingerprints and identifiers
# ---------------------------------------------------------------------------


def test_the_result_records_the_configuration_and_catalog_fingerprints(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    result = engine.run(dataset(catalog))
    assert result.configuration_fingerprint == DetectionConfig().fingerprint()
    assert result.rule_catalog_fingerprint == RULE_CATALOG.fingerprint()
    assert result.detection_schema_version == "1.0.0"


def test_a_semantic_configuration_change_changes_the_run_fingerprint(
    catalog: FeatureCatalog,
) -> None:
    rows = dataset(catalog)
    baseline = DetectionEngine(DetectionConfig(), feature_catalog=catalog).run(rows)
    retuned_config = DetectionConfig(
        rules={"PAD-BF-001": RuleSettings(parameters={"min_pair_failures": 3})}
    )
    retuned = DetectionEngine(retuned_config, feature_catalog=catalog).run(rows)
    assert retuned.configuration_fingerprint != baseline.configuration_fingerprint


def test_detection_identifiers_are_stable_across_runs(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = dataset(catalog)
    first = [detection.detection_id for detection in engine.run(rows).fired_detections]
    second = [
        detection.detection_id
        for detection in DetectionEngine(DetectionConfig(), feature_catalog=catalog)
        .run(list(reversed(rows)))
        .fired_detections
    ]
    assert first == second
    assert len(set(first)) == len(first)


def test_a_different_rule_version_yields_a_different_detection_identifier(
    catalog: FeatureCatalog,
) -> None:
    """Detection identity is (anchor, rule, rule version).

    A rule whose logic changed enough to warrant a version bump produces a
    distinct detection rather than silently overwriting the previous one.
    """
    from password_attack_detector.detection.schemas import detection_identifier

    first = detection_identifier("anchor-0000", "PAD-BF-001", "1.0.0")
    second = detection_identifier("anchor-0000", "PAD-BF-001", "2.0.0")
    assert first != second


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_an_empty_input_returns_a_valid_empty_result(engine: DetectionEngine) -> None:
    """A quiet interval is an ordinary state, not a failure."""
    result = engine.run([])
    assert isinstance(result, EngineResult)
    assert result.fired_detections == ()
    assert result.evaluated_snapshot_count == 0
    assert result.total_rule_evaluation_count == 0
    assert result.configuration_fingerprint


def test_a_duplicate_anchor_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    rows = [
        factories.brute_force_row(catalog, anchor_event_id="anchor-0001"),
        factories.spraying_row(
            catalog,
            anchor_event_id="anchor-0001",
            anchor_event_time=factories.WHEN + timedelta(minutes=1),
        ),
    ]
    with pytest.raises(DataValidationError, match="duplicate"):
        engine.run(rows)


@pytest.mark.parametrize(
    "column",
    ["label", "attack_class", "campaign_id", "split", "model_probability", "malicious"],
)
def test_a_prohibited_column_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog, column: str
) -> None:
    row = {**factories.brute_force_row(catalog), column: "value"}
    with pytest.raises(DataValidationError, match="prohibited column"):
        engine.run([row])


def test_the_prohibition_message_names_columns_and_nothing_else(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    row = {**factories.brute_force_row(catalog), "label": "malicious"}
    with pytest.raises(DataValidationError) as caught:
        engine.run([row])
    message = str(caught.value)
    assert "label" in message
    assert factories.ANCHOR not in message
    assert "malicious" not in message


def test_a_missing_required_column_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    row = factories.brute_force_row(catalog)
    dropped = next(iter(engine.required_columns))
    del row[dropped]
    with pytest.raises(DataValidationError, match="missing required column"):
        engine.run([row])


def test_an_incompatible_feature_schema_version_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    row = factories.brute_force_row(catalog, feature_schema_version="9.9.9")
    with pytest.raises(DataValidationError, match="incompatible"):
        engine.run([row])


@pytest.mark.parametrize("value", [None, "", 42])
def test_an_invalid_anchor_identifier_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog, value: Any
) -> None:
    row = factories.brute_force_row(catalog)
    row["anchor_event_id"] = value
    with pytest.raises(DataValidationError, match="non-empty string"):
        engine.run([row])


def test_a_timezone_naive_anchor_time_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    from datetime import datetime

    row = factories.brute_force_row(catalog)
    row["anchor_event_time"] = datetime(2026, 3, 1, 12, 0)
    with pytest.raises(DataValidationError, match="timezone-naive"):
        engine.run([row])


def test_a_non_datetime_anchor_time_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    row = factories.brute_force_row(catalog)
    row["anchor_event_time"] = "2026-03-01T12:00:00Z"
    with pytest.raises(DataValidationError, match="non-datetime"):
        engine.run([row])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_rule_input_is_refused(
    engine: DetectionEngine, catalog: FeatureCatalog, value: float
) -> None:
    row = factories.brute_force_row(catalog, pair_failure_rate__5m=value)
    with pytest.raises(DataValidationError, match="NaN or infinity"):
        engine.run([row])


def test_the_non_finite_message_names_the_column_only(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    row = factories.brute_force_row(catalog, pair_failure_rate__5m=math.nan)
    with pytest.raises(DataValidationError) as caught:
        engine.run([row])
    assert "pair_failure_rate__5m" in str(caught.value)
    assert factories.ANCHOR not in str(caught.value)


def test_a_non_finite_value_in_an_unread_column_is_ignored(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    """Validation covers what the run reads, which is what can affect it."""
    unread = "hour_sin"
    assert unread not in engine.required_columns
    row = factories.brute_force_row(catalog, **{unread: math.nan})
    assert engine.run([row]).fired_count > 0


def test_validation_runs_before_any_rule_sees_a_row(
    engine: DetectionEngine, catalog: FeatureCatalog
) -> None:
    """A valid row alongside an invalid one must not produce a partial result."""
    rows = [
        factories.brute_force_row(catalog, anchor_event_id="anchor-0001"),
        factories.brute_force_row(
            catalog,
            anchor_event_id="anchor-0002",
            anchor_event_time=factories.WHEN + timedelta(minutes=1),
            pair_failure_rate__5m=math.inf,
        ),
    ]
    with pytest.raises(DataValidationError):
        engine.run(rows)


# ---------------------------------------------------------------------------
# The convenience entry point
# ---------------------------------------------------------------------------


def test_evaluate_snapshots_matches_a_reused_engine(catalog: FeatureCatalog) -> None:
    rows = dataset(catalog)
    once = evaluate_snapshots(DetectionConfig(), rows, feature_catalog=catalog)
    reused = DetectionEngine(DetectionConfig(), feature_catalog=catalog).run(rows)
    assert once == reused
