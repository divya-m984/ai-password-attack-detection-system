"""Structural complexity tests for the detection engine.

These assert *asymptotic behaviour*, never wall-clock time.  A timing assertion
in CI is a flaky test waiting to happen: it fails on a loaded runner and passes
on a quiet one, which tells you nothing about the code.  Counting operations
tells you exactly what you want to know.

The target is ``O(snapshots x enabled rules)``.  What that rules out is a rule
rescanning the dataset, a per-row preparation, or a per-rule pass over every
row -- each of which turns a linear run quadratic and each of which shows up
here as a count that grew faster than the input.

The 2,000-snapshot case is marked ``slow`` and deselected by default; run it
with ``uv run pytest -m slow``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.engine import DetectionEngine
from password_attack_detector.features.catalog import FeatureCatalog
from tests.unit.detection import factories

#: Rotated so a run exercises every code path -- firing, clean negative, and
#: unavailable history -- rather than one cheap branch repeated.
_BUILDERS = (
    factories.brute_force_row,
    factories.spraying_row,
    factories.quiet_row,
    factories.stuffing_row,
    factories.snapshot,
    factories.bot_row,
)


@pytest.fixture(scope="module")
def catalog() -> FeatureCatalog:
    return factories.feature_catalog()


def _stream(count: int, catalog: FeatureCatalog) -> list[dict[str, Any]]:
    """A reproducible snapshot stream with distinct anchors and timestamps."""
    return [
        _BUILDERS[index % len(_BUILDERS)](
            catalog,
            anchor_event_id=f"anchor-{index:06d}",
            anchor_event_time=factories.WHEN + timedelta(seconds=index),
        )
        for index in range(count)
    ]


def test_evaluation_count_is_exactly_linear_in_rows(catalog: FeatureCatalog) -> None:
    engine = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    rules = len(engine.enabled_rule_ids)

    counts = {
        size: engine.run(_stream(size, catalog)).stats.rule_evaluations
        for size in (10, 100, 200)
    }
    assert counts == {size: size * rules for size in counts}


def test_preparation_count_does_not_grow_with_rows(catalog: FeatureCatalog) -> None:
    """Preparation is per run, not per row: the whole point of the two-phase
    rule contract."""
    engine = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    expected = len(engine.enabled_rule_ids)
    for size in (1, 50, 200):
        assert engine.run(_stream(size, catalog)).stats.rule_preparations == expected


def test_validation_cost_is_linear_in_rows_and_read_columns(
    catalog: FeatureCatalog,
) -> None:
    engine = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    columns = len(engine.required_columns)
    for size in (10, 100, 200):
        result = engine.run(_stream(size, catalog))
        assert result.stats.validation_column_checks == size * columns


def test_evaluation_count_scales_with_enabled_rules_not_the_catalog(
    catalog: FeatureCatalog,
) -> None:
    """Disabling rules must reduce work, not merely suppress output."""
    rows = _stream(100, catalog)
    full = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    trimmed = DetectionEngine(
        DetectionConfig(enabled_rule_ids=("PAD-BF-001", "PAD-PS-001")),
        feature_catalog=catalog,
    )
    assert trimmed.run(rows).stats.rule_evaluations == len(rows) * 2
    assert full.run(rows).stats.rule_evaluations == len(rows) * len(
        full.enabled_rule_ids
    )


def test_a_tenfold_input_costs_tenfold_and_no_more(catalog: FeatureCatalog) -> None:
    """The exact statement of "no quadratic dataset scan"."""
    engine = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    small = engine.run(_stream(20, catalog)).stats
    large = engine.run(_stream(200, catalog)).stats
    assert large.rule_evaluations == small.rule_evaluations * 10
    assert large.validation_column_checks == small.validation_column_checks * 10
    assert large.rule_preparations == small.rule_preparations


@pytest.mark.slow
def test_two_thousand_snapshots_stay_linear(catalog: FeatureCatalog) -> None:
    engine = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    rows = _stream(2000, catalog)
    result = engine.run(rows)

    assert result.evaluated_snapshot_count == 2000
    assert result.total_rule_evaluation_count == 2000 * len(engine.enabled_rule_ids)
    assert result.stats.rule_preparations == len(engine.enabled_rule_ids)
    assert sum(result.status_counts.values()) == result.total_rule_evaluation_count
    assert result.fired_count > 0


@pytest.mark.slow
def test_a_large_run_stays_deterministic_under_reordering(
    catalog: FeatureCatalog,
) -> None:
    engine = DetectionEngine(DetectionConfig(), feature_catalog=catalog)
    rows = _stream(1000, catalog)
    assert engine.run(list(reversed(rows))) == engine.run(rows)
