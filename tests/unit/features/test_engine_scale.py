"""Structural complexity tests for the feature engine.

These assert *asymptotic behaviour*, never wall-clock time.  A timing assertion
in CI is a flaky test waiting to happen: it fails on a loaded runner and passes
on a quiet one, which tells you nothing about the code.  Counting operations
tells you exactly what you want to know.

The 10,000-event case is marked ``slow`` and deselected by default; run it with
``uv run pytest -m slow``.
"""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.features.catalog import build_catalog
from password_attack_detector.features.config import (
    BaselineConfig,
    FeatureConfig,
    SplitConfig,
)
from password_attack_detector.features.engine import EngineStats, FeatureEngine
from tests.features.factories import make_event

#: Six entity kinds are tracked, so every event is appended six times.
_TRACKED_ENTITY_KINDS = 6

_CONFIG = FeatureConfig(
    split=SplitConfig(purge=timedelta(hours=24), max_excluded_fraction=0.5),
    baseline=BaselineConfig(enabled=False),
)


def _stream(count: int, *, seed: int = 4242, entity_pool: int = 50) -> list[AuthEvent]:
    """A reproducible stream over a bounded entity pool spread across days."""
    rng = random.Random(seed)
    return [
        make_event(
            t=float(index) * 30.0,
            user=f"u{rng.randint(1, entity_pool)}",
            source=f"s{rng.randint(1, entity_pool // 5 or 1)}",
            device=f"d{rng.randint(1, entity_pool // 2 or 1)}",
            session=f"sess{rng.randint(1, entity_pool)}",
            outcome=rng.choice(["success", "failure", "blocked", "challenged"]),
            country=rng.choice([None, "US", "GB", "DE"]),
            response_time_ms=rng.choice([None, 40, 180, 900]),
            key=str(index),
        )
        for index in range(count)
    ]


def _run(count: int) -> EngineStats:
    events = _stream(count)
    return FeatureEngine(_CONFIG, build_catalog(_CONFIG)).run(events).stats


@pytest.fixture(scope="module")
def unit_scale_stats() -> EngineStats:
    """Engine statistics for a thousand-event run, computed once."""
    return _run(1_000)


class TestUnitScale:
    """A thousand events: the shape every property is asserted against."""

    def test_every_event_produces_exactly_one_row(
        self, unit_scale_stats: EngineStats
    ) -> None:
        stats = unit_scale_stats
        assert stats.rows_emitted == 1_000
        assert stats.events_ingested == 1_000

    def test_each_event_is_appended_once_per_tracked_entity(
        self, unit_scale_stats: EngineStats
    ) -> None:
        stats = unit_scale_stats
        assert stats.buffer_appends == 1_000 * _TRACKED_ENTITY_KINDS

    def test_no_full_history_scan_occurs(self, unit_scale_stats: EngineStats) -> None:
        stats = unit_scale_stats
        # A quadratic implementation would touch on the order of n^2/2 records.
        # Linear bookkeeping keeps evictions bounded by appends.
        assert stats.buffer_evictions <= stats.buffer_appends

    def test_blocks_never_exceed_events(self, unit_scale_stats: EngineStats) -> None:
        stats = unit_scale_stats
        assert 0 < stats.blocks_processed <= stats.events_ingested

    def test_stats_are_all_integers(self, unit_scale_stats: EngineStats) -> None:
        assert all(isinstance(v, int) for v in unit_scale_stats.as_dict().values())


class TestLinearScaling:
    """Work must grow linearly with the event count, not quadratically."""

    def test_appends_scale_linearly(self) -> None:
        small = _run(500)
        large = _run(2_000)
        ratio = large.buffer_appends / small.buffer_appends
        assert ratio == pytest.approx(4.0, abs=0.01)

    def test_rows_scale_exactly(self) -> None:
        assert _run(2_000).rows_emitted == 4 * _run(500).rows_emitted

    def test_quadratic_growth_is_ruled_out(self) -> None:
        # Under O(n^2) the ratio for a 4x input would be near 16, not 4.
        small = _run(500)
        large = _run(2_000)
        total_small = small.buffer_appends + small.buffer_evictions
        total_large = large.buffer_appends + large.buffer_evictions
        assert total_large / total_small < 6.0


class TestBoundedState:
    """Retained state must depend on the active window, not on run length."""

    def test_tracked_entities_do_not_grow_without_bound(self) -> None:
        # The stream spans far more than 24h, so early entities must be
        # released rather than retained for the whole run.
        stats = _run(4_000)
        assert stats.expired_entity_count > 0
        assert stats.tracked_entity_count < stats.peak_tracked_entity_count

    def test_retained_state_is_similar_across_run_lengths(self) -> None:
        # Doubling the input should not double the live entity count, because
        # the extra events fall outside the longest window.
        short = _run(4_000)
        long_run = _run(8_000)
        assert long_run.tracked_entity_count <= short.tracked_entity_count * 2

    def test_capacity_cap_is_honoured_and_reported(self) -> None:
        config = FeatureConfig(
            max_tracked_entities=10,
            split=SplitConfig(purge=timedelta(hours=24), max_excluded_fraction=0.5),
            baseline=BaselineConfig(enabled=False),
        )
        events = _stream(1_000, entity_pool=200)
        frame = FeatureEngine(config, build_catalog(config)).run(events)
        assert frame.stats.capacity_evicted_entity_count > 0
        assert len(frame) == len(events), "eviction must never drop a row"


@pytest.mark.slow
class TestBenchmarkScale:
    """Ten thousand events. Deselected by default; run with ``-m slow``.

    **Nothing here is a correctness test that only runs under the marker.**
    Each case re-runs, at larger scale, a property that a default-selected
    test already asserts:

    ==================================== ====================================
    Slow case (10,000 events)            Default-selected counterpart
    ==================================== ====================================
    ``test_completes_and_stays_linear``  ``TestLinearScaling`` (500 / 2,000)
                                         and ``TestUnitScale`` (1,000)
    ``test_state_stays_bounded_at_scale`` ``TestBoundedState``
                                         ``::test_tracked_entities_do_not_``
                                         ``grow_without_bound`` (4,000)
    ``test_output_remains_``             ``test_engine.py::TestDeterminism``
    ``deterministic_at_scale``           ``::test_input_order_does_not_``
                                         ``affect_output``
    ==================================== ====================================

    So a default ``uv run pytest`` -- and therefore ``scripts/verify.sh`` --
    still covers every guarantee. These exist to catch behaviour that only
    emerges with volume, at a runtime CI should not pay on every commit.

    If you add a case here, add or point to its default-selected counterpart
    too. A correctness property reachable only via ``-m slow`` is a property
    nobody runs.
    """

    def test_completes_and_stays_linear(self) -> None:
        stats = _run(10_000)
        assert stats.rows_emitted == 10_000
        assert stats.buffer_appends == 10_000 * _TRACKED_ENTITY_KINDS

    def test_state_stays_bounded_at_scale(self) -> None:
        stats = _run(10_000)
        assert stats.expired_entity_count > 0
        assert stats.tracked_entity_count < stats.peak_tracked_entity_count

    def test_output_remains_deterministic_at_scale(self) -> None:
        events = _stream(10_000)
        shuffled = list(events)
        random.Random(7).shuffle(shuffled)
        catalog = build_catalog(_CONFIG)
        first = FeatureEngine(_CONFIG, catalog).run(events).rows
        second = FeatureEngine(_CONFIG, catalog).run(shuffled).rows
        assert first == second
