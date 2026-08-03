"""Tests for the point-in-time temporal primitives."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta, timezone
from itertools import pairwise

import pytest

from password_attack_detector.features.temporal import (
    MICROSECONDS_PER_SECOND,
    MISSING_DIM,
    NULL_RESPONSE_TIME,
    EntityBuffer,
    WindowAccumulator,
    calendar_features,
    coefficient_of_variation,
    from_microseconds,
    iter_timestamp_blocks,
    mean_std,
    to_microseconds,
)
from tests.features.factories import BASE_TIME, make_event, make_stream

_MINUTE = 60 * MICROSECONDS_PER_SECOND

# Outcome codes used throughout these tests: 0 success, 1 failure.
_SUCCESS = 0
_FAILURE = 1
_N_OUTCOMES = 4


def _buffer(
    *widths_seconds: int,
    n_dims: int = 0,
    cardinality: frozenset[int] = frozenset(),
) -> EntityBuffer:
    return EntityBuffer(
        [s * MICROSECONDS_PER_SECOND for s in widths_seconds],
        n_outcomes=_N_OUTCOMES,
        n_dims=n_dims,
        cardinality_windows=cardinality,
    )


# --- time conversion -------------------------------------------------------


class TestTimeConversion:
    def test_round_trips_exactly(self) -> None:
        moment = datetime(2024, 3, 4, 12, 34, 56, 789012, tzinfo=UTC)
        assert from_microseconds(to_microseconds(moment)) == moment

    def test_microsecond_precision_is_preserved(self) -> None:
        a = datetime(2024, 1, 1, tzinfo=UTC)
        b = a + timedelta(microseconds=1)
        assert to_microseconds(b) - to_microseconds(a) == 1

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            to_microseconds(datetime(2024, 1, 1))

    def test_non_utc_timezone_is_normalised(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        local = datetime(2024, 1, 1, 5, tzinfo=eastern)
        assert to_microseconds(local) == to_microseconds(local.astimezone(UTC))


# --- timestamp blocking ----------------------------------------------------


class TestTimestampBlocks:
    def test_empty_input_yields_nothing(self) -> None:
        assert list(iter_timestamp_blocks([])) == []

    def test_distinct_timestamps_yield_singleton_blocks(self) -> None:
        events = make_stream("0 u1 s1 failure\n30 u1 s1 failure\n60 u1 s1 failure")
        blocks = list(iter_timestamp_blocks(events))
        assert [len(members) for _, members in blocks] == [1, 1, 1]

    def test_equal_timestamps_are_grouped(self) -> None:
        events = make_stream(
            "0 u1 s1 failure\n60 u1 s2 failure\n60 u1 s3 failure\n90 u1 s1 failure"
        )
        blocks = list(iter_timestamp_blocks(events))
        assert [len(members) for _, members in blocks] == [1, 2, 1]

    def test_block_timestamps_are_ascending(self) -> None:
        events = make_stream("0 u1 s1 failure\n60 u1 s2 failure\n60 u1 s3 failure")
        stamps = [ts for ts, _ in iter_timestamp_blocks(events)]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_every_event_appears_exactly_once(self) -> None:
        events = make_stream(
            "0 u1 s1 failure\n0 u1 s2 failure\n60 u1 s3 failure\n60 u1 s4 failure"
        )
        emitted = [e for _, members in iter_timestamp_blocks(events) for e in members]
        assert [e.event_id for e in emitted] == [e.event_id for e in events]

    def test_unsorted_input_rejected(self) -> None:
        events = [make_event(t=60.0, key="a"), make_event(t=0.0, key="b")]
        with pytest.raises(ValueError, match="sorted"):
            list(iter_timestamp_blocks(events))

    def test_single_event_yields_one_block(self) -> None:
        blocks = list(iter_timestamp_blocks([make_event(t=0.0)]))
        assert len(blocks) == 1
        assert len(blocks[0][1]) == 1


# --- exact mean and standard deviation -------------------------------------


class TestMeanStd:
    def test_empty_sample_is_null(self) -> None:
        assert mean_std(0, 0, 0) == (None, None)

    def test_single_observation_has_mean_but_null_std(self) -> None:
        mean, std = mean_std(1, 7, 49)
        assert mean == pytest.approx(7.0)
        assert std is None, "std of one sample is undefined, not 0.0"

    def test_matches_the_textbook_sample_formula(self) -> None:
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        n = len(values)
        mean, std = mean_std(n, sum(values), sum(v * v for v in values))
        expected_mean = sum(values) / n
        expected_var = sum((v - expected_mean) ** 2 for v in values) / (n - 1)
        assert mean == pytest.approx(expected_mean)
        assert std is not None
        assert std == pytest.approx(math.sqrt(expected_var))

    def test_constant_sample_has_zero_std(self) -> None:
        _, std = mean_std(4, 20, 100)
        assert std == pytest.approx(0.0)

    def test_variance_numerator_is_never_negative(self) -> None:
        # Exact integer arithmetic makes the numerator provably non-negative,
        # so no clamping is needed and sqrt never sees a negative argument.
        rng = random.Random(20240304)
        for _ in range(200):
            values = [rng.randint(0, 30_000) for _ in range(rng.randint(2, 40))]
            _, std = mean_std(len(values), sum(values), sum(v * v for v in values))
            assert std is not None
            assert std >= 0.0

    def test_large_microsecond_magnitudes_stay_exact(self) -> None:
        day = 86_400 * MICROSECONDS_PER_SECOND
        values = [day, 2 * day, 3 * day]
        mean, std = mean_std(3, sum(values), sum(v * v for v in values))
        assert mean == pytest.approx(2 * day)
        assert std is not None
        assert std == pytest.approx(day)


class TestCoefficientOfVariation:
    def test_returns_std_over_mean(self) -> None:
        assert coefficient_of_variation(4.0, 2.0) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "mean,std", [(None, 1.0), (1.0, None), (None, None), (0.0, 1.0)]
    )
    def test_undefined_cases_are_null(
        self, mean: float | None, std: float | None
    ) -> None:
        assert coefficient_of_variation(mean, std) is None


# --- window boundary semantics ---------------------------------------------


class TestWindowBoundary:
    def test_interval_is_closed_on_the_left(self) -> None:
        buffer = _buffer(60)
        anchor = 10 * _MINUTE
        buffer.append(anchor - 60 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.advance(anchor)
        assert buffer.accumulators[0].n == 1, "an event at exactly t - w is in window"

    def test_interval_is_open_on_the_right(self) -> None:
        buffer = _buffer(60)
        anchor = 10 * _MINUTE
        buffer.append(anchor, outcome=_FAILURE)
        buffer.advance(anchor)
        assert buffer.accumulators[0].n == 1

    def test_one_microsecond_before_the_left_edge_is_excluded(self) -> None:
        buffer = _buffer(60)
        anchor = 10 * _MINUTE
        buffer.append(anchor - 60 * MICROSECONDS_PER_SECOND - 1, outcome=_FAILURE)
        buffer.advance(anchor)
        assert buffer.accumulators[0].n == 0

    def test_exact_boundary_triple(self) -> None:
        buffer = _buffer(60)
        anchor = 10 * _MINUTE
        width = 60 * MICROSECONDS_PER_SECOND
        buffer.append(anchor - width - 1, outcome=_FAILURE)  # out
        buffer.append(anchor - width, outcome=_FAILURE)  # in
        buffer.append(anchor - 1, outcome=_FAILURE)  # in
        buffer.advance(anchor)
        assert buffer.accumulators[0].n == 2

    def test_each_window_evicts_independently(self) -> None:
        buffer = _buffer(60, 300)
        anchor = 100 * _MINUTE
        buffer.append(anchor - 200 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.append(anchor - 30 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.advance(anchor)
        assert buffer.accumulators[0].n == 1
        assert buffer.accumulators[1].n == 2

    def test_eviction_never_resurrects_a_record(self) -> None:
        # Raw head indices are not monotonic because compaction rebases them;
        # the invariant that matters is that the oldest retained timestamp
        # never moves backward.
        buffer = _buffer(60)
        for offset in range(0, 600, 30):
            buffer.append(offset * MICROSECONDS_PER_SECOND, outcome=_FAILURE)

        oldest_seen = -1
        for anchor in range(0, 900, 60):
            buffer.advance(anchor * MICROSECONDS_PER_SECOND)
            oldest = buffer.oldest_in_window_ts(0)
            if oldest is not None:
                assert oldest >= oldest_seen
                oldest_seen = oldest

    def test_empty_window_reports_no_oldest_record(self) -> None:
        buffer = _buffer(60)
        assert buffer.oldest_in_window_ts(0) is None
        buffer.append(0, outcome=_FAILURE)
        buffer.advance(3600 * MICROSECONDS_PER_SECOND)
        assert buffer.oldest_in_window_ts(0) is None


# --- counting and rates ----------------------------------------------------


class TestCounting:
    def test_counts_by_outcome(self) -> None:
        buffer = _buffer(3600)
        for _ in range(3):
            buffer.append(0, outcome=_FAILURE)
        buffer.append(0, outcome=_SUCCESS)
        buffer.advance(_MINUTE)
        accumulator = buffer.accumulators[0]
        assert accumulator.n == 4
        assert accumulator.n_by_outcome[_FAILURE] == 3
        assert accumulator.n_by_outcome[_SUCCESS] == 1

    def test_counts_return_to_zero_after_full_eviction(self) -> None:
        buffer = _buffer(60)
        buffer.append(0, outcome=_FAILURE)
        buffer.advance(3600 * MICROSECONDS_PER_SECOND)
        accumulator = buffer.accumulators[0]
        assert accumulator.n == 0
        assert accumulator.n_by_outcome == [0, 0, 0, 0]

    def test_mfa_failures_are_tracked(self) -> None:
        buffer = _buffer(3600)
        buffer.append(0, outcome=_FAILURE, mfa_failed=True)
        buffer.append(1, outcome=_FAILURE, mfa_failed=False)
        buffer.advance(_MINUTE)
        assert buffer.accumulators[0].n_mfa_failed == 1

    def test_rate_is_null_for_an_empty_window(self) -> None:
        buffer = _buffer(60)
        buffer.advance(_MINUTE)
        accumulator = buffer.accumulators[0]
        assert accumulator.rate(0, min_count=1) is None, "0/0 must be null, not 0.0"

    def test_rate_divides_by_the_window_count(self) -> None:
        buffer = _buffer(3600)
        for _ in range(3):
            buffer.append(0, outcome=_FAILURE)
        buffer.append(0, outcome=_SUCCESS)
        buffer.advance(_MINUTE)
        accumulator = buffer.accumulators[0]
        assert accumulator.rate(3, min_count=1) == pytest.approx(0.75)

    def test_rate_respects_min_count(self) -> None:
        buffer = _buffer(3600)
        buffer.append(0, outcome=_FAILURE)
        buffer.advance(_MINUTE)
        accumulator = buffer.accumulators[0]
        assert accumulator.rate(1, min_count=1) == pytest.approx(1.0)
        assert accumulator.rate(1, min_count=5) is None


# --- unique cardinality ----------------------------------------------------


class TestCardinality:
    def test_counts_distinct_values(self) -> None:
        buffer = _buffer(3600, n_dims=1, cardinality=frozenset({0}))
        for code in (7, 7, 8, 9):
            buffer.append(0, outcome=_FAILURE, dims=(code,))
        buffer.advance(_MINUTE)
        assert buffer.accumulators[0].unique_count(0) == 3

    def test_evicted_values_stop_counting(self) -> None:
        buffer = _buffer(60, n_dims=1, cardinality=frozenset({0}))
        buffer.append(0, outcome=_FAILURE, dims=(7,))
        buffer.append(120 * MICROSECONDS_PER_SECOND, outcome=_FAILURE, dims=(8,))
        buffer.advance(150 * MICROSECONDS_PER_SECOND)
        assert buffer.accumulators[0].unique_count(0) == 1

    def test_repeated_value_survives_partial_eviction(self) -> None:
        buffer = _buffer(60, n_dims=1, cardinality=frozenset({0}))
        buffer.append(0, outcome=_FAILURE, dims=(7,))
        buffer.append(120 * MICROSECONDS_PER_SECOND, outcome=_FAILURE, dims=(7,))
        buffer.advance(150 * MICROSECONDS_PER_SECOND)
        assert buffer.accumulators[0].unique_count(0) == 1

    def test_counter_is_emptied_not_merely_zeroed(self) -> None:
        # Without deleting on zero, len(counter) would keep counting values
        # that have fully aged out.
        buffer = _buffer(60, n_dims=1, cardinality=frozenset({0}))
        buffer.append(0, outcome=_FAILURE, dims=(7,))
        buffer.advance(3600 * MICROSECONDS_PER_SECOND)
        assert buffer.accumulators[0].unique_count(0) == 0
        assert buffer.accumulators[0].cardinality[0] == {}

    def test_missing_dimension_is_not_counted(self) -> None:
        buffer = _buffer(3600, n_dims=1, cardinality=frozenset({0}))
        buffer.append(0, outcome=_FAILURE, dims=(MISSING_DIM,))
        buffer.append(1, outcome=_FAILURE, dims=(5,))
        buffer.advance(_MINUTE)
        assert buffer.accumulators[0].unique_count(0) == 1

    def test_multiple_dimensions_are_independent(self) -> None:
        buffer = _buffer(3600, n_dims=2, cardinality=frozenset({0}))
        buffer.append(0, outcome=_FAILURE, dims=(1, 100))
        buffer.append(1, outcome=_FAILURE, dims=(1, 200))
        buffer.advance(_MINUTE)
        assert buffer.accumulators[0].unique_count(0) == 1
        assert buffer.accumulators[0].unique_count(1) == 2

    def test_untracked_window_rejects_cardinality_reads(self) -> None:
        buffer = _buffer(60, n_dims=1)
        with pytest.raises(ValueError, match="does not track cardinality"):
            buffer.accumulators[0].unique_count(0)


# --- response-time statistics ----------------------------------------------


class TestResponseTimeStats:
    def test_null_response_times_are_excluded(self) -> None:
        buffer = _buffer(3600)
        buffer.append(0, outcome=_FAILURE, rt_ms=100)
        buffer.append(1, outcome=_FAILURE, rt_ms=NULL_RESPONSE_TIME)
        buffer.append(2, outcome=_FAILURE, rt_ms=200)
        buffer.advance(_MINUTE)
        mean, _ = buffer.accumulators[0].response_time_stats()
        assert mean == pytest.approx(150.0), "absent values must not count as zero"

    def test_empty_window_is_null(self) -> None:
        buffer = _buffer(60)
        buffer.advance(_MINUTE)
        assert buffer.accumulators[0].response_time_stats() == (None, None)

    def test_single_observation_has_null_std(self) -> None:
        buffer = _buffer(3600)
        buffer.append(0, outcome=_FAILURE, rt_ms=100)
        buffer.advance(_MINUTE)
        mean, std = buffer.accumulators[0].response_time_stats()
        assert mean == pytest.approx(100.0)
        assert std is None

    def test_stats_are_exact_after_eviction(self) -> None:
        buffer = _buffer(60)
        buffer.append(0, outcome=_FAILURE, rt_ms=1000)
        anchor = 120 * MICROSECONDS_PER_SECOND
        buffer.append(anchor - 10, outcome=_FAILURE, rt_ms=100)
        buffer.append(anchor - 5, outcome=_FAILURE, rt_ms=200)
        buffer.advance(anchor)
        mean, _ = buffer.accumulators[0].response_time_stats()
        assert mean == pytest.approx(150.0)


# --- interarrival statistics -----------------------------------------------


class TestInterarrivalStats:
    def test_no_samples_for_a_single_event(self) -> None:
        buffer = _buffer(3600)
        buffer.append(0, outcome=_FAILURE)
        buffer.advance(_MINUTE)
        assert buffer.accumulators[0].interarrival_stats_seconds() == (None, None)

    def test_mean_gap_is_reported_in_seconds(self) -> None:
        buffer = _buffer(3600)
        for offset in (0, 10, 20, 30):
            buffer.append(offset * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.advance(_MINUTE)
        mean, std = buffer.accumulators[0].interarrival_stats_seconds()
        assert mean == pytest.approx(10.0)
        assert std == pytest.approx(0.0)

    def test_sample_count_is_always_one_less_than_the_event_count(self) -> None:
        buffer = _buffer(120)
        for step in range(20):
            buffer.append(step * 10 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
            buffer.advance(step * 10 * MICROSECONDS_PER_SECOND)
            accumulator = buffer.accumulators[0]
            assert accumulator.ia_n == max(0, accumulator.n - 1)

    def test_gap_spanning_the_boundary_is_not_a_sample(self) -> None:
        buffer = _buffer(60)
        buffer.append(0, outcome=_FAILURE)
        buffer.append(100 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.append(110 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.advance(120 * MICROSECONDS_PER_SECOND)
        accumulator = buffer.accumulators[0]
        assert accumulator.n == 2
        mean, _ = accumulator.interarrival_stats_seconds()
        assert mean == pytest.approx(10.0), "the 100s gap crosses the window edge"

    def test_gaps_are_measured_against_the_previous_event_not_the_window(
        self,
    ) -> None:
        buffer = _buffer(3600)
        buffer.append(0, outcome=_FAILURE)
        buffer.append(5 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.append(35 * MICROSECONDS_PER_SECOND, outcome=_FAILURE)
        buffer.advance(_MINUTE)
        mean, _ = buffer.accumulators[0].interarrival_stats_seconds()
        assert mean == pytest.approx(17.5)

    def test_matches_a_brute_force_recomputation(self) -> None:
        rng = random.Random(4711)
        width_s = 60
        buffer = _buffer(width_s)
        history: list[int] = []
        stamp = 0
        for _ in range(200):
            stamp += rng.randint(1, 20) * MICROSECONDS_PER_SECOND
            buffer.append(stamp, outcome=_FAILURE)
            history.append(stamp)

            anchor = stamp + MICROSECONDS_PER_SECOND
            buffer.advance(anchor)

            cutoff = anchor - width_s * MICROSECONDS_PER_SECOND
            in_window = [s for s in history if cutoff <= s < anchor]
            gaps = [b - a for a, b in pairwise(in_window)]
            expected = (
                None if not gaps else sum(gaps) / len(gaps) / MICROSECONDS_PER_SECOND
            )
            mean, _ = buffer.accumulators[0].interarrival_stats_seconds()
            if expected is None:
                assert mean is None
            else:
                assert mean == pytest.approx(expected)


# --- bounded state ---------------------------------------------------------


class TestBoundedState:
    def test_buffer_is_compacted_as_records_age_out(self) -> None:
        buffer = _buffer(60)
        for step in range(500):
            stamp = step * 10 * MICROSECONDS_PER_SECOND
            buffer.append(stamp, outcome=_FAILURE)
            buffer.advance(stamp)
        # A 60s window holds at most 7 ten-second records; the shared buffer
        # must not grow with the total number of events processed.
        assert buffer.record_count < 40

    def test_records_are_retained_for_the_longest_window(self) -> None:
        buffer = _buffer(60, 600)
        for step in range(200):
            stamp = step * 10 * MICROSECONDS_PER_SECOND
            buffer.append(stamp, outcome=_FAILURE)
            buffer.advance(stamp)
        assert buffer.accumulators[1].n == 61

    def test_expiry_is_reported_after_the_longest_window(self) -> None:
        buffer = _buffer(60)
        buffer.append(0, outcome=_FAILURE)
        width = 60 * MICROSECONDS_PER_SECOND
        assert not buffer.is_expired(width, width)
        assert buffer.is_expired(width + 1, width)

    def test_a_never_used_buffer_is_not_expired(self) -> None:
        assert not _buffer(60).is_expired(10**12, 60 * MICROSECONDS_PER_SECOND)

    def test_compaction_preserves_aggregates(self) -> None:
        compacting = _buffer(60)
        reference: list[int] = []
        for step in range(300):
            stamp = step * 5 * MICROSECONDS_PER_SECOND
            compacting.append(stamp, outcome=_FAILURE)
            compacting.advance(stamp)
            reference.append(stamp)
            cutoff = stamp - 60 * MICROSECONDS_PER_SECOND
            expected = sum(1 for s in reference if cutoff <= s <= stamp)
            assert compacting.accumulators[0].n == expected


class TestAccumulatorConstruction:
    def test_outcome_slots_are_preallocated(self) -> None:
        accumulator = WindowAccumulator(width_us=1, n_outcomes=4, n_dims=0)
        assert accumulator.n_by_outcome == [0, 0, 0, 0]

    def test_cardinality_dicts_created_only_when_tracked(self) -> None:
        tracked = WindowAccumulator(
            width_us=1, n_outcomes=4, n_dims=3, track_cardinality=True
        )
        untracked = WindowAccumulator(width_us=1, n_outcomes=4, n_dims=3)
        assert len(tracked.cardinality) == 3
        assert untracked.cardinality == []


# --- calendar features -----------------------------------------------------


class TestCalendarFeatures:
    def test_extracts_utc_hour_and_weekday(self) -> None:
        features = calendar_features(datetime(2024, 3, 4, 13, 30, tzinfo=UTC))
        assert features["hour_of_day"] == 13
        assert features["day_of_week"] == 0  # 2024-03-04 is a Monday

    def test_weekend_detection(self) -> None:
        assert calendar_features(datetime(2024, 3, 9, tzinfo=UTC))["is_weekend"]
        assert calendar_features(datetime(2024, 3, 10, tzinfo=UTC))["is_weekend"]
        assert not calendar_features(datetime(2024, 3, 11, tzinfo=UTC))["is_weekend"]

    def test_cyclical_encodings_are_on_the_unit_circle(self) -> None:
        for hour in range(24):
            features = calendar_features(datetime(2024, 3, 4, hour, tzinfo=UTC))
            radius = features["hour_sin"] ** 2 + features["hour_cos"] ** 2
            assert radius == pytest.approx(1.0)

    def test_cyclical_encodings_stay_within_range(self) -> None:
        for day in range(1, 8):
            features = calendar_features(datetime(2024, 3, day, tzinfo=UTC))
            for key in ("hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"):
                assert -1.0 <= features[key] <= 1.0

    def test_hour_zero_and_twenty_four_are_adjacent(self) -> None:
        midnight = calendar_features(datetime(2024, 3, 4, 0, tzinfo=UTC))
        late = calendar_features(datetime(2024, 3, 4, 23, tzinfo=UTC))
        gap = math.dist(
            (midnight["hour_sin"], midnight["hour_cos"]),
            (late["hour_sin"], late["hour_cos"]),
        )
        assert gap < 0.3, "the encoding must wrap around midnight"

    def test_non_utc_input_is_converted_before_extraction(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        local = datetime(2024, 3, 4, 20, tzinfo=eastern)  # 01:00 UTC on the 5th
        features = calendar_features(local)
        assert features["hour_of_day"] == 1
        assert features["day_of_week"] == 1  # Tuesday in UTC

    def test_base_time_fixture_is_consistent(self) -> None:
        features = calendar_features(BASE_TIME)
        assert features["hour_of_day"] == 12
        assert not features["is_weekend"]
