"""Tests for the point-in-time feature engine.

The invariance tests in this module are the phase's core guarantee.  If they
pass, the timing contract in ``docs/temporal-semantics.md`` holds; if any of
them fails, every downstream detection metric is meaningless.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

import pytest

from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    ClientType,
    FailureReason,
    MFAOutcome,
)
from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.exceptions import FeatureComputationError
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    LeakageClass,
    build_catalog,
)
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    BaselineConfig,
    FeatureConfig,
    GeospatialConfig,
    SplitConfig,
)
from password_attack_detector.features.engine import (
    FeatureEngine,
    FeatureFrame,
    sort_events_canonically,
)
from tests.features.factories import make_event, make_stream
from tests.features.reference_engine import reference_columns, reference_rows

_OUTCOMES = ("success", "failure", "blocked", "challenged")


@pytest.fixture()
def config() -> FeatureConfig:
    return FeatureConfig()


def _run(events: list[AuthEvent], config: FeatureConfig | None = None) -> FeatureFrame:
    resolved = config if config is not None else FeatureConfig()
    return FeatureEngine(resolved).run(events)


def _prior_only_columns(frame: FeatureFrame) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in frame.catalog.specs_for_leakage_class(LeakageClass.PRIOR_ONLY)
    )


def _random_stream(seed: int, count: int) -> list[AuthEvent]:
    """Build a pseudo-random but reproducible stream with dense collisions.

    Timestamps are drawn from a small grid so simultaneous events are common:
    the same-timestamp path must be exercised heavily, not incidentally.
    """
    rng = random.Random(seed)
    events: list[AuthEvent] = []
    for index in range(count):
        events.append(
            make_event(
                t=float(rng.randrange(0, 4000, 10)),
                user=f"u{rng.randint(1, 6)}",
                source=f"s{rng.randint(1, 5)}",
                device=f"d{rng.randint(1, 4)}",
                session=f"sess{rng.randint(1, 5)}",
                application=f"app-{rng.randint(0, 2)}",
                outcome=rng.choice(_OUTCOMES),
                method=rng.choice([AuthMethod.PASSWORD, AuthMethod.SSO]),
                mfa_outcome=rng.choice([None, MFAOutcome.PASSED, MFAOutcome.FAILED]),
                country=rng.choice([None, "US", "GB", "DE"]),
                user_agent=rng.choice([None, "chrome", "firefox"]),
                client_type=rng.choice([None, ClientType.WEB_BROWSER]),
                response_time_ms=rng.choice([None, 50, 120, 900, 4000]),
                key=str(index),
            )
        )
    return events


# --- output shape ----------------------------------------------------------


class TestOutputShape:
    def test_one_row_per_event(self) -> None:
        events = _random_stream(1, 40)
        assert len(_run(events)) == len(events)

    def test_columns_match_the_catalog_exactly_and_in_order(
        self, config: FeatureConfig
    ) -> None:
        frame = _run(_random_stream(2, 20), config)
        expected = build_catalog(config).column_order()
        for row in frame.rows:
            assert tuple(row) == expected

    def test_rows_are_in_canonical_order(self) -> None:
        events = _random_stream(3, 40)
        frame = _run(events)
        expected = [str(e.event_id) for e in sort_events_canonically(events)]
        assert frame.column(ANCHOR_EVENT_ID) == expected

    def test_anchor_times_are_non_decreasing(self) -> None:
        times = _run(_random_stream(4, 40)).column(ANCHOR_EVENT_TIME)
        assert times == sorted(times)

    def test_anchor_ids_are_unique(self) -> None:
        ids = _run(_random_stream(5, 60)).column(ANCHOR_EVENT_ID)
        assert len(set(ids)) == len(ids)

    def test_schema_version_is_stamped_on_every_row(self) -> None:
        frame = _run(_random_stream(6, 10))
        assert set(frame.column("feature_schema_version")) == {FEATURE_SCHEMA_VERSION}

    def test_empty_input_yields_no_rows(self) -> None:
        assert len(_run([])) == 0

    def test_unknown_column_lookup_raises(self) -> None:
        with pytest.raises(FeatureComputationError):
            _run(_random_stream(7, 3)).column("not_a_feature")


# --- the point-in-time contract -------------------------------------------


class TestAnchorExclusion:
    def test_anchor_never_enters_its_own_history(self) -> None:
        frame = _run([make_event(t=0.0, outcome="failure")])
        row = frame.rows[0]
        assert row["user_attempt_count__5m"] == 0
        assert row["user_failure_count__5m"] == 0
        assert row["source_attempt_count__1m"] == 0

    def test_history_grows_only_with_earlier_events(self) -> None:
        frame = _run(make_stream("0 u1 s1 failure\n10 u1 s1 failure\n20 u1 s1 failure"))
        assert frame.column("user_attempt_count__5m") == [0, 1, 2]

    def test_current_outcome_is_context_not_history(self) -> None:
        frame = _run([make_event(t=0.0, outcome="failure")])
        row = frame.rows[0]
        assert row["current_authentication_outcome"] == "failure"
        assert row["user_failure_count__1m"] == 0
        assert row["prior_consecutive_user_failures"] == 0


class TestSameTimestampExclusion:
    def test_simultaneous_peers_see_identical_history(self) -> None:
        # Two earlier events, then three at one instant.  All three must see
        # exactly the two earlier events -- not each other.
        events = make_stream(
            """
            0   u1 s1 failure
            10  u1 s1 failure
            60  u1 s2 failure
            60  u1 s3 failure
            60  u1 s4 failure
            """
        )
        frame = _run(events)
        simultaneous = frame.rows[2:]
        assert len(simultaneous) == 3
        for row in simultaneous:
            assert row["user_attempt_count__5m"] == 2

    def test_every_prior_only_column_agrees_across_a_block(self) -> None:
        events = make_stream(
            """
            0   u1 s1 failure
            10  u1 s1 success
            60  u1 s1 failure
            60  u1 s1 failure
            60  u1 s1 failure
            """
        )
        frame = _run(events)
        block = frame.rows[2:]
        for column in _prior_only_columns(frame):
            values = [row[column] for row in block]
            assert all(v == values[0] for v in values), column

    def test_a_peer_does_not_appear_in_unique_counts(self) -> None:
        events = make_stream(
            """
            0   u1 s1 failure
            60  u1 s2 failure
            60  u1 s3 failure
            """
        )
        frame = _run(events)
        for row in frame.rows[1:]:
            assert row["user_unique_source_count__5m"] == 1

    def test_a_peer_does_not_appear_in_sequence_features(self) -> None:
        events = make_stream("0 u1 s1 success\n60 u1 s1 failure\n60 u1 s1 failure")
        frame = _run(events)
        for row in frame.rows[1:]:
            assert row["prior_consecutive_user_failures"] == 0
            assert row["previous_user_outcome"] == "success"

    def test_simultaneous_events_have_zero_elapsed_semantics(self) -> None:
        events = make_stream("0 u1 s1 failure\n0 u1 s1 failure")
        frame = _run(events)
        # Neither sees the other, so neither has a previous event at all.
        for row in frame.rows:
            assert row["seconds_since_user_previous_event"] is None

    def test_block_size_does_not_change_earlier_rows(self) -> None:
        base = make_stream("0 u1 s1 failure\n60 u1 s2 failure")
        wider = [*base, make_event(t=60.0, user="u1", source="s9", key="extra")]
        first = _run(base).rows[0]
        second = _run(wider).rows[0]
        assert first == second


class TestWindowBoundaries:
    def test_left_edge_is_inclusive_and_right_edge_exclusive(self) -> None:
        # Anchor at t=300. A 5m window is [0, 300).
        events = [
            make_event(t=-0.000001, user="u1", key="before"),
            make_event(t=0.0, user="u1", key="left_edge"),
            make_event(t=299.999999, user="u1", key="inside"),
            make_event(t=300.0, user="u1", key="anchor"),
        ]
        frame = _run(events)
        anchor_row = frame.rows[-1]
        assert anchor_row["user_attempt_count__5m"] == 2

    def test_each_window_sees_its_own_span(self) -> None:
        events = [
            make_event(t=-3000.0, user="u1", key="old"),
            make_event(t=-100.0, user="u1", key="recent"),
            make_event(t=-10.0, user="u1", key="fresh"),
            make_event(t=0.0, user="u1", key="anchor"),
        ]
        row = _run(events).rows[-1]
        assert row["user_attempt_count__1m"] == 1
        assert row["user_attempt_count__5m"] == 2
        assert row["user_attempt_count__1h"] == 3

    def test_events_outside_every_window_are_forgotten(self) -> None:
        events = [
            make_event(t=0.0, user="u1", key="ancient"),
            make_event(t=200_000.0, user="u1", key="anchor"),
        ]
        row = _run(events).rows[-1]
        assert row["user_attempt_count__24h"] == 0


# --- invariance ------------------------------------------------------------


def _pinned_config() -> FeatureConfig:
    """A config with no fraction-derived boundaries and no baseline.

    Invariance under future mutation is a property of the *engine*.  With
    fraction-derived split boundaries and a fitted baseline in play, appending
    an event would legitimately move the boundary and therefore the baseline,
    changing baseline-derived columns for early rows.  Pinning both isolates
    the property under test.
    """
    return FeatureConfig(
        baseline=BaselineConfig(enabled=False),
        split=SplitConfig(strict_isolation=False),
    )


class TestFutureMutationInvariance:
    def test_appending_a_later_event_changes_nothing_earlier(self) -> None:
        config = _pinned_config()
        base = _random_stream(11, 60)
        cutoff = max(e.event_time for e in base)

        extended = [*base, make_event(t=cutoff + timedelta(hours=1), key="future")]

        before = {r[ANCHOR_EVENT_ID]: r for r in _run(base, config).rows}
        after = {r[ANCHOR_EVENT_ID]: r for r in _run(extended, config).rows}

        for event_id, row in before.items():
            assert after[event_id] == row, f"row for {event_id} changed"

    def test_inserting_a_mid_stream_event_changes_nothing_at_or_before_it(
        self,
    ) -> None:
        config = _pinned_config()
        base = sort_events_canonically(_random_stream(12, 80))
        pivot = base[40].event_time

        inserted = make_event(t=pivot, user="u1", source="s1", key="inserted")
        extended = [*base, inserted]

        before = {r[ANCHOR_EVENT_ID]: r for r in _run(base, config).rows}
        after = {r[ANCHOR_EVENT_ID]: r for r in _run(extended, config).rows}

        unchanged = [e for e in base if e.event_time <= pivot]
        assert len(unchanged) > 5, "the fixture must actually exercise the boundary"
        for event in unchanged:
            key = str(event.event_id)
            assert after[key] == before[key], (
                "an event inserted at t must not change any row at or before t"
            )

    def test_modifying_a_later_event_changes_nothing_earlier(self) -> None:
        config = _pinned_config()
        base = sort_events_canonically(_random_stream(13, 60))
        pivot = base[30].event_time

        # Rewrite every later event's outcome, MFA result, and response time.
        # Between them these drive counts, rates, response-time statistics, and
        # every sequence counter -- so if any of it leaked backwards, an
        # earlier row would move.
        mutated = [
            e
            if e.event_time <= pivot
            else e.model_copy(
                update={
                    "authentication_outcome": AuthOutcome.FAILURE,
                    "failure_reason": FailureReason.INVALID_CREDENTIALS,
                    "mfa_outcome": MFAOutcome.FAILED,
                    "response_time_ms": 12345,
                }
            )
            for e in base
        ]

        before = {r[ANCHOR_EVENT_ID]: r for r in _run(base, config).rows}
        after = {r[ANCHOR_EVENT_ID]: r for r in _run(mutated, config).rows}

        for event in base:
            if event.event_time <= pivot:
                key = str(event.event_id)
                assert after[key] == before[key]

    def test_truncating_the_future_changes_nothing_earlier(self) -> None:
        config = _pinned_config()
        base = sort_events_canonically(_random_stream(14, 70))
        head = base[:40]

        full = {r[ANCHOR_EVENT_ID]: r for r in _run(base, config).rows}
        partial = {r[ANCHOR_EVENT_ID]: r for r in _run(head, config).rows}

        for event in head:
            key = str(event.event_id)
            assert partial[key] == full[key]


class TestDeterminism:
    def test_input_order_does_not_affect_output(self) -> None:
        events = _random_stream(21, 70)
        shuffled = list(events)
        random.Random(99).shuffle(shuffled)
        assert _run(events).rows == _run(shuffled).rows

    def test_repeated_runs_are_identical(self) -> None:
        events = _random_stream(22, 50)
        assert _run(events).rows == _run(events).rows

    def test_a_reused_engine_instance_does_not_carry_state(
        self, config: FeatureConfig
    ) -> None:
        engine = FeatureEngine(config)
        events = _random_stream(23, 40)
        first = engine.run(events)
        second = engine.run(events)
        assert first.rows == second.rows


class TestReferenceEquivalence:
    """The highest-value test in the phase.

    A naive O(n^2) implementation recomputes every windowed and sequence
    feature from scratch for each anchor.  Both sides accumulate in exact
    integers, so equality is exact -- no tolerance is permitted.
    """

    @pytest.mark.parametrize("seed", [101, 202, 303])
    def test_matches_the_naive_implementation_exactly(self, seed: int) -> None:
        config = _pinned_config()
        catalog = build_catalog(config)
        events = _random_stream(seed, 120)

        actual = FeatureEngine(config, catalog).run(events).rows
        expected = reference_rows(events, config, catalog)
        covered = sorted(reference_columns(catalog))

        assert len(actual) == len(expected)
        assert len(covered) > 100, "the oracle must cover the bulk of the catalog"

        for index, (produced, reference) in enumerate(
            zip(actual, expected, strict=True)
        ):
            for column in covered:
                assert produced[column] == reference[column], (
                    f"row {index}, column {column}: "
                    f"{produced[column]!r} != {reference[column]!r}"
                )

    def test_matches_on_a_dense_simultaneous_stream(self) -> None:
        # Every event lands on one of three timestamps, so almost all history
        # decisions are same-timestamp decisions.
        rng = random.Random(555)
        events = [
            make_event(
                t=float(rng.choice([0, 60, 120])),
                user=f"u{rng.randint(1, 3)}",
                source=f"s{rng.randint(1, 3)}",
                outcome=rng.choice(_OUTCOMES),
                response_time_ms=rng.choice([None, 10, 20]),
                key=str(index),
            )
            for index in range(40)
        ]
        config = _pinned_config()
        catalog = build_catalog(config)

        actual = FeatureEngine(config, catalog).run(events).rows
        expected = reference_rows(events, config, catalog)
        for produced, reference in zip(actual, expected, strict=True):
            for column in reference_columns(catalog):
                assert produced[column] == reference[column], column


# --- null and zero semantics ----------------------------------------------


class TestNullSemantics:
    @pytest.fixture()
    def cold_row(self) -> dict[str, Any]:
        return _run([make_event(t=0.0)]).rows[0]

    def test_counts_are_zero_for_a_cold_entity(self, cold_row: dict[str, Any]) -> None:
        for column in (
            "user_attempt_count__5m",
            "source_attempt_count__1m",
            "pair_attempt_count__5m",
            "user_unique_source_count__5m",
        ):
            assert cold_row[column] == 0, column

    def test_rates_are_null_not_zero_for_a_cold_entity(
        self, cold_row: dict[str, Any]
    ) -> None:
        assert cold_row["user_failure_rate__5m"] is None
        assert cold_row["user_success_rate__5m"] is None
        assert cold_row["source_failure_rate__1m"] is None

    def test_statistics_are_null_for_a_cold_entity(
        self, cold_row: dict[str, Any]
    ) -> None:
        for column in (
            "user_mean_response_time_ms__5m",
            "user_response_time_std_ms__5m",
            "user_mean_interarrival_seconds__5m",
            "source_interarrival_coefficient_of_variation__5m",
        ):
            assert cold_row[column] is None, column

    def test_elapsed_features_are_null_for_a_cold_entity(
        self, cold_row: dict[str, Any]
    ) -> None:
        for column in (
            "seconds_since_user_previous_event",
            "seconds_since_user_previous_success",
            "seconds_since_user_previous_failure",
            "seconds_since_pair_previous_event",
        ):
            assert cold_row[column] is None, column

    def test_sequence_counters_are_zero_for_a_cold_entity(
        self, cold_row: dict[str, Any]
    ) -> None:
        assert cold_row["prior_consecutive_user_failures"] == 0
        assert cold_row["prior_failures_since_user_success"] == 0

    def test_previous_outcome_is_null_for_a_cold_entity(
        self, cold_row: dict[str, Any]
    ) -> None:
        assert cold_row["previous_user_outcome"] is None
        assert cold_row["previous_pair_outcome"] is None

    def test_standard_deviation_of_one_observation_is_null(self) -> None:
        events = make_stream("0 u1 s1 failure response_time_ms=100\n10 u1 s1 failure")
        row = _run(events).rows[1]
        assert row["user_mean_response_time_ms__5m"] == pytest.approx(100.0)
        assert row["user_response_time_std_ms__5m"] is None

    def test_absent_response_times_are_not_treated_as_zero(self) -> None:
        events = make_stream(
            """
            0  u1 s1 failure response_time_ms=100
            10 u1 s1 failure
            20 u1 s1 failure response_time_ms=200
            30 u1 s1 failure
            """
        )
        row = _run(events).rows[-1]
        assert row["user_mean_response_time_ms__5m"] == pytest.approx(150.0)

    def test_rate_denominator_uses_all_attempts(self) -> None:
        events = make_stream(
            "0 u1 s1 failure\n1 u1 s1 failure\n2 u1 s1 success\n3 u1 s1 failure"
        )
        row = _run(events).rows[-1]
        assert row["user_failure_rate__5m"] == pytest.approx(2 / 3)
        assert row["user_success_rate__5m"] == pytest.approx(1 / 3)

    def test_min_count_for_rate_suppresses_thin_denominators(self) -> None:
        config = FeatureConfig(min_count_for_rate=5)
        events = make_stream("0 u1 s1 failure\n10 u1 s1 failure")
        row = _run(events, config).rows[-1]
        assert row["user_failure_rate__5m"] is None


# --- sequence features -----------------------------------------------------


class TestSequenceFeatures:
    def test_consecutive_failures_accumulate(self) -> None:
        frame = _run(make_stream("0 u1 s1 failure\n1 u1 s1 failure\n2 u1 s1 failure"))
        assert frame.column("prior_consecutive_user_failures") == [0, 1, 2]

    def test_a_success_breaks_the_run(self) -> None:
        frame = _run(
            make_stream(
                "0 u1 s1 failure\n1 u1 s1 failure\n2 u1 s1 success\n3 u1 s1 failure"
            )
        )
        assert frame.column("prior_consecutive_user_failures") == [0, 1, 2, 0]

    def test_a_blocked_outcome_breaks_the_run(self) -> None:
        frame = _run(make_stream("0 u1 s1 failure\n1 u1 s1 blocked\n2 u1 s1 failure"))
        assert frame.column("prior_consecutive_user_failures") == [0, 1, 0]

    def test_failures_since_success_ignores_non_success_outcomes(self) -> None:
        frame = _run(
            make_stream(
                "0 u1 s1 failure\n1 u1 s1 blocked\n2 u1 s1 failure\n3 u1 s1 failure"
            )
        )
        assert frame.column("prior_failures_since_user_success") == [0, 1, 1, 2]

    def test_failures_since_success_resets_on_success(self) -> None:
        frame = _run(
            make_stream(
                "0 u1 s1 failure\n1 u1 s1 failure\n2 u1 s1 success\n3 u1 s1 failure"
            )
        )
        assert frame.column("prior_failures_since_user_success") == [0, 1, 2, 0]

    def test_source_counters_are_independent_of_user_counters(self) -> None:
        frame = _run(make_stream("0 u1 s1 failure\n1 u2 s1 failure\n2 u1 s1 failure"))
        assert frame.column("prior_consecutive_source_failures") == [0, 1, 2]
        assert frame.column("prior_consecutive_user_failures") == [0, 0, 1]

    def test_elapsed_since_previous_event(self) -> None:
        frame = _run(make_stream("0 u1 s1 failure\n30 u1 s1 failure"))
        assert frame.rows[1]["seconds_since_user_previous_event"] == pytest.approx(30.0)

    def test_elapsed_since_previous_success_and_failure_differ(self) -> None:
        frame = _run(make_stream("0 u1 s1 success\n10 u1 s1 failure\n30 u1 s1 failure"))
        row = frame.rows[-1]
        assert row["seconds_since_user_previous_success"] == pytest.approx(30.0)
        assert row["seconds_since_user_previous_failure"] == pytest.approx(20.0)

    def test_previous_outcome_reflects_the_prior_event(self) -> None:
        frame = _run(make_stream("0 u1 s1 success\n10 u1 s1 failure"))
        assert frame.column("previous_user_outcome") == [None, "success"]

    def test_pair_features_track_the_user_source_combination(self) -> None:
        frame = _run(make_stream("0 u1 s1 failure\n1 u1 s2 failure\n2 u1 s1 failure"))
        assert frame.column("prior_failures_since_pair_success") == [0, 0, 1]

    def test_no_prior_field_observes_the_current_outcome(self) -> None:
        # A lone failure must not report itself as a prior failure anywhere.
        row = _run([make_event(t=0.0, outcome="failure")]).rows[0]
        for column, value in row.items():
            if column.startswith(("prior_", "previous_", "seconds_since_")):
                assert value in (0, None), column


# --- current-event context and calendar -----------------------------------


class TestCurrentContext:
    def test_reports_the_anchor_fields(self) -> None:
        event = make_event(
            t=0.0,
            outcome="success",
            method=AuthMethod.SSO,
            mfa_outcome=MFAOutcome.PASSED,
            client_type=ClientType.WEB_BROWSER,
            country="US",
            response_time_ms=250,
        )
        row = _run([event]).rows[0]
        assert row["current_authentication_outcome"] == "success"
        assert row["current_authentication_method"] == "sso"
        assert row["current_mfa_outcome"] == "passed"
        assert row["current_client_type"] == "web_browser"
        assert row["current_country_code"] == "US"
        assert row["current_response_time_ms"] == 250

    def test_absent_optional_fields_are_null(self) -> None:
        row = _run([make_event(t=0.0)]).rows[0]
        assert row["current_mfa_outcome"] is None
        assert row["current_client_type"] is None
        assert row["current_country_code"] is None
        assert row["current_response_time_ms"] is None

    def test_location_presence_is_reported(self) -> None:
        with_location = make_event(t=0.0, latitude=37.8, longitude=-122.4, key="a")
        without = make_event(t=10.0, key="b")
        frame = _run([with_location, without])
        assert frame.column("current_has_location") == [True, False]

    def test_calendar_features_use_utc(self) -> None:
        row = _run([make_event(t="2024-03-04T13:30:00Z")]).rows[0]
        assert row["hour_of_day"] == 13
        assert row["day_of_week"] == 0
        assert row["is_weekend"] is False


# --- geospatial integration -----------------------------------------------


class TestGeospatialIntegration:
    def test_no_prior_success_is_reported_as_such(self) -> None:
        row = _run([make_event(t=0.0, latitude=37.8, longitude=-122.4)]).rows[0]
        assert row["user_previous_success_geo__status"] == "no_prior_success"
        assert row["distance_km_from_user_previous_success"] is None

    def test_distance_is_measured_from_the_previous_located_success(self) -> None:
        events = [
            make_event(
                t=0.0,
                outcome="success",
                latitude=51.5,
                longitude=-0.1,
                country="GB",
                key="london",
            ),
            make_event(
                t=3600.0,
                outcome="failure",
                latitude=48.9,
                longitude=2.35,
                country="FR",
                key="paris",
            ),
        ]
        row = _run(events).rows[1]
        assert row["user_previous_success_geo__status"] == "ok"
        assert row["distance_km_from_user_previous_success"] == pytest.approx(
            343.0, abs=10.0
        )
        assert row["country_changed_since_previous_success"] is True

    def test_missing_current_location_is_distinguished_from_no_success(self) -> None:
        events = [
            make_event(t=0.0, outcome="success", latitude=51.5, longitude=-0.1),
            make_event(t=60.0, outcome="failure", key="nowhere"),
        ]
        row = _run(events).rows[1]
        assert row["user_previous_success_geo__status"] == "missing_current_location"

    def test_missing_prior_location_is_distinguished(self) -> None:
        events = [
            make_event(t=0.0, outcome="success", key="unlocated"),
            make_event(t=60.0, latitude=51.5, longitude=-0.1, key="located"),
        ]
        row = _run(events).rows[1]
        assert row["user_previous_success_geo__status"] == "missing_prior_location"

    def test_velocity_is_capped_rather_than_unbounded(self) -> None:
        config = FeatureConfig(
            geospatial=GeospatialConfig(max_plausible_velocity_kmh=100.0)
        )
        events = [
            make_event(t=0.0, outcome="success", latitude=51.5, longitude=-0.1),
            make_event(t=60.0, latitude=-33.9, longitude=151.2, key="sydney"),
        ]
        row = _run(events, config).rows[1]
        assert row["implied_velocity__status"] == "capped"
        assert row["implied_velocity_kmh_from_previous_success"] == pytest.approx(100.0)

    def test_simultaneous_success_yields_zero_elapsed_not_a_division_error(
        self,
    ) -> None:
        # The prior success is at the same instant, so it is invisible to the
        # anchor; the status must say so rather than divide by zero.
        events = [
            make_event(
                t=0.0, outcome="success", latitude=51.5, longitude=-0.1, key="a"
            ),
            make_event(t=0.0, latitude=48.9, longitude=2.35, key="b"),
        ]
        row = _run(events).rows[1]
        assert row["user_previous_success_geo__status"] == "no_prior_success"
        assert row["implied_velocity_kmh_from_previous_success"] is None

    def test_no_column_asserts_impossible_travel(self) -> None:
        frame = _run(_random_stream(31, 10))
        for column in frame.rows[0]:
            assert "impossible" not in column


# --- baseline placeholder --------------------------------------------------


class TestBaselineAbsent:
    def test_baseline_columns_are_null_without_a_fitted_baseline(self) -> None:
        row = _run([make_event(t=0.0)]).rows[0]
        assert row["is_new_device_for_user"] is None
        assert row["response_time_zscore"] is None

    def test_coverage_flags_are_false_without_a_baseline(self) -> None:
        row = _run([make_event(t=0.0)]).rows[0]
        assert row["user_in_baseline"] is False
        assert row["source_in_baseline"] is False

    def test_a_cold_entity_is_never_reported_as_novel(self) -> None:
        # Null, not True: "never seen this user" is not "this device is new".
        row = _run([make_event(t=0.0)]).rows[0]
        for column in (
            "is_new_device_for_user",
            "is_new_source_for_user",
            "is_new_country_for_user",
        ):
            assert row[column] is not True, column


# --- privacy ---------------------------------------------------------------


class TestNoLeakageInOutput:
    def test_no_pseudonyms_appear_in_any_row(self) -> None:
        events = _random_stream(41, 20)
        frame = _run(events)
        identifiers = {e.user_id for e in events} | {e.source_id for e in events}
        for row in frame.rows:
            for value in row.values():
                if isinstance(value, str):
                    assert value not in identifiers

    def test_no_coordinates_are_echoed(self) -> None:
        event = make_event(t=0.0, latitude=37.8123, longitude=-122.4567)
        row = _run([event]).rows[0]
        assert 37.8123 not in row.values()
        assert -122.4567 not in row.values()

    def test_the_only_identifier_is_the_anchor_event_id(self) -> None:
        events = _random_stream(42, 10)
        frame = _run(events)
        expected = {str(e.event_id) for e in events}
        for row in frame.rows:
            assert row[ANCHOR_EVENT_ID] in expected


# --- bounded state and complexity -----------------------------------------


class TestBoundedState:
    def test_entity_state_is_released_after_the_longest_window(self) -> None:
        # Two bursts separated by more than 24h: state from the first burst
        # must be released rather than retained for the whole run.
        early = [make_event(t=float(i), user=f"u{i}", key=f"e{i}") for i in range(30)]
        late = [
            make_event(t=200_000.0 + i, user=f"v{i}", key=f"l{i}") for i in range(5)
        ]
        frame = _run([*early, *late])
        assert frame.stats.expired_entity_count > 0
        assert frame.stats.tracked_entity_count < frame.stats.peak_tracked_entity_count

    def test_each_event_is_appended_once_per_tracked_entity(self) -> None:
        events = _random_stream(51, 50)
        frame = _run(events)
        # Six entity kinds are tracked; every event is appended to each exactly
        # once.  A quadratic implementation would exceed this by orders of
        # magnitude.
        assert frame.stats.buffer_appends == len(events) * 6

    def test_work_grows_linearly_not_quadratically(self) -> None:
        small = _run(_random_stream(52, 50)).stats
        large = _run(_random_stream(52, 200)).stats
        ratio = large.buffer_appends / small.buffer_appends
        assert ratio == pytest.approx(4.0, abs=0.01)

    def test_capacity_cap_evicts_and_reports(self) -> None:
        config = FeatureConfig(max_tracked_entities=3)
        events = [
            make_event(t=float(i), user=f"u{i}", source=f"s{i}", key=str(i))
            for i in range(40)
        ]
        frame = _run(events, config)
        assert frame.stats.capacity_evicted_entity_count > 0
        assert len(frame) == len(events), "eviction must not drop rows"

    def test_eviction_is_never_silent(self) -> None:
        config = FeatureConfig(max_tracked_entities=2)
        events = [make_event(t=float(i), user=f"u{i}", key=str(i)) for i in range(20)]
        stats = _run(events, config).stats
        assert stats.capacity_evicted_entity_count > 0
        assert "capacity_evicted_entity_count" in stats.as_dict()

    def test_stats_are_reported_as_a_plain_mapping(self) -> None:
        stats = _run(_random_stream(53, 10)).stats.as_dict()
        assert stats["rows_emitted"] == 10
        assert stats["events_ingested"] == 10
        assert all(isinstance(v, int) for v in stats.values())


# --- catalog agreement -----------------------------------------------------


class TestCatalogAgreement:
    def test_trimmed_catalog_produces_trimmed_output(self) -> None:
        config = FeatureConfig(
            cardinality_windows=("24h",),
            baseline=BaselineConfig(rate_reference_window="24h"),
        )
        frame = _run(_random_stream(61, 10), config)
        assert "user_unique_source_count__24h" in frame.rows[0]
        assert "user_unique_source_count__5m" not in frame.rows[0]

    def test_disabling_geospatial_removes_those_columns(self) -> None:
        config = FeatureConfig(geospatial=GeospatialConfig(enabled=False))
        frame = _run(_random_stream(62, 10), config)
        assert "implied_velocity__status" not in frame.rows[0]

    def test_disabling_baseline_removes_those_columns(self) -> None:
        config = FeatureConfig(baseline=BaselineConfig(enabled=False))
        frame = _run(_random_stream(63, 10), config)
        assert "user_in_baseline" not in frame.rows[0]

    def test_engine_rejects_a_catalog_it_cannot_implement(
        self, config: FeatureConfig
    ) -> None:
        from password_attack_detector.features.catalog import FeatureCatalog

        catalog = build_catalog(config)
        broken = catalog.get("user_attempt_count__5m").model_copy(
            update={"name": "user_mystery_measure__5m"}
        )
        specs = tuple(
            broken if s.name == "user_attempt_count__5m" else s for s in catalog.specs
        )
        with pytest.raises(FeatureComputationError, match="no implementation"):
            FeatureEngine(
                config,
                FeatureCatalog(specs, config_fingerprint=config.fingerprint()),
            )
