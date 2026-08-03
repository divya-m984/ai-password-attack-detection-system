"""Tests for behavioral baseline fitting and transformation."""

from __future__ import annotations

import json
import random
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.data.enums import AuthMethod, ClientType, MFAOutcome
from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.exceptions import ArtifactNotFoundError, BaselineFitError
from password_attack_detector.features.baselines import (
    BASELINE_COLUMNS,
    BASELINE_JSON,
    BehavioralBaselineModel,
    fit_baseline,
)
from password_attack_detector.features.config import BaselineConfig, FeatureConfig
from password_attack_detector.features.engine import FeatureEngine
from tests.features.factories import BASE_TIME, make_event

_INTERVAL = (BASE_TIME - timedelta(hours=1), BASE_TIME + timedelta(days=2))


def _training_events(seed: int = 7, count: int = 120) -> list[AuthEvent]:
    """A reproducible stream with enough volume to clear the fit thresholds."""
    rng = random.Random(seed)
    return [
        make_event(
            t=float(index * 60),
            user=f"u{rng.randint(1, 4)}",
            source=f"s{rng.randint(1, 3)}",
            device=f"d{rng.randint(1, 3)}",
            application=f"app-{rng.randint(0, 1)}",
            outcome=rng.choice(["success", "success", "failure"]),
            method=rng.choice([AuthMethod.PASSWORD, AuthMethod.SSO]),
            country=rng.choice(["US", "GB"]),
            client_type=rng.choice([ClientType.WEB_BROWSER, ClientType.MOBILE_APP]),
            user_agent=rng.choice(["chrome", "firefox"]),
            response_time_ms=rng.randint(50, 500),
            latitude=rng.choice([37.8, 51.5]),
            longitude=rng.choice([-122.4, -0.1]),
            key=str(index),
        )
        for index in range(count)
    ]


def _fit(
    events: list[AuthEvent] | None = None,
    config: BaselineConfig | None = None,
) -> BehavioralBaselineModel:
    resolved = _training_events() if events is None else events
    model = BehavioralBaselineModel(config if config is not None else BaselineConfig())
    return model.fit(
        resolved,
        permitted_event_ids=frozenset(e.event_id for e in resolved),
        interval=_INTERVAL,
    )


# --- fit permissions -------------------------------------------------------


class TestFitPermissions:
    def test_fit_populates_the_model(self) -> None:
        model = _fit()
        assert model.is_fitted
        assert model.user_count > 0
        assert model.source_count > 0

    def test_unfitted_model_reports_so(self) -> None:
        model = BehavioralBaselineModel(BaselineConfig())
        assert not model.is_fitted
        with pytest.raises(BaselineFitError, match="not been fitted"):
            _ = model.artifact

    def test_an_event_outside_the_permitted_set_is_rejected(self) -> None:
        events = _training_events(count=20)
        model = BehavioralBaselineModel(BaselineConfig())
        permitted = frozenset(e.event_id for e in events[:-1])
        with pytest.raises(BaselineFitError, match="not in the permitted set"):
            model.fit(events, permitted_event_ids=permitted, interval=_INTERVAL)

    def test_rejection_is_not_a_silent_skip(self) -> None:
        # The whole point of the permitted set is to surface the mistake.  A
        # baseline that quietly dropped the offending events would still be
        # fitted, and the error would go unnoticed.
        events = _training_events(count=20)
        model = BehavioralBaselineModel(BaselineConfig())
        with pytest.raises(BaselineFitError):
            model.fit(events, permitted_event_ids=frozenset(), interval=_INTERVAL)
        assert not model.is_fitted

    def test_an_event_outside_the_interval_is_rejected(self) -> None:
        events = _training_events(count=20)
        narrow = (BASE_TIME, BASE_TIME + timedelta(minutes=5))
        model = BehavioralBaselineModel(BaselineConfig())
        with pytest.raises(BaselineFitError, match="outside the fitted interval"):
            model.fit(
                events,
                permitted_event_ids=frozenset(e.event_id for e in events),
                interval=narrow,
            )

    def test_inverted_interval_is_rejected(self) -> None:
        events = _training_events(count=5)
        model = BehavioralBaselineModel(BaselineConfig())
        with pytest.raises(BaselineFitError, match="non-empty and increasing"):
            model.fit(
                events,
                permitted_event_ids=frozenset(e.event_id for e in events),
                interval=(_INTERVAL[1], _INTERVAL[0]),
            )

    def test_artifact_records_what_was_consumed(self) -> None:
        events = _training_events(count=40)
        artifact = _fit(events).artifact
        assert artifact.total_fit_events == 40
        assert len(artifact.fitted_source_fingerprint) == 64

    def test_source_fingerprint_identifies_the_fit_events(self) -> None:
        # The leakage auditor recomputes this independently from the split
        # table, so a baseline fitted on the wrong events is detectable.
        from password_attack_detector.data.serialization import (
            compute_events_fingerprint,
        )

        events = _training_events(count=30)
        assert _fit(events).artifact.fitted_source_fingerprint == (
            compute_events_fingerprint(events)
        )

    def test_in_sample_caveat_is_recorded(self) -> None:
        assert _fit().artifact.baseline_in_sample_for_train is True


# --- fitted content --------------------------------------------------------


class TestFittedContent:
    def test_known_sets_capture_observed_values(self) -> None:
        events = [
            make_event(t=float(i), user="u1", device=f"d{i % 3}", key=str(i))
            for i in range(20)
        ]
        model = _fit(events)
        assert model.has_user(events[0].user_id)

    def test_hour_histogram_sums_to_one(self) -> None:
        model = _fit()
        table = model._user_table_data()
        for histogram in table["hour_histogram"]:
            assert sum(histogram) == pytest.approx(1.0)

    def test_hour_histogram_has_one_bucket_per_hour(self) -> None:
        model = _fit()
        for histogram in model._user_table_data()["hour_histogram"]:
            assert len(histogram) == 24

    def test_thin_entities_get_null_rates(self) -> None:
        config = BaselineConfig(min_events_per_user=100, min_events_per_source=100)
        model = _fit(_training_events(count=20), config)
        for rate in model._user_table_data()["success_rate"]:
            assert rate is None

    def test_known_sets_are_capped(self) -> None:
        events = [
            make_event(t=float(i), user="u1", device=f"d{i}", key=str(i))
            for i in range(40)
        ]
        model = _fit(events, BaselineConfig(known_set_max_size=5))
        devices = model._user_table_data()["known_device_ids"][0]
        assert len(devices) == 5
        assert model.artifact.truncated_set_count > 0

    def test_truncation_is_recorded_not_silent(self) -> None:
        events = [
            make_event(t=float(i), user="u1", device=f"d{i}", key=str(i))
            for i in range(40)
        ]
        model = _fit(events, BaselineConfig(known_set_max_size=3))
        assert "known_device_ids" in model._user_table_data()["truncated_sets"][0]

    def test_min_occurrences_filters_rare_values(self) -> None:
        events = [
            make_event(t=float(i), user="u1", device="d1", key=str(i))
            for i in range(10)
        ]
        events.append(make_event(t=999.0, user="u1", device="d_rare", key="rare"))
        model = _fit(events, BaselineConfig(known_set_min_occurrences=2))
        devices = model._user_table_data()["known_device_ids"][0]
        assert len(devices) == 1

    def test_source_tracks_targeted_user_count(self) -> None:
        events = [
            make_event(t=float(i), user=f"u{i}", source="s1", key=str(i))
            for i in range(7)
        ]
        model = _fit(events)
        assert model._source_table_data()["targeted_user_count"][0] == 7

    def test_entropy_is_zero_for_a_single_observed_value(self) -> None:
        events = [
            make_event(t=float(i), source="s1", client_type=ClientType.BOT, key=str(i))
            for i in range(10)
        ]
        model = _fit(events)
        assert model._source_table_data()["client_type_entropy"][0] == pytest.approx(
            0.0
        )

    def test_entropy_is_one_bit_for_an_even_two_way_split(self) -> None:
        events = [
            make_event(
                t=float(i),
                source="s1",
                client_type=(ClientType.BOT if i % 2 else ClientType.WEB_BROWSER),
                key=str(i),
            )
            for i in range(10)
        ]
        model = _fit(events)
        assert model._source_table_data()["client_type_entropy"][0] == pytest.approx(
            1.0
        )

    def test_centroid_uses_only_located_events(self) -> None:
        events = [
            make_event(t=0.0, user="u1", latitude=10.0, longitude=20.0, key="a"),
            make_event(t=1.0, user="u1", latitude=30.0, longitude=40.0, key="b"),
            make_event(t=2.0, user="u1", key="unlocated"),
        ]
        model = _fit(events)
        table = model._user_table_data()
        assert table["centroid_latitude"][0] == pytest.approx(20.0)
        assert table["located_event_count"][0] == 2


# --- determinism -----------------------------------------------------------


class TestDeterminism:
    def test_fingerprint_is_independent_of_input_order(self) -> None:
        events = _training_events(count=80)
        shuffled = list(events)
        random.Random(1234).shuffle(shuffled)
        assert (
            _fit(events).artifact.content_fingerprint
            == _fit(shuffled).artifact.content_fingerprint
        )

    def test_fingerprint_excludes_the_creation_timestamp(self) -> None:
        events = _training_events(count=40)
        first = _fit(events).artifact
        second = _fit(events).artifact
        assert first.content_fingerprint == second.content_fingerprint

    def test_fingerprint_reacts_to_a_content_change(self) -> None:
        base = _training_events(count=40)
        extended = [*base, make_event(t=99999.0, user="u_new", key="extra")]
        assert (
            _fit(base).artifact.content_fingerprint
            != _fit(extended).artifact.content_fingerprint
        )

    def test_fingerprint_reacts_to_a_config_change(self) -> None:
        events = _training_events(count=40)
        assert (
            _fit(events).artifact.content_fingerprint
            != _fit(
                events, BaselineConfig(known_set_max_size=2)
            ).artifact.content_fingerprint
        )

    def test_fingerprint_is_hex_sha256(self) -> None:
        fingerprint = _fit().artifact.content_fingerprint
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")


# --- transform purity ------------------------------------------------------


class TestTransformPurity:
    def test_transform_does_not_change_the_fitted_state(self) -> None:
        model = _fit()
        before = model.artifact.content_fingerprint
        model.transform(_training_events(seed=99, count=60))
        assert model.artifact.content_fingerprint == before

    def test_transforming_unseen_entities_does_not_add_them(self) -> None:
        model = _fit()
        users_before = model.user_count
        model.transform([make_event(t=0.0, user="stranger", source="unknown")])
        assert model.user_count == users_before

    def test_evaluation_data_cannot_move_a_training_baseline(self) -> None:
        model = _fit()
        before = model.artifact.content_fingerprint
        evaluation = _training_events(seed=555, count=200)
        model.transform(evaluation)
        assert model.artifact.content_fingerprint == before

    def test_fitted_state_is_frozen(self) -> None:
        model = _fit()
        baseline = next(iter(model._users.values()))
        with pytest.raises(AttributeError):
            baseline.event_count = 999  # type: ignore[misc]


# --- transform output ------------------------------------------------------


class TestTransformOutput:
    def test_produces_exactly_the_declared_columns(self) -> None:
        model = _fit()
        row = model.transform_one(make_event(t=0.0))
        assert set(row) == set(BASELINE_COLUMNS)

    def test_unknown_entities_get_nulls_and_false_coverage(self) -> None:
        model = _fit()
        row = model.transform_one(make_event(t=0.0, user="stranger", source="ghost"))
        assert row["user_in_baseline"] is False
        assert row["source_in_baseline"] is False
        for column in BASELINE_COLUMNS:
            if column.endswith("_in_baseline"):
                continue
            assert row[column] is None, column

    def test_a_cold_user_is_never_reported_as_having_a_new_device(self) -> None:
        # Null, not True: "never seen this user" is a different observation
        # from "know this user, and this device is new".
        model = _fit()
        row = model.transform_one(make_event(t=0.0, user="stranger"))
        assert row["is_new_device_for_user"] is None

    def test_a_known_device_is_not_flagged_as_new(self) -> None:
        events = [
            make_event(t=float(i), user="u1", device="d1", key=str(i))
            for i in range(10)
        ]
        model = _fit(events)
        row = model.transform_one(make_event(t=500.0, user="u1", device="d1"))
        assert row["user_in_baseline"] is True
        assert row["is_new_device_for_user"] is False

    def test_an_unseen_device_is_flagged_as_new(self) -> None:
        events = [
            make_event(t=float(i), user="u1", device="d1", key=str(i))
            for i in range(10)
        ]
        model = _fit(events)
        row = model.transform_one(make_event(t=500.0, user="u1", device="d_novel"))
        assert row["is_new_device_for_user"] is True

    def test_novelty_for_an_absent_country_is_null(self) -> None:
        events = [
            make_event(t=float(i), user="u1", country="US", key=str(i))
            for i in range(10)
        ]
        model = _fit(events)
        row = model.transform_one(make_event(t=500.0, user="u1"))
        assert row["is_new_country_for_user"] is None

    def test_login_hour_deviation_is_within_the_unit_interval(self) -> None:
        model = _fit()
        row = model.transform_one(make_event(t=0.0, user="u1"))
        assert row["user_in_baseline"] is True
        assert 0.0 <= row["login_hour_deviation"] <= 1.0

    def test_response_time_zscore_is_clipped(self) -> None:
        events = [
            make_event(t=float(i), user="u1", response_time_ms=100 + i, key=str(i))
            for i in range(20)
        ]
        model = _fit(events, BaselineConfig(max_response_time_zscore=3.0))
        row = model.transform_one(
            make_event(t=500.0, user="u1", response_time_ms=30000)
        )
        assert row["response_time_zscore"] == pytest.approx(3.0)

    def test_zscore_is_null_without_a_response_time(self) -> None:
        model = _fit()
        row = model.transform_one(make_event(t=0.0, user="u1"))
        assert row["response_time_zscore"] is None

    def test_centroid_distance_is_reported(self) -> None:
        events = [
            make_event(t=float(i), user="u1", latitude=51.5, longitude=-0.1, key=str(i))
            for i in range(10)
        ]
        model = _fit(events)
        row = model.transform_one(
            make_event(t=500.0, user="u1", latitude=48.9, longitude=2.35)
        )
        assert row["distance_from_user_baseline_centroid_km"] == pytest.approx(
            343.0, rel=0.05
        )

    def test_ratio_features_are_null_without_windowed_context(self) -> None:
        model = _fit()
        row = model.transform_one(make_event(t=0.0, user="u1"))
        assert row["user_event_rate_ratio"] is None

    def test_ratio_features_use_the_reference_window(self) -> None:
        model = _fit()
        event = make_event(t=0.0, user="u1")
        row = model.transform_one(
            event,
            {"user_attempt_count__1h": 10, "source_attempt_count__1h": 4},
        )
        assert row["user_event_rate_ratio"] is not None
        assert row["user_event_rate_ratio"] > 0.0

    def test_no_column_asserts_an_attack(self) -> None:
        for column in BASELINE_COLUMNS:
            for banned in ("attack", "takeover", "spraying", "bot_detected"):
                assert banned not in column


# --- engine integration ----------------------------------------------------


class TestEngineIntegration:
    def test_engine_uses_a_supplied_baseline(self) -> None:
        events = _training_events(count=60)
        model = _fit(events)
        config = FeatureConfig()
        frame = FeatureEngine(config, baseline=model).run(events)
        assert any(frame.column("user_in_baseline"))

    def test_engine_output_still_matches_the_catalog(self) -> None:
        events = _training_events(count=40)
        config = FeatureConfig()
        frame = FeatureEngine(config, baseline=_fit(events)).run(events)
        assert tuple(frame.rows[0]) == frame.catalog.column_order()

    def test_baseline_ratios_are_populated_through_the_engine(self) -> None:
        events = _training_events(count=90)
        frame = FeatureEngine(FeatureConfig(), baseline=_fit(events)).run(events)
        ratios = [v for v in frame.column("user_event_rate_ratio") if v is not None]
        assert ratios, "the reference window should yield some populated ratios"

    def test_unknown_entities_stay_null_through_the_engine(self) -> None:
        model = _fit(_training_events(count=40))
        stranger = make_event(t=99999.0, user="ghost", source="phantom")
        frame = FeatureEngine(FeatureConfig(), baseline=model).run([stranger])
        row = frame.rows[0]
        assert row["user_in_baseline"] is False
        assert row["is_new_device_for_user"] is None


# --- persistence -----------------------------------------------------------


class TestPersistence:
    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        model = _fit()
        model.save(tmp_path / "baseline")
        loaded = BehavioralBaselineModel.load(tmp_path / "baseline")
        assert loaded.user_count == model.user_count
        assert loaded.source_count == model.source_count
        assert loaded.artifact.content_fingerprint == model.artifact.content_fingerprint

    def test_loaded_model_transforms_identically(self, tmp_path: Path) -> None:
        events = _training_events(count=60)
        model = _fit(events)
        model.save(tmp_path / "baseline")
        loaded = BehavioralBaselineModel.load(tmp_path / "baseline")
        probe = make_event(t=500.0, user="u1", device="d1", response_time_ms=200)
        assert loaded.transform_one(probe) == model.transform_one(probe)

    def test_overwrite_is_refused_by_default(self, tmp_path: Path) -> None:
        model = _fit()
        model.save(tmp_path / "baseline")
        with pytest.raises(BaselineFitError, match="already exist"):
            model.save(tmp_path / "baseline")

    def test_overwrite_is_allowed_when_requested(self, tmp_path: Path) -> None:
        model = _fit()
        model.save(tmp_path / "baseline")
        model.save(tmp_path / "baseline", overwrite=True)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactNotFoundError):
            BehavioralBaselineModel.load(tmp_path / "absent")

    def test_unfitted_model_cannot_be_saved(self, tmp_path: Path) -> None:
        with pytest.raises(BaselineFitError):
            BehavioralBaselineModel(BaselineConfig()).save(tmp_path / "baseline")


class TestPersistencePrivacy:
    @pytest.fixture()
    def saved(self, tmp_path: Path) -> Path:
        directory = tmp_path / "baseline"
        _fit().save(directory)
        return directory

    def test_metadata_file_contains_no_identifiers(self, saved: Path) -> None:
        payload: dict[str, Any] = json.loads(
            (saved / BASELINE_JSON).read_text(encoding="utf-8")
        )
        text = json.dumps(payload)
        assert "u:" not in text
        assert "s:" not in text
        assert "d:" not in text

    def test_metadata_file_reports_only_aggregates(self, saved: Path) -> None:
        payload = json.loads((saved / BASELINE_JSON).read_text(encoding="utf-8"))
        assert payload["user_count"] > 0
        assert set(payload) == {
            "baseline_schema_version",
            "fitted_interval_start",
            "fitted_interval_end",
            "fitted_source_fingerprint",
            "config_fingerprint",
            "content_fingerprint",
            "user_count",
            "source_count",
            "total_fit_events",
            "truncated_set_count",
            "baseline_in_sample_for_train",
            "created_at",
        }

    @pytest.mark.parametrize(
        "name", ["user_baselines.parquet", "source_baselines.parquet"]
    )
    def test_pseudonym_bearing_tables_are_private(self, saved: Path, name: str) -> None:
        mode = stat.S_IMODE((saved / name).stat().st_mode)
        assert mode == 0o600, "pseudonym-bearing state must not be world-readable"

    def test_metadata_file_stays_readable(self, saved: Path) -> None:
        # Reports and CLI output read this file, so it must not be locked down.
        assert (saved / BASELINE_JSON).read_text(encoding="utf-8")


# --- convenience wrapper ---------------------------------------------------


class TestFitBaselineHelper:
    def test_fits_from_a_full_feature_config(self) -> None:
        events = _training_events(count=40)
        model = fit_baseline(
            events,
            FeatureConfig(),
            permitted_event_ids=frozenset(e.event_id for e in events),
            interval=_INTERVAL,
        )
        assert model.is_fitted

    def test_uses_the_nested_baseline_configuration(self) -> None:
        events = _training_events(count=40)
        config = FeatureConfig(baseline=BaselineConfig(known_set_max_size=2))
        model = fit_baseline(
            events,
            config,
            permitted_event_ids=frozenset(e.event_id for e in events),
            interval=_INTERVAL,
        )
        assert model.config.known_set_max_size == 2


class TestMfaAndInterval:
    def test_interval_length_drives_the_event_rate(self) -> None:
        events = [make_event(t=float(i * 60), user="u1", key=str(i)) for i in range(10)]
        start = datetime(2024, 3, 4, 11, tzinfo=UTC)
        model = BehavioralBaselineModel(BaselineConfig()).fit(
            events,
            permitted_event_ids=frozenset(e.event_id for e in events),
            interval=(start, start + timedelta(hours=10)),
        )
        assert model._user_table_data()["event_rate_per_hour"][0] == pytest.approx(1.0)

    def test_mfa_outcome_does_not_break_fitting(self) -> None:
        events = [
            make_event(
                t=float(i),
                user="u1",
                outcome="failure",
                mfa_outcome=MFAOutcome.FAILED,
                key=str(i),
            )
            for i in range(10)
        ]
        assert _fit(events).is_fitted
