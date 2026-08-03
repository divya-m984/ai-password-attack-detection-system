"""Tests for the feature-engineering configuration layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from password_attack_detector.exceptions import ConfigurationError
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    FINGERPRINT_EXCLUDED_FIELDS,
    AggregateKind,
    BaselineConfig,
    EntityKind,
    FeatureConfig,
    GeospatialConfig,
    SplitConfig,
    duration_to_microseconds,
    format_duration,
    load_feature_config,
    parse_duration,
)

_REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "features"


# --- duration parsing ------------------------------------------------------


class TestParseDuration:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("30s", timedelta(seconds=30)),
            ("1m", timedelta(minutes=1)),
            ("5m", timedelta(minutes=5)),
            ("15m", timedelta(minutes=15)),
            ("1h", timedelta(hours=1)),
            ("24h", timedelta(hours=24)),
            ("7d", timedelta(days=7)),
        ],
    )
    def test_valid_durations(self, text: str, expected: timedelta) -> None:
        assert parse_duration(text) == expected

    def test_surrounding_whitespace_tolerated(self) -> None:
        assert parse_duration("  15m ") == timedelta(minutes=15)

    @pytest.mark.parametrize(
        "text", ["", "m", "15", "15x", "-5m", "1.5h", "5 m", "1w", "abc"]
    )
    def test_malformed_durations_rejected(self, text: str) -> None:
        with pytest.raises(ConfigurationError):
            parse_duration(text)

    def test_zero_duration_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_duration("0m")

    def test_microsecond_conversion_is_exact(self) -> None:
        assert duration_to_microseconds(parse_duration("24h")) == 86_400_000_000
        assert duration_to_microseconds(parse_duration("1m")) == 60_000_000

    @pytest.mark.parametrize("text", ["30s", "5m", "15m", "1h"])
    def test_format_duration_round_trips(self, text: str) -> None:
        assert format_duration(parse_duration(text)) == text


# --- window validation -----------------------------------------------------


class TestWindowValidation:
    def test_default_windows_accepted(self) -> None:
        config = FeatureConfig()
        assert config.windows == ("1m", "5m", "15m", "1h", "24h")

    def test_empty_windows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(windows=())

    def test_duplicate_windows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(windows=("1m", "5m", "5m", "1h", "24h"))

    def test_descending_windows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(windows=("24h", "1h", "15m", "5m", "1m"))

    def test_unsorted_windows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(windows=("5m", "1m", "15m", "1h", "24h"))

    def test_zero_length_window_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(windows=("0m", "5m"))

    def test_unparseable_window_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(windows=("1m", "5minutes"))

    def test_equal_durations_in_different_units_rejected(self) -> None:
        # "60s" and "1m" are the same duration, so the tuple is not ascending.
        with pytest.raises(ValidationError):
            FeatureConfig(windows=("60s", "1m"))

    def test_subset_window_not_in_windows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(
                windows=("1m", "5m"),
                cardinality_windows=("5m", "1h"),
                dispersion_windows=("5m",),
                device_session_windows=("5m",),
                pair_windows=("5m",),
                split=SplitConfig(strict_isolation=False),
            )


# --- windows_for trimming --------------------------------------------------


class TestWindowsFor:
    @pytest.fixture()
    def config(self) -> FeatureConfig:
        return FeatureConfig()

    def test_user_counts_use_full_ladder(self, config: FeatureConfig) -> None:
        assert config.windows_for(EntityKind.USER, AggregateKind.COUNT) == (
            "1m",
            "5m",
            "15m",
            "1h",
            "24h",
        )

    def test_unique_counts_restricted_to_cardinality_windows(
        self, config: FeatureConfig
    ) -> None:
        assert config.windows_for(EntityKind.USER, AggregateKind.UNIQUE_COUNT) == (
            "5m",
            "1h",
            "24h",
        )

    def test_std_restricted_to_dispersion_windows(self, config: FeatureConfig) -> None:
        assert config.windows_for(EntityKind.SOURCE, AggregateKind.STD) == (
            "5m",
            "15m",
            "1h",
            "24h",
        )

    def test_cov_restricted_to_dispersion_windows(self, config: FeatureConfig) -> None:
        assert config.windows_for(
            EntityKind.SOURCE, AggregateKind.COV
        ) == config.windows_for(EntityKind.SOURCE, AggregateKind.STD)

    def test_pair_restricted_to_pair_windows(self, config: FeatureConfig) -> None:
        assert config.windows_for(EntityKind.USER_SOURCE, AggregateKind.COUNT) == (
            "5m",
            "1h",
            "24h",
        )

    @pytest.mark.parametrize(
        "kind", [EntityKind.USER_DEVICE, EntityKind.DEVICE, EntityKind.SESSION]
    )
    def test_device_session_restricted(
        self, config: FeatureConfig, kind: EntityKind
    ) -> None:
        assert config.windows_for(kind, AggregateKind.COUNT) == ("5m", "1h")

    def test_entity_and_aggregate_restrictions_intersect(
        self, config: FeatureConfig
    ) -> None:
        # pair_windows is (5m, 1h, 24h); dispersion is (5m, 15m, 1h, 24h).
        assert config.windows_for(EntityKind.USER_SOURCE, AggregateKind.STD) == (
            "5m",
            "1h",
            "24h",
        )

    def test_result_always_follows_ascending_window_order(
        self, config: FeatureConfig
    ) -> None:
        for kind in EntityKind:
            for aggregate in AggregateKind:
                result = config.windows_for(kind, aggregate)
                indices = [config.windows.index(w) for w in result]
                assert indices == sorted(indices)

    def test_trimming_is_configuration_not_code(self) -> None:
        config = FeatureConfig(
            cardinality_windows=("24h",),
            baseline=BaselineConfig(rate_reference_window="24h"),
        )
        assert config.windows_for(EntityKind.USER, AggregateKind.UNIQUE_COUNT) == (
            "24h",
        )


# --- split configuration ---------------------------------------------------


class TestSplitConfig:
    def test_default_fractions_sum_to_one(self) -> None:
        config = SplitConfig()
        total = (
            config.train_fraction + config.validation_fraction + config.test_fraction
        )
        assert total == pytest.approx(1.0)

    def test_fractions_not_summing_to_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SplitConfig(train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2)

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_fraction_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError):
            SplitConfig(train_fraction=value)

    def test_boundaries_mode_requires_all_boundaries(self) -> None:
        with pytest.raises(ValidationError):
            SplitConfig(mode="boundaries", train_end=datetime(2024, 1, 1, tzinfo=UTC))

    def test_boundaries_must_be_strictly_increasing(self) -> None:
        with pytest.raises(ValidationError):
            SplitConfig(
                mode="boundaries",
                train_end=datetime(2024, 1, 3, tzinfo=UTC),
                validation_end=datetime(2024, 1, 2, tzinfo=UTC),
                test_end=datetime(2024, 1, 4, tzinfo=UTC),
            )

    def test_equal_boundaries_rejected(self) -> None:
        moment = datetime(2024, 1, 2, tzinfo=UTC)
        with pytest.raises(ValidationError):
            SplitConfig(
                mode="boundaries",
                train_end=moment,
                validation_end=moment,
                test_end=datetime(2024, 1, 4, tzinfo=UTC),
            )

    def test_valid_boundaries_accepted_and_normalized_to_utc(self) -> None:
        config = SplitConfig(
            mode="boundaries",
            train_end=datetime(2024, 1, 1, tzinfo=UTC),
            validation_end=datetime(2024, 1, 2, tzinfo=UTC),
            test_end=datetime(2024, 1, 3, tzinfo=UTC),
        )
        assert config.train_end is not None
        assert config.train_end.tzinfo == UTC

    def test_duration_strings_accepted_for_purge_and_embargo(self) -> None:
        config = SplitConfig(purge="24h", embargo="30s")  # type: ignore[arg-type]
        assert config.purge == timedelta(hours=24)
        assert config.embargo == timedelta(seconds=30)

    def test_negative_purge_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SplitConfig(purge=timedelta(seconds=-1))

    def test_malformed_purge_string_rejected(self) -> None:
        with pytest.raises((ValidationError, ConfigurationError)):
            SplitConfig(purge="soon")  # type: ignore[arg-type]

    def test_default_boundary_policy_is_exclude(self) -> None:
        assert SplitConfig().boundary_campaign_policy == "exclude"

    def test_default_normal_grouping_is_singleton(self) -> None:
        assert SplitConfig().normal_grouping == "singleton"


class TestBaselineReferenceWindow:
    def test_default_reference_window_is_configured(self) -> None:
        config = FeatureConfig()
        assert config.baseline.rate_reference_window in config.windows
        assert config.baseline.rate_reference_window in config.cardinality_windows

    def test_reference_window_outside_windows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(baseline=BaselineConfig(rate_reference_window="7d"))

    def test_reference_window_without_cardinality_rejected(self) -> None:
        # The source fan-out ratio reads a unique count from this window.
        with pytest.raises(ValidationError):
            FeatureConfig(baseline=BaselineConfig(rate_reference_window="1m"))

    def test_reference_window_unchecked_when_baseline_disabled(self) -> None:
        config = FeatureConfig(
            baseline=BaselineConfig(enabled=False, rate_reference_window="1m")
        )
        assert config.baseline.enabled is False


class TestPurgeAgainstMaxWindow:
    def test_purge_shorter_than_max_window_rejected_under_strict_isolation(
        self,
    ) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(split=SplitConfig(purge=timedelta(hours=1)))

    def test_purge_equal_to_max_window_accepted(self) -> None:
        config = FeatureConfig(split=SplitConfig(purge=timedelta(hours=24)))
        assert config.split.purge == config.max_window

    def test_short_purge_allowed_when_strict_isolation_disabled(self) -> None:
        config = FeatureConfig(
            split=SplitConfig(purge=timedelta(minutes=1), strict_isolation=False)
        )
        assert config.split.purge == timedelta(minutes=1)


# --- schema-version compatibility ------------------------------------------


class TestSchemaVersions:
    def test_matching_source_schema_version_accepted(self) -> None:
        assert FeatureConfig().source_schema_version == "1.0.0"

    def test_incompatible_source_schema_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(source_schema_version="2.0.0")

    def test_feature_schema_version_is_pinned(self) -> None:
        assert FEATURE_SCHEMA_VERSION == "1.0.0"
        assert FeatureConfig().feature_schema_version == FEATURE_SCHEMA_VERSION


# --- paths -----------------------------------------------------------------


class TestOutputPaths:
    def test_relative_output_dir_accepted(self) -> None:
        config = FeatureConfig(output_dir=Path("data/processed"))
        assert config.output_dir == Path("data/processed")

    def test_parent_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(output_dir=Path("../escape"))

    def test_nested_parent_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(output_dir=Path("data/../../escape"))


# --- fingerprinting --------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_is_hex_sha256(self) -> None:
        fingerprint = FeatureConfig().fingerprint()
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")

    def test_identical_config_produces_identical_fingerprint(self) -> None:
        assert FeatureConfig().fingerprint() == FeatureConfig().fingerprint()

    def test_fingerprint_is_independent_of_output_dir(self) -> None:
        a = FeatureConfig(output_dir=Path("/tmp/run-a"))
        b = FeatureConfig(output_dir=Path("data/processed"))
        c = FeatureConfig(output_dir=None)
        assert a.fingerprint() == b.fingerprint() == c.fingerprint()

    def test_fingerprint_is_independent_of_overwrite_flag(self) -> None:
        assert (
            FeatureConfig(overwrite=True).fingerprint()
            == FeatureConfig(overwrite=False).fingerprint()
        )

    def test_window_change_changes_fingerprint(self) -> None:
        baseline = FeatureConfig().fingerprint()
        changed = FeatureConfig(
            windows=("1m", "5m", "15m", "1h", "24h", "48h"),
            split=SplitConfig(purge=timedelta(hours=48)),
        ).fingerprint()
        assert baseline != changed

    def test_cardinality_trim_changes_fingerprint(self) -> None:
        assert (
            FeatureConfig(
                cardinality_windows=("24h",),
                baseline=BaselineConfig(rate_reference_window="24h"),
            ).fingerprint()
            != FeatureConfig().fingerprint()
        )

    def test_min_count_for_rate_changes_fingerprint(self) -> None:
        assert (
            FeatureConfig(min_count_for_rate=3).fingerprint()
            != FeatureConfig().fingerprint()
        )

    def test_nested_baseline_change_changes_fingerprint(self) -> None:
        changed = FeatureConfig(baseline=BaselineConfig(known_set_max_size=8))
        assert changed.fingerprint() != FeatureConfig().fingerprint()

    def test_nested_split_change_changes_fingerprint(self) -> None:
        changed = FeatureConfig(
            split=SplitConfig(boundary_campaign_policy="assign_by_first_event")
        )
        assert changed.fingerprint() != FeatureConfig().fingerprint()

    def test_nested_geospatial_change_changes_fingerprint(self) -> None:
        changed = FeatureConfig(
            geospatial=GeospatialConfig(max_plausible_velocity_kmh=900.0)
        )
        assert changed.fingerprint() != FeatureConfig().fingerprint()

    def test_component_fingerprints_are_hex_sha256(self) -> None:
        for fingerprint in (
            BaselineConfig().fingerprint(),
            SplitConfig().fingerprint(),
        ):
            assert len(fingerprint) == 64
            assert set(fingerprint) <= set("0123456789abcdef")

    def test_every_model_field_has_a_fingerprint_decision(self) -> None:
        # Drift guard: a field added to FeatureConfig must either appear in the
        # fingerprint payload or be explicitly listed as excluded.  Neither is
        # allowed to happen by accident.
        model_fields = set(type(FeatureConfig()).model_fields)
        payload_fields = set(FeatureConfig().fingerprint_data())
        assert payload_fields | FINGERPRINT_EXCLUDED_FIELDS == model_fields

    def test_excluded_fields_are_absent_from_the_payload(self) -> None:
        payload = FeatureConfig(
            output_dir=Path("/tmp/anywhere"), overwrite=True
        ).fingerprint_data()
        assert FINGERPRINT_EXCLUDED_FIELDS.isdisjoint(payload)

    def test_excluded_fields_describe_only_output_location(self) -> None:
        assert {"output_dir", "overwrite"} == FINGERPRINT_EXCLUDED_FIELDS


# --- immutability ----------------------------------------------------------


class TestImmutability:
    def test_feature_config_is_frozen(self) -> None:
        config = FeatureConfig()
        with pytest.raises(ValidationError):
            config.min_count_for_rate = 5

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(unknown_setting=1)  # type: ignore[call-arg]

    def test_split_config_is_frozen(self) -> None:
        config = SplitConfig()
        with pytest.raises(ValidationError):
            config.purge = timedelta(hours=1)


# --- max window helpers ----------------------------------------------------


class TestMaxWindow:
    def test_max_window_is_longest_configured_window(self) -> None:
        assert FeatureConfig().max_window == timedelta(hours=24)

    def test_max_window_microseconds_is_exact(self) -> None:
        assert FeatureConfig().max_window_microseconds == 86_400_000_000

    def test_window_microseconds_pairs_are_ascending(self) -> None:
        pairs = FeatureConfig().window_microseconds()
        assert [label for label, _ in pairs] == ["1m", "5m", "15m", "1h", "24h"]
        values = [us for _, us in pairs]
        assert values == sorted(values)


# --- YAML loading ----------------------------------------------------------


class TestLoadFeatureConfig:
    def test_missing_file_raises_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_feature_config(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("windows: [1m,\n  - broken", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_feature_config(path)

    def test_non_mapping_yaml_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_feature_config(path)

    def test_empty_yaml_yields_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_feature_config(path).fingerprint() == FeatureConfig().fingerprint()

    def test_invalid_values_raise_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad-windows.yaml"
        path.write_text('windows: ["5m", "1m"]\n', encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_feature_config(path)

    def test_unknown_key_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.yaml"
        path.write_text("mystery_option: 1\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_feature_config(path)


class TestShippedConfigs:
    @pytest.mark.parametrize(
        "name", ["feature-testing.yaml", "feature-development.yaml"]
    )
    def test_shipped_config_loads(self, name: str) -> None:
        config = load_feature_config(_REPO_CONFIGS / name)
        assert config.windows == ("1m", "5m", "15m", "1h", "24h")

    def test_development_config_enforces_strict_isolation(self) -> None:
        config = load_feature_config(_REPO_CONFIGS / "feature-development.yaml")
        assert config.split.strict_isolation is True
        assert config.split.purge >= config.max_window

    def test_testing_config_relaxes_isolation_for_small_datasets(self) -> None:
        config = load_feature_config(_REPO_CONFIGS / "feature-testing.yaml")
        assert config.split.strict_isolation is False

    def test_shipped_configs_have_distinct_fingerprints(self) -> None:
        testing = load_feature_config(_REPO_CONFIGS / "feature-testing.yaml")
        development = load_feature_config(_REPO_CONFIGS / "feature-development.yaml")
        assert testing.fingerprint() != development.fingerprint()

    def test_shipped_config_fingerprint_is_path_independent(
        self, tmp_path: Path
    ) -> None:
        source = _REPO_CONFIGS / "feature-development.yaml"
        copied = tmp_path / "elsewhere.yaml"
        copied.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        assert (
            load_feature_config(source).fingerprint()
            == load_feature_config(copied).fingerprint()
        )
