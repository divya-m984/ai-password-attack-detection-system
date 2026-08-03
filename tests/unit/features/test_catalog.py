"""Tests for the versioned feature catalog."""

from __future__ import annotations

import re

import pytest

from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS
from password_attack_detector.exceptions import FeatureComputationError
from password_attack_detector.features.catalog import (
    AMBIGUOUS_FEATURE_NAMES,
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    PROHIBITED_FEATURE_COLUMNS,
    FeatureCatalog,
    FeatureDType,
    FeatureGroup,
    FeatureSpec,
    LeakageClass,
    build_catalog,
    catalog_to_markdown,
)
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    AggregateKind,
    BaselineConfig,
    EntityKind,
    FeatureConfig,
    GeospatialConfig,
)


def _narrow_cardinality() -> FeatureConfig:
    """A config that tracks cardinality only at 24h.

    The baseline's rate reference window must move with it: the source
    fan-out ratio reads a unique count, so the reference window has to be one
    that actually tracks cardinality.
    """
    return FeatureConfig(
        cardinality_windows=("24h",),
        baseline=BaselineConfig(rate_reference_window="24h"),
    )


_WINDOWED_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*__(?:\d+[smhd])$")
_PLAIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@pytest.fixture()
def config() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture()
def catalog(config: FeatureConfig) -> FeatureCatalog:
    return build_catalog(config)


# --- structure -------------------------------------------------------------


class TestCatalogStructure:
    def test_catalog_is_non_empty(self, catalog: FeatureCatalog) -> None:
        assert len(catalog) > 100

    def test_feature_names_are_unique(self, catalog: FeatureCatalog) -> None:
        names = catalog.column_order()
        assert len(names) == len(set(names))

    def test_column_order_is_deterministic(self, config: FeatureConfig) -> None:
        assert (
            build_catalog(config).column_order() == build_catalog(config).column_order()
        )

    def test_key_columns_come_first(self, catalog: FeatureCatalog) -> None:
        assert catalog.column_order()[:3] == (
            "feature_schema_version",
            ANCHOR_EVENT_ID,
            ANCHOR_EVENT_TIME,
        )

    def test_duplicate_names_rejected(self, catalog: FeatureCatalog) -> None:
        first = catalog.specs[0]
        with pytest.raises(FeatureComputationError, match="Duplicate feature name"):
            FeatureCatalog((first, first), config_fingerprint="x")

    def test_get_returns_the_spec(self, catalog: FeatureCatalog) -> None:
        assert catalog.get(ANCHOR_EVENT_ID).name == ANCHOR_EVENT_ID

    def test_get_unknown_name_raises(self, catalog: FeatureCatalog) -> None:
        with pytest.raises(FeatureComputationError, match="Undeclared feature"):
            catalog.get("not_a_feature")

    def test_has_reports_membership(self, catalog: FeatureCatalog) -> None:
        assert catalog.has(ANCHOR_EVENT_ID)
        assert not catalog.has("not_a_feature")

    def test_group_counts_sum_to_catalog_size(self, catalog: FeatureCatalog) -> None:
        assert sum(catalog.group_counts().values()) == len(catalog)

    def test_every_group_is_populated(self, catalog: FeatureCatalog) -> None:
        for group in FeatureGroup:
            assert catalog.specs_for_group(group), f"{group} is empty"

    def test_catalog_is_iterable_in_column_order(self, catalog: FeatureCatalog) -> None:
        assert tuple(s.name for s in catalog) == catalog.column_order()


# --- naming ----------------------------------------------------------------


class TestFeatureNaming:
    def test_names_are_machine_readable(self, catalog: FeatureCatalog) -> None:
        for name in catalog.column_order():
            assert _WINDOWED_NAME_RE.match(name) or _PLAIN_NAME_RE.match(name), name

    def test_windowed_names_carry_their_window_suffix(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.window is None:
                assert "__" not in spec.name or spec.companion_of is not None
            else:
                assert spec.name.endswith(f"__{spec.window}")

    def test_expected_names_are_present(self, catalog: FeatureCatalog) -> None:
        for name in (
            "source_attempt_count__1m",
            "user_failure_count__5m",
            "source_unique_user_count__5m",
            "user_unique_source_count__1h",
            "pair_failure_rate__5m",
            "prior_consecutive_user_failures",
            "hour_sin",
            "is_new_device_for_user",
        ):
            assert catalog.has(name), name

    def test_no_prohibited_ground_truth_columns(self, catalog: FeatureCatalog) -> None:
        assert not set(catalog.column_order()) & PROHIBITED_GT_COLUMNS

    def test_no_prohibited_feature_columns(self, catalog: FeatureCatalog) -> None:
        assert not set(catalog.column_order()) & PROHIBITED_FEATURE_COLUMNS

    def test_prohibited_set_covers_split_and_campaign_fields(self) -> None:
        for name in ("campaign_id", "scenario", "malicious", "split", "attack_class"):
            assert name in PROHIBITED_FEATURE_COLUMNS

    def test_no_ambiguous_or_conclusion_asserting_names(
        self, catalog: FeatureCatalog
    ) -> None:
        for name in catalog.column_order():
            tokens = set(name.split("__")[0].split("_"))
            assert not tokens & AMBIGUOUS_FEATURE_NAMES, name

    def test_no_column_carries_raw_identifiers(self, catalog: FeatureCatalog) -> None:
        # anchor_event_id is the sole identifier and is classified as a key.
        identifier_like = {
            name
            for name in catalog.column_order()
            if name.endswith("_id") or name.endswith("_ids")
        }
        assert identifier_like == {ANCHOR_EVENT_ID}

    def test_no_coordinate_columns_are_emitted(self, catalog: FeatureCatalog) -> None:
        for name in catalog.column_order():
            assert "latitude" not in name
            assert "longitude" not in name


class TestUnsafeNameRejection:
    def test_ambiguous_name_rejected_at_build(self) -> None:
        from password_attack_detector.features import catalog as catalog_module

        spec = FeatureSpec(
            name="user_anomaly_score__5m",
            group=FeatureGroup.USER_HISTORY,
            entity=EntityKind.USER,
            dtype=FeatureDType.FLOAT64,
            nullable=True,
            leakage_class=LeakageClass.PRIOR_ONLY,
            null_semantics="n/a",
            description="n/a",
        )
        with pytest.raises(FeatureComputationError, match="ambiguous"):
            catalog_module._reject_unsafe_names((spec,))

    def test_prohibited_name_rejected_at_build(self) -> None:
        from password_attack_detector.features import catalog as catalog_module

        spec = FeatureSpec(
            name="malicious",
            group=FeatureGroup.USER_HISTORY,
            entity=EntityKind.USER,
            dtype=FeatureDType.BOOL,
            nullable=False,
            leakage_class=LeakageClass.PRIOR_ONLY,
            null_semantics="n/a",
            description="n/a",
        )
        with pytest.raises(FeatureComputationError, match="prohibited"):
            catalog_module._reject_unsafe_names((spec,))


# --- metadata completeness -------------------------------------------------


class TestMetadataCompleteness:
    def test_every_spec_documents_null_semantics(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            assert spec.null_semantics.strip()

    def test_every_spec_has_a_description(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            assert len(spec.description.strip()) > 10

    def test_every_spec_declares_a_leakage_class(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            assert isinstance(spec.leakage_class, LeakageClass)

    def test_every_spec_declares_privacy_class(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            assert spec.privacy_class in {"non_sensitive", "operational_metadata"}

    def test_no_feature_is_deprecated_in_the_first_version(
        self, catalog: FeatureCatalog
    ) -> None:
        assert not [s for s in catalog.specs if s.deprecated]

    def test_rate_features_declare_min_count(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            if spec.aggregate is AggregateKind.RATE:
                assert spec.min_count is not None

    def test_rates_declare_the_unit_interval(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            if spec.aggregate is AggregateKind.RATE:
                assert spec.value_range == (0.0, 1.0)

    def test_cyclical_features_declare_symmetric_range(
        self, catalog: FeatureCatalog
    ) -> None:
        for name in ("hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"):
            assert catalog.get(name).value_range == (-1.0, 1.0)

    def test_std_features_require_two_observations(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.aggregate in {AggregateKind.STD, AggregateKind.COV}:
                assert spec.min_observations == 2

    def test_baseline_features_are_flagged(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs_for_group(FeatureGroup.BASELINE):
            assert spec.requires_baseline
            assert spec.leakage_class is LeakageClass.BASELINE_DERIVED

    def test_only_baseline_features_require_a_baseline(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.requires_baseline:
                assert spec.leakage_class is LeakageClass.BASELINE_DERIVED

    def test_companion_columns_reference_a_declared_feature(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.companion_of is not None:
                assert catalog.has(spec.companion_of)

    def test_companion_columns_are_few(self, catalog: FeatureCatalog) -> None:
        # Nullability is the validity signal; companions exist only where null
        # is genuinely ambiguous.  Guard against __is_valid proliferation.
        companions = [s for s in catalog.specs if s.companion_of is not None]
        assert len(companions) <= 4


# --- leakage classification ------------------------------------------------


class TestLeakageClassification:
    def test_current_context_columns_use_the_current_event(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs_for_group(FeatureGroup.CURRENT_CONTEXT):
            assert spec.leakage_class is LeakageClass.CURRENT_EVENT_CONTEXT
            assert spec.uses_current_event

    def test_every_current_prefixed_column_is_current_context(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.name.startswith("current_"):
                assert spec.leakage_class is LeakageClass.CURRENT_EVENT_CONTEXT

    def test_no_prior_column_uses_the_current_event(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.name.startswith("prior_") or spec.name.startswith("previous_"):
                assert spec.leakage_class is LeakageClass.PRIOR_ONLY
                assert not spec.uses_current_event

    def test_no_windowed_column_uses_the_current_event(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.window is not None:
                assert spec.leakage_class is LeakageClass.PRIOR_ONLY
                assert not spec.uses_current_event

    def test_windowed_descriptions_state_the_half_open_interval(
        self, catalog: FeatureCatalog
    ) -> None:
        for spec in catalog.specs:
            if spec.window is not None:
                assert f"[t - {spec.window}, t)" in spec.description

    def test_prior_only_class_is_the_largest(self, catalog: FeatureCatalog) -> None:
        prior = catalog.specs_for_leakage_class(LeakageClass.PRIOR_ONLY)
        assert len(prior) > len(catalog) // 2

    def test_every_spec_falls_in_exactly_one_leakage_class(
        self, catalog: FeatureCatalog
    ) -> None:
        total = sum(len(catalog.specs_for_leakage_class(cls)) for cls in LeakageClass)
        assert total == len(catalog)


# --- null and zero semantics ----------------------------------------------


class TestNullSemantics:
    def test_count_features_are_never_nullable(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            if spec.is_count_like:
                assert not spec.nullable, spec.name

    def test_rate_mean_std_features_are_nullable(self, catalog: FeatureCatalog) -> None:
        nullable_aggregates = {
            AggregateKind.RATE,
            AggregateKind.MEAN,
            AggregateKind.STD,
            AggregateKind.COV,
        }
        for spec in catalog.specs:
            if spec.aggregate in nullable_aggregates:
                assert spec.nullable, spec.name

    def test_sequence_counters_are_not_nullable(self, catalog: FeatureCatalog) -> None:
        for name in (
            "prior_consecutive_user_failures",
            "prior_consecutive_source_failures",
            "prior_failures_since_user_success",
            "prior_failures_since_pair_success",
        ):
            assert not catalog.get(name).nullable

    def test_seconds_since_features_are_nullable(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs:
            if spec.name.startswith("seconds_since_"):
                assert spec.nullable, spec.name

    def test_previous_outcome_features_are_nullable(
        self, catalog: FeatureCatalog
    ) -> None:
        assert catalog.get("previous_user_outcome").nullable
        assert catalog.get("previous_pair_outcome").nullable

    def test_baseline_coverage_columns_are_not_nullable(
        self, catalog: FeatureCatalog
    ) -> None:
        assert not catalog.get("user_in_baseline").nullable
        assert not catalog.get("source_in_baseline").nullable

    def test_baseline_novelty_columns_are_nullable(
        self, catalog: FeatureCatalog
    ) -> None:
        # A cold user must be null, never reported as having a "new" device.
        for name in (
            "is_new_device_for_user",
            "is_new_source_for_user",
            "is_new_country_for_user",
        ):
            assert catalog.get(name).nullable

    def test_geospatial_status_columns_are_not_nullable(
        self, catalog: FeatureCatalog
    ) -> None:
        assert not catalog.get("user_previous_success_geo__status").nullable
        assert not catalog.get("implied_velocity__status").nullable

    def test_key_columns_are_not_nullable(self, catalog: FeatureCatalog) -> None:
        for spec in catalog.specs_for_group(FeatureGroup.KEY):
            assert not spec.nullable


class TestEmptySnapshot:
    def test_empty_snapshot_covers_windowed_user_features(
        self, catalog: FeatureCatalog
    ) -> None:
        snapshot = catalog.empty_snapshot(EntityKind.USER)
        assert snapshot["user_attempt_count__5m"] == 0
        assert snapshot["user_failure_rate__5m"] is None

    def test_empty_snapshot_counts_are_zero_not_null(
        self, catalog: FeatureCatalog
    ) -> None:
        snapshot = catalog.empty_snapshot(EntityKind.SOURCE)
        for spec in catalog.specs_for_entity(EntityKind.SOURCE):
            if spec.window is None:
                continue
            if spec.is_count_like:
                assert snapshot[spec.name] == 0
            elif spec.nullable:
                assert snapshot[spec.name] is None

    def test_empty_snapshot_excludes_baseline_features(
        self, catalog: FeatureCatalog
    ) -> None:
        snapshot = catalog.empty_snapshot(EntityKind.USER)
        assert "is_new_device_for_user" not in snapshot

    def test_empty_snapshot_covers_only_windowed_features(
        self, catalog: FeatureCatalog
    ) -> None:
        # Sequence and geospatial features come from separate state that always
        # exists, so they are not part of an empty *window* snapshot.
        snapshot = catalog.empty_snapshot(EntityKind.USER)
        assert "prior_consecutive_user_failures" not in snapshot
        assert all(catalog.get(name).window is not None for name in snapshot)

    def test_empty_snapshot_keys_are_declared_features(
        self, catalog: FeatureCatalog
    ) -> None:
        for kind in EntityKind:
            for name in catalog.empty_snapshot(kind):
                assert catalog.has(name)


# --- configuration-driven expansion ----------------------------------------


class TestConfigDrivenExpansion:
    def test_default_windows_expand_counts_five_ways(
        self, catalog: FeatureCatalog
    ) -> None:
        names = [
            n for n in catalog.column_order() if n.startswith("user_attempt_count__")
        ]
        assert names == [
            "user_attempt_count__1m",
            "user_attempt_count__5m",
            "user_attempt_count__15m",
            "user_attempt_count__1h",
            "user_attempt_count__24h",
        ]

    def test_cardinality_trim_reduces_unique_count_columns(self) -> None:
        wide = build_catalog(FeatureConfig())
        narrow = build_catalog(_narrow_cardinality())
        assert len(narrow) < len(wide)
        assert narrow.has("user_unique_source_count__24h")
        assert not narrow.has("user_unique_source_count__5m")

    def test_dispersion_trim_reduces_std_columns(self) -> None:
        narrow = build_catalog(FeatureConfig(dispersion_windows=("24h",)))
        assert narrow.has("user_response_time_std_ms__24h")
        assert not narrow.has("user_response_time_std_ms__5m")

    def test_pair_windows_control_pair_columns(self, catalog: FeatureCatalog) -> None:
        names = [
            n for n in catalog.column_order() if n.startswith("pair_attempt_count")
        ]
        assert names == [
            "pair_attempt_count__5m",
            "pair_attempt_count__1h",
            "pair_attempt_count__24h",
        ]

    def test_device_session_windows_control_session_columns(
        self, catalog: FeatureCatalog
    ) -> None:
        names = [
            n for n in catalog.column_order() if n.startswith("session_event_count")
        ]
        assert names == ["session_event_count__5m", "session_event_count__1h"]

    def test_disabling_geospatial_drops_those_columns(self) -> None:
        catalog = build_catalog(
            FeatureConfig(geospatial=GeospatialConfig(enabled=False))
        )
        assert not catalog.specs_for_group(FeatureGroup.GEOSPATIAL)
        assert not catalog.has("implied_velocity_kmh_from_previous_success")

    def test_disabling_baseline_drops_those_columns(self) -> None:
        catalog = build_catalog(FeatureConfig(baseline=BaselineConfig(enabled=False)))
        assert not catalog.specs_for_group(FeatureGroup.BASELINE)
        assert not catalog.has("user_in_baseline")

    def test_baseline_min_observations_follow_config(self) -> None:
        catalog = build_catalog(
            FeatureConfig(baseline=BaselineConfig(min_events_per_user=11))
        )
        assert catalog.get("is_new_device_for_user").min_observations == 11

    def test_min_count_for_rate_follows_config(self) -> None:
        catalog = build_catalog(FeatureConfig(min_count_for_rate=4))
        assert catalog.get("user_failure_rate__5m").min_count == 4


# --- fingerprinting --------------------------------------------------------


class TestCatalogFingerprint:
    def test_fingerprint_is_hex_sha256(self, catalog: FeatureCatalog) -> None:
        fingerprint = catalog.fingerprint()
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")

    def test_fingerprint_is_stable_across_builds(self, config: FeatureConfig) -> None:
        assert (
            build_catalog(config).fingerprint() == build_catalog(config).fingerprint()
        )

    def test_window_trim_changes_the_fingerprint(self) -> None:
        assert (
            build_catalog(_narrow_cardinality()).fingerprint()
            != build_catalog(FeatureConfig()).fingerprint()
        )

    def test_min_count_change_changes_the_fingerprint(self) -> None:
        assert (
            build_catalog(FeatureConfig(min_count_for_rate=9)).fingerprint()
            != build_catalog(FeatureConfig()).fingerprint()
        )

    def test_fingerprint_ignores_prose_changes(self, catalog: FeatureCatalog) -> None:
        # Correcting a typo in a description must not invalidate every artifact
        # that recorded the catalog fingerprint.
        edited = tuple(
            spec.model_copy(update={"description": spec.description + " (reworded)"})
            for spec in catalog.specs
        )
        rebuilt = FeatureCatalog(edited, config_fingerprint=catalog.config_fingerprint)
        assert rebuilt.fingerprint() == catalog.fingerprint()

    def test_fingerprint_ignores_null_semantics_prose(
        self, catalog: FeatureCatalog
    ) -> None:
        edited = tuple(
            spec.model_copy(update={"null_semantics": "reworded"})
            for spec in catalog.specs
        )
        rebuilt = FeatureCatalog(edited, config_fingerprint=catalog.config_fingerprint)
        assert rebuilt.fingerprint() == catalog.fingerprint()

    def test_fingerprint_reacts_to_dtype_change(self, catalog: FeatureCatalog) -> None:
        edited = list(catalog.specs)
        index = next(
            i for i, s in enumerate(edited) if s.name == "user_attempt_count__5m"
        )
        edited[index] = edited[index].model_copy(update={"dtype": FeatureDType.FLOAT64})
        rebuilt = FeatureCatalog(
            tuple(edited), config_fingerprint=catalog.config_fingerprint
        )
        assert rebuilt.fingerprint() != catalog.fingerprint()

    def test_fingerprint_reacts_to_nullability_change(
        self, catalog: FeatureCatalog
    ) -> None:
        edited = list(catalog.specs)
        index = next(
            i for i, s in enumerate(edited) if s.name == "user_attempt_count__5m"
        )
        edited[index] = edited[index].model_copy(update={"nullable": True})
        rebuilt = FeatureCatalog(
            tuple(edited), config_fingerprint=catalog.config_fingerprint
        )
        assert rebuilt.fingerprint() != catalog.fingerprint()

    def test_fingerprint_is_independent_of_spec_order(
        self, catalog: FeatureCatalog
    ) -> None:
        reversed_catalog = FeatureCatalog(
            tuple(reversed(catalog.specs)),
            config_fingerprint=catalog.config_fingerprint,
        )
        assert reversed_catalog.fingerprint() == catalog.fingerprint()

    def test_catalog_records_the_config_fingerprint(
        self, config: FeatureConfig
    ) -> None:
        assert build_catalog(config).config_fingerprint == config.fingerprint()


# --- generated documentation -----------------------------------------------


class TestGeneratedMarkdown:
    @pytest.fixture()
    def markdown(self, catalog: FeatureCatalog) -> str:
        return catalog_to_markdown(catalog)

    def test_has_a_title_and_version(self, markdown: str) -> None:
        assert markdown.startswith("# Feature Catalog")
        assert FEATURE_SCHEMA_VERSION in markdown

    def test_records_both_fingerprints(
        self, markdown: str, catalog: FeatureCatalog
    ) -> None:
        assert catalog.fingerprint() in markdown
        assert catalog.config_fingerprint in markdown

    def test_documents_every_feature(
        self, markdown: str, catalog: FeatureCatalog
    ) -> None:
        for name in catalog.column_order():
            assert f"`{name}`" in markdown

    def test_ends_with_known_limitations(self, markdown: str) -> None:
        assert "## Known limitations" in markdown

    def test_explains_the_leakage_classes(self, markdown: str) -> None:
        assert "`prior_only`" in markdown
        assert "`current_event_context`" in markdown
        assert "[t - window, t)" in markdown

    def test_makes_no_detection_or_performance_claim(self, markdown: str) -> None:
        lowered = markdown.lower()
        for phrase in ("accuracy", "precision of", "detects attacks", "faster than"):
            assert phrase not in lowered

    def test_contains_no_absolute_paths(self, markdown: str) -> None:
        assert "/home/" not in markdown
        assert "/Users/" not in markdown

    def test_is_deterministic(self, catalog: FeatureCatalog) -> None:
        assert catalog_to_markdown(catalog) == catalog_to_markdown(catalog)
