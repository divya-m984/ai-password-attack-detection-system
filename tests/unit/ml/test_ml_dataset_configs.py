"""Tests for the tracked 30-day ML development configurations.

These validate ``configs/data/synthetic-ml-development.yaml`` and
``configs/features/feature-ml-development.yaml`` **without generating a single
event**. Building the dataset they describe is a slow, opt-in workflow that
never runs in CI; what CI must guarantee is that the two files are mutually
compatible and satisfy the contracts they declare against.

The central property is arithmetic: strict split isolation costs
``2 * purge / duration`` of the dataset regardless of the fractions chosen, and
every evaluation window must be comfortably longer than the purge interval. At
168 hours a 24h purge leaves about 18 usable hours per window; at 720 hours it
leaves about 120. That difference is the entire reason these files exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.data.synthetic.config import SyntheticConfig
from password_attack_detector.features.config import FeatureConfig, load_feature_config

#: The intended span, in hours.
EXPECTED_DURATION_HOURS = 720

#: Every scenario the generator can produce, keyed by its campaign-parameter
#: attribute.  Derived from the enum so a new scenario fails here rather than
#: being silently absent from the ML dataset.
ATTACK_SCENARIOS = tuple(
    scenario.value for scenario in ScenarioType if scenario is not ScenarioType.NORMAL
)


def _repo_root() -> Path:
    """Return the repository root, located from this test file."""
    return Path(__file__).resolve().parents[3]


def _synthetic_path() -> Path:
    """Return the path to the 30-day synthetic configuration."""
    return _repo_root() / "configs" / "data" / "synthetic-ml-development.yaml"


def _feature_path() -> Path:
    """Return the path to the matching feature configuration."""
    return _repo_root() / "configs" / "features" / "feature-ml-development.yaml"


@pytest.fixture(scope="module")
def synthetic() -> SyntheticConfig:
    """Return the parsed 30-day synthetic configuration."""
    data = yaml.safe_load(_synthetic_path().read_text(encoding="utf-8"))
    return SyntheticConfig(**data)


@pytest.fixture(scope="module")
def features() -> FeatureConfig:
    """Return the parsed matching feature configuration."""
    return load_feature_config(_feature_path())


# ---------------------------------------------------------------------------
# The files exist and validate
# ---------------------------------------------------------------------------


def test_both_configurations_are_tracked() -> None:
    """The configuration is committed even though its output never is."""
    assert _synthetic_path().exists()
    assert _feature_path().exists()


def test_the_synthetic_configuration_validates(synthetic: SyntheticConfig) -> None:
    """It must satisfy the unchanged Phase 2 contract."""
    assert len(synthetic.fingerprint()) == 64


def test_the_feature_configuration_validates(features: FeatureConfig) -> None:
    """It must satisfy the unchanged Phase 3 contract."""
    assert len(features.fingerprint()) == 64
    assert len(features.split.fingerprint()) == 64


# ---------------------------------------------------------------------------
# Span, seed, and reuse of the existing schemas
# ---------------------------------------------------------------------------


def test_the_dataset_spans_thirty_days(synthetic: SyntheticConfig) -> None:
    """720 hours is the point of this configuration."""
    assert synthetic.duration_hours == EXPECTED_DURATION_HOURS


def test_the_seed_is_deterministic_and_distinct(synthetic: SyntheticConfig) -> None:
    """A fixed seed, and not the one the 7-day configuration uses.

    Sharing a seed would draw the same entity population, so a result that
    only held for that one population would be invisible.
    """
    assert isinstance(synthetic.seed, int)
    seven_day = yaml.safe_load(
        (_repo_root() / "configs" / "data" / "synthetic-development.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert synthetic.seed != seven_day["seed"]


def test_the_generation_start_time_is_fixed_and_utc(
    synthetic: SyntheticConfig,
) -> None:
    """The generator never reads the system clock."""
    assert synthetic.start_time.utcoffset() is not None
    assert synthetic.start_time.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


def test_the_existing_schema_versions_are_reused(
    synthetic: SyntheticConfig, features: FeatureConfig
) -> None:
    """New configuration, unchanged contracts.

    Phase 2 and Phase 3 runtime behaviour must not be altered to accommodate a
    longer dataset; only the values supplied to them change.
    """
    from password_attack_detector.data.schemas import SCHEMA_VERSION
    from password_attack_detector.features.config import FEATURE_SCHEMA_VERSION

    assert synthetic.schema_version == SCHEMA_VERSION
    assert features.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert features.source_schema_version == SCHEMA_VERSION


def test_every_scenario_is_enabled(synthetic: SyntheticConfig) -> None:
    """A model comparison needs every category the label schema can produce."""
    for scenario in ScenarioType:
        assert getattr(synthetic.enabled_scenarios, scenario.value) is True, scenario


def test_the_configuration_covers_every_declared_scenario(
    synthetic: SyntheticConfig,
) -> None:
    """A scenario added to the enum must be given campaign parameters here."""
    for scenario in ATTACK_SCENARIOS:
        assert hasattr(synthetic.campaign_parameters, scenario), scenario


# ---------------------------------------------------------------------------
# Split arithmetic -- the reason for the longer span
# ---------------------------------------------------------------------------


def test_the_split_is_sixty_twenty_twenty(features: FeatureConfig) -> None:
    """The approved fractions for the longer dataset."""
    assert features.split.train_fraction == pytest.approx(0.60)
    assert features.split.validation_fraction == pytest.approx(0.20)
    assert features.split.test_fraction == pytest.approx(0.20)


def test_the_fractions_sum_to_one(features: FeatureConfig) -> None:
    """A shortfall would silently discard the tail of the dataset."""
    total = (
        features.split.train_fraction
        + features.split.validation_fraction
        + features.split.test_fraction
    )
    assert total == pytest.approx(1.0)


def test_strict_isolation_is_enabled_with_a_twenty_four_hour_purge(
    features: FeatureConfig,
) -> None:
    """The approved purge, and the mode that makes it mandatory."""
    assert features.split.strict_isolation is True
    assert features.split.purge.total_seconds() == 24 * 3600


def test_the_purge_covers_the_longest_feature_window(
    features: FeatureConfig,
) -> None:
    """A purge shorter than the longest lookback would isolate nothing.

    Phase 3 rejects that combination outright under strict isolation; this
    asserts the shipped values actually satisfy it rather than relying on the
    build to notice.
    """
    assert features.split.purge >= features.max_window


def test_each_evaluation_window_comfortably_exceeds_the_purge(
    synthetic: SyntheticConfig, features: FeatureConfig
) -> None:
    """The property the 7-day configuration cannot satisfy.

    At 168 hours under 50/25/25 a 24h purge leaves roughly 18 usable hours per
    evaluation window. Here each window spans 144 hours and about 120 survive,
    which is what makes a per-scenario metric worth computing.
    """
    duration = synthetic.duration_hours
    purge_hours = features.split.purge.total_seconds() / 3600

    for fraction in (
        features.split.validation_fraction,
        features.split.test_fraction,
    ):
        window_hours = duration * fraction
        usable = window_hours - purge_hours
        assert window_hours == pytest.approx(144.0)
        assert usable >= 4 * purge_hours, usable


def test_the_exclusion_tolerance_exceeds_the_cost_of_the_purge(
    synthetic: SyntheticConfig, features: FeatureConfig
) -> None:
    """``max_excluded_fraction`` must exceed ``2 * purge / duration``.

    Below that floor every build raises regardless of how well the split is
    otherwise formed, because the purge alone spends more than the tolerance
    allows.
    """
    purge_hours = features.split.purge.total_seconds() / 3600
    purge_cost = 2 * purge_hours / synthetic.duration_hours
    assert purge_cost == pytest.approx(48 / 720)
    assert features.split.max_excluded_fraction > purge_cost


def test_the_exclusion_tolerance_leaves_headroom_for_boundary_campaigns(
    synthetic: SyntheticConfig, features: FeatureConfig
) -> None:
    """Whole campaigns crossing a boundary are excluded too, and cost more."""
    purge_hours = features.split.purge.total_seconds() / 3600
    purge_cost = 2 * purge_hours / synthetic.duration_hours
    headroom = features.split.max_excluded_fraction - purge_cost
    assert headroom > 0.10, headroom


def test_the_tolerance_is_tighter_than_the_seven_day_configuration(
    features: FeatureConfig,
) -> None:
    """The purge costs far less here, so the tolerance should not be as loose.

    A tolerance carried over unchanged would stop being a guard.
    """
    seven_day = load_feature_config(
        _repo_root() / "configs" / "features" / "feature-development.yaml"
    )
    assert features.split.max_excluded_fraction < seven_day.split.max_excluded_fraction


def test_the_boundary_policy_is_the_zero_leakage_one(
    features: FeatureConfig,
) -> None:
    """Excluding a straddling campaign is the only trivially auditable policy."""
    assert features.split.boundary_campaign_policy == "exclude"
    assert features.split.normal_grouping == "singleton"


def test_no_embargo_is_configured(features: FeatureConfig) -> None:
    """Features look backwards, so contamination flows forward across a boundary.

    The purge -- which excludes the later side -- is the one that closes it.
    """
    assert features.split.embargo.total_seconds() == 0


# ---------------------------------------------------------------------------
# Support intent
# ---------------------------------------------------------------------------


def test_every_attack_scenario_declares_multiple_campaigns(
    synthetic: SyntheticConfig,
) -> None:
    """One campaign per scenario cannot survive the boundary-exclusion policy.

    A scenario with a single campaign loses everything if that campaign
    happens to straddle a split boundary.
    """
    for scenario in ATTACK_SCENARIOS:
        parameters = getattr(synthetic.campaign_parameters, scenario)
        assert parameters.num_campaigns >= 5, scenario


def test_the_campaign_counts_exceed_the_seven_day_configuration(
    synthetic: SyntheticConfig,
) -> None:
    """A longer span with the same campaign count would not add support."""
    seven_day = SyntheticConfig(
        **yaml.safe_load(
            (
                _repo_root() / "configs" / "data" / "synthetic-development.yaml"
            ).read_text(encoding="utf-8")
        )
    )
    for scenario in ATTACK_SCENARIOS:
        longer = getattr(synthetic.campaign_parameters, scenario).num_campaigns
        shorter = getattr(seven_day.campaign_parameters, scenario).num_campaigns
        assert longer > shorter, scenario


def test_the_entity_population_is_wider_than_the_seven_day_configuration(
    synthetic: SyntheticConfig,
) -> None:
    """A 30-day span should not simply replay the same 200 users four times."""
    seven_day = yaml.safe_load(
        (_repo_root() / "configs" / "data" / "synthetic-development.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert synthetic.num_users > seven_day["num_users"]
    assert synthetic.num_sources > seven_day["num_sources"]
    assert synthetic.num_devices > seven_day["num_devices"]


def test_the_novel_holdout_has_more_than_one_campaign(
    synthetic: SyntheticConfig,
) -> None:
    """The generalisation probe needs more than a single campaign to probe."""
    assert synthetic.campaign_parameters.novel_anomaly_holdout.num_campaigns >= 2


# ---------------------------------------------------------------------------
# Safety and hygiene
# ---------------------------------------------------------------------------


def test_neither_configuration_references_real_data() -> None:
    """Synthetic generation only: no ingestion source, no real traffic."""
    for path in (_synthetic_path(), _feature_path()):
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in ("http://", "https://", "ldap", "ad_server", "s3://"):
            assert forbidden not in text, (path.name, forbidden)


def test_neither_configuration_carries_a_credential_shaped_key() -> None:
    """A generator configuration has no use for a secret."""
    for path in (_synthetic_path(), _feature_path()):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        _assert_no_secret_keys(data, path.name)


#: Keys drawn from project vocabulary rather than from a free-form name.
#: ``password_spraying`` and ``credential_stuffing`` are attack scenarios, not
#: credentials; the detection layer's own scanner exempts its registries the
#: same way, and building the exemption from the enum keeps it correct as the
#: enum grows.
_CONTROLLED_VOCABULARY = frozenset(scenario.value for scenario in ScenarioType)


def _assert_no_secret_keys(node: object, where: str) -> None:
    """Walk a parsed configuration and assert no key looks like a credential."""
    forbidden = {"password", "secret", "token", "credential", "apikey", "salt"}
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key) not in _CONTROLLED_VOCABULARY:
                tokens = set(str(key).lower().replace("-", "_").split("_"))
                assert not tokens & forbidden, f"{where}: {key}"
            _assert_no_secret_keys(value, where)
    elif isinstance(node, list):
        for item in node:
            _assert_no_secret_keys(item, where)


def test_no_dataset_generated_from_these_configurations_is_committed() -> None:
    """The configuration is tracked; its output never is."""
    root = _repo_root()
    for directory in ("data/raw", "data/interim", "data/processed", "models"):
        present = {
            path.name
            for path in (root / directory).iterdir()
            if path.name != ".gitkeep"
        }
        assert not present, f"{directory} contains {sorted(present)}"


def test_generating_the_dataset_is_not_triggered_by_these_tests(
    synthetic: SyntheticConfig,
) -> None:
    """Parsing a configuration must never generate an event.

    Building this dataset is a slow, opt-in workflow. These tests read the
    declaration and compute arithmetic over it; nothing here calls the
    generator, and the assertion below is what keeps that true.
    """
    assert synthetic.duration_hours * synthetic.events_per_hour > 0
    for directory in ("data/raw", "data/interim"):
        contents = {
            path.name
            for path in (_repo_root() / directory).iterdir()
            if path.name != ".gitkeep"
        }
        assert not contents
