"""Unit tests for password_attack_detector.data.synthetic.config."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from password_attack_detector.data.schemas import SCHEMA_VERSION
from password_attack_detector.data.synthetic.config import (
    BotActivityParams,
    BruteForceParams,
    CampaignParameters,
    CredentialStuffingParams,
    DistributedBruteForceParams,
    EnabledScenarios,
    ImpossibleTravelParams,
    NovelAnomalyParams,
    PasswordSprayingParams,
    SyntheticConfig,
)

_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


def _make_config(**kwargs: object) -> SyntheticConfig:
    defaults: dict[str, object] = {
        "seed": 42,
        "start_time": _START,
        "duration_hours": 1,
        "num_users": 3,
        "num_sources": 2,
        "num_devices": 4,
        "num_applications": 1,
        "events_per_hour": 5,
    }
    defaults.update(kwargs)
    return SyntheticConfig(**defaults)  # type: ignore[arg-type]


class TestSyntheticConfigConstruction:
    def test_minimal_config_constructs(self) -> None:
        cfg = _make_config()
        assert cfg.seed == 42
        assert cfg.generator_version == "1.0.0"
        assert cfg.schema_version == SCHEMA_VERSION

    def test_start_time_normalized_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        t = datetime(2024, 6, 1, 12, 0, 0, tzinfo=eastern)
        cfg = _make_config(start_time=t)
        assert cfg.start_time.tzinfo == UTC

    def test_start_time_requires_timezone(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(start_time=datetime(2024, 1, 1))

    def test_num_users_must_be_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(num_users=0)

    def test_events_per_hour_must_be_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(events_per_hour=0)

    def test_duration_hours_must_be_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(duration_hours=0)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(nonexistent_field=True)

    def test_config_is_frozen(self) -> None:
        cfg = _make_config()
        with pytest.raises(ValidationError):
            cfg.seed = 99

    def test_enabled_scenarios_defaults_all_true(self) -> None:
        cfg = _make_config()
        es = cfg.enabled_scenarios
        assert es.normal is True
        assert es.brute_force is True
        assert es.novel_anomaly_holdout is True

    def test_custom_enabled_scenarios(self) -> None:
        es = EnabledScenarios(
            normal=True,
            brute_force=False,
            password_spraying=False,
            credential_stuffing=False,
            distributed_brute_force=False,
            account_takeover_indicator=False,
            impossible_travel=False,
            bot_activity=False,
            novel_anomaly_holdout=False,
        )
        cfg = _make_config(enabled_scenarios=es)
        assert cfg.enabled_scenarios.brute_force is False
        assert cfg.enabled_scenarios.normal is True


class TestSyntheticConfigFingerprint:
    def test_fingerprint_is_64_hex_chars(self) -> None:
        fp = _make_config().fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_same_config_same_fingerprint(self) -> None:
        cfg1 = _make_config(seed=7)
        cfg2 = _make_config(seed=7)
        assert cfg1.fingerprint() == cfg2.fingerprint()

    def test_different_seed_different_fingerprint(self) -> None:
        fp1 = _make_config(seed=1).fingerprint()
        fp2 = _make_config(seed=2).fingerprint()
        assert fp1 != fp2

    def test_different_duration_different_fingerprint(self) -> None:
        fp1 = _make_config(duration_hours=1).fingerprint()
        fp2 = _make_config(duration_hours=2).fingerprint()
        assert fp1 != fp2

    def test_different_user_count_different_fingerprint(self) -> None:
        fp1 = _make_config(num_users=5).fingerprint()
        fp2 = _make_config(num_users=10).fingerprint()
        assert fp1 != fp2

    def test_different_start_time_different_fingerprint(self) -> None:
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)
        fp1 = _make_config(start_time=t1).fingerprint()
        fp2 = _make_config(start_time=t2).fingerprint()
        assert fp1 != fp2

    def test_fingerprint_stable_across_calls(self) -> None:
        cfg = _make_config()
        assert cfg.fingerprint() == cfg.fingerprint()

    def test_fingerprint_includes_schema_version(self) -> None:
        cfg = _make_config()
        assert cfg.fingerprint() != ""


class TestCampaignSubParams:
    def test_brute_force_params_defaults(self) -> None:
        p = BruteForceParams()
        assert p.attempts_per_campaign >= 1
        assert p.num_campaigns >= 1

    def test_brute_force_params_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            BruteForceParams(attempts_per_campaign=0)

    def test_password_spraying_params(self) -> None:
        p = PasswordSprayingParams(passwords_per_round=3, num_campaigns=2)
        assert p.passwords_per_round == 3

    def test_distributed_brute_force_params(self) -> None:
        p = DistributedBruteForceParams(attempts_per_source=3, num_sources=3)
        assert p.num_sources == 3

    def test_distributed_brute_force_num_sources_ge_2(self) -> None:
        with pytest.raises(ValidationError):
            DistributedBruteForceParams(attempts_per_source=5, num_sources=1)

    def test_bot_activity_params(self) -> None:
        p = BotActivityParams(events_per_campaign=10, num_campaigns=1)
        assert p.events_per_campaign == 10

    def test_campaign_parameters_contains_all_scenarios(self) -> None:
        cp = CampaignParameters()
        assert isinstance(cp.brute_force, BruteForceParams)
        assert isinstance(cp.password_spraying, PasswordSprayingParams)
        assert isinstance(cp.credential_stuffing, CredentialStuffingParams)
        assert isinstance(cp.distributed_brute_force, DistributedBruteForceParams)
        assert isinstance(cp.impossible_travel, ImpossibleTravelParams)
        assert isinstance(cp.bot_activity, BotActivityParams)
        assert isinstance(cp.novel_anomaly_holdout, NovelAnomalyParams)

    def test_model_dump_includes_all_scenario_keys(self) -> None:
        cp = CampaignParameters()
        dumped = cp.model_dump()
        assert "brute_force" in dumped
        assert "novel_anomaly_holdout" in dumped
