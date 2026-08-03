"""Unit tests for password_attack_detector.data.synthetic.generator."""

from __future__ import annotations

from datetime import UTC, datetime

from password_attack_detector.data.enums import (
    AuthOutcome,
    ScenarioType,
)
from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS
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
from password_attack_detector.data.synthetic.generator import (
    GenerationResult,
    generate_dataset,
)

_START = datetime(2024, 1, 1, tzinfo=UTC)


def _tiny_config(*, seed: int = 42, all_scenarios: bool = False) -> SyntheticConfig:
    cp = CampaignParameters(
        brute_force=BruteForceParams(attempts_per_campaign=3, num_campaigns=1),
        password_spraying=PasswordSprayingParams(
            passwords_per_round=2, num_campaigns=1
        ),
        credential_stuffing=CredentialStuffingParams(
            credentials_per_batch=3, num_campaigns=1
        ),
        distributed_brute_force=DistributedBruteForceParams(
            attempts_per_source=2, num_sources=2, num_campaigns=1
        ),
        impossible_travel=ImpossibleTravelParams(num_campaigns=1),
        bot_activity=BotActivityParams(events_per_campaign=3, num_campaigns=1),
        novel_anomaly_holdout=NovelAnomalyParams(num_campaigns=1),
    )
    es = EnabledScenarios(
        normal=True,
        brute_force=all_scenarios,
        password_spraying=all_scenarios,
        credential_stuffing=all_scenarios,
        distributed_brute_force=all_scenarios,
        account_takeover_indicator=all_scenarios,
        impossible_travel=all_scenarios,
        bot_activity=all_scenarios,
        novel_anomaly_holdout=all_scenarios,
    )
    return SyntheticConfig(
        seed=seed,
        start_time=_START,
        duration_hours=1,
        num_users=4,
        num_sources=3,
        num_devices=6,
        num_applications=2,
        events_per_hour=5,
        campaign_parameters=cp,
        enabled_scenarios=es,
    )


def _full_config(*, seed: int = 42) -> SyntheticConfig:
    return _tiny_config(seed=seed, all_scenarios=True)


class TestGenerationResult:
    def test_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(GenerationResult)

    def test_is_frozen(self) -> None:
        result = generate_dataset(_tiny_config())
        try:
            result.events = ()  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception as exc:
            msg = str(exc).lower()
            assert "frozen" in msg or "can't" in msg or "cannot" in msg

    def test_has_expected_fields(self) -> None:
        result = generate_dataset(_tiny_config())
        assert hasattr(result, "events")
        assert hasattr(result, "labels")
        assert hasattr(result, "config")
        assert hasattr(result, "config_fingerprint")

    def test_config_fingerprint_matches_config(self) -> None:
        cfg = _tiny_config()
        result = generate_dataset(cfg)
        assert result.config_fingerprint == cfg.fingerprint()


class TestGenerateDataset:
    def test_returns_generation_result(self) -> None:
        result = generate_dataset(_tiny_config())
        assert isinstance(result, GenerationResult)

    def test_events_non_empty(self) -> None:
        result = generate_dataset(_tiny_config())
        assert len(result.events) > 0

    def test_labels_non_empty(self) -> None:
        result = generate_dataset(_tiny_config())
        assert len(result.labels) > 0

    def test_one_to_one_event_label_relationship(self) -> None:
        result = generate_dataset(_full_config())
        assert len(result.events) == len(result.labels)

    def test_event_label_ids_match(self) -> None:
        result = generate_dataset(_full_config())
        for event, label in zip(result.events, result.labels, strict=True):
            assert event.event_id == label.event_id

    def test_event_ids_globally_unique(self) -> None:
        result = generate_dataset(_full_config())
        ids = [str(e.event_id) for e in result.events]
        assert len(ids) == len(set(ids)), "Duplicate event IDs found"

    def test_no_gt_columns_in_any_event(self) -> None:
        result = generate_dataset(_full_config())
        event_fields = set(type(result.events[0]).model_fields.keys())
        for col in PROHIBITED_GT_COLUMNS:
            assert col not in event_fields, f"GT column {col!r} in AuthEvent"

    def test_all_failure_events_have_failure_reason(self) -> None:
        result = generate_dataset(_full_config())
        for event in result.events:
            if event.authentication_outcome in (
                AuthOutcome.FAILURE,
                AuthOutcome.BLOCKED,
            ):
                assert event.failure_reason is not None, (
                    f"Event {event.event_id} has outcome {event.authentication_outcome} "
                    "but failure_reason is None"
                )

    def test_all_success_events_have_null_failure_reason(self) -> None:
        result = generate_dataset(_full_config())
        for event in result.events:
            if event.authentication_outcome == AuthOutcome.SUCCESS:
                assert event.failure_reason is None

    def test_novel_anomaly_not_supervised_training_eligible(self) -> None:
        result = generate_dataset(_full_config())
        for label in result.labels:
            if label.scenario == ScenarioType.NOVEL_ANOMALY_HOLDOUT:
                assert label.supervised_training_eligible is False, (
                    "Novel anomaly must not be supervised_training_eligible"
                )

    def test_normal_events_are_supervised_training_eligible(self) -> None:
        result = generate_dataset(_full_config())
        for label in result.labels:
            if label.scenario == ScenarioType.NORMAL:
                assert label.supervised_training_eligible is True

    def test_determinism_same_seed(self) -> None:
        r1 = generate_dataset(_tiny_config(seed=99))
        r2 = generate_dataset(_tiny_config(seed=99))
        ids1 = [str(e.event_id) for e in r1.events]
        ids2 = [str(e.event_id) for e in r2.events]
        assert ids1 == ids2

    def test_different_seeds_produce_different_events(self) -> None:
        r1 = generate_dataset(_tiny_config(seed=1))
        r2 = generate_dataset(_tiny_config(seed=2))
        ids1 = {str(e.event_id) for e in r1.events}
        ids2 = {str(e.event_id) for e in r2.events}
        # With different seeds, IDs are derived from different UUID5 inputs
        assert ids1 != ids2

    def test_disabled_scenario_produces_no_events_for_that_scenario(self) -> None:
        cfg = _tiny_config()  # brute_force disabled
        result = generate_dataset(cfg)
        for label in result.labels:
            assert label.scenario != ScenarioType.BRUTE_FORCE

    def test_all_scenarios_produce_events_when_enabled(self) -> None:
        result = generate_dataset(_full_config())
        scenarios = {label.scenario for label in result.labels}
        assert ScenarioType.NORMAL in scenarios
        assert ScenarioType.BRUTE_FORCE in scenarios
        assert ScenarioType.NOVEL_ANOMALY_HOLDOUT in scenarios

    def test_event_times_within_duration(self) -> None:
        cfg = _tiny_config()
        from datetime import timedelta

        end_time = cfg.start_time + timedelta(hours=cfg.duration_hours)
        result = generate_dataset(cfg)
        for event in result.events:
            assert cfg.start_time <= event.event_time <= end_time + timedelta(hours=1)

    def test_config_stored_in_result(self) -> None:
        cfg = _tiny_config()
        result = generate_dataset(cfg)
        assert result.config is cfg

    def test_all_events_are_valid_auth_events(self) -> None:
        from password_attack_detector.data.schemas import AuthEvent

        result = generate_dataset(_full_config())
        for event in result.events:
            assert isinstance(event, AuthEvent)
