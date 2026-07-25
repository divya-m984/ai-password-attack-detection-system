"""Unit tests for password_attack_detector.data.synthetic.campaigns."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np

from password_attack_detector.data.enums import (
    AuthOutcome,
    ClientType,
    ScenarioType,
)
from password_attack_detector.data.schemas import (
    PROHIBITED_GT_COLUMNS,
    AuthEvent,
)
from password_attack_detector.data.synthetic.campaigns import (
    generate_account_takeover_indicator,
    generate_bot_activity,
    generate_brute_force,
    generate_credential_stuffing,
    generate_distributed_brute_force,
    generate_impossible_travel,
    generate_normal,
    generate_novel_anomaly_holdout,
    generate_password_spraying,
)
from password_attack_detector.data.synthetic.config import (
    BotActivityParams,
    BruteForceParams,
    CampaignParameters,
    CredentialStuffingParams,
    DistributedBruteForceParams,
    ImpossibleTravelParams,
    NovelAnomalyParams,
    PasswordSprayingParams,
    SyntheticConfig,
)
from password_attack_detector.data.synthetic.entities import (
    EntityPopulation,
    build_entity_population,
)

_START = datetime(2024, 1, 1, tzinfo=UTC)
_PSEUDONYM_RE = re.compile(r"^(u|s|d|sess):[0-9a-f]{32}$")


def _small_config(**kwargs: object) -> SyntheticConfig:
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
        bot_activity=BotActivityParams(events_per_campaign=4, num_campaigns=1),
        novel_anomaly_holdout=NovelAnomalyParams(num_campaigns=1),
    )
    defaults: dict[str, object] = {
        "seed": 42,
        "start_time": _START,
        "duration_hours": 1,
        "num_users": 5,
        "num_sources": 4,
        "num_devices": 8,
        "num_applications": 2,
        "events_per_hour": 5,
        "campaign_parameters": cp,
    }
    defaults.update(kwargs)
    return SyntheticConfig(**defaults)  # type: ignore[arg-type]


def _make_rng_and_pop(
    cfg: SyntheticConfig,
) -> tuple[np.random.Generator, EntityPopulation]:
    rng = np.random.default_rng(cfg.seed)
    pop = build_entity_population(cfg, rng)
    return rng, pop


def _assert_no_gt_columns(event: AuthEvent) -> None:
    """Assert that none of the PROHIBITED_GT_COLUMNS appear in the event."""
    event_fields = set(type(event).model_fields.keys())
    for col in PROHIBITED_GT_COLUMNS:
        assert col not in event_fields, f"GT column {col!r} found in AuthEvent"


def _assert_failure_reason_consistent(event: AuthEvent) -> None:
    if event.authentication_outcome in (AuthOutcome.FAILURE, AuthOutcome.BLOCKED):
        assert event.failure_reason is not None, (
            f"failure_reason must not be None for {event.authentication_outcome}"
        )
    elif event.authentication_outcome in (AuthOutcome.SUCCESS, AuthOutcome.CHALLENGED):
        assert event.failure_reason is None, (
            f"failure_reason must be None for {event.authentication_outcome}"
        )


class TestNormalGenerator:
    def test_produces_events(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        assert len(pairs) == cfg.events_per_hour * cfg.duration_hours

    def test_all_events_are_normal_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        for _, label in pairs:
            assert label.scenario == ScenarioType.NORMAL

    def test_no_gt_columns_in_events(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        for event, _ in pairs:
            _assert_no_gt_columns(event)

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_normal_events_not_malicious(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        for _, label in pairs:
            assert label.malicious is False

    def test_supervised_training_eligible(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        for _, label in pairs:
            assert label.supervised_training_eligible is True

    def test_event_ids_unique(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        ids = [str(e.event_id) for e, _ in pairs]
        assert len(ids) == len(set(ids))

    def test_one_to_one_event_label_mapping(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_normal(rng, cfg, pop, [0])
        for event, label in pairs:
            assert event.event_id == label.event_id


class TestBruteForceGenerator:
    def test_produces_events(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        assert len(pairs) > 0

    def test_all_brute_force_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.BRUTE_FORCE

    def test_malicious_true(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        for _, label in pairs:
            assert label.malicious is True

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_no_gt_columns(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        for event, _ in pairs:
            _assert_no_gt_columns(event)

    def test_uses_password_method(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        from password_attack_detector.data.enums import AuthMethod

        for event, _ in pairs:
            assert event.authentication_method == AuthMethod.PASSWORD

    def test_event_label_ids_match(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        for event, label in pairs:
            assert event.event_id == label.event_id

    def test_expected_event_count(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_brute_force(rng, cfg, pop, [1000])
        params = cfg.campaign_parameters.brute_force
        expected = params.attempts_per_campaign * params.num_campaigns
        assert len(pairs) == expected


class TestPasswordSprayingGenerator:
    def test_all_password_spraying_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_password_spraying(rng, cfg, pop, [2000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.PASSWORD_SPRAYING

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_password_spraying(rng, cfg, pop, [2000])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_malicious_true(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_password_spraying(rng, cfg, pop, [2000])
        for _, label in pairs:
            assert label.malicious is True


class TestCredentialStuffingGenerator:
    def test_all_credential_stuffing_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_credential_stuffing(rng, cfg, pop, [3000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.CREDENTIAL_STUFFING

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_credential_stuffing(rng, cfg, pop, [3000])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_event_label_ids_match(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_credential_stuffing(rng, cfg, pop, [3000])
        for event, label in pairs:
            assert event.event_id == label.event_id


class TestDistributedBruteForceGenerator:
    def test_all_distributed_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_distributed_brute_force(rng, cfg, pop, [4000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.DISTRIBUTED_BRUTE_FORCE

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_distributed_brute_force(rng, cfg, pop, [4000])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_multiple_sources_used(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_distributed_brute_force(rng, cfg, pop, [4000])
        source_ids = {e.source_id for e, _ in pairs}
        params = cfg.campaign_parameters.distributed_brute_force
        assert len(source_ids) >= params.num_sources


class TestAccountTakeoverGenerator:
    def test_all_ato_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_account_takeover_indicator(rng, cfg, pop, [5000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.ACCOUNT_TAKEOVER_INDICATOR

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_account_takeover_indicator(rng, cfg, pop, [5000])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_contains_success_event(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_account_takeover_indicator(rng, cfg, pop, [5000])
        outcomes = [e.authentication_outcome for e, _ in pairs]
        assert AuthOutcome.SUCCESS in outcomes


class TestImpossibleTravelGenerator:
    def test_all_impossible_travel_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_impossible_travel(rng, cfg, pop, [6000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.IMPOSSIBLE_TRAVEL

    def test_no_gt_columns_in_events(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_impossible_travel(rng, cfg, pop, [6000])
        for event, _ in pairs:
            _assert_no_gt_columns(event)

    def test_events_have_coarse_location(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_impossible_travel(rng, cfg, pop, [6000])
        for event, _ in pairs:
            assert event.coarse_latitude is not None
            assert event.coarse_longitude is not None
            assert event.country_code is not None

    def test_two_events_per_campaign(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_impossible_travel(rng, cfg, pop, [6000])
        params = cfg.campaign_parameters.impossible_travel
        assert len(pairs) == params.num_campaigns * 2

    def test_locations_differ_between_events(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_impossible_travel(rng, cfg, pop, [6000])
        if len(pairs) >= 2:
            e1, e2 = pairs[0][0], pairs[1][0]
            assert (e1.country_code != e2.country_code) or (
                e1.coarse_latitude != e2.coarse_latitude
            )

    def test_all_events_success(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_impossible_travel(rng, cfg, pop, [6000])
        for event, _ in pairs:
            assert event.authentication_outcome == AuthOutcome.SUCCESS
            assert event.failure_reason is None


class TestBotActivityGenerator:
    def test_all_bot_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_bot_activity(rng, cfg, pop, [7000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.BOT_ACTIVITY

    def test_client_type_is_bot(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_bot_activity(rng, cfg, pop, [7000])
        for event, _ in pairs:
            assert event.client_type == ClientType.BOT

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_bot_activity(rng, cfg, pop, [7000])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_expected_event_count(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_bot_activity(rng, cfg, pop, [7000])
        params = cfg.campaign_parameters.bot_activity
        assert len(pairs) == params.events_per_campaign * params.num_campaigns


class TestNovelAnomalyGenerator:
    def test_all_novel_anomaly_scenario(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_novel_anomaly_holdout(rng, cfg, pop, [8000])
        for _, label in pairs:
            assert label.scenario == ScenarioType.NOVEL_ANOMALY_HOLDOUT

    def test_supervised_training_eligible_is_false(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_novel_anomaly_holdout(rng, cfg, pop, [8000])
        for _, label in pairs:
            assert label.supervised_training_eligible is False

    def test_malicious_true(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_novel_anomaly_holdout(rng, cfg, pop, [8000])
        for _, label in pairs:
            assert label.malicious is True

    def test_failure_reason_consistent(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_novel_anomaly_holdout(rng, cfg, pop, [8000])
        for event, _ in pairs:
            _assert_failure_reason_consistent(event)

    def test_no_gt_columns(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        pairs = generate_novel_anomaly_holdout(rng, cfg, pop, [8000])
        for event, _ in pairs:
            _assert_no_gt_columns(event)


class TestCounterAdvancement:
    def test_counter_incremented_per_event(self) -> None:
        cfg = _small_config()
        rng, pop = _make_rng_and_pop(cfg)
        counter: list[int] = [0]
        pairs = generate_normal(rng, cfg, pop, counter)
        assert counter[0] == len(pairs)

    def test_no_counter_overlap_between_generators(self) -> None:
        cfg = _small_config()
        rng1 = np.random.default_rng(cfg.seed)
        pop1 = build_entity_population(cfg, rng1)
        counter: list[int] = [0]
        p_normal = generate_normal(rng1, cfg, pop1, counter)
        p_bf = generate_brute_force(rng1, cfg, pop1, counter)
        ids_normal = {str(e.event_id) for e, _ in p_normal}
        ids_bf = {str(e.event_id) for e, _ in p_bf}
        assert ids_normal.isdisjoint(ids_bf)
