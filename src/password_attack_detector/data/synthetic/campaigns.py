"""Synthetic event campaign generators for all nine scenario types.

Each public ``generate_*`` function produces a list of (AuthEvent,
GroundTruthLabel) pairs for one scenario.  The canonical ``AuthEvent``
contains **no** ground-truth columns (scenario, malicious, campaign_id,
risk score, attack probability, etc.).  Ground truth is held exclusively
in ``GroundTruthLabel`` and joined by ``event_id``.

Determinism guarantee
---------------------
All event identifiers (event_id, session_id) are produced via UUIDv5
with fixed project-internal namespaces keyed on ``(seed, counter)``.
Timing offsets use the caller-supplied ``np.random.Generator`` only.
The system clock is never read.

Failure-reason contract
-----------------------
Every ``AuthEvent`` with outcome ``FAILURE`` or ``BLOCKED`` carries a
non-null ``failure_reason``.  Events with outcome ``SUCCESS`` or
``CHALLENGED`` carry ``failure_reason=None``.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import numpy as np

from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    CampaignStage,
    ClientType,
    FailureReason,
    ScenarioType,
)
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.data.synthetic.config import SyntheticConfig
from password_attack_detector.data.synthetic.entities import (
    EntityPopulation,
    make_session_id,
    make_source_id,
)
from password_attack_detector.data.synthetic.profiles import UserProfile

__all__ = [
    "generate_account_takeover_indicator",
    "generate_bot_activity",
    "generate_brute_force",
    "generate_credential_stuffing",
    "generate_distributed_brute_force",
    "generate_impossible_travel",
    "generate_normal",
    "generate_novel_anomaly_holdout",
    "generate_password_spraying",
]

# Fixed event-ID namespace.  Must not change between generator versions.
_NS_EVENT = uuid.UUID("0b18e7d4-5f6c-5021-d293-0e1f2a3b4c5d")

EventPair = tuple[AuthEvent, GroundTruthLabel]
EventPairs = list[EventPair]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _event_id(seed: int, counter: int) -> uuid.UUID:
    return uuid.uuid5(_NS_EVENT, f"{seed}:{counter}")


def _random_offset(rng: np.random.Generator, duration_seconds: float) -> float:
    return float(rng.uniform(0.0, duration_seconds))


def _pick(rng: np.random.Generator, seq: tuple[object, ...]) -> object:
    idx = int(rng.integers(0, len(seq)))
    return seq[idx]


def _build_event(
    *,
    seed: int,
    counter: int,
    config: SyntheticConfig,
    offset_seconds: float,
    user: UserProfile,
    source_id: str,
    device_id: str,
    app_id: str,
    auth_method: AuthMethod,
    outcome: AuthOutcome,
    failure_reason: FailureReason | None = None,
    country_code: str | None = None,
    coarse_latitude: float | None = None,
    coarse_longitude: float | None = None,
    response_time_ms: int | None = None,
    client_type: ClientType | None = None,
) -> AuthEvent:
    event_time = config.start_time + timedelta(seconds=offset_seconds)
    session_id = make_session_id(seed, counter)
    return AuthEvent(
        event_id=_event_id(seed, counter),
        event_time=event_time,
        user_id=user.user_id,
        source_id=source_id,
        device_id=device_id,
        session_id=session_id,
        application_id=app_id,
        authentication_method=auth_method,
        authentication_outcome=outcome,
        failure_reason=failure_reason,
        country_code=country_code,
        coarse_latitude=coarse_latitude,
        coarse_longitude=coarse_longitude,
        response_time_ms=response_time_ms,
        client_type=client_type,
    )


def _gt(
    event: AuthEvent,
    *,
    campaign_id: str,
    scenario: ScenarioType,
    malicious: bool,
    supervised_training_eligible: bool = True,
    scenario_variant: str | None = None,
    campaign_stage: CampaignStage | None = None,
) -> GroundTruthLabel:
    return GroundTruthLabel(
        event_id=event.event_id,
        campaign_id=campaign_id,
        scenario=scenario,
        malicious=malicious,
        supervised_training_eligible=supervised_training_eligible,
        generator_version="1.0.0",
        scenario_variant=scenario_variant,
        campaign_stage=campaign_stage,
    )


def _duration(config: SyntheticConfig) -> float:
    return float(config.duration_hours * 3600)


# ---------------------------------------------------------------------------
# 1. Normal (background traffic)
# ---------------------------------------------------------------------------


def generate_normal(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate benign background authentication events."""
    total = config.events_per_hour * config.duration_hours
    dur = _duration(config)
    pairs: EventPairs = []

    for _ in range(total):
        c = counter[0]
        counter[0] += 1

        user_idx = int(rng.integers(0, len(population.users)))
        user = population.users[user_idx]

        source_id = (
            str(_pick(rng, user.known_source_ids))
            if user.known_source_ids
            else population.sources[0].source_id
        )
        device_id = (
            str(_pick(rng, user.known_device_ids))
            if user.known_device_ids
            else population.devices[0].device_id
        )
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]
        method = user.preferred_auth_methods[
            int(rng.integers(0, len(user.preferred_auth_methods)))
        ]

        succeed = float(rng.random()) < user.baseline_success_rate
        outcome = AuthOutcome.SUCCESS if succeed else AuthOutcome.FAILURE
        failure_reason = None if succeed else FailureReason.INVALID_CREDENTIALS

        src_profile = next(
            (s for s in population.sources if s.source_id == source_id),
            population.sources[0],
        )

        offset = _random_offset(rng, dur)
        event = _build_event(
            seed=config.seed,
            counter=c,
            config=config,
            offset_seconds=offset,
            user=user,
            source_id=source_id,
            device_id=device_id,
            app_id=app.application_id,
            auth_method=method,
            outcome=outcome,
            failure_reason=failure_reason,
            country_code=src_profile.country_code,
            coarse_latitude=src_profile.coarse_latitude,
            coarse_longitude=src_profile.coarse_longitude,
            response_time_ms=int(rng.integers(50, 2000)),
        )
        label = _gt(
            event,
            campaign_id=f"normal-{config.seed}",
            scenario=ScenarioType.NORMAL,
            malicious=False,
        )
        pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 2. Brute Force (single-source, single-target)
# ---------------------------------------------------------------------------


def generate_brute_force(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate single-source brute-force attack campaigns."""
    params = config.campaign_parameters.brute_force
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"bf-{config.seed}-{camp_i}"
        target_user = population.users[int(rng.integers(0, len(population.users)))]
        # Attacker source: beyond-population index for deterministic unique source
        attacker_src_id = make_source_id(config.seed, 10000 + camp_i)
        device_id = population.devices[
            int(rng.integers(0, len(population.devices)))
        ].device_id
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]

        attack_start = float(rng.uniform(0.0, dur * 0.7))

        for attempt in range(params.attempts_per_campaign):
            c = counter[0]
            counter[0] += 1

            offset = attack_start + attempt * 2.0  # 2 seconds between attempts
            if attempt < params.attempts_per_campaign - 1:
                outcome = AuthOutcome.FAILURE
                fr: FailureReason | None = (
                    FailureReason.ACCOUNT_LOCKED
                    if attempt >= params.attempts_per_campaign // 2
                    else FailureReason.INVALID_CREDENTIALS
                )
                stage = CampaignStage.ACTIVE
            else:
                outcome = AuthOutcome.SUCCESS
                fr = None
                stage = CampaignStage.EXFIL

            event = _build_event(
                seed=config.seed,
                counter=c,
                config=config,
                offset_seconds=offset,
                user=target_user,
                source_id=attacker_src_id,
                device_id=device_id,
                app_id=app.application_id,
                auth_method=AuthMethod.PASSWORD,
                outcome=outcome,
                failure_reason=fr,
                response_time_ms=int(rng.integers(80, 300)),
                client_type=ClientType.API_CLIENT,
            )
            label = _gt(
                event,
                campaign_id=campaign_id,
                scenario=ScenarioType.BRUTE_FORCE,
                malicious=True,
                campaign_stage=stage,
            )
            pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 3. Password Spraying (one attacker, many targets, few passwords)
# ---------------------------------------------------------------------------


def generate_password_spraying(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate password-spraying attack campaigns."""
    params = config.campaign_parameters.password_spraying
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"ps-{config.seed}-{camp_i}"
        attacker_src_id = make_source_id(config.seed, 20000 + camp_i)
        device_id = population.devices[
            int(rng.integers(0, len(population.devices)))
        ].device_id
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]

        attack_start = float(rng.uniform(0.0, dur * 0.6))
        num_targets = min(len(population.users), params.passwords_per_round * 2)
        target_indices = [
            int(x)
            for x in rng.choice(len(population.users), size=num_targets, replace=False)
        ]

        for round_i in range(params.passwords_per_round):
            for target_idx in target_indices:
                c = counter[0]
                counter[0] += 1

                target_user = population.users[target_idx]
                offset = (
                    attack_start
                    + (round_i * num_targets + target_indices.index(target_idx)) * 30.0
                )

                event = _build_event(
                    seed=config.seed,
                    counter=c,
                    config=config,
                    offset_seconds=offset,
                    user=target_user,
                    source_id=attacker_src_id,
                    device_id=device_id,
                    app_id=app.application_id,
                    auth_method=AuthMethod.PASSWORD,
                    outcome=AuthOutcome.FAILURE,
                    failure_reason=FailureReason.INVALID_CREDENTIALS,
                    response_time_ms=int(rng.integers(80, 300)),
                    client_type=ClientType.API_CLIENT,
                )
                label = _gt(
                    event,
                    campaign_id=campaign_id,
                    scenario=ScenarioType.PASSWORD_SPRAYING,
                    malicious=True,
                    campaign_stage=CampaignStage.ACTIVE,
                )
                pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 4. Credential Stuffing (automated, multiple sources)
# ---------------------------------------------------------------------------


def generate_credential_stuffing(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate credential-stuffing attack campaigns."""
    params = config.campaign_parameters.credential_stuffing
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"cs-{config.seed}-{camp_i}"
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]
        attack_start = float(rng.uniform(0.0, dur * 0.6))

        for cred_i in range(params.credentials_per_batch):
            c = counter[0]
            counter[0] += 1

            target_user = population.users[cred_i % len(population.users)]
            attacker_src_id = make_source_id(config.seed, 30000 + camp_i * 100 + cred_i)
            device_id = population.devices[cred_i % len(population.devices)].device_id

            succeed = cred_i == 0  # first credential hits
            outcome = AuthOutcome.SUCCESS if succeed else AuthOutcome.FAILURE
            fr_cs: FailureReason | None = (
                None if succeed else FailureReason.INVALID_CREDENTIALS
            )

            offset = attack_start + cred_i * 0.5

            event = _build_event(
                seed=config.seed,
                counter=c,
                config=config,
                offset_seconds=offset,
                user=target_user,
                source_id=attacker_src_id,
                device_id=device_id,
                app_id=app.application_id,
                auth_method=AuthMethod.PASSWORD,
                outcome=outcome,
                failure_reason=fr_cs,
                response_time_ms=int(rng.integers(80, 300)),
                client_type=ClientType.API_CLIENT,
            )
            label = _gt(
                event,
                campaign_id=campaign_id,
                scenario=ScenarioType.CREDENTIAL_STUFFING,
                malicious=True,
                campaign_stage=CampaignStage.ACTIVE
                if not succeed
                else CampaignStage.EXFIL,
            )
            pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 5. Distributed Brute Force (many sources, single target)
# ---------------------------------------------------------------------------


def generate_distributed_brute_force(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate distributed brute-force attack campaigns."""
    params = config.campaign_parameters.distributed_brute_force
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"dbf-{config.seed}-{camp_i}"
        target_user = population.users[int(rng.integers(0, len(population.users)))]
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]
        attack_start = float(rng.uniform(0.0, dur * 0.6))

        for src_i in range(params.num_sources):
            attacker_src_id = make_source_id(config.seed, 40000 + camp_i * 100 + src_i)
            device_id = population.devices[src_i % len(population.devices)].device_id

            for attempt in range(params.attempts_per_source):
                c = counter[0]
                counter[0] += 1

                offset = (
                    attack_start + (src_i * params.attempts_per_source + attempt) * 5.0
                )
                is_last_overall = (
                    src_i == params.num_sources - 1
                    and attempt == params.attempts_per_source - 1
                )
                outcome = (
                    AuthOutcome.SUCCESS if is_last_overall else AuthOutcome.FAILURE
                )
                fr_dbf: FailureReason | None = (
                    None if is_last_overall else FailureReason.INVALID_CREDENTIALS
                )

                event = _build_event(
                    seed=config.seed,
                    counter=c,
                    config=config,
                    offset_seconds=offset,
                    user=target_user,
                    source_id=attacker_src_id,
                    device_id=device_id,
                    app_id=app.application_id,
                    auth_method=AuthMethod.PASSWORD,
                    outcome=outcome,
                    failure_reason=fr_dbf,
                    response_time_ms=int(rng.integers(80, 300)),
                    client_type=ClientType.API_CLIENT,
                )
                label = _gt(
                    event,
                    campaign_id=campaign_id,
                    scenario=ScenarioType.DISTRIBUTED_BRUTE_FORCE,
                    malicious=True,
                    campaign_stage=CampaignStage.ACTIVE,
                )
                pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 6. Account Takeover Indicator
# ---------------------------------------------------------------------------


def generate_account_takeover_indicator(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate account-takeover-indicator campaigns.

    Pattern: failed attempts from an unfamiliar source, followed by a
    successful login from that same source (indicator of compromised credentials).
    """
    params = config.campaign_parameters.account_takeover_indicator
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"ato-{config.seed}-{camp_i}"
        target_user = population.users[int(rng.integers(0, len(population.users)))]
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]
        attacker_src_id = make_source_id(config.seed, 50000 + camp_i)
        device_id = population.devices[
            int(rng.integers(0, len(population.devices)))
        ].device_id
        attack_start = float(rng.uniform(0.0, dur * 0.7))

        # 3 failures followed by 1 success
        for attempt in range(4):
            c = counter[0]
            counter[0] += 1

            offset = attack_start + attempt * 60.0
            succeed_ato = attempt == 3
            outcome = AuthOutcome.SUCCESS if succeed_ato else AuthOutcome.FAILURE
            fr_ato: FailureReason | None = (
                None if succeed_ato else FailureReason.INVALID_CREDENTIALS
            )
            stage = CampaignStage.EXFIL if succeed_ato else CampaignStage.ACTIVE

            event = _build_event(
                seed=config.seed,
                counter=c,
                config=config,
                offset_seconds=offset,
                user=target_user,
                source_id=attacker_src_id,
                device_id=device_id,
                app_id=app.application_id,
                auth_method=AuthMethod.PASSWORD,
                outcome=outcome,
                failure_reason=fr_ato,
                response_time_ms=int(rng.integers(80, 300)),
                client_type=ClientType.WEB_BROWSER,
            )
            label = _gt(
                event,
                campaign_id=campaign_id,
                scenario=ScenarioType.ACCOUNT_TAKEOVER_INDICATOR,
                malicious=True,
                campaign_stage=stage,
            )
            pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 7. Impossible Travel
# ---------------------------------------------------------------------------

# Two geographically distant country pairs for impossible-travel events.
_TRAVEL_LOCATIONS: list[tuple[str, float, float]] = [
    ("US", 38.0, -97.0),
    ("AU", -25.0, 133.0),
    ("SG", 1.0, 104.0),
    ("GB", 55.0, -3.0),
]


def generate_impossible_travel(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate impossible-travel campaigns.

    The canonical ``AuthEvent`` records only timestamps and coarse geo
    coordinates.  No impossible-travel flag appears in the canonical record;
    the anomaly is detectable purely from the temporal and spatial data.
    """
    params = config.campaign_parameters.impossible_travel
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"it-{config.seed}-{camp_i}"
        target_user = population.users[int(rng.integers(0, len(population.users)))]
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]
        device_id = population.devices[
            int(rng.integers(0, len(population.devices)))
        ].device_id

        loc_a = _TRAVEL_LOCATIONS[camp_i % len(_TRAVEL_LOCATIONS)]
        loc_b = _TRAVEL_LOCATIONS[(camp_i + 2) % len(_TRAVEL_LOCATIONS)]

        attack_start = float(rng.uniform(0.0, dur * 0.8))

        src_a = make_source_id(config.seed, 60000 + camp_i * 2)
        src_b = make_source_id(config.seed, 60000 + camp_i * 2 + 1)

        for loc_i, (src_id, loc) in enumerate(((src_a, loc_a), (src_b, loc_b))):
            c = counter[0]
            counter[0] += 1

            # Only 15 minutes apart — physically impossible given the distance
            offset = attack_start + loc_i * 900.0

            event = _build_event(
                seed=config.seed,
                counter=c,
                config=config,
                offset_seconds=offset,
                user=target_user,
                source_id=src_id,
                device_id=device_id,
                app_id=app.application_id,
                auth_method=AuthMethod.PASSWORD,
                outcome=AuthOutcome.SUCCESS,
                failure_reason=None,
                country_code=loc[0],
                coarse_latitude=loc[1],
                coarse_longitude=loc[2],
                response_time_ms=int(rng.integers(80, 500)),
                client_type=ClientType.WEB_BROWSER,
            )
            label = _gt(
                event,
                campaign_id=campaign_id,
                scenario=ScenarioType.IMPOSSIBLE_TRAVEL,
                malicious=True,
                campaign_stage=CampaignStage.ACTIVE,
            )
            pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 8. Bot Activity
# ---------------------------------------------------------------------------


def generate_bot_activity(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate bot-activity campaigns (mechanical, high-frequency requests)."""
    params = config.campaign_parameters.bot_activity
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"bot-{config.seed}-{camp_i}"
        target_user = population.users[int(rng.integers(0, len(population.users)))]
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]
        bot_src_id = make_source_id(config.seed, 70000 + camp_i)
        bot_device_id = population.devices[
            int(rng.integers(0, len(population.devices)))
        ].device_id

        attack_start = float(rng.uniform(0.0, dur * 0.5))

        for bot_i in range(params.events_per_campaign):
            c = counter[0]
            counter[0] += 1

            # Highly regular inter-event interval (0.1 seconds apart)
            offset = attack_start + bot_i * 0.1
            succeed_bot = bot_i % 10 == 0  # occasional success
            outcome = AuthOutcome.SUCCESS if succeed_bot else AuthOutcome.FAILURE
            fr_bot: FailureReason | None = (
                None if succeed_bot else FailureReason.INVALID_CREDENTIALS
            )

            event = _build_event(
                seed=config.seed,
                counter=c,
                config=config,
                offset_seconds=offset,
                user=target_user,
                source_id=bot_src_id,
                device_id=bot_device_id,
                app_id=app.application_id,
                auth_method=AuthMethod.PASSWORD,
                outcome=outcome,
                failure_reason=fr_bot,
                response_time_ms=int(rng.integers(10, 50)),
                client_type=ClientType.BOT,
            )
            label = _gt(
                event,
                campaign_id=campaign_id,
                scenario=ScenarioType.BOT_ACTIVITY,
                malicious=True,
                campaign_stage=CampaignStage.ACTIVE,
            )
            pairs.append((event, label))

    return pairs


# ---------------------------------------------------------------------------
# 9. Novel Anomaly Holdout
# ---------------------------------------------------------------------------


def generate_novel_anomaly_holdout(
    rng: np.random.Generator,
    config: SyntheticConfig,
    population: EntityPopulation,
    counter: list[int],
) -> EventPairs:
    """Generate novel-anomaly holdout campaigns.

    Events are reserved for unsupervised/anomaly-detection evaluation.
    ``supervised_training_eligible=False`` for every event in this scenario.
    """
    params = config.campaign_parameters.novel_anomaly_holdout
    dur = _duration(config)
    pairs: EventPairs = []

    for camp_i in range(params.num_campaigns):
        campaign_id = f"na-{config.seed}-{camp_i}"
        target_user = population.users[int(rng.integers(0, len(population.users)))]
        app = population.applications[
            int(rng.integers(0, len(population.applications)))
        ]
        novel_src_id = make_source_id(config.seed, 80000 + camp_i)
        device_id = population.devices[
            int(rng.integers(0, len(population.devices)))
        ].device_id
        attack_start = float(rng.uniform(0.0, dur * 0.7))

        # Unusual pattern: random auth method, unusual timing, mixed outcomes
        for ev_i in range(3):
            c = counter[0]
            counter[0] += 1

            offset = attack_start + ev_i * 3600.0  # 1 hour gaps (unusual)
            method = AuthMethod.CERTIFICATE  # unusual method
            succeed_na = ev_i == 2
            outcome = AuthOutcome.SUCCESS if succeed_na else AuthOutcome.CHALLENGED
            fr_na: FailureReason | None = None  # CHALLENGED → None

            event = _build_event(
                seed=config.seed,
                counter=c,
                config=config,
                offset_seconds=offset,
                user=target_user,
                source_id=novel_src_id,
                device_id=device_id,
                app_id=app.application_id,
                auth_method=method,
                outcome=outcome,
                failure_reason=fr_na,
                response_time_ms=int(rng.integers(500, 5000)),
                client_type=ClientType.CLI_TOOL,
            )
            label = _gt(
                event,
                campaign_id=campaign_id,
                scenario=ScenarioType.NOVEL_ANOMALY_HOLDOUT,
                malicious=True,
                supervised_training_eligible=False,
                campaign_stage=CampaignStage.ACTIVE,
            )
            pairs.append((event, label))

    return pairs
