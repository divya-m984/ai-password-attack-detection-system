"""Deterministic factories for building small authentication-event fixtures.

Hand-building an ``AuthEvent`` is verbose: identifiers must match the
pseudonym pattern ``^(u|s|d|sess):[0-9a-f]{32}$`` and the outcome must be
consistent with ``failure_reason`` and ``mfa_outcome``.  These helpers let a
temporal test read as a specification of behaviour rather than a wall of
fixture setup.

Short logical names (``"u1"``, ``"s2"``) are hashed into valid pseudonyms
deterministically, so the same logical name always yields the same identifier
across runs and across tests.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    CampaignStage,
    ClientType,
    FailureReason,
    MFAOutcome,
    ScenarioType,
)
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel

__all__ = [
    "BASE_TIME",
    "make_event",
    "make_labels",
    "make_stream",
    "pseudonym",
    "shift",
]

#: Fixed namespace so generated event ids are deterministic across runs.
_NS_TEST_EVENT = uuid.UUID("3f9a1c52-6b7d-5e84-9a10-2c4d6e8f0a12")

#: Default anchor time for fixtures; arbitrary but fixed and timezone-aware.
BASE_TIME: datetime = datetime(2024, 3, 4, 12, 0, 0, tzinfo=UTC)

_DOMAIN_PREFIX: dict[str, str] = {
    "user": "u",
    "source": "s",
    "device": "d",
    "session": "sess",
}

#: Failure reasons that satisfy the schema's BLOCKED consistency validator.
_BLOCKED_REASON = FailureReason.IP_BLOCKED
_FAILURE_REASON = FailureReason.INVALID_CREDENTIALS


def pseudonym(domain: str, logical_name: str) -> str:
    """Return a deterministic, schema-valid pseudonym for a logical name.

    ``pseudonym("user", "u1")`` always returns the same ``u:<32 hex>`` string.
    This is a test helper only; it is not the production pseudonymization
    scheme, which is keyed HMAC (see ``data/privacy.py``).
    """
    try:
        prefix = _DOMAIN_PREFIX[domain]
    except KeyError:
        raise ValueError(f"Unknown pseudonym domain: {domain!r}") from None
    digest = hashlib.sha256(f"{domain}:{logical_name}".encode()).hexdigest()[:32]
    return f"{prefix}:{digest}"


def shift(seconds: float, *, base: datetime = BASE_TIME) -> datetime:
    """Return *base* offset by *seconds*, preserving UTC."""
    return base.fromtimestamp(base.timestamp() + seconds, tz=UTC)


def _coerce_time(value: str | datetime | float | int) -> datetime:
    """Accept an ISO string, a datetime, or an offset in seconds from BASE_TIME."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Event times must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, int | float):
        return shift(float(value))
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"Event time {value!r} must be timezone-aware")
    return parsed.astimezone(UTC)


def _coerce_outcome(value: AuthOutcome | str) -> AuthOutcome:
    return value if isinstance(value, AuthOutcome) else AuthOutcome(value)


def _default_failure_reason(outcome: AuthOutcome) -> FailureReason | None:
    """Return a failure reason consistent with *outcome*.

    The schema requires one for FAILURE, requires a specific subset for
    BLOCKED, and forbids one for SUCCESS and CHALLENGED.
    """
    if outcome is AuthOutcome.FAILURE:
        return _FAILURE_REASON
    if outcome is AuthOutcome.BLOCKED:
        return _BLOCKED_REASON
    return None


def make_event(
    *,
    t: str | datetime | float | int = BASE_TIME,
    user: str = "u1",
    source: str = "s1",
    device: str = "d1",
    session: str = "sess1",
    application: str = "app-00",
    outcome: AuthOutcome | str = AuthOutcome.FAILURE,
    method: AuthMethod = AuthMethod.PASSWORD,
    failure_reason: FailureReason | str | None = "auto",
    mfa_outcome: MFAOutcome | None = None,
    country: str | None = None,
    region: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    user_agent: str | None = None,
    operating_system: str | None = None,
    client_type: ClientType | None = None,
    response_time_ms: int | None = None,
    event_id: uuid.UUID | None = None,
    key: str | None = None,
) -> AuthEvent:
    """Build one canonical ``AuthEvent`` from short logical names.

    ``t`` accepts a datetime, an ISO-8601 string, or a float offset in seconds
    from :data:`BASE_TIME`.  ``failure_reason`` defaults to a value consistent
    with ``outcome``; pass ``None`` explicitly to force it absent.

    ``event_id`` is derived deterministically from the event's content unless
    given, so tests that reorder input rows still produce stable identifiers.
    Pass ``key`` to disambiguate two events that are otherwise identical.
    """
    resolved_outcome = _coerce_outcome(outcome)
    event_time = _coerce_time(t)

    resolved_reason: FailureReason | None
    if failure_reason == "auto":
        resolved_reason = _default_failure_reason(resolved_outcome)
    elif isinstance(failure_reason, str):
        resolved_reason = FailureReason(failure_reason)
    else:
        resolved_reason = failure_reason

    if event_id is None:
        seed = "|".join(
            [
                event_time.isoformat(),
                user,
                source,
                device,
                session,
                application,
                str(resolved_outcome),
                str(method),
                key or "",
            ]
        )
        event_id = uuid.uuid5(_NS_TEST_EVENT, seed)

    return AuthEvent(
        event_id=event_id,
        event_time=event_time,
        user_id=pseudonym("user", user),
        source_id=pseudonym("source", source),
        device_id=pseudonym("device", device),
        session_id=pseudonym("session", session),
        application_id=application,
        authentication_method=method,
        authentication_outcome=resolved_outcome,
        failure_reason=resolved_reason,
        mfa_outcome=mfa_outcome,
        country_code=country,
        region_code=region,
        coarse_latitude=latitude,
        coarse_longitude=longitude,
        user_agent_family=user_agent,
        operating_system_family=operating_system,
        client_type=client_type,
        response_time_ms=response_time_ms,
    )


def make_stream(spec: str) -> list[AuthEvent]:
    """Build a list of events from a compact one-event-per-line DSL.

    Each non-empty, non-comment line is::

        <time> <user> <source> <outcome> [key=value ...]

    where ``<time>`` is either an ISO-8601 timestamp or a float offset in
    seconds from :data:`BASE_TIME`.  Two lines sharing a timestamp form a
    same-timestamp block, which is the case the point-in-time contract is most
    concerned with.

    Example::

        make_stream('''
            0    u1 s1 failure
            30   u1 s1 failure
            60   u1 s2 success   device=d2 country=US
            60   u1 s3 failure           # same timestamp as the line above
        ''')
    """
    events: list[AuthEvent] = []
    for index, raw_line in enumerate(spec.strip().splitlines()):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"Line {index + 1} needs at least '<time> <user> <source> "
                f"<outcome>', got {raw_line!r}"
            )

        time_token, user, source, outcome = parts[:4]
        kwargs: dict[str, Any] = {}
        for token in parts[4:]:
            if "=" not in token:
                raise ValueError(
                    f"Expected key=value on line {index + 1}, got {token!r}"
                )
            name, _, value = token.partition("=")
            kwargs[name] = _coerce_option(name, value)

        try:
            when: str | float = float(time_token)
        except ValueError:
            when = time_token

        events.append(
            make_event(
                t=when,
                user=user,
                source=source,
                outcome=outcome,
                key=kwargs.pop("key", str(index)),
                **kwargs,
            )
        )
    return events


_FLOAT_OPTIONS = frozenset({"latitude", "longitude"})
_INT_OPTIONS = frozenset({"response_time_ms"})
_ENUM_OPTIONS: dict[str, type[Any]] = {
    "method": AuthMethod,
    "mfa_outcome": MFAOutcome,
    "client_type": ClientType,
    "failure_reason": FailureReason,
}


def _coerce_option(name: str, value: str) -> Any:
    """Convert a ``key=value`` DSL token into the right Python type."""
    if value == "none":
        return None
    if name in _FLOAT_OPTIONS:
        return float(value)
    if name in _INT_OPTIONS:
        return int(value)
    if name in _ENUM_OPTIONS:
        return _ENUM_OPTIONS[name](value)
    return value


def make_labels(
    events: Sequence[AuthEvent],
    *,
    scenario: ScenarioType = ScenarioType.NORMAL,
    campaign_id: str = "c-normal",
    malicious: bool | None = None,
    supervised_training_eligible: bool | None = None,
    generator_version: str = "1.0.0",
    campaign_stage: CampaignStage | None = None,
) -> list[GroundTruthLabel]:
    """Build ground-truth labels for *events*.

    ``malicious`` defaults to ``scenario is not NORMAL``, and
    ``supervised_training_eligible`` defaults to ``False`` only for the novel
    anomaly holdout, matching the Phase 2 generator's conventions.
    """
    is_malicious = (
        scenario is not ScenarioType.NORMAL if malicious is None else malicious
    )
    eligible = (
        scenario is not ScenarioType.NOVEL_ANOMALY_HOLDOUT
        if supervised_training_eligible is None
        else supervised_training_eligible
    )
    return [
        GroundTruthLabel(
            event_id=event.event_id,
            campaign_id=campaign_id,
            scenario=scenario,
            malicious=is_malicious,
            supervised_training_eligible=eligible,
            generator_version=generator_version,
            campaign_stage=campaign_stage,
        )
        for event in events
    ]
