"""Deterministic request bodies for the serving-layer suites.

Every helper here produces a JSON-ready mapping in exactly the shape a client
would send: no ``AuthEvent`` is constructed and then dumped, because that would
let a request pass a test by carrying a field the wire schema does not accept.

Identifiers are hashed from short logical names, so ``"u1"`` is the same user in
every test and in every run.  Times are offsets from a fixed base; nothing here
reads a wall clock.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from tests.features.factories import pseudonym

__all__ = [
    "BASE_TIME",
    "brute_force_window",
    "event",
    "normal_window",
    "spraying_window",
]

#: Fixed anchor time for every request fixture.  Arbitrary, timezone-aware, and
#: never derived from the clock.
BASE_TIME: datetime = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)

#: Namespace for deterministic event identifiers.
_NS_API_EVENT = uuid.UUID("6b1d7a4e-9c02-5f31-88ad-4e7f0b3c9d15")


def event(
    key: str,
    *,
    offset_seconds: float = 0.0,
    user: str = "u1",
    source: str = "s1",
    device: str = "d1",
    session: str = "sess1",
    outcome: str = "failure",
    application: str = "app-00",
    method: str = "password",
    response_time_ms: int | None = 120,
    country: str | None = "US",
    **overrides: Any,
) -> dict[str, Any]:
    """Return one request-shaped authentication event.

    *key* seeds the event identifier, so two calls with the same key produce the
    same event and a window can be rebuilt byte-for-byte.
    """
    body: dict[str, Any] = {
        "event_id": str(uuid.uuid5(_NS_API_EVENT, key)),
        "event_time": (BASE_TIME + timedelta(seconds=offset_seconds)).isoformat(),
        "user_id": pseudonym("user", user),
        "source_id": pseudonym("source", source),
        "device_id": pseudonym("device", device),
        "session_id": pseudonym("session", session),
        "application_id": application,
        "authentication_method": method,
        "authentication_outcome": outcome,
        "response_time_ms": response_time_ms,
        "country_code": country,
    }
    if outcome == "failure":
        body["failure_reason"] = "invalid_credentials"
    elif outcome == "blocked":
        body["failure_reason"] = "ip_blocked"
    body.update(overrides)
    return body


def brute_force_window(count: int = 30, *, user: str = "u1") -> list[dict[str, Any]]:
    """Return one source hammering one account with failures, ten seconds apart.

    The shape the brute-force rules describe: a single user/source pair, an
    unbroken run of failures, and enough of them inside one window to clear the
    configured count and rate thresholds.
    """
    return [
        event(
            f"bf-{user}-{index}",
            offset_seconds=10.0 * index,
            user=user,
            source="s-attacker",
            outcome="failure",
            response_time_ms=40,
        )
        for index in range(count)
    ]


def spraying_window(
    user_count: int = 24, *, source: str = "s-sprayer"
) -> list[dict[str, Any]]:
    """Return one source trying one failed attempt against many distinct accounts.

    The shape the spraying rule describes: high account fan-out from a single
    source, rather than repeated attempts against any single account.
    """
    return [
        event(
            f"ps-{index}",
            offset_seconds=5.0 * index,
            user=f"sprayed-{index:03d}",
            source=source,
            device=f"d-{index % 3}",
            session=f"sess-{index}",
            outcome="failure",
        )
        for index in range(user_count)
    ]


def normal_window(count: int = 12) -> list[dict[str, Any]]:
    """Return ordinary successful traffic, well spaced, from one stable identity."""
    return [
        event(
            f"ok-{index}",
            offset_seconds=600.0 * index,
            user="u-regular",
            source="s-office",
            outcome="success",
            response_time_ms=210 + index,
        )
        for index in range(count)
    ]
