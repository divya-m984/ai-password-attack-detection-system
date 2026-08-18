"""Safe synthetic windows, and the vocabulary the console builds events from.

**These are documentation fixtures, not attacks.**  Every one is a list of
request bodies describing authentication activity that *did not happen*, aimed at
a service the operator is running themselves.  Nothing here contacts a login
endpoint, tries a credential, or carries one: the API refuses credential material
under any spelling, and there is no field below through which one could be sent.
Identities are synthetic labels hashed to the pseudonym shape the wire contract
requires, and where an address appears it is drawn from the ranges RFC 5737
reserves for documentation.

The templates exist because a detection console with an empty form is a console
nobody can evaluate.  Loading one **populates the request form and does nothing
else** -- the analyst still presses submit, and the request still goes through
the API like any other.  Nothing here bypasses the service or scores anything.

Full replay automation is a later milestone.  These are three fixed windows.

The enumeration constants are the wire contract's own vocabularies, restated
here rather than imported from
:mod:`password_attack_detector.data.enums`: this package talks to the API over a
socket and does not import the detection stack.  A test pins each list against
the server's enum, so a value added there is a failing test rather than a form
that silently offers eleven of twelve options.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

__all__ = [
    "APPLICATION_IDS",
    "AUTHENTICATION_METHODS",
    "AUTHENTICATION_OUTCOMES",
    "BLOCKED_FAILURE_REASONS",
    "CLIENT_TYPES",
    "COUNTRY_CODES",
    "DOCUMENTATION_ADDRESSES",
    "FAILURE_REASONS",
    "MFA_OUTCOMES",
    "PROHIBITED_EVENT_FIELDS",
    "SCENARIOS",
    "Scenario",
    "build_event",
    "prohibited_field_names",
    "pseudonym",
    "scenario_events",
]

#: Field names that would carry credential material, **normalised**: lowercased,
#: with hyphens, underscores and spaces removed and camelCase split first, so
#: ``password_hash``, ``passwordHash`` and ``Password-Hash`` are one entry.
#:
#: The same normalisation the project's ingestion scanner uses, and a superset of
#: the names it refuses -- a test asserts the containment. Restated here rather
#: than imported from :mod:`password_attack_detector.data.privacy` for the same
#: reason the wire contract is restated: this package imports no part of the
#: detection stack.
#:
#: The service refuses these regardless. This list exists so one can never be
#: *stored* in a browser session or echoed back onto a page in the first place,
#: both of which would happen before the service ever saw the request.
PROHIBITED_EVENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "authtoken",
        "cookie",
        "credential",
        "credentials",
        "hash",
        "otp",
        "passcode",
        "passphrase",
        "passwd",
        "password",
        "passwordhash",
        "pin",
        "privatekey",
        "pwd",
        "refreshtoken",
        "secret",
        "sessiontoken",
        "token",
    }
)

_CAMEL_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[-_\s]+")


def _normalized(name: str) -> str:
    """Return *name* in the form :data:`PROHIBITED_EVENT_FIELDS` is written in."""
    return _SEPARATORS.sub("", _CAMEL_BOUNDARY.sub(" ", name)).strip().lower()


def prohibited_field_names(event: Any) -> tuple[str, ...]:
    """Return the credential-shaped keys *event* carries, in sorted order.

    Reads **names only**. A prohibited field's value is never read, copied, or
    included in any message built from this -- which is the same discipline the
    service's own scanner follows, and for the same reason: a scrubber has to be
    right every time, and not reading has to be right once.
    """
    if not isinstance(event, Mapping):
        return ()
    return tuple(
        sorted(
            str(key)
            for key in event
            if _normalized(str(key)) in PROHIBITED_EVENT_FIELDS
        )
    )


#: Authentication methods the wire contract accepts.
AUTHENTICATION_METHODS: Final[tuple[str, ...]] = (
    "password",
    "mfa_totp",
    "mfa_sms",
    "mfa_email",
    "sso",
    "oauth2",
    "api_key",
    "certificate",
    "biometric",
    "passkey",
)

#: Authentication outcomes the wire contract accepts.
AUTHENTICATION_OUTCOMES: Final[tuple[str, ...]] = (
    "success",
    "failure",
    "blocked",
    "challenged",
)

#: Failure reasons the wire contract accepts.
FAILURE_REASONS: Final[tuple[str, ...]] = (
    "invalid_credentials",
    "account_locked",
    "account_disabled",
    "account_not_found",
    "mfa_failed",
    "mfa_expired",
    "token_expired",
    "ip_blocked",
    "rate_limited",
    "suspicious_activity",
    "unknown",
)

#: The subset of :data:`FAILURE_REASONS` the canonical schema admits alongside a
#: ``blocked`` outcome.  Offering the others there would build a window the API
#: correctly refuses, which reads as a dashboard bug.
BLOCKED_FAILURE_REASONS: Final[tuple[str, ...]] = (
    "ip_blocked",
    "rate_limited",
    "suspicious_activity",
    "account_locked",
    "account_disabled",
    "unknown",
)

#: MFA outcomes the wire contract accepts.
MFA_OUTCOMES: Final[tuple[str, ...]] = (
    "passed",
    "failed",
    "bypassed",
    "not_required",
    "not_enrolled",
)

#: Client types the wire contract accepts.
CLIENT_TYPES: Final[tuple[str, ...]] = (
    "web_browser",
    "mobile_app",
    "desktop_app",
    "api_client",
    "cli_tool",
    "bot",
    "unknown",
)

#: Application labels the console offers.  Arbitrary synthetic names; the schema
#: accepts any bounded string, and offering a list keeps a demo consistent.
APPLICATION_IDS: Final[tuple[str, ...]] = (
    "app-00",
    "app-01",
    "corporate-vpn",
    "mail-gateway",
    "internal-portal",
)

#: Country codes the console offers.  A short list, not the ISO register: this
#: is a picker for a demonstration, not a geography reference.
COUNTRY_CODES: Final[tuple[str, ...]] = ("US", "GB", "DE", "IN", "BR", "JP", "AU")

#: Addresses RFC 5737 reserves for documentation.  The only literal addresses
#: this package contains, and the defaults the console's address field offers.
DOCUMENTATION_ADDRESSES: Final[tuple[str, ...]] = (
    "192.0.2.10",
    "198.51.100.23",
    "203.0.113.47",
)

#: Namespace for deterministic synthetic event identifiers.  Fixed, so the same
#: template produces the same window on every machine and in every session.
_NS_SCENARIO: Final[uuid.UUID] = uuid.UUID("3f5c1a8e-6d24-5b90-9f13-2a7c4e8b0d61")

#: Fixed base instant every template's offsets are measured from.  Deliberately
#: not "now": a template that moved with the clock would produce a different
#: window on every page load, and two runs of a demo could not be compared.
BASE_TIME: Final[datetime] = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)

#: Domain prefix each pseudonymous identifier field requires.
_PREFIX: Final[dict[str, str]] = {
    "user": "u",
    "source": "s",
    "device": "d",
    "session": "sess",
}


def pseudonym(domain: str, label: str) -> str:
    """Return a synthetic identifier in the pseudonym shape the API requires.

    **This is not the project's pseudonymization.**  That is a keyed HMAC over
    real identifiers, it lives in
    :mod:`password_attack_detector.data.privacy`, and it never runs here: there
    is no real identifier in this module to protect.  This function fabricates a
    stable opaque label for made-up demo entities, so ``"attacker"`` is the same
    non-existent source in every session and every run.

    Raises:
        KeyError: when *domain* is not one the wire contract defines.
    """
    prefix = _PREFIX[domain]
    digest = hashlib.sha256(f"pad-dashboard-demo:{domain}:{label}".encode()).hexdigest()
    return f"{prefix}:{digest[:32]}"


def build_event(
    key: str,
    *,
    offset_seconds: float = 0.0,
    user: str = "u1",
    source: str = "s1",
    device: str = "d1",
    session: str = "sess1",
    outcome: str = "failure",
    failure_reason: str | None = None,
    method: str = "password",
    application: str = "app-00",
    country: str | None = "US",
    response_time_ms: int | None = 120,
    mfa_outcome: str | None = None,
    client_type: str | None = None,
    source_ip: str | None = None,
    base_time: datetime = BASE_TIME,
) -> dict[str, Any]:
    """Return one request-shaped authentication event.

    *key* seeds the event identifier, so the same key always produces the same
    event and a template is byte-identical between sessions.

    The outcome/failure-reason pairing is filled in to match the canonical
    schema's rule -- a failure names a reason, a success names none -- because a
    template that built an invalid window would be indistinguishable, to whoever
    is running the demo, from a broken API.

    Supplying *source_ip* replaces the pseudonymous source rather than
    accompanying it: the schema requires exactly one of the two.
    """
    body: dict[str, Any] = {
        "event_id": str(uuid.uuid5(_NS_SCENARIO, key)),
        "event_time": (base_time + timedelta(seconds=offset_seconds)).isoformat(),
        "user_id": pseudonym("user", user),
        "device_id": pseudonym("device", device),
        "session_id": pseudonym("session", session),
        "application_id": application,
        "authentication_method": method,
        "authentication_outcome": outcome,
    }
    if source_ip is not None:
        body["source_ip"] = source_ip
    else:
        body["source_id"] = pseudonym("source", source)
    if outcome in {"failure", "blocked"}:
        default = "invalid_credentials" if outcome == "failure" else "ip_blocked"
        body["failure_reason"] = failure_reason or default
    elif failure_reason is not None:
        # The schema forbids a reason on a success or a challenge. Dropping it
        # here rather than passing it on keeps an inconsistent form state from
        # becoming an API refusal the analyst has to decode.
        pass
    if country is not None:
        body["country_code"] = country
    if response_time_ms is not None:
        body["response_time_ms"] = response_time_ms
    if mfa_outcome is not None:
        body["mfa_outcome"] = mfa_outcome
    if client_type is not None:
        body["client_type"] = client_type
    return body


class Scenario:
    """One named synthetic window, and what it is meant to demonstrate."""

    __slots__ = ("builder", "description", "expectation", "key", "label")

    def __init__(
        self,
        key: str,
        label: str,
        description: str,
        expectation: str,
        builder: Any,
    ) -> None:
        self.key = key
        self.label = label
        self.description = description
        #: What the detection system is expected to make of it, in words. A
        #: statement of intent, not a prediction the dashboard then checks: the
        #: verdict is whatever the frozen system returns.
        self.expectation = expectation
        self.builder = builder

    def events(self) -> list[dict[str, Any]]:
        """Return this scenario's window."""
        built: list[dict[str, Any]] = self.builder()
        return built


def _normal_window(count: int = 12) -> list[dict[str, Any]]:
    """Ordinary successful sign-ins from one stable identity, ten minutes apart."""
    return [
        build_event(
            f"normal-{index}",
            offset_seconds=600.0 * index,
            user="regular-analyst",
            source="office-egress",
            device="laptop-01",
            session=f"day-session-{index}",
            outcome="success",
            application="internal-portal",
            response_time_ms=210 + index,
            mfa_outcome="passed",
            client_type="web_browser",
        )
        for index in range(count)
    ]


def _brute_force_window(count: int = 30) -> list[dict[str, Any]]:
    """One source hammering one account with failures, ten seconds apart.

    The shape the brute-force rules describe: a single user/source pair, an
    unbroken run of failures, and enough of them inside one window to clear the
    configured count and rate thresholds.
    """
    return [
        build_event(
            f"brute-{index}",
            offset_seconds=10.0 * index,
            user="targeted-account",
            source="hostile-egress",
            device="unknown-device",
            session=f"burst-{index}",
            outcome="failure",
            failure_reason="invalid_credentials",
            application="corporate-vpn",
            response_time_ms=40,
            client_type="api_client",
        )
        for index in range(count)
    ]


def _spraying_window(count: int = 24) -> list[dict[str, Any]]:
    """One source trying a single failed attempt against many distinct accounts.

    The shape the spraying rule describes: high account fan-out from one source,
    rather than repeated attempts against any single account.
    """
    return [
        build_event(
            f"spray-{index}",
            offset_seconds=5.0 * index,
            user=f"sprayed-{index:03d}",
            source="spray-egress",
            device=f"rotating-{index % 3}",
            session=f"spray-session-{index}",
            outcome="failure",
            failure_reason="invalid_credentials",
            application="mail-gateway",
            response_time_ms=95,
            client_type="bot",
        )
        for index in range(count)
    ]


#: The templates the console offers, in the order it offers them.
SCENARIOS: Final[tuple[Scenario, ...]] = (
    Scenario(
        key="normal",
        label="Normal login activity",
        description=(
            "Twelve successful sign-ins from one account, one device and one "
            "source, ten minutes apart, each passing MFA."
        ),
        expectation=(
            "Nothing to raise. No rule describes well-spaced successes from a "
            "stable identity."
        ),
        builder=_normal_window,
    ),
    Scenario(
        key="brute_force",
        label="Brute-force-like failure burst",
        description=(
            "Thirty consecutive failed attempts against one account from one "
            "source, ten seconds apart, over five minutes."
        ),
        expectation=(
            "Depth against a single account: the brute-force family's count and "
            "rate conditions."
        ),
        builder=_brute_force_window,
    ),
    Scenario(
        key="spraying",
        label="Password-spraying-like fan-out",
        description=(
            "One source attempting a single failed sign-in against each of "
            "twenty-four distinct accounts, five seconds apart."
        ),
        expectation=(
            "Breadth rather than depth: no account sees enough attempts to look "
            "like a brute-force, and the source sees many accounts."
        ),
        builder=_spraying_window,
    ),
)


def scenario_events(key: str) -> list[dict[str, Any]]:
    """Return the window for the scenario named *key*.

    Raises:
        KeyError: when no template carries that key.
    """
    for scenario in SCENARIOS:
        if scenario.key == key:
            return scenario.events()
    raise KeyError(f"no scenario named {key!r}")


def _assert_the_templates_carry_no_credential_field() -> None:
    """Fail at import if a template ever grows a credential-shaped key.

    The API refuses these outright, so a template carrying one would be a
    template that cannot be submitted -- but the point is the earlier one: a
    demonstration fixture is exactly the kind of file somebody eventually pastes
    a real password into to "test the refusal".
    """
    for scenario in SCENARIOS:
        for item in scenario.events():
            offending = list(prohibited_field_names(item))
            if offending:
                raise ValueError(
                    f"scenario {scenario.key!r} carries prohibited field(s) "
                    f"{offending}; this package builds demonstration windows and "
                    f"never credential material"
                )


_assert_the_templates_carry_no_credential_field()


def _assert_every_address_is_reserved_for_documentation() -> None:
    """Fail at import if a literal address outside RFC 5737 appears here."""
    import ipaddress

    allowed = (
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
    )
    for text in DOCUMENTATION_ADDRESSES:
        address = ipaddress.ip_address(text)
        if not any(address in network for network in allowed):
            raise ValueError(
                f"{text} is not in a range RFC 5737 reserves for documentation; "
                f"a demonstration fixture must not name a routable host"
            )


_assert_every_address_is_reserved_for_documentation()


def scenario_summary(events: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Return the counts a console shows before a window is submitted."""
    outcomes: dict[str, int] = {}
    for item in events:
        outcome = str(item.get("authentication_outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "events": len(events),
        "distinct_users": len({str(item.get("user_id")) for item in events}),
        "distinct_sources": len(
            {str(item.get("source_id") or item.get("source_ip")) for item in events}
        ),
        **{f"outcome_{name}": count for name, count in sorted(outcomes.items())},
    }
