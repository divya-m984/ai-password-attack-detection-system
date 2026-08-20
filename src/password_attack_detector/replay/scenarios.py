"""The reviewed catalog of synthetic replay scenarios.

**Every scenario is a fabrication.**  Each one is a list of authentication events
describing activity that did not happen, involving accounts, devices and sources
that do not exist, replayed into a detector the operator is running themselves.
Nothing here contacts a login endpoint, tries a credential, carries one, or names
a routable host: where an address appears at all it comes from the ranges RFC
5737 and RFC 3849 reserve for documentation, and the identities are opaque labels
derived from made-up names.  Two import-time guards assert both properties, and
tests assert them again from outside.

**Every scenario is deterministic.**  The event identifiers are UUIDv5 values
over a fixed namespace, the timestamps are offsets from a fixed base instant, and
nothing is drawn from a clock or a random source.  :func:`scenario_fingerprint`
binds the schema version, the identifier, the revision and the canonical event
content into one digest, so "the same scenario" is a checkable claim rather than
a shared name.  That is what makes the pace-independence property meaningful:
:data:`~password_attack_detector.replay.enums.ReplayPace` changes when a step is
presented and can change nothing a detector reads.

**What a scenario claims about the detector is bounded by what a test proved.**
:attr:`Scenario.expected_rule_ids` lists the rules that actually fire when the
scenario is replayed through the frozen system, and
``tests/integration/test_replay_detection.py`` asserts the observed set equals
the declared one.  A rule the catalog would *like* to demonstrate but that does
not fire is named in :attr:`Scenario.limitations` instead, in the honest form:
what the rule needs, and why this deployment does not supply it.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from password_attack_detector.data.privacy import scan_prohibited_keys
from password_attack_detector.replay.enums import ScenarioId

__all__ = [
    "BASE_TIME",
    "DOCUMENTATION_ADDRESSES",
    "MAX_CATALOG_EVENTS",
    "SCENARIOS",
    "SCENARIO_SCHEMA_VERSION",
    "Scenario",
    "scenario",
    "scenario_fingerprint",
]

#: The replay catalog's own contract version.  Independent of the event schema
#: version and the API schema version: what a scenario *is* can change without
#: either of those changing, and a client comparing runs across releases needs to
#: know which of the three moved.
SCENARIO_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The ceiling on how many events any built-in scenario may carry.  The store
#: enforces its own limit too; this one keeps the catalog itself honest, because
#: a scenario is replayed one step at a time and each step re-scores the whole
#: prefix -- so event count is quadratic in work, not linear.
MAX_CATALOG_EVENTS: Final[int] = 64

#: Fixed base instant every scenario's offsets are measured from.  Deliberately
#: not "now": a scenario that moved with the clock would produce different
#: features on every run, and two replays of it could not be compared.
BASE_TIME: Final[datetime] = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)

#: Namespace for deterministic synthetic event identifiers.
_NS_REPLAY: Final[uuid.UUID] = uuid.UUID("6b2f9d14-0c7a-5e83-9a61-4d8e0f37b2c5")

#: Addresses reserved for documentation.  Present as a vocabulary the scenarios
#: *may* draw on; the built-in scenarios use pseudonymous source identifiers
#: instead, so no address reaches the wire at all.
DOCUMENTATION_ADDRESSES: Final[tuple[str, ...]] = (
    "192.0.2.10",
    "198.51.100.23",
    "203.0.113.47",
    "2001:db8::1",
)

#: Domain prefix each pseudonymous identifier field requires, matching the wire
#: contract's per-domain check.
_PREFIX: Final[dict[str, str]] = {
    "user": "u",
    "source": "s",
    "device": "d",
    "session": "sess",
}


def _pseudonym(domain: str, label: str) -> str:
    """Return a stable opaque identifier for a made-up entity.

    **Not the project's pseudonymization.**  That is a keyed HMAC over real
    identifiers and it never runs here, because there is no real identifier in
    this module to protect.  This is a deterministic label for a fictional
    entity, so ``"targeted-account"`` is the same non-existent account in every
    run on every machine.
    """
    digest = hashlib.sha256(f"pad-replay:{domain}:{label}".encode()).hexdigest()
    return f"{_PREFIX[domain]}:{digest[:32]}"


def _event(
    scenario_id: ScenarioId,
    key: str,
    *,
    offset_seconds: float,
    user: str,
    source: str,
    device: str,
    session: str,
    outcome: str,
    failure_reason: str | None = None,
    method: str = "password",
    application: str = "corporate-portal",
    country: str | None = "US",
    response_time_ms: int | None = 140,
    mfa_outcome: str | None = None,
    client_type: str | None = None,
    user_agent_family: str | None = None,
) -> dict[str, Any]:
    """Return one wire-shaped authentication event.

    The keys are exactly the ones
    :class:`~password_attack_detector.api.schemas.AuthEventRequest` declares, and
    the outcome/reason pairing follows the canonical schema's rule -- a failure
    names a reason, a success and a challenge name none.  A scenario that built
    an invalid window would be indistinguishable, to whoever is watching the
    demonstration, from a broken service.
    """
    body: dict[str, Any] = {
        "event_id": str(uuid.uuid5(_NS_REPLAY, f"{scenario_id.value}:{key}")),
        "event_time": (BASE_TIME + timedelta(seconds=offset_seconds)).isoformat(),
        "user_id": _pseudonym("user", user),
        "source_id": _pseudonym("source", source),
        "device_id": _pseudonym("device", device),
        "session_id": _pseudonym("session", session),
        "application_id": application,
        "authentication_method": method,
        "authentication_outcome": outcome,
    }
    if outcome == "failure":
        body["failure_reason"] = failure_reason or "invalid_credentials"
    elif outcome == "blocked":
        body["failure_reason"] = failure_reason or "ip_blocked"
    if country is not None:
        body["country_code"] = country
    if response_time_ms is not None:
        body["response_time_ms"] = response_time_ms
    if mfa_outcome is not None:
        body["mfa_outcome"] = mfa_outcome
    if client_type is not None:
        body["client_type"] = client_type
    if user_agent_family is not None:
        body["user_agent_family"] = user_agent_family
    return body


# ---------------------------------------------------------------------------
# The scenario record
# ---------------------------------------------------------------------------


class Scenario:
    """One synthetic scenario: its events, and what it is honestly good for.

    Immutable by construction: the events are built once and handed out as a
    fresh list per call, so a caller that mutates what it received cannot change
    what the next caller sees.
    """

    __slots__ = (
        "_events",
        "demonstrates_hybrid",
        "demonstrates_ml",
        "description",
        "expected_rule_families",
        "expected_rule_ids",
        "expected_severity_at_least",
        "limitations",
        "name",
        "purpose",
        "revision",
        "scenario_id",
    )

    def __init__(
        self,
        scenario_id: ScenarioId,
        *,
        name: str,
        description: str,
        purpose: str,
        revision: int,
        events: Sequence[dict[str, Any]],
        expected_rule_ids: Sequence[str] = (),
        expected_rule_families: Sequence[str] = (),
        expected_severity_at_least: str | None = None,
        demonstrates_ml: bool = True,
        demonstrates_hybrid: bool = True,
        limitations: str = "",
    ) -> None:
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        #: What an analyst is meant to learn from watching it run.
        self.purpose = purpose
        #: Bumped whenever the event content changes. Part of the fingerprint, so
        #: two runs of "brute_force" from different releases are distinguishable
        #: rather than silently comparable.
        self.revision = revision
        self._events = tuple(dict(item) for item in events)
        #: Rules a test proved fire somewhere in this scenario's timeline, under
        #: the repository's demo rule configuration.
        self.expected_rule_ids = tuple(expected_rule_ids)
        self.expected_rule_families = tuple(expected_rule_families)
        #: The weakest severity the run's worst step is guaranteed to reach.
        #: ``None`` where the scenario makes no severity claim.
        self.expected_severity_at_least = expected_severity_at_least
        #: Whether the frozen model layer produces a verdict on this scenario.
        #: A verdict, not a *positive* verdict: the model's answer is whatever
        #: the frozen champion returns, and the catalog does not predict it.
        self.demonstrates_ml = demonstrates_ml
        #: Whether the frozen fusion strategy produces a fused verdict.
        self.demonstrates_hybrid = demonstrates_hybrid
        #: What this scenario would demonstrate and cannot, stated plainly.
        self.limitations = limitations

    @property
    def event_count(self) -> int:
        """Return how many events the scenario emits."""
        return len(self._events)

    @property
    def duration_seconds(self) -> float:
        """Return the simulated span from the first event to the last.

        Simulated, not wall-clock: this is a property of the scenario's own
        timeline and is unaffected by the pace it is replayed at.
        """
        if not self._events:  # pragma: no cover - no empty scenario is admitted
            return 0.0
        times = [
            datetime.fromisoformat(str(item["event_time"])) for item in self._events
        ]
        return (max(times) - min(times)).total_seconds()

    def events(self) -> list[dict[str, Any]]:
        """Return a fresh copy of this scenario's events, in emission order."""
        return [dict(item) for item in self._events]

    def fingerprint(self) -> str:
        """Return the content identity of this scenario."""
        return scenario_fingerprint(self)


def scenario_fingerprint(item: Scenario) -> str:
    """Return a SHA-256 digest binding a scenario's identity to its content.

    Covers the catalog schema version, the identifier, the revision and every
    event, serialised canonically.  Two processes that agree on this digest are
    replaying the same events in the same order with the same timestamps -- which
    is the precondition for comparing their verdicts at all.
    """
    payload = json.dumps(
        {
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_id": item.scenario_id.value,
            "revision": item.revision,
            "events": item.events(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------

#: Irregular gaps, in seconds, between one analyst's ordinary sign-ins. Not a
#: uniform cadence: real people are not metronomes, and a perfectly regular
#: sequence of successes is a *bot* signature rather than a normal one.
_NORMAL_GAPS: Final[tuple[int, ...]] = (0, 28, 15, 41, 19, 34, 11, 47, 23, 17, 38, 22)


def _normal_activity() -> list[dict[str, Any]]:
    """One analyst signing in repeatedly over a few minutes, always succeeding."""
    offset = 0.0
    built: list[dict[str, Any]] = []
    for index, gap in enumerate(_NORMAL_GAPS):
        offset += gap
        built.append(
            _event(
                ScenarioId.NORMAL_ACTIVITY,
                f"normal-{index:02d}",
                offset_seconds=offset,
                user="regular-analyst",
                source="office-egress",
                device="issued-laptop",
                session=f"workday-{index:02d}",
                outcome="success",
                application="internal-portal",
                response_time_ms=190 + index * 3,
                mfa_outcome="passed",
                client_type="web_browser",
                user_agent_family="pad-browser",
            )
        )
    return built


def _brute_force() -> list[dict[str, Any]]:
    """One source failing against one account, every ten seconds, for five minutes."""
    return [
        _event(
            ScenarioId.BRUTE_FORCE,
            f"brute-{index:02d}",
            offset_seconds=10.0 * index,
            user="targeted-account",
            source="hostile-egress",
            device="unknown-host",
            session=f"burst-{index:02d}",
            outcome="failure",
            failure_reason="invalid_credentials",
            application="corporate-vpn",
            response_time_ms=40,
            client_type="api_client",
        )
        for index in range(30)
    ]


def _password_spraying() -> list[dict[str, Any]]:
    """One source, one failed attempt each against twenty-four distinct accounts."""
    return [
        _event(
            ScenarioId.PASSWORD_SPRAYING,
            f"spray-{index:02d}",
            offset_seconds=5.0 * index,
            user=f"sprayed-{index:03d}",
            source="spray-egress",
            device=f"spray-host-{index % 2}",
            session=f"spray-{index:02d}",
            outcome="failure",
            failure_reason="invalid_credentials",
            application="mail-gateway",
            response_time_ms=95,
            client_type="bot",
        )
        for index in range(24)
    ]


#: The client identities the stuffing scenario rotates through.  Client diversity
#: is the signature that separates stuffing from spraying: a spraying source
#: looks like one tool, a stuffing source looks like many browsers.
_STUFFING_CLIENTS: Final[tuple[tuple[str, str], ...]] = (
    ("pad-browser-a", "web_browser"),
    ("pad-browser-b", "web_browser"),
    ("pad-browser-c", "mobile_app"),
    ("pad-browser-d", "desktop_app"),
)

#: Positions in the stuffing timeline where a reused credential happens to work.
#: The mixed-outcome signature: mostly failures, occasionally not.
_STUFFING_SUCCESSES: Final[frozenset[int]] = frozenset({6, 13, 19})


def _credential_stuffing() -> list[dict[str, Any]]:
    """Broad account fan-out from one source, varied clients, mixed outcomes.

    No credential value, password list, or reuse dataset appears anywhere: the
    scenario models the *observable shape* of the behaviour -- many accounts, a
    few of them succeeding, from clients that keep changing -- which is all the
    detection system reads.
    """
    built: list[dict[str, Any]] = []
    for index in range(21):
        agent, client = _STUFFING_CLIENTS[index % len(_STUFFING_CLIENTS)]
        succeeded = index in _STUFFING_SUCCESSES
        built.append(
            _event(
                ScenarioId.CREDENTIAL_STUFFING,
                f"stuff-{index:02d}",
                offset_seconds=14.0 * index,
                user=f"reused-{index % 18:03d}",
                source="stuffing-egress",
                device=f"stuffing-host-{index % 4}",
                session=f"stuffing-{index:02d}",
                outcome="success" if succeeded else "failure",
                failure_reason=None if succeeded else "invalid_credentials",
                application="customer-portal",
                country="DE" if index % 3 else "US",
                response_time_ms=120 + (index % 5) * 11,
                client_type=client,
                user_agent_family=agent,
            )
        )
    return built


def _account_takeover() -> list[dict[str, Any]]:
    """Ordinary sign-ins, then a failure burst, then a success from a new context."""
    built: list[dict[str, Any]] = [
        _event(
            ScenarioId.ACCOUNT_TAKEOVER,
            f"ato-baseline-{index}",
            offset_seconds=20.0 * index,
            user="compromised-operator",
            source="office-egress",
            device="issued-laptop",
            session=f"ato-normal-{index}",
            outcome="success",
            application="finance-console",
            response_time_ms=205,
            mfa_outcome="passed",
            client_type="web_browser",
            user_agent_family="pad-browser",
        )
        for index in range(3)
    ]
    built += [
        _event(
            ScenarioId.ACCOUNT_TAKEOVER,
            f"ato-burst-{index:02d}",
            offset_seconds=60.0 + 10.0 * index,
            user="compromised-operator",
            source="takeover-egress",
            device="unknown-host",
            session=f"ato-burst-{index:02d}",
            outcome="failure",
            failure_reason="invalid_credentials",
            application="finance-console",
            response_time_ms=38,
            client_type="api_client",
        )
        for index in range(10)
    ]
    built.append(
        _event(
            ScenarioId.ACCOUNT_TAKEOVER,
            "ato-success",
            offset_seconds=160.0,
            user="compromised-operator",
            source="takeover-egress",
            device="unknown-host",
            session="ato-success",
            outcome="success",
            application="finance-console",
            country="BR",
            response_time_ms=44,
            mfa_outcome="bypassed",
            client_type="api_client",
        )
    )
    return built


def _bot_activity() -> list[dict[str, Any]]:
    """A service identity authenticating on a metronome, mostly successfully.

    Deliberately *not* an attack shape: the accounts are few, the outcomes are
    mostly successes, and what stands out is the cadence and the single unchanging
    client.  It is the scenario that shows the automation rule firing on its own,
    with no brute-force or spraying rule beside it.
    """
    return [
        _event(
            ScenarioId.BOT_ACTIVITY,
            f"bot-{index:02d}",
            offset_seconds=12.0 * index,
            user=f"service-account-{index % 5}",
            source="automation-egress",
            device="automation-host",
            session=f"automation-{index:02d}",
            outcome="failure" if index % 6 == 5 else "success",
            failure_reason="invalid_credentials" if index % 6 == 5 else None,
            application="batch-gateway",
            response_time_ms=60,
            client_type="api_client",
            user_agent_family="pad-agent",
        )
        for index in range(24)
    ]


def _mixed_attack() -> list[dict[str, Any]]:
    """Spraying, then a concentrated burst, then the success that followed it."""
    built: list[dict[str, Any]] = [
        _event(
            ScenarioId.MIXED_ATTACK,
            f"mixed-spray-{index:02d}",
            offset_seconds=4.0 * index,
            user=f"mixed-sprayed-{index:03d}",
            source="mixed-spray-egress",
            device=f"mixed-spray-host-{index % 2}",
            session=f"mixed-spray-{index:02d}",
            outcome="failure",
            failure_reason="invalid_credentials",
            application="mail-gateway",
            response_time_ms=88,
            client_type="bot",
        )
        for index in range(16)
    ]
    built += [
        _event(
            ScenarioId.MIXED_ATTACK,
            f"mixed-brute-{index:02d}",
            offset_seconds=80.0 + 10.0 * index,
            user="mixed-target",
            source="mixed-brute-egress",
            device="mixed-brute-host",
            session=f"mixed-brute-{index:02d}",
            outcome="failure",
            failure_reason="invalid_credentials",
            application="corporate-vpn",
            response_time_ms=42,
            client_type="api_client",
        )
        for index in range(12)
    ]
    built.append(
        _event(
            ScenarioId.MIXED_ATTACK,
            "mixed-success",
            offset_seconds=200.0,
            user="mixed-target",
            source="mixed-brute-egress",
            device="mixed-new-host",
            session="mixed-success",
            outcome="success",
            application="corporate-vpn",
            country="JP",
            response_time_ms=51,
            mfa_outcome="bypassed",
            client_type="api_client",
        )
    )
    return built


#: The stated reason the two baseline-dependent rules cannot fire on a live
#: request.  Written once, because it is one fact about the serving path rather
#: than a property of either scenario.
_NO_BASELINE: Final[str] = (
    "PAD-CS-001 and PAD-ATO-001 both gate on a fitted behavioural baseline "
    "(user_in_baseline, and the is_new_*_for_user novelty flags). The serving "
    "path computes point-in-time features from the supplied window alone and "
    "loads no baseline artifact, so those rules report insufficient data on "
    "every live request rather than firing. What this scenario demonstrates is "
    "therefore the behavioural shape, the rules that read the window directly, "
    "and the frozen model and hybrid layers."
)

#: The catalog, in the order it is offered.
SCENARIOS: Final[tuple[Scenario, ...]] = (
    Scenario(
        ScenarioId.NORMAL_ACTIVITY,
        name="Normal login activity",
        description=(
            "Twelve successful sign-ins by one analyst from one device and one "
            "source over about five minutes, at irregular human intervals, each "
            "passing multi-factor authentication."
        ),
        purpose=(
            "The control. Shows that ordinary behaviour produces no rule "
            "evidence, so a flagged timeline elsewhere means something."
        ),
        revision=1,
        events=_normal_activity(),
        expected_rule_ids=(),
        expected_rule_families=(),
        expected_severity_at_least=None,
        limitations=(
            "Demonstrates an absence. A run that flags nothing is the expected "
            "outcome, not a failed replay."
        ),
    ),
    Scenario(
        ScenarioId.BRUTE_FORCE,
        name="Concentrated brute force",
        description=(
            "Thirty consecutive failed attempts against one account from one "
            "source, ten seconds apart, over five minutes."
        ),
        purpose=(
            "Depth against a single account: the count, rate and consecutive-run "
            "conditions of the brute-force family, plus the machine-cadence "
            "automation indicator that a fixed ten-second gap also satisfies."
        ),
        revision=1,
        events=_brute_force(),
        expected_rule_ids=("PAD-BF-001", "PAD-BOT-001"),
        expected_rule_families=("automation", "brute_force"),
        expected_severity_at_least="high",
    ),
    Scenario(
        ScenarioId.PASSWORD_SPRAYING,
        name="Password spraying",
        description=(
            "One source attempting a single failed sign-in against each of "
            "twenty-four distinct accounts, five seconds apart."
        ),
        purpose=(
            "Breadth rather than depth: no account sees enough attempts to look "
            "like brute force, and the source touches enough accounts to clear "
            "the spraying fan-out condition."
        ),
        revision=1,
        events=_password_spraying(),
        expected_rule_ids=("PAD-BOT-001", "PAD-PS-001"),
        expected_rule_families=("automation", "spraying"),
        expected_severity_at_least="high",
    ),
    Scenario(
        ScenarioId.CREDENTIAL_STUFFING,
        name="Credential stuffing",
        description=(
            "One source touching eighteen distinct accounts in twenty-one "
            "attempts over five minutes, from four rotating clients, with three "
            "of the attempts succeeding."
        ),
        purpose=(
            "The mixed-outcome, many-client, many-account shape. Contrast it "
            "with the spraying scenario: the same fan-out condition fires, and "
            "the automation indicator does not, because the client keeps "
            "changing."
        ),
        revision=1,
        events=_credential_stuffing(),
        expected_rule_ids=("PAD-PS-001",),
        expected_rule_families=("spraying",),
        expected_severity_at_least="medium",
        limitations=_NO_BASELINE,
    ),
    Scenario(
        ScenarioId.ACCOUNT_TAKEOVER,
        name="Account-takeover indicator",
        description=(
            "Three ordinary sign-ins for one account, then ten failed attempts "
            "from a new source and device, then a success from that same new "
            "context in a different country with multi-factor bypassed."
        ),
        purpose=(
            "The sequence an analyst actually triages: the burst, and then the "
            "success that followed it. The successful-authentication-after-"
            "failure-burst rule is what marks the transition."
        ),
        revision=1,
        events=_account_takeover(),
        expected_rule_ids=("PAD-BF-001", "PAD-BF-002"),
        expected_rule_families=("brute_force",),
        expected_severity_at_least="medium",
        limitations=_NO_BASELINE,
    ),
    Scenario(
        ScenarioId.BOT_ACTIVITY,
        name="Automated client activity",
        description=(
            "A service identity authenticating every twelve seconds across five "
            "accounts from one host and one client family, mostly succeeding."
        ),
        purpose=(
            "The automation indicator in isolation. Too few accounts for "
            "spraying and too few failures for brute force, so what remains is "
            "the cadence and the unchanging client."
        ),
        revision=1,
        events=_bot_activity(),
        expected_rule_ids=("PAD-BOT-001",),
        expected_rule_families=("automation",),
        expected_severity_at_least="low",
    ),
    Scenario(
        ScenarioId.MIXED_ATTACK,
        name="Mixed attack timeline",
        description=(
            "Sixteen sprayed accounts from one source, then twelve concentrated "
            "failures against one account from a second source, then a success "
            "for that account from a new device in a different country."
        ),
        purpose=(
            "A timeline with more than one thing in it, which is what an analyst "
            "console has to stay readable under: different rules fire on "
            "different steps, and the severity moves as the run progresses."
        ),
        revision=1,
        events=_mixed_attack(),
        expected_rule_ids=("PAD-BF-001", "PAD-BF-002", "PAD-PS-001"),
        expected_rule_families=("brute_force", "spraying"),
        expected_severity_at_least="medium",
    ),
)

#: The catalog keyed by identifier, built once.
_BY_ID: Final[dict[ScenarioId, Scenario]] = {
    item.scenario_id: item for item in SCENARIOS
}


def scenario(scenario_id: str) -> Scenario | None:
    """Return the scenario named *scenario_id*, or ``None`` when there is none.

    Returns rather than raises: an unknown identifier is an ordinary client
    mistake that the API layer turns into a stable error code, not an exceptional
    condition inside the catalog.
    """
    try:
        key = ScenarioId(scenario_id)
    except ValueError:
        return None
    return _BY_ID.get(key)


# ---------------------------------------------------------------------------
# Import-time guards
# ---------------------------------------------------------------------------


def _assert_the_catalog_is_complete_and_bounded() -> None:
    """Fail at import if a scenario is missing, duplicated, or oversized."""
    declared = [item.scenario_id for item in SCENARIOS]
    if len(set(declared)) != len(declared):
        raise ValueError("two scenarios declare the same identifier")
    missing = sorted(str(item) for item in ScenarioId if item not in set(declared))
    if missing:
        raise ValueError(f"scenario id(s) {missing} are declared but not built")
    for item in SCENARIOS:
        if not item.event_count:
            raise ValueError(f"scenario {item.scenario_id!s} carries no events")
        if item.event_count > MAX_CATALOG_EVENTS:
            raise ValueError(
                f"scenario {item.scenario_id!s} carries {item.event_count} events, "
                f"above the catalog ceiling of {MAX_CATALOG_EVENTS}"
            )


_assert_the_catalog_is_complete_and_bounded()


def _assert_no_scenario_carries_credential_material() -> None:
    """Fail at import if a scenario event ever grows a credential-shaped key.

    Checked with the project's *own* ingestion scanner rather than a local list,
    so a name added there is refused here without anything being kept in step by
    hand.  The API refuses such a request regardless -- the point is the earlier
    one: a demonstration fixture is exactly the kind of file somebody eventually
    pastes a real password into to "test the refusal".
    """
    for item in SCENARIOS:
        for event in item.events():
            offending = scan_prohibited_keys(event)
            if offending:
                raise ValueError(
                    f"scenario {item.scenario_id!s} carries prohibited field "
                    f"name(s) {sorted(offending)}; the replay catalog describes "
                    f"authentication behaviour and never credential material"
                )


_assert_no_scenario_carries_credential_material()


def _assert_every_literal_address_is_reserved_for_documentation() -> None:
    """Fail at import if a routable address appears anywhere in this module.

    Covers both the documentation vocabulary and the scenarios themselves: a
    scenario that named a real host would be a demonstration fixture pointing at
    somebody's infrastructure, whether or not anything ever dialled it.
    """
    reserved = (
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
        ipaddress.ip_network("2001:db8::/32"),
    )

    def check(text: str) -> None:
        address = ipaddress.ip_address(text)
        if not any(address in network for network in reserved):
            raise ValueError(
                f"{text} is not reserved for documentation; a replay scenario "
                f"must not name a routable host"
            )

    for text in DOCUMENTATION_ADDRESSES:
        check(text)
    for item in SCENARIOS:
        for event in item.events():
            raw = event.get("source_ip")
            if raw is not None:
                check(str(raw))


_assert_every_literal_address_is_reserved_for_documentation()
