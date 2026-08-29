"""Tests for the reviewed scenario catalog.

Four properties, and the catalog is only useful if all four hold:

* **Deterministic.**  The same events, the same identifiers, the same timestamps,
  every time and on every machine -- otherwise "the same scenario produced the
  same verdicts" is not a checkable claim.
* **Valid.**  Every event is one the serving layer's own request schema accepts,
  because a scenario that built an invalid window would look, to whoever is
  watching the demonstration, exactly like a broken service.
* **Safe.**  No credential-shaped field name anywhere, no routable address
  anywhere, and no field the wire contract does not declare.
* **Bounded.**  Every scenario fits inside limits the store can actually serve.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from password_attack_detector.api.schemas import (
    AnchorSelection,
    DetectionWindowRequest,
)
from password_attack_detector.data.privacy import scan_prohibited_keys
from password_attack_detector.replay import scenarios as catalog
from password_attack_detector.replay.enums import ScenarioId

# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_every_scenario_id_has_exactly_one_catalog_entry() -> None:
    """The enumeration is the input surface, so it must be totally covered."""
    declared = [item.scenario_id for item in catalog.SCENARIOS]
    assert sorted(declared) == sorted(ScenarioId)
    assert len(set(declared)) == len(declared)


def test_the_catalog_covers_every_scenario_the_milestone_named() -> None:
    """The seven shapes the milestone asked for, by name."""
    assert {str(item) for item in ScenarioId} == {
        "normal_activity",
        "brute_force",
        "password_spraying",
        "credential_stuffing",
        "account_takeover",
        "bot_activity",
        "mixed_attack",
    }


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_a_scenario_is_bounded(scenario: catalog.Scenario) -> None:
    """Each step re-scores the whole prefix, so event count is quadratic in work."""
    assert 0 < scenario.event_count <= catalog.MAX_CATALOG_EVENTS


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_a_scenario_fits_inside_five_minutes(scenario: catalog.Scenario) -> None:
    """Deliberate: it makes the rule expectations robust to the window settings.

    Every windowed rule condition in the catalog reads a window of five minutes
    or more, so a scenario whose whole timeline fits in five minutes produces the
    same counts under the repository's demo rule configuration and under the
    rule catalog's own defaults. A scenario that spanned an hour would fire
    different rules under the two, and the published expectations would be true
    of only one deployment.
    """
    assert scenario.duration_seconds <= 300.0


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_a_scenario_declares_what_it_demonstrates(scenario: catalog.Scenario) -> None:
    """Metadata a client renders must actually be there."""
    assert scenario.name
    assert scenario.description
    assert scenario.purpose
    assert scenario.revision >= 1


def test_a_scenario_that_names_no_rule_says_why_in_its_metadata() -> None:
    """The control scenario demonstrates an absence, and says so."""
    normal = catalog.scenario("normal_activity")
    assert normal is not None
    assert normal.expected_rule_ids == ()
    assert "absence" in normal.limitations


def test_the_baseline_dependent_scenarios_state_their_limitation() -> None:
    """Stuffing and takeover both want a rule the serving path cannot run.

    PAD-CS-001 and PAD-ATO-001 gate on a fitted behavioural baseline, and the
    serving path loads none. Rather than quietly claiming those rules, the two
    scenarios say what they need and why this deployment does not supply it.
    """
    for name in ("credential_stuffing", "account_takeover"):
        found = catalog.scenario(name)
        assert found is not None
        assert "baseline" in found.limitations
        assert "PAD-CS-001" not in found.expected_rule_ids
        assert "PAD-ATO-001" not in found.expected_rule_ids


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_a_scenario_builds_identically_every_time(scenario: catalog.Scenario) -> None:
    """No clock, no random source, no dictionary ordering."""
    assert scenario.events() == scenario.events()
    assert scenario.fingerprint() == scenario.fingerprint()


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_a_scenario_hands_out_copies(scenario: catalog.Scenario) -> None:
    """A caller that mutates what it received cannot change the catalog."""
    first = scenario.events()
    first[0]["application_id"] = "mutated"
    first.clear()
    assert scenario.events()[0]["application_id"] != "mutated"
    assert len(scenario.events()) == scenario.event_count


def test_two_scenarios_never_share_a_fingerprint() -> None:
    """The fingerprint is a content identity, so distinct content is distinct."""
    prints = {item.fingerprint() for item in catalog.SCENARIOS}
    assert len(prints) == len(catalog.SCENARIOS)


def test_the_fingerprint_covers_the_revision() -> None:
    """A revision bump has to change the identity, or it records nothing."""
    original = catalog.SCENARIOS[1]
    bumped = catalog.Scenario(
        original.scenario_id,
        name=original.name,
        description=original.description,
        purpose=original.purpose,
        revision=original.revision + 1,
        events=original.events(),
    )
    assert bumped.fingerprint() != original.fingerprint()


def test_the_fingerprint_covers_the_events() -> None:
    """Two scenarios with the same identity and different events are different."""
    original = catalog.SCENARIOS[1]
    altered = original.events()
    altered[0]["application_id"] = "somewhere-else"
    changed = catalog.Scenario(
        original.scenario_id,
        name=original.name,
        description=original.description,
        purpose=original.purpose,
        revision=original.revision,
        events=altered,
    )
    assert changed.fingerprint() != original.fingerprint()


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_event_times_are_offsets_from_the_fixed_base(
    scenario: catalog.Scenario,
) -> None:
    """Nothing here moves with the wall clock."""
    times = [
        datetime.fromisoformat(str(item["event_time"])) for item in scenario.events()
    ]
    assert min(times) >= catalog.BASE_TIME
    assert times == sorted(times), "a window is supplied in non-decreasing order"


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_event_identifiers_are_unique_within_a_scenario(
    scenario: catalog.Scenario,
) -> None:
    """The window contract refuses a repeated identity, so the catalog must not."""
    ids = [item["event_id"] for item in scenario.events()]
    assert len(set(ids)) == len(ids)


def test_no_two_scenarios_share_an_event_identifier() -> None:
    """The scenario id seeds the UUID, so two catalogs' events never collide."""
    seen: set[str] = set()
    for scenario in catalog.SCENARIOS:
        ids = {str(item["event_id"]) for item in scenario.events()}
        assert not (ids & seen)
        seen |= ids


# ---------------------------------------------------------------------------
# Validity against the serving contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_every_scenario_builds_a_window_the_service_accepts(
    scenario: catalog.Scenario,
) -> None:
    """The whole scenario, through the *same* schema an HTTP body goes through.

    This is the check that makes a demonstration trustworthy: the replayed
    events are not a parallel format that happens to resemble a request, they
    are requests.
    """
    events = scenario.events()
    request = DetectionWindowRequest.model_validate(
        {
            "events": events,
            "anchor_selection": AnchorSelection.EXPLICIT.value,
            "anchor_event_ids": [events[-1]["event_id"]],
        }
    )
    assert len(request.events) == scenario.event_count
    assert request.resolved_anchor_ids() == (str(events[-1]["event_id"]),)


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_every_prefix_of_a_scenario_is_a_valid_window(
    scenario: catalog.Scenario,
) -> None:
    """A replay scores prefixes, so every prefix has to be admissible.

    Not just the whole scenario: the engine emits step by step, and step *n* is
    scored against the first *n* events. A scenario whose third prefix were
    invalid would fail a third of the way through a demonstration.
    """
    events = scenario.events()
    for index in range(len(events)):
        prefix = events[: index + 1]
        request = DetectionWindowRequest.model_validate(
            {
                "events": prefix,
                "anchor_selection": AnchorSelection.EXPLICIT.value,
                "anchor_event_ids": [prefix[-1]["event_id"]],
            }
        )
        assert len(request.events) == index + 1


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_every_event_becomes_a_canonical_authentication_event(
    scenario: catalog.Scenario,
) -> None:
    """The canonical schema's cross-field rules, not just the wire schema's.

    Outcome against failure reason, and the multi-factor consistency rule. A
    scenario that satisfied the request schema and failed the canonical one
    would be refused mid-replay with an event-validity code.
    """
    request = DetectionWindowRequest.model_validate(
        {"events": scenario.events(), "anchor_selection": "last"}
    )
    for event in request.events:
        assert event.source_id is not None
        canonical = event.to_canonical_event(source_id=event.source_id)
        assert canonical.event_id == event.event_id


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

#: Exactly the fields ``AuthEventRequest`` declares. The schema forbids extras,
#: so an invented key would be refused -- this asserts the catalog does not even
#: try.
_WIRE_FIELDS = {
    "event_id",
    "event_time",
    "user_id",
    "source_id",
    "source_ip",
    "device_id",
    "session_id",
    "application_id",
    "authentication_method",
    "authentication_outcome",
    "failure_reason",
    "mfa_outcome",
    "country_code",
    "region_code",
    "coarse_latitude",
    "coarse_longitude",
    "user_agent_family",
    "operating_system_family",
    "client_type",
    "response_time_ms",
}


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_no_scenario_event_carries_credential_material(
    scenario: catalog.Scenario,
) -> None:
    """Checked with the project's own scanner, not a local copy of its list."""
    for event in scenario.events():
        assert scan_prohibited_keys(event) == []


@pytest.mark.parametrize(
    "scenario", catalog.SCENARIOS, ids=lambda item: str(item.scenario_id)
)
def test_no_scenario_event_carries_an_undeclared_field(
    scenario: catalog.Scenario,
) -> None:
    """A demonstration fixture must not offer a field the contract refuses."""
    for event in scenario.events():
        assert set(event) <= _WIRE_FIELDS, set(event) - _WIRE_FIELDS


def test_no_scenario_names_an_address_at_all() -> None:
    """The built-in scenarios use pseudonymous sources, so none reaches the wire.

    The address vocabulary exists and is reserved for documentation; the
    scenarios simply do not need it, which is the stronger position.
    """
    for scenario in catalog.SCENARIOS:
        for event in scenario.events():
            assert "source_ip" not in event
            assert str(event["source_id"]).startswith("s:")


def test_the_documentation_addresses_are_all_reserved() -> None:
    """RFC 5737 and RFC 3849 ranges only."""
    import ipaddress

    reserved = (
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
        ipaddress.ip_network("2001:db8::/32"),
    )
    for text in catalog.DOCUMENTATION_ADDRESSES:
        address = ipaddress.ip_address(text)
        assert any(address in network for network in reserved), text


def test_a_pseudonym_is_a_stable_opaque_label() -> None:
    """Not the project's keyed pseudonymization -- there is no real identity here."""
    first = catalog._pseudonym("user", "somebody")
    assert first == catalog._pseudonym("user", "somebody")
    assert first != catalog._pseudonym("user", "somebody-else")
    assert first.startswith("u:")
    assert "somebody" not in first


# ---------------------------------------------------------------------------
# The import-time guards
# ---------------------------------------------------------------------------


def test_the_credential_guard_fires_on_a_scenario_that_carries_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard nobody has seen fire is a guard nobody knows works."""
    offending = catalog.Scenario(
        ScenarioId.NORMAL_ACTIVITY,
        name="x",
        description="x",
        purpose="x",
        revision=1,
        events=[{"event_id": "1", "password": "hunter2"}],
    )
    monkeypatch.setattr(catalog, "SCENARIOS", (offending,))
    with pytest.raises(ValueError, match="prohibited field"):
        catalog._assert_no_scenario_carries_credential_material()


def test_the_address_guard_fires_on_a_routable_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A demonstration fixture must not name somebody's infrastructure."""
    monkeypatch.setattr(catalog, "DOCUMENTATION_ADDRESSES", ("198.51.101.1",))
    with pytest.raises(ValueError, match="reserved for documentation"):
        catalog._assert_every_literal_address_is_reserved_for_documentation()


def test_the_bounds_guard_fires_on_an_oversized_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalog ceiling is enforced at import, not merely documented."""
    big = catalog.Scenario(
        ScenarioId.NORMAL_ACTIVITY,
        name="x",
        description="x",
        purpose="x",
        revision=1,
        events=[{"event_id": str(index)} for index in range(500)],
    )
    # A *complete* catalog with one oversized member, so the bound is what fires
    # rather than the completeness check that runs before it.
    others = tuple(
        item
        for item in catalog.SCENARIOS
        if item.scenario_id is not ScenarioId.NORMAL_ACTIVITY
    )
    monkeypatch.setattr(catalog, "SCENARIOS", (big, *others))
    with pytest.raises(ValueError, match="above the catalog ceiling"):
        catalog._assert_the_catalog_is_complete_and_bounded()


def test_the_completeness_guard_fires_on_a_missing_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enumeration member with no entry would be a 404 nobody could fix."""
    monkeypatch.setattr(catalog, "SCENARIOS", (catalog.SCENARIOS[0],))
    with pytest.raises(ValueError, match="declared but not built"):
        catalog._assert_the_catalog_is_complete_and_bounded()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_a_known_scenario_is_found_by_its_identifier() -> None:
    """The lookup the service performs on every start request."""
    found = catalog.scenario("brute_force")
    assert found is not None
    assert found.scenario_id is ScenarioId.BRUTE_FORCE


@pytest.mark.parametrize(
    "name",
    ["", "unknown", "BRUTE_FORCE", "../../etc/passwd", "http://example.invalid"],
)
def test_an_unknown_scenario_returns_none_rather_than_raising(name: str) -> None:
    """An unknown identifier is an ordinary client mistake, not an exception."""
    assert catalog.scenario(name) is None


def test_the_fingerprint_is_stable_across_a_json_round_trip() -> None:
    """The digest is over canonical JSON, so serialising cannot perturb it."""
    scenario = catalog.SCENARIOS[2]
    reloaded = catalog.Scenario(
        scenario.scenario_id,
        name=scenario.name,
        description=scenario.description,
        purpose=scenario.purpose,
        revision=scenario.revision,
        events=json.loads(json.dumps(scenario.events())),
    )
    assert reloaded.fingerprint() == scenario.fingerprint()
