"""The replay HTTP surface, against the genuinely frozen deployment.

Every assertion below runs against the same session-scoped fixture the detection
and attribution suites use: a real trained, selected, frozen champion with a
materialized stacked hybrid.  So a replay run here is not a simulation of a
demonstration -- it *is* one, scored by the artifacts a deployment would serve.

What this module checks is the **contract**: the codes, the cursor, the
lifecycle, the bounds.  What the detector actually said about each scenario is
``test_replay_detection.py``'s job.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from password_attack_detector.replay.enums import ScenarioId
from password_attack_detector.replay.scenarios import SCENARIOS, scenario

#: How long a test waits for a run to reach a terminal state. Generous: the
#: fixture's champion is real, and a thirty-step scenario is thirty scored
#: windows.
_TERMINAL_TIMEOUT = 180.0


def _await_terminal(
    client: Any, run_id: str, *, timeout: float = _TERMINAL_TIMEOUT
) -> Any:
    """Poll a run until it stops producing, and return its final document."""
    deadline = time.monotonic() + timeout
    document: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/demo/runs/{run_id}")
        assert response.status_code == 200, response.text
        document = response.json()
        if not document["more_expected"]:
            return document
        time.sleep(0.05)
    raise AssertionError(f"replay run {run_id} did not finish: {document.get('state')}")


def _start(client: Any, scenario_id: str, *, pace: str = "instant") -> dict[str, Any]:
    """Start one run and return the created document."""
    response = client.post(
        "/api/v1/demo/runs", json={"scenario_id": scenario_id, "pace": pace}
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


def test_the_catalog_endpoint_publishes_every_scenario(client: Any) -> None:
    """The whole admissible input surface, in one document."""
    response = client.get("/api/v1/demo/scenarios")
    assert response.status_code == 200
    body = response.json()
    assert body["scenario_count"] == len(SCENARIOS)
    assert {item["scenario_id"] for item in body["scenarios"]} == {
        str(member) for member in ScenarioId
    }


def test_each_published_scenario_carries_what_a_client_renders(client: Any) -> None:
    """Name, purpose, size, fingerprint, and its honest limitations."""
    body = client.get("/api/v1/demo/scenarios").json()
    for entry in body["scenarios"]:
        source = scenario(entry["scenario_id"])
        assert source is not None
        assert entry["name"] == source.name
        assert entry["event_count"] == source.event_count
        assert entry["scenario_fingerprint"] == source.fingerprint()
        assert entry["duration_seconds"] == pytest.approx(source.duration_seconds)
        assert entry["revision"] >= 1


def test_the_catalog_carries_no_event_body(client: Any) -> None:
    """A client picks a scenario by name and never receives its events."""
    payload = client.get("/api/v1/demo/scenarios").text
    for item in SCENARIOS:
        first = item.events()[0]
        assert str(first["event_id"]) not in payload
        assert str(first["user_id"]) not in payload
        assert str(first["session_id"]) not in payload


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------


def test_the_system_status_reports_replay_as_an_optional_facility(
    client: Any,
) -> None:
    """Available, not required, and not a detection layer."""
    body = client.get("/api/v1/system/status").json()
    assert body["replay_enabled"] is True
    assert body["replay_available"] is True
    assert body["replay_required"] is False
    assert body["replay_unavailable_reason"] is None
    assert body["replay_scenario_count"] == len(SCENARIOS)
    assert body["max_active_replay_runs"] >= 1


def test_readiness_reports_replay_without_depending_on_it(client: Any) -> None:
    """A detection service does not become unready because a demo facility did."""
    body = client.get("/ready").json()
    replay = next(item for item in body["components"] if item["component"] == "replay")
    assert replay["required"] is False
    assert replay["state"] == "ready"
    assert body["status"] == "ready"


def test_health_stays_cheap(client: Any) -> None:
    """Replay is reported on the status document, never on liveness."""
    body = client.get("/health").json()
    assert set(body) == {"status", "service", "version"}


# ---------------------------------------------------------------------------
# The run lifecycle
# ---------------------------------------------------------------------------


def test_a_run_completes_and_reports_its_own_progress(client: Any) -> None:
    """The ordinary path, end to end, through the real detector."""
    source = scenario("account_takeover")
    assert source is not None
    created = _start(client, "account_takeover")

    assert created["state"] == "running"
    assert created["event_count"] == source.event_count
    assert created["scenario_fingerprint"] == source.fingerprint()
    assert created["failure_reason"] is None

    final = _await_terminal(client, created["run_id"])
    assert final["state"] == "completed"
    assert final["emitted_count"] == source.event_count
    assert final["more_expected"] is False
    assert final["next_sequence"] == source.event_count
    assert final["summary"]["detection_count"] == source.event_count


def test_a_run_identifier_is_opaque(client: Any) -> None:
    """No host path, process identifier, port, user, or secret contributes to it."""
    created = _start(client, "account_takeover")
    run_id = created["run_id"]
    _await_terminal(client, run_id)

    assert run_id.startswith("run_")
    token = run_id.removeprefix("run_")
    assert len(token) == 32
    assert all(character in "0123456789abcdef" for character in token)


def test_two_runs_are_isolated(client: Any) -> None:
    """Concurrent demonstrations do not see one another's timelines."""
    first = _start(client, "account_takeover")
    second = _start(client, "normal_activity")
    _await_terminal(client, first["run_id"])
    _await_terminal(client, second["run_id"])

    left = client.get(f"/api/v1/demo/runs/{first['run_id']}/timeline").json()
    right = client.get(f"/api/v1/demo/runs/{second['run_id']}/timeline").json()
    assert {item["run_id"] for item in left["records"]} == {first["run_id"]}
    assert {item["run_id"] for item in right["records"]} == {second["run_id"]}
    assert {item["scenario_id"] for item in left["records"]} == {"account_takeover"}
    assert {item["scenario_id"] for item in right["records"]} == {"normal_activity"}


def test_the_listing_reports_this_process_and_its_bounds(client: Any) -> None:
    """Named for what it is: a bounded window over one process's memory."""
    created = _start(client, "account_takeover")
    _await_terminal(client, created["run_id"])

    body = client.get("/api/v1/demo/runs").json()
    assert created["run_id"] in {item["run_id"] for item in body["runs"]}
    assert body["max_active_runs"] >= 1
    assert body["max_retained_runs"] >= body["max_active_runs"]
    assert body["run_count"] == len(body["runs"])


# ---------------------------------------------------------------------------
# The cursor
# ---------------------------------------------------------------------------


def test_the_timeline_cursor_returns_only_new_records(client: Any) -> None:
    """The property that makes polling a long run affordable."""
    created = _start(client, "account_takeover")
    final = _await_terminal(client, created["run_id"])
    total = final["emitted_count"]

    first = client.get(
        f"/api/v1/demo/runs/{created['run_id']}/timeline", params={"limit": 5}
    ).json()
    assert first["after_sequence"] == 0
    assert first["record_count"] == 5
    assert [item["sequence"] for item in first["records"]] == [1, 2, 3, 4, 5]
    assert first["next_sequence"] == 5
    assert first["more_expected"] is True

    second = client.get(
        f"/api/v1/demo/runs/{created['run_id']}/timeline",
        params={"after_sequence": first["next_sequence"], "limit": 100},
    ).json()
    assert [item["sequence"] for item in second["records"]] == list(range(6, total + 1))
    assert second["more_expected"] is False


def test_a_cursor_at_the_end_returns_an_empty_page(client: Any) -> None:
    """A client that is up to date asks for nothing and gets it."""
    created = _start(client, "account_takeover")
    final = _await_terminal(client, created["run_id"])
    page = client.get(
        f"/api/v1/demo/runs/{created['run_id']}/timeline",
        params={"after_sequence": final["next_sequence"]},
    ).json()
    assert page["record_count"] == 0
    assert page["records"] == []
    assert page["next_sequence"] == final["next_sequence"]
    assert page["more_expected"] is False


def test_a_negative_cursor_is_refused_with_its_own_code(client: Any) -> None:
    """Not a position a cursor can occupy."""
    created = _start(client, "account_takeover")
    _await_terminal(client, created["run_id"])
    response = client.get(
        f"/api/v1/demo/runs/{created['run_id']}/timeline",
        params={"after_sequence": -1},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "API021"


@pytest.mark.parametrize("limit", [0, -1, 10_000])
def test_an_out_of_range_page_size_is_refused(client: Any, limit: int) -> None:
    """The page bound is enforced by the schema, before any handler runs."""
    created = _start(client, "account_takeover")
    _await_terminal(client, created["run_id"])
    response = client.get(
        f"/api/v1/demo/runs/{created['run_id']}/timeline", params={"limit": limit}
    )
    assert response.status_code == 422


def test_sequences_are_strictly_increasing_and_contiguous(client: Any) -> None:
    """A gap would mean a record was produced and never delivered."""
    created = _start(client, "account_takeover")
    final = _await_terminal(client, created["run_id"])
    page = client.get(
        f"/api/v1/demo/runs/{created['run_id']}/timeline", params={"limit": 100}
    ).json()
    assert [item["sequence"] for item in page["records"]] == list(
        range(1, final["emitted_count"] + 1)
    )
    assert [item["window_event_count"] for item in page["records"]] == list(
        range(1, final["emitted_count"] + 1)
    )


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------


def test_stopping_a_run_ends_it_and_emits_nothing_further(client: Any) -> None:
    """The acknowledgement is honest: the timeline is final when it returns."""
    created = _start(client, "brute_force", pace="slow")
    response = client.post(f"/api/v1/demo/runs/{created['run_id']}/stop")
    assert response.status_code == 200
    stopped = response.json()

    assert stopped["state"] == "stopped"
    assert stopped["more_expected"] is False
    emitted = stopped["emitted_count"]
    assert emitted < stopped["event_count"], "a slow run is stopped before it finishes"

    time.sleep(0.5)
    after = client.get(f"/api/v1/demo/runs/{created['run_id']}").json()
    assert after["emitted_count"] == emitted
    assert after["state"] == "stopped"


def test_stopping_is_idempotent(client: Any) -> None:
    """A second press reports the state the run is in and changes nothing."""
    created = _start(client, "account_takeover")
    _await_terminal(client, created["run_id"])
    first = client.post(f"/api/v1/demo/runs/{created['run_id']}/stop").json()
    second = client.post(f"/api/v1/demo/runs/{created['run_id']}/stop").json()
    assert first["state"] == second["state"]
    assert first["emitted_count"] == second["emitted_count"]


def test_stopping_one_run_does_not_disturb_another(client: Any) -> None:
    """The addressed run and no other."""
    slow = _start(client, "brute_force", pace="slow")
    quick = _start(client, "normal_activity")

    client.post(f"/api/v1/demo/runs/{slow['run_id']}/stop")
    final = _await_terminal(client, quick["run_id"])
    assert final["state"] == "completed"
    assert final["emitted_count"] == final["event_count"]


def test_a_stopped_run_cannot_be_restarted(client: Any) -> None:
    """A finished run stays finished; a second execution is a new run."""
    created = _start(client, "brute_force", pace="slow")
    client.post(f"/api/v1/demo/runs/{created['run_id']}/stop")
    again = _start(client, "brute_force", pace="instant")
    assert again["run_id"] != created["run_id"]
    assert again["state"] == "running"
    client.post(f"/api/v1/demo/runs/{again['run_id']}/stop")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_unknown_run_is_a_stable_not_found(client: Any) -> None:
    """Distinct from an unavailable subsystem: a different fault to fix."""
    for path in (
        "/api/v1/demo/runs/run_00000000000000000000000000000000",
        "/api/v1/demo/runs/run_00000000000000000000000000000000/timeline",
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "API018"
    response = client.post(
        "/api/v1/demo/runs/run_00000000000000000000000000000000/stop"
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "API018"


@pytest.mark.parametrize(
    "run_id",
    ["../../../etc/passwd", "run id", "run/../x", "%2e%2e%2f", "a" * 200, "run.id"],
)
def test_a_run_identifier_that_is_not_one_is_refused(client: Any, run_id: str) -> None:
    """A value that could read as a path segment never reaches a handler.

    The empty identifier is deliberately not in this list: ``/demo/runs/`` is
    the listing route with a trailing slash, and answering it is correct routing
    rather than a run lookup that leaked.
    """
    response = client.get(f"/api/v1/demo/runs/{run_id}")
    assert response.status_code in {404, 422}
    assert "error" in response.json()


def test_an_unknown_scenario_is_refused(client: Any) -> None:
    """Scenarios cannot be uploaded; the catalog is the whole admissible set."""
    response = client.post(
        "/api/v1/demo/runs", json={"scenario_id": "definitely_not_a_scenario"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "API001"


def test_an_unknown_pace_is_refused(client: Any) -> None:
    """A pace is a word from a four-word vocabulary, never a duration."""
    for pace in ("0.001", "immediately", "-1", "99999"):
        response = client.post(
            "/api/v1/demo/runs",
            json={"scenario_id": "normal_activity", "pace": pace},
        )
        assert response.status_code == 422, pace


def test_the_pace_defaults_rather_than_being_required(client: Any) -> None:
    """A caller who does not care gets the middle of the vocabulary."""
    response = client.post("/api/v1/demo/runs", json={"scenario_id": "normal_activity"})
    assert response.status_code == 201
    body = response.json()
    assert body["pace"] == "normal"
    client.post(f"/api/v1/demo/runs/{body['run_id']}/stop")


def test_every_new_error_code_is_documented(client: Any) -> None:
    """A code is a contract, so every one this milestone added carries a message."""
    from password_attack_detector.api.errors import ERROR_MESSAGES, ErrorCode

    for code in (
        ErrorCode.REPLAY_UNAVAILABLE,
        ErrorCode.REPLAY_SCENARIO_NOT_FOUND,
        ErrorCode.REPLAY_RUN_NOT_FOUND,
        ErrorCode.REPLAY_LIMIT_REACHED,
        ErrorCode.REPLAY_INVALID_TRANSITION,
        ErrorCode.REPLAY_CURSOR_INVALID,
    ):
        assert ERROR_MESSAGES[code]
        assert code.value.startswith("API")


def test_the_existing_error_codes_did_not_move(client: Any) -> None:
    """A published code never changes meaning; new failures take new members."""
    from password_attack_detector.api.errors import ErrorCode

    assert ErrorCode.MALFORMED_REQUEST.value == "API001"
    assert ErrorCode.CREDENTIAL_FIELD_REJECTED.value == "API013"
    assert ErrorCode.NOT_FOUND.value == "API015"
    assert ErrorCode.INTERNAL_ERROR.value == "API099"
