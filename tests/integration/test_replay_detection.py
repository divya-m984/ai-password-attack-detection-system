"""What the frozen system actually says about each built-in scenario.

This is the module that turns the catalog's published expectations from claims
into facts.  Every scenario is replayed end to end through the real deployment --
the trained, selected, frozen champion with its materialized **stacked** hybrid --
and what the rules, the model and the fusion produced is asserted against what
``GET /api/v1/demo/scenarios`` says they will.

Two disciplines the milestone asked for, and both are load-bearing here:

* **Nothing is claimed that is not proved.**  ``expected_rule_ids`` is asserted as
  an *equality*, not a subset: a scenario cannot quietly over-claim, and it
  cannot quietly acquire a rule nobody documented either.
* **Nothing is tuned to make the demonstration dramatic.**  Where a scenario does
  not produce the rule its name suggests, that is recorded here as the finding it
  is -- see :func:`test_the_baseline_dependent_rules_never_fire_on_a_live_request`
  -- rather than fixed by moving a threshold. The scientific artifacts are frozen
  and this milestone did not touch one.

Every scenario is replayed once, in a module-scoped fixture: seven runs of a real
champion is the expensive part, and thirty assertions over cached timelines is
not.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from password_attack_detector.api.app import create_app
from password_attack_detector.api.config import APISettings
from password_attack_detector.api.services import build_runtime
from password_attack_detector.detection.enums import SEVERITY_ORDER, Severity
from password_attack_detector.replay.scenarios import SCENARIOS, Scenario

_TERMINAL_TIMEOUT = 300.0


@pytest.fixture(scope="module")
def demo(served: APISettings) -> Iterator[TestClient]:
    """One application, started once, for every replay in this module."""
    with TestClient(
        create_app(settings=served, runtime=build_runtime(served))
    ) as ready:
        yield ready


def _replay(client: TestClient, scenario_id: str, *, pace: str = "instant") -> Any:
    """Run one scenario to completion and return ``(run, records)``."""
    created = client.post(
        "/api/v1/demo/runs", json={"scenario_id": scenario_id, "pace": pace}
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["run_id"]

    deadline = time.monotonic() + _TERMINAL_TIMEOUT
    document: dict[str, Any] = {}
    while time.monotonic() < deadline:
        document = client.get(f"/api/v1/demo/runs/{run_id}").json()
        if not document["more_expected"]:
            break
        time.sleep(0.05)
    else:  # pragma: no cover - only on a pathologically slow machine
        raise AssertionError(f"{scenario_id} did not finish")

    records: list[dict[str, Any]] = []
    cursor = 0
    while True:
        page = client.get(
            f"/api/v1/demo/runs/{run_id}/timeline",
            params={"after_sequence": cursor, "limit": 100},
        ).json()
        records.extend(page["records"])
        if not page["records"]:
            break
        cursor = page["next_sequence"]
        if not page["more_expected"]:
            break
    return (document, records)


@pytest.fixture(scope="module")
def replays(demo: TestClient) -> dict[str, Any]:
    """Replay every catalog scenario once, and cache what came back."""
    return {
        str(item.scenario_id): _replay(demo, str(item.scenario_id))
        for item in SCENARIOS
    }


def _fired(records: list[dict[str, Any]]) -> set[str]:
    """Return every public rule identifier that fired anywhere in a run."""
    return {
        rule_id
        for record in records
        for rule_id in record["detection"]["rule"]["fired_rule_ids"]
    }


def _worst(records: list[dict[str, Any]]) -> Severity:
    """Return the worst severity any step reached, on the Phase 4 ordinal scale."""
    return max(
        (Severity(record["detection"]["severity"]) for record in records),
        key=lambda item: SEVERITY_ORDER[item],
    )


# ---------------------------------------------------------------------------
# Every scenario, against its own published expectation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: str(item.scenario_id))
def test_a_scenario_fires_exactly_the_rules_it_claims(
    replays: dict[str, Any], scenario: Scenario
) -> None:
    """Equality, not containment.

    A subset assertion would let a scenario silently acquire a rule nobody
    documented, and a superset one would let it claim a rule that never fires.
    The catalog's published expectation is the run's actual result or the test
    fails.
    """
    _, records = replays[str(scenario.scenario_id)]
    assert _fired(records) == set(scenario.expected_rule_ids)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: str(item.scenario_id))
def test_a_scenario_reaches_the_severity_it_claims(
    replays: dict[str, Any], scenario: Scenario
) -> None:
    """ "At least this severe": a floor, so a stronger result is not a failure."""
    _, records = replays[str(scenario.scenario_id)]
    if scenario.expected_severity_at_least is None:
        return
    floor = Severity(scenario.expected_severity_at_least)
    assert SEVERITY_ORDER[_worst(records)] >= SEVERITY_ORDER[floor]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: str(item.scenario_id))
def test_a_scenario_emits_every_event_it_declares(
    replays: dict[str, Any], scenario: Scenario
) -> None:
    """One scored step per event, and one record per step."""
    run, records = replays[str(scenario.scenario_id)]
    assert run["state"] == "completed"
    assert run["emitted_count"] == scenario.event_count
    assert len(records) == scenario.event_count


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: str(item.scenario_id))
def test_every_step_carries_all_three_layers(
    replays: dict[str, Any], scenario: Scenario
) -> None:
    """The frozen champion and the frozen hybrid answer on every window."""
    _, records = replays[str(scenario.scenario_id)]
    for record in records:
        detection = record["detection"]
        assert detection["ml"]["available"] is True, "the frozen champion is loaded"
        assert detection["ml"]["flagged"] is not None
        assert detection["hybrid"]["available"] is True
        assert detection["hybrid"]["flagged"] is not None


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: str(item.scenario_id))
def test_every_step_is_fused_by_the_frozen_stacked_hybrid(
    replays: dict[str, Any], scenario: Scenario
) -> None:
    """The real STACKED path, with no fallback anywhere in it.

    ``stacked`` is the interesting case: it is the only strategy that needs a
    fitted artifact, and it runs only from a serving bundle whose reconstruction
    recomputed the fingerprint Phase 5 sealed. A replay that fell back to a gate
    would produce a perfectly plausible timeline of a hybrid nobody selected.
    """
    _, records = replays[str(scenario.scenario_id)]
    strategies = {record["detection"]["hybrid"]["strategy"] for record in records}
    assert strategies == {"stacked"}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: str(item.scenario_id))
def test_the_run_summary_agrees_with_its_own_records(
    replays: dict[str, Any], scenario: Scenario
) -> None:
    """Every figure in the summary is derived from the timeline, not beside it."""
    run, records = replays[str(scenario.scenario_id)]
    summary = run["summary"]
    assert summary["detection_count"] == len(records)
    assert summary["rule_flagged_count"] == sum(
        1 for item in records if item["detection"]["rule"]["flagged"]
    )
    assert summary["ml_flagged_count"] == sum(
        1 for item in records if item["detection"]["ml"]["flagged"]
    )
    assert summary["hybrid_flagged_count"] == sum(
        1 for item in records if item["detection"]["hybrid"]["flagged"]
    )
    assert summary["highest_severity"] == str(_worst(records))
    assert set(summary["triggered_rule_counts"]) == _fired(records)
    assert summary["fusion_strategies"] == ["stacked"]


# ---------------------------------------------------------------------------
# The individual scenarios, named
# ---------------------------------------------------------------------------


def test_normal_activity_produces_no_attack_demonstration(
    replays: dict[str, Any],
) -> None:
    """The control. Ordinary behaviour must not look like an incident."""
    run, records = replays["normal_activity"]
    assert _fired(records) == set()
    assert _worst(records) is Severity.LOW
    assert run["summary"]["rule_flagged_count"] == 0
    assert all(record["detection"]["rule"]["risk_score"] == 0.0 for record in records)


def test_brute_force_produces_rule_evidence_that_grows_with_the_burst(
    replays: dict[str, Any],
) -> None:
    """Depth against one account, and evidence an analyst can read."""
    _, records = replays["brute_force"]
    assert "PAD-BF-001" in _fired(records)
    assert _worst(records) is Severity.CRITICAL
    flagged = [item for item in records if item["detection"]["rule"]["flagged"]]
    assert flagged, "the burst is long enough to clear the rule's thresholds"
    assert flagged[-1]["detection"]["rule"]["evidence"], "sanitized behavioral evidence"
    scores = [item["detection"]["rule"]["risk_score"] for item in flagged]
    assert scores[-1] >= scores[0], "a longer burst is not scored lower"


def test_password_spraying_produces_the_spraying_rule(
    replays: dict[str, Any],
) -> None:
    """Breadth rather than depth: the fan-out condition, not the burst one."""
    _, records = replays["password_spraying"]
    fired = _fired(records)
    assert "PAD-PS-001" in fired
    assert "PAD-BF-001" not in fired, "no account sees enough attempts"


def test_credential_stuffing_is_reported_by_the_fan_out_rule(
    replays: dict[str, Any],
) -> None:
    """The honest result, and the reason it is not PAD-CS-001.

    Stuffing and spraying are close relatives at the rule layer: both are one
    source touching many accounts with a high failure share. What separates them
    -- an unfamiliar device or country *for the targeted account* -- is a
    baseline-derived signal the serving path does not carry, so the fan-out rule
    is what fires. The scenario's published ``limitations`` say exactly this.
    """
    _, records = replays["credential_stuffing"]
    fired = _fired(records)
    assert fired == {"PAD-PS-001"}
    assert "PAD-CS-001" not in fired
    assert "PAD-BOT-001" not in fired, "four rotating clients is not one tool"


def test_account_takeover_shows_the_burst_and_the_success_that_followed(
    replays: dict[str, Any],
) -> None:
    """The sequence an analyst triages, even though PAD-ATO-001 cannot fire."""
    _, records = replays["account_takeover"]
    fired = _fired(records)
    assert "PAD-BF-001" in fired, "the failure burst"
    assert "PAD-BF-002" in fired, "the success that followed it"
    assert "PAD-ATO-001" not in fired

    last = records[-1]
    assert last["authentication_outcome"] == "success"
    assert "PAD-BF-002" in last["detection"]["rule"]["fired_rule_ids"]


def test_bot_activity_isolates_the_automation_indicator(
    replays: dict[str, Any],
) -> None:
    """Too few accounts for spraying, too few failures for brute force."""
    _, records = replays["bot_activity"]
    assert _fired(records) == {"PAD-BOT-001"}


def test_mixed_attack_produces_more_than_one_kind_of_finding(
    replays: dict[str, Any],
) -> None:
    """A timeline an analyst has to read, not a single repeated verdict."""
    _, records = replays["mixed_attack"]
    fired = _fired(records)
    assert {"PAD-PS-001", "PAD-BF-001", "PAD-BF-002"} <= fired
    # Different rules fire on different steps, which is the point of the
    # scenario: the timeline changes character as the run progresses.
    per_step = [
        frozenset(item["detection"]["rule"]["fired_rule_ids"])
        for item in records
        if item["detection"]["rule"]["flagged"]
    ]
    assert len(set(per_step)) > 1


def test_the_baseline_dependent_rules_never_fire_on_a_live_request(
    replays: dict[str, Any],
) -> None:
    """A finding about the serving path, recorded rather than worked around.

    PAD-CS-001 gates on ``user_in_baseline`` and PAD-ATO-001 counts the
    ``is_new_*_for_user`` novelty flags. Both come from a fitted behavioural
    baseline, and the serving path computes point-in-time features from the
    supplied window alone with no baseline artifact loaded -- so both rules
    report insufficient data on every live request, in every scenario, whatever
    the events look like.

    This is stated in the affected scenarios' published ``limitations`` and in
    ``docs/live-replay.md``. It was **not** addressed by loosening a rule or
    moving a threshold: the Phase 4 configuration and the Phase 5 artifacts are
    frozen, and a demonstration is not a reason to unfreeze one.
    """
    for scenario_id, (_, records) in replays.items():
        fired = _fired(records)
        assert "PAD-CS-001" not in fired, scenario_id
        assert "PAD-ATO-001" not in fired, scenario_id


# ---------------------------------------------------------------------------
# Determinism and pace independence
# ---------------------------------------------------------------------------


def _scientific(records: list[dict[str, Any]]) -> list[Any]:
    """Return the timeline with every presentation-only field removed.

    ``run_id``, ``emitted_at`` and ``replay_state`` are the fields that describe
    *this execution*; everything else describes what the detector was given and
    what it decided.
    """
    return [
        {
            key: value
            for key, value in record.items()
            if key not in {"run_id", "emitted_at", "replay_state"}
        }
        for record in records
    ]


def test_two_runs_of_one_scenario_produce_identical_results(
    demo: TestClient, replays: dict[str, Any]
) -> None:
    """Same scenario, same bundle, same build: the same verdicts."""
    _, first = replays["account_takeover"]
    _, second = _replay(demo, "account_takeover")
    assert _scientific(second) == _scientific(first)


@pytest.mark.parametrize("pace", ["fast", "normal", "slow"])
def test_the_result_is_identical_at_every_pace(
    demo: TestClient, replays: dict[str, Any], pace: str
) -> None:
    """Pace changes presentation timing and nothing a detector reads.

    Asserted against the real serving path rather than a stub: the events, the
    windows, the rule verdicts, the model's score and the fused decision are all
    byte-identical whether the run took a fraction of a second or twenty.
    """
    _, baseline = replays["account_takeover"]
    _, paced = _replay(demo, "account_takeover", pace=pace)
    assert _scientific(paced) == _scientific(baseline)


def test_the_scenario_fingerprint_is_the_same_across_runs(
    demo: TestClient, replays: dict[str, Any]
) -> None:
    """Content identity: what makes "the same scenario" a checkable claim."""
    run, _ = replays["brute_force"]
    again, _ = _replay(demo, "brute_force")
    assert again["scenario_fingerprint"] == run["scenario_fingerprint"]
    assert again["run_id"] != run["run_id"], "instance identity is not content identity"


def test_a_replayed_step_matches_the_detect_endpoint_on_the_same_window(
    demo: TestClient, replays: dict[str, Any]
) -> None:
    """The strongest form of "there is no second detection path".

    The window a replay step was scored against is reconstructed here and posted
    to ``POST /api/v1/detect``, and the two verdicts are compared field by field.
    They agree because they are the *same* call: the replay engine's detector is
    one invocation of the function the endpoint invokes.
    """
    from password_attack_detector.replay.scenarios import scenario

    source = scenario("account_takeover")
    assert source is not None
    _, records = replays["account_takeover"]
    events = source.events()

    for index in (0, len(events) // 2, len(events) - 1):
        window = events[: index + 1]
        response = demo.post(
            "/api/v1/detect",
            json={
                "events": window,
                "anchor_selection": "explicit",
                "anchor_event_ids": [window[-1]["event_id"]],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["anchor"] == records[index]["detection"]
