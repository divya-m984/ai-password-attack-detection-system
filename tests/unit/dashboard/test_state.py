"""Tests for what one browser session remembers.

The session is a scratchpad, and the tests below are mostly about keeping it
honestly labelled as one: bounded, resettable, holding no request body, and
computing no verdict of its own.

``observed_at`` is passed into every recording call rather than read from a
clock, which is what lets these tests state exactly what a session looks like
instead of asserting around wall time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from password_attack_detector.dashboard.contracts import (
    BatchDetectionDocument,
    DetectionDocument,
)
from password_attack_detector.dashboard.state import (
    MAX_HISTORY,
    DashboardSession,
    DetectionRecord,
    severity_counts,
    triggered_rule_counts,
)

WHEN = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)


def _document(
    *,
    anchor: str = "anchor-1",
    severity: str = "high",
    rule_flagged: bool = True,
    fired: tuple[str, ...] = ("PAD-BF-001",),
    ml_flagged: bool | None = True,
    hybrid_flagged: bool | None = True,
    strategy: str | None = "stacked",
    risk: float = 72.0,
) -> DetectionDocument:
    """Return a detection response in the shape the API publishes."""
    body: dict[str, Any] = {
        "api_schema_version": "1.0.0",
        "window": {"event_count": 30, "anchor_count": 1},
        "anchor": {
            "anchor_event_id": anchor,
            "anchor_event_time": WHEN,
            "severity": severity,
            "rule": {
                "flagged": rule_flagged,
                "risk_score": risk,
                "severity": severity,
                "fired_rule_ids": list(fired),
                "fired_rule_count": len(fired),
                "scoring_version": "1.0.0",
            },
            "ml": {
                "available": ml_flagged is not None,
                "flagged": ml_flagged,
                "score_kind": "calibrated_probability",
                "decision_score": 0.8,
                "probability": 0.8,
                "decision_threshold": 0.5,
                "unavailable_reason": None if ml_flagged is not None else "no_champion",
            },
            "hybrid": {
                "available": hybrid_flagged is not None,
                "flagged": hybrid_flagged,
                "strategy": strategy,
                "unavailable_reason": (
                    None if hybrid_flagged is not None else "no_fusion_selection"
                ),
            },
        },
    }
    return DetectionDocument.model_validate(body)


def _batch(anchors: tuple[str, ...]) -> BatchDetectionDocument:
    """Return a batch response carrying one verdict per named anchor."""
    return BatchDetectionDocument.model_validate(
        {
            "api_schema_version": "1.0.0",
            "window": {"event_count": len(anchors), "anchor_count": len(anchors)},
            "anchors": [_document(anchor=name).anchor.model_dump() for name in anchors],
        }
    )


# ---------------------------------------------------------------------------
# The draft window
# ---------------------------------------------------------------------------


def test_a_new_session_holds_nothing() -> None:
    """A fresh browser session starts empty, and says so everywhere."""
    session = DashboardSession()
    assert session.draft_events == []
    assert session.history == []
    assert session.latest is None
    assert session.last_result is None
    assert session.detection_count == 0


def test_events_are_appended_in_order() -> None:
    """The window's order is the caller's; nothing here sorts it."""
    session = DashboardSession()
    session.extend_events([{"event_id": "a"}, {"event_id": "b"}])
    session.add_event({"event_id": "c"})
    assert [item["event_id"] for item in session.draft_events] == ["a", "b", "c"]


def test_an_added_event_is_copied_not_aliased() -> None:
    """A caller mutating its own mapping must not rewrite the session's window."""
    session = DashboardSession()
    original = {"event_id": "a"}
    session.add_event(original)
    original["event_id"] = "b"
    assert session.draft_events[0]["event_id"] == "a"


@pytest.mark.parametrize(
    "field",
    ["password", "Password", "api-key", "secret", "token", "PASSWORD_HASH", "pwd"],
)
def test_a_credential_shaped_field_never_enters_the_session(field: str) -> None:
    """Refused at the door, under any spelling and any case.

    The service refuses such a request too, but by then the value would already
    be in browser-session state and in the console's JSON preview. This is the
    check that keeps it out of both.
    """
    session = DashboardSession()
    with pytest.raises(ValueError, match="no credential material"):
        session.add_event({"event_id": "a", field: "irrelevant"})
    assert session.draft_events == []


def test_a_refusal_names_the_field_count_and_not_the_value() -> None:
    """A message quoting the secret would be the leak the check exists to stop."""
    session = DashboardSession()
    with pytest.raises(ValueError) as caught:
        session.add_event({"event_id": "a", "password": "hunter2"})
    assert "hunter2" not in str(caught.value)
    assert "password" not in str(caught.value)
    assert "1 prohibited field name(s)" in str(caught.value)


def test_a_template_carrying_a_credential_leaves_no_partial_window() -> None:
    """The draft is cleared before the new events are admitted, not after."""
    session = DashboardSession()
    session.add_event({"event_id": "existing"})
    with pytest.raises(ValueError, match="no credential material"):
        session.load_scenario(
            "hostile", [{"event_id": "a"}, {"event_id": "b", "token": "x"}]
        )
    assert [item["event_id"] for item in session.draft_events] == ["a"]


def test_removing_an_index_that_is_not_there_does_nothing() -> None:
    """A stale index from a re-rendered form must not raise onto the page."""
    session = DashboardSession()
    session.add_event({"event_id": "a"})
    session.remove_event(7)
    session.remove_event(-1)
    assert len(session.draft_events) == 1


def test_clearing_the_window_forgets_the_template_too() -> None:
    """A cleared window did not come from a scenario."""
    session = DashboardSession()
    session.load_scenario("brute_force", [{"event_id": "a"}])
    session.clear_window()
    assert session.draft_events == []
    assert session.selected_scenario == "custom"


def test_loading_a_scenario_replaces_rather_than_appends() -> None:
    """A template is a complete scenario; merging would describe neither."""
    session = DashboardSession()
    session.add_event({"event_id": "manual"})
    session.load_scenario("normal", [{"event_id": "a"}, {"event_id": "b"}])
    assert [item["event_id"] for item in session.draft_events] == ["a", "b"]
    assert session.selected_scenario == "normal"


# ---------------------------------------------------------------------------
# Recording results
# ---------------------------------------------------------------------------


def test_a_record_reports_what_the_service_returned() -> None:
    """The session copies verdicts; it never derives one."""
    session = DashboardSession()
    session.selected_scenario = "brute_force"
    record = session.record(_document(), observed_at=WHEN)
    assert record.sequence == 1
    assert record.scenario == "brute_force"
    assert record.severity == "high"
    assert record.rule_flagged is True
    assert record.ml_flagged is True
    assert record.hybrid_flagged is True
    assert record.hybrid_strategy == "stacked"
    assert record.fired_rule_ids == ("PAD-BF-001",)


def test_sequence_numbers_count_up_from_one() -> None:
    """Local to the tab, and stable within it."""
    session = DashboardSession()
    for expected in (1, 2, 3):
        assert session.record(_document(), observed_at=WHEN).sequence == expected


def test_the_latest_full_document_is_kept_and_the_previous_one_is_not() -> None:
    """A session holding every full response would grow without bound."""
    session = DashboardSession()
    session.record(_document(anchor="first"), observed_at=WHEN)
    session.record(_document(anchor="second"), observed_at=WHEN)
    assert session.last_result is not None
    assert session.last_result.anchor.anchor_event_id == "second"


def test_a_batch_is_recorded_as_one_entry_per_anchor() -> None:
    """A batch that rendered results without recording them would report an
    empty session on the alerts page while showing verdicts on screen."""
    session = DashboardSession()
    session.selected_scenario = "spraying"
    stored = session.record_batch(
        _batch(("anchor-a", "anchor-b", "anchor-c")), observed_at=WHEN
    )
    assert [item.sequence for item in stored] == [1, 2, 3]
    assert [item.anchor_event_id for item in session.history] == [
        "anchor-a",
        "anchor-b",
        "anchor-c",
    ]
    assert all(item.scenario == "spraying" for item in session.history)
    assert session.detection_count == 3


def test_a_batch_leaves_the_latest_single_result_alone() -> None:
    """The detailed panels render one anchor; a batch has no single latest one."""
    session = DashboardSession()
    session.record(_document(anchor="single"), observed_at=WHEN)
    session.record_batch(_batch(("a", "b")), observed_at=WHEN)
    assert session.last_result is not None
    assert session.last_result.anchor.anchor_event_id == "single"


def test_a_batch_respects_the_history_bound() -> None:
    """One code path stores every record, so one bound applies to both."""
    session = DashboardSession()
    session.record_batch(
        _batch(tuple(f"a-{index}" for index in range(MAX_HISTORY + 10))),
        observed_at=WHEN,
    )
    assert len(session.history) == MAX_HISTORY
    assert session.detection_count == MAX_HISTORY + 10


def test_the_history_is_bounded() -> None:
    """A session that never forgets is a memory leak with a chart on top."""
    session = DashboardSession()
    for _ in range(MAX_HISTORY + 25):
        session.record(_document(), observed_at=WHEN)
    assert len(session.history) == MAX_HISTORY


def test_the_counter_keeps_counting_past_the_bound() -> None:
    """Dropping the oldest rows must not renumber the ones that remain."""
    session = DashboardSession()
    for _ in range(MAX_HISTORY + 5):
        session.record(_document(), observed_at=WHEN)
    assert session.detection_count == MAX_HISTORY + 5
    assert session.history[-1].sequence == MAX_HISTORY + 5
    assert session.history[0].sequence == 6


def test_clearing_the_history_resets_the_numbering() -> None:
    """Otherwise the next detection is #57 in a list whose first entry is #57."""
    session = DashboardSession()
    for _ in range(5):
        session.record(_document(), observed_at=WHEN)
    session.clear_history()
    assert session.history == []
    assert session.last_result is None
    assert session.last_explanation is None
    assert session.record(_document(), observed_at=WHEN).sequence == 1


def test_clearing_the_history_leaves_the_draft_window_alone() -> None:
    """Forgetting results is not the same act as discarding the request."""
    session = DashboardSession()
    session.add_event({"event_id": "a"})
    session.record(_document(), observed_at=WHEN)
    session.clear_history()
    assert len(session.draft_events) == 1


# ---------------------------------------------------------------------------
# What a record is, and is not
# ---------------------------------------------------------------------------


def test_a_record_carries_no_request_body() -> None:
    """Identifiers, addresses and application names stay in the visible draft."""
    declared = set(DetectionRecord.__slots__)
    assert not declared & {
        "events",
        "request",
        "body",
        "user_id",
        "source_id",
        "source_ip",
        "device_id",
        "session_id",
        "application_id",
    }


def test_a_record_carries_no_artifact_identity() -> None:
    """Nothing here is an artifact path, a fingerprint, or a key."""
    declared = set(DetectionRecord.__slots__)
    assert not declared & {
        "artifact_root",
        "model_path",
        "stacked_state_fingerprint",
        "pseudonymization_key",
        "champion_scope_key",
    }


def test_the_disjunction_is_for_grouping_and_is_not_a_verdict() -> None:
    """``any_layer_flagged`` groups rows; the fused verdict is the server's."""
    record = DetectionRecord.from_document(
        _document(rule_flagged=False, ml_flagged=False, hybrid_flagged=True),
        sequence=1,
        observed_at=WHEN,
        scenario="custom",
    )
    assert record.any_layer_flagged is True
    # The hybrid verdict is reported separately and is unchanged by the above.
    assert record.hybrid_flagged is True

    quiet = DetectionRecord.from_document(
        _document(rule_flagged=False, ml_flagged=False, hybrid_flagged=False),
        sequence=2,
        observed_at=WHEN,
        scenario="custom",
    )
    assert quiet.any_layer_flagged is False


def test_an_unavailable_layer_contributes_no_verdict() -> None:
    """An absent model has not 'declined to flag'; it produced nothing."""
    record = DetectionRecord.from_document(
        _document(ml_flagged=None, hybrid_flagged=None, strategy=None),
        sequence=1,
        observed_at=WHEN,
        scenario="custom",
    )
    assert record.ml_available is False
    assert record.ml_flagged is None
    assert record.hybrid_available is False
    assert record.hybrid_flagged is None


def test_a_record_is_immutable() -> None:
    """A rendered row must not be rewritten by the page rendering it."""
    record = DetectionRecord.from_document(
        _document(), sequence=1, observed_at=WHEN, scenario="custom"
    )
    with pytest.raises(Exception, match="cannot assign"):
        record.severity = "critical"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_severity_counts_follow_the_ordinal_scale() -> None:
    """A chart of it reads left to right as increasing seriousness."""
    session = DashboardSession()
    for severity in ("critical", "low", "low", "high"):
        session.record(_document(severity=severity), observed_at=WHEN)
    counts = severity_counts(session.history)
    assert list(counts) == ["none", "low", "medium", "high", "critical"]
    assert counts == {"none": 0, "low": 2, "medium": 0, "high": 1, "critical": 1}


def test_severity_counts_of_an_empty_session_are_all_zero() -> None:
    """Zeroes for a session with no detections, not an absent mapping."""
    assert set(severity_counts([]).values()) == {0}


def test_rule_counts_are_ordered_by_frequency_then_identifier() -> None:
    """The same session always renders the same chart."""
    session = DashboardSession()
    session.record(_document(fired=("PAD-BF-001", "PAD-PS-001")), observed_at=WHEN)
    session.record(_document(fired=("PAD-BF-001",)), observed_at=WHEN)
    session.record(_document(fired=("PAD-AA-001",)), observed_at=WHEN)
    counts = triggered_rule_counts(session.history)
    assert list(counts) == ["PAD-BF-001", "PAD-AA-001", "PAD-PS-001"]
    assert counts["PAD-BF-001"] == 2


def test_rule_counts_of_a_session_with_no_flags_are_empty() -> None:
    """An empty mapping, so the chart shows its own empty state."""
    session = DashboardSession()
    session.record(_document(rule_flagged=False, fired=()), observed_at=WHEN)
    assert triggered_rule_counts(session.history) == {}


def test_the_flagged_history_is_a_subset_of_the_history() -> None:
    """The filter selects; it does not recompute anything."""
    session = DashboardSession()
    session.record(_document(), observed_at=WHEN)
    session.record(
        _document(rule_flagged=False, ml_flagged=False, hybrid_flagged=False),
        observed_at=WHEN + timedelta(minutes=1),
    )
    assert len(session.history) == 2
    assert len(session.flagged_history) == 1
