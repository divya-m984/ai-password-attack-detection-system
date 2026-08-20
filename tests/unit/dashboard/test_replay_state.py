"""What one browser session knows about a server-side replay run.

The distinction this module's subject exists to keep visible: the run lives on
the server and the session holds a *view* of it.  So the tests are about the
things a view can get wrong -- double-counting a repeated page, carrying one
run's records into another, losing the label that says where a record came from
-- rather than about anything the run itself does.

No backend here. ``tests/integration/test_replay_dashboard.py`` drives the same
object against the real deployment; this file states the invariants directly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from password_attack_detector.dashboard.contracts import (
    ReplayRunDocument,
    ReplayTimelineDocument,
)
from password_attack_detector.dashboard.state import (
    MAX_REPLAY_RECORDS,
    REPLAY_SCENARIO_PREFIX,
    DashboardSession,
    ReplaySession,
)
from password_attack_detector.dashboard.views.replay import (
    LIVE_POLL_SECONDS,
    PACES,
    poll_interval,
)

_BASE = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)


def _run(
    run_id: str = "run_0001",
    *,
    state: str = "running",
    emitted: int = 0,
    total: int = 10,
    more: bool = True,
) -> ReplayRunDocument:
    """Return a run document in the shape the service publishes."""
    return ReplayRunDocument.model_validate(
        {
            "run_id": run_id,
            "scenario_id": "brute_force",
            "scenario_name": "Concentrated brute force",
            "pace": "instant",
            "state": state,
            "event_count": total,
            "emitted_count": emitted,
            "next_sequence": emitted,
            "more_expected": more,
        }
    )


def _record(sequence: int, *, severity: str = "high", flagged: bool = True) -> Any:
    """Return one timeline record, embedding a serving-shaped verdict."""
    return {
        "sequence": sequence,
        "run_id": "run_0001",
        "scenario_id": "brute_force",
        "replay_state": "running",
        "event_index": sequence - 1,
        "window_event_count": sequence,
        "source_event_time": (_BASE + timedelta(seconds=10 * sequence)).isoformat(),
        "emitted_at": _BASE.isoformat(),
        "authentication_outcome": "failure",
        "detection": {
            "anchor_event_id": f"anchor-{sequence}",
            "anchor_event_time": (_BASE + timedelta(seconds=10 * sequence)).isoformat(),
            "rule": {
                "flagged": flagged,
                "risk_score": 80.0 if flagged else 0.0,
                "severity": severity,
                "fired_rule_ids": ["PAD-BF-001"] if flagged else [],
            },
            "ml": {"available": True, "flagged": flagged, "decision_score": 1.0},
            "hybrid": {"available": True, "flagged": flagged, "strategy": "stacked"},
            "severity": severity,
        },
    }


def _page(
    *sequences: int, after: int = 0, more: bool = False
) -> ReplayTimelineDocument:
    """Return one timeline page carrying the given sequences."""
    return ReplayTimelineDocument.model_validate(
        {
            "run_id": "run_0001",
            "scenario_id": "brute_force",
            "state": "running",
            "after_sequence": after,
            "next_sequence": max(sequences, default=after),
            "record_count": len(sequences),
            "records": [_record(item) for item in sequences],
            "more_expected": more,
            "emitted_count": max(sequences, default=0),
            "event_count": 10,
        }
    )


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------


def test_a_fresh_session_follows_nothing() -> None:
    """The default, and the state a reload returns to."""
    session = ReplaySession()
    assert session.attached is False
    assert session.active is False
    assert session.records == []
    assert session.cursor == 0


def test_attaching_records_the_run_and_starts_a_fresh_cursor() -> None:
    """A new run's sequences start at 1, so the cursor must start at 0."""
    session = ReplaySession()
    session.attach(_run())
    assert session.attached is True
    assert session.active is True
    assert session.run_id == "run_0001"
    assert session.cursor == 0


def test_attaching_a_second_run_discards_the_first_ones_records() -> None:
    """Two runs share a numbering that starts at 1; merging them means nothing."""
    session = ReplaySession()
    session.attach(_run())
    session.absorb(_page(1, 2, 3))
    assert len(session.records) == 3

    session.attach(_run("run_0002"))
    assert session.records == []
    assert session.cursor == 0
    assert session.run_id == "run_0002"


def test_detaching_forgets_everything() -> None:
    """Nothing survives to be rendered beside a run that is no longer followed."""
    session = ReplaySession()
    session.attach(_run())
    session.absorb(_page(1, 2))
    session.detach()
    assert session.attached is False
    assert session.run is None
    assert session.records == []
    assert session.cursor == 0


def test_observing_updates_the_state_without_touching_the_records() -> None:
    """A poll refreshes what the run *is*; the timeline is fetched separately."""
    session = ReplaySession()
    session.attach(_run(emitted=0))
    session.absorb(_page(1, 2))
    session.observe(_run(emitted=5))
    assert session.run is not None
    assert session.run.emitted_count == 5
    assert len(session.records) == 2


# ---------------------------------------------------------------------------
# Absorbing pages
# ---------------------------------------------------------------------------


def test_absorbing_a_page_advances_the_cursor() -> None:
    """The cursor is the highest sequence held, which is what the next poll asks past."""
    session = ReplaySession()
    session.attach(_run())
    assert session.absorb(_page(1, 2, 3)) == 3
    assert session.cursor == 3
    assert [item.sequence for item in session.records] == [1, 2, 3]


def test_absorbing_the_same_page_twice_adds_nothing() -> None:
    """A repeated page must not make a chart grow while the run stands still."""
    session = ReplaySession()
    session.attach(_run())
    page = _page(1, 2, 3)
    assert session.absorb(page) == 3
    assert session.absorb(page) == 0
    assert len(session.records) == 3


def test_absorbing_an_overlapping_page_keeps_only_what_is_new() -> None:
    """A cursor that slipped backwards would otherwise duplicate rows."""
    session = ReplaySession()
    session.attach(_run())
    session.absorb(_page(1, 2, 3))
    assert session.absorb(_page(2, 3, 4, 5, after=1)) == 2
    assert [item.sequence for item in session.records] == [1, 2, 3, 4, 5]


def test_absorbing_an_empty_page_is_harmless() -> None:
    """What a client that is up to date receives on every poll."""
    session = ReplaySession()
    session.attach(_run())
    session.absorb(_page(1))
    assert session.absorb(_page(after=1)) == 0
    assert session.cursor == 1


def test_the_record_buffer_is_bounded() -> None:
    """A buffer whose size is decided by the other end is not a buffer."""
    session = ReplaySession()
    session.attach(_run())
    for start in range(1, MAX_REPLAY_RECORDS + 60, 20):
        session.absorb(_page(*range(start, start + 20), after=start - 1))
    assert len(session.records) <= MAX_REPLAY_RECORDS


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def test_derived_records_carry_the_replay_prefix() -> None:
    """The one thing this console must never do is merge the two sources."""
    session = ReplaySession()
    session.attach(_run())
    session.absorb(_page(1, 2))
    derived = session.detection_records()
    assert len(derived) == 2
    assert all(
        item.scenario == f"{REPLAY_SCENARIO_PREFIX}brute_force" for item in derived
    )


def test_derived_records_carry_the_verdict_the_service_returned() -> None:
    """Assembled from the response, never re-derived."""
    session = ReplaySession()
    session.attach(_run())
    session.absorb(_page(1))
    record = session.detection_records()[0]
    assert record.severity == "high"
    assert record.rule_flagged is True
    assert record.fired_rule_ids == ("PAD-BF-001",)
    assert record.ml_available is True
    assert record.hybrid_strategy == "stacked"
    assert record.sequence == 1
    assert record.event_count == 1, "the window size at that step"


def test_the_replay_view_is_separate_from_the_manual_history() -> None:
    """Two objects, because they are two things with two lifetimes."""
    session = DashboardSession()
    session.replay.attach(_run())
    session.replay.absorb(_page(1, 2, 3))
    assert session.history == []
    assert session.detection_count == 0
    assert len(session.replay.detection_records()) == 3


def test_clearing_the_manual_history_leaves_the_replay_view_alone() -> None:
    """They are cleared by different controls because they are different data."""
    session = DashboardSession()
    session.replay.attach(_run())
    session.replay.absorb(_page(1))
    session.clear_history()
    assert len(session.replay.records) == 1


# ---------------------------------------------------------------------------
# Polling policy
# ---------------------------------------------------------------------------


def test_nothing_attached_means_no_polling() -> None:
    """A page that is not following a run does not call the backend on a timer."""
    assert poll_interval(ReplaySession()) is None


def test_an_active_run_polls_at_the_bounded_interval() -> None:
    """Fast enough to show every step, slow enough not to be a load source."""
    session = ReplaySession()
    session.attach(_run(more=True))
    assert poll_interval(session) == LIVE_POLL_SECONDS


@pytest.mark.parametrize("state", ["completed", "stopped", "failed"])
def test_a_terminal_run_stops_the_polling(state: str) -> None:
    """The property that keeps the console from calling a finished run forever."""
    session = ReplaySession()
    session.attach(_run(state=state, more=False))
    assert session.active is False
    assert poll_interval(session) is None


def test_the_console_reads_terminality_from_the_service() -> None:
    """A client that decided for itself when a run was over could disagree with it."""
    assert _run(state="running", more=True).terminal is False
    assert _run(state="completed", more=False).terminal is True
    # Even a state name the console does not recognise is handled, because the
    # console branches on ``more_expected`` rather than on the word.
    assert ReplayRunDocument(
        run_id="x", state="something_new", more_expected=False
    ).terminal


def test_progress_is_bounded_and_safe_with_no_events() -> None:
    """A malformed document must not produce a division error on a page."""
    assert _run(emitted=5, total=10).progress == pytest.approx(0.5)
    assert _run(emitted=10, total=10).progress == pytest.approx(1.0)
    assert ReplayRunDocument(run_id="x", event_count=0).progress == 0.0
    assert _run(emitted=99, total=10).progress == pytest.approx(1.0)


def test_the_console_pace_vocabulary_is_the_four_bounded_words() -> None:
    """Never a duration, and never an expression."""
    assert PACES == ("instant", "fast", "normal", "slow")
