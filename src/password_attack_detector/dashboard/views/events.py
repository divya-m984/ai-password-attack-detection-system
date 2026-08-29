"""Authentication events: the ones this browser session composed.

There is no event store yet, so this page shows the window currently being
composed and nothing else.  It is not a search over historical traffic, it does
not claim to be, and the alternative -- populating it with plausible-looking rows
-- would make a demonstration of a capability that does not exist.

What it *does* offer is the part an analyst genuinely needs while building a
request: seeing the exact events that will be sent, dropping one, clearing the
window, and re-sending a window they can still read.  A resend is one explicit
press on visible content; nothing here re-submits on its own.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.replay import render_run_banner
from password_attack_detector.dashboard.components.status import (
    Connectivity,
    render_problem,
)
from password_attack_detector.dashboard.formatting import format_timestamp
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title
from password_attack_detector.dashboard.views.detection import (
    synthetic_identity_note,
)

__all__ = ["render"]

#: Fields the table shows.  Every one is something the analyst typed or a
#: template supplied; no field is added here that the request does not carry.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("event_time", "Event time"),
    ("authentication_outcome", "Outcome"),
    ("authentication_method", "Method"),
    ("failure_reason", "Failure reason"),
    ("mfa_outcome", "MFA"),
    ("application_id", "Application"),
    ("user_id", "User"),
    ("device_id", "Device"),
    ("session_id", "Session"),
    ("country_code", "Country"),
    ("client_type", "Client"),
    ("response_time_ms", "Response ms"),
)


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the current session's authentication events."""
    st.markdown(section_title("Current dashboard session"), unsafe_allow_html=True)
    st.info(
        "This view shows the window composed in **this browser session**. "
        "Persistent event storage and live streaming are a later milestone; "
        "no historical traffic is available to display.",
    )

    events = session.draft_events
    if not events:
        st.caption(
            "No events in this session. Build a window on the **Detection Console**."
        )
        _render_replay_events(session)
        return

    st.dataframe(
        [
            {
                label: item.get(key, "—") if item.get(key) is not None else "—"
                for key, label in _COLUMNS
            }
            | {"Source": item.get("source_id") or item.get("source_ip") or "—"}
            for item in events
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(synthetic_identity_note())
    st.caption(
        "No credential, key, artifact path, or internal identifier appears "
        "here, because none is ever collected: the session refuses a "
        "credential-shaped field name before storing it."
    )

    with st.expander("Inspect the exact request body"):
        st.code(json.dumps({"events": events}, indent=2), language="json")

    resend, clear = st.columns(2)
    with resend:
        if st.button(
            "Re-submit this window",
            help="Sends the window above once. Nothing is resent automatically.",
            width="stretch",
            disabled=not status.ready,
        ):
            _resend(client, session)
    with clear:
        if st.button("Clear session events", width="stretch"):
            session.clear_window()
            st.rerun()

    _render_replay_events(session)


def _render_replay_events(session: DashboardSession) -> None:
    """Render the events a followed demo run has emitted so far.

    Under its own heading, and describing only what the timeline records carry:
    when each fabricated event claimed to occur, its outcome, and how large the
    window was by the time it was scored. The run's *event bodies* are not
    fetched -- the console has no endpoint that returns them and no reason to
    want one -- so this is honestly a view of the run's steps rather than a copy
    of its input.
    """
    replay = session.replay
    if not replay.attached or not replay.records:
        return
    st.markdown(section_title("Server-side demo replay run"), unsafe_allow_html=True)
    render_run_banner(replay)
    st.dataframe(
        [
            {
                "Step": item.sequence,
                "Event time": format_timestamp(item.source_event_time),
                "Outcome": item.authentication_outcome,
                "Window size": item.window_event_count,
                "Severity": str(item.detection.severity),
            }
            for item in replay.records
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "These events were emitted by the **service**, from its own reviewed "
        "scenario catalog. They are synthetic, they carry no credential, and "
        "they were never submitted from this browser."
    )


def _resend(client: DashboardAPIClient, session: DashboardSession) -> None:
    """Send the visible window once, and record the result."""
    result = client.detect(session.draft_events, anchor_selection="last")
    if not result.ok:
        assert result.problem is not None
        render_problem(result.problem, context="The window was not re-submitted.")
        return
    record = session.record(result.unwrap(), observed_at=datetime.now(UTC))
    st.success(f"Detection #{record.sequence} complete.", icon="✅")
