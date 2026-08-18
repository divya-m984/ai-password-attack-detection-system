"""Security alerts: what this session's detections reported.

Every entry here is a detection somebody performed from this browser tab.  That
is stated at the top of the page and repeated on the table, because the single
most misleading thing a security console can do is present a session buffer as
though it were an alert database.

The sequence numbers are local.  They count submissions in this tab; they are not
alert identifiers, they are not stable across a reload, and nothing on the server
knows about them.  When persistent alerting arrives, these become what they
already are -- a scratchpad -- and the real identifiers come from the store.
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.alerts import render_alert_history
from password_attack_detector.dashboard.components.metrics import render_session_cards
from password_attack_detector.dashboard.components.status import Connectivity
from password_attack_detector.dashboard.formatting import (
    SEVERITY_COLORS,
    severity_rank,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import badge, section_title

__all__ = ["render"]


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render this session's alert history."""
    st.markdown(section_title("Current dashboard session"), unsafe_allow_html=True)
    st.info(
        "These are the detections performed from **this browser session**. "
        "There is no persistent alert store yet, so nothing here survives a "
        "reload and nothing here came from the server's history.",
    )

    render_session_cards(session.history)

    if not session.history:
        st.caption(
            "No detection activity in this dashboard session. Submit a window "
            "on the **Detection Console** to populate this page."
        )
        return

    st.markdown(section_title("Session results"), unsafe_allow_html=True)
    only_flagged = st.checkbox(
        "Show only results where a layer raised a flag", value=False
    )
    history = session.flagged_history if only_flagged else tuple(session.history)
    render_alert_history(history)

    st.markdown(
        section_title("Highest severity in this session"), unsafe_allow_html=True
    )
    ranked = sorted(session.history, key=lambda item: -severity_rank(item.severity))
    for item in ranked[:3]:
        color = SEVERITY_COLORS.get(item.severity, "#8b949e")
        st.markdown(
            f"{badge(item.severity, color=color)} "
            f'<span class="pad-mono">#{item.sequence} · '
            f"{', '.join(item.fired_rule_ids) or 'no rule fired'}</span>",
            unsafe_allow_html=True,
        )

    if st.button("Clear session history"):
        session.clear_history()
        st.rerun()
