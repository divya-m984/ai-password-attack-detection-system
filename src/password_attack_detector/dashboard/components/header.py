"""The page header and the sidebar: identity, connectivity, and navigation.

The header states two things at all times, because they are the two an analyst
checks before believing anything else on the screen: whether the API answered,
and whether the system it fronts is ready.  They are separate badges because they
are separate facts -- a service can be perfectly alive and unable to serve a
single detection, and collapsing that into one light would hide the case an
operator most needs to see.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from password_attack_detector.dashboard.components.status import (
    Connectivity,
    render_header_status,
    render_version_caption,
)
from password_attack_detector.dashboard.config import DashboardSettings
from password_attack_detector.dashboard.formatting import escape_text

__all__ = ["PAGES", "render_header", "render_sidebar"]

#: The navigation labels, in order.  Stable: they are how the documentation, the
#: tests, and a demo script all refer to a view.
#:
#: "Live Replay" sits directly under the Detection Console because the two are
#: the same act at two scales -- one window somebody composed, and a whole
#: scenario the server walks through -- and an analyst comparing them should not
#: have to cross the rest of the navigation to do it.
PAGES: Final[tuple[str, ...]] = (
    "Overview",
    "Detection Console",
    "Live Replay",
    "Authentication Events",
    "Security Alerts",
    "Attack Analytics",
    "Rule vs ML vs Hybrid",
    "Explainability",
    "Drift Monitoring",
    "System & Model",
)

_TITLE: Final[str] = "AI-Powered Password Attack Detection System"
_SUBTITLE: Final[str] = "Security Analytics &amp; Hybrid Detection Console"


def render_header(status: Connectivity, settings: DashboardSettings) -> None:
    """Render the title block and the connectivity badges."""
    title, indicators = st.columns([0.68, 0.32])
    with title:
        st.markdown(
            f'<div class="pad-header">'
            f'<p class="pad-title">{escape_text(settings.page_title)}</p>'
            f'<p class="pad-subtitle">{_SUBTITLE}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with indicators:
        st.markdown(
            f'<div style="text-align:right;padding-top:0.9rem">'
            f"{render_header_status(status)}</div>",
            unsafe_allow_html=True,
        )
    render_version_caption(status, settings.display_api_url)


def render_sidebar(status: Connectivity, settings: DashboardSettings) -> str:
    """Render navigation and connection controls, and return the chosen page.

    The refresh control is a **button**, not a timer. An auto-refreshing console
    is a client that keeps calling a service nobody is looking at, and the
    configured interval is shown as guidance for how often a manual re-read is
    worth doing rather than used to schedule one.
    """
    with st.sidebar:
        st.markdown(
            '<p class="pad-subtitle" style="margin-bottom:0.6rem">'
            "Detection console</p>",
            unsafe_allow_html=True,
        )
        page = st.radio("View", PAGES, label_visibility="collapsed")
        st.divider()
        st.markdown(
            f'<div class="pad-card-label">API endpoint</div>'
            f'<div class="pad-mono">{escape_text(settings.display_api_url)}</div>',
            unsafe_allow_html=True,
        )
        state = (
            "online, ready"
            if status.ready
            else "online, not ready"
            if status.online
            else "offline"
        )
        st.caption(f"Status: {state}")
        if st.button("Retry connection", width="stretch"):
            st.rerun()
        st.caption(
            f"Suggested re-read interval: {settings.refresh_seconds}s. "
            f"Refresh is manual; this console does not poll."
        )
        st.divider()
        st.caption(
            "History on every page is **this browser session only**. "
            "No server-side alert store exists yet."
        )
    return str(page)
