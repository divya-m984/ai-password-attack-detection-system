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
)
from password_attack_detector.dashboard.config import DashboardSettings
from password_attack_detector.dashboard.formatting import escape_text

__all__ = [
    "ADVANCED_PAGES",
    "PAGES",
    "PRIMARY_PAGES",
    "render_header",
    "render_sidebar",
]

#: Primary navigation labels -- the views a non-technical viewer uses.
PRIMARY_PAGES: Final[tuple[str, ...]] = (
    "Overview",
    "Live Replay",
    "Alerts",
    "Analytics",
    "Explainability",
    "Drift Monitoring",
)

#: Advanced navigation labels -- technical and debugging views.
ADVANCED_PAGES: Final[tuple[str, ...]] = (
    "Detection Console",
    "Authentication Events",
    "Rule vs ML vs Hybrid",
    "System & Model",
)

#: All navigation labels, in order. Stable: tests, docs and demo scripts
#: refer to a view by its label. Primary pages first, then advanced, then about.
PAGES: Final[tuple[str, ...]] = (
    *PRIMARY_PAGES,
    *ADVANCED_PAGES,
    "About System",
)

_TITLE: Final[str] = "AI-Powered Password Attack Detection"
_SUBTITLE: Final[str] = "Hybrid security analytics"


def render_header(status: Connectivity, settings: DashboardSettings) -> None:
    """Render the title block and the connectivity badges."""
    title, indicators = st.columns([0.7, 0.3])
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


def render_sidebar(status: Connectivity, settings: DashboardSettings) -> str:
    """Render navigation and connection controls, and return the chosen page.

    Navigation reads as three groups -- primary, advanced, about -- and *is* one
    radio. The grouping is drawn by the stylesheet above the first option of
    each group rather than by splitting the control into three widgets, because
    three widgets would be three selections: picking an advanced view would have
    to clear the primary one, and the state that decides which of the three is
    the live answer is exactly the ambiguity this console does not want in its
    navigation. One widget, one value, and :data:`PAGES` is its order.
    """
    with st.sidebar:
        # Allow programmatic navigation from other views (e.g. the CTA).
        nav_target = st.session_state.pop("_pad_nav_target", None)
        default_index = PAGES.index(nav_target) if nav_target in PAGES else 0
        page = st.radio(
            "View", PAGES, index=default_index, label_visibility="collapsed"
        )

        st.divider()

        # Connection status -- human-readable, no raw URL
        _conn = (
            "Online"
            if status.ready
            else "Online, not ready"
            if status.online
            else "Offline"
        )
        st.markdown(
            f'<div class="pad-card-label">Backend</div>'
            f'<div class="pad-card-value" style="font-size:0.95rem">'
            f"{escape_text(_conn)}</div>",
            unsafe_allow_html=True,
        )

        if st.button("Retry connection", use_container_width=True):
            st.rerun()

        st.caption("History is this browser session only.")
    return str(page)
