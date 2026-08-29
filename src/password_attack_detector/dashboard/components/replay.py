"""Where replay data appears on a page that is not the Live Replay page.

One module, because the same sentence has to be said on four of them and saying
it four times is how it eventually gets said three times.

The sentence is: **this data came from a server-side demonstration run, not from
this browser session, and not from any persistent store.**  Those are three
different origins with three different lifetimes, and a security console that let
them blur would be presenting a demo as history.  So every page that shows replay
data shows this banner above it, and every page that could show *both* sources
makes the viewer choose which -- there is no view anywhere in this console where
manual and replay results are silently summed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import streamlit as st

from password_attack_detector.dashboard.formatting import escape_text
from password_attack_detector.dashboard.state import DashboardSession, ReplaySession
from password_attack_detector.dashboard.theme import badge

__all__ = [
    "SOURCE_LABELS",
    "DataSource",
    "render_run_banner",
    "select_source",
]

_MUTED: Final[str] = "#8b949e"
_BLUE: Final[str] = "#58a6ff"
_GREEN: Final[str] = "#3fb950"


class DataSource(StrEnum):
    """Which origin a page is currently displaying."""

    #: Windows submitted from the Detection Console in this browser tab.
    MANUAL = "manual"
    #: The server-side demonstration run this session is following.
    REPLAY = "replay"
    #: Both, each row still carrying its own origin.
    BOTH = "both"


#: What each source is called on screen.  Long labels on purpose: this is the
#: control that stops two very different things being read as one number.
SOURCE_LABELS: Final[dict[DataSource, str]] = {
    DataSource.MANUAL: "This dashboard session (manual submissions)",
    DataSource.REPLAY: "Active demo replay run (server-side)",
    DataSource.BOTH: "Both, labelled by source",
}


def render_run_banner(replay: ReplaySession) -> bool:
    """Render what demo run this session is following, and return whether there is one."""
    if not replay.attached or replay.run is None:
        return False
    run = replay.run
    state_color = _BLUE if not run.terminal else _GREEN
    st.markdown(
        f"{badge('demo replay', color=_MUTED)} "
        f"{badge(run.state, color=state_color)} "
        f'<span class="pad-mono">{escape_text(run.scenario_name or run.scenario_id)} '
        f"· {escape_text(run.emitted_count)}/{escape_text(run.event_count)} events"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This run is executing on the **server**, not in this browser. It is a "
        "synthetic demonstration, it is held in that process's memory only, and "
        "it is gone when the service restarts."
    )
    return True


def select_source(session: DashboardSession, *, key: str) -> DataSource:
    """Render the source selector and return what the viewer chose.

    Offered only when there is actually a second source to choose. With no demo
    run attached the page has one origin, the control would be a decision with
    one option, and its absence is not a silent merge.
    """
    if not session.replay.attached or not session.replay.records:
        return DataSource.MANUAL
    options = list(DataSource)
    chosen = st.radio(
        "Data source",
        options=options,
        format_func=lambda item: SOURCE_LABELS[item],
        index=options.index(DataSource.MANUAL),
        horizontal=False,
        key=key,
    )
    return chosen if isinstance(chosen, DataSource) else DataSource.MANUAL
