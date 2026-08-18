"""Attack analytics over this session's detections.

Four small charts, all drawn from :class:`~password_attack_detector.dashboard
.state.DetectionRecord` values the service returned.  Nothing is simulated,
nothing is seeded, and an empty session draws no chart at all -- a chart with no
data behind it still looks like a measurement, and that is exactly the confusion
worth designing against on a console like this.

The layers are never plotted as one stacked total.  Rule flags, model flags and
hybrid flags are three separate counts of three separate decisions, and a stacked
bar would suggest they are parts of a whole.
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.charts import (
    render_layer_agreement_chart,
    render_rule_frequency_chart,
    render_session_timeline,
    render_severity_chart,
)
from password_attack_detector.dashboard.components.status import Connectivity
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title

__all__ = ["render"]


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the analytics view."""
    st.markdown(section_title("Current dashboard session"), unsafe_allow_html=True)
    st.info(
        "Every figure below is computed from the detections performed in **this "
        "browser session**. No historical or aggregate traffic data exists yet, "
        "and none is simulated to fill the space.",
    )

    if not session.history:
        st.caption(
            "No detection activity in this dashboard session. Submit a few "
            "windows on the **Detection Console** and these charts populate "
            "from the results."
        )
        return

    st.caption(f"{len(session.history)} detection(s) in this session.")

    left, right = st.columns(2)
    with left:
        st.markdown(section_title("Detections by severity"), unsafe_allow_html=True)
        render_severity_chart(session.history)
        st.caption("Phase 4 ordinal severity, in the scale's own order.")
    with right:
        st.markdown(section_title("Triggered rules"), unsafe_allow_html=True)
        render_rule_frequency_chart(session.history)
        st.caption("How often each rule fired across this session's windows.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown(section_title("Flags raised, by layer"), unsafe_allow_html=True)
        render_layer_agreement_chart(session.history)
        st.caption(
            "Three independent counts. Not a stacked total: the layers are not "
            "parts of one quantity."
        )
    with lower_right:
        st.markdown(section_title("Session activity"), unsafe_allow_html=True)
        render_session_timeline(session.history)

    st.markdown(section_title("Scenarios submitted"), unsafe_allow_html=True)
    counts: dict[str, int] = {}
    for item in session.history:
        counts[item.scenario] = counts.get(item.scenario, 0) + 1
    st.dataframe(
        [
            {"Scenario": name, "Detections": count}
            for name, count in sorted(
                counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ],
        width="stretch",
        hide_index=True,
    )
