"""About System: what this project is, how it works, and what it does not do.

An approachable project summary assembled from verified facts.  Nothing here
is read from the API; it is static prose about the system's architecture and
the scope of this demonstration.
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.status import Connectivity
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title

__all__ = ["render"]

_DIAGRAM = """
<div style="background:#131a29;border:1px solid #1f2a3d;border-radius:8px;
            padding:1.2rem;font-size:0.88rem;color:#c9d1d9;
            line-height:2;text-align:center;max-width:28rem;margin:0 auto">
  <div style="color:#58a6ff;font-weight:600">Authentication events</div>
  <div style="color:#8b949e">&#8595;</div>
  <div style="color:#58a6ff">Point-in-time features</div>
  <div style="color:#8b949e">&#8595;</div>
  <div>
    <span style="color:#f0883e">Rules</span>
    &nbsp;+&nbsp;
    <span style="color:#3fb950">ML</span>
  </div>
  <div style="color:#8b949e">&#8595;</div>
  <div style="color:#58a6ff;font-weight:600">Stacked hybrid decision</div>
  <div style="color:#8b949e">&#8595;</div>
  <div style="color:#f85149;font-weight:600">Alert</div>
</div>
"""


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the About System page."""
    st.markdown(section_title("About this system"), unsafe_allow_html=True)

    st.markdown(
        '<div class="pad-intro">'
        "The AI-Powered Password Attack Detection System evaluates "
        "authentication activity using three complementary approaches: "
        "rule-based detection for known suspicious patterns, a trained "
        "machine-learning model for patterns no rule was written for, and "
        "frozen hybrid fusion that combines both signals into a single "
        "decision."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacing

    left, right = st.columns([0.45, 0.55])
    with left:
        st.markdown(section_title("How it works"), unsafe_allow_html=True)
        st.markdown(_DIAGRAM, unsafe_allow_html=True)

    with right:
        st.markdown(section_title("Capabilities"), unsafe_allow_html=True)
        st.markdown(
            """
- **Rule-based detection** -- deterministic conditions over authentication
  behaviour, each naming the pattern it found
- **Machine-learning detection** -- a frozen champion model applied at a
  frozen operating point
- **Hybrid fusion** -- the strategy validation selected, combining both
  layers into the final verdict
- **Synthetic replay** -- built-in scenarios replayed through the live
  detection path for demonstration
- **Explainability** -- per-anchor decomposition showing which factors
  influenced the model's decision
- **Drift monitoring** -- tracks whether live feature distributions move
  away from the training baseline
            """.strip()
        )

    st.markdown(section_title("This demonstration"), unsafe_allow_html=True)
    st.markdown(
        '<div class="pad-intro">'
        "This public deployment uses synthetic authentication scenarios for "
        "demonstration. All identities are fabricated, all events are "
        "generated from reviewed templates, and no real authentication "
        "traffic is processed. The detection path is the same one a "
        "production deployment would use."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(section_title("Limitations"), unsafe_allow_html=True)
    st.markdown(
        """
- **Session-only history.** There is no persistent alert store or event
  database. Every count and chart describes this browser session and is
  gone on reload.
- **No serving drift report.** Drift is computed offline; the serving
  layer publishes no live drift endpoint in this milestone.
- **Per-anchor attribution only.** Explainability answers about one
  anchor of one window, not population-level patterns.
- **Synthetic data only.** All scenarios are fabricated documentation
  fixtures. No real credential or real user activity is involved.
- **Some rules require baseline context.** Two rules (PAD-CS-001,
  PAD-ATO-001) need behavioural baselines that are not available for
  all synthetic live-replay identities.
        """.strip()
    )
