"""Rule vs ML vs hybrid: what each layer is for, and what each one did.

The architecture diagram is static CSS in this module -- no external diagram
service, no image fetch, no rendering library.  A console that reached out to a
third party to draw its own architecture would be sending a description of the
deployment somewhere the operator did not choose.

The comparison table below it is per-layer and never a scoreboard.  "The rule
layer flagged nine and the model flagged seven" is a description of two
mechanisms on this session's windows; it is not an accuracy comparison, and it
could not be -- these windows carry no labels, and the one evaluation permitted
to read labels was locked in Phase 5.
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.charts import (
    layer_flag_counts,
    render_layer_agreement_chart,
)
from password_attack_detector.dashboard.components.status import (
    Connectivity,
    render_problem,
    require_backend,
)
from password_attack_detector.dashboard.contracts import SystemStatusDocument
from password_attack_detector.dashboard.formatting import (
    format_fusion_strategy,
    format_reason_code,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title

__all__ = ["render"]

#: The pipeline, drawn in static CSS.  A constant: nothing is interpolated into
#: it, so no response value can become markup.
_DIAGRAM = """
<div style="background:#131a29;border:1px solid #1f2a3d;border-radius:4px;
            padding:1.1rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
            font-size:0.82rem;color:#c9d1d9;line-height:1.85;text-align:center">
  <div style="color:#58a6ff">Authentication events</div>
  <div style="color:#8b949e">&#8595;</div>
  <div style="color:#58a6ff">Point-in-time features</div>
  <div style="color:#8b949e">&#8601;&nbsp;&nbsp;&nbsp;&#8595;&nbsp;&nbsp;&nbsp;&#8600;</div>
  <div>
    <span style="color:#f0883e">Rules</span>
    &nbsp;&nbsp;&nbsp;<span style="color:#3fb950">ML</span>
    &nbsp;&nbsp;&nbsp;<span style="color:#8b949e">evidence</span>
  </div>
  <div style="color:#8b949e">&#8600;&nbsp;&nbsp;&nbsp;&#8595;&nbsp;&nbsp;&nbsp;&#8601;</div>
  <div style="color:#58a6ff">Frozen hybrid fusion</div>
  <div style="color:#8b949e">&#8595;</div>
  <div style="color:#f85149">Alert</div>
</div>
"""


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the three-layer comparison."""
    if not require_backend(status, needs_ready=False):
        return
    result = client.system_status()
    if result.problem is not None:
        render_problem(result.problem, context="Could not read the system status.")
    system = result.document

    left, right = st.columns([0.45, 0.55])
    with left:
        st.markdown(section_title("How detection works"), unsafe_allow_html=True)
        st.markdown(_DIAGRAM, unsafe_allow_html=True)
        st.caption(
            "Every stage runs on the server. This console renders what the "
            "pipeline returned and computes no part of it."
        )
    with right:
        st.markdown(section_title("Detection layers"), unsafe_allow_html=True)
        _render_layer_table(system, session)

    _render_hybrid_panel(system)

    st.markdown(section_title("Session flag comparison"), unsafe_allow_html=True)
    if not session.history:
        st.caption(
            "No detection activity in this dashboard session. This chart "
            "populates once windows have been submitted from the Detection "
            "Console."
        )
    else:
        render_layer_agreement_chart(session.history)
        st.caption(
            "A description of what each mechanism did on this session's "
            "windows. Not an accuracy comparison: these windows carry no "
            "labels, and the one labelled evaluation was locked in Phase 5."
        )


def _render_layer_table(
    system: SystemStatusDocument | None, session: DashboardSession
) -> None:
    """Render each layer's availability, session flag count, and role."""
    counts = layer_flag_counts(session.history)
    rows = [
        {
            "Layer": "Rule-based",
            "Available": _available(system, "rule"),
            "Flagged (session)": counts["rule"],
            "Role": "Known suspicious patterns matched against authentication behaviour",
        },
        {
            "Layer": "ML-based",
            "Available": _available(system, "ml"),
            "Flagged (session)": counts["ml"],
            "Role": "Patterns learned from historical data, applied at a frozen operating point",
        },
        {
            "Layer": "Hybrid fusion",
            "Available": _available(system, "hybrid"),
            "Flagged (session)": counts["hybrid"],
            "Role": "Combines rule and ML signals into the final deployment decision",
        },
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def _available(system: SystemStatusDocument | None, layer: str) -> str:
    """Return whether a layer is running, as a word."""
    if system is None:
        return "unknown"
    enabled = {
        "rule": system.rule_detection_enabled,
        "ml": system.ml_detection_enabled,
        "hybrid": system.hybrid_detection_enabled,
    }[layer]
    return "yes" if enabled else "no"


def _render_hybrid_panel(system: SystemStatusDocument | None) -> None:
    """Render what the hybrid arm is, in this deployment, and why."""
    st.markdown(section_title("Hybrid strategy"), unsafe_allow_html=True)
    if system is None:
        st.caption("The system status could not be read.")
        return

    strategy = system.fusion_strategy
    if strategy == "stacked":
        st.success(
            "The frozen **stacked** fusion state is being served. The fitted "
            "meta-learner was reconstructed offline from pre-evaluation "
            "lineage, its fingerprint was checked against the one Phase 5 "
            "sealed, and this process verified it at startup rather than "
            "fitting anything.",
            icon="🧩",
        )
        if system.stacked_state_fingerprint:
            st.markdown(
                f"**Loaded state fingerprint**  \n`{system.stacked_state_fingerprint}`"
            )
            st.caption(
                "An identity, so an operator can confirm which stacker is live. "
                "The state's parameters are not exposed by the API and are not "
                "reachable from this console."
            )
    elif strategy in {"or_gate", "and_gate"}:
        st.success(
            f"The frozen **{format_fusion_strategy(strategy)}** is being "
            "served. A gate is a function of the two layers' booleans and needs "
            "no fitted artifact.",
            icon="🧩",
        )
    elif system.hybrid_required:
        st.error(
            "A hybrid strategy was frozen for this lineage and cannot execute: "
            f"{format_reason_code(system.fusion_unavailable_reason)}. There is "
            "no fallback strategy — substituting one would deploy a hybrid "
            "nobody selected.",
            icon="⚠️",
        )
    else:
        st.info(
            "No hybrid qualified on validation for this lineage, so none is "
            "required and none is running. That is a measured scientific "
            "outcome rather than a missing artifact.",
        )
