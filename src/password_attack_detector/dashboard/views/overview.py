"""A calm, approachable landing page for the SOC console.

Every operational value on this page comes from a live API read.  Nothing is
hard-coded, and nothing is carried over from a previous render.

The page is structured for a first-time viewer: a handful of status cards in
plain language, a single call to action, a short explanation of what the system
does, a session activity summary, and then an expander for the technical details
an operator might want.

It renders no title of its own.  The global header above it already states the
product name and what it is, and a landing page that repeated both would put the
same sentence on the screen twice before the first number.
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.alerts import render_anchor_result
from password_attack_detector.dashboard.components.metrics import render_session_cards
from password_attack_detector.dashboard.components.replay import render_run_banner
from password_attack_detector.dashboard.components.status import (
    Connectivity,
    render_component_table,
    render_problem,
    require_backend,
)
from password_attack_detector.dashboard.contracts import (
    ModelInfoDocument,
    RuleCatalogDocument,
    SystemStatusDocument,
)
from password_attack_detector.dashboard.formatting import (
    escape_text,
    format_fusion_strategy,
    format_fusion_strategy_short,
    format_model_family,
    format_reason_code,
    truncate_identifier,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import card, section_title

__all__ = ["render"]


# ---------------------------------------------------------------------------
# Primary status cards
# ---------------------------------------------------------------------------


def _count_active_layers(system: SystemStatusDocument | None) -> str:
    """Return a human-readable count of running detection layers."""
    if system is None:
        return "unknown"
    count = sum(
        [
            system.rule_detection_enabled,
            system.ml_detection_enabled,
            system.hybrid_detection_enabled,
        ]
    )
    if count == 1:
        return "1 layer active"
    return f"{count} layers active"


def _system_readiness(status: Connectivity) -> str:
    """Return 'Ready' or 'Not ready' from the connectivity state."""
    if status.ready:
        return "Ready"
    return "Not ready"


def _demo_readiness(system: SystemStatusDocument | None) -> str:
    """Return a short demo-readiness label."""
    if system is None:
        return "Unknown"
    if system.replay_available:
        return "Ready"
    if system.replay_enabled:
        return "Not available"
    return "Not offered"


def _render_status_cards(
    status: Connectivity,
    system: SystemStatusDocument | None,
    model: ModelInfoDocument | None,
    rules: RuleCatalogDocument | None,
) -> None:
    """Render the primary status cards: compact, human-readable values."""
    cols = st.columns(6)
    with cols[0]:
        ready = _system_readiness(status)
        accent = "#3fb950" if ready == "Ready" else "#f85149"
        st.markdown(card("System", ready, accent=accent), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            card("Detection", _count_active_layers(system)),
            unsafe_allow_html=True,
        )
    with cols[2]:
        strategy = system.frozen_fusion_strategy if system else None
        st.markdown(
            card("Fusion", format_fusion_strategy_short(strategy)),
            unsafe_allow_html=True,
        )
    with cols[3]:
        rule_label = f"{rules.enabled_rule_count} configured" if rules else "unknown"
        st.markdown(card("Rules", rule_label), unsafe_allow_html=True)
    with cols[4]:
        family = model.model_family if model else None
        st.markdown(
            card("Model", format_model_family(family)),
            unsafe_allow_html=True,
        )
    with cols[5]:
        demo = _demo_readiness(system)
        demo_accent = "#3fb950" if demo == "Ready" else "#8b949e"
        st.markdown(card("Demo", demo, accent=demo_accent), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Explanation section
# ---------------------------------------------------------------------------

#: The landing page's explanation, as a fragment of the styled intro block.
#:
#: **This is HTML, not Markdown.** It is rendered inside ``pad-intro`` with
#: ``unsafe_allow_html``, and Streamlit does not run the Markdown parser over a
#: block it is given as raw HTML -- so ``**rule engine**`` in this string would
#: reach the page as four literal asterisks. The emphasis is ``<strong>``.
#: A test asserts no ``**`` survives in any raw-HTML block this page emits.
#:
#: Safe to interpolate because it is a module constant: no API value, and no
#: value a viewer typed, is any part of it.
_WHAT_THIS_SYSTEM_DOES = (
    "This system detects password attacks using three complementary layers. "
    "A <strong>rule engine</strong> matches known attack patterns such as "
    "brute force, credential stuffing and password spraying. A "
    "<strong>machine-learning model</strong> scores authentication windows "
    "for anomalies the rules do not cover. A <strong>hybrid fusion layer</strong> "
    "combines both verdicts into a single decision calibrated on validation "
    "data. Each layer can operate independently, and the dashboard shows "
    "exactly what each one contributed to every verdict."
)


# ---------------------------------------------------------------------------
# Recent activity
# ---------------------------------------------------------------------------


def _render_recent_activity(session: DashboardSession) -> None:
    """Show a compact summary of session detections, or a friendly empty state."""
    st.markdown(section_title("Recent activity"), unsafe_allow_html=True)
    render_session_cards(session.history)
    latest = session.latest
    if latest is None:
        st.info("No activity in this browser session yet.", icon="📭")
        st.caption(
            "Submit a window from the **Detection Console** or run a "
            "**Live Replay** scenario to see results here."
        )
    else:
        st.caption(
            f"Most recent detection in this session (#{latest.sequence}). "
            f"Full result below."
        )
        if session.last_result is not None:
            render_anchor_result(session.last_result.anchor)


# ---------------------------------------------------------------------------
# Advanced details (inside an expander)
# ---------------------------------------------------------------------------


def _render_system_panel(status: Connectivity) -> None:
    """Render readiness and the component breakdown."""
    st.markdown(section_title("System status"), unsafe_allow_html=True)
    if status.readiness is None:
        st.caption("Readiness could not be read.")
        return
    render_component_table(status.readiness)


def _render_architecture_panel(system: SystemStatusDocument | None) -> None:
    """Render what the detection architecture is doing in this deployment."""
    st.markdown(section_title("Detection architecture"), unsafe_allow_html=True)
    if system is None:
        st.caption("System status could not be read.")
        return
    rows = [
        ("Rule layer", "running" if system.rule_detection_enabled else "not running"),
        ("Model layer", "running" if system.ml_detection_enabled else "not running"),
        (
            "Hybrid fusion",
            "running" if system.hybrid_detection_enabled else "not running",
        ),
        ("Frozen strategy", format_fusion_strategy(system.frozen_fusion_strategy)),
        ("Hybrid required", "yes" if system.hybrid_required else "no"),
        ("Max events per request", str(system.max_batch_events)),
    ]
    st.dataframe(
        [{"Component": name, "State": value} for name, value in rows],
        width="stretch",
        hide_index=True,
    )
    if system.hybrid_required and not system.hybrid_detection_enabled:
        st.warning(
            "A hybrid strategy was frozen for this deployment and cannot "
            f"execute: {format_reason_code(system.fusion_unavailable_reason)}. "
            "The service correctly reports itself not ready.",
            icon="⚠️",
        )
    elif not system.hybrid_required:
        st.caption(
            "No hybrid qualified on validation for this lineage. That is a "
            "measured scientific outcome, not a missing artifact."
        )


def _render_model_panel(model: ModelInfoDocument | None) -> None:
    """Render the champion summary."""
    st.markdown(section_title("Champion model"), unsafe_allow_html=True)
    if model is None:
        st.caption("Model information could not be read.")
        return
    if not model.available:
        st.warning(
            f"No frozen champion is loaded: "
            f"{format_reason_code(model.unavailable_reason)}",
            icon="⚠️",
        )
        return
    st.dataframe(
        [
            {"Field": "Family", "Value": model.model_family or "—"},
            {"Field": "Task", "Value": model.task or "—"},
            {"Field": "Score kind", "Value": model.score_kind or "—"},
            {
                "Field": "Calibrated",
                "Value": "yes" if model.calibrated else "no",
            },
            {
                "Field": "Model id",
                "Value": truncate_identifier(model.model_id or "—", keep=20),
            },
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Identity and decision semantics only. The API publishes no model "
        "parameters and no artifact location."
    )


def _render_rule_panel(rules: RuleCatalogDocument | None) -> None:
    """Render the rule catalog summary."""
    st.markdown(section_title("Rule catalog"), unsafe_allow_html=True)
    if rules is None:
        st.caption("The rule catalog could not be read.")
        return
    st.markdown(
        f'<div class="pad-note">'
        f"{escape_text(rules.enabled_rule_count)} of "
        f"{escape_text(rules.rule_count)} registered rules are enabled in this "
        f"deployment (detection schema "
        f"{escape_text(rules.detection_schema_version)}).</div>",
        unsafe_allow_html=True,
    )
    families: dict[str, int] = {}
    for rule in rules.rules:
        if rule.enabled:
            families[rule.family] = families.get(rule.family, 0) + 1
    if families:
        st.dataframe(
            [
                {"Family": name, "Enabled rules": count}
                for name, count in sorted(families.items())
            ],
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Replay panel (unchanged contract)
# ---------------------------------------------------------------------------


def _render_replay_panel(session: DashboardSession) -> None:
    """Render the demo run this session is following, when there is one.

    Only when there is one. An overview that always carried a "replay: idle" card
    would be spending a landing-page slot on the absence of an optional
    demonstration facility.
    """
    replay = session.replay
    if not replay.attached or replay.run is None:
        return
    st.markdown(section_title("Attached demo replay run"), unsafe_allow_html=True)
    render_run_banner(replay)
    summary = replay.run.summary
    st.caption(
        f"{summary.detection_count} step(s) scored · "
        f"{summary.rule_flagged_count} rule-flagged · "
        f"{summary.ml_flagged_count} model-flagged · "
        f"{summary.hybrid_flagged_count} hybrid-flagged · "
        f"worst severity {summary.highest_severity or '—'}. "
        f"Open **Live Replay** for the timeline."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the overview."""
    if not require_backend(status, needs_ready=False):
        return

    # -- Fetch live data ----------------------------------------------------
    #
    # No title block. The product name and its one-line description are the
    # global header's job, and this page renders directly below it: a second
    # near-identical heading would say the same thing twice on one screen.

    system = client.system_status()
    model = client.model_info()
    rules = client.rules()

    for result, name in (
        (system, "system status"),
        (model, "model information"),
        (rules, "rule catalog"),
    ):
        if result.problem is not None:
            render_problem(result.problem, context=f"Could not read the {name}.")

    # -- Primary status cards -----------------------------------------------

    _render_status_cards(status, system.document, model.document, rules.document)

    # -- Primary CTA --------------------------------------------------------

    # Still the only primary-styled control on the page, but held to about a
    # fifth of the width rather than two fifths: it should read as the obvious
    # next step, not as the page's masthead.
    st.markdown("")
    _left, _cta, _right = st.columns([0.4, 0.2, 0.4])
    with _cta:
        if st.button(
            "Start a live demo",
            type="primary",
            use_container_width=True,
            help="Opens the Live Replay page where you can run a detection scenario.",
        ):
            st.session_state["_pad_nav_target"] = "Live Replay"
            st.rerun()

    # -- What this system does ----------------------------------------------

    st.markdown(section_title("What this system does"), unsafe_allow_html=True)
    st.markdown(
        f'<div class="pad-intro">{_WHAT_THIS_SYSTEM_DOES}</div>', unsafe_allow_html=True
    )

    # -- Recent activity ----------------------------------------------------

    _render_recent_activity(session)

    # -- Replay panel (when attached) ---------------------------------------

    _render_replay_panel(session)

    # -- Technical details (collapsed) --------------------------------------

    with st.expander("Advanced system details"):
        left, right = st.columns([0.55, 0.45])
        with left:
            _render_system_panel(status)
            _render_architecture_panel(system.document)
        with right:
            _render_model_panel(model.document)
            _render_rule_panel(rules.document)
