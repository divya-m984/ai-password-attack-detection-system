"""The SOC landing page: what this deployment is, right now.

Every operational value on this page comes from a live API read.  Nothing is
hard-coded, and nothing is carried over from a previous render.

The one thing it deliberately does *not* show is a global event or alert total.
There is no persistent event store and no alert database yet, so any such figure
would be invented.  Where a console would normally put "12,503 events today"
this page puts the session's own count, labelled as the session's own count.
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.alerts import render_anchor_result
from password_attack_detector.dashboard.components.metrics import (
    render_posture_cards,
    render_session_cards,
)
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
    format_reason_code,
    truncate_identifier,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title

__all__ = ["render"]


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the overview."""
    if not require_backend(status, needs_ready=False):
        return

    system = client.system_status()
    model = client.model_info()
    rules = client.rules()

    render_posture_cards(status, system.document, model.document)

    for result, name in (
        (system, "system status"),
        (model, "model information"),
        (rules, "rule catalog"),
    ):
        if result.problem is not None:
            render_problem(result.problem, context=f"Could not read the {name}.")

    left, right = st.columns([0.55, 0.45])
    with left:
        _render_system_panel(status)
        _render_architecture_panel(system.document)
    with right:
        _render_model_panel(model.document)
        _render_rule_panel(rules.document)

    st.markdown(section_title("This dashboard session"), unsafe_allow_html=True)
    render_session_cards(session.history)
    latest = session.latest
    if latest is None:
        st.info("No detection activity in this dashboard session.", icon="📭")
        st.caption(
            "Submit a window from the **Detection Console** to populate the "
            "session pages. Nothing is stored on the server."
        )
    else:
        st.caption(
            f"Most recent detection in this session (#{latest.sequence}). "
            f"Full result below."
        )
        if session.last_result is not None:
            render_anchor_result(session.last_result.anchor)


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
