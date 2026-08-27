"""System and model: every operational document the API publishes, in one place.

Four reads -- ``/version``, ``/api/v1/system/status``, ``/api/v1/model/info`` and
``/api/v1/rules`` -- rendered as they arrive.

The page shows the frozen operating point, because the API deliberately publishes
it: an operating point nobody can see is one nobody can audit.  It is displayed
as a number and never as a control.  There is no widget on this console that
writes a threshold, and there is no endpoint that would accept one.

What the API does not publish, this page cannot show: no coefficient, no tree
array, no HMAC key, no artifact path, no host path, and no environment.  Those
absences are structural -- the response schemas forbid extra fields, so there is
nothing here to filter out.
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
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
    VersionDocument,
)
from password_attack_detector.dashboard.formatting import (
    format_fusion_strategy,
    format_reason_code,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title

__all__ = ["render"]


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the system and model view."""
    if not require_backend(status, needs_ready=False):
        return

    version = client.version()
    system = client.system_status()
    model = client.model_info()
    rules = client.rules()
    for result, name in (
        (version, "version document"),
        (system, "system status"),
        (model, "model information"),
        (rules, "rule catalog"),
    ):
        if result.problem is not None:
            render_problem(result.problem, context=f"Could not read the {name}.")

    _render_system(version.document, system.document, status)
    _render_model(model.document)
    _render_rules(rules.document)


def _render_system(
    version: VersionDocument | None,
    system: SystemStatusDocument | None,
    status: Connectivity,
) -> None:
    """Render service versions, enabled layers, and component readiness."""
    st.markdown(section_title("System health"), unsafe_allow_html=True)
    rows: list[dict[str, str]] = []
    if version is not None:
        rows += [
            {"Field": "Service", "Value": version.service},
            {"Field": "Package version", "Value": version.package_version},
            {"Field": "API schema", "Value": version.api_schema_version},
            {"Field": "Event schema", "Value": version.event_schema_version},
            {"Field": "Feature schema", "Value": version.feature_schema_version},
            {"Field": "Detection schema", "Value": version.detection_schema_version},
            {"Field": "Scoring version", "Value": version.scoring_version},
            {"Field": "ML schema", "Value": version.ml_schema_version},
            {"Field": "Fusion schema", "Value": version.fusion_schema_version},
        ]
    if system is not None:
        rows += [
            {"Field": "Readiness", "Value": system.status},
            {
                "Field": "Rule layer",
                "Value": "enabled" if system.rule_detection_enabled else "disabled",
            },
            {
                "Field": "Model layer",
                "Value": "enabled" if system.ml_detection_enabled else "disabled",
            },
            {
                "Field": "Hybrid layer",
                "Value": "enabled" if system.hybrid_detection_enabled else "disabled",
            },
            {
                "Field": "Executing strategy",
                "Value": format_fusion_strategy(system.fusion_strategy),
            },
            {
                "Field": "Frozen strategy",
                "Value": format_fusion_strategy(system.frozen_fusion_strategy),
            },
            {
                "Field": "Hybrid required",
                "Value": "yes" if system.hybrid_required else "no",
            },
            {
                "Field": "Max events per request",
                "Value": str(system.max_batch_events),
            },
        ]
        if system.fusion_unavailable_reason:
            rows.append(
                {
                    "Field": "Hybrid unavailable",
                    "Value": format_reason_code(system.fusion_unavailable_reason),
                }
            )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Scientific lineage"):
        if system is not None and system.stacked_state_fingerprint:
            st.markdown("**Loaded stacked state fingerprint**")
            st.code(system.stacked_state_fingerprint, language="text")
            st.caption(
                "An identity for the fitted meta-learner this process loaded, so an "
                "operator can confirm which stacker is live. Its parameters are not "
                "published."
            )
    if status.readiness is not None:
        render_component_table(status.readiness)


def _render_model(model: ModelInfoDocument | None) -> None:
    """Render the frozen champion's identity, operating point, and lineage."""
    st.markdown(section_title("Model details"), unsafe_allow_html=True)
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
    with st.expander("Full model identity"):
        st.dataframe(
            [
                {"Field": "Family", "Value": model.model_family or "—"},
                {"Field": "Catalog model id", "Value": model.catalog_model_id or "—"},
                {"Field": "Model id", "Value": model.model_id or "—"},
                {"Field": "Task", "Value": model.task or "—"},
                {"Field": "Score kind", "Value": model.score_kind or "—"},
                {
                    "Field": "Calibrated",
                    "Value": "yes" if model.calibrated else "no",
                },
                {
                    "Field": "Decision threshold",
                    "Value": (
                        "—"
                        if model.decision_threshold is None
                        else f"{model.decision_threshold:.6f}"
                    ),
                },
                {
                    "Field": "Champion scope key",
                    "Value": model.champion_scope_key or "—",
                },
                {"Field": "Freeze record id", "Value": model.freeze_record_id or "—"},
                {"Field": "Training run id", "Value": model.training_run_id or "—"},
                {
                    "Field": "Validation selection id",
                    "Value": model.validation_selection_id or "—",
                },
                {
                    "Field": "Required feature schema",
                    "Value": model.required_feature_schema_version or "—",
                },
                {
                    "Field": "Category head",
                    "Value": "available" if model.category_head_available else "none",
                },
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "The threshold is published for transparency and is frozen. It cannot "
            "be changed through the API and there is no control for it on this "
            "console. No model parameter, artifact path, or environment value is "
            "published, so none can be shown."
        )


def _render_rules(rules: RuleCatalogDocument | None) -> None:
    """Render the full public rule catalog."""
    st.markdown(section_title("Detection rules"), unsafe_allow_html=True)
    if rules is None:
        st.caption("The rule catalog could not be read.")
        return
    st.caption(
        f"{rules.enabled_rule_count} of {rules.rule_count} registered rules are "
        f"enabled (detection schema {rules.detection_schema_version}). The "
        f"catalog carries no thresholds and no feature names."
    )
    st.dataframe(
        [
            {
                "Rule": rule.rule_id,
                "Version": rule.rule_version,
                "Name": rule.name,
                "Family": rule.family,
                "Category": rule.attack_category,
                "Default severity": rule.default_severity,
                "Enabled": "yes" if rule.enabled else "no",
                "Deprecated": "yes" if rule.deprecated else "no",
                "Description": rule.description,
            }
            for rule in rules.rules
        ],
        width="stretch",
        hide_index=True,
    )
