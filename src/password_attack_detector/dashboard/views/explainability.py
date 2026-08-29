"""Explainability: which transformed columns moved the model's decision.

This page calls ``POST /api/v1/explain``.  It does **not** import
:mod:`password_attack_detector.ml.explain`, and the distinction is not
bureaucratic: the decomposition is arithmetic over a fitted model's own
parameters, and a dashboard that could perform it would be a dashboard holding
the model.

Three things the page is careful to say, because each is a way an explanation
view routinely misleads:

* **The contributions sum to a decision value, not to a probability.**  A logit,
  a mean leaf score, or a threshold step, depending on the champion's family.
  Never a percentage, and never rendered as one.
* **The list is ranked and truncated.**  The service reports how many columns it
  omitted, and that number is shown next to the table rather than left out.
* **A contribution is not a cause.**  It describes how a fitted function
  decomposes over the columns it was handed. The wording stays flat --
  "contribution", never "driver", "importance", or "because".
"""

from __future__ import annotations

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.status import (
    Connectivity,
    render_problem,
    require_backend,
)
from password_attack_detector.dashboard.contracts import ExplanationDocument
from password_attack_detector.dashboard.formatting import (
    escape_text,
    format_reason_code,
    format_timestamp,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title

__all__ = ["render"]

_METHODS: dict[str, str] = {
    "linear_logit_contribution": (
        "Linear logit contribution — value x coefficient per column, with the "
        "intercept carried separately as the baseline. Sums to the model's logit."
    ),
    "tree_path_contribution": (
        "Tree path contribution — each split on the root-to-leaf path is "
        "credited with the change it makes to the node's stored class "
        "distribution, averaged over trees, against the ensemble-mean root value."
    ),
    "single_feature_step_contribution": (
        "Single-feature step — one reviewed column carries the whole decision, "
        "and every other column contributes exactly zero because the model does "
        "not read it."
    ),
}


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the explainability view."""
    st.markdown(section_title("Explainability"), unsafe_allow_html=True)
    st.markdown(
        '<div class="pad-intro">'
        "See which factors most influenced the model's decision for a "
        "given authentication window. The explanation is computed from "
        "the frozen model's own parameters."
        "</div>",
        unsafe_allow_html=True,
    )

    if not require_backend(status):
        return

    events = session.draft_events
    if not events:
        st.info(
            "Build a window on the **Detection Console** first. An attribution "
            "explains one anchor of one window, so there has to be a window.",
            icon="🪟",
        )
        _render_method_documentation()
        return

    st.caption(
        f"The current window carries {len(events)} event(s). The attribution "
        f"answers about its last event, the same anchor `/api/v1/detect` "
        f"scores by default."
    )
    if st.button("Explain this window's anchor", type="primary"):
        result = client.explain(events, anchor_selection="last")
        if not result.ok:
            assert result.problem is not None
            render_problem(result.problem, context="No attribution was produced.")
            return
        session.last_explanation = result.unwrap()

    if session.last_explanation is not None:
        _render_explanation(session.last_explanation)
    _render_method_documentation()


def _render_explanation(document: ExplanationDocument) -> None:
    """Render one attribution document."""
    st.markdown(section_title("Attribution"), unsafe_allow_html=True)
    st.markdown(
        f'<span class="pad-mono">anchor {escape_text(document.anchor_event_id)} · '
        f"{escape_text(format_timestamp(document.anchor_event_time))}</span>",
        unsafe_allow_html=True,
    )

    if not document.available:
        st.warning(
            "No exact decomposition is available for this champion: "
            f"{format_reason_code(document.unavailable_reason)}. An approximate "
            "per-feature number is not published in its place — it would read "
            "exactly like an exact one once it is in a table.",
        )
        return

    first, second, third = st.columns(3)
    first.metric("Decision value", f"{document.decision_value:.6f}")
    second.metric("Baseline", f"{document.baseline_value:.6f}")
    third.metric(
        "Columns decomposed",
        str(document.transformed_feature_count),
        delta=f"{document.omitted_contribution_count} not shown",
        delta_color="off",
    )
    st.caption(
        "The decision function's own quantity — not a probability, not a "
        "percentage, and not comparable with the rule layer's 0-100 ordinal "
        "magnitude."
    )

    st.dataframe(
        [
            {
                "Transformed column": item.transformed_feature,
                "Contribution": item.contribution,
                "Direction": "toward malicious"
                if item.contribution > 0
                else "away from malicious"
                if item.contribution < 0
                else "none",
            }
            for item in document.contributions
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Ranked by magnitude and bounded; "
        f"{document.omitted_contribution_count} further column(s) contributed "
        f"less and are not listed. Column names are approved engineered "
        f"features; no feature *value* is disclosed."
    )

    with st.expander("Advanced verification"):
        if document.reconstruction_residual is not None:
            st.markdown(
                f"**Reconstruction residual** `{document.reconstruction_residual:.3e}`"
            )
            st.caption(
                "decision value - (baseline + sum of the full decomposition), "
                "checked by the service against its declared tolerance before this "
                "document was built. An attribution that does not add up is refused "
                "rather than published with a caveat."
            )


def _render_method_documentation() -> None:
    """Render what the exact methods are, and what attribution does not claim."""
    st.markdown(section_title("Explanation methods"), unsafe_allow_html=True)
    for name, description in _METHODS.items():
        st.markdown(f"**{escape_text(name)}**  \n{escape_text(description)}")
    st.markdown(section_title("Important caveats"), unsafe_allow_html=True)
    st.markdown(
        '<div class="pad-note">'
        "A contribution describes how a fitted function decomposes over the "
        "columns it was handed. It does not say the behaviour caused the "
        "outcome, that changing it would change an attacker's success, or that "
        "the model is right. Families outside the exact set report the "
        "attribution unavailable rather than an approximation."
        "</div>",
        unsafe_allow_html=True,
    )
