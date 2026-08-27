"""Rendering a detection result, with the three layers kept visibly apart.

The layout is the argument.  Rule, model and hybrid get one column each, at the
same width, with their own headings -- because they are three separate verdicts
about the same anchor, produced by three separate mechanisms, and the moment they
share a cell somebody reads them as one number.

Four things this module will not do, each of which would be easy and wrong:

* **No arithmetic.**  Nothing here averages, weights, or blends the rule layer's
  ordinal magnitude with the model's probability.  They are quantities on
  different scales; a combined figure would have units nobody defined.  The
  fused verdict is the server's, produced by the frozen strategy.
* **No renaming.**  A decision score is called a decision score.  The label comes
  from the ``score_kind`` the service declared, via
  :func:`~password_attack_detector.dashboard.formatting.format_model_score`.
* **No threshold arithmetic.**  The frozen operating point is displayed as a
  number the service published.  Nothing here compares it to a score to re-derive
  a flag, or infers where it sits from a set of observed decisions.
* **No fabricated severity.**  The severity shown is the Phase 4 ordinal the
  service returned for the anchor.  It is not recomputed from the layers.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from password_attack_detector.dashboard.contracts import AnchorDetection
from password_attack_detector.dashboard.formatting import (
    SEVERITY_COLORS,
    escape_text,
    format_model_score,
    format_reason_code,
    format_risk_score,
    format_timestamp,
    format_verdict,
)
from password_attack_detector.dashboard.state import DetectionRecord
from password_attack_detector.dashboard.theme import badge, chip, section_title

__all__ = [
    "render_alert_history",
    "render_anchor_result",
    "render_final_assessment",
]

_GREEN = "#3fb950"
_MUTED = "#8b949e"


def render_anchor_result(anchor: AnchorDetection) -> None:
    """Render one anchor's three-layer verdict, then the overall assessment."""
    st.markdown(section_title("Detection result"), unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        f'<span class="pad-mono">anchor {escape_text(anchor.anchor_event_id)} · '
        f"{escape_text(format_timestamp(anchor.anchor_event_time))}</span>",
        unsafe_allow_html=True,
    )
    rule_col, ml_col, hybrid_col = st.columns(3)
    with rule_col:
        _render_rule_layer(anchor)
    with ml_col:
        _render_ml_layer(anchor)
    with hybrid_col:
        _render_hybrid_layer(anchor)
    render_final_assessment(anchor)


def _render_rule_layer(anchor: AnchorDetection) -> None:
    """Render the Phase 4 rule verdict."""
    rule = anchor.rule
    st.markdown("**Rule detection**")
    st.markdown(
        badge(format_verdict(rule.flagged), color=_flag_color(rule.flagged)),
        unsafe_allow_html=True,
    )
    st.metric("Risk score", format_risk_score(rule.risk_score))
    st.caption("Ordinal severity magnitude on a 0-100 scale. Not a probability.")
    st.markdown(f"**Severity** {escape_text(rule.severity)}")
    if rule.primary_attack_category:
        st.markdown(f"**Category** {escape_text(rule.primary_attack_category)}")
    if rule.fired_rule_ids:
        st.markdown(
            "".join(chip(item) for item in rule.fired_rule_ids),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No rule fired.")
    if rule.insufficient_data_count:
        st.caption(
            f"{rule.insufficient_data_count} rule(s) could not see the history "
            f"they need. Distinct from a clean negative."
        )
    if rule.evidence:
        with st.expander(f"Evidence ({len(rule.evidence)})"):
            st.dataframe(
                [
                    {
                        "Code": item.evidence_code,
                        "Observation": item.message,
                        "Value": item.observed_value,
                    }
                    for item in rule.evidence
                ],
                width="stretch",
                hide_index=True,
            )


def _render_ml_layer(anchor: AnchorDetection) -> None:
    """Render the frozen champion's verdict."""
    ml = anchor.ml
    st.markdown("**ML detection**")
    if not ml.available:
        st.markdown(badge("unavailable", color="#f85149"), unsafe_allow_html=True)
        st.caption(format_reason_code(ml.unavailable_reason))
        return
    st.markdown(
        badge(format_verdict(ml.flagged), color=_flag_color(ml.flagged)),
        unsafe_allow_html=True,
    )
    label, value = format_model_score(
        score_kind=ml.score_kind,
        decision_score=ml.decision_score,
        probability=ml.probability,
    )
    st.metric(label, value)
    st.caption(
        "Reported under the score kind the frozen operating point was selected against."
    )
    if ml.decision_threshold is not None:
        st.markdown(
            f'**Frozen operating point** <span class="pad-mono">'
            f"{ml.decision_threshold:.6f}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Frozen before deployment. Not settable from this console.")


def _render_hybrid_layer(anchor: AnchorDetection) -> None:
    """Render the fused verdict, or why there is not one."""
    hybrid = anchor.hybrid
    st.markdown("**Hybrid detection**")
    if not hybrid.available:
        st.markdown(badge("unavailable", color=_MUTED), unsafe_allow_html=True)
        st.caption(format_reason_code(hybrid.unavailable_reason))
        return
    st.markdown(
        badge(format_verdict(hybrid.flagged), color=_flag_color(hybrid.flagged)),
        unsafe_allow_html=True,
    )
    from password_attack_detector.dashboard.formatting import format_fusion_strategy

    st.markdown(f"**Strategy** {escape_text(format_fusion_strategy(hybrid.strategy))}")
    if hybrid.strategy == "stacked":
        st.caption(
            "The frozen stacked fusion state, materialized offline and verified "
            "at startup. Its parameters are not exposed."
        )
    else:
        st.caption("The strategy validation selected before the locked evaluation.")


def render_final_assessment(anchor: AnchorDetection) -> None:
    """Render the overall assessment, in words, without inventing a verdict.

    Describes what the system decided and which layer decided it. Where the
    layers disagree, the disagreement is stated rather than resolved: resolving
    it is the fusion strategy's job, it has already been done on the server, and
    a second opinion rendered here would be the dashboard detecting.
    """
    st.markdown(section_title("Security assessment"), unsafe_allow_html=True)
    severity = anchor.severity
    color = SEVERITY_COLORS.get(severity, _MUTED)
    st.markdown(badge(f"severity: {severity}", color=color), unsafe_allow_html=True)
    st.markdown(
        f'<div class="pad-note">{escape_text(_narrative(anchor))}</div>',
        unsafe_allow_html=True,
    )


def _narrative(anchor: AnchorDetection) -> str:
    """Return a plain-language account of what the system decided."""
    parts: list[str] = []
    if anchor.rule.flagged:
        fired = ", ".join(anchor.rule.fired_rule_ids) or "at least one rule"
        parts.append(
            f"The rule layer flagged this anchor ({fired}) at severity "
            f"{anchor.severity}."
        )
    else:
        parts.append("No rule fired on this anchor.")

    if not anchor.ml.available:
        parts.append("The model layer produced no verdict for this deployment.")
    elif anchor.ml.flagged:
        parts.append("The frozen model flagged it at its frozen operating point.")
    else:
        parts.append("The frozen model did not flag it at its frozen operating point.")

    if not anchor.hybrid.available:
        parts.append(
            "No hybrid verdict is available, so the two layers stand on their own."
        )
    elif anchor.hybrid.flagged:
        parts.append(
            f"The frozen {anchor.hybrid.strategy} fusion, which is what this "
            f"deployment decides on, flagged it."
        )
    else:
        parts.append(
            f"The frozen {anchor.hybrid.strategy} fusion, which is what this "
            f"deployment decides on, did not flag it."
        )
    return " ".join(parts)


def _flag_color(flagged: bool | None) -> str:
    """Return the accent for a verdict, with absence distinct from a negative."""
    if flagged is None:
        return _MUTED
    return "#f85149" if flagged else _GREEN


def render_alert_history(history: Sequence[DetectionRecord]) -> None:
    """Render this session's results as a table, newest first.

    Dashboard-safe metadata only. The sequence number is local to this browser
    session and is explicitly not a server-side alert identifier -- there is no
    server-side alert store yet, and numbering these as though there were would
    be the single most misleading thing this console could do.
    """
    if not history:
        st.info(
            "No detection activity in this dashboard session.",
            icon="📭",
        )
        return
    st.dataframe(
        [
            {
                "#": item.sequence,
                "Observed (local)": format_timestamp(item.observed_at),
                "Anchor time": format_timestamp(item.anchor_event_time),
                "Scenario": item.scenario,
                "Severity": item.severity,
                "Rule": _cell(item.rule_flagged),
                "ML": _cell(item.ml_flagged),
                "Hybrid": _cell(item.hybrid_flagged),
                "Strategy": item.hybrid_strategy or "—",
                "Rules fired": ", ".join(item.fired_rule_ids) or "—",
            }
            for item in reversed(history)
        ],
        width="stretch",
        hide_index=True,
    )


def _cell(flagged: bool | None) -> str:
    """Return a compact table cell for one layer's verdict."""
    if flagged is None:
        return "—"
    return "flagged" if flagged else "clear"
