"""The compact cards the pages open with.

One function per card row, so a page states *what* it wants shown and this module
decides how.  The rule the whole file exists to keep: a card shows a value the
service reported, or it shows an em dash.  There is no third case -- no default,
no last-known-good, no zero standing in for "not measured" -- because on a
security console those are indistinguishable from a real reading of zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from password_attack_detector.dashboard.components.status import Connectivity
from password_attack_detector.dashboard.contracts import (
    ModelInfoDocument,
    SystemStatusDocument,
)
from password_attack_detector.dashboard.formatting import (
    SEVERITY_COLORS,
    format_fusion_strategy,
    format_fusion_strategy_short,
    format_model_family,
)
from password_attack_detector.dashboard.state import DetectionRecord
from password_attack_detector.dashboard.theme import card

__all__ = ["render_posture_cards", "render_session_cards"]

_GREEN = "#3fb950"
_RED = "#f85149"
_AMBER = "#d29922"
_MUTED = "#8b949e"
_ACCENT = "#58a6ff"


def render_posture_cards(
    status: Connectivity,
    system: SystemStatusDocument | None,
    model: ModelInfoDocument | None,
) -> None:
    """Render the six cards describing what this deployment currently is."""
    columns = st.columns(6)
    with columns[0]:
        st.markdown(
            card(
                "System",
                "Online" if status.online else "Offline",
                accent=_GREEN if status.online else _RED,
            ),
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            card(
                "Readiness",
                "Ready" if status.ready else "Not ready",
                accent=_GREEN if status.ready else _RED,
                note=(
                    ""
                    if status.ready or status.readiness is None
                    else f"{len(status.readiness.blocking)} blocking"
                ),
            ),
            unsafe_allow_html=True,
        )
    layers = _active_layers(system)
    with columns[2]:
        st.markdown(
            card(
                "Detection",
                f"{len(layers)} layers active" if system is not None else "—",
                accent=_ACCENT,
            ),
            unsafe_allow_html=True,
        )
    with columns[3]:
        st.markdown(
            card(
                "Hybrid strategy",
                format_fusion_strategy_short(
                    None if system is None else system.fusion_strategy
                ),
                accent=_fusion_accent(system),
            ),
            unsafe_allow_html=True,
        )
    with columns[4]:
        st.markdown(
            card(
                "Rules",
                "—" if system is None else f"{system.enabled_rule_count} configured",
                accent=_ACCENT,
            ),
            unsafe_allow_html=True,
        )
    with columns[5]:
        available = model is not None and model.available
        st.markdown(
            card(
                "Model",
                format_model_family(model.model_family)
                if available and model is not None
                else "Unavailable",
                accent=_ACCENT if available else _RED,
            ),
            unsafe_allow_html=True,
        )


def _active_layers(system: SystemStatusDocument | None) -> list[str]:
    """Return the names of the detection layers this deployment is running."""
    if system is None:
        return []
    names = []
    if system.rule_detection_enabled:
        names.append("rules")
    if system.ml_detection_enabled:
        names.append("model")
    if system.hybrid_detection_enabled:
        names.append("hybrid")
    return names


def _fusion_accent(system: SystemStatusDocument | None) -> str:
    """Return the accent for the fusion card.

    Amber, not red, when a hybrid was frozen and cannot run: it is a real fault,
    but the service has already refused to call itself ready, and painting the
    same fact red twice on one row says nothing extra. Muted when none was
    frozen, because that is a measured outcome rather than a fault.
    """
    if system is None:
        return _MUTED
    if system.hybrid_detection_enabled:
        return _GREEN
    return _AMBER if system.hybrid_required else _MUTED


def _fusion_note(system: SystemStatusDocument | None) -> str:
    """Return the fusion card's note, distinguishing absence from failure."""
    if system is None:
        return ""
    if system.hybrid_detection_enabled:
        return "frozen selection, executing"
    if system.hybrid_required:
        frozen = format_fusion_strategy(system.frozen_fusion_strategy)
        return f"{frozen} was selected and cannot run"
    return "no hybrid qualified on validation"


def render_session_cards(history: Sequence[DetectionRecord]) -> None:
    """Render the counts describing what happened in *this* browser session.

    Labelled as session counts everywhere they appear. They are not a measure of
    traffic, of alerts raised, or of anything that happened on the server: they
    count the windows somebody submitted from this tab.
    """
    columns = st.columns(4)
    flagged = [item for item in history if item.any_layer_flagged]
    worst = _worst_severity(history)
    with columns[0]:
        st.markdown(
            card("Detections", str(len(history)), accent=_ACCENT),
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            card(
                "Flagged",
                str(len(flagged)),
                accent=_AMBER if flagged else _GREEN,
            ),
            unsafe_allow_html=True,
        )
    with columns[2]:
        st.markdown(
            card(
                "Highest severity",
                worst or "—",
                accent=SEVERITY_COLORS.get(worst or "none", _MUTED),
            ),
            unsafe_allow_html=True,
        )
    with columns[3]:
        strategies = {item.hybrid_strategy for item in history if item.hybrid_strategy}
        st.markdown(
            card(
                "Strategy",
                format_fusion_strategy_short(next(iter(strategies)))
                if strategies
                else "—",
                accent=_ACCENT if strategies else _MUTED,
            ),
            unsafe_allow_html=True,
        )


def _worst_severity(history: Sequence[DetectionRecord]) -> str | None:
    """Return the highest severity in *history*, on the Phase 4 ordinal scale."""
    from password_attack_detector.dashboard.formatting import severity_rank

    if not history:
        return None
    return max((item.severity for item in history), key=severity_rank)
