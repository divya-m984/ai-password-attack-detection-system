"""Charts, drawn from session results and from nothing else.

Streamlit's built-in charts, deliberately: they cover bars over a small
categorical axis and a count over session time, which is everything the analyst
pages actually plot.  Adding a charting library for this would be a dependency
carried for styling.

The rule every function here follows: **an empty session draws no chart.**  Not
an empty axis, not a zeroed series, not seeded example data -- an explicit
message saying there is nothing yet.  A chart with no data behind it still looks
like a measurement, and on a security console that is the failure mode worth
designing against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from password_attack_detector.dashboard.state import (
    DetectionRecord,
    severity_counts,
    triggered_rule_counts,
)

__all__ = [
    "render_layer_agreement_chart",
    "render_replay_activity_chart",
    "render_rule_frequency_chart",
    "render_session_timeline",
    "render_severity_chart",
]

_EMPTY = "Not enough detections in this session yet."


def _empty(message: str = _EMPTY) -> None:
    """Render the one empty state every chart falls back to."""
    st.caption(message)


def render_severity_chart(history: Sequence[DetectionRecord]) -> None:
    """Render detections by severity, on the Phase 4 ordinal scale's own order."""
    if not history:
        _empty()
        return
    counts = severity_counts(history)
    frame = pd.DataFrame(
        {"severity": list(counts), "detections": list(counts.values())}
    ).set_index("severity")
    st.bar_chart(frame, height=240, color="#58a6ff")


def render_rule_frequency_chart(history: Sequence[DetectionRecord]) -> None:
    """Render how often each rule fired across this session's detections."""
    counts = triggered_rule_counts(history)
    if not counts:
        _empty("No rule has fired in this session.")
        return
    frame = pd.DataFrame(
        {"rule": list(counts), "times fired": list(counts.values())}
    ).set_index("rule")
    st.bar_chart(frame, height=240, color="#f0883e")


def render_layer_agreement_chart(history: Sequence[DetectionRecord]) -> None:
    """Render how many detections each layer flagged.

    Three independent counts side by side, never a stacked or summed total: the
    layers are not parts of one quantity, and a stacked bar would suggest they
    are.
    """
    if not history:
        _empty()
        return
    counts = layer_flag_counts(history)
    frame = pd.DataFrame(
        {"layer": list(counts), "flagged": list(counts.values())}
    ).set_index("layer")
    st.bar_chart(frame, height=240, color="#3fb950")


def layer_flag_counts(history: Sequence[DetectionRecord]) -> Mapping[str, int]:
    """Return how many session detections each layer flagged.

    A layer that produced no verdict contributes nothing rather than a zero: an
    unavailable model has not "declined to flag" anything.
    """
    return {
        "rule": sum(1 for item in history if item.rule_flagged),
        "ml": sum(1 for item in history if item.ml_flagged),
        "hybrid": sum(1 for item in history if item.hybrid_flagged),
    }


def render_replay_activity_chart(history: Sequence[DetectionRecord]) -> None:
    """Render each layer's cumulative flag count across a replay run.

    Three running totals against the run's own step number, which is the shape
    that makes a replay legible: an analyst watches the moment a layer starts
    firing and how the three then diverge or agree.

    Cumulative rather than per-step because a per-step boolean plotted as a line
    is a square wave nobody can read, and cumulative counts over the *step* axis
    -- not over wall-clock time -- keep the picture identical at every pace.
    Every point is a count of records the service returned; nothing is smoothed,
    interpolated, or projected forward.
    """
    if len(history) < 2:
        _empty("A replay chart needs at least two scored steps.")
        return
    rule = ml = hybrid = 0
    steps: list[int] = []
    rules: list[int] = []
    models: list[int] = []
    hybrids: list[int] = []
    for item in history:
        rule += 1 if item.rule_flagged else 0
        ml += 1 if item.ml_flagged else 0
        hybrid += 1 if item.hybrid_flagged else 0
        steps.append(item.sequence)
        rules.append(rule)
        models.append(ml)
        hybrids.append(hybrid)
    frame = pd.DataFrame(
        {
            "step": steps,
            "rule flags": rules,
            "model flags": models,
            "hybrid flags": hybrids,
        }
    ).set_index("step")
    st.line_chart(frame, height=260, color=["#3fb950", "#58a6ff", "#f0883e"])


def render_session_timeline(history: Sequence[DetectionRecord]) -> None:
    """Render detection activity across this dashboard session.

    The x axis is the session's own sequence, not wall-clock time: submissions
    are manual and arbitrarily spaced, so a time axis would render as a few
    points with long empty stretches between them and would invite reading a
    rate into something that has none.
    """
    if len(history) < 2:
        _empty("A timeline needs at least two detections in this session.")
        return
    frame = pd.DataFrame(
        {
            "detection": [item.sequence for item in history],
            "rule risk score": [item.rule_risk_score for item in history],
        }
    ).set_index("detection")
    st.line_chart(frame, height=240, color="#58a6ff")
    st.caption(
        "Rule-layer ordinal magnitude per detection, in submission order. "
        "The model's score is on a different scale and is not plotted with it."
    )
