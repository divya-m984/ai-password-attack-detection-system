"""Live Replay: watch a synthetic scenario go through the real detector.

This is the page the whole console is for.  Every other view shows a state or a
result; this one shows the *sequence* -- rules starting to fire as a burst
lengthens, the model's score moving with it, the frozen hybrid fusing the two --
which is the thing a static screenshot of a detection system cannot convey.

Three properties it is built around.

**The run happens on the server.**  This page starts one, polls it, and stops it.
It does not schedule anything, hold a timer that survives it, or compute a single
value the service did not return.  A browser reload loses the *view* and not the
run; another tab polling the same identifier sees the same timeline.

**Polling is bounded and it ends.**  There is no ``while`` loop and no busy wait.
While a run is active the timeline block is a Streamlit fragment with a fixed
:data:`LIVE_POLL_SECONDS` interval; the moment the service reports the run
terminal the fragment is not used at all and the page stops calling the backend
until somebody asks it to.  The interval is a constant tied to the *pace*
vocabulary rather than to the console's health-refresh preference, because what
it has to keep up with is a replay step and not an operator's idea of how often
to re-read a status.

**Nothing starts or stops without a press.**  Start and Stop are buttons outside
the polling fragment.  A run identifier is written into session state the instant
one is created, and the Start control is disabled while a run is attached and
active, so a page that reruns -- which Streamlit does constantly -- cannot start
a second run behind the first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.alerts import render_anchor_result
from password_attack_detector.dashboard.components.charts import (
    render_replay_activity_chart,
)
from password_attack_detector.dashboard.components.status import (
    Connectivity,
    render_problem,
    require_backend,
)
from password_attack_detector.dashboard.contracts import (
    ReplayRunDocument,
    ScenarioDocument,
    SystemStatusDocument,
)
from password_attack_detector.dashboard.formatting import (
    SEVERITY_COLORS,
    escape_text,
    format_fusion_strategy,
    format_reason_code,
    format_risk_score,
    format_timestamp,
    format_verdict,
    truncate_identifier,
)
from password_attack_detector.dashboard.state import DashboardSession, ReplaySession
from password_attack_detector.dashboard.theme import badge, card, chip, section_title

__all__ = ["LIVE_POLL_SECONDS", "PACES", "poll_interval", "render"]

#: How often the live timeline block re-polls while a run is active.
#:
#: A constant, and deliberately not the console's ``refresh_seconds`` setting:
#: that one is a preference about how often to re-read a *health* document, and
#: this one has to keep up with a replay step. The service's slowest pace puts
#: 1.75 s between steps, so polling a little faster than that shows every step
#: without ever making the console a load source -- one small ``GET`` carrying
#: only what the console has not already seen.
LIVE_POLL_SECONDS: Final[float] = 1.5

#: How many timeline records one poll asks for. Comfortably more than a run can
#: produce between two polls at any pace, so the console never falls behind.
POLL_LIMIT: Final[int] = 100

#: The pace vocabulary, in the order the control offers it. Restated here rather
#: than imported: this package talks to the API over a socket. A test pins the
#: list against the service's own enumeration.
PACES: Final[tuple[str, ...]] = ("instant", "fast", "normal", "slow")

#: What each pace means, for the viewer choosing one.
_PACE_NOTES: Final[dict[str, str]] = {
    "instant": "No delay. The whole run completes as fast as it can be scored.",
    "fast": "About a quarter-second per step.",
    "normal": "About three-quarters of a second per step.",
    "slow": "About one and three-quarter seconds per step, for narrating over.",
}

_MUTED: Final[str] = "#8b949e"
_GREEN: Final[str] = "#3fb950"
_BLUE: Final[str] = "#58a6ff"
_RED: Final[str] = "#f85149"

#: Accent per run state. Terminal-but-fine states read neutral rather than green:
#: a stopped run is neither healthy nor broken, it is what somebody asked for.
_STATE_COLORS: Final[dict[str, str]] = {
    "created": _MUTED,
    "running": _BLUE,
    "completed": _GREEN,
    "stopped": _MUTED,
    "failed": _RED,
}


def poll_interval(replay: ReplaySession) -> float | None:
    """Return how often to poll, or ``None`` when polling should stop.

    The whole of the console's polling policy, extracted so it can be tested
    without Streamlit running. ``None`` for a run that is not attached and for
    one the service has reported terminal -- which is what makes "polling stops
    at a terminal state" a property rather than an intention.
    """
    if not replay.attached or not replay.active:
        return None
    return LIVE_POLL_SECONDS


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the Live Replay page."""
    st.markdown(section_title("Live replay"), unsafe_allow_html=True)
    st.info(
        "Replays a **built-in synthetic scenario** through the same detection "
        "path as the Detection Console: one fabricated event at a time, into "
        "the running service, scored by the frozen rules, model and hybrid. "
        "Nothing here contacts an external system, and no scenario can be "
        "uploaded or edited.",
    )
    if not require_backend(status, needs_ready=False):
        return

    system = client.system_status()
    if system.problem is not None:
        render_problem(system.problem, context="Could not read the system status.")
    if not _replay_offered(system.document):
        return

    catalog = client.demo_scenarios()
    if catalog.problem is not None:
        render_problem(catalog.problem, context="Could not read the scenario catalog.")
        return
    scenarios = catalog.unwrap().scenarios
    if not scenarios:
        st.warning("The service published an empty scenario catalog.", icon="⚠️")
        return

    _render_controls(client, session, scenarios, ready=status.ready)
    _render_run(client, session, scenarios)


def _replay_offered(system: SystemStatusDocument | None) -> bool:
    """Render why replay is not usable here, and return whether it is."""
    if system is None:
        st.caption("The system status could not be read, so replay cannot be offered.")
        return False
    if system.replay_available:
        return True
    if not system.replay_enabled:
        st.info(
            "This deployment does not serve the demonstration replay endpoints. "
            "That is a configuration choice and not a fault: detection is "
            "unaffected, and every other view on this console works normally.",
        )
        return False
    st.error(
        "The replay subsystem is enabled on this deployment and is not "
        f"available: {format_reason_code(system.replay_unavailable_reason)}.",
        icon="⚠️",
    )
    return False


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _render_controls(
    client: DashboardAPIClient,
    session: DashboardSession,
    scenarios: Sequence[ScenarioDocument],
    *,
    ready: bool,
) -> None:
    """Render the scenario picker, the pace picker, and the three buttons."""
    replay = session.replay
    running = replay.attached and replay.active

    left, right = st.columns([0.58, 0.42])
    with left:
        chosen = st.selectbox(
            "Scenario",
            options=[item.scenario_id for item in scenarios],
            format_func=lambda value: _label(scenarios, value),
            disabled=running,
            help="Built-in and reviewed. Scenarios cannot be uploaded or edited.",
        )
        replay.selected_scenario = str(chosen)
    with right:
        pace = st.selectbox(
            "Pace",
            options=PACES,
            index=PACES.index(replay.selected_pace)
            if replay.selected_pace in PACES
            else PACES.index("normal"),
            disabled=running,
            help=(
                "Presentation timing only. The scenario's event times and every "
                "verdict are identical at every pace."
            ),
        )
        replay.selected_pace = str(pace)
        st.caption(_PACE_NOTES.get(str(pace), ""))

    _render_scenario_brief(_find(scenarios, replay.selected_scenario))

    start, stop, refresh = st.columns(3)
    with start:
        if st.button(
            "▶  Start replay",
            width="stretch",
            type="primary",
            disabled=running or not ready,
            help=(
                "Starts one run on the server. Disabled while a run is active, "
                "so this page cannot start a second one behind the first."
            ),
        ):
            _start(client, session)
    with stop:
        if st.button(
            "■  Stop replay",
            width="stretch",
            disabled=not running,
            help="Stops this run and no other. Records already emitted are kept.",
        ):
            _stop(client, session)
    with refresh:
        if st.button(
            "↻  Refresh",
            width="stretch",
            disabled=not replay.attached,
            help="Re-reads the run and any timeline records not yet fetched.",
        ):
            _poll(client, session)
            st.rerun()

    if not ready:
        st.warning(
            "The service is running but not ready to serve detection, so no "
            "replay can be started. The **System & Model** view names the "
            "component that is missing.",
            icon="⚠️",
        )


def _label(scenarios: Sequence[ScenarioDocument], scenario_id: str) -> str:
    """Return the display label for one scenario identifier."""
    found = _find(scenarios, scenario_id)
    return (
        scenario_id if found is None else f"{found.name}  ({found.event_count} events)"
    )


def _find(
    scenarios: Sequence[ScenarioDocument], scenario_id: str
) -> ScenarioDocument | None:
    """Return the catalog entry with this identifier, or ``None``."""
    return next((item for item in scenarios if item.scenario_id == scenario_id), None)


def _render_scenario_brief(scenario: ScenarioDocument | None) -> None:
    """Render what the selected scenario is, and what it is honestly good for."""
    if scenario is None:
        return
    st.markdown(
        f'<div class="pad-note">{escape_text(scenario.description)}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("What this scenario demonstrates"):
        st.markdown(f"**Purpose.** {escape_text(scenario.purpose)}")
        if scenario.expected_rule_ids:
            st.markdown(
                "**Rules the service expects to fire.** "
                + " ".join(chip(item) for item in scenario.expected_rule_ids),
                unsafe_allow_html=True,
            )
            st.caption(
                "The service's own stated expectation, backed by its own tests. "
                "This console renders it beside the run's actual result and "
                "never checks one against the other."
            )
        else:
            st.markdown(
                "**Rules the service expects to fire.** None — this scenario "
                "demonstrates an absence."
            )
        st.markdown(
            f"**Simulated span.** {scenario.duration_seconds:.0f} s across "
            f"{scenario.event_count} events. "
            f"**Content fingerprint.** `{escape_text(truncate_identifier(scenario.scenario_fingerprint, keep=16))}`"
        )
        if scenario.limitations:
            st.warning(scenario.limitations, icon="⚠️")


def _start(client: DashboardAPIClient, session: DashboardSession) -> None:
    """Start one run, on one press, and attach this session to it."""
    replay = session.replay
    result = client.start_demo_run(replay.selected_scenario, pace=replay.selected_pace)
    if not result.ok:
        assert result.problem is not None
        render_problem(result.problem, context="The replay run was not started.")
        return
    replay.attach(result.unwrap())
    st.rerun()


def _stop(client: DashboardAPIClient, session: DashboardSession) -> None:
    """Stop the attached run, and record the final state the service reported."""
    replay = session.replay
    if replay.run_id is None:  # pragma: no cover - the control is disabled
        return
    result = client.stop_demo_run(replay.run_id)
    if not result.ok:
        assert result.problem is not None
        render_problem(result.problem, context="The replay run was not stopped.")
        return
    replay.observe(result.unwrap())
    _poll(client, session)
    st.rerun()


def _poll(client: DashboardAPIClient, session: DashboardSession) -> None:
    """Re-read the run and fetch whatever timeline records are new.

    Two ``GET``s, both safe to repeat and neither repeated here. The timeline
    request carries the session's cursor, so a long run is never re-transmitted
    and a record is never absorbed twice.
    """
    replay = session.replay
    if replay.run_id is None:  # pragma: no cover - callers check
        return
    run = client.demo_run(replay.run_id)
    if run.problem is not None:
        render_problem(run.problem, context="Could not re-read the replay run.")
        return
    replay.observe(run.unwrap())
    page = client.demo_timeline(
        replay.run_id, after_sequence=replay.cursor, limit=POLL_LIMIT
    )
    if page.problem is not None:
        render_problem(page.problem, context="Could not read the replay timeline.")
        return
    replay.absorb(page.unwrap())


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _render_run(
    client: DashboardAPIClient,
    session: DashboardSession,
    scenarios: Sequence[ScenarioDocument],
) -> None:
    """Render the attached run, polling while it is active."""
    replay = session.replay
    if not replay.attached:
        st.caption(
            "No replay run is attached to this browser session. Choose a "
            "scenario and a pace above, then press **Start replay**."
        )
        return

    interval = poll_interval(replay)
    if interval is None:
        # Terminal, or nothing to follow. Rendered directly, with no fragment
        # and therefore no schedule: the page stops calling the backend until
        # somebody presses something.
        _render_live(client, session, scenarios)
    else:
        st.fragment(_render_live, run_every=interval)(client, session, scenarios)


def _render_live(
    client: DashboardAPIClient,
    session: DashboardSession,
    scenarios: Sequence[ScenarioDocument],
) -> None:
    """Poll once and render the run's status, timeline, chart, and summary.

    This is the fragment body while a run is active. It performs the two reads,
    renders what came back, and returns; there is no loop in it, and Streamlit's
    own ``run_every`` is what brings it round again.
    """
    replay = session.replay
    if replay.active:
        _poll(client, session)
    run = replay.run
    if run is None:
        st.caption("The replay run could not be read.")
        return

    _render_status_cards(run, replay)
    _render_timeline(replay)
    _render_chart(replay)
    if run.terminal:
        _render_summary(run, scenarios)


def _render_status_cards(run: ReplayRunDocument, replay: ReplaySession) -> None:
    """Render scenario, pace, state, progress, fusion, and current severity."""
    latest = replay.records[-1] if replay.records else None
    severity = None if latest is None else str(latest.detection.severity)
    strategy = None if latest is None else latest.detection.hybrid.strategy

    columns = st.columns(6)
    values = (
        ("Scenario", run.scenario_name or run.scenario_id, _BLUE, ""),
        ("Pace", run.pace or "—", _MUTED, "presentation timing only"),
        (
            "Status",
            run.state,
            _STATE_COLORS.get(run.state, _MUTED),
            format_reason_code(run.failure_reason) if run.failure_reason else "",
        ),
        (
            "Events emitted",
            f"{run.emitted_count} / {run.event_count}",
            _BLUE,
            "of this scenario's own timeline",
        ),
        (
            "Fusion strategy",
            format_fusion_strategy(strategy) if strategy else "—",
            _GREEN if strategy else _MUTED,
            "frozen; the service chose it, not this page",
        ),
        (
            "Current severity",
            severity or "—",
            SEVERITY_COLORS.get(severity or "", _MUTED),
            "the most recent scored step",
        ),
    )
    for column, (label, value, accent, note) in zip(columns, values, strict=True):
        with column:
            st.markdown(
                card(label, str(value), accent=accent, note=note),
                unsafe_allow_html=True,
            )
    st.progress(run.progress, text=f"{run.emitted_count} of {run.event_count} events")


def _render_timeline(replay: ReplaySession) -> None:
    """Render the run's timeline: the latest steps in colour, then the table."""
    st.markdown(section_title("Replay timeline"), unsafe_allow_html=True)
    if not replay.records:
        st.caption(
            "No steps scored yet. The first record appears as soon as the "
            "service has scored the run's first event."
        )
        return

    for item in reversed(replay.records[-3:]):
        anchor = item.detection
        severity = str(anchor.severity)
        st.markdown(
            f"{badge(f'step {item.sequence}', color=_MUTED)} "
            f"{badge(severity, color=SEVERITY_COLORS.get(severity, _MUTED))} "
            f'<span class="pad-mono">'
            f"{escape_text(format_timestamp(item.source_event_time))} · "
            f"{escape_text(item.authentication_outcome)} · "
            f"rule {escape_text(format_verdict(anchor.rule.flagged))} · "
            f"model {escape_text(format_verdict(anchor.ml.flagged))} · "
            f"hybrid {escape_text(format_verdict(anchor.hybrid.flagged))}"
            f"</span>",
            unsafe_allow_html=True,
        )

    st.dataframe(
        [
            {
                "Step": item.sequence,
                "Event time": format_timestamp(item.source_event_time),
                "Outcome": item.authentication_outcome,
                "Window": item.window_event_count,
                "Rule": _verdict_cell(item.detection.rule.flagged),
                "Risk": format_risk_score(item.detection.rule.risk_score),
                "ML": _verdict_cell(item.detection.ml.flagged),
                "Hybrid": _verdict_cell(item.detection.hybrid.flagged),
                "Severity": str(item.detection.severity),
                "Rules fired": ", ".join(item.detection.rule.fired_rule_ids) or "—",
            }
            for item in reversed(replay.records)
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Newest first. **Window** is how many events the detection window "
        "carried at that step — every event emitted so far, which is the "
        "anchor's own strictly-prior history. No history is fabricated."
    )

    with st.expander("Full result for the most recent step"):
        render_anchor_result(replay.records[-1].detection)


def _verdict_cell(flagged: bool | None) -> str:
    """Return a compact table cell for one layer's verdict."""
    if flagged is None:
        return "—"
    return "flagged" if flagged else "clear"


def _render_chart(replay: ReplaySession) -> None:
    """Render the layer activity chart, from the run's own records."""
    st.markdown(section_title("Layer activity across the run"), unsafe_allow_html=True)
    render_replay_activity_chart(replay.detection_records())
    st.caption(
        "Cumulative flags per layer against the run's step number. Three "
        "independent counts of three independent decisions, plotted against the "
        "step rather than the clock — so the picture is identical at every pace."
    )


def _render_summary(
    run: ReplayRunDocument, scenarios: Sequence[ScenarioDocument]
) -> None:
    """Render the finished run's summary, as the service computed it."""
    st.markdown(section_title("Demo run summary"), unsafe_allow_html=True)
    summary = run.summary
    st.markdown(
        f"{badge(run.state, color=_STATE_COLORS.get(run.state, _MUTED))} "
        f'<span class="pad-mono">{escape_text(run.scenario_name)} · run '
        f"{escape_text(truncate_identifier(run.run_id, keep=16))}</span>",
        unsafe_allow_html=True,
    )

    rows = [
        {"Measure": "Events emitted", "Value": str(run.emitted_count)},
        {
            "Measure": "Detection windows processed",
            "Value": str(summary.detection_count),
        },
        {"Measure": "Rule-flagged steps", "Value": str(summary.rule_flagged_count)},
        {"Measure": "Model-flagged steps", "Value": str(summary.ml_flagged_count)},
        {"Measure": "Hybrid-flagged steps", "Value": str(summary.hybrid_flagged_count)},
        {"Measure": "Highest severity", "Value": summary.highest_severity or "—"},
        {
            "Measure": "Frozen fusion strategy",
            "Value": ", ".join(
                format_fusion_strategy(item) for item in summary.fusion_strategies
            )
            or "—",
        },
    ]
    left, right = st.columns(2)
    with left:
        st.dataframe(rows, width="stretch", hide_index=True)
        if summary.ml_unavailable_count:
            st.caption(
                f"The model layer produced no verdict on "
                f"{summary.ml_unavailable_count} step(s). An unavailable layer "
                f"has not declined to flag anything."
            )
    with right:
        st.markdown("**Severities observed**")
        st.dataframe(
            [
                {"Severity": name, "Steps": count}
                for name, count in summary.severity_counts.items()
            ]
            or [{"Severity": "—", "Steps": 0}],
            width="stretch",
            hide_index=True,
        )
        st.markdown("**Rules triggered**")
        st.dataframe(
            [
                {"Rule": name, "Times fired": count}
                for name, count in summary.triggered_rule_counts.items()
            ]
            or [{"Rule": "none fired", "Times fired": 0}],
            width="stretch",
            hide_index=True,
        )

    scenario = _find(scenarios, run.scenario_id)
    if scenario is not None and scenario.expected_rule_ids:
        st.caption(
            "The service stated in advance that this scenario fires "
            + ", ".join(scenario.expected_rule_ids)
            + ". The table above is what this run actually produced; the two are "
            "shown side by side and neither is derived from the other."
        )
    st.caption(
        "**Demo run summary.** Every figure is derived from this run's own "
        "timeline records. It describes one demonstration in one server "
        "process: there is no persistent replay history, and a restart clears "
        "every run."
    )
