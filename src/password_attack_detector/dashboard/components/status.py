"""Connectivity, readiness, and the one place a failed call becomes a page.

Every page that needs the backend calls :func:`connectivity` once and then
renders from what it returned.  Two calls per render -- liveness and readiness --
and no page makes its own: a dashboard where each panel probes independently
produces a page that is partly online, and a load pattern that grows with the
number of panels.

:func:`render_problem` is the deterministic error contract.  It renders the fixed
message for a
:class:`~password_attack_detector.dashboard.api_client.ProblemKind`, the API's own
stable code where the service supplied one, and nothing else.  No exception, no
URL, no traceback.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from password_attack_detector.dashboard.api_client import (
    DashboardAPIClient,
    Problem,
    ProblemKind,
)
from password_attack_detector.dashboard.contracts import (
    HealthDocument,
    ReadinessDocument,
)
from password_attack_detector.dashboard.formatting import (
    format_component_state,
    format_detail,
    format_reason_code,
)
from password_attack_detector.dashboard.theme import badge, section_title

__all__ = [
    "Connectivity",
    "connectivity",
    "render_component_table",
    "render_header_status",
    "render_problem",
    "require_backend",
]


@dataclass(frozen=True, slots=True)
class Connectivity:
    """What one render knows about the backend.

    Three states, kept apart because they call for different things from the
    viewer: unreachable (start the service), reachable but not ready (look at
    the component table), and ready.
    """

    health: HealthDocument | None
    readiness: ReadinessDocument | None
    problem: Problem | None

    @property
    def online(self) -> bool:
        """Return whether the service answered at all."""
        return self.health is not None

    @property
    def ready(self) -> bool:
        """Return whether the service reported itself able to serve detection."""
        return self.readiness is not None and self.readiness.is_ready


def connectivity(client: DashboardAPIClient) -> Connectivity:
    """Probe liveness and readiness once, and report both.

    Readiness is attempted even when it will fail, so the component table can
    say *which* part is missing rather than only that something is.
    """
    health = client.health()
    if not health.ok:
        return Connectivity(health=None, readiness=None, problem=health.problem)
    readiness = client.readiness()
    return Connectivity(
        health=health.unwrap(),
        readiness=readiness.document,
        problem=readiness.problem,
    )


def render_problem(problem: Problem, *, context: str = "") -> None:
    """Render one failed call, in fixed text and with the service's own code.

    Never an exception and never a URL. The prose comes from
    :data:`~password_attack_detector.dashboard.api_client.PROBLEM_MESSAGES`, the
    code from the service's error envelope, and the detail from the aggregate
    context the service chose to publish.
    """
    lines = [problem.summary]
    if context:
        lines.insert(0, context)
    if problem.code:
        lines.append(f"API error code: {problem.code}")
    detail = format_detail(problem.detail)
    if detail:
        lines.append(detail)
    body = "  \n".join(lines)
    if problem.kind in {ProblemKind.OFFLINE, ProblemKind.TIMEOUT}:
        st.warning(body, icon="🔌")
    else:
        st.error(body, icon="⚠️")


def require_backend(status: Connectivity, *, needs_ready: bool = True) -> bool:
    """Render the reason this page cannot load, and return whether it can.

    The single gate every backend-dependent page passes through, so "the API is
    down" looks and reads the same on all nine of them. A page that returns
    ``False`` here renders nothing further; it does not fall back to placeholder
    numbers, because a placeholder on a security console is a lie with a chart
    around it.
    """
    if not status.online:
        if status.problem is not None:
            render_problem(status.problem, context="This view needs the detection API.")
        else:  # pragma: no cover - connectivity always carries one or the other
            st.warning("The detection API is not reachable.", icon="🔌")
        st.caption(
            "No values are shown rather than stale or placeholder ones. "
            "Use **Retry connection** in the sidebar once the service is up."
        )
        return False
    if needs_ready and not status.ready:
        st.error(
            "The detection API is running but is not ready to serve detection. "
            "The component table below names what is missing.",
            icon="⚠️",
        )
        if status.readiness is not None:
            render_component_table(status.readiness)
        return False
    return True


def render_component_table(readiness: ReadinessDocument) -> None:
    """Render each runtime component's state, reason, and whether it blocks.

    Reason codes are shown verbatim beside their prose. They are the service's
    stable vocabulary -- ``serving_bundle_lineage_mismatch`` means one specific
    thing -- and rewording them here would break the one string an operator can
    search a log for.
    """
    st.markdown(section_title("Runtime components"), unsafe_allow_html=True)
    st.dataframe(
        [
            {
                "Component": item.component,
                "State": format_component_state(item.state),
                "Required": "yes" if item.required else "no",
                "Reason": format_reason_code(item.reason),
            }
            for item in readiness.components
        ],
        width="stretch",
        hide_index=True,
    )


def render_header_status(status: Connectivity) -> str:
    """Return the header's connectivity badges as one HTML fragment.

    Rendered in sentence case rather than shouted. These two badges are on every
    screen at all times; ``ONLINE READY`` reads as an alarm, and the state they
    report is the unremarkable one.
    """
    if not status.online:
        return (
            badge("Offline", color="#f85149", caps=False)
            + " "
            + badge("Unknown", color="#8b949e", caps=False)
        )
    online = badge("Online", color="#3fb950", caps=False)
    if status.ready:
        return f"{online} {badge('Ready', color='#3fb950', caps=False)}"
    return f"{online} {badge('Not ready', color='#f0883e', caps=False)}"
