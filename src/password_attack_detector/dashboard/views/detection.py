"""The detection console: build a window, submit it, read the verdict.

The one page where the dashboard sends something rather than reads something, so
it is where the boundary matters most.

**The form builds a request body and nothing else.**  Every control below writes
into a mapping that is posted verbatim to ``/api/v1/detect``.  The console never
pre-validates a window against its own copy of the rules, never computes a
feature, and never predicts what the verdict will be: the API's schema is the
authority on what a valid event is, and a second opinion here would eventually
disagree with it.

**There is no credential field, and there cannot be one.**  Not a password box
that is ignored, not a disabled input -- no control on this page writes a
credential-shaped key, and the service refuses one under any spelling regardless.
A test asserts the built body's key set against the wire contract.

**Nothing is submitted without a press.**  No template auto-submits, no result
auto-refreshes, and a resend is a separate deliberate action on a window the
analyst can still see.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, time

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.alerts import render_anchor_result
from password_attack_detector.dashboard.components.status import (
    Connectivity,
    render_problem,
    require_backend,
)
from password_attack_detector.dashboard.contracts import SystemStatusDocument
from password_attack_detector.dashboard.formatting import escape_text
from password_attack_detector.dashboard.scenarios import (
    APPLICATION_IDS,
    AUTHENTICATION_METHODS,
    AUTHENTICATION_OUTCOMES,
    BLOCKED_FAILURE_REASONS,
    CLIENT_TYPES,
    COUNTRY_CODES,
    DOCUMENTATION_ADDRESSES,
    FAILURE_REASONS,
    MFA_OUTCOMES,
    SCENARIOS,
    build_event,
    pseudonym,
    scenario_summary,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import section_title

__all__ = ["render", "synthetic_identity_note"]

_ANCHOR_MODES: dict[str, str] = {
    "last": "Last event in the window (default)",
    "all": "Every event in the window",
}


def render(
    client: DashboardAPIClient, status: Connectivity, session: DashboardSession
) -> None:
    """Render the detection console."""
    online = require_backend(status)
    system = client.system_status().document if online else None

    _render_templates(session)
    st.divider()
    _render_event_builder(session)
    st.divider()
    _render_window(session, system)
    st.divider()
    _render_submit(client, session, online=online, system=system)

    if session.last_result is not None:
        st.divider()
        render_anchor_result(session.last_result.anchor)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _render_templates(session: DashboardSession) -> None:
    """Render the safe synthetic scenario buttons."""
    st.markdown(section_title("Synthetic scenarios"), unsafe_allow_html=True)
    st.caption(
        "Documentation fixtures describing activity that did not happen, built "
        "from synthetic identities. Loading one fills the window below; it does "
        "not submit anything and does not bypass the API."
    )
    columns = st.columns(len(SCENARIOS))
    for column, scenario in zip(columns, SCENARIOS, strict=True):
        with column:
            st.markdown(f"**{escape_text(scenario.label)}**")
            st.caption(scenario.description)
            st.caption(f"_Expected shape:_ {scenario.expectation}")
            if st.button(
                f"Load {scenario.label.lower()}",
                key=f"scenario-{scenario.key}",
                width="stretch",
            ):
                session.load_scenario(scenario.key, scenario.events())
                st.rerun()


# ---------------------------------------------------------------------------
# Building one event
# ---------------------------------------------------------------------------


def _render_event_builder(session: DashboardSession) -> None:
    """Render the form that appends one event to the window."""
    st.markdown(section_title("Add an authentication event"), unsafe_allow_html=True)
    st.caption(
        "Field names and vocabularies are the API's own. This console accepts "
        "no password, hash, token, or secret, and the service refuses one under "
        "any spelling."
    )
    st.caption(synthetic_identity_note())
    with st.form("event-builder", clear_on_submit=False):
        first, second, third = st.columns(3)
        with first:
            event_date = st.date_input("Event date", value=datetime.now(UTC).date())
            event_time = st.time_input("Event time (UTC)", value=time(12, 0))
            outcome = st.selectbox("Authentication outcome", AUTHENTICATION_OUTCOMES)
            method = st.selectbox("Authentication method", AUTHENTICATION_METHODS)
        with second:
            user_label = st.text_input("User label (synthetic)", value="analyst-01")
            device_label = st.text_input("Device label (synthetic)", value="laptop-01")
            session_label = st.text_input(
                "Session label (synthetic)", value="session-01"
            )
            application = st.selectbox("Application", APPLICATION_IDS)
        with third:
            identity_mode = st.radio(
                "Source identity",
                ("Pseudonymous source id", "Source IP address"),
                help=(
                    "The schema requires exactly one. An address is "
                    "pseudonymized by the service on arrival and never returned; "
                    "that path needs the deployment to hold a pseudonymization "
                    "key."
                ),
            )
            source_label = st.text_input("Source label (synthetic)", value="office-01")
            source_ip = st.selectbox(
                "Source address (RFC 5737 documentation range)",
                DOCUMENTATION_ADDRESSES,
            )
            country = st.selectbox("Country code", ("(none)", *COUNTRY_CODES))

        fourth, fifth, sixth = st.columns(3)
        with fourth:
            failure_reason = st.selectbox(
                "Failure reason",
                ("(automatic)", *FAILURE_REASONS),
                help=(
                    "The canonical schema requires a reason on a failure and "
                    "forbids one on a success. Left automatic, a valid reason is "
                    "chosen for the outcome."
                ),
            )
        with fifth:
            mfa_outcome = st.selectbox("MFA outcome", ("(none)", *MFA_OUTCOMES))
        with sixth:
            client_type = st.selectbox("Client type", ("(none)", *CLIENT_TYPES))
            response_time = st.number_input(
                "Response time (ms)", min_value=0, max_value=30_000, value=150
            )

        submitted = st.form_submit_button("Add event to window", width="stretch")

    if not submitted:
        return
    if outcome == "blocked" and failure_reason not in (
        "(automatic)",
        *BLOCKED_FAILURE_REASONS,
    ):
        st.error(
            "That failure reason is not valid alongside a blocked outcome. "
            f"Valid values: {', '.join(BLOCKED_FAILURE_REASONS)}.",
            icon="⚠️",
        )
        return
    moment = datetime.combine(event_date, event_time, tzinfo=UTC)
    event = build_event(
        f"manual-{uuid.uuid4()}",
        base_time=moment,
        user=user_label,
        source=source_label,
        device=device_label,
        session=session_label,
        outcome=str(outcome),
        failure_reason=None if failure_reason == "(automatic)" else str(failure_reason),
        method=str(method),
        application=str(application),
        country=None if country == "(none)" else str(country),
        response_time_ms=int(response_time),
        mfa_outcome=None if mfa_outcome == "(none)" else str(mfa_outcome),
        client_type=None if client_type == "(none)" else str(client_type),
        source_ip=str(source_ip) if identity_mode == "Source IP address" else None,
    )
    session.add_event(event)
    session.selected_scenario = "custom"
    st.rerun()


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def _render_window(
    session: DashboardSession, system: SystemStatusDocument | None
) -> None:
    """Render the composed window, in table form and as JSON."""
    st.markdown(section_title("Detection window"), unsafe_allow_html=True)
    events = session.draft_events
    if not events:
        st.info(
            "The window is empty. Load a synthetic scenario above, or add "
            "events one at a time.",
            icon="🪟",
        )
        return

    st.caption(
        "Events must be in non-decreasing time order; the API refuses a window "
        "that is not. A window is history plus the anchors it supports — the "
        "service fabricates no history for a caller who supplies none."
    )
    table_tab, json_tab = st.tabs(["Form view", "JSON preview"])
    with table_tab:
        st.dataframe(
            [
                {
                    "#": index,
                    "Event time": item.get("event_time", ""),
                    "Outcome": item.get("authentication_outcome", ""),
                    "Method": item.get("authentication_method", ""),
                    "Application": item.get("application_id", ""),
                    "User": item.get("user_id", ""),
                    "Source": item.get("source_id") or item.get("source_ip") or "",
                    "Failure reason": item.get("failure_reason", "—"),
                }
                for index, item in enumerate(events)
            ],
            width="stretch",
            hide_index=True,
        )
        remove, clear = st.columns([0.6, 0.4])
        with remove:
            index = st.number_input(
                "Remove event #",
                min_value=0,
                max_value=max(len(events) - 1, 0),
                value=0,
                step=1,
            )
            if st.button("Remove event", width="stretch"):
                session.remove_event(int(index))
                st.rerun()
        with clear:
            st.write("")
            st.write("")
            if st.button("Clear window", width="stretch"):
                session.clear_window()
                st.rerun()
    with json_tab:
        st.caption(
            "The exact body this console will post. Nothing is added, renamed, "
            "or filled in on the way out."
        )
        st.code(
            json.dumps(
                {"events": events, "anchor_selection": _anchor_mode()},
                indent=2,
                sort_keys=False,
            ),
            language="json",
        )

    summary = scenario_summary(events)
    if system is not None and len(events) > system.max_batch_events:
        st.error(
            f"This window carries {len(events)} events and the deployment "
            f"accepts at most {system.max_batch_events}. The API will refuse it.",
            icon="⚠️",
        )
    st.caption(
        f"Distinct users: {summary['distinct_users']} · "
        f"distinct sources: {summary['distinct_sources']}"
    )


def _anchor_mode() -> str:
    """Return the anchor selection currently chosen, defaulting to ``last``."""
    return str(st.session_state.get("anchor-mode", "last"))


# ---------------------------------------------------------------------------
# Submitting
# ---------------------------------------------------------------------------


def _render_submit(
    client: DashboardAPIClient,
    session: DashboardSession,
    *,
    online: bool,
    system: SystemStatusDocument | None,
) -> None:
    """Render the pre-submission summary and the submit control."""
    st.markdown(section_title("Submit"), unsafe_allow_html=True)
    events = session.draft_events
    st.selectbox(
        "Anchor mode",
        tuple(_ANCHOR_MODES),
        format_func=lambda key: _ANCHOR_MODES[key],
        key="anchor-mode",
    )
    mode = _anchor_mode()

    st.markdown(
        f'<div class="pad-note">Events: <b>{len(events)}</b> · '
        f"Anchor mode: <b>{escape_text(_ANCHOR_MODES[mode])}</b> · "
        f"API: <b>{'connected' if online else 'not connected'}</b></div>",
        unsafe_allow_html=True,
    )

    disabled = not online or not events
    if not events:
        st.caption("Add at least one event before submitting.")
    elif not online:
        st.caption(
            "Submission is disabled while the detection API is unreachable. "
            "Nothing is queued and nothing is scored locally."
        )

    if not st.button("Run detection", type="primary", disabled=disabled):
        return

    if mode == "all":
        batch = client.detect_batch(events, anchor_selection="all")
        if not batch.ok:
            assert batch.problem is not None
            render_problem(batch.problem, context="The detection was not performed.")
            return
        document = batch.unwrap()
        # Recorded, one entry per anchor: the analyst performed these detections,
        # and a batch that rendered results without recording them would leave the
        # alerts and analytics pages reporting an empty session.
        stored = session.record_batch(document, observed_at=datetime.now(UTC))
        numbering = (
            f" (session detections #{stored[0].sequence}-#{stored[-1].sequence})"
            if stored
            else ""
        )
        st.success(
            f"Scored {len(document.anchors)} anchors in one window{numbering}.",
            icon="✅",
        )
        for anchor in document.anchors:
            render_anchor_result(anchor)
        return

    single = client.detect(events, anchor_selection="last")
    if not single.ok:
        assert single.problem is not None
        render_problem(single.problem, context="The detection was not performed.")
        st.caption(
            "The request was sent once and is not retried automatically: a "
            "request that timed out may already have been evaluated."
        )
        return
    record = session.record(single.unwrap(), observed_at=datetime.now(UTC))
    st.success(f"Detection #{record.sequence} complete.", icon="✅")
    if system is not None and system.hybrid_detection_enabled:
        st.caption(
            f"Fused under the frozen {system.fusion_strategy} strategy this "
            f"deployment loaded at startup."
        )


def synthetic_identity_note() -> str:
    """Return the note explaining what the console's identifiers are.

    Shared with the events view, where the same question comes up about the
    table it renders: the two pages must not answer it differently.
    """
    return (
        "Identifiers are synthetic labels hashed into the pseudonym shape the "
        f"wire contract requires (for example {pseudonym('user', 'example')}). "
        "They name entities that do not exist; the project's keyed "
        "pseudonymization of real identifiers happens on the server, not here."
    )
