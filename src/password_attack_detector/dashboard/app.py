"""The Streamlit entrypoint.

    uv run streamlit run src/password_attack_detector/dashboard/app.py \\
      --server.address 127.0.0.1 --server.port 8501

Responsible for exactly three things: resolving configuration once, holding the
session's state and client across reruns, and dispatching to a view.  Every
decision about what to display lives in the view; every decision about what the
detection system *is* lives on the other side of the API.

**A rerun is not a new session.**  Streamlit re-executes this module top to
bottom on every interaction, so the client and the session object are cached in
``st.session_state`` rather than rebuilt -- otherwise every button press would
open a fresh connection pool and forget the session's history.

**A misconfiguration is a page, not a traceback.**  If the settings do not
validate, the app renders the failure and stops.  Streamlit renders an uncaught
exception into the browser in full, which is exactly what the project's error
contract exists to prevent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import streamlit as st

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.header import (
    render_header,
    render_sidebar,
)
from password_attack_detector.dashboard.components.status import (
    Connectivity,
    connectivity,
)
from password_attack_detector.dashboard.config import (
    DashboardSettings,
    load_dashboard_settings,
)
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import STYLESHEET
from password_attack_detector.dashboard.views import (
    about,
    alerts,
    analytics,
    comparison,
    detection,
    drift,
    events,
    explainability,
    overview,
    replay,
    system,
)
from password_attack_detector.exceptions import ConfigurationError

__all__ = ["VIEWS", "View", "main"]

_CLIENT_KEY: Final[str] = "pad_api_client"
_SESSION_KEY: Final[str] = "pad_session"

#: What every view's entry point is.  One signature for all ten, so no view
#: acquires its own way of reaching the backend or its own idea of what a
#: session is: the client, this render's connectivity, and the session are
#: handed to each of them and nothing else is.
View = Callable[[DashboardAPIClient, Connectivity, DashboardSession], None]

#: Navigation label to the function that renders it.  The labels are the ones
#: :data:`~password_attack_detector.dashboard.components.header.PAGES` declares;
#: a test asserts the two agree, so a renamed label cannot leave a blank page.
VIEWS: Final[dict[str, View]] = {
    "Overview": overview.render,
    "Live Replay": replay.render,
    "Alerts": alerts.render,
    "Analytics": analytics.render,
    "Explainability": explainability.render,
    "Drift Monitoring": drift.render,
    "Detection Console": detection.render,
    "Authentication Events": events.render,
    "Rule vs ML vs Hybrid": comparison.render,
    "System & Model": system.render,
    "About System": about.render,
}


def _session() -> DashboardSession:
    """Return this browser session's state, creating it on first render."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = DashboardSession()
    stored: DashboardSession = st.session_state[_SESSION_KEY]
    return stored


def _client(settings: DashboardSettings) -> DashboardAPIClient:
    """Return this session's API client, creating it on first render.

    Cached across reruns so a page render reuses one connection pool. Not
    ``st.cache_resource``: that would share a client between browser sessions,
    and the client holds the configured timeout rather than being stateless.

    Rebuilt if the resolved settings differ from the ones the cached client was
    built with. A cached client pointed at the previous URL would keep reporting
    the previous deployment's state under the new configuration, which is exactly
    the kind of stale answer this console must not give.
    """
    stored: DashboardAPIClient | None = st.session_state.get(_CLIENT_KEY)
    if stored is not None and stored.settings == settings:
        return stored
    if stored is not None:
        stored.close()
    built = DashboardAPIClient(settings)
    st.session_state[_CLIENT_KEY] = built
    return built


def main() -> None:
    """Render one pass of the console."""
    try:
        settings = load_dashboard_settings()
    except ConfigurationError as exc:
        st.set_page_config(page_title="Detection console", layout="wide")
        st.error(
            "The dashboard configuration is not valid. Correct the "
            "`PAD_DASHBOARD_*` environment and restart.",
            icon="⚠️",
        )
        # Safe to render because ``load_dashboard_settings`` builds this message
        # from field names and validation rules only. It deliberately does not
        # use ``str(ValidationError)``, which appends ``input_value=...`` -- and
        # the value that most often breaks ``api_url`` is a URL somebody pasted a
        # token into.
        st.code(str(exc), language="text")
        return

    st.set_page_config(
        page_title=settings.page_title,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLESHEET, unsafe_allow_html=True)

    session = _session()
    client = _client(settings)
    status = connectivity(client)

    render_header(status, settings)
    page = render_sidebar(status, settings)

    view = VIEWS.get(page, VIEWS["Overview"])
    view(client, status, session)


if __name__ == "__main__":
    # Streamlit executes the entry script with ``__name__ == "__main__"``, so
    # this runs under ``streamlit run`` and does nothing on import. Importing
    # the module -- which the tests do, to check the dispatch table against the
    # navigation labels -- must not start rendering.
    main()
