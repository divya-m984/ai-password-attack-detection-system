"""Every view rendered, twice: against the real deployment, and with it gone.

Driven by Streamlit's own ``AppTest``, which executes the entrypoint script in
process and exposes the elements it produced.  That matters more here than the
convenience: the properties being asserted are about what a *viewer sees*, and
the only honest way to check "no traceback appears on the page" is to render the
page and look.

Two fixtures, and the pairing is the point:

* ``wired`` -- the dashboard's client is given a transport into the genuinely
  frozen deployment the session fixture built, so every online assertion runs
  against real documents.
* ``offline`` -- the client is pointed at a port nothing is listening on, which
  is what an analyst gets when they open the console before starting the API.

Every view is rendered in both states, and in neither may an exception reach the
page.  Streamlit renders an uncaught exception into the browser in full, which is
exactly what the project's error contract exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.header import PAGES
from password_attack_detector.dashboard.config import DashboardSettings
from password_attack_detector.dashboard.scenarios import prohibited_field_names
from tests.integration.conftest import ServingTransport

APP = str(
    Path(__file__).resolve().parents[2]
    / "src"
    / "password_attack_detector"
    / "dashboard"
    / "app.py"
)

#: A loopback port nothing binds.  Chosen from the ephemeral range so a
#: developer's own service is not what the offline tests accidentally reach.
DEAD_PORT = 59_431


@pytest.fixture()
def wired(
    serving_transport: ServingTransport, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Point every client the app builds at the real deployment.

    ``AppTest`` re-executes the entrypoint as a fresh module, so patching the
    app module's own namespace would patch a module the run does not use. The
    constructor is patched instead: whichever module imports the class, it is
    this class.
    """
    original = DashboardAPIClient.__init__

    def patched(
        self: DashboardAPIClient,
        settings: DashboardSettings,
        *,
        transport: Any = None,
    ) -> None:
        original(self, settings, transport=serving_transport)

    monkeypatch.setattr(DashboardAPIClient, "__init__", patched)
    monkeypatch.setenv("PAD_DASHBOARD_API_URL", "http://testserver")
    monkeypatch.setenv("PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS", "30")
    yield


@pytest.fixture()
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the dashboard at a port nothing is listening on."""
    monkeypatch.setenv("PAD_DASHBOARD_API_URL", f"http://127.0.0.1:{DEAD_PORT}")
    monkeypatch.setenv("PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS", "2")
    yield


def _render(page: str) -> AppTest:
    """Render one view and return the resulting app state."""
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    return app


def _raw_html(app: AppTest) -> list[str]:
    """Return every block the page emitted as HTML rather than as Markdown.

    These are the blocks rendered with ``unsafe_allow_html``, which Streamlit
    passes through without running the Markdown parser over them -- so anything
    in one that *looks* like Markdown reaches the viewer as punctuation.
    """
    return [str(item.value) for item in app.markdown if "<div" in str(item.value)]


def _text(app: AppTest) -> str:
    """Return every rendered string, for the sweeps below."""
    parts: list[str] = []
    for group in (
        app.markdown,
        app.caption,
        app.warning,
        app.error,
        app.info,
        app.success,
        app.code,
        app.title,
        app.header,
        app.subheader,
    ):
        parts.extend(str(item.value) for item in group)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Online: every view renders against the frozen deployment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page", PAGES)
def test_every_view_renders_without_an_exception(page: str, wired: None) -> None:
    """Nine views, real documents, and nothing raised onto the page."""
    app = _render(page)
    assert not app.exception, [str(item.value) for item in app.exception]


def test_the_overview_shows_system_readiness_and_no_offline_state(
    wired: None,
) -> None:
    """The two facts an analyst checks before believing anything else."""
    app = _render("Overview")
    rendered = _text(app)
    assert "System" in rendered
    assert "Ready" in rendered
    assert "Not ready" not in rendered


def test_the_overview_reports_the_real_layers_and_strategy(wired: None) -> None:
    """Operational values come from the API, never from a constant."""
    rendered = _text(_render("Overview"))
    assert "layers active" in rendered
    assert "Stacked" in rendered


def test_the_overview_invents_no_global_event_total(wired: None) -> None:
    """With no session activity there is nothing to count, and nothing is."""
    rendered = _text(_render("Overview"))
    assert "No activity in this browser session yet." in rendered


def test_the_overview_states_the_product_name_once(wired: None) -> None:
    """One product heading per screen.

    The global header carries the name; the landing page renders directly below
    it and adds none of its own. Two ``pad-title`` blocks means the page has
    grown a second masthead saying almost the same words as the first.
    """
    app = _render("Overview")
    titles = [block for block in _raw_html(app) if 'class="pad-title"' in block]
    assert len(titles) == 1, titles


def test_the_overview_prose_renders_no_literal_markdown(wired: None) -> None:
    """Emphasis in a raw-HTML block would reach the viewer as asterisks.

    Every default block this page emits with ``unsafe_allow_html`` is swept,
    not only the one the bug was in: the trap is a property of the rendering
    mode rather than of that one constant.
    """
    for block in _raw_html(_render("Overview")):
        assert "**" not in block, block


def test_the_console_navigates_through_exactly_one_control(wired: None) -> None:
    """The sidebar groups are drawn, not wired: one widget holds one page."""
    app = _render("Overview")
    assert len(app.sidebar.radio) == 1
    assert list(app.sidebar.radio[0].options) == list(PAGES)


def test_the_system_view_reports_the_frozen_model_and_rules(wired: None) -> None:
    """Every row on that page comes from ``/version``, ``/model/info``, ``/rules``."""
    app = _render("System & Model")
    assert not app.exception
    rendered = _text(app)
    # The loaded stacker's identity is published as a digest, in a code block.
    assert any(len(str(item.value)) == 64 for item in app.code)
    # The system table is a dataframe; check section titles are present.
    assert "System health" in rendered
    assert "Model details" in rendered


def test_the_comparison_view_names_the_served_stacked_state(wired: None) -> None:
    """When STACKED is active the page says the frozen state is being served."""
    rendered = _text(_render("Rule vs ML vs Hybrid"))
    assert "stacked" in rendered.lower()
    assert "frozen" in rendered.lower()


def test_the_drift_view_loads_no_report_and_fabricates_no_figure(
    wired: None,
) -> None:
    """There is no serving drift endpoint, and the page says exactly that."""
    app = _render("Drift Monitoring")
    assert not app.exception
    rendered = _text(app)
    assert "No drift report" in rendered
    assert "PSI >= 0.10" in rendered.replace("≥", ">=")
    assert "PSI >= 0.25" in rendered.replace("≥", ">=")


def test_the_session_pages_start_empty_and_say_so(wired: None) -> None:
    """No seeded data, on any of the three session-backed views."""
    for page in ("Alerts", "Analytics", "Authentication Events"):
        rendered = _text(_render(page))
        assert "session" in rendered.lower()
        assert "12,503" not in rendered


# ---------------------------------------------------------------------------
# Offline: the console still loads, and says why it is empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page", PAGES)
def test_every_view_survives_the_api_being_down(page: str, offline: None) -> None:
    """The single most likely state during a demonstration."""
    app = _render(page)
    assert not app.exception, [str(item.value) for item in app.exception]


def test_the_header_reports_the_api_offline(offline: None) -> None:
    """The badge is the first thing a viewer looks at, so it must be right."""
    rendered = _text(_render("Overview"))
    assert "Offline" in rendered
    assert "Online" not in rendered


def test_an_offline_page_shows_no_traceback(offline: None) -> None:
    """A traceback in a browser is the failure this contract exists to prevent."""
    for page in PAGES:
        app = _render(page)
        rendered = _text(app)
        assert "Traceback" not in rendered
        assert "ConnectError" not in rendered
        assert "httpx" not in rendered
        assert 'File "' not in rendered


def test_an_offline_page_fabricates_no_metadata(offline: None) -> None:
    """No placeholder model, no invented rule count, no default strategy."""
    rendered = _text(_render("System & Model"))
    assert "not reachable" in rendered
    for invented in ("logistic_regression", "random_forest", "or_gate", "and_gate"):
        assert invented not in rendered


def test_an_offline_console_offers_a_retry(offline: None) -> None:
    """The recovery path is a control, not a page reload the viewer has to guess."""
    app = _render("Overview")
    labels = {item.label for item in app.sidebar.button}
    assert "Retry connection" in labels


def test_an_offline_detection_console_disables_submission(offline: None) -> None:
    """Nothing is queued, and nothing is scored locally instead."""
    app = _render("Detection Console")
    assert not app.exception
    rendered = _text(app)
    assert "not connected" in rendered
    submit = [item for item in app.button if item.label == "Run detection"]
    assert submit and submit[0].disabled


def test_the_drift_view_needs_no_backend_at_all(offline: None) -> None:
    """It documents a contract; there is no live figure for the API to supply."""
    rendered = _text(_render("Drift Monitoring"))
    assert "No drift report" in rendered


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_the_console_recovers_when_the_api_comes_back(
    serving_transport: ServingTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline, then online, in one process: the second render is fully live.

    A console that cached its first failure would need a restart to notice the
    service coming up, which during a demonstration reads as the dashboard being
    broken rather than the API having been.
    """
    monkeypatch.setenv("PAD_DASHBOARD_API_URL", f"http://127.0.0.1:{DEAD_PORT}")
    monkeypatch.setenv("PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS", "2")
    down = AppTest.from_file(APP, default_timeout=120)
    down.run()
    assert "Offline" in _text(down)

    original = DashboardAPIClient.__init__

    def patched(
        self: DashboardAPIClient,
        settings: DashboardSettings,
        *,
        transport: Any = None,
    ) -> None:
        original(self, settings, transport=serving_transport)

    monkeypatch.setattr(DashboardAPIClient, "__init__", patched)
    monkeypatch.setenv("PAD_DASHBOARD_API_URL", "http://testserver")
    monkeypatch.setenv("PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS", "30")

    up = AppTest.from_file(APP, default_timeout=120)
    up.run()
    rendered = _text(up)
    assert "Online" in rendered
    assert "Ready" in rendered


# ---------------------------------------------------------------------------
# The console actually detects
# ---------------------------------------------------------------------------


def test_a_template_submits_through_the_api_and_populates_the_session(
    wired: None,
) -> None:
    """The primary demonstration path, end to end through the real service."""
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()

    load = [item for item in app.button if item.key == "scenario-brute_force"]
    assert load, [item.key for item in app.button]
    load[0].click().run()

    submit = [item for item in app.button if item.label == "Run detection"]
    assert submit and not submit[0].disabled
    submit[0].click().run()

    assert not app.exception, [str(item.value) for item in app.exception]
    rendered = _text(app)
    assert "Detection #1 complete." in rendered
    # The three layers, kept apart on the page as they are in the response.
    assert "Rule detection" in rendered
    assert "ML detection" in rendered
    assert "Hybrid detection" in rendered
    assert "Security assessment" in rendered


def test_a_submitted_detection_reaches_the_alerts_and_analytics_views(
    wired: None,
) -> None:
    """Session state carries one result across three views in one session."""
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()
    next(item for item in app.button if item.key == "scenario-spraying").click().run()
    next(item for item in app.button if item.label == "Run detection").click().run()

    app.sidebar.radio[0].set_value("Alerts").run()
    assert not app.exception
    assert "No detection activity in this dashboard session." not in _text(app)

    app.sidebar.radio[0].set_value("Analytics").run()
    assert not app.exception
    # The caption names its source since Milestone 3: with a replay run also
    # possible, "1 detection(s)" alone would no longer say whose.
    text = _text(app)
    assert "1 detection(s)." in text
    assert "This dashboard session (manual submissions)" in text


def test_a_batch_submission_records_one_alert_per_anchor(wired: None) -> None:
    """Every anchor scored is an entry in the session, and the counts agree.

    A batch that rendered verdicts on screen while the alerts page reported an
    empty session would be the console disagreeing with itself.
    """
    app = AppTest.from_file(APP, default_timeout=240)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()
    next(item for item in app.button if item.key == "scenario-normal").click().run()

    app.selectbox("anchor-mode").set_value("all").run()
    next(item for item in app.button if item.label == "Run detection").click().run()
    assert not app.exception, [str(item.value) for item in app.exception]

    events = len(app.session_state["pad_session"].draft_events)
    history = app.session_state["pad_session"].history
    assert len(history) == events
    assert [item.sequence for item in history] == list(range(1, events + 1))
    assert f"Scored {events} anchors in one window" in _text(app)

    app.sidebar.radio[0].set_value("Analytics").run()
    text = _text(app)
    assert f"{events} detection(s)." in text
    assert "This dashboard session (manual submissions)" in text


def test_the_events_view_shows_the_composed_window_and_can_resend_it(
    wired: None,
) -> None:
    """The session's own events, the exact request body, and one explicit resend.

    A resend is a separate deliberate action on a window the analyst can still
    read; nothing on this page re-submits on its own.
    """
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()
    next(item for item in app.button if item.key == "scenario-normal").click().run()

    app.sidebar.radio[0].set_value("Authentication Events").run()
    assert not app.exception
    rendered = _text(app)
    assert "Current dashboard session" in rendered
    assert "synthetic labels" in rendered
    # The exact body, shown so an analyst can inspect what will be sent.
    assert any('"events"' in str(item.value) for item in app.code)

    resend = next(item for item in app.button if item.label == "Re-submit this window")
    assert not resend.disabled
    resend.click().run()
    assert not app.exception, [str(item.value) for item in app.exception]
    assert "Detection #1 complete." in _text(app)

    clear = next(item for item in app.button if item.label == "Clear session events")
    clear.click().run()
    assert not app.exception
    assert "No events in this session." in _text(app)


def test_the_events_view_carries_no_credential_field_or_internal_location(
    wired: None,
) -> None:
    """The sweep the page's own caption promises.

    Checked against the rendered *request body* for credential keys rather than
    against the whole page for the word: ``password`` legitimately appears on
    every page as part of the product's name, and ``"authentication_method":
    "password"`` is a value the wire contract defines. What must not appear is a
    credential-shaped **key**.
    """
    import json

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()
    next(item for item in app.button if item.key == "scenario-spraying").click().run()
    app.sidebar.radio[0].set_value("Authentication Events").run()

    bodies = [
        json.loads(str(item.value))
        for item in app.code
        if str(item.value).lstrip().startswith("{")
    ]
    assert bodies, "the page renders the exact request body"
    for body in bodies:
        for event in body["events"]:
            assert prohibited_field_names(event) == ()

    rendered = _text(app)
    for forbidden in ("/home/", "/tmp/", "champion.lock", ".parquet", "Traceback"):
        assert forbidden not in rendered


def test_the_explainability_view_attributes_a_submitted_window(
    wired: None,
) -> None:
    """The console's explain path, against the frozen champion's own arrays."""
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()
    next(
        item for item in app.button if item.key == "scenario-brute_force"
    ).click().run()

    app.sidebar.radio[0].set_value("Explainability").run()
    explain = [
        item for item in app.button if item.label == "Explain this window's anchor"
    ]
    assert explain
    explain[0].click().run()

    assert not app.exception, [str(item.value) for item in app.exception]
    rendered = _text(app)
    assert "Reconstruction residual" in rendered
    assert "not a probability" in rendered


def test_a_credential_never_enters_a_session_and_so_never_reaches_a_page(
    wired: None,
) -> None:
    """The console cannot compose a window carrying credential material.

    The refusal is at the session's door rather than at the service's: the
    service would refuse the *request*, but by then the value would already have
    been written into browser-session state and echoed back in the JSON preview.
    So a prohibited key is refused before it is stored, and the draft is provably
    free of one -- which is what makes both the preview and the outgoing body
    safe.
    """
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()
    next(item for item in app.button if item.key == "scenario-normal").click().run()

    session = app.session_state["pad_session"]
    with pytest.raises(ValueError, match="no credential material"):
        session.add_event({"event_id": "x", "password": "not-a-real-secret"})

    assert not any("password" in event for event in session.draft_events)
    assert "not-a-real-secret" not in _text(app)


def test_a_service_refusal_renders_as_its_stable_code(
    wired: None,
) -> None:
    """A refusal reaches the page as the API's own code, with no traceback.

    Driven with a duplicate event identity, which the console has no control to
    produce and the service refuses as ``API003``. The point is the rendering
    contract, not the particular refusal.
    """
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Detection Console").run()
    next(item for item in app.button if item.key == "scenario-normal").click().run()

    session = app.session_state["pad_session"]
    session.draft_events[1]["event_id"] = session.draft_events[0]["event_id"]

    next(item for item in app.button if item.label == "Run detection").click().run()
    assert not app.exception
    rendered = _text(app)
    assert "API error code: API003" in rendered
    assert "Traceback" not in rendered
    assert "The detection was not performed." in rendered
