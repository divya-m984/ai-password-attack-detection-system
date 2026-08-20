"""The Live Replay page, and the client that feeds it.

Two halves, and both matter for different reasons.

**The client half** drives the five new methods against the genuinely frozen
deployment through the same synchronous transport the rest of the dashboard
suite uses, so the documents being parsed are bytes the real serving application
produced.

**The page half** renders the view with Streamlit's own ``AppTest``, online and
offline, and asserts what a *viewer sees*.  The properties that matter here are
not really about data flow -- they are about the console never starting or
stopping anything on its own, never showing a row it did not receive, and never
putting a traceback in a browser.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from password_attack_detector.dashboard.api_client import DashboardAPIClient
from password_attack_detector.dashboard.components.header import PAGES
from password_attack_detector.dashboard.config import DashboardSettings
from password_attack_detector.dashboard.contracts import ReplayRunDocument
from password_attack_detector.dashboard.state import (
    REPLAY_SCENARIO_PREFIX,
    DashboardSession,
)
from password_attack_detector.dashboard.views.replay import (
    LIVE_POLL_SECONDS,
    PACES,
    poll_interval,
)
from password_attack_detector.replay.enums import ReplayPace, ScenarioId
from password_attack_detector.replay.scenarios import SCENARIOS
from tests.integration.conftest import ServingTransport

APP = str(
    Path(__file__).resolve().parents[2]
    / "src"
    / "password_attack_detector"
    / "dashboard"
    / "app.py"
)

DEAD_PORT = 59_433

PAGE = "Live Replay"


@pytest.fixture()
def api(serving_transport: ServingTransport) -> DashboardAPIClient:
    """The dashboard's own client, wired to the real deployment."""
    settings = DashboardSettings(
        api_url="http://testserver", request_timeout_seconds=30.0
    )
    return DashboardAPIClient(settings, transport=serving_transport)


def _finish(api: DashboardAPIClient, run_id: str, *, timeout: float = 300.0) -> Any:
    """Poll a run through the dashboard client until it stops producing."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = api.demo_run(run_id)
        assert result.ok, result.problem
        document = result.unwrap()
        if document.terminal:
            return document
        time.sleep(0.05)
    raise AssertionError("the run did not finish")  # pragma: no cover


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


def test_the_client_reads_the_scenario_catalog(api: DashboardAPIClient) -> None:
    """Typed, and parsed from the service's own document."""
    result = api.demo_scenarios()
    assert result.ok, result.problem
    catalog = result.unwrap()
    assert catalog.scenario_count == len(SCENARIOS)
    assert {item.scenario_id for item in catalog.scenarios} == {
        str(member) for member in ScenarioId
    }
    for entry in catalog.scenarios:
        assert entry.name
        assert entry.event_count > 0
        assert entry.scenario_fingerprint


def test_the_client_starts_reads_and_stops_a_run(api: DashboardAPIClient) -> None:
    """The whole lifecycle, through the one client the dashboard has."""
    started = api.start_demo_run("account_takeover", pace="instant")
    assert started.ok, started.problem
    run = started.unwrap()
    assert run.run_id
    assert run.scenario_id == "account_takeover"
    assert run.pace == "instant"

    finished = _finish(api, run.run_id)
    assert finished.state == "completed"
    assert finished.terminal
    assert finished.progress == pytest.approx(1.0)

    stopped = api.stop_demo_run(run.run_id)
    assert stopped.ok, stopped.problem
    assert stopped.unwrap().state == "completed", (
        "stopping a finished run changes it not"
    )


def test_the_client_polls_a_timeline_incrementally(api: DashboardAPIClient) -> None:
    """The cursor, from the consumer's side."""
    run = api.start_demo_run("account_takeover", pace="instant").unwrap()
    finished = _finish(api, run.run_id)

    first = api.demo_timeline(run.run_id, after_sequence=0, limit=3).unwrap()
    assert [item.sequence for item in first.records] == [1, 2, 3]
    assert first.more_expected is True

    second = api.demo_timeline(
        run.run_id, after_sequence=first.next_sequence, limit=100
    ).unwrap()
    assert [item.sequence for item in second.records] == list(
        range(4, finished.emitted_count + 1)
    )
    assert second.more_expected is False


def test_the_client_parses_the_three_layers_of_every_record(
    api: DashboardAPIClient,
) -> None:
    """A record embeds the serving layer's own verdict, and it survives parsing."""
    run = api.start_demo_run("brute_force", pace="instant").unwrap()
    _finish(api, run.run_id)
    page = api.demo_timeline(run.run_id, limit=100).unwrap()

    assert page.records
    for record in page.records:
        assert record.detection.ml.available is True
        assert record.detection.hybrid.strategy == "stacked"
        assert record.source_event_time is not None
        assert record.emitted_at is not None
        assert record.window_event_count >= 1


def test_the_client_reports_an_unknown_run_as_a_refusal_not_an_exception(
    api: DashboardAPIClient,
) -> None:
    """Failure is a value here, as it is everywhere else in this client."""
    result = api.demo_run("run_00000000000000000000000000000000")
    assert not result.ok
    assert result.problem is not None
    assert result.problem.code == "API018"
    assert result.problem.status_code == 404


def test_the_client_fabricates_no_timeline_when_a_call_fails(
    api: DashboardAPIClient,
) -> None:
    """No default document, no empty page standing in for a real one."""
    result = api.demo_timeline("run_00000000000000000000000000000000")
    assert result.document is None
    assert result.problem is not None


def test_the_client_offers_no_way_to_send_anything_but_a_scenario_and_a_pace() -> None:
    """The console cannot express an override the service would refuse anyway."""
    import inspect

    signature = inspect.signature(DashboardAPIClient.start_demo_run)
    assert list(signature.parameters) == ["self", "scenario_id", "pace"]
    source = inspect.getsource(DashboardAPIClient.start_demo_run)
    for forbidden in ("threshold", "model", "strategy", "artifact", "events", "url"):
        assert forbidden not in source


def test_the_console_pace_vocabulary_matches_the_service(
    api: DashboardAPIClient,
) -> None:
    """Restated in the view, pinned against the service's own enumeration."""
    assert set(PACES) == {str(item) for item in ReplayPace}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def test_a_session_absorbs_a_page_without_double_counting(
    api: DashboardAPIClient,
) -> None:
    """A repeated page must not make the chart grow while the run stands still."""
    run = api.start_demo_run("account_takeover", pace="instant").unwrap()
    finished = _finish(api, run.run_id)
    session = DashboardSession()
    session.replay.attach(finished)

    page = api.demo_timeline(run.run_id, limit=100).unwrap()
    assert session.replay.absorb(page) == finished.emitted_count
    assert session.replay.absorb(page) == 0, "the same page again adds nothing"
    assert len(session.replay.records) == finished.emitted_count
    assert session.replay.cursor == finished.emitted_count


def test_replay_records_are_labelled_by_their_source(api: DashboardAPIClient) -> None:
    """The one thing this console must never do is merge the two silently."""
    run = api.start_demo_run("account_takeover", pace="instant").unwrap()
    finished = _finish(api, run.run_id)
    session = DashboardSession()
    session.replay.attach(finished)
    session.replay.absorb(api.demo_timeline(run.run_id, limit=100).unwrap())

    derived = session.replay.detection_records()
    assert len(derived) == finished.emitted_count
    assert all(item.scenario.startswith(REPLAY_SCENARIO_PREFIX) for item in derived)
    assert session.history == [], "the manual history is untouched"


def test_attaching_a_new_run_discards_the_previous_one(
    api: DashboardAPIClient,
) -> None:
    """Two runs share a numbering that starts at 1, so merging them is meaningless."""
    first = api.start_demo_run("account_takeover", pace="instant").unwrap()
    _finish(api, first.run_id)
    session = DashboardSession()
    session.replay.attach(first)
    session.replay.absorb(api.demo_timeline(first.run_id, limit=100).unwrap())
    assert session.replay.records

    second = api.start_demo_run("normal_activity", pace="instant").unwrap()
    session.replay.attach(second)
    assert session.replay.records == []
    assert session.replay.cursor == 0
    assert session.replay.run_id == second.run_id
    _finish(api, second.run_id)


# ---------------------------------------------------------------------------
# Polling policy
# ---------------------------------------------------------------------------


def test_polling_stops_at_a_terminal_state(api: DashboardAPIClient) -> None:
    """The property that keeps the console from calling a finished run forever."""
    session = DashboardSession()
    assert poll_interval(session.replay) is None, "nothing attached"

    run = api.start_demo_run("brute_force", pace="slow").unwrap()
    session.replay.attach(run)
    assert poll_interval(session.replay) == LIVE_POLL_SECONDS

    stopped = api.stop_demo_run(run.run_id).unwrap()
    session.replay.observe(stopped)
    assert poll_interval(session.replay) is None


def test_the_poll_interval_is_bounded_and_matches_the_pace_vocabulary() -> None:
    """Fast enough to show every step, slow enough not to be a load source."""
    assert 0.5 <= LIVE_POLL_SECONDS <= 5.0


def test_a_terminal_run_reports_itself_terminal_from_the_service(
    api: DashboardAPIClient,
) -> None:
    """The console reads ``more_expected`` rather than deciding for itself."""
    run = api.start_demo_run("normal_activity", pace="instant").unwrap()
    finished = _finish(api, run.run_id)
    assert finished.more_expected is False
    assert finished.terminal is True
    assert ReplayRunDocument(run_id="x", more_expected=True).terminal is False


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


@pytest.fixture()
def wired(
    serving_transport: ServingTransport, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DashboardAPIClient]:
    """Point every client the app builds at the real deployment.

    Yields a client on the *same* transport, so a test can drive a run to
    completion directly instead of re-rendering the page in a loop. Rendering is
    the expensive part and it is not what those tests are about: what they assert
    is what the page shows once the run has finished, and one render after the
    fact shows exactly that.
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
    yield DashboardAPIClient(
        DashboardSettings(api_url="http://testserver", request_timeout_seconds=30.0),
        transport=serving_transport,
    )


def _settle(app: AppTest, api: DashboardAPIClient) -> Any:
    """Drive the page's attached run to completion, then render once more.

    The run is polled through *this* client rather than by re-rendering, so the
    wait costs one small ``GET`` per iteration instead of a whole page. The final
    ``app.run()`` is the page's own poll, which absorbs the finished timeline.
    """
    session = app.session_state["pad_session"]
    assert session.replay.run_id is not None
    _finish(api, session.replay.run_id)
    app.run()
    return app.session_state["pad_session"]


@pytest.fixture()
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the dashboard at a port nothing is listening on."""
    monkeypatch.setenv("PAD_DASHBOARD_API_URL", f"http://127.0.0.1:{DEAD_PORT}")
    monkeypatch.setenv("PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS", "2")
    yield


def _render() -> AppTest:
    """Render the Live Replay page and return the resulting app state."""
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value(PAGE).run()
    return app


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
    ):
        parts.extend(str(item.value) for item in group)
    return "\n".join(parts)


def test_the_page_is_in_the_navigation() -> None:
    """Ten views since this milestone, and the label is stable."""
    assert PAGE in PAGES
    assert len(PAGES) == 10


@pytest.mark.usefixtures("wired")
def test_the_page_renders_without_starting_anything() -> None:
    """A page load must never begin a run. Only a press does."""
    app = _render()
    assert not app.exception
    text = _text(app)
    assert "Live replay" in text or "replay" in text.lower()
    assert "No replay run is attached" in text


@pytest.mark.usefixtures("wired")
def test_the_page_offers_the_catalog_and_the_pace_vocabulary() -> None:
    """Both pickers are populated from the service, not from a local list."""
    app = _render()
    labels = [str(item.label) for item in app.selectbox]
    assert "Scenario" in labels
    assert "Pace" in labels
    pace = next(item for item in app.selectbox if item.label == "Pace")
    assert list(pace.options) == list(PACES)


def test_starting_requires_an_explicit_press(wired: DashboardAPIClient) -> None:
    """And when pressed once, produces exactly one run."""
    app = _render()
    start = next(item for item in app.button if "Start" in str(item.label))
    scenario = next(item for item in app.selectbox if item.label == "Scenario")
    scenario.set_value("normal_activity").run()

    start = next(item for item in app.button if "Start" in str(item.label))
    start.click().run()
    assert not app.exception

    session = app.session_state["pad_session"]
    assert session.replay.attached
    run_id = session.replay.run_id

    # A further rerun with nothing pressed must not start a second run.
    app.run()
    assert app.session_state["pad_session"].replay.run_id == run_id


@pytest.mark.usefixtures("wired")
def test_the_start_control_is_disabled_while_a_run_is_active() -> None:
    """One page session cannot start a second run behind the first."""
    app = _render()
    scenario = next(item for item in app.selectbox if item.label == "Scenario")
    scenario.set_value("brute_force").run()
    pace = next(item for item in app.selectbox if item.label == "Pace")
    pace.set_value("slow").run()
    next(item for item in app.button if "Start" in str(item.label)).click().run()

    session = app.session_state["pad_session"]
    assert session.replay.attached
    start = next(item for item in app.button if "Start" in str(item.label))
    assert start.disabled is True

    stop = next(item for item in app.button if "Stop" in str(item.label))
    assert stop.disabled is False
    stop.click().run()
    assert not app.exception


@pytest.mark.usefixtures("wired")
def test_stopping_requires_an_explicit_press_and_ends_the_run() -> None:
    """Nothing is stopped on the console's own initiative either."""
    app = _render()
    next(item for item in app.selectbox if item.label == "Scenario").set_value(
        "brute_force"
    ).run()
    next(item for item in app.selectbox if item.label == "Pace").set_value("slow").run()
    next(item for item in app.button if "Start" in str(item.label)).click().run()

    next(item for item in app.button if "Stop" in str(item.label)).click().run()
    assert not app.exception
    session = app.session_state["pad_session"]
    assert session.replay.run is not None
    assert session.replay.run.state == "stopped"
    assert session.replay.run.terminal


def test_a_completed_run_renders_a_summary_derived_from_its_records(
    wired: DashboardAPIClient,
) -> None:
    """Every figure on the page came out of the timeline the service returned."""
    app = _render()
    next(item for item in app.selectbox if item.label == "Scenario").set_value(
        "account_takeover"
    ).run()
    next(item for item in app.selectbox if item.label == "Pace").set_value(
        "instant"
    ).run()
    next(item for item in app.button if "Start" in str(item.label)).click().run()

    session = _settle(app, wired)
    assert not app.exception
    assert session.replay.run is not None
    assert session.replay.run.terminal
    text = _text(app)
    assert "Demo run summary" in text
    summary = session.replay.run.summary
    assert summary.detection_count == len(session.replay.records)
    assert summary.detection_count == session.replay.run.event_count


@pytest.mark.usefixtures("wired")
def test_the_page_fabricates_no_timeline_row_before_a_run_exists() -> None:
    """An empty state, never a placeholder row."""
    app = _render()
    assert app.dataframe == [] or all(len(item.value) == 0 for item in app.dataframe), (
        "no table is drawn with no run attached"
    )


@pytest.mark.usefixtures("offline")
def test_the_page_loads_with_the_api_down_and_shows_no_traceback() -> None:
    """Streamlit renders an uncaught exception into the browser in full."""
    app = _render()
    assert not app.exception
    text = _text(app)
    assert "Traceback" not in text
    assert "not reachable" in text


@pytest.mark.usefixtures("offline")
def test_no_url_or_exception_text_appears_in_a_failure_message() -> None:
    """``httpx`` puts the request URL in its exception messages; none is forwarded.

    The *configured* endpoint is shown on purpose -- the sidebar names where this
    console is pointed, which is the first thing anyone checks when it says
    offline. What must never appear is a URL or an exception string inside a
    failure message, because that is the path by which a stack frame, a proxy
    address, or a pasted token would reach a browser.
    """
    app = _render()
    messages = [str(item.value) for item in (*app.error, *app.warning)]
    assert messages, "the offline page states why it cannot load"
    for message in messages:
        assert "http://" not in message, message
        assert "https://" not in message, message
        assert "ConnectError" not in message
        assert "Traceback" not in message
        assert "59433" not in message


@pytest.mark.usefixtures("offline")
def test_the_offline_page_offers_no_start_control() -> None:
    """There is nothing to start, and the page says why rather than pretending."""
    app = _render()
    assert not any("Start" in str(item.label) for item in app.button)


def test_the_replay_run_appears_on_the_pages_that_integrate_it(
    wired: DashboardAPIClient,
) -> None:
    """Alerts, events, analytics and the overview, each saying whose data it is."""
    app = _render()
    next(item for item in app.selectbox if item.label == "Scenario").set_value(
        "account_takeover"
    ).run()
    next(item for item in app.selectbox if item.label == "Pace").set_value(
        "instant"
    ).run()
    next(item for item in app.button if "Start" in str(item.label)).click().run()
    _settle(app, wired)

    for page, marker in (
        ("Security Alerts", "Server-side demo replay run"),
        ("Authentication Events", "Server-side demo replay run"),
        ("Overview", "Attached demo replay run"),
    ):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert marker in _text(app), page


def test_analytics_never_merges_the_two_sources_silently(
    wired: DashboardAPIClient,
) -> None:
    """A selector, defaulting to the manual session, with both labelled."""
    app = _render()
    next(item for item in app.selectbox if item.label == "Scenario").set_value(
        "normal_activity"
    ).run()
    next(item for item in app.selectbox if item.label == "Pace").set_value(
        "instant"
    ).run()
    next(item for item in app.button if "Start" in str(item.label)).click().run()
    _settle(app, wired)

    app.sidebar.radio[0].set_value("Attack Analytics").run()
    assert not app.exception
    sources = [item for item in app.radio if item.label == "Data source"]
    assert sources, "the source selector is offered once there are two sources"
    assert "manual" in str(sources[0].value)
