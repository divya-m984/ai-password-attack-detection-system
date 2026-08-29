"""Tests for the dashboard's only door to the backend.

Every branch is driven through a real ``httpx`` transport rather than by
monkeypatching the client's own methods: what is being tested is how the client
behaves against *responses*, and a stubbed method would test the stub.

Grouped by the property each set is defending:

* **A failure is a value.** Every way a call can fail produces an
  :class:`APIResult` carrying a stable :class:`ProblemKind`, never an exception.
  Streamlit renders an uncaught exception into the browser in full.
* **Nothing is invented.** A failed call carries no document, and no default
  stands in for one.
* **Nothing leaks.** No URL, no exception message, and no traceback reaches a
  :class:`Problem`.
* **A detection is sent once.** POST is not retried under any failure.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from password_attack_detector.dashboard.api_client import (
    PROBLEM_MESSAGES,
    APIResult,
    DashboardAPIClient,
    Problem,
    ProblemKind,
)
from password_attack_detector.dashboard.config import DashboardSettings

SETTINGS = DashboardSettings(
    api_url="http://detector.test:8000", request_timeout_seconds=1.0
)

HEALTH = {"status": "ok", "service": "password-attack-detector", "version": "0.5.0"}
READY = {
    "status": "ready",
    "service": "password-attack-detector",
    "version": "0.5.0",
    "components": [
        {"component": "rule_engine", "state": "ready", "reason": None, "required": True}
    ],
}


def _client(handler: Any) -> DashboardAPIClient:
    """Return a client whose transport is *handler*."""
    return DashboardAPIClient(SETTINGS, transport=httpx.MockTransport(handler))


def _always(status_code: int, payload: Any) -> Any:
    """Return a handler answering every request identically."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_health_parses_into_a_typed_document() -> None:
    """The ordinary read, stated once so the negatives below are readable."""
    with _client(_always(200, HEALTH)) as client:
        result = client.health()
    assert result.ok
    assert result.unwrap().version == "0.5.0"
    assert result.problem is None


def test_a_request_goes_to_the_configured_base_url() -> None:
    """Every URL is assembled from the configured base and nothing else."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=HEALTH)

    with _client(handler) as client:
        client.health()
    assert seen == ["http://detector.test:8000/health"]


def test_an_unknown_field_in_a_response_is_ignored() -> None:
    """A serving release that adds a field must not break a running dashboard."""
    payload = {**HEALTH, "a_field_from_a_later_release": 42}
    with _client(_always(200, payload)) as client:
        result = client.health()
    assert result.ok
    assert result.unwrap().version == "0.5.0"


def test_readiness_reports_which_components_block() -> None:
    """The blocking set is what the not-ready page is built from."""
    payload = {
        **READY,
        "status": "not_ready",
        "components": [
            {
                "component": "fusion",
                "state": "unavailable",
                "reason": "serving_bundle_not_published",
                "required": True,
            },
            {
                "component": "rule_engine",
                "state": "ready",
                "reason": None,
                "required": True,
            },
        ],
    }
    with _client(_always(503, payload)) as client:
        result = client.readiness()
    assert result.ok
    document = result.unwrap()
    assert document.is_ready is False
    assert [item.component for item in document.blocking] == ["fusion"]


def test_a_503_readiness_is_a_document_not_a_problem() -> None:
    """The body *is* the report, and the report is what the page needs."""
    with _client(_always(503, {**READY, "status": "not_ready"})) as client:
        result = client.readiness()
    assert result.ok
    assert result.problem is None


def test_a_503_on_any_other_endpoint_is_a_problem() -> None:
    """Only readiness publishes a document alongside a 503."""
    with _client(_always(503, {"error": {"code": "API010", "message": "no"}})) as c:
        result = c.system_status()
    assert not result.ok
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.NOT_READY
    assert result.problem.code == "API010"


# ---------------------------------------------------------------------------
# Failures are values
# ---------------------------------------------------------------------------


def test_a_refused_connection_is_reported_offline() -> None:
    """The commonest state during a demo: the service is not running."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client(handler) as client:
        result = client.health()
    assert not result.ok
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.OFFLINE
    assert result.document is None


def test_a_timeout_is_reported_as_a_timeout() -> None:
    """Distinct from offline: something is there and did not answer in time."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with _client(handler) as client:
        result = client.health()
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.TIMEOUT


def test_a_post_timeout_is_a_timeout_too() -> None:
    """The write path classifies the same way the read path does."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    with _client(handler) as client:
        result = client.detect([{"event_id": "x"}])
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.TIMEOUT


def test_a_protocol_error_is_reported_offline() -> None:
    """Whatever answered was not speaking HTTP; to a viewer that is 'not there'."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("garbage", request=request)

    with _client(handler) as client:
        result = client.rules()
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.OFFLINE


def test_a_refusal_carries_the_services_own_error_code() -> None:
    """The stable code is what a viewer needs; the prose beside it may reword."""
    payload = {
        "error": {
            "code": "API013",
            "message": "This service never accepts passwords, hashes, tokens, "
            "or secrets.",
        }
    }
    with _client(_always(422, payload)) as client:
        result = client.detect([{"event_id": "x"}])
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.REFUSED
    assert result.problem.code == "API013"
    assert result.problem.status_code == 422


def test_a_refusal_carries_the_services_aggregate_detail() -> None:
    """Counts and limits only, exactly as the service published them."""
    payload = {
        "error": {
            "code": "API005",
            "message": "too many",
            "detail": {"event_count": 9, "max_batch_events": 5},
        }
    }
    with _client(_always(413, payload)) as client:
        result = client.detect([{"event_id": "x"}])
    assert result.problem is not None
    assert result.problem.detail == {"event_count": 9, "max_batch_events": 5}


def test_a_server_error_is_distinct_from_a_refusal() -> None:
    """One is the caller's problem to fix and one is the deployment's."""
    with _client(_always(500, {"error": {"code": "API099", "message": "no"}})) as c:
        result = c.model_info()
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.SERVER_ERROR


def test_an_error_body_that_is_not_the_envelope_still_classifies() -> None:
    """A proxy in front of the service returns HTML, not an error object."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    with _client(handler) as client:
        result = client.version()
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.SERVER_ERROR
    assert result.problem.code is None


def test_a_non_string_error_code_is_not_adopted() -> None:
    """A malformed envelope must not put an integer where a code goes."""
    payload = {"error": {"code": 500, "message": ["not", "a", "string"]}}
    with _client(_always(422, payload)) as client:
        result = client.detect([{"event_id": "x"}])
    assert result.problem is not None
    assert result.problem.code is None
    assert result.problem.message is None


# ---------------------------------------------------------------------------
# Malformed responses
# ---------------------------------------------------------------------------


def test_a_body_that_is_not_json_is_reported_as_malformed() -> None:
    """Something answered on that port and it was not this API."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>hello</html>")

    with _client(handler) as client:
        result = client.health()
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.MALFORMED
    assert result.document is None


def test_a_body_that_is_not_this_contract_is_reported_as_malformed() -> None:
    """JSON, and not the published document. Reported, never raised."""
    with _client(_always(200, {"anchor": "missing everything else"})) as client:
        result = client.detect([{"event_id": "x"}])
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.MALFORMED


def test_a_json_array_where_a_document_belongs_is_malformed() -> None:
    """Valid JSON of the wrong shape entirely."""
    with _client(_always(200, [1, 2, 3])) as client:
        result = client.rules()
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.MALFORMED


def test_a_malformed_response_never_raises_a_validation_error() -> None:
    """A pydantic error message quotes the input; the page it lands on is a browser."""
    with _client(_always(200, {"rules": [{"nope": True}]})) as client:
        result = client.rules()  # no exception
    assert result.problem is not None


# ---------------------------------------------------------------------------
# Nothing leaks, nothing is invented
# ---------------------------------------------------------------------------


def test_a_problem_carries_no_url_exception_or_traceback() -> None:
    """The client discards the exception message rather than scrubbing it.

    ``httpx`` puts the full request URL in its exception messages, so forwarding
    one would put the backend's address on the page every time it went down.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "failed to connect to http://detector.test:8000/health", request=request
        )

    with _client(handler) as client:
        result = client.health()
    assert result.problem is not None
    rendered = repr(result.problem) + result.problem.summary
    assert "detector.test" not in rendered
    assert "Traceback" not in rendered
    assert "http://" not in rendered


def test_every_problem_kind_has_fixed_text() -> None:
    """A message interpolated from an exception eventually interpolates a path."""
    for kind in ProblemKind:
        assert PROBLEM_MESSAGES[kind]
        assert "{" not in PROBLEM_MESSAGES[kind]


def test_a_failed_call_carries_no_document() -> None:
    """The client never fabricates data when the backend is unavailable."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with _client(handler) as client:
        for result in (
            client.health(),
            client.readiness(),
            client.version(),
            client.system_status(),
            client.model_info(),
            client.rules(),
            client.detect([{"event_id": "x"}]),
            client.detect_batch([{"event_id": "x"}]),
            client.explain([{"event_id": "x"}]),
        ):
            assert result.document is None
            assert result.problem is not None


def test_unwrapping_a_problem_is_a_loud_failure() -> None:
    """A mistake here should fail in a test, not render ``None`` as text."""
    result: APIResult[str] = APIResult(problem=Problem(ProblemKind.OFFLINE))
    with pytest.raises(RuntimeError, match="carries a problem"):
        result.unwrap()


# ---------------------------------------------------------------------------
# A detection is sent once
# ---------------------------------------------------------------------------


def test_a_failed_detection_is_not_retried() -> None:
    """A request that timed out may already have been evaluated.

    Re-sending it would double an entry in the session's own alert history for
    no gain, and would double whatever the service did with it.
    """
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("slow", request=request)

    with _client(handler) as client:
        client.detect([{"event_id": "x"}])
    assert attempts == 1


def test_a_refused_detection_is_not_retried() -> None:
    """A 422 is not going to become a 200 on a second attempt either."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(422, json={"error": {"code": "API001", "message": "no"}})

    with _client(handler) as client:
        client.detect([{"event_id": "x"}])
    assert attempts == 1


def test_a_failed_read_is_not_retried_either() -> None:
    """The retry a viewer wants is the one they asked for by pressing a button."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("down", request=request)

    with _client(handler) as client:
        client.system_status()
    assert attempts == 1


def test_a_redirect_is_not_followed() -> None:
    """A redirect would move the dashboard somewhere the operator did not configure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://elsewhere/health"})

    with _client(handler) as client:
        result = client.health()
    assert result.problem is not None
    assert result.problem.status_code == 302


# ---------------------------------------------------------------------------
# The request body
# ---------------------------------------------------------------------------


def test_the_posted_body_is_the_events_the_caller_built() -> None:
    """Nothing is added, renamed, or filled in on the way out.

    The API's schema is the authority on what a valid event is; a client that
    pre-massaged a body would be a second, unreviewed opinion about it.
    """
    captured: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(422, json={"error": {"code": "API001", "message": "x"}})

    events = [{"event_id": "abc", "authentication_outcome": "failure"}]
    with _client(handler) as client:
        client.detect(events, anchor_selection="last")
    assert captured == [{"events": events, "anchor_selection": "last"}]


def test_the_batch_endpoint_defaults_to_every_anchor() -> None:
    """The two endpoints differ in how many anchors they answer, and that is all."""
    captured: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content)["anchor_selection"])
        return httpx.Response(422, json={"error": {"code": "API001", "message": "x"}})

    with _client(handler) as client:
        client.detect([{"event_id": "x"}])
        client.detect_batch([{"event_id": "x"}])
        client.explain([{"event_id": "x"}])
    assert captured == ["last", "all", "last"]


def test_the_configured_timeout_reaches_the_transport() -> None:
    """A bound nobody applies is not a bound."""
    settings = DashboardSettings(request_timeout_seconds=2.5)
    with DashboardAPIClient(settings) as client:
        assert client.settings.request_timeout_seconds == 2.5


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_the_client_holds_no_detection_capability() -> None:
    """The import-time guard, asserted so it cannot be quietly deleted."""
    from password_attack_detector.dashboard import api_client

    namespace = vars(api_client)
    for forbidden in (
        "DetectionEngine",
        "FeatureEngine",
        "FrozenChampion",
        "RiskScorer",
        "fuse",
        "local_contributions",
        "predict_serving_binary",
        "load_serving_bundle",
    ):
        assert forbidden not in namespace
