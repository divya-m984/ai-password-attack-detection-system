"""The serving layer's security surface.

Four claims, each tested against the running application rather than against the
code that implements it:

* **A request cannot restate a frozen decision.**  No model, threshold, fusion
  strategy, feature set, or artifact location can be named by a client, under
  any key, on any route.
* **A request is bounded.**  Event count and body size both have ceilings the
  service enforces itself.
* **A failure says what, never where.**  No path, no traceback, no internal
  message reaches a client, including when something unforeseen goes wrong.
* **No credential is accepted, and none is echoed.**
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from password_attack_detector.api.app import create_app
from password_attack_detector.api.config import APISettings
from password_attack_detector.api.errors import ERROR_MESSAGES, ErrorCode
from tests.api.factories import brute_force_window, event, normal_window


@pytest.fixture()
def client() -> Any:
    """A rule-only deployment: ready, and with a small configured ceiling."""
    settings = APISettings(require_ml_champion=False, max_batch_events=25)
    with TestClient(create_app(settings=settings)) as connected:
        yield connected


@pytest.fixture()
def tolerant_client() -> Any:
    """The same deployment, with server exceptions rendered rather than raised."""
    settings = APISettings(require_ml_champion=False)
    with TestClient(
        create_app(settings=settings), raise_server_exceptions=False
    ) as connected:
        yield connected


def _detect(client: Any, **body: Any) -> Any:
    """Post a detection request built from *body*."""
    return client.post("/api/v1/detect", json=body)


# ---------------------------------------------------------------------------
# Nothing scientific is client-settable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_id", "some-other-model"),
        ("catalog_model_id", "M-999"),
        ("model_family", "gradient_boosting"),
        ("champion_scope_key", "0" * 64),
        ("decision_threshold", 0.01),
        ("threshold", 0.99),
        ("score_kind", "decision_score"),
        ("fusion_strategy", "or_gate"),
        ("strategy", "and_gate"),
        ("artifact_root", "/etc"),
        ("allowlist_path", "/etc/passwd"),
        ("feature_config_path", "/tmp/anything.yaml"),
        ("features", {"user_failure_count__5m": 999}),
        ("risk_score", 100.0),
    ],
)
def test_a_request_cannot_name_a_frozen_quantity(
    client: Any, key: str, value: Any
) -> None:
    """Every one of these is an undeclared key, and undeclared keys are refused."""
    response = _detect(client, events=brute_force_window(3), **{key: value})
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"API001", "API013"}


def test_an_override_attempt_changes_nothing_about_the_served_model(
    client: Any,
) -> None:
    """A refused request must also be an inert one."""
    before = client.get("/api/v1/model/info").json()
    _detect(client, events=brute_force_window(3), decision_threshold=0.0)
    _detect(client, events=brute_force_window(3), model_id="something-else")
    assert client.get("/api/v1/model/info").json() == before


def test_an_override_attempt_changes_nothing_about_the_rule_catalog(
    client: Any,
) -> None:
    """The enabled rule set is deployment configuration, not request input."""
    before = client.get("/api/v1/rules").json()
    _detect(client, events=brute_force_window(3), rules=["PAD-BF-001"])
    _detect(client, events=brute_force_window(3), enabled_rule_ids=[])
    assert client.get("/api/v1/rules").json() == before


def test_an_event_cannot_carry_a_computed_feature(client: Any) -> None:
    """Features are computed from events; supplying one would bypass that."""
    contaminated = event("a") | {"user_failure_count__5m": 99}
    response = _detect(client, events=[contaminated])
    assert response.status_code == 422


def test_an_event_cannot_carry_a_verdict(client: Any) -> None:
    """A request that could assert its own answer is not a detection request."""
    for key in ("flagged", "risk_score", "malicious", "label", "severity"):
        response = _detect(client, events=[event("a") | {key: 1}])
        assert response.status_code == 422, key


def test_no_route_accepts_a_query_parameter(client: Any) -> None:
    """Nothing is configurable per request, so nothing is read from the URL."""
    schema = client.get("/openapi.json").json()
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            assert not operation.get("parameters"), (path, method)


# ---------------------------------------------------------------------------
# Requests are bounded
# ---------------------------------------------------------------------------


def test_a_window_over_the_configured_ceiling_is_refused(client: Any) -> None:
    """The deployment's own limit, reported back so a client can adapt."""
    response = _detect(client, events=brute_force_window(40))
    assert response.status_code == 413
    body = response.json()["error"]
    assert body["code"] == ErrorCode.BATCH_LIMIT_EXCEEDED.value
    assert body["detail"] == {"event_count": 40, "max_batch_events": 25}


def test_a_body_over_the_byte_ceiling_is_refused() -> None:
    """Enforced by this service, not assumed of a proxy that may not be there."""
    settings = APISettings(require_ml_champion=False, max_request_bytes=2_048)
    with TestClient(create_app(settings=settings)) as small:
        response = small.post("/api/v1/detect", json={"events": normal_window(12)})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == ErrorCode.PAYLOAD_TOO_LARGE.value


def test_an_undeclared_oversized_body_is_still_refused() -> None:
    """A body streamed without a Content-Length is counted as it arrives."""
    settings = APISettings(require_ml_champion=False, max_request_bytes=2_048)
    payload = json.dumps({"events": normal_window(12)}).encode()

    def chunks() -> Any:
        for start in range(0, len(payload), 256):
            yield payload[start : start + 256]

    with TestClient(create_app(settings=settings)) as small:
        response = small.post(
            "/api/v1/detect",
            content=chunks(),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == ErrorCode.PAYLOAD_TOO_LARGE.value


def test_a_body_within_the_ceiling_is_served(client: Any) -> None:
    """The limit refuses excess, not ordinary use."""
    assert _detect(client, events=brute_force_window(5)).status_code == 200


# ---------------------------------------------------------------------------
# Failures say what, never where
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"events": []},
        {"events": [{"event_id": "not-a-uuid"}]},
        {"events": [{}]},
        {"nonsense": True},
        {},
    ],
)
def test_a_malformed_request_gets_the_stable_envelope(
    client: Any, body: dict[str, Any]
) -> None:
    """One shape for every failure: an ``error`` object with a stable code."""
    response = client.post("/api/v1/detect", json=body)
    assert response.status_code in {413, 422}
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) <= {"code", "message", "detail"}
    assert payload["error"]["code"] in {item.value for item in ErrorCode}


def test_a_non_json_body_gets_the_stable_envelope(client: Any) -> None:
    """Unparseable input is a client error, not a server one."""
    response = client.post(
        "/api/v1/detect",
        content=b"{ not json at all",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == ErrorCode.MALFORMED_REQUEST.value


def test_no_error_response_carries_a_path_or_a_traceback(client: Any) -> None:
    """A client learns what went wrong; it never learns where this process lives."""
    bodies: tuple[dict[str, Any], ...] = (
        {"events": []},
        {"events": [{"event_id": "x"}]},
        {"junk": 1},
    )
    for body in bodies:
        text = client.post("/api/v1/detect", json=body).text
        assert "Traceback" not in text
        assert 'File "' not in text
        assert "/home/" not in text
        assert "site-packages" not in text
        assert ".py" not in text


def test_no_error_message_quotes_the_submitted_value(client: Any) -> None:
    """Echoing the input is how a validation error becomes a disclosure."""
    response = client.post(
        "/api/v1/detect",
        json={"events": [event("a", application_id="MARKER-VALUE-9df3")]},
    )
    assert "MARKER-VALUE-9df3" not in response.text


def test_an_unforeseen_failure_is_rendered_without_its_message(
    tolerant_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Debug is off, so an internal message never becomes a response body."""

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("internal detail /var/lib/secret/champion.lock")

    monkeypatch.setattr(
        "password_attack_detector.api.routes.detection.detect_single", explode
    )
    response = tolerant_client.post(
        "/api/v1/detect", json={"events": brute_force_window(3)}
    )
    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": ErrorCode.INTERNAL_ERROR.value,
            "message": ERROR_MESSAGES[ErrorCode.INTERNAL_ERROR],
        }
    }
    assert "champion.lock" not in response.text
    assert "RuntimeError" not in response.text


def test_every_error_code_has_a_fixed_message() -> None:
    """A message interpolated from an exception eventually interpolates a path."""
    for code in ErrorCode:
        assert code in ERROR_MESSAGES
        message = ERROR_MESSAGES[code]
        assert message and "/" not in message.replace("and/or", "")


# ---------------------------------------------------------------------------
# Credentials and identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["password", "password_hash", "passwordHash", "secret", "token", "cookie"],
)
def test_credential_material_is_refused_with_its_own_code(
    client: Any, field: str
) -> None:
    """One code for one thing, so a client cannot mistake it for a typo."""
    response = _detect(client, events=[event("a") | {field: "value"}])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == ErrorCode.CREDENTIAL_FIELD_REJECTED.value


def test_the_credential_refusal_echoes_neither_value_nor_count(client: Any) -> None:
    """How many secrets a request carried is itself a fact about the secrets."""
    offered = event("a") | {"password": "SECRET-VALUE-1", "token": "SECRET-VALUE-2"}
    response = _detect(client, events=[offered])
    assert "SECRET-VALUE" not in response.text
    assert response.json()["error"].get("detail") is None


def test_a_successful_response_carries_no_credential_field(client: Any) -> None:
    """There is no field for one, on any layer, in any response."""
    text = client.post("/api/v1/detect", json={"events": brute_force_window(5)}).text
    for term in ("password", "secret", "token", "credential", "api_key"):
        assert term not in text.lower()


def test_a_successful_response_carries_no_pseudonym(client: Any) -> None:
    """Entity identity is input, not output; only the caller's own anchor returns."""
    events = brute_force_window(5)
    body = client.post("/api/v1/detect", json={"events": events}).json()
    text = json.dumps(body)
    assert events[0]["user_id"] not in text
    assert events[0]["source_id"] not in text
    assert events[0]["device_id"] not in text
    assert events[0]["session_id"] not in text
    assert body["anchor"]["anchor_event_id"] == events[-1]["event_id"]


def test_a_source_address_is_never_returned() -> None:
    """Supplied for processing under the privacy contract, and never echoed."""
    settings = APISettings(require_ml_champion=False)
    body: list[dict[str, Any]] = [
        {k: v for k, v in item.items() if k != "source_id"}
        | {"source_ip": "198.51.100.77"}
        for item in brute_force_window(4)
    ]
    with TestClient(create_app(settings=settings)) as connected:
        response = connected.post("/api/v1/detect", json={"events": body})
        # With no pseudonymization key the request is refused; with one it is
        # served. Either way the address must not come back.
        assert "198.51.100.77" not in response.text
        assert response.status_code in {200, 422}


def test_no_successful_response_carries_a_filesystem_path(client: Any) -> None:
    """Nothing in a verdict describes where this deployment keeps anything."""
    for path in ("/health", "/ready", "/version", "/api/v1/system/status"):
        assert "/home/" not in client.get(path).text
    detection = client.post(
        "/api/v1/detect/batch", json={"events": brute_force_window(5)}
    ).text
    assert "/home/" not in detection
    assert ".yaml" not in detection
    assert ".parquet" not in detection


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_no_permissive_cross_origin_header_is_sent(client: Any) -> None:
    """No dashboard origin exists yet, so no origin is trusted yet."""
    response = client.get("/health", headers={"Origin": "https://example.invalid"})
    assert "access-control-allow-origin" not in {
        name.lower() for name in response.headers
    }


def test_the_application_is_not_in_debug_mode() -> None:
    """A debug application renders tracebacks into responses."""
    application = create_app(settings=APISettings(require_ml_champion=False))
    assert application.debug is False
