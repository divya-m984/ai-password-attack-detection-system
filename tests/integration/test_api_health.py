"""The operational surface, over HTTP.

Liveness, readiness, version, and the OpenAPI documents.  No frozen champion is
needed for any of it -- which is the point: an operator whose model failed to
load has to be able to ask this service what happened, and get an answer that
names the failure without naming the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from password_attack_detector import __version__
from password_attack_detector.api.app import create_app
from password_attack_detector.api.config import APISettings


@pytest.fixture()
def rule_only_client() -> Any:
    """A client for a rule-only deployment: ready, with the model layer off."""
    with TestClient(create_app(settings=APISettings(require_ml_champion=False))) as c:
        yield c


@pytest.fixture()
def unready_client(tmp_path: Path) -> Any:
    """A client for a deployment whose required champion could not be found."""
    settings = APISettings(
        require_ml_champion=True, artifact_root=tmp_path / "nothing-here"
    )
    with TestClient(create_app(settings=settings)) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_is_two_hundred_and_deterministic(rule_only_client: Any) -> None:
    """The same three keys, the same three values, every time."""
    first = rule_only_client.get("/health")
    second = rule_only_client.get("/health")
    assert first.status_code == 200
    assert first.json() == {
        "status": "ok",
        "service": "password-attack-detector",
        "version": __version__,
    }
    assert first.json() == second.json()


def test_health_answers_even_when_the_runtime_is_broken(unready_client: Any) -> None:
    """Liveness is about the process, so a missing model does not affect it."""
    assert unready_client.get("/health").status_code == 200


def test_health_names_no_component(rule_only_client: Any) -> None:
    """It performs no artifact check, so it has nothing to report about one."""
    body = rule_only_client.get("/health").json()
    assert set(body) == {"status", "service", "version"}


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_a_ready_runtime_reports_every_component(rule_only_client: Any) -> None:
    """Ready, with each component's own state visible beside the aggregate."""
    response = rule_only_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    names = {item["component"] for item in body["components"]}
    assert names == {
        "feature_contract",
        "rule_engine",
        "model_artifacts",
        "ml_champion",
        "fusion",
        # The optional demonstration replay subsystem. Reported like every other
        # component and required like none of them: a detection service does not
        # become unready because a demo facility did not initialise.
        "replay",
    }
    replay = next(item for item in body["components"] if item["component"] == "replay")
    assert replay["required"] is False


def test_a_missing_required_component_makes_readiness_fail(
    unready_client: Any,
) -> None:
    """503, not 200 with a caveat: a probe reads the status line."""
    response = unready_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    blocking = [
        item
        for item in body["components"]
        if item["required"] and item["state"] != "ready"
    ]
    assert blocking


def test_every_unready_component_names_a_stable_reason(unready_client: Any) -> None:
    """A code a client can branch on, not prose a client has to parse."""
    for item in unready_client.get("/ready").json()["components"]:
        if item["state"] == "ready":
            assert item["reason"] is None
        else:
            assert item["reason"]
            assert item["reason"] == item["reason"].lower()
            assert " " not in item["reason"]


def test_the_readiness_document_leaks_no_location(
    unready_client: Any, tmp_path: Path
) -> None:
    """The failure is named; where it happened is not."""
    text = unready_client.get("/ready").text
    assert str(tmp_path) not in text
    assert "Traceback" not in text
    assert "/home/" not in text


def test_readiness_does_not_reload_artifacts(rule_only_client: Any) -> None:
    """A probe that reloaded a model would be a denial of service with a tick."""
    first = rule_only_client.get("/ready").json()
    for _ in range(5):
        assert rule_only_client.get("/ready").json() == first


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version_reports_the_release(rule_only_client: Any) -> None:
    """Milestone 1 ships against 0.5.0 and moves no contract version."""
    body = rule_only_client.get("/version").json()
    assert body["package_version"] == __version__ == "0.5.0"
    assert body["api_schema_version"] == "1.0.0"
    assert body["event_schema_version"] == "1.0.0"
    assert body["feature_schema_version"] == "1.0.0"
    assert body["detection_schema_version"] == "1.0.0"
    assert body["scoring_version"] == "1.0.0"
    assert body["ml_schema_version"] == "1.0.0"
    assert body["fusion_schema_version"] == "1.0.0"


def test_version_names_no_host(rule_only_client: Any) -> None:
    """Two machines running this build answer identically."""
    body = rule_only_client.get("/version").json()
    for key in body:
        assert key not in {"hostname", "host", "pid", "started_at", "path"}
    assert "/" not in "".join(str(value) for value in body.values())


def test_version_answers_without_a_runtime(unready_client: Any) -> None:
    """The contract versions are build properties, not artifact properties."""
    assert unready_client.get("/version").status_code == 200


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------


def test_the_interactive_documents_are_served(rule_only_client: Any) -> None:
    """The demo needs Swagger; it is on by default and configurable."""
    assert rule_only_client.get("/docs").status_code == 200
    assert rule_only_client.get("/openapi.json").status_code == 200


def test_the_schema_is_titled_versioned_and_tagged(rule_only_client: Any) -> None:
    """A schema browser is only useful if the groups mean something."""
    schema = rule_only_client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Password Attack Detector API"
    assert schema["info"]["version"] == __version__
    assert schema["info"]["description"]
    assert {tag["name"] for tag in schema["tags"]} == {
        "Health",
        "Detection",
        "System",
        # The synthetic replay demonstration, grouped separately so a reader can
        # see at a glance that it is not part of the detection contract.
        "Demo",
    }


def test_every_route_is_tagged(rule_only_client: Any) -> None:
    """An untagged route is invisible in the group a reader is looking at."""
    schema = rule_only_client.get("/openapi.json").json()
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            assert operation.get("tags"), (path, method)


def test_the_documents_can_be_switched_off() -> None:
    """A public port that should not expose a schema browser can turn it off."""
    settings = APISettings(require_ml_champion=False, docs_enabled=False)
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/health").status_code == 200


def test_an_unknown_route_uses_the_error_envelope(rule_only_client: Any) -> None:
    """One envelope for every failure, including the ones the router produces."""
    body = rule_only_client.get("/not-a-route").json()
    assert body["error"]["code"] == "API015"
    assert "Traceback" not in rule_only_client.get("/not-a-route").text


def test_a_wrong_method_is_refused_with_its_own_status(rule_only_client: Any) -> None:
    """405 stays 405; the envelope does not flatten the status it wraps."""
    response = rule_only_client.get("/api/v1/detect")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "API015"
