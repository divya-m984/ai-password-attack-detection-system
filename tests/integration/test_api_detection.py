"""Detection over HTTP, against a genuinely frozen champion.

The fixture runs the real pipeline once: publish a feature dataset, train,
select, freeze, predict, detect, and evaluate.  Nothing is hand-assembled --
a serving test that passed against a hand-built artifact would be testing a
shape the commands never produce.

What the assertions are actually about:

* the adapter reaches all three layers and keeps them apart;
* a known attack scenario reaches the rule layer as that scenario;
* ordinary traffic does not;
* the same window scored twice gives the same answer;
* nothing in a response is a path, a parameter, or an identity the privacy
  model prohibits.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from password_attack_detector.api.app import create_app
from password_attack_detector.api.config import APISettings
from tests.api.factories import brute_force_window, normal_window, spraying_window
from tests.integration.ml_workspace import (
    ML_CONFIG,
    build_workspace,
    detect,
    evaluate,
    freeze,
    materialize,
    predict,
    prediction_ids,
)


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> APISettings:
    """Run the whole Phase 5 pipeline once, then materialize its hybrid.

    The pipeline's own selection on this fixture is ``stacked``, which is the
    interesting case: it is the strategy that needs a fitted artifact, and the
    one a serving layer cannot execute from a receipt alone. So the fixture ends
    with ``deploy materialize``, and every assertion below runs against a
    deployment whose hybrid is the reconstructed meta-learner Phase 5 fitted.
    """
    base = tmp_path_factory.mktemp("api-detection")
    (base / "workspace").mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(base / "workspace")
    root = base / "artifacts"
    reports = base / "reports"
    freeze(workspace, root, reports)

    for split in ("train", "validation", "test"):
        scored = predict(workspace, root, split=split)
        assert scored.exit_code == 0, scored.output

    detection = base / "detection"
    assessed = detect(workspace, detection)
    assert assessed.exit_code == 0, assessed.output

    ids = prediction_ids(root)
    evaluated = evaluate(
        workspace,
        root,
        detection,
        reports,
        **{
            "--prediction": ids["test"],
            "--validation-prediction": ids["validation"],
        },
    )
    assert evaluated.exit_code == 0, evaluated.output

    published = materialize(
        workspace, root, detection, validation_prediction=ids["validation"]
    )
    assert published.exit_code == 0, published.output

    return APISettings(
        artifact_root=root,
        allowlist_path=workspace / "allowlist.yaml",
        feature_config_path=workspace / "features.yaml",
        ml_config_path=Path(ML_CONFIG),
        detection_config_path=detection / "rules.yaml",
    )


@pytest.fixture()
def client(served: APISettings) -> Any:
    """A client bound to the frozen champion the fixture produced."""
    with TestClient(create_app(settings=served)) as connected:
        yield connected


def _anchor(client: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Post a window and return the single anchor's verdict."""
    response = client.post("/api/v1/detect", json={"events": events})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()["anchor"]
    return body


# ---------------------------------------------------------------------------
# The runtime the pipeline produced
# ---------------------------------------------------------------------------


def test_a_frozen_champion_makes_the_runtime_ready(client: Any) -> None:
    """Every required component resolved against real artifacts."""
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    states = {item["component"]: item["state"] for item in body["components"]}
    assert states["rule_engine"] == "ready"
    assert states["feature_contract"] == "ready"
    assert states["model_artifacts"] == "ready"
    assert states["ml_champion"] == "ready"
    # The hybrid is required here, because validation froze one.
    assert states["fusion"] == "ready"
    required = {item["component"] for item in body["components"] if item["required"]}
    assert "fusion" in required


def test_the_model_document_describes_the_frozen_champion(client: Any) -> None:
    """Identity and operating point, read off the lock the freeze produced."""
    body = client.get("/api/v1/model/info").json()
    assert body["available"] is True
    assert body["task"] == "binary_malicious"
    assert body["model_family"]
    assert body["catalog_model_id"]
    assert body["model_id"]
    assert body["freeze_record_id"]
    assert body["validation_selection_id"]
    assert isinstance(body["decision_threshold"], float)
    assert body["calibrated"] is (body["score_kind"] == "calibrated_probability")


def test_the_system_status_reports_all_three_layers(client: Any) -> None:
    """Rule, model, and the materialized stacked hybrid are all running."""
    body = client.get("/api/v1/system/status").json()
    assert body["rule_detection_enabled"] is True
    assert body["ml_detection_enabled"] is True
    assert body["champion_model_family"]
    assert body["hybrid_detection_enabled"] is True
    assert body["hybrid_required"] is True
    assert body["frozen_fusion_strategy"] == "stacked"
    assert body["fusion_strategy"] == "stacked"
    assert body["fusion_unavailable_reason"] is None
    # Identity of the fitted meta-learner this process loaded, so an operator can
    # confirm which stacker is live. A digest, never a coefficient.
    assert len(body["stacked_state_fingerprint"]) == 64


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_a_brute_force_window_is_reported_as_brute_force(client: Any) -> None:
    """The scenario the rules were written for, over HTTP, end to end."""
    anchor = _anchor(client, brute_force_window(30))
    assert anchor["rule"]["flagged"] is True
    assert anchor["rule"]["primary_attack_category"] == "brute_force"
    assert "PAD-BF-001" in anchor["rule"]["fired_rule_ids"]
    assert anchor["rule"]["risk_score"] > 0.0
    assert anchor["severity"] in {"low", "medium", "high", "critical"}


def test_a_brute_force_window_carries_its_evidence(client: Any) -> None:
    """A flagged anchor explains itself in sanitized behavioral terms."""
    anchor = _anchor(client, brute_force_window(30))
    assert anchor["rule"]["evidence"]
    for item in anchor["rule"]["evidence"]:
        assert item["evidence_code"].isupper()
        assert item["message"]
        assert isinstance(item["observed_value"], bool | int | float | str)


def test_a_password_spraying_window_reaches_the_spraying_rule(client: Any) -> None:
    """One source, many accounts, one failure each: fan-out rather than depth."""
    anchor = _anchor(client, spraying_window(24))
    assert anchor["rule"]["flagged"] is True
    assert "PAD-PS-001" in anchor["rule"]["fired_rule_ids"]


def test_ordinary_traffic_is_not_flagged_by_the_rule_layer(client: Any) -> None:
    """Well-spaced successes from one stable identity are not an attack."""
    anchor = _anchor(client, normal_window(12))
    assert anchor["rule"]["flagged"] is False
    assert anchor["rule"]["risk_score"] == 0.0
    assert anchor["rule"]["fired_rule_ids"] == []


# ---------------------------------------------------------------------------
# Layer separation
# ---------------------------------------------------------------------------


def test_the_three_layers_are_separate_objects(client: Any) -> None:
    """Rule, model, and hybrid each report on their own terms."""
    anchor = _anchor(client, brute_force_window(20))
    assert set(anchor) == {
        "anchor_event_id",
        "anchor_event_time",
        "rule",
        "ml",
        "hybrid",
        "severity",
    }
    assert set(anchor["rule"]) >= {"flagged", "risk_score", "severity"}
    assert set(anchor["ml"]) >= {"available", "flagged", "decision_score"}
    assert set(anchor["hybrid"]) >= {"available", "flagged", "strategy"}


def test_the_model_layer_scores_every_anchor(client: Any) -> None:
    """A ready champion produces a decision, a score, and its own threshold."""
    anchor = _anchor(client, brute_force_window(20))
    layer = anchor["ml"]
    assert layer["available"] is True
    assert layer["unavailable_reason"] is None
    assert isinstance(layer["flagged"], bool)
    assert isinstance(layer["decision_score"], float)
    assert isinstance(layer["decision_threshold"], float)
    if layer["score_kind"] == "calibrated_probability":
        assert 0.0 <= layer["probability"] <= 1.0
    else:
        assert layer["probability"] is None


def test_the_hybrid_layer_executes_the_frozen_stacked_strategy(client: Any) -> None:
    """The strategy validation selected, running the state it selected."""
    hybrid = _anchor(client, brute_force_window(20))["hybrid"]
    assert hybrid["available"] is True
    assert hybrid["strategy"] == "stacked"
    assert isinstance(hybrid["flagged"], bool)
    assert hybrid["unavailable_reason"] is None


def _copied_deployment(served: APISettings, destination: Path) -> Path:
    """Return a private copy of the served artifact root, for tampering with."""
    assert served.artifact_root is not None
    root = destination / "artifacts"
    shutil.copytree(served.artifact_root, root)
    return root


def test_a_deployment_with_no_evaluation_has_no_hybrid(
    served: APISettings, tmp_path: Path
) -> None:
    """No locked evaluation means nothing selected a strategy, so none is used.

    And nothing is *required*: this is the honest negative, so the deployment is
    still ready and reports the hybrid as unavailable by scientific outcome.
    """
    root = _copied_deployment(served, tmp_path)
    shutil.rmtree(root / "evaluations")
    settings = served.model_copy(update={"artifact_root": root})
    with TestClient(create_app(settings=settings)) as client:
        body = client.get("/api/v1/system/status").json()
        assert body["hybrid_detection_enabled"] is False
        assert body["hybrid_required"] is False
        assert body["frozen_fusion_strategy"] is None
        assert body["fusion_unavailable_reason"] == "no_fusion_selection"
        assert body["ml_detection_enabled"] is True
        assert client.get("/ready").status_code == 200


def test_an_unreadable_evaluation_receipt_fails_the_deployment_closed(
    served: APISettings, tmp_path: Path
) -> None:
    """A receipt that does not verify is refused, not partially believed.

    The receipt is sealed, so a tampered one cannot be read back at all -- which
    is also why no test here can forge a selection: a strategy this service
    applies must have been selected by a real validation run.

    It fails **closed**: the unreadable receipt may be the one that selected a
    hybrid, so the hybrid is treated as required and unloadable rather than as
    absent. Calling this deployment ready would report a system as
    hybrid-capable on the strength of a file nobody can read.
    """
    root = _copied_deployment(served, tmp_path)
    receipt = next((root / "evaluations").glob("*/test_evaluation.json"))
    receipt.write_text('{"not": "a receipt"}', encoding="utf-8")
    settings = served.model_copy(update={"artifact_root": root})
    with TestClient(create_app(settings=settings)) as client:
        body = client.get("/api/v1/system/status").json()
        assert body["fusion_unavailable_reason"] == "evaluation_receipt_unreadable"
        assert body["hybrid_detection_enabled"] is False
        assert body["hybrid_required"] is True
        # The model layer is untouched: a broken hybrid does not break scoring.
        assert body["ml_detection_enabled"] is True
        assert client.get("/ready").status_code == 503


def test_a_missing_serving_bundle_fails_a_stacked_deployment_closed(
    served: APISettings, tmp_path: Path
) -> None:
    """A selected stacker with nothing to load is not a ready deployment."""
    root = _copied_deployment(served, tmp_path)
    shutil.rmtree(root / "serving")
    settings = served.model_copy(update={"artifact_root": root})
    with TestClient(create_app(settings=settings)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        states = {
            item["component"]: (item["state"], item["reason"], item["required"])
            for item in ready.json()["components"]
        }
        assert states["fusion"] == (
            "unavailable",
            "serving_bundle_not_published",
            True,
        )
        # And no gate is quietly substituted for the strategy that was selected.
        body = client.get("/api/v1/system/status").json()
        assert body["frozen_fusion_strategy"] == "stacked"
        assert body["fusion_strategy"] is None
        assert body["hybrid_detection_enabled"] is False


def test_a_tampered_stacked_state_fails_the_deployment_closed(
    served: APISettings, tmp_path: Path
) -> None:
    """A stacker that does not verify is never served, at any coefficient."""
    root = _copied_deployment(served, tmp_path)
    state = next((root / "serving").glob("*/fusion_stacked_state.json"))
    payload = json.loads(state.read_text(encoding="utf-8"))
    payload["coefficients"] = [99.0, 99.0]
    state.write_text(json.dumps(payload), encoding="utf-8")
    settings = served.model_copy(update={"artifact_root": root})
    with TestClient(create_app(settings=settings)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        reasons = {
            item["component"]: item["reason"] for item in ready.json()["components"]
        }
        assert reasons["fusion"] == "serving_bundle_unverifiable"
        # Detection is refused outright: the hybrid is a required component here,
        # so answering with two of three layers would be answering a different
        # question than the one this deployment claims to answer.
        refused = client.post("/api/v1/detect", json={"events": brute_force_window(6)})
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "API010"


def test_the_rule_score_and_the_model_probability_are_never_combined(
    client: Any,
) -> None:
    """Two scales, two fields, and nowhere in the response for a third number."""
    anchor = _anchor(client, brute_force_window(20))
    assert "risk_score" in anchor["rule"]
    assert "probability" in anchor["ml"]
    for layer in (anchor["rule"], anchor["ml"], anchor["hybrid"], anchor):
        assert not {
            "combined_score",
            "blended_score",
            "fused_score",
            "overall_score",
        } & set(layer)


# ---------------------------------------------------------------------------
# Determinism and batching
# ---------------------------------------------------------------------------


def test_the_same_window_scores_identically_twice(client: Any) -> None:
    """No clock, no random state, no process identity reaches a verdict."""
    events = brute_force_window(15)
    first = client.post("/api/v1/detect", json={"events": events}).json()
    second = client.post("/api/v1/detect", json={"events": events}).json()
    assert first == second


def test_the_batch_endpoint_answers_every_anchor(client: Any) -> None:
    """One validated ordered batch, one verdict per event in it."""
    events = brute_force_window(12)
    response = client.post("/api/v1/detect/batch", json={"events": events})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window"]["event_count"] == 12
    assert body["window"]["anchor_count"] == 12
    assert len(body["anchors"]) == 12


def test_the_batch_is_ordered_canonically(client: Any) -> None:
    """Anchor order comes from the events, not from the request."""
    events = brute_force_window(8)
    body = client.post("/api/v1/detect/batch", json={"events": events}).json()
    times = [anchor["anchor_event_time"] for anchor in body["anchors"]]
    assert times == sorted(times)


def test_a_batch_answer_matches_the_single_answer(client: Any) -> None:
    """The endpoints differ in how many anchors they answer, not in how."""
    events = brute_force_window(10)
    single = client.post("/api/v1/detect", json={"events": events}).json()["anchor"]
    batch = client.post("/api/v1/detect/batch", json={"events": events}).json()
    assert batch["anchors"][-1] == single


def test_an_explicit_anchor_is_answered_alone(client: Any) -> None:
    """The window is context; the anchor is the question."""
    events = brute_force_window(10)
    response = client.post(
        "/api/v1/detect",
        json={
            "events": events,
            "anchor_selection": "explicit",
            "anchor_event_ids": [events[4]["event_id"]],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["anchor"]["anchor_event_id"] == events[4]["event_id"]


def test_the_window_summary_reports_what_was_evaluated(client: Any) -> None:
    """Counts and contract versions, and nothing located or private."""
    body = client.post("/api/v1/detect", json={"events": brute_force_window(7)}).json()
    window = body["window"]
    assert window["event_count"] == 7
    assert window["evaluated_snapshot_count"] == 7
    assert window["anchor_count"] == 1
    assert window["feature_schema_version"] == "1.0.0"
    assert window["enabled_rule_count"] > 0


# ---------------------------------------------------------------------------
# Regression: this milestone adds no way to change what is served
# ---------------------------------------------------------------------------


def test_the_service_exposes_no_retraining_promotion_or_upload_route(
    client: Any,
) -> None:
    """The API reads frozen state. There is no endpoint that could write any."""
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {
        "/health",
        "/ready",
        "/version",
        "/api/v1/detect",
        "/api/v1/detect/batch",
        "/api/v1/system/status",
        "/api/v1/model/info",
        "/api/v1/rules",
    }
    for forbidden in ("train", "retrain", "promote", "freeze", "upload", "select"):
        assert not any(forbidden in path for path in paths)


def test_no_document_from_a_loaded_runtime_names_a_location(client: Any) -> None:
    """A loaded champion is the case where a path could plausibly escape."""
    for path in ("/ready", "/version", "/api/v1/system/status", "/api/v1/model/info"):
        text = client.get(path).text
        assert "/home/" not in text
        assert "/tmp/" not in text
        assert ".parquet" not in text
        assert ".yaml" not in text
        assert "champion.lock" not in text


def test_the_model_document_carries_no_parameters(client: Any) -> None:
    """Identity and decision semantics; never a coefficient or a tree."""
    body = client.get("/api/v1/model/info").json()
    for forbidden in (
        "coefficients",
        "intercept",
        "parameters",
        "arrays",
        "feature_names",
        "raw_feature_names",
        "transformed_feature_names",
        "hyperparameters",
    ):
        assert forbidden not in body


def test_a_detection_response_from_a_loaded_runtime_leaks_nothing(
    client: Any,
) -> None:
    """The full three-layer payload, swept for identity and location."""
    events = brute_force_window(8)
    text = client.post("/api/v1/detect/batch", json={"events": events}).text
    for identifier in ("user_id", "source_id", "device_id", "session_id"):
        assert events[0][identifier] not in text
    assert "/home/" not in text
    assert "Traceback" not in text
    for term in ("password", "secret", "token", "credential"):
        assert term not in text.lower()


def test_only_detection_routes_accept_a_body(client: Any) -> None:
    """Every other route is a read; a read has nothing to accept."""
    schema = client.get("/openapi.json").json()
    with_bodies = {
        path
        for path, operations in schema["paths"].items()
        for operation in operations.values()
        if "requestBody" in operation
    }
    assert with_bodies == {"/api/v1/detect", "/api/v1/detect/batch"}
