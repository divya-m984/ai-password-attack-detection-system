"""The dashboard client against the real serving application.

Not a mock transport: the client here talks to the actual FastAPI app, over
``httpx``'s ASGI transport, backed by the genuinely frozen champion the session
fixture produced.  That is the point -- the contract in
:mod:`password_attack_detector.dashboard.contracts` is re-declared rather than
imported, so the only thing that can prove it matches is a real response.

Also proved here: the console's three synthetic templates build windows the
service accepts, and the ones meant to look like attacks reach the rules that
describe them.  A template that quietly stopped validating would look, during a
demonstration, exactly like a broken API.
"""

from __future__ import annotations

from typing import Any

import pytest

from password_attack_detector.dashboard.api_client import (
    DashboardAPIClient,
    ProblemKind,
)
from password_attack_detector.dashboard.components.status import connectivity
from password_attack_detector.dashboard.config import DashboardSettings
from password_attack_detector.dashboard.scenarios import SCENARIOS, scenario_events
from password_attack_detector.dashboard.state import DashboardSession
from tests.api.factories import BASE_TIME
from tests.integration.conftest import ServingTransport


@pytest.fixture()
def dashboard_client(serving_transport: ServingTransport) -> Any:
    """A dashboard client wired straight into the real serving application."""
    settings = DashboardSettings(
        api_url="http://testserver", request_timeout_seconds=30.0
    )
    with DashboardAPIClient(settings, transport=serving_transport) as client:
        yield client


# ---------------------------------------------------------------------------
# The contract the dashboard re-declares matches the one the service publishes
# ---------------------------------------------------------------------------


def test_every_read_endpoint_parses_into_its_declared_contract(
    dashboard_client: DashboardAPIClient,
) -> None:
    """Six documents, all parsed by the client's own re-declared shapes."""
    for result in (
        dashboard_client.health(),
        dashboard_client.readiness(),
        dashboard_client.version(),
        dashboard_client.system_status(),
        dashboard_client.model_info(),
        dashboard_client.rules(),
    ):
        assert result.ok, result.problem
        assert result.problem is None


def test_health_and_readiness_report_the_frozen_deployment(
    dashboard_client: DashboardAPIClient,
) -> None:
    """Connectivity is what every page gates on, so it is checked end to end."""
    status = connectivity(dashboard_client)
    assert status.online is True
    assert status.ready is True
    assert status.readiness is not None
    assert status.readiness.blocking == ()
    states = {item.component: item.state for item in status.readiness.components}
    assert states["fusion"] == "ready"


def test_the_system_status_reports_the_stacked_hybrid(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The console's fusion panels read these three fields; they must arrive."""
    document = dashboard_client.system_status().unwrap()
    assert document.rule_detection_enabled is True
    assert document.ml_detection_enabled is True
    assert document.hybrid_detection_enabled is True
    assert document.frozen_fusion_strategy == "stacked"
    assert document.fusion_strategy == "stacked"
    assert document.hybrid_required is True
    assert document.stacked_state_fingerprint is not None
    assert len(document.stacked_state_fingerprint) == 64


def test_the_model_document_carries_the_fields_the_console_renders(
    dashboard_client: DashboardAPIClient,
) -> None:
    """Every row of the System & Model table comes from one of these."""
    document = dashboard_client.model_info().unwrap()
    assert document.available is True
    assert document.model_family
    assert document.model_id
    assert document.task == "binary_malicious"
    assert document.decision_threshold is not None
    assert document.freeze_record_id


def test_the_rule_catalog_carries_every_field_the_table_shows(
    dashboard_client: DashboardAPIClient,
) -> None:
    """A missing description or family would render as a column of blanks."""
    document = dashboard_client.rules().unwrap()
    assert document.rule_count > 0
    assert document.enabled_rule_count > 0
    for rule in document.rules:
        assert rule.rule_id
        assert rule.name
        assert rule.description
        assert rule.family
        assert rule.attack_category
        assert rule.default_severity


# ---------------------------------------------------------------------------
# The templates build windows the service accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.key)
def test_every_synthetic_template_is_accepted_by_the_service(
    dashboard_client: DashboardAPIClient, scenario: Any
) -> None:
    """A template that stopped validating would look like a broken API."""
    result = dashboard_client.detect(scenario.events())
    assert result.ok, result.problem


def test_the_brute_force_template_reaches_the_brute_force_rules(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The template is meant to look like depth against one account. It does."""
    document = dashboard_client.detect(scenario_events("brute_force")).unwrap()
    anchor = document.anchor
    assert anchor.rule.flagged is True
    assert anchor.rule.primary_attack_category == "brute_force"
    assert "PAD-BF-001" in anchor.rule.fired_rule_ids


def test_the_spraying_template_reaches_the_spraying_rule(
    dashboard_client: DashboardAPIClient,
) -> None:
    """Breadth rather than depth, and the fan-out rule is the one that fires."""
    document = dashboard_client.detect(scenario_events("spraying")).unwrap()
    assert document.anchor.rule.flagged is True
    assert "PAD-PS-001" in document.anchor.rule.fired_rule_ids


def test_the_normal_template_is_not_flagged_by_the_rule_layer(
    dashboard_client: DashboardAPIClient,
) -> None:
    """A template that flagged everything would demonstrate nothing."""
    document = dashboard_client.detect(scenario_events("normal")).unwrap()
    assert document.anchor.rule.flagged is False
    assert document.anchor.rule.fired_rule_ids == ()


def test_a_detection_reports_all_three_layers(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The console renders three columns; all three have to be populated."""
    anchor = dashboard_client.detect(scenario_events("brute_force")).unwrap().anchor
    assert anchor.ml.available is True
    assert anchor.ml.decision_score is not None
    assert anchor.ml.decision_threshold is not None
    assert anchor.hybrid.available is True
    assert anchor.hybrid.strategy == "stacked"
    assert isinstance(anchor.hybrid.flagged, bool)


def test_the_batch_endpoint_returns_an_anchor_per_event(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The console's 'every event' anchor mode."""
    events = scenario_events("normal")
    document = dashboard_client.detect_batch(events).unwrap()
    assert len(document.anchors) == len(events)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_the_explain_endpoint_parses_into_the_declared_contract(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The explainability page renders exactly these fields."""
    document = dashboard_client.explain(scenario_events("brute_force")).unwrap()
    assert document.available is True
    assert document.method
    assert document.decision_value is not None
    assert document.baseline_value is not None
    assert document.reconstruction_residual is not None
    assert document.contributions
    assert document.transformed_feature_count > 0


def test_an_attribution_carries_no_feature_value(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The console has nowhere to render one, and the service sends none."""
    document = dashboard_client.explain(scenario_events("brute_force")).unwrap()
    for item in document.contributions:
        assert set(item.model_dump()) == {"transformed_feature", "contribution"}


# ---------------------------------------------------------------------------
# Refusals reach the console as stable codes
# ---------------------------------------------------------------------------


def test_a_credential_field_reaches_the_console_as_api013(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The console shows the code; the viewer learns what was refused and why."""
    events = [dict(item) for item in scenario_events("normal")]
    events[0]["password"] = "irrelevant"
    result = dashboard_client.detect(events)
    assert not result.ok
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.REFUSED
    assert result.problem.code == "API013"


def test_a_misordered_window_reaches_the_console_as_api004(
    dashboard_client: DashboardAPIClient,
) -> None:
    """The console does not pre-validate ordering; the service is the authority."""
    events = list(reversed(scenario_events("brute_force")))
    result = dashboard_client.detect(events)
    assert result.problem is not None
    assert result.problem.code == "API004"


def test_an_empty_window_is_refused_rather_than_scored(
    dashboard_client: DashboardAPIClient,
) -> None:
    """A window with no events has no anchor and no history."""
    result = dashboard_client.detect([])
    assert result.problem is not None
    assert result.problem.kind is ProblemKind.REFUSED


# ---------------------------------------------------------------------------
# The session records what the service returned
# ---------------------------------------------------------------------------


def test_a_session_records_the_services_own_verdicts(
    dashboard_client: DashboardAPIClient,
) -> None:
    """End to end: template, request, response, session record."""
    session = DashboardSession()
    session.load_scenario("brute_force", scenario_events("brute_force"))
    document = dashboard_client.detect(session.draft_events).unwrap()
    record = session.record(document, observed_at=BASE_TIME)

    assert record.sequence == 1
    assert record.scenario == "brute_force"
    assert record.severity == document.anchor.severity
    assert record.rule_flagged == document.anchor.rule.flagged
    assert record.ml_flagged == document.anchor.ml.flagged
    assert record.hybrid_flagged == document.anchor.hybrid.flagged
    assert record.hybrid_strategy == "stacked"
    assert record.fired_rule_ids == document.anchor.rule.fired_rule_ids


def test_the_session_never_alters_a_verdict(
    dashboard_client: DashboardAPIClient,
) -> None:
    """Whatever the frozen system decided is what the session holds."""
    session = DashboardSession()
    for key in ("normal", "brute_force", "spraying"):
        session.load_scenario(key, scenario_events(key))
        document = dashboard_client.detect(session.draft_events).unwrap()
        record = session.record(document, observed_at=BASE_TIME)
        assert record.hybrid_flagged is document.anchor.hybrid.flagged
        assert record.rule_risk_score == document.anchor.rule.risk_score
