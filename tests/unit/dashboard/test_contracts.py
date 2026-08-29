"""The dashboard restates several things it deliberately does not import.

Each restatement buys a real boundary -- the console can talk to the API over a
socket and could not compute a verdict if it tried -- and each one costs the
possibility of silently drifting from the thing it restates.  These tests are
what makes that trade honest: the drift becomes a failing test rather than a form
offering eleven of twelve options, or a page quoting a threshold nobody uses any
more.

The imports below are the ones the *dashboard* may not make.  A test is not the
dashboard: it is exactly the right place to hold the two definitions side by
side.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from password_attack_detector.api.schemas import (
    API_SCHEMA_VERSION,
    AuthEventRequest,
    DetectionResponse,
    ExplanationResponse,
    ModelInfoResponse,
    ReadinessResponse,
    RuleCatalogResponse,
    RuleSummary,
    SystemStatusResponse,
    VersionResponse,
)
from password_attack_detector.dashboard import contracts
from password_attack_detector.dashboard.components.header import (
    ADVANCED_PAGES,
    PAGES,
    PRIMARY_PAGES,
)
from password_attack_detector.dashboard.scenarios import (
    AUTHENTICATION_METHODS,
    AUTHENTICATION_OUTCOMES,
    BLOCKED_FAILURE_REASONS,
    CLIENT_TYPES,
    FAILURE_REASONS,
    MFA_OUTCOMES,
    PROHIBITED_EVENT_FIELDS,
)
from password_attack_detector.dashboard.views import drift as drift_view
from password_attack_detector.data.enums import (
    AuthMethod,
    AuthOutcome,
    ClientType,
    FailureReason,
    MFAOutcome,
)
from password_attack_detector.data.privacy import (
    PROHIBITED_NORMALIZED,
    scan_prohibited_keys,
)
from password_attack_detector.ml.config import DriftConfig

# ---------------------------------------------------------------------------
# The console's vocabularies match the wire contract's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "enumeration"),
    [
        (AUTHENTICATION_METHODS, AuthMethod),
        (AUTHENTICATION_OUTCOMES, AuthOutcome),
        (FAILURE_REASONS, FailureReason),
        (MFA_OUTCOMES, MFAOutcome),
        (CLIENT_TYPES, ClientType),
    ],
    ids=["method", "outcome", "failure_reason", "mfa", "client_type"],
)
def test_a_console_vocabulary_matches_the_wire_contract(
    declared: tuple[str, ...], enumeration: Any
) -> None:
    """A value added to the schema must not leave the form offering the old set."""
    assert set(declared) == {member.value for member in enumeration}


def test_the_blocked_failure_reasons_are_a_subset_the_schema_accepts() -> None:
    """Offering the rest would build a window the API correctly refuses.

    Checked by construction rather than against a private constant. The *wire*
    schema admits every reason -- it is the canonical event that carries the
    outcome/reason rule -- so each candidate is taken all the way through
    ``to_canonical_event``, which is exactly the conversion the service performs
    on arrival.
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    def accepted(reason: str) -> bool:
        request = AuthEventRequest.model_validate(
            {
                "event_id": str(uuid4()),
                "event_time": datetime(2026, 3, 4, tzinfo=UTC).isoformat(),
                "user_id": f"u:{'a' * 32}",
                "source_id": f"s:{'b' * 32}",
                "device_id": f"d:{'c' * 32}",
                "session_id": f"sess:{'d' * 32}",
                "application_id": "app-00",
                "authentication_method": "password",
                "authentication_outcome": "blocked",
                "failure_reason": reason,
            }
        )
        try:
            request.to_canonical_event(source_id=f"s:{'b' * 32}")
        except ValueError:
            return False
        return True

    assert {reason for reason in FAILURE_REASONS if accepted(reason)} == set(
        BLOCKED_FAILURE_REASONS
    )


def test_the_consoles_prohibited_field_list_covers_the_projects_own() -> None:
    """A name the project refuses must not be one this console would store.

    A superset is correct and a subset is not: the console may refuse more than
    the service does -- it is the one holding a browser session -- but a name the
    service treats as credential material must never be written into one.
    """
    assert PROHIBITED_NORMALIZED <= PROHIBITED_EVENT_FIELDS


@pytest.mark.parametrize(
    "spelling",
    [
        "password",
        "Password",
        "password_hash",
        "passwordHash",
        "Password-Hash",
        "access_token",
        "accessToken",
        "privateKey",
        "authorization",
        "cookie",
    ],
)
def test_both_scanners_refuse_the_same_spelling(spelling: str) -> None:
    """The console's normalisation is the project's, not a looser one of its own."""
    from password_attack_detector.dashboard.scenarios import prohibited_field_names

    event = {"event_id": "a", spelling: "irrelevant"}
    assert scan_prohibited_keys(event) == [spelling]
    assert prohibited_field_names(event) == (spelling,)


def test_an_ordinary_event_field_is_refused_by_neither() -> None:
    """The check must not refuse the wire contract's own fields."""
    from password_attack_detector.dashboard.scenarios import prohibited_field_names

    event = {
        "event_id": "a",
        "user_id": "u:1",
        "authentication_method": "password",
        "authentication_outcome": "failure",
        "failure_reason": "invalid_credentials",
    }
    assert scan_prohibited_keys(event) == []
    assert prohibited_field_names(event) == ()


# ---------------------------------------------------------------------------
# The re-declared documents match the ones the service publishes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("client_model", "server_model"),
    [
        (contracts.ReadinessDocument, ReadinessResponse),
        (contracts.VersionDocument, VersionResponse),
        (contracts.SystemStatusDocument, SystemStatusResponse),
        (contracts.ModelInfoDocument, ModelInfoResponse),
        (contracts.RuleCatalogDocument, RuleCatalogResponse),
        (contracts.RuleSummary, RuleSummary),
        (contracts.DetectionDocument, DetectionResponse),
        (contracts.ExplanationDocument, ExplanationResponse),
    ],
    ids=lambda item: getattr(item, "__name__", str(item)),
)
def test_a_client_document_declares_no_field_the_service_does_not(
    client_model: Any, server_model: Any
) -> None:
    """A field the client expects and the service never sends renders as a blank.

    The reverse is allowed and deliberate: the client declares only what a page
    displays, so the server having *more* fields is the normal state.
    """
    extra = set(client_model.model_fields) - set(server_model.model_fields)
    assert not extra, extra


def test_the_client_ignores_unknown_fields_and_the_service_forbids_them() -> None:
    """Opposite policies, each correct for its side.

    The service forbids because an undeclared key on the way *in* is an attack
    surface. A client that forbade would refuse to run against a newer service.
    """
    assert contracts.HealthDocument.model_config["extra"] == "ignore"
    assert ReadinessResponse.model_config["extra"] == "forbid"


def test_the_client_carries_the_api_schema_version_it_was_written_against() -> None:
    """A version bump should be a visible decision, not a silent divergence."""
    assert API_SCHEMA_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# The drift page's documented thresholds
# ---------------------------------------------------------------------------


def test_the_drift_page_quotes_the_configured_thresholds() -> None:
    """The page restates ``DriftConfig``'s defaults; they must be the same numbers.

    There is no serving drift endpoint to read them from, so the page documents
    them. A value changed in the configuration must be a failing test here rather
    than a console quoting the old one.
    """
    configured = DriftConfig()
    assert configured.psi_warn_threshold == drift_view.PSI_WARN_THRESHOLD
    assert configured.psi_alert_threshold == drift_view.PSI_ALERT_THRESHOLD
    assert configured.reference_source == drift_view.REFERENCE_SOURCE


def test_the_drift_thresholds_are_ordered_as_the_configuration_requires() -> None:
    """Alert strictly above warning, which ``DriftConfig`` enforces."""
    assert drift_view.PSI_WARN_THRESHOLD < drift_view.PSI_ALERT_THRESHOLD


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def test_every_navigation_label_has_a_view() -> None:
    """A renamed label must not leave a blank page behind it."""
    from password_attack_detector.dashboard.app import VIEWS

    assert set(VIEWS) == set(PAGES)
    assert list(VIEWS) == list(PAGES)


def test_the_navigation_offers_the_eleven_declared_areas() -> None:
    """The labels are stable: documentation and demo scripts refer to them.

    Eleven since the UI polish pass. Primary views first, then advanced,
    then About System.
    """
    assert PAGES == (
        "Overview",
        "Live Replay",
        "Alerts",
        "Analytics",
        "Explainability",
        "Drift Monitoring",
        "Detection Console",
        "Authentication Events",
        "Rule vs ML vs Hybrid",
        "System & Model",
        "About System",
    )


def test_every_navigation_target_a_view_can_request_is_a_real_page() -> None:
    """A call-to-action that navigates nowhere fails silently, which is the worst
    way for it to fail.

    The Overview's primary button navigates by writing a label into session state,
    and the sidebar falls back to index 0 for a label it does not recognise. So a
    renamed page turns "Start a live demo" into "reload the Overview", with no
    error anywhere -- on the one control a first-time viewer is most likely to
    press. The literals are read out of the source rather than restated, because
    restating them is exactly the drift being guarded against.
    """
    import ast
    from pathlib import Path

    views = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "password_attack_detector"
        / "dashboard"
        / "views"
    )
    targets: set[str] = set()
    for path in sorted(views.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "_pad_nav_target"
                    and isinstance(node.value, ast.Constant)
                ):
                    targets.add(str(node.value.value))

    assert targets, "at least one view offers a navigation shortcut"
    assert targets <= set(PAGES), sorted(targets - set(PAGES))
    # And the one the Overview's primary call to action names, specifically.
    assert "Live Replay" in targets


def test_the_navigation_groups_partition_the_navigation() -> None:
    """Primary, advanced and About are the whole of it, in that order."""
    assert (*PRIMARY_PAGES, *ADVANCED_PAGES, "About System") == PAGES
    assert not set(PRIMARY_PAGES) & set(ADVANCED_PAGES)


def test_the_sidebar_group_headings_sit_on_the_group_boundaries() -> None:
    """The stylesheet draws the headings; the ordinals must match the groups.

    The grouping is CSS over one radio rather than three radios, which keeps
    navigation to a single value -- and makes the position of each heading a
    number in a stylesheet. That number is the thing that can silently rot when
    a view is added, so it is asserted against the groups it is meant to sit on
    rather than trusted.
    """
    from password_attack_detector.dashboard.theme import STYLESHEET

    found = {
        heading: int(ordinal)
        for ordinal, heading in re.findall(
            r'nth-child\((\d+)\)::before \{\s*content: "([A-Z]+)";',
            STYLESHEET,
        )
    }
    assert found == {
        "PRIMARY": 1,
        "ADVANCED": len(PRIMARY_PAGES) + 1,
        "ABOUT": len(PRIMARY_PAGES) + len(ADVANCED_PAGES) + 1,
    }
    # And the last group is the single About page, so nothing falls below it.
    assert len(PAGES) == len(PRIMARY_PAGES) + len(ADVANCED_PAGES) + 1


def test_the_overview_prose_carries_no_literal_markdown_markers() -> None:
    """The intro block is raw HTML, so Markdown emphasis would render literally.

    ``st.markdown(..., unsafe_allow_html=True)`` on a block that is already HTML
    does not run the Markdown parser over it: ``**rule engine**`` reaches the
    page as four asterisks around the words. This is the constant that bug was
    in, and this is the assertion that keeps it out.
    """
    from password_attack_detector.dashboard.views.overview import (
        _WHAT_THIS_SYSTEM_DOES,
    )

    assert "**" not in _WHAT_THIS_SYSTEM_DOES
    assert "__" not in _WHAT_THIS_SYSTEM_DOES
    assert "<strong>" in _WHAT_THIS_SYSTEM_DOES


def test_every_view_is_callable_with_the_one_signature() -> None:
    """One signature for all ten, so no view invents its own dependencies."""
    import inspect

    from password_attack_detector.dashboard.app import VIEWS

    for label, view in VIEWS.items():
        parameters = list(inspect.signature(view).parameters)
        assert parameters == ["client", "status", "session"], label
