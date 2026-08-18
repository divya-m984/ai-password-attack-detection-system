"""Tests for the dashboard configuration.

Two things are being pinned here, and only one of them is ordinary validation.

The ordinary part: defaults, bounds, and the URL rules -- checked because a
misconfigured base URL should be a startup error naming the field rather than a
connection failure that reads like the API being down.

The part that matters: the configuration **cannot restate a frozen scientific
decision, and cannot hold a secret**. A model picker or a threshold slider on a
console would produce screenshots of a system nobody deployed, and an API key
field would be a credential with nowhere to go and a file to leak out of. The
import-time guard refuses both; these tests keep the guard honest.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from password_attack_detector.dashboard.config import (
    ALLOWED_API_SCHEMES,
    DEFAULT_API_URL,
    MAX_REFRESH_SECONDS,
    MAX_REQUEST_TIMEOUT_SECONDS,
    MIN_REFRESH_SECONDS,
    PROHIBITED_SETTING_NAMES,
    DashboardSettings,
    load_dashboard_settings,
)
from password_attack_detector.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_the_defaults_describe_a_local_two_terminal_demo() -> None:
    """The out-of-the-box configuration is the documented local setup."""
    settings = DashboardSettings()
    assert settings.api_url == DEFAULT_API_URL
    assert settings.api_url == "http://127.0.0.1:8000"
    assert settings.request_timeout_seconds == 5.0
    assert settings.refresh_seconds == 30
    assert settings.page_title == "AI-Powered Password Attack Detection System"


def test_the_environment_supplies_every_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each field is reachable under its documented ``PAD_DASHBOARD_`` name."""
    monkeypatch.setenv("PAD_DASHBOARD_API_URL", "https://detector.internal:9443")
    monkeypatch.setenv("PAD_DASHBOARD_REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("PAD_DASHBOARD_REFRESH_SECONDS", "120")
    monkeypatch.setenv("PAD_DASHBOARD_PAGE_TITLE", "Console")
    settings = load_dashboard_settings()
    assert settings.api_url == "https://detector.internal:9443"
    assert settings.request_timeout_seconds == 12.5
    assert settings.refresh_seconds == 120
    assert settings.page_title == "Console"


def test_a_constructor_override_beats_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """How a test drives a client without touching the process environment."""
    monkeypatch.setenv("PAD_DASHBOARD_API_URL", "http://127.0.0.1:8000")
    settings = load_dashboard_settings(api_url="http://127.0.0.1:9999")
    assert settings.api_url == "http://127.0.0.1:9999"


# ---------------------------------------------------------------------------
# The URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
        ("  http://127.0.0.1:8000  ", "http://127.0.0.1:8000"),
        ("https://host/api/", "https://host/api"),
    ],
)
def test_a_url_is_normalised(given: str, expected: str) -> None:
    """A trailing slash and surrounding space are not two different services."""
    assert DashboardSettings(api_url=given).api_url == expected


@pytest.mark.parametrize(
    "given",
    [
        "file:///etc/passwd",
        "ftp://host/",
        "127.0.0.1:8000",
        "/api/v1",
        "http://",
        "http://host/?query=1",
        "http://host/#fragment",
        "",
    ],
)
def test_an_unusable_url_is_a_configuration_error(given: str) -> None:
    """A scheme the client will not call is refused at load, not at request."""
    with pytest.raises(ConfigurationError):
        load_dashboard_settings(api_url=given)


def test_only_http_schemes_are_admitted() -> None:
    """Stated as data as well as behaviour, so the list cannot drift silently."""
    assert frozenset({"http", "https"}) == ALLOWED_API_SCHEMES


def test_an_endpoint_is_built_from_the_configured_base() -> None:
    """The single place a request URL is assembled."""
    settings = DashboardSettings(api_url="http://127.0.0.1:8000")
    assert settings.endpoint("/health") == "http://127.0.0.1:8000/health"
    assert settings.endpoint("health") == "http://127.0.0.1:8000/health"
    assert settings.endpoint("/api/v1/detect") == "http://127.0.0.1:8000/api/v1/detect"


def test_a_configuration_failure_message_carries_no_offending_value() -> None:
    """The message is rendered onto the page, so it must not quote the input.

    ``str(ValidationError)`` appends ``input_value=...``. The value that most
    often breaks ``api_url`` is a URL somebody pasted a token into, so the
    message is built from field names and rules instead and the value is dropped
    rather than scrubbed.
    """
    with pytest.raises(ConfigurationError) as caught:
        load_dashboard_settings(api_url="ftp://user:s3cr3t-token@host/path")
    message = str(caught.value)
    assert "s3cr3t-token" not in message
    assert "input_value" not in message
    assert "host" not in message
    # It still says which field failed and which rule it broke.
    assert "api_url" in message
    assert "http" in message


def test_a_configuration_failure_names_every_failing_field() -> None:
    """An operator fixing two mistakes should be told about both."""
    with pytest.raises(ConfigurationError) as caught:
        load_dashboard_settings(api_url="ftp://host/", refresh_seconds=1)
    message = str(caught.value)
    assert "api_url" in message
    assert "refresh_seconds" in message


def test_the_displayed_url_carries_no_userinfo() -> None:
    """The one place a URL is rendered onto a page strips any credential.

    Nothing should ever put one there -- the settings schema declares no
    credential field -- but a URL is exactly the kind of string somebody
    eventually pastes a token into.
    """
    settings = DashboardSettings(api_url="https://user:s3cret@host:9443/api")
    assert settings.display_api_url == "https://host:9443/api"
    assert "s3cret" not in settings.display_api_url
    assert "user" not in settings.display_api_url


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("given", [0.0, -1.0, MAX_REQUEST_TIMEOUT_SECONDS + 1])
def test_an_unbounded_timeout_is_refused(given: float) -> None:
    """A timeout an operator could set to 'wait forever' is not a timeout."""
    with pytest.raises(ConfigurationError):
        load_dashboard_settings(request_timeout_seconds=given)


@pytest.mark.parametrize("given", [0, MIN_REFRESH_SECONDS - 1, MAX_REFRESH_SECONDS + 1])
def test_a_refresh_interval_outside_the_bounds_is_refused(given: int) -> None:
    """The lower bound is what keeps a refresh control from becoming a load source."""
    with pytest.raises(ConfigurationError):
        load_dashboard_settings(refresh_seconds=given)


def test_an_empty_page_title_is_refused() -> None:
    """A blank header is a page that does not say what it is."""
    with pytest.raises(ConfigurationError):
        load_dashboard_settings(page_title="")


def test_the_settings_are_frozen() -> None:
    """A page that could rewrite the configuration mid-render is a page that will."""
    settings = DashboardSettings()
    with pytest.raises(ValidationError, match=r"frozen|immutable"):
        settings.api_url = "http://elsewhere"


# ---------------------------------------------------------------------------
# What cannot be configured
# ---------------------------------------------------------------------------


def test_no_scientific_setting_is_declared() -> None:
    """The guard's own claim, asserted so the guard cannot be quietly deleted."""
    declared = set(DashboardSettings.model_fields)
    assert not declared & PROHIBITED_SETTING_NAMES


@pytest.mark.parametrize(
    "name",
    [
        "model_id",
        "model_family",
        "decision_threshold",
        "threshold",
        "fusion_strategy",
        "score_kind",
        "champion_scope_key",
        "artifact_root",
        "serving_bundle_root",
    ],
)
def test_the_prohibited_list_names_every_scientific_override(name: str) -> None:
    """Each of these would let a console publish a system nobody deployed."""
    assert name in PROHIBITED_SETTING_NAMES


@pytest.mark.parametrize("name", ["api_key", "token", "secret", "password"])
def test_the_prohibited_list_names_every_credential_field(name: str) -> None:
    """The service accepts no credential, so the client needs none to hold."""
    assert name in PROHIBITED_SETTING_NAMES


def test_an_unknown_environment_variable_is_ignored_not_adopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``PAD_DASHBOARD_DECISION_THRESHOLD`` sets nothing, because nothing reads it."""
    monkeypatch.setenv("PAD_DASHBOARD_DECISION_THRESHOLD", "0.05")
    monkeypatch.setenv("PAD_DASHBOARD_MODEL_ID", "something-else")
    settings = load_dashboard_settings()
    assert not hasattr(settings, "decision_threshold")
    assert not hasattr(settings, "model_id")
    assert settings.api_url == DEFAULT_API_URL


def test_the_import_guard_fires_on_a_field_set_that_offends() -> None:
    """The guard itself, exercised against a field set that does declare one.

    A guard nobody has ever seen fire is a guard nobody knows works, so it takes
    the field names as a parameter rather than always reading its own module's.
    """
    from password_attack_detector.dashboard import config as module

    with pytest.raises(ValueError, match="never which system it is"):
        module._assert_no_scientific_override_field(
            {"api_url", "decision_threshold", "page_title"}
        )


def test_the_import_guard_passes_on_the_declared_field_set() -> None:
    """Called with no argument it reads this module's own fields, and is quiet."""
    from password_attack_detector.dashboard import config as module

    module._assert_no_scientific_override_field()
    module._assert_no_scientific_override_field(DashboardSettings.model_fields)
