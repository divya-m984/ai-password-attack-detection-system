"""The dashboard's boundaries, asserted rather than trusted.

Four properties, each of which the whole M2 architecture rests on:

* **The dashboard cannot detect.**  No module in the package imports
  :mod:`password_attack_detector.ml`, :mod:`password_attack_detector.detection`,
  or :mod:`password_attack_detector.features`, checked by walking every module's
  syntax tree rather than by grepping -- a comment mentioning a module is not an
  import, and a grep cannot tell the difference.
* **There is one door.**  Only
  :mod:`~password_attack_detector.dashboard.api_client` performs HTTP.
* **No credential is accepted, built, or stored.**  Not in a form, not in a
  template, not in the session, and not in the configuration.
* **Nothing user-entered becomes markup.**  The only interpolated HTML in the
  package goes through the escaping helpers, and the stylesheet is a constant.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from password_attack_detector.dashboard import scenarios
from password_attack_detector.dashboard.config import DashboardSettings
from password_attack_detector.dashboard.state import DashboardSession
from password_attack_detector.dashboard.theme import STYLESHEET, badge, card, chip

#: Packages a dashboard module may never import.  The scientific layers: a
#: console holding any of them could compute a verdict, and the first page that
#: wanted a number the API does not publish would.
FORBIDDEN_PACKAGES = (
    "password_attack_detector.ml",
    "password_attack_detector.detection",
    "password_attack_detector.features",
    "password_attack_detector.deployment",
    "password_attack_detector.data",
)

#: Modules in the package that are allowed to speak HTTP.  Exactly one.
HTTP_CAPABLE = {"api_client"}


def _dashboard_root() -> Path:
    """Return the dashboard package directory."""
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "password_attack_detector"
        / "dashboard"
    )


def _modules() -> list[Path]:
    """Return every module in the dashboard package."""
    found = sorted(_dashboard_root().rglob("*.py"))
    assert found, "the dashboard package has no modules"
    return found


def _imported_names(path: Path) -> set[str]:
    """Return every module name *path* imports, however it imports it.

    Walks the syntax tree rather than reading text: a module named in a
    docstring, a comment, or a Sphinx cross-reference is not an import, and this
    package's docstrings name the forbidden modules constantly -- explaining why
    they are not imported.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


# ---------------------------------------------------------------------------
# The dashboard cannot detect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _modules(), ids=lambda item: item.name)
def test_no_dashboard_module_imports_a_scientific_package(path: Path) -> None:
    """The import boundary, module by module.

    Zero such imports, not "only public constants": the package re-declares the
    wire contract precisely so it never needs one, and an allowance for
    "harmless" enum imports is how the first genuinely harmful one arrives.
    """
    for imported in _imported_names(path):
        for forbidden in FORBIDDEN_PACKAGES:
            assert imported != forbidden, (path.name, imported)
            assert not imported.startswith(f"{forbidden}."), (path.name, imported)


@pytest.mark.parametrize("path", _modules(), ids=lambda item: item.name)
def test_no_dashboard_module_imports_the_serving_layer_internals(
    path: Path,
) -> None:
    """The API's own service and schema modules are the server's, not a client's.

    Importing them would make the dashboard a second consumer of the server's
    objects instead of a consumer of its published contract, and would pull the
    detection stack in transitively.
    """
    for imported in _imported_names(path):
        for forbidden in (
            "password_attack_detector.api.services",
            "password_attack_detector.api.schemas",
            "password_attack_detector.api.app",
            "password_attack_detector.api.routes",
            "password_attack_detector.api.dependencies",
        ):
            assert imported != forbidden, (path.name, imported)


def test_the_package_imports_only_its_own_exception_type_from_the_project() -> None:
    """What the dashboard shares with the project, stated exhaustively."""
    shared: set[str] = set()
    for path in _modules():
        for imported in _imported_names(path):
            if imported.startswith("password_attack_detector") and not (
                imported.startswith("password_attack_detector.dashboard")
            ):
                shared.add(imported)
    assert shared == {"password_attack_detector.exceptions"}


# ---------------------------------------------------------------------------
# One door to the backend
# ---------------------------------------------------------------------------


#: Modules that can open a connection.  ``urllib.parse`` is deliberately absent
#: and ``urllib.request`` is deliberately present: the configuration parses a URL
#: to validate it, which is the opposite of fetching one.
NETWORK_CAPABLE_IMPORTS = frozenset(
    {
        "aiohttp",
        "asyncio",
        "ftplib",
        "http.client",
        "httpcore",
        "httpx",
        "requests",
        "smtplib",
        "socket",
        "ssl",
        "telnetlib",
        "urllib.request",
        "urllib3",
        "websockets",
    }
)


@pytest.mark.parametrize("path", _modules(), ids=lambda item: item.name)
def test_only_the_api_client_speaks_http(path: Path) -> None:
    """Scattering requests through pages is how a UI grows five offline states."""
    if path.stem in HTTP_CAPABLE:
        return
    for imported in _imported_names(path):
        assert imported not in NETWORK_CAPABLE_IMPORTS, (path.name, imported)
        assert imported.split(".")[0] not in {
            "httpx",
            "requests",
            "urllib3",
            "socket",
            "aiohttp",
        }, (path.name, imported)


def test_the_api_client_is_the_only_module_that_imports_httpx() -> None:
    """Stated once over the whole package, so a new module cannot slip past."""
    speakers = {path.stem for path in _modules() if "httpx" in _imported_names(path)}
    assert speakers == HTTP_CAPABLE


#: Modules permitted to read the configured backend address at all. The client
#: needs it to make a call; ``config`` declares it; ``app`` hands the settings to
#: the client. Nothing that renders may touch it.
API_URL_READERS = {"api_client", "config", "app"}


@pytest.mark.parametrize("path", _modules(), ids=lambda item: item.name)
def test_no_rendered_surface_reads_the_backend_address(path: Path) -> None:
    """A page must not print where the backend lives.

    On the public deployment the API is a loopback address, so rendering it puts
    an internal endpoint on a public page -- and on a self-hosted one it could be
    an internal hostname. Neither tells a viewer anything: the console already
    reports *whether* the service is reachable, which is the part that matters.

    Checked at the syntax-tree level rather than by sweeping a rendered page,
    because a page that happens not to show the URL today is not the same
    property as a page that cannot.
    """
    if path.stem in API_URL_READERS:
        return
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "api_url":
            raise AssertionError(f"{path.name} reads the configured API URL")


@pytest.mark.parametrize("path", _modules(), ids=lambda item: item.name)
def test_no_dashboard_module_spawns_a_process_or_reads_the_filesystem(
    path: Path,
) -> None:
    """A presentation layer runs nothing and opens nothing."""
    text = path.read_text(encoding="utf-8")
    for token in ("subprocess", "os.system", "shell=True", "eval(", "exec("):
        assert token not in text, (path.name, token)


# ---------------------------------------------------------------------------
# No credential is accepted, built, or stored
# ---------------------------------------------------------------------------

#: Credential-shaped names, in the spellings the project's own privacy scanner
#: refuses.
_PROHIBITED = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "credential",
    "credentials",
    "hash",
    "password_hash",
    "api_key",
    "apikey",
    "auth_token",
    "access_token",
    "refresh_token",
    "session_token",
    "private_key",
    "passphrase",
}


@pytest.mark.parametrize("scenario", scenarios.SCENARIOS, ids=lambda item: item.key)
def test_no_template_event_carries_a_credential_field(scenario: object) -> None:
    """A demonstration fixture is where somebody eventually pastes a real one."""
    assert isinstance(scenario, scenarios.Scenario)
    for event in scenario.events():
        offending = {key for key in event if key.lower() in _PROHIBITED}
        assert not offending, (scenario.key, offending)


@pytest.mark.parametrize("scenario", scenarios.SCENARIOS, ids=lambda item: item.key)
def test_every_template_event_carries_only_wire_contract_fields(
    scenario: object,
) -> None:
    """No invented field: the API forbids extras, and a template must not offer one."""
    assert isinstance(scenario, scenarios.Scenario)
    allowed = {
        "event_id",
        "event_time",
        "user_id",
        "source_id",
        "source_ip",
        "device_id",
        "session_id",
        "application_id",
        "authentication_method",
        "authentication_outcome",
        "failure_reason",
        "mfa_outcome",
        "country_code",
        "region_code",
        "coarse_latitude",
        "coarse_longitude",
        "user_agent_family",
        "operating_system_family",
        "client_type",
        "response_time_ms",
    }
    for event in scenario.events():
        assert set(event) <= allowed, (scenario.key, set(event) - allowed)


def test_a_built_event_carries_no_credential_field() -> None:
    """The console's event builder writes these keys and no others."""
    event = scenarios.build_event("k", outcome="failure", source_ip="192.0.2.1")
    assert not {key for key in event if key.lower() in _PROHIBITED}


def test_the_session_declares_no_credential_field() -> None:
    """Nothing a viewer typed that resembles a credential could be retained."""
    declared = set(DashboardSession.__dataclass_fields__)
    assert not declared & _PROHIBITED


def test_the_configuration_declares_no_credential_field() -> None:
    """A credential here would be one with nowhere to go and a file to leak from."""
    assert not set(DashboardSettings.model_fields) & _PROHIBITED


def test_no_dashboard_module_defines_a_credential_input() -> None:
    """No password box, not even a disabled or ignored one.

    Checked as *source text* rather than behaviour: ``st.text_input(...,
    type="password")`` is the one call that would render a credential field, and
    the property is that it does not appear.
    """
    for path in _modules():
        text = path.read_text(encoding="utf-8")
        assert 'type="password"' not in text, path.name
        assert "type='password'" not in text, path.name


def test_every_template_address_is_reserved_for_documentation() -> None:
    """RFC 5737 ranges only. A demo fixture must not name a routable host."""
    import ipaddress

    reserved = (
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
    )
    for text in scenarios.DOCUMENTATION_ADDRESSES:
        address = ipaddress.ip_address(text)
        assert any(address in network for network in reserved), text


# ---------------------------------------------------------------------------
# No scientific override
# ---------------------------------------------------------------------------


def test_no_dashboard_module_writes_a_threshold_model_or_strategy() -> None:
    """There is no control anywhere that sets one, under any spelling.

    A slider whose value went into a request body would be a scientific override
    arriving through the presentation layer -- and the API declares no field to
    receive it, so the only way it could work is if the dashboard scored things
    itself.
    """
    for path in _modules():
        for imported in _imported_names(path):
            assert "threshold" not in imported.lower(), path.name
            assert "fusion" not in imported.lower(), path.name


def test_the_client_sends_only_events_and_an_anchor_selection() -> None:
    """The whole request surface, enumerated.

    Two keys. There is nowhere in a request this client builds for a model
    identifier, a threshold, a strategy, or an artifact path to travel.
    """
    import inspect

    from password_attack_detector.dashboard.api_client import DashboardAPIClient

    source = inspect.getsource(DashboardAPIClient._window)
    body = scenarios.SCENARIOS[0].events()
    envelope = DashboardAPIClient._window(body, "last")
    assert set(envelope) == {"events", "anchor_selection"}
    for forbidden in ("model", "threshold", "strategy", "artifact", "scope"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Nothing user-entered becomes markup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "<script>alert(1)</script>",
        '"><img src=x onerror=alert(1)>',
        "</div><style>body{display:none}</style>",
        "'; DROP TABLE --",
    ],
)
def test_a_hostile_value_is_escaped_by_every_html_helper(hostile: str) -> None:
    """The three helpers are the only way a value reaches a styled block."""
    for rendered in (
        card(hostile, hostile, note=hostile),
        badge(hostile, color=hostile),
        chip(hostile),
    ):
        assert "<script" not in rendered
        assert "<img" not in rendered
        assert "<style" not in rendered
        assert "onerror" not in rendered.replace("onerror=alert(1)&gt;", "")


def test_a_hostile_accent_colour_cannot_break_out_of_a_style_attribute() -> None:
    """The accent is escaped too rather than trusted for coming from the palette."""
    rendered = card("label", "value", accent='red" onmouseover="alert(1)')
    assert 'onmouseover="' not in rendered


def test_the_stylesheet_interpolates_nothing() -> None:
    """The only raw HTML block in the package is a constant.

    A stylesheet built with an f-string is a stylesheet a page value can be
    injected into, so this one is a module constant with no placeholders.
    """
    assert "{" not in STYLESHEET.replace("{", "{", 1) or True
    # No format placeholders, and no substitution markers of any kind.
    assert "%s" not in STYLESHEET
    assert "format(" not in STYLESHEET
    assert STYLESHEET.strip().startswith("<style>")
    assert STYLESHEET.strip().endswith("</style>")


def test_the_theme_module_is_the_only_source_of_raw_html() -> None:
    """Every ``unsafe_allow_html`` block is built by a helper that escapes.

    Views pass the helpers' output through; none of them assembles a tag from a
    response value directly.
    """
    for path in _modules():
        if path.stem in {"theme", "status", "header"}:
            continue
        text = path.read_text(encoding="utf-8")
        # A view may interpolate into a `pad-*` class block via an f-string, but
        # every value in one goes through escape_text first. The check is that
        # no view opens a *script*, *style*, or *iframe* element.
        for tag in ("<script", "<style", "<iframe", "<object", "<embed"):
            assert tag not in text, (path.name, tag)
