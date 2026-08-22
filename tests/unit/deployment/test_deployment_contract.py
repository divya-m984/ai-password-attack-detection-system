"""What the public deployment promises, asserted against the files themselves.

The sibling module ``test_container_contract.py`` covers the local
``docker compose up`` deployment. This one covers the perimeter that
``compose.deploy.yaml`` adds on top of it: a reverse proxy as the only public
listener, the API and the console un-published, bounded logs, and a routing
policy that is a reviewed file rather than something assembled from environment
variables.

Nothing here starts a container. The claims are all readable out of
``compose.deploy.yaml``, ``deploy/caddy/*``, ``.env.deploy.example``,
``.gitignore`` and ``scripts/deploy/bootstrap_server.sh`` -- and every one of
them is a claim that is expensive to discover was false *after* a machine is
facing the internet.

The claims that need a real merge or a real proxy -- that
``docker compose config`` actually drops the published ports, that Caddy accepts
both routing policies, that a request to ``127.0.0.1:8000`` is refused -- live in
``tests/integration/test_deployment_topology.py``.

One structural note. ``compose.deploy.yaml`` uses Compose's ``!reset`` tag to
remove a key rather than override it, because Compose *appends* sequences when
it merges files: an override cannot un-publish a port by restating a shorter
list. ``yaml.safe_load`` refuses an unknown tag, so :class:`_ComposeLoader`
below preserves it as a marker instead -- which is also what lets
:func:`test_the_deployment_override_resets_rather_than_restates_the_ports`
assert that the tag is the mechanism in use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from password_attack_detector import __version__
from password_attack_detector.api.config import (
    PROHIBITED_SETTING_NAMES as API_PROHIBITED,
)
from password_attack_detector.dashboard.config import (
    PROHIBITED_SETTING_NAMES as DASHBOARD_PROHIBITED,
)

ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = ROOT / "compose.yaml"
DEPLOY_COMPOSE = ROOT / "compose.deploy.yaml"
CADDYFILE = ROOT / "deploy" / "caddy" / "Caddyfile"
CADDYFILE_API_DOCS = ROOT / "deploy" / "caddy" / "Caddyfile.api-docs"
ENV_EXAMPLE = ROOT / ".env.deploy.example"
BOOTSTRAP = ROOT / "scripts" / "deploy" / "bootstrap_server.sh"
GITIGNORE = ROOT / ".gitignore"
DOCKERIGNORE = ROOT / ".dockerignore"

#: The ports that may reach the internet, and the only ones.
PUBLIC_PORTS = {80, 443}

#: The ports that must not, whatever else changes.
PRIVATE_PORTS = {8000, 8501}

#: The upstreams the proxy is allowed to name. Both are Compose service names on
#: the project network; neither is reachable from outside it.
ADMISSIBLE_UPSTREAMS = {"api:8000", "dashboard:8501"}

#: Headers the public boundary must set on every response.
REQUIRED_HEADERS = (
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Permissions-Policy",
    "X-Frame-Options",
    "Content-Security-Policy",
    "Strict-Transport-Security",
)


class _Tagged:
    """A YAML node carrying a Compose merge tag, kept rather than resolved."""

    def __init__(self, tag: str, value: object) -> None:
        self.tag = tag
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"!{self.tag} {self.value!r}"


class _ComposeLoader(yaml.SafeLoader):
    """A safe loader that preserves Compose's ``!reset`` / ``!override`` tags."""


def _construct_tagged(loader: yaml.Loader, suffix: str, node: yaml.Node) -> _Tagged:
    """Return an unknown ``!tag`` as a marker instead of refusing the document."""
    value: object
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node)
    else:  # pragma: no cover - YAML has exactly the three node kinds above
        value = None
    return _Tagged(suffix, value)


# PyYAML ships no annotation for this classmethod, so a strict check reads the
# call as untyped. Narrow ignore: the argument types are checked above.
_ComposeLoader.add_multi_constructor("!", _construct_tagged)  # type: ignore[no-untyped-call]


def _load(path: Path) -> dict[str, Any]:
    """Return a Compose document, merge tags preserved."""
    # `yaml.load` with an explicit loader, not `yaml.unsafe_load`:
    # `_ComposeLoader` derives from `SafeLoader` and adds one constructor that
    # turns an unknown `!tag` into an inert marker object. Nothing in a document
    # can name a Python type.
    document: dict[str, Any] = yaml.load(
        path.read_text(encoding="utf-8"), Loader=_ComposeLoader
    )
    return document


@pytest.fixture(scope="module")
def deploy() -> dict[str, Any]:
    """Return the parsed deployment override."""
    return _load(DEPLOY_COMPOSE)


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    """Return the parsed local compose file."""
    return _load(BASE_COMPOSE)


@pytest.fixture(scope="module")
def deploy_services(deploy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the deployment override's services."""
    services: dict[str, dict[str, Any]] = deploy["services"]
    return services


@pytest.fixture(scope="module")
def proxy(deploy_services: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the proxy service definition."""
    return deploy_services["proxy"]


@pytest.fixture(scope="module")
def caddyfiles() -> dict[str, str]:
    """Return both routing policies, keyed by file name."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in (CADDYFILE, CADDYFILE_API_DOCS)
    }


def _directive_lines(text: str) -> list[str]:
    """Return a Caddyfile's instruction lines, comments and blanks removed.

    A Caddyfile's comments explain what is *not* routed as much as what is, so a
    test that scanned the raw text would forbid its own explanation.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# The files exist, and the local workflow is untouched
# ---------------------------------------------------------------------------


def test_every_deployment_file_exists() -> None:
    """A documented deployment needs all of its parts present."""
    for path in (DEPLOY_COMPOSE, CADDYFILE, CADDYFILE_API_DOCS, ENV_EXAMPLE, BOOTSTRAP):
        assert path.is_file(), f"missing: {path.relative_to(ROOT)}"


def test_the_package_version_is_unchanged() -> None:
    """Preparing to deploy is not releasing. Nothing here bumps the version."""
    assert __version__ == "0.5.0"


def test_the_local_compose_file_still_publishes_to_loopback(
    base: dict[str, Any],
) -> None:
    """``docker compose up --build`` must keep working exactly as it did.

    The deployment override is additive. If it had been achieved by editing the
    base file, the local two-terminal workflow every other document describes
    would have quietly changed underneath it.
    """
    services = base["services"]
    assert services["api"]["ports"] == ["127.0.0.1:8000:8000"]
    assert services["dashboard"]["ports"] == ["127.0.0.1:8501:8501"]
    assert "proxy" not in services


def test_the_deployment_override_declares_no_service_of_its_own_invention(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """It retunes three services and adds exactly one."""
    assert set(deploy_services) == {"prepare", "api", "dashboard", "proxy"}


def test_the_override_carries_no_build_instruction_for_the_application(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """The image the deployment runs is the image the local workflow builds.

    An override with its own ``build`` block could produce a different artifact
    from the same source, which would make "verified locally" mean nothing.
    """
    for name in ("prepare", "api", "dashboard"):
        assert "build" not in deploy_services[name], name
        assert "image" not in deploy_services[name], name


# ---------------------------------------------------------------------------
# Public ports
# ---------------------------------------------------------------------------


def test_the_deployment_override_resets_rather_than_restates_the_ports(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """``!reset`` removes the key; a shorter list would be appended to."""
    for name in ("api", "dashboard"):
        declared = deploy_services[name]["ports"]
        assert isinstance(declared, _Tagged), f"{name} must use a merge tag"
        assert declared.tag == "reset", f"{name} uses !{declared.tag}, not !reset"


def test_only_the_proxy_publishes_anything(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """Exactly one service in the deployment has a port on the host."""
    publishing = {
        name
        for name, service in deploy_services.items()
        if isinstance(service.get("ports"), list)
    }
    assert publishing == {"proxy"}


def _container_port(specification: str) -> int:
    """Return the container-side port of a Compose publish specification."""
    return int(specification.rsplit(":", 1)[1].split("/")[0])


def test_the_proxy_publishes_only_http_and_https(proxy: dict[str, Any]) -> None:
    """80 and 443, and nothing else, on the container side."""
    published = {_container_port(entry) for entry in proxy["ports"]}
    assert published == PUBLIC_PORTS


def test_the_default_host_ports_are_eighty_and_four_four_three(
    proxy: dict[str, Any],
) -> None:
    """The documented default must not depend on an environment file existing."""
    defaults = [re.sub(r"\$\{[A-Z_]+:-([^}]*)\}", r"\1", e) for e in proxy["ports"]]
    assert defaults == ["80:80", "443:443"]


def test_no_deployment_service_publishes_the_application_ports(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """8000 and 8501 reach the Compose network and stop there.

    Asserted over every publish specification in the override, on both sides of
    the mapping: a host-side 8000 would be as public as a container-side one.
    """
    for name, service in deploy_services.items():
        for entry in (
            service.get("ports", []) if isinstance(service.get("ports"), list) else []
        ):
            numbers = {int(part) for part in re.findall(r"\d+", str(entry))}
            assert not (numbers & PRIVATE_PORTS), f"{name} publishes {entry}"


def test_the_publish_specifications_are_the_only_parameterised_ports(
    proxy: dict[str, Any],
) -> None:
    """A variable may choose where to publish; it may not choose what to expose.

    Both entries interpolate the *host* side only. The container side is a
    literal, so no environment file can point the proxy's own listener somewhere
    else.
    """
    for entry in proxy["ports"]:
        host, container = str(entry).rsplit(":", 1)
        assert "${" in host, entry
        assert "${" not in container, entry


# ---------------------------------------------------------------------------
# The proxy container
# ---------------------------------------------------------------------------


def test_the_proxy_image_is_pinned_by_digest(proxy: dict[str, Any]) -> None:
    """The same reasoning as the application base: a tag can be moved."""
    image = proxy["image"]
    assert image.startswith("caddy:2.")
    assert ":latest" not in image
    digest = image.split("@sha256:")[1]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_the_proxy_drops_every_capability_but_the_one_it_must_have(
    proxy: dict[str, Any],
) -> None:
    """Binding a port below 1024 needs NET_BIND_SERVICE. Nothing else is added."""
    assert proxy["cap_drop"] == ["ALL"]
    assert proxy["cap_add"] == ["NET_BIND_SERVICE"]


def test_the_proxy_is_hardened_like_everything_else(proxy: dict[str, Any]) -> None:
    """Read-only root, no new privileges, not privileged, not on the host."""
    assert proxy["read_only"] is True
    assert "no-new-privileges:true" in proxy["security_opt"]
    assert "privileged" not in proxy
    assert "network_mode" not in proxy
    assert proxy["networks"] == ["demo"]


def test_the_proxy_waits_for_the_thing_it_routes_to(proxy: dict[str, Any]) -> None:
    """A proxy that started first would answer 502 while Streamlit booted."""
    assert proxy["depends_on"]["dashboard"]["condition"] == "service_healthy"


def test_the_proxy_healthcheck_installs_nothing(proxy: dict[str, Any]) -> None:
    """busybox wget ships in the Alpine image; curl would be a package to patch."""
    test = proxy["healthcheck"]["test"]
    assert test[0] == "CMD"
    assert test[1] == "wget"
    assert "curl" not in " ".join(test)


# ---------------------------------------------------------------------------
# The proxy holds no scientific state, and neither does the console
# ---------------------------------------------------------------------------


def test_the_proxy_mounts_no_serving_state(proxy: dict[str, Any]) -> None:
    """Certificates and its own config. Not the frozen bundle, in any mode."""
    sources = [str(entry).split(":")[0] for entry in proxy["volumes"]]
    assert "serving-state" not in sources
    assert set(sources) <= {"${PAD_PROXY_CADDYFILE", "caddy-data", "caddy-config"}


def test_the_proxy_mounts_its_routing_policy_read_only(proxy: dict[str, Any]) -> None:
    """A container that could rewrite its own routing policy has no policy."""
    caddyfile = next(e for e in proxy["volumes"] if "Caddyfile" in str(e))
    assert str(caddyfile).endswith(":/etc/caddy/Caddyfile:ro")


def test_the_override_does_not_give_the_console_a_volume(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """The console holds no artifact in the local deployment and none here."""
    assert "volumes" not in deploy_services["dashboard"]


def test_the_override_does_not_touch_the_serving_state_mount(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """Read-only into the API, absent everywhere else, and not restated here.

    Compose merges volume lists by appending, so an override that mentioned the
    volume at all could only add a second mount of it -- which is how a
    deployment would accidentally acquire a writable one.
    """
    for name in ("prepare", "api"):
        assert "volumes" not in deploy_services[name], name


def test_the_docker_socket_is_mounted_nowhere(deploy: dict[str, Any]) -> None:
    """The one mount that would make every other boundary decorative."""
    assert "docker.sock" not in DEPLOY_COMPOSE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Restart, logs, resources
# ---------------------------------------------------------------------------


def test_the_preparation_job_is_never_given_a_restart_policy(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """A one-shot job that restarts is not a one-shot job.

    It would also mean the machine retrained on every reboot, which is the one
    thing the offline-preparation contract exists to prevent.
    """
    assert deploy_services["prepare"].get("restart", "no") == "no"


@pytest.mark.parametrize("name", ["api", "dashboard", "proxy"])
def test_the_long_lived_services_survive_a_reboot(
    deploy_services: dict[str, dict[str, Any]], name: str
) -> None:
    """``unless-stopped`` restarts the same images; it fetches no new code."""
    assert deploy_services[name]["restart"] == "unless-stopped"


@pytest.mark.parametrize("name", ["prepare", "api", "dashboard", "proxy"])
def test_every_service_has_bounded_logs(
    deploy_services: dict[str, dict[str, Any]], name: str
) -> None:
    """Docker's default keeps every byte forever; a demo VM has 25 GB."""
    logging = deploy_services[name]["logging"]
    assert logging["driver"] in {"json-file", "local"}
    options = logging["options"]
    assert options["max-size"].endswith("m")
    assert int(options["max-file"]) >= 1


@pytest.mark.parametrize(
    ("name", "ceiling_mib"),
    [("prepare", 1024), ("api", 768), ("dashboard", 384), ("proxy", 128)],
)
def test_the_memory_ceilings_fit_a_two_gibibyte_machine(
    deploy_services: dict[str, dict[str, Any]], name: str, ceiling_mib: int
) -> None:
    """Carried from what M4 measured, not from what sounds generous."""
    limit = deploy_services[name]["deploy"]["resources"]["limits"]["memory"]
    number = int(re.sub(r"[^0-9]", "", str(limit)))
    unit = str(limit)[-1].lower()
    assert unit in {"m", "g"}
    assert (number * 1024 if unit == "g" else number) == ceiling_mib


def test_the_long_lived_ceilings_leave_room_on_a_two_gibibyte_machine(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """api + dashboard + proxy must not add up to the whole machine.

    ``prepare`` is excluded because it is the only thing running while it runs:
    the API does not start until it has exited.
    """
    total = 0
    for name in ("api", "dashboard", "proxy"):
        limit = str(deploy_services[name]["deploy"]["resources"]["limits"]["memory"])
        number = int(re.sub(r"[^0-9]", "", limit))
        total += number * 1024 if limit[-1].lower() == "g" else number
    assert total <= 1408, f"{total} MiB of ceilings on a 2048 MiB machine"


# ---------------------------------------------------------------------------
# Nothing scientific crosses the perimeter
# ---------------------------------------------------------------------------


def _environment_names(services: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """Return every (service, environment variable name) pair in a document."""
    pairs: list[tuple[str, str]] = []
    for service, definition in services.items():
        environment = definition.get("environment", {})
        names = (
            environment
            if isinstance(environment, dict)
            else [str(entry).split("=", 1)[0] for entry in environment]
        )
        pairs.extend((service, str(name)) for name in names)
    return pairs


def test_the_override_declares_no_scientific_override(
    deploy_services: dict[str, dict[str, Any]],
) -> None:
    """The same guard the local file gets, applied to the perimeter."""
    for service, name in _environment_names(deploy_services):
        if name.startswith("PAD_API_"):
            assert name.removeprefix("PAD_API_").lower() not in API_PROHIBITED, service
        if name.startswith("PAD_DASHBOARD_"):
            suffix = name.removeprefix("PAD_DASHBOARD_").lower()
            assert suffix not in DASHBOARD_PROHIBITED, service


@pytest.mark.parametrize(
    "forbidden",
    ["MODEL", "THRESHOLD", "FUSION", "CHAMPION", "CALIBRAT", "SCORE", "ARTIFACT"],
)
def test_no_deployment_variable_even_resembles_a_scientific_control(
    deploy_services: dict[str, dict[str, Any]], forbidden: str
) -> None:
    """Stricter than the settings guard, and deliberately so.

    ``APISettings`` refuses to *declare* such a field, so a variable named after
    one would be ignored rather than obeyed. This refuses the name anyway: a
    deployment file that appears to set a threshold documents a lie even when the
    process quietly discards it.
    """
    for service, name in _environment_names(deploy_services):
        assert forbidden not in name.upper(), f"{service} declares {name}"


@pytest.mark.parametrize(
    "forbidden", ["PASSWORD", "SECRET", "TOKEN", "API_KEY", "CREDENTIAL", "PRIVATE_KEY"]
)
def test_no_deployment_variable_is_shaped_like_a_credential(
    deploy_services: dict[str, dict[str, Any]], forbidden: str
) -> None:
    """This system accepts no credential material anywhere in its stack."""
    for service, name in _environment_names(deploy_services):
        assert forbidden not in name.upper(), f"{service} declares {name}"


def test_the_deployment_variables_are_the_documented_four(
    deploy: dict[str, Any],
) -> None:
    """Every substitution point in the override, enumerated.

    Equality rather than containment: a new variable is a new thing a deployment
    can change, and it should have to be named here before it can be one.
    """
    referenced = set(
        re.findall(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)", DEPLOY_COMPOSE.read_text(encoding="utf-8")
        )
    )
    assert referenced == {
        "PAD_SITE_ADDRESS",
        "PAD_HSTS",
        "PAD_PROXY_CADDYFILE",
        "PAD_PROXY_HTTP_PUBLISH",
        "PAD_PROXY_HTTPS_PUBLISH",
    }


# ---------------------------------------------------------------------------
# The routing policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Caddyfile", "Caddyfile.api-docs"])
def test_the_policy_names_only_internal_upstreams(
    caddyfiles: dict[str, str], name: str
) -> None:
    """No environment variable can compose an upstream URL.

    This is the difference between "a reverse proxy in front of two containers"
    and "an open forwarder anybody can point anywhere". Every ``reverse_proxy``
    target is a literal Compose service name.
    """
    targets = [
        line.split(None, 1)[1].split("{")[0].strip()
        for line in _directive_lines(caddyfiles[name])
        if line.startswith("reverse_proxy ")
    ]
    assert targets, "a proxy with no upstream is not a proxy"
    assert set(targets) <= ADMISSIBLE_UPSTREAMS, targets
    for target in targets:
        assert "$" not in target, target


@pytest.mark.parametrize("name", ["Caddyfile", "Caddyfile.api-docs"])
def test_the_only_environment_placeholders_are_the_two_documented_ones(
    caddyfiles: dict[str, str], name: str
) -> None:
    """A site address and an HSTS max-age. Neither can name a host to talk to."""
    placeholders = set(re.findall(r"\{\$([A-Za-z_][A-Za-z0-9_]*)", caddyfiles[name]))
    assert placeholders == {"PAD_SITE_ADDRESS", "PAD_HSTS"}


@pytest.mark.parametrize("name", ["Caddyfile", "Caddyfile.api-docs"])
@pytest.mark.parametrize("header", REQUIRED_HEADERS)
def test_the_policy_sets_the_security_headers(
    caddyfiles: dict[str, str], name: str, header: str
) -> None:
    """Set in a shared snippet, so the two policies cannot drift apart."""
    assert any(
        line.startswith(header) for line in _directive_lines(caddyfiles[name])
    ), f"{name} does not set {header}"


@pytest.mark.parametrize("name", ["Caddyfile", "Caddyfile.api-docs"])
def test_the_content_security_policy_is_frame_ancestors_only(
    caddyfiles: dict[str, str], name: str
) -> None:
    """The audited compromise, pinned so it cannot be "improved" into a break.

    A ``script-src`` tight enough to be worth having stops Streamlit's bundled
    client from running; one loose enough to keep it working would have to permit
    ``unsafe-inline`` and ``unsafe-eval``, which claims a protection it does not
    provide. ``frame-ancestors`` is the directive that is both meaningful here
    and compatible. See docs/deployment.md for the audit.
    """
    line = next(
        entry
        for entry in _directive_lines(caddyfiles[name])
        if entry.startswith("Content-Security-Policy")
    )
    assert "frame-ancestors 'none'" in line
    assert "unsafe-inline" not in line
    assert "unsafe-eval" not in line


@pytest.mark.parametrize("name", ["Caddyfile", "Caddyfile.api-docs"])
def test_the_policy_bounds_a_request_body(
    caddyfiles: dict[str, str], name: str
) -> None:
    """Matched to the service's own ceiling rather than chosen independently."""
    assert any(
        line.startswith("max_size ") for line in _directive_lines(caddyfiles[name])
    )


@pytest.mark.parametrize("name", ["Caddyfile", "Caddyfile.api-docs"])
def test_the_admin_api_is_off(caddyfiles: dict[str, str], name: str) -> None:
    """An unauthenticated endpoint that can rewrite the routing policy."""
    assert "admin off" in _directive_lines(caddyfiles[name])


def test_the_default_policy_routes_nothing_to_the_api(
    caddyfiles: dict[str, str],
) -> None:
    """One route, to the console. The API is unreachable from the internet."""
    targets = [
        line
        for line in _directive_lines(caddyfiles["Caddyfile"])
        if line.startswith("reverse_proxy ")
    ]
    assert targets == ["reverse_proxy dashboard:8501"]


def test_neither_policy_publishes_a_scoring_or_replay_control_route(
    caddyfiles: dict[str, str],
) -> None:
    """Bounded is not the same as free.

    Every scoring request builds a point-in-time feature window and runs a rule
    pass and a model pass. Until this deployment has rate limiting -- deferred,
    and documented as deferred -- an unauthenticated public POST that costs CPU
    is a denial-of-service surface on a 2 GiB machine. The console reaches these
    over the internal network, so nothing about the demonstration is lost.
    """
    refused = ("/detect", "/explain", "/demo/runs")
    for name, text in caddyfiles.items():
        for line in _directive_lines(text):
            if not line.startswith(("path ", "@")):
                continue
            for endpoint in refused:
                assert endpoint not in line, f"{name} routes {endpoint}"


def test_the_documented_policy_files_are_the_only_two(
    proxy: dict[str, Any],
) -> None:
    """The variable selects between reviewed files; it does not compose a path.

    A default that is a tracked file means a deployment with no environment file
    still gets an audited policy rather than a missing mount.
    """
    entry = next(str(e) for e in proxy["volumes"] if "Caddyfile" in str(e))
    default = entry.split(":-", 1)[1].split("}", 1)[0]
    assert default == "./deploy/caddy/Caddyfile"
    assert (ROOT / default.removeprefix("./")).is_file()


# ---------------------------------------------------------------------------
# Secrets: none, anywhere, in anything tracked
# ---------------------------------------------------------------------------

#: Patterns that would mean a tracked file had grown a credential.
#:
#: Written as anchored regular expressions rather than substrings because the
#: substring forms produce false positives against ordinary prose -- ``sk-``
#: alone matches "disk-exhaustion", and a test that fires on its own explanation
#: gets deleted rather than fixed.
CREDENTIAL_PATTERNS = (
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bghp_[A-Za-z0-9]{20,}",
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bsk-[A-Za-z0-9]{20,}",
    r"\bxoxb-[0-9A-Za-z-]{20,}",
    r"\bdop_v1_[a-f0-9]{40,}",  # a DigitalOcean personal access token
    r"Authorization:\s*Bearer\s+\S+",
)


@pytest.mark.parametrize(
    "relative",
    [
        ".env.deploy.example",
        "compose.deploy.yaml",
        "deploy/caddy/Caddyfile",
        "deploy/caddy/Caddyfile.api-docs",
        "scripts/deploy/bootstrap_server.sh",
    ],
)
def test_no_tracked_deployment_file_carries_credential_material(
    relative: str,
) -> None:
    """Read as text, marker by marker.

    This project's services accept no credential of any kind, so anything
    matching here is either a mistake or a new capability nobody reviewed.
    """
    text = (ROOT / relative).read_text(encoding="utf-8")
    for pattern in CREDENTIAL_PATTERNS:
        assert not re.search(pattern, text), f"{relative} matches {pattern}"


def test_the_environment_template_assigns_nothing_secret_looking() -> None:
    """Every assignment in the template is an operational value.

    Checked by name and by shape: a variable whose name suggests a secret, and a
    value long and random enough to be one, are both refused.
    """
    suspicious = ("PASSWORD", "SECRET", "TOKEN", "KEY", "CREDENTIAL")
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        assert not any(word in name.upper() for word in suspicious), line
        assert not re.fullmatch(r"[A-Za-z0-9+/_-]{24,}", value), line


def test_the_real_deployment_environment_file_is_refused_by_git() -> None:
    """The template is tracked; the machine's own file must never be."""
    ignored = GITIGNORE.read_text(encoding="utf-8").splitlines()
    assert ".env.deploy" in [line.strip() for line in ignored]


def test_no_deployment_environment_file_exists_in_the_working_tree() -> None:
    """Ignored is not the same as absent, and this is a repository, not a server."""
    assert not (ROOT / ".env.deploy").exists()


def test_the_check_yaml_hook_is_narrowed_and_not_disarmed() -> None:
    """One file is excluded from ``check-yaml``, and it is not ``--unsafe``.

    ``compose.deploy.yaml`` carries Compose's ``!reset`` tag, which that hook
    refuses because it has no constructor for it. The fix is an exclusion naming
    exactly that file. The fix is *not* ``--unsafe``, which is a global flag: it
    would downgrade every YAML in the repository -- including the configurations
    declaring what gets trained -- from "parses and constructs" to "is
    syntactically YAML-shaped".

    Worth pinning because the failure is invisible until somebody commits: the
    hook skips untracked files, so a full ``pre-commit run --all-files`` passes
    right up to the moment the file enters the index.
    """
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text("utf-8"))
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    check_yaml = next(hook for hook in hooks if hook["id"] == "check-yaml")
    assert check_yaml.get("exclude") == r"^compose\.deploy\.yaml$"
    assert "--unsafe" not in check_yaml.get("args", [])
    # The base compose file has no merge tag and stays covered.
    assert not re.search(r"^\s*\S+:\s*!", BASE_COMPOSE.read_text("utf-8"), re.MULTILINE)


def test_the_deployment_directory_never_enters_the_application_build_context() -> None:
    """``.dockerignore`` is an allowlist, and ``deploy/`` is not on it.

    The routing policy is mounted into the proxy at run time from the checkout.
    Baking it into the application image would put a public routing decision
    inside an artifact whose provenance is meant to be scientific only.
    """
    admitted = {
        line.strip().removeprefix("!")
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("!")
    }
    assert "deploy" not in admitted
    assert not any(entry.startswith("deploy/") for entry in admitted)
    assert "scripts/prepare_demo_bundle.py" in admitted
    assert not any(entry.startswith("scripts/deploy") for entry in admitted)


# ---------------------------------------------------------------------------
# The server bootstrap script
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bootstrap() -> str:
    """Return the bootstrap script's text."""
    return BOOTSTRAP.read_text(encoding="utf-8")


def _shell_instructions(text: str) -> list[str]:
    """Return a shell script's executable lines: no comments, no heredoc bodies.

    The heredoc part matters here. This script *prints* the deployment steps for
    a human to run, and one of them is ``git clone``. A scan that read the
    printed text as instructions would conclude the script clones a repository,
    which is the opposite of true -- it deliberately does not, so that what lands
    on a server is a checkout somebody made at a reviewed commit.
    """
    instructions: list[str] = []
    terminator: str | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if terminator is not None:
            if stripped == terminator:
                terminator = None
            continue
        if stripped.startswith("#"):
            continue
        opener = re.search(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", stripped)
        if opener is not None:
            terminator = opener.group(1)
            continue
        if stripped:
            instructions.append(stripped)
    return instructions


def test_the_bootstrap_script_is_executable() -> None:
    """It is documented as `bash scripts/...`, but a mode bit costs nothing."""
    assert BOOTSTRAP.stat().st_mode & 0o111


def test_the_bootstrap_script_refuses_to_run_by_accident(bootstrap: str) -> None:
    """Three guards, and the machine-shape ones come first.

    A developer who runs this on a laptop should be told it is the wrong machine,
    not that they need sudo -- so the Ubuntu check and the graphical-session check
    precede the root check.
    """
    assert "refusing to run without --confirm" in bootstrap
    assert 'ID:-}" = "ubuntu"' in bootstrap
    assert bootstrap.index("xsessions") < bootstrap.index('id -u)" -eq 0')


def test_the_bootstrap_script_never_edits_the_ssh_configuration(
    bootstrap: str,
) -> None:
    """It reads ``sshd -T`` and warns. It writes nothing under /etc/ssh.

    A bootstrap that rewrites sshd on a machine you are connected to over SSH is
    a bootstrap that can lock you out of it.
    """
    assert "sshd -T" in bootstrap
    for line in _shell_instructions(bootstrap):
        assert "sshd_config" not in line or line.startswith("warn "), line
    assert "PasswordAuthentication yes" not in bootstrap


def test_the_bootstrap_script_creates_a_passwordless_unprivileged_account(
    bootstrap: str,
) -> None:
    """Key authentication only; no password is set and none may be."""
    assert "--disabled-password" in bootstrap
    assert 'DEPLOY_USER" != "root"' in bootstrap


def test_the_bootstrap_script_is_honest_about_the_docker_group(
    bootstrap: str,
) -> None:
    """Membership in ``docker`` is root-equivalent, and the script says so."""
    assert "usermod -aG docker" in bootstrap
    assert "root-equivalent" in bootstrap


def test_the_bootstrap_script_opens_no_arbitrary_port(bootstrap: str) -> None:
    """Firewall changes are opt-in and are exactly SSH, HTTP and HTTPS."""
    allowed = re.findall(r"^\s*ufw allow (\S+)$", bootstrap, re.MULTILINE)
    assert allowed == ["22/tcp", "80/tcp", "443/tcp"]
    assert "--with-ufw" in bootstrap


def test_the_bootstrap_script_performs_no_git_operation(bootstrap: str) -> None:
    """It installs the package. It does not clone, pull, configure, or commit.

    What lands on a server should be a checkout somebody made at a reviewed
    commit, recorded with ``git rev-parse HEAD``. A bootstrap that cloned for you
    would decide which revision is deployed, and would decide it silently.
    """
    for line in _shell_instructions(bootstrap):
        assert not re.match(r"^git (clone|pull|fetch|config|checkout|remote)", line)


def test_the_bootstrap_script_talks_to_no_cloud_provider(bootstrap: str) -> None:
    """No provider API, no token, no resource creation. It prepares a host."""
    lowered = bootstrap.lower()
    assert "api.digitalocean.com" not in lowered
    assert "doctl" not in lowered
    # Only *invocations* are checked -- naming curl in an apt-get argument list
    # installs it, and the single place it is then run is Docker's own host.
    for line in _shell_instructions(bootstrap):
        if re.search(r"(^|[|(]\s*)curl\s", line):
            assert "download.docker.com" in line, line


def test_the_deployment_starts_nothing_automatically(bootstrap: str) -> None:
    """The deployment user runs Compose deliberately; the script prints how.

    The printed instructions are in a heredoc, so they are text rather than
    commands -- which is exactly the distinction this asserts.
    """
    assert "docker compose --env-file .env.deploy" in bootstrap
    for line in _shell_instructions(bootstrap):
        assert not re.match(r"^docker compose .*\bup\b", line), line
