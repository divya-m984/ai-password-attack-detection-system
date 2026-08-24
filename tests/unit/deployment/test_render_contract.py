"""What the Render deployment promises, asserted against the files themselves.

None of this starts a container, builds an image, or contacts Render. Every
claim here is one that can be read out of ``render.yaml``,
``Dockerfile.render``, ``deploy/render/Caddyfile`` and
``scripts/render_entrypoint.py`` -- and each is a claim that would be expensive
to discover was false only after a public URL existed.

The Docker-requiring half lives in ``tests/integration/test_render_container.py``,
which builds the image, runs it under a 512 MiB ceiling, and checks that the
bundle it carries is fingerprint-for-fingerprint the one the Compose
preparation job produces.

Four groups of claim, and each exists for a different failure:

**The blueprint stays within the free tier.** A blueprint that quietly asks for
a disk, a database, a second service or a paid plan does not fail -- it deploys,
and bills, or is rejected at sync time with a message nobody reads until then.

**The image bakes a frozen bundle and fits nothing.** The Render deployment is
the one place in this project where a trained model lives inside an image. That
is a deliberate deviation from ``Dockerfile``'s rule, made because Render Free
has no persistent volume for a preparation job to write into, and it is only
defensible while the preparation still happens at build time, in a stage that is
not the runtime image, from the same tracked script and configurations.

**Only the proxy is public.** One container runs three processes. If the API
bound anything but loopback, or if the proxy routed anything but the console and
one liveness path, the detection and replay-control endpoints would be on the
public internet -- which is exactly what the VPS deployment's routing policy
refuses, for the same reasons.

**The VPS deployment is untouched.** M5B adapts; it does not replace. The
Compose overlay, its Caddyfile and its documentation must all still be there.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
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

RENDER_BLUEPRINT = ROOT / "render.yaml"
RENDER_DOCKERFILE = ROOT / "Dockerfile.render"
RENDER_CADDYFILE = ROOT / "deploy" / "render" / "Caddyfile"
RENDER_ENTRYPOINT = ROOT / "scripts" / "render_entrypoint.py"
BUNDLE_VERIFIER = ROOT / "scripts" / "verify_serving_bundle.py"

VPS_DOCKERFILE = ROOT / "Dockerfile"
VPS_COMPOSE = ROOT / "compose.yaml"
VPS_DEPLOY_OVERLAY = ROOT / "compose.deploy.yaml"
VPS_CADDYFILE = ROOT / "deploy" / "caddy" / "Caddyfile"
VPS_DOCS = ROOT / "docs" / "deployment.md"
RENDER_DOCS = ROOT / "docs" / "render-deployment.md"

#: The account the runtime image serves as.
RUNTIME_UID = "10001"

#: The loopback endpoints the three processes agree on. Not settings: they are
#: addresses inside one container's network namespace, and nothing outside the
#: container can reach either.
API_UPSTREAM = "127.0.0.1:8000"
DASHBOARD_UPSTREAM = "127.0.0.1:8501"

#: The one API path the proxy publishes, and what it is rewritten to.
PUBLIC_HEALTH_PATH = "/healthz"
API_LIVENESS_PATH = "/health"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def blueprint() -> dict[str, Any]:
    """Return the parsed Render blueprint."""
    document: dict[str, Any] = yaml.safe_load(
        RENDER_BLUEPRINT.read_text(encoding="utf-8")
    )
    return document


@pytest.fixture(scope="module")
def web_service(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Return the blueprint's single service."""
    services: list[dict[str, Any]] = blueprint["services"]
    assert len(services) == 1
    return services[0]


@pytest.fixture(scope="module")
def render_dockerfile() -> str:
    """Return the Render Dockerfile's text."""
    return RENDER_DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def render_caddyfile() -> str:
    """Return the Render Caddyfile's text."""
    return RENDER_CADDYFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def entrypoint() -> Any:
    """Import the supervisor by path.

    It lives under ``scripts/`` rather than in the package for the same reason
    the preparation script does: it is an operator-facing process, not an
    importable API, and nothing in the serving path should be able to reach it.
    """
    spec = importlib.util.spec_from_file_location(
        "_render_entrypoint", RENDER_ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _instructions(text: str) -> list[str]:
    """Return a Dockerfile's instructions, comments and continuations resolved."""
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1].strip() + " "
            continue
        joined.append((buffer + line).strip())
        buffer = ""
    if buffer:
        joined.append(buffer.strip())
    return joined


def _env_pairs(text: str) -> dict[str, str]:
    """Return every ``ENV KEY=VALUE`` pair the Render Dockerfile declares."""
    found: dict[str, str] = {}
    for instruction in _instructions(text):
        if not instruction.startswith("ENV "):
            continue
        for token in instruction[4:].split():
            if "=" in token:
                key, _, value = token.partition("=")
                found[key] = value
    return found


def _caddy_directives(text: str) -> list[str]:
    """Return the Caddyfile's non-comment lines.

    Both routing files in this repository explain their own policy in comments,
    and those comments name the very paths the policy refuses to publish. A test
    that grepped the raw text would read the explanation as the configuration.
    """
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _caddy_body(text: str) -> str:
    """Return the Caddyfile's directives as one string, comments removed."""
    return "\n".join(_caddy_directives(text))


# ---------------------------------------------------------------------------
# The files exist, and the VPS deployment still does too
# ---------------------------------------------------------------------------


def test_every_render_file_exists() -> None:
    """The blueprint, the image definition, the proxy policy and the supervisor."""
    for path in (
        RENDER_BLUEPRINT,
        RENDER_DOCKERFILE,
        RENDER_CADDYFILE,
        RENDER_ENTRYPOINT,
        BUNDLE_VERIFIER,
        RENDER_DOCS,
    ):
        assert path.is_file(), path


def test_the_vps_deployment_is_preserved_intact() -> None:
    """M5B adapts the deployment target; it does not replace the existing one.

    A Render adapter that quietly deleted the Compose overlay would leave the
    project with one deployment path and a documentation set describing two.
    """
    for path in (
        VPS_DOCKERFILE,
        VPS_COMPOSE,
        VPS_DEPLOY_OVERLAY,
        VPS_CADDYFILE,
        VPS_DOCS,
    ):
        assert path.is_file(), path


def test_the_vps_overlay_still_removes_the_published_ports() -> None:
    """The single load-bearing mechanism of the VPS perimeter is still in place.

    ``!reset`` is what un-publishes 8000 and 8501 when the deployment overlay is
    merged; Compose appends sequences, so a shorter ``ports`` list would not do
    it. Asserted here as well as in the M5A suite because this milestone edits
    the deployment surface and that is exactly the kind of thing a refactor
    quietly loses.
    """
    text = VPS_DEPLOY_OVERLAY.read_text(encoding="utf-8")
    assert text.count("ports: !reset null") == 2


def test_the_package_version_is_unchanged() -> None:
    """M5B is a deployment adapter, and adapters do not rename the system."""
    assert __version__ == "0.5.0"
    assert 'version = "0.5.0"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The blueprint: exactly what Render Free supports, and nothing more
# ---------------------------------------------------------------------------


def test_the_blueprint_parses_as_yaml(blueprint: dict[str, Any]) -> None:
    """A blueprint that does not parse is rejected at sync time, not at review."""
    assert isinstance(blueprint, dict)
    assert isinstance(blueprint.get("services"), list)


def test_the_blueprint_declares_exactly_one_service(
    blueprint: dict[str, Any],
) -> None:
    """One service, which is the whole architectural decision of this milestone.

    Two free services would double instance-hour consumption against a shared
    allowance, would require the detection API to be publicly reachable so the
    console could call it, and would give a viewer two independent cold starts.
    """
    assert len(blueprint["services"]) == 1


def test_the_service_is_a_free_web_service(web_service: dict[str, Any]) -> None:
    """Type, plan and runtime, all three stated rather than defaulted."""
    assert web_service["type"] == "web"
    assert web_service["plan"] == "free"
    assert web_service["runtime"] == "docker"


def test_the_blueprint_builds_the_render_dockerfile(
    web_service: dict[str, Any],
) -> None:
    """And the file it names is the one in this repository."""
    named = str(web_service["dockerfilePath"]).removeprefix("./")
    assert (ROOT / named) == RENDER_DOCKERFILE
    assert web_service["dockerContext"] == "."


def test_the_blueprint_names_no_start_command(web_service: dict[str, Any]) -> None:
    """The image's CMD is the single definition of the process model.

    A ``dockerCommand`` here would be a second one, and the two would drift.
    """
    assert "dockerCommand" not in web_service
    assert "startCommand" not in web_service


@pytest.mark.parametrize(
    "section",
    ["databases", "envVarGroups", "previews", "version"],
)
def test_the_blueprint_declares_no_paid_or_unsupported_section(
    blueprint: dict[str, Any], section: str
) -> None:
    """Nothing outside a single free web service is asked for."""
    assert section not in blueprint


@pytest.mark.parametrize(
    "key", ["disk", "numInstances", "scaling", "maxShutdownDelaySeconds"]
)
def test_the_service_asks_for_nothing_the_free_tier_does_not_give(
    web_service: dict[str, Any], key: str
) -> None:
    """A disk in particular: the runtime is proven to need no persistence.

    The scientific state is baked into the image and read-only, and replay
    history is process memory that the console says out loud will not survive a
    restart. Asking for a disk would be asking for a paid plan to store nothing.
    """
    assert key not in web_service


def test_no_service_is_a_worker_private_service_or_cron_job(
    blueprint: dict[str, Any],
) -> None:
    """Only ``type: web`` exists, so there is nothing to communicate privately with."""
    for service in blueprint["services"]:
        assert service["type"] == "web"


def test_the_blueprint_declares_no_environment_variables(
    web_service: dict[str, Any],
) -> None:
    """Every setting is baked into the image, and that is a security property.

    A blueprint variable can be edited afterwards in a web dashboard by anyone
    with access to it; an image layer cannot. Since none of the deployment's
    settings is declared here, there is no place in this file where a model, a
    threshold, a calibration or a fusion strategy could be introduced.
    """
    assert "envVars" not in web_service


def test_the_blueprint_configures_a_health_check(web_service: dict[str, Any]) -> None:
    """Render needs one, and it is the single public API path."""
    assert web_service["healthCheckPath"] == PUBLIC_HEALTH_PATH


def test_deploys_are_not_automatic(web_service: dict[str, Any]) -> None:
    """A push to a branch does not publish a new public demonstration on its own.

    The same reasoning as the VPS deployment's refusal to pull from git at
    container startup: a deployment should be an act, not a side effect.
    """
    assert web_service["autoDeploy"] is False


# ---------------------------------------------------------------------------
# The image: same interpreter, same lockfile, bundle baked at build time
# ---------------------------------------------------------------------------


def test_the_render_image_pins_the_same_base_as_the_vps_image(
    render_dockerfile: str,
) -> None:
    """One interpreter across both deployments, pinned by digest in both.

    Two deployments of "the same system" resolving onto different CPython builds
    would make every comparison between them approximate.
    """
    pattern = r"ARG PYTHON_IMAGE=(\S+)"
    render = re.search(pattern, render_dockerfile)
    vps = re.search(pattern, VPS_DOCKERFILE.read_text(encoding="utf-8"))
    assert render is not None and vps is not None
    assert render.group(1) == vps.group(1)
    assert "python:3.12-slim@sha256:" in render.group(1)


def test_the_render_image_pins_the_same_installer_as_the_vps_image(
    render_dockerfile: str,
) -> None:
    """An installer that resolved differently would defeat the shared lockfile."""
    pattern = r"ARG UV_VERSION=(\S+)"
    render = re.search(pattern, render_dockerfile)
    vps = re.search(pattern, VPS_DOCKERFILE.read_text(encoding="utf-8"))
    assert render is not None and vps is not None
    assert render.group(1) == vps.group(1)


def test_the_proxy_binary_is_pinned_by_digest(render_dockerfile: str) -> None:
    """And it is the same Caddy the VPS deployment runs."""
    match = re.search(r"ARG CADDY_IMAGE=(\S+)", render_dockerfile)
    assert match is not None
    assert "@sha256:" in match.group(1)
    overlay = VPS_DEPLOY_OVERLAY.read_text(encoding="utf-8")
    assert match.group(1) in overlay


def test_no_stage_uses_a_floating_tag(render_dockerfile: str) -> None:
    """A tag somebody else can move is not a pin."""
    for instruction in _instructions(render_dockerfile):
        if instruction.startswith("FROM "):
            reference = instruction.split()[1]
            assert ":latest" not in reference
            assert reference.startswith("${") or "@sha256:" in reference


def test_the_image_installs_no_development_dependencies(
    render_dockerfile: str,
) -> None:
    """pytest, mypy, ruff and pre-commit have no business in a runtime image."""
    syncs = [i for i in _instructions(render_dockerfile) if "uv sync" in i]
    assert syncs
    for instruction in syncs:
        assert "--no-dev" in instruction
        assert "--frozen" in instruction


def test_the_build_runs_the_tracked_preparation_script(
    render_dockerfile: str,
) -> None:
    """The real pipeline, from the same script the Compose job runs.

    Not a reimplementation and not a subset: if this stopped naming
    ``prepare_demo_bundle.py``, the image would be carrying a bundle produced by
    something other than the project's own pipeline.
    """
    text = render_dockerfile
    assert "scripts/prepare_demo_bundle.py --state-root /srv/state" in text
    compose_command = VPS_COMPOSE.read_text(encoding="utf-8")
    assert "/app/scripts/prepare_demo_bundle.py" in compose_command


def test_the_build_fails_when_the_bundle_does_not_verify(
    render_dockerfile: str,
) -> None:
    """Verification is a build step, so an unverifiable bundle never ships.

    Run twice: once on what the pipeline produced, and once after the prune, so
    a prune that removed something serving needs fails the build rather than the
    deployment.
    """
    verifications = [
        instruction
        for instruction in _instructions(render_dockerfile)
        if "verify_serving_bundle.py" in instruction and instruction.startswith("RUN")
    ]
    assert len(verifications) >= 2


def test_the_preparation_happens_in_a_stage_that_is_not_the_runtime_image(
    render_dockerfile: str,
) -> None:
    """The training scratch exists only in a discarded layer.

    A runtime stage that ran the preparation itself would be a service that fits
    models, which is the one thing this project's serving contract forbids.
    """
    instructions = _instructions(render_dockerfile)
    stages = [i for i in instructions if i.startswith("FROM ")]
    assert any(i.endswith("AS prepare") for i in stages)
    assert any(i.endswith("AS runtime") for i in stages)

    runtime_index = next(
        index for index, i in enumerate(instructions) if i.endswith("AS runtime")
    )
    after_runtime = instructions[runtime_index:]
    for instruction in after_runtime:
        assert "prepare_demo_bundle.py" not in instruction, instruction


def test_the_runtime_image_receives_the_prepared_state_by_copy(
    render_dockerfile: str,
) -> None:
    """``COPY --from=prepare``: the bundle is baked, immutably, and owned by root."""
    copies = [
        instruction
        for instruction in _instructions(render_dockerfile)
        if instruction.startswith("COPY") and "--from=prepare" in instruction
    ]
    assert len(copies) == 1
    assert "/srv/state" in copies[0]
    assert "--chown=root:root" in copies[0]


def test_the_build_prunes_what_serving_does_not_read(
    render_dockerfile: str,
) -> None:
    """Datasets, feature snapshots, rule output and reports are build inputs."""
    removals = [
        instruction
        for instruction in _instructions(render_dockerfile)
        if instruction.startswith("RUN ") and "rm -rf" in instruction
    ]
    assert len(removals) == 1
    for pruned in ("dataset", "processed", "detection", "reports"):
        assert f"/srv/state/{pruned}" in removals[0]
    # ...and the two things serving actually reads are never among them.
    assert "/srv/state/artifacts" not in removals[0]
    assert "/srv/state/allowlist.yaml" not in removals[0]


def test_the_runtime_account_is_unprivileged_and_cannot_log_in(
    render_dockerfile: str,
) -> None:
    """Nothing in this container needs root, including PID 1."""
    text = render_dockerfile
    assert "USER pad:pad" in text
    assert '--uid "${APP_UID}"' in text
    assert f"APP_UID={RUNTIME_UID}" in text
    assert "--shell /usr/sbin/nologin" in text


def test_nothing_after_the_user_switch_needs_root(render_dockerfile: str) -> None:
    """A build step running as root after the switch would silently undo it."""
    instructions = _instructions(render_dockerfile)
    index = instructions.index("USER pad:pad")
    for instruction in instructions[index + 1 :]:
        assert not instruction.startswith("RUN "), instruction
        assert not instruction.startswith("COPY "), instruction


def test_the_image_carries_no_source_tree_or_tests(render_dockerfile: str) -> None:
    """The runtime stage installs the built package and copies no ``src/``."""
    instructions = _instructions(render_dockerfile)
    index = next(
        i for i, line in enumerate(instructions) if line.endswith("AS runtime")
    )
    for instruction in instructions[index:]:
        if instruction.startswith("COPY") and "--from=" not in instruction:
            assert " src" not in f" {instruction}"
            assert "tests" not in instruction


def test_no_stage_copies_the_whole_context(render_dockerfile: str) -> None:
    """``COPY . .`` would make .dockerignore the only thing standing between the
    build and every local file in the working tree."""
    for instruction in _instructions(render_dockerfile):
        if instruction.startswith("COPY") and "--from=" not in instruction:
            sources = [p for p in instruction.split()[1:-1] if not p.startswith("--")]
            assert "." not in sources, instruction


def test_the_command_is_declared_in_exec_form(render_dockerfile: str) -> None:
    """So the supervisor is PID 1 and receives SIGTERM directly.

    A shell-form CMD would put ``/bin/sh`` at PID 1, which does not forward
    signals: the three processes behind it would be killed rather than stopped,
    and the API's graceful shutdown would never run.
    """
    commands = [i for i in _instructions(render_dockerfile) if i.startswith("CMD ")]
    assert len(commands) == 1
    assert commands[0] == 'CMD ["python", "/app/scripts/render_entrypoint.py"]'


def test_the_image_uses_no_docker_socket_and_no_nested_docker(
    render_dockerfile: str,
) -> None:
    """One container, three processes. No Compose inside Render, no Docker-in-Docker."""
    text = render_dockerfile.lower()
    for forbidden in (
        "docker.sock",
        "docker-compose",
        "dind",
        "apt-get install docker",
    ):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# The image's environment: locations and facility switches only
# ---------------------------------------------------------------------------


def test_no_environment_variable_restates_a_frozen_scientific_decision(
    render_dockerfile: str,
) -> None:
    """The same guard the API and the console enforce on their own settings."""
    by_prefix = {
        "PAD_API_": {name.upper() for name in API_PROHIBITED},
        "PAD_DASHBOARD_": {name.upper() for name in DASHBOARD_PROHIBITED},
    }
    for key in _env_pairs(render_dockerfile):
        for prefix, prohibited in by_prefix.items():
            if key.startswith(prefix):
                assert key.removeprefix(prefix) not in prohibited, key


def test_no_environment_variable_even_sounds_like_a_scientific_control(
    render_dockerfile: str,
) -> None:
    """A broader net than the prohibited list, cast on purpose."""
    suspicious = ("MODEL", "THRESHOLD", "FUSION", "STRATEGY", "CALIBRAT", "SCORE")
    for key in _env_pairs(render_dockerfile):
        if key.startswith(("PAD_API_", "PAD_DASHBOARD_")):
            assert not any(word in key for word in suspicious), key


def test_the_api_binds_loopback_only(render_dockerfile: str) -> None:
    """The security boundary the whole single-container design rests on."""
    env = _env_pairs(render_dockerfile)
    assert env["PAD_API_HOST"] == "127.0.0.1"
    assert env["PAD_API_PORT"] == "8000"


def test_the_console_binds_loopback_only(render_dockerfile: str) -> None:
    """Restated in the environment as well as in .streamlit/config.toml."""
    env = _env_pairs(render_dockerfile)
    assert env["STREAMLIT_SERVER_ADDRESS"] == "127.0.0.1"
    assert env["STREAMLIT_SERVER_PORT"] == "8501"
    assert env["STREAMLIT_SERVER_HEADLESS"] == "true"


def test_the_console_calls_the_api_over_loopback(render_dockerfile: str) -> None:
    """Never the public Render URL.

    Routing an internal call out through Render's edge and back would double the
    latency, put internal traffic on the public internet, and require the API to
    be publicly proxied -- which the routing policy refuses.
    """
    env = _env_pairs(render_dockerfile)
    assert env["PAD_DASHBOARD_API_URL"] == f"http://{API_UPSTREAM}"
    assert "onrender.com" not in render_dockerfile


def test_the_interactive_schema_browser_is_off(render_dockerfile: str) -> None:
    """The API is not publicly proxied, so /docs is unreachable regardless.

    Turned off anyway: the smallest useful surface is the one that does not rely
    on a proxy rule to stay small.
    """
    assert _env_pairs(render_dockerfile)["PAD_API_DOCS_ENABLED"] == "false"


def test_the_replay_demonstration_is_a_facility_switch(
    render_dockerfile: str,
) -> None:
    """On, and not required for readiness -- as on the VPS deployment."""
    env = _env_pairs(render_dockerfile)
    assert env["PAD_API_REPLAY_ENABLED"] == "true"
    assert env["PAD_API_REPLAY_REQUIRED"] == "false"


def test_every_configured_artifact_path_is_absolute(render_dockerfile: str) -> None:
    """``APISettings`` refuses a relative one; failing here is cheaper."""
    env = _env_pairs(render_dockerfile)
    for key, value in env.items():
        if key.startswith("PAD_API_") and ("PATH" in key or key.endswith("_ROOT")):
            assert value.startswith("/"), f"{key}={value}"


def test_the_image_declares_no_public_port_literal(render_dockerfile: str) -> None:
    """Render assigns the port; nothing here may guess it.

    10000 is what Render happens to use today. An image that baked it would pass
    every local test and be unreachable the day that changed.
    """
    env = _env_pairs(render_dockerfile)
    assert "PORT" not in env
    for instruction in _instructions(render_dockerfile):
        assert not instruction.startswith("EXPOSE "), instruction


def test_the_image_carries_no_credential_material(render_dockerfile: str) -> None:
    """The service accepts none, so a secret here would have nowhere to go."""
    text = render_dockerfile
    for pattern in (
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\brnd_[A-Za-z0-9]{20,}",
        r"\bghp_[A-Za-z0-9]{20,}",
        r"Authorization:\s*Bearer\s+\S+",
    ):
        assert re.search(pattern, text) is None, pattern
    env = _env_pairs(text)
    for key in env:
        assert not any(
            word in key.upper() for word in ("SECRET", "TOKEN", "PASSWORD", "API_KEY")
        ), key


# ---------------------------------------------------------------------------
# The proxy: one public listener, one public API path
# ---------------------------------------------------------------------------


def test_the_proxy_listens_on_the_platform_supplied_port(
    render_caddyfile: str,
) -> None:
    """``:{$PORT}``, with no default.

    A default would let a misconfigured container come up quietly on a port
    nobody asked for; the entrypoint validates PORT before Caddy is started, so
    there is nothing for a default to rescue.
    """
    assert ":{$PORT}" in render_caddyfile
    assert ":{$PORT:" not in render_caddyfile
    assert "10000" not in render_caddyfile


def test_the_proxy_admin_api_is_off(render_caddyfile: str) -> None:
    """An unauthenticated endpoint that can rewrite the routing policy."""
    assert "admin off" in _caddy_directives(render_caddyfile)


def test_the_proxy_does_not_attempt_certificate_issuance(
    render_caddyfile: str,
) -> None:
    """Render terminates TLS at its edge; there is no name here to obtain one for."""
    assert "auto_https off" in _caddy_directives(render_caddyfile)


def test_the_proxy_reaches_both_processes_over_loopback(
    render_caddyfile: str,
) -> None:
    """And over loopback only -- no service name, no external host."""
    upstreams = re.findall(r"reverse_proxy\s+(\S+)", render_caddyfile)
    assert set(upstreams) == {API_UPSTREAM, DASHBOARD_UPSTREAM}


def test_the_console_is_the_default_route(render_caddyfile: str) -> None:
    """Anything not explicitly matched goes to the console, never to the API."""
    body = render_caddyfile[render_caddyfile.index(":{$PORT}") :]
    fallback = body[body.rindex("handle {") :]
    assert f"reverse_proxy {DASHBOARD_UPSTREAM}" in fallback


def test_the_only_public_api_route_is_liveness(render_caddyfile: str) -> None:
    """One matcher, one path, rewritten to the API's liveness endpoint.

    ``/health`` reads no artifact, no model and no filesystem, and its body is a
    status and the package version -- both already on the console's own page.
    """
    matchers = re.findall(r"@\w+\s+path\s+(.+)", render_caddyfile)
    assert matchers == [PUBLIC_HEALTH_PATH]
    assert f"rewrite * {API_LIVENESS_PATH}" in render_caddyfile


@pytest.mark.parametrize(
    "path",
    [
        "/ready",
        "/version",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/detect",
        "/api/v1/explain",
        "/api/v1/demo",
        "/api/v1/system",
        "/api/v1/model",
        "/api/v1/rules",
    ],
)
def test_no_detection_replay_or_system_route_is_publicly_proxied(
    render_caddyfile: str, path: str
) -> None:
    """The routing policy the VPS deployment established, carried forward.

    ``/ready`` is on this list deliberately: it names which scientific component
    is unavailable and why, which is diagnostic detail for an operator reading
    logs rather than for an anonymous caller. The console renders the same
    information for a viewer, having asked for it over loopback.
    """
    assert path not in _caddy_body(render_caddyfile)


@pytest.mark.parametrize(
    "header",
    [
        "X-Content-Type-Options",
        "Referrer-Policy",
        "Permissions-Policy",
        "X-Frame-Options",
        "Content-Security-Policy",
        "Strict-Transport-Security",
    ],
)
def test_the_proxy_hardens_every_response(render_caddyfile: str, header: str) -> None:
    """The same header set the VPS deployment offers.

    A header offered on one deployment of this console and not the other would
    be a difference nobody decided.
    """
    assert header in render_caddyfile
    assert header in VPS_CADDYFILE.read_text(encoding="utf-8")


def test_the_content_security_policy_is_the_audited_one(
    render_caddyfile: str,
) -> None:
    """``frame-ancestors`` only, and no 'unsafe-inline' anywhere.

    Streamlit's client evaluates generated code and installs inline styles: a
    script-src tight enough to be worth having breaks the console, and one loose
    enough to keep it working claims a protection it does not provide.
    """
    body = _caddy_body(render_caddyfile)
    assert "Content-Security-Policy \"frame-ancestors 'none'\"" in body
    assert "unsafe-inline" not in body
    assert "unsafe-eval" not in body


def test_the_proxy_bounds_the_request_body(render_caddyfile: str) -> None:
    """Matched to the API's own ceiling rather than chosen independently."""
    assert "max_size 1MB" in render_caddyfile


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


def test_the_supervisor_imports_nothing_from_the_project(entrypoint: Any) -> None:
    """PID 1 lives for the container's lifetime on a 512 MiB budget.

    Importing the project package here would pull numpy, pandas and
    scikit-learn into a process whose entire job is to wait, and hold their
    memory for the whole deployment.
    """
    text = RENDER_ENTRYPOINT.read_text(encoding="utf-8")
    assert "password_attack_detector" not in text.replace(
        "password_attack_detector.api.app:app", ""
    ).replace("password_attack_detector/dashboard/app.py", ""), (
        "the supervisor imports the project package"
    )
    assert not hasattr(entrypoint, "np")


@pytest.mark.parametrize("value", ["1", "80", "8080", "10000", "34567", "65535"])
def test_any_valid_port_is_accepted(entrypoint: Any, value: str) -> None:
    """Render's assignment is read, never assumed."""
    assert entrypoint.resolve_public_port({"PORT": value}) == int(value)


def test_a_port_with_surrounding_whitespace_is_accepted(entrypoint: Any) -> None:
    """An environment variable that arrived with a stray newline still works."""
    assert entrypoint.resolve_public_port({"PORT": " 10000\n"}) == 10000


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"PORT": ""},
        {"PORT": "   "},
        {"PORT": "http"},
        {"PORT": "80.5"},
        {"PORT": "-1"},
        {"PORT": "0"},
        {"PORT": "65536"},
        {"PORT": "99999999"},
    ],
    ids=[
        "unset",
        "empty",
        "blank",
        "not-a-number",
        "not-an-integer",
        "negative",
        "zero",
        "one-past-the-end",
        "far-too-large",
    ],
)
def test_a_missing_or_invalid_port_refuses_to_start(
    entrypoint: Any, environ: dict[str, str]
) -> None:
    """Loudly, with a message, rather than by binding something arbitrary.

    Port 0 is refused with the rest: it means "any free port", which for a
    service whose entire contract is *this* port is a failure dressed as success.
    """
    with pytest.raises(ValueError):
        entrypoint.resolve_public_port(environ)


def test_the_supervisor_never_names_a_default_port(entrypoint: Any) -> None:
    """Not even as a fallback constant somebody could later start using.

    Checked over the parsed syntax tree rather than the text, because the module
    explains in prose *why* it refuses to default to 10000 and a text search
    would read the explanation as the behaviour.
    """
    tree = ast.parse(RENDER_ENTRYPOINT.read_text(encoding="utf-8"))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }
    assert 10_000 not in literals
    assert 8_080 not in literals


def test_the_api_is_started_on_loopback_without_a_reloader(entrypoint: Any) -> None:
    """``--reload`` would watch a source tree the runtime image does not carry,
    and would fork a worker this supervisor does not know about."""
    command = entrypoint.api_command()
    assert "--host" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8000"
    assert "--reload" not in command
    assert "--no-server-header" in command


def test_the_api_is_given_time_to_shut_down_gracefully(entrypoint: Any) -> None:
    """So in-flight replay tasks are cancelled and recorded as stopped."""
    command = entrypoint.api_command()
    assert command[command.index("--timeout-graceful-shutdown") + 1] == "20"


def test_the_console_is_started_on_loopback(entrypoint: Any) -> None:
    """The console is reachable through the proxy, and from nowhere else."""
    command = entrypoint.dashboard_command()
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--server.port") + 1] == "8501"


def test_the_supervisor_grace_period_outlasts_the_api_shutdown_budget(
    entrypoint: Any,
) -> None:
    """Killing at exactly the API's own deadline would race it."""
    assert entrypoint.SHUTDOWN_GRACE > 20.0


def test_the_supervisor_starts_the_proxy_from_the_baked_configuration(
    entrypoint: Any,
) -> None:
    """One file, at a fixed path, copied into the image."""
    command = entrypoint.proxy_command()
    assert command[0] == "caddy"
    assert entrypoint.CADDYFILE in command
    assert entrypoint.CADDYFILE == "/etc/caddy/Caddyfile"


def test_the_supervisor_verifies_the_bundle_before_starting_anything(
    entrypoint: Any,
) -> None:
    """And does it in a subprocess, so the memory is given back."""
    command = entrypoint.verify_bundle_command()
    assert entrypoint.VERIFIER in command
    assert entrypoint.STATE_ROOT in command
    source = RENDER_ENTRYPOINT.read_text(encoding="utf-8")
    assert "subprocess.run(verify_bundle_command()" in source


def test_the_supervisor_distinguishes_its_failure_modes(entrypoint: Any) -> None:
    """So a Render log says which invariant broke, not merely that one did."""
    codes = {
        entrypoint.EXIT_OK,
        entrypoint.EXIT_STARTUP_FAILED,
        entrypoint.EXIT_BAD_PORT,
        entrypoint.EXIT_BUNDLE_UNVERIFIED,
        entrypoint.EXIT_CHILD_DIED,
    }
    assert len(codes) == 5
    assert entrypoint.EXIT_OK == 0
    assert all(code > 0 for code in codes - {entrypoint.EXIT_OK})


def test_the_supervisor_fits_nothing_and_names_no_scientific_decision() -> None:
    """It starts processes. It does not select, train, calibrate or threshold."""
    text = RENDER_ENTRYPOINT.read_text(encoding="utf-8").lower()
    for word in ("fit(", "train(", "sklearn", "joblib", "threshold=", "champion="):
        assert word not in text, word


def test_the_supervisor_waits_without_polling() -> None:
    """A blocking select rather than a sleep loop.

    On a 0.1 CPU instance the difference between a process that wakes up
    continuously and one that does not is the difference between a demonstration
    that responds and one that does not.
    """
    text = RENDER_ENTRYPOINT.read_text(encoding="utf-8")
    assert "select.select" in text
    assert "wait_for_event(None)" in text
    assert "time.sleep" not in text


def test_the_supervisor_reaps_orphans_as_well_as_its_own_children() -> None:
    """PID 1 inherits the container's orphans; reaping only known pids leaves
    zombies in a process that never restarts."""
    text = RENDER_ENTRYPOINT.read_text(encoding="utf-8")
    assert "os.waitpid(-1, os.WNOHANG)" in text


def test_the_supervisor_signals_process_groups() -> None:
    """Each child leads its own session, so a grandchild cannot outlive a stop."""
    text = RENDER_ENTRYPOINT.read_text(encoding="utf-8")
    assert "os.setsid()" in text
    assert "os.killpg" in text


# ---------------------------------------------------------------------------
# The bundle verifier
# ---------------------------------------------------------------------------


def test_the_verifier_cannot_name_a_strategy_model_or_threshold() -> None:
    """It reports what the bundle declares. Reporting is not deciding.

    A verifier with a ``--require-strategy`` option would be making the fusion
    selection that the locked evaluation already made.
    """
    text = BUNDLE_VERIFIER.read_text(encoding="utf-8")
    options = re.findall(r'add_argument\(\s*"(--[a-z-]+)"', text)
    assert set(options) == {"--state-root", "--json"}


def test_the_verifier_uses_the_projects_own_bundle_loader() -> None:
    """Not a reimplementation of the checks it is standing in for."""
    text = BUNDLE_VERIFIER.read_text(encoding="utf-8")
    assert (
        "from password_attack_detector.deployment.bundle import load_serving_bundle"
        in text
    )


def test_the_verifier_reads_the_scope_key_from_the_receipt() -> None:
    """Rather than accepting one, so it cannot be pointed at another champion."""
    text = BUNDLE_VERIFIER.read_text(encoding="utf-8")
    assert 'receipt["champion_scope_key"]' in text
    assert "--champion" not in text
    assert "--scope" not in text


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


def test_the_render_documentation_does_not_claim_a_live_deployment() -> None:
    """Nothing is deployed. Saying otherwise in a document is how it stops being
    obvious that nothing is deployed."""
    text = RENDER_DOCS.read_text(encoding="utf-8").lower()
    assert "not deployed" in text or "no render service" in text
    assert "https://pad-demo.onrender.com" not in text


def test_the_render_documentation_states_the_baseline_limitation() -> None:
    """Two rules cannot fire on any live request, and the document says so."""
    text = RENDER_DOCS.read_text(encoding="utf-8")
    assert "PAD-CS-001" in text
    assert "PAD-ATO-001" in text


def test_the_render_documentation_states_that_nothing_persists() -> None:
    """A viewer who runs a replay and reloads later must not be surprised."""
    text = RENDER_DOCS.read_text(encoding="utf-8").lower()
    assert "ephemeral" in text
    assert "replay history" in text
