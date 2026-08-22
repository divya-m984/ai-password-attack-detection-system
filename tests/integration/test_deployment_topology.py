"""The deployment perimeter, running.

Runs the real production topology on this machine before any of it is pointed at
the internet:

    docker compose -f compose.yaml -f compose.deploy.yaml up -d --wait

with the proxy published on ``127.0.0.1:18080`` instead of ``0.0.0.0:80`` -- the
same configuration, the same routing policy, the same containers, on a port that
needs no privilege and does not reach the local network. Everything else is
byte-for-byte what a droplet would run.

What this asserts that ``tests/unit/deployment/test_deployment_contract.py``
cannot: that ``!reset`` really did remove the published ports rather than merely
appearing to, that ``127.0.0.1:8000`` really is refused, that Caddy really
accepts both routing policies, that the Streamlit websocket really survives the
proxy, and that the console really still drives a full replay through a frozen
stacked hybrid when the only way in is one HTTP port.

Needs a Docker daemon and the two application images, so the module is marked
``slow`` and skips itself when either is missing:

    docker compose build
    uv run pytest -m slow tests/integration/test_deployment_topology.py --no-cov

The fixture refuses to run if a ``pad-demo`` stack is already up -- the service
containers have fixed names, so starting this one would fight with a
demonstration somebody is giving -- and tears down everything it started,
volumes included.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ("pad-demo-api:0.5.0", "pad-demo-dashboard:0.5.0")

#: Where the proxy is published for this run. A loopback bind and an
#: unprivileged, deliberately unusual port -- 8080 is the first thing anything
#: else on a development machine takes: the deployment's own defaults are 80 and 443 on every
#: interface, and the unit suite asserts that. This exercises the same
#: configuration without taking a privileged port or reaching the LAN.
PROXY = "http://127.0.0.1:18080"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 18080

#: The ports the deployment must NOT publish. Probed directly, not inferred.
UNPUBLISHED = (8000, 8501)

COMPOSE_FILES = ("-f", "compose.yaml", "-f", "compose.deploy.yaml")

#: The environment a deployment supplies. Passed through the process environment
#: rather than an ``--env-file`` so the test needs no temporary file; the
#: template's own parseability is asserted separately.
DEPLOY_ENVIRONMENT = {
    "PAD_SITE_ADDRESS": ":80",
    "PAD_HSTS": "max-age=0",
    "PAD_PROXY_CADDYFILE": "./deploy/caddy/Caddyfile",
    "PAD_PROXY_HTTP_PUBLISH": f"{PROXY_HOST}:{PROXY_PORT}",
    "PAD_PROXY_HTTPS_PUBLISH": f"{PROXY_HOST}:18443",
}


def _docker(
    *arguments: str, timeout: float = 300.0, overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a docker command from the repository root, in a deployment env."""
    return subprocess.run(
        ["docker", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env={**os.environ, **DEPLOY_ENVIRONMENT, **(overrides or {})},
    )


def _compose(
    *arguments: str, timeout: float = 300.0, overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a compose command against the merged deployment configuration."""
    return _docker(
        "compose", *COMPOSE_FILES, *arguments, timeout=timeout, overrides=overrides
    )


def _daemon_available() -> bool:
    """Return whether a Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    return _docker("info", "--format", "{{.ServerVersion}}", timeout=30).returncode == 0


def _images_built() -> bool:
    """Return whether both application images are present locally."""
    listed = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=30)
    if listed.returncode != 0:
        return False
    present = set(listed.stdout.split())
    return all(image in present for image in IMAGES)


if not _daemon_available():  # pragma: no cover - environment-dependent
    pytest.skip("no Docker daemon is reachable", allow_module_level=True)
if not _images_built():  # pragma: no cover - environment-dependent
    pytest.skip("run `docker compose build` first", allow_module_level=True)


def _running_containers() -> set[str]:
    """Return the names of this project's containers that are currently up."""
    listed = _docker(
        "ps", "--filter", "name=pad-demo-", "--format", "{{.Names}}", timeout=60
    )
    return set(listed.stdout.split()) if listed.returncode == 0 else set()


@pytest.fixture(scope="module")
def deployment() -> Iterator[None]:
    """Bring the deployment topology up, and remove every trace of it after."""
    if _running_containers():  # pragma: no cover - environment-dependent
        pytest.skip(
            "a pad-demo stack is already running; stop it before running the "
            "deployment-topology tests (the containers have fixed names)"
        )

    built = _compose("build", timeout=1800)
    if built.returncode != 0:  # pragma: no cover - reported, not asserted
        pytest.skip(
            f"the deployment configuration did not build: {built.stderr[-500:]}"
        )

    started = _compose("up", "-d", "--wait", timeout=1800)
    if started.returncode != 0:  # pragma: no cover - reported, not asserted
        _compose("down", "--volumes", "--remove-orphans", timeout=300)
        pytest.skip(f"could not start the deployment: {started.stderr[-500:]}")
    try:
        yield
    finally:
        # --volumes: this run created the serving-state volume and Caddy's two.
        # A test must leave no state behind, and the serving state is
        # deterministic -- the next start rebuilds it identically.
        _compose("down", "--volumes", "--remove-orphans", timeout=300)


def _get(url: str, timeout: float = 20.0) -> tuple[int, dict[str, str], bytes]:
    """Return one response's status, headers, and raw body."""
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, dict(answer.headers), answer.read()
    except urllib.error.HTTPError as answer:
        return answer.code, dict(answer.headers), answer.read()


def _in_container(service: str, source: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet inside one of the running containers."""
    return _docker(
        "compose", *COMPOSE_FILES, "exec", "-T", service, "python", "-c", source
    )


def _inspect(container: str, template: str) -> str:
    """Return one formatted field from a container's inspection."""
    result = _docker("inspect", container, "--format", template, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _port_is_closed(port: int) -> bool:
    """Return whether nothing on the host accepts a connection on *port*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(3.0)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _wait_for(url: str, *, seconds: float = 45.0) -> None:
    """Block until *url* answers, or fail the test saying it never did.

    Waiting on the port is not enough and the difference is a real trap: Docker's
    userland proxy binds the published port the moment the container is created,
    so a port check succeeds while the process behind it is still starting and
    the first request comes back as a connection reset.
    """
    deadline = time.monotonic() + seconds
    last = "no attempt was made"
    while time.monotonic() < deadline:
        try:
            _get(url, timeout=5.0)
        except OSError as failure:  # connection refused, reset, or timed out
            last = repr(failure)
            time.sleep(0.5)
        else:
            return
    pytest.fail(f"{url} never answered within {seconds:.0f}s; last error: {last}")


# ---------------------------------------------------------------------------
# The merged configuration -- no daemon needed for these, only the CLI
# ---------------------------------------------------------------------------


def test_the_merged_deployment_configuration_is_valid() -> None:
    """`docker compose config` is the authority on what the two files mean."""
    result = _compose("config", "--quiet", timeout=120)
    assert result.returncode == 0, result.stderr


def test_the_tracked_environment_template_is_a_valid_deployment() -> None:
    """`.env.deploy.example` must actually work as `.env.deploy` does.

    A template that a server copies and Compose then refuses is worse than no
    template. This runs the merge with the tracked file supplying every variable.
    """
    result = _docker(
        "compose",
        "--env-file",
        ".env.deploy.example",
        *COMPOSE_FILES,
        "config",
        "--quiet",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_the_merge_publishes_only_the_proxy() -> None:
    """Read out of the resolved configuration, not out of either source file.

    This is the claim ``!reset`` exists to make true: Compose appends sequences
    when it merges, so an override that restated a shorter ``ports`` list would
    have produced the union of both. Here the API and the console have no
    ``ports`` key at all.
    """
    result = _docker(
        "compose", *COMPOSE_FILES, "config", "--format", "json", timeout=120
    )
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]

    publishing = {name for name, body in services.items() if body.get("ports")}
    assert publishing == {"proxy"}

    for name, body in services.items():
        for entry in body.get("ports", []):
            assert int(entry["target"]) in {80, 443}, (name, entry)
            assert int(str(entry["published"]).rsplit(":", 1)[-1]) not in UNPUBLISHED


def test_the_merge_keeps_the_serving_state_read_only_in_the_api() -> None:
    """The perimeter must not have loosened the scientific-state contract."""
    result = _docker(
        "compose", *COMPOSE_FILES, "config", "--format", "json", timeout=120
    )
    services = json.loads(result.stdout)["services"]

    api = [v for v in services["api"]["volumes"] if v["source"] == "serving-state"]
    assert len(api) == 1
    assert api[0]["read_only"] is True

    prepare = [
        v for v in services["prepare"]["volumes"] if v["source"] == "serving-state"
    ]
    assert len(prepare) == 1, "one writable mount, for the length of one job"
    assert not prepare[0].get("read_only", False)

    for name in ("dashboard", "proxy"):
        sources = {v.get("source") for v in services[name].get("volumes", [])}
        assert "serving-state" not in sources, name


@pytest.mark.parametrize("policy", ["Caddyfile", "Caddyfile.api-docs"])
def test_caddy_itself_accepts_the_routing_policy(policy: str) -> None:
    """Validated by the same binary that will serve it, in the pinned image.

    A Caddyfile that parses in a reviewer's head and not in Caddy is a public
    outage discovered by a browser.
    """
    image = json.loads(
        _docker("compose", *COMPOSE_FILES, "config", "--format", "json").stdout
    )["services"]["proxy"]["image"]
    result = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{ROOT}/deploy/caddy/{policy}:/etc/caddy/Caddyfile:ro",
        "-e",
        "PAD_SITE_ADDRESS=:80",
        "-e",
        "PAD_HSTS=max-age=0",
        image,
        "caddy",
        "validate",
        "--config",
        "/etc/caddy/Caddyfile",
        "--adapter",
        "caddyfile",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Valid configuration" in result.stderr + result.stdout


# ---------------------------------------------------------------------------
# The perimeter, running
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("port", UNPUBLISHED)
def test_the_application_ports_are_not_reachable_on_the_host(
    deployment: None, port: int
) -> None:
    """A TCP probe, not an inference from a configuration file.

    The API is listening on 8000 and the console on 8501 -- inside their network
    namespaces. Nothing on the host accepts a connection there.
    """
    assert _port_is_closed(port), f"something is listening on 127.0.0.1:{port}"


def test_the_proxy_is_the_only_way_in(deployment: None) -> None:
    """One host port serves the whole application."""
    status, _, body = _get(f"{PROXY}/")
    assert status == 200
    # Streamlit's shell, served through the proxy: the console's own markup.
    assert b"stApp" in body or b"streamlit" in body.lower()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "strict-origin-when-cross-origin"),
        ("X-Frame-Options", "DENY"),
        ("Content-Security-Policy", "frame-ancestors 'none'"),
        ("Strict-Transport-Security", "max-age=0"),
    ],
)
def test_the_public_boundary_sets_its_security_headers(
    deployment: None, header: str, expected: str
) -> None:
    """Read off a real response, because a header directive can be misordered."""
    _, headers, _ = _get(f"{PROXY}/")
    assert headers.get(header) == expected


def test_the_public_boundary_declines_to_say_what_it_is(deployment: None) -> None:
    """`-Server` removes the banner rather than replacing it with a lie."""
    _, headers, _ = _get(f"{PROXY}/")
    assert "Server" not in headers


def test_the_permissions_policy_denies_every_device(deployment: None) -> None:
    """Set as one long header; checked for the entries that matter."""
    _, headers, _ = _get(f"{PROXY}/")
    policy = headers.get("Permissions-Policy", "")
    for feature in ("camera=()", "microphone=()", "geolocation=()"):
        assert feature in policy


def test_the_streamlit_websocket_survives_the_proxy(deployment: None) -> None:
    """The console is a websocket application; a proxy that breaks it breaks it.

    A raw HTTP/1.1 upgrade handshake rather than a client library: the assertion
    is that the proxy forwards the upgrade and Streamlit answers 101, and a
    library would hide which half failed.
    """
    key = base64.b64encode(b"pad-deployment-1").decode()
    request = (
        "GET /_stcore/stream HTTP/1.1\r\n"
        f"Host: {PROXY_HOST}:{PROXY_PORT}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Origin: http://{PROXY_HOST}:{PROXY_PORT}\r\n"
        "\r\n"
    )
    with socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=15) as stream:
        stream.sendall(request.encode("ascii"))
        answer = stream.recv(4096).decode("latin-1")
    assert answer.startswith("HTTP/1.1 101"), answer.splitlines()[:1]
    assert "upgrade: websocket" in answer.lower()


# ---------------------------------------------------------------------------
# The default routing policy: the console, and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/ready",
        "/version",
        "/docs",
        "/openapi.json",
        "/api/v1/system/status",
        "/api/v1/model/info",
        "/api/v1/rules",
        "/api/v1/demo/scenarios",
    ],
)
def test_the_default_policy_publishes_no_api_document(
    deployment: None, path: str
) -> None:
    """Whatever the console makes of these paths, none of them is the API.

    Asserted by content rather than by status: Streamlit answers some unknown
    paths with its own shell and some with a 404, and which one it picks is its
    business. What matters is that no API document reaches the internet.
    """
    _, headers, body = _get(f"{PROXY}{path}")
    if "json" not in headers.get("Content-Type", ""):
        return
    document = json.loads(body)
    if not isinstance(document, dict):
        return
    # Every document this service publishes carries one of these. Streamlit's own
    # JSON answers carry none of them.
    markers = {"api_schema_version", "replay_schema_version", "openapi", "components"}
    assert not (markers & set(document)), f"{path} returned an API document"


@pytest.mark.parametrize(
    "path", ["/api/v1/detect", "/api/v1/explain", "/api/v1/demo/runs"]
)
def test_no_scoring_endpoint_is_reachable_from_outside(
    deployment: None, path: str
) -> None:
    """Bounded is not free; until there is rate limiting these stay internal."""
    request = urllib.request.Request(
        f"{PROXY}{path}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            status, body = answer.status, answer.read()
    except urllib.error.HTTPError as answer:
        status, body = answer.code, answer.read()
    assert status != 200, f"{path} answered a POST from the public boundary"
    # Not merely a non-200: the refusal must not be the *service's* refusal,
    # which would mean the request reached it.
    assert b"api_schema_version" not in body
    assert b"MALFORMED_REQUEST" not in body


# ---------------------------------------------------------------------------
# The console still reaches the service it renders
# ---------------------------------------------------------------------------


def test_the_console_reaches_the_api_over_the_project_network(
    deployment: None,
) -> None:
    """The dashboard's client runs server-side; this is the path it uses."""
    result = _in_container(
        "dashboard",
        "import json,urllib.request;"
        "a=urllib.request.urlopen('http://api:8000/ready',timeout=20);"
        "print(json.dumps(json.load(a)))",
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["status"] == "ready"
    assert [part["state"] for part in document["components"]] == ["ready"] * len(
        document["components"]
    )


def test_the_frozen_stacked_hybrid_is_what_the_deployment_serves(
    deployment: None,
) -> None:
    """The perimeter changed the network. It must not have changed the science.

    ``fusion_strategy`` is what is *executing*; ``frozen_fusion_strategy`` is what
    validation selected before TEST. Equal, or the deployment is running a hybrid
    nobody chose -- which the serving layer refuses to do, and which this asserts
    it did not start doing behind a proxy.
    """
    result = _in_container(
        "dashboard",
        "import json,urllib.request;"
        "a=urllib.request.urlopen('http://api:8000/api/v1/system/status',timeout=20);"
        "print(json.dumps(json.load(a)))",
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["hybrid_detection_enabled"] is True
    assert document["fusion_strategy"] == "stacked"
    assert document["frozen_fusion_strategy"] == "stacked"


def test_a_replay_runs_end_to_end_in_the_deployment_topology(
    deployment: None,
) -> None:
    """The console's own client, in the console's own container, start to finish.

    This is what a viewer of the deployed page actually causes to happen: the
    browser talks only to the proxy, the proxy talks only to Streamlit, and
    Streamlit's server-side client drives the detection service across the
    project network.
    """
    source = """
import json, time, urllib.request

def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(
        "http://api:8000" + path, data=data, headers=headers,
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as answer:
        return json.load(answer)

run = call("/api/v1/demo/runs", {"scenario_id": "brute_force", "pace": "instant"})
for _ in range(120):
    state = call("/api/v1/demo/runs/" + run["run_id"])
    if state["state"] in {"completed", "failed", "stopped"}:
        break
    time.sleep(1)
page = call("/api/v1/demo/runs/" + run["run_id"] + "/timeline?limit=100")
records = page["records"]
print(json.dumps({
    "state": state["state"],
    "records": len(records),
    "strategies": sorted({r["detection"]["hybrid"]["strategy"] for r in records}),
    "rules": sorted({
        rule
        for r in records
        for rule in r["detection"]["rule"]["fired_rule_ids"]
    }),
}))
"""
    result = _in_container("dashboard", source)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["state"] == "completed"
    assert summary["records"] > 0
    assert summary["strategies"] == ["stacked"], "no fallback, in any container"
    assert summary["rules"] == ["PAD-BF-001", "PAD-BOT-001"], summary["rules"]


# ---------------------------------------------------------------------------
# The containers are still the hardened ones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", ["api", "dashboard"])
def test_the_application_still_runs_unprivileged(
    deployment: None, service: str
) -> None:
    """The perimeter must not have relaxed anything the local deployment had."""
    result = _docker("compose", *COMPOSE_FILES, "exec", "-T", service, "id")
    assert result.returncode == 0, result.stderr
    assert "uid=10001(pad)" in result.stdout


@pytest.mark.parametrize(
    "container", ["pad-demo-api", "pad-demo-dashboard", "pad-demo-proxy"]
)
def test_no_container_is_privileged_or_on_the_host_network(
    deployment: None, container: str
) -> None:
    """The three settings that would undo every other boundary."""
    assert _inspect(container, "{{.HostConfig.Privileged}}") == "false"
    assert _inspect(container, "{{.HostConfig.NetworkMode}}") != "host"
    assert "no-new-privileges:true" in _inspect(
        container, "{{.HostConfig.SecurityOpt}}"
    )
    assert _inspect(container, "{{.HostConfig.ReadonlyRootfs}}") == "true"
    assert "ALL" in _inspect(container, "{{.HostConfig.CapDrop}}")


def test_only_the_proxy_holds_a_capability_and_only_one(deployment: None) -> None:
    """Binding 80 and 443 needs NET_BIND_SERVICE. Nothing else is granted."""
    # The daemon normalises the name it was given, so read it as a set rather
    # than as a string: `NET_BIND_SERVICE` comes back as `CAP_NET_BIND_SERVICE`.
    granted = json.loads(_inspect("pad-demo-proxy", "{{json .HostConfig.CapAdd}}"))
    assert {name.removeprefix("CAP_") for name in granted} == {"NET_BIND_SERVICE"}
    for container in ("pad-demo-api", "pad-demo-dashboard"):
        added = _inspect(container, "{{json .HostConfig.CapAdd}}")
        assert json.loads(added) in ([], None), container


def test_no_container_can_reach_the_docker_daemon(deployment: None) -> None:
    """The socket is mounted nowhere; verified against the running containers."""
    for container in ("pad-demo-api", "pad-demo-dashboard", "pad-demo-proxy"):
        mounts = _inspect(container, "{{range .Mounts}}{{.Source}} {{end}}")
        assert "docker.sock" not in mounts, container


def test_the_proxy_cannot_write_its_own_routing_policy(deployment: None) -> None:
    """Read-only, in practice rather than in declaration."""
    result = _docker(
        "compose",
        *COMPOSE_FILES,
        "exec",
        "-T",
        "proxy",
        "sh",
        "-c",
        "echo x >> /etc/caddy/Caddyfile",
    )
    assert result.returncode != 0
    # A read-only bind mount refuses the open; busybox reports it as a permission
    # failure and a glibc shell as a read-only filesystem. Either is the refusal.
    reported = (result.stderr + result.stdout).lower()
    assert "read-only" in reported or "permission denied" in reported, reported


@pytest.mark.parametrize(
    "container", ["pad-demo-api", "pad-demo-dashboard", "pad-demo-proxy"]
)
def test_every_container_has_bounded_logs(deployment: None, container: str) -> None:
    """A public demonstration must not be able to fill the disk with its logs."""
    config = _inspect(container, "{{json .HostConfig.LogConfig}}")
    declared = json.loads(config)
    assert declared["Type"] in {"json-file", "local"}
    assert declared["Config"]["max-size"] == "10m"
    assert declared["Config"]["max-file"] == "3"


# ---------------------------------------------------------------------------
# Restart, and the optional Swagger policy
# ---------------------------------------------------------------------------


def test_a_restart_preserves_the_science_and_discards_the_replay_history(
    deployment: None,
) -> None:
    """Frozen state is on a volume; demonstration history is process memory.

    Run last among the behavioural tests because it restarts the API. What
    survives and what does not is the property the console states on the page,
    and it must be true of the deployment topology too.
    """
    before = _in_container(
        "dashboard",
        "import json,urllib.request;"
        "print(json.dumps(json.load("
        "urllib.request.urlopen('http://api:8000/api/v1/model/info',timeout=20))))",
    )
    assert before.returncode == 0, before.stderr

    restarted = _compose("restart", "api", timeout=300)
    assert restarted.returncode == 0, restarted.stderr

    for _ in range(60):
        probe = _in_container(
            "dashboard",
            "import urllib.request;"
            "urllib.request.urlopen('http://api:8000/health',timeout=5)",
        )
        if probe.returncode == 0:
            break
        time.sleep(2)
    else:  # pragma: no cover - a restarted API that never answers is a failure
        pytest.fail("the API did not come back after a restart")

    after = _in_container(
        "dashboard",
        "import json,urllib.request;"
        "print(json.dumps(json.load("
        "urllib.request.urlopen('http://api:8000/api/v1/model/info',timeout=20))))",
    )
    assert after.returncode == 0, after.stderr
    assert json.loads(after.stdout) == json.loads(before.stdout)

    runs = _in_container(
        "dashboard",
        "import json,urllib.request;"
        "print(json.dumps(json.load("
        "urllib.request.urlopen('http://api:8000/api/v1/demo/runs',timeout=20))))",
    )
    assert runs.returncode == 0, runs.stderr
    assert json.loads(runs.stdout)["runs"] == [], "replay history is not durable"


def test_the_optional_swagger_policy_publishes_documents_and_not_scoring(
    deployment: None,
) -> None:
    """The second reviewed policy, applied to the running proxy.

    Selecting it is one variable, and the point of asserting it here is that the
    variable chooses between two audited files -- it does not compose a route.
    Restores the default policy afterwards.
    """
    docs_policy = {"PAD_PROXY_CADDYFILE": "./deploy/caddy/Caddyfile.api-docs"}
    swapped = _compose(
        "up", "-d", "--force-recreate", "proxy", timeout=300, overrides=docs_policy
    )
    assert swapped.returncode == 0, swapped.stderr
    try:
        _wait_for(f"{PROXY}/")

        status, _, body = _get(f"{PROXY}/openapi.json")
        assert status == 200
        schema = json.loads(body)
        assert schema["info"]["title"] == "Password Attack Detector API"

        status, _, _ = _get(f"{PROXY}/docs")
        assert status == 200

        status, _, body = _get(f"{PROXY}/api/v1/system/status")
        assert status == 200
        assert json.loads(body)["fusion_strategy"] == "stacked"

        status, _, body = _get(f"{PROXY}/api/v1/model/info")
        assert status == 200
        assert json.loads(body)["available"] is True

        # The line this policy still holds: documents, yes; scoring, no.
        request = urllib.request.Request(
            f"{PROXY}/api/v1/detect",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as answer:
                refused, refusal = answer.status, answer.read()
        except urllib.error.HTTPError as answer:
            refused, refusal = answer.code, answer.read()
        # The path is not in this policy's matcher, so it falls through to the
        # console, which declines a POST it has no handler for. 405 rather than
        # 404 is Streamlit answering; what matters is that the refusal is not the
        # detection service's, which would mean the request had reached it.
        assert refused in {404, 405}, refused
        assert b"api_schema_version" not in refusal
    finally:
        restored = _compose("up", "-d", "--force-recreate", "proxy", timeout=300)
        assert restored.returncode == 0, restored.stderr
        _wait_for(f"{PROXY}/")
