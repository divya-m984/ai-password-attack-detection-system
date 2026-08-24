"""The Render deployment, running, under the free tier's actual limits.

Builds nothing and assumes nothing: it runs the image ``Dockerfile.render``
produces, in one container, with a 512 MiB ceiling, no swap, a read-only root
filesystem and no volume of any kind -- which is as close as a laptop gets to a
Render Free web service.

    docker build -f Dockerfile.render -t pad-render:0.5.0 .
    uv run pytest -m slow tests/integration/test_render_container.py --no-cov

What this asserts that ``tests/unit/deployment/test_render_contract.py`` cannot:

**That the baked bundle is the same bundle.** The headline claim of this
milestone is that moving preparation from a one-shot Compose job into a build
stage changed nothing scientific. ``test_the_baked_bundle_is_the_compose_prepared
_bundle`` runs the Compose preparation from scratch, in a separate project, and
compares every fingerprint the manifest carries against the ones inside the
image. It is slow because the only honest way to prove it is to do it.

**That the API is genuinely unreachable.** A status-code check is not enough
here: Streamlit answers 200 with its own single-page shell for *any* path it
does not recognise, so ``/api/v1/detect`` through the public port returns 200
whether the routing policy works or not. Every routing test therefore inspects
the body.

**That the ceiling is not merely survived but comfortable.** The container's own
cgroup is read for its peak and its OOM-event counters, rather than a sampled
``docker stats`` figure that can miss a spike.

The fixtures refuse to run beside a live demonstration and remove every
container they start.
"""

from __future__ import annotations

import ast
import base64
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]

#: The image under test, and the two the Compose reference preparation needs.
RENDER_IMAGE = "pad-render:0.5.0"
COMPOSE_IMAGE = "pad-demo-api:0.5.0"

#: The public port for the long-lived container. Deliberately not 10000: the
#: whole point of the port contract is that the value is Render's to choose, so
#: the test picks an arbitrary one and the deployment must cope.
SERVED_PORT = 19080
SERVED = f"http://127.0.0.1:{SERVED_PORT}"
SERVED_NAME = "pad-render-itest"

#: A second, unusual port for the lifecycle tests, chosen from a different range
#: again so nothing can quietly depend on either number.
LIFECYCLE_PORT = 41573

#: The free tier's runtime budget. Swap is pinned to the same value, because
#: ``--memory-swap`` defaulting to twice ``--memory`` would let the container
#: page out of a limit it is supposed to be tested against.
MEMORY_LIMIT = "512m"

#: The supervisor's exit codes, restated here rather than imported: this suite
#: observes the container from outside, and importing the module under test
#: would make the contract agree with itself.
EXIT_OK = 0
EXIT_BAD_PORT = 2
EXIT_BUNDLE_UNVERIFIED = 3
EXIT_CHILD_DIED = 4

#: Every scenario the catalog publishes must complete. Read from the running
#: service rather than hardcoded, but the count is pinned so a catalog that
#: silently shrank would not make this suite pass by testing less.
EXPECTED_SCENARIO_COUNT = 7


def _docker(
    *arguments: str, timeout: float = 300.0
) -> subprocess.CompletedProcess[str]:
    """Run one docker command from the repository root."""
    return subprocess.run(
        ["docker", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _daemon_available() -> bool:
    """Return whether a Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    return _docker("info", "--format", "{{.ServerVersion}}", timeout=30).returncode == 0


def _image_present(reference: str) -> bool:
    """Return whether *reference* is present locally."""
    listed = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=30)
    return reference in listed.stdout.split() if listed.returncode == 0 else False


if not _daemon_available():  # pragma: no cover - environment-dependent
    pytest.skip("no Docker daemon is reachable", allow_module_level=True)
if not _image_present(RENDER_IMAGE):  # pragma: no cover - environment-dependent
    pytest.skip(
        f"build the image first: docker build -f Dockerfile.render -t {RENDER_IMAGE} .",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Running the container
# ---------------------------------------------------------------------------


def _run_arguments(name: str, port: int, *, cpus: str | None = None) -> list[str]:
    """Return the ``docker run`` arguments for a free-tier-shaped container.

    Read-only root, no volume, no bind mount, no capability, and two tmpfs
    mounts for the paths frameworks insist on writing to. Render's own
    filesystem is writable but ephemeral; running read-only here proves the
    stronger property, which is that nothing scientific needs to be written at
    all.
    """
    arguments = [
        "run",
        "--detach",
        "--name",
        name,
        "--memory",
        MEMORY_LIMIT,
        "--memory-swap",
        MEMORY_LIMIT,
        "--read-only",
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/tmp:mode=1777,size=128m",
        "--tmpfs",
        "/home/pad:mode=0755,uid=10001,gid=10001,size=64m",
        "--env",
        f"PORT={port}",
        "--publish",
        f"127.0.0.1:{port}:{port}",
    ]
    if cpus is not None:
        arguments += ["--cpus", cpus]
    return [*arguments, RENDER_IMAGE]


def _wait_for(url: str, *, seconds: float = 420.0) -> None:
    """Block until *url* answers 200, or fail.

    Waits on the HTTP response rather than on the port. Docker's userland proxy
    binds a published port when the container is created, before anything inside
    is listening, so a connect check would succeed against a container that
    cannot answer yet.

    The budget is generous because a cold start on a throttled container is
    dominated by importing the scientific stack.
    """
    deadline = time.monotonic() + seconds
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as answer:
                if answer.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last = repr(exc)
        time.sleep(0.5)
    pytest.fail(f"{url} did not answer within {seconds:.0f}s (last: {last})")


def _remove(name: str) -> None:
    """Remove a container, running or not."""
    _docker("rm", "--force", "--volumes", name, timeout=120)


def _live_demonstration_running() -> bool:
    """Return whether a demonstration stack is already up on this machine."""
    listed = _docker("ps", "--format", "{{.Names}}", timeout=60)
    names = set(listed.stdout.split())
    return any(name.startswith(("pad-demo-", "pad-render")) for name in names)


@pytest.fixture(scope="module")
def served() -> Iterator[str]:
    """Start the deployment once and yield its public base URL."""
    if _live_demonstration_running():  # pragma: no cover - environment-dependent
        pytest.skip("a demonstration is already running on this machine")
    name = SERVED_NAME
    _remove(name)
    started = _docker(*_run_arguments(name, SERVED_PORT), timeout=180)
    assert started.returncode == 0, started.stderr
    try:
        _wait_for(f"{SERVED}/healthz")
        yield SERVED
    finally:
        logs = _docker("logs", name, timeout=60)
        if os.environ.get("PAD_KEEP_LOGS"):  # pragma: no cover - debugging aid
            print(logs.stdout, logs.stderr)
        _remove(name)


@pytest.fixture
def lifecycle() -> Iterator[Callable[[], str]]:
    """Yield a factory that starts one throwaway container and returns its name.

    Function-scoped and separate from ``served`` because the tests that use it
    stop, kill and restart the thing they are testing, and a shared container
    would make every later test depend on the order they ran in.
    """
    names: list[str] = []

    def start(*, cpus: str | None = None) -> str:
        name = f"pad-render-life-{len(names)}"
        _remove(name)
        started = _docker(*_run_arguments(name, LIFECYCLE_PORT, cpus=cpus), timeout=180)
        assert started.returncode == 0, started.stderr
        names.append(name)
        _wait_for(f"http://127.0.0.1:{LIFECYCLE_PORT}/healthz")
        return name

    try:
        yield start
    finally:
        for name in names:
            _remove(name)


# ---------------------------------------------------------------------------
# Talking to the deployment
# ---------------------------------------------------------------------------


def _get(url: str, *, timeout: float = 60.0) -> tuple[int, bytes, dict[str, str]]:
    """Return the status, body and headers of a GET, without raising on 4xx/5xx."""
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, answer.read(), dict(answer.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _inside(container: str, script: str, *, timeout: float = 600.0) -> str:
    """Run a Python *script* inside *container* and return its stdout.

    The detection API is bound to loopback inside the container, which is the
    property under test. Reaching it therefore means running there -- exactly as
    the console does.
    """
    completed = subprocess.run(
        ["docker", "exec", "--interactive", container, "python", "-"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


_API_HELPER = """
import json, time, urllib.request, urllib.error
BASE = "http://127.0.0.1:8000"

def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as answer:
            return answer.status, json.load(answer)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")
"""


def _api(container: str, path: str) -> Any:
    """Return the parsed body of an internal API GET performed inside *container*."""
    output = _inside(
        container,
        _API_HELPER + f"\nprint(json.dumps(call({path!r})[1]))\n",
    )
    return json.loads(output.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# The bundle is the one the Compose preparation produces
# ---------------------------------------------------------------------------


def _baked_identity() -> dict[str, Any]:
    """Return what the image's own verifier says about the bundle it carries."""
    completed = _docker(
        "run",
        "--rm",
        "--entrypoint",
        "python",
        RENDER_IMAGE,
        "/app/scripts/verify_serving_bundle.py",
        "--state-root",
        "/srv/state",
        "--json",
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    identity: dict[str, Any] = json.loads(completed.stdout)
    return identity


@pytest.fixture(scope="module")
def baked() -> dict[str, Any]:
    """Return the verified identity of the bundle baked into the image."""
    return _baked_identity()


def test_the_image_carries_a_bundle_that_verifies(baked: dict[str, Any]) -> None:
    """Every seal recomputed, every declared payload digested, before anything runs."""
    assert baked["strategy"] == "stacked"
    assert baked["stacked"] is True
    assert len(baked["manifest_fingerprint"]) == 64
    assert len(baked["stacked_state_fingerprint"]) == 64


@pytest.mark.skipif(
    not _image_present(COMPOSE_IMAGE),
    reason=f"the Compose reference needs {COMPOSE_IMAGE}: run `docker compose build`",
)
def test_the_baked_bundle_is_the_compose_prepared_bundle(
    baked: dict[str, Any],
) -> None:
    """The headline claim of this milestone, proved by doing it.

    Runs ``compose.yaml``'s preparation job from scratch, in its own project so
    it cannot collide with a demonstration, and compares every fingerprint the
    serving manifest carries against the ones inside the Render image.

    Equality here is what makes "the same system, deployed two ways" a
    statement about the artifacts rather than about the intention: the champion,
    its calibration, its operating point, the feature contract it was fitted
    under, the frozen fusion selection and the fitted stacker are each pinned by
    a digest, and the manifest fingerprint covers all of them at once.
    """
    project = "pad-render-reference"
    volume = f"{project}_serving-state"
    try:
        prepared = _docker(
            "compose",
            "--project-name",
            project,
            "--file",
            "compose.yaml",
            "run",
            "--rm",
            "--no-deps",
            "prepare",
            timeout=1800,
        )
        assert prepared.returncode == 0, prepared.stdout + prepared.stderr

        reference = _docker(
            "run",
            "--rm",
            "--volume",
            f"{volume}:/srv/state:ro",
            "--entrypoint",
            "python",
            RENDER_IMAGE,
            "/app/scripts/verify_serving_bundle.py",
            "--state-root",
            "/srv/state",
            "--json",
            timeout=300,
        )
        assert reference.returncode == 0, reference.stderr
        compose_identity = json.loads(reference.stdout)
    finally:
        _docker(
            "compose",
            "--project-name",
            project,
            "--file",
            "compose.yaml",
            "down",
            "--volumes",
            "--remove-orphans",
            timeout=300,
        )

    assert compose_identity == baked


def test_the_served_fingerprint_is_the_baked_fingerprint(
    served: str, baked: dict[str, Any]
) -> None:
    """And what the running service reports is what the image carries.

    A bundle that verified on disk and a service that reported something else
    would mean the verification was checking a file the service does not read.
    """
    status = _api(SERVED_NAME, "/api/v1/system/status")
    assert status["fusion_strategy"] == "stacked"
    assert status["frozen_fusion_strategy"] == "stacked"
    assert status["stacked_state_fingerprint"] == baked["stacked_state_fingerprint"]
    assert status["fusion_unavailable_reason"] is None


def test_the_package_version_is_unchanged(served: str) -> None:
    """A deployment adapter does not rename the system it deploys."""
    _, body, _ = _get(f"{served}/healthz")
    assert json.loads(body)["version"] == "0.5.0"


# ---------------------------------------------------------------------------
# The runtime fits nothing and needs no persistence
# ---------------------------------------------------------------------------


def test_the_runtime_carries_no_training_inputs() -> None:
    """The dataset, the feature snapshots and the reports stayed in the build.

    Their absence is also the proof that the runtime cannot re-run the pipeline:
    there is nothing for it to run the pipeline over.
    """
    listing = _docker(
        "run",
        "--rm",
        "--entrypoint",
        "python",
        RENDER_IMAGE,
        "-c",
        "import pathlib,json;"
        "print(json.dumps(sorted(p.name for p in pathlib.Path('/srv/state').iterdir())))",
        timeout=120,
    )
    assert listing.returncode == 0, listing.stderr
    present = set(json.loads(listing.stdout))
    assert present == {"artifacts", "allowlist.yaml", "prepared.json"}


def test_the_runtime_image_carries_no_preparation_entry_point() -> None:
    """``prepare_demo_bundle.py`` is a build-stage tool and is not shipped."""
    probe = _docker(
        "run",
        "--rm",
        "--entrypoint",
        "python",
        RENDER_IMAGE,
        "-c",
        "import pathlib;print(pathlib.Path('/app/scripts/prepare_demo_bundle.py').exists())",
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False"


def test_the_service_is_ready_without_fitting_anything(served: str) -> None:
    """Every required component available, from artifacts alone."""
    document = _api(SERVED_NAME, "/ready")
    assert document["status"] == "ready"
    states = {item["component"]: item["state"] for item in document["components"]}
    for component in (
        "feature_contract",
        "rule_engine",
        "model_artifacts",
        "ml_champion",
        "fusion",
    ):
        assert states[component] == "ready", (component, states)


def test_the_container_mounts_no_volume_and_no_host_path() -> None:
    """Nothing persists, and nothing from the host is visible.

    Render Free offers no disk, so a deployment that quietly needed one would
    work in every local test and fail the first time it was deployed.
    """
    inspected = _docker(
        "inspect", "--format", "{{json .Mounts}}", SERVED_NAME, timeout=60
    )
    assert inspected.returncode == 0, inspected.stderr
    mounts = json.loads(inspected.stdout)
    assert [m for m in mounts if m.get("Type") not in ("tmpfs",)] == []


def test_the_root_filesystem_is_read_only(served: str) -> None:
    """Including the serving state, which is root-owned and never written."""
    script = """
import pathlib
for target in ("/srv/state/probe", "/app/probe", "/etc/caddy/probe"):
    try:
        pathlib.Path(target).write_text("x")
        print(target, "WRITABLE")
    except OSError as exc:
        print(target, type(exc).__name__)
"""
    lines = [line for line in _inside(SERVED_NAME, script).splitlines() if line.strip()]
    assert len(lines) == 3, lines
    for line in lines:
        assert "WRITABLE" not in line, line
        assert line.split()[-1].endswith("Error"), line


# ---------------------------------------------------------------------------
# The process model
# ---------------------------------------------------------------------------


def test_the_supervisor_is_pid_one(served: str) -> None:
    """So Docker's SIGTERM reaches something that knows how to stop three
    processes, rather than a shell that forwards nothing."""
    output = _inside(
        SERVED_NAME,
        "import pathlib;print(pathlib.Path('/proc/1/cmdline').read_bytes()"
        ".decode().replace(chr(0),' ').strip())",
    )
    assert output.strip().endswith("render_entrypoint.py")


def test_exactly_three_processes_are_supervised(served: str) -> None:
    """The proxy, the API and the console, and nothing else long-lived."""
    script = """
import pathlib
found = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        command = (entry / "cmdline").read_bytes().decode().split(chr(0))
    except OSError:
        continue
    joined = " ".join(part for part in command if part)
    if "uvicorn" in joined:
        found.append("api")
    elif "streamlit" in joined and " run " in joined:
        found.append("dashboard")
    elif joined.startswith("caddy"):
        found.append("proxy")
    elif "render_entrypoint" in joined:
        found.append("supervisor")
print(sorted(found))
"""
    found = _inside(SERVED_NAME, script).strip()
    assert found == "['api', 'dashboard', 'proxy', 'supervisor']", found


def test_only_the_proxy_listens_on_a_public_interface(served: str) -> None:
    """Read out of the kernel's own socket table, not inferred from configuration.

    The API and the console are bound to 127.0.0.1 inside this container's
    network namespace, so nothing outside it can reach either -- which is the
    entire security argument for putting three processes in one container.
    """
    script = """
import pathlib
listening = []
for name in ("tcp", "tcp6"):
    try:
        lines = pathlib.Path(f"/proc/net/{name}").read_text().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        fields = line.split()
        if fields[3] != "0A":
            continue
        address, port = fields[1].rsplit(":", 1)
        listening.append((address, int(port, 16)))
print(sorted(listening))
"""
    listening = ast.literal_eval(_inside(SERVED_NAME, script).strip())
    loopback_v4 = "0100007F"
    bound = {port: address for address, port in listening}
    assert bound[8000] == loopback_v4
    assert bound[8501] == loopback_v4
    assert SERVED_PORT in bound
    assert set(bound) == {8000, 8501, SERVED_PORT}


def test_the_internal_ports_are_not_published_to_the_host(served: str) -> None:
    """Probed directly. One public port, and it is the one Render assigned."""
    for port in (8000, 8501):
        with socket.socket() as probe:
            probe.settimeout(2.0)
            assert probe.connect_ex(("127.0.0.1", port)) != 0, port
    with socket.socket() as probe:
        probe.settimeout(5.0)
        assert probe.connect_ex(("127.0.0.1", SERVED_PORT)) == 0


# ---------------------------------------------------------------------------
# The public routing policy
# ---------------------------------------------------------------------------


def _is_console_shell(body: bytes) -> bool:
    """Return whether a response body is Streamlit's single-page shell."""
    return b"streamlit" in body.lower() and b"<!doctype html" in body.lower()


def test_the_public_root_serves_the_console(served: str) -> None:
    """A viewer arriving at the service gets the analyst console."""
    status, body, _ = _get(f"{served}/")
    assert status == 200
    assert _is_console_shell(body)


def test_the_health_endpoint_is_the_api_and_says_nothing_scientific(
    served: str,
) -> None:
    """Render's check, and the one API route the internet can reach."""
    status, body, headers = _get(f"{served}/healthz")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    document = json.loads(body)
    assert document["status"] == "ok"
    assert set(document) == {"status", "service", "version"}


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/ready",
        "/version",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/rules",
        "/api/v1/model/info",
        "/api/v1/system/status",
        "/api/v1/demo/scenarios",
        "/api/v1/demo/runs",
        "/api/v1/detect",
        "/api/v1/explain",
    ],
)
def test_no_api_route_answers_on_the_public_port(served: str, path: str) -> None:
    """The body, not the status code, is what proves this.

    Streamlit answers 200 with its own shell for any path it does not recognise,
    so every one of these returns 200 whether the routing policy holds or not.
    What must be true is that the answer came from the console and not from the
    detection service.
    """
    _, body, _ = _get(f"{served}{path}")
    assert _is_console_shell(body), path
    assert b'"api_schema_version"' not in body
    assert b"swagger" not in body.lower()


def test_a_post_to_a_detection_route_reaches_no_detector(served: str) -> None:
    """Not even with a well-formed body: there is nothing behind that path."""
    request = urllib.request.Request(
        f"{served}/api/v1/detect",
        data=b'{"events": []}',
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            body = answer.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
    assert b'"api_schema_version"' not in body
    assert b'"detection"' not in body


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Content-Security-Policy", "frame-ancestors 'none'"),
        ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ],
)
def test_every_response_is_hardened(served: str, header: str, expected: str) -> None:
    """The same header set the VPS deployment offers."""
    _, _, headers = _get(f"{served}/")
    assert headers.get(header) == expected


def test_the_proxy_does_not_advertise_what_it_is(served: str) -> None:
    """No ``Server`` header. ``Via`` remains, as it does on the VPS deployment:
    it is a hop-by-hop header the proxy is specified to add."""
    _, _, headers = _get(f"{served}/")
    assert "Server" not in headers


def test_the_admin_api_is_not_listening(served: str) -> None:
    """Caddy's admin endpoint can rewrite the entire routing policy."""
    with socket.socket() as probe:
        probe.settimeout(2.0)
        assert probe.connect_ex(("127.0.0.1", 2019)) != 0


# ---------------------------------------------------------------------------
# The console's websocket survives two proxies
# ---------------------------------------------------------------------------


def _handshake(host: str, origin: str, path: str = "/_stcore/stream") -> str:
    """Perform a raw websocket upgrade and return the response's status line."""
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Origin: {origin}\r\n"
        "X-Forwarded-Proto: https\r\n\r\n"
    )
    with socket.create_connection(("127.0.0.1", SERVED_PORT), timeout=15) as stream:
        stream.sendall(request.encode())
        return stream.recv(4096).decode(errors="replace").splitlines()[0]


def test_the_console_websocket_upgrades_through_the_proxy(served: str) -> None:
    """Without it the console renders once and then never updates."""
    assert (
        _handshake(f"127.0.0.1:{SERVED_PORT}", f"http://127.0.0.1:{SERVED_PORT}")[:12]
        == "HTTP/1.1 101"
    )


def test_the_websocket_survives_tls_termination_at_the_edge(served: str) -> None:
    """The case a local HTTP test would otherwise miss.

    On Render a browser speaks HTTPS to the edge and the edge speaks plain HTTP
    to this container, so Streamlit sees an ``Origin`` whose scheme does not
    match the connection it arrived on. If its origin check rejected that, the
    console would load and then sit disconnected -- and only on the real
    deployment.
    """
    status = _handshake("pad-demo.onrender.com", "https://pad-demo.onrender.com")
    assert status[:12] == "HTTP/1.1 101", status


# ---------------------------------------------------------------------------
# The demonstration itself
# ---------------------------------------------------------------------------


_REPLAY_SCRIPT = (
    _API_HELPER
    + """
_, catalog = call("/api/v1/demo/scenarios")
outcomes = []
for scenario in catalog["scenarios"]:
    code, run = call(
        "/api/v1/demo/runs",
        {"scenario_id": scenario["scenario_id"], "pace": "instant"},
    )
    assert code == 201, (scenario["scenario_id"], code, run)
    while True:
        _, state = call("/api/v1/demo/runs/" + run["run_id"])
        if state["state"] not in ("pending", "running"):
            break
        time.sleep(0.2)
    _, timeline = call("/api/v1/demo/runs/" + run["run_id"] + "/timeline")
    strategies, unavailable = set(), set()
    for record in timeline["records"]:
        hybrid = (record.get("detection") or {}).get("hybrid") or {}
        if hybrid.get("strategy"):
            strategies.add(hybrid["strategy"])
        if hybrid.get("unavailable_reason"):
            unavailable.add(hybrid["unavailable_reason"])
    outcomes.append({
        "scenario_id": scenario["scenario_id"],
        "expected_rule_ids": scenario["expected_rule_ids"],
        "state": state["state"],
        "emitted": state["emitted_count"],
        "event_count": scenario["event_count"],
        "triggered": state["summary"]["triggered_rule_counts"],
        "strategies": sorted(strategies),
        "unavailable": sorted(unavailable),
    })
print(json.dumps(outcomes))
"""
)


@pytest.fixture(scope="module")
def replayed(served: str) -> list[dict[str, Any]]:
    """Run every catalogued scenario once and return what each produced."""
    output = _inside(SERVED_NAME, _REPLAY_SCRIPT, timeout=1200)
    outcomes: list[dict[str, Any]] = json.loads(output.strip().splitlines()[-1])
    return outcomes


def test_every_scenario_completes(replayed: list[dict[str, Any]]) -> None:
    """Including the control, whose expected outcome is that nothing fires."""
    assert len(replayed) == EXPECTED_SCENARIO_COUNT
    for outcome in replayed:
        assert outcome["state"] == "completed", outcome
        assert outcome["emitted"] == outcome["event_count"], outcome


def test_every_scenario_is_fused_by_the_frozen_stacked_hybrid(
    replayed: list[dict[str, Any]], baked: dict[str, Any]
) -> None:
    """No step falls back to another strategy, and none reports one unavailable.

    A degraded deployment is allowed to say so -- that is what
    ``unavailable_reason`` is for -- but it must not silently produce a verdict
    from a strategy nobody selected.
    """
    for outcome in replayed:
        assert outcome["strategies"] == ["stacked"], outcome
        assert outcome["unavailable"] == [], outcome


def test_each_scenario_triggers_the_rules_its_catalogue_entry_claims(
    replayed: list[dict[str, Any]],
) -> None:
    """The catalog publishes an expectation; the deployment has to meet it."""
    for outcome in replayed:
        triggered = set(outcome["triggered"])
        expected = set(outcome["expected_rule_ids"])
        assert expected <= triggered, outcome


@pytest.mark.parametrize(
    "scenario_id",
    ["normal_activity", "brute_force", "password_spraying", "mixed_attack"],
)
def test_the_named_demonstrations_are_present(
    replayed: list[dict[str, Any]], scenario_id: str
) -> None:
    """The four a viewer is most likely to be shown."""
    assert any(outcome["scenario_id"] == scenario_id for outcome in replayed)


def test_the_control_scenario_fires_no_rule(replayed: list[dict[str, Any]]) -> None:
    """A flagged timeline elsewhere only means something if this one is clean."""
    control = next(o for o in replayed if o["scenario_id"] == "normal_activity")
    assert control["triggered"] == {}


def test_an_explanation_is_available_for_a_scored_anchor(served: str) -> None:
    """The model layer decomposes its own decision, and the parts add up."""
    script = (
        _API_HELPER
        + """
import uuid
from datetime import UTC, datetime, timedelta

start = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
events = [
    {
        "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"render-explain-{index}")),
        "event_time": (start + timedelta(seconds=10 * index)).isoformat(),
        "user_id": "u:" + "a1" * 16,
        "source_id": "s:" + "b2" * 16,
        "device_id": "d:" + "c3" * 16,
        "session_id": "sess:" + format(index, "032x"),
        "application_id": "vpn-gateway",
        "authentication_method": "password",
        "authentication_outcome": "failure",
        "failure_reason": "invalid_credentials",
        "country_code": "US",
    }
    for index in range(20)
]
code, explanation = call("/api/v1/explain", {"events": events})
print(json.dumps({"code": code, "explanation": explanation}))
"""
    )
    payload = json.loads(_inside(SERVED_NAME, script).strip().splitlines()[-1])
    assert payload["code"] == 200, payload
    explanation = payload["explanation"]
    assert explanation["available"] is True
    assert explanation["unavailable_reason"] is None
    assert explanation["contributions"]
    assert abs(explanation["reconstruction_residual"]) < 1e-6


def test_the_baseline_dependent_rules_do_not_fire_and_are_not_pretended_to(
    replayed: list[dict[str, Any]],
) -> None:
    """PAD-CS-001 and PAD-ATO-001 need a behavioural baseline the serving path
    does not have, so no scenario may claim them.

    Asserted rather than assumed because "fix the demo by lowering a threshold"
    is exactly the shortcut this would otherwise invite. See
    docs/render-deployment.md and docs/deployment.md §16.
    """
    for outcome in replayed:
        assert "PAD-CS-001" not in outcome["triggered"], outcome
        assert "PAD-ATO-001" not in outcome["triggered"], outcome
        assert "PAD-CS-001" not in outcome["expected_rule_ids"]
        assert "PAD-ATO-001" not in outcome["expected_rule_ids"]


# ---------------------------------------------------------------------------
# The memory budget
# ---------------------------------------------------------------------------


def _cgroup(container: str, name: str) -> str:
    """Return the contents of one cgroup file inside *container*."""
    return _inside(
        container,
        f"import pathlib;print(pathlib.Path('/sys/fs/cgroup/{name}').read_text().strip())",
    ).strip()


def test_the_deployment_never_approached_the_free_tier_ceiling(
    served: str, replayed: list[dict[str, Any]]
) -> None:
    """Read from the container's own cgroup, after the full demonstration has run.

    ``memory.peak`` is the high-water mark the kernel recorded, not a sample --
    a ``docker stats`` figure taken once a second can miss the spike that
    matters. Depends on ``replayed`` so the peak covers the work, not an idle
    process.
    """
    peak_bytes = int(_cgroup(SERVED_NAME, "memory.peak"))
    limit_bytes = 512 * 1024 * 1024
    assert peak_bytes < limit_bytes
    # A margin, not merely a pass: a deployment that peaked at 500 MiB would
    # survive this test and be killed by the first thing that used 12 MiB more.
    assert peak_bytes < limit_bytes * 0.75, f"{peak_bytes / 1048576:.1f} MiB"


def test_nothing_was_killed_for_using_too_much_memory(
    served: str, replayed: list[dict[str, Any]]
) -> None:
    """The counter the kernel keeps, rather than an inference from a survivor."""
    events = dict(
        line.split() for line in _cgroup(SERVED_NAME, "memory.events").splitlines()
    )
    assert events["oom"] == "0", events
    assert events["oom_kill"] == "0", events


def test_the_container_is_still_running_after_the_whole_demonstration(
    served: str, replayed: list[dict[str, Any]]
) -> None:
    """No restart, no crash loop, no silent replacement of a dead process."""
    inspected = _docker(
        "inspect",
        "--format",
        "{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}",
        SERVED_NAME,
        timeout=60,
    )
    assert inspected.stdout.split() == ["true", "false", "0"]


# ---------------------------------------------------------------------------
# Lifecycle: stopping, failing, restarting
# ---------------------------------------------------------------------------


def test_sigterm_stops_every_process_cleanly(
    lifecycle: Callable[..., str],
) -> None:
    """With a replay in flight, which is when a careless shutdown shows.

    The API is given a graceful-shutdown budget so its lifespan runs: active
    replay runs are cancelled and recorded as stopped, rather than a process
    exiting while its last published state still says a run is happening.
    """
    name = lifecycle()
    _inside(
        name,
        _API_HELPER
        + '\ncall("/api/v1/demo/runs", {"scenario_id": "brute_force", "pace": "slow"})\n',
    )
    started = time.monotonic()
    stopped = _docker("stop", "--time", "60", name, timeout=120)
    elapsed = time.monotonic() - started
    assert stopped.returncode == 0, stopped.stderr
    # Well inside Docker's own kill deadline: a container that only stops when
    # the runtime kills it is not stopping, it is being killed.
    assert elapsed < 30.0, elapsed

    inspected = _docker("inspect", "--format", "{{.State.ExitCode}}", name, timeout=60)
    assert inspected.stdout.strip() == str(EXIT_OK)

    logs = _docker("logs", name, timeout=60)
    combined = logs.stdout + logs.stderr
    # Reverse order: the proxy stops accepting before the console it fronts goes
    # away, and the console goes before the API it reads from.
    for marker in ("stopping proxy", "stopping dashboard", "stopping api"):
        assert marker in combined, combined[-2000:]
    assert "Application shutdown complete" in combined


def test_a_child_that_dies_takes_the_container_with_it(
    lifecycle: Callable[..., str],
) -> None:
    """No half-deployment. Render restarts the container; a supervisor that
    quietly restarted one child would leave the service reporting healthy while
    running a combination nobody deployed."""
    name = lifecycle()
    _inside(
        name,
        """
import os, pathlib
for entry in pathlib.Path("/proc").iterdir():
    if entry.name.isdigit():
        try:
            command = (entry / "cmdline").read_bytes().decode()
        except OSError:
            continue
        if "uvicorn" in command:
            os.kill(int(entry.name), 9)
            break
""",
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        inspected = _docker(
            "inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", name
        )
        if inspected.stdout.split()[:1] == ["false"]:
            break
        time.sleep(0.5)
    else:  # pragma: no cover - a failure path
        pytest.fail("the container kept running after a supervised process died")

    assert inspected.stdout.split() == ["false", str(EXIT_CHILD_DIED)]

    with socket.socket() as probe:
        probe.settimeout(2.0)
        assert probe.connect_ex(("127.0.0.1", LIFECYCLE_PORT)) != 0


def test_a_restart_keeps_the_champion_and_forgets_the_replays(
    lifecycle: Callable[..., str], baked: dict[str, Any]
) -> None:
    """The two halves of the ephemeral-filesystem claim, in one test.

    The scientific state is in the image and survives anything. Replay history
    is process memory and survives nothing -- which the console states on the
    page rather than hiding.
    """
    name = lifecycle()
    _inside(
        name,
        _API_HELPER
        + """
code, run = call("/api/v1/demo/runs", {"scenario_id": "brute_force", "pace": "instant"})
assert code == 201, run
while True:
    _, state = call("/api/v1/demo/runs/" + run["run_id"])
    if state["state"] not in ("pending", "running"):
        break
    time.sleep(0.2)
_, runs = call("/api/v1/demo/runs")
print(len(runs["runs"]))
""",
    )
    before = _api(name, "/api/v1/demo/runs")
    assert before["runs"]

    assert _docker("restart", "--time", "60", name, timeout=180).returncode == 0
    _wait_for(f"http://127.0.0.1:{LIFECYCLE_PORT}/healthz")

    after = _api(name, "/api/v1/demo/runs")
    assert after["runs"] == []

    status = _api(name, "/api/v1/system/status")
    assert status["fusion_strategy"] == "stacked"
    assert status["stacked_state_fingerprint"] == baked["stacked_state_fingerprint"]


# ---------------------------------------------------------------------------
# Refusing to start
# ---------------------------------------------------------------------------


def _run_to_completion(*arguments: str, timeout: float = 300.0) -> int:
    """Run the image once, in the foreground, and return its exit code."""
    completed = _docker(
        "run",
        "--rm",
        "--memory",
        MEMORY_LIMIT,
        *arguments,
        RENDER_IMAGE,
        timeout=timeout,
    )
    return completed.returncode


@pytest.mark.parametrize(
    "environment",
    [
        [],
        ["--env", "PORT="],
        ["--env", "PORT=http"],
        ["--env", "PORT=0"],
        ["--env", "PORT=65536"],
    ],
    ids=["unset", "empty", "not-a-number", "zero", "one-past-the-end"],
)
def test_a_missing_or_invalid_port_refuses_to_start(environment: list[str]) -> None:
    """Loudly, rather than by binding something arbitrary and looking healthy."""
    assert _run_to_completion(*environment) == EXIT_BAD_PORT


def test_an_unverifiable_bundle_refuses_to_start() -> None:
    """Simulated by shadowing the baked state with an empty writable mount.

    The important half is that the refusal happens *before* uvicorn is started:
    a service that came up and then reported itself un-ready would look like a
    deployment problem rather than an image that must be rebuilt.
    """
    assert (
        _run_to_completion(
            "--env", f"PORT={LIFECYCLE_PORT}", "--tmpfs", "/srv/state:mode=0755"
        )
        == EXIT_BUNDLE_UNVERIFIED
    )
