"""The container deployment, running.

Everything here needs a Docker daemon and the two images already built, so the
whole module is marked ``slow`` and skips itself when either is missing:

    docker compose build
    uv run pytest -m slow tests/integration/test_docker_compose.py --no-cov

What these assert that ``tests/unit/deployment/test_container_contract.py``
cannot: that the declarations in ``compose.yaml`` actually took effect. A file
can say ``read_only: true`` and a daemon can decline to apply it; a service can
be given ``http://api:8000`` and the name can fail to resolve. The unit suite
reads the intent, and this one reads the running container.

The fixture brings the stack up only if it is not already up, and tears down
only what it started -- a test run must not destroy a stack somebody is
demonstrating from.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8000"
DASHBOARD = "http://127.0.0.1:8501"
IMAGES = ("pad-demo-api:0.5.0", "pad-demo-dashboard:0.5.0")


def _docker(
    *arguments: str, timeout: float = 300.0
) -> subprocess.CompletedProcess[str]:
    """Run a docker command from the repository root."""
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


def _images_built() -> bool:
    """Return whether both runtime images are present locally."""
    listed = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=30)
    if listed.returncode != 0:
        return False
    present = set(listed.stdout.split())
    return all(image in present for image in IMAGES)


if not _daemon_available():  # pragma: no cover - environment-dependent
    pytest.skip("no Docker daemon is reachable", allow_module_level=True)
if not _images_built():  # pragma: no cover - environment-dependent
    pytest.skip("run `docker compose build` first", allow_module_level=True)


def _running_services() -> set[str]:
    """Return the names of this project's services that are currently up."""
    listed = _docker("compose", "ps", "--format", "json", timeout=60)
    if listed.returncode != 0:
        return set()
    names: set[str] = set()
    for line in listed.stdout.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("State") == "running":
            names.add(str(entry.get("Service")))
    return names


def _get(url: str, timeout: float = 20.0) -> tuple[int, Any]:
    """Return one response's status and decoded body."""
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            body = answer.read()
            status = answer.status
    except urllib.error.HTTPError as answer:  # a 503 readiness report is a document
        body = answer.read()
        status = answer.code
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def _post(path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    """POST JSON to the API and return the status and decoded body."""
    request = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            return answer.status, json.loads(answer.read())
    except urllib.error.HTTPError as answer:
        return answer.code, json.loads(answer.read())


@pytest.fixture(scope="module")
def stack() -> Iterator[None]:
    """Ensure the deployment is up, and leave it as it was found."""
    already = _running_services() >= {"api", "dashboard"}
    if not already:
        started = _docker("compose", "up", "-d", "--wait", timeout=900)
        if started.returncode != 0:  # pragma: no cover - reported, not asserted
            pytest.skip(f"could not start the stack: {started.stderr[-500:]}")
    yield
    if not already:
        _docker("compose", "down", "-v", timeout=300)


# ---------------------------------------------------------------------------
# The declarations in compose.yaml actually took effect
# ---------------------------------------------------------------------------


def _inspect(container: str, template: str) -> str:
    """Return one formatted field from a container's inspection."""
    result = _docker("inspect", container, "--format", template, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("container", ["pad-demo-api", "pad-demo-dashboard"])
def test_the_application_runs_as_an_unprivileged_account(
    stack: None, container: str
) -> None:
    """Read out of the running process, not out of the Dockerfile."""
    result = _docker("compose", "exec", "-T", container.removeprefix("pad-demo-"), "id")
    assert result.returncode == 0, result.stderr
    assert "uid=10001(pad)" in result.stdout
    assert "uid=0(root)" not in result.stdout


@pytest.mark.parametrize("container", ["pad-demo-api", "pad-demo-dashboard"])
def test_no_container_is_privileged_or_on_the_host_network(
    stack: None, container: str
) -> None:
    """The three settings that would undo every other boundary."""
    assert _inspect(container, "{{.HostConfig.Privileged}}") == "false"
    assert _inspect(container, "{{.HostConfig.NetworkMode}}") != "host"
    assert _inspect(container, "{{.HostConfig.CapAdd}}") in {"[]", "<no value>"}
    assert "no-new-privileges:true" in _inspect(
        container, "{{.HostConfig.SecurityOpt}}"
    )


@pytest.mark.parametrize("container", ["pad-demo-api", "pad-demo-dashboard"])
def test_every_root_filesystem_is_read_only_in_practice(
    stack: None, container: str
) -> None:
    """A declaration the daemon declined to apply would be worth knowing about."""
    assert _inspect(container, "{{.HostConfig.ReadonlyRootfs}}") == "true"
    service = container.removeprefix("pad-demo-")
    attempt = _docker("compose", "exec", "-T", service, "touch", "/app/probe")
    assert attempt.returncode != 0
    assert "read-only" in (attempt.stderr + attempt.stdout).lower()


def test_the_serving_state_is_mounted_read_only_into_the_api(stack: None) -> None:
    """The API loads a frozen decision. It must not be able to rewrite one."""
    mounts = _inspect(
        "pad-demo-api", "{{range .Mounts}}{{.Destination}}:{{.RW}} {{end}}"
    )
    assert "/srv/state:false" in mounts
    refused = _docker("compose", "exec", "-T", "api", "touch", "/srv/state/probe")
    assert refused.returncode != 0


def test_the_console_holds_no_artifact_mount_at_all(stack: None) -> None:
    """Nothing to read means nothing to be tempted to read."""
    mounts = _inspect("pad-demo-dashboard", "{{range .Mounts}}{{.Destination}} {{end}}")
    assert mounts == ""


def test_no_container_can_reach_the_docker_socket(stack: None) -> None:
    """A container that can reach the daemon is not a container."""
    for service in ("api", "dashboard"):
        probe = _docker(
            "compose", "exec", "-T", service, "test", "-S", "/var/run/docker.sock"
        )
        assert probe.returncode != 0, service


# ---------------------------------------------------------------------------
# The image carries nothing local, private, or secret
# ---------------------------------------------------------------------------


def test_the_image_holds_exactly_four_things_under_app(stack: None) -> None:
    """An enumeration of what IS there, which is stronger than a list of what
    is not: nothing from the working tree can be in the image without appearing
    here, including a local file nobody thought to exclude by name."""
    listed = _docker("compose", "exec", "-T", "api", "ls", "-A", "/app")
    assert listed.returncode == 0, listed.stderr
    assert set(listed.stdout.split()) == {".streamlit", ".venv", "configs", "scripts"}
    scripts = _docker("compose", "exec", "-T", "api", "ls", "-A", "/app/scripts")
    assert set(scripts.stdout.split()) == {"prepare_demo_bundle.py"}


@pytest.mark.parametrize(
    "path",
    [
        "/app/.git",
        "/app/tests",
        "/app/dist",
        "/app/.env",
        "/app/.streamlit/secrets.toml",
    ],
)
def test_no_local_or_private_file_is_in_the_image(stack: None, path: str) -> None:
    """Asserted against the built image rather than against `.dockerignore`."""
    found = _docker("compose", "exec", "-T", "api", "test", "-e", path)
    assert found.returncode != 0, f"{path} is present in the image"


def test_no_ssh_key_or_compiler_is_in_the_image(stack: None) -> None:
    """A runtime image needs neither, and each is a way in or a way out."""
    for binary in ("gcc", "cc", "git", "ssh", "curl"):
        probe = _docker(
            "compose", "exec", "-T", "api", "sh", "-c", f"command -v {binary}"
        )
        assert probe.returncode != 0, f"{binary} is on PATH in the runtime image"


# ---------------------------------------------------------------------------
# The service, running
# ---------------------------------------------------------------------------


def test_health_readiness_and_the_schema_browser_all_answer(stack: None) -> None:
    """The three URLs the workflow tells a reader to open."""
    assert _get(f"{API}/health")[0] == 200
    status, ready = _get(f"{API}/ready")
    assert status == 200
    assert ready["status"] == "ready"
    assert {component["component"] for component in ready["components"]} >= {
        "feature_contract",
        "rule_engine",
        "model_artifacts",
        "ml_champion",
        "fusion",
        "replay",
    }
    assert _get(f"{API}/docs")[0] == 200


def test_the_console_answers_on_its_own_port(stack: None) -> None:
    """Both the health endpoint the container check uses and the page itself."""
    assert _get(f"{DASHBOARD}/_stcore/health")[0] == 200
    assert _get(DASHBOARD)[0] == 200


def test_the_deployment_serves_the_frozen_stacked_hybrid(stack: None) -> None:
    """The interesting case: the strategy that needs a materialized artifact."""
    _, status = _get(f"{API}/api/v1/system/status")
    assert status["fusion_strategy"] == "stacked"
    assert status["frozen_fusion_strategy"] == "stacked"
    assert status["fusion_unavailable_reason"] is None
    assert len(status["stacked_state_fingerprint"]) == 64
    assert status["hybrid_detection_enabled"] is True


def test_the_model_layer_reports_a_real_frozen_champion(stack: None) -> None:
    """Not a stand-in, and not a family the deployment chose at startup."""
    _, info = _get(f"{API}/api/v1/model/info")
    assert info["available"] is True
    assert info["model_family"] == "logistic_regression"
    assert info["score_kind"] == "calibrated_probability"


def test_the_console_reaches_the_api_by_docker_service_name(stack: None) -> None:
    """Driven through the console's own client, inside the console's container.

    Anything less would prove the API is reachable from the *host*, which is a
    different network and not the one the console uses.
    """
    probe = _docker(
        "compose",
        "exec",
        "-T",
        "dashboard",
        "python",
        "-c",
        (
            "from password_attack_detector.dashboard.config import "
            "load_dashboard_settings;"
            "from password_attack_detector.dashboard.api_client import "
            "DashboardAPIClient;"
            "s = load_dashboard_settings();"
            "r = DashboardAPIClient(s).system_status();"
            "print(s.api_url, r.ok, r.document.fusion_strategy)"
        ),
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.split() == ["http://api:8000", "True", "stacked"]


# ---------------------------------------------------------------------------
# Replay, through the containerized detection path
# ---------------------------------------------------------------------------


def _run_to_completion(scenario_id: str, pace: str = "instant") -> list[dict[str, Any]]:
    """Start one replay run and return its records once nothing more will arrive."""
    status, run = _post("/api/v1/demo/runs", {"scenario_id": scenario_id, "pace": pace})
    assert status == 201, run
    identifier = run["run_id"]
    deadline = time.monotonic() + 120.0
    page: dict[str, Any] = {"records": [], "more_expected": True}
    while time.monotonic() < deadline:
        _, page = _get(f"{API}/api/v1/demo/runs/{identifier}/timeline?limit=100")
        if not page["more_expected"]:
            break
        time.sleep(0.05)
    assert not page["more_expected"], f"{scenario_id} did not finish"
    return list(page["records"])


def test_the_scenario_catalog_is_served(stack: None) -> None:
    """Seven reviewed scenarios, and no way to add an eighth over the wire."""
    status, catalog = _get(f"{API}/api/v1/demo/scenarios")
    assert status == 200
    assert catalog["scenario_count"] == 7


@pytest.mark.parametrize(
    "scenario_id",
    [
        "normal_activity",
        "brute_force",
        "password_spraying",
        "credential_stuffing",
        "account_takeover",
        "bot_activity",
        "mixed_attack",
    ],
)
def test_every_scenario_triggers_exactly_the_rules_it_publishes(
    stack: None, scenario_id: str
) -> None:
    """An equality, not a superset.

    The catalog states what each scenario is expected to trigger, and the
    container has to be a deployment that description is true of -- including
    ``normal_activity``, whose published expectation is that nothing fires.
    """
    _, catalog = _get(f"{API}/api/v1/demo/scenarios")
    published = next(
        entry for entry in catalog["scenarios"] if entry["scenario_id"] == scenario_id
    )
    records = _run_to_completion(scenario_id)
    assert len(records) == published["event_count"]
    fired = {
        rule
        for record in records
        for rule in record["detection"]["rule"]["fired_rule_ids"]
    }
    assert sorted(fired) == sorted(published["expected_rule_ids"])


def test_every_replayed_step_is_fused_by_the_frozen_stacked_hybrid(
    stack: None,
) -> None:
    """No fallback anywhere: a substituted gate is unconstructible by design,
    and this is the check that the design held in a container."""
    records = _run_to_completion("mixed_attack")
    strategies = {record["detection"]["hybrid"]["strategy"] for record in records}
    assert strategies == {"stacked"}
    assert all(record["detection"]["hybrid"]["available"] for record in records)
    assert all(
        record["detection"]["hybrid"]["unavailable_reason"] is None
        for record in records
    )


def test_a_running_replay_can_be_stopped_and_emits_nothing_afterwards(
    stack: None,
) -> None:
    """Stop is an acknowledgement, so nothing may arrive after it returns."""
    _, run = _post("/api/v1/demo/runs", {"scenario_id": "brute_force", "pace": "slow"})
    identifier = run["run_id"]
    time.sleep(2.0)
    status, stopped = _post(f"/api/v1/demo/runs/{identifier}/stop", {})
    assert status == 200
    assert stopped["state"] == "stopped"
    _, first = _get(f"{API}/api/v1/demo/runs/{identifier}/timeline?limit=100")
    time.sleep(3.0)
    _, second = _get(f"{API}/api/v1/demo/runs/{identifier}/timeline?limit=100")
    assert len(second["records"]) == len(first["records"])
    assert second["more_expected"] is False
    again = _post(f"/api/v1/demo/runs/{identifier}/stop", {})
    assert again[0] == 200, "stop is idempotent"


# ---------------------------------------------------------------------------
# Restart: the science survives, the demo history does not
# ---------------------------------------------------------------------------


def test_restarting_the_api_preserves_the_science_and_clears_the_replay_history(
    stack: None,
) -> None:
    """Both halves matter, and the second is the one worth being explicit about.

    The serving bundle is on a read-only volume and comes back byte-identical.
    The replay store is process memory and does not come back at all -- which is
    a property the console states on the page rather than one it hides.
    """
    _run_to_completion("bot_activity")
    _, before_status = _get(f"{API}/api/v1/system/status")
    _, before_model = _get(f"{API}/api/v1/model/info")
    _, before_runs = _get(f"{API}/api/v1/demo/runs")
    assert before_runs["run_count"] > 0

    restarted = _docker("compose", "restart", "api", timeout=300)
    assert restarted.returncode == 0, restarted.stderr

    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            if _get(f"{API}/health", timeout=3)[0] == 200:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1.0)
    else:  # pragma: no cover - the API failed to come back
        pytest.fail("the API did not answer after a restart")

    _, after_status = _get(f"{API}/api/v1/system/status")
    _, after_model = _get(f"{API}/api/v1/model/info")
    _, after_runs = _get(f"{API}/api/v1/demo/runs")

    assert after_status == before_status
    assert after_model == before_model
    assert after_runs["run_count"] == 0


# ---------------------------------------------------------------------------
# The refusals still refuse, over a real socket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"scenario_id": "brute_force", "pace": "normal", "password": "x"},
        {"scenario_id": "brute_force", "pace": "normal", "model_id": "M-002"},
        {"scenario_id": "brute_force", "pace": "normal", "decision_threshold": 0.1},
        {"scenario_id": "brute_force", "pace": "normal", "fusion_strategy": "or_gate"},
        {"scenario_id": "brute_force", "pace": "normal", "artifact_root": "/srv"},
        {"scenario_id": "brute_force", "pace": "normal", "target_url": "http://x"},
        {"scenario_id": "../../etc/passwd", "pace": "normal"},
        {"scenario_id": "brute_force", "pace": "0.001"},
    ],
)
def test_the_replay_endpoint_refuses_every_shape_it_should(
    stack: None, body: dict[str, Any]
) -> None:
    """Credentials, overrides, paths, external targets, arbitrary timing."""
    status, answer = _post("/api/v1/demo/runs", body)
    assert status in {404, 422}, answer
    assert "error" in answer or "code" in json.dumps(answer)


def test_no_response_carries_a_filesystem_path_or_a_traceback(stack: None) -> None:
    """A refusal must not describe the machine it was refused on."""
    _, answer = _post(
        "/api/v1/demo/runs", {"scenario_id": "brute_force", "artifact_root": "/srv"}
    )
    rendered = json.dumps(answer)
    assert "/srv" not in rendered
    assert "/app" not in rendered
    assert "Traceback" not in rendered


def test_the_api_log_carries_no_traceback_and_no_host_path(stack: None) -> None:
    """Container logs are the operator's window; a traceback in one is a leak."""
    logs = _docker("compose", "logs", "--no-color", "api", timeout=120)
    assert logs.returncode == 0
    assert "Traceback (most recent call last)" not in logs.stdout
    assert str(ROOT) not in logs.stdout
