"""What the container deployment promises, asserted against the files themselves.

None of this starts a container. Every claim here is one that can be read out of
``Dockerfile``, ``compose.yaml``, ``.dockerignore`` and the tracked demonstration
configurations, and each is a claim that would be expensive to discover was
false: an image running as root, a source tree bind-mounted over the installed
package, a scientific override arriving through the environment, a writable
serving bundle, a base image pinned only by a tag that somebody else can move.

The Docker-requiring half lives in ``tests/integration/test_docker_compose.py``.

Two of these tests deserve a word about *why* they exist:

``test_the_demonstration_rule_configuration_matches_the_integration_fixture``
    The live/replay scenario catalog publishes which rules each scenario is
    expected to trigger, and those expectations are asserted against the
    integration fixture's rule configuration. The container serves
    ``configs/detection/rules-demo.yaml``. If the two drifted apart, the
    container would compute one thing honestly and document another.

``test_the_dockerignore_admits_everything_the_dockerfile_copies``
    A ``.dockerignore`` written as "exclude everything, then re-admit" is only
    safe while somebody remembers to re-admit. A build that silently lost
    ``configs/`` would fail at container start rather than at build time.
"""

from __future__ import annotations

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


def _repository_root() -> Path:
    """Return the repository root, located from this file."""
    return Path(__file__).resolve().parents[3]


ROOT = _repository_root()
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "compose.yaml"

#: The account both runtime targets run as. A fixed number rather than whatever
#: the distribution assigns next, because the named volume's ownership depends
#: on it.
RUNTIME_UID = "10001"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    """Return the Dockerfile's text."""
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    """Return the parsed compose file."""
    document: dict[str, Any] = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return document


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the compose services, with the shared anchors already merged."""
    resolved: dict[str, dict[str, Any]] = compose["services"]
    return resolved


@pytest.fixture(scope="module")
def dockerignore() -> list[str]:
    """Return the non-comment, non-blank lines of ``.dockerignore``."""
    lines = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# ---------------------------------------------------------------------------
# The files exist and are the ones the workflow names
# ---------------------------------------------------------------------------


def test_the_deployment_files_exist() -> None:
    """A one-command workflow needs all three files present at the root."""
    assert DOCKERFILE.is_file()
    assert DOCKERIGNORE.is_file()
    assert COMPOSE.is_file()
    # docker compose discovers `compose.yaml` first; a second file under the
    # older name would make which one runs a question of the tool's version.
    assert not (ROOT / "docker-compose.yml").exists()
    assert not (ROOT / "docker-compose.yaml").exists()


def test_the_package_version_is_unchanged() -> None:
    """Containerization is not a release. Nothing here bumps the version."""
    assert __version__ == "0.5.0"


# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------


def test_the_base_image_is_python_312_and_pinned_by_digest(dockerfile: str) -> None:
    """A tag says what it is; a digest is what actually gets pulled."""
    declared = re.findall(r"^ARG PYTHON_IMAGE=(\S+)$", dockerfile, re.MULTILINE)
    assert len(declared) == 1, "exactly one base image, declared once"
    image = declared[0]
    assert image.startswith("python:3.12")
    assert "@sha256:" in image
    digest = image.split("@sha256:")[1]
    assert re.fullmatch(r"[0-9a-f]{64}", digest), "a full 64-hex digest"


def test_no_stage_uses_a_floating_latest_tag(dockerfile: str) -> None:
    """`latest` is whatever it happened to be on the day somebody built."""
    for line in dockerfile.splitlines():
        if line.startswith("FROM "):
            assert ":latest" not in line, line


def test_every_stage_builds_from_the_one_declared_base(dockerfile: str) -> None:
    """No second base image sneaks in for a runtime stage."""
    external = [
        line.split()[1]
        for line in dockerfile.splitlines()
        if line.startswith("FROM ") and not line.split()[1].startswith("${")
    ]
    internal = {"builder", "runtime"}
    assert set(external) <= internal, f"unexpected external base(s): {external}"


# ---------------------------------------------------------------------------
# The runtime account
# ---------------------------------------------------------------------------


def test_the_runtime_stage_switches_to_an_unprivileged_account(
    dockerfile: str,
) -> None:
    """Root is the default, so not-root has to be stated."""
    assert re.search(r"^USER pad:pad$", dockerfile, re.MULTILINE)
    assert f"APP_UID={RUNTIME_UID}" in dockerfile
    assert "useradd" in dockerfile


def test_nothing_after_the_user_switch_needs_root(dockerfile: str) -> None:
    """A RUN after ``USER`` would either fail or need a privilege we dropped."""
    after = dockerfile.split("USER pad:pad", 1)[1]
    assert not re.search(r"^RUN ", after, re.MULTILINE)


def test_the_runtime_account_cannot_log_in(dockerfile: str) -> None:
    """A service account with a shell is a shell somebody can reach for."""
    assert "--shell /usr/sbin/nologin" in dockerfile


# ---------------------------------------------------------------------------
# What the images contain, and what they must not
# ---------------------------------------------------------------------------


def test_the_runtime_image_installs_no_development_dependencies(
    dockerfile: str,
) -> None:
    """pytest, mypy, ruff and pre-commit have no business in a runtime image."""
    syncs = re.findall(r"^RUN uv sync.*$", dockerfile, re.MULTILINE)
    assert syncs, "the image installs from the lockfile"
    for command in syncs:
        assert "--no-dev" in command, command
        assert "--frozen" in command, command


def test_the_runtime_image_carries_no_source_tree_or_tests(dockerfile: str) -> None:
    """Only the builder sees ``src/``; the runtime gets the installed package."""
    runtime = dockerfile.split("AS runtime", 1)[1]
    assert "COPY src" not in runtime
    assert "tests" not in runtime


def test_no_stage_copies_the_whole_context(dockerfile: str) -> None:
    """``COPY . .`` would make `.dockerignore` the only thing standing between
    an image and every local file that happens to be lying around."""
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if stripped.startswith("COPY") and "--from=" not in stripped:
            parts = stripped.split()
            assert parts[1] != ".", stripped


@pytest.mark.parametrize(
    "forbidden", [".git", "dist", "htmlcov", ".coverage", "tests", "presentations"]
)
def test_no_instruction_names_a_local_private_or_generated_path(
    dockerfile: str, forbidden: str
) -> None:
    """Nothing local, private, or generated is copied in by name.

    Instructions only. The comments in the Dockerfile name some of these
    deliberately -- saying what is *not* in the image is most of what that prose
    is for -- and a check that read them would forbid the explanation.

    This is the weaker half of the guarantee. The strong half is
    ``test_the_dockerfile_copies_only_from_the_admitted_allowlist``: an
    enumeration cannot cover a local file nobody has thought of yet, and an
    allowlist covers all of them.
    """
    instructions = [
        line
        for line in dockerfile.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in instructions:
        assert forbidden not in line, line


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


def test_the_dockerignore_excludes_everything_by_default(
    dockerignore: list[str],
) -> None:
    """The first rule is the one that makes the rest an allowlist."""
    assert dockerignore[0] == "*"


def test_the_dockerignore_admits_everything_the_dockerfile_copies(
    dockerfile: str, dockerignore: list[str]
) -> None:
    """Every COPY source is re-admitted, so no build silently loses an input."""
    admitted = {line[1:] for line in dockerignore if line.startswith("!")}
    sources: set[str] = set()
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if stripped.startswith("COPY") and "--from=" not in stripped:
            # COPY [--chown=...] <src>... <dest>
            parts = [p for p in stripped.split()[1:] if not p.startswith("--")]
            sources.update(parts[:-1])
    assert sources, "the Dockerfile copies something"
    for source in sources:
        # removeprefix, not lstrip: lstrip removes *characters*, so "./.streamlit"
        # would come back as "streamlit" and match nothing.
        top = source.removeprefix("./")
        assert top in admitted or any(
            top.startswith(f"{entry}/") for entry in admitted
        ), f"{source!r} is copied by the Dockerfile but excluded from the context"


@pytest.mark.parametrize(
    "pattern",
    [
        "**/__pycache__",
        "**/.pytest_cache",
        "**/.mypy_cache",
        "**/.ruff_cache",
        "**/.env",
        "**/secrets.toml",
        "**/*.parquet",
    ],
)
def test_the_dockerignore_re_excludes_local_and_generated_content(
    dockerignore: list[str], pattern: str
) -> None:
    """A re-admitted directory must not smuggle a cache or an artifact in."""
    assert pattern in dockerignore


#: Everything either deployment's build may read, enumerated. The last three
#: entries are used only by ``Dockerfile.render``; the VPS image copies none of
#: them, and ``test_the_dockerfile_copies_only_from_the_admitted_allowlist``
#: below checks that this widening did not widen what the VPS image takes.
ADMITTED_CONTEXT = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "src",
        "configs",
        ".streamlit",
        "scripts/prepare_demo_bundle.py",
        "scripts/verify_serving_bundle.py",
        "scripts/render_entrypoint.py",
        "deploy/render/Caddyfile",
    }
)


def test_the_dockerignore_admits_exactly_the_expected_paths(
    dockerignore: list[str],
) -> None:
    """The whole build context, enumerated.

    An equality rather than a list of things to keep out, and that is the point:
    a denylist can only refuse what somebody thought of, while this refuses
    everything else in the working tree -- editor state, local tool
    configuration, scratch files, and anything created after this was written.
    Widening the context is then a visible edit to this set.
    """
    admitted = {line[1:] for line in dockerignore if line.startswith("!")}
    assert admitted == set(ADMITTED_CONTEXT)


def test_the_dockerfile_copies_only_from_the_admitted_allowlist(
    dockerfile: str,
) -> None:
    """And the build uses no more of the context than the allowlist admits."""
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY") or "--from=" in stripped:
            continue
        parts = [p for p in stripped.split()[1:] if not p.startswith("--")]
        for source in parts[:-1]:
            top = source.removeprefix("./")
            assert top in ADMITTED_CONTEXT or any(
                top.startswith(f"{entry}/") for entry in ADMITTED_CONTEXT
            ), stripped


# ---------------------------------------------------------------------------
# Compose: shape
# ---------------------------------------------------------------------------


def test_the_project_declares_its_own_name(compose: dict[str, Any]) -> None:
    """Otherwise the project is named after whatever directory it sits in."""
    assert compose["name"] == "pad-demo"


def test_the_expected_services_exist(services: dict[str, dict[str, Any]]) -> None:
    """Two services and one job, and nothing else."""
    assert set(services) == {"prepare", "api", "dashboard"}


def test_the_startup_order_is_declared_rather_than_hoped_for(
    services: dict[str, dict[str, Any]],
) -> None:
    """The API must not start against a state root a failed job left behind."""
    assert services["api"]["depends_on"] == {
        "prepare": {"condition": "service_completed_successfully"}
    }
    assert services["dashboard"]["depends_on"] == {
        "api": {"condition": "service_healthy"}
    }
    assert "depends_on" not in services["prepare"]


def test_the_preparation_job_does_not_restart(
    services: dict[str, dict[str, Any]],
) -> None:
    """A one-shot job with a restart policy is a loop."""
    assert services["prepare"]["restart"] == "no"


@pytest.mark.parametrize("service", ["api", "dashboard"])
def test_a_served_container_recovers_from_a_crash_but_not_from_a_reboot(
    services: dict[str, dict[str, Any]], service: str
) -> None:
    """``unless-stopped`` would bring a demonstration back up on every Docker
    daemon start. A bounded ``on-failure`` recovers a transient crash and lets a
    persistent one stop and show itself."""
    assert services[service]["restart"] == "on-failure:3"


def test_the_published_ports_are_the_documented_ones(
    services: dict[str, dict[str, Any]],
) -> None:
    """8000 for the API, 8501 for the console, and both on loopback only."""
    assert services["api"]["ports"] == ["127.0.0.1:8000:8000"]
    assert services["dashboard"]["ports"] == ["127.0.0.1:8501:8501"]
    assert "ports" not in services["prepare"], "a job serves nothing"


def test_the_console_reaches_the_api_by_service_name(
    services: dict[str, dict[str, Any]],
) -> None:
    """Inside the network, ``127.0.0.1`` would name the console's own container."""
    url = services["dashboard"]["environment"]["PAD_DASHBOARD_API_URL"]
    assert url == "http://api:8000"


# ---------------------------------------------------------------------------
# Compose: hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service", ["prepare", "api", "dashboard"])
def test_no_service_is_privileged_or_gains_capabilities(
    services: dict[str, dict[str, Any]], service: str
) -> None:
    """The three things that would undo every other boundary here."""
    declared = services[service]
    assert "privileged" not in declared
    assert declared["cap_drop"] == ["ALL"]
    assert "cap_add" not in declared
    assert "no-new-privileges:true" in declared["security_opt"]


@pytest.mark.parametrize("service", ["prepare", "api", "dashboard"])
def test_every_service_runs_on_the_project_network(
    services: dict[str, dict[str, Any]], service: str
) -> None:
    """Host networking would put these processes on the host's interfaces."""
    assert services[service]["networks"] == ["demo"]
    assert "network_mode" not in services[service]


@pytest.mark.parametrize("service", ["prepare", "api", "dashboard"])
def test_every_root_filesystem_is_read_only(
    services: dict[str, dict[str, Any]], service: str
) -> None:
    """Nothing writes into its own image; tmpfs covers what a framework needs."""
    assert services[service]["read_only"] is True
    assert services[service]["tmpfs"], "a read-only root needs somewhere for /tmp"


def test_no_service_mounts_the_docker_socket_or_a_host_path(
    services: dict[str, dict[str, Any]],
) -> None:
    """A container that can reach the daemon is not a container."""
    for name, declared in services.items():
        for mount in declared.get("volumes", []):
            source = str(mount).split(":", 1)[0]
            assert "docker.sock" not in str(mount), name
            assert not source.startswith(("/", ".", "~")), (
                f"{name} bind-mounts a host path: {mount!r}"
            )


def test_the_source_tree_is_not_mounted_over_the_installed_package(
    services: dict[str, dict[str, Any]],
) -> None:
    """A development bind mount would make the image's contents irrelevant."""
    for name, declared in services.items():
        for mount in declared.get("volumes", []):
            assert ":/app" not in str(mount), name


def test_only_the_preparation_job_may_write_the_serving_state(
    services: dict[str, dict[str, Any]],
) -> None:
    """The API reads a frozen decision; it has no business rewriting one."""
    assert services["prepare"]["volumes"] == ["serving-state:/srv/state"]
    assert services["api"]["volumes"] == ["serving-state:/srv/state:ro"]
    assert "volumes" not in services["dashboard"], "the console holds no artifact"


def test_the_state_volume_is_named_rather_than_bound(
    compose: dict[str, Any],
) -> None:
    """A named volume needs no host path, so `up` works on a fresh machine."""
    assert "serving-state" in compose["volumes"]
    assert not compose["volumes"]["serving-state"], "no driver_opts, no device"


def test_every_service_reaps_its_children(
    services: dict[str, dict[str, Any]],
) -> None:
    """``init: true`` gives PID 1 a reaper, so a signal reaches the process."""
    for name, declared in services.items():
        assert declared["init"] is True, name


# ---------------------------------------------------------------------------
# Compose: no scientific control may arrive through the environment
# ---------------------------------------------------------------------------


def _environment_names(services: dict[str, dict[str, Any]]) -> set[str]:
    """Return every environment variable name any service declares."""
    names: set[str] = set()
    for declared in services.values():
        names.update(declared.get("environment", {}))
    return names


def test_no_service_declares_a_scientific_override(
    services: dict[str, dict[str, Any]],
) -> None:
    """The names both settings modules already refuse, refused here too.

    ``APISettings`` would ignore an unknown ``PAD_API_*`` variable rather than
    fail, so a scientific-sounding one in this file would be quietly inert --
    and it would still read, to anybody opening the file, as a control somebody
    could turn.

    Each prefix is checked against *its own* prohibition list. They differ on
    purpose: ``artifact_root`` is a legitimate serving setting -- a location, and
    only a location -- and is forbidden on the console, which holds no artifact
    and must not look as though it could.
    """
    for name in _environment_names(services):
        if name.startswith("PAD_API_"):
            assert name.removeprefix("PAD_API_").lower() not in API_PROHIBITED, name
        if name.startswith("PAD_DASHBOARD_"):
            suffix = name.removeprefix("PAD_DASHBOARD_").lower()
            assert suffix not in DASHBOARD_PROHIBITED, name


@pytest.mark.parametrize(
    "fragment",
    [
        "MODEL",
        "THRESHOLD",
        "FUSION",
        "CHAMPION",
        "CALIBRAT",
        "SCORE",
        "PASSWORD",
        "SECRET",
        "TOKEN",
        "API_KEY",
    ],
)
def test_no_environment_variable_even_sounds_like_a_scientific_control(
    services: dict[str, dict[str, Any]], fragment: str
) -> None:
    """A stricter net than the prohibition list, cast over the file as written."""
    for name in _environment_names(services):
        assert fragment not in name.upper(), name


def test_the_api_is_given_only_locations_and_facility_switches(
    services: dict[str, dict[str, Any]],
) -> None:
    """Every PAD_API_ variable here answers *where* or *how much*, never *which*."""
    assert set(services["api"]["environment"]) == {
        "PAD_API_HOST",
        "PAD_API_PORT",
        "PAD_API_LOG_LEVEL",
        "PAD_API_ENVIRONMENT",
        "PAD_API_DOCS_ENABLED",
        "PAD_API_ARTIFACT_ROOT",
        "PAD_API_ALLOWLIST_PATH",
        "PAD_API_FEATURE_CONFIG_PATH",
        "PAD_API_ML_CONFIG_PATH",
        "PAD_API_DETECTION_CONFIG_PATH",
        "PAD_API_REPLAY_ENABLED",
        "PAD_API_REPLAY_REQUIRED",
    }


def test_every_configured_path_is_absolute(
    services: dict[str, dict[str, Any]],
) -> None:
    """``APISettings`` refuses a relative path; catching it here is cheaper."""
    environment = services["api"]["environment"]
    for name, value in environment.items():
        if name.endswith(("_ROOT", "_PATH")):
            assert str(value).startswith("/"), f"{name}={value!r}"
            assert ".." not in str(value), f"{name}={value!r}"


def test_the_console_is_given_no_artifact_location_at_all(
    services: dict[str, dict[str, Any]],
) -> None:
    """It renders what the API returns. An artifact path would invite more."""
    for name in services["dashboard"].get("environment", {}):
        assert not name.startswith("PAD_API_"), name


# ---------------------------------------------------------------------------
# Compose: commands and healthchecks
# ---------------------------------------------------------------------------


def test_the_api_serves_without_an_auto_reloader(
    dockerfile: str, services: dict[str, dict[str, Any]]
) -> None:
    """``--reload`` watches the filesystem and restarts on a write. In a
    deployment it is a way for a serving process to change underneath itself."""
    assert "--reload" not in dockerfile
    assert "--reload" not in yaml.safe_dump(services)


def test_the_api_command_binds_every_interface_inside_its_namespace(
    dockerfile: str,
) -> None:
    """A container binding 127.0.0.1 is one nothing else can reach.

    Safe here precisely because the namespace *is* the boundary: the published
    port is bound to the host's loopback, so the service is reachable from the
    machine running it and from nowhere else.
    """
    assert '"--host", "0.0.0.0", "--port", "8000"' in dockerfile


def test_every_command_is_declared_in_exec_form(dockerfile: str) -> None:
    """Shell form puts /bin/sh at PID 1, which does not forward SIGTERM -- the
    lifespan shutdown would never run and replay tasks would be killed."""
    commands = re.findall(r"^CMD (.+)$", dockerfile, re.MULTILINE)
    assert len(commands) == 2, "one per runtime target"
    for command in commands:
        assert command.lstrip().startswith("["), command


def test_the_api_shuts_down_gracefully_rather_than_being_killed(
    dockerfile: str,
) -> None:
    """Long enough for the lifespan to cancel active replay runs."""
    assert '"--timeout-graceful-shutdown", "20"' in dockerfile


@pytest.mark.parametrize(
    ("service", "endpoint"),
    [("api", "/health"), ("dashboard", "/_stcore/health")],
)
def test_each_served_container_has_a_healthcheck_on_its_own_endpoint(
    services: dict[str, dict[str, Any]], service: str, endpoint: str
) -> None:
    """And the API's is liveness, not readiness -- see the note in compose.yaml."""
    check = services[service]["healthcheck"]
    assert check["test"][0] == "CMD"
    assert endpoint in yaml.safe_dump(check["test"])
    assert "127.0.0.1" in yaml.safe_dump(check["test"]), "a health check is local"


def test_no_healthcheck_installs_a_binary_to_make_an_http_request(
    services: dict[str, dict[str, Any]], dockerfile: str
) -> None:
    """The image ships a Python interpreter; curl would be a CVE surface."""
    assert "curl" not in dockerfile
    assert "wget" not in dockerfile
    for name in ("api", "dashboard"):
        assert services[name]["healthcheck"]["test"][1] == "python", name


@pytest.mark.parametrize("service", ["api", "dashboard"])
def test_healthcheck_intervals_are_not_aggressive(
    services: dict[str, dict[str, Any]], service: str
) -> None:
    """A health check every second is a load generator with a nice name."""
    check = services[service]["healthcheck"]
    assert check["interval"] == "15s"
    assert check["start_period"] == "30s"


@pytest.mark.parametrize("service", ["prepare", "api", "dashboard"])
def test_every_service_declares_a_memory_ceiling(
    services: dict[str, dict[str, Any]], service: str
) -> None:
    """Sized from measurement; the numbers and how they were taken are in
    docs/docker.md."""
    limits = services[service]["deploy"]["resources"]["limits"]
    assert limits["memory"] in {"1g", "2g"}


def test_the_preparation_job_runs_the_tracked_script(
    services: dict[str, dict[str, Any]],
) -> None:
    """Not an inline shell pipeline: a pipeline in YAML is code nobody reviews."""
    command = services["prepare"]["command"]
    assert command[:2] == ["python", "/app/scripts/prepare_demo_bundle.py"]
    assert "--state-root" in command


# ---------------------------------------------------------------------------
# The demonstration configurations the deployment is defined by
# ---------------------------------------------------------------------------


DEMO_CONFIGS = (
    "configs/data/synthetic-demo.yaml",
    "configs/features/feature-demo.yaml",
    "configs/detection/rules-demo.yaml",
    "configs/ml/model-demo.yaml",
)


@pytest.mark.parametrize("relative", DEMO_CONFIGS)
def test_every_demonstration_configuration_is_tracked_and_loadable(
    relative: str,
) -> None:
    """The deployment is defined by reviewed files, not by generated ones."""
    path = ROOT / relative
    assert path.is_file()
    assert yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_compose_file_names_the_demonstration_configurations(
    services: dict[str, dict[str, Any]],
) -> None:
    """What the API reads at startup is what the preparation step trained under."""
    environment = services["api"]["environment"]
    assert environment["PAD_API_FEATURE_CONFIG_PATH"].endswith("feature-demo.yaml")
    assert environment["PAD_API_ML_CONFIG_PATH"].endswith("model-demo.yaml")
    assert environment["PAD_API_DETECTION_CONFIG_PATH"].endswith("rules-demo.yaml")


def test_the_demonstration_rule_configuration_matches_the_integration_fixture() -> None:
    """The container must serve the configuration the catalog was verified under.

    ``replay/scenarios.py`` publishes an ``expected_rule_ids`` per scenario, and
    ``tests/integration/test_replay_detection.py`` asserts those as an equality
    against the fixture's rule configuration. Serving a different one would make
    the catalog's published expectations describe a deployment nobody runs.
    """
    from tests.integration.ml_workspace import rule_config

    published = yaml.safe_load(
        (ROOT / "configs/detection/rules-demo.yaml").read_text(encoding="utf-8")
    )
    assert published == rule_config()


def test_the_demonstration_model_configuration_excludes_baseline_features() -> None:
    """The serving path fits no behavioural baseline, so a model trained on
    baseline-derived columns would read a constant at serving time and call it
    evidence. See docs/docker.md and the note in the configuration itself."""
    document = yaml.safe_load(
        (ROOT / "configs/ml/model-demo.yaml").read_text(encoding="utf-8")
    )
    classes = document["preprocessing"]["include_leakage_classes"]
    assert "baseline_derived" not in classes
    assert set(classes) == {"prior_only", "current_event_context"}


def test_the_demonstration_dataset_is_synthetic_and_seeded() -> None:
    """Determinism is what lets two machines build the same champion."""
    document = yaml.safe_load(
        (ROOT / "configs/data/synthetic-demo.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(document["seed"], int)
    assert document["start_time"].startswith("2024-")
    assert document["duration_hours"] == 4


# ---------------------------------------------------------------------------
# The preparation script
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepare_module() -> Any:
    """Import the preparation script by path.

    It lives under ``scripts/`` rather than in the package because it is an
    operator action rather than an importable API, and because nothing in the
    serving path may be able to reach a function that fits a model.
    """
    path = ROOT / "scripts" / "prepare_demo_bundle.py"
    spec = importlib.util.spec_from_file_location("_prepare_demo_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_preparation_script_names_only_tracked_configurations(
    prepare_module: Any,
) -> None:
    """Every input it reads is a reviewed file in the repository."""
    named = {
        prepare_module.SYNTHETIC_CONFIG,
        prepare_module.FEATURE_CONFIG,
        prepare_module.RULE_CONFIG,
        prepare_module.ML_CONFIG,
    }
    assert named == set(DEMO_CONFIGS)
    for relative in named:
        assert (ROOT / relative).is_file()


def test_the_preparation_script_can_locate_the_configurations(
    prepare_module: Any,
) -> None:
    """In a checkout it finds them beside the sources; in the image, beside it."""
    assert prepare_module._configuration_root() == ROOT


def test_an_absent_receipt_means_not_prepared(
    prepare_module: Any, tmp_path: Path
) -> None:
    """The check is what makes a second `up` skip a fifty-second pipeline."""
    assert prepare_module.already_prepared(tmp_path) is None


def test_a_receipt_without_its_bundle_is_treated_as_absent(
    prepare_module: Any, tmp_path: Path
) -> None:
    """A state root rebuilt underneath a receipt must not be served."""
    (tmp_path / prepare_module.RECEIPT_FILE).write_text(
        '{"receipt_schema_version": "1.0.0", "bundle_manifest": "gone.json"}',
        encoding="utf-8",
    )
    assert prepare_module.already_prepared(tmp_path) is None


def test_a_receipt_from_another_layout_is_treated_as_absent(
    prepare_module: Any, tmp_path: Path
) -> None:
    """A future change to what a receipt means must not be silently reused."""
    manifest = tmp_path / "bundle.json"
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / prepare_module.RECEIPT_FILE).write_text(
        '{"receipt_schema_version": "0.9.0", "bundle_manifest": "bundle.json"}',
        encoding="utf-8",
    )
    assert prepare_module.already_prepared(tmp_path) is None


def test_an_unreadable_receipt_is_treated_as_absent(
    prepare_module: Any, tmp_path: Path
) -> None:
    """Rebuilding is cheap; serving a half-published bundle is not."""
    (tmp_path / prepare_module.RECEIPT_FILE).write_text("{ not json", encoding="utf-8")
    assert prepare_module.already_prepared(tmp_path) is None


def test_a_complete_receipt_is_honoured(prepare_module: Any, tmp_path: Path) -> None:
    """The positive case, so the three refusals above are not vacuous."""
    manifest = tmp_path / "artifacts" / "serving" / "scope" / "serving_bundle.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / prepare_module.RECEIPT_FILE).write_text(
        '{"receipt_schema_version": "1.0.0", "bundle_manifest": '
        '"artifacts/serving/scope/serving_bundle.json"}',
        encoding="utf-8",
    )
    receipt = prepare_module.already_prepared(tmp_path)
    assert receipt is not None


def test_the_preparation_script_has_no_option_naming_a_scientific_decision(
    prepare_module: Any,
) -> None:
    """It reports what the pipeline chose. It cannot ask for a different answer."""
    source = (ROOT / "scripts" / "prepare_demo_bundle.py").read_text(encoding="utf-8")
    for flag in (
        "--model",
        "--threshold",
        "--fusion",
        "--strategy",
        "--champion",
        "--calibrat",
    ):
        assert f'"{flag}' not in source, flag


def test_the_preparation_script_invokes_the_real_cli(prepare_module: Any) -> None:
    """Not a reimplementation of the pipeline: the pipeline's own entry point."""
    source = (ROOT / "scripts" / "prepare_demo_bundle.py").read_text(encoding="utf-8")
    assert '"-m", "password_attack_detector"' in source
    for stage in (
        '"data",\n            "generate"',
        '"features",\n            "build"',
        '"ml",\n            "train"',
        '"ml",\n            "select"',
        '"deploy",\n            "materialize"',
    ):
        assert stage in source or stage.replace("\n            ", " ") in source


def test_the_receipt_records_no_absolute_path(prepare_module: Any) -> None:
    """A receipt carrying one would publish the layout of the machine that
    built it, into a volume the API then reads."""
    source = (ROOT / "scripts" / "prepare_demo_bundle.py").read_text(encoding="utf-8")
    assert "relative_to(state)" in source
