"""The replay layer's boundaries, asserted rather than trusted.

A demonstration facility is the part of a security system most likely to acquire
a convenience that undoes it: an "upload your own scenario" field, a "point it at
this host" parameter, a "try this threshold" slider.  Each would be one small
edit, and each would turn a synthetic demo into something else entirely.

So the properties are enforced structurally and asserted from outside:

* **No external reach.**  No replay module imports a network client, and no
  request field can name a host, a URL, or a path.
* **No scientific override.**  The create-run contract has two fields, both
  closed vocabularies, and ``extra="forbid"``.
* **No credential, anywhere.**  Not in the scenarios, not in the request, not in
  the response.
* **No filesystem.**  Nothing in the package opens, spawns, or evaluates.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.replay.scenarios import SCENARIOS
from password_attack_detector.replay.schemas import CreateReplayRunRequest

# ---------------------------------------------------------------------------
# The module boundary
# ---------------------------------------------------------------------------


def _replay_root() -> Path:
    """Return the replay package directory."""
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "password_attack_detector"
        / "replay"
    )


def _modules() -> list[Path]:
    """Return every module in the replay package."""
    found = sorted(_replay_root().rglob("*.py"))
    assert found, "the replay package has no modules"
    return found


def _imported_names(path: Path) -> set[str]:
    """Return every module name *path* imports, however it imports it.

    Walks the syntax tree rather than the text: this package's docstrings name
    forbidden modules constantly, explaining why they are not imported, and a
    grep cannot tell a mention from an import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


#: Modules that could open a connection to something outside this process.
#: ``asyncio`` is deliberately absent: replay *schedules* with it, which is the
#: opposite of dialling out with it.
NETWORK_CAPABLE_IMPORTS = frozenset(
    {
        "aiohttp",
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
def test_no_replay_module_imports_a_network_client(path: Path) -> None:
    """A replay run reaches nothing outside the process it runs in."""
    for imported in _imported_names(path):
        assert imported not in NETWORK_CAPABLE_IMPORTS, (path.name, imported)
        assert imported.split(".")[0] not in {
            "httpx",
            "requests",
            "urllib3",
            "socket",
            "aiohttp",
        }, (path.name, imported)


@pytest.mark.parametrize("path", _modules(), ids=lambda item: item.name)
def test_no_replay_module_spawns_a_process_or_evaluates_source(path: Path) -> None:
    """Scenario data is data. There is nothing here that could execute one."""
    text = path.read_text(encoding="utf-8")
    for token in ("subprocess", "os.system", "shell=True", "eval(", "exec(", "pickle"):
        assert token not in text, (path.name, token)


@pytest.mark.parametrize("path", _modules(), ids=lambda item: item.name)
def test_no_replay_module_reads_or_writes_the_filesystem(path: Path) -> None:
    """The store is in memory, and the catalog is source. Neither opens a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in ("open", "write_text", "write_bytes", "read_text", "read_bytes"):
        assert forbidden not in called, (path.name, forbidden)


def test_the_replay_layer_imports_no_writer_of_frozen_state() -> None:
    """A demonstration must not be able to change what it is demonstrating."""
    forbidden = {
        "freeze_champion",
        "fit_stacked_fusion",
        "materialize_serving_bundle",
        "select_binary_threshold",
        "select_fusion_strategy",
        "train_model",
        "write_serving_bundle",
    }
    for path in _modules():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        imported = {
            alias.asname or alias.name.split(".")[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom | ast.Import)
            for alias in node.names
        }
        assert not (imported & forbidden), (path.name, imported & forbidden)


def test_the_engine_holds_no_detection_capability() -> None:
    """The detector is injected; the engine cannot construct one."""
    from password_attack_detector.replay import engine

    namespace = vars(engine)
    for forbidden in (
        "DetectionEngine",
        "FeatureEngine",
        "FrozenChampion",
        "RiskScorer",
        "fuse",
        "predict_serving_binary",
    ):
        assert forbidden not in namespace


# ---------------------------------------------------------------------------
# The request surface
# ---------------------------------------------------------------------------


def test_the_create_request_declares_exactly_two_fields() -> None:
    """The whole admissible input surface, enumerated."""
    assert set(CreateReplayRunRequest.model_fields) == {"scenario_id", "pace"}


@pytest.mark.parametrize(
    "extra",
    [
        {"model_id": "some-model"},
        {"decision_threshold": 0.1},
        {"threshold": 0.1},
        {"fusion_strategy": "or_gate"},
        {"artifact_root": "/var/lib/pad"},
        {"serving_bundle_root": "/tmp"},
        {"target_url": "http://example.invalid/login"},
        {"target": "example.invalid"},
        {"source_ip": "203.0.113.9"},
        {"events": [{"event_id": "x"}]},
        {"event_count": 10_000},
        {"pace_seconds": 0.0001},
        {"schedule": "* * * * *"},
        {"path": "../../etc/passwd"},
        {"script": "import os; os.system('id')"},
        {"callback": "http://example.invalid"},
    ],
)
def test_the_create_request_refuses_every_field_it_does_not_declare(
    extra: dict[str, Any],
) -> None:
    """``extra="forbid"``: offering one is a refusal, not a silently ignored key."""
    with pytest.raises(ValueError):
        CreateReplayRunRequest.model_validate(
            {"scenario_id": "normal_activity", "pace": "instant", **extra}
        )


@pytest.mark.parametrize(
    "credential",
    [
        {"password": "hunter2"},
        {"passwordHash": "abc"},
        {"password_hash": "abc"},
        {"secret": "x"},
        {"token": "x"},
        {"access_token": "x"},
        {"refresh_token": "x"},
        {"authorization": "Bearer x"},
        {"cookie": "session=x"},
        {"private_key": "x"},
    ],
)
def test_the_create_request_refuses_credential_material_by_name(
    credential: dict[str, Any],
) -> None:
    """The project's own ingestion scanner, run before anything else.

    These are the spellings ``scan_prohibited_keys`` refuses, so they are claimed
    as ``API013`` -- "this service accepts no credential material" -- rather than
    as an ordinary unexpected key.
    """
    with pytest.raises(ValueError) as caught:
        CreateReplayRunRequest.model_validate(
            {"scenario_id": "normal_activity", **credential}
        )
    assert "[API013]" in str(caught.value)


@pytest.mark.parametrize(
    "credential",
    [{"api_key": "sk-x"}, {"credentials": ["a", "b"]}, {"passphrase": "x"}],
)
def test_a_credential_shaped_name_the_scanner_does_not_list_is_still_refused(
    credential: dict[str, Any],
) -> None:
    """Refused by ``extra="forbid"`` instead, and refused is what matters.

    ``api_key``, ``credentials`` and ``passphrase`` are not in the project's
    prohibited-name list, so they are rejected as unexpected keys rather than
    claimed as ``API013``. The request still does not proceed, and no value is
    read: the two paths differ in which stable code the client is told, not in
    whether the body is admitted.
    """
    with pytest.raises(ValueError) as caught:
        CreateReplayRunRequest.model_validate(
            {"scenario_id": "normal_activity", **credential}
        )
    assert "extra_forbidden" in str(caught.value)


@pytest.mark.parametrize(
    "scenario_id",
    [
        "../../etc/passwd",
        "http://example.invalid/scenario.json",
        "file:///etc/shadow",
        "normal_activity; DROP TABLE runs",
        "NORMAL_ACTIVITY",
        "",
        "a" * 500,
    ],
)
def test_the_scenario_identifier_admits_only_catalog_members(
    scenario_id: str,
) -> None:
    """A name, from a reviewed list. Never a location and never a payload."""
    with pytest.raises(ValueError):
        CreateReplayRunRequest.model_validate({"scenario_id": scenario_id})


# ---------------------------------------------------------------------------
# The endpoints, live
# ---------------------------------------------------------------------------


def test_the_replay_endpoints_reject_a_credential_body(client: Any) -> None:
    """The wire, not just the model."""
    response = client.post(
        "/api/v1/demo/runs",
        json={"scenario_id": "normal_activity", "password": "hunter2"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "API013"
    assert "hunter2" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {"scenario_id": "normal_activity", "password": "correct-horse-battery"},
        {"scenario_id": "normal_activity", "api_key": "correct-horse-battery"},
        {"scenario_id": "correct-horse-battery"},
        {"scenario_id": "normal_activity", "pace": "correct-horse-battery"},
    ],
)
def test_no_submitted_value_is_ever_echoed_back(client: Any, body: Any) -> None:
    """The response is the boundary, and no value crosses it.

    Worth stating where the guarantee actually lives. Pydantic's own
    ``ValidationError`` *string* embeds ``input_value``, and for a
    ``mode="before"`` validator that value is the whole submitted mapping -- so
    the exception text does carry what was sent. What reaches a client does not:
    the application's validation handler reads the error list for two things
    only, the problem count and the code a validator claimed, and renders fixed
    text. Not one byte of a submitted value is forwarded, which is the property
    a caller can rely on.
    """
    response = client.post("/api/v1/demo/runs", json=body)
    assert response.status_code == 422
    assert "correct-horse-battery" not in response.text
    assert "input_value" not in response.text
    assert "Traceback" not in response.text


def test_the_replay_endpoints_reject_a_scientific_override(client: Any) -> None:
    """There is nowhere in a replay request for a threshold to travel."""
    for extra in (
        {"decision_threshold": 0.05},
        {"model_id": "x"},
        {"fusion_strategy": "or_gate"},
        {"artifact_root": "/tmp"},
    ):
        response = client.post(
            "/api/v1/demo/runs", json={"scenario_id": "normal_activity", **extra}
        )
        assert response.status_code == 422, extra
        assert response.json()["error"]["code"] == "API001"


def test_the_replay_endpoints_reject_a_supplied_event_stream(client: Any) -> None:
    """A replay endpoint that ran caller-supplied events would be a work queue."""
    response = client.post(
        "/api/v1/demo/runs",
        json={
            "scenario_id": "normal_activity",
            "events": [{"event_id": "00000000-0000-0000-0000-000000000001"}],
        },
    )
    assert response.status_code == 422


def test_the_replay_endpoints_reject_an_external_target(client: Any) -> None:
    """No field names a host, and no code path dials one."""
    for extra in (
        {"target_url": "http://example.invalid/login"},
        {"callback_url": "http://example.invalid"},
        {"host": "example.invalid"},
    ):
        response = client.post(
            "/api/v1/demo/runs", json={"scenario_id": "brute_force", **extra}
        )
        assert response.status_code == 422, extra


def test_an_oversized_replay_body_is_refused(client: Any) -> None:
    """The service's own body ceiling, not a proxy's."""
    response = client.post(
        "/api/v1/demo/runs",
        json={"scenario_id": "normal_activity", "filler": "x" * 4_000_000},
    )
    assert response.status_code in {413, 422}


#: Tokens that must not appear in *any* replay payload, prose included. Chosen so
#: the sweep survives a catalog that legitimately talks about security: the word
#: "password" appears in a scenario *named* ``password_spraying``, and a rule
#: description may say "credential". What must never appear is a location, a key,
#: or a trace.
_NEVER_ANYWHERE = (
    "/home/",
    "/tmp/",
    "/usr/",
    "/var/",
    "artifact_root",
    "api_key",
    "hmac",
    "pseudonymization_key",
    "traceback",
    "coefficient",
    'file "',
)

#: Additionally forbidden in a *run* or *timeline* payload, which carries no
#: prose at all -- only identifiers, counts, verdicts, and timestamps.
_NEVER_IN_A_RUN = ("password", "secret", "token", "credential")


def test_no_replay_response_carries_a_path_or_a_secret(client: Any) -> None:
    """Aggregate, public-safe values only, all the way through a run."""
    import time

    created = client.post(
        "/api/v1/demo/runs", json={"scenario_id": "normal_activity", "pace": "instant"}
    ).json()
    run_id = created["run_id"]

    for _ in range(400):
        run = client.get(f"/api/v1/demo/runs/{run_id}")
        if not run.json()["more_expected"]:
            break
        time.sleep(0.05)
    timeline = client.get(f"/api/v1/demo/runs/{run_id}/timeline").text
    catalog = client.get("/api/v1/demo/scenarios").text

    for payload in (run.text, timeline, catalog):
        lowered = payload.lower()
        for forbidden in _NEVER_ANYWHERE:
            assert forbidden not in lowered, forbidden

    for payload in (run.text, timeline):
        lowered = payload.lower()
        for forbidden in _NEVER_IN_A_RUN:
            assert forbidden not in lowered, forbidden


def test_the_catalog_prose_is_the_only_place_a_security_word_appears() -> None:
    """And it appears as vocabulary, never as a key or a value.

    Scenario names and descriptions legitimately contain "password" and
    "credential" -- they are the names of attack shapes. What the catalog must
    not contain is a *field* named like a credential, which is a different claim
    and is checked with the project's own scanner.
    """
    from password_attack_detector.data.privacy import scan_prohibited_keys
    from password_attack_detector.replay.service import scenario_catalog_document

    document = scenario_catalog_document().model_dump(mode="json")
    assert scan_prohibited_keys(document) == []


def test_the_built_in_scenarios_carry_no_credential_material() -> None:
    """The catalog is exactly the kind of file somebody pastes a real one into."""
    from password_attack_detector.data.privacy import scan_prohibited_keys

    for scenario in SCENARIOS:
        for event in scenario.events():
            assert scan_prohibited_keys(event) == []


def test_the_built_in_scenarios_name_no_host_or_path() -> None:
    """Pseudonymous sources only: no address, URL, or path reaches the wire."""
    for scenario in SCENARIOS:
        for event in scenario.events():
            assert "source_ip" not in event
            for value in event.values():
                text = str(value)
                assert "://" not in text, text
                assert not text.startswith("/"), text
                assert ".." not in text, text
