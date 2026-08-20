"""The one genuinely frozen deployment every serving suite runs against.

Building it means running the real pipeline -- publish a feature dataset, train,
select, freeze, predict, detect, evaluate, materialize -- which takes long enough
that doing it per module would be the slowest thing in the suite.  So it is
session-scoped and shared: the detection suite, the attribution suite, and the
dashboard's integration suite all serve the *same* artifacts, which is a
stronger property than each building its own, because a document one suite reads
is then provably the document another suite scored against.

Nothing here is hand-assembled.  A serving test that passed against a hand-built
artifact would be testing a shape the commands never produce.

**Why the pipeline helpers are imported inside the fixture.**  A ``conftest.py``
is imported during *collection*, which is before ``pytest_configure`` -- and
``pytest_configure`` in ``tests/conftest.py`` is where the terminal-colour scrub
happens, because ``rich.Console`` latches its colour system in ``__init__`` and
the CLI builds its consoles at import time.  ``tests.integration.ml_workspace``
imports the CLI.  Importing it at the top of this file would therefore latch an
inherited ``FORCE_COLOR`` before the scrub could run, and
``test_cli_color_isolation.py`` catches exactly that.  The FastAPI imports below
build no console and are safe where they are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from password_attack_detector.api.app import create_app
from password_attack_detector.api.config import APISettings
from password_attack_detector.api.services import build_runtime


@pytest.fixture(scope="session")
def served(tmp_path_factory: pytest.TempPathFactory) -> APISettings:
    """Run the whole Phase 5 pipeline once, then materialize its hybrid.

    The pipeline's own selection on this fixture is ``stacked``, which is the
    interesting case: it is the strategy that needs a fitted artifact, and the
    one a serving layer cannot execute from a receipt alone.  So the fixture ends
    with ``deploy materialize``, and every suite that depends on it runs against
    a deployment whose hybrid is the reconstructed meta-learner Phase 5 fitted.

    ``ML_CONFIG`` is the testing configuration rather than the development one:
    the champion is frozen under it, and the development config resolves a
    different feature contract, which the runtime correctly refuses.
    """
    # Imported here rather than at module scope; see the module docstring.
    from tests.integration.ml_workspace import (
        ML_CONFIG,
        build_workspace,
        detect,
        evaluate,
        freeze,
        materialize,
        predict,
        prediction_ids,
    )

    base = tmp_path_factory.mktemp("serving")
    (base / "workspace").mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(base / "workspace")
    root = base / "artifacts"
    reports = base / "reports"
    freeze(workspace, root, reports)

    for split in ("train", "validation", "test"):
        scored = predict(workspace, root, split=split)
        assert scored.exit_code == 0, scored.output

    detection = base / "detection"
    assessed = detect(workspace, detection)
    assert assessed.exit_code == 0, assessed.output

    ids = prediction_ids(root)
    evaluated = evaluate(
        workspace,
        root,
        detection,
        reports,
        **{
            "--prediction": ids["test"],
            "--validation-prediction": ids["validation"],
        },
    )
    assert evaluated.exit_code == 0, evaluated.output

    published = materialize(
        workspace, root, detection, validation_prediction=ids["validation"]
    )
    assert published.exit_code == 0, published.output

    return APISettings(
        artifact_root=root,
        allowlist_path=workspace / "allowlist.yaml",
        feature_config_path=workspace / "features.yaml",
        ml_config_path=Path(ML_CONFIG),
        detection_config_path=detection / "rules.yaml",
    )


@pytest.fixture()
def client(served: APISettings) -> Any:
    """A client bound to the frozen champion the pipeline produced."""
    with TestClient(create_app(settings=served)) as connected:
        yield connected


class ServingTransport(httpx.BaseTransport):
    """A synchronous transport that dispatches into the real ASGI application.

    ``httpx.ASGITransport`` is async-only, and the dashboard's client is
    deliberately synchronous -- Streamlit renders synchronously, and an async
    client under it would buy nothing but an event loop to manage.  So requests
    are handed to Starlette's ``TestClient``, which owns the portal that runs the
    ASGI app, and its response is handed back.

    The point of doing this rather than mocking: every dashboard assertion runs
    against bytes the actual serving application produced.

    **The client must be entered as a context manager.**  A ``TestClient`` that
    has not been entered creates a *fresh event loop per request* and tears it
    down when the response is returned -- which is invisible for a request/
    response endpoint and fatal for a background task.  A replay run started
    through such a client is scheduled on a loop that stops existing a moment
    later, so the run stays at ``running`` with nothing driving it, forever.
    Under uvicorn there is one loop for the life of the process and the question
    does not arise; the fixture below enters the client so the test harness
    matches.
    """

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Dispatch one request into the application and return its response."""
        answered = self._client.request(
            request.method,
            str(request.url),
            content=request.read(),
            headers=dict(request.headers),
        )
        return httpx.Response(
            answered.status_code,
            headers=answered.headers,
            content=answered.content,
            request=request,
        )


@pytest.fixture()
def serving_transport(served: APISettings) -> Any:
    """A transport into the frozen deployment, for the dashboard's own client.

    The runtime is built and injected rather than resolved from scratch, because
    a dashboard talking to an application whose artifacts were never loaded would
    exercise the "not ready" path on every assertion. The lifespan still runs --
    it adopts the injected runtime -- which is what gives this client the single
    persistent event loop a replay run needs to make progress, and what runs the
    engine's shutdown when the test ends.

    Function-scoped on purpose: each test gets its own replay store, so one
    test's runs are invisible to the next and the bounded-capacity assertions
    mean something.
    """
    application = create_app(settings=served, runtime=build_runtime(served))
    with TestClient(application) as connected:
        yield ServingTransport(connected)
