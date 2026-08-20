"""The synthetic live/replay demonstration endpoints.

Thin, like every other route module here: each handler validates nothing itself,
decides nothing itself, and calls exactly one typed service function.

These handlers are ``async def`` where the detection routes are ``def``, and that
is not incidental.  A replay run is a background task on the event loop, the
store is guarded by an ``asyncio`` lock, and a stop *awaits* the run's task before
answering -- so these handlers have to be on the loop rather than in the
threadpool.  The detection work each step performs is dispatched back off the
loop by the binding in
:func:`~password_attack_detector.api.services.build_replay_detector`, so a
demonstration never stalls the process it is running in.

**Nothing here reaches outside this process.**  There is no field on any request
below for an address, a URL, a filesystem path, an event definition, a schedule,
a model, a threshold, or a fusion strategy.  The only inputs are a scenario name
from the reviewed catalog, a pace from a four-word vocabulary, an opaque run
identifier, and a cursor.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query, status

from password_attack_detector.api.dependencies import ReadyRuntime, Runtime
from password_attack_detector.api.errors import ErrorResponse
from password_attack_detector.replay.schemas import (
    CreateReplayRunRequest,
    ReplayRunListResponse,
    ReplayRunResponse,
    ReplayTimelineResponse,
    ScenarioCatalogResponse,
)
from password_attack_detector.replay.service import (
    DEFAULT_TIMELINE_LIMIT,
    get_run,
    get_timeline,
    list_runs,
    scenario_catalog_document,
    start_run,
    stop_run,
)
from password_attack_detector.replay.store import DEFAULT_LIMITS

__all__ = ["replay_router"]

replay_router = APIRouter(prefix="/api/v1/demo", tags=["Demo"])

#: The shape a run identifier may take.  Opaque, bounded, and alphanumeric: a
#: value that could be read as a path segment, a traversal, or a query is refused
#: by the schema before any handler sees it.
RunId = Annotated[
    str,
    Path(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Opaque, process-local run identifier.",
    ),
]

_REPLAY_FAILURES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No such run in this process, or no such scenario.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "The replay subsystem is not available on this deployment.",
    },
}


@replay_router.get(
    "/scenarios",
    response_model=ScenarioCatalogResponse,
    summary="The built-in synthetic scenario catalog",
    description=(
        "Reports every scenario that can be replayed, with its content "
        "fingerprint, event count, simulated duration, and what replaying it has "
        "been shown to demonstrate. Scenarios cannot be uploaded or "
        "parameterised: this catalog is the whole admissible input surface. "
        "Answered even where replay is switched off, so a client can see what "
        "would be runnable."
    ),
)
async def scenarios() -> ScenarioCatalogResponse:
    """Return the reviewed scenario catalog."""
    return scenario_catalog_document()


@replay_router.post(
    "/runs",
    response_model=ReplayRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start one built-in scenario",
    description=(
        "Registers a run and begins emitting the scenario's events one at a "
        "time into the same serving orchestration POST /api/v1/detect uses. The "
        "request accepts a scenario identifier and a pace, and nothing else: "
        "there is no field for an event, an address, a path, a model, a "
        "threshold, or a fusion strategy. The pace affects presentation timing "
        "only -- the scenario's event times and every verdict are identical at "
        "every pace. Runs are process-local, bounded, and not retained across a "
        "restart."
    ),
    responses={
        **_REPLAY_FAILURES,
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The run could not be started from the state it is in.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "The request is malformed, or names an unknown scenario.",
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": (
                "The concurrent-run ceiling is reached. No existing run was "
                "stopped to make room."
            ),
        },
    },
)
async def create_run(
    request: CreateReplayRunRequest, runtime: ReadyRuntime
) -> ReplayRunResponse:
    """Start one scenario and return the run that was registered.

    Gated on a *ready* runtime deliberately: a replay against a runtime that
    cannot serve detection would fail on its first step, and refusing up front
    with the readiness code says what is actually wrong.
    """
    return await start_run(runtime.replay, request)


@replay_router.get(
    "/runs",
    response_model=ReplayRunListResponse,
    summary="Runs this process is retaining",
    description=(
        "Reports the runs held in this process's memory, newest first, without "
        "their timelines. This is not a history: it is a bounded window over one "
        "process, cleared by a restart, from which finished runs are evicted "
        "oldest-first once the retention bound is reached."
    ),
    responses=_REPLAY_FAILURES,
)
async def runs(runtime: Runtime) -> ReplayRunListResponse:
    """Return every retained run."""
    return await list_runs(runtime.replay)


@replay_router.get(
    "/runs/{run_id}",
    response_model=ReplayRunResponse,
    summary="One run's state and summary",
    description=(
        "Reports a run's lifecycle state, progress, timeline cursor, and the "
        "aggregate summary derived from the records it produced."
    ),
    responses=_REPLAY_FAILURES,
)
async def run(run_id: RunId, runtime: Runtime) -> ReplayRunResponse:
    """Return one run."""
    return await get_run(runtime.replay, run_id)


@replay_router.get(
    "/runs/{run_id}/timeline",
    response_model=ReplayTimelineResponse,
    summary="One bounded page of a run's timeline",
    description=(
        "Returns the records after `after_sequence`, bounded by `limit`. A "
        "client polls with the `next_sequence` it was last given and receives "
        "only what it has not seen, so a long run does not re-transmit its whole "
        "timeline on every poll. `more_expected` is false once the run has "
        "reached a terminal state and this page carried everything left; that is "
        "the signal to stop polling."
    ),
    responses={
        **_REPLAY_FAILURES,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "The cursor is not a position a cursor can occupy.",
        },
    },
)
async def timeline(
    run_id: RunId,
    runtime: Runtime,
    after_sequence: Annotated[
        int,
        Query(
            description=(
                "Return only records with a strictly greater sequence. Zero, the "
                "default, returns from the beginning."
            )
        ),
    ] = 0,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=DEFAULT_LIMITS.max_timeline_page,
            description="Maximum records to return in this page.",
        ),
    ] = DEFAULT_TIMELINE_LIMIT,
) -> ReplayTimelineResponse:
    """Return one page of a run's timeline."""
    return await get_timeline(
        runtime.replay, run_id, after_sequence=after_sequence, limit=limit
    )


@replay_router.post(
    "/runs/{run_id}/stop",
    response_model=ReplayRunResponse,
    summary="Stop one run",
    description=(
        "Stops the addressed run and no other. Idempotent: stopping a run that "
        "has already finished reports the state it is in and changes nothing. "
        "Records already emitted are preserved, and by the time this responds "
        "the run is terminal -- so no further record can appear after the stop "
        "was acknowledged. Nothing is killed at the process level."
    ),
    responses=_REPLAY_FAILURES,
)
async def stop(run_id: RunId, runtime: Runtime) -> ReplayRunResponse:
    """Stop one run and return its final state."""
    return await stop_run(runtime.replay, run_id)
