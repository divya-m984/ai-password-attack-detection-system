"""Replay orchestration: the operations the API namespace is a thin shell over.

Every function here takes the resolved replay runtime and returns a wire
document or raises an :class:`~password_attack_detector.api.errors.APIError` with
a stable code.  Nothing below scores anything, computes a feature, reads a
threshold, or fuses two layers -- the engine's detector does all of that by
calling the *same* serving orchestration ``POST /api/v1/detect`` calls, and the
binding for it lives in :mod:`password_attack_detector.api.services` beside the
function it binds.

Two decisions worth stating.

**A missing run and an unavailable subsystem are different codes.**  A run this
process never had, a run that finished and was evicted under the retention
bound, and a deployment with replay switched off are three different things an
operator would fix three different ways, so they are three different codes rather
than one 404.

**The catalog is the whole input surface.**  There is no code path from a request
to an event definition, a schedule, an address, a path, or a scientific
parameter: :func:`start_run` takes a scenario identifier, looks it up in the
reviewed catalog, and refuses anything that is not there.
"""

from __future__ import annotations

from password_attack_detector.api.errors import APIError, ErrorCode
from password_attack_detector.exceptions import ReplayCapacityError, ReplayStateError
from password_attack_detector.logging_config import get_logger
from password_attack_detector.replay.engine import ReplayRuntime
from password_attack_detector.replay.scenarios import (
    SCENARIO_SCHEMA_VERSION,
    SCENARIOS,
    Scenario,
    scenario,
)
from password_attack_detector.replay.schemas import (
    CreateReplayRunRequest,
    ReplayRunListResponse,
    ReplayRunResponse,
    ReplayTimelineResponse,
    ScenarioCatalogResponse,
    ScenarioSummary,
)
from password_attack_detector.replay.store import ReplayRun, summarize

__all__ = [
    "DEFAULT_TIMELINE_LIMIT",
    "get_run",
    "get_timeline",
    "list_runs",
    "run_document",
    "scenario_catalog_document",
    "start_run",
    "stop_run",
]

_log = get_logger(__name__)

#: How many timeline records one poll returns when the caller does not say. Sized
#: so a client that has fallen a few steps behind catches up in one request, and
#: a client polling steadily always gets everything new in one.
DEFAULT_TIMELINE_LIMIT: int = 50


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


def _scenario_summary(item: Scenario) -> ScenarioSummary:
    """Return one catalog entry's public description."""
    return ScenarioSummary(
        scenario_id=item.scenario_id,
        name=item.name,
        description=item.description,
        purpose=item.purpose,
        scenario_schema_version=SCENARIO_SCHEMA_VERSION,
        revision=item.revision,
        scenario_fingerprint=item.fingerprint(),
        event_count=item.event_count,
        duration_seconds=item.duration_seconds,
        expected_rule_ids=item.expected_rule_ids,
        expected_rule_families=item.expected_rule_families,
        expected_severity_at_least=item.expected_severity_at_least,
        demonstrates_ml=item.demonstrates_ml,
        demonstrates_hybrid=item.demonstrates_hybrid,
        limitations=item.limitations,
    )


def scenario_catalog_document() -> ScenarioCatalogResponse:
    """Return the built-in scenario catalog.

    Needs no runtime: the catalog is a property of the build, and publishing it
    on a deployment whose replay subsystem is switched off is useful rather than
    misleading -- it tells a client what *would* be runnable.
    """
    return ScenarioCatalogResponse(
        scenario_schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_count=len(SCENARIOS),
        scenarios=tuple(_scenario_summary(item) for item in SCENARIOS),
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def run_document(run: ReplayRun) -> ReplayRunResponse:
    """Return one run's public state and the summary of what it produced."""
    return ReplayRunResponse(
        run_id=run.run_id,
        scenario_id=run.scenario_id,
        scenario_name=run.scenario_name,
        scenario_revision=run.scenario_revision,
        scenario_fingerprint=run.scenario_fingerprint,
        pace=run.pace,
        state=run.state,
        event_count=run.event_count,
        emitted_count=run.emitted_count,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        next_sequence=run.next_sequence,
        more_expected=not run.terminal,
        failure_reason=run.failure_reason,
        summary=summarize(run.records),
    )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _require_available(replay: ReplayRuntime | None) -> ReplayRuntime:
    """Return the replay runtime, or refuse because this deployment has none."""
    if replay is None or not replay.available:
        reason = (
            "replay_disabled"
            if replay is None or not replay.enabled
            else replay.unavailable_reason or "replay_unavailable"
        )
        raise APIError(ErrorCode.REPLAY_UNAVAILABLE, detail={"reason": reason})
    return replay


async def _require_run(replay: ReplayRuntime, run_id: str) -> ReplayRun:
    """Return a stable snapshot of one run, or refuse because there is none."""
    run = await replay.store.snapshot(run_id)
    if run is None:
        raise APIError(ErrorCode.REPLAY_RUN_NOT_FOUND)
    return run


async def start_run(
    replay: ReplayRuntime | None, request: CreateReplayRunRequest
) -> ReplayRunResponse:
    """Register and start one built-in scenario.

    Raises:
        APIError: ``API016`` when replay is unavailable, ``API017`` when the
            scenario is not in the catalog, ``API019`` when the active-run
            ceiling is reached, and ``API020`` when the run could not be started
            -- which, for a freshly created run, means something raced it.
    """
    resolved = _require_available(replay)
    found = scenario(str(request.scenario_id))
    if found is None:  # pragma: no cover - the enum admits only catalog members
        raise APIError(ErrorCode.REPLAY_SCENARIO_NOT_FOUND)

    try:
        run = await resolved.store.create(
            scenario_id=found.scenario_id,
            scenario_name=found.name,
            scenario_revision=found.revision,
            scenario_fingerprint=found.fingerprint(),
            pace=request.pace,
            event_count=found.event_count,
        )
    except ReplayCapacityError:
        raise APIError(
            ErrorCode.REPLAY_LIMIT_REACHED,
            detail={"max_active_runs": resolved.store.limits.max_active_runs},
        ) from None

    assert resolved.engine is not None  # guaranteed by ``available``
    try:
        started = await resolved.engine.start(run, found)
    except ReplayStateError:
        raise APIError(ErrorCode.REPLAY_INVALID_TRANSITION) from None
    _log.info(
        "replay run started",
        run_id=started.run_id,
        scenario_id=str(started.scenario_id),
        pace=str(started.pace),
    )
    return run_document(await _require_run(resolved, started.run_id))


async def get_run(replay: ReplayRuntime | None, run_id: str) -> ReplayRunResponse:
    """Return one run's current state and summary."""
    resolved = _require_available(replay)
    return run_document(await _require_run(resolved, run_id))


async def list_runs(replay: ReplayRuntime | None) -> ReplayRunListResponse:
    """Return every run this process retains, newest first.

    The runs are returned without their records: a listing that inlined every
    timeline would grow without bound in exactly the situation -- several runs
    retained -- where a client is least likely to want them.
    """
    resolved = _require_available(replay)
    runs = await resolved.store.all_runs()
    limits = resolved.store.limits
    return ReplayRunListResponse(
        run_count=len(runs),
        active_run_count=sum(1 for item in runs if not item.terminal),
        max_active_runs=limits.max_active_runs,
        max_retained_runs=limits.max_retained_runs,
        runs=tuple(run_document(item) for item in runs),
    )


async def get_timeline(
    replay: ReplayRuntime | None,
    run_id: str,
    *,
    after_sequence: int = 0,
    limit: int = DEFAULT_TIMELINE_LIMIT,
) -> ReplayTimelineResponse:
    """Return one bounded page of a run's timeline, after *after_sequence*.

    Incremental polling is the whole point of the cursor: a client passes back
    the ``next_sequence`` it was last given and receives only what it has not
    seen, so a long run does not re-transmit its entire timeline on every poll.

    Raises:
        APIError: ``API021`` when the cursor is negative, ``API018`` when there
            is no such run, ``API016`` when replay is unavailable.
    """
    resolved = _require_available(replay)
    if after_sequence < 0:
        raise APIError(
            ErrorCode.REPLAY_CURSOR_INVALID, detail={"after_sequence": after_sequence}
        )
    run = await _require_run(resolved, run_id)
    page = await resolved.store.timeline(
        run_id, after_sequence=after_sequence, limit=limit
    )
    if page is None:  # pragma: no cover - the snapshot above already found it
        raise APIError(ErrorCode.REPLAY_RUN_NOT_FOUND)
    records, more = page
    return ReplayTimelineResponse(
        run_id=run.run_id,
        scenario_id=run.scenario_id,
        state=run.state,
        after_sequence=after_sequence,
        # The cursor advances to the last record delivered, and stays put when
        # a page is empty. Advancing it past a gap would silently skip records
        # that arrive between two polls.
        next_sequence=records[-1].sequence if records else after_sequence,
        record_count=len(records),
        records=records,
        more_expected=more,
        emitted_count=run.emitted_count,
        event_count=run.event_count,
    )


async def stop_run(replay: ReplayRuntime | None, run_id: str) -> ReplayRunResponse:
    """Stop one run, idempotently, and return it once it is terminal.

    Stopping a run that has already finished is a request that succeeded: it
    reports the state the run is in and changes nothing. Only the addressed run
    is affected -- every other run keeps its own task, its own stop event, and
    its own timeline.
    """
    resolved = _require_available(replay)
    await _require_run(resolved, run_id)
    assert resolved.engine is not None  # guaranteed by ``available``
    stopped = await resolved.engine.stop(run_id)
    if stopped is None:  # pragma: no cover - the snapshot above already found it
        raise APIError(ErrorCode.REPLAY_RUN_NOT_FOUND)
    _log.info("replay run stopped", run_id=run_id, state=str(stopped.state))
    return run_document(await _require_run(resolved, run_id))


def _assert_the_replay_layer_computes_no_verdict() -> None:
    """Fail at import if this module acquires a detection capability.

    The replay layer decides *when* something is scored and never *what the
    score is*.  A scorer, a rule engine, a preprocessor, a calibrator or a fusion
    function reachable from here would make "replay duplicates no scientific
    logic" a convention rather than a fact -- and the first scenario that wanted
    a number the serving layer does not return would compute one.
    """
    import sys

    forbidden = {
        "DetectionEngine",
        "FeatureEngine",
        "FrozenChampion",
        "RiskScorer",
        "StackedFusionState",
        "apply_frozen_binary_decision",
        "fuse",
        "local_contributions",
        "predict_serving_binary",
        "quantize",
    }
    offending = sorted(forbidden & set(vars(sys.modules[__name__])))
    if offending:
        raise ValueError(
            f"{__name__} imported {offending}; the replay layer schedules "
            f"detections and must not be able to produce one itself"
        )


_assert_the_replay_layer_computes_no_verdict()
