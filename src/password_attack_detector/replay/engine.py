"""The replay scheduler: one state machine, one bounded delay, no scoring.

What this module does is decide **when** a step happens and **what state the run
is in**.  What it deliberately does not do is decide anything about the step's
content: the detector is injected, and the only thing the engine knows about it
is that it takes a window and an anchor and returns the serving layer's own
verdict.  That is what keeps "the replay layer duplicates no scoring logic" a
structural fact -- there is no feature, threshold, rule, calibrator or fusion
function reachable from here, and a test asserts the module's namespace.

Three properties this module exists to make true.

**The pace changes presentation and nothing else.**  Each step appends the next
scenario event to the window and scores that window -- so step *n* is always
scored against exactly the *n* events emitted so far, whatever the wall clock
did between them.  The delay is applied *before* the detection, is drawn from a
closed vocabulary, and is bounded.  Replaying the same scenario at ``instant``
and at ``slow`` therefore produces identical verdicts, which
``tests/unit/replay/test_engine.py`` asserts directly.

**A stop takes effect at a step boundary, and the acknowledgement is honest.**
The pace delay is an interruptible wait on a stop event rather than a sleep, so a
stop does not have to wait out a slow pace.  :meth:`ReplayEngine.stop` then
*awaits the run's task* before returning, so by the time a caller sees the
response the run is terminal and no further record can appear.  A record that was
already computed is kept: discarding it would lose a detection that genuinely
happened.

**A failure is a state, not a traceback.**  Anything the detector raises leaves
the run ``failed`` with a stable reason code.  The exception is logged by type
and its message is discarded, exactly as the serving layer's error contract
requires.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from password_attack_detector.api.schemas import AnchorDetection
from password_attack_detector.exceptions import ReplayCapacityError, ReplayStateError
from password_attack_detector.logging_config import get_logger
from password_attack_detector.replay.enums import (
    PACE_INTERVAL_SECONDS,
    ReplayState,
)
from password_attack_detector.replay.scenarios import Scenario
from password_attack_detector.replay.schemas import ReplayTimelineRecord
from password_attack_detector.replay.store import ReplayRun, ReplayStore

__all__ = [
    "DEADLINE_EXCEEDED",
    "DETECTION_FAILED",
    "RECORD_LIMIT_REACHED",
    "SHUTDOWN",
    "STOP_GRACE_SECONDS",
    "ReplayEngine",
    "ReplayRuntime",
    "StepDetector",
]

_log = get_logger(__name__)

#: Stable reason codes a failed or shut-down run reports.  Codes, not messages:
#: an operator greps for one of these, and none of them can carry a path, a
#: value, or a stack frame.
DETECTION_FAILED: Final[str] = "replay_detection_failed"
DEADLINE_EXCEEDED: Final[str] = "replay_deadline_exceeded"
RECORD_LIMIT_REACHED: Final[str] = "replay_record_limit_reached"
SHUTDOWN: Final[str] = "replay_shutdown"

#: How long :meth:`ReplayEngine.stop` waits for a run to finish before cancelling
#: it. Long enough for one in-flight detection to complete and be recorded;
#: short enough that a stop request always answers promptly.
STOP_GRACE_SECONDS: Final[float] = 10.0


class StepDetector(Protocol):
    """What the engine needs from the serving layer, and the whole of it.

    One awaitable that takes the window emitted so far and the anchor to report
    on, and returns the serving layer's own verdict for that anchor.  Declared
    structurally so the engine holds no import of the detection stack: the
    binding lives in :mod:`password_attack_detector.replay.service`.
    """

    async def __call__(
        self, events: Sequence[Mapping[str, Any]], *, anchor_event_id: str
    ) -> AnchorDetection:
        """Score *events* and return the verdict for *anchor_event_id*."""
        ...  # pragma: no cover - a protocol body is never executed


def _now() -> datetime:
    """Return the current instant, timezone-aware."""
    return datetime.now(UTC)


class ReplayEngine:
    """Drives replay runs, one background task each."""

    def __init__(
        self,
        store: ReplayStore,
        *,
        detector: StepDetector,
        pace_scale: float = 1.0,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        """Build an engine over *store*.

        Args:
            store: where runs and their timelines live.
            detector: the serving-layer binding. The engine never constructs one.
            pace_scale: a multiplier on every pace's interval, in ``[0, 1]``.
                One in production. A test sets it near zero to drive the
                ``normal`` and ``slow`` paces through their real code path --
                the interruptible wait, the stop race, the step ordering --
                without spending their wall-clock time. It can only make a run
                *faster*, so no setting of it can extend how long a run holds a
                slot.
            clock: what "now" means for the presentation timestamps.

        Raises:
            ValueError: when *pace_scale* is outside ``[0, 1]``.
        """
        if not 0.0 <= pace_scale <= 1.0:
            raise ValueError(
                f"pace_scale must be in [0, 1], got {pace_scale}; a scale above "
                f"one would let a run outlast the pace vocabulary's own ceiling"
            )
        self._store = store
        self._detector = detector
        self._pace_scale = pace_scale
        self._clock = clock
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stops: dict[str, asyncio.Event] = {}

    @property
    def active_run_ids(self) -> tuple[str, ...]:
        """Return the runs this engine currently has a task for."""
        return tuple(
            sorted(run_id for run_id, task in self._tasks.items() if not task.done())
        )

    # -- lifecycle ----------------------------------------------------------

    async def start(self, run: ReplayRun, scenario: Scenario) -> ReplayRun:
        """Move a run to ``running`` and schedule its background task.

        The transition happens *before* the task is scheduled, so two callers
        racing to start the same run cannot both win: the second one meets the
        state machine, which has no ``running -> running`` edge.

        Raises:
            ReplayStateError: when the run is not in a state it can be started
                from.
        """
        started = await self._store.transition(run.run_id, ReplayState.RUNNING)
        self._stops[run.run_id] = asyncio.Event()
        task = asyncio.create_task(
            self._drive(
                started.run_id,
                scenario,
                PACE_INTERVAL_SECONDS[started.pace] * self._pace_scale,
            ),
            name=f"replay:{started.run_id}",
        )
        self._tasks[started.run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(started.run_id, None))
        return started

    async def stop(self, run_id: str) -> ReplayRun | None:
        """Stop one run and return it once it is terminal.

        Idempotent: stopping an already-finished run changes nothing and reports
        the state it is in. Stopping one that never started moves it straight to
        ``stopped``.

        The wait is what makes the acknowledgement honest -- when this returns,
        the run is terminal and its timeline is final, so a client that polls
        immediately afterwards cannot see a record appear after the stop it was
        told had happened.
        """
        run = await self._store.request_stop(run_id)
        if run is None:
            return None
        event = self._stops.get(run_id)
        if event is not None:
            event.set()
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=STOP_GRACE_SECONDS)
            except TimeoutError:
                # A detection genuinely in flight outlived the grace period. The
                # task is abandoned rather than waited on indefinitely; the run
                # is finished below, and the abandoned step's result is discarded
                # rather than appended to a timeline that has already ended.
                task.cancel()
            except asyncio.CancelledError:  # pragma: no cover - shutdown race
                pass
        return await self._finish(run_id, ReplayState.STOPPED)

    async def shutdown(self) -> None:
        """Cancel every active run and leave each one terminal.

        Called from the application's lifespan.  A process that exited with runs
        still marked ``running`` would leave its last state describing something
        that is not happening; every abandoned run is recorded as stopped with a
        stable shutdown reason instead.
        """
        for event in self._stops.values():
            event.set()
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        # ``return_exceptions`` because a cancelled or already-failed task has
        # nothing left to report -- the run's own state carries the outcome --
        # and re-raising here would abort the rest of the shutdown.
        await asyncio.gather(*tasks, return_exceptions=True)
        for run in await self._store.all_runs():
            if not run.terminal:
                await self._finish(run.run_id, ReplayState.STOPPED)
        self._tasks.clear()
        self._stops.clear()

    async def _finish(self, run_id: str, state: ReplayState) -> ReplayRun | None:
        """Move a run to a terminal state, tolerating one that already is."""
        try:
            return await self._store.transition(run_id, state)
        except ReplayStateError:
            # Already terminal. Which is the outcome the caller wanted.
            return await self._store.get(run_id)

    # -- the run loop -------------------------------------------------------

    async def _drive(self, run_id: str, scenario: Scenario, interval: float) -> None:
        """Emit and score every event in *scenario*, one step at a time.

        Never raises. Every exit is a recorded terminal state, because a
        background task that died with an exception would leave a run reporting
        ``running`` forever.
        """
        events = scenario.events()
        deadline = (
            asyncio.get_running_loop().time() + self._store.limits.max_run_seconds
        )
        window: list[Mapping[str, Any]] = []

        try:
            for index, event in enumerate(events):
                if await self._paused_or_stopped(run_id, interval, first=index == 0):
                    await self._finish(run_id, ReplayState.STOPPED)
                    return
                if asyncio.get_running_loop().time() > deadline:
                    _log.warning("replay run exceeded its deadline", run_id=run_id)
                    await self._fail(run_id, DEADLINE_EXCEEDED)
                    return
                window.append(event)
                record = await self._step(
                    run_id, scenario, index=index, event=event, window=window
                )
                if record is None:
                    return
        except asyncio.CancelledError:
            # Shutdown or a stop that outran its grace period. The terminal
            # transition is performed by whoever cancelled, so nothing is done
            # here beyond letting the cancellation propagate.
            raise
        await self._finish(run_id, ReplayState.COMPLETED)

    async def _paused_or_stopped(
        self, run_id: str, interval: float, *, first: bool
    ) -> bool:
        """Wait out this step's pace delay, and report whether a stop arrived.

        The delay is a race between the interval and the stop event rather than a
        plain sleep, so a stop at ``slow`` pace is acted on immediately instead of
        after up to two seconds. The first step is not delayed: a viewer who
        pressed start wants to see something happen.
        """
        event = self._stops.get(run_id)
        if event is not None and event.is_set():
            return True
        if not first and interval > 0.0 and event is not None:
            try:
                await asyncio.wait_for(event.wait(), timeout=interval)
                return True
            except TimeoutError:
                pass
        return await self._store.stop_requested(run_id)

    async def _step(
        self,
        run_id: str,
        scenario: Scenario,
        *,
        index: int,
        event: Mapping[str, Any],
        window: Sequence[Mapping[str, Any]],
    ) -> ReplayTimelineRecord | None:
        """Score one step and append its record, or finish the run and return None."""
        try:
            anchor = await self._detector(
                window, anchor_event_id=str(event["event_id"])
            )
        except Exception as exc:
            # Deliberately broad, and deliberately not re-raised. This runs in a
            # background task: an exception escaping it would leave the run
            # reporting ``running`` forever with nothing driving it. Every
            # failure mode the detector has -- a refusal from the serving layer,
            # a schema rejection, an unavailable component -- is the same fact
            # from here, which is that this run cannot continue. The exception's
            # *type* is logged and its message is discarded, exactly as the
            # serving layer's error contract requires. ``CancelledError`` is a
            # ``BaseException`` and is therefore not caught here.
            _log.warning(
                "replay step could not be scored",
                run_id=run_id,
                error_type=type(exc).__name__,
            )
            await self._fail(run_id, DETECTION_FAILED)
            return None

        record = ReplayTimelineRecord(
            sequence=index + 1,
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            replay_state=ReplayState.RUNNING,
            event_index=index,
            window_event_count=len(window),
            source_event_time=datetime.fromisoformat(str(event["event_time"])),
            emitted_at=self._clock(),
            authentication_outcome=str(event["authentication_outcome"]),
            detection=anchor,
        )
        try:
            await self._store.append(run_id, record)
        except ReplayStateError:
            # The run reached a terminal state while this step was in flight --
            # a stop, or shutdown. The step is discarded rather than appended to
            # a timeline that has already been declared final.
            return None
        except ReplayCapacityError:
            await self._fail(run_id, RECORD_LIMIT_REACHED)
            return None
        return record

    async def _fail(self, run_id: str, reason: str) -> None:
        """Move a run to ``failed`` with a stable reason, tolerating a race.

        A run that is already terminal stays as it is: a stop that landed first
        is the outcome the operator asked for, and overwriting it with a failure
        would report a fault where there was an instruction.
        """
        with suppress(ReplayStateError):
            await self._store.transition(
                run_id, ReplayState.FAILED, failure_reason=reason
            )


@dataclass(slots=True)
class ReplayRuntime:
    """What one process resolved for the optional replay subsystem.

    Mutable in exactly one field and exactly once.  :attr:`engine` is attached
    after the serving runtime exists, because the engine's detector is bound to
    that runtime and the runtime carries this object -- so one of the two has to
    be completed after the other.  Doing it this way keeps the serving runtime
    frozen and keeps the detector bound to the *final* runtime rather than to a
    partially-assembled copy whose readiness could differ.
    """

    store: ReplayStore
    #: Whether this deployment offers replay at all.
    enabled: bool
    #: Whether overall readiness depends on it.  False by default: replay is a
    #: demonstration facility, and a detection service that refused to serve
    #: because its demo history could not initialise would have its priorities
    #: backwards.
    required: bool
    #: Stable reason code when replay is not available.
    unavailable_reason: str | None = None
    engine: ReplayEngine | None = None

    @property
    def available(self) -> bool:
        """Return whether a replay run can actually be started."""
        return self.enabled and self.engine is not None
