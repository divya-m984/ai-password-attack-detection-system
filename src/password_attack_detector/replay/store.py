"""Bounded, process-local storage for demonstration replay runs.

**This is not persistence, and nothing here pretends otherwise.**  Every run and
every timeline record lives in one process's memory.  Restarting the API clears
all of it; two API processes behind a load balancer would not see each other's
runs; nothing is written to disk, to a database, or to a queue.  A replay run is
a demonstration somebody is watching, not a record anybody should later rely on,
and the API documents and the dashboard both say so where a viewer can read it.

**Every bound is a refusal, never an eviction of something live.**  Reaching the
active-run ceiling refuses the new run; it does not stop somebody else's
demonstration to make room.  Only runs that have already finished are evicted,
oldest first, and only to keep the retained history bounded.  That asymmetry is
the point: a store that silently killed the run you were watching to serve a
request you did not make would be worse than one that said no.

**One lock, held only around mutations.**  The engine appends from a background
task while the API reads timelines from request handlers, and both run on the
same event loop.  The lock makes "read a consistent snapshot while another run is
writing" a property rather than a hope, and every method below either takes it or
operates on a copy that was taken under it.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from password_attack_detector.detection.enums import SEVERITY_ORDER, Severity
from password_attack_detector.exceptions import ReplayCapacityError, ReplayStateError
from password_attack_detector.replay.enums import (
    TERMINAL_STATES,
    ReplayPace,
    ReplayState,
    ScenarioId,
    can_transition,
)
from password_attack_detector.replay.schemas import (
    ReplayRunSummary,
    ReplayTimelineRecord,
)

__all__ = [
    "DEFAULT_LIMITS",
    "ReplayLimits",
    "ReplayRun",
    "ReplayStore",
    "summarize",
]


@dataclass(frozen=True, slots=True)
class ReplayLimits:
    """What one process will spend on demonstration replay.

    Deliberately conservative.  Replay is an optional demonstration facility
    sharing a process with a detection service, and every one of these numbers
    is chosen so that a caller hammering the replay endpoints cannot degrade the
    thing the service is actually for.
    """

    #: How many runs may be in flight at once. Each holds a background task and
    #: re-scores a growing window on every step, so this is the number that
    #: bounds CPU.
    max_active_runs: int = 4
    #: How many runs are retained in total, finished ones included. Oldest
    #: *finished* runs are dropped first; an active run is never evicted.
    max_retained_runs: int = 24
    #: Records one run may accumulate. A backstop: the catalog's largest
    #: scenario is far below it, so reaching this means something is wrong.
    max_records_per_run: int = 256
    #: Wall-clock ceiling on one run. A run that outlives it is abandoned with a
    #: stable reason rather than left holding a slot.
    max_run_seconds: float = 300.0
    #: The largest timeline page one request may draw.
    max_timeline_page: int = 100

    def __post_init__(self) -> None:
        """Refuse a limit set that does not actually limit anything."""
        for name in (
            "max_active_runs",
            "max_retained_runs",
            "max_records_per_run",
            "max_timeline_page",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.max_run_seconds <= 0.0:
            raise ValueError("max_run_seconds must be positive")
        if self.max_retained_runs < self.max_active_runs:
            raise ValueError(
                "max_retained_runs must be at least max_active_runs, or an active "
                "run would have to be evicted to admit another"
            )


#: The bounds a deployment gets unless it says otherwise.
DEFAULT_LIMITS: Final[ReplayLimits] = ReplayLimits()


@dataclass
class ReplayRun:
    """One demonstration run: its identity, its lifecycle, and what it produced.

    Mutable, and mutated only through :class:`ReplayStore` under its lock.  The
    two identities are separate on purpose -- see
    :class:`~password_attack_detector.replay.schemas.ReplayRunResponse` for what
    each one means.
    """

    run_id: str
    scenario_id: ScenarioId
    scenario_name: str
    scenario_revision: int
    scenario_fingerprint: str
    pace: ReplayPace
    event_count: int
    created_at: datetime
    state: ReplayState = ReplayState.CREATED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    records: list[ReplayTimelineRecord] = field(default_factory=list)
    failure_reason: str | None = None
    #: Set by a stop request. The engine checks it between steps, so a stop takes
    #: effect at a step boundary rather than mid-detection -- which is what makes
    #: "no record is emitted after the stop is acknowledged" true without
    #: abandoning a half-finished one.
    stop_requested: bool = False

    @property
    def emitted_count(self) -> int:
        """Return how many steps have been emitted and scored."""
        return len(self.records)

    @property
    def next_sequence(self) -> int:
        """Return the cursor a client should poll from to receive only new records."""
        return self.records[-1].sequence if self.records else 0

    @property
    def terminal(self) -> bool:
        """Return whether this run has finished and will emit nothing further."""
        return self.state in TERMINAL_STATES


def summarize(records: Sequence[ReplayTimelineRecord]) -> ReplayRunSummary:
    """Return the aggregate view of one run's timeline.

    Derived entirely from the records the run produced.  A layer that produced no
    verdict is counted as unavailable rather than as an unflagged negative: an
    absent model has not "declined to flag" anything, and collapsing the two
    would make a degraded deployment look like a quiet one.
    """
    severity_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    strategies: list[str] = []
    rule_flagged = ml_flagged = ml_absent = hybrid_flagged = hybrid_absent = 0
    worst: Severity | None = None

    for item in records:
        anchor = item.detection
        severity_counts[str(anchor.severity)] = (
            severity_counts.get(str(anchor.severity), 0) + 1
        )
        if worst is None or SEVERITY_ORDER[anchor.severity] > SEVERITY_ORDER[worst]:
            worst = anchor.severity
        if anchor.rule.flagged:
            rule_flagged += 1
        for rule_id in anchor.rule.fired_rule_ids:
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        if anchor.ml.available:
            ml_flagged += 1 if anchor.ml.flagged else 0
        else:
            ml_absent += 1
        if anchor.hybrid.available:
            hybrid_flagged += 1 if anchor.hybrid.flagged else 0
            if anchor.hybrid.strategy is not None:
                strategies.append(str(anchor.hybrid.strategy))
        else:
            hybrid_absent += 1

    return ReplayRunSummary(
        detection_count=len(records),
        rule_flagged_count=rule_flagged,
        ml_flagged_count=ml_flagged,
        ml_unavailable_count=ml_absent,
        hybrid_flagged_count=hybrid_flagged,
        hybrid_unavailable_count=hybrid_absent,
        highest_severity=None if worst is None else str(worst),
        severity_counts={
            name: severity_counts[name]
            for name in (str(member) for member in Severity)
            if name in severity_counts
        },
        triggered_rule_counts=dict(
            sorted(rule_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        fusion_strategies=tuple(sorted(set(strategies))),
    )


def _now() -> datetime:
    """Return the current instant, timezone-aware."""
    return datetime.now(UTC)


class ReplayStore:
    """Every demonstration run this process knows about, and nothing more."""

    def __init__(
        self,
        limits: ReplayLimits = DEFAULT_LIMITS,
        *,
        clock: Callable[[], datetime] = _now,
        identity: Callable[[], str] | None = None,
    ) -> None:
        """Build an empty store.

        Args:
            limits: the bounds this store enforces.
            clock: what "now" means. Injectable so a test can state exactly when
                a run was created and finished rather than asserting around a
                real clock.
            identity: how a run identifier is minted. Defaults to a random
                opaque token; a test may supply a deterministic one. Whatever it
                returns must carry nothing about the host -- no path, no process
                identifier, no port, no user, no secret.
        """
        self._limits = limits
        self._clock = clock
        self._identity = identity if identity is not None else self._mint
        self._runs: dict[str, ReplayRun] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def _mint() -> str:
        """Return an opaque run identifier.

        A random token rather than a counter: a counter would publish how many
        demonstrations this process has served, which is a small fact about the
        deployment that nothing needs.
        """
        return f"run_{uuid.uuid4().hex}"

    @property
    def limits(self) -> ReplayLimits:
        """Return the bounds this store enforces."""
        return self._limits

    # -- lifecycle ----------------------------------------------------------

    async def create(
        self,
        *,
        scenario_id: ScenarioId,
        scenario_name: str,
        scenario_revision: int,
        scenario_fingerprint: str,
        pace: ReplayPace,
        event_count: int,
    ) -> ReplayRun:
        """Register a new run in :attr:`ReplayState.CREATED`.

        Raises:
            ReplayCapacityError: when the active-run ceiling is already reached.
                Refused rather than absorbed: the alternative is stopping a run
                somebody is watching to start one they did not ask for.
        """
        async with self._lock:
            active = sum(1 for item in self._runs.values() if not item.terminal)
            if active >= self._limits.max_active_runs:
                raise ReplayCapacityError(
                    f"this deployment runs at most {self._limits.max_active_runs} "
                    f"replay runs at once"
                )
            self._evict_finished()
            run = ReplayRun(
                run_id=self._identity(),
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                scenario_revision=scenario_revision,
                scenario_fingerprint=scenario_fingerprint,
                pace=pace,
                event_count=event_count,
                created_at=self._clock(),
            )
            self._runs[run.run_id] = run
            self._order.append(run.run_id)
            return run

    def _evict_finished(self) -> None:
        """Drop the oldest finished runs until the retention bound is satisfied.

        Called with the lock held.  Deterministic: oldest first, by creation
        order, and only runs that have already reached a terminal state.  If
        every retained run is still active the store simply stays at its bound --
        the active ceiling above is what stops it growing further.
        """
        while len(self._runs) >= self._limits.max_retained_runs:
            victim = next(
                (
                    run_id
                    for run_id in self._order
                    if run_id in self._runs and self._runs[run_id].terminal
                ),
                None,
            )
            if victim is None:
                return
            del self._runs[victim]
            self._order.remove(victim)

    async def get(self, run_id: str) -> ReplayRun | None:
        """Return the run, or ``None`` when this process has no such run.

        The returned object is the live one.  Callers that need a stable view
        while other runs advance use :meth:`snapshot`.
        """
        async with self._lock:
            return self._runs.get(run_id)

    async def snapshot(self, run_id: str) -> ReplayRun | None:
        """Return a copy of one run whose record list cannot change underneath it."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            copied = ReplayRun(**{**vars(run), "records": list(run.records)})
            return copied

    async def all_runs(self) -> tuple[ReplayRun, ...]:
        """Return a snapshot of every retained run, newest first."""
        async with self._lock:
            return tuple(
                ReplayRun(**{**vars(self._runs[run_id]), "records": []})
                for run_id in reversed(self._order)
                if run_id in self._runs
            )

    # -- transitions --------------------------------------------------------

    async def transition(
        self, run_id: str, target: ReplayState, *, failure_reason: str | None = None
    ) -> ReplayRun:
        """Move a run to *target*, or refuse because the lifecycle forbids it.

        Raises:
            ReplayStateError: when the run is unknown, or when the edge is not in
                the transition graph -- which is how a completed run is prevented
                from restarting and a stopped one from completing.
        """
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ReplayStateError(f"no replay run {run_id!r} in this process")
            if not can_transition(run.state, target):
                raise ReplayStateError(
                    f"a replay run cannot move from {run.state!s} to {target!s}"
                )
            if (target is ReplayState.FAILED) != (failure_reason is not None):
                raise ReplayStateError(
                    "a failed run names a stable reason code, and only a failed "
                    "one does"
                )
            run.state = target
            if target is ReplayState.RUNNING:
                run.started_at = self._clock()
            if target in TERMINAL_STATES:
                run.finished_at = self._clock()
                run.failure_reason = failure_reason
            return run

    async def request_stop(self, run_id: str) -> ReplayRun | None:
        """Mark a run for stopping, idempotently, and return it.

        Returns the run unchanged when it has already finished: "stop a stopped
        run" is a request that succeeded, not one that failed. The engine sees
        the flag at the next step boundary and finishes the run itself, so a
        record that was already being computed is still recorded.
        """
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            if not run.terminal:
                run.stop_requested = True
            return run

    async def stop_requested(self, run_id: str) -> bool:
        """Return whether a stop has been requested for this run."""
        async with self._lock:
            run = self._runs.get(run_id)
            return run is not None and run.stop_requested

    # -- records ------------------------------------------------------------

    async def append(self, run_id: str, record: ReplayTimelineRecord) -> None:
        """Append one timeline record to a run.

        Raises:
            ReplayStateError: when the run is unknown or has already finished. A
                terminal run's timeline is final; appending to one would let a
                stopped run keep producing rows after its stop was acknowledged.
            ReplayCapacityError: when the per-run record bound is reached.
        """
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ReplayStateError(f"no replay run {run_id!r} in this process")
            if run.terminal:
                raise ReplayStateError(
                    f"replay run {run_id!r} is {run.state!s} and its timeline is final"
                )
            if len(run.records) >= self._limits.max_records_per_run:
                raise ReplayCapacityError(
                    f"a replay run retains at most "
                    f"{self._limits.max_records_per_run} timeline records"
                )
            run.records.append(record)

    async def timeline(
        self, run_id: str, *, after_sequence: int, limit: int
    ) -> tuple[tuple[ReplayTimelineRecord, ...], bool] | None:
        """Return one bounded page after *after_sequence*, and whether more follow.

        ``None`` when there is no such run.  The boolean is true when this page
        was truncated *or* the run may still produce more, which is the single
        thing a polling client needs in order to decide whether to poll again.
        """
        bounded = max(1, min(limit, self._limits.max_timeline_page))
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            pending = [item for item in run.records if item.sequence > after_sequence]
            page = tuple(pending[:bounded])
            truncated = len(pending) > len(page)
            return (page, truncated or not run.terminal)
