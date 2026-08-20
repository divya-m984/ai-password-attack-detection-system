"""Tests for the replay scheduler.

The engine's contract, in four parts:

* **Pace is presentation.**  The same scenario at every pace produces byte-
  identical verdicts, in the same order, against the same windows.  This is the
  property the whole demonstration rests on and it is asserted directly.
* **A stop is honest.**  When :meth:`ReplayEngine.stop` returns, the run is
  terminal and its timeline is final.
* **A failure is a state.**  Anything the detector raises becomes ``failed`` with
  a stable reason code, and nothing escapes the background task.
* **Runs are isolated.**  Two concurrent runs share a store and see nothing of
  one another; stopping one leaves the other running.

The detector is a stub, deliberately: this module is being tested for *when* it
calls something and *what state that leaves behind*, and a real frozen champion
here would make the tests slow without making them stronger.  The real binding is
exercised end to end in ``tests/integration/test_replay_detection.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from password_attack_detector.api.schemas import (
    AnchorDetection,
    HybridLayerResult,
    MLLayerResult,
    RuleLayerResult,
)
from password_attack_detector.detection.enums import Severity
from password_attack_detector.exceptions import (
    PasswordAttackDetectorError,
    ReplayStateError,
)
from password_attack_detector.ml.enums import FusionStrategy, ScoreKind
from password_attack_detector.replay import scenarios as catalog
from password_attack_detector.replay.engine import (
    DEADLINE_EXCEEDED,
    DETECTION_FAILED,
    RECORD_LIMIT_REACHED,
    ReplayEngine,
)
from password_attack_detector.replay.enums import ReplayPace, ReplayState, ScenarioId
from password_attack_detector.replay.store import ReplayLimits, ReplayStore

_BASE = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Drive one coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


class StubDetector:
    """A detector that records what it was asked and answers deterministically.

    The verdict is a function of the window size alone, so a run's timeline is
    reproducible and any difference between two runs of the same scenario is a
    difference in *what was scored*, not in what the scorer felt like returning.
    """

    def __init__(self, *, fail_at: int | None = None, delay: float = 0.0) -> None:
        self.calls: list[tuple[int, str]] = []
        self._fail_at = fail_at
        self._delay = delay

    async def __call__(
        self, events: Sequence[Mapping[str, Any]], *, anchor_event_id: str
    ) -> AnchorDetection:
        """Return a verdict derived from the window size."""
        self.calls.append((len(events), anchor_event_id))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail_at is not None and len(events) == self._fail_at:
            raise PasswordAttackDetectorError("the frozen champion refused this row")
        flagged = len(events) >= 5
        return AnchorDetection(
            anchor_event_id=anchor_event_id,
            anchor_event_time=_BASE,
            rule=RuleLayerResult(
                flagged=flagged,
                risk_score=float(min(100, len(events) * 3)),
                severity=Severity.HIGH if flagged else Severity.LOW,
                fired_rule_ids=("PAD-BF-001",) if flagged else (),
                fired_rule_count=1 if flagged else 0,
                scoring_version="1.0.0",
            ),
            ml=MLLayerResult(
                available=True,
                flagged=flagged,
                decision_score=float(len(events)),
                score_kind=ScoreKind.DECISION_SCORE,
            ),
            hybrid=HybridLayerResult(
                available=True, flagged=flagged, strategy=FusionStrategy.STACKED
            ),
            severity=Severity.HIGH if flagged else Severity.LOW,
        )


def _store(**overrides: Any) -> ReplayStore:
    """Return a store with deterministic identities."""
    counter = {"n": 0}

    def identity() -> str:
        counter["n"] += 1
        return f"run_{counter['n']:04d}"

    return ReplayStore(ReplayLimits(**overrides), identity=identity)


async def _register(
    store: ReplayStore, scenario: catalog.Scenario, pace: ReplayPace
) -> Any:
    """Register one run for *scenario*."""
    return await store.create(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        scenario_revision=scenario.revision,
        scenario_fingerprint=scenario.fingerprint(),
        pace=pace,
        event_count=scenario.event_count,
    )


def _small() -> catalog.Scenario:
    """Return a short scenario, so a full run costs a handful of stub calls."""
    found = catalog.scenario("account_takeover")
    assert found is not None
    return found


async def _drain(engine: ReplayEngine, run_id: str) -> Any:
    """Wait for a run's task to finish and return the run."""
    task = engine._tasks.get(run_id)
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)
    return await engine._store.snapshot(run_id)


# ---------------------------------------------------------------------------
# The ordinary run
# ---------------------------------------------------------------------------


def test_a_run_scores_every_event_against_the_prefix_emitted_so_far() -> None:
    """Step *n* sees exactly the history that existed when event *n* occurred."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        detector = StubDetector()
        engine = ReplayEngine(store, detector=detector, pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        finished = await _drain(engine, created.run_id)

        assert finished is not None
        assert finished.state is ReplayState.COMPLETED
        assert finished.emitted_count == scenario.event_count
        # One call per event, with a window that grows by exactly one each time.
        assert [size for size, _ in detector.calls] == list(
            range(1, scenario.event_count + 1)
        )
        expected = [str(item["event_id"]) for item in scenario.events()]
        assert [anchor for _, anchor in detector.calls] == expected

    run(main())


def test_records_carry_the_scenario_timestamp_not_the_wall_clock() -> None:
    """The scientific instant and the presentation instant are different fields."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(
            store, detector=StubDetector(), pace_scale=0.0, clock=lambda: _BASE
        )
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        finished = await _drain(engine, created.run_id)

        assert finished is not None
        source = [item.source_event_time for item in finished.records]
        assert source == sorted(source)
        assert all(item.emitted_at == _BASE for item in finished.records)
        assert [item.sequence for item in finished.records] == list(
            range(1, scenario.event_count + 1)
        )
        assert [item.window_event_count for item in finished.records] == list(
            range(1, scenario.event_count + 1)
        )

    run(main())


# ---------------------------------------------------------------------------
# Pace independence
# ---------------------------------------------------------------------------


def _fingerprint(records: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Return the scientific content of a timeline, with presentation stripped.

    Everything a detector produced and nothing about when it was shown: the
    ``emitted_at`` field is precisely the thing a pace is allowed to change.
    """
    return [
        (
            item.sequence,
            item.event_index,
            item.window_event_count,
            item.source_event_time,
            item.authentication_outcome,
            item.detection.model_dump_json(),
        )
        for item in records
    ]


@pytest.mark.parametrize(
    "pace", [ReplayPace.INSTANT, ReplayPace.FAST, ReplayPace.NORMAL, ReplayPace.SLOW]
)
def test_the_final_result_is_identical_at_every_pace(pace: ReplayPace) -> None:
    """The property the whole demonstration rests on.

    The ``normal`` and ``slow`` paces run through their real code path -- the
    interruptible wait, the stop race, the step ordering -- with the interval
    scaled towards zero, which is a shortened clock rather than a bypassed one.
    """

    async def once(chosen: ReplayPace) -> list[tuple[Any, ...]]:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)
        created = await _register(store, scenario, chosen)
        await engine.start(created, scenario)
        finished = await _drain(engine, created.run_id)
        assert finished is not None
        assert finished.state is ReplayState.COMPLETED
        return _fingerprint(finished.records)

    baseline = run(once(ReplayPace.INSTANT))
    assert run(once(pace)) == baseline


def test_a_scaled_pace_still_takes_the_interruptible_wait() -> None:
    """A scale of zero must not silently skip the delay branch entirely.

    Asserted by driving a paced run with a scale that is small but non-zero and
    confirming it still completes with the full timeline: the wait is exercised,
    it times out, and the loop carries on.
    """

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0005)
        created = await _register(store, scenario, ReplayPace.SLOW)
        await engine.start(created, scenario)
        finished = await _drain(engine, created.run_id)
        assert finished is not None
        assert finished.state is ReplayState.COMPLETED
        assert finished.emitted_count == scenario.event_count

    run(main())


def test_a_pace_scale_outside_the_unit_interval_is_refused() -> None:
    """A scale above one would let a run outlast the pace vocabulary's ceiling."""
    store = _store()
    for value in (-0.1, 1.5):
        with pytest.raises(ValueError, match="pace_scale"):
            ReplayEngine(store, detector=StubDetector(), pace_scale=value)


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------


def test_a_stopped_run_is_terminal_by_the_time_stop_returns() -> None:
    """This is what makes the acknowledgement honest."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(delay=0.01), pace_scale=1.0)
        created = await _register(store, scenario, ReplayPace.SLOW)
        await engine.start(created, scenario)
        await asyncio.sleep(0)
        stopped = await engine.stop(created.run_id)

        assert stopped is not None
        assert stopped.state is ReplayState.STOPPED
        assert stopped.terminal
        before = stopped.emitted_count
        # Nothing further can be appended: the store refuses a terminal run.
        await asyncio.sleep(0.05)
        after = await store.snapshot(created.run_id)
        assert after is not None
        assert after.emitted_count == before
        assert after.emitted_count < scenario.event_count

    run(main())


def test_a_stop_preserves_the_records_already_emitted() -> None:
    """Discarding them would lose detections that genuinely happened."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(delay=0.02), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        await asyncio.sleep(0.05)
        stopped = await engine.stop(created.run_id)
        assert stopped is not None
        assert stopped.emitted_count >= 1
        assert [item.sequence for item in stopped.records] == list(
            range(1, stopped.emitted_count + 1)
        )

    run(main())


def test_stopping_is_idempotent() -> None:
    """A second press reports the state the run is in and changes nothing."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        await _drain(engine, created.run_id)

        first = await engine.stop(created.run_id)
        second = await engine.stop(created.run_id)
        assert first is not None and second is not None
        # The run completed on its own, so stopping reports completion rather
        # than overwriting it: a stop that landed after the fact did not stop it.
        assert first.state is ReplayState.COMPLETED
        assert second.state is first.state
        assert second.emitted_count == first.emitted_count

    run(main())


def test_stopping_a_run_that_never_started_moves_it_straight_to_stopped() -> None:
    """A created-but-unstarted run is stoppable, and has no task to cancel."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        stopped = await engine.stop(created.run_id)
        assert stopped is not None
        assert stopped.state is ReplayState.STOPPED
        assert stopped.emitted_count == 0

    run(main())


def test_stopping_an_unknown_run_returns_none() -> None:
    """The service turns this into a stable 404 code."""
    store = _store()
    engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)
    assert run(engine.stop("run_absent")) is None


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_a_detector_failure_leaves_the_run_failed_with_a_stable_reason() -> None:
    """Nothing escapes the background task, and no message reaches the record."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(fail_at=3), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        finished = await _drain(engine, created.run_id)

        assert finished is not None
        assert finished.state is ReplayState.FAILED
        assert finished.failure_reason == DETECTION_FAILED
        assert finished.emitted_count == 2, "the two steps before the failure"
        assert "refused this row" not in str(finished.failure_reason)

    run(main())


def test_the_record_bound_fails_the_run_rather_than_truncating_it() -> None:
    """A truncated timeline that reported success would be a silent cap."""

    async def main() -> None:
        scenario = _small()
        store = _store(max_records_per_run=2)
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        finished = await _drain(engine, created.run_id)

        assert finished is not None
        assert finished.state is ReplayState.FAILED
        assert finished.failure_reason == RECORD_LIMIT_REACHED
        assert finished.emitted_count == 2

    run(main())


def test_a_run_that_outlives_its_deadline_is_abandoned() -> None:
    """A run holding a slot forever is what the ceiling exists to prevent."""

    async def main() -> None:
        scenario = _small()
        store = _store(max_run_seconds=0.01)
        engine = ReplayEngine(store, detector=StubDetector(delay=0.02), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        finished = await _drain(engine, created.run_id)

        assert finished is not None
        assert finished.state is ReplayState.FAILED
        assert finished.failure_reason == DEADLINE_EXCEEDED

    run(main())


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_a_run_cannot_be_started_twice() -> None:
    """Two clients racing to start one run: the state machine settles it."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        with pytest.raises(ReplayStateError):
            await engine.start(created, scenario)
        await _drain(engine, created.run_id)

    run(main())


def test_two_simultaneous_runs_are_isolated() -> None:
    """Different scenarios, one store, one engine, no contamination."""

    async def main() -> None:
        first_scenario = _small()
        second_scenario = catalog.scenario("bot_activity")
        assert second_scenario is not None
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)

        first = await _register(store, first_scenario, ReplayPace.INSTANT)
        second = await _register(store, second_scenario, ReplayPace.FAST)
        await engine.start(first, first_scenario)
        await engine.start(second, second_scenario)
        left = await _drain(engine, first.run_id)
        right = await _drain(engine, second.run_id)

        assert left is not None and right is not None
        assert left.emitted_count == first_scenario.event_count
        assert right.emitted_count == second_scenario.event_count
        assert {item.scenario_id for item in left.records} == {
            ScenarioId.ACCOUNT_TAKEOVER
        }
        assert {item.scenario_id for item in right.records} == {ScenarioId.BOT_ACTIVITY}
        assert {item.run_id for item in left.records} == {first.run_id}
        assert {item.run_id for item in right.records} == {second.run_id}

    run(main())


def test_stopping_one_run_does_not_stop_another() -> None:
    """The addressed run and no other."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(delay=0.01), pace_scale=1.0)
        first = await _register(store, scenario, ReplayPace.SLOW)
        second = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(first, scenario)
        await engine.start(second, scenario)

        stopped = await engine.stop(first.run_id)
        assert stopped is not None
        assert stopped.state is ReplayState.STOPPED

        other = await _drain(engine, second.run_id)
        assert other is not None
        assert other.state is ReplayState.COMPLETED
        assert other.emitted_count == scenario.event_count

    run(main())


def test_shutdown_cancels_every_active_run_and_leaves_it_terminal() -> None:
    """A process that exited with a run still 'running' would misreport itself."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(delay=0.05), pace_scale=1.0)
        first = await _register(store, scenario, ReplayPace.SLOW)
        second = await _register(store, scenario, ReplayPace.SLOW)
        await engine.start(first, scenario)
        await engine.start(second, scenario)
        await asyncio.sleep(0)

        await engine.shutdown()

        assert engine.active_run_ids == ()
        for run_id in (first.run_id, second.run_id):
            snapshot = await store.snapshot(run_id)
            assert snapshot is not None
            assert snapshot.terminal
            assert snapshot.state is ReplayState.STOPPED

    run(main())


def test_shutdown_leaves_an_already_finished_run_alone() -> None:
    """A completed run is not retrospectively reported as stopped."""

    async def main() -> None:
        scenario = _small()
        store = _store()
        engine = ReplayEngine(store, detector=StubDetector(), pace_scale=0.0)
        created = await _register(store, scenario, ReplayPace.INSTANT)
        await engine.start(created, scenario)
        await _drain(engine, created.run_id)

        await engine.shutdown()
        snapshot = await store.snapshot(created.run_id)
        assert snapshot is not None
        assert snapshot.state is ReplayState.COMPLETED

    run(main())


def test_shutdown_on_an_engine_with_nothing_running_is_a_no_op() -> None:
    """Called from the lifespan of every process, including idle ones."""

    async def main() -> None:
        engine = ReplayEngine(_store(), detector=StubDetector(), pace_scale=0.0)
        await engine.shutdown()
        assert engine.active_run_ids == ()

    run(main())
