"""Tests for the bounded, process-local replay run store.

The store's whole job is to be *honest about being small*: it bounds what one
process will hold, refuses rather than evicting anything live, and keeps a
finished run's timeline final. Each of those is asserted directly here.

``asyncio.run`` per test rather than a plugin: the store's lock is the only
asynchronous thing in it, each test builds its own store, and adding a test
dependency to exercise five ``await``s would be a dependency carried for syntax.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from password_attack_detector.api.schemas import (
    AnchorDetection,
    HybridLayerResult,
    MLLayerResult,
    RuleLayerResult,
)
from password_attack_detector.detection.enums import Severity
from password_attack_detector.exceptions import ReplayCapacityError, ReplayStateError
from password_attack_detector.ml.enums import FusionStrategy, ScoreKind
from password_attack_detector.replay.enums import ReplayPace, ReplayState, ScenarioId
from password_attack_detector.replay.schemas import ReplayTimelineRecord
from password_attack_detector.replay.store import (
    ReplayLimits,
    ReplayStore,
    summarize,
)

_BASE = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Drive one coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


def _clock() -> datetime:
    """Return a fixed instant, so a test can assert timestamps exactly."""
    return _BASE


def _store(**overrides: Any) -> ReplayStore:
    """Return a store with a fixed clock and deterministic identities."""
    counter = {"n": 0}

    def identity() -> str:
        counter["n"] += 1
        return f"run_{counter['n']:04d}"

    return ReplayStore(ReplayLimits(**overrides), clock=_clock, identity=identity)


async def _create(store: ReplayStore, *, events: int = 3) -> Any:
    """Register one run with plausible scenario metadata."""
    return await store.create(
        scenario_id=ScenarioId.BRUTE_FORCE,
        scenario_name="Concentrated brute force",
        scenario_revision=1,
        scenario_fingerprint="f" * 64,
        pace=ReplayPace.INSTANT,
        event_count=events,
    )


def _anchor(
    *,
    severity: Severity = Severity.LOW,
    rule_flagged: bool = False,
    fired: tuple[str, ...] = (),
    ml: MLLayerResult | None = None,
    hybrid: HybridLayerResult | None = None,
) -> AnchorDetection:
    """Return one anchor verdict, in the serving layer's own contract."""
    return AnchorDetection(
        anchor_event_id="00000000-0000-0000-0000-000000000001",
        anchor_event_time=_BASE,
        rule=RuleLayerResult(
            flagged=rule_flagged,
            risk_score=42.0 if rule_flagged else 0.0,
            severity=severity,
            fired_rule_ids=fired,
            fired_rule_count=len(fired),
            scoring_version="1.0.0",
        ),
        ml=ml or MLLayerResult(available=False, unavailable_reason="ml_disabled"),
        hybrid=hybrid
        or HybridLayerResult(available=False, unavailable_reason="ml_disabled"),
        severity=severity,
    )


def _record(
    sequence: int, run_id: str = "run_0001", **kwargs: Any
) -> ReplayTimelineRecord:
    """Return one timeline record at *sequence*."""
    return ReplayTimelineRecord(
        sequence=sequence,
        run_id=run_id,
        scenario_id=ScenarioId.BRUTE_FORCE,
        replay_state=ReplayState.RUNNING,
        event_index=sequence - 1,
        window_event_count=sequence,
        source_event_time=_BASE + timedelta(seconds=10 * sequence),
        emitted_at=_BASE,
        authentication_outcome="failure",
        detection=_anchor(**kwargs),
    )


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_limits_must_actually_limit() -> None:
    """A bound of zero is not a bound."""
    for name in (
        "max_active_runs",
        "max_retained_runs",
        "max_records_per_run",
        "max_timeline_page",
    ):
        with pytest.raises(ValueError, match=name):
            ReplayLimits(**{name: 0})
    with pytest.raises(ValueError, match="max_run_seconds"):
        ReplayLimits(max_run_seconds=0.0)


def test_retention_must_be_at_least_the_active_ceiling() -> None:
    """Otherwise an active run would have to be evicted to admit another."""
    with pytest.raises(ValueError, match="max_retained_runs"):
        ReplayLimits(max_active_runs=4, max_retained_runs=2)


# ---------------------------------------------------------------------------
# Creation and capacity
# ---------------------------------------------------------------------------


def test_a_created_run_starts_in_created_with_no_records() -> None:
    """Registration emits nothing; the engine does that."""
    store = _store()
    run_ = run(_create(store))
    assert run_.state is ReplayState.CREATED
    assert run_.records == []
    assert run_.emitted_count == 0
    assert run_.next_sequence == 0
    assert run_.created_at == _BASE
    assert run_.started_at is None
    assert not run_.terminal


def test_each_run_gets_its_own_identity() -> None:
    """Two runs of the same scenario are two runs, not one."""

    async def main() -> tuple[str, str]:
        store = _store()
        first = await _create(store)
        second = await _create(store)
        return (first.run_id, second.run_id)

    first, second = run(main())
    assert first != second


def test_the_default_identity_carries_nothing_about_the_host() -> None:
    """A run identifier is opaque: no path, no process id, no port, no user."""
    minted = {ReplayStore._mint() for _ in range(20)}
    assert len(minted) == 20
    for value in minted:
        assert value.startswith("run_")
        token = value.removeprefix("run_")
        assert len(token) == 32
        assert all(character in "0123456789abcdef" for character in token)


def test_the_active_ceiling_refuses_rather_than_evicting() -> None:
    """Admitting the request would mean stopping a run somebody is watching."""

    async def main() -> None:
        store = _store(max_active_runs=2, max_retained_runs=8)
        await _create(store)
        await _create(store)
        with pytest.raises(ReplayCapacityError, match="at most 2"):
            await _create(store)
        # Nothing was evicted or stopped to make room.
        assert len(await store.all_runs()) == 2

    run(main())


def test_a_finished_run_frees_its_slot() -> None:
    """The ceiling is on *active* runs, not on runs ever started."""

    async def main() -> None:
        store = _store(max_active_runs=1, max_retained_runs=8)
        first = await _create(store)
        await store.transition(first.run_id, ReplayState.RUNNING)
        await store.transition(first.run_id, ReplayState.COMPLETED)
        second = await _create(store)
        assert second.run_id != first.run_id

    run(main())


def test_retention_evicts_the_oldest_finished_run_first() -> None:
    """Deterministic FIFO over finished runs, and only over finished runs."""

    async def main() -> None:
        store = _store(max_active_runs=1, max_retained_runs=3)
        ids = []
        for _ in range(4):
            created = await _create(store)
            ids.append(created.run_id)
            await store.transition(created.run_id, ReplayState.RUNNING)
            await store.transition(created.run_id, ReplayState.COMPLETED)
        retained = [item.run_id for item in await store.all_runs()]
        assert ids[0] not in retained
        assert ids[-1] in retained
        assert len(retained) <= 3

    run(main())


def test_retention_never_evicts_an_active_run() -> None:
    """The store stays at its bound rather than dropping something live."""

    async def main() -> None:
        store = _store(max_active_runs=3, max_retained_runs=3)
        first = await _create(store)
        await _create(store)
        await _create(store)
        # Every retained run is active, so nothing is evictable and the active
        # ceiling is what refuses the next one.
        with pytest.raises(ReplayCapacityError):
            await _create(store)
        assert first.run_id in {item.run_id for item in await store.all_runs()}

    run(main())


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def test_a_run_moves_from_created_to_running_to_completed() -> None:
    """The ordinary path."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        started = await store.transition(created.run_id, ReplayState.RUNNING)
        assert started.started_at == _BASE
        done = await store.transition(created.run_id, ReplayState.COMPLETED)
        assert done.state is ReplayState.COMPLETED
        assert done.finished_at == _BASE
        assert done.terminal

    run(main())


@pytest.mark.parametrize(
    ("terminal", "target"),
    [
        (ReplayState.COMPLETED, ReplayState.RUNNING),
        (ReplayState.STOPPED, ReplayState.RUNNING),
        (ReplayState.FAILED, ReplayState.RUNNING),
        (ReplayState.COMPLETED, ReplayState.STOPPED),
        (ReplayState.STOPPED, ReplayState.COMPLETED),
    ],
)
def test_a_terminal_run_never_transitions_again(
    terminal: ReplayState, target: ReplayState
) -> None:
    """A finished run stays finished. A second execution is a new run."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        await store.transition(
            created.run_id,
            terminal,
            failure_reason="x" if terminal is ReplayState.FAILED else None,
        )
        with pytest.raises(ReplayStateError, match="cannot move"):
            await store.transition(created.run_id, target)

    run(main())


def test_a_run_cannot_be_started_twice() -> None:
    """Two clients racing to start one run: the state machine settles it."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        with pytest.raises(ReplayStateError):
            await store.transition(created.run_id, ReplayState.RUNNING)

    run(main())


def test_transitioning_an_unknown_run_is_refused() -> None:
    """A run this process does not have cannot be moved."""

    async def main() -> None:
        store = _store()
        with pytest.raises(ReplayStateError, match="no replay run"):
            await store.transition("run_nope", ReplayState.RUNNING)

    run(main())


def test_a_failure_must_name_a_reason_and_only_a_failure_may() -> None:
    """The reason code is what an operator greps for."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        with pytest.raises(ReplayStateError, match="stable reason"):
            await store.transition(created.run_id, ReplayState.FAILED)
        with pytest.raises(ReplayStateError, match="stable reason"):
            await store.transition(
                created.run_id, ReplayState.STOPPED, failure_reason="why"
            )
        failed = await store.transition(
            created.run_id, ReplayState.FAILED, failure_reason="replay_x"
        )
        assert failed.failure_reason == "replay_x"

    run(main())


def test_stop_is_idempotent_and_survives_a_finished_run() -> None:
    """ "Stop a stopped run" is a request that succeeded."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        assert await store.request_stop(created.run_id) is not None
        assert await store.stop_requested(created.run_id) is True
        assert await store.request_stop(created.run_id) is not None
        await store.transition(created.run_id, ReplayState.STOPPED)
        again = await store.request_stop(created.run_id)
        assert again is not None
        assert again.state is ReplayState.STOPPED

    run(main())


def test_stopping_an_unknown_run_reports_nothing_rather_than_raising() -> None:
    """The service turns this into a stable 404 code."""
    store = _store()
    assert run(store.request_stop("run_absent")) is None
    assert run(store.stop_requested("run_absent")) is False


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_records_accumulate_in_sequence_order() -> None:
    """The timeline is what the cursor is a position in."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        for index in (1, 2, 3):
            await store.append(created.run_id, _record(index, created.run_id))
        snapshot = await store.snapshot(created.run_id)
        assert snapshot is not None
        assert [item.sequence for item in snapshot.records] == [1, 2, 3]
        assert snapshot.next_sequence == 3
        assert snapshot.emitted_count == 3

    run(main())


def test_a_terminal_run_accepts_no_further_record() -> None:
    """This is what makes a stop acknowledgement honest."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        await store.append(created.run_id, _record(1, created.run_id))
        await store.transition(created.run_id, ReplayState.STOPPED)
        with pytest.raises(ReplayStateError, match="timeline is final"):
            await store.append(created.run_id, _record(2, created.run_id))

    run(main())


def test_appending_to_an_unknown_run_is_refused() -> None:
    """An evicted run cannot acquire new records."""

    async def main() -> None:
        store = _store()
        with pytest.raises(ReplayStateError, match="no replay run"):
            await store.append("run_gone", _record(1))

    run(main())


def test_the_per_run_record_bound_is_enforced() -> None:
    """A backstop, far above any catalog scenario."""

    async def main() -> None:
        store = _store(max_records_per_run=2)
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        await store.append(created.run_id, _record(1, created.run_id))
        await store.append(created.run_id, _record(2, created.run_id))
        with pytest.raises(ReplayCapacityError, match="at most 2"):
            await store.append(created.run_id, _record(3, created.run_id))

    run(main())


def test_a_snapshot_does_not_change_while_the_run_advances() -> None:
    """A page rendered from a snapshot describes one moment, not two."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        await store.append(created.run_id, _record(1, created.run_id))
        snapshot = await store.snapshot(created.run_id)
        await store.append(created.run_id, _record(2, created.run_id))
        assert snapshot is not None
        assert len(snapshot.records) == 1

    run(main())


def test_two_runs_never_see_one_anothers_records() -> None:
    """Cross-run contamination would make every timeline unreadable."""

    async def main() -> None:
        store = _store()
        first = await _create(store)
        second = await _create(store)
        await store.transition(first.run_id, ReplayState.RUNNING)
        await store.transition(second.run_id, ReplayState.RUNNING)
        await store.append(first.run_id, _record(1, first.run_id))
        await store.append(first.run_id, _record(2, first.run_id))
        await store.append(second.run_id, _record(1, second.run_id))
        left = await store.snapshot(first.run_id)
        right = await store.snapshot(second.run_id)
        assert left is not None and right is not None
        assert len(left.records) == 2
        assert len(right.records) == 1
        assert {item.run_id for item in right.records} == {second.run_id}

    run(main())


def test_stopping_one_run_leaves_another_alone() -> None:
    """The addressed run and no other."""

    async def main() -> None:
        store = _store()
        first = await _create(store)
        second = await _create(store)
        await store.transition(first.run_id, ReplayState.RUNNING)
        await store.transition(second.run_id, ReplayState.RUNNING)
        await store.request_stop(first.run_id)
        await store.transition(first.run_id, ReplayState.STOPPED)
        other = await store.get(second.run_id)
        assert other is not None
        assert other.state is ReplayState.RUNNING
        assert await store.stop_requested(second.run_id) is False

    run(main())


# ---------------------------------------------------------------------------
# The cursor
# ---------------------------------------------------------------------------


def test_the_timeline_returns_only_records_after_the_cursor() -> None:
    """Incremental polling is the whole point of the cursor."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        for index in range(1, 6):
            await store.append(created.run_id, _record(index, created.run_id))
        page = await store.timeline(created.run_id, after_sequence=3, limit=10)
        assert page is not None
        records, more = page
        assert [item.sequence for item in records] == [4, 5]
        assert more is True, "the run has not finished"

    run(main())


def test_the_timeline_page_is_bounded_and_reports_truncation() -> None:
    """A poll never returns an unbounded payload."""

    async def main() -> None:
        store = _store(max_timeline_page=2)
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        for index in range(1, 6):
            await store.append(created.run_id, _record(index, created.run_id))
        await store.transition(created.run_id, ReplayState.COMPLETED)
        page = await store.timeline(created.run_id, after_sequence=0, limit=1000)
        assert page is not None
        records, more = page
        assert [item.sequence for item in records] == [1, 2]
        assert more is True, "this page was truncated"

    run(main())


def test_a_terminal_run_with_nothing_left_reports_no_more() -> None:
    """The signal a polling client stops on."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        await store.append(created.run_id, _record(1, created.run_id))
        await store.transition(created.run_id, ReplayState.COMPLETED)
        page = await store.timeline(created.run_id, after_sequence=0, limit=10)
        assert page is not None
        _, more = page
        assert more is False

    run(main())


def test_a_cursor_beyond_the_end_returns_an_empty_page() -> None:
    """Not an error: a client that is up to date asks for nothing and gets it."""

    async def main() -> None:
        store = _store()
        created = await _create(store)
        await store.transition(created.run_id, ReplayState.RUNNING)
        await store.append(created.run_id, _record(1, created.run_id))
        await store.transition(created.run_id, ReplayState.COMPLETED)
        page = await store.timeline(created.run_id, after_sequence=99, limit=10)
        assert page is not None
        records, more = page
        assert records == ()
        assert more is False

    run(main())


def test_the_timeline_of_an_unknown_run_is_none() -> None:
    """The service turns this into a stable 404 code."""
    store = _store()
    assert run(store.timeline("run_absent", after_sequence=0, limit=10)) is None


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def test_an_empty_run_summarises_to_zero_and_no_severity() -> None:
    """Absence is reported as absence, not as a zero severity."""
    summary = summarize([])
    assert summary.detection_count == 0
    assert summary.highest_severity is None
    assert summary.severity_counts == {}
    assert summary.triggered_rule_counts == {}


def test_the_summary_counts_each_layer_separately() -> None:
    """Three independent decisions, never one total."""
    records = [
        _record(
            1,
            rule_flagged=True,
            fired=("PAD-BF-001",),
            ml=MLLayerResult(
                available=True,
                flagged=True,
                decision_score=1.0,
                score_kind=ScoreKind.DECISION_SCORE,
            ),
            hybrid=HybridLayerResult(
                available=True, flagged=True, strategy=FusionStrategy.STACKED
            ),
        ),
        _record(
            2,
            rule_flagged=False,
            ml=MLLayerResult(
                available=True,
                flagged=False,
                decision_score=-1.0,
                score_kind=ScoreKind.DECISION_SCORE,
            ),
            hybrid=HybridLayerResult(
                available=True, flagged=False, strategy=FusionStrategy.STACKED
            ),
        ),
    ]
    summary = summarize(records)
    assert summary.detection_count == 2
    assert summary.rule_flagged_count == 1
    assert summary.ml_flagged_count == 1
    assert summary.hybrid_flagged_count == 1
    assert summary.ml_unavailable_count == 0
    assert summary.fusion_strategies == ("stacked",)


def test_an_unavailable_layer_is_counted_as_absent_not_as_a_negative() -> None:
    """A model that produced no verdict has not declined to flag anything."""
    summary = summarize([_record(1), _record(2)])
    assert summary.ml_flagged_count == 0
    assert summary.ml_unavailable_count == 2
    assert summary.hybrid_unavailable_count == 2
    assert summary.fusion_strategies == ()


def test_the_summary_reports_the_worst_severity_on_the_ordinal_scale() -> None:
    """Worst, not last and not most frequent."""
    summary = summarize(
        [
            _record(1, severity=Severity.CRITICAL),
            _record(2, severity=Severity.LOW),
            _record(3, severity=Severity.MEDIUM),
        ]
    )
    assert summary.highest_severity == "critical"


def test_severity_counts_are_ordered_by_the_phase_four_scale() -> None:
    """So a chart of them reads left to right as increasing seriousness."""
    summary = summarize(
        [
            _record(1, severity=Severity.HIGH),
            _record(2, severity=Severity.LOW),
            _record(3, severity=Severity.HIGH),
        ]
    )
    assert list(summary.severity_counts) == ["low", "high"]
    assert summary.severity_counts == {"low": 1, "high": 2}


def test_rule_counts_are_ordered_by_frequency_then_identifier() -> None:
    """The same run always renders the same table."""
    summary = summarize(
        [
            _record(1, rule_flagged=True, fired=("PAD-BOT-001", "PAD-BF-001")),
            _record(2, rule_flagged=True, fired=("PAD-BF-001",)),
        ]
    )
    assert list(summary.triggered_rule_counts) == ["PAD-BF-001", "PAD-BOT-001"]
