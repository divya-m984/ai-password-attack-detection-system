"""Tests for the replay orchestration and its error contract.

What matters here is that every refusal carries the code a client can branch on,
and that a document reports the run rather than restating it.  The service is the
one layer that turns the store's and the engine's typed failures into the API's
stable envelope, so the mapping from each of those to each code is asserted
directly rather than inferred from an HTTP status somewhere else.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from password_attack_detector.api.errors import APIError, ErrorCode
from password_attack_detector.api.schemas import (
    AnchorDetection,
    HybridLayerResult,
    MLLayerResult,
    RuleLayerResult,
)
from password_attack_detector.detection.enums import Severity
from password_attack_detector.replay import scenarios as catalog
from password_attack_detector.replay import service
from password_attack_detector.replay.engine import ReplayEngine, ReplayRuntime
from password_attack_detector.replay.enums import ReplayPace, ReplayState, ScenarioId
from password_attack_detector.replay.schemas import CreateReplayRunRequest
from password_attack_detector.replay.store import ReplayLimits, ReplayStore

_BASE = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Drive one coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


async def _detector(
    events: Sequence[Mapping[str, Any]], *, anchor_event_id: str
) -> AnchorDetection:
    """Return a fixed, unflagged verdict for any window."""
    return AnchorDetection(
        anchor_event_id=anchor_event_id,
        anchor_event_time=_BASE,
        rule=RuleLayerResult(
            flagged=False,
            risk_score=0.0,
            severity=Severity.LOW,
            scoring_version="1.0.0",
        ),
        ml=MLLayerResult(available=False, unavailable_reason="ml_disabled"),
        hybrid=HybridLayerResult(available=False, unavailable_reason="ml_disabled"),
        severity=Severity.LOW,
    )


def _runtime(**limits: Any) -> ReplayRuntime:
    """Return an available replay runtime over a stub detector."""
    store = ReplayStore(ReplayLimits(**limits))
    replay = ReplayRuntime(store=store, enabled=True, required=False)
    replay.engine = ReplayEngine(store, detector=_detector, pace_scale=0.0)
    return replay


def _request(scenario: str = "account_takeover", pace: str = "instant") -> Any:
    """Return a validated create-run request."""
    return CreateReplayRunRequest.model_validate(
        {"scenario_id": scenario, "pace": pace}
    )


async def _finish(replay: ReplayRuntime, run_id: str) -> None:
    """Wait for a run's background task to complete."""
    assert replay.engine is not None
    task = replay.engine._tasks.get(run_id)
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


def test_the_catalog_document_publishes_every_scenario() -> None:
    """Needs no runtime: the catalog is a property of the build."""
    document = service.scenario_catalog_document()
    assert document.scenario_count == len(catalog.SCENARIOS)
    assert {item.scenario_id for item in document.scenarios} == set(ScenarioId)
    assert document.replay_schema_version
    assert document.scenario_schema_version == catalog.SCENARIO_SCHEMA_VERSION


def test_each_catalog_entry_carries_its_content_fingerprint() -> None:
    """Two processes agreeing on this replayed identical timelines."""
    document = service.scenario_catalog_document()
    for entry in document.scenarios:
        source = catalog.scenario(str(entry.scenario_id))
        assert source is not None
        assert entry.scenario_fingerprint == source.fingerprint()
        assert entry.event_count == source.event_count


def test_the_catalog_publishes_no_event_bodies() -> None:
    """A client picks a scenario by name; it never receives the events."""
    document = service.scenario_catalog_document()
    payload = document.model_dump_json()
    for scenario in catalog.SCENARIOS:
        first = scenario.events()[0]
        assert str(first["event_id"]) not in payload
        assert str(first["user_id"]) not in payload


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "replay",
    [
        None,
        ReplayRuntime(store=ReplayStore(), enabled=False, required=False),
        ReplayRuntime(store=ReplayStore(), enabled=True, required=False),
    ],
    ids=["absent", "disabled", "unbound"],
)
def test_every_operation_refuses_when_replay_is_not_available(
    replay: ReplayRuntime | None,
) -> None:
    """One stable code covers all three ways replay can be missing."""

    async def main() -> None:
        for call in (
            service.start_run(replay, _request()),
            service.get_run(replay, "run_x"),
            service.list_runs(replay),
            service.get_timeline(replay, "run_x"),
            service.stop_run(replay, "run_x"),
        ):
            with pytest.raises(APIError) as caught:
                await call
            assert caught.value.code is ErrorCode.REPLAY_UNAVAILABLE
            assert caught.value.status_code == 503

    run(main())


def test_the_unavailable_refusal_names_a_reason_and_no_more() -> None:
    """Aggregate context only: a code, never a path or a message."""

    async def main() -> None:
        with pytest.raises(APIError) as caught:
            await service.get_run(None, "run_x")
        assert caught.value.detail == {"reason": "replay_disabled"}

    run(main())


# ---------------------------------------------------------------------------
# Starting
# ---------------------------------------------------------------------------


def test_a_started_run_reports_both_identities() -> None:
    """Content identity and instance identity, kept apart."""

    async def main() -> None:
        replay = _runtime()
        document = await service.start_run(replay, _request())
        source = catalog.scenario("account_takeover")
        assert source is not None

        assert document.run_id.startswith("run_")
        assert document.scenario_fingerprint == source.fingerprint()
        assert document.scenario_id is ScenarioId.ACCOUNT_TAKEOVER
        assert document.scenario_name == source.name
        assert document.event_count == source.event_count
        assert document.pace is ReplayPace.INSTANT
        assert document.state in {ReplayState.RUNNING, ReplayState.COMPLETED}
        await _finish(replay, document.run_id)

    run(main())


def test_two_runs_of_one_scenario_share_a_fingerprint_and_not_an_identity() -> None:
    """Same content, different executions."""

    async def main() -> None:
        replay = _runtime()
        first = await service.start_run(replay, _request())
        second = await service.start_run(replay, _request())
        assert first.scenario_fingerprint == second.scenario_fingerprint
        assert first.run_id != second.run_id
        await _finish(replay, first.run_id)
        await _finish(replay, second.run_id)

    run(main())


def test_the_active_ceiling_is_reported_as_a_limit_code() -> None:
    """429, not 503: the deployment is healthy and is refusing this request."""

    async def main() -> None:
        replay = _runtime(max_active_runs=1, max_retained_runs=4)
        # A paced run so the first one is still active when the second arrives.
        assert replay.engine is not None
        replay.engine._pace_scale = 1.0
        first = await service.start_run(replay, _request(pace="slow"))
        with pytest.raises(APIError) as caught:
            await service.start_run(replay, _request(pace="slow"))
        assert caught.value.code is ErrorCode.REPLAY_LIMIT_REACHED
        assert caught.value.status_code == 429
        assert caught.value.detail == {"max_active_runs": 1}
        assert replay.engine is not None
        await replay.engine.stop(first.run_id)

    run(main())


def test_an_unknown_scenario_is_refused_by_the_request_schema() -> None:
    """The enumeration is the input surface; nothing else reaches the service."""
    for name in ("unknown", "../../etc/passwd", "http://example.invalid", ""):
        with pytest.raises(ValueError):
            CreateReplayRunRequest.model_validate({"scenario_id": name})


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_an_unknown_run_is_refused_with_its_own_code() -> None:
    """Distinct from an unavailable subsystem: different fault, different fix."""

    async def main() -> None:
        replay = _runtime()
        for call in (
            service.get_run(replay, "run_absent"),
            service.get_timeline(replay, "run_absent"),
            service.stop_run(replay, "run_absent"),
        ):
            with pytest.raises(APIError) as caught:
                await call
            assert caught.value.code is ErrorCode.REPLAY_RUN_NOT_FOUND
            assert caught.value.status_code == 404

    run(main())


def test_a_negative_cursor_is_refused_with_the_cursor_code() -> None:
    """Not a position a cursor can occupy."""

    async def main() -> None:
        replay = _runtime()
        started = await service.start_run(replay, _request())
        await _finish(replay, started.run_id)
        with pytest.raises(APIError) as caught:
            await service.get_timeline(replay, started.run_id, after_sequence=-1)
        assert caught.value.code is ErrorCode.REPLAY_CURSOR_INVALID
        assert caught.value.detail == {"after_sequence": -1}

    run(main())


def test_the_cursor_returns_only_what_has_not_been_seen() -> None:
    """Incremental polling, which is the whole point of the cursor."""

    async def main() -> None:
        replay = _runtime()
        started = await service.start_run(replay, _request())
        await _finish(replay, started.run_id)

        first = await service.get_timeline(replay, started.run_id, after_sequence=0)
        assert first.after_sequence == 0
        assert first.record_count == started.event_count
        assert first.next_sequence == started.event_count
        assert first.more_expected is False

        second = await service.get_timeline(
            replay, started.run_id, after_sequence=first.next_sequence
        )
        assert second.record_count == 0
        assert second.next_sequence == first.next_sequence
        assert second.more_expected is False

    run(main())


def test_a_truncated_page_reports_that_more_is_coming() -> None:
    """A client that stopped polling on a truncated page would lose records."""

    async def main() -> None:
        replay = _runtime()
        started = await service.start_run(replay, _request())
        await _finish(replay, started.run_id)

        page = await service.get_timeline(replay, started.run_id, limit=3)
        assert page.record_count == 3
        assert page.next_sequence == 3
        assert page.more_expected is True

    run(main())


def test_the_page_size_is_clamped_to_the_store_bound() -> None:
    """An unbounded payload on every poll is what the bound exists to prevent."""

    async def main() -> None:
        replay = _runtime(max_timeline_page=2)
        started = await service.start_run(replay, _request())
        await _finish(replay, started.run_id)
        page = await service.get_timeline(replay, started.run_id, limit=1_000_000)
        assert page.record_count == 2

    run(main())


def test_a_run_document_summarises_only_what_the_run_produced() -> None:
    """No global total, and no figure from anywhere but the timeline records."""

    async def main() -> None:
        replay = _runtime()
        started = await service.start_run(replay, _request())
        await _finish(replay, started.run_id)
        document = await service.get_run(replay, started.run_id)

        assert document.state is ReplayState.COMPLETED
        assert document.more_expected is False
        assert document.emitted_count == document.event_count
        assert document.summary.detection_count == document.event_count
        # The stub flags nothing and offers no model, so the summary says so
        # rather than reporting zeros that read like negatives.
        assert document.summary.rule_flagged_count == 0
        assert document.summary.ml_unavailable_count == document.event_count
        assert document.summary.fusion_strategies == ()

    run(main())


def test_a_listing_carries_the_bounds_it_is_operating_under() -> None:
    """An absent run may have been evicted rather than never have existed."""

    async def main() -> None:
        replay = _runtime(max_active_runs=2, max_retained_runs=5)
        first = await service.start_run(replay, _request())
        await _finish(replay, first.run_id)
        second = await service.start_run(replay, _request(scenario="bot_activity"))
        await _finish(replay, second.run_id)

        listing = await service.list_runs(replay)
        assert listing.run_count == 2
        assert listing.max_active_runs == 2
        assert listing.max_retained_runs == 5
        assert listing.active_run_count == 0
        # Newest first.
        assert listing.runs[0].run_id == second.run_id

    run(main())


def test_a_listing_omits_the_timelines() -> None:
    """A listing that inlined every timeline would grow without bound."""

    async def main() -> None:
        replay = _runtime()
        started = await service.start_run(replay, _request())
        await _finish(replay, started.run_id)
        listing = await service.list_runs(replay)
        assert listing.runs[0].summary.detection_count == 0
        assert listing.runs[0].emitted_count == 0

    run(main())


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------


def test_stop_returns_a_terminal_run() -> None:
    """By the time a caller sees the response, the timeline is final."""

    async def main() -> None:
        replay = _runtime()
        assert replay.engine is not None
        replay.engine._pace_scale = 1.0
        started = await service.start_run(replay, _request(pace="slow"))
        stopped = await service.stop_run(replay, started.run_id)
        assert stopped.state is ReplayState.STOPPED
        assert stopped.more_expected is False

    run(main())


def test_stop_is_idempotent_at_the_service_layer() -> None:
    """A second press is a request that succeeded."""

    async def main() -> None:
        replay = _runtime()
        started = await service.start_run(replay, _request())
        await _finish(replay, started.run_id)
        first = await service.stop_run(replay, started.run_id)
        second = await service.stop_run(replay, started.run_id)
        assert first.state is second.state
        assert first.emitted_count == second.emitted_count

    run(main())


# ---------------------------------------------------------------------------
# The import guard
# ---------------------------------------------------------------------------


def test_the_replay_layer_holds_no_detection_capability() -> None:
    """A scorer reachable from here would make the boundary a convention."""
    service._assert_the_replay_layer_computes_no_verdict()
    namespace = vars(service)
    for forbidden in ("fuse", "RiskScorer", "FeatureEngine", "predict_serving_binary"):
        assert forbidden not in namespace
