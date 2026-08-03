"""A deliberately naive reference implementation of the feature engine.

This recomputes every windowed and sequence feature from scratch for each
anchor by filtering the full event list with an explicit predicate.  It is
O(n^2) and unusably slow on real data; that is the point.  It expresses the
point-in-time contract as directly as possible, with no incremental state, no
eviction, and no bookkeeping that could be subtly wrong.

The production engine is checked against it for exact equality.  Because both
sides accumulate in exact integers, "exact" means exact -- no tolerance.  Any
divergence is a real defect in the incremental accumulators, not float noise.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import pairwise
from typing import Any

from password_attack_detector.data.enums import AuthOutcome, MFAOutcome
from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.features.catalog import FeatureCatalog
from password_attack_detector.features.config import (
    EntityKind,
    FeatureConfig,
    duration_to_microseconds,
    parse_duration,
)
from password_attack_detector.features.engine import sort_events_canonically
from password_attack_detector.features.temporal import (
    MICROSECONDS_PER_SECOND,
    coefficient_of_variation,
    mean_std,
    to_microseconds,
)

__all__ = ["reference_columns", "reference_rows"]


def _entity_key(kind: EntityKind, event: AuthEvent) -> Hashable:
    if kind is EntityKind.USER:
        return event.user_id
    if kind is EntityKind.SOURCE:
        return event.source_id
    if kind is EntityKind.USER_SOURCE:
        return (event.user_id, event.source_id)
    if kind is EntityKind.USER_DEVICE:
        return (event.user_id, event.device_id)
    if kind is EntityKind.SESSION:
        return event.session_id
    return event.device_id


def _dimension_value(name: str, event: AuthEvent) -> str | None:
    if name == "source":
        return event.source_id
    if name == "user":
        return event.user_id
    if name == "device":
        return event.device_id
    if name == "country":
        return event.country_code
    if name == "application":
        return event.application_id
    if name == "auth_method":
        return str(event.authentication_method)
    if name == "user_agent":
        return event.user_agent_family
    raise AssertionError(f"Unknown dimension {name!r}")


_ENTITY_PREFIX_TO_KIND: dict[str, EntityKind] = {
    "user_device": EntityKind.USER_DEVICE,
    "user_source": EntityKind.USER_SOURCE,
    "pair": EntityKind.USER_SOURCE,
    "user": EntityKind.USER,
    "source": EntityKind.SOURCE,
    "session": EntityKind.SESSION,
    "device": EntityKind.DEVICE,
}

_UNIQUE_MEASURES: dict[str, str] = {
    "unique_source_count": "source",
    "unique_device_count": "device",
    "unique_country_count": "country",
    "unique_application_count": "application",
    "unique_auth_method_count": "auth_method",
    "unique_user_count": "user",
    "unique_user_agent_count": "user_agent",
    "user_count": "user",
}

_OUTCOME_MEASURES: dict[str, AuthOutcome] = {
    "success_count": AuthOutcome.SUCCESS,
    "failure_count": AuthOutcome.FAILURE,
    "blocked_count": AuthOutcome.BLOCKED,
    "challenge_count": AuthOutcome.CHALLENGED,
}

_RATE_MEASURES: dict[str, AuthOutcome] = {
    "failure_rate": AuthOutcome.FAILURE,
    "success_rate": AuthOutcome.SUCCESS,
}


def _measure_of(name: str, window: str) -> tuple[EntityKind, str]:
    """Split a windowed feature name into its entity kind and measure."""
    stem = name[: -(len(window) + 2)]
    for prefix, kind in _ENTITY_PREFIX_TO_KIND.items():
        if stem.startswith(prefix + "_"):
            return kind, stem[len(prefix) + 1 :]
    raise AssertionError(f"Cannot decompose feature name {name!r}")


def _windowed_value(
    measure: str,
    history: Sequence[AuthEvent],
    *,
    min_count: int,
) -> Any:
    """Compute one windowed aggregate by brute force over *history*."""
    n = len(history)

    if measure in {"attempt_count", "event_count"}:
        return n
    if measure in _OUTCOME_MEASURES:
        target = _OUTCOME_MEASURES[measure]
        return sum(1 for e in history if e.authentication_outcome is target)
    if measure == "mfa_failure_count":
        return sum(1 for e in history if e.mfa_outcome is MFAOutcome.FAILED)
    if measure in _RATE_MEASURES:
        if n < min_count or n == 0:
            return None
        target = _RATE_MEASURES[measure]
        return sum(1 for e in history if e.authentication_outcome is target) / n
    if measure in _UNIQUE_MEASURES:
        dimension = _UNIQUE_MEASURES[measure]
        values = {_dimension_value(dimension, e) for e in history}
        values.discard(None)
        return len(values)

    if measure in {"mean_response_time_ms", "response_time_std_ms"}:
        samples = [
            e.response_time_ms for e in history if e.response_time_ms is not None
        ]
        mean, std = mean_std(len(samples), sum(samples), sum(v * v for v in samples))
        return mean if measure == "mean_response_time_ms" else std

    if measure in {
        "mean_interarrival_seconds",
        "interarrival_std_seconds",
        "interarrival_coefficient_of_variation",
    }:
        stamps = [to_microseconds(e.event_time) for e in history]
        gaps = [b - a for a, b in pairwise(stamps)]
        mean, std = mean_std(len(gaps), sum(gaps), sum(g * g for g in gaps))
        mean_s = None if mean is None else mean / MICROSECONDS_PER_SECOND
        std_s = None if std is None else std / MICROSECONDS_PER_SECOND
        if measure == "mean_interarrival_seconds":
            return mean_s
        if measure == "interarrival_std_seconds":
            return std_s
        return coefficient_of_variation(mean_s, std_s)

    raise AssertionError(f"Reference engine cannot compute measure {measure!r}")


def _sequence_values(
    anchor: AuthEvent, ordered: Sequence[AuthEvent], anchor_ts: int
) -> dict[str, Any]:
    """Compute the sequence features by folding all strictly-earlier events."""

    def prior_for(kind: EntityKind) -> list[AuthEvent]:
        key = _entity_key(kind, anchor)
        return [
            e
            for e in ordered
            if _entity_key(kind, e) == key and to_microseconds(e.event_time) < anchor_ts
        ]

    def consecutive_failures(history: Sequence[AuthEvent]) -> int:
        count = 0
        for event in reversed(history):
            if event.authentication_outcome is AuthOutcome.FAILURE:
                count += 1
            else:
                break
        return count

    def failures_since_success(history: Sequence[AuthEvent]) -> int:
        count = 0
        for event in reversed(history):
            if event.authentication_outcome is AuthOutcome.SUCCESS:
                break
            if event.authentication_outcome is AuthOutcome.FAILURE:
                count += 1
        return count

    def elapsed_since_last(
        history: Sequence[AuthEvent], outcome: AuthOutcome | None = None
    ) -> float | None:
        for event in reversed(history):
            if outcome is None or event.authentication_outcome is outcome:
                gap = anchor_ts - to_microseconds(event.event_time)
                return gap / MICROSECONDS_PER_SECOND
        return None

    def last_outcome(history: Sequence[AuthEvent]) -> str | None:
        return None if not history else str(history[-1].authentication_outcome)

    user = prior_for(EntityKind.USER)
    source = prior_for(EntityKind.SOURCE)
    pair = prior_for(EntityKind.USER_SOURCE)

    return {
        "prior_consecutive_user_failures": consecutive_failures(user),
        "prior_consecutive_source_failures": consecutive_failures(source),
        "prior_failures_since_user_success": failures_since_success(user),
        "prior_failures_since_pair_success": failures_since_success(pair),
        "seconds_since_user_previous_event": elapsed_since_last(user),
        "seconds_since_source_previous_event": elapsed_since_last(source),
        "seconds_since_user_previous_success": elapsed_since_last(
            user, AuthOutcome.SUCCESS
        ),
        "seconds_since_user_previous_failure": elapsed_since_last(
            user, AuthOutcome.FAILURE
        ),
        "seconds_since_pair_previous_event": elapsed_since_last(pair),
        "previous_user_outcome": last_outcome(user),
        "previous_pair_outcome": last_outcome(pair),
    }


def reference_columns(catalog: FeatureCatalog) -> tuple[str, ...]:
    """Return the columns the reference implementation covers.

    Every windowed feature plus every sequence feature -- the parts with
    non-trivial temporal bookkeeping.  Calendar and current-event context are
    direct field reads, and geospatial and baseline features have their own
    dedicated tests.
    """
    windowed = tuple(s.name for s in catalog.specs if s.window is not None)
    sequence = tuple(
        s.name
        for s in catalog.specs
        if s.name.startswith(("prior_", "previous_", "seconds_since_"))
        and s.window is None
        and not s.requires_baseline
        and "location" not in s.name
    )
    return windowed + sequence


def reference_rows(
    events: Sequence[AuthEvent], config: FeatureConfig, catalog: FeatureCatalog
) -> list[dict[str, Any]]:
    """Recompute the reference columns for every event, by brute force."""
    ordered = sort_events_canonically(events)
    covered = set(reference_columns(catalog))
    rows: list[dict[str, Any]] = []

    for anchor in ordered:
        anchor_ts = to_microseconds(anchor.event_time)
        row: dict[str, Any] = {}

        for spec in catalog.specs:
            if spec.window is None or spec.name not in covered:
                continue
            kind, measure = _measure_of(spec.name, spec.window)
            width = duration_to_microseconds(parse_duration(spec.window))
            key = _entity_key(kind, anchor)
            lower = anchor_ts - width

            history = [
                e
                for e in ordered
                if _entity_key(kind, e) == key
                and lower <= to_microseconds(e.event_time) < anchor_ts
            ]
            row[spec.name] = _windowed_value(
                measure, history, min_count=spec.min_count or 1
            )

        row.update(_sequence_values(anchor, ordered, anchor_ts))
        rows.append({name: row[name] for name in sorted(covered)})

    return rows
