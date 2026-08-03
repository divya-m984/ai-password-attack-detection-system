"""The point-in-time feature engine.

The engine walks canonical authentication events once, in canonical order, and
emits one feature snapshot per event.

**The timing contract is enforced structurally, not by comparisons.**  Events
are grouped into *timestamp blocks* -- maximal runs sharing an event time.  For
each block the engine runs two strictly separated phases:

1. **Emit.** Every event in the block produces its row by reading entity state.
   State is read-only in this phase.
2. **Ingest.** Only after every row in the block has been emitted does any
   event in the block update state.

Because blocks are processed in ascending time order, the invariant "entity
state contains only events strictly earlier than the current block" holds at
the top of every iteration.  It follows that:

* the anchor never enters its own history;
* events sharing the anchor's timestamp never enter it either, in either
  direction -- simultaneous events are mutually invisible;
* adding or modifying any event after time ``t`` cannot change any feature for
  an anchor at or before ``t``.

The ``event_id`` tie-break in the canonical sort governs **output row order
only**.  It never affects state content, so it cannot let simultaneous events
see one another.

See ``docs/temporal-semantics.md`` for the contract in prose.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from password_attack_detector.data.enums import AuthOutcome, MFAOutcome
from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.exceptions import FeatureComputationError
from password_attack_detector.features.catalog import (
    _ENTITY_PREFIX,
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    FeatureCatalog,
    LeakageClass,
    build_catalog,
)
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    AggregateKind,
    EntityKind,
    FeatureConfig,
)
from password_attack_detector.features.geospatial import (
    previous_success_geo,
    valid_coordinates,
)
from password_attack_detector.features.temporal import (
    MICROSECONDS_PER_SECOND,
    MISSING_DIM,
    NULL_RESPONSE_TIME,
    EntityBuffer,
    calendar_features,
    coefficient_of_variation,
    from_microseconds,
    iter_timestamp_blocks,
)

__all__ = [
    "BaselineTransform",
    "EngineStats",
    "FeatureEngine",
    "FeatureFrame",
    "sort_events_canonically",
]


# --- outcome coding --------------------------------------------------------

#: Fixed integer codes for authentication outcomes.  Coding categoricals to
#: integers before the loop keeps the hot path free of string hashing.
_OUTCOME_ORDER: tuple[AuthOutcome, ...] = (
    AuthOutcome.SUCCESS,
    AuthOutcome.FAILURE,
    AuthOutcome.BLOCKED,
    AuthOutcome.CHALLENGED,
)
_OUTCOME_CODE: dict[AuthOutcome, int] = {
    outcome: index for index, outcome in enumerate(_OUTCOME_ORDER)
}
_SUCCESS = _OUTCOME_CODE[AuthOutcome.SUCCESS]
_FAILURE = _OUTCOME_CODE[AuthOutcome.FAILURE]
_N_OUTCOMES = len(_OUTCOME_ORDER)


# --- entity configuration --------------------------------------------------

#: Entity kinds the engine maintains rolling history for, in a fixed order.
_TRACKED_ENTITIES: tuple[EntityKind, ...] = (
    EntityKind.USER,
    EntityKind.SOURCE,
    EntityKind.USER_SOURCE,
    EntityKind.USER_DEVICE,
    EntityKind.SESSION,
    EntityKind.DEVICE,
)

#: Cardinality dimensions tracked per entity kind, in accumulator order.
_ENTITY_DIMS: dict[EntityKind, tuple[str, ...]] = {
    EntityKind.USER: ("source", "device", "country", "application", "auth_method"),
    EntityKind.SOURCE: ("user", "device", "country", "application", "user_agent"),
    EntityKind.DEVICE: ("user",),
    EntityKind.USER_SOURCE: (),
    EntityKind.USER_DEVICE: (),
    EntityKind.SESSION: (),
}

#: Entity kinds whose sequence state the engine maintains.
_SEQUENCE_ENTITIES: tuple[EntityKind, ...] = (
    EntityKind.USER,
    EntityKind.SOURCE,
    EntityKind.USER_SOURCE,
)


# --- read operations -------------------------------------------------------

_OP_COUNT_TOTAL = 0
_OP_COUNT_OUTCOME = 1
_OP_COUNT_MFA = 2
_OP_RATE = 3
_OP_UNIQUE = 4
_OP_RT_MEAN = 5
_OP_RT_STD = 6
_OP_IA_MEAN = 7
_OP_IA_STD = 8
_OP_IA_COV = 9

#: Maps a catalog measure name onto a read operation and its argument.  The
#: argument is an outcome code for counts and rates, and a dimension name for
#: unique counts (resolved to an index per entity kind).
_MEASURE_OPS: dict[str, tuple[int, Any]] = {
    "attempt_count": (_OP_COUNT_TOTAL, 0),
    "event_count": (_OP_COUNT_TOTAL, 0),
    "success_count": (_OP_COUNT_OUTCOME, AuthOutcome.SUCCESS),
    "failure_count": (_OP_COUNT_OUTCOME, AuthOutcome.FAILURE),
    "blocked_count": (_OP_COUNT_OUTCOME, AuthOutcome.BLOCKED),
    "challenge_count": (_OP_COUNT_OUTCOME, AuthOutcome.CHALLENGED),
    "mfa_failure_count": (_OP_COUNT_MFA, 0),
    "failure_rate": (_OP_RATE, AuthOutcome.FAILURE),
    "success_rate": (_OP_RATE, AuthOutcome.SUCCESS),
    "unique_source_count": (_OP_UNIQUE, "source"),
    "unique_device_count": (_OP_UNIQUE, "device"),
    "unique_country_count": (_OP_UNIQUE, "country"),
    "unique_application_count": (_OP_UNIQUE, "application"),
    "unique_auth_method_count": (_OP_UNIQUE, "auth_method"),
    "unique_user_count": (_OP_UNIQUE, "user"),
    "unique_user_agent_count": (_OP_UNIQUE, "user_agent"),
    "user_count": (_OP_UNIQUE, "user"),
    "mean_response_time_ms": (_OP_RT_MEAN, 0),
    "response_time_std_ms": (_OP_RT_STD, 0),
    "mean_interarrival_seconds": (_OP_IA_MEAN, 0),
    "interarrival_std_seconds": (_OP_IA_STD, 0),
    "interarrival_coefficient_of_variation": (_OP_IA_COV, 0),
}


class BaselineTransform(Protocol):
    """The minimal surface the engine needs from a fitted behavioral baseline.

    Declaring it structurally keeps the engine independent of the baselines
    module, so the two can be tested and reasoned about separately.
    """

    def transform_one(
        self, event: AuthEvent, features: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return the baseline-derived columns for one event, without mutating.

        *features* is the point-in-time row built so far, so ratio features can
        compare an observed windowed count against a fitted rate without ever
        looking outside ``[t - window, t)``.
        """
        ...


@dataclass(frozen=True, slots=True)
class _Instruction:
    """A precompiled instruction for reading one windowed feature."""

    name: str
    window_index: int
    op: int
    arg: int
    min_count: int


# --- per-entity non-windowed state ----------------------------------------


@dataclass(slots=True)
class _SequenceState:
    """Strictly-prior sequence state for one entity.

    Updated only during the ingest phase, so a field named ``prior_*`` can
    never observe the anchor's own outcome.
    """

    last_ts_us: int = -1
    last_success_ts_us: int = -1
    last_failure_ts_us: int = -1
    consecutive_failures: int = 0
    failures_since_success: int = 0
    previous_outcome: int = -1

    def update(self, ts_us: int, outcome: int) -> None:
        """Fold an event into the sequence state.

        ``consecutive_failures`` counts an unbroken run of FAILURE outcomes;
        any other outcome, including BLOCKED, ends the run.
        ``failures_since_success`` resets only on SUCCESS.
        """
        if outcome == _FAILURE:
            self.consecutive_failures += 1
            self.failures_since_success += 1
            self.last_failure_ts_us = ts_us
        else:
            self.consecutive_failures = 0
            if outcome == _SUCCESS:
                self.failures_since_success = 0
                self.last_success_ts_us = ts_us
        self.previous_outcome = outcome
        self.last_ts_us = ts_us


@dataclass(slots=True)
class _GeoState:
    """The most recent successful authentication for one user."""

    #: Most recent success of any kind, used for country and region change.
    success_ts_us: int = -1
    success_country: str | None = None
    success_region: str | None = None

    #: Most recent success that carried usable coordinates.
    located_ts_us: int = -1
    located_latitude: float | None = None
    located_longitude: float | None = None

    def update(
        self,
        ts_us: int,
        *,
        country: str | None,
        region: str | None,
        latitude: float | None,
        longitude: float | None,
    ) -> None:
        """Record a successful authentication."""
        self.success_ts_us = ts_us
        self.success_country = country
        self.success_region = region
        if valid_coordinates(latitude, longitude):
            self.located_ts_us = ts_us
            self.located_latitude = latitude
            self.located_longitude = longitude


@dataclass(slots=True)
class EngineStats:
    """Observable counters describing one engine run.

    These are the basis for the structural complexity tests: they assert the
    engine's asymptotic behaviour without ever asserting wall-clock time.
    """

    rows_emitted: int = 0
    events_ingested: int = 0
    blocks_processed: int = 0
    buffer_appends: int = 0
    buffer_evictions: int = 0
    tracked_entity_count: int = 0
    peak_tracked_entity_count: int = 0
    expired_entity_count: int = 0
    capacity_evicted_entity_count: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain mapping, for reporting."""
        return {
            "rows_emitted": self.rows_emitted,
            "events_ingested": self.events_ingested,
            "blocks_processed": self.blocks_processed,
            "buffer_appends": self.buffer_appends,
            "buffer_evictions": self.buffer_evictions,
            "tracked_entity_count": self.tracked_entity_count,
            "peak_tracked_entity_count": self.peak_tracked_entity_count,
            "expired_entity_count": self.expired_entity_count,
            "capacity_evicted_entity_count": self.capacity_evicted_entity_count,
        }


@dataclass(frozen=True)
class FeatureFrame:
    """The result of one engine run: rows in catalog column order."""

    rows: tuple[dict[str, Any], ...]
    catalog: FeatureCatalog
    stats: EngineStats

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> list[Any]:
        """Return one column's values across all rows, in row order."""
        if not self.catalog.has(name):
            raise FeatureComputationError(f"Undeclared feature column: {name!r}")
        return [row[name] for row in self.rows]

    def row_for(self, event_id: Any) -> dict[str, Any]:
        """Return the row whose anchor is *event_id*."""
        target = str(event_id)
        for row in self.rows:
            if row[ANCHOR_EVENT_ID] == target:
                return row
        raise FeatureComputationError("No feature row for the requested anchor")


def sort_events_canonically(events: Iterable[AuthEvent]) -> list[AuthEvent]:
    """Return events in canonical order: event time ascending, then event id.

    The ``event_id`` tie-break makes output row order deterministic under
    input reordering.  It has no effect on feature values: simultaneous events
    are grouped into one block and none of them contributes to another's
    history regardless of id.
    """
    return sorted(events, key=lambda e: (e.event_time, str(e.event_id)))


class FeatureEngine:
    """Computes point-in-time feature snapshots from canonical events.

    The engine holds no module-level state, reads no environment, and never
    consults the working directory.  Two engines built from the same config
    and given the same events produce byte-identical output.
    """

    def __init__(
        self,
        config: FeatureConfig,
        catalog: FeatureCatalog | None = None,
        *,
        baseline: BaselineTransform | None = None,
    ) -> None:
        self._config = config
        self._catalog = catalog if catalog is not None else build_catalog(config)
        self._baseline = baseline
        self._column_order = self._catalog.column_order()

        self._window_labels: dict[EntityKind, tuple[str, ...]] = {}
        self._window_widths: dict[EntityKind, tuple[int, ...]] = {}
        self._cardinality_indices: dict[EntityKind, frozenset[int]] = {}
        self._plans: dict[EntityKind, tuple[_Instruction, ...]] = {}
        self._compile()

        self._max_window_us = config.max_window_microseconds
        self._baseline_columns = tuple(
            spec.name
            for spec in self._catalog.specs
            if spec.leakage_class is LeakageClass.BASELINE_DERIVED
        )

    # -- compilation --------------------------------------------------------

    def _entity_windows(self, kind: EntityKind) -> tuple[str, ...]:
        """Return the windows this entity kind maintains accumulators for.

        This is the union of the windows any of its features needs, so a single
        buffer serves every aggregate for the entity.
        """
        labels: set[str] = set()
        for aggregate in AggregateKind:
            labels.update(self._config.windows_for(kind, aggregate))
        return tuple(w for w in self._config.windows if w in labels)

    def _compile(self) -> None:
        """Precompile the per-entity read plans from the catalog.

        Deriving the plan from the catalog is what keeps the engine and the
        declared schema in step: an undeclared measure fails here, at
        construction, rather than producing a silently misnamed column.
        """
        widths = dict(self._config.window_microseconds())

        for kind in _TRACKED_ENTITIES:
            labels = self._entity_windows(kind)
            self._window_labels[kind] = labels
            self._window_widths[kind] = tuple(widths[label] for label in labels)
            self._cardinality_indices[kind] = frozenset(
                index
                for index, label in enumerate(labels)
                if label in self._config.cardinality_windows
            )

            dims = _ENTITY_DIMS[kind]
            instructions: list[_Instruction] = []
            for spec in self._catalog.specs_for_entity(kind):
                if spec.window is None:
                    continue
                measure = self._measure_of(spec.name, kind, spec.window)
                try:
                    op, raw_arg = _MEASURE_OPS[measure]
                except KeyError:
                    raise FeatureComputationError(
                        f"Catalog declares {spec.name!r}, but the engine has no "
                        f"implementation for measure {measure!r}"
                    ) from None

                if op == _OP_UNIQUE:
                    if raw_arg not in dims:
                        raise FeatureComputationError(
                            f"Feature {spec.name!r} needs dimension {raw_arg!r}, "
                            f"which {kind} does not track"
                        )
                    arg = dims.index(raw_arg)
                elif op in (_OP_COUNT_OUTCOME, _OP_RATE):
                    arg = _OUTCOME_CODE[raw_arg]
                else:
                    arg = 0

                instructions.append(
                    _Instruction(
                        name=spec.name,
                        window_index=labels.index(spec.window),
                        op=op,
                        arg=arg,
                        min_count=spec.min_count or 1,
                    )
                )
            self._plans[kind] = tuple(instructions)

    @staticmethod
    def _measure_of(name: str, kind: EntityKind, window: str) -> str:
        """Strip the entity prefix and window suffix from a feature name."""
        prefix = _ENTITY_PREFIX[kind]
        return name[len(prefix) + 1 : -(len(window) + 2)]

    # -- entity keys and dimensions ----------------------------------------

    @staticmethod
    def _entity_key(kind: EntityKind, event: AuthEvent) -> Hashable:
        """Return the state key for one entity kind."""
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

    def _dims_for(self, kind: EntityKind, event: AuthEvent) -> tuple[int, ...]:
        """Return the integer dimension codes for one entity kind.

        Absent categorical values become ``MISSING_DIM`` and never contribute
        to a unique count: "no country recorded" is not a distinct country.
        """
        if kind is EntityKind.USER:
            return (
                self._code(event.source_id),
                self._code(event.device_id),
                self._code(event.country_code),
                self._code(event.application_id),
                self._code(str(event.authentication_method)),
            )
        if kind is EntityKind.SOURCE:
            return (
                self._code(event.user_id),
                self._code(event.device_id),
                self._code(event.country_code),
                self._code(event.application_id),
                self._code(event.user_agent_family),
            )
        if kind is EntityKind.DEVICE:
            return (self._code(event.user_id),)
        return ()

    def _code(self, value: str | None) -> int:
        """Intern a categorical value to a stable integer code."""
        if value is None:
            return MISSING_DIM
        codes = self._codes
        code = codes.get(value)
        if code is None:
            code = len(codes)
            codes[value] = code
        return code

    # -- reading ------------------------------------------------------------

    @staticmethod
    def _read(buffer: EntityBuffer, instruction: _Instruction) -> Any:
        """Execute one precompiled read against an entity's accumulators."""
        accumulator = buffer.accumulators[instruction.window_index]
        op = instruction.op

        if op == _OP_COUNT_TOTAL:
            return accumulator.n
        if op == _OP_COUNT_OUTCOME:
            return accumulator.n_by_outcome[instruction.arg]
        if op == _OP_COUNT_MFA:
            return accumulator.n_mfa_failed
        if op == _OP_RATE:
            return accumulator.rate(
                accumulator.n_by_outcome[instruction.arg],
                min_count=instruction.min_count,
            )
        if op == _OP_UNIQUE:
            return accumulator.unique_count(instruction.arg)
        if op == _OP_RT_MEAN:
            return accumulator.response_time_stats()[0]
        if op == _OP_RT_STD:
            return accumulator.response_time_stats()[1]
        if op == _OP_IA_MEAN:
            return accumulator.interarrival_stats_seconds()[0]
        if op == _OP_IA_STD:
            return accumulator.interarrival_stats_seconds()[1]
        mean, std = accumulator.interarrival_stats_seconds()
        return coefficient_of_variation(mean, std)

    # -- the main loop ------------------------------------------------------

    def run(self, events: Sequence[AuthEvent]) -> FeatureFrame:
        """Compute one feature snapshot per event.

        *events* are sorted canonically before processing, so the caller may
        pass them in any order and receive identical output.

        The engine deliberately processes the **entire** stream, ignoring any
        split assignment.  Isolation between splits is achieved by purging
        events whose lookback windows cross a boundary, never by resetting
        engine state: a reset would manufacture cold-start artifacts at the
        start of validation and test that do not occur in production.
        """
        ordered = sort_events_canonically(events)
        self._reset()

        rows: list[dict[str, Any]] = []
        stats = self._stats

        for ts_us, block in iter_timestamp_blocks(ordered):
            stats.blocks_processed += 1

            # Phase 1 -- EMIT.  State is read-only here, and still contains
            # only events strictly earlier than ts_us.
            for event in block:
                rows.append(self._emit(event, ts_us))

            # Phase 2 -- INGEST.  Only now may block members touch state, so
            # they were invisible to one another during emission.
            for event in block:
                self._ingest(event, ts_us)

            self._expire(ts_us)

        stats.rows_emitted = len(rows)
        stats.tracked_entity_count = sum(len(d) for d in self._buffers.values())
        return FeatureFrame(rows=tuple(rows), catalog=self._catalog, stats=stats)

    def _reset(self) -> None:
        """Clear all per-run state so an engine instance can be reused safely."""
        self._codes: dict[str, int] = {}
        self._buffers: dict[EntityKind, OrderedDict[Hashable, EntityBuffer]] = {
            kind: OrderedDict() for kind in _TRACKED_ENTITIES
        }
        self._sequences: dict[EntityKind, OrderedDict[Hashable, _SequenceState]] = {
            kind: OrderedDict() for kind in _SEQUENCE_ENTITIES
        }
        self._geo: dict[Hashable, _GeoState] = {}
        self._touch_queue: deque[tuple[int, EntityKind, Hashable]] = deque()
        self._last_seen: dict[tuple[EntityKind, Hashable], int] = {}
        self._stats = EngineStats()

    def _buffer_for(self, kind: EntityKind, key: Hashable) -> EntityBuffer:
        """Return this entity's buffer, creating an empty one on first sight.

        A freshly created buffer reports zero counts and null statistics
        through the ordinary read path, so cold entities need no separate code
        path that could drift from the declared schema.
        """
        buffers = self._buffers[kind]
        buffer = buffers.get(key)
        if buffer is None:
            buffer = EntityBuffer(
                self._window_widths[kind],
                n_outcomes=_N_OUTCOMES,
                n_dims=len(_ENTITY_DIMS[kind]),
                cardinality_windows=self._cardinality_indices[kind],
            )
            buffers[key] = buffer
            self._stats.peak_tracked_entity_count = max(
                self._stats.peak_tracked_entity_count,
                sum(len(d) for d in self._buffers.values()),
            )
        return buffer

    def _sequence_for(self, kind: EntityKind, key: Hashable) -> _SequenceState:
        """Return this entity's sequence state, creating it on first sight."""
        sequences = self._sequences[kind]
        state = sequences.get(key)
        if state is None:
            state = _SequenceState()
            sequences[key] = state
        return state

    def _emit(self, event: AuthEvent, ts_us: int) -> dict[str, Any]:
        """Build one feature row.  Reads state; never writes it."""
        row: dict[str, Any] = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            ANCHOR_EVENT_ID: str(event.event_id),
            ANCHOR_EVENT_TIME: from_microseconds(ts_us),
        }
        row.update(self._current_context(event))
        row.update(calendar_features(event.event_time))

        for kind in _TRACKED_ENTITIES:
            plan = self._plans[kind]
            if not plan:
                continue
            buffer = self._buffer_for(kind, self._entity_key(kind, event))
            buffer.advance(ts_us)
            for instruction in plan:
                row[instruction.name] = self._read(buffer, instruction)

        row.update(self._sequence_features(event, ts_us))
        if self._config.geospatial.enabled:
            row.update(self._geospatial_features(event, ts_us))
        if self._baseline_columns:
            # Passed the row built so far: baseline ratio features compare an
            # observed windowed count against a fitted rate, and those counts
            # are themselves strictly point-in-time.
            row.update(self._baseline_features(event, row))

        return self._project(row)

    def _project(self, row: dict[str, Any]) -> dict[str, Any]:
        """Reorder a row into catalog column order, rejecting any drift.

        A missing or undeclared key is a hard error rather than a silently
        different schema: the catalog is the single source of truth, and the
        engine must agree with it exactly.
        """
        missing = [name for name in self._column_order if name not in row]
        if missing:
            raise FeatureComputationError(
                f"Engine produced no value for {len(missing)} declared "
                f"feature(s), starting with {missing[0]!r}"
            )
        extra = sorted(set(row) - set(self._column_order))
        if extra:
            raise FeatureComputationError(
                f"Engine produced {len(extra)} undeclared column(s), starting "
                f"with {extra[0]!r}"
            )
        return {name: row[name] for name in self._column_order}

    @staticmethod
    def _current_context(event: AuthEvent) -> dict[str, Any]:
        """Return the anchor event's own recorded fields.

        These are legitimate model inputs because detection runs after the
        event has completed and been recorded.  They are the only place the
        anchor's own values appear.
        """
        return {
            "current_authentication_outcome": str(event.authentication_outcome),
            "current_authentication_method": str(event.authentication_method),
            "current_mfa_outcome": (
                None if event.mfa_outcome is None else str(event.mfa_outcome)
            ),
            "current_client_type": (
                None if event.client_type is None else str(event.client_type)
            ),
            "current_response_time_ms": event.response_time_ms,
            "current_country_code": event.country_code,
            "current_has_device": bool(event.device_id),
            "current_has_location": valid_coordinates(
                event.coarse_latitude, event.coarse_longitude
            ),
        }

    def _sequence_features(self, event: AuthEvent, ts_us: int) -> dict[str, Any]:
        """Return strictly-prior sequence features for the anchor."""
        user = self._sequence_for(EntityKind.USER, event.user_id)
        source = self._sequence_for(EntityKind.SOURCE, event.source_id)
        pair = self._sequence_for(
            EntityKind.USER_SOURCE, (event.user_id, event.source_id)
        )

        def elapsed(since_us: int) -> float | None:
            if since_us < 0:
                return None
            return (ts_us - since_us) / MICROSECONDS_PER_SECOND

        def outcome_name(code: int) -> str | None:
            return None if code < 0 else str(_OUTCOME_ORDER[code])

        return {
            "prior_consecutive_user_failures": user.consecutive_failures,
            "prior_consecutive_source_failures": source.consecutive_failures,
            "prior_failures_since_user_success": user.failures_since_success,
            "prior_failures_since_pair_success": pair.failures_since_success,
            "seconds_since_user_previous_event": elapsed(user.last_ts_us),
            "seconds_since_source_previous_event": elapsed(source.last_ts_us),
            "seconds_since_user_previous_success": elapsed(user.last_success_ts_us),
            "seconds_since_user_previous_failure": elapsed(user.last_failure_ts_us),
            "seconds_since_pair_previous_event": elapsed(pair.last_ts_us),
            "previous_user_outcome": outcome_name(user.previous_outcome),
            "previous_pair_outcome": outcome_name(pair.previous_outcome),
        }

    def _geospatial_features(self, event: AuthEvent, ts_us: int) -> dict[str, Any]:
        """Return coarse-location features relative to the user's last success."""
        state = self._geo.get(event.user_id)
        geospatial = self._config.geospatial

        if state is None:
            result = previous_success_geo(
                current_latitude=event.coarse_latitude,
                current_longitude=event.coarse_longitude,
                prior_latitude=None,
                prior_longitude=None,
                prior_elapsed_seconds=None,
                has_prior_success=False,
                max_plausible_kmh=geospatial.max_plausible_velocity_kmh,
                min_distance_km=geospatial.min_distance_km_for_velocity,
            )
            country_changed: bool | None = None
            region_changed: bool | None = None
            elapsed_located: float | None = None
        else:
            elapsed_located = (
                None
                if state.located_ts_us < 0
                else (ts_us - state.located_ts_us) / MICROSECONDS_PER_SECOND
            )
            result = previous_success_geo(
                current_latitude=event.coarse_latitude,
                current_longitude=event.coarse_longitude,
                prior_latitude=state.located_latitude,
                prior_longitude=state.located_longitude,
                prior_elapsed_seconds=elapsed_located,
                has_prior_success=state.success_ts_us >= 0,
                max_plausible_kmh=geospatial.max_plausible_velocity_kmh,
                min_distance_km=geospatial.min_distance_km_for_velocity,
            )
            country_changed = (
                None
                if state.success_country is None or event.country_code is None
                else state.success_country != event.country_code
            )
            region_changed = (
                None
                if state.success_region is None or event.region_code is None
                else state.success_region != event.region_code
            )

        return {
            "distance_km_from_user_previous_success": result.distance_km,
            "seconds_since_user_previous_success_with_location": elapsed_located,
            "implied_velocity_kmh_from_previous_success": result.velocity_kmh,
            "country_changed_since_previous_success": country_changed,
            "region_changed_since_previous_success": region_changed,
            "user_previous_success_geo__status": str(result.status),
            "implied_velocity__status": str(result.velocity_status),
        }

    def _baseline_features(
        self, event: AuthEvent, features: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return baseline-derived features, or nulls when no baseline is fitted.

        With no baseline, every derived column is null and both coverage flags
        are false.  A cold entity is never reported as having a *new* device:
        that would conflate "never seen this entity" with "know this entity,
        and this value is new".
        """
        if self._baseline is None:
            values: dict[str, Any] = dict.fromkeys(self._baseline_columns)
            values["user_in_baseline"] = False
            values["source_in_baseline"] = False
            return values
        return self._baseline.transform_one(event, features)

    def _ingest(self, event: AuthEvent, ts_us: int) -> None:
        """Fold one event into entity state, after its row has been emitted."""
        outcome = _OUTCOME_CODE[event.authentication_outcome]
        mfa_failed = event.mfa_outcome is MFAOutcome.FAILED
        rt_ms = (
            NULL_RESPONSE_TIME
            if event.response_time_ms is None
            else event.response_time_ms
        )

        for kind in _TRACKED_ENTITIES:
            if not self._plans[kind]:
                continue
            key = self._entity_key(kind, event)
            buffer = self._buffer_for(kind, key)
            buffer.append(
                ts_us,
                outcome=outcome,
                mfa_failed=mfa_failed,
                rt_ms=rt_ms,
                dims=self._dims_for(kind, event),
            )
            self._stats.buffer_appends += 1
            self._touch(kind, key, ts_us)

        for kind in _SEQUENCE_ENTITIES:
            self._sequence_for(kind, self._entity_key(kind, event)).update(
                ts_us, outcome
            )

        if outcome == _SUCCESS:
            state = self._geo.get(event.user_id)
            if state is None:
                state = _GeoState()
                self._geo[event.user_id] = state
            state.update(
                ts_us,
                country=event.country_code,
                region=event.region_code,
                latitude=event.coarse_latitude,
                longitude=event.coarse_longitude,
            )

        self._stats.events_ingested += 1

    # -- bounded state ------------------------------------------------------

    def _touch(self, kind: EntityKind, key: Hashable, ts_us: int) -> None:
        """Record that an entity was seen, for expiry and capacity management."""
        self._touch_queue.append((ts_us, kind, key))
        self._last_seen[(kind, key)] = ts_us
        if self._config.max_tracked_entities is not None:
            self._buffers[kind].move_to_end(key)

    def _expire(self, ts_us: int) -> None:
        """Release windowed state for entities that have aged out.

        Uses a lazy expiry queue with a stale-entry guard, so the cost is
        amortised O(1) per event rather than an O(entities) sweep per block.
        Sequence state has an unbounded horizon by definition and is retained;
        it is bounded only by ``max_tracked_entities``.
        """
        horizon = ts_us - self._max_window_us
        queue = self._touch_queue
        while queue and queue[0][0] < horizon:
            seen_at, kind, key = queue.popleft()
            if self._last_seen.get((kind, key)) != seen_at:
                continue  # the entity was seen again later; this entry is stale
            buffer = self._buffers[kind].pop(key, None)
            if buffer is not None:
                self._stats.buffer_evictions += buffer.record_count
                self._stats.expired_entity_count += 1
            self._last_seen.pop((kind, key), None)

        cap = self._config.max_tracked_entities
        if cap is None:
            return
        for kind in _TRACKED_ENTITIES:
            buffers = self._buffers[kind]
            while len(buffers) > cap:
                evicted_key, evicted = buffers.popitem(last=False)
                self._stats.buffer_evictions += evicted.record_count
                self._stats.capacity_evicted_entity_count += 1
                self._last_seen.pop((kind, evicted_key), None)
                if kind in self._sequences:
                    self._sequences[kind].pop(evicted_key, None)
