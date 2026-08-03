"""Point-in-time primitives: timestamp blocks, rolling accumulators, calendar.

This module is domain-free.  It knows about timestamps, windows, and exact
integer aggregation; it knows nothing about authentication.

Two design decisions here carry the whole phase:

**One buffer, several heads.**  An entity keeps a single append-only list of
records covering the longest configured window, plus one
:class:`WindowAccumulator` per window holding an integer index into that list.
Heads and the tail advance monotonically, so each record is appended once and
evicted at most once per window: the engine is O(n * k) for k windows, never
O(n^2).

**Exact integer accumulators.**  Response times are integers and event times
are integer microseconds, so sums and sums-of-squares are accumulated in
Python ``int`` (arbitrary precision) and converted to float only at read time.
This is not primarily about numerical stability: float addition is not
associative, so an incremental accumulator and a recompute-from-buffer would
disagree in the last bits.  Exact integers make the engine bit-for-bit
reproducible, which is what every invariance test in this package relies on.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "MICROSECONDS_PER_SECOND",
    "MISSING_DIM",
    "NULL_RESPONSE_TIME",
    "EntityBuffer",
    "EventRecord",
    "WindowAccumulator",
    "calendar_features",
    "coefficient_of_variation",
    "from_microseconds",
    "iter_timestamp_blocks",
    "mean_std",
    "to_microseconds",
]

MICROSECONDS_PER_SECOND: int = 1_000_000

#: Sentinel for a categorical dimension the source event did not record.  A
#: missing value never contributes to a unique-cardinality count.
MISSING_DIM: int = -1

#: Sentinel for an absent response time.  Absent values are excluded from the
#: mean and standard deviation rather than treated as zero.
NULL_RESPONSE_TIME: int = -1

#: Sentinel for "this entity has no earlier event", used for interarrival gaps.
_NO_GAP: int = -1


def to_microseconds(moment: datetime) -> int:
    """Return *moment* as integer microseconds since the Unix epoch, in UTC.

    Integer microseconds are the canonical internal time representation: they
    make window arithmetic and interarrival sums exact.
    """
    if moment.tzinfo is None:
        raise ValueError("Event times must be timezone-aware")
    utc = moment.astimezone(UTC)
    seconds = int(utc.timestamp())
    return seconds * MICROSECONDS_PER_SECOND + utc.microsecond


def from_microseconds(value: int) -> datetime:
    """Return the UTC datetime for integer microseconds since the epoch."""
    seconds, micros = divmod(value, MICROSECONDS_PER_SECOND)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=micros)


def iter_timestamp_blocks(
    events: Sequence[Any],
) -> Iterator[tuple[int, Sequence[Any]]]:
    """Yield ``(timestamp_microseconds, events)`` for each maximal equal-time run.

    *events* must already be in canonical order (event time ascending, then
    event id ascending).  Grouping simultaneous events into a block is what
    makes same-timestamp mutual exclusion structural rather than a comparison
    the engine has to remember to make: the engine emits every row in a block
    before ingesting any of them, so block members are invisible to each other.

    Raises:
        ValueError: if *events* is not sorted by event time ascending.
    """
    if not events:
        return

    start = 0
    current = to_microseconds(events[0].event_time)
    previous = current

    for index in range(1, len(events)):
        moment = to_microseconds(events[index].event_time)
        if moment < previous:
            raise ValueError(
                "Events must be sorted by event_time ascending before blocking"
            )
        if moment != current:
            yield current, events[start:index]
            start = index
            current = moment
        previous = moment

    yield current, events[start:]


def mean_std(n: int, total: int, total_sq: int) -> tuple[float | None, float | None]:
    """Return the sample mean and standard deviation from exact integer sums.

    Returns ``(None, None)`` for an empty sample and ``(mean, None)`` for a
    single observation: the standard deviation of one value is undefined, not
    zero.  Encoding it as ``0.0`` would be indistinguishable from a genuinely
    constant window.

    The variance numerator ``n * total_sq - total * total`` is an exact integer
    and is mathematically non-negative, so no clamping or epsilon is needed.
    """
    if n <= 0:
        return None, None
    mean = total / n
    if n == 1:
        return mean, None
    numerator = n * total_sq - total * total
    return mean, math.sqrt(numerator / (n * (n - 1)))


def coefficient_of_variation(mean: float | None, std: float | None) -> float | None:
    """Return ``std / mean``, or ``None`` when it is undefined.

    Undefined when there are too few observations for a standard deviation, or
    when the mean is zero.
    """
    if mean is None or std is None or mean == 0.0:
        return None
    return std / mean


@dataclass(slots=True)
class EventRecord:
    """A compact, pre-encoded event held in an entity's rolling buffer.

    Every categorical value is an integer code assigned once before the engine
    loop, so the hot path never hashes a string.
    """

    ts_us: int
    #: Microseconds since this entity's previous event; ``_NO_GAP`` if first.
    gap_us: int
    #: Index into the engine's outcome code table.
    outcome: int
    mfa_failed: bool
    #: Response time in milliseconds, or ``NULL_RESPONSE_TIME`` if absent.
    rt_ms: int
    #: Integer codes for this entity's cardinality dimensions.
    dims: tuple[int, ...]


@dataclass(slots=True)
class WindowAccumulator:
    """Incremental aggregates over one half-open window ``[t - width, t)``.

    ``head`` indexes the oldest in-window record in the owning
    :class:`EntityBuffer`.  It only ever moves forward.
    """

    width_us: int
    n_outcomes: int
    n_dims: int
    track_cardinality: bool = False

    head: int = 0
    n: int = 0
    n_by_outcome: list[int] = field(default_factory=list)
    n_mfa_failed: int = 0

    rt_n: int = 0
    rt_sum: int = 0
    rt_sumsq: int = 0

    #: Interarrival gaps, maintained so that ``ia_n == max(0, n - 1)`` exactly.
    ia_n: int = 0
    ia_sum: int = 0
    ia_sumsq: int = 0

    cardinality: list[dict[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.n_by_outcome:
            self.n_by_outcome = [0] * self.n_outcomes
        if self.track_cardinality and not self.cardinality:
            self.cardinality = [{} for _ in range(self.n_dims)]

    def add(self, record: EventRecord, *, has_in_window_predecessor: bool) -> None:
        """Fold *record* into the aggregates.

        ``has_in_window_predecessor`` says whether the record immediately
        before this one is still inside this window.  If it is, this record's
        gap becomes an in-window interarrival sample; if not, the gap spans the
        window boundary and is not a sample.
        """
        self.n += 1
        self.n_by_outcome[record.outcome] += 1
        if record.mfa_failed:
            self.n_mfa_failed += 1

        if record.rt_ms != NULL_RESPONSE_TIME:
            self.rt_n += 1
            self.rt_sum += record.rt_ms
            self.rt_sumsq += record.rt_ms * record.rt_ms

        if has_in_window_predecessor and record.gap_us != _NO_GAP:
            self.ia_n += 1
            self.ia_sum += record.gap_us
            self.ia_sumsq += record.gap_us * record.gap_us

        if self.track_cardinality:
            for index, value in enumerate(record.dims):
                if value == MISSING_DIM:
                    continue
                counter = self.cardinality[index]
                counter[value] = counter.get(value, 0) + 1

    def remove(self, record: EventRecord, *, next_record: EventRecord | None) -> None:
        """Remove *record*, the current head, from the aggregates.

        *next_record* is the record that follows it in the buffer, if any.  Its
        gap was measured against *record*; once *record* leaves the window that
        gap spans the boundary and stops being an in-window sample.
        """
        self.n -= 1
        self.n_by_outcome[record.outcome] -= 1
        if record.mfa_failed:
            self.n_mfa_failed -= 1

        if record.rt_ms != NULL_RESPONSE_TIME:
            self.rt_n -= 1
            self.rt_sum -= record.rt_ms
            self.rt_sumsq -= record.rt_ms * record.rt_ms

        if next_record is not None and next_record.gap_us != _NO_GAP:
            self.ia_n -= 1
            self.ia_sum -= next_record.gap_us
            self.ia_sumsq -= next_record.gap_us * next_record.gap_us

        if self.track_cardinality:
            for index, value in enumerate(record.dims):
                if value == MISSING_DIM:
                    continue
                counter = self.cardinality[index]
                remaining = counter[value] - 1
                if remaining:
                    counter[value] = remaining
                else:
                    # Deleting on zero is mandatory, not an optimisation:
                    # len(counter) is the unique count, so a fully aged-out
                    # value left behind would be counted forever.
                    del counter[value]

    def unique_count(self, dim: int) -> int:
        """Return the number of distinct values seen in dimension *dim*."""
        if not self.track_cardinality:
            raise ValueError("This window does not track cardinality")
        return len(self.cardinality[dim])

    def response_time_stats(self) -> tuple[float | None, float | None]:
        """Return the in-window response-time mean and standard deviation."""
        return mean_std(self.rt_n, self.rt_sum, self.rt_sumsq)

    def interarrival_stats_seconds(self) -> tuple[float | None, float | None]:
        """Return the in-window interarrival mean and standard deviation, in seconds."""
        mean, std = mean_std(self.ia_n, self.ia_sum, self.ia_sumsq)
        return (
            None if mean is None else mean / MICROSECONDS_PER_SECOND,
            None if std is None else std / MICROSECONDS_PER_SECOND,
        )

    def rate(self, numerator: int, *, min_count: int) -> float | None:
        """Return ``numerator / n``, or ``None`` when the denominator is too small.

        Returning ``None`` rather than ``0.0`` for an empty denominator is
        deliberate: zero attempts is not the same observation as zero failures
        out of many attempts.
        """
        if self.n < min_count or self.n == 0:
            return None
        return numerator / self.n


class EntityBuffer:
    """One entity's rolling history: a shared record list and several windows.

    Records are appended in non-decreasing time order.  Call :meth:`advance`
    with the anchor time before reading any accumulator, so every window holds
    exactly ``[t - width, t)``.
    """

    __slots__ = ("_accumulators", "_records", "last_ts_us")

    #: Compact the shared buffer once this fraction of it has aged out of every
    #: window.  Amortises to O(1) per record.
    _COMPACTION_RATIO: int = 2

    def __init__(
        self,
        window_widths_us: Sequence[int],
        *,
        n_outcomes: int,
        n_dims: int = 0,
        cardinality_windows: frozenset[int] = frozenset(),
    ) -> None:
        self._records: list[EventRecord] = []
        self._accumulators: list[WindowAccumulator] = [
            WindowAccumulator(
                width_us=width,
                n_outcomes=n_outcomes,
                n_dims=n_dims,
                track_cardinality=index in cardinality_windows,
            )
            for index, width in enumerate(window_widths_us)
        ]
        self.last_ts_us: int = -1

    @property
    def accumulators(self) -> list[WindowAccumulator]:
        """Return the per-window accumulators, in configured window order."""
        return self._accumulators

    @property
    def record_count(self) -> int:
        """Return the number of records currently retained in the shared buffer."""
        return len(self._records)

    def append(
        self,
        ts_us: int,
        *,
        outcome: int,
        mfa_failed: bool = False,
        rt_ms: int = NULL_RESPONSE_TIME,
        dims: tuple[int, ...] = (),
    ) -> None:
        """Append one event to this entity's history.

        The interarrival gap is measured against this entity's previous event,
        whether or not that event is still inside any window; each accumulator
        then decides independently whether the gap is an in-window sample.
        """
        gap_us = _NO_GAP if self.last_ts_us < 0 else ts_us - self.last_ts_us
        record = EventRecord(
            ts_us=ts_us,
            gap_us=gap_us,
            outcome=outcome,
            mfa_failed=mfa_failed,
            rt_ms=rt_ms,
            dims=dims,
        )
        index = len(self._records)
        self._records.append(record)
        self.last_ts_us = ts_us

        for accumulator in self._accumulators:
            accumulator.add(record, has_in_window_predecessor=accumulator.head < index)

    def advance(self, t_us: int) -> None:
        """Evict records older than ``t_us - width`` from every window.

        The comparison is strict (``ts < cutoff``), so an event at exactly
        ``t - width`` remains inside the window: the interval is
        ``[t - width, t)``, closed on the left.
        """
        records = self._records
        total = len(records)
        for accumulator in self._accumulators:
            cutoff = t_us - accumulator.width_us
            head = accumulator.head
            while head < total and records[head].ts_us < cutoff:
                accumulator.remove(
                    records[head],
                    next_record=records[head + 1] if head + 1 < total else None,
                )
                head += 1
            accumulator.head = head
        self._compact()

    def _compact(self) -> None:
        """Drop records that have aged out of every window and rebase the heads."""
        if not self._records:
            return
        min_head = min(accumulator.head for accumulator in self._accumulators)
        if min_head == 0 or min_head * self._COMPACTION_RATIO < len(self._records):
            return
        del self._records[:min_head]
        for accumulator in self._accumulators:
            accumulator.head -= min_head

    def oldest_in_window_ts(self, window_index: int) -> int | None:
        """Return the timestamp of the oldest record still inside a window.

        ``None`` when the window is empty.  This value is monotonically
        non-decreasing across :meth:`advance` calls, which is the eviction
        invariant that matters; raw head indices are not, because compaction
        rebases them.
        """
        accumulator = self._accumulators[window_index]
        if accumulator.head >= len(self._records):
            return None
        return self._records[accumulator.head].ts_us

    def is_expired(self, t_us: int, max_width_us: int) -> bool:
        """Return whether this entity's windowed state can be released.

        Windowed state is droppable once the entity's most recent event has
        fallen out of the longest window.  Sequence state has an unbounded
        horizon by definition and is tracked separately.
        """
        return self.last_ts_us >= 0 and self.last_ts_us < t_us - max_width_us


_HOUR_RADIANS: float = 2.0 * math.pi / 24.0
_WEEKDAY_RADIANS: float = 2.0 * math.pi / 7.0


def calendar_features(moment: datetime) -> dict[str, Any]:
    """Return calendar features for *moment*, always evaluated in UTC.

    A user's local timezone is never inferred from their country: a country
    can span several offsets and the mapping is not part of the canonical
    event schema.  Introducing local time would require a trusted per-user
    timezone source, which this phase does not have.
    """
    utc = moment.astimezone(UTC)
    hour = utc.hour
    weekday = utc.weekday()
    return {
        "hour_of_day": hour,
        "day_of_week": weekday,
        "is_weekend": weekday >= 5,
        "hour_sin": math.sin(hour * _HOUR_RADIANS),
        "hour_cos": math.cos(hour * _HOUR_RADIANS),
        "day_of_week_sin": math.sin(weekday * _WEEKDAY_RADIANS),
        "day_of_week_cos": math.cos(weekday * _WEEKDAY_RADIANS),
    }
