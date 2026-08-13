"""Canonical row order for every fit, transform, and prediction in this layer.

One key, used everywhere::

    (anchor_event_time, str(anchor_event_id))

The same key ``features/engine.py`` sorts events by and ``detection/engine.py``
sorts snapshots by.  Reusing it rather than inventing a third ordering is the
point: three orderings would agree on most inputs and diverge on the ones that
matter.

**The guarantee is the trainer's, not the estimators'.**  Several scikit-learn
estimators are sensitive to row order -- ties broken during a tree split,
floating-point accumulation in a summation, the sequence a solver visits samples
in.  Nothing here makes them order-invariant.  What this module does is remove
the *input's* order from the equation: rows arrive from Parquet in whatever
order they were written, get sorted once, and every downstream stage asserts
they are still sorted.  Reproducibility then follows from the canonical order
plus a fixed seed, and the documentation says so in those terms rather than
claiming an invariance the estimators do not have.

Three things this module refuses to do:

* **No** :func:`hash`.  Python's string hash is randomised per process, so any
  ordering derived from it would differ between two runs of the same code.
* **No** locale-sensitive comparison.  ``str`` comparison in Python is by code
  point, and no collation is consulted, so the tie-break is the same on every
  machine.
* **No** wall clock.  Nothing here reads the current time; the only timestamps
  are the anchors' own recorded event times.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from password_attack_detector.exceptions import DataValidationError, ModelTrainingError

__all__ = [
    "AnchoredRow",
    "assert_canonical",
    "canonical_key",
    "canonicalize_rows",
    "is_canonical",
]


@runtime_checkable
class AnchoredRow(Protocol):
    """Anything carrying an anchor identifier and its event time."""

    @property
    def anchor_event_id(self) -> str: ...

    @property
    def anchor_event_time(self) -> datetime: ...


def _validated_time(row: AnchoredRow) -> datetime:
    """Return *row*'s anchor time as a UTC instant, or raise.

    A naive datetime is refused rather than assumed to be UTC.  Assuming would
    silently place a row hours away from where it belongs, and the resulting
    order would be wrong in a way no downstream check could detect.
    """
    value = row.anchor_event_time
    if not isinstance(value, datetime):
        raise DataValidationError(
            f"anchor_event_time must be a datetime, got {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataValidationError(
            "anchor_event_time must be timezone-aware; a naive timestamp has no "
            "position on the timeline this ordering depends on"
        )
    return value.astimezone(UTC)


def canonical_key(row: AnchoredRow) -> tuple[datetime, str]:
    """Return the canonical sort key for *row*.

    The identifier is compared as a string rather than as a UUID: string
    comparison is total, code-point ordered, and identical across platforms,
    whereas ``UUID`` comparison would depend on the type surviving a Parquet
    round trip.
    """
    return (_validated_time(row), str(row.anchor_event_id))


def _reject_duplicate_anchors(rows: Sequence[AnchoredRow]) -> None:
    """Raise when two rows share an anchor identifier.

    Checked **before** sorting, deliberately.  After a sort, duplicates sit next
    to each other and look like a legitimate tie; the sort would succeed and the
    duplicate would ride into a design matrix as two rows describing one event.
    The count is reported, never the identifier.
    """
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        identifier = str(row.anchor_event_id)
        if identifier in seen:
            duplicates += 1
        seen.add(identifier)
    if duplicates:
        raise DataValidationError(
            f"{duplicates} duplicate anchor identifier(s) in {len(rows)} row(s); "
            f"each anchor event must contribute exactly one row"
        )


def canonicalize_rows[RowT: AnchoredRow](rows: Sequence[RowT]) -> tuple[RowT, ...]:
    """Return *rows* in canonical order.

    Duplicate anchors and invalid timestamps raise before any sorting happens,
    so a caller never receives a neatly ordered sequence that was not valid to
    order in the first place.

    Raises:
        DataValidationError: on a duplicate anchor, a non-datetime anchor time,
            or a naive one.
    """
    _reject_duplicate_anchors(rows)
    for row in rows:
        _validated_time(row)
    return tuple(sorted(rows, key=canonical_key))


def is_canonical(rows: Sequence[AnchoredRow]) -> bool:
    """Return whether *rows* are already in canonical order.

    Does not check for duplicates: this answers the ordering question alone, so
    a caller can report the two failures separately rather than conflating an
    unsorted input with an invalid one.
    """
    previous: tuple[datetime, str] | None = None
    for row in rows:
        current = canonical_key(row)
        if previous is not None and current < previous:
            return False
        previous = current
    return True


def assert_canonical(rows: Sequence[AnchoredRow], *, stage: str) -> None:
    """Raise unless *rows* are canonically ordered and uniquely anchored.

    The reusable entry-point guard for every later milestone: preprocessing fit
    and transform, class weighting, each ``ModelAdapter.fit``, calibration,
    out-of-fold construction, threshold selection, and prediction all call this
    on entry.  Re-asserting at each stage rather than trusting the loader means
    a stage that reorders rows internally is caught by the next one.

    Args:
        rows: the rows about to be consumed.
        stage: what is about to consume them, named in the error message so a
            failure says where the order was lost.

    Raises:
        ModelTrainingError: if the rows are unordered or carry a duplicate
            anchor.
    """
    try:
        _reject_duplicate_anchors(rows)
    except DataValidationError as exc:
        raise ModelTrainingError(f"{stage} received invalid rows: {exc}") from None

    if not is_canonical(rows):
        raise ModelTrainingError(
            f"{stage} received {len(rows)} row(s) that are not in canonical "
            f"order; rows must be sorted by anchor event time, then by anchor "
            f"identifier"
        )
