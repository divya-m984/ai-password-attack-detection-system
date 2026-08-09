"""Canonical row ordering: the one property every later stage rests on."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import pytest

from password_attack_detector.exceptions import DataValidationError, ModelTrainingError
from password_attack_detector.ml.ordering import (
    assert_canonical,
    canonical_key,
    canonicalize_rows,
    is_canonical,
)
from tests.ml.factories import EPOCH, anchor_id, at


@dataclass(frozen=True, slots=True)
class Row:
    """A minimal anchored row."""

    anchor_event_id: str
    anchor_event_time: datetime


def rows(*pairs: tuple[str, datetime]) -> list[Row]:
    """Return rows from ``(identifier, time)`` pairs."""
    return [Row(anchor_event_id=name, anchor_event_time=when) for name, when in pairs]


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_the_key_is_time_then_identifier() -> None:
    """Stated as a fact, so a later edit that reorders it fails here."""
    row = Row(anchor_event_id="e0007", anchor_event_time=at(3))
    assert canonical_key(row) == (at(3), "e0007")


def test_the_identifier_is_compared_as_a_string() -> None:
    """String comparison is total and platform-independent; UUID typing is not."""
    ordered = canonicalize_rows(rows(("e0010", at(0)), ("e0002", at(0))))
    assert [row.anchor_event_id for row in ordered] == ["e0002", "e0010"]


def test_equivalent_instants_in_different_zones_order_together() -> None:
    """Ordering is by instant, not by the offset a timestamp was written in."""
    utc = Row(anchor_event_id="b", anchor_event_time=EPOCH + timedelta(hours=1))
    plus_two = Row(
        anchor_event_id="a",
        anchor_event_time=(EPOCH + timedelta(hours=1)).astimezone(
            timezone(timedelta(hours=2))
        ),
    )
    ordered = canonicalize_rows([utc, plus_two])
    assert [row.anchor_event_id for row in ordered] == ["a", "b"]


# ---------------------------------------------------------------------------
# Order invariance of the input
# ---------------------------------------------------------------------------


def test_input_file_order_does_not_matter() -> None:
    """The whole point: rows arrive in write order and leave in canonical order."""
    original = [Row(anchor_id(index), at(index)) for index in range(12)]
    shuffled = list(original)
    random.Random(20260807).shuffle(shuffled)
    assert canonicalize_rows(shuffled) == tuple(original)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_every_shuffle_produces_the_same_order(seed: int) -> None:
    """Not one lucky permutation: any permutation lands in the same place."""
    original = [Row(anchor_id(index), at(index // 3)) for index in range(15)]
    shuffled = list(original)
    random.Random(seed).shuffle(shuffled)
    assert canonicalize_rows(shuffled) == canonicalize_rows(original)


def test_exact_timestamp_ties_break_on_the_identifier() -> None:
    """Simultaneous events still have exactly one canonical order."""
    tied = rows(("e0003", at(5)), ("e0001", at(5)), ("e0002", at(5)))
    ordered = canonicalize_rows(tied)
    assert [row.anchor_event_id for row in ordered] == ["e0001", "e0002", "e0003"]


def test_a_tie_is_stable_across_input_orders() -> None:
    """A tie-break that depended on input order would not be one."""
    forward = rows(("e0001", at(5)), ("e0002", at(5)))
    backward = list(reversed(forward))
    assert canonicalize_rows(forward) == canonicalize_rows(backward)


def test_all_rows_tied_at_one_instant_order_by_identifier() -> None:
    """The degenerate case: a single instant, ordered entirely by the tie-break."""
    tied = [Row(anchor_id(index), EPOCH) for index in reversed(range(8))]
    ordered = canonicalize_rows(tied)
    assert [row.anchor_event_id for row in ordered] == [
        anchor_id(index) for index in range(8)
    ]


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_duplicate_anchors_fail_before_sorting() -> None:
    """After a sort, duplicates look like a legitimate tie. So they are caught first."""
    with pytest.raises(DataValidationError, match="duplicate anchor"):
        canonicalize_rows(rows(("e0001", at(1)), ("e0001", at(2))))


def test_the_duplicate_message_reports_a_count_not_an_identifier() -> None:
    """A validation message is not a place to publish an event identifier."""
    with pytest.raises(DataValidationError) as caught:
        canonicalize_rows(rows(("e0001", at(1)), ("e0001", at(2))))
    assert "e0001" not in str(caught.value)


def test_a_naive_timestamp_is_refused_rather_than_assumed_utc() -> None:
    """Assuming would place a row hours from where it belongs, undetectably."""
    naive = [Row("e0001", datetime(2026, 3, 1, 0, 0, 0))]
    with pytest.raises(DataValidationError, match="timezone-aware"):
        canonicalize_rows(naive)


def test_a_non_datetime_timestamp_is_refused() -> None:
    """A string that looks like a time is not a time."""
    bad = [Row("e0001", "2026-03-01T00:00:00Z")]  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="must be a datetime"):
        canonicalize_rows(bad)


def test_invalid_timestamps_are_rejected_even_when_already_sorted() -> None:
    """Validity is checked independently of whether a sort would have succeeded."""
    naive = [
        Row("e0001", datetime(2026, 3, 1, tzinfo=UTC)),
        Row("e0002", datetime(2026, 3, 2)),
    ]
    with pytest.raises(DataValidationError, match="timezone-aware"):
        canonicalize_rows(naive)


# ---------------------------------------------------------------------------
# The reusable assertion
# ---------------------------------------------------------------------------


def test_canonical_rows_pass_the_assertion() -> None:
    """The guard later milestones call at every fit entry point."""
    assert_canonical(
        canonicalize_rows([Row(anchor_id(i), at(i)) for i in range(5)]), stage="a fit"
    )


def test_the_assertion_names_the_stage_that_lost_the_order() -> None:
    """A failure should say where the order was lost, not merely that it was."""
    unordered = rows(("e0002", at(9)), ("e0001", at(1)))
    with pytest.raises(ModelTrainingError, match="calibration fit"):
        assert_canonical(unordered, stage="calibration fit")


def test_the_assertion_rejects_a_duplicate_anchor() -> None:
    """Ordered but duplicated is still invalid."""
    with pytest.raises(ModelTrainingError, match="duplicate anchor"):
        assert_canonical(rows(("e0001", at(1)), ("e0001", at(2))), stage="a fit")


def test_an_empty_sequence_is_canonical() -> None:
    """An empty split is vacuously ordered; the guard must not invent a failure."""
    assert is_canonical([])
    assert_canonical([], stage="a fit")


def test_a_single_row_is_canonical() -> None:
    """One row cannot be out of order."""
    assert is_canonical(rows(("e0001", at(3))))


def test_is_canonical_answers_the_ordering_question_only() -> None:
    """Duplicates are a separate failure, reported separately."""
    assert is_canonical(rows(("e0001", at(1)), ("e0001", at(2))))


def test_is_canonical_detects_an_out_of_order_pair() -> None:
    """One inversion anywhere is enough."""
    assert not is_canonical(rows(("e0001", at(1)), ("e0003", at(9)), ("e0002", at(4))))


def test_is_canonical_detects_an_out_of_order_tie_break() -> None:
    """Times ascending is not sufficient; the tie-break is part of the order."""
    assert not is_canonical(rows(("e0002", at(5)), ("e0001", at(5))))


# ---------------------------------------------------------------------------
# What the module refuses to use
# ---------------------------------------------------------------------------


def test_the_module_uses_no_python_hash_and_no_wall_clock() -> None:
    """Randomised hashing and wall-clock reads would both break reproducibility.

    Asserted over the syntax tree rather than the text. The module's docstring
    legitimately *names* the things it refuses to use, and a substring scan
    would flag the very prose that documents the guarantee.
    """
    import ast
    import inspect

    from password_attack_detector.ml import ordering

    tree = ast.parse(inspect.getsource(ordering))

    called: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                called.add(target.id)
            elif isinstance(target, ast.Attribute):
                called.add(target.attr)
        elif isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "hash" not in called
    assert "now" not in called
    assert "utcnow" not in called
    assert "time" not in called
    assert "locale" not in imported
    assert "random" not in imported
