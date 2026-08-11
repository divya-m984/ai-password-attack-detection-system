"""Deterministic builders for the Milestone 5 calibration and threshold suites.

Everything here is hand-specified and reproducible: literal timestamps, a fixed
seed where randomness is used at all, and digest stand-ins made of repeated hex
characters so a fingerprint mismatch in a failure message is instantly readable.

Nothing here fits a model.  Milestone 5 consumes *scores*, and a test that had
to fit a random forest to produce a hundred numbers would be testing the forest.
The one place a real fitted model appears is the parity suite, which needs the
genuine article precisely because that is what it is comparing against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from password_attack_detector.ml.calibration import BinaryScoreSample, ScoreSampleSource
from password_attack_detector.ml.enums import (
    MLSplit,
    ScoreKind,
    ValidationPartition,
)
from password_attack_detector.ml.thresholds import (
    AnomalyScoreSample,
    CategoryScoreSample,
)

#: The instant every fixture counts from.  A literal, never ``now``.
EPOCH: datetime = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)

#: Digest stand-ins.  Distinct repeated characters, so a provenance failure
#: names which link broke without anybody having to compare sixty-four hex
#: digits by eye.
MODEL_FINGERPRINT: str = "a" * 64
PARTITION_FINGERPRINT: str = "b" * 64
PREPROCESSOR_FINGERPRINT: str = "c" * 64
CONFIG_FINGERPRINT: str = "d" * 64
TRAIN_FINGERPRINT: str = "e" * 64
OTHER_FINGERPRINT: str = "f" * 64

#: The deterministic known-category class order, sorted as the configuration
#: declares it must be.
CATEGORY_ORDER: tuple[str, ...] = (
    "brute_force",
    "credential_stuffing",
    "password_spraying",
)


@dataclass(frozen=True, slots=True)
class Anchor:
    """The only two anchor fields Milestone 5 is allowed to see."""

    anchor_event_id: str
    anchor_event_time: datetime


def anchors(count: int, *, start: int = 0) -> tuple[Anchor, ...]:
    """Return *count* canonically ordered anchors."""
    return tuple(
        Anchor(f"e{index:05d}", EPOCH + timedelta(minutes=index))
        for index in range(start, start + count)
    )


def source(
    partition: ValidationPartition | None = ValidationPartition.VALIDATION_A,
    *,
    split: MLSplit = MLSplit.VALIDATION,
    fingerprint: str = PARTITION_FINGERPRINT,
) -> ScoreSampleSource:
    """Return typed provenance for a batch of scores.

    A half is only meaningful for the validation split, so one supplied
    alongside any other split is dropped rather than refused: it lets a
    firewall test say ``split=MLSplit.TEST`` without also having to remember to
    clear the default.
    """
    return ScoreSampleSource(
        split=split,
        partition=partition if split is MLSplit.VALIDATION else None,
        source_fingerprint=fingerprint,
    )


def train_source(fingerprint: str = TRAIN_FINGERPRINT) -> ScoreSampleSource:
    """Return typed provenance for training rows."""
    return ScoreSampleSource(
        split=MLSplit.TRAIN, partition=None, source_fingerprint=fingerprint
    )


def graded_scores(count: int) -> tuple[float, ...]:
    """Return *count* evenly spaced scores in ``(0, 1)``.

    Evenly spaced rather than random, so every count in a confusion matrix can
    be worked out by hand from the index of the threshold.
    """
    return tuple(round((index + 0.5) / count, 9) for index in range(count))


def graded_labels(
    scores: Sequence[float], *, cut: float = 0.5, noise_every: int = 7
) -> tuple[bool, ...]:
    """Return labels that follow *scores* with a fixed, reproducible disagreement.

    Perfectly separated labels would let a Platt fit run off to infinity and
    would make every threshold objective agree, which hides exactly the
    behaviour these suites exist to pin.  Every ``noise_every``-th row is
    flipped: enough overlap to be interesting, and entirely deterministic.
    """
    return tuple(
        (score >= cut) != (index % noise_every == 0)
        for index, score in enumerate(scores)
    )


def binary_sample(
    scores: Sequence[float],
    malicious: Sequence[bool],
    *,
    partition: ValidationPartition | None = ValidationPartition.VALIDATION_A,
    split: MLSplit = MLSplit.VALIDATION,
    fingerprint: str = PARTITION_FINGERPRINT,
    score_kind: ScoreKind = ScoreKind.DECISION_SCORE,
    model_fingerprint: str = MODEL_FINGERPRINT,
    preprocessor_fingerprint: str = PREPROCESSOR_FINGERPRINT,
    model_id: str = "model-under-test",
    **extra: Any,
) -> BinaryScoreSample:
    """Return a binary score sample over *scores* and *malicious*."""
    return BinaryScoreSample(
        source=source(partition, split=split, fingerprint=fingerprint),
        anchors=anchors(len(scores)),
        scores=tuple(scores),
        malicious=tuple(malicious),
        score_kind=score_kind,
        model_id=model_id,
        model_content_fingerprint=model_fingerprint,
        preprocessor_fingerprint=preprocessor_fingerprint,
        **extra,
    )


def validation_a(count: int = 400, **kwargs: Any) -> BinaryScoreSample:
    """Return a validation-A sample large enough for the default support floors."""
    scores = graded_scores(count)
    return binary_sample(
        scores,
        graded_labels(scores),
        partition=ValidationPartition.VALIDATION_A,
        **kwargs,
    )


def validation_b(count: int = 400, **kwargs: Any) -> BinaryScoreSample:
    """Return a validation-B sample large enough for the default support floors."""
    scores = graded_scores(count)
    return binary_sample(
        scores,
        graded_labels(scores),
        partition=ValidationPartition.VALIDATION_B,
        **kwargs,
    )


def category_sample(
    rows: Sequence[tuple[tuple[float, ...], str]],
    *,
    partition: ValidationPartition | None = ValidationPartition.VALIDATION_B,
    split: MLSplit = MLSplit.VALIDATION,
    fingerprint: str = PARTITION_FINGERPRINT,
    class_order: Sequence[str] = CATEGORY_ORDER,
    malicious: Sequence[bool] | None = None,
    **extra: Any,
) -> CategoryScoreSample:
    """Return a category sample from ``(class scores, true category)`` pairs."""
    return CategoryScoreSample(
        source=source(partition, split=split, fingerprint=fingerprint),
        anchors=anchors(len(rows)),
        class_order=tuple(class_order),
        class_scores=tuple(scores for scores, _ in rows),
        true_category=tuple(name for _, name in rows),
        malicious=tuple(malicious) if malicious is not None else (True,) * len(rows),
        score_kind=ScoreKind.CLASS_SCORE,
        model_id="category-model",
        model_content_fingerprint=MODEL_FINGERPRINT,
        preprocessor_fingerprint=PREPROCESSOR_FINGERPRINT,
        **extra,
    )


def category_rows(
    count: int, *, correct_every: int = 5
) -> list[tuple[tuple[float, ...], str]]:
    """Return category rows whose confidence and correctness both vary by index.

    Row ``i`` puts its peak on class ``i % 3`` with a confidence that climbs
    with the index, and every ``correct_every``-th row is *mislabelled* so the
    precision at a low threshold is below one and rises as the threshold does.
    That is the shape an abstention threshold exists to exploit, so it is the
    shape the fixture has.
    """
    rows: list[tuple[tuple[float, ...], str]] = []
    for index in range(count):
        peak = index % len(CATEGORY_ORDER)
        confidence = round(0.34 + 0.65 * (index / max(count - 1, 1)), 9)
        remainder = round((1.0 - confidence) / 2.0, 9)
        scores = tuple(
            confidence if position == peak else remainder
            for position in range(len(CATEGORY_ORDER))
        )
        truth = CATEGORY_ORDER[peak]
        if index % correct_every == 0:
            truth = CATEGORY_ORDER[(peak + 1) % len(CATEGORY_ORDER)]
        rows.append((scores, truth))
    return rows


def anomaly_sample(
    scores: Sequence[float],
    *,
    split: MLSplit = MLSplit.TRAIN,
    partition: ValidationPartition | None = None,
    fingerprint: str = TRAIN_FINGERPRINT,
    malicious: Sequence[bool] | None = None,
    **extra: Any,
) -> AnomalyScoreSample:
    """Return a benign anomaly-score sample."""
    return AnomalyScoreSample(
        source=source(partition, split=split, fingerprint=fingerprint),
        anchors=anchors(len(scores)),
        scores=tuple(scores),
        malicious=(
            tuple(malicious) if malicious is not None else (False,) * len(scores)
        ),
        score_kind=ScoreKind.ANOMALY_SCORE,
        model_id="anomaly-probe",
        model_content_fingerprint=MODEL_FINGERPRINT,
        preprocessor_fingerprint=PREPROCESSOR_FINGERPRINT,
        **extra,
    )


def anomaly_scores(count: int) -> tuple[float, ...]:
    """Return *count* distinct anomaly scores, most anomalous first.

    Negative and increasing, matching the scikit-learn convention this project
    inherits: a lower value is more anomalous.
    """
    return tuple(round(-1.0 + index / count, 9) for index in range(count))
