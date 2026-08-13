"""Shared builders for the Milestone 4 model suites.

One place that assembles a training batch, fits a family, and publishes an
artifact, so a test can say what it is checking instead of restating the whole
pipeline. Everything is deterministic: fixed seeds, literal timestamps, and
``n_jobs=1`` throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from password_attack_detector.ml.catalog import MODEL_CATALOG
from password_attack_detector.ml.config import ImbalanceConfig, PreprocessingConfig
from password_attack_detector.ml.enums import MLSplit, MLTask
from password_attack_detector.ml.features import resolve_eligible_features
from password_attack_detector.ml.imbalance import (
    BINARY_CLASS_ORDER,
    compute_class_weights,
)
from password_attack_detector.ml.manifest import build_model_manifest
from password_attack_detector.ml.models.base import FittedModel, TrainingBatch
from password_attack_detector.ml.preprocessing import (
    FittedPreprocessor,
    TransformedMatrix,
    fit_preprocessor,
)
from password_attack_detector.ml.serialization import (
    build_model_document,
    write_model_directory,
)
from tests.ml import factories as fx

#: The instant every fixture counts from. A literal, never ``now``.
EPOCH: datetime = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)

ALL_CLASSES = ("prior_only", "current_event_context", "baseline_derived")

CATEGORY_ORDER: tuple[str, ...] = (
    "brute_force",
    "credential_stuffing",
    "password_spraying",
)


@dataclass(frozen=True, slots=True)
class Anchor:
    """The only two anchor fields a model layer is allowed to see."""

    anchor_event_id: str
    anchor_event_time: datetime


@dataclass(frozen=True, slots=True)
class Frame:
    """A minimal frame satisfying the preprocessing protocol."""

    split: MLSplit
    feature_names: tuple[str, ...]
    anchors: tuple[Anchor, ...]
    feature_matrix: tuple[tuple[Any, ...], ...]


def raw_rows(count: int = 160, *, seed: int = 7) -> list[tuple[Any, ...]]:
    """Return deterministic raw feature rows in ``PREPROCESSING_FEATURES`` order."""
    rng = np.random.default_rng(seed)
    return [
        (
            float(rng.normal()),
            int(rng.integers(0, 20)),
            "success" if index % 3 else "failure",
            ("us", "gb", "de")[index % 3],
            bool(index % 2),
            True,
        )
        for index in range(count)
    ]


def frame_for(
    rows: Sequence[tuple[Any, ...]], *, split: MLSplit = MLSplit.TRAIN
) -> Frame:
    """Return a canonically ordered frame over *rows*."""
    return Frame(
        split=split,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=tuple(
            Anchor(fx.anchor_id(index), EPOCH + timedelta(minutes=index))
            for index in range(len(rows))
        ),
        feature_matrix=tuple(tuple(row) for row in rows),
    )


def binary_targets(rows: Sequence[tuple[Any, ...]]) -> tuple[str, ...]:
    """Return a separable-but-not-trivial binary target column.

    A null first column counts as benign rather than raising: several fixtures
    deliberately make that column mostly missing, and the target is a property
    of the fixture rather than of the feature.
    """
    return tuple(
        "malicious" if row[0] is not None and float(row[0]) > 0.2 else "benign"
        for row in rows
    )


def category_targets(rows: Sequence[tuple[Any, ...]]) -> tuple[str, ...]:
    """Return a three-class target column for the triage head."""
    return tuple(CATEGORY_ORDER[index % 3] for index in range(len(rows)))


@dataclass(frozen=True, slots=True)
class Prepared:
    """A batch and everything needed to publish or re-score it."""

    batch: TrainingBatch
    matrix: TransformedMatrix
    preprocessor: FittedPreprocessor
    frame: Frame
    catalog: Any
    eligible: Any


def prepare(
    *,
    count: int = 160,
    seed: int = 7,
    task: MLTask = MLTask.BINARY_MALICIOUS,
    weighted: bool = True,
    rows: Sequence[tuple[Any, ...]] | None = None,
) -> Prepared:
    """Return a fitted preprocessor and an assembled training batch."""
    source = list(rows) if rows is not None else raw_rows(count, seed=seed)
    catalog = fx.preprocessing_catalog()
    eligible = resolve_eligible_features(
        catalog, fx.allowlist_for(catalog), include_leakage_classes=ALL_CLASSES
    )
    frame = frame_for(source)
    preprocessor = fit_preprocessor(
        frame,
        catalog=catalog,
        eligible=eligible,
        config=PreprocessingConfig(min_category_frequency=2),
    )
    matrix = preprocessor.transform(frame)

    if task is MLTask.ANOMALY:
        targets: tuple[str, ...] = ()
        order: tuple[str, ...] = ()
        weights = None
    else:
        targets = (
            binary_targets(source)
            if task is MLTask.BINARY_MALICIOUS
            else category_targets(source)
        )
        order = (
            BINARY_CLASS_ORDER if task is MLTask.BINARY_MALICIOUS else CATEGORY_ORDER
        )
        weights = (
            compute_class_weights(
                list(targets), task=task, class_order=order, config=ImbalanceConfig()
            )
            if weighted
            else None
        )

    batch = TrainingBatch(
        split=MLSplit.TRAIN,
        anchors=frame.anchors,
        transformed_feature_names=matrix.output_feature_names,
        matrix=matrix.rows,
        targets=targets,
        class_order=order,
        preprocessor=preprocessor,
        class_weights=weights,
    )
    return Prepared(
        batch=batch,
        matrix=matrix,
        preprocessor=preprocessor,
        frame=frame,
        catalog=catalog,
        eligible=eligible,
    )


def publish(
    target: Path,
    fitted: FittedModel,
    preprocessor: FittedPreprocessor,
    *,
    overwrite: bool = False,
    published_at: str | None = None,
    **manifest_overrides: Any,
) -> Path:
    """Write a complete model directory at *target*."""
    document = build_model_document(fitted)

    def builder(digests: dict[str, str], sizes: dict[str, int]) -> Any:
        return build_model_manifest(
            document=document,
            digests=digests,
            sizes=sizes,
            seed=42,
            required_feature_schema_version="1.0.0",
            model_catalog_fingerprint=MODEL_CATALOG.fingerprint(),
            published_at=published_at,
            **manifest_overrides,
        )

    return write_model_directory(
        target,
        fitted=fitted,
        preprocessor=preprocessor,
        manifest_builder=builder,
        overwrite=overwrite,
    )
