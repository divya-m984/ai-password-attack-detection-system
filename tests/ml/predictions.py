"""Deterministic builders for the Milestone 8 prediction suites.

Built on the Milestone 6 fixture rather than beside it: a prediction test needs a
*frozen champion*, and a frozen champion is the end of a real pipeline -- train,
publish, select, freeze. A hand-built lock would let these suites pass against a
shape the pipeline never produces, which is exactly the failure the whole
verification chain exists to catch.

Nothing here is a test. It is the fixture the prediction suites share, kept in
one place so the inference, serialization, manifest, publisher, and quality
suites cannot drift into disagreeing about what a frozen champion looks like.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from password_attack_detector.ml.dataset import (
    InferenceDataset,
    assemble_inference_dataset,
)
from password_attack_detector.ml.enums import MLSplit
from password_attack_detector.ml.ledger import ExperimentLedger
from password_attack_detector.ml.predictions import FrozenChampion
from tests.ml import runs as rx

__all__ = [
    "Prepared",
    "champion_of",
    "inference_dataset",
    "prepare",
    "shuffled_inference_dataset",
]


@dataclass(frozen=True, slots=True)
class Prepared:
    """One published, selected, and frozen experiment in a writable root."""

    root: Path
    ledger: ExperimentLedger
    rows: rx.Rows
    scope_key: str

    def champion(self, **overrides: Any) -> FrozenChampion:
        """Return the verified frozen champion under this root."""
        settings: dict[str, Any] = {"ledger": self.ledger, "scope_key": self.scope_key}
        settings.update(overrides)
        return FrozenChampion.load(self.root, **settings)


def prepare(
    destination: Path,
    *,
    source: rx.Experiment | None = None,
    rows: rx.Rows | None = None,
) -> Prepared:
    """Train, publish, select, and freeze a champion under *destination*.

    Args:
        destination: a writable root. Prediction writes, and a suite that
            corrupted a shared run directory would decide what every later test
            saw.
        source: an already published experiment to copy, so the expensive fit
            happens once per module rather than once per test.
        rows: the raw tables to train from when no *source* is supplied.
    """
    from password_attack_detector.ml.champion import freeze_champion
    from password_attack_detector.ml.selection import (
        load_candidate_evidence,
        publish_selection,
        select_binary_champion,
        select_category_head,
    )

    built = rows if rows is not None else rx.build_rows()
    if source is None:
        source = rx.publish_experiment(
            destination.parent / f"{destination.name}-source", rows=built
        )
    root = destination
    shutil.copytree(source.root, root)

    ledger = ExperimentLedger(root / "ledger")
    evidence = load_candidate_evidence(root, ledger=ledger)
    config = rx.config()
    binary = select_binary_champion(evidence, config=config)
    category = select_category_head(evidence, config=config)
    publish_selection(binary, root=root, ledger=ledger)
    publish_selection(category, root=root, ledger=ledger)
    publication = freeze_champion(
        binary.record,
        evidence={item.run_id: item for item in evidence},
        config=config,
        root=root,
        ledger=ledger,
        category=category.record,
    )
    return Prepared(
        root=root, ledger=ledger, rows=built, scope_key=publication.scope_key
    )


def champion_of(prepared: Prepared) -> FrozenChampion:
    """Return the verified frozen champion, for the common case."""
    return prepared.champion()


def inference_dataset(
    prepared: Prepared,
    *,
    scope: MLSplit = MLSplit.TEST,
    rows: rx.Rows | None = None,
) -> InferenceDataset:
    """Return the label-free inference input for one scope.

    Assembled from the feature and split tables alone. The fixture's label and
    campaign tables are right there in ``rows`` and are deliberately not passed:
    there is no parameter for them, which is the firewall stated as a signature.
    """
    source = rows if rows is not None else prepared.rows
    return assemble_inference_dataset(
        feature_rows=source.features,
        splits=source.splits,
        eligible=rx.eligible_features(),
        scope=scope,
        feature_catalog_fingerprint=rx.feature_catalog().fingerprint(),
    )


def shuffled_inference_dataset(
    prepared: Prepared, *, scope: MLSplit = MLSplit.TEST, step: int = 7
) -> InferenceDataset:
    """Return the same inference input from physically reordered source rows."""
    return inference_dataset(
        prepared, scope=scope, rows=rx.shuffled(prepared.rows, step=step)
    )
