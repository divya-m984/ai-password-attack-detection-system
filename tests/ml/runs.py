"""Deterministic builders for the Milestone 6 training and ledger suites.

One small dataset, hand-specified, spanning every split the orchestration reads
and every split it must refuse. Literal timestamps, integer-derived feature
values, and no generator anywhere: a training test that needed 720 hours of
synthetic traffic would be testing the generator.

The dataset is deliberately shaped so that the interesting outcomes are all
reachable. Logistic regression produces distinct scores and completes; the prior
baseline produces one score for every row and cannot support a calibrator, which
is a real outcome the run statuses exist to describe rather than a defect in the
fixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from password_attack_detector.features.catalog import FeatureCatalog
from password_attack_detector.ml.config import (
    AnomalyConfig,
    CalibrationConfig,
    CategoryConfig,
    ChampionGateConfig,
    ImbalanceConfig,
    MLConfig,
    PreprocessingConfig,
    ThresholdConfig,
    ValidationPartitionConfig,
)
from password_attack_detector.ml.dataset import (
    CampaignRow,
    LabelRow,
    MLDataset,
    SplitRow,
    assemble_ml_dataset,
)
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    MLSplit,
    ModelFamily,
    ThresholdObjective,
)
from password_attack_detector.ml.features import (
    EligibleFeatureList,
    resolve_eligible_features,
)
from password_attack_detector.ml.partition import (
    ValidationPartitionResult,
    partition_validation,
)
from password_attack_detector.ml.schemas import SupportRequirement
from password_attack_detector.ml.training import TrainingContext
from tests.ml import factories as fx

#: The known malicious categories this fixture uses. Two, because the category
#: head needs at least two represented classes and a third would add rows
#: without adding coverage.
CATEGORIES: tuple[str, ...] = ("brute_force", "password_spraying")

#: The transformed column M-001 is configured to threshold. Named here so the
#: fixture and the configuration cannot drift apart.
BASELINE_COLUMN: str = "user_failure_rate"

#: How many rows each split gets. Sized to clear the floors below with room to
#: spare, and no larger: every row is fitted on in every test that runs.
TRAIN_ROWS: int = 160
VALIDATION_ROWS: int = 120
TEST_ROWS: int = 40
HOLDOUT_ROWS: int = 20


def support() -> SupportRequirement:
    """Return CI-sized support floors."""
    return SupportRequirement(
        min_train_positive_rows=5,
        min_validation_positive_rows=5,
        min_validation_benign_rows=10,
        min_rows_per_category=2,
    )


def config(**overrides: Any) -> MLConfig:
    """Return a CI-sized ML configuration with the M6 policies pinned.

    Every *policy* matches the development configuration -- calibration reads
    validation-A, thresholds read validation-B, missingness indicators are on,
    resampling is off. Only the counts are shrunk, so a test cannot pass on a
    pipeline development would reject.
    """
    settings: dict[str, Any] = {
        "seed": 7,
        "single_feature_baseline_column": BASELINE_COLUMN,
        "enabled_model_families": (
            ModelFamily.PRIOR_BASELINE,
            ModelFamily.SINGLE_FEATURE_THRESHOLD,
            ModelFamily.LOGISTIC_REGRESSION,
            ModelFamily.ISOLATION_FOREST,
        ),
        "preprocessing": PreprocessingConfig(
            min_category_frequency=1,
            max_category_cardinality=8,
            include_leakage_classes=(
                "prior_only",
                "current_event_context",
                "baseline_derived",
            ),
        ),
        "imbalance": ImbalanceConfig(),
        "validation_partition": ValidationPartitionConfig(
            min_partition_rows=10, min_partition_positive_rows=2
        ),
        "calibration": CalibrationConfig(
            # Platt rather than isotonic: a CI-sized validation half carries far
            # too few rows for a monotone piecewise fit to mean anything.
            method=CalibrationMethod.PLATT,
            max_expected_calibration_error=0.5,
            reliability_bin_count=4,
            min_calibration_rows=10,
            min_reliability_bin_rows=1,
            min_isotonic_distinct_scores=2,
        ),
        "thresholds": ThresholdConfig(
            objective=ThresholdObjective.MAX_F1,
            max_false_positive_rate=0.5,
            min_detection_rate=0.1,
            search_grid_size=64,
        ),
        "category": CategoryConfig(
            min_category_score=0.2,
            min_rows_per_category=2,
            min_known_category_precision=0.4,
            min_known_malicious_rows=5,
            abstention_search_grid_size=64,
        ),
        "anomaly": AnomalyConfig(quantile=0.9, min_fit_rows=10),
        "support": support(),
        # Loosened to match the loosened threshold search above. The gate is
        # not exercised here -- Milestone 6 promotes nothing -- but the
        # configuration refuses a gate stricter than the search it judges, and
        # that coherence check is worth keeping rather than working around.
        "gates": ChampionGateConfig(
            min_pr_auc_gain_over_baseline=0.0,
            max_false_positive_rate=0.5,
            min_detection_rate=0.1,
            max_expected_calibration_error=0.5,
        ),
    }
    settings.update(overrides)
    return MLConfig(**settings)


def feature_catalog() -> FeatureCatalog:
    """Return the six-feature catalog every fixture is built from."""
    return fx.preprocessing_catalog()


def eligible_features(catalog: FeatureCatalog | None = None) -> EligibleFeatureList:
    """Return the resolved reviewed feature contract."""
    source = catalog or feature_catalog()
    return resolve_eligible_features(
        source,
        fx.allowlist_for(source),
        include_leakage_classes=(
            "prior_only",
            "current_event_context",
            "baseline_derived",
        ),
    )


@dataclass(frozen=True, slots=True)
class Rows:
    """The raw tables one dataset is assembled from."""

    features: list[dict[str, Any]]
    labels: list[LabelRow]
    splits: list[SplitRow]
    campaigns: list[CampaignRow]


def _values(index: int, *, malicious: bool) -> tuple[Any, ...]:
    """Return one row's raw feature values.

    Separable but not perfectly so: the failure rate tracks the label with a
    fixed, reproducible overlap on every seventh row. Perfect separation would
    let a Platt fit run to infinity and would make every threshold objective
    agree, which hides exactly the behaviour these suites check.
    """
    overlap = index % 7 == 0
    signal = 0.9 if malicious != overlap else 0.1
    return (
        round(signal + (index % 5) * 0.01, 9),
        int(index % 11),
        "failure" if malicious else "success",
        ("us", "gb", "de")[index % 3],
        bool(index % 2),
        True,
    )


def _rows(
    *,
    start: int,
    count: int,
    split: MLSplit,
    malicious_every: int,
    campaign_size: int = 5,
    campaign_prefix: str = "c",
) -> Rows:
    """Return *count* rows of one split, with campaigns over the malicious ones."""
    names = fx.PREPROCESSING_FEATURES
    features: list[dict[str, Any]] = []
    labels: list[LabelRow] = []
    splits: list[SplitRow] = []
    campaigns: list[CampaignRow] = []
    malicious_seen = 0

    for offset in range(count):
        index = start + offset
        malicious = offset % malicious_every == 0
        features.append(
            fx.feature_row(
                index,
                names=names,
                minutes=index,
                values=_values(index, malicious=malicious),
            )
        )
        attack_class = (
            CATEGORIES[malicious_seen % len(CATEGORIES)] if malicious else "normal"
        )
        labels.append(
            LabelRow(
                event_id=fx.anchor_id(index),
                attack_class=attack_class,
                malicious=malicious,
                supervised_training_eligible=True,
            )
        )
        splits.append(SplitRow(event_id=fx.anchor_id(index), split=str(split)))
        if malicious:
            campaigns.append(
                CampaignRow(
                    event_id=fx.anchor_id(index),
                    campaign_id=(
                        f"{campaign_prefix}{malicious_seen // campaign_size:03d}"
                    ),
                    campaign_stage=None,
                )
            )
            malicious_seen += 1
    return Rows(features=features, labels=labels, splits=splits, campaigns=campaigns)


def build_rows(
    *,
    train_rows: int = TRAIN_ROWS,
    validation_rows: int = VALIDATION_ROWS,
    test_rows: int = TEST_ROWS,
    holdout_rows: int = HOLDOUT_ROWS,
    train_signal: float | None = None,
) -> Rows:
    """Return every raw table for one complete dataset.

    Each split occupies its own contiguous index range, so a test that wants to
    perturb one split can rebuild that range and leave the others byte-identical.
    """
    parts = [
        _rows(
            start=0,
            count=train_rows,
            split=MLSplit.TRAIN,
            malicious_every=3,
            campaign_prefix="t",
        ),
        _rows(
            start=1000,
            count=validation_rows,
            split=MLSplit.VALIDATION,
            malicious_every=3,
            campaign_prefix="v",
        ),
        _rows(
            start=2000,
            count=test_rows,
            split=MLSplit.TEST,
            malicious_every=3,
            campaign_prefix="s",
        ),
        _rows(
            start=3000,
            count=holdout_rows,
            split=MLSplit.NOVEL_ANOMALY_HOLDOUT,
            malicious_every=2,
            campaign_prefix="n",
        ),
    ]
    rows = Rows(
        features=[row for part in parts for row in part.features],
        labels=[row for part in parts for row in part.labels],
        splits=[row for part in parts for row in part.splits],
        campaigns=[row for part in parts for row in part.campaigns],
    )
    if train_signal is not None:
        for row in rows.features:
            if int(row["anchor_event_id"][1:]) < train_rows:
                row["user_failure_rate"] = train_signal
    return rows


def dataset(rows: Rows | None = None, **kwargs: Any) -> MLDataset:
    """Return an assembled dataset over *rows*."""
    source = rows if rows is not None else build_rows(**kwargs)
    catalog = feature_catalog()
    return assemble_ml_dataset(
        feature_rows=source.features,
        labels=source.labels,
        splits=source.splits,
        campaigns=source.campaigns,
        eligible=eligible_features(catalog),
        feature_catalog_fingerprint=catalog.fingerprint(),
    )


def partition(
    data: MLDataset, settings: MLConfig | None = None
) -> ValidationPartitionResult:
    """Return the campaign-disjoint validation partition for *data*."""
    resolved = settings or config()
    return partition_validation(
        data.for_split(MLSplit.VALIDATION),
        config=resolved.validation_partition,
        support=resolved.support,
        campaign_metadata_supplied=True,
    )


def context(
    data: MLDataset | None = None,
    settings: MLConfig | None = None,
    **kwargs: Any,
) -> TrainingContext:
    """Return a prepared training context over a fixture dataset."""
    resolved = settings or config()
    catalog = feature_catalog()
    built = data if data is not None else dataset(**kwargs)
    return TrainingContext.prepare(
        built,
        config=resolved,
        eligible=eligible_features(catalog),
        feature_catalog=catalog,
        partition=partition(built, resolved),
        allowlist_fingerprint=fx.allowlist_for(catalog).fingerprint(),
    )


def shuffled(rows: Rows, *, step: int = 7) -> Rows:
    """Return *rows* with every table reordered deterministically.

    A fixed stride rather than a shuffle: the point is that the *source* order
    differs while the assembled dataset does not, and a random permutation
    would make a failure depend on a seed nobody wrote down.
    """

    def rotate(items: Sequence[Any]) -> list[Any]:
        return (
            [items[(index * step) % len(items)] for index in range(len(items))]
            if (len(items) and _coprime(step, len(items)))
            else list(reversed(items))
        )

    return Rows(
        features=rotate(rows.features),
        labels=rotate(rows.labels),
        splits=rotate(rows.splits),
        campaigns=rotate(rows.campaigns),
    )


def _coprime(left: int, right: int) -> bool:
    """Return whether *left* and *right* share no factor above one."""
    while right:
        left, right = right, left % right
    return left == 1


def anchors_in(rows: Rows, split: MLSplit) -> list[str]:
    """Return the anchor identifiers assigned to *split*, in table order."""
    return [row.event_id for row in rows.splits if row.split == str(split)]


def malicious_anchors(rows: Rows) -> set[str]:
    """Return the anchor identifiers of malicious rows."""
    return {row.event_id for row in rows.labels if row.malicious}


def _copy(rows: Rows) -> Rows:
    """Return a deep-enough copy that a mutation cannot reach the original."""
    return Rows(
        features=[dict(row) for row in rows.features],
        labels=list(rows.labels),
        splits=list(rows.splits),
        campaigns=list(rows.campaigns),
    )


def revalue_anchor(rows: Rows, anchor: str, *, value: float) -> Rows:
    """Return *rows* with one row's failure rate changed.

    One row and one column, so a test that asserts a fingerprint moved can
    point at the single thing that moved it.
    """
    copied = _copy(rows)
    for row in copied.features:
        if row["anchor_event_id"] == anchor:
            row["user_failure_rate"] = value
    return copied


def relabel_anchor(rows: Rows, anchor: str) -> Rows:
    """Return *rows* with one row's malicious label flipped.

    The attack class follows the label, because a malicious row with no
    scenario and a benign row carrying one are both states the Phase 2 contract
    does not produce.
    """
    return Rows(
        features=[dict(row) for row in rows.features],
        labels=[
            LabelRow(
                event_id=row.event_id,
                attack_class=("normal" if row.malicious else CATEGORIES[0]),
                malicious=not row.malicious,
                supervised_training_eligible=row.supervised_training_eligible,
            )
            if row.event_id == anchor
            else row
            for row in rows.labels
        ],
        splits=list(rows.splits),
        campaigns=list(rows.campaigns),
    )


def reassign_anchor(rows: Rows, anchor: str, *, split: MLSplit) -> Rows:
    """Return *rows* with one row moved to *split*."""
    return Rows(
        features=[dict(row) for row in rows.features],
        labels=list(rows.labels),
        splits=[
            SplitRow(event_id=row.event_id, split=str(split))
            if row.event_id == anchor
            else row
            for row in rows.splits
        ],
        campaigns=list(rows.campaigns),
    )
