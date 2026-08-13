"""Dataset assembly: the joins, the class space, and what stays out of X."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

import pytest

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.ml.dataset import (
    KNOWN_CATEGORY_CLASSES,
    UNKNOWN_CAMPAIGN_REFERENCE_CODE,
    CampaignRow,
    MLDataset,
    assemble_ml_dataset,
    campaign_association,
    declared_campaign_ids,
)
from password_attack_detector.ml.enums import MLSplit
from password_attack_detector.ml.features import resolve_eligible_features
from password_attack_detector.ml.ordering import is_canonical
from tests.ml import factories as fx

#: One planned row: index, split, malicious, attack class, campaign.
Plan = list[tuple[int, str, bool, str, str | None]]

ALL_CLASSES = ("prior_only", "current_event_context", "baseline_derived")


@pytest.fixture
def eligible() -> Any:
    """Return the resolved four-feature contract used across this module."""
    catalog = fx.small_catalog()
    return resolve_eligible_features(
        catalog, fx.allowlist_for(catalog), include_leakage_classes=ALL_CLASSES
    )


def build(
    eligible: Any,
    *,
    plan: Plan,
    campaigns: bool = True,
    feature_rows: list[dict[str, Any]] | None = None,
    campaign_rows: list[CampaignRow] | None = None,
    **kwargs: Any,
) -> MLDataset:
    """Assemble a dataset from a compact ``(index, split, malicious, class, campaign)`` plan."""
    names = eligible.feature_names
    rows = feature_rows
    if rows is None:
        rows = [fx.feature_row(index, names=names) for index, *_ in plan]
    return assemble_ml_dataset(
        feature_rows=rows,
        labels=[
            fx.label_row(index, malicious=malicious, attack_class=attack_class)
            for index, _, malicious, attack_class, _ in plan
        ],
        splits=[fx.split_row(index, split) for index, split, *_ in plan],
        campaigns=(
            campaign_rows
            if campaign_rows is not None
            else (
                [
                    fx.campaign_row(index, campaign)
                    for index, _, _, _, campaign in plan
                    if campaign is not None
                ]
                if campaigns
                else None
            )
        ),
        eligible=eligible,
        **kwargs,
    )


def simple_plan(count: int = 6) -> Plan:
    """Return a plan alternating benign train rows and malicious validation rows."""
    plan: Plan = []
    for index in range(count):
        if index % 2:
            plan.append((index, "validation", True, "brute_force", "c1"))
        else:
            plan.append((index, "train", False, "normal", None))
    return plan


# ---------------------------------------------------------------------------
# The class space
# ---------------------------------------------------------------------------


def test_the_known_category_space_is_derived_from_the_label_schema() -> None:
    """Derived at runtime; a renamed scenario fails here rather than scoring wrong."""
    expected = tuple(
        sorted(
            str(scenario)
            for scenario in ScenarioType
            if scenario not in {ScenarioType.NORMAL, ScenarioType.NOVEL_ANOMALY_HOLDOUT}
        )
    )
    assert expected == KNOWN_CATEGORY_CLASSES
    assert len(KNOWN_CATEGORY_CLASSES) == 7


def test_normal_and_the_novel_holdout_are_not_classes() -> None:
    """The head is fitted on malicious rows, and the holdout measures the unfitted."""
    assert "normal" not in KNOWN_CATEGORY_CLASSES
    assert "novel_anomaly_holdout" not in KNOWN_CATEGORY_CLASSES


def test_the_class_order_is_deterministic_and_sorted() -> None:
    """A serialized class order must mean the same thing on every machine."""
    assert list(KNOWN_CATEGORY_CLASSES) == sorted(KNOWN_CATEGORY_CLASSES)


def test_a_benign_row_carries_no_known_category(eligible: Any) -> None:
    """Benign rows are not a category; they are the absence of one."""
    dataset = build(eligible, plan=simple_plan())
    train = dataset.for_split(MLSplit.TRAIN)
    assert set(train.known_category) == {None}


def test_a_novel_holdout_row_carries_no_known_category(eligible: Any) -> None:
    """Its whole purpose is to be an attack no class was fitted for."""
    dataset = build(
        eligible,
        plan=[
            (0, "novel_anomaly_holdout", True, "novel_anomaly_holdout", None),
            (1, "train", False, "normal", None),
        ],
    )
    holdout = dataset.for_split(MLSplit.NOVEL_ANOMALY_HOLDOUT)
    assert holdout.known_category == (None,)


def test_an_unrecognised_class_stays_representable_rather_than_forced(
    eligible: Any,
) -> None:
    """A class outside the space is ``None``, never rounded to the nearest known one."""
    dataset = build(
        eligible,
        plan=[
            (0, "train", True, "a_scenario_nobody_declared", "c1"),
            (1, "train", False, "normal", None),
        ],
    )
    assert dataset.for_split(MLSplit.TRAIN).known_category == (None, None)


def test_category_counts_are_zero_filled(eligible: Any) -> None:
    """A class with no rows reads as zero, not as absent."""
    dataset = build(eligible, plan=simple_plan())
    counts = dataset.for_split(MLSplit.VALIDATION).category_counts()
    assert set(counts) == set(KNOWN_CATEGORY_CLASSES)
    assert counts["brute_force"] == 3
    assert counts["bot_activity"] == 0


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def test_a_one_to_one_join_produces_one_row_per_anchor(eligible: Any) -> None:
    """The ordinary case, stated so the failures below have a baseline."""
    dataset = build(eligible, plan=simple_plan())
    assert dataset.joined_row_count == 6
    assert dataset.for_split(MLSplit.TRAIN).row_count == 3
    assert dataset.for_split(MLSplit.VALIDATION).row_count == 3


def test_a_duplicate_feature_anchor_fails(eligible: Any) -> None:
    """One event contributes one row, or the join is not one-to-one."""
    names = eligible.feature_names
    rows = [fx.feature_row(0, names=names), fx.feature_row(0, names=names, minutes=9)]
    with pytest.raises(DataValidationError, match="duplicate identifier"):
        build(
            eligible,
            plan=[(0, "train", False, "normal", None)],
            feature_rows=rows,
        )


def test_a_duplicate_label_fails(eligible: Any) -> None:
    """Two labels for one event is two answers to one question."""
    with pytest.raises(DataValidationError, match="label table"):
        assemble_ml_dataset(
            feature_rows=[fx.feature_row(0, names=eligible.feature_names)],
            labels=[fx.label_row(0), fx.label_row(0, malicious=True)],
            splits=[fx.split_row(0, "train")],
            eligible=eligible,
        )


def test_a_duplicate_split_assignment_fails(eligible: Any) -> None:
    """An event assigned to two splits belongs to neither."""
    with pytest.raises(DataValidationError, match="split table"):
        assemble_ml_dataset(
            feature_rows=[fx.feature_row(0, names=eligible.feature_names)],
            labels=[fx.label_row(0)],
            splits=[fx.split_row(0, "train"), fx.split_row(0, "test")],
            eligible=eligible,
        )


def test_a_duplicate_campaign_mapping_fails(eligible: Any) -> None:
    """One event belongs to at most one campaign."""
    with pytest.raises(DataValidationError, match="campaign table"):
        assemble_ml_dataset(
            feature_rows=[fx.feature_row(0, names=eligible.feature_names)],
            labels=[fx.label_row(0)],
            splits=[fx.split_row(0, "train")],
            campaigns=[fx.campaign_row(0, "c1"), fx.campaign_row(0, "c2")],
            eligible=eligible,
        )


def test_an_anchor_with_no_label_fails(eligible: Any) -> None:
    """An unlabelled row cannot be silently dropped or silently assumed benign."""
    with pytest.raises(DataValidationError, match="do not describe the same events"):
        assemble_ml_dataset(
            feature_rows=[
                fx.feature_row(index, names=eligible.feature_names) for index in (0, 1)
            ],
            labels=[fx.label_row(0)],
            splits=[fx.split_row(index, "train") for index in (0, 1)],
            eligible=eligible,
        )


def test_a_label_with_no_anchor_fails(eligible: Any) -> None:
    """The relationship is symmetric; an orphan label means a missing feature row."""
    with pytest.raises(DataValidationError, match="do not describe the same events"):
        assemble_ml_dataset(
            feature_rows=[fx.feature_row(0, names=eligible.feature_names)],
            labels=[fx.label_row(0), fx.label_row(1)],
            splits=[fx.split_row(0, "train")],
            eligible=eligible,
        )


def test_an_anchor_with_no_split_fails(eligible: Any) -> None:
    """A row with no split has no permitted use."""
    with pytest.raises(DataValidationError, match="split tables do not describe"):
        assemble_ml_dataset(
            feature_rows=[
                fx.feature_row(index, names=eligible.feature_names) for index in (0, 1)
            ],
            labels=[fx.label_row(index) for index in (0, 1)],
            splits=[fx.split_row(0, "train")],
            eligible=eligible,
        )


def test_an_orphan_campaign_row_fails(eligible: Any) -> None:
    """Campaign metadata must describe the same events, not a different run's."""
    with pytest.raises(DataValidationError, match="no matching feature anchor"):
        assemble_ml_dataset(
            feature_rows=[fx.feature_row(0, names=eligible.feature_names)],
            labels=[fx.label_row(0)],
            splits=[fx.split_row(0, "train")],
            campaigns=[fx.campaign_row(0, "c1"), fx.campaign_row(7, "c2")],
            eligible=eligible,
        )


def test_a_missing_feature_column_fails(eligible: Any) -> None:
    """An admitted feature the table does not carry is a contract mismatch."""
    row = fx.feature_row(0, names=eligible.feature_names)
    del row["login_hour_deviation"]
    with pytest.raises(DataValidationError, match="missing 1 admitted feature"):
        build(eligible, plan=[(0, "train", False, "normal", None)], feature_rows=[row])


def test_campaign_metadata_is_optional_at_assembly(eligible: Any) -> None:
    """Assembly tolerates its absence; the partitioner is what refuses to guess."""
    dataset = build(eligible, plan=simple_plan(), campaigns=False)
    assert dataset.campaign_fingerprint is None
    assert all(
        anchor.campaign_id is None
        for split in MLSplit
        for anchor in dataset.for_split(split).anchors
    )


# ---------------------------------------------------------------------------
# Campaign resolution: the metadata is the authority, not the spelling
# ---------------------------------------------------------------------------


def test_a_staged_identifier_is_declared_however_it_is_spelled() -> None:
    """A recorded stage is a positive declaration and outranks the shape."""
    rows = [
        fx.campaign_row(0, "normal-123", stage="active"),
        fx.campaign_row(1, "normal-864209"),
    ]
    assert declared_campaign_ids(rows) == frozenset({"normal-123"})


def test_an_ordinary_campaign_table_declares_everything_in_it() -> None:
    """No stages anywhere is what a plain campaign table looks like.

    Absence of a stage declares nothing either way, so it can never demote an
    identifier on its own; only the documented placeholder shape does that.
    """
    rows = [fx.campaign_row(0, "bf-1-0"), fx.campaign_row(1, "ps-1-2")]
    assert declared_campaign_ids(rows) == frozenset({"bf-1-0", "ps-1-2"})


def test_the_placeholder_is_the_only_shape_that_goes_undeclared() -> None:
    """Filler is recognised; everything else in the table is a campaign."""
    rows = [
        fx.campaign_row(0, "normal-864209"),
        fx.campaign_row(1, "normalish-7"),
        fx.campaign_row(2, "normal-"),
    ]
    assert declared_campaign_ids(rows) == frozenset({"normalish-7", "normal-"})


def test_a_declared_identifier_resolves_to_itself() -> None:
    """Rule 1: presence in the metadata settles it."""
    declared = frozenset({"normal-123"})
    assert campaign_association("normal-123", declared=declared) == "normal-123"


def test_an_undeclared_placeholder_resolves_to_no_association() -> None:
    """Rule 2: both conditions -- documented shape *and* undeclared."""
    assert campaign_association("normal-864209", declared=frozenset()) is None


def test_an_undeclared_identifier_raises_rather_than_resolving() -> None:
    """Rule 3: a dangling reference is a failure, never a singleton."""
    with pytest.raises(DataValidationError, match=UNKNOWN_CAMPAIGN_REFERENCE_CODE):
        campaign_association("bf-1-0", declared=frozenset({"ps-1-2"}))


@pytest.mark.parametrize("value", [None, "", "   "])
def test_an_absent_identifier_resolves_to_no_association(value: object) -> None:
    """Nothing recorded is not a dangling reference; it is nothing recorded."""
    assert campaign_association(value, declared=frozenset()) is None


def test_the_campaign_fingerprint_covers_the_declaration(eligible: Any) -> None:
    """Two tables declaring different campaigns must not fingerprint alike."""
    plan: Plan = [(0, "train", False, "normal", None), (1, "train", False, "x", "c1")]
    plain = build(eligible, plan=plan)
    staged = build(
        eligible,
        plan=plan,
        campaign_rows=[fx.campaign_row(1, "c1", stage="active")],
    )
    assert plain.campaign_fingerprint != staged.campaign_fingerprint


# ---------------------------------------------------------------------------
# Validation of cells and timestamps
# ---------------------------------------------------------------------------


def test_an_unsupported_split_label_fails(eligible: Any) -> None:
    """A split nobody declared is refused, with the supported set named."""
    with pytest.raises(DataValidationError, match="Unsupported split"):
        build(eligible, plan=[(0, "holdout_v2", False, "normal", None)])


def test_a_naive_anchor_timestamp_fails(eligible: Any) -> None:
    """A naive timestamp has no position on the timeline the split rests on."""
    row = fx.feature_row(0, names=eligible.feature_names)
    row["anchor_event_time"] = datetime(2026, 3, 1)
    with pytest.raises(DataValidationError, match="timezone-aware"):
        build(eligible, plan=[(0, "train", False, "normal", None)], feature_rows=[row])


def test_a_non_timestamp_anchor_time_fails(eligible: Any) -> None:
    """A string that looks like a time is not one."""
    row = fx.feature_row(0, names=eligible.feature_names)
    row["anchor_event_time"] = "2026-03-01T00:00:00Z"
    with pytest.raises(DataValidationError, match="must be a timestamp"):
        build(eligible, plan=[(0, "train", False, "normal", None)], feature_rows=[row])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_feature_value_fails(eligible: Any, bad: float) -> None:
    """A missing observation is null, which means undefined. NaN blurs that."""
    row = fx.feature_row(0, names=eligible.feature_names)
    row["user_failure_rate"] = bad
    with pytest.raises(DataValidationError, match="non-finite"):
        build(eligible, plan=[(0, "train", False, "normal", None)], feature_rows=[row])


def test_a_null_feature_value_is_carried_unchanged(eligible: Any) -> None:
    """Null means undefined and survives assembly; imputation is Milestone 3's job."""
    row = fx.feature_row(0, names=eligible.feature_names)
    row["user_failure_rate"] = None
    dataset = build(
        eligible, plan=[(0, "train", False, "normal", None)], feature_rows=[row]
    )
    assert dataset.for_split(MLSplit.TRAIN).feature_matrix[0][0] is None


def test_an_empty_feature_table_fails(eligible: Any) -> None:
    """Zero rows is a mistake upstream, not an empty dataset."""
    with pytest.raises(DataValidationError, match="No feature rows"):
        assemble_ml_dataset(feature_rows=[], labels=[], splits=[], eligible=eligible)


# ---------------------------------------------------------------------------
# What must not reach the matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    ["malicious", "attack_class", "split", "campaign_id", "malicious_probability"],
)
def test_a_prohibited_column_in_the_feature_table_fails(
    eligible: Any, column: str
) -> None:
    """Refused at the source table, before any matrix is built."""
    row = fx.feature_row(0, names=eligible.feature_names)
    row[column] = "anything"
    with pytest.raises(DataValidationError, match="must never reach a model"):
        build(eligible, plan=[(0, "train", False, "normal", None)], feature_rows=[row])


def test_a_hand_built_feature_list_cannot_smuggle_a_label_column(
    eligible: Any,
) -> None:
    """A second line of defence behind the allowlist's own validators."""
    contaminated = eligible.model_copy(
        update={
            "feature_names": (*eligible.feature_names, "malicious"),
            "decision_points": (*eligible.decision_points, eligible.decision_points[0]),
            "leakage_classes": (*eligible.leakage_classes, eligible.leakage_classes[0]),
            "feature_groups": (*eligible.feature_groups, eligible.feature_groups[0]),
        }
    )
    with pytest.raises(DataValidationError, match="name a label, split, campaign"):
        build(contaminated, plan=[(0, "train", False, "normal", None)])


def test_the_matrix_carries_exactly_the_eligible_features(eligible: Any) -> None:
    """Four admitted features, four columns, in the resolved order."""
    dataset = build(eligible, plan=simple_plan())
    assert dataset.feature_names == eligible.feature_names
    for split in MLSplit:
        scoped = dataset.for_split(split)
        assert all(len(row) == 4 for row in scoped.feature_matrix)


def test_identity_and_labels_live_beside_the_matrix_not_in_it(eligible: Any) -> None:
    """They are needed, so they are named fields rather than columns."""
    dataset = build(eligible, plan=simple_plan())
    anchor = dataset.for_split(MLSplit.VALIDATION).anchors[0]
    assert anchor.campaign_id == "c1"
    assert anchor.malicious is True
    assert anchor.split is MLSplit.VALIDATION
    assert "campaign_id" not in dataset.feature_names


# ---------------------------------------------------------------------------
# Ordering and splits
# ---------------------------------------------------------------------------


def test_every_split_is_canonically_ordered(eligible: Any) -> None:
    """Ordering is applied once, immediately after assembly."""
    plan = simple_plan(10)
    shuffled = list(plan)
    random.Random(11).shuffle(shuffled)
    dataset = build(
        eligible,
        plan=shuffled,
        feature_rows=[
            fx.feature_row(index, names=eligible.feature_names)
            for index, *_ in shuffled
        ],
    )
    for split in MLSplit:
        assert is_canonical(dataset.for_split(split).anchors)


def test_input_row_order_does_not_change_the_training_data_fingerprint(
    eligible: Any,
) -> None:
    """The identity of what a fit consumes cannot depend on write order."""
    plan = simple_plan(10)
    forward = build(eligible, plan=plan)
    shuffled = list(plan)
    random.Random(3).shuffle(shuffled)
    backward = build(
        eligible,
        plan=shuffled,
        feature_rows=[
            fx.feature_row(index, names=eligible.feature_names)
            for index, *_ in shuffled
        ],
    )
    assert forward.training_data_fingerprint == backward.training_data_fingerprint
    assert forward.label_fingerprint == backward.label_fingerprint
    assert forward.split_fingerprint == backward.split_fingerprint
    assert forward.campaign_fingerprint == backward.campaign_fingerprint


def test_different_data_produces_a_different_fingerprint(eligible: Any) -> None:
    """A digest that could not tell two datasets apart would pin nothing."""
    plan = simple_plan()
    rows = [fx.feature_row(index, names=eligible.feature_names) for index, *_ in plan]
    baseline = build(eligible, plan=plan, feature_rows=rows)

    perturbed = [dict(row) for row in rows]
    perturbed[0]["user_failure_rate"] = 99.0
    changed = build(eligible, plan=plan, feature_rows=perturbed)
    assert baseline.training_data_fingerprint != changed.training_data_fingerprint


def test_the_splits_partition_every_row(eligible: Any) -> None:
    """Every joined row lands in exactly one split."""
    plan: Plan = [
        (0, "train", False, "normal", None),
        (1, "validation", True, "brute_force", "c1"),
        (2, "test", True, "bot_activity", "c2"),
        (3, "novel_anomaly_holdout", True, "novel_anomaly_holdout", None),
        (4, "excluded", False, "normal", None),
    ]
    dataset = build(eligible, plan=plan)
    assert sum(dataset.for_split(split).row_count for split in MLSplit) == 5
    assert all(dataset.for_split(split).row_count == 1 for split in MLSplit)


def test_support_counts_are_aggregate_only(eligible: Any) -> None:
    """Counts, keyed by split; nothing identifying."""
    dataset = build(eligible, plan=simple_plan())
    counts = dataset.support_counts()
    assert counts["rows"] == 6
    assert counts["train_benign_rows"] == 3
    assert counts["validation_malicious_rows"] == 3
    assert counts["validation_campaigns"] == 1
    assert all(isinstance(value, int) for value in counts.values())


def test_split_scoped_row_counts_agree_with_the_anchors(eligible: Any) -> None:
    """Every parallel tuple in a split describes the same rows."""
    dataset = build(eligible, plan=simple_plan())
    for split in MLSplit:
        scoped = dataset.for_split(split)
        assert (
            scoped.row_count
            == len(scoped.feature_matrix)
            == len(scoped.malicious)
            == len(scoped.known_category)
            == len(scoped.supervised_training_eligible)
        )
        assert scoped.positive_row_count + scoped.benign_row_count == scoped.row_count


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_the_dataset_records_the_feature_contract_it_was_built_from(
    eligible: Any,
) -> None:
    """A fitted model records these, and inference rejects a disagreeing one."""
    dataset = build(eligible, plan=simple_plan(), feature_catalog_fingerprint="a" * 64)
    assert dataset.eligible_feature_list_fingerprint == eligible.fingerprint()
    assert dataset.allowlist_id == eligible.allowlist_id
    assert dataset.feature_catalog_fingerprint == "a" * 64


def test_timestamps_are_normalised_to_utc(eligible: Any) -> None:
    """Two zones, one instant, one recorded time."""
    row = fx.feature_row(0, names=eligible.feature_names)
    row["anchor_event_time"] = fx.at(0).astimezone(UTC)
    dataset = build(
        eligible, plan=[(0, "train", False, "normal", None)], feature_rows=[row]
    )
    assert dataset.for_split(MLSplit.TRAIN).anchors[0].anchor_event_time.tzinfo is UTC
