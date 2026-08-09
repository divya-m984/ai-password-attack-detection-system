"""The validation partition and the fifteen-check eligibility audit.

Two properties recur. The partition's is that **the boundary falls between
whole campaign groups or the run is reported as unsupported** -- there is no
third outcome. The audit's is that **a skipped check is not a pass**, so every
named failure is exercised individually and the overall status is asserted
alongside it.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from typing import Any

import pytest

from password_attack_detector.exceptions import DataValidationError, ModelTrainingError
from password_attack_detector.ml.config import MLConfig, ValidationPartitionConfig
from password_attack_detector.ml.dataset import (
    UNKNOWN_CAMPAIGN_REFERENCE_CODE,
    MLDataset,
    assemble_ml_dataset,
)
from password_attack_detector.ml.eligibility import (
    CHECK_NAMES,
    MLEligibilityAuditor,
    ml_audit_result_to_markdown,
)
from password_attack_detector.ml.enums import (
    AuditCheckStatus,
    AuditStatus,
    MLSplit,
    ValidationPartition,
    ValidationPartitionStatus,
)
from password_attack_detector.ml.features import resolve_eligible_features
from password_attack_detector.ml.partition import partition_validation
from password_attack_detector.ml.schemas import SupportRequirement
from tests.ml import factories as fx

#: One planned row: index, split, malicious, attack class, campaign.
Plan = list[tuple[int, str, bool, str, str | None]]

ALL_CLASSES = ("prior_only", "current_event_context", "baseline_derived")

#: A partition policy sized for hand-written fixtures. Only the *sizes* are
#: relaxed; every policy field keeps its production value, matching the
#: reasoning in ``configs/ml/model-testing.yaml``.
TINY_PARTITION = ValidationPartitionConfig(
    min_partition_rows=1, min_partition_positive_rows=1
)
TINY_SUPPORT = SupportRequirement(
    min_train_positive_rows=1,
    min_validation_positive_rows=1,
    min_validation_benign_rows=1,
    min_rows_per_category=1,
)

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")


@pytest.fixture
def catalog() -> Any:
    """Return the shared four-feature test catalog."""
    return fx.small_catalog()


@pytest.fixture
def allowlist(catalog: Any) -> Any:
    """Return an allowlist admitting every feature of that catalog."""
    return fx.allowlist_for(catalog)


@pytest.fixture
def eligible(catalog: Any, allowlist: Any) -> Any:
    """Return the resolved feature contract."""
    return resolve_eligible_features(
        catalog, allowlist, include_leakage_classes=ALL_CLASSES
    )


@pytest.fixture
def config() -> MLConfig:
    """Return a default ML configuration."""
    return MLConfig()


def manifest_for(catalog: Any) -> dict[str, Any]:
    """Return a feature manifest that agrees with *catalog*."""
    return {
        "feature_catalog_fingerprint": catalog.fingerprint(),
        "feature_schema_version": "1.0.0",
    }


def dataset_from(
    eligible: Any,
    plan: Plan,
    *,
    campaigns: bool = True,
    catalog_fingerprint: str | None = None,
    stages: Mapping[str, str] | None = None,
    declared: frozenset[str] | None = None,
) -> MLDataset:
    """Assemble a dataset from a compact plan.

    *stages* declares campaigns through the metadata's own stage field; *declared*
    states the authoritative campaign set directly.
    """
    staged = stages or {}
    return assemble_ml_dataset(
        feature_rows=[
            fx.feature_row(index, names=eligible.feature_names) for index, *_ in plan
        ],
        labels=[
            fx.label_row(index, malicious=malicious, attack_class=attack_class)
            for index, _, malicious, attack_class, _ in plan
        ],
        splits=[fx.split_row(index, split) for index, split, *_ in plan],
        campaigns=(
            [
                fx.campaign_row(index, campaign, stage=staged.get(campaign))
                for index, _, _, _, campaign in plan
                if campaign is not None
            ]
            if campaigns
            else None
        ),
        declared_campaigns=declared,
        eligible=eligible,
        feature_catalog_fingerprint=catalog_fingerprint,
    )


def healthy_plan() -> Plan:
    """Return a plan with every split populated and two validation campaigns.

    Validation rows 10-19: campaign ``c1`` early, benign singletons between, and
    campaign ``c2`` late, so the group boundary has somewhere meaningful to fall.
    """
    plan: Plan = [
        (0, "train", False, "normal", None),
        (1, "train", True, "brute_force", "t1"),
        (2, "train", False, "normal", None),
        (3, "test", True, "bot_activity", "x1"),
        (4, "novel_anomaly_holdout", True, "novel_anomaly_holdout", None),
        (5, "excluded", False, "normal", None),
    ]
    plan += [
        (10, "validation", True, "brute_force", "c1"),
        (11, "validation", True, "brute_force", "c1"),
        (12, "validation", False, "normal", None),
        (13, "validation", False, "normal", None),
        (14, "validation", True, "password_spraying", "c2"),
        (15, "validation", True, "password_spraying", "c2"),
        (16, "validation", False, "normal", None),
        (17, "validation", False, "normal", None),
    ]
    return plan


def partition_for(dataset: MLDataset, **kwargs: Any) -> Any:
    """Partition the validation split with the tiny fixture policy."""
    kwargs.setdefault("config", TINY_PARTITION)
    kwargs.setdefault("support", TINY_SUPPORT)
    kwargs.setdefault("campaign_metadata_supplied", True)
    return partition_validation(dataset.for_split(MLSplit.VALIDATION), **kwargs)


def audit_of(
    dataset: MLDataset,
    catalog: Any,
    allowlist: Any,
    eligible: Any,
    config: MLConfig,
    **kwargs: Any,
) -> Any:
    """Run the auditor with the manifest and partition supplied by default."""
    kwargs.setdefault("feature_manifest", manifest_for(catalog))
    kwargs.setdefault("partition", partition_for(dataset))
    return MLEligibilityAuditor(
        catalog=catalog,
        allowlist=allowlist,
        eligible=eligible,
        config=config,
        **kwargs,
    ).audit(dataset)


# ---------------------------------------------------------------------------
# Partition: grouping and the boundary
# ---------------------------------------------------------------------------


def test_no_campaign_appears_in_both_halves(eligible: Any) -> None:
    """The central guarantee, checked from the assignment rather than the code."""
    dataset = dataset_from(eligible, healthy_plan())
    result = partition_for(dataset)
    assert result.status is ValidationPartitionStatus.PARTITIONED
    assert result.campaigns_are_disjoint(dataset.for_split(MLSplit.VALIDATION).anchors)


def test_a_campaigns_rows_land_together(eligible: Any) -> None:
    """Every row of one campaign gets the same half, whatever their times."""
    dataset = dataset_from(eligible, healthy_plan())
    result = partition_for(dataset)
    by_campaign: dict[str, set[ValidationPartition]] = {}
    for anchor in dataset.for_split(MLSplit.VALIDATION).anchors:
        if anchor.campaign_id is not None:
            by_campaign.setdefault(anchor.campaign_id, set()).add(
                result.assignment[anchor.anchor_event_id]
            )
    assert by_campaign
    assert all(len(halves) == 1 for halves in by_campaign.values())


def test_benign_rows_form_deterministic_singletons(eligible: Any) -> None:
    """A row with no campaign association is its own group, keyed by its anchor."""
    plan: Plan = [
        (index, "validation", False, "normal", None) for index in range(10, 18)
    ]
    plan.append((0, "train", False, "normal", None))
    dataset = dataset_from(eligible, plan)
    result = partition_for(dataset)
    assert result.partition_a_group_count + result.partition_b_group_count == 8
    assert result.partition_a_campaign_count == 0
    assert result.partition_b_campaign_count == 0


# ---------------------------------------------------------------------------
# Regression: the campaign-group invariant
#
# Every non-null campaign_id is one indivisible group, whatever its rows are
# labelled. Only rows with no campaign association become singletons. No
# row-level label overrides campaign membership.
# ---------------------------------------------------------------------------


def _mixed_campaign_plan() -> Plan:
    """Return a plan whose campaign ``mixed`` holds benign *and* malicious rows.

    Fixture A. The campaign's benign rows sit at the start of the validation
    stream and its malicious rows at the end, with unaffiliated benign rows in
    between -- the arrangement a label-keyed grouping would split, because it
    would see the benign members as ungrouped singletons free to land anywhere.
    """
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [(10, "validation", False, "normal", "mixed")]
    plan += [(11, "validation", False, "normal", "mixed")]
    plan += [(12, "validation", False, "normal", None)]
    plan += [(13, "validation", False, "normal", None)]
    plan += [(14, "validation", True, "brute_force", "other")]
    plan += [(15, "validation", False, "normal", None)]
    plan += [(16, "validation", True, "brute_force", "mixed")]
    plan += [(17, "validation", True, "brute_force", "mixed")]
    return plan


def _halves_of(result: Any, dataset: MLDataset, campaign: str) -> set[Any]:
    """Return the validation halves *campaign*'s rows were assigned to."""
    return {
        result.assignment[anchor.anchor_event_id]
        for anchor in dataset.for_split(MLSplit.VALIDATION).anchors
        if anchor.campaign_id == campaign
    }


def test_a_mixed_benign_and_malicious_campaign_stays_in_one_half(
    eligible: Any,
) -> None:
    """Fixture A: a label never overrides campaign membership.

    The campaign holds two benign rows at the front of the stream and two
    malicious rows at the back. All four must land in the same half, or the
    calibrator and the threshold search have both seen the same campaign.
    """
    dataset = dataset_from(eligible, _mixed_campaign_plan())
    result = partition_for(dataset)

    halves = _halves_of(result, dataset, "mixed")
    assert len(halves) == 1, halves
    assert result.campaigns_are_disjoint(dataset.for_split(MLSplit.VALIDATION).anchors)


def test_the_benign_members_of_a_campaign_are_not_singletons(eligible: Any) -> None:
    """Fixture A, converse: they are grouped, not merely co-located by luck.

    Counting groups distinguishes "the four rows happened to land together" from
    "the four rows are one indivisible group". Four campaign rows, one campaign
    of one malicious row, and three unaffiliated benign singletons make five
    groups; a label-keyed grouping would produce seven.
    """
    dataset = dataset_from(eligible, _mixed_campaign_plan())
    result = partition_for(dataset)
    assert result.partition_a_group_count + result.partition_b_group_count == 5
    assert result.partition_a_campaign_count + result.partition_b_campaign_count == 2


def test_two_campaigns_stay_indivisible_beside_benign_singletons(
    eligible: Any,
) -> None:
    """Fixture B: campaigns are whole, unaffiliated benign rows are singletons."""
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", True, "brute_force", "c1") for index in range(3)
    ]
    plan += [(20 + index, "validation", False, "normal", None) for index in range(4)]
    plan += [
        (30 + index, "validation", True, "password_spraying", "c2")
        for index in range(3)
    ]
    dataset = dataset_from(eligible, plan)
    result = partition_for(dataset)

    for campaign in ("c1", "c2"):
        assert len(_halves_of(result, dataset, campaign)) == 1, campaign
    # Two campaign groups plus four benign singletons.
    assert result.partition_a_group_count + result.partition_b_group_count == 6
    assert result.partition_a_campaign_count + result.partition_b_campaign_count == 2
    assert result.campaigns_are_disjoint(dataset.for_split(MLSplit.VALIDATION).anchors)


@pytest.mark.parametrize("seed", [1, 2, 3, 17, 20260807])
def test_shuffled_input_yields_the_identical_partition_fingerprint(
    eligible: Any, seed: int
) -> None:
    """Fixture C: the fingerprint depends on the data, not on the write order."""
    plan = _mixed_campaign_plan()
    baseline = partition_for(dataset_from(eligible, plan))

    shuffled = list(plan)
    random.Random(seed).shuffle(shuffled)
    reordered = partition_for(dataset_from(eligible, shuffled))

    assert reordered.fingerprint == baseline.fingerprint
    assert reordered.assignment == baseline.assignment
    assert reordered.to_dict() == baseline.to_dict()


def test_a_campaign_that_would_cross_the_target_stays_whole(eligible: Any) -> None:
    """Fixture D: boundary pressure never splits a campaign.

    A four-row campaign starts before the halfway point of a nine-row validation
    split and finishes after it. The boundary has to fall either before or after
    the whole campaign; the target fraction does not get a vote on splitting it.
    """
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [(10 + index, "validation", False, "normal", None) for index in range(3)]
    plan += [
        (20 + index, "validation", True, "brute_force", "straddler")
        for index in range(4)
    ]
    plan += [(30 + index, "validation", False, "normal", None) for index in range(2)]
    dataset = dataset_from(eligible, plan)
    result = partition_for(dataset)

    assert len(_halves_of(result, dataset, "straddler")) == 1
    # Three singletons then the whole four-row campaign: the target of 4.5 rows
    # is first reached at seven, so the campaign lands entirely in validation-A.
    assert result.partition_a_row_count == 7
    assert result.partition_b_row_count == 2
    assert result.campaigns_are_disjoint(dataset.for_split(MLSplit.VALIDATION).anchors)


def test_the_audit_inspects_benign_campaign_rows_too(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Fixture E: the disjointness check looks at every non-null campaign id.

    A check restricted to malicious rows would pass this straddle, because the
    campaign's benign rows are the ones that moved.
    """
    from dataclasses import replace

    plan = _mixed_campaign_plan()
    dataset = dataset_from(eligible, plan, catalog_fingerprint=catalog.fingerprint())
    sound = partition_for(dataset)

    benign_members = [
        anchor
        for anchor in dataset.for_split(MLSplit.VALIDATION).anchors
        if anchor.campaign_id == "mixed" and not anchor.malicious
    ]
    assert benign_members, "the fixture must carry benign campaign rows"

    straddling = dict(sound.assignment)
    for anchor in benign_members:
        straddling[anchor.anchor_event_id] = (
            ValidationPartition.VALIDATION_A
            if straddling[anchor.anchor_event_id] is ValidationPartition.VALIDATION_B
            else ValidationPartition.VALIDATION_B
        )

    moved = replace(sound, assignment=straddling)
    assert not moved.campaigns_are_disjoint(
        dataset.for_split(MLSplit.VALIDATION).anchors
    )
    result = audit_of(dataset, catalog, allowlist, eligible, config, partition=moved)
    _assert_fails(result, "VALIDATION_HALVES_CAMPAIGN_DISJOINT")


def test_a_placeholder_campaign_identifier_does_not_group_rows(
    eligible: Any,
) -> None:
    """The Phase 2 filler id denotes no campaign, so it groups nothing.

    ``GroundTruthLabel.campaign_id`` is required, so the generator writes
    ``normal-<seed>`` on every benign row. Treating that one shared value as a
    campaign would fuse the whole benign class into a single indivisible group.
    The exclusion is by identifier, never by label -- see the companion test
    below, where a real campaign's benign rows stay grouped.
    """
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", False, "normal", "normal-42") for index in range(6)
    ]
    plan += [(20, "validation", True, "brute_force", "c1")]
    dataset = dataset_from(eligible, plan)

    assert all(
        anchor.campaign_id is None
        for anchor in dataset.for_split(MLSplit.VALIDATION).anchors
        if not anchor.malicious
    )
    result = partition_for(dataset)
    assert result.partition_a_group_count + result.partition_b_group_count == 7


def test_the_placeholder_exclusion_is_by_identifier_not_by_label(
    eligible: Any,
) -> None:
    """A benign row in a *named* campaign keeps its campaign association."""
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [(10, "validation", False, "normal", "c1")]
    plan += [(11, "validation", True, "brute_force", "c1")]
    dataset = dataset_from(eligible, plan)
    assert {
        anchor.campaign_id for anchor in dataset.for_split(MLSplit.VALIDATION).anchors
    } == {"c1"}


# ---------------------------------------------------------------------------
# Regression: campaign metadata, not identifier spelling, is the authority
#
# The placeholder pattern is a recogniser of last resort. A declared identifier
# is a campaign however it is spelled; an undeclared one that is not the
# documented placeholder is a dangling reference, never a quiet singleton.
# ---------------------------------------------------------------------------


def test_a_declared_campaign_named_like_the_placeholder_stays_a_campaign(
    eligible: Any,
) -> None:
    """Fixture A: spelling loses to the metadata.

    ``normal-123`` matches the placeholder shape exactly, but the metadata
    declares it, so it is a campaign and its rows are one indivisible group.
    """
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", True, "brute_force", "normal-123")
        for index in range(3)
    ]
    plan += [(20 + index, "validation", False, "normal", None) for index in range(4)]
    dataset = dataset_from(eligible, plan, stages={"normal-123": "active"})

    anchors = dataset.for_split(MLSplit.VALIDATION).anchors
    assert sum(1 for anchor in anchors if anchor.campaign_id == "normal-123") == 3
    result = partition_for(dataset)
    assert len(_halves_of(result, dataset, "normal-123")) == 1
    # Three campaign rows as one group, plus four unaffiliated singletons.
    assert result.partition_a_group_count + result.partition_b_group_count == 5
    assert result.partition_a_campaign_count + result.partition_b_campaign_count == 1


def test_the_same_identifier_declared_or_not_resolves_differently(
    eligible: Any,
) -> None:
    """Fixture A, sharpened: only the declaration differs between these two runs.

    Identical rows, identical identifier, identical labels. Declared, it groups;
    undeclared, it is the documented placeholder and groups nothing. Nothing
    about the *string* decided it.
    """
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", True, "brute_force", "normal-123")
        for index in range(3)
    ]
    plan += [(20 + index, "validation", False, "normal", None) for index in range(4)]

    declared = dataset_from(eligible, plan, stages={"normal-123": "active"})
    undeclared = dataset_from(eligible, plan)

    assert declared.for_split(MLSplit.VALIDATION).campaign_count == 1
    assert undeclared.for_split(MLSplit.VALIDATION).campaign_count == 0


def test_the_generator_placeholder_absent_from_metadata_groups_nothing(
    eligible: Any,
) -> None:
    """Fixture B: the documented filler still resolves to no association."""
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", False, "normal", "normal-864209")
        for index in range(6)
    ]
    plan += [(20, "validation", True, "brute_force", "c1")]
    dataset = dataset_from(eligible, plan)

    anchors = dataset.for_split(MLSplit.VALIDATION).anchors
    assert all(anchor.campaign_id is None for anchor in anchors if not anchor.malicious)
    result = partition_for(dataset)
    # Six deterministic singletons plus the one real campaign.
    assert result.partition_a_group_count + result.partition_b_group_count == 7


def test_an_undeclared_campaign_reference_fails_with_a_stable_code(
    eligible: Any,
) -> None:
    """Fixture C: a dangling reference stops assembly; it never becomes a row.

    Silently demoting an unknown identifier to a singleton would scatter
    whatever it names across both halves, and the partition report would look
    perfectly healthy afterwards.
    """
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", True, "brute_force", "bf-1-0") for index in range(2)
    ]
    plan += [(20, "validation", True, "password_spraying", "ps-9-3")]
    plan += [(30 + index, "validation", False, "normal", None) for index in range(3)]

    with pytest.raises(DataValidationError) as excinfo:
        dataset_from(eligible, plan, declared=frozenset({"bf-1-0"}))

    assert UNKNOWN_CAMPAIGN_REFERENCE_CODE in str(excinfo.value)
    # The message names the boundary, never the identifier it refused.
    assert "ps-9-3" not in str(excinfo.value)


def test_a_declared_campaign_with_mixed_members_stays_indivisible(
    eligible: Any,
) -> None:
    """Fixture D: declaration and label are independent questions."""
    plan = _mixed_campaign_plan()
    dataset = dataset_from(
        eligible, plan, stages={"mixed": "active", "other": "active"}
    )
    result = partition_for(dataset)

    assert len(_halves_of(result, dataset, "mixed")) == 1
    assert result.partition_a_group_count + result.partition_b_group_count == 5
    assert result.campaigns_are_disjoint(dataset.for_split(MLSplit.VALIDATION).anchors)


@pytest.mark.parametrize("seed", [3, 11, 29, 20260809])
def test_shuffled_input_resolves_and_partitions_identically(
    eligible: Any, seed: int
) -> None:
    """Fixture E: resolution, partition, and fingerprint all ignore write order."""
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", True, "brute_force", "normal-123")
        for index in range(3)
    ]
    plan += [
        (20 + index, "validation", False, "normal", "normal-864209")
        for index in range(3)
    ]
    plan += [(30 + index, "validation", False, "normal", None) for index in range(2)]
    stages = {"normal-123": "active"}

    baseline = dataset_from(eligible, plan, stages=stages)
    shuffled = list(plan)
    random.Random(seed).shuffle(shuffled)
    reordered = dataset_from(eligible, shuffled, stages=stages)

    def associations(dataset: MLDataset) -> dict[str, str | None]:
        return {
            anchor.anchor_event_id: anchor.campaign_id
            for anchor in dataset.for_split(MLSplit.VALIDATION).anchors
        }

    assert associations(reordered) == associations(baseline)
    assert set(associations(baseline).values()) == {None, "normal-123"}

    forward, backward = partition_for(baseline), partition_for(reordered)
    assert backward.fingerprint == forward.fingerprint
    assert backward.assignment == forward.assignment
    assert backward.to_dict() == forward.to_dict()


def test_the_partition_is_deterministic_under_input_reordering(eligible: Any) -> None:
    """Two runs over the same rows in different write order agree exactly."""
    plan = healthy_plan()
    forward = partition_for(dataset_from(eligible, plan))
    shuffled = list(plan)
    random.Random(7).shuffle(shuffled)
    backward = partition_for(dataset_from(eligible, shuffled))
    assert forward.fingerprint == backward.fingerprint
    assert forward.assignment == backward.assignment


def test_the_fingerprint_includes_the_support_policy(eligible: Any) -> None:
    """Two runs that reached different conclusions must not digest identically."""
    dataset = dataset_from(eligible, healthy_plan())
    lenient = partition_for(dataset)
    strict = partition_for(
        dataset,
        support=SupportRequirement(
            min_train_positive_rows=1,
            min_validation_positive_rows=1,
            min_validation_benign_rows=2,
            min_rows_per_category=1,
        ),
    )
    assert lenient.fingerprint != strict.fingerprint


def test_the_fingerprint_includes_the_grouping_policy(eligible: Any) -> None:
    """A different target fraction is a different partition contract."""
    dataset = dataset_from(eligible, healthy_plan())
    half = partition_for(dataset)
    quarter = partition_for(
        dataset,
        config=ValidationPartitionConfig(
            min_partition_rows=1,
            min_partition_positive_rows=1,
            target_partition_a_fraction=0.25,
        ),
    )
    assert half.fingerprint != quarter.fingerprint


def test_the_boundary_falls_between_groups_never_inside_one(eligible: Any) -> None:
    """A campaign larger than the target still lands whole on one side."""
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", True, "brute_force", "big") for index in range(8)
    ]
    plan += [(30, "validation", False, "normal", None)]
    dataset = dataset_from(eligible, plan)
    result = partition_for(dataset)
    assert result.partition_a_row_count == 8
    assert result.partition_b_row_count == 1
    assert result.campaigns_are_disjoint(dataset.for_split(MLSplit.VALIDATION).anchors)


def test_a_single_group_yields_insufficient_support_not_a_row_cut(
    eligible: Any,
) -> None:
    """The failure mode a midpoint fallback would have hidden."""
    plan: Plan = [(0, "train", False, "normal", None)]
    plan += [
        (10 + index, "validation", True, "brute_force", "only") for index in range(6)
    ]
    result = partition_for(dataset_from(eligible, plan))
    assert result.status is ValidationPartitionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert not result.usable
    assert result.partition_a_row_count == 0 or result.partition_b_row_count == 0


def test_support_is_checked_independently_for_each_half(eligible: Any) -> None:
    """A combined check would pass a partition where one half carries everything."""
    dataset = dataset_from(eligible, healthy_plan())
    result = partition_for(
        dataset,
        config=ValidationPartitionConfig(
            min_partition_rows=1, min_partition_positive_rows=3
        ),
    )
    assert result.status is ValidationPartitionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    assert "validation_a:min_partition_positive_rows" in result.failing_requirements
    assert "validation_b:min_partition_positive_rows" in result.failing_requirements


def test_the_failing_requirements_are_stable_codes(eligible: Any) -> None:
    """A caller has to be able to branch on the reason, not parse a sentence."""
    dataset = dataset_from(eligible, healthy_plan())
    result = partition_for(
        dataset,
        support=SupportRequirement(
            min_train_positive_rows=1,
            min_validation_positive_rows=1,
            min_validation_benign_rows=99,
            min_rows_per_category=1,
        ),
    )
    assert result.failing_requirements == (
        "validation_a:min_validation_benign_rows",
        "validation_b:min_validation_benign_rows",
    )


# ---------------------------------------------------------------------------
# Partition: refusals
# ---------------------------------------------------------------------------


def test_missing_campaign_metadata_refuses_rather_than_falls_back(
    eligible: Any,
) -> None:
    """A row cut would deliver the very leak this partition prevents."""
    dataset = dataset_from(eligible, healthy_plan(), campaigns=False)
    with pytest.raises(ModelTrainingError, match="no ungrouped fallback"):
        partition_for(dataset, campaign_metadata_supplied=False)


@pytest.mark.parametrize(
    "split", [MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT, MLSplit.TRAIN]
)
def test_only_validation_rows_may_be_partitioned(eligible: Any, split: MLSplit) -> None:
    """Test and holdout rows never enter either half."""
    dataset = dataset_from(eligible, healthy_plan())
    with pytest.raises(ModelTrainingError, match="never enter either half"):
        partition_validation(
            dataset.for_split(split),
            config=TINY_PARTITION,
            support=TINY_SUPPORT,
            campaign_metadata_supplied=True,
        )


def test_an_empty_validation_split_refuses(eligible: Any) -> None:
    """There is nothing to partition, and pretending otherwise helps nobody."""
    dataset = dataset_from(eligible, [(0, "train", False, "normal", None)])
    with pytest.raises(ModelTrainingError, match="nothing to partition"):
        partition_for(dataset)


def test_the_partition_summary_carries_no_identifier(eligible: Any) -> None:
    """``to_dict`` is what a report may embed; the assignment is not."""
    dataset = dataset_from(eligible, healthy_plan())
    summary = partition_for(dataset).to_dict()
    rendered = json.dumps(summary)
    assert "c1" not in rendered
    assert "e0010" not in rendered
    assert "assignment" not in summary


# ---------------------------------------------------------------------------
# Audit: the happy path
# ---------------------------------------------------------------------------


def test_a_sound_dataset_passes_every_check(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """The baseline the failure tests below are measured against."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(dataset, catalog, allowlist, eligible, config)
    assert result.status is AuditStatus.PASS, result.failures
    assert result.passed
    assert result.failures == ()


def test_the_result_carries_exactly_the_declared_checks(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """A check cannot be dropped by deleting its implementation."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(dataset, catalog, allowlist, eligible, config)
    assert tuple(check.name for check in result.checks) == CHECK_NAMES
    assert len(CHECK_NAMES) == 15


def test_the_audit_is_deterministic(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Two runs over one dataset produce identical reports."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    first = audit_of(dataset, catalog, allowlist, eligible, config)
    second = audit_of(dataset, catalog, allowlist, eligible, config)
    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# Audit: a skipped check is not a pass
# ---------------------------------------------------------------------------


def test_omitting_the_manifest_skips_a_check_and_fails_the_audit(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Fourteen passes and one omission must not read as fifteen passes."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(
        dataset, catalog, allowlist, eligible, config, feature_manifest=None
    )
    skipped = next(
        check
        for check in result.checks
        if check.name == "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH"
    )
    assert skipped.status is AuditCheckStatus.SKIPPED
    assert not skipped.passed
    assert result.status is AuditStatus.FAIL
    assert "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH" in result.failures


def test_omitting_the_partition_skips_two_checks_and_fails_the_audit(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Both partition checks depend on the same unsupplied input."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(dataset, catalog, allowlist, eligible, config, partition=None)
    skipped = {
        check.name
        for check in result.checks
        if check.status is AuditCheckStatus.SKIPPED
    }
    assert skipped == {
        "VALIDATION_HALVES_CAMPAIGN_DISJOINT",
        "VALIDATION_SUPPORT_SUFFICIENT",
    }
    assert result.status is AuditStatus.FAIL


# ---------------------------------------------------------------------------
# Audit: every named failure
# ---------------------------------------------------------------------------


def _assert_fails(result: Any, check_name: str) -> None:
    """Assert *check_name* failed and took the overall status with it."""
    named = next(check for check in result.checks if check.name == check_name)
    assert named.status is AuditCheckStatus.FAIL, named.message
    assert result.status is AuditStatus.FAIL
    assert check_name in result.failures


def _contaminate(eligible: Any, name: str) -> Any:
    """Return the eligible list with *name* appended as a bogus column."""
    return eligible.model_copy(
        update={
            "feature_names": (*eligible.feature_names, name),
            "decision_points": (*eligible.decision_points, eligible.decision_points[0]),
            "leakage_classes": (*eligible.leakage_classes, eligible.leakage_classes[0]),
            "feature_groups": (*eligible.feature_groups, eligible.feature_groups[0]),
        }
    )


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("malicious_probability", "NO_PROHIBITED_COLUMNS"),
        ("split", "NO_LABEL_OR_SPLIT_IN_MATRIX"),
        ("campaign_id", "NO_CAMPAIGN_IDENTIFIER_IN_MATRIX"),
        ("a_column_nobody_declared", "EVERY_COLUMN_TRACES_TO_CATALOG"),
    ],
)
def test_a_contaminated_matrix_fails_the_matching_check(
    eligible: Any,
    catalog: Any,
    allowlist: Any,
    config: MLConfig,
    column: str,
    expected: str,
) -> None:
    """The auditor is checked against a matrix the loader would have refused."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    contaminated = _contaminate(eligible, column)
    poisoned = MLDataset(
        feature_names=contaminated.feature_names,
        known_category_classes=dataset.known_category_classes,
        splits=dataset.splits,
        eligible_feature_list_fingerprint=contaminated.fingerprint(),
        allowlist_id=dataset.allowlist_id,
        allowlist_version=dataset.allowlist_version,
        label_fingerprint=dataset.label_fingerprint,
        split_fingerprint=dataset.split_fingerprint,
        campaign_fingerprint=dataset.campaign_fingerprint,
        feature_catalog_fingerprint=dataset.feature_catalog_fingerprint,
        training_data_fingerprint=dataset.training_data_fingerprint,
        joined_row_count=dataset.joined_row_count,
    )
    result = audit_of(poisoned, catalog, allowlist, contaminated, config)
    _assert_fails(result, expected)


def test_a_key_class_column_fails_its_own_check(
    eligible: Any, allowlist: Any, config: MLConfig
) -> None:
    """A join key that reached the matrix is named separately from 'undeclared'."""
    with_key = fx.catalog(
        [
            *fx.small_catalog().specs,
            fx.spec(
                "some_join_column",
                group="key",
                leakage_class="key",
                window=None,
            ),
        ]
    )
    contaminated = _contaminate(eligible, "some_join_column")
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=with_key.fingerprint()
    )
    poisoned = MLDataset(
        **{
            **dataset.__dict__,
            "feature_names": contaminated.feature_names,
        }
    )
    result = audit_of(
        poisoned,
        with_key,
        allowlist,
        contaminated,
        config,
        feature_manifest=manifest_for(with_key),
    )
    _assert_fails(result, "NO_KEY_CLASS_COLUMNS")
    _assert_fails(result, "ALLOWLIST_COVERS_EVERY_MATRIX_COLUMN")


def test_an_unreviewed_catalog_feature_fails_by_name(
    eligible: Any, catalog: Any, config: MLConfig
) -> None:
    """The message names the feature so the reviewer knows what to rule on."""
    narrower = fx.allowlist(
        [
            fx.admission_for(item)
            for item in catalog.specs
            if item.name != "source_failure_rate"
        ],
        catalog_fingerprints=(catalog.fingerprint(),),
    )
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(dataset, catalog, narrower, eligible, config)
    _assert_fails(result, "NO_UNREVIEWED_CATALOG_FEATURE")
    named = next(
        check
        for check in result.checks
        if check.name == "NO_UNREVIEWED_CATALOG_FEATURE"
    )
    assert "source_failure_rate" in named.message


def test_unordered_rows_fail_the_ordering_check(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """The auditor re-checks the order rather than trusting the loader applied it."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    train = dataset.for_split(MLSplit.TRAIN)
    reversed_train = type(train)(
        split=train.split,
        feature_names=train.feature_names,
        anchors=tuple(reversed(train.anchors)),
        feature_matrix=tuple(reversed(train.feature_matrix)),
        malicious=tuple(reversed(train.malicious)),
        known_category=tuple(reversed(train.known_category)),
        supervised_training_eligible=tuple(
            reversed(train.supervised_training_eligible)
        ),
    )
    poisoned = MLDataset(
        **{
            **dataset.__dict__,
            "splits": {**dataset.splits, MLSplit.TRAIN: reversed_train},
        }
    )
    result = audit_of(poisoned, catalog, allowlist, eligible, config)
    _assert_fails(result, "ROWS_ARE_CANONICALLY_ORDERED")


def test_an_anchor_in_two_splits_fails_disjointness(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Split sets must partition the rows, not merely cover them."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    train = dataset.for_split(MLSplit.TRAIN)
    poisoned = MLDataset(
        **{**dataset.__dict__, "splits": {**dataset.splits, MLSplit.TEST: train}}
    )
    result = audit_of(poisoned, catalog, allowlist, eligible, config)
    _assert_fails(result, "SPLIT_SETS_DISJOINT")


def test_a_holdout_row_in_the_train_split_fails(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Holdout rows must not reach a fittable split under any label."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    holdout = dataset.for_split(MLSplit.NOVEL_ANOMALY_HOLDOUT)
    train = dataset.for_split(MLSplit.TRAIN)
    merged = type(train)(
        split=MLSplit.TRAIN,
        feature_names=train.feature_names,
        anchors=(*train.anchors, *holdout.anchors),
        feature_matrix=(*train.feature_matrix, *holdout.feature_matrix),
        malicious=(*train.malicious, *holdout.malicious),
        known_category=(*train.known_category, *holdout.known_category),
        supervised_training_eligible=(
            *train.supervised_training_eligible,
            *holdout.supervised_training_eligible,
        ),
    )
    poisoned = MLDataset(
        **{**dataset.__dict__, "splits": {**dataset.splits, MLSplit.TRAIN: merged}}
    )
    result = audit_of(poisoned, catalog, allowlist, eligible, config)
    _assert_fails(result, "HOLDOUT_AND_EXCLUDED_ABSENT_FROM_FIT")


@pytest.mark.parametrize(
    "manifest",
    [
        {"feature_catalog_fingerprint": "b" * 64, "feature_schema_version": "1.0.0"},
        {"feature_catalog_fingerprint": None, "feature_schema_version": "9.9.9"},
    ],
)
def test_a_disagreeing_manifest_fails_the_fingerprint_check(
    eligible: Any,
    catalog: Any,
    allowlist: Any,
    config: MLConfig,
    manifest: dict[str, Any],
) -> None:
    """The manifest, the catalog, and the allowlist must describe one contract."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(
        dataset, catalog, allowlist, eligible, config, feature_manifest=manifest
    )
    _assert_fails(result, "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH")


def test_an_allowlist_pinning_another_catalog_fails(
    eligible: Any, catalog: Any, config: MLConfig
) -> None:
    """A reviewed allowlist has to have been reviewed against *this* catalog."""
    elsewhere = fx.allowlist(
        [fx.admission_for(item) for item in catalog.specs],
        catalog_fingerprints=("c" * 64,),
    )
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(dataset, catalog, elsewhere, eligible, config)
    _assert_fails(result, "SCHEMA_AND_CATALOG_FINGERPRINT_MATCH")


def test_a_row_accounting_mismatch_fails_join_integrity(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Rows joined, rows across splits, and distinct anchors must all agree."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    poisoned = MLDataset(**{**dataset.__dict__, "joined_row_count": 999})
    result = audit_of(poisoned, catalog, allowlist, eligible, config)
    _assert_fails(result, "JOIN_KEY_INTEGRITY")


def test_an_insufficiently_supported_partition_fails_its_check(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """The partition's negative outcome propagates into the audit."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    starved = partition_for(
        dataset,
        support=SupportRequirement(
            min_train_positive_rows=1,
            min_validation_positive_rows=1,
            min_validation_benign_rows=99,
            min_rows_per_category=1,
        ),
    )
    result = audit_of(dataset, catalog, allowlist, eligible, config, partition=starved)
    _assert_fails(result, "VALIDATION_SUPPORT_SUFFICIENT")


def test_a_campaign_straddling_the_boundary_fails_its_check(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Asserted against a hand-built violation the partitioner cannot produce."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    sound = partition_for(dataset)
    anchors = dataset.for_split(MLSplit.VALIDATION).anchors
    campaign_rows = [a for a in anchors if a.campaign_id == "c1"]
    assert len(campaign_rows) == 2
    straddling = dict(sound.assignment)
    straddling[campaign_rows[0].anchor_event_id] = ValidationPartition.VALIDATION_A
    straddling[campaign_rows[1].anchor_event_id] = ValidationPartition.VALIDATION_B

    from dataclasses import replace

    result = audit_of(
        dataset,
        catalog,
        allowlist,
        eligible,
        config,
        partition=replace(sound, assignment=straddling),
    )
    _assert_fails(result, "VALIDATION_HALVES_CAMPAIGN_DISJOINT")


def test_a_test_row_inside_a_validation_half_fails_selection_provenance(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Nothing selected may draw on a test or holdout row."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    sound = partition_for(dataset)
    test_anchor = dataset.for_split(MLSplit.TEST).anchors[0]
    leaked = dict(sound.assignment)
    leaked[test_anchor.anchor_event_id] = ValidationPartition.VALIDATION_B

    from dataclasses import replace

    result = audit_of(
        dataset,
        catalog,
        allowlist,
        eligible,
        config,
        partition=replace(sound, assignment=leaked),
    )
    _assert_fails(result, "NO_TEST_OR_HOLDOUT_SELECTION_SOURCE")


def test_the_validation_partition_enum_names_no_evaluation_split() -> None:
    """The absence of the member is the enforcement, so the absence is asserted."""
    named = {str(item) for item in ValidationPartition}
    assert named == {"validation_a", "validation_b"}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def test_the_json_report_carries_no_identifier_or_campaign(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """A privacy sweep over the machine-readable output."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    rendered = json.dumps(
        audit_of(dataset, catalog, allowlist, eligible, config).to_dict()
    )
    assert not _UUID_RE.search(rendered)
    assert not _PSEUDONYM_RE.search(rendered)
    for forbidden in ("e0010", "e0000", "c1", "c2", "t1", "x1"):
        assert f'"{forbidden}"' not in rendered, forbidden


def test_the_markdown_report_carries_no_identifier_or_campaign(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """The same sweep over the human-readable output."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    rendered = ml_audit_result_to_markdown(
        audit_of(dataset, catalog, allowlist, eligible, config)
    )
    assert not _UUID_RE.search(rendered)
    assert not _PSEUDONYM_RE.search(rendered)
    assert "e0010" not in rendered


def test_the_markdown_report_states_the_skipped_check_rule(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """A reader must not have to know the doctrine to read the table correctly."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    rendered = ml_audit_result_to_markdown(
        audit_of(dataset, catalog, allowlist, eligible, config)
    )
    lowered = rendered.lower()
    assert "skipped" in lowered
    assert "not a passed check" in lowered
    assert "says nothing about detection effectiveness" in lowered
    assert "no model has been fitted" in lowered


def test_the_markdown_report_marks_a_skipped_check_distinctly(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Skipped must not render the same as passed."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    rendered = ml_audit_result_to_markdown(
        audit_of(dataset, catalog, allowlist, eligible, config, partition=None)
    )
    assert "SKIPPED" in rendered


def test_the_reports_render_every_check(
    eligible: Any, catalog: Any, allowlist: Any, config: MLConfig
) -> None:
    """Nothing is summarised away."""
    dataset = dataset_from(
        eligible, healthy_plan(), catalog_fingerprint=catalog.fingerprint()
    )
    result = audit_of(dataset, catalog, allowlist, eligible, config)
    rendered = ml_audit_result_to_markdown(result)
    payload = result.to_dict()
    for name in CHECK_NAMES:
        assert name in rendered
        assert any(check["name"] == name for check in payload["checks"])
