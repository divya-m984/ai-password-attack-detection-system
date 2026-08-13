"""Campaign-disjoint, chronological partitioning of the validation split.

Validation is used for two different jobs, and doing both on the same rows is a
leak: a calibrator fitted on rows that also chose the operating point makes the
operating point look better calibrated than it is.  So validation is cut in two.

**Validation-A** fits calibration, and is the only permitted source for an
anomaly threshold selected from benign flag rate.  **Validation-B** selects the
binary decision threshold, the category abstention threshold, the model
candidate, and the fusion strategy.

The cut is placed **between campaign groups**, never between rows.  A campaign
is a coordinated burst by one operator; its early events and its late events are
the same behaviour twice.  A row-midpoint cut would put some of them in the half
that fits the calibrator and the rest in the half that chooses the threshold,
and the threshold would be chosen partly by memorising a campaign the calibrator
had already seen.

The invariant, exactly:

* every non-null ``campaign_id`` defines **one indivisible group**, whatever mix
  of benign and malicious rows it holds;
* only rows with **no campaign association** become deterministic singletons,
  keyed by anchor identifier;
* no row-level label ever overrides campaign membership;
* grouping happens **before** support is tallied and before the boundary is
  placed, so nothing downstream can divide a campaign to hit a target;
* groups are ordered by ``(minimum anchor_event_time, minimum anchor_event_id)``
  and whole groups accumulate into validation-A until the target fraction is
  first reached;
* no campaign appears in both halves, and there is no midpoint, row-halving, or
  campaign-straddling fallback.

**This is stricter than the Phase 3 splitter, deliberately.**  Phase 3 applies
``normal_grouping: singleton``, keyed on the recorded scenario, so a benign row
is ungrouped there even when it carries a campaign identifier.  That is
defensible for the train/validation/test split, whose benign class would
otherwise be discarded wholesale by the exclusion policy.  It is not good enough
for *this* partition: validation-A fits the calibrator and validation-B chooses
the operating point, so a campaign with rows on both sides means the operating
point was chosen partly by memorising activity the calibrator had already fitted
on.  Phase 5 keeps every campaign whole and accepts the cost.  The Phase 5
invariant is not weakened to match Phase 3.

Two consequences worth stating:

* **Campaign metadata is required.**  ``campaign_id`` is not published in any
  Phase 3 feature, label, or split table -- the splitter reads it internally and
  never puts it beside a model input.  So a campaign-grouped partition has to be
  handed the Phase 2 label table explicitly, and refuses to run without it.
  There is no ungrouped fallback: falling back to a row cut would silently
  deliver the leak this module exists to prevent.
* **Insufficient support is an outcome, not an exception.**  Two halves that are
  each too small to measure anything are reported as
  :attr:`~password_attack_detector.ml.enums.ValidationPartitionStatus.INSUFFICIENT_VALIDATION_SUPPORT`,
  naming the failing requirements.  Halving the rows anyway would hand back
  something that looks usable and is not.

Campaign identifiers reach this module and stop here.  They group rows and enter
the partition fingerprint; they never reach a design matrix, and nothing this
module returns for reporting contains one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.config import ValidationPartitionConfig
from password_attack_detector.ml.dataset import AnchorMetadata, SplitDataset
from password_attack_detector.ml.enums import (
    MLSplit,
    ValidationPartition,
    ValidationPartitionStatus,
)
from password_attack_detector.ml.ordering import assert_canonical
from password_attack_detector.ml.schemas import SupportRequirement

__all__ = [
    "ValidationPartitionResult",
    "partition_validation",
]


@dataclass(frozen=True, slots=True)
class _Group:
    """One indivisible unit of the partition.

    A campaign, or a single benign row.  ``is_campaign`` is carried so the
    fingerprint can distinguish "one campaign of forty rows" from "forty
    singletons" without publishing which campaign it was.
    """

    group_id: str
    is_campaign: bool
    row_indices: tuple[int, ...]
    min_event_time: datetime
    min_anchor_id: str
    row_count: int
    positive_count: int

    @property
    def sort_key(self) -> tuple[datetime, str]:
        """Return the deterministic group ordering key."""
        return (self.min_event_time, self.min_anchor_id)


def _has_campaign(anchor: AnchorMetadata) -> bool:
    """Return whether *anchor* belongs to a campaign that must not be divided.

    The rule is exactly one condition: **a non-null campaign identifier**.  No
    label is consulted.  A campaign containing a mix of benign and malicious
    events is still one campaign, and splitting it because some of its rows
    happen to be benign would put the same coordinated activity on both sides of
    the boundary -- precisely the leak this partition exists to prevent.

    Whether a *recorded value* denotes a campaign association at all is decided
    earlier, at the reader boundary in :mod:`password_attack_detector.ml.dataset`.
    By the time an anchor reaches here, ``campaign_id`` is either a real campaign
    or ``None``, and this predicate does not second-guess it.

    The single definition of "campaign group", used by the grouping *and* by the
    disjointness check, so the two cannot disagree about what they are grouping.
    """
    return anchor.campaign_id is not None


@dataclass(frozen=True)
class ValidationPartitionResult:
    """The outcome of partitioning the validation split.

    ``assignment`` maps an anchor identifier to its half.  It is an in-memory
    working structure for the trainer, not report content: everything published
    about a partition comes from the counts and the fingerprint below.
    """

    status: ValidationPartitionStatus
    assignment: Mapping[str, ValidationPartition]
    partition_a_row_count: int
    partition_b_row_count: int
    partition_a_positive_count: int
    partition_b_positive_count: int
    partition_a_benign_count: int
    partition_b_benign_count: int
    partition_a_group_count: int
    partition_b_group_count: int
    partition_a_campaign_count: int
    partition_b_campaign_count: int
    failing_requirements: tuple[str, ...]
    fingerprint: str

    @property
    def usable(self) -> bool:
        """Return whether both halves carry enough support to mean anything."""
        return self.status is ValidationPartitionStatus.PARTITIONED

    def campaigns_are_disjoint(self, anchors: Sequence[AnchorMetadata]) -> bool:
        """Return whether any campaign has rows in both halves.

        Inspects **every non-null campaign identifier**, benign and malicious
        alike.  A check that looked only at malicious rows would miss the case
        that matters most: a campaign whose benign rows landed in one half and
        whose malicious rows landed in the other would be reported as disjoint
        while being exactly the leak in question.

        Recomputed from the assignment rather than asserted by construction.
        The construction does guarantee it, which is exactly why it is worth
        checking independently: a property that can only be verified by reading
        the code that provides it is not an audit.

        Uses :func:`_has_campaign`, the same predicate the grouping used.  A
        check with its own notion of "campaign" would not be checking the
        partition -- it would be checking a different partition nobody built.
        """
        seen: dict[str, set[ValidationPartition]] = {}
        for anchor in anchors:
            if not _has_campaign(anchor):
                continue
            half = self.assignment.get(anchor.anchor_event_id)
            if half is None:
                continue
            assert anchor.campaign_id is not None
            seen.setdefault(anchor.campaign_id, set()).add(half)
        return all(len(halves) == 1 for halves in seen.values())

    def to_dict(self) -> dict[str, Any]:
        """Return an aggregate-only, JSON-serialisable summary.

        Counts and a digest.  No anchor identifier, no campaign identifier, and
        no assignment mapping: this is what a report may carry.
        """
        return {
            "status": str(self.status),
            "partition_a_rows": self.partition_a_row_count,
            "partition_b_rows": self.partition_b_row_count,
            "partition_a_malicious_rows": self.partition_a_positive_count,
            "partition_b_malicious_rows": self.partition_b_positive_count,
            "partition_a_benign_rows": self.partition_a_benign_count,
            "partition_b_benign_rows": self.partition_b_benign_count,
            "partition_a_groups": self.partition_a_group_count,
            "partition_b_groups": self.partition_b_group_count,
            "partition_a_campaigns": self.partition_a_campaign_count,
            "partition_b_campaigns": self.partition_b_campaign_count,
            "failing_requirements": list(self.failing_requirements),
            "fingerprint": self.fingerprint,
        }


def _build_groups(
    anchors: Sequence[AnchorMetadata], malicious: Sequence[bool]
) -> tuple[_Group, ...]:
    """Return the indivisible groups, sorted deterministically.

    Every non-null ``campaign_id`` becomes **one indivisible group**, whatever
    its size and whatever mix of benign and malicious rows it holds.  Only rows
    with no campaign association become singletons, keyed by their anchor
    identifier.

    Grouping happens here, before support is tallied and before the boundary is
    placed, so no later step can divide a campaign to satisfy a target fraction
    or a support floor.

    **This is deliberately stricter than the Phase 3 splitter.**  Phase 3 applies
    ``normal_grouping: singleton``, which keys on the recorded scenario and
    therefore treats a benign row as ungrouped even when it carries a campaign
    identifier.  That is a reasonable rule for the train/validation/test split,
    whose benign class would otherwise be discarded wholesale.  It is *not* good
    enough here: validation-A fits the calibrator and validation-B chooses the
    operating point, so a campaign with rows on both sides means the operating
    point is chosen partly by memorising activity the calibrator already saw.
    Phase 5 keeps the whole campaign together and accepts the cost.
    """
    members: dict[str, list[int]] = {}
    is_campaign: dict[str, bool] = {}
    for index, anchor in enumerate(anchors):
        if _has_campaign(anchor):
            key = f"campaign:{anchor.campaign_id}"
            is_campaign[key] = True
        else:
            key = f"row:{anchor.anchor_event_id}"
            is_campaign[key] = False
        members.setdefault(key, []).append(index)

    groups = [
        _Group(
            group_id=key,
            is_campaign=is_campaign[key],
            row_indices=tuple(indices),
            min_event_time=min(anchors[i].anchor_event_time for i in indices),
            min_anchor_id=min(anchors[i].anchor_event_id for i in indices),
            row_count=len(indices),
            positive_count=sum(1 for i in indices if malicious[i]),
        )
        for key, indices in members.items()
    ]
    return tuple(sorted(groups, key=lambda group: group.sort_key))


def _fingerprint(
    config: ValidationPartitionConfig,
    support: SupportRequirement,
    ordered_groups: Sequence[_Group],
    boundary: int,
) -> str:
    """Return a digest covering the grouping, the policy, and the assignment.

    The support policy is included on purpose.  Two runs that partitioned the
    same rows the same way but under different support floors reached different
    *conclusions* about whether those halves were usable, and a digest that
    could not tell them apart would let one run's partition be mistaken for the
    other's.
    """
    payload = {
        "grouping": config.fingerprint_data(),
        "support": support.fingerprint_data(),
        "groups": [
            {
                "group_id": group.group_id,
                "is_campaign": group.is_campaign,
                "row_count": group.row_count,
                "partition": (
                    str(ValidationPartition.VALIDATION_A)
                    if index < boundary
                    else str(ValidationPartition.VALIDATION_B)
                ),
            }
            for index, group in enumerate(ordered_groups)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _support_failures(
    half: str,
    *,
    tally: _HalfTally,
    config: ValidationPartitionConfig,
    support: SupportRequirement,
) -> list[str]:
    """Return the stable requirement codes *half* fails.

    Checked independently for each half.  A combined check would pass on a
    partition where one half carries everything, which is the failure mode most
    worth catching.
    """
    failures: list[str] = []
    if tally.rows < config.min_partition_rows:
        failures.append(f"{half}:min_partition_rows")
    if tally.positives < config.min_partition_positive_rows:
        failures.append(f"{half}:min_partition_positive_rows")
    if tally.positives < support.min_validation_positive_rows:
        failures.append(f"{half}:min_validation_positive_rows")
    if tally.benign < support.min_validation_benign_rows:
        failures.append(f"{half}:min_validation_benign_rows")
    return failures


@dataclass
class _HalfTally:
    """Running counts for one half, filled as whole groups are assigned."""

    rows: int = 0
    positives: int = 0
    benign: int = 0
    groups: int = 0

    def add(self, group: _Group) -> None:
        """Fold one whole group into this half."""
        self.rows += group.row_count
        self.positives += group.positive_count
        self.benign += group.row_count - group.positive_count
        self.groups += 1


def partition_validation(
    validation: SplitDataset,
    *,
    config: ValidationPartitionConfig,
    support: SupportRequirement,
    campaign_metadata_supplied: bool,
) -> ValidationPartitionResult:
    """Partition the validation split into calibration and selection halves.

    Args:
        validation: the validation split's rows, canonically ordered.
        config: the grouping and boundary policy.
        support: the minimum support each half must carry.
        campaign_metadata_supplied: whether the Phase 2 campaign table was
            actually provided.  Passed explicitly rather than inferred from the
            rows, because "no row carries a campaign" and "nobody supplied the
            campaign table" look identical from here and mean opposite things.

    Returns:
        A result whose ``status`` is ``PARTITIONED`` when both halves carry
        enough support, and ``INSUFFICIENT_VALIDATION_SUPPORT`` otherwise.  The
        assignment is returned either way, because a caller reporting the
        negative outcome still wants the counts that produced it.

    Raises:
        ModelTrainingError: if handed rows from a split other than validation,
            rows that are not canonically ordered, an empty split, or no
            campaign metadata.
    """
    if validation.split is not MLSplit.VALIDATION:
        raise ModelTrainingError(
            f"The validation partitioner was handed {str(validation.split)!r} "
            f"rows; test and holdout rows never enter either half"
        )
    if not campaign_metadata_supplied:
        raise ModelTrainingError(
            "Campaign-grouped validation partitioning requires campaign "
            "metadata, and none was supplied. Campaign identifiers are "
            "deliberately absent from the Phase 3 feature, label, and split "
            "tables, so the Phase 2 label table must be passed explicitly. "
            "There is no ungrouped fallback: a row cut would place one "
            "campaign's events on both sides of the boundary, which is the "
            "leak this partition exists to prevent"
        )
    assert_canonical(validation.anchors, stage="validation partitioning")

    anchors = validation.anchors
    if not anchors:
        raise ModelTrainingError(
            "The validation split carries no rows; there is nothing to partition"
        )

    groups = _build_groups(anchors, validation.malicious)
    target = config.target_partition_a_fraction * len(anchors)

    # Accumulate whole groups until the target is first reached. The boundary is
    # an index into the group list, so it can only ever fall between groups --
    # there is no code path that splits one, and none that adjusts the index
    # afterwards. A boundary that leaves one half empty stays where it landed
    # and is reported as insufficient support; nudging it would be exactly the
    # silent fallback this module refuses to have.
    accumulated = 0
    boundary = 0
    for index, group in enumerate(groups):
        if accumulated >= target:
            break
        accumulated += group.row_count
        boundary = index + 1

    assignment: dict[str, ValidationPartition] = {}
    tallies = {
        ValidationPartition.VALIDATION_A: _HalfTally(),
        ValidationPartition.VALIDATION_B: _HalfTally(),
    }
    campaigns: dict[ValidationPartition, set[str]] = {
        ValidationPartition.VALIDATION_A: set(),
        ValidationPartition.VALIDATION_B: set(),
    }
    for index, group in enumerate(groups):
        half = (
            ValidationPartition.VALIDATION_A
            if index < boundary
            else ValidationPartition.VALIDATION_B
        )
        tallies[half].add(group)
        if group.is_campaign:
            campaigns[half].add(group.group_id)
        for row_index in group.row_indices:
            assignment[anchors[row_index].anchor_event_id] = half

    first = tallies[ValidationPartition.VALIDATION_A]
    second = tallies[ValidationPartition.VALIDATION_B]

    failures = _support_failures(
        "validation_a", tally=first, config=config, support=support
    ) + _support_failures("validation_b", tally=second, config=config, support=support)

    status = (
        ValidationPartitionStatus.PARTITIONED
        if not failures
        else ValidationPartitionStatus.INSUFFICIENT_VALIDATION_SUPPORT
    )

    return ValidationPartitionResult(
        status=status,
        assignment=assignment,
        partition_a_row_count=first.rows,
        partition_b_row_count=second.rows,
        partition_a_positive_count=first.positives,
        partition_b_positive_count=second.positives,
        partition_a_benign_count=first.benign,
        partition_b_benign_count=second.benign,
        partition_a_group_count=first.groups,
        partition_b_group_count=second.groups,
        partition_a_campaign_count=len(campaigns[ValidationPartition.VALIDATION_A]),
        partition_b_campaign_count=len(campaigns[ValidationPartition.VALIDATION_B]),
        failing_requirements=tuple(sorted(failures)),
        fingerprint=_fingerprint(config, support, groups, boundary),
    )
