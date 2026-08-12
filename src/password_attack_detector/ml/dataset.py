"""Dataset assembly -- **the only ML module permitted to read ground truth**.

Labels, split assignments, and campaign metadata enter the machine-learning
layer here and nowhere else.  ``ml.features``, ``ml.ordering``,
``ml.eligibility``, ``ml.partition``, and every module later milestones add are
forbidden from importing a label reader, and an AST test over the whole source
tree asserts the allowlist is exactly two modules: this one and
``detection.evaluation``.

The reason it is *this* module rather than the trainer: a boundary is only worth
having if it is narrow.  One module that reads labels, joins them, and hands
back a structure with the labels in named fields is auditable.  A layer where
several modules each read "just the split column" is not, and the second reader
is always the one that seems harmless.

**What leaves this module, and what does not.**  The design matrix carries the
reviewed eligible features and nothing else.  Anchor identifiers, anchor times,
split assignments, campaign identifiers, ground-truth labels, and supervised
eligibility all leave too -- as *separate, named fields* alongside the matrix.
They are needed: the partitioner groups by campaign, the auditor checks split
disjointness, the trainer selects rows by split.  What must never happen is one
of them becoming a *column*, so :class:`SplitDataset` gives them nowhere to hide
and the auditor checks the matrix against every prohibited name anyway.

**Campaign metadata is the authority on campaign membership -- not identifier
spelling.**  Resolving a recorded ``campaign_id`` into either a campaign or
``None`` happens here, once, before any grouping, so the partitioner receives
already-resolved values and stays entirely label-agnostic.  A declared
identifier is a campaign however it is spelled; only the generator's documented
benign placeholder resolves to no association, and only while the metadata
declares nothing about it; anything else undeclared is a dangling reference and
stops the assembly.  See :func:`declared_campaign_ids` and
:func:`campaign_association`.

Null semantics are Phase 3's, unchanged: a null means the quantity is
**undefined** for this row -- no attempts were made, no baseline was fitted --
and zero means it was **observed to be zero**.  Nothing here imputes; that is
Milestone 3's job, and it will ship a missingness indicator beside every
imputed column so the distinction survives.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.exceptions import DataValidationError
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    PROHIBITED_FEATURE_COLUMNS,
)
from password_attack_detector.features.serialization import (
    FEATURE_LABEL_COLUMNS,
    FEATURE_SPLIT_COLUMNS,
)
from password_attack_detector.features.splitting import SplitLabel
from password_attack_detector.ml.enums import MLSplit
from password_attack_detector.ml.features import (
    ML_OUTPUT_COLUMNS,
    RESERVED_MATRIX_COLUMNS,
    EligibleFeatureList,
)
from password_attack_detector.ml.ordering import canonicalize_rows

__all__ = [
    "CAMPAIGN_COLUMNS",
    "GENERATOR_PLACEHOLDER_PATTERN",
    "KNOWN_CATEGORY_CLASSES",
    "UNKNOWN_CAMPAIGN_REFERENCE_CODE",
    "AnchorMetadata",
    "CampaignRow",
    "InferenceAnchor",
    "InferenceDataset",
    "InferenceFrame",
    "LabelRow",
    "MLDataset",
    "SplitDataset",
    "SplitRow",
    "assemble_inference_dataset",
    "assemble_ml_dataset",
    "campaign_association",
    "declared_campaign_ids",
    "load_inference_dataset",
    "load_ml_dataset",
]

#: Column names that carry campaign membership.  Present in the Phase 2 label
#: table, absent from every published feature table, and never a model input.
CAMPAIGN_COLUMNS: Final[frozenset[str]] = frozenset({"campaign_id", "campaign_stage"})

#: The Phase 2 generator's documented placeholder identifier shape.
#:
#: ``GroundTruthLabel.campaign_id`` is a required field, so the generator has to
#: write *something* on every row -- including the ordinary background traffic
#: that belongs to no campaign at all.  It writes ``normal-<seed>``
#: (``data/synthetic/campaigns.py``): one value shared by every benign event in
#: the dataset, naming no coordinated activity.  Every campaign the generator
#: actually runs is declared in the metadata it emits alongside those rows.
#:
#: **This pattern is a recogniser, never an authority.**  Matching it is a
#: *necessary* condition for resolving an identifier to "no campaign", never a
#: sufficient one: the supplied campaign metadata is consulted first, and an
#: identifier the metadata declares stays a campaign however it is spelled.
#: Deciding otherwise would let a naming coincidence dissolve a real campaign
#: into singleton rows, which is exactly the leak this layer exists to prevent.
GENERATOR_PLACEHOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^normal-\d+$")

#: Stable code for a campaign reference the supplied metadata does not declare.
#:
#: ``MLD`` marks the ML *dataset-assembly* boundary, distinct from the ``M0xx``
#: codes reserved for prediction-artifact validation in a later milestone.
UNKNOWN_CAMPAIGN_REFERENCE_CODE: Final[str] = "MLD001"


def declared_campaign_ids(campaigns: Sequence[CampaignRow]) -> frozenset[str]:
    """Return the campaigns the supplied metadata declares.

    The metadata is the authority on what a campaign is; this function only
    reads the declaration out of it.  An identifier is declared when the
    metadata records campaign structure for it -- a stage, which the generator
    writes on every campaign it runs and on nothing else -- **or** when it is not
    the documented benign placeholder.  The second clause is what keeps an
    ordinary campaign table, which carries no stages at all, working exactly as
    before: everything in it is a campaign unless it is the one identifier shape
    the generator documents as filler.

    The asymmetry is deliberate.  A recorded stage is a positive declaration and
    outranks the placeholder shape, so a real campaign that happens to be named
    ``normal-123`` is still a campaign.  Absence of a stage declares nothing
    either way, so it can never *demote* an identifier on its own.

    No label is read here or anywhere downstream of it.  Whether a row is benign
    or malicious has no bearing on which campaign it belongs to.
    """
    staged = {
        row.campaign_id.strip()
        for row in campaigns
        if row.campaign_stage is not None and row.campaign_id.strip()
    }
    unstaged = {
        text
        for row in campaigns
        if (text := row.campaign_id.strip())
        and not GENERATOR_PLACEHOLDER_PATTERN.match(text)
    }
    return frozenset(staged | unstaged)


def campaign_association(value: object, *, declared: frozenset[str]) -> str | None:
    """Return the campaign *value* refers to, or ``None`` when it refers to none.

    *declared* is the authoritative set of campaigns, read from the supplied
    campaign metadata.  Resolution consults it before anything else:

    1. an identifier the metadata declares is a campaign, whatever its spelling;
    2. an undeclared identifier matching the generator's documented placeholder
       contract denotes no campaign association;
    3. any other undeclared identifier is a dangling reference and raises.

    Rule 3 is the point of the exercise.  Quietly turning an identifier nobody
    declared into a singleton row would scatter whatever it names across both
    validation halves, and the resulting partition would look perfectly healthy
    -- the failure mode this layer is built to make impossible.

    Raises:
        DataValidationError: on an undeclared, non-placeholder identifier.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in declared:
        return text
    if GENERATOR_PLACEHOLDER_PATTERN.match(text):
        return None
    raise DataValidationError(
        f"[{UNKNOWN_CAMPAIGN_REFERENCE_CODE}] A row references a campaign the "
        f"supplied campaign metadata does not declare, and which is not the "
        f"generator's documented benign placeholder; campaign metadata is the "
        f"authority on campaign membership, so the reference cannot be resolved"
    )


def _known_category_classes() -> tuple[str, ...]:
    """Return the category head's class space, derived from the label schema.

    Every malicious :class:`~password_attack_detector.data.enums.ScenarioType`
    except the novel-anomaly holdout, sorted by value.  Derived at runtime
    rather than written down, so adding a scenario to the generator extends the
    class space and renaming one fails the test that pins this tuple, instead of
    scoring quietly against a class that no longer exists.

    ``normal`` is excluded because the category head is fitted on malicious rows
    only; ``novel_anomaly_holdout`` is excluded because its whole purpose is to
    measure behaviour on attacks no class was fitted for, and admitting it as a
    class would destroy that measurement.
    """
    return tuple(
        sorted(
            str(scenario)
            for scenario in ScenarioType
            if scenario not in {ScenarioType.NORMAL, ScenarioType.NOVEL_ANOMALY_HOLDOUT}
        )
    )


#: The secondary head's deterministic class order.
KNOWN_CATEGORY_CLASSES: Final[tuple[str, ...]] = _known_category_classes()


def _assert_split_enums_agree() -> None:
    """Fail at import if the ML split mirror has drifted from Phase 3's.

    ``ml.enums.MLSplit`` mirrors ``features.splitting.SplitLabel`` so that no ML
    module except this one has to import the label-bearing module.  A mirror can
    drift, so the one module that sees both checks them -- loudly, at import,
    rather than by producing rows filed under a split that no longer exists.
    """
    mirrored = {str(item) for item in MLSplit}
    declared = {str(item) for item in SplitLabel}
    if mirrored != declared:
        raise DataValidationError(
            f"MLSplit has drifted from SplitLabel: only in MLSplit "
            f"{sorted(mirrored - declared)}, only in SplitLabel "
            f"{sorted(declared - mirrored)}"
        )


def _assert_reserved_columns_cover_the_published_tables() -> None:
    """Fail at import if a published label or split column is not reserved.

    ``ml.features.RESERVED_MATRIX_COLUMNS`` is what the rest of the layer checks
    a design matrix against, precisely so no other ML module has to import the
    label-bearing Phase 3 modules for a tuple of column names.  That indirection
    is only safe while it stays complete, so the one module permitted to see
    both ends checks it here rather than trusting two lists to be edited
    together.
    """
    published = set(FEATURE_LABEL_COLUMNS) | set(FEATURE_SPLIT_COLUMNS)
    missing = sorted(published - RESERVED_MATRIX_COLUMNS)
    if missing:
        raise DataValidationError(
            f"Published label or split column(s) {missing} are absent from "
            f"RESERVED_MATRIX_COLUMNS; the design-matrix guard would not "
            f"reject them"
        )


_assert_split_enums_agree()
_assert_reserved_columns_cover_the_published_tables()


@dataclass(frozen=True, slots=True)
class LabelRow:
    """One ground-truth label, as published by Phase 3."""

    event_id: str
    attack_class: str
    malicious: bool
    supervised_training_eligible: bool = True


@dataclass(frozen=True, slots=True)
class SplitRow:
    """One split assignment, as published by Phase 3."""

    event_id: str
    split: str
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignRow:
    """One event's campaign membership, from the Phase 2 label table.

    A separate input because ``FEATURE_LABEL_COLUMNS`` deliberately omits
    ``campaign_id``: the Phase 3 splitter reads it internally for group
    isolation, but it is never published beside a model input.  Campaign-grouped
    validation partitioning therefore *requires* this table to be handed over
    explicitly, and refuses to run without it rather than degrading to a row cut.

    ``campaign_stage`` is carried because it is the metadata's own declaration
    that a campaign exists -- see :func:`declared_campaign_ids`.  It is never a
    feature, never a label, and never leaves this module.
    """

    event_id: str
    campaign_id: str
    campaign_stage: str | None = None


@dataclass(frozen=True, slots=True)
class AnchorMetadata:
    """Everything about a row that is not a feature.

    Kept beside the matrix rather than in it.  The partitioner needs the
    campaign, the auditor needs the split, the trainer needs supervised
    eligibility -- and none of them is a model input.
    """

    anchor_event_id: str
    anchor_event_time: datetime
    split: MLSplit
    malicious: bool
    attack_class: str
    known_category: str | None
    supervised_training_eligible: bool
    campaign_id: str | None


@dataclass(frozen=True, slots=True)
class SplitDataset:
    """One split's rows: metadata, design matrix, and targets, all row-aligned.

    Every tuple here has the same length and the same canonical row order, so
    index *i* means the same event in all of them.
    """

    split: MLSplit
    feature_names: tuple[str, ...]
    anchors: tuple[AnchorMetadata, ...]
    feature_matrix: tuple[tuple[Any, ...], ...]
    malicious: tuple[bool, ...]
    known_category: tuple[str | None, ...]
    supervised_training_eligible: tuple[bool, ...]

    @property
    def row_count(self) -> int:
        """Return the number of rows in this split."""
        return len(self.anchors)

    @property
    def positive_row_count(self) -> int:
        """Return the number of malicious rows."""
        return sum(1 for value in self.malicious if value)

    @property
    def benign_row_count(self) -> int:
        """Return the number of benign rows."""
        return self.row_count - self.positive_row_count

    @property
    def campaign_count(self) -> int:
        """Return the number of distinct campaigns represented."""
        return len({a.campaign_id for a in self.anchors if a.campaign_id is not None})

    def category_counts(self) -> dict[str, int]:
        """Return per-known-category row counts, zero-filled over the class space.

        Zero-filled on purpose: a class with no rows must appear as ``0`` rather
        than be absent, or a support check reading this mapping would mistake
        "no support" for "not applicable".
        """
        counts = dict.fromkeys(KNOWN_CATEGORY_CLASSES, 0)
        for value in self.known_category:
            if value is not None:
                counts[value] = counts.get(value, 0) + 1
        return counts


@dataclass(frozen=True)
class MLDataset:
    """The assembled, canonically ordered, split-scoped dataset.

    Carries the fingerprints that give a training run its identity.  A model
    fitted from this dataset records them, and inference refuses a model whose
    recorded fingerprints disagree with the catalog it is handed.
    """

    feature_names: tuple[str, ...]
    known_category_classes: tuple[str, ...]
    splits: Mapping[MLSplit, SplitDataset]
    eligible_feature_list_fingerprint: str
    allowlist_id: str
    allowlist_version: str
    label_fingerprint: str
    split_fingerprint: str
    campaign_fingerprint: str | None
    feature_catalog_fingerprint: str | None
    training_data_fingerprint: str
    joined_row_count: int

    def for_split(self, split: MLSplit) -> SplitDataset:
        """Return the rows assigned to *split*, possibly empty."""
        return self.splits[split]

    def support_counts(self) -> dict[str, int]:
        """Return aggregate row counts per split and class.

        Counts only.  Safe to render in a report: no identifier, no pseudonym,
        no campaign, no feature value.
        """
        counts: dict[str, int] = {"rows": self.joined_row_count}
        for split in MLSplit:
            dataset = self.splits[split]
            counts[f"{split}_rows"] = dataset.row_count
            counts[f"{split}_malicious_rows"] = dataset.positive_row_count
            counts[f"{split}_benign_rows"] = dataset.benign_row_count
            counts[f"{split}_campaigns"] = dataset.campaign_count
        return counts


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _digest(payload: object) -> str:
    """Return a SHA-256 digest over a canonical JSON rendering of *payload*."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _scalar(value: object) -> Any:
    """Render one cell into a JSON-stable, fingerprint-safe scalar.

    Floats are formatted at nine decimals, matching every other fingerprint in
    this project, so a digest does not change with the last bit of a repr.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _unique_or_raise(identifiers: Sequence[str], what: str) -> None:
    """Raise when *identifiers* are not unique, reporting counts only."""
    seen: set[str] = set()
    duplicates = 0
    for identifier in identifiers:
        if identifier in seen:
            duplicates += 1
        seen.add(identifier)
    if duplicates:
        raise DataValidationError(
            f"{what} carries {duplicates} duplicate identifier(s) across "
            f"{len(identifiers)} row(s); the join must be one-to-one"
        )


def _reject_prohibited_columns(columns: Sequence[str]) -> None:
    """Raise if a feature row carries a label, split, campaign, or output column."""
    present = set(columns)
    forbidden = (
        PROHIBITED_FEATURE_COLUMNS
        | ML_OUTPUT_COLUMNS
        | CAMPAIGN_COLUMNS
        | set(FEATURE_LABEL_COLUMNS)
        | set(FEATURE_SPLIT_COLUMNS)
    ) - {"event_id"}
    offending = sorted(present & forbidden)
    if offending:
        raise DataValidationError(
            f"Feature rows carry column(s) that must never reach a model: {offending}"
        )


def _reject_matrix_contamination(feature_names: Sequence[str]) -> None:
    """Raise if the resolved feature order names anything that is not a feature.

    A second line of defence behind the allowlist's own validators.  The
    allowlist refuses to *admit* a reserved name; this refuses to *build a
    matrix* from one, so a hand-constructed feature list cannot bypass the file.
    """
    offending = sorted(
        set(feature_names)
        & (
            PROHIBITED_FEATURE_COLUMNS
            | ML_OUTPUT_COLUMNS
            | CAMPAIGN_COLUMNS
            | RESERVED_MATRIX_COLUMNS
            | set(FEATURE_LABEL_COLUMNS)
            | set(FEATURE_SPLIT_COLUMNS)
        )
    )
    if offending:
        raise DataValidationError(
            f"Design matrix column(s) {offending} name a label, split, campaign, "
            f"identifier, or model output"
        )


def _validated_cell(value: object, column: str) -> Any:
    """Return *value* unless it is a non-finite float.

    ``NaN`` and infinity are refused rather than carried.  ``NaN != NaN`` breaks
    every exact-equality comparison this layer's determinism guarantees rest on,
    and a null already has a precise meaning here -- undefined -- that ``NaN``
    would blur.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise DataValidationError(
            f"Feature column {column!r} carries a non-finite value; a missing "
            f"observation must be null, which means undefined, not NaN"
        )
    return value


def _validated_anchor_time(value: object) -> datetime:
    """Return *value* as a UTC-aware datetime, or raise."""
    if not isinstance(value, datetime):
        raise DataValidationError(
            f"{ANCHOR_EVENT_TIME} must be a timestamp, got {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataValidationError(
            f"{ANCHOR_EVENT_TIME} must be timezone-aware; a naive timestamp "
            f"cannot be placed on the timeline the split contract depends on"
        )
    return value.astimezone(UTC)


def _validated_split(value: str) -> MLSplit:
    """Return *value* as an :class:`MLSplit`, or raise naming the supported set."""
    try:
        return MLSplit(value)
    except ValueError:
        raise DataValidationError(
            f"Unsupported split {value!r}; supported: "
            f"{sorted(str(item) for item in MLSplit)}"
        ) from None


def _known_category_for(attack_class: str, malicious: bool) -> str | None:
    """Return the known category for a row, or ``None`` when there is not one.

    ``None`` covers three genuinely different situations -- a benign row, the
    novel-anomaly holdout, and a malicious row whose class is not in the known
    space -- and the category head treats all three the same way: it does not
    learn from them.  What it must never do is force one into the nearest known
    class, which is why
    :data:`~password_attack_detector.ml.enums.UNKNOWN_CATEGORY` exists as a
    first-class *prediction* outcome even though it is never a *fitted* class.
    """
    if not malicious:
        return None
    if attack_class in KNOWN_CATEGORY_CLASSES:
        return attack_class
    return None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _JoinedRow:
    """One fully joined row, before it is split-scoped."""

    anchor_event_id: str
    anchor_event_time: datetime
    values: tuple[Any, ...]
    split: MLSplit
    malicious: bool
    attack_class: str
    known_category: str | None
    supervised_training_eligible: bool
    campaign_id: str | None


def assemble_ml_dataset(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[LabelRow],
    splits: Sequence[SplitRow],
    eligible: EligibleFeatureList,
    campaigns: Sequence[CampaignRow] | None = None,
    declared_campaigns: frozenset[str] | None = None,
    feature_catalog_fingerprint: str | None = None,
) -> MLDataset:
    """Join Phase 3 features, labels, splits, and campaigns into one dataset.

    Every relationship is one-to-one on the anchor identifier, and every
    validation below runs *before* a single design-matrix cell is built.  An
    assembly that has already produced a matrix and then discovers a duplicate
    key has to decide what to do with the matrix; one that never built it does
    not.

    Args:
        feature_rows: Phase 3 feature snapshot rows, in any order.
        labels: Phase 3 ground-truth labels, one per event.
        splits: Phase 3 split assignments, one per event.
        eligible: the resolved, reviewed feature contract.
        campaigns: Phase 2 campaign membership.  Optional here; required by the
            partitioner, which refuses to guess.
        declared_campaigns: the authoritative set of campaigns.  Read from
            *campaigns* when omitted; supply it to state the declaration
            directly rather than have it inferred.
        feature_catalog_fingerprint: recorded for provenance when supplied.

    Raises:
        DataValidationError: on a duplicate key, an asymmetric relationship, a
            prohibited column, an unsupported split, a naive timestamp, a
            non-finite value, a missing declared feature, or a campaign
            reference the metadata does not declare
            (:data:`UNKNOWN_CAMPAIGN_REFERENCE_CODE`).
    """
    _reject_matrix_contamination(eligible.feature_names)

    if not feature_rows:
        raise DataValidationError("No feature rows were supplied")

    _reject_prohibited_columns(sorted({key for row in feature_rows for key in row}))

    anchor_ids = [str(row[ANCHOR_EVENT_ID]) for row in feature_rows]
    _unique_or_raise(anchor_ids, "The feature table")
    _unique_or_raise([label.event_id for label in labels], "The label table")
    _unique_or_raise([split.event_id for split in splits], "The split table")
    if campaigns is not None:
        _unique_or_raise([row.event_id for row in campaigns], "The campaign table")

    label_by_id = {label.event_id: label for label in labels}
    split_by_id = {split.event_id: split for split in splits}
    # Resolved here, once, against the supplied metadata and before any
    # grouping: a declared identifier is a campaign whatever its spelling, the
    # documented benign placeholder is dropped, and anything else the metadata
    # does not declare stops the assembly rather than becoming a singleton.
    declared = (
        frozenset()
        if campaigns is None
        else (
            declared_campaign_ids(campaigns)
            if declared_campaigns is None
            else declared_campaigns
        )
    )
    campaign_by_id = (
        {}
        if campaigns is None
        else {
            row.event_id: association
            for row in campaigns
            if (association := campaign_association(row.campaign_id, declared=declared))
            is not None
        }
    )

    anchor_set = set(anchor_ids)
    _require_symmetry(anchor_set, set(label_by_id), "label")
    _require_symmetry(anchor_set, set(split_by_id), "split")
    if campaigns is not None:
        # Checked against every supplied row, not against the resolved mapping:
        # an orphan carrying a placeholder identifier is still an orphan, and
        # filtering placeholders first would hide it.
        orphans = len({row.event_id for row in campaigns} - anchor_set)
        if orphans:
            raise DataValidationError(
                f"The campaign table carries {orphans} row(s) with no matching "
                f"feature anchor; campaign metadata must describe the same events"
            )

    joined: list[_JoinedRow] = []
    for row in feature_rows:
        anchor_id = str(row[ANCHOR_EVENT_ID])
        label = label_by_id[anchor_id]
        split_record = split_by_id[anchor_id]
        missing = [name for name in eligible.feature_names if name not in row]
        if missing:
            raise DataValidationError(
                f"The feature table is missing {len(missing)} admitted feature "
                f"column(s), including {sorted(missing)[:5]}"
            )
        values = tuple(
            _validated_cell(row[name], name) for name in eligible.feature_names
        )
        joined.append(
            _JoinedRow(
                anchor_event_id=anchor_id,
                anchor_event_time=_validated_anchor_time(row[ANCHOR_EVENT_TIME]),
                values=values,
                split=_validated_split(split_record.split),
                malicious=bool(label.malicious),
                attack_class=label.attack_class,
                known_category=_known_category_for(
                    label.attack_class, bool(label.malicious)
                ),
                supervised_training_eligible=bool(label.supervised_training_eligible),
                campaign_id=campaign_by_id.get(anchor_id),
            )
        )

    ordered = canonicalize_rows(joined)

    split_datasets = {
        split: _scope_to_split(ordered, split, eligible.feature_names)
        for split in MLSplit
    }

    return MLDataset(
        feature_names=eligible.feature_names,
        known_category_classes=KNOWN_CATEGORY_CLASSES,
        splits=split_datasets,
        eligible_feature_list_fingerprint=eligible.fingerprint(),
        allowlist_id=eligible.allowlist_id,
        allowlist_version=eligible.allowlist_version,
        label_fingerprint=_label_fingerprint(labels),
        split_fingerprint=_split_fingerprint(splits),
        campaign_fingerprint=(
            None if campaigns is None else _campaign_fingerprint(campaigns)
        ),
        feature_catalog_fingerprint=feature_catalog_fingerprint,
        training_data_fingerprint=_training_data_fingerprint(ordered, eligible),
        joined_row_count=len(ordered),
    )


def _require_symmetry(anchors: set[str], other: set[str], what: str) -> None:
    """Raise unless the feature anchors and *other* describe the same events."""
    missing = len(anchors - other)
    extra = len(other - anchors)
    if missing or extra:
        raise DataValidationError(
            f"The feature and {what} tables do not describe the same events: "
            f"{missing} anchor(s) have no {what}, {extra} {what} row(s) have no "
            f"anchor"
        )


def _scope_to_split(
    rows: Sequence[_JoinedRow], split: MLSplit, feature_names: tuple[str, ...]
) -> SplitDataset:
    """Return the subset of *rows* assigned to *split*, order preserved."""
    selected = [row for row in rows if row.split is split]
    return SplitDataset(
        split=split,
        feature_names=feature_names,
        anchors=tuple(
            AnchorMetadata(
                anchor_event_id=row.anchor_event_id,
                anchor_event_time=row.anchor_event_time,
                split=row.split,
                malicious=row.malicious,
                attack_class=row.attack_class,
                known_category=row.known_category,
                supervised_training_eligible=row.supervised_training_eligible,
                campaign_id=row.campaign_id,
            )
            for row in selected
        ),
        feature_matrix=tuple(row.values for row in selected),
        malicious=tuple(row.malicious for row in selected),
        known_category=tuple(row.known_category for row in selected),
        supervised_training_eligible=tuple(
            row.supervised_training_eligible for row in selected
        ),
    )


def _label_fingerprint(labels: Sequence[LabelRow]) -> str:
    """Return a digest of the label table, independent of row order."""
    return _digest(
        [
            {
                "event_id": label.event_id,
                "attack_class": label.attack_class,
                "malicious": bool(label.malicious),
                "supervised_training_eligible": bool(
                    label.supervised_training_eligible
                ),
            }
            for label in sorted(labels, key=lambda item: item.event_id)
        ]
    )


def _split_fingerprint(splits: Sequence[SplitRow]) -> str:
    """Return a digest of the split table, independent of row order."""
    return _digest(
        [
            {
                "event_id": row.event_id,
                "split": row.split,
                "exclusion_reason": row.exclusion_reason,
            }
            for row in sorted(splits, key=lambda item: item.event_id)
        ]
    )


def _campaign_fingerprint(campaigns: Sequence[CampaignRow]) -> str:
    """Return a digest of the campaign table, independent of row order.

    Covers the stage as well as the identifier: the stage is what declares a
    campaign, so a table that declares a different set of campaigns must not
    fingerprint the same as one that does not.
    """
    return _digest(
        [
            {
                "event_id": row.event_id,
                "campaign_id": row.campaign_id,
                "campaign_stage": row.campaign_stage,
            }
            for row in sorted(campaigns, key=lambda item: item.event_id)
        ]
    )


def _training_data_fingerprint(
    rows: Sequence[_JoinedRow], eligible: EligibleFeatureList
) -> str:
    """Return a digest of exactly what a fit would consume.

    Covers the resolved feature contract, the canonical row order, and every
    cell.  Two runs over the same data in different file order produce the same
    digest because the rows are already canonically sorted; two runs over
    different data do not.
    """
    return _digest(
        {
            "eligible_feature_list_fingerprint": eligible.fingerprint(),
            "feature_names": list(eligible.feature_names),
            "rows": [
                {
                    "anchor_event_id": row.anchor_event_id,
                    "anchor_event_time": row.anchor_event_time.isoformat(),
                    "split": str(row.split),
                    "malicious": row.malicious,
                    "known_category": row.known_category,
                    "supervised_training_eligible": row.supervised_training_eligible,
                    "values": [_scalar(value) for value in row.values],
                }
                for row in rows
            ],
        }
    )


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------


def _read_rows(path: Path, what: str) -> list[dict[str, Any]]:
    """Return every row of a Parquet table as a mapping.

    The exception message names the failure *type* only.  A pyarrow error can
    quote a file path, and a run log is not where a personal directory should
    turn up.
    """
    import pyarrow.parquet as pq

    try:
        return list(pq.read_table(path).to_pylist())
    except Exception as exc:
        raise DataValidationError(
            f"Cannot read the {what} ({type(exc).__name__})"
        ) from None


def load_ml_dataset(
    *,
    features_path: Path,
    labels_path: Path,
    splits_path: Path,
    eligible: EligibleFeatureList,
    campaign_labels_path: Path | None = None,
    feature_catalog_fingerprint: str | None = None,
) -> MLDataset:
    """Read the Phase 3 tables from disk and assemble them.

    A thin shell over :func:`assemble_ml_dataset`: it reads Parquet and converts
    rows into typed records, and does no joining or validating of its own.  Unit
    tests exercise the assembly directly on in-memory rows, which is why none of
    them needs a Parquet fixture.

    Raises:
        DataValidationError: if a file cannot be read or a required column is
            absent, in addition to every assembly failure.
    """
    feature_rows = _read_rows(features_path, "feature snapshot table")
    label_rows = _read_rows(labels_path, "label table")
    split_rows = _read_rows(splits_path, "split table")

    for row in feature_rows:
        for column in (ANCHOR_EVENT_ID, ANCHOR_EVENT_TIME):
            if column not in row:
                raise DataValidationError(
                    f"The feature snapshot table has no {column!r} column"
                )
        break

    labels = [
        LabelRow(
            event_id=str(row["event_id"]),
            attack_class=str(row.get("attack_class") or str(ScenarioType.NORMAL)),
            malicious=bool(row.get("malicious")),
            supervised_training_eligible=bool(
                row.get("supervised_training_eligible", True)
            ),
        )
        for row in label_rows
    ]
    split_records = [
        SplitRow(
            event_id=str(row["event_id"]),
            split=str(row.get("split") or ""),
            exclusion_reason=(
                None
                if row.get("exclusion_reason") is None
                else str(row["exclusion_reason"])
            ),
        )
        for row in split_rows
    ]

    campaigns = None
    if campaign_labels_path is not None:
        campaign_rows = _read_rows(campaign_labels_path, "campaign label table")
        campaigns = [
            CampaignRow(
                event_id=str(row["event_id"]),
                campaign_id=str(row["campaign_id"]),
                campaign_stage=(
                    None
                    if row.get("campaign_stage") is None
                    else str(row["campaign_stage"])
                ),
            )
            for row in campaign_rows
            if row.get("campaign_id")
        ]

    return assemble_ml_dataset(
        feature_rows=feature_rows,
        labels=labels,
        splits=split_records,
        campaigns=campaigns,
        eligible=eligible,
        feature_catalog_fingerprint=feature_catalog_fingerprint,
    )


# ---------------------------------------------------------------------------
# Inference input -- features and split membership, and no ground truth at all
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InferenceAnchor:
    """A row's join identity, and the only identity inference carries.

    Satisfies :class:`~password_attack_detector.ml.ordering.AnchoredRow`, which
    is all the ordering guard and the preprocessor need.  There is no label
    field here, no campaign, and no supervised-eligibility flag, because the
    loader that builds these never opened a table containing any of them.
    """

    anchor_event_id: str
    anchor_event_time: datetime


@dataclass(frozen=True, slots=True)
class _InferenceRow:
    """One scoped inference row, before it becomes a frame."""

    anchor_event_id: str
    anchor_event_time: datetime
    values: tuple[Any, ...]
    split: MLSplit


@dataclass(frozen=True, slots=True)
class InferenceFrame:
    """The rows a frozen model may be scored over.

    Structurally satisfies
    :class:`~password_attack_detector.ml.preprocessing.FeatureFrame` and nothing
    wider.  A frozen preprocessor accepts it exactly as it accepts a training
    split, and there is no field on it a label could arrive in.
    """

    split: MLSplit
    feature_names: tuple[str, ...]
    anchors: tuple[InferenceAnchor, ...]
    feature_matrix: tuple[tuple[Any, ...], ...]

    @property
    def row_count(self) -> int:
        """Return the number of rows to be scored."""
        return len(self.anchors)


@dataclass(frozen=True)
class InferenceDataset:
    """One scope's inference input, canonically ordered and fingerprinted.

    The deliberate counterpart of :class:`MLDataset`: same feature contract,
    same canonical ordering, same fingerprint discipline -- and no
    ``label_fingerprint``, because nothing here read a label to fingerprint.

    ``inference_input_fingerprint`` covers the resolved feature contract, the
    requested scope, and every anchor and cell that will be scored. It is what
    makes a prediction's identity move when the input moves, and it cannot move
    when a label changes.
    """

    scope: MLSplit
    frame: InferenceFrame
    feature_names: tuple[str, ...]
    eligible_feature_list_fingerprint: str
    allowlist_id: str
    allowlist_version: str
    feature_catalog_fingerprint: str | None
    split_membership_fingerprint: str
    inference_input_fingerprint: str
    scoped_row_count: int
    source_row_count: int

    @property
    def row_count(self) -> int:
        """Return the number of rows in scope."""
        return self.scoped_row_count


def assemble_inference_dataset(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    splits: Sequence[SplitRow],
    eligible: EligibleFeatureList,
    scope: MLSplit,
    feature_catalog_fingerprint: str | None = None,
) -> InferenceDataset:
    """Join features to split membership for one scope, reading no ground truth.

    The narrow inference counterpart of :func:`assemble_ml_dataset`. It takes no
    labels, no campaign metadata, and no supervised-eligibility flags, and there
    is no parameter through which any of them could be supplied -- which is the
    firewall stated as a signature rather than as a convention. Split membership
    *is* read, because scoring "the test rows" requires knowing which rows those
    are, and split assignment is not an outcome.

    Args:
        feature_rows: Phase 3 feature snapshot rows, in any order.
        splits: Phase 3 split assignments, one per event.
        eligible: the resolved, reviewed feature contract the frozen model was
            fitted under.
        scope: the split whose rows are to be scored.
        feature_catalog_fingerprint: recorded for provenance when supplied.

    Raises:
        DataValidationError: on a duplicate anchor, an asymmetric split table, a
            prohibited column, a missing admitted feature, an unsupported split,
            a naive timestamp, a non-finite value, or an empty scope.
    """
    _reject_matrix_contamination(eligible.feature_names)
    if not feature_rows:
        raise DataValidationError("No feature rows were supplied")
    _reject_prohibited_columns(sorted({key for row in feature_rows for key in row}))

    anchor_ids = [str(row[ANCHOR_EVENT_ID]) for row in feature_rows]
    _unique_or_raise(anchor_ids, "The feature table")
    _unique_or_raise([split.event_id for split in splits], "The split table")
    split_by_id = {split.event_id: split for split in splits}
    _require_symmetry(set(anchor_ids), set(split_by_id), "split")

    rows: list[_InferenceRow] = []
    for row in feature_rows:
        anchor_id = str(row[ANCHOR_EVENT_ID])
        missing = [name for name in eligible.feature_names if name not in row]
        if missing:
            raise DataValidationError(
                f"The feature table is missing {len(missing)} admitted feature "
                f"column(s), including {sorted(missing)[:5]}"
            )
        rows.append(
            _InferenceRow(
                anchor_event_id=anchor_id,
                anchor_event_time=_validated_anchor_time(row[ANCHOR_EVENT_TIME]),
                values=tuple(
                    _validated_cell(row[name], name) for name in eligible.feature_names
                ),
                split=_validated_split(split_by_id[anchor_id].split),
            )
        )

    # Canonicalised over the whole table before scoping, so the order a row ends
    # up in does not depend on which other rows happened to share its scope.
    ordered = canonicalize_rows(rows)
    scoped = [row for row in ordered if row.split is scope]
    if not scoped:
        raise DataValidationError(
            f"No rows are assigned to {str(scope)!r}; there is nothing to score"
        )

    frame = InferenceFrame(
        split=scope,
        feature_names=eligible.feature_names,
        anchors=tuple(
            InferenceAnchor(
                anchor_event_id=row.anchor_event_id,
                anchor_event_time=row.anchor_event_time,
            )
            for row in scoped
        ),
        feature_matrix=tuple(row.values for row in scoped),
    )
    return InferenceDataset(
        scope=scope,
        frame=frame,
        feature_names=eligible.feature_names,
        eligible_feature_list_fingerprint=eligible.fingerprint(),
        allowlist_id=eligible.allowlist_id,
        allowlist_version=eligible.allowlist_version,
        feature_catalog_fingerprint=feature_catalog_fingerprint,
        split_membership_fingerprint=_split_fingerprint(splits),
        inference_input_fingerprint=_inference_input_fingerprint(
            scoped, eligible=eligible, scope=scope
        ),
        scoped_row_count=len(scoped),
        source_row_count=len(ordered),
    )


def _inference_input_fingerprint(
    rows: Sequence[_InferenceRow], *, eligible: EligibleFeatureList, scope: MLSplit
) -> str:
    """Return a digest of exactly what will be scored.

    The feature contract, the scope, the canonical row order, and every cell.
    Physical file order cannot reach it -- the rows are already canonically
    sorted -- and neither can a label, because none was read.
    """
    return _digest(
        {
            "eligible_feature_list_fingerprint": eligible.fingerprint(),
            "feature_names": list(eligible.feature_names),
            "scope": str(scope),
            "rows": [
                {
                    "anchor_event_id": row.anchor_event_id,
                    "anchor_event_time": row.anchor_event_time.isoformat(),
                    "values": [_scalar(value) for value in row.values],
                }
                for row in rows
            ],
        }
    )


def load_inference_dataset(
    *,
    features_path: Path,
    splits_path: Path,
    eligible: EligibleFeatureList,
    scope: MLSplit,
    feature_catalog_fingerprint: str | None = None,
) -> InferenceDataset:
    """Read the feature and split tables from disk and scope them for inference.

    Two paths, and there is no third: the label table is not a parameter here,
    so a caller cannot supply one by mistake and a reviewer does not have to
    check whether one was.

    Raises:
        DataValidationError: if a file cannot be read or a required column is
            absent, in addition to every assembly failure.
    """
    feature_rows = _read_rows(features_path, "feature snapshot table")
    split_rows = _read_rows(splits_path, "split table")

    for row in feature_rows:
        for column in (ANCHOR_EVENT_ID, ANCHOR_EVENT_TIME):
            if column not in row:
                raise DataValidationError(
                    f"The feature snapshot table has no {column!r} column"
                )
        break

    return assemble_inference_dataset(
        feature_rows=feature_rows,
        splits=[
            SplitRow(
                event_id=str(row["event_id"]),
                split=str(row.get("split") or ""),
                exclusion_reason=(
                    None
                    if row.get("exclusion_reason") is None
                    else str(row["exclusion_reason"])
                ),
            )
            for row in split_rows
        ],
        eligible=eligible,
        scope=scope,
        feature_catalog_fingerprint=feature_catalog_fingerprint,
    )
