"""Deterministic chronological, campaign-aware dataset splitting.

Three ideas do the work here.

**Chronological, never random.**  Splits are cut on time.  A random split would
place an attack campaign's early events in training and its later events in
test, letting a model memorise the campaign rather than learn the behaviour.

**Campaign-aware.**  All events of one synthetic attack campaign must land in
the same supervised split.  A campaign whose events straddle a boundary is, by
default, excluded entirely -- the only policy with no leakage risk and the only
one that is trivially auditable.

**Purged.**  Features look *backwards*, so contamination flows *forward across*
a boundary: an event just after ``train_end`` has a lookback window reaching
into training data.  ``purge`` therefore excludes events on the **later** side
of each boundary, and ``embargo`` excludes them on the **earlier** side.

Nothing is ever dropped silently.  Every input event receives an assignment,
and that is asserted, not merely intended.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.exceptions import SplitConfigurationError
from password_attack_detector.features.config import FeatureConfig, SplitConfig
from password_attack_detector.features.engine import sort_events_canonically

__all__ = [
    "SPLIT_SCHEMA_VERSION",
    "SUPERVISED_SPLITS",
    "ChronologicalSplitter",
    "ExclusionReason",
    "SplitAssignment",
    "SplitLabel",
    "SplitManifest",
    "SplitResult",
]

SPLIT_SCHEMA_VERSION: str = "1.0.0"


class SplitLabel(StrEnum):
    """Where an event may be used."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    #: Evaluation-only pool for anomaly detection.  Never a supervised class.
    NOVEL_ANOMALY_HOLDOUT = "novel_anomaly_holdout"
    #: Retained in the tables, but not usable for training or evaluation.
    EXCLUDED = "excluded"


#: The splits a supervised model may be trained or evaluated on.
SUPERVISED_SPLITS: tuple[SplitLabel, ...] = (
    SplitLabel.TRAIN,
    SplitLabel.VALIDATION,
    SplitLabel.TEST,
)


class ExclusionReason(StrEnum):
    """Why an event was excluded.  Every exclusion carries one of these."""

    AFTER_TEST_END = "after_test_end"
    BEFORE_TRAIN_START = "before_train_start"
    CAMPAIGN_CROSSES_SPLIT_BOUNDARY = "campaign_crosses_split_boundary"
    PURGED_AFTER_BOUNDARY = "purged_after_boundary"
    EMBARGOED_BEFORE_BOUNDARY = "embargoed_before_boundary"


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One event's split assignment.

    Kept in its own table, keyed by ``event_id``.  The split is never a model
    feature: a column that says which split a row is in is a perfect predictor
    of nothing useful and a perfect vector for leakage.
    """

    event_id: str
    split: SplitLabel
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class SplitManifest:
    """Aggregate description of one split.

    Reports aggregate counts only.  No event identifier ever appears here.
    """

    split_schema_version: str
    source_feature_fingerprint: str
    label_fingerprint: str
    split_config_fingerprint: str
    train_end: str | None
    validation_end: str | None
    test_end: str | None
    purge_seconds: int
    embargo_seconds: int
    total_events: int
    split_counts: dict[str, int]
    campaign_counts: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    scenario_distribution: dict[str, dict[str, int]]
    exclusion_counts: dict[str, int]
    holdout_count: int
    excluded_fraction: float
    boundary_campaign_policy: str
    normal_grouping: str
    validation_status: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-serialisable mapping."""
        return {
            "split_schema_version": self.split_schema_version,
            "source_feature_fingerprint": self.source_feature_fingerprint,
            "label_fingerprint": self.label_fingerprint,
            "split_config_fingerprint": self.split_config_fingerprint,
            "train_end": self.train_end,
            "validation_end": self.validation_end,
            "test_end": self.test_end,
            "purge_seconds": self.purge_seconds,
            "embargo_seconds": self.embargo_seconds,
            "total_events": self.total_events,
            "split_counts": self.split_counts,
            "campaign_counts": self.campaign_counts,
            "class_distribution": self.class_distribution,
            "scenario_distribution": self.scenario_distribution,
            "exclusion_counts": self.exclusion_counts,
            "holdout_count": self.holdout_count,
            "excluded_fraction": self.excluded_fraction,
            "boundary_campaign_policy": self.boundary_campaign_policy,
            "normal_grouping": self.normal_grouping,
            "validation_status": self.validation_status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SplitResult:
    """Assignments plus the manifest describing them."""

    assignments: tuple[SplitAssignment, ...]
    manifest: SplitManifest

    def by_event_id(self) -> dict[str, SplitAssignment]:
        """Return the assignments keyed by event id."""
        return {a.event_id: a for a in self.assignments}

    def event_ids_for(self, split: SplitLabel) -> frozenset[str]:
        """Return the event ids assigned to *split*."""
        return frozenset(a.event_id for a in self.assignments if a.split is split)


def label_fingerprint(labels: Sequence[GroundTruthLabel]) -> str:
    """Return an order-independent digest of a ground-truth label set."""
    rows = sorted(
        (
            {
                "event_id": str(label.event_id),
                "campaign_id": label.campaign_id,
                "scenario": str(label.scenario),
                "malicious": label.malicious,
                "supervised_training_eligible": label.supervised_training_eligible,
            }
            for label in labels
        ),
        key=lambda row: str(row["event_id"]),
    )
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ChronologicalSplitter:
    """Assigns every event to exactly one split, deterministically."""

    def __init__(self, config: SplitConfig) -> None:
        self._config = config

    # -- boundaries ---------------------------------------------------------

    def resolve_boundaries(
        self, events: Sequence[AuthEvent]
    ) -> tuple[datetime, datetime, datetime]:
        """Return ``(train_end, validation_end, test_end)`` for *events*.

        In ``fractions`` mode the boundaries are the event times at the
        cumulative fraction positions.  Because assignment compares
        ``event_time < boundary``, a boundary that lands inside a group of
        simultaneous events still keeps that whole group on one side: events
        with byte-identical histories are never split apart.

        Raises:
            SplitConfigurationError: if the events cannot produce three
                non-empty, strictly increasing intervals.
        """
        config = self._config
        if config.mode == "boundaries":
            if (
                config.train_end is None
                or config.validation_end is None
                or config.test_end is None
            ):  # pragma: no cover - guarded by SplitConfig validation
                raise SplitConfigurationError(
                    "mode='boundaries' requires all three boundaries"
                )
            return config.train_end, config.validation_end, config.test_end

        if not events:
            raise SplitConfigurationError("Cannot split an empty event set")

        ordered = sort_events_canonically(events)
        total = len(ordered)

        def boundary_at(fraction: float) -> datetime:
            index = min(total - 1, max(0, int(total * fraction)))
            return ordered[index].event_time

        train_end = boundary_at(config.train_fraction)
        validation_end = boundary_at(config.train_fraction + config.validation_fraction)
        test_end = ordered[-1].event_time + timedelta(microseconds=1)

        if not (train_end < validation_end < test_end):
            raise SplitConfigurationError(
                "The requested fractions do not yield three non-empty "
                "chronological intervals for this dataset; the events are too "
                "few or too concentrated in time"
            )
        return train_end, validation_end, test_end

    # -- assignment ---------------------------------------------------------

    def split(
        self,
        events: Sequence[AuthEvent],
        labels: Sequence[GroundTruthLabel],
        *,
        source_feature_fingerprint: str = "",
    ) -> SplitResult:
        """Assign every event to exactly one split.

        Raises:
            SplitConfigurationError: if boundaries cannot be resolved, or if
                the resulting exclusion fraction exceeds the configured limit.
        """
        config = self._config
        ordered = sort_events_canonically(events)
        by_id: dict[UUID, GroundTruthLabel] = {
            label.event_id: label for label in labels
        }

        train_end, validation_end, test_end = self.resolve_boundaries(ordered)

        assignments: dict[str, SplitLabel] = {}
        reasons: dict[str, str | None] = {}

        # Step 1 -- base temporal assignment.
        for event in ordered:
            key = str(event.event_id)
            moment = event.event_time
            if moment < train_end:
                assignments[key] = SplitLabel.TRAIN
            elif moment < validation_end:
                assignments[key] = SplitLabel.VALIDATION
            elif moment < test_end:
                assignments[key] = SplitLabel.TEST
            else:
                assignments[key] = SplitLabel.EXCLUDED
                reasons[key] = str(ExclusionReason.AFTER_TEST_END)

        # Step 2 -- holdout routing, applied before grouping so a holdout
        # campaign cannot drag co-located events through the grouping logic.
        # The holdout is an evaluation-only pool: time-slicing it would only
        # shrink it without improving isolation.
        for event in ordered:
            key = str(event.event_id)
            label = by_id.get(event.event_id)
            if label is None:
                continue
            if (
                label.scenario in config.holdout_scenarios
                or not label.supervised_training_eligible
            ):
                assignments[key] = SplitLabel.NOVEL_ANOMALY_HOLDOUT
                reasons.pop(key, None)

        # Step 3 -- group formation and boundary-crossing policy.
        self._apply_group_policy(ordered, by_id, assignments, reasons)

        # Step 4 -- purge and embargo around each boundary.
        self._apply_purge_and_embargo(
            ordered, (train_end, validation_end), assignments, reasons
        )

        result = tuple(
            SplitAssignment(
                event_id=str(event.event_id),
                split=assignments[str(event.event_id)],
                exclusion_reason=reasons.get(str(event.event_id)),
            )
            for event in ordered
        )

        # Step 5 -- totality.  Raised, not logged: a silently dropped row is
        # exactly the failure this whole module exists to prevent.
        if len(result) != len(ordered):  # pragma: no cover - defensive
            raise SplitConfigurationError(
                "Split produced a different number of assignments than events"
            )
        if {a.event_id for a in result} != {str(e.event_id) for e in ordered}:
            raise SplitConfigurationError(  # pragma: no cover - defensive
                "Split assignments do not cover exactly the input events"
            )

        manifest = self._build_manifest(
            result,
            ordered,
            by_id,
            boundaries=(train_end, validation_end, test_end),
            labels=labels,
            source_feature_fingerprint=source_feature_fingerprint,
        )

        excluded_fraction = manifest.excluded_fraction
        if excluded_fraction > config.max_excluded_fraction:
            raise SplitConfigurationError(
                f"Split excluded {excluded_fraction:.1%} of events, above the "
                f"configured limit of {config.max_excluded_fraction:.1%}. "
                f"Badly placed boundaries should be loud, not quiet: widen the "
                f"dataset, shorten the purge, or raise max_excluded_fraction "
                f"deliberately."
            )

        return SplitResult(assignments=result, manifest=manifest)

    def _group_key(
        self, event: AuthEvent, label: GroundTruthLabel | None
    ) -> str | None:
        """Return the isolation group for an event, or ``None`` for no grouping.

        Normal activity defaults to ``singleton``: each benign event is its own
        group.  The leakage that campaign-awareness addresses comes from
        *coordinated* activity whose members share structure a model could
        memorise; benign traffic has no such coordination.  Grouping benign
        events by campaign would create enormous groups straddling every
        boundary and, under the default exclude policy, discard the entire
        benign class.
        """
        if label is None:
            return None
        if label.scenario is ScenarioType.NORMAL:
            if self._config.normal_grouping == "singleton":
                return None
            return f"campaign:{label.campaign_id}"
        return f"campaign:{label.campaign_id}"

    def _apply_group_policy(
        self,
        ordered: Sequence[AuthEvent],
        by_id: Mapping[UUID, GroundTruthLabel],
        assignments: dict[str, SplitLabel],
        reasons: dict[str, str | None],
    ) -> None:
        """Force each isolation group into a single supervised split."""
        groups: dict[str, list[AuthEvent]] = defaultdict(list)
        for event in ordered:
            key = self._group_key(event, by_id.get(event.event_id))
            if key is not None:
                groups[key].append(event)

        for members in groups.values():
            supervised = {
                assignments[str(e.event_id)]
                for e in members
                if assignments[str(e.event_id)] in SUPERVISED_SPLITS
            }
            if len(supervised) <= 1:
                continue

            if self._config.boundary_campaign_policy == "exclude":
                for event in members:
                    key = str(event.event_id)
                    if assignments[key] in SUPERVISED_SPLITS:
                        assignments[key] = SplitLabel.EXCLUDED
                        reasons[key] = str(
                            ExclusionReason.CAMPAIGN_CROSSES_SPLIT_BOUNDARY
                        )
            else:
                # assign_by_first_event: the group takes the split of its
                # earliest member in canonical order.  Fully deterministic --
                # the (event_time, event_id) ordering leaves no tie to break.
                first = min(members, key=lambda e: (e.event_time, str(e.event_id)))
                target = assignments[str(first.event_id)]
                for event in members:
                    key = str(event.event_id)
                    if assignments[key] in SUPERVISED_SPLITS:
                        assignments[key] = target
                        reasons.pop(key, None)

    def _apply_purge_and_embargo(
        self,
        ordered: Sequence[AuthEvent],
        boundaries: tuple[datetime, datetime],
        assignments: dict[str, SplitLabel],
        reasons: dict[str, str | None],
    ) -> None:
        """Exclude events whose windows straddle a boundary.

        Direction is the easy thing to get backwards.  Features look
        *backwards*, so an event just **after** a boundary is the contaminated
        one -- its lookback window reaches into the previous split.  Purge
        therefore applies to the later side; embargo, which guards against
        forward-looking effects, applies to the earlier side.
        """
        purge = self._config.purge
        embargo = self._config.embargo
        if purge <= timedelta(0) and embargo <= timedelta(0):
            return

        for event in ordered:
            key = str(event.event_id)
            if assignments[key] not in SUPERVISED_SPLITS:
                continue
            moment = event.event_time

            for boundary in boundaries:
                if purge > timedelta(0) and boundary <= moment < boundary + purge:
                    assignments[key] = SplitLabel.EXCLUDED
                    reasons[key] = str(ExclusionReason.PURGED_AFTER_BOUNDARY)
                    break
                if embargo > timedelta(0) and boundary - embargo <= moment < boundary:
                    assignments[key] = SplitLabel.EXCLUDED
                    reasons[key] = str(ExclusionReason.EMBARGOED_BEFORE_BOUNDARY)
                    break

    # -- manifest -----------------------------------------------------------

    def _build_manifest(
        self,
        assignments: Sequence[SplitAssignment],
        ordered: Sequence[AuthEvent],
        by_id: Mapping[UUID, GroundTruthLabel],
        *,
        boundaries: tuple[datetime, datetime, datetime],
        labels: Sequence[GroundTruthLabel],
        source_feature_fingerprint: str,
    ) -> SplitManifest:
        """Summarise the split in aggregate terms only."""
        split_counts: dict[str, int] = {str(label): 0 for label in SplitLabel}
        class_distribution: dict[str, dict[str, int]] = {
            str(label): {"malicious": 0, "benign": 0, "unlabelled": 0}
            for label in SplitLabel
        }
        scenario_distribution: dict[str, dict[str, int]] = {
            str(label): {} for label in SplitLabel
        }
        campaigns: dict[str, set[str]] = {str(label): set() for label in SplitLabel}
        exclusion_counts: dict[str, int] = {}

        by_event = {str(e.event_id): e for e in ordered}

        for assignment in assignments:
            split = str(assignment.split)
            split_counts[split] += 1

            if assignment.exclusion_reason is not None:
                exclusion_counts[assignment.exclusion_reason] = (
                    exclusion_counts.get(assignment.exclusion_reason, 0) + 1
                )

            event = by_event[assignment.event_id]
            label = by_id.get(event.event_id)
            if label is None:
                class_distribution[split]["unlabelled"] += 1
                continue

            class_distribution[split]["malicious" if label.malicious else "benign"] += 1
            scenario = str(label.scenario)
            scenario_distribution[split][scenario] = (
                scenario_distribution[split].get(scenario, 0) + 1
            )
            if label.scenario is not ScenarioType.NORMAL:
                campaigns[split].add(label.campaign_id)

        total = len(assignments)
        excluded = split_counts[str(SplitLabel.EXCLUDED)]

        return SplitManifest(
            split_schema_version=SPLIT_SCHEMA_VERSION,
            source_feature_fingerprint=source_feature_fingerprint,
            label_fingerprint=label_fingerprint(labels),
            split_config_fingerprint=self._config.fingerprint(),
            train_end=boundaries[0].astimezone(UTC).isoformat(),
            validation_end=boundaries[1].astimezone(UTC).isoformat(),
            test_end=boundaries[2].astimezone(UTC).isoformat(),
            purge_seconds=int(self._config.purge.total_seconds()),
            embargo_seconds=int(self._config.embargo.total_seconds()),
            total_events=total,
            split_counts=split_counts,
            campaign_counts={name: len(values) for name, values in campaigns.items()},
            class_distribution=class_distribution,
            scenario_distribution=scenario_distribution,
            exclusion_counts=exclusion_counts,
            holdout_count=split_counts[str(SplitLabel.NOVEL_ANOMALY_HOLDOUT)],
            excluded_fraction=excluded / total if total else 0.0,
            boundary_campaign_policy=self._config.boundary_campaign_policy,
            normal_grouping=self._config.normal_grouping,
            validation_status="valid",
            created_at=datetime.now(UTC).isoformat(),
        )


def split_dataset(
    events: Sequence[AuthEvent],
    labels: Sequence[GroundTruthLabel],
    config: FeatureConfig,
    *,
    source_feature_fingerprint: str = "",
) -> SplitResult:
    """Convenience wrapper: split using a full feature configuration."""
    return ChronologicalSplitter(config.split).split(
        events, labels, source_feature_fingerprint=source_feature_fingerprint
    )
