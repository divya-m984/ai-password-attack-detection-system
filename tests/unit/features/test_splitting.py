"""Tests for chronological, campaign-aware dataset splitting."""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.exceptions import SplitConfigurationError
from password_attack_detector.features.config import FeatureConfig, SplitConfig
from password_attack_detector.features.splitting import (
    SPLIT_SCHEMA_VERSION,
    SUPERVISED_SPLITS,
    ChronologicalSplitter,
    ExclusionReason,
    SplitLabel,
    SplitResult,
    label_fingerprint,
    split_dataset,
)
from tests.features.factories import BASE_TIME, make_event, make_labels

_NO_PURGE = SplitConfig(purge=timedelta(0), strict_isolation=False)


def _stream(count: int = 100, *, step_seconds: float = 600.0) -> list[AuthEvent]:
    """Evenly spaced benign events over a long enough span to split."""
    return [
        make_event(
            t=float(index) * step_seconds,
            user=f"u{index % 5}",
            source=f"s{index % 3}",
            key=str(index),
        )
        for index in range(count)
    ]


def _split(
    events: list[AuthEvent],
    labels: list[GroundTruthLabel],
    config: SplitConfig | None = None,
) -> SplitResult:
    return ChronologicalSplitter(config if config is not None else _NO_PURGE).split(
        events, labels
    )


# --- totality --------------------------------------------------------------


class TestTotality:
    def test_every_event_receives_exactly_one_assignment(self) -> None:
        events = _stream(120)
        result = _split(events, make_labels(events))
        assert len(result.assignments) == len(events)
        assert len({a.event_id for a in result.assignments}) == len(events)

    def test_assignments_cover_exactly_the_input_events(self) -> None:
        events = _stream(80)
        result = _split(events, make_labels(events))
        assert {a.event_id for a in result.assignments} == {
            str(e.event_id) for e in events
        }

    def test_no_row_is_silently_dropped(self) -> None:
        events = _stream(150)
        result = _split(events, make_labels(events))
        assert result.manifest.total_events == len(events)
        assert sum(result.manifest.split_counts.values()) == len(events)

    def test_every_exclusion_carries_a_reason(self) -> None:
        events = _stream(120)
        config = SplitConfig(
            purge=timedelta(hours=2), strict_isolation=False, max_excluded_fraction=0.5
        )
        result = _split(events, make_labels(events), config)
        for assignment in result.assignments:
            if assignment.split is SplitLabel.EXCLUDED:
                assert assignment.exclusion_reason is not None

    def test_no_included_event_carries_a_reason(self) -> None:
        events = _stream(100)
        result = _split(events, make_labels(events))
        for assignment in result.assignments:
            if assignment.split is not SplitLabel.EXCLUDED:
                assert assignment.exclusion_reason is None


# --- chronology ------------------------------------------------------------


class TestChronology:
    def test_splits_are_ordered_in_time(self) -> None:
        events = _stream(120)
        result = _split(events, make_labels(events))
        by_id = {str(e.event_id): e for e in events}

        latest_train = max(
            by_id[a.event_id].event_time
            for a in result.assignments
            if a.split is SplitLabel.TRAIN
        )
        earliest_test = min(
            by_id[a.event_id].event_time
            for a in result.assignments
            if a.split is SplitLabel.TEST
        )
        assert latest_train < earliest_test

    def test_all_three_supervised_splits_are_populated(self) -> None:
        events = _stream(120)
        result = _split(events, make_labels(events))
        for split in SUPERVISED_SPLITS:
            assert result.manifest.split_counts[str(split)] > 0

    def test_fractions_are_approximately_honoured(self) -> None:
        events = _stream(200)
        result = _split(events, make_labels(events))
        train = result.manifest.split_counts[str(SplitLabel.TRAIN)]
        assert train == pytest.approx(140, abs=5)

    def test_assignment_is_deterministic(self) -> None:
        events = _stream(120)
        labels = make_labels(events)
        first = _split(events, labels).by_event_id()
        second = _split(events, labels).by_event_id()
        assert first == second

    def test_input_order_does_not_change_assignments(self) -> None:
        events = _stream(120)
        labels = make_labels(events)
        shuffled = list(events)
        random.Random(4242).shuffle(shuffled)
        assert (
            _split(events, labels).by_event_id()
            == _split(shuffled, labels).by_event_id()
        )

    def test_simultaneous_events_are_never_split_apart(self) -> None:
        # Events with byte-identical histories must share a split, otherwise
        # the two sides of a boundary are coupled.
        events = [
            make_event(t=float(index // 4) * 600.0, user="u1", key=str(index))
            for index in range(120)
        ]
        result = _split(events, make_labels(events))
        by_id = {str(e.event_id): e for e in events}
        per_time: dict[object, set[SplitLabel]] = {}
        for assignment in result.assignments:
            moment = by_id[assignment.event_id].event_time
            per_time.setdefault(moment, set()).add(assignment.split)
        for splits in per_time.values():
            assert len(splits) == 1


class TestExplicitBoundaries:
    def test_explicit_boundaries_are_used_verbatim(self) -> None:
        events = _stream(100)
        config = SplitConfig(
            mode="boundaries",
            train_end=BASE_TIME + timedelta(hours=5),
            validation_end=BASE_TIME + timedelta(hours=10),
            test_end=BASE_TIME + timedelta(hours=20),
            purge=timedelta(0),
            strict_isolation=False,
        )
        result = _split(events, make_labels(events), config)
        assert result.manifest.train_end is not None
        assert result.manifest.train_end.startswith("2024-03-04T17:00")

    def test_events_after_test_end_are_excluded_with_a_reason(self) -> None:
        events = _stream(100)
        config = SplitConfig(
            mode="boundaries",
            train_end=BASE_TIME + timedelta(hours=2),
            validation_end=BASE_TIME + timedelta(hours=4),
            test_end=BASE_TIME + timedelta(hours=6),
            purge=timedelta(0),
            strict_isolation=False,
            max_excluded_fraction=1.0,
        )
        result = _split(events, make_labels(events), config)
        reasons = result.manifest.exclusion_counts
        assert reasons[str(ExclusionReason.AFTER_TEST_END)] > 0

    def test_empty_event_set_is_rejected(self) -> None:
        with pytest.raises(SplitConfigurationError, match="empty event set"):
            _split([], [])

    def test_too_concentrated_a_dataset_is_rejected(self) -> None:
        # Every event at the same instant cannot yield three intervals.
        events = [make_event(t=0.0, key=str(i)) for i in range(50)]
        with pytest.raises(SplitConfigurationError, match="non-empty"):
            _split(events, make_labels(events))


# --- holdout routing -------------------------------------------------------


class TestHoldoutRouting:
    def _mixed(self) -> tuple[list[AuthEvent], list[GroundTruthLabel]]:
        benign = _stream(100)
        novel = [
            make_event(t=float(i) * 600.0 + 30.0, user="u9", key=f"n{i}")
            for i in range(10)
        ]
        labels = make_labels(benign) + make_labels(
            novel,
            scenario=ScenarioType.NOVEL_ANOMALY_HOLDOUT,
            campaign_id="c-novel",
        )
        return benign + novel, labels

    def test_novel_anomaly_events_go_to_the_holdout(self) -> None:
        events, labels = self._mixed()
        result = _split(events, labels)
        assert result.manifest.holdout_count == 10

    def test_holdout_events_never_enter_a_supervised_split(self) -> None:
        events, labels = self._mixed()
        result = _split(events, labels)
        holdout_ids = {
            str(label.event_id)
            for label in labels
            if not label.supervised_training_eligible
        }
        for assignment in result.assignments:
            if assignment.event_id in holdout_ids:
                assert assignment.split is SplitLabel.NOVEL_ANOMALY_HOLDOUT

    def test_ineligible_events_are_routed_regardless_of_scenario(self) -> None:
        events = _stream(100)
        labels = make_labels(events[:5], supervised_training_eligible=False)
        labels += make_labels(events[5:])
        result = _split(events, labels)
        assert result.manifest.holdout_count == 5

    def test_holdout_is_not_converted_into_a_supervised_class(self) -> None:
        events, labels = self._mixed()
        result = _split(events, labels)
        counts = result.manifest.split_counts
        assert counts[str(SplitLabel.NOVEL_ANOMALY_HOLDOUT)] == 10
        for split in SUPERVISED_SPLITS:
            scenarios = result.manifest.scenario_distribution[str(split)]
            assert str(ScenarioType.NOVEL_ANOMALY_HOLDOUT) not in scenarios

    def test_holdout_remains_available_for_later_evaluation(self) -> None:
        events, labels = self._mixed()
        result = _split(events, labels)
        assert len(result.event_ids_for(SplitLabel.NOVEL_ANOMALY_HOLDOUT)) == 10


# --- campaign isolation ----------------------------------------------------


class TestCampaignIsolation:
    def _straddling(self) -> tuple[list[AuthEvent], list[GroundTruthLabel]]:
        """A campaign whose events span the whole timeline, crossing every cut."""
        benign = _stream(100)
        campaign = [
            make_event(t=float(i) * 6000.0 + 5.0, user="u_target", key=f"c{i}")
            for i in range(10)
        ]
        labels = make_labels(benign) + make_labels(
            campaign, scenario=ScenarioType.BRUTE_FORCE, campaign_id="c-bf-1"
        )
        return benign + campaign, labels

    def test_a_straddling_campaign_is_excluded_by_default(self) -> None:
        events, labels = self._straddling()
        result = _split(
            events,
            labels,
            SplitConfig(
                purge=timedelta(0), strict_isolation=False, max_excluded_fraction=0.5
            ),
        )
        campaign_ids = {
            str(label.event_id) for label in labels if label.campaign_id == "c-bf-1"
        }
        for assignment in result.assignments:
            if assignment.event_id in campaign_ids:
                assert assignment.split is SplitLabel.EXCLUDED
                assert assignment.exclusion_reason == str(
                    ExclusionReason.CAMPAIGN_CROSSES_SPLIT_BOUNDARY
                )

    def test_assign_by_first_event_keeps_the_campaign_together(self) -> None:
        events, labels = self._straddling()
        config = SplitConfig(
            purge=timedelta(0),
            strict_isolation=False,
            boundary_campaign_policy="assign_by_first_event",
            max_excluded_fraction=0.5,
        )
        result = _split(events, labels, config)
        campaign_splits = {
            assignment.split
            for assignment in result.assignments
            if assignment.event_id
            in {str(x.event_id) for x in labels if x.campaign_id == "c-bf-1"}
        }
        assert campaign_splits == {SplitLabel.TRAIN}

    def test_assign_by_first_event_is_deterministic(self) -> None:
        events, labels = self._straddling()
        config = SplitConfig(
            purge=timedelta(0),
            strict_isolation=False,
            boundary_campaign_policy="assign_by_first_event",
            max_excluded_fraction=0.5,
        )
        assert (
            _split(events, labels, config).by_event_id()
            == _split(events, labels, config).by_event_id()
        )

    def test_no_campaign_spans_two_supervised_splits(self) -> None:
        benign = _stream(120)
        labels = make_labels(benign)
        events = list(benign)
        for campaign_index in range(4):
            members = [
                make_event(
                    t=float(campaign_index) * 15000.0 + i * 10.0,
                    user=f"u_t{campaign_index}",
                    key=f"c{campaign_index}_{i}",
                )
                for i in range(6)
            ]
            events += members
            labels += make_labels(
                members,
                scenario=ScenarioType.BRUTE_FORCE,
                campaign_id=f"c-{campaign_index}",
            )

        result = _split(
            events,
            labels,
            SplitConfig(
                purge=timedelta(0), strict_isolation=False, max_excluded_fraction=0.5
            ),
        )
        by_id = {str(x.event_id): x for x in labels}
        per_campaign: dict[str, set[SplitLabel]] = {}
        for assignment in result.assignments:
            label = by_id[assignment.event_id]
            if label.scenario is ScenarioType.NORMAL:
                continue
            if assignment.split not in SUPERVISED_SPLITS:
                continue
            per_campaign.setdefault(label.campaign_id, set()).add(assignment.split)

        for splits in per_campaign.values():
            assert len(splits) == 1

    def test_a_contained_campaign_is_not_disturbed(self) -> None:
        benign = _stream(120)
        contained = [
            make_event(t=100.0 + i * 5.0, user="u_t", key=f"tight{i}") for i in range(5)
        ]
        labels = make_labels(benign) + make_labels(
            contained, scenario=ScenarioType.BRUTE_FORCE, campaign_id="c-tight"
        )
        result = _split(benign + contained, labels)
        contained_ids = {str(e.event_id) for e in contained}
        splits = {a.split for a in result.assignments if a.event_id in contained_ids}
        assert splits == {SplitLabel.TRAIN}

    def test_normal_events_are_grouped_as_singletons_by_default(self) -> None:
        # Grouping benign traffic by campaign would make every group straddle
        # every boundary and, under exclude, discard the whole benign class.
        events = _stream(120)
        result = _split(events, make_labels(events))
        assert result.manifest.split_counts[str(SplitLabel.EXCLUDED)] == 0

    def test_normal_grouping_by_campaign_is_available(self) -> None:
        events = _stream(120)
        config = SplitConfig(
            purge=timedelta(0),
            strict_isolation=False,
            normal_grouping="campaign_id",
            max_excluded_fraction=1.0,
        )
        result = _split(events, make_labels(events), config)
        assert result.manifest.normal_grouping == "campaign_id"
        assert result.manifest.split_counts[str(SplitLabel.EXCLUDED)] > 0


# --- purge and embargo -----------------------------------------------------


class TestPurgeAndEmbargo:
    def test_purge_excludes_events_after_a_boundary(self) -> None:
        events = _stream(200, step_seconds=300.0)
        config = SplitConfig(
            purge=timedelta(hours=1), strict_isolation=False, max_excluded_fraction=0.5
        )
        result = _split(events, make_labels(events), config)
        assert (
            result.manifest.exclusion_counts[str(ExclusionReason.PURGED_AFTER_BOUNDARY)]
            > 0
        )

    def test_purge_applies_to_the_later_side_of_the_boundary(self) -> None:
        # Features look backwards, so the contaminated events are the ones
        # whose lookback windows reach back across the cut -- the events just
        # *after* a boundary, not before it.
        from datetime import datetime

        purge = timedelta(hours=1)
        events = _stream(200, step_seconds=300.0)
        config = SplitConfig(
            purge=purge, strict_isolation=False, max_excluded_fraction=0.5
        )
        result = _split(events, make_labels(events), config)
        by_id = {str(e.event_id): e for e in events}

        assert result.manifest.train_end is not None
        assert result.manifest.validation_end is not None
        boundaries = [
            datetime.fromisoformat(result.manifest.train_end),
            datetime.fromisoformat(result.manifest.validation_end),
        ]

        purged = [
            by_id[a.event_id].event_time
            for a in result.assignments
            if a.exclusion_reason == str(ExclusionReason.PURGED_AFTER_BOUNDARY)
        ]
        assert purged
        for moment in purged:
            assert any(
                boundary <= moment < boundary + purge for boundary in boundaries
            ), "a purged event must sit in [boundary, boundary + purge)"

    def test_embargo_excludes_events_before_a_boundary(self) -> None:
        events = _stream(200, step_seconds=300.0)
        config = SplitConfig(
            purge=timedelta(0),
            embargo=timedelta(hours=1),
            strict_isolation=False,
            max_excluded_fraction=0.5,
        )
        result = _split(events, make_labels(events), config)
        assert (
            result.manifest.exclusion_counts[
                str(ExclusionReason.EMBARGOED_BEFORE_BOUNDARY)
            ]
            > 0
        )

    def test_purge_and_embargo_counts_are_reported(self) -> None:
        events = _stream(300, step_seconds=300.0)
        config = SplitConfig(
            purge=timedelta(hours=1),
            embargo=timedelta(minutes=30),
            strict_isolation=False,
            max_excluded_fraction=0.5,
        )
        result = _split(events, make_labels(events), config)
        counts = result.manifest.exclusion_counts
        assert counts[str(ExclusionReason.PURGED_AFTER_BOUNDARY)] > 0
        assert counts[str(ExclusionReason.EMBARGOED_BEFORE_BOUNDARY)] > 0

    def test_zero_purge_and_embargo_exclude_nothing(self) -> None:
        events = _stream(120)
        result = _split(events, make_labels(events))
        assert result.manifest.split_counts[str(SplitLabel.EXCLUDED)] == 0

    def test_strict_isolation_requires_purge_to_cover_the_max_window(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="shorter than the maximum"):
            FeatureConfig(split=SplitConfig(purge=timedelta(minutes=5)))

    def test_over_exclusion_is_loud(self) -> None:
        events = _stream(120, step_seconds=60.0)
        config = SplitConfig(
            purge=timedelta(hours=1),
            strict_isolation=False,
            max_excluded_fraction=0.01,
        )
        with pytest.raises(SplitConfigurationError, match="above the configured limit"):
            _split(events, make_labels(events), config)

    def test_holdout_is_not_purged(self) -> None:
        benign = _stream(150, step_seconds=300.0)
        novel = [
            make_event(t=float(i) * 3000.0, user="u9", key=f"n{i}") for i in range(8)
        ]
        labels = make_labels(benign) + make_labels(
            novel, scenario=ScenarioType.NOVEL_ANOMALY_HOLDOUT, campaign_id="c-novel"
        )
        config = SplitConfig(
            purge=timedelta(hours=1), strict_isolation=False, max_excluded_fraction=0.5
        )
        result = _split(benign + novel, labels, config)
        assert result.manifest.holdout_count == 8


# --- manifest --------------------------------------------------------------


class TestSplitManifest:
    @pytest.fixture()
    def result(self) -> SplitResult:
        benign = _stream(120)
        attack = [
            make_event(t=200.0 + i * 5.0, user="u_t", key=f"a{i}") for i in range(8)
        ]
        labels = make_labels(benign) + make_labels(
            attack, scenario=ScenarioType.BRUTE_FORCE, campaign_id="c-a"
        )
        return _split(benign + attack, labels)

    def test_records_the_schema_version(self, result: SplitResult) -> None:
        assert result.manifest.split_schema_version == SPLIT_SCHEMA_VERSION

    def test_records_the_boundaries(self, result: SplitResult) -> None:
        manifest = result.manifest
        assert manifest.train_end is not None
        assert manifest.validation_end is not None
        assert manifest.train_end < manifest.validation_end

    def test_records_the_config_fingerprint(self, result: SplitResult) -> None:
        assert len(result.manifest.split_config_fingerprint) == 64

    def test_records_the_label_fingerprint(self, result: SplitResult) -> None:
        assert len(result.manifest.label_fingerprint) == 64

    def test_reports_class_distribution_per_split(self, result: SplitResult) -> None:
        train = result.manifest.class_distribution[str(SplitLabel.TRAIN)]
        assert train["benign"] > 0
        assert train["malicious"] > 0

    def test_reports_campaign_counts_per_split(self, result: SplitResult) -> None:
        assert result.manifest.campaign_counts[str(SplitLabel.TRAIN)] == 1

    def test_reports_scenario_distribution(self, result: SplitResult) -> None:
        scenarios = result.manifest.scenario_distribution[str(SplitLabel.TRAIN)]
        assert str(ScenarioType.BRUTE_FORCE) in scenarios

    def test_serialises_to_a_plain_mapping(self, result: SplitResult) -> None:
        import json

        payload = result.manifest.to_dict()
        assert json.loads(json.dumps(payload))["split_schema_version"] == (
            SPLIT_SCHEMA_VERSION
        )

    def test_contains_no_event_identifiers(self, result: SplitResult) -> None:
        import json

        text = json.dumps(result.manifest.to_dict())
        for assignment in result.assignments:
            assert assignment.event_id not in text

    def test_contains_no_pseudonyms(self, result: SplitResult) -> None:
        import json

        text = json.dumps(result.manifest.to_dict())
        assert "u:" not in text
        assert "s:" not in text

    def test_excluded_fraction_is_reported(self, result: SplitResult) -> None:
        assert 0.0 <= result.manifest.excluded_fraction <= 1.0


class TestLabelFingerprint:
    def test_is_order_independent(self) -> None:
        events = _stream(40)
        labels = make_labels(events)
        shuffled = list(labels)
        random.Random(7).shuffle(shuffled)
        assert label_fingerprint(labels) == label_fingerprint(shuffled)

    def test_reacts_to_a_label_change(self) -> None:
        events = _stream(40)
        original = make_labels(events)
        changed = make_labels(events, scenario=ScenarioType.BRUTE_FORCE)
        assert label_fingerprint(original) != label_fingerprint(changed)

    def test_is_hex_sha256(self) -> None:
        fingerprint = label_fingerprint(make_labels(_stream(10)))
        assert len(fingerprint) == 64


# --- convenience wrapper ---------------------------------------------------


class TestSplitDataset:
    def test_uses_the_nested_split_configuration(self) -> None:
        # The shipped default pairs a 24h purge with strict isolation, so the
        # stream has to be long enough for that purge to be a small fraction.
        events = _stream(600, step_seconds=3600.0)
        config = FeatureConfig()
        result = split_dataset(events, make_labels(events), config)
        assert result.manifest.purge_seconds == 86400
        assert result.manifest.excluded_fraction < 0.10

    def test_result_helpers_expose_the_assignments(self) -> None:
        events = _stream(120)
        result = _split(events, make_labels(events))
        train_ids = result.event_ids_for(SplitLabel.TRAIN)
        assert train_ids
        assert all(isinstance(value, str) for value in train_ids)
