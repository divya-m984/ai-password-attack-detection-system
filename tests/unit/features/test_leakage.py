"""Tests for the leakage auditor.

Several tests here deliberately construct a *broken* dataset and assert the
auditor catches it.  An auditor that only ever sees correct input proves
nothing.
"""

from __future__ import annotations

import json
import random
import re
from datetime import timedelta

import pytest

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.features.baselines import BehavioralBaselineModel
from password_attack_detector.features.catalog import FeatureCatalog, build_catalog
from password_attack_detector.features.config import (
    BaselineConfig,
    FeatureConfig,
    SplitConfig,
)
from password_attack_detector.features.engine import FeatureEngine, FeatureFrame
from password_attack_detector.features.leakage import (
    CHECK_NAMES,
    LeakageAuditor,
    LeakageAuditResult,
    audit_result_to_markdown,
)
from password_attack_detector.features.splitting import (
    ChronologicalSplitter,
    SplitAssignment,
    SplitLabel,
)
from tests.features.factories import BASE_TIME, make_event, make_labels

#: Short windows plus a matching purge, so a small fixture can still satisfy
#: strict isolation.  A config with purge=0 would (correctly) fail the purge
#: check, which is exercised separately below.
_CONFIG = FeatureConfig(
    windows=("1m", "5m"),
    cardinality_windows=("5m",),
    dispersion_windows=("5m",),
    device_session_windows=("5m",),
    pair_windows=("5m",),
    baseline=BaselineConfig(rate_reference_window="5m"),
    split=SplitConfig(purge=timedelta(minutes=5), max_excluded_fraction=0.5),
)

#: A deliberately unpurged configuration, used to prove the purge check bites.
_UNPURGED = FeatureConfig(
    windows=("1m", "5m"),
    cardinality_windows=("5m",),
    dispersion_windows=("5m",),
    device_session_windows=("5m",),
    pair_windows=("5m",),
    baseline=BaselineConfig(rate_reference_window="5m"),
    split=SplitConfig(purge=timedelta(0), strict_isolation=False),
)

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")


def _events(count: int = 60) -> list[AuthEvent]:
    rng = random.Random(31337)
    return [
        make_event(
            t=float(index) * 600.0,
            user=f"u{rng.randint(1, 4)}",
            source=f"s{rng.randint(1, 3)}",
            outcome=rng.choice(["success", "failure"]),
            response_time_ms=rng.randint(20, 400),
            key=str(index),
        )
        for index in range(count)
    ]


def _frame(events: list[AuthEvent], config: FeatureConfig = _CONFIG) -> FeatureFrame:
    return FeatureEngine(config, build_catalog(config)).run(events)


def _auditor(config: FeatureConfig = _CONFIG) -> LeakageAuditor:
    return LeakageAuditor(config, build_catalog(config))


def _split(
    events: list[AuthEvent], labels: list[GroundTruthLabel]
) -> list[SplitAssignment]:
    return list(ChronologicalSplitter(_CONFIG.split).split(events, labels).assignments)


# --- overall shape ---------------------------------------------------------


def _fitted_baseline(
    events: list[AuthEvent], assignments: list[SplitAssignment]
) -> BehavioralBaselineModel:
    """Fit a baseline on exactly the training split, as the pipeline does."""
    train_ids = {a.event_id for a in assignments if a.split is SplitLabel.TRAIN}
    train = [e for e in events if str(e.event_id) in train_ids]
    return BehavioralBaselineModel(_CONFIG.baseline).fit(
        train,
        permitted_event_ids=frozenset(e.event_id for e in train),
        interval=(BASE_TIME - timedelta(hours=1), BASE_TIME + timedelta(days=30)),
    )


class TestAuditShape:
    def test_a_fully_specified_clean_dataset_passes(self) -> None:
        events = _events()
        labels = make_labels(events)
        assignments = _split(events, labels)
        model = _fitted_baseline(events, assignments)
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=assignments,
            baseline_artifact=model.artifact,
        )
        assert result.status == "pass"
        assert result.passed
        assert not result.errors
        assert not any(c.skipped for c in result.checks)

    def test_omitting_the_baseline_downgrades_to_a_warning(self) -> None:
        # Not a failure -- the dataset is fine -- but the audit must say which
        # checks it could not run rather than reporting a clean pass.
        events = _events()
        labels = make_labels(events)
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=_split(events, labels),
        )
        assert result.status == "warning"
        assert not result.errors
        assert result.passed

    def test_every_named_check_is_reported(self) -> None:
        events = _events(20)
        result = _auditor().audit(_frame(events), events=events)
        assert tuple(c.name for c in result.checks) == CHECK_NAMES

    def test_missing_inputs_are_skipped_not_silently_passed(self) -> None:
        # An audit that quietly degrades to a subset of its checks is worse
        # than one that says what it could not verify.
        result = _auditor().audit(_frame(_events(10)))
        skipped = [c.name for c in result.checks if c.skipped]
        assert skipped
        assert result.status == "warning"
        assert any("Skipped" in w for w in result.warnings)

    def test_reports_row_and_feature_counts(self) -> None:
        events = _events(30)
        result = _auditor().audit(_frame(events), events=events)
        assert result.checked_row_count == 30
        assert result.checked_feature_count > 100

    def test_reports_the_split_summary(self) -> None:
        events = _events(80)
        labels = make_labels(events)
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=_split(events, labels),
        )
        assert result.split_summary[str(SplitLabel.TRAIN)] > 0

    def test_serialises_to_a_plain_mapping(self) -> None:
        events = _events(20)
        result = _auditor().audit(_frame(events), events=events)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["status"] in {"pass", "warning", "fail"}
        assert len(payload["checks"]) == len(CHECK_NAMES)


# --- column checks ---------------------------------------------------------


class TestColumnChecks:
    def _named(self, result: LeakageAuditResult, name: str) -> bool:
        return next(c.passed for c in result.checks if c.name == name)

    def test_no_ground_truth_columns(self) -> None:
        result = _auditor().audit(_frame(_events(20)))
        assert self._named(result, "NO_GROUND_TRUTH_COLUMNS")

    def test_no_campaign_columns(self) -> None:
        result = _auditor().audit(_frame(_events(20)))
        assert self._named(result, "NO_CAMPAIGN_COLUMNS")

    def test_no_split_columns(self) -> None:
        result = _auditor().audit(_frame(_events(20)))
        assert self._named(result, "NO_SPLIT_COLUMNS")

    def test_prohibited_findings_are_empty_for_a_clean_dataset(self) -> None:
        result = _auditor().audit(_frame(_events(20)))
        assert result.prohibited_column_findings == ()

    def test_an_injected_label_column_is_detected(self) -> None:
        # Construct a catalog that smuggles a ground-truth column in, and
        # verify the auditor refuses it.
        config = _CONFIG
        catalog = build_catalog(config)
        smuggled = catalog.get("user_attempt_count__5m").model_copy(
            update={"name": "malicious"}
        )
        broken = FeatureCatalog(
            tuple(
                smuggled if s.name == "user_attempt_count__5m" else s
                for s in catalog.specs
            ),
            config_fingerprint=config.fingerprint(),
        )
        frame = FeatureFrame(
            rows=({"malicious": 1},),
            catalog=broken,
            stats=_frame(_events(2)).stats,
        )
        result = LeakageAuditor(config, broken).audit(frame)
        assert not next(
            c.passed for c in result.checks if c.name == "NO_GROUND_TRUTH_COLUMNS"
        )
        assert result.status == "fail"
        assert "malicious" in result.prohibited_column_findings


# --- behavioural checks ----------------------------------------------------


class TestBehaviouralChecks:
    def test_future_contribution_check_passes_on_the_real_engine(self) -> None:
        events = _events(40)
        result = _auditor().audit(_frame(events), events=events)
        check = next(c for c in result.checks if c.name == "NO_FUTURE_CONTRIBUTION")
        assert check.passed
        assert not check.skipped

    def test_same_timestamp_check_passes_on_the_real_engine(self) -> None:
        events = _events(40)
        result = _auditor().audit(_frame(events), events=events)
        check = next(
            c for c in result.checks if c.name == "NO_SAME_TIMESTAMP_CONTRIBUTION"
        )
        assert check.passed
        assert not check.skipped

    def test_behavioural_checks_are_skipped_without_events(self) -> None:
        result = _auditor().audit(_frame(_events(10)))
        for name in ("NO_FUTURE_CONTRIBUTION", "NO_SAME_TIMESTAMP_CONTRIBUTION"):
            assert next(c.skipped for c in result.checks if c.name == name)

    def test_behavioural_checks_survive_a_dense_simultaneous_stream(self) -> None:
        events = [
            make_event(t=float(i // 5) * 600.0, user=f"u{i % 3}", key=str(i))
            for i in range(50)
        ]
        result = _auditor().audit(_frame(events), events=events)
        for name in ("NO_FUTURE_CONTRIBUTION", "NO_SAME_TIMESTAMP_CONTRIBUTION"):
            assert next(c.passed for c in result.checks if c.name == name), name


# --- baseline provenance ---------------------------------------------------


class TestBaselineProvenance:
    def _fitted(
        self, events: list[AuthEvent], assignments: list[SplitAssignment]
    ) -> BehavioralBaselineModel:
        train_ids = {a.event_id for a in assignments if a.split is SplitLabel.TRAIN}
        train = [e for e in events if str(e.event_id) in train_ids]
        return BehavioralBaselineModel(BaselineConfig()).fit(
            train,
            permitted_event_ids=frozenset(e.event_id for e in train),
            interval=(
                BASE_TIME - timedelta(hours=1),
                BASE_TIME + timedelta(days=30),
            ),
        )

    def test_a_correctly_fitted_baseline_passes(self) -> None:
        events = _events(80)
        labels = make_labels(events)
        assignments = _split(events, labels)
        model = self._fitted(events, assignments)

        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=assignments,
            baseline_artifact=model.artifact,
        )
        assert next(
            c.passed
            for c in result.checks
            if c.name == "BASELINE_SOURCE_FINGERPRINT_MATCHES_TRAIN"
        )

    def test_a_baseline_fitted_on_everything_is_caught(self) -> None:
        # The auditor recomputes the expected fingerprint from the split table
        # independently, so an over-broad fit cannot hide.
        events = _events(80)
        labels = make_labels(events)
        assignments = _split(events, labels)

        over_broad = BehavioralBaselineModel(BaselineConfig()).fit(
            events,
            permitted_event_ids=frozenset(e.event_id for e in events),
            interval=(BASE_TIME - timedelta(hours=1), BASE_TIME + timedelta(days=30)),
        )

        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=assignments,
            baseline_artifact=over_broad.artifact,
        )
        assert not next(
            c.passed
            for c in result.checks
            if c.name == "BASELINE_SOURCE_FINGERPRINT_MATCHES_TRAIN"
        )
        assert not next(
            c.passed
            for c in result.checks
            if c.name == "BASELINE_NOT_FIT_ON_EVALUATION_DATA"
        )
        assert result.status == "fail"

    def test_a_tampered_fingerprint_is_caught(self) -> None:
        events = _events(80)
        labels = make_labels(events)
        assignments = _split(events, labels)
        model = self._fitted(events, assignments)
        tampered = model.artifact.__class__(
            **{
                **model.artifact.to_dict(),
                "fitted_source_fingerprint": "0" * 64,
            }
        )
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=assignments,
            baseline_artifact=tampered,
        )
        assert result.status == "fail"

    def test_baseline_interval_is_reported(self) -> None:
        events = _events(60)
        labels = make_labels(events)
        assignments = _split(events, labels)
        model = self._fitted(events, assignments)
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=assignments,
            baseline_artifact=model.artifact,
        )
        assert result.baseline_interval is not None
        assert len(result.baseline_interval) == 2


# --- split integrity -------------------------------------------------------


class TestSplitIntegrity:
    def test_purge_check_passes_with_an_adequate_purge(self) -> None:
        events = [
            make_event(t=float(i) * 60.0, user=f"u{i % 3}", key=str(i))
            for i in range(200)
        ]
        labels = make_labels(events)
        assignments = list(
            ChronologicalSplitter(_CONFIG.split).split(events, labels).assignments
        )
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=assignments,
        )
        assert next(
            c.passed for c in result.checks if c.name == "PURGE_INTERVAL_RESPECTED"
        )

    def test_purge_check_catches_an_absent_purge(self) -> None:
        # With no purge, events just after a boundary have lookback windows
        # reaching back into the previous split.  The check must notice.
        events = [
            make_event(t=float(i) * 60.0, user=f"u{i % 3}", key=str(i))
            for i in range(200)
        ]
        labels = make_labels(events)
        assignments = list(
            ChronologicalSplitter(_UNPURGED.split).split(events, labels).assignments
        )
        result = LeakageAuditor(_UNPURGED, build_catalog(_UNPURGED)).audit(
            _frame(events, _UNPURGED),
            events=events,
            labels=labels,
            assignments=assignments,
        )
        assert not next(
            c.passed for c in result.checks if c.name == "PURGE_INTERVAL_RESPECTED"
        )

    def test_campaign_isolation_passes_for_a_contained_campaign(self) -> None:
        benign = _events(100)
        campaign = [
            make_event(t=200.0 + i * 5.0, user="u_t", key=f"c{i}") for i in range(6)
        ]
        labels = make_labels(benign) + make_labels(
            campaign, scenario=ScenarioType.BRUTE_FORCE, campaign_id="c-1"
        )
        events = benign + campaign
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=_split(events, labels),
        )
        assert next(
            c.passed for c in result.checks if c.name == "CAMPAIGN_GROUPS_ISOLATED"
        )

    def test_campaign_isolation_catches_a_straddling_campaign(self) -> None:
        benign = _events(100)
        campaign = [
            make_event(t=float(i) * 6000.0, user="u_t", key=f"c{i}") for i in range(8)
        ]
        labels = make_labels(benign) + make_labels(
            campaign, scenario=ScenarioType.BRUTE_FORCE, campaign_id="c-1"
        )
        events = benign + campaign

        # Hand-built assignments that deliberately break the isolation rule.
        broken = [
            SplitAssignment(
                event_id=str(event.event_id),
                split=SplitLabel.TRAIN if index % 2 else SplitLabel.TEST,
            )
            for index, event in enumerate(events)
        ]
        result = _auditor().audit(
            _frame(events), events=events, labels=labels, assignments=broken
        )
        assert not next(
            c.passed for c in result.checks if c.name == "CAMPAIGN_GROUPS_ISOLATED"
        )

    def test_holdout_leak_is_caught(self) -> None:
        events = _events(40)
        labels = make_labels(events, supervised_training_eligible=False)
        leaked = [
            SplitAssignment(event_id=str(e.event_id), split=SplitLabel.TRAIN)
            for e in events
        ]
        result = _auditor().audit(
            _frame(events), events=events, labels=labels, assignments=leaked
        )
        assert not next(
            c.passed
            for c in result.checks
            if c.name == "HOLDOUT_EXCLUDED_FROM_SUPERVISED"
        )

    def test_holdout_routed_correctly_passes(self) -> None:
        benign = _events(80)
        novel = [
            make_event(t=float(i) * 700.0, user="u9", key=f"n{i}") for i in range(6)
        ]
        labels = make_labels(benign) + make_labels(
            novel, scenario=ScenarioType.NOVEL_ANOMALY_HOLDOUT, campaign_id="c-novel"
        )
        events = benign + novel
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=_split(events, labels),
        )
        assert next(
            c.passed
            for c in result.checks
            if c.name == "HOLDOUT_EXCLUDED_FROM_SUPERVISED"
        )


# --- schema checks ---------------------------------------------------------


class TestSchemaChecks:
    def test_current_versus_prior_passes_for_the_shipped_catalog(self) -> None:
        result = _auditor().audit(_frame(_events(10)))
        assert next(
            c.passed
            for c in result.checks
            if c.name == "CURRENT_VS_PRIOR_FIELDS_DISTINGUISHED"
        )

    def test_a_mislabelled_current_column_is_caught(self) -> None:
        from password_attack_detector.features.catalog import LeakageClass

        config = _CONFIG
        catalog = build_catalog(config)
        broken_spec = catalog.get("current_country_code").model_copy(
            update={"leakage_class": LeakageClass.PRIOR_ONLY}
        )
        broken = FeatureCatalog(
            tuple(
                broken_spec if s.name == "current_country_code" else s
                for s in catalog.specs
            ),
            config_fingerprint=config.fingerprint(),
        )
        result = LeakageAuditor(config, broken).audit(_frame(_events(5)))
        assert not next(
            c.passed
            for c in result.checks
            if c.name == "CURRENT_VS_PRIOR_FIELDS_DISTINGUISHED"
        )

    def test_a_windowed_column_claiming_current_context_is_caught(self) -> None:
        from password_attack_detector.features.catalog import LeakageClass

        config = _CONFIG
        catalog = build_catalog(config)
        broken_spec = catalog.get("user_attempt_count__5m").model_copy(
            update={
                "leakage_class": LeakageClass.CURRENT_EVENT_CONTEXT,
                "uses_current_event": True,
            }
        )
        broken = FeatureCatalog(
            tuple(
                broken_spec if s.name == "user_attempt_count__5m" else s
                for s in catalog.specs
            ),
            config_fingerprint=config.fingerprint(),
        )
        result = LeakageAuditor(config, broken).audit(_frame(_events(5)))
        assert not next(
            c.passed
            for c in result.checks
            if c.name == "CURRENT_VS_PRIOR_FIELDS_DISTINGUISHED"
        )

    def test_join_integrity_passes_for_matched_tables(self) -> None:
        events = _events(40)
        labels = make_labels(events)
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=_split(events, labels),
        )
        assert next(c.passed for c in result.checks if c.name == "JOIN_KEY_INTEGRITY")

    def test_join_integrity_catches_a_missing_label(self) -> None:
        events = _events(40)
        labels = make_labels(events[:-5])
        result = _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=_split(events, make_labels(events)),
        )
        assert not next(
            c.passed for c in result.checks if c.name == "JOIN_KEY_INTEGRITY"
        )

    def test_join_integrity_catches_a_missing_split_row(self) -> None:
        events = _events(40)
        labels = make_labels(events)
        assignments = _split(events, labels)[:-3]
        result = _auditor().audit(
            _frame(events), events=events, labels=labels, assignments=assignments
        )
        assert not next(
            c.passed for c in result.checks if c.name == "JOIN_KEY_INTEGRITY"
        )


# --- privacy of the report -------------------------------------------------


class TestReportPrivacy:
    @pytest.fixture()
    def result(self) -> LeakageAuditResult:
        events = _events(40)
        labels = make_labels(events)
        return _auditor().audit(
            _frame(events),
            events=events,
            labels=labels,
            assignments=_split(events, labels),
        )

    def test_result_contains_no_event_identifiers(
        self, result: LeakageAuditResult
    ) -> None:
        text = json.dumps(result.to_dict())
        for event in _events(40):
            assert str(event.event_id) not in text

    def test_result_contains_no_pseudonyms(self, result: LeakageAuditResult) -> None:
        text = json.dumps(result.to_dict())
        assert not _PSEUDONYM_RE.search(text)

    def test_markdown_reports_aggregates_only(self, result: LeakageAuditResult) -> None:
        markdown = audit_result_to_markdown(result)
        for event in _events(40):
            assert str(event.event_id) not in markdown

    def test_markdown_lists_every_check(self, result: LeakageAuditResult) -> None:
        markdown = audit_result_to_markdown(result)
        for name in CHECK_NAMES:
            assert name in markdown

    def test_markdown_ends_with_known_limitations(
        self, result: LeakageAuditResult
    ) -> None:
        assert "## Known limitations" in audit_result_to_markdown(result)

    def test_markdown_makes_no_effectiveness_claim(
        self, result: LeakageAuditResult
    ) -> None:
        markdown = audit_result_to_markdown(result).lower()
        assert "says nothing about detection effectiveness" in markdown
