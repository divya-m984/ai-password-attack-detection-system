"""Tests for ground-truth evaluation.

Most metrics are checked against values computed by hand on a small fixture,
not against the code that produced them -- a test that recomputes the
implementation's own arithmetic proves nothing.  The remainder covers the
boundaries that matter operationally: split isolation, holdout separation,
empty denominators, deterministic class order, and the standing guarantee that
nothing here tunes a threshold.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, replace
from datetime import timedelta
from types import ModuleType
from typing import Any

import pytest

from password_attack_detector.detection import alerts as alerts_module
from password_attack_detector.detection import engine as engine_module
from password_attack_detector.detection import evaluation as evaluation_module
from password_attack_detector.detection import scoring as scoring_module
from password_attack_detector.detection.alerts import AlertBuilder
from password_attack_detector.detection.config import DetectionConfig
from password_attack_detector.detection.engine import DetectionEngine
from password_attack_detector.detection.enums import AttackCategory
from password_attack_detector.detection.evaluation import (
    CATEGORY_TO_SCENARIO,
    NOVEL_HOLDOUT_SPLIT,
    SYNTHETIC_CAVEAT,
    CampaignRecord,
    EvaluationReport,
    LabelRecord,
    SplitRecord,
    evaluate_detection_run,
    report_to_json,
    report_to_markdown,
)
from password_attack_detector.detection.scoring import RiskScorer
from tests.unit.detection import factories

WHEN = factories.WHEN
CONFIG = DetectionConfig()

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_PSEUDONYM_RE = re.compile(r"(u|s|d|sess):[0-9a-f]{32}")

#: The fixture: four repeating scenarios, three of them malicious.
_SCENARIOS = ("brute_force", "password_spraying", "credential_stuffing", "normal")
_BUILDERS = (
    factories.brute_force_row,
    factories.spraying_row,
    factories.stuffing_row,
    factories.quiet_row,
)


@dataclass(frozen=True)
class Run:
    """One pipeline run paired with its hand-assigned ground truth."""

    detections: list[Any]
    assessments: list[Any]
    alerts: list[Any]
    labels: list[LabelRecord]
    splits: list[SplitRecord]
    campaigns: list[CampaignRecord]


def build_run(
    count: int = 12,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    with_campaigns: bool = True,
) -> Run:
    """Run the real pipeline and pair it with hand-assigned ground truth."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    rows = []
    labels = []
    split_records = []
    campaigns = []
    for index in range(count):
        kind = index % 4
        anchor = f"a{index}"
        rows.append(
            _BUILDERS[kind](
                catalog,
                anchor_event_id=anchor,
                anchor_event_time=WHEN + timedelta(minutes=index * 3),
            )
        )
        labels.append(
            LabelRecord(
                event_id=anchor,
                attack_class=_SCENARIOS[kind],
                malicious=kind != 3,
            )
        )
        split_records.append(
            SplitRecord(event_id=anchor, split=splits[index % len(splits)])
        )
        if kind != 3 and with_campaigns:
            campaigns.append(CampaignRecord(event_id=anchor, campaign_id=f"c{kind}"))

    detections = list(engine.run(rows).fired_detections)
    scored = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    alerting = AlertBuilder(CONFIG).build(scored.assessments, detections=detections)
    return Run(
        detections=detections,
        assessments=list(scored.assessments),
        alerts=list(alerting.alerts),
        labels=labels,
        splits=split_records,
        campaigns=campaigns,
    )


def evaluate(run: Run, **overrides: Any) -> EvaluationReport:
    """Evaluate a prepared run."""
    kwargs: dict[str, Any] = {
        "detections": run.detections,
        "assessments": run.assessments,
        "alerts": run.alerts,
        "labels": run.labels,
        "splits": run.splits,
        "campaigns": run.campaigns,
        "configuration_fingerprint": CONFIG.fingerprint(),
    }
    kwargs.update(overrides)
    return evaluate_detection_run(**kwargs)


@pytest.fixture(scope="module")
def run() -> Run:
    return build_run()


# ---------------------------------------------------------------------------
# Exact event-level metrics
# ---------------------------------------------------------------------------


def test_event_counts_match_the_hand_assigned_labels(run: Run) -> None:
    """Twelve events, nine malicious across three scenarios, three benign."""
    overall = evaluate(run).overall
    assert overall.event_count == 12
    assert overall.malicious_count == 9
    assert overall.benign_count == 3


def test_the_fixture_detects_every_malicious_event_and_no_benign_one(
    run: Run,
) -> None:
    overall = evaluate(run).overall
    assert overall.detection_rate == 1.0
    assert overall.false_positive_rate == 0.0


def test_a_benign_event_that_fires_raises_the_false_positive_rate() -> None:
    """One of the three benign events relabelled from a firing scenario."""
    run = build_run()
    labels = [
        LabelRecord(event_id=item.event_id, attack_class="normal", malicious=False)
        if item.event_id == "a0"
        else item
        for item in run.labels
    ]
    overall = evaluate(run, labels=labels).overall
    assert overall.benign_count == 4
    assert overall.false_positive_rate == pytest.approx(0.25)


def test_an_undetected_malicious_event_lowers_the_detection_rate() -> None:
    """A quiet event relabelled malicious is a miss, and must show as one."""
    run = build_run()
    labels = [
        LabelRecord(event_id=item.event_id, attack_class="brute_force", malicious=True)
        if item.event_id == "a3"
        else item
        for item in run.labels
    ]
    overall = evaluate(run, labels=labels).overall
    assert overall.malicious_count == 10
    assert overall.detection_rate == pytest.approx(0.9)


def test_per_scenario_detection_rates_are_reported_for_every_scenario(
    run: Run,
) -> None:
    overall = evaluate(run).overall
    assert set(overall.per_scenario_detection_rate) == {
        "brute_force",
        "password_spraying",
        "credential_stuffing",
    }
    assert all(rate == 1.0 for rate in overall.per_scenario_detection_rate.values())


def test_the_duplicate_reduction_ratio_is_detections_over_alerts(
    run: Run,
) -> None:
    report = evaluate(run)
    expected = len(run.detections) / len(run.alerts)
    assert report.overall.duplicate_reduction_ratio == pytest.approx(expected)


def test_alerts_per_thousand_events_is_scaled_from_the_event_count(
    run: Run,
) -> None:
    report = evaluate(run)
    expected = 1000.0 * len(run.alerts) / 12
    assert report.overall.alerts_per_thousand_events == pytest.approx(expected)


def test_the_insufficient_data_rate_is_per_assessment(
    run: Run,
) -> None:
    report = evaluate(run)
    total = sum(item.insufficient_data_count for item in run.assessments)
    assert report.overall.insufficient_data_rate == pytest.approx(total / 12)


# ---------------------------------------------------------------------------
# Category metrics and the confusion matrix
# ---------------------------------------------------------------------------


def test_the_confusion_matrix_class_order_is_deterministic(
    run: Run,
) -> None:
    """Sorted scenarios, then ``none`` -- so two renderings read alike."""
    overall = evaluate(run).overall
    expected = sorted({str(value) for value in CATEGORY_TO_SCENARIO.values()})
    assert overall.confusion_matrix_classes == [*expected, "none"]
    assert (
        overall.confusion_matrix_classes
        == evaluate(run).overall.confusion_matrix_classes
    )


def test_the_confusion_matrix_totals_every_event(run: Run) -> None:
    overall = evaluate(run).overall
    total = sum(sum(row.values()) for row in overall.confusion_matrix.values())
    assert total == overall.event_count


def test_the_fixture_produces_perfect_category_metrics(
    run: Run,
) -> None:
    overall = evaluate(run).overall
    for name in ("brute_force", "password_spraying", "credential_stuffing"):
        metric = overall.category_metrics[name]
        assert metric.support == 3
        assert metric.true_positives == 3
        assert metric.precision == 1.0
        assert metric.recall == 1.0
        assert metric.f1 == 1.0


def test_a_category_nothing_claimed_reports_null_precision(
    run: Run,
) -> None:
    """No prediction and no support means unmeasured, not zero."""
    overall = evaluate(run).overall
    metric = overall.category_metrics["impossible_travel"]
    assert metric.support == 0
    assert metric.predicted == 0
    assert metric.precision is None
    assert metric.recall is None
    assert metric.f1 is None


def test_macro_and_weighted_summaries_are_reported(run: Run) -> None:
    overall = evaluate(run).overall
    assert overall.macro_f1 == pytest.approx(1.0)
    assert overall.weighted_f1 == pytest.approx(1.0)
    assert overall.macro_precision == pytest.approx(1.0)
    assert overall.weighted_recall == pytest.approx(1.0)


def test_every_claimable_category_maps_to_a_synthetic_scenario() -> None:
    from password_attack_detector.data.enums import ScenarioType

    for category, scenario in CATEGORY_TO_SCENARIO.items():
        assert isinstance(category, AttackCategory)
        assert isinstance(scenario, ScenarioType)


def test_the_mfa_category_claims_no_synthetic_scenario() -> None:
    """The generator has no MFA scenario, so the category is not scored as one."""
    assert AttackCategory.MFA_SEQUENCE_ANOMALY not in CATEGORY_TO_SCENARIO


# ---------------------------------------------------------------------------
# Splits and the holdout
# ---------------------------------------------------------------------------


def test_metrics_are_reported_per_split(run: Run) -> None:
    report = evaluate(run)
    assert sorted(report.by_split) == ["test", "train", "validation"]
    assert sum(m.event_count for m in report.by_split.values()) == 12


def test_a_split_sees_only_its_own_events(run: Run) -> None:
    """Split isolation: one split's metrics cannot include another's events."""
    report = evaluate(run)
    for metrics in report.by_split.values():
        assert metrics.event_count == 4


def test_the_novel_holdout_is_reported_separately_and_never_folded_in() -> None:
    run = build_run(count=12, splits=("train", "test", NOVEL_HOLDOUT_SPLIT))
    report = evaluate(run)
    assert NOVEL_HOLDOUT_SPLIT not in report.by_split
    assert report.novel_holdout is not None
    assert report.novel_holdout.split == NOVEL_HOLDOUT_SPLIT
    assert sum(m.event_count for m in report.by_split.values()) < 12


def test_an_absent_holdout_reports_nothing_rather_than_an_empty_split(
    run: Run,
) -> None:
    assert evaluate(run).novel_holdout is None


def test_the_holdout_is_not_a_supervised_category() -> None:
    """Its whole purpose is anomalies no rule was written for."""
    assert NOVEL_HOLDOUT_SPLIT not in {
        str(value) for value in CATEGORY_TO_SCENARIO.values()
    }


# ---------------------------------------------------------------------------
# Campaign metrics
# ---------------------------------------------------------------------------


def test_campaign_coverage_counts_each_campaign_once(run: Run) -> None:
    """Repeated detections inside one campaign cannot inflate coverage."""
    campaign = evaluate(run).campaign_metrics
    assert campaign.available is True
    assert campaign.campaign_count == 3
    assert campaign.detected_campaign_count == 3
    assert campaign.coverage == 1.0
    assert campaign.false_negative_count == 0


def test_time_to_first_detection_is_zero_when_the_first_event_fires(
    run: Run,
) -> None:
    campaign = evaluate(run).campaign_metrics
    assert campaign.time_to_first_detection_seconds["min"] == 0.0
    assert campaign.time_to_first_detection_seconds["max"] == 0.0


def test_time_to_first_detection_measures_the_real_gap() -> None:
    """A campaign whose first event is quiet detects later, by exactly the gap."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    rows = [
        factories.quiet_row(catalog, anchor_event_id="q1", anchor_event_time=WHEN),
        factories.brute_force_row(
            catalog,
            anchor_event_id="q2",
            anchor_event_time=WHEN + timedelta(minutes=10),
        ),
    ]
    detections = list(engine.run(rows).fired_detections)
    scored = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    report = evaluate_detection_run(
        detections=detections,
        assessments=list(scored.assessments),
        alerts=[],
        labels=[
            LabelRecord(event_id="q1", attack_class="brute_force", malicious=True),
            LabelRecord(event_id="q2", attack_class="brute_force", malicious=True),
        ],
        splits=[
            SplitRecord(event_id="q1", split="test"),
            SplitRecord(event_id="q2", split="test"),
        ],
        campaigns=[
            CampaignRecord(event_id="q1", campaign_id="c1"),
            CampaignRecord(event_id="q2", campaign_id="c1"),
        ],
    )
    assert report.campaign_metrics.time_to_first_detection_seconds["min"] == 600.0


def test_an_undetected_campaign_is_a_false_negative() -> None:
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    rows = [factories.quiet_row(catalog, anchor_event_id="q1")]
    scored = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    report = evaluate_detection_run(
        detections=[],
        assessments=list(scored.assessments),
        alerts=[],
        labels=[LabelRecord(event_id="q1", attack_class="brute_force", malicious=True)],
        splits=[SplitRecord(event_id="q1", split="test")],
        campaigns=[CampaignRecord(event_id="q1", campaign_id="c1")],
    )
    assert report.campaign_metrics.campaign_count == 1
    assert report.campaign_metrics.detected_campaign_count == 0
    assert report.campaign_metrics.coverage == 0.0
    assert report.campaign_metrics.false_negative_count == 1


def test_campaign_metrics_are_unavailable_without_a_campaign_table() -> None:
    """Reported as unavailable rather than fabricated."""
    run = build_run(with_campaigns=False)
    campaign = evaluate(run, campaigns=None).campaign_metrics
    assert campaign.available is False
    assert campaign.coverage is None
    assert campaign.time_to_first_detection_seconds["median"] is None


# ---------------------------------------------------------------------------
# Null semantics and edge cases
# ---------------------------------------------------------------------------


def test_an_empty_benign_class_yields_a_null_false_positive_rate() -> None:
    run = build_run(count=3)
    labels = [
        LabelRecord(event_id=item.event_id, attack_class="brute_force", malicious=True)
        for item in run.labels
    ]
    overall = evaluate(run, labels=labels).overall
    assert overall.benign_count == 0
    assert overall.false_positive_rate is None


def test_an_empty_malicious_class_yields_a_null_detection_rate() -> None:
    run = build_run(count=3)
    labels = [
        LabelRecord(event_id=item.event_id, attack_class="normal", malicious=False)
        for item in run.labels
    ]
    overall = evaluate(run, labels=labels).overall
    assert overall.malicious_count == 0
    assert overall.detection_rate is None


def test_an_empty_alert_set_yields_null_alert_metrics(run: Run) -> None:
    report = evaluate(run, alerts=[])
    assert report.overall.alert_precision is None
    assert report.overall.alert_recall is None
    assert report.overall.duplicate_reduction_ratio is None


def test_an_assessment_without_a_label_is_excluded_rather_than_guessed() -> None:
    """A missing relationship drops the row; it never becomes a benign one."""
    run = build_run(count=8)
    labels = [item for item in run.labels if item.event_id != "a0"]
    overall = evaluate(run, labels=labels).overall
    assert overall.event_count == 7


def test_evaluation_of_nothing_is_well_formed() -> None:
    report = evaluate_detection_run(
        detections=[], assessments=[], alerts=[], labels=[], splits=[]
    )
    assert report.overall.event_count == 0
    assert report.overall.detection_rate is None
    assert report.overall.false_positive_rate is None
    assert report.by_split == {}
    assert report.novel_holdout is None


# ---------------------------------------------------------------------------
# Determinism and the boundary
# ---------------------------------------------------------------------------


def test_input_row_order_does_not_change_any_metric(run: Run) -> None:
    forward = evaluate(run)
    reversed_run = replace(
        run,
        detections=list(reversed(run.detections)),
        assessments=list(reversed(run.assessments)),
        alerts=list(reversed(run.alerts)),
        labels=list(reversed(run.labels)),
        splits=list(reversed(run.splits)),
        campaigns=list(reversed(run.campaigns)),
    )
    assert evaluate(reversed_run).to_dict() == forward.to_dict()


def test_no_threshold_optimisation_path_exists(run: Run) -> None:
    """The flag is a literal, and no code here writes a configuration value."""
    assert evaluate(run).threshold_optimization_performed is False
    source = inspect.getsource(evaluation_module)
    for forbidden in ("DetectionConfig(", "model_copy", "min_alert_risk_score ="):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("module", [engine_module, scoring_module, alerts_module])
def test_the_detection_path_never_imports_evaluation_or_labels(
    module: ModuleType,
) -> None:
    source = inspect.getsource(module)
    for forbidden in (
        "detection.evaluation",
        "LabelRecord",
        "SplitRecord",
        "CampaignRecord",
        "features.splitting",
    ):
        assert forbidden not in source, forbidden


def _imported_modules(module: ModuleType) -> set[str]:
    """Return every module name *module* imports, from its syntax tree.

    Parsed rather than text-matched: these modules' prose names the things
    they refuse to read, so a substring scan would flag the very docstring
    that documents the guarantee.
    """
    import ast

    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_evaluation_is_the_only_label_reader() -> None:
    """Stated as an import-graph fact, not left to convention."""
    from password_attack_detector.detection import quality as quality_module
    from password_attack_detector.detection import validation as validation_module

    forbidden = {
        "password_attack_detector.detection.evaluation",
        "password_attack_detector.features.splitting",
        "password_attack_detector.features.serialization",
    }
    for module in (
        engine_module,
        scoring_module,
        alerts_module,
        quality_module,
        validation_module,
    ):
        imported = _imported_modules(module)
        assert not imported & forbidden, (module, imported & forbidden)
        assert not any(
            name.endswith(("LabelRecord", "SplitRecord", "CampaignRecord"))
            for name in imported
        )


def test_evaluation_does_import_the_label_types() -> None:
    """The converse: the one module that may read ground truth, does."""
    assert "password_attack_detector.data.enums" in _imported_modules(evaluation_module)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_every_report_is_labelled_synthetic(run: Run) -> None:
    report = evaluate(run)
    assert report.synthetic_caveat == SYNTHETIC_CAVEAT
    assert "synthetic" in SYNTHETIC_CAVEAT.lower()
    assert "not evidence of real-world" in SYNTHETIC_CAVEAT.lower()
    assert SYNTHETIC_CAVEAT in report_to_markdown(report)
    assert SYNTHETIC_CAVEAT in report_to_json(report)


def test_no_report_claims_real_world_effectiveness(run: Run) -> None:
    markdown = report_to_markdown(evaluate(run)).lower()
    for phrase in ("production ready", "proven effective", "guarantees detection"):
        assert phrase not in markdown


def test_the_renderers_are_deterministic(run: Run) -> None:
    first, second = evaluate(run), evaluate(run)
    assert report_to_json(first) == report_to_json(second)
    assert report_to_markdown(first) == report_to_markdown(second)


def test_the_json_report_round_trips(run: Run) -> None:
    payload = json.loads(report_to_json(evaluate(run)))
    assert payload["overall"]["event_count"] == 12
    assert payload["threshold_optimization_performed"] is False


def test_no_report_carries_an_identifier() -> None:
    """Aggregates only: the anchors in this fixture are UUID-shaped on purpose."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    anchor = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
    rows = [factories.brute_force_row(catalog, anchor_event_id=anchor)]
    detections = list(engine.run(rows).fired_detections)
    scored = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    report = evaluate_detection_run(
        detections=detections,
        assessments=list(scored.assessments),
        alerts=[],
        labels=[
            LabelRecord(event_id=anchor, attack_class="brute_force", malicious=True)
        ],
        splits=[SplitRecord(event_id=anchor, split="test")],
        campaigns=[CampaignRecord(event_id=anchor, campaign_id="campaign-0001")],
    )
    for rendered in (report_to_json(report), report_to_markdown(report)):
        assert not _UUID_RE.search(rendered)
        assert not _PSEUDONYM_RE.search(rendered)
        assert "campaign-0001" not in rendered
        assert "/home/" not in rendered


def test_the_markdown_report_is_well_formed(run: Run) -> None:
    markdown = report_to_markdown(evaluate(run))
    assert markdown.startswith("# Rule Evaluation Report")
    assert markdown.endswith("\n")
    for heading in ("## Overall", "## By split", "## Campaigns", "## Novel-anomaly"):
        assert heading in markdown


def test_unavailable_metrics_render_as_unavailable() -> None:
    run = build_run(count=3, with_campaigns=False)
    markdown = report_to_markdown(evaluate(run, campaigns=None, alerts=[]))
    assert "unavailable" in markdown


def test_a_category_claiming_no_synthetic_scenario_predicts_none(
    run: Run,
) -> None:
    """The MFA category maps to no generated scenario, so it predicts ``none``."""
    reassigned = [
        item.model_construct(
            **{
                **item.__dict__,
                "primary_attack_category": AttackCategory.MFA_SEQUENCE_ANOMALY,
                "contributing_categories": (AttackCategory.MFA_SEQUENCE_ANOMALY,),
            }
        )
        if item.fired_rule_count > 0
        else item
        for item in run.assessments
    ]
    overall = evaluate(run, assessments=reassigned).overall
    # Every malicious event now predicts ``none``, so no category scores.
    assert all(
        metric.true_positives == 0 for metric in overall.category_metrics.values()
    )
    assert overall.confusion_matrix["brute_force"]["none"] == 3


def test_the_holdout_section_renders_its_metrics() -> None:
    run = build_run(count=12, splits=("train", "test", NOVEL_HOLDOUT_SPLIT))
    markdown = report_to_markdown(evaluate(run))
    assert "Reported separately" in markdown
    assert f"### `{NOVEL_HOLDOUT_SPLIT}`" in markdown
