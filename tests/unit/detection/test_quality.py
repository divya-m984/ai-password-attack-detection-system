"""Tests for the detection quality report.

Metrics are checked against a hand-computed fixture rather than against the
code that produced them, and both rendered formats are swept for anything that
should never leave the run directory.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any

import pytest

from password_attack_detector.detection.alerts import AlertBuilder, AlertingResult
from password_attack_detector.detection.config import (
    AlertingConfig,
    DetectionConfig,
    SeverityThresholds,
)
from password_attack_detector.detection.engine import DetectionEngine, EngineResult
from password_attack_detector.detection.enums import RuleStatus, Severity
from password_attack_detector.detection.quality import (
    DEFINITIONS,
    generate_detection_quality_report,
    report_to_json,
    report_to_markdown,
)
from password_attack_detector.detection.scoring import RiskScorer, ScoringResult
from password_attack_detector.detection.serialization import (
    compute_alert_fingerprint,
    compute_detection_fingerprint,
    compute_risk_fingerprint,
)
from password_attack_detector.detection.validation import DetectionValidator
from tests.unit.detection import factories

WHEN = factories.WHEN
CONFIG = DetectionConfig()

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_PSEUDONYM_RE = re.compile(r"(u|s|d|sess):[0-9a-f]{32}")


@pytest.fixture(scope="module")
def pipeline() -> tuple[EngineResult, ScoringResult, AlertingResult, int]:
    """Run the real pipeline over a fixed twelve-snapshot dataset."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    builders = [
        factories.brute_force_row,
        factories.spraying_row,
        factories.stuffing_row,
        factories.quiet_row,
    ]
    rows = [
        builders[index % 4](
            catalog,
            anchor_event_id=f"anchor-{index:04d}",
            anchor_event_time=WHEN + timedelta(minutes=index * 3),
        )
        for index in range(12)
    ]
    engine_result = engine.run(rows)
    scoring_result = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    alerting_result = AlertBuilder(CONFIG).build(
        scoring_result.assessments, detections=list(engine_result.fired_detections)
    )
    return engine_result, scoring_result, alerting_result, len(rows)


def build_report(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
    **overrides: Any,
) -> Any:
    """Build a quality report from a pipeline run."""
    engine_result, scoring_result, alerting_result, snapshot_count = pipeline
    kwargs: dict[str, Any] = {
        "engine_result": engine_result,
        "scoring_result": scoring_result,
        "alerting_result": alerting_result,
        "config": CONFIG,
        "input_snapshot_count": snapshot_count,
        "detection_fingerprint": compute_detection_fingerprint(
            engine_result.fired_detections
        ),
        "risk_fingerprint": compute_risk_fingerprint(scoring_result.assessments),
        "alert_fingerprint": compute_alert_fingerprint(alerting_result.alerts),
    }
    kwargs.update(overrides)
    return generate_detection_quality_report(**kwargs)


# ---------------------------------------------------------------------------
# Exact metrics
# ---------------------------------------------------------------------------


def test_evaluation_counts_match_the_engine(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    engine_result, _, _, snapshot_count = pipeline
    report = build_report(pipeline)

    assert report.input_snapshot_count == snapshot_count
    assert report.evaluated_snapshot_count == 12
    assert report.total_rule_evaluation_count == 12 * 9
    assert (
        report.fired_count + report.not_fired_count + report.insufficient_data_count
        == report.total_rule_evaluation_count
    )
    assert report.fired_count == engine_result.status_counts[str(RuleStatus.FIRED)]


def test_per_rule_and_per_family_counts_agree(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    report = build_report(pipeline)
    assert sum(report.per_rule_trigger_counts.values()) == report.fired_count
    assert sum(report.per_family_trigger_counts.values()) == report.fired_count
    assert sum(report.attack_category_distribution.values()) == report.fired_count


def test_the_severity_distribution_covers_every_band(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    report = build_report(pipeline)
    assert set(report.severity_distribution) == {str(item) for item in Severity}
    assert sum(report.severity_distribution.values()) == report.evaluated_snapshot_count


def test_the_alerting_counts_come_from_the_alerting_statistics(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    _, _, alerting_result, _ = pipeline
    report = build_report(pipeline)
    stats = alerting_result.stats

    assert report.alert_count == len(alerting_result.alerts)
    assert report.grouped_detection_count == stats.grouped_detection_count
    assert report.suppressed_total == stats.suppressed_total
    assert report.escalated_count == stats.escalated_count
    assert (
        report.entity_scoped_count + report.category_scoped_count == report.alert_count
    )
    assert sum(report.grouping_mode_counts.values()) == report.alert_count


def test_the_risk_summary_is_computed_from_the_assessments(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    _, scoring_result, _, _ = pipeline
    report = build_report(pipeline)
    scores = [item.risk_score for item in scoring_result.assessments]

    assert report.risk_score_summary["min"] == min(scores)
    assert report.risk_score_summary["max"] == max(scores)
    assert report.risk_score_summary["mean"] == pytest.approx(sum(scores) / len(scores))


def test_the_average_events_per_alert_is_exact(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    _, _, alerting_result, _ = pipeline
    report = build_report(pipeline)
    expected = sum(
        alert.contributing_event_count for alert in alerting_result.alerts
    ) / len(alerting_result.alerts)
    assert report.average_events_per_alert == pytest.approx(expected)


def test_an_empty_alert_set_reports_nulls_not_zeros() -> None:
    """A statistic over nothing is unmeasured, not zero."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    rows = [factories.quiet_row(catalog, anchor_event_id="a1")]
    engine_result = engine.run(rows)
    scoring_result = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    alerting_result = AlertBuilder(CONFIG).build(scoring_result.assessments)

    report = generate_detection_quality_report(
        engine_result=engine_result,
        scoring_result=scoring_result,
        alerting_result=alerting_result,
        config=CONFIG,
    )
    assert report.alert_count == 0
    assert report.average_events_per_alert is None
    assert report.alert_duration_seconds_summary["mean"] is None
    assert report.aggregate_risk_summary["max"] is None
    assert report.peak_risk_summary["min"] is None


def test_the_configuration_gates_are_reported(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    report = build_report(pipeline)
    assert report.min_alert_risk_score == CONFIG.alerting.min_alert_risk_score
    assert report.min_alert_severity == str(CONFIG.alerting.min_alert_severity)
    assert report.low_alert_reachable is CONFIG.low_alert_reachable


def test_an_unreachable_low_band_is_recorded(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """Raising the floors past the LOW band is visible, not surprising."""
    config = DetectionConfig(
        alerting=AlertingConfig(min_alert_severity=Severity.MEDIUM),
        severity_thresholds=SeverityThresholds(),
    )
    report = build_report(pipeline, config=config)
    assert report.low_alert_reachable is False


def test_the_validation_result_is_carried_through(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    engine_result, scoring_result, alerting_result, _ = pipeline
    validation = DetectionValidator(CONFIG).validate(
        list(engine_result.fired_detections),
        list(scoring_result.assessments),
        list(alerting_result.alerts),
    )
    report = build_report(pipeline, validation_result=validation)

    assert report.validation_status == str(validation.status)
    assert report.validation_result is not None
    assert report.warning_summary == [item.code for item in validation.warnings]


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def test_the_report_defines_what_its_numbers_mean(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    report = build_report(pipeline)
    assert set(report.definitions) == set(DEFINITIONS)
    assert set(report.definitions) >= {
        "risk_score",
        "signal_strength",
        "aggregate_risk_score",
        "peak_risk_score",
        "evidence",
    }


@pytest.mark.parametrize("term", ["risk_score", "signal_strength"])
def test_the_score_definitions_deny_being_probabilities(term: str) -> None:
    assert "not a probability" in DEFINITIONS[term].lower()


def test_the_aggregate_definition_names_the_arithmetic_mean() -> None:
    assert "arithmetic mean" in DEFINITIONS["aggregate_risk_score"].lower()


def test_the_peak_definition_names_the_maximum() -> None:
    assert "maximum" in DEFINITIONS["peak_risk_score"].lower()


def test_the_evidence_definition_denies_causal_proof() -> None:
    text = DEFINITIONS["evidence"].lower()
    assert "not causal proof" in text
    assert "indicator" in text


def test_both_renderers_carry_the_definitions(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    report = build_report(pipeline)
    markdown = report_to_markdown(report)
    payload = json.loads(report_to_json(report))
    assert "not a probability" in markdown
    assert "arithmetic mean" in markdown
    assert payload["definitions"] == report.definitions


# ---------------------------------------------------------------------------
# Rendering and privacy
# ---------------------------------------------------------------------------


def test_the_json_renderer_round_trips(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    report = build_report(pipeline)
    assert json.loads(report_to_json(report)) == json.loads(
        json.dumps(report.to_dict(), default=str)
    )


def test_the_renderers_are_deterministic(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    report = build_report(pipeline)
    assert report_to_json(report) == report_to_json(build_report(pipeline))
    assert report_to_markdown(report) == report_to_markdown(build_report(pipeline))


def test_the_markdown_report_is_well_formed(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    markdown = report_to_markdown(build_report(pipeline))
    assert markdown.startswith("# Detection Quality Report")
    assert markdown.endswith("\n")
    for heading in ("## Overview", "## Alerting", "## Fingerprints"):
        assert heading in markdown


def test_no_rendered_report_carries_an_identifier_or_a_scope_value() -> None:
    """The privacy sweep, over a run whose inputs are deliberately identifying."""
    catalog = factories.feature_catalog()
    engine = DetectionEngine(CONFIG, feature_catalog=catalog)
    rows = [
        factories.brute_force_row(
            catalog,
            anchor_event_id="3f2504e0-4f89-41d3-9a0c-0305e82c3301",
            anchor_event_time=WHEN,
        ),
        factories.spraying_row(
            catalog,
            anchor_event_id="7a1b2c3d-4e5f-4a6b-8c9d-0e1f2a3b4c5d",
            anchor_event_time=WHEN + timedelta(minutes=4),
        ),
    ]
    engine_result = engine.run(rows)
    scoring_result = RiskScorer(CONFIG).score(engine.run_diagnostic(rows))
    scope = [
        factories.scope_record(row["anchor_event_id"], user="ab" * 16, source="cd" * 16)
        for row in rows
    ]
    from password_attack_detector.detection.alerts import build_entity_scope_table

    alerting_result = AlertBuilder(CONFIG).build(
        scoring_result.assessments,
        entity_scope=build_entity_scope_table(scope),
    )
    assert any(alert.scope_value for alert in alerting_result.alerts)

    report = generate_detection_quality_report(
        engine_result=engine_result,
        scoring_result=scoring_result,
        alerting_result=alerting_result,
        config=CONFIG,
    )
    for rendered in (report_to_json(report), report_to_markdown(report)):
        assert not _UUID_RE.search(rendered)
        assert not _PSEUDONYM_RE.search(rendered)
        assert "ab" * 16 not in rendered
        assert "cd" * 16 not in rendered
        assert "/home/" not in rendered


def test_no_rendered_report_carries_an_evidence_message(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """Aggregates only: a rendered evidence sentence would be a raw row."""
    engine_result, _, _, _ = pipeline
    sample = engine_result.fired_detections[0].evidence[0].message
    report = build_report(pipeline)
    for rendered in (report_to_json(report), report_to_markdown(report)):
        assert sample not in rendered


def test_the_report_records_the_run_fingerprints(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    engine_result, scoring_result, alerting_result, _ = pipeline
    report = build_report(pipeline)
    assert report.configuration_fingerprint == engine_result.configuration_fingerprint
    assert report.rule_catalog_fingerprint == engine_result.rule_catalog_fingerprint
    assert report.detection_fingerprint == compute_detection_fingerprint(
        engine_result.fired_detections
    )
    assert report.risk_fingerprint == compute_risk_fingerprint(
        scoring_result.assessments
    )
    assert report.alert_fingerprint == compute_alert_fingerprint(alerting_result.alerts)


def test_an_empty_distribution_renders_no_table(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """A heading with no rows under it would read as a missing measurement."""
    report = build_report(pipeline)
    empty = type(report)(**{**report.__dict__, "attack_category_distribution": {}})
    assert "## Attack categories" not in report_to_markdown(empty)
    assert "## Attack categories" in report_to_markdown(report)


def test_validation_warnings_are_rendered_when_present(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    engine_result, scoring_result, alerting_result, _ = pipeline
    validation = DetectionValidator(CONFIG).validate(
        list(engine_result.fired_detections),
        list(scoring_result.assessments),
        list(alerting_result.alerts),
    )
    assert validation.warnings
    markdown = report_to_markdown(build_report(pipeline, validation_result=validation))
    assert "## Validation warnings" in markdown
    for finding in validation.warnings:
        assert f"`{finding.code}`" in markdown
