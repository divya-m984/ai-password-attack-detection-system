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


# ---------------------------------------------------------------------------
# Unavailable is not zero
#
# A live run observes every rule evaluation and every suppression decision. A
# report rebuilt from published artifacts sees only what those artifacts
# persist. Reporting zero for a counter nobody recorded would assert the event
# never happened -- a stronger claim than "this artifact set does not say".
# ---------------------------------------------------------------------------


def reconstructed(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
    **overrides: Any,
) -> Any:
    """Rebuild a quality report from a pipeline run's published artifacts."""
    from password_attack_detector.detection.quality import (
        reconstruct_detection_quality_report,
    )

    engine_result, scoring_result, alerting_result, _ = pipeline
    kwargs: dict[str, Any] = {
        "detections": list(engine_result.fired_detections),
        "assessments": list(scoring_result.assessments),
        "alerts": list(alerting_result.alerts),
        "config": CONFIG,
    }
    kwargs.update(overrides)
    return reconstruct_detection_quality_report(**kwargs)


def test_a_live_run_report_is_fully_populated(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """Every counter is an integer; nothing is unavailable."""
    from password_attack_detector.detection.quality import (
        UNAVAILABLE_WHEN_RECONSTRUCTED,
        QualityReportSource,
    )

    report = build_report(pipeline)
    assert report.report_source is QualityReportSource.LIVE_RUN
    assert report.is_reconstructed is False
    assert report.unavailable_metrics == []
    assert report.reconstruction_note is None
    for name in UNAVAILABLE_WHEN_RECONSTRUCTED:
        assert isinstance(getattr(report, name), int), name


def test_a_live_run_preserves_a_measured_zero(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """This fixture suppresses nothing, and the report must say zero."""
    _, _, alerting_result, _ = pipeline
    assert alerting_result.stats.suppressed_total == 0

    report = build_report(pipeline)
    assert report.suppressed_total == 0
    assert report.suppressed_by_cooldown_count == 0
    assert report.suppressed_by_rate_limit_count == 0
    assert report.escalated_count == 0
    assert report.scope_missing_count == 0
    # A measured zero is an integer, not a null.
    assert report.suppressed_total is not None


def test_a_measured_zero_renders_as_zero_not_unavailable(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    from password_attack_detector.detection.quality import UNAVAILABLE_LABEL

    markdown = report_to_markdown(build_report(pipeline))
    assert "| Suppressed total | 0 |" in markdown
    assert UNAVAILABLE_LABEL not in markdown


@pytest.mark.parametrize(
    "name",
    [
        "input_snapshot_count",
        "total_rule_evaluation_count",
        "not_fired_count",
        "disabled_rule_count",
        "grouped_detection_count",
        "suppressed_by_cooldown_count",
        "suppressed_by_rate_limit_count",
        "suppressed_total",
        "escalated_count",
        "below_score_floor_count",
        "below_severity_floor_count",
        "scope_missing_count",
    ],
)
def test_a_reconstructed_report_reports_null_not_zero(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int], name: str
) -> None:
    report = reconstructed(pipeline)
    assert getattr(report, name) is None, name


def test_the_unavailable_set_is_exactly_what_the_report_declares(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """The declared list and the actual nulls cannot drift apart."""
    from password_attack_detector.detection.quality import (
        UNAVAILABLE_WHEN_RECONSTRUCTED,
    )

    report = reconstructed(pipeline)
    assert report.unavailable_metrics == list(UNAVAILABLE_WHEN_RECONSTRUCTED)
    actually_null = {
        name for name in UNAVAILABLE_WHEN_RECONSTRUCTED if getattr(report, name) is None
    }
    assert actually_null == set(UNAVAILABLE_WHEN_RECONSTRUCTED)


def test_derivable_counts_stay_exact_under_reconstruction(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """What the tables do record must match the live run exactly."""
    live = build_report(pipeline)
    rebuilt = reconstructed(pipeline)

    assert rebuilt.evaluated_snapshot_count == live.evaluated_snapshot_count
    assert rebuilt.fired_count == live.fired_count
    assert rebuilt.insufficient_data_count == live.insufficient_data_count
    assert rebuilt.zero_score_assessment_count == live.zero_score_assessment_count
    assert rebuilt.alert_count == live.alert_count
    assert rebuilt.per_rule_trigger_counts == live.per_rule_trigger_counts
    assert rebuilt.per_family_trigger_counts == live.per_family_trigger_counts
    assert rebuilt.attack_category_distribution == live.attack_category_distribution
    assert rebuilt.severity_distribution == live.severity_distribution
    assert rebuilt.risk_score_summary == live.risk_score_summary
    assert rebuilt.entity_scoped_count == live.entity_scoped_count
    assert rebuilt.category_scoped_count == live.category_scoped_count
    assert rebuilt.average_events_per_alert == live.average_events_per_alert
    assert rebuilt.grouping_mode_counts == live.grouping_mode_counts


def test_a_rule_that_never_fired_is_a_measured_zero_trigger_count(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    """The detection table proves a rule triggered zero times."""
    report = reconstructed(pipeline)
    silent = [
        rule_id
        for rule_id, count in report.per_rule_trigger_counts.items()
        if count == 0
    ]
    assert silent
    assert set(report.per_rule_trigger_counts) == set(CONFIG.enabled_rule_ids)


def test_the_reconstruction_is_recorded_as_the_report_source(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    from password_attack_detector.detection.quality import (
        RECONSTRUCTED_WARNING_CODE,
        RECONSTRUCTION_NOTE,
        QualityReportSource,
    )

    report = reconstructed(pipeline)
    assert report.report_source is QualityReportSource.PUBLISHED_ARTIFACTS
    assert report.is_reconstructed is True
    assert report.reconstruction_note == RECONSTRUCTION_NOTE
    assert RECONSTRUCTED_WARNING_CODE in report.warning_summary


def test_the_json_report_uses_null_for_unavailable_counters(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    from password_attack_detector.detection.quality import (
        UNAVAILABLE_WHEN_RECONSTRUCTED,
    )

    payload = json.loads(report_to_json(reconstructed(pipeline)))
    for name in UNAVAILABLE_WHEN_RECONSTRUCTED:
        assert payload[name] is None, name
    assert payload["report_source"] == "published_artifacts"
    assert payload["reconstruction_note"]
    assert payload["unavailable_metrics"] == list(UNAVAILABLE_WHEN_RECONSTRUCTED)
    # Derivable counts stay integers in the same payload.
    assert isinstance(payload["fired_count"], int)
    assert isinstance(payload["evaluated_snapshot_count"], int)


def test_the_markdown_report_says_unavailable_rather_than_zero(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    from password_attack_detector.detection.quality import (
        RECONSTRUCTED_WARNING_CODE,
        UNAVAILABLE_LABEL,
    )

    markdown = report_to_markdown(reconstructed(pipeline))
    assert RECONSTRUCTED_WARNING_CODE in markdown
    assert "reconstructed from published artifacts" in markdown
    assert f"| Not fired | {UNAVAILABLE_LABEL} |" in markdown
    assert f"| Suppressed total | {UNAVAILABLE_LABEL} |" in markdown
    assert f"| Rule evaluations | {UNAVAILABLE_LABEL} |" in markdown
    # And a derivable count is still a number in the same table.
    assert "| Fired | " in markdown
    assert f"| Fired | {UNAVAILABLE_LABEL} |" not in markdown


def test_both_formats_name_every_unavailable_counter(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    from password_attack_detector.detection.quality import (
        UNAVAILABLE_WHEN_RECONSTRUCTED,
    )

    report = reconstructed(pipeline)
    markdown = report_to_markdown(report)
    for name in UNAVAILABLE_WHEN_RECONSTRUCTED:
        assert f"`{name}`" in markdown, name


def test_a_reconstructed_report_carries_no_identifier() -> None:
    """The privacy sweep applies to the reconstructed form too."""
    from password_attack_detector.detection.alerts import build_entity_scope_table
    from password_attack_detector.detection.quality import (
        reconstruct_detection_quality_report,
    )

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
    scope = build_entity_scope_table(
        [factories.scope_record(row["anchor_event_id"], user="ab" * 16) for row in rows]
    )
    alerting_result = AlertBuilder(CONFIG).build(
        scoring_result.assessments, entity_scope=scope
    )
    report = reconstruct_detection_quality_report(
        detections=list(engine_result.fired_detections),
        assessments=list(scoring_result.assessments),
        alerts=list(alerting_result.alerts),
        config=CONFIG,
    )
    for rendered in (report_to_json(report), report_to_markdown(report)):
        assert not _UUID_RE.search(rendered)
        assert not _PSEUDONYM_RE.search(rendered)
        assert "ab" * 16 not in rendered
        assert "/home/" not in rendered


def test_reconstruction_of_an_empty_artifact_set_is_well_formed() -> None:
    from password_attack_detector.detection.quality import (
        reconstruct_detection_quality_report,
    )

    report = reconstruct_detection_quality_report(
        detections=[], assessments=[], alerts=[], config=CONFIG
    )
    assert report.evaluated_snapshot_count == 0
    assert report.fired_count == 0
    assert report.alert_count == 0
    assert report.not_fired_count is None
    assert report.average_events_per_alert is None
    assert report_to_markdown(report).startswith("# Detection Quality Report")


def test_reconstruction_carries_the_validation_result_through(
    pipeline: tuple[EngineResult, ScoringResult, AlertingResult, int],
) -> None:
    engine_result, scoring_result, alerting_result, _ = pipeline
    validation = DetectionValidator(CONFIG).validate(
        list(engine_result.fired_detections),
        list(scoring_result.assessments),
        list(alerting_result.alerts),
    )
    report = reconstructed(pipeline, validation_result=validation)
    assert report.validation_status == str(validation.status)
    assert report.warning_summary[0] == "Q001"
    for finding in validation.warnings:
        assert finding.code in report.warning_summary
