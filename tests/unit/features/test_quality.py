"""Tests for aggregate feature quality reporting."""

from __future__ import annotations

import json
import random
import re
from datetime import timedelta

import pytest

from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.features.catalog import build_catalog
from password_attack_detector.features.config import (
    BaselineConfig,
    FeatureConfig,
    SplitConfig,
)
from password_attack_detector.features.engine import FeatureEngine, FeatureFrame
from password_attack_detector.features.quality import (
    FeatureQualityReport,
    generate_feature_quality_report,
    report_to_json,
    report_to_markdown,
)
from password_attack_detector.features.serialization import (
    compute_features_fingerprint,
)
from password_attack_detector.features.splitting import ChronologicalSplitter
from tests.features.factories import make_event, make_labels

_CONFIG = FeatureConfig(
    windows=("1m", "5m"),
    cardinality_windows=("5m",),
    dispersion_windows=("5m",),
    device_session_windows=("5m",),
    pair_windows=("5m",),
    baseline=BaselineConfig(rate_reference_window="5m"),
    split=SplitConfig(purge=timedelta(minutes=5), max_excluded_fraction=0.5),
)

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)


def _events(count: int = 50) -> list[AuthEvent]:
    rng = random.Random(13579)
    return [
        make_event(
            t=float(index) * 120.0,
            user=f"u{rng.randint(1, 3)}",
            source=f"s{rng.randint(1, 2)}",
            outcome=rng.choice(["success", "failure"]),
            response_time_ms=rng.choice([None, 60, 240]),
            country=rng.choice([None, "US", "GB"]),
            latitude=rng.choice([None, 37.8, 51.5]),
            longitude=rng.choice([None, -122.4, -0.1]),
            key=str(index),
        )
        for index in range(count)
    ]


def _frame(events: list[AuthEvent] | None = None) -> FeatureFrame:
    resolved = _events() if events is None else events
    return FeatureEngine(_CONFIG, build_catalog(_CONFIG)).run(resolved)


@pytest.fixture()
def report() -> FeatureQualityReport:
    events = _events()
    frame = _frame(events)
    labels = make_labels(events)
    result = ChronologicalSplitter(_CONFIG.split).split(events, labels)
    return generate_feature_quality_report(
        frame,
        feature_fingerprint=compute_features_fingerprint(frame),
        assignments=list(result.assignments),
        class_distribution_by_split=result.manifest.class_distribution,
        exclusion_counts=result.manifest.exclusion_counts,
    )


# --- content ---------------------------------------------------------------


class TestReportContent:
    def test_reports_the_shape(self, report: FeatureQualityReport) -> None:
        assert report.row_count == 50
        assert report.feature_count > 50

    def test_reports_the_temporal_range(self, report: FeatureQualityReport) -> None:
        assert report.earliest_anchor_time is not None
        assert report.latest_anchor_time is not None
        assert report.earliest_anchor_time < report.latest_anchor_time

    def test_reports_feature_group_counts(self, report: FeatureQualityReport) -> None:
        assert sum(report.feature_group_counts.values()) == report.feature_count

    def test_reports_leakage_class_counts(self, report: FeatureQualityReport) -> None:
        assert sum(report.leakage_class_counts.values()) == report.feature_count

    def test_reports_null_rates_for_every_column(
        self, report: FeatureQualityReport
    ) -> None:
        assert len(report.null_rates) == report.feature_count
        assert all(0.0 <= rate <= 1.0 for rate in report.null_rates.values())

    def test_reports_numeric_ranges(self, report: FeatureQualityReport) -> None:
        summary = report.numeric_ranges["user_attempt_count__5m"]
        assert summary["min"] <= summary["mean"] <= summary["max"]

    def test_counts_infinite_values(self, report: FeatureQualityReport) -> None:
        assert report.infinite_value_count == 0

    def test_identifies_zero_variance_features(
        self, report: FeatureQualityReport
    ) -> None:
        # device_id is mandatory in the canonical schema, so current_has_device
        # is constant.  Reporting it is the point; removing it is a later
        # modelling decision.
        assert "current_has_device" in report.zero_variance_features

    def test_reports_baseline_coverage(self, report: FeatureQualityReport) -> None:
        assert set(report.baseline_coverage) == {
            "user_in_baseline",
            "source_in_baseline",
        }

    def test_reports_novelty_rates(self, report: FeatureQualityReport) -> None:
        assert "is_new_device_for_user" in report.novelty_rates

    def test_reports_geospatial_status_rates(
        self, report: FeatureQualityReport
    ) -> None:
        rates = report.geospatial_status_rates["user_previous_success_geo__status"]
        assert rates
        assert sum(rates.values()) == pytest.approx(1.0)

    def test_reports_split_sizes(self, report: FeatureQualityReport) -> None:
        assert report.split_counts["train"] > 0

    def test_reports_class_distribution_by_split(
        self, report: FeatureQualityReport
    ) -> None:
        assert "train" in report.class_distribution_by_split

    def test_records_all_three_fingerprints(self, report: FeatureQualityReport) -> None:
        for fingerprint in (
            report.feature_fingerprint,
            report.config_fingerprint,
            report.catalog_fingerprint,
        ):
            assert len(fingerprint) == 64

    def test_no_feature_importance_is_calculated(
        self, report: FeatureQualityReport
    ) -> None:
        # This phase trains no model, so nothing can rank features.
        payload = report.to_dict()
        for key in payload:
            assert "importance" not in key
            assert "shap" not in key


# --- rendering -------------------------------------------------------------


class TestJsonRendering:
    def test_is_valid_json(self, report: FeatureQualityReport) -> None:
        payload = json.loads(report_to_json(report))
        assert payload["row_count"] == 50

    def test_is_deterministic(self, report: FeatureQualityReport) -> None:
        assert report_to_json(report) == report_to_json(report)

    def test_keys_are_sorted(self, report: FeatureQualityReport) -> None:
        payload = json.loads(report_to_json(report))
        assert list(payload) == sorted(payload)


class TestMarkdownRendering:
    @pytest.fixture()
    def markdown(self, report: FeatureQualityReport) -> str:
        return report_to_markdown(report)

    def test_has_a_title(self, markdown: str) -> None:
        assert markdown.startswith("# Feature Quality Report")

    def test_uses_metric_value_tables(self, markdown: str) -> None:
        assert "| Metric | Value |" in markdown
        assert "|--------|-------|" in markdown

    def test_reports_the_row_count(self, markdown: str) -> None:
        assert "| Feature snapshots | 50 |" in markdown

    def test_includes_the_fingerprints(
        self, markdown: str, report: FeatureQualityReport
    ) -> None:
        assert report.feature_fingerprint in markdown
        assert report.catalog_fingerprint in markdown

    def test_includes_split_sizes(self, markdown: str) -> None:
        assert "## Split sizes" in markdown

    def test_ends_with_known_limitations(self, markdown: str) -> None:
        assert "## Known limitations" in markdown

    def test_states_that_no_model_is_trained(self, markdown: str) -> None:
        assert "trains no model" in markdown

    def test_makes_no_effectiveness_claim(self, markdown: str) -> None:
        assert "nothing about detection effectiveness" in markdown

    def test_is_deterministic(self, report: FeatureQualityReport) -> None:
        assert report_to_markdown(report) == report_to_markdown(report)


# --- privacy ---------------------------------------------------------------


class TestReportPrivacy:
    @pytest.fixture()
    def rendered(self, report: FeatureQualityReport) -> str:
        return report_to_json(report) + report_to_markdown(report)

    def test_contains_no_event_identifiers(self, rendered: str) -> None:
        assert not _UUID_RE.search(rendered)

    def test_contains_no_pseudonyms(self, rendered: str) -> None:
        assert not _PSEUDONYM_RE.search(rendered)

    def test_contains_no_coordinates(self, rendered: str) -> None:
        for coordinate in ("37.8", "51.5", "-122.4"):
            assert coordinate not in rendered

    def test_contains_no_absolute_paths(self, rendered: str) -> None:
        assert "/home/" not in rendered
        assert "/tmp/" not in rendered

    def test_contains_no_event_rows(self, report: FeatureQualityReport) -> None:
        # Only aggregates: no key in the payload holds a per-row list.
        payload = report.to_dict()
        for key, value in payload.items():
            if isinstance(value, list):
                assert len(value) < report.row_count, key


# --- edge cases ------------------------------------------------------------


class TestEdgeCases:
    def test_a_single_row_dataset_profiles(self) -> None:
        frame = _frame([make_event(t=0.0)])
        report = generate_feature_quality_report(frame, feature_fingerprint="0" * 64)
        assert report.row_count == 1

    def test_missing_split_information_is_tolerated(self) -> None:
        frame = _frame()
        report = generate_feature_quality_report(frame, feature_fingerprint="0" * 64)
        assert report.split_counts == {}
        assert report.class_distribution_by_split == {}

    def test_markdown_omits_absent_sections(self) -> None:
        frame = _frame()
        markdown = report_to_markdown(
            generate_feature_quality_report(frame, feature_fingerprint="0" * 64)
        )
        assert "## Split sizes" not in markdown
        assert "## Known limitations" in markdown
