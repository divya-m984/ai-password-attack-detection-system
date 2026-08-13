"""The aggregate profile, and the line it is not allowed to cross.

Half the file is arithmetic: counts sum, rates are computed over the right
denominators, and a quantity nothing could produce is ``None`` rather than
``0.0``. The other half is prohibition -- no field, no rendered line, and no
JSON key may carry a figure that would have required a label to compute, because
the whole publication was produced without opening one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.ml.enums import UNKNOWN_CATEGORY, AuditStatus, MLSplit
from password_attack_detector.ml.prediction_manifest import (
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    PredictionManifest,
)
from password_attack_detector.ml.prediction_publisher import publish_predictions
from password_attack_detector.ml.prediction_serialization import (
    read_binary_predictions,
    read_category_predictions,
)
from password_attack_detector.ml.prediction_validation import validate_publication
from password_attack_detector.ml.quality import (
    QUALITY_SCHEMA_VERSION,
    REPORTED_QUANTILES,
    AnomalyDistribution,
    BinaryDistribution,
    CategoryClassCount,
    CategoryDistribution,
    MLQualityReport,
    build_quality_report,
    quality_report_to_markdown,
)
from tests.ml import predictions as px
from tests.ml import runs as rx
from tests.unit.ml.test_prediction_serialization import (
    CLASS_ORDER,
    anomaly_rows,
    binary_rows,
    category_rows,
)

#: Every figure a report is forbidden to carry, in the spellings a reader would
#: search for. Asserted against the fields, the JSON, and the Markdown.
OUTCOME_TERMS = (
    "accuracy",
    "precision_against_truth",
    "recall_against_truth",
    "f1_score",
    "false_positive_rate",
    "true_positive",
    "false_negative",
    "pr_auc",
    "roc_auc",
    "brier",
    "expected_calibration_error",
    "confusion_matrix",
)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> rx.Experiment:
    """Publish one experiment, shared by every test in this module."""
    return rx.publish_experiment(tmp_path_factory.mktemp("quality-source"))


@pytest.fixture
def published(source: rx.Experiment, tmp_path: Path) -> tuple[Path, MLQualityReport]:
    """Return one published prediction directory and its rebuilt report."""
    prepared = px.prepare(tmp_path / "root", source=source)
    publication = publish_predictions(
        champion=prepared.champion(),
        dataset=px.inference_dataset(prepared),
        root=prepared.root,
    )
    directory = prepared.root / PREDICTIONS_DIR / publication.prediction_id
    return (directory, rebuild(directory))


def rebuild(directory: Path) -> MLQualityReport:
    """Rebuild the report from the publication alone, as a later reader would."""
    manifest = PredictionManifest.from_json(
        (directory / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    declared = {item.logical_name for item in manifest.files}
    return build_quality_report(
        manifest=manifest,
        validation=validate_publication(directory),
        binary=read_binary_predictions(directory / "binary_predictions.parquet"),
        category=(
            read_category_predictions(directory / "category_predictions.parquet")
            if "category_predictions.parquet" in declared
            else None
        ),
        anomaly=None,
    )


def binary_distribution(**overrides: Any) -> BinaryDistribution:
    """Return a valid uncalibrated binary distribution."""
    fields: dict[str, Any] = {
        "total_rows": 10,
        "flagged_count": 4,
        "flagged_rate": 0.4,
        "unflagged_count": 6,
        "unflagged_rate": 0.6,
        "score_kind": "decision_score",
        "decision_threshold": 0.2,
        "decision_score_available": True,
        "decision_score_minimum": 0.0,
        "decision_score_maximum": 0.9,
        "decision_score_mean": 0.45,
        "decision_score_quantiles": None,
        "calibrated_probability_available": False,
        "probability_minimum": None,
        "probability_maximum": None,
        "probability_mean": None,
        "probability_quantiles": None,
        "null_probability_count": 10,
    }
    fields.update(overrides)
    return BinaryDistribution(**fields)


# ---------------------------------------------------------------------------
# What the report says
# ---------------------------------------------------------------------------


def test_the_report_is_rebuildable_from_the_publication_alone(
    published: tuple[Path, MLQualityReport],
) -> None:
    """No training data, no labels, no model directory -- just the artifacts."""
    directory, report = published
    assert rebuild(directory).to_json() == report.to_json()


def test_the_counts_sum_and_the_rates_follow(
    published: tuple[Path, MLQualityReport],
) -> None:
    """Arithmetic over the published rows, not a summary of a summary."""
    _, report = published
    binary = report.binary
    assert binary.flagged_count + binary.unflagged_count == binary.total_rows
    assert binary.flagged_rate == pytest.approx(
        binary.flagged_count / binary.total_rows, abs=1e-9
    )


def test_the_report_records_what_validation_found(
    published: tuple[Path, MLQualityReport],
) -> None:
    """A profile is never a clean bill of health for a failing artifact."""
    _, report = published
    assert report.validation_status is AuditStatus.PASS
    assert report.validation_failures == ()
    assert report.validation_check_count >= 20


def test_the_report_names_the_publication_it_profiles(
    published: tuple[Path, MLQualityReport],
) -> None:
    """Bound to the identity and the content, so the two cannot be paired wrongly."""
    directory, report = published
    manifest = PredictionManifest.from_json(
        (directory / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert report.prediction_id == manifest.prediction_id
    assert report.prediction_content_fingerprint == (
        manifest.prediction_content_fingerprint
    )


def test_the_category_denominator_is_the_applicable_population(
    published: tuple[Path, MLQualityReport],
) -> None:
    """Property E: rates are over binary-positive rows, never over the table.

    Both populations are reported, and the two absences stay apart: a row the
    binary head never flagged is *not applicable*, and a row it flagged whose
    best class score fell short is ``unknown``.
    """
    _, report = published
    category = report.category
    binary = report.binary
    assert category is not None
    assert category.binary_row_count == binary.total_rows
    assert category.applicable_row_count == binary.flagged_count
    assert category.not_applicable_count == binary.unflagged_count
    assert category.not_applicable_count > 0, "the fixture must leave rows unflagged"
    assert category.known_count + category.unknown_count == (
        category.applicable_row_count
    )
    assert sum(item.predicted_count for item in category.class_counts) == (
        category.known_count
    )
    assert category.unknown_rate == pytest.approx(
        category.unknown_count / category.applicable_row_count, abs=1e-9
    )


def test_an_unflagged_row_is_absent_from_the_category_artifact(
    published: tuple[Path, MLQualityReport],
) -> None:
    """The table's membership is what records applicability."""
    directory, report = published
    binary = read_binary_predictions(directory / "binary_predictions.parquet")
    category = read_category_predictions(directory / "category_predictions.parquet")
    flagged = {row.anchor_event_id for row in binary if row.flagged_malicious}
    assert {row.anchor_event_id for row in category} == flagged
    assert report.category is not None
    assert report.category.applicable_row_count == len(flagged)


def test_zero_applicable_rows_report_no_rate_rather_than_zero() -> None:
    """Nothing was routed to triage, so nothing abstained."""
    distribution = CategoryDistribution(
        binary_row_count=12,
        applicable_row_count=0,
        not_applicable_count=12,
        known_count=0,
        known_rate=None,
        unknown_count=0,
        unknown_rate=None,
        min_category_score=0.35,
        class_order=CLASS_ORDER,
        class_counts=tuple(
            CategoryClassCount(class_name=name, predicted_count=0)
            for name in CLASS_ORDER
        ),
        max_score_minimum=None,
        max_score_maximum=None,
        max_score_mean=None,
    )
    assert distribution.unknown_rate is None
    assert distribution.known_rate is None
    assert distribution.min_category_score == 0.35


def test_a_distribution_that_counts_unflagged_rows_as_abstentions_is_refused() -> None:
    """The two absences cannot be collapsed into one."""
    with pytest.raises(ValueError, match="is not an abstention"):
        CategoryDistribution(
            binary_row_count=10,
            applicable_row_count=4,
            not_applicable_count=6,
            known_count=2,
            known_rate=0.5,
            unknown_count=8,
            unknown_rate=2.0,
            min_category_score=0.3,
            class_order=CLASS_ORDER,
            class_counts=(
                CategoryClassCount(class_name=CLASS_ORDER[0], predicted_count=2),
                CategoryClassCount(class_name=CLASS_ORDER[1], predicted_count=0),
            ),
            max_score_minimum=0.1,
            max_score_maximum=0.9,
            max_score_mean=0.5,
        )


def test_populations_that_do_not_sum_are_refused() -> None:
    """Applicable plus not-applicable is every row that was scored."""
    with pytest.raises(ValueError, match="do not sum to the rows scored"):
        CategoryDistribution(
            binary_row_count=10,
            applicable_row_count=4,
            not_applicable_count=5,
            known_count=4,
            known_rate=1.0,
            unknown_count=0,
            unknown_rate=0.0,
            min_category_score=0.3,
            class_order=CLASS_ORDER,
            class_counts=(
                CategoryClassCount(class_name=CLASS_ORDER[0], predicted_count=4),
                CategoryClassCount(class_name=CLASS_ORDER[1], predicted_count=0),
            ),
            max_score_minimum=0.1,
            max_score_maximum=0.9,
            max_score_mean=0.5,
        )


# ---------------------------------------------------------------------------
# Unavailable is not zero
# ---------------------------------------------------------------------------


def test_an_uncalibrated_publication_reports_no_probability_statistic() -> None:
    """A mean of zero would describe a model certain every row was benign."""
    rows = binary_rows(6)
    distribution = build_quality_report(
        manifest=_manifest_stub(rows),
        validation=_validation_stub(),
        binary=rows,
        category=None,
        anomaly=None,
    ).binary
    assert distribution.calibrated_probability_available is False
    assert distribution.probability_minimum is None
    assert distribution.probability_mean is None
    assert distribution.probability_quantiles is None
    assert distribution.null_probability_count == len(rows)


def test_a_calibrated_publication_reports_the_probability_distribution() -> None:
    """The positive case, so the nulls above mean something."""
    rows = binary_rows(6, calibrated=True)
    distribution = build_quality_report(
        manifest=_manifest_stub(rows),
        validation=_validation_stub(),
        binary=rows,
        category=None,
        anomaly=None,
    ).binary
    assert distribution.calibrated_probability_available is True
    assert distribution.probability_minimum is not None
    assert distribution.null_probability_count == 0


def test_a_probability_statistic_without_a_calibrator_is_refused() -> None:
    """The absence is structural, not a rendering convention."""
    with pytest.raises(ValueError, match="not a distribution of zeros"):
        binary_distribution(probability_mean=0.0)


def test_a_partially_populated_probability_column_is_refused() -> None:
    """A calibrator that failed on some rows, with nothing saying so."""
    with pytest.raises(ValueError, match="carries a probability on every row"):
        binary_distribution(
            score_kind="calibrated_probability",
            calibrated_probability_available=True,
            probability_minimum=0.1,
            probability_maximum=0.9,
            probability_mean=0.5,
            null_probability_count=3,
        )


def test_counts_that_do_not_sum_are_refused() -> None:
    """A distribution that lost a row is not a distribution."""
    with pytest.raises(ValueError, match="do not sum"):
        binary_distribution(flagged_count=4, unflagged_count=3)


def test_a_class_count_outside_the_declared_order_is_refused() -> None:
    """Every declared class, and nothing else."""
    with pytest.raises(ValueError, match="in the frozen class order"):
        CategoryDistribution(
            binary_row_count=2,
            applicable_row_count=2,
            not_applicable_count=0,
            known_count=2,
            known_rate=1.0,
            unknown_count=0,
            unknown_rate=0.0,
            min_category_score=0.3,
            class_order=CLASS_ORDER,
            class_counts=(
                CategoryClassCount(class_name="something_else", predicted_count=2),
            ),
            max_score_minimum=0.4,
            max_score_maximum=0.9,
            max_score_mean=0.65,
        )


def test_the_abstention_label_is_not_a_declared_class() -> None:
    """``unknown`` is counted separately, never as a class the head predicts."""
    with pytest.raises(ValueError, match="abstention outcome"):
        CategoryDistribution(
            binary_row_count=1,
            applicable_row_count=1,
            not_applicable_count=0,
            known_count=1,
            known_rate=1.0,
            unknown_count=0,
            unknown_rate=0.0,
            min_category_score=0.3,
            class_order=(UNKNOWN_CATEGORY, "brute_force"),
            class_counts=(
                CategoryClassCount(class_name=UNKNOWN_CATEGORY, predicted_count=0),
                CategoryClassCount(class_name="brute_force", predicted_count=1),
            ),
            max_score_minimum=0.4,
            max_score_maximum=0.4,
            max_score_mean=0.4,
        )


def test_an_anomaly_flag_count_needs_a_threshold() -> None:
    """A count without the threshold that produced it is uninterpretable."""
    with pytest.raises(ValueError, match="reported together"):
        AnomalyDistribution(
            total_rows=3,
            score_minimum=-0.5,
            score_maximum=-0.3,
            score_mean=-0.4,
            score_quantiles=None,
            anomaly_threshold=None,
            flagged_count=2,
            flagged_rate=0.66,
        )


def test_an_anomaly_distribution_is_permanently_experimental() -> None:
    """It cannot claim influence over anything."""
    with pytest.raises(ValueError, match="never influences champion selection"):
        AnomalyDistribution(
            total_rows=3,
            score_minimum=-0.5,
            score_maximum=-0.3,
            score_mean=-0.4,
            score_quantiles=None,
            anomaly_threshold=None,
            influences_champion_selection=True,
        )


def test_an_anomaly_distribution_summarises_the_magnitudes() -> None:
    """A range, a mean, quantiles, and a flag count where a threshold exists."""
    rows = anomaly_rows(5)
    report = build_quality_report(
        manifest=_manifest_stub(binary_rows(5)),
        validation=_validation_stub(),
        binary=binary_rows(5),
        category=None,
        anomaly=rows,
    )
    assert report.anomaly is not None
    assert report.anomaly.total_rows == 5
    assert report.anomaly.anomaly_threshold == rows[0].anomaly_threshold
    assert report.anomaly.flagged_count == sum(
        1 for row in rows if row.flagged_anomalous
    )
    assert report.anomaly.experimental is True


# ---------------------------------------------------------------------------
# Determinism and rendering
# ---------------------------------------------------------------------------


def test_the_json_rendering_is_deterministic(
    published: tuple[Path, MLQualityReport],
) -> None:
    """Two renderings of one report are the same bytes."""
    _, report = published
    assert json.dumps(report.to_dict(), sort_keys=True) == json.dumps(
        rebuild(published[0]).to_dict(), sort_keys=True
    )


def test_the_markdown_rendering_is_deterministic(
    published: tuple[Path, MLQualityReport],
) -> None:
    """Including across two rebuilds of the same publication."""
    directory, report = published
    assert quality_report_to_markdown(report) == quality_report_to_markdown(
        rebuild(directory)
    )


def test_the_markdown_names_no_anchor(
    published: tuple[Path, MLQualityReport],
) -> None:
    """A profile is aggregate; the join keys stay in the Parquet."""
    directory, report = published
    rendered = quality_report_to_markdown(report)
    rows = read_binary_predictions(directory / "binary_predictions.parquet")
    for row in rows:
        assert row.anchor_event_id not in rendered


def test_the_markdown_says_what_it_is_not(
    published: tuple[Path, MLQualityReport],
) -> None:
    """The distinction between structural validity and predictive quality."""
    _, report = published
    rendered = quality_report_to_markdown(report)
    assert "What this report is not" in rendered
    assert "Structural validity is not predictive quality" in rendered
    assert "No label was read" in rendered
    assert "synthetic" in rendered


def test_an_unavailable_number_renders_as_unavailable() -> None:
    """Never as a zero, in the JSON or in the Markdown."""
    rows = binary_rows(4)
    report = build_quality_report(
        manifest=_manifest_stub(rows),
        validation=_validation_stub(),
        binary=rows,
        category=None,
        anomaly=None,
    )
    rendered = quality_report_to_markdown(report)
    assert "| Probability mean | unavailable |" in rendered
    assert report.to_dict()["binary"]["probability_mean"] is None


# ---------------------------------------------------------------------------
# What a profile must never contain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term", OUTCOME_TERMS)
def test_no_schema_declares_an_outcome_field(term: str) -> None:
    """Each requires a label, and no label was read."""
    models = (
        MLQualityReport,
        BinaryDistribution,
        CategoryDistribution,
        CategoryClassCount,
        AnomalyDistribution,
    )
    for model in models:
        assert not any(term in name for name in model.model_fields), (
            model.__name__,
            term,
        )


@pytest.mark.parametrize("term", OUTCOME_TERMS)
def test_no_rendering_mentions_an_outcome_figure(
    published: tuple[Path, MLQualityReport], term: str
) -> None:
    """Except in the paragraph that says none of them is computable."""
    _, report = published
    payload = json.dumps(report.to_dict()).lower()
    assert term not in payload, term
    body = quality_report_to_markdown(report).lower()
    head, _, disclaimer = body.partition("## what this report is not")
    assert term not in head, term
    assert disclaimer


def test_the_quantile_set_is_declared_and_fixed() -> None:
    """Two reports over different data remain comparable."""
    assert REPORTED_QUANTILES == (0.05, 0.25, 0.50, 0.75, 0.95)


def test_the_quality_contract_version_is_pinned() -> None:
    """A change to what a profile reports is a visible edit."""
    assert QUALITY_SCHEMA_VERSION == "1.0.0"


def test_a_failing_validation_is_carried_into_the_report() -> None:
    """A profile of a broken artifact says so rather than describing it."""
    with pytest.raises(ValueError, match="names no failing check"):
        MLQualityReport.seal(
            prediction_id="00000000-0000-5000-8000-000000000001",
            prediction_content_fingerprint="a" * 64,
            scope=MLSplit.TEST,
            scope_role=_scope_role(),
            binary=binary_distribution(),
            category=None,
            anomaly=None,
            validation_status=AuditStatus.PASS,
            validation_failures=("M009",),
            validation_check_count=28,
        )


# ---------------------------------------------------------------------------
# Small stubs
# ---------------------------------------------------------------------------


def _scope_role() -> Any:
    """Return the supervised scope role."""
    from password_attack_detector.ml.prediction_manifest import scope_role_for

    return scope_role_for(MLSplit.TEST)


def _manifest_stub(rows: Any) -> Any:
    """Return a manifest over *rows*, built by the manifest suite's own factory."""
    from tests.unit.ml.test_prediction_manifest import manifest

    return manifest(list(rows))


def _validation_stub() -> Any:
    """Return a passing staged validation result over the same rows."""
    from password_attack_detector.ml.prediction_validation import (
        validate_staged_predictions,
    )

    return validate_staged_predictions(
        binary=binary_rows(1),
        category=None,
        anomaly=None,
        scope=MLSplit.TEST,
        binary_schema_matches=True,
    )


def test_the_category_rows_helper_is_shared_not_duplicated() -> None:
    """The suites agree on what a category prediction looks like."""
    assert len(category_rows(2)) == 2
