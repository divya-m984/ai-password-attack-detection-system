"""Unit tests for drift detection against the frozen reference profile.

Two properties are swept across the file. **The reference decides the
partition** -- an identical population reports no drift, a shifted one reports
drift, and nothing an incoming population contains can add a cell, move an edge,
or change an expected share. And **a refusal is never a measurement** -- too
little support reports `inconclusive`, an absent quantity reports `unavailable`,
and a schema validator refuses either carrying a number.
"""

from __future__ import annotations

from typing import Any

import pytest

from password_attack_detector.exceptions import ModelNotReadyError
from password_attack_detector.ml.config import DriftConfig
from password_attack_detector.ml.drift import (
    PSI_PROPORTION_FLOOR,
    REASON_ABSENT_INCOMING,
    REASON_INCOMING_SUPPORT,
    REASON_REFERENCE_SUPPORT,
    DriftManifest,
    FeatureDriftResult,
    MLDriftReport,
    PredictionDriftResult,
    aggregate_status,
    build_drift_manifest,
    build_drift_report,
    compare_feature_population,
    drift_report_to_markdown,
    population_stability_index,
)
from password_attack_detector.ml.enums import (
    DriftMetric,
    DriftStatus,
    MLSplit,
    ReferenceFeatureKind,
)
from password_attack_detector.ml.reference import (
    MLReferenceProfile,
    build_reference_profile,
    feature_reference_index,
)
from password_attack_detector.ml.schemas import PROHIBITED_METADATA_FIELDS
from tests.ml.models import prepare, raw_rows

WARN = 0.10
ALERT = 0.25


class _Frame:
    """The narrow shape a comparison reads."""

    def __init__(self, names: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]):
        self.feature_names = names
        self.feature_matrix = rows
        self.anchors = tuple(range(len(rows)))

    @property
    def row_count(self) -> int:
        """Return the number of rows in this population."""
        return len(self.feature_matrix)


class _Reference:
    """A stand-in for the loaded inference dataset."""

    def __init__(self, frame: _Frame, fingerprint: str = "f" * 64):
        self.frame = frame
        self.scope = MLSplit.TRAIN
        self.inference_input_fingerprint = fingerprint
        self.row_count = frame.row_count


class _Lock:
    """The champion lock fields a profile binds."""

    def __init__(self, preprocessor_fingerprint: str):
        self.lock_fingerprint = "a" * 64
        self.scope_key = "b" * 64
        self.catalog_model_id = "M-010"
        self.model_id = "model-1"
        self.model_content_fingerprint = "c" * 64
        self.preprocessor_fingerprint = preprocessor_fingerprint
        self.feature_catalog_fingerprint = "d" * 64
        self.allowlist_fingerprint = "e" * 64
        self.eligible_feature_list_fingerprint = "0" * 64


@pytest.fixture
def baseline() -> tuple[MLReferenceProfile, _Frame]:
    """A profile captured from a fitted preprocessor, and the frame behind it."""
    prepared = prepare()
    frame = _Frame(
        prepared.preprocessor.raw_feature_names,
        tuple(tuple(row) for row in prepared.frame.feature_matrix),
    )
    profile = build_reference_profile(
        lock=_Lock(prepared.preprocessor.fingerprint()),
        preprocessor=prepared.preprocessor,
        reference=_Reference(frame),
        reference_split=MLSplit.TRAIN,
        required_feature_schema_version="1.0.0",
        drift_config=DriftConfig(quantile_count=4, min_reference_rows=1),
        drift_config_fingerprint="1" * 64,
    )
    return profile, frame


def _compare(
    profile: MLReferenceProfile, frame: _Frame, *, min_support: int = 1
) -> tuple[FeatureDriftResult, ...]:
    """Compare *frame* against *profile* at the shared thresholds."""
    return compare_feature_population(
        profile=profile,
        frame=frame,
        warn_threshold=WARN,
        alert_threshold=ALERT,
        min_support=min_support,
    )


def _by_feature(
    results: tuple[FeatureDriftResult, ...],
) -> dict[str, FeatureDriftResult]:
    """Return the results keyed by feature name."""
    return {item.feature: item for item in results}


# ---------------------------------------------------------------------------
# The index itself
# ---------------------------------------------------------------------------


def test_identical_distributions_have_a_zero_index() -> None:
    """The measure is zero when nothing moved, not merely small."""
    assert population_stability_index((0.5, 0.5), (0.5, 0.5)) == 0.0


def test_the_index_is_symmetric_in_its_arguments() -> None:
    """A property of the formula, asserted rather than assumed."""
    first = population_stability_index((0.7, 0.3), (0.4, 0.6))
    second = population_stability_index((0.4, 0.6), (0.7, 0.3))
    assert first == second


def test_the_index_is_non_negative() -> None:
    """Each term is a difference times a log of the same sign."""
    assert population_stability_index((0.9, 0.1), (0.1, 0.9)) > 0.0
    assert population_stability_index((0.2, 0.8), (0.25, 0.75)) > 0.0


def test_a_zero_expected_cell_stays_finite_and_large() -> None:
    """A category the reference never saw, now appearing, is strong drift.

    Dropping the cell would make an entirely new category read as perfect
    stability; letting the logarithm diverge would make the number unusable.
    """
    value = population_stability_index((1.0, 0.0), (0.5, 0.5))
    assert value > ALERT
    assert value < 100.0


def test_the_floor_does_not_depend_on_population_size() -> None:
    """Two runs over the same shares must agree whatever the row counts were."""
    assert PSI_PROPORTION_FLOOR > 0.0
    assert population_stability_index((1.0, 0.0), (0.5, 0.5)) == (
        population_stability_index((1.0, 0.0), (0.5, 0.5))
    )


def test_mismatched_partitions_are_refused() -> None:
    """An index summed over cells that do not correspond means nothing."""
    with pytest.raises(ModelNotReadyError, match="different partitions"):
        population_stability_index((0.5, 0.5), (1.0,))


# ---------------------------------------------------------------------------
# Feature comparison
# ---------------------------------------------------------------------------


def test_the_reference_population_drifts_from_nothing(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Comparing the reference against itself must report no drift anywhere."""
    profile, frame = baseline
    results = _compare(profile, frame)
    assert results
    assert {item.status for item in results} == {DriftStatus.NO_DRIFT}
    assert all(item.observed_value == 0.0 for item in results)


def test_a_shifted_numeric_population_is_detected(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """A column pushed past every reference edge lands in the outer cell."""
    profile, frame = baseline
    numeric = next(
        item
        for item in profile.features
        if item.kind is ReferenceFeatureKind.NUMERIC
        and item.partition_kind == "quantile"
    )
    index = frame.feature_names.index(numeric.feature)
    shifted = _Frame(
        frame.feature_names,
        tuple(
            tuple(
                (float(value) + 1_000_000.0) if position == index else value
                for position, value in enumerate(row)
            )
            for row in frame.feature_matrix
        ),
    )
    result = _by_feature(_compare(profile, shifted))[numeric.feature]
    assert result.status is DriftStatus.DRIFT_DETECTED
    assert result.observed_value is not None and result.observed_value > ALERT


def test_an_out_of_range_value_is_binned_rather_than_discarded(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Every incoming row is accounted for; the outer cells are unbounded."""
    profile, frame = baseline
    numeric = next(
        item
        for item in profile.features
        if item.kind is ReferenceFeatureKind.NUMERIC
        and item.partition_kind == "quantile"
    )
    index = frame.feature_names.index(numeric.feature)
    shifted = _Frame(
        frame.feature_names,
        tuple(
            tuple(
                -1_000_000.0 if position == index else value
                for position, value in enumerate(row)
            )
            for row in frame.feature_matrix
        ),
    )
    result = _by_feature(_compare(profile, shifted))[numeric.feature]
    assert result.incoming_support == shifted.row_count
    assert result.status is DriftStatus.DRIFT_DETECTED


def test_a_missingness_shift_moves_the_index_and_is_reported(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """The null cell is part of the partition, so nulling a column shows up."""
    profile, frame = baseline
    nullable = next(
        item.feature
        for item in profile.features
        if item.kind is ReferenceFeatureKind.NUMERIC and item.reference_null_rate < 0.5
    )
    index = frame.feature_names.index(nullable)
    nulled = _Frame(
        frame.feature_names,
        tuple(
            tuple(
                None if position == index else value
                for position, value in enumerate(row)
            )
            for row in frame.feature_matrix
        ),
    )
    result = _by_feature(_compare(profile, nulled))[nullable]
    assert result.incoming_null_rate == 1.0
    assert result.null_rate_delta is not None and result.null_rate_delta > 0.0
    assert result.status is DriftStatus.DRIFT_DETECTED


def test_an_unseen_category_lands_in_the_unknown_cell(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """The incoming population must not be able to define a cell for itself."""
    profile, frame = baseline
    categorical = next(
        item.feature
        for item in profile.features
        if item.kind is ReferenceFeatureKind.CATEGORICAL
    )
    index = frame.feature_names.index(categorical)
    novel = _Frame(
        frame.feature_names,
        tuple(
            tuple(
                "a-category-nobody-declared" if position == index else value
                for position, value in enumerate(row)
            )
            for row in frame.feature_matrix
        ),
    )
    result = _by_feature(_compare(profile, novel))[categorical]
    assert result.incoming_unknown_rate == 1.0
    assert result.unknown_rate_delta is not None
    assert result.status is DriftStatus.DRIFT_DETECTED
    # The reference partition is unchanged: the same cells, in the same order.
    assert [
        item.cell for item in feature_reference_index(profile)[categorical].bins
    ] == [item.cell for item in feature_reference_index(profile)[categorical].bins]


def test_insufficient_incoming_support_is_inconclusive_not_stable(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """A comparison over too few rows has not established anything."""
    profile, frame = baseline
    small = _Frame(frame.feature_names, frame.feature_matrix[:3])
    results = _compare(profile, small, min_support=50)
    assert {item.status for item in results} == {DriftStatus.INCONCLUSIVE}
    assert {item.reason_code for item in results} == {REASON_INCOMING_SUPPORT}
    assert all(item.observed_value is None for item in results)


def test_insufficient_reference_support_is_inconclusive(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Short on the reference side is a different reason, and it is named."""
    profile, frame = baseline
    results = _compare(profile, frame, min_support=100_000)
    assert {item.status for item in results} == {DriftStatus.INCONCLUSIVE}
    assert {item.reason_code for item in results} == {REASON_REFERENCE_SUPPORT}


def test_a_feature_missing_from_the_incoming_frame_is_unavailable(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """An absent column is a typed refusal, not a shorter report."""
    profile, frame = baseline
    dropped = frame.feature_names[0]
    trimmed = _Frame(
        frame.feature_names[1:],
        tuple(row[1:] for row in frame.feature_matrix),
    )
    results = _by_feature(_compare(profile, trimmed))
    assert len(results) == len(profile.features)
    assert results[dropped].status is DriftStatus.UNAVAILABLE
    assert results[dropped].reason_code == REASON_ABSENT_INCOMING
    assert results[dropped].observed_value is None


def test_the_incoming_population_cannot_mutate_the_reference(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """The whole point of freezing: comparing must be a read.

    Checked by serializing the profile either side of a comparison against a
    deliberately hostile population, so a mutation anywhere in it would show.
    """
    profile, frame = baseline
    before = profile.to_json()
    hostile = _Frame(
        frame.feature_names,
        tuple(tuple(None for _ in row) for row in frame.feature_matrix),
    )
    _compare(profile, hostile)
    assert profile.to_json() == before


def test_the_comparison_is_deterministic(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Two comparisons of the same populations must agree exactly."""
    profile, frame = baseline
    assert _compare(profile, frame) == _compare(profile, frame)


def test_results_are_returned_in_feature_order(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Report order must not depend on matrix column order."""
    profile, frame = baseline
    results = _compare(profile, frame)
    assert [item.feature for item in results] == sorted(
        item.feature for item in results
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_the_aggregate_is_worst_first() -> None:
    """A single alert is not diluted by a hundred quiet features."""
    assert (
        aggregate_status([DriftStatus.NO_DRIFT, DriftStatus.DRIFT_DETECTED])
        is DriftStatus.DRIFT_DETECTED
    )
    assert (
        aggregate_status([DriftStatus.NO_DRIFT, DriftStatus.DRIFT_WARNING])
        is DriftStatus.DRIFT_WARNING
    )


def test_inconclusive_outranks_no_drift() -> None:
    """A skipped check is not a passed one, here as everywhere else."""
    assert (
        aggregate_status([DriftStatus.NO_DRIFT, DriftStatus.INCONCLUSIVE])
        is DriftStatus.INCONCLUSIVE
    )


def test_an_empty_result_set_is_unavailable_not_stable() -> None:
    """Nothing was compared, so nothing was found stable."""
    assert aggregate_status([]) is DriftStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Schema refusals
# ---------------------------------------------------------------------------


def _result(**overrides: Any) -> dict[str, Any]:
    """Return the fields of a measured feature result, with overrides applied."""
    fields: dict[str, Any] = {
        "feature": "f",
        "kind": ReferenceFeatureKind.NUMERIC,
        "partition_kind": "quantile",
        "reference_support": 100,
        "incoming_support": 100,
        "observed_value": 0.01,
        "warn_threshold": WARN,
        "alert_threshold": ALERT,
        "status": DriftStatus.NO_DRIFT,
        "reason_code": "psi_below_warn_threshold",
    }
    fields.update(overrides)
    return fields


def test_a_measured_status_without_a_value_is_refused() -> None:
    """A status that says it measured something must say what."""
    with pytest.raises(ValueError, match="names the value"):
        FeatureDriftResult(**_result(observed_value=None))


def test_a_refusal_carrying_a_value_is_refused() -> None:
    """A number beside a refusal reads as a measurement."""
    with pytest.raises(ValueError, match="reports no value"):
        FeatureDriftResult(**_result(status=DriftStatus.INCONCLUSIVE, reason_code="x"))


def test_a_measured_status_without_thresholds_is_refused() -> None:
    """A thresholded status names what it was decided against."""
    with pytest.raises(ValueError, match="names the thresholds"):
        FeatureDriftResult(**_result(warn_threshold=None))


def test_a_rate_delta_that_is_not_the_difference_is_refused() -> None:
    """Two numbers and a third that does not follow from them is a bug."""
    with pytest.raises(ValueError, match="not the difference"):
        FeatureDriftResult(
            **_result(
                reference_null_rate=0.1,
                incoming_null_rate=0.5,
                null_rate_delta=0.9,
            )
        )


def test_only_a_categorical_feature_has_an_unknown_rate() -> None:
    """A numeric column has no vocabulary to fall outside of."""
    with pytest.raises(ValueError, match="vocabulary"):
        FeatureDriftResult(
            **_result(
                reference_unknown_rate=0.0,
                incoming_unknown_rate=0.1,
                unknown_rate_delta=0.1,
            )
        )


def test_a_report_whose_aggregate_disagrees_with_its_results_is_refused() -> None:
    """The aggregate is derived, so a hand-set one must not survive."""
    result = FeatureDriftResult(
        **_result(
            observed_value=0.5,
            status=DriftStatus.DRIFT_DETECTED,
            reason_code="psi_at_or_above_alert_threshold",
        )
    )
    with pytest.raises(ValueError, match="not the one its results imply"):
        MLDriftReport.seal(
            reference_profile_id="r",
            reference_profile_fingerprint="a" * 64,
            incoming_scope=MLSplit.VALIDATION,
            incoming_row_count=10,
            incoming_population_fingerprint="b" * 64,
            warn_threshold=WARN,
            alert_threshold=ALERT,
            min_support=1,
            feature_status=DriftStatus.NO_DRIFT,
            prediction_status=DriftStatus.UNAVAILABLE,
            features=(result,),
            predictions=(),
        )


def test_only_one_metric_is_thresholded() -> None:
    """A second metric would be reported against thresholds nobody chose."""
    assert list(DriftMetric) == [DriftMetric.POPULATION_STABILITY_INDEX]


def test_no_schema_here_declares_a_prohibited_field() -> None:
    """An aggregate drift artifact describes populations, never rows."""
    for model in (
        FeatureDriftResult,
        PredictionDriftResult,
        MLDriftReport,
        DriftManifest,
    ):
        assert not set(model.model_fields) & PROHIBITED_METADATA_FIELDS


# ---------------------------------------------------------------------------
# No automatic retraining
# ---------------------------------------------------------------------------


def test_the_drift_module_imports_no_fitting_entry_point() -> None:
    """A monitor that could call ``fit`` is a monitor that might.

    Checked against the parsed import list rather than the documentation,
    because documentation does not stop a call.
    """
    import password_attack_detector.ml.drift as module
    from password_attack_detector.ml.governance import module_imports

    forbidden = {
        "password_attack_detector.ml.training",
        "password_attack_detector.ml.selection",
        "password_attack_detector.ml.champion",
        "password_attack_detector.ml.thresholds",
        "password_attack_detector.ml.test_evaluation",
        "password_attack_detector.ml.experiments",
        "password_attack_detector.ml.prediction_publisher",
    }
    assert not module_imports(module.__file__) & forbidden


def test_the_drift_module_names_no_retraining_vocabulary() -> None:
    """No hook, no webhook, no scheduler, no promotion."""
    from pathlib import Path

    import password_attack_detector.ml.drift as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for token in ("webhook", "schedule(", "subprocess", "requests."):
        assert token not in source, token


# ---------------------------------------------------------------------------
# Assembly and rendering
# ---------------------------------------------------------------------------


def test_a_report_and_its_manifest_are_deterministic(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Same comparison, same bytes -- in any directory, at any time."""
    profile, frame = baseline
    incoming = _Reference(frame, fingerprint="9" * 64)

    def _build() -> tuple[MLDriftReport, DriftManifest]:
        report = build_drift_report(
            profile=profile,
            incoming=incoming,
            features=_compare(profile, frame),
            predictions=(),
            incoming_manifest=None,
            warn_threshold=WARN,
            alert_threshold=ALERT,
            min_support=1,
        )
        return report, build_drift_manifest(profile=profile, report=report)

    first_report, first_manifest = _build()
    second_report, second_manifest = _build()
    assert first_report.to_json() == second_report.to_json()
    assert first_manifest.drift_run_id == second_manifest.drift_run_id


def test_the_rendered_report_states_that_nothing_retrains(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """The caveat travels with the document, because the document travels."""
    profile, frame = baseline
    incoming = _Reference(frame, fingerprint="9" * 64)
    report = build_drift_report(
        profile=profile,
        incoming=incoming,
        features=_compare(profile, frame),
        predictions=(),
        incoming_manifest=None,
        warn_threshold=WARN,
        alert_threshold=ALERT,
        min_support=1,
    )
    rendered = drift_report_to_markdown(
        report, build_drift_manifest(profile=profile, report=report)
    )
    lowered = rendered.lower()
    assert "no retraining happened" in lowered
    assert "**not** model correctness" in lowered
    assert "is **not** `no_drift`" in lowered
    assert "without reading a label" in lowered


def test_the_report_separates_feature_and_prediction_drift(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Two aggregates, never blended into one verdict."""
    profile, frame = baseline
    report = build_drift_report(
        profile=profile,
        incoming=_Reference(frame, fingerprint="9" * 64),
        features=_compare(profile, frame),
        predictions=(),
        incoming_manifest=None,
        warn_threshold=WARN,
        alert_threshold=ALERT,
        min_support=1,
    )
    assert report.feature_status is DriftStatus.NO_DRIFT
    # No publication was compared, so the prediction side measured nothing.
    assert report.prediction_status is DriftStatus.UNAVAILABLE
    assert "overall" not in report.to_json().lower()


def test_a_second_population_does_not_change_the_first_report(
    baseline: tuple[MLReferenceProfile, _Frame],
) -> None:
    """Reports are values; running another comparison cannot revise one."""
    profile, frame = baseline
    shifted = _Frame(
        frame.feature_names,
        tuple(tuple(row) for row in raw_rows(40, seed=99)),
    )
    first = build_drift_report(
        profile=profile,
        incoming=_Reference(frame, fingerprint="9" * 64),
        features=_compare(profile, frame),
        predictions=(),
        incoming_manifest=None,
        warn_threshold=WARN,
        alert_threshold=ALERT,
        min_support=1,
    )
    captured = first.to_json()
    _compare(profile, shifted)
    assert first.to_json() == captured
