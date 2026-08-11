"""Training orchestration: three tracks, kept apart, on the splits they may read.

This suite fits real models on a small hand-specified dataset, because the thing
under test is the composition and a mocked composition would test nothing. What
it does not do is measure them: no assertion here reads an accuracy, a
threshold value, or a score, and the orchestration produces no ranking for one
to come from.

The two arguments that carry the most weight are at the end. The firewall suite
proves the frozen splits cannot reach a fitted quantity, and the reproducibility
suite proves that two runs of identical semantics produce identical identities
and identical bytes.
"""

from __future__ import annotations

from typing import Any

import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    CalibrationStatus,
    MLSplit,
    MLTask,
    ModelFamily,
    SelectionStatus,
    TrainingRunStatus,
)
from password_attack_detector.ml.training import (
    NON_CONSTRUCTOR_HYPERPARAMETERS,
    TrainingContext,
    enumerate_candidates,
    train_all,
    train_candidate,
)
from tests.ml import runs


@pytest.fixture(scope="module")
def context() -> TrainingContext:
    """Return one prepared context, shared by the tests that only read it."""
    return runs.context()


@pytest.fixture(scope="module")
def outcomes(context: TrainingContext) -> tuple[Any, ...]:
    """Return every candidate's outcome, trained once for the whole module."""
    return train_all(context)


def by_label(outcomes: tuple[Any, ...], label: str) -> Any:
    """Return the outcome for one candidate label."""
    return next(item for item in outcomes if item.candidate.label == label)


def candidate(config: Any, label: str) -> Any:
    """Return the enumerated candidate with one label."""
    return next(item for item in enumerate_candidates(config) if item.label == label)


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------


def test_candidates_are_enumerated_deterministically() -> None:
    """Same configuration, same sequence -- ordered by task then catalog entry."""
    config = runs.config()
    first = enumerate_candidates(config)
    second = enumerate_candidates(config)
    assert [item.label for item in first] == [item.label for item in second]
    assert [item.label for item in first] == [
        "M-000/binary_malicious",
        "M-001/binary_malicious",
        "M-010/binary_malicious",
        "M-000/attack_category",
        "M-010/attack_category",
        "M-030/anomaly",
    ]


def test_a_candidate_fingerprint_covers_its_effective_settings() -> None:
    """Two candidates that would fit differently are different candidates."""
    baseline = enumerate_candidates(runs.config())
    changed = enumerate_candidates(
        runs.config(
            family_hyperparameters={ModelFamily.LOGISTIC_REGRESSION: {"max_iter": 500}}
        )
    )

    def fingerprint(items: Any, label: str) -> Any:
        return next(item for item in items if item.label == label).candidate_fingerprint

    assert fingerprint(baseline, "M-010/binary_malicious") != fingerprint(
        changed, "M-010/binary_malicious"
    )
    # And a candidate nobody changed keeps its identity.
    assert fingerprint(baseline, "M-000/binary_malicious") == fingerprint(
        changed, "M-000/binary_malicious"
    )


def test_no_wall_clock_takes_part_in_candidate_identity() -> None:
    """Enumerated twice at different moments, and identical both times."""
    first = enumerate_candidates(runs.config())
    second = enumerate_candidates(runs.config())
    assert [item.candidate_fingerprint for item in first] == [
        item.candidate_fingerprint for item in second
    ]


def test_disabling_a_head_removes_its_candidates() -> None:
    """The category head and the anomaly probe are configuration, not defaults."""
    from password_attack_detector.ml.config import AnomalyConfig, CategoryConfig

    without = enumerate_candidates(
        runs.config(
            category=CategoryConfig(enabled=False),
            anomaly=AnomalyConfig(enabled=False, quantile=0.9, min_fit_rows=10),
        )
    )
    tasks = {item.task for item in without}
    assert tasks == {MLTask.BINARY_MALICIOUS}


def test_no_search_procedure_exists() -> None:
    """No grid, no sampler, no optimiser: a candidate exists because it was written.

    Asserted on the module's own surface rather than in prose, so a search
    helper added later fails here.
    """
    import password_attack_detector.ml.training as module

    for banned in (
        "optuna",
        "random_search",
        "grid_search",
        "bayesian",
        "tune",
        "optimize",
        "search_space",
    ):
        assert not hasattr(module, banned)


# ---------------------------------------------------------------------------
# Binary track
# ---------------------------------------------------------------------------


def test_the_binary_track_produces_a_complete_run(outcomes: tuple[Any, ...]) -> None:
    """Fitted, calibrated on validation-A, measured and thresholded on validation-B."""
    outcome = by_label(outcomes, "M-010/binary_malicious")
    assert outcome.status is TrainingRunStatus.COMPLETED
    assert outcome.fitted is not None
    assert outcome.fitted.task is MLTask.BINARY_MALICIOUS
    assert outcome.class_weights is not None

    assert outcome.calibration is not None
    assert outcome.calibration.status is CalibrationStatus.FITTED
    assert outcome.calibration_state is not None
    assert outcome.calibration_diagnostic is not None
    assert outcome.calibration_quality is not None
    assert outcome.binary_threshold is not None
    assert outcome.binary_threshold.status is SelectionStatus.SELECTED

    # And nothing belonging to another track.
    assert outcome.category_abstention is None
    assert outcome.anomaly_threshold is None


def test_the_calibrator_is_fitted_on_validation_a_and_measured_on_b(
    outcomes: tuple[Any, ...],
) -> None:
    """The Milestone 5 split discipline, carried through the orchestration."""
    from password_attack_detector.ml.enums import (
        CalibrationEvaluationKind,
        ValidationPartition,
    )

    outcome = by_label(outcomes, "M-010/binary_malicious")
    assert (
        outcome.calibration_state.fit_source_partition
        is ValidationPartition.VALIDATION_A
    )
    assert (
        outcome.calibration_diagnostic.evaluation_kind
        is CalibrationEvaluationKind.IN_SAMPLE_FIT_DIAGNOSTIC
    )
    assert (
        outcome.calibration_quality.evaluation_kind
        is CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION
    )
    assert (
        outcome.calibration_quality.source_partition is ValidationPartition.VALIDATION_B
    )
    assert outcome.binary_threshold.source_partition is ValidationPartition.VALIDATION_B


def test_the_binary_threshold_is_chosen_on_calibrated_probabilities(
    outcomes: tuple[Any, ...],
) -> None:
    """And names the calibrator that produced them."""
    from password_attack_detector.ml.enums import ScoreKind

    outcome = by_label(outcomes, "M-010/binary_malicious")
    selection = outcome.binary_threshold
    assert selection.score_kind is ScoreKind.CALIBRATED_PROBABILITY
    assert (
        selection.calibration_state_fingerprint
        == outcome.calibration_state.calibration_state_fingerprint
    )


def test_the_reference_baseline_is_trained_and_stays_non_champion(
    context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """M-000 is fitted, published, and permanently outside the contest."""
    outcome = by_label(outcomes, "M-000/binary_malicious")
    assert outcome.fitted is not None
    assert outcome.fitted.champion_eligible is False
    spec = context.catalog.for_family(ModelFamily.PRIOR_BASELINE)
    assert spec.reference_baseline is True
    assert spec.champion_eligible is False


def test_the_reference_baseline_completes_without_a_calibrator(
    outcomes: tuple[Any, ...],
) -> None:
    """The mandatory comparator must be usable, and it is -- uncalibrated.

    M-000 emits the training class prior for every row, so there is no score
    variation for a calibrator to map and both Milestone 5 methods require
    distinct scores. Rather than leave the comparator every candidate is
    measured against permanently unusable, calibration is declared *not
    applicable* for a reference baseline: the run keeps its ``decision_score``,
    chooses an operating point on it, and completes.
    """
    outcome = by_label(outcomes, "M-000/binary_malicious")
    assert outcome.status is TrainingRunStatus.COMPLETED
    assert outcome.fitted is not None
    assert outcome.binary_threshold is not None
    assert outcome.binary_threshold.status is SelectionStatus.SELECTED


def test_the_reference_baseline_fakes_no_calibrator(
    outcomes: tuple[Any, ...],
) -> None:
    """Not applicable is recorded as such, and no state is invented."""
    from password_attack_detector.ml.enums import ScoreKind

    outcome = by_label(outcomes, "M-000/binary_malicious")
    assert outcome.calibration is not None
    assert outcome.calibration.status is CalibrationStatus.NOT_CALIBRATED
    assert outcome.calibration.method is CalibrationMethod.NONE
    assert outcome.calibration_state is None
    assert outcome.calibration_diagnostic is None
    assert outcome.calibration_quality is None

    # The vocabulary is untouched: an uncalibrated score is never relabelled.
    assert outcome.fitted.score_semantics.score_kind is ScoreKind.DECISION_SCORE
    assert outcome.binary_threshold.score_kind is ScoreKind.DECISION_SCORE
    assert outcome.binary_threshold.calibration_state_fingerprint is None
    assert outcome.binary_threshold.calibration_method is CalibrationMethod.NONE


def test_the_reference_baseline_is_still_not_champion_eligible(
    context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Completing a run is not promotion, and never becomes one."""
    outcome = by_label(outcomes, "M-000/binary_malicious")
    assert outcome.fitted.champion_eligible is False
    spec = context.catalog.for_family(ModelFamily.PRIOR_BASELINE)
    assert spec.reference_baseline is True
    assert spec.champion_eligible is False


def test_an_ordinary_candidate_still_fails_when_calibration_does() -> None:
    """The exemption is for the comparator alone, not a general relaxation.

    A champion candidate whose configured protocol requires calibration and
    cannot fit one reports ``calibration_unavailable`` exactly as before.
    """
    from password_attack_detector.ml.config import CalibrationConfig

    settings = runs.config(
        calibration=CalibrationConfig(
            method=CalibrationMethod.PLATT,
            max_expected_calibration_error=0.5,
            reliability_bin_count=4,
            min_calibration_rows=100_000,
            min_reliability_bin_rows=1,
            min_isotonic_distinct_scores=2,
        )
    )
    context = runs.context(settings=settings)
    outcome = train_candidate(
        candidate(settings, "M-010/binary_malicious"), context=context
    )
    assert outcome.status is TrainingRunStatus.CALIBRATION_UNAVAILABLE
    assert outcome.fitted is not None
    assert outcome.calibration is not None
    assert outcome.calibration.state is None
    assert "min_calibration_rows" in outcome.failing_requirements
    assert outcome.binary_threshold is None


def test_the_threshold_baseline_needs_a_reviewed_column(
    context: TrainingContext,
) -> None:
    """Unconfigured, M-001 reports unavailable rather than picking a column.

    Scanning for the best column would be a model-selection procedure run on
    training data and reported as a baseline.
    """
    unconfigured = runs.context(
        settings=runs.config(single_feature_baseline_column=None)
    )
    outcome = train_candidate(
        candidate(context.config, "M-001/binary_malicious"), context=unconfigured
    )
    assert outcome.status is TrainingRunStatus.UNAVAILABLE
    assert outcome.failing_requirements == ("model_fit",)
    assert outcome.fitted is None


# ---------------------------------------------------------------------------
# Category track
# ---------------------------------------------------------------------------


def test_the_category_track_fits_known_malicious_rows_only(
    context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Benign rows are excluded, and so is anything with no known category."""
    outcome = by_label(outcomes, "M-010/attack_category")
    assert outcome.status is TrainingRunStatus.COMPLETED
    assert outcome.fitted is not None
    assert outcome.fitted.task is MLTask.ATTACK_CATEGORY

    population = context.category_train()
    assert population.row_count == outcome.train_row_count
    assert all(population.malicious)
    assert all(value is not None for value in population.known_category)
    assert population.row_count < context.supervised_train().row_count


def test_the_category_class_space_comes_from_the_scenario_contract(
    outcomes: tuple[Any, ...],
) -> None:
    """Derived from the Phase 2 label schema, never from a Phase 4 rule name."""
    from password_attack_detector.ml.dataset import KNOWN_CATEGORY_CLASSES

    outcome = by_label(outcomes, "M-010/attack_category")
    assert set(outcome.fitted.class_order) <= set(KNOWN_CATEGORY_CLASSES)
    assert list(outcome.fitted.class_order) == sorted(outcome.fitted.class_order)


def test_the_category_track_publishes_only_an_abstention_point(
    outcomes: tuple[Any, ...],
) -> None:
    """A category head never inherits the binary threshold, and never a calibrator."""
    outcome = by_label(outcomes, "M-010/attack_category")
    assert outcome.category_abstention is not None
    assert outcome.binary_threshold is None
    assert outcome.anomaly_threshold is None
    assert outcome.calibration is None
    assert outcome.calibration_state is None


def test_a_thin_category_reports_insufficient_support() -> None:
    """A precision measured over two rows of a class is not a measurement."""
    from password_attack_detector.ml.config import CategoryConfig

    settings = runs.config(
        category=CategoryConfig(
            min_category_score=0.2,
            min_rows_per_category=5_000,
            min_known_category_precision=0.4,
            min_known_malicious_rows=5,
        )
    )
    context = runs.context(settings=settings)
    outcome = train_candidate(
        candidate(settings, "M-010/attack_category"), context=context
    )
    assert outcome.status is TrainingRunStatus.INSUFFICIENT_SUPPORT
    assert outcome.failing_requirements == ("min_rows_per_category",)


# ---------------------------------------------------------------------------
# Anomaly track
# ---------------------------------------------------------------------------


def test_the_anomaly_track_fits_benign_training_rows_with_no_target(
    context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Which rows are benign is decided here; the estimator sees a matrix only."""
    outcome = by_label(outcomes, "M-030/anomaly")
    assert outcome.status is TrainingRunStatus.COMPLETED
    assert outcome.fitted is not None
    assert outcome.fitted.task is MLTask.ANOMALY
    assert outcome.fitted.class_order == ()
    assert outcome.class_weights is None

    population = context.benign_train()
    assert not any(population.malicious)
    assert population.row_count == outcome.train_row_count


def test_the_anomaly_track_carries_no_calibrator(outcomes: tuple[Any, ...]) -> None:
    """An unsupervised magnitude is never a calibrated probability."""
    from password_attack_detector.ml.enums import ScoreKind

    outcome = by_label(outcomes, "M-030/anomaly")
    assert outcome.calibration is None
    assert outcome.calibration_state is None
    assert outcome.calibration_quality is None
    assert outcome.fitted.score_semantics.score_kind is ScoreKind.ANOMALY_SCORE
    assert outcome.anomaly_threshold is not None
    assert outcome.anomaly_threshold.score_kind is ScoreKind.ANOMALY_SCORE


def test_the_anomaly_threshold_reads_only_benign_training_scores(
    outcomes: tuple[Any, ...],
) -> None:
    """Its declared provenance, and no validation outcome anywhere in it."""
    from password_attack_detector.ml.enums import AnomalyThresholdMethod

    selection = by_label(outcomes, "M-030/anomaly").anomaly_threshold
    assert selection.method is AnomalyThresholdMethod.TRAIN_BENIGN_QUANTILE
    assert selection.source_split is MLSplit.TRAIN
    assert selection.source_partition is None
    assert selection.influences_champion_selection is False


def test_the_anomaly_probe_can_read_validation_a_benign_rows_instead() -> None:
    """The other permitted provenance, and it still touches no malicious row."""
    from password_attack_detector.ml.config import AnomalyConfig
    from password_attack_detector.ml.enums import (
        AnomalyThresholdMethod,
        ValidationPartition,
    )

    settings = runs.config(
        anomaly=AnomalyConfig(
            threshold_method=AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR,
            target_benign_flag_rate=0.2,
            quantile=0.9,
            min_fit_rows=10,
        )
    )
    context = runs.context(settings=settings)
    outcome = train_candidate(candidate(settings, "M-030/anomaly"), context=context)
    selection = outcome.anomaly_threshold
    assert selection is not None
    assert selection.method is AnomalyThresholdMethod.VALIDATION_A_BENIGN_FPR
    assert selection.source_partition is ValidationPartition.VALIDATION_A
    assert (
        context.validation_benign(ValidationPartition.VALIDATION_A).positive_count == 0
    )


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------


def test_every_configured_candidate_produces_an_outcome(
    context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """A candidate list that silently shrinks is a comparison nobody can audit."""
    assert len(outcomes) == len(enumerate_candidates(context.config))
    assert {item.candidate.label for item in outcomes} == {
        item.label for item in enumerate_candidates(context.config)
    }


def test_an_unpublishable_family_reports_serializer_unavailable() -> None:
    """M-021 may be fitted in process and never stored, so it never completes."""
    from password_attack_detector.ml.config import AnomalyConfig

    settings = runs.config(
        enabled_model_families=(
            ModelFamily.PRIOR_BASELINE,
            ModelFamily.LOGISTIC_REGRESSION,
            ModelFamily.HISTOGRAM_GRADIENT_BOOSTING,
        ),
        anomaly=AnomalyConfig(enabled=False, quantile=0.9, min_fit_rows=10),
    )
    context = runs.context(settings=settings)
    outcome = train_candidate(
        candidate(settings, "M-021/binary_malicious"), context=context
    )
    assert outcome.status is TrainingRunStatus.SERIALIZER_UNAVAILABLE
    assert outcome.failing_requirements == ("publishable_family",)
    assert outcome.fitted is None


def test_thin_training_support_stops_a_candidate_before_it_fits() -> None:
    """A model fitted below the support floor would be a number nobody can use."""
    from password_attack_detector.ml.schemas import SupportRequirement

    settings = runs.config(
        support=SupportRequirement(
            min_train_positive_rows=5_000,
            min_validation_positive_rows=5,
            min_validation_benign_rows=10,
            min_rows_per_category=2,
        )
    )
    context = runs.context(settings=settings)
    outcome = train_candidate(
        candidate(settings, "M-010/binary_malicious"), context=context
    )
    assert outcome.status is TrainingRunStatus.INSUFFICIENT_SUPPORT
    assert outcome.failing_requirements == ("min_train_positive_rows",)
    assert outcome.fitted is None


def test_an_unusable_validation_partition_stops_the_whole_run() -> None:
    """Not a per-candidate outcome: no candidate could recover from it."""
    from password_attack_detector.ml.config import ValidationPartitionConfig

    with pytest.raises(ModelTrainingError, match="could not be partitioned"):
        runs.context(
            settings=runs.config(
                validation_partition=ValidationPartitionConfig(
                    min_partition_rows=100_000, min_partition_positive_rows=2
                )
            )
        )


def test_completed_is_the_only_status_meaning_everything_is_present(
    outcomes: tuple[Any, ...],
) -> None:
    """Every other member names a specific missing piece."""
    for outcome in outcomes:
        assert outcome.complete is (outcome.status is TrainingRunStatus.COMPLETED)
        if outcome.complete:
            assert outcome.failing_requirements == ()
        else:
            assert outcome.failing_requirements


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


def test_a_declared_hyperparameter_no_adapter_reads_is_refused() -> None:
    """Silently dropping one is the behaviour this project refuses.

    ``class_weight`` and ``selection_metric`` are named exceptions with stated
    reasons; anything else the catalog declares must be a constructor argument.
    """
    assert (
        frozenset({"class_weight", "selection_metric"})
        == NON_CONSTRUCTOR_HYPERPARAMETERS
    )


def test_class_weights_reach_the_fit_as_sample_weights(
    outcomes: tuple[Any, ...],
) -> None:
    """Computed from training counts, and recorded beside the model that used them."""
    outcome = by_label(outcomes, "M-010/binary_malicious")
    assert outcome.class_weights is not None
    assert outcome.class_weights.computed_from == "train"
    assert (
        outcome.fitted.class_weight_fingerprint == outcome.class_weights.fingerprint()
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_each_track_has_its_own_fitted_preprocessing_state(
    context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Three populations, three fitted states, three fingerprints.

    A fitted preprocessor learns imputation constants, category vocabularies,
    rare-value buckets, and scaling statistics, so its fitting population is
    part of the learned pipeline. Sharing one across the three tracks would let
    the anomaly probe's encoder be shaped by malicious rows it is supposed to be
    blind to.
    """
    by_task = {
        outcome.candidate.task: outcome.fitted.preprocessor_fingerprint
        for outcome in outcomes
        if outcome.fitted is not None
    }
    assert len(set(by_task.values())) == 3
    assert by_task[MLTask.BINARY_MALICIOUS] == context.preprocessor.fingerprint()
    for task, fingerprint in by_task.items():
        assert fingerprint == context.preprocessor_for(task).fingerprint()


def test_families_within_one_task_share_that_task_s_state(
    outcomes: tuple[Any, ...],
) -> None:
    """Identical preprocessing configuration, so one fitted state and one identity."""
    binary = {
        outcome.fitted.preprocessor_fingerprint
        for outcome in outcomes
        if outcome.fitted is not None
        and outcome.candidate.task is MLTask.BINARY_MALICIOUS
    }
    assert len(binary) == 1


def test_each_track_s_preprocessing_sees_only_its_own_population(
    context: TrainingContext,
) -> None:
    """Asserted on the populations themselves, not only on the fingerprints."""
    binary = context.training_population(MLTask.BINARY_MALICIOUS)
    category = context.training_population(MLTask.ATTACK_CATEGORY)
    anomaly = context.training_population(MLTask.ANOMALY)

    assert all(category.malicious)
    assert all(value is not None for value in category.known_category)
    assert not any(anomaly.malicious)
    assert category.row_count < binary.row_count
    assert anomaly.row_count < binary.row_count
    # The two specialised populations are disjoint: one is every malicious row
    # with a known category, the other is every benign row.
    assert category.row_count + anomaly.row_count <= binary.row_count


def test_preprocessing_sees_training_rows_only(context: TrainingContext) -> None:
    """Enforced by the preprocessor itself, which refuses any other split."""
    from password_attack_detector.ml.preprocessing import fit_preprocessor

    validation = context.validation(
        __import__(
            "password_attack_detector.ml.enums", fromlist=["ValidationPartition"]
        ).ValidationPartition.VALIDATION_A
    )
    with pytest.raises(ModelTrainingError, match="may be fitted on"):
        fit_preprocessor(
            validation.frame,
            catalog=context.feature_catalog,
            eligible=context.eligible,
            config=context.config.preprocessing,
        )


# ---------------------------------------------------------------------------
# The test and holdout firewall
# ---------------------------------------------------------------------------


def _identities(context: TrainingContext) -> dict[str, Any]:
    """Return every fitted identity one context produces."""
    from password_attack_detector.ml.experiments import build_training_run_record

    result: dict[str, Any] = {"preprocessor": context.preprocessor.fingerprint()}
    for outcome in train_all(context):
        label = outcome.candidate.label
        record = build_training_run_record(outcome, context=context)
        result[f"{label}:run_id"] = record.run_id
        result[f"{label}:record"] = record.to_json()
        if outcome.fitted is not None:
            result[f"{label}:model"] = outcome.fitted.content_fingerprint()
        if outcome.class_weights is not None:
            result[f"{label}:weights"] = outcome.class_weights.fingerprint()
        if outcome.calibration_state is not None:
            result[f"{label}:calibrator"] = outcome.calibration_state.to_json()
        if outcome.calibration_quality is not None:
            result[f"{label}:quality"] = outcome.calibration_quality.to_json()
        if outcome.binary_threshold is not None:
            result[f"{label}:threshold"] = outcome.binary_threshold.to_json()
        if outcome.category_abstention is not None:
            result[f"{label}:abstention"] = outcome.category_abstention.to_json()
        if outcome.anomaly_threshold is not None:
            result[f"{label}:anomaly"] = outcome.anomaly_threshold.to_json()
    return result


def test_mutating_test_and_holdout_changes_nothing_at_all() -> None:
    """The central acceptance boundary, stated as an equality over everything.

    Train, validation-A and validation-B are held fixed and the frozen splits
    are replaced wholesale -- different sizes, different label mix, different
    feature values. Every preprocessing statistic, every fitted model, every
    calibrator, every report, every operating point, and every run identifier
    comes out byte-identical.
    """
    baseline = _identities(runs.context())
    mutated = _identities(
        runs.context(
            rows=runs.build_rows(test_rows=4, holdout_rows=100),
        )
    )
    assert mutated == baseline


def test_mutating_train_moves_the_model_and_the_run_identity() -> None:
    """The converse: a genuine semantic change must be visible."""
    baseline = _identities(runs.context())
    mutated = _identities(runs.context(rows=runs.build_rows(train_signal=0.42)))
    assert mutated["preprocessor"] != baseline["preprocessor"]
    assert (
        mutated["M-010/binary_malicious:model"]
        != baseline["M-010/binary_malicious:model"]
    )
    assert (
        mutated["M-010/binary_malicious:run_id"]
        != baseline["M-010/binary_malicious:run_id"]
    )


def test_mutating_validation_moves_the_calibrator_and_not_the_model() -> None:
    """Validation-A calibrates; it does not fit."""
    baseline = _identities(runs.context())
    mutated = _identities(runs.context(rows=runs.build_rows(validation_rows=90)))
    assert (
        mutated["M-010/binary_malicious:model"]
        == baseline["M-010/binary_malicious:model"]
    )
    assert mutated["preprocessor"] == baseline["preprocessor"]
    assert (
        mutated["M-010/binary_malicious:calibrator"]
        != baseline["M-010/binary_malicious:calibrator"]
    )
    assert (
        mutated["M-010/binary_malicious:run_id"]
        != baseline["M-010/binary_malicious:run_id"]
    )


def test_no_training_entry_point_can_be_handed_test_rows() -> None:
    """There is no parameter through which they could arrive.

    Row selection happens inside the orchestration, from a dataset that already
    knows which split each row belongs to, so a caller cannot substitute one.
    """
    import inspect

    for function in (TrainingContext.prepare, train_candidate, train_all):
        parameters = set(inspect.signature(function).parameters)
        assert not any(
            "test" in name or "holdout" in name or "novel" in name
            for name in parameters
        )


def test_the_frozen_splits_are_present_and_simply_never_read(
    context: TrainingContext,
) -> None:
    """The dataset carries them, which is what makes the firewall meaningful.

    A firewall over splits that were not there would prove nothing.
    """
    assert context.dataset.for_split(MLSplit.TEST).row_count > 0
    assert context.dataset.for_split(MLSplit.NOVEL_ANOMALY_HOLDOUT).row_count > 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_two_runs_of_identical_semantics_are_identical() -> None:
    """Same inputs, same everything -- fingerprints, identities, and bytes."""
    assert _identities(runs.context()) == _identities(runs.context())


def test_shuffling_the_source_rows_changes_nothing() -> None:
    """Canonical ordering is applied once, on the way out of dataset assembly."""
    rows = runs.build_rows()
    baseline = _identities(runs.context(rows=rows))
    reordered = _identities(runs.context(rows=runs.shuffled(rows)))
    assert reordered == baseline


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 11},
        {"single_feature_baseline_column": "user_attempt_count"},
    ],
)
def test_a_semantic_configuration_change_moves_run_identity(
    change: dict[str, Any],
) -> None:
    """Anything that changes what would be fitted changes what it is called."""
    baseline = _identities(runs.context())
    mutated = _identities(runs.context(settings=runs.config(**change)))
    changed = [
        key
        for key in baseline
        if key.endswith(":run_id") and mutated.get(key) != baseline[key]
    ]
    assert changed


# ---------------------------------------------------------------------------
# Role-scoped lineage
# ---------------------------------------------------------------------------


def _lineage(context: TrainingContext, task: MLTask) -> Any:
    """Return the role-scoped lineage digests for one task."""
    return context.readable_lineage(task)


def _run_ids(context: TrainingContext) -> dict[str, str]:
    """Return each candidate's run identifier."""
    from password_attack_detector.ml.experiments import build_training_run_record

    return {
        outcome.candidate.label: build_training_run_record(
            outcome, context=context
        ).run_id
        for outcome in train_all(context)
    }


@pytest.fixture(scope="module")
def base_rows() -> Any:
    """Return the unmodified fixture tables."""
    return runs.build_rows()


@pytest.fixture(scope="module")
def anchors(base_rows: Any) -> dict[str, str]:
    """Return one representative anchor from each population that matters."""
    from password_attack_detector.ml.enums import ValidationPartition

    context = runs.context(rows=base_rows)
    assignment = context.partition.assignment
    malicious = runs.malicious_anchors(base_rows)
    train = runs.anchors_in(base_rows, MLSplit.TRAIN)
    validation_a = [
        key
        for key, half in assignment.items()
        if half is ValidationPartition.VALIDATION_A
    ]
    validation_b = [
        key
        for key, half in assignment.items()
        if half is ValidationPartition.VALIDATION_B
    ]
    return {
        "train_malicious": next(item for item in train if item in malicious),
        "train_benign": next(item for item in train if item not in malicious),
        "validation_a": sorted(validation_a)[0],
        "validation_b": sorted(validation_b)[0],
        "test": runs.anchors_in(base_rows, MLSplit.TEST)[0],
        "holdout": runs.anchors_in(base_rows, MLSplit.NOVEL_ANOMALY_HOLDOUT)[0],
    }


def test_the_three_lineage_digests_are_separate(context: TrainingContext) -> None:
    """One digest each for data, labels, and role membership.

    Kept apart so a moved fingerprint names its own cause. Collapsing them would
    tell a reader that *something* about the lineage changed and never what.
    """
    lineage = _lineage(context, MLTask.BINARY_MALICIOUS)
    assert len({lineage.training_data, lineage.labels, lineage.split}) == 3


def test_each_track_scopes_its_lineage_to_the_rows_it_reads(
    context: TrainingContext,
) -> None:
    """A digest over "everything the run touched" would move for the wrong reasons."""
    binary = _lineage(context, MLTask.BINARY_MALICIOUS)
    category = _lineage(context, MLTask.ATTACK_CATEGORY)
    anomaly = _lineage(context, MLTask.ANOMALY)
    assert (
        len({binary.training_data, category.training_data, anomaly.training_data}) == 3
    )

    roles = {
        task: [role for role, _ in context.readable_roles(task)]
        for task in (MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY, MLTask.ANOMALY)
    }
    assert roles[MLTask.BINARY_MALICIOUS] == ["train", "validation_a", "validation_b"]
    # The category head fits no calibrator, so it never reads validation-A.
    assert roles[MLTask.ATTACK_CATEGORY] == ["train", "validation_b"]
    assert roles[MLTask.ANOMALY] == ["train"]


def test_a_train_feature_value_moves_the_training_data_lineage(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case A: the data digest moves, and the run identity with it."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(
        rows=runs.revalue_anchor(base_rows, anchors["train_benign"], value=0.4242)
    )
    before = _lineage(baseline, MLTask.BINARY_MALICIOUS)
    after = _lineage(mutated, MLTask.BINARY_MALICIOUS)
    assert after.training_data != before.training_data
    assert after.labels == before.labels
    assert after.split == before.split
    assert (
        _run_ids(mutated)["M-010/binary_malicious"]
        != _run_ids(baseline)["M-010/binary_malicious"]
    )


def test_a_train_label_moves_the_label_lineage(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case B: the label digest moves and the data digest does not."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(rows=runs.relabel_anchor(base_rows, anchors["train_benign"]))
    before = _lineage(baseline, MLTask.BINARY_MALICIOUS)
    after = _lineage(mutated, MLTask.BINARY_MALICIOUS)
    assert after.labels != before.labels
    assert after.training_data == before.training_data
    assert (
        _run_ids(mutated)["M-010/binary_malicious"]
        != _run_ids(baseline)["M-010/binary_malicious"]
    )


def test_a_validation_a_label_moves_the_lineage_and_the_calibrator(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case C: validation-A is where the calibrator is fitted, so both move."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(rows=runs.relabel_anchor(base_rows, anchors["validation_a"]))
    assert (
        _lineage(mutated, MLTask.BINARY_MALICIOUS).labels
        != _lineage(baseline, MLTask.BINARY_MALICIOUS).labels
    )

    def calibrator(context: TrainingContext) -> Any:
        outcome = by_label(train_all(context), "M-010/binary_malicious")
        assert outcome.calibration_state is not None
        return outcome.calibration_state.calibration_state_fingerprint

    assert calibrator(mutated) != calibrator(baseline)
    assert (
        _run_ids(mutated)["M-010/binary_malicious"]
        != _run_ids(baseline)["M-010/binary_malicious"]
    )


def test_a_validation_b_label_moves_the_lineage_and_the_operating_point(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case D: validation-B chooses the threshold and measures the calibrator."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(rows=runs.relabel_anchor(base_rows, anchors["validation_b"]))
    assert (
        _lineage(mutated, MLTask.BINARY_MALICIOUS).labels
        != _lineage(baseline, MLTask.BINARY_MALICIOUS).labels
    )

    def artifacts(context: TrainingContext) -> tuple[str, str]:
        outcome = by_label(train_all(context), "M-010/binary_malicious")
        assert outcome.binary_threshold is not None
        assert outcome.calibration_quality is not None
        return (
            outcome.binary_threshold.selection_fingerprint,
            outcome.calibration_quality.report_fingerprint,
        )

    assert artifacts(mutated) != artifacts(baseline)
    assert (
        _run_ids(mutated)["M-010/binary_malicious"]
        != _run_ids(baseline)["M-010/binary_malicious"]
    )


def test_a_readable_split_reassignment_moves_the_split_lineage(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case E: a row leaving a readable role is a lineage change, not a data one."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(
        rows=runs.reassign_anchor(
            base_rows, anchors["train_benign"], split=MLSplit.EXCLUDED
        )
    )
    before = _lineage(baseline, MLTask.BINARY_MALICIOUS)
    after = _lineage(mutated, MLTask.BINARY_MALICIOUS)
    assert after.split != before.split
    assert after.training_data != before.training_data
    assert (
        _run_ids(mutated)["M-010/binary_malicious"]
        != _run_ids(baseline)["M-010/binary_malicious"]
    )


@pytest.mark.parametrize("frozen", ["test", "holdout"])
def test_perturbing_a_frozen_split_moves_nothing(
    base_rows: Any, anchors: dict[str, str], frozen: str
) -> None:
    """Cases F and G: features, labels, and split assignment, one at a time.

    Each perturbation is a real change to the dataset and a change to the
    dataset's own whole-row fingerprints. None of them reaches a Milestone 6
    identity, because none of those rows is in a readable role.
    """
    anchor = anchors[frozen]
    other = MLSplit.NOVEL_ANOMALY_HOLDOUT if frozen == "test" else MLSplit.TEST
    baseline = runs.context(rows=base_rows)
    expected = _run_ids(baseline)
    lineages = {
        task: _lineage(baseline, task)
        for task in (MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY, MLTask.ANOMALY)
    }

    for rows in (
        runs.revalue_anchor(base_rows, anchor, value=0.9999),
        runs.relabel_anchor(base_rows, anchor),
        runs.reassign_anchor(base_rows, anchor, split=other),
    ):
        mutated = runs.context(rows=rows)
        assert _run_ids(mutated) == expected
        for task, lineage in lineages.items():
            assert _lineage(mutated, task) == lineage, task


def test_the_whole_dataset_fingerprints_do_move_for_frozen_rows(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """The firewall is scoping, not blindness -- and this is the difference.

    The dataset's own digests cover every row it holds, so they *do* move when a
    test row changes. That is exactly why none of them may enter a training-run
    identity, and why the scoped digests exist.
    """
    baseline = runs.dataset(base_rows)
    mutated = runs.dataset(runs.revalue_anchor(base_rows, anchors["test"], value=0.5))
    assert mutated.training_data_fingerprint != baseline.training_data_fingerprint

    relabelled = runs.dataset(runs.relabel_anchor(base_rows, anchors["test"]))
    assert relabelled.label_fingerprint != baseline.label_fingerprint


def test_no_run_record_publishes_a_whole_dataset_fingerprint(
    context: TrainingContext,
) -> None:
    """They are left unset rather than published as non-semantic decoration."""
    from password_attack_detector.ml.experiments import build_training_run_record

    for outcome in train_all(context):
        identity = build_training_run_record(outcome, context=context).identity
        assert identity.training_data_fingerprint is None
        assert identity.label_fingerprint is None
        assert identity.split_config_fingerprint is None
        assert identity.readable_training_data_fingerprint is not None
        assert identity.readable_label_fingerprint is not None
        assert identity.readable_split_fingerprint is not None


def test_no_lineage_digest_publishes_a_row_identifier(
    context: TrainingContext,
) -> None:
    """Row identity is used inside the digest and leaves as a digest."""
    lineage = _lineage(context, MLTask.BINARY_MALICIOUS)
    for digest in (lineage.training_data, lineage.labels, lineage.split):
        assert len(digest) == 64
        assert "e0000" not in digest


# ---------------------------------------------------------------------------
# Per-track preprocessing populations
# ---------------------------------------------------------------------------


def _anomaly_identity(context: TrainingContext) -> tuple[str, str, str]:
    """Return the anomaly track's preprocessing, model, and threshold identity."""
    outcome = by_label(train_all(context), "M-030/anomaly")
    assert outcome.fitted is not None
    assert outcome.anomaly_threshold is not None
    return (
        outcome.fitted.preprocessor_fingerprint,
        outcome.fitted.content_fingerprint(),
        outcome.anomaly_threshold.selection_fingerprint,
    )


def _category_identity(context: TrainingContext) -> tuple[str, str]:
    """Return the category track's preprocessing and model identity."""
    outcome = by_label(train_all(context), "M-010/attack_category")
    assert outcome.fitted is not None
    return (
        outcome.fitted.preprocessor_fingerprint,
        outcome.fitted.content_fingerprint(),
    )


def test_malicious_train_rows_do_not_reach_the_anomaly_probe(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case A: change a malicious training row; the benign-only probe is unmoved.

    Its preprocessing, its model, and its threshold all come from benign rows,
    so a malicious row changing underneath it must change none of them. Under a
    shared preprocessor this would have moved every one.
    """
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(
        rows=runs.revalue_anchor(base_rows, anchors["train_malicious"], value=0.1234)
    )
    assert _anomaly_identity(mutated) == _anomaly_identity(baseline)
    assert _run_ids(mutated)["M-030/anomaly"] == _run_ids(baseline)["M-030/anomaly"]
    # And the binary track, which does read that row, moves.
    assert (
        _run_ids(mutated)["M-010/binary_malicious"]
        != _run_ids(baseline)["M-010/binary_malicious"]
    )


def test_benign_train_rows_do_reach_the_anomaly_probe(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case B: the rows it is fitted on are the rows that move it."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(
        rows=runs.revalue_anchor(base_rows, anchors["train_benign"], value=0.1234)
    )
    assert _anomaly_identity(mutated) != _anomaly_identity(baseline)
    assert _run_ids(mutated)["M-030/anomaly"] != _run_ids(baseline)["M-030/anomaly"]


def test_benign_train_rows_do_not_reach_the_category_head(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case C: the head answers "which attack", so benign rows are not its business."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(
        rows=runs.revalue_anchor(base_rows, anchors["train_benign"], value=0.777)
    )
    assert _category_identity(mutated) == _category_identity(baseline)
    assert (
        _run_ids(mutated)["M-010/attack_category"]
        == _run_ids(baseline)["M-010/attack_category"]
    )


def test_known_malicious_train_rows_do_reach_the_category_head(
    base_rows: Any, anchors: dict[str, str]
) -> None:
    """Case D: the rows it is fitted on are the rows that move it."""
    baseline = runs.context(rows=base_rows)
    mutated = runs.context(
        rows=runs.revalue_anchor(base_rows, anchors["train_malicious"], value=0.777)
    )
    assert _category_identity(mutated) != _category_identity(baseline)
    assert (
        _run_ids(mutated)["M-010/attack_category"]
        != _run_ids(baseline)["M-010/attack_category"]
    )


def test_the_binary_track_keeps_the_full_supervised_population(
    context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Case E: unchanged -- the binary head reads every supervised training row."""
    outcome = by_label(outcomes, "M-010/binary_malicious")
    population = context.supervised_train()
    assert outcome.train_row_count == population.row_count
    assert population.positive_count > 0
    assert population.benign_count > 0
    assert (
        outcome.fitted.preprocessor_fingerprint
        == context.preprocessor_for(MLTask.BINARY_MALICIOUS).fingerprint()
    )
