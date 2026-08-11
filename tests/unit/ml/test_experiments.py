"""Publishing a run: staged, verified, promoted, indexed -- or nothing at all.

The assertions that matter are the failure ones. A publication that succeeds is
easy; what makes a run directory evidence is that a failed publication leaves
the destination and the ledger exactly as they were, and that a second
publication of a *different* run under the same identifier is refused rather
than absorbed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import (
    ExperimentPublicationError,
    LedgerConflictError,
)
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    MLTask,
    SelectionStatus,
    TrainingRunStatus,
)
from password_attack_detector.ml.experiments import (
    CALIBRATION_DIR,
    MODEL_DIR,
    RUNS_DIR,
    THRESHOLD_DIR,
    TRAINING_RUN_FILE,
    build_training_run_record,
    publish_training_run,
    reconcile,
    summarize,
)
from password_attack_detector.ml.ledger import ExperimentLedger, TrainingRunRecord
from password_attack_detector.ml.preprocessing import FittedPreprocessor
from password_attack_detector.ml.training import TrainingContext, train_all
from tests.ml import runs


@pytest.fixture(scope="module")
def context() -> TrainingContext:
    """Return one prepared context, shared across the module."""
    return runs.context()


@pytest.fixture(scope="module")
def outcomes(context: TrainingContext) -> tuple[Any, ...]:
    """Return every candidate's outcome, trained once."""
    return train_all(context)


def by_label(outcomes: tuple[Any, ...], label: str) -> Any:
    """Return the outcome for one candidate label."""
    return next(item for item in outcomes if item.candidate.label == label)


def publish_everything(
    root: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> tuple[ExperimentLedger, list[Any]]:
    """Publish every outcome under *root* and return the ledger and results."""
    ledger = ExperimentLedger(root / "ledger")
    return ledger, [
        publish_training_run(outcome, context=context, root=root, ledger=ledger)
        for outcome in outcomes
    ]


def files_in(directory: Path) -> set[str]:
    """Return every file under *directory*, relative and sorted."""
    return {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# The artifact set
# ---------------------------------------------------------------------------


def test_a_binary_run_publishes_a_model_a_calibrator_and_a_threshold(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Exactly what the binary task means, and nothing that belongs elsewhere."""
    outcome = by_label(outcomes, "M-010/binary_malicious")
    ledger = ExperimentLedger(tmp_path / "ledger")
    publication = publish_training_run(
        outcome, context=context, root=tmp_path, ledger=ledger
    )
    directory = tmp_path / RUNS_DIR / publication.run_id
    assert files_in(directory) == {
        TRAINING_RUN_FILE,
        f"{MODEL_DIR}/model.json",
        f"{MODEL_DIR}/arrays.npz",
        f"{MODEL_DIR}/preprocessor.json",
        f"{MODEL_DIR}/model_manifest.json",
        f"{CALIBRATION_DIR}/calibration_state.json",
        f"{CALIBRATION_DIR}/calibration_fit_diagnostic.json",
        f"{CALIBRATION_DIR}/calibration_validation_report.json",
        f"{THRESHOLD_DIR}/binary_threshold.json",
    }


def test_a_category_run_publishes_an_abstention_point_and_no_calibrator(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """No empty ``calibration/`` standing in for a calibrator that cannot exist."""
    outcome = by_label(outcomes, "M-010/attack_category")
    ledger = ExperimentLedger(tmp_path / "ledger")
    publication = publish_training_run(
        outcome, context=context, root=tmp_path, ledger=ledger
    )
    directory = tmp_path / RUNS_DIR / publication.run_id
    assert files_in(directory) == {
        TRAINING_RUN_FILE,
        f"{MODEL_DIR}/model.json",
        f"{MODEL_DIR}/arrays.npz",
        f"{MODEL_DIR}/preprocessor.json",
        f"{MODEL_DIR}/model_manifest.json",
        f"{THRESHOLD_DIR}/category_abstention.json",
    }
    assert not (directory / CALIBRATION_DIR).exists()


def test_an_anomaly_run_publishes_only_its_own_threshold(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """An unsupervised probe has no calibrator and no binary operating point."""
    outcome = by_label(outcomes, "M-030/anomaly")
    ledger = ExperimentLedger(tmp_path / "ledger")
    publication = publish_training_run(
        outcome, context=context, root=tmp_path, ledger=ledger
    )
    directory = tmp_path / RUNS_DIR / publication.run_id
    assert f"{THRESHOLD_DIR}/anomaly_threshold.json" in files_in(directory)
    assert not (directory / CALIBRATION_DIR).exists()
    assert not (directory / THRESHOLD_DIR / "binary_threshold.json").exists()


def test_a_candidate_with_no_model_is_recorded_and_not_published(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Nothing to write, and still part of the run history."""
    settings = runs.config(single_feature_baseline_column=None)
    unconfigured = runs.context(settings=settings)
    outcome = next(
        item
        for item in train_all(unconfigured)
        if item.candidate.label == "M-001/binary_malicious"
    )
    assert outcome.fitted is None

    ledger = ExperimentLedger(tmp_path / "ledger")
    publication = publish_training_run(
        outcome, context=unconfigured, root=tmp_path, ledger=ledger
    )
    assert publication.created is False
    assert publication.ledger.created is True
    assert not (tmp_path / RUNS_DIR / publication.run_id).exists()
    assert ledger.read(publication.run_id).status is TrainingRunStatus.UNAVAILABLE


def test_the_model_artifact_passes_the_milestone_four_verifier(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """The same checker a loader uses, not a lighter one written for publication."""
    from password_attack_detector.ml.manifest import verify_model_artifact

    _, publications = publish_everything(tmp_path, context, outcomes)
    for publication in publications:
        if not publication.created:
            continue
        directory = tmp_path / RUNS_DIR / publication.run_id / MODEL_DIR
        assert verify_model_artifact(directory).passed


def test_a_published_run_reloads_as_an_inference_model(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """The artifact is not merely well-formed; it is usable."""
    from password_attack_detector.ml.inference import InferenceModel

    outcome = by_label(outcomes, "M-010/binary_malicious")
    ledger = ExperimentLedger(tmp_path / "ledger")
    publication = publish_training_run(
        outcome, context=context, root=tmp_path, ledger=ledger
    )
    loaded = InferenceModel.load(tmp_path / RUNS_DIR / publication.run_id / MODEL_DIR)
    assert loaded.task is MLTask.BINARY_MALICIOUS
    assert loaded.document.model_content_fingerprint == (
        outcome.fitted.content_fingerprint()
    )


def test_every_published_file_is_digested_in_the_record(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """The receipt covers the bytes it was written beside."""
    import hashlib

    outcome = by_label(outcomes, "M-010/binary_malicious")
    ledger = ExperimentLedger(tmp_path / "ledger")
    publication = publish_training_run(
        outcome, context=context, root=tmp_path, ledger=ledger
    )
    directory = tmp_path / RUNS_DIR / publication.run_id
    record = TrainingRunRecord.from_json(
        (directory / TRAINING_RUN_FILE).read_text(encoding="utf-8")
    )
    digests = dict(record.artifact_digests)
    assert set(digests) == files_in(directory) - {TRAINING_RUN_FILE}
    for relative, digest in digests.items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == digest


# ---------------------------------------------------------------------------
# Idempotency and conflict
# ---------------------------------------------------------------------------


def test_publishing_the_same_run_twice_changes_nothing(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """The second call confirms the first rather than repeating it."""
    ledger, first = publish_everything(tmp_path, context, outcomes)
    snapshot = {
        path: path.read_bytes()
        for path in (tmp_path / RUNS_DIR).rglob("*")
        if path.is_file()
    }
    second = [
        publish_training_run(outcome, context=context, root=tmp_path, ledger=ledger)
        for outcome in train_all(context)
    ]
    assert [item.run_id for item in second] == [item.run_id for item in first]
    assert all(item.created is False for item in second)
    assert all(item.ledger.created is False for item in second)
    assert {
        path: path.read_bytes()
        for path in (tmp_path / RUNS_DIR).rglob("*")
        if path.is_file()
    } == snapshot


def test_a_different_run_under_the_same_identifier_is_refused(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """A published run is evidence, and evidence is never overwritten."""
    outcome = by_label(outcomes, "M-010/binary_malicious")
    ledger = ExperimentLedger(tmp_path / "ledger")
    publication = publish_training_run(
        outcome, context=context, root=tmp_path, ledger=ledger
    )
    directory = tmp_path / RUNS_DIR / publication.run_id
    before = {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }

    # Same identity, different reported status -- the status is not part of the
    # identifier, which is precisely why this has to be caught here.
    from dataclasses import replace

    contradicting = replace(
        outcome,
        status=TrainingRunStatus.THRESHOLD_UNAVAILABLE,
        failing_requirements=("max_false_positive_rate",),
    )
    with pytest.raises(ExperimentPublicationError, match="never overwritten"):
        publish_training_run(
            contradicting, context=context, root=tmp_path, ledger=ledger
        )
    assert {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    } == before


def test_a_run_directory_without_a_receipt_is_never_completed_in_place(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """An incomplete directory is refused rather than filled in."""
    outcome = by_label(outcomes, "M-010/binary_malicious")
    ledger = ExperimentLedger(tmp_path / "ledger")
    record = build_training_run_record(outcome, context=context)
    (tmp_path / RUNS_DIR / record.run_id).mkdir(parents=True)

    with pytest.raises(ExperimentPublicationError, match="without a training-run"):
        publish_training_run(outcome, context=context, root=tmp_path, ledger=ledger)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_a_staging_failure_leaves_no_partial_run(
    tmp_path: Path,
    context: TrainingContext,
    outcomes: tuple[Any, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing promoted, nothing indexed, and no staging directory left behind."""
    import password_attack_detector.ml.experiments as module

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("the disk is on fire")

    monkeypatch.setattr(module, "stage_model_directory", explode)
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = by_label(outcomes, "M-010/binary_malicious")
    with pytest.raises(ExperimentPublicationError, match="could not be staged"):
        publish_training_run(outcome, context=context, root=tmp_path, ledger=ledger)

    assert list((tmp_path / RUNS_DIR).iterdir()) == []
    assert ledger.training_runs() == ()


def test_a_verification_failure_leaves_no_partial_run(
    tmp_path: Path,
    context: TrainingContext,
    outcomes: tuple[Any, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged run that fails its own check is refused, not promoted."""
    import password_attack_detector.ml.experiments as module
    from password_attack_detector.ml.manifest import VerificationOutcome

    def failed(directory: Path) -> VerificationOutcome:
        return VerificationOutcome(
            passed=False,
            error_code="M999",
            error_detail="a deliberately failed verification",
            file_count=0,
            checks_run=1,
        )

    monkeypatch.setattr(module, "verify_model_artifact", failed)
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = by_label(outcomes, "M-010/binary_malicious")
    with pytest.raises(ExperimentPublicationError, match="failed verification"):
        publish_training_run(outcome, context=context, root=tmp_path, ledger=ledger)

    assert list((tmp_path / RUNS_DIR).iterdir()) == []
    assert ledger.training_runs() == ()


def test_an_existing_run_survives_a_failed_publication_of_another(
    tmp_path: Path,
    context: TrainingContext,
    outcomes: tuple[Any, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One run's failure never touches another's bytes."""
    import password_attack_detector.ml.experiments as module

    ledger = ExperimentLedger(tmp_path / "ledger")
    good = by_label(outcomes, "M-010/binary_malicious")
    published = publish_training_run(
        good, context=context, root=tmp_path, ledger=ledger
    )
    directory = tmp_path / RUNS_DIR / published.run_id
    before = {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("the disk is on fire")

    monkeypatch.setattr(module, "stage_model_directory", explode)
    with pytest.raises(ExperimentPublicationError):
        publish_training_run(
            by_label(outcomes, "M-030/anomaly"),
            context=context,
            root=tmp_path,
            ledger=ledger,
        )
    assert {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    } == before
    assert len(ledger.training_runs()) == 1


# ---------------------------------------------------------------------------
# Ledger ordering and reconciliation
# ---------------------------------------------------------------------------


def test_the_ledger_is_appended_only_after_the_run_is_on_disk(
    tmp_path: Path,
    context: TrainingContext,
    outcomes: tuple[Any, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure mode this ordering exists to prevent, asserted directly.

    A ledger appended first could assert a run that does not exist, and the only
    repair would be deleting an immutable record. So the ledger goes last, and a
    ledger failure leaves a complete run that is merely unindexed.
    """
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = by_label(outcomes, "M-010/binary_malicious")

    def refuse(record: Any) -> None:
        raise LedgerConflictError("a deliberately failed append")

    monkeypatch.setattr(ledger, "append", refuse)
    with pytest.raises(LedgerConflictError):
        publish_training_run(outcome, context=context, root=tmp_path, ledger=ledger)

    monkeypatch.undo()
    published = list((tmp_path / RUNS_DIR).iterdir())
    assert len(published) == 1
    assert (published[0] / TRAINING_RUN_FILE).is_file()
    assert ledger.training_runs() == ()


def test_reconcile_indexes_a_valid_unindexed_run(
    tmp_path: Path,
    context: TrainingContext,
    outcomes: tuple[Any, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deterministic recovery: read the run's own record and append it."""
    ledger = ExperimentLedger(tmp_path / "ledger")
    outcome = by_label(outcomes, "M-010/binary_malicious")

    def refuse(record: Any) -> None:
        raise LedgerConflictError("a deliberately failed append")

    monkeypatch.setattr(ledger, "append", refuse)
    with pytest.raises(LedgerConflictError):
        publish_training_run(outcome, context=context, root=tmp_path, ledger=ledger)
    monkeypatch.undo()

    appended = reconcile(root=tmp_path, ledger=ledger)
    assert len(appended) == 1
    assert len(ledger.training_runs()) == 1
    # And it is the record the run was published with, not a rebuilt one.
    directory = tmp_path / RUNS_DIR / appended[0]
    assert ledger.read(appended[0]).to_json() + "\n" == (
        directory / TRAINING_RUN_FILE
    ).read_text(encoding="utf-8")


def test_reconcile_is_idempotent(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Nothing to index means nothing appended, and nothing rewritten."""
    ledger, _ = publish_everything(tmp_path, context, outcomes)
    before = len(ledger.training_runs())
    assert reconcile(root=tmp_path, ledger=ledger) == ()
    assert len(ledger.training_runs()) == before


def test_reconcile_on_an_empty_root_does_nothing(tmp_path: Path) -> None:
    """A root with no runs is not an error."""
    ledger = ExperimentLedger(tmp_path / "ledger")
    assert reconcile(root=tmp_path, ledger=ledger) == ()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _published_bytes(root: Path) -> dict[str, bytes]:
    """Return every published artifact keyed by its run-relative path."""
    payloads: dict[str, bytes] = {}
    for directory in sorted((root / RUNS_DIR).iterdir()):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                payloads[f"{directory.name}/{path.relative_to(directory)}"] = (
                    path.read_bytes()
                )
    return payloads


def test_two_runs_in_two_directories_publish_identical_bytes(
    tmp_path: Path,
) -> None:
    """Same semantics, different directories, byte-identical artifacts.

    Including the model manifest: no publication timestamp is written, because
    an observational field would make identical runs merely equivalent rather
    than identical.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        context = runs.context()
        publish_everything(root, context, train_all(context))
    assert _published_bytes(first) == _published_bytes(second)
    assert _published_bytes(first)


def test_shuffling_the_source_rows_publishes_identical_bytes(
    tmp_path: Path,
) -> None:
    """Canonical ordering is applied once, and the artifacts prove it."""
    rows = runs.build_rows()
    first = tmp_path / "ordered"
    second = tmp_path / "shuffled"
    for root, source in ((first, rows), (second, runs.shuffled(rows))):
        context = runs.context(rows=source)
        publish_everything(root, context, train_all(context))
    assert _published_bytes(first) == _published_bytes(second)


def test_no_published_artifact_carries_a_timestamp(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Observational metadata is absent rather than merely excluded from a digest."""
    publish_everything(tmp_path, context, outcomes)
    manifest = next((tmp_path / RUNS_DIR).rglob("model_manifest.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["published_at"] is None


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_a_summary_carries_statuses_and_no_metric(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """A listing that ranked runs would be a champion selection under another name."""
    ledger, _ = publish_everything(tmp_path, context, outcomes)
    summaries = [summarize(record) for record in ledger.training_runs()]
    assert len(summaries) == len(outcomes)
    for summary in summaries:
        assert isinstance(summary.status, TrainingRunStatus)
        for attribute in type(summary).__slots__:
            assert "metric" not in attribute
            assert "score" not in attribute
    labels = {(item.catalog_model_id, item.task) for item in summaries}
    assert (("M-030"), MLTask.ANOMALY) in labels


def test_a_summary_marks_the_reference_baseline_and_the_experiment(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """M-000 stays reference-only and M-030 stays experimental, in the ledger too."""
    ledger, _ = publish_everything(tmp_path, context, outcomes)
    summaries = {
        (item.catalog_model_id, item.task): item
        for item in (summarize(record) for record in ledger.training_runs())
    }
    baseline = summaries[("M-000", MLTask.BINARY_MALICIOUS)]
    assert baseline.reference_baseline is True
    assert baseline.champion_eligible is False

    probe = summaries[("M-030", MLTask.ANOMALY)]
    assert probe.experimental is True
    assert probe.champion_eligible is False
    assert probe.calibration_method is CalibrationMethod.NONE


def test_a_run_that_stopped_early_is_visible_in_the_ledger(tmp_path: Path) -> None:
    """A candidate absent from a later comparison must be distinguishable.

    A false-positive ceiling of 0.1% cannot be resolved by sixty benign
    validation-B rows -- the coarsest observable rate is 1/60 -- so every binary
    candidate fits a model and then reports that its operating point could not
    be chosen. The models are still published; the ledger records where each run
    stopped.
    """
    from password_attack_detector.ml.config import ThresholdConfig

    settings = runs.config(
        thresholds=ThresholdConfig(
            max_false_positive_rate=0.001, min_detection_rate=0.1, search_grid_size=64
        )
    )
    context = runs.context(settings=settings)
    outcomes = train_all(context)
    ledger, _ = publish_everything(tmp_path, context, outcomes)
    statuses = {
        (record.catalog_model_id, record.task): record.status
        for record in ledger.training_runs()
    }
    assert statuses[("M-010", MLTask.BINARY_MALICIOUS)] is (
        TrainingRunStatus.THRESHOLD_UNAVAILABLE
    )
    # The category and anomaly tracks read no binary ceiling and are unaffected.
    assert statuses[("M-010", MLTask.ATTACK_CATEGORY)] is TrainingRunStatus.COMPLETED
    assert statuses[("M-030", MLTask.ANOMALY)] is TrainingRunStatus.COMPLETED


def test_the_reference_baseline_publishes_a_complete_run_without_a_calibrator(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """The mandatory comparator reaches the ledger as a completed run.

    No calibration directory is written, because there is no calibrator: an
    empty one, or a fabricated state, would each be worse than the absence.
    """
    ledger, _ = publish_everything(tmp_path, context, outcomes)
    record = next(
        item
        for item in ledger.training_runs()
        if item.catalog_model_id == "M-000" and item.task is MLTask.BINARY_MALICIOUS
    )
    assert record.status is TrainingRunStatus.COMPLETED
    assert record.reference_baseline is True
    assert record.champion_eligible is False
    assert record.calibration_method is CalibrationMethod.NONE
    assert record.binary_threshold_status is SelectionStatus.SELECTED

    directory = tmp_path / RUNS_DIR / record.run_id
    assert not (directory / CALIBRATION_DIR).exists()
    assert (directory / THRESHOLD_DIR / "binary_threshold.json").is_file()
    assert not any("calibration" in relative for relative, _ in record.artifact_digests)


def test_each_published_run_carries_its_own_preprocessing_state(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """The model and the encoder that built its matrix are published together.

    Staging refuses a mismatch, so a category head paired with the binary
    track's encoder would fail publication rather than produce a directory whose
    own fingerprints disagree.
    """
    import json

    ledger, _ = publish_everything(tmp_path, context, outcomes)
    fingerprints: dict[MLTask, set[str]] = {}
    for record in ledger.training_runs():
        directory = tmp_path / RUNS_DIR / record.run_id
        if not directory.exists():
            continue
        document = json.loads(
            (directory / MODEL_DIR / "model.json").read_text(encoding="utf-8")
        )
        stored = FittedPreprocessor.from_json(
            (directory / MODEL_DIR / "preprocessor.json").read_text(encoding="utf-8")
        )
        assert stored.fingerprint() == document["preprocessor_fingerprint"]
        assert record.identity.preprocessor_fingerprint == stored.fingerprint()
        fingerprints.setdefault(record.task, set()).add(stored.fingerprint())

    assert len(fingerprints) == 3
    for task, values in fingerprints.items():
        assert len(values) == 1, task
    assert len({next(iter(values)) for values in fingerprints.values()}) == 3


def test_the_category_abstention_provenance_survives_into_the_ledger(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """A predeclared fallback must stay distinguishable from a data selection."""
    ledger, _ = publish_everything(tmp_path, context, outcomes)
    record = next(
        item for item in ledger.training_runs() if item.task is MLTask.ATTACK_CATEGORY
    )
    assert record.category_abstention_status in set(SelectionStatus)
    assert record.category_threshold_data_selected is not None
    assert record.binary_threshold_status is None


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "banned", ["e0000", "campaign", "u:", "anchor_event_id", "/home/"]
)
def test_no_identity_reaches_a_run_receipt_or_a_ledger_record(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...], banned: str
) -> None:
    """Rows go in; fingerprints, counts, and statuses come out."""
    publish_everything(tmp_path, context, outcomes)
    for receipt in (tmp_path / RUNS_DIR).rglob(TRAINING_RUN_FILE):
        assert banned not in receipt.read_text(encoding="utf-8")
    for record in (tmp_path / "ledger" / "training_run").glob("*.json"):
        assert banned not in record.read_text(encoding="utf-8")


def test_no_run_receipt_names_an_absolute_path(
    tmp_path: Path, context: TrainingContext, outcomes: tuple[Any, ...]
) -> None:
    """Artifact digests are relative to the run directory and nowhere else."""
    publish_everything(tmp_path, context, outcomes)
    for receipt in (tmp_path / RUNS_DIR).rglob(TRAINING_RUN_FILE):
        record = TrainingRunRecord.from_json(receipt.read_text(encoding="utf-8"))
        for relative, _ in record.artifact_digests:
            assert not relative.startswith("/")
            assert ".." not in relative.split("/")
        assert str(tmp_path) not in receipt.read_text(encoding="utf-8")
