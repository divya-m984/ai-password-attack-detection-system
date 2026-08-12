"""End-to-end tests for ``ml train`` and ``ml experiments`` over a real dataset.

The whole chain, nothing mocked: generate a micro event stream, publish it
through ``features build``, draft an allowlist, train every configured
candidate, publish immutable runs, and list the ledger.

The dataset is deliberately tiny -- 160 events over about two and a half hours.
The 720-hour development workflow is not run here and never will be: a contract
test that needs a month of traffic to express itself is testing the generator.
What *is* asserted end to end is the orchestration, the artifact set, the
sanitisation of terminal output, and the absences -- no champion, no test
evaluation, no lock file.

The dataset and the training invocation live in
:mod:`tests.integration.ml_workspace`, shared with the Milestone 7 command
suites so the three cannot drift into disagreeing about what a published
experiment looks like.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from password_attack_detector.ml.experiments import RUNS_DIR, TRAINING_RUN_FILE
from password_attack_detector.ml.ledger import LEDGER_FILE
from tests.integration.ml_workspace import (
    PSEUDONYM_RE,
    build_workspace,
    invoke,
    repo_root,
    train,
)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a feature dataset and a drafted allowlist once for the module."""
    return build_workspace(tmp_path_factory.mktemp("train-cli"))


@pytest.fixture(scope="module")
def trained(workspace: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train once and return the artifact root."""
    root = tmp_path_factory.mktemp("ml-artifacts")
    result = train(workspace, root)
    assert result.exit_code == 0, result.output
    return root


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_training_completes_on_a_published_dataset(
    workspace: Path, tmp_path: Path
) -> None:
    """The whole chain, end to end, on real published artifacts."""
    result = train(workspace, tmp_path / "artifacts")
    assert result.exit_code == 0, result.output
    assert "Training runs" in result.stdout


def test_every_configured_candidate_appears_in_the_output(trained: Path) -> None:
    """A candidate that could not be trained is reported, never dropped."""
    ledger = trained / "ledger" / "training_run"
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(ledger.glob("*.json"))
    ]
    assert records
    entries = {(item["catalog_model_id"], item["task"]) for item in records}
    assert ("M-010", "binary_malicious") in entries
    assert ("M-000", "binary_malicious") in entries
    assert ("M-001", "binary_malicious") in entries

    # M-001's reviewed column is named in the shipped configuration, so the
    # baseline is trained rather than reported unavailable for want of a
    # setting. On a 160-event micro-dataset its cut point puts every
    # validation-A row on one side, so calibration then has one distinct score
    # to work with and says so -- a real downstream outcome, not a missing
    # configuration. The path where it completes is exercised in the unit suite,
    # on a fixture sized for it.
    baseline = {
        item["status"] for item in records if item["catalog_model_id"] == "M-001"
    }
    assert "unavailable" not in baseline
    assert baseline == {"calibration_unavailable"}
    fitted = [item for item in records if item["catalog_model_id"] == "M-001"]
    assert all(
        item["failing_requirements"] == ["min_distinct_scores"] for item in fitted
    )


def test_the_reference_baseline_is_trained_and_recorded(trained: Path) -> None:
    """M-000 is fitted, published, and never calibrated -- by contract.

    Under this configuration's false-positive ceiling a constant score has no
    feasible operating point, so the run stops at ``threshold_unavailable``.
    What matters here is what it did *not* do: it was not stopped by the
    absence of a calibrator, and no calibration artifact was invented for it.
    """
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((trained / "ledger" / "training_run").glob("*.json"))
    ]
    baseline = next(
        item
        for item in records
        if item["catalog_model_id"] == "M-000" and item["task"] == "binary_malicious"
    )
    assert baseline["reference_baseline"] is True
    assert baseline["champion_eligible"] is False
    assert baseline["calibration_method"] == "none"
    assert baseline["calibration_status"] == "not_calibrated"
    assert baseline["status"] != "calibration_unavailable"
    assert not any(
        "calibration" in relative for relative, _ in baseline["artifact_digests"]
    )


def test_a_completed_run_publishes_its_whole_artifact_set(trained: Path) -> None:
    """Model, calibrator, reports, threshold, and a receipt written last."""
    completed = [
        path.parent
        for path in (trained / RUNS_DIR).rglob(TRAINING_RUN_FILE)
        if json.loads(path.read_text(encoding="utf-8"))["status"] == "completed"
        and json.loads(path.read_text(encoding="utf-8"))["task"] == "binary_malicious"
    ]
    assert completed
    directory = completed[0]
    for relative in (
        "model/model.json",
        "model/arrays.npz",
        "model/preprocessor.json",
        "model/model_manifest.json",
        "calibration/calibration_state.json",
        "calibration/calibration_fit_diagnostic.json",
        "calibration/calibration_validation_report.json",
        "thresholds/binary_threshold.json",
        TRAINING_RUN_FILE,
    ):
        assert (directory / relative).is_file(), relative


def test_the_ledger_is_written_and_readable(trained: Path) -> None:
    """One file per record, plus the ledger's own contract file."""
    assert (trained / "ledger" / LEDGER_FILE).is_file()
    records = sorted((trained / "ledger" / "training_run").glob("*.json"))
    assert records
    for path in records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["record_type"] == "training_run"
        assert path.stem == payload["identity"]["run_id"]


def test_training_twice_is_idempotent(workspace: Path, tmp_path: Path) -> None:
    """The second run confirms the first rather than republishing it."""
    root = tmp_path / "artifacts"
    first = train(workspace, root)
    assert first.exit_code == 0, first.output
    snapshot = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    second = train(workspace, root)
    assert second.exit_code == 0, second.output
    assert {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == snapshot


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_missing_input_is_refused_without_a_traceback(
    workspace: Path, tmp_path: Path
) -> None:
    """A typo names the missing input and nothing else."""
    result = train(
        workspace,
        tmp_path / "artifacts",
        **{"--labels": str(tmp_path / "absent.parquet")},
    )
    assert result.exit_code == 1
    assert "Input not found" in result.output
    assert "Traceback" not in result.output


def test_a_failed_audit_refuses_totrain(workspace: Path, tmp_path: Path) -> None:
    """A model fitted on a dataset that failed the audit is one nobody should use.

    The manifest is what the fingerprint provenance check compares against, so a
    manifest describing a different dataset fails the audit -- and training stops
    before anything is fitted.
    """
    wrong = tmp_path / "wrong_manifest.json"
    payload = json.loads(
        (workspace / "processed" / "feature_manifest.json").read_text(encoding="utf-8")
    )
    payload["feature_catalog_fingerprint"] = "0" * 64
    wrong.write_text(json.dumps(payload), encoding="utf-8")

    root = tmp_path / "artifacts"
    result = train(workspace, root, **{"--feature-manifest": str(wrong)})
    assert result.exit_code == 1
    assert "Training refused" in result.output
    assert not (root / RUNS_DIR).exists()


# ---------------------------------------------------------------------------
# What the output must not contain
# ---------------------------------------------------------------------------


def test_the_output_carries_no_identity_or_absolute_path(
    workspace: Path, tmp_path: Path
) -> None:
    """Run identifiers and statuses; never a row, a pseudonym, or a home directory."""
    result = train(workspace, tmp_path / "artifacts")
    assert result.exit_code == 0
    output = result.stdout
    assert not PSEUDONYM_RE.search(output)
    assert "/home/" not in output
    assert str(Path.home()) not in output
    for banned in ("campaign-", "normal-864209", "coefficient", "intercept"):
        assert banned not in output, banned


def test_the_output_reports_no_performance_figure(
    trained: Path, workspace: Path
) -> None:
    """Milestone 6 records what was run, not how well it did.

    Matched on whole words: run identifiers are hexadecimal, so a substring
    search for a short metric name finds one in roughly every other UUID.
    """
    result = train(workspace, trained)
    output = result.stdout.lower()
    for banned in (
        "accuracy",
        "precision",
        "recall",
        "brier",
        "auc",
        "calibration error",
    ):
        assert not re.search(rf"\b{banned}\b", output), banned


def test_no_champion_is_selected_and_no_lock_is_written(trained: Path) -> None:
    """The absences Milestone 6 is defined by."""
    assert not list(trained.rglob("champion.lock"))
    assert not list(trained.rglob("champion*.json"))
    for path in (trained / "ledger").rglob("*.json"):
        payload = path.read_text(encoding="utf-8")
        assert "champion_status" not in payload
        assert '"is_champion"' not in payload


def test_no_test_evaluation_is_produced(trained: Path) -> None:
    """No test record, no test metric, and nowhere to put one."""
    assert not (trained / "ledger" / "test_evaluation").exists()
    assert not (trained / "ledger" / "champion_freeze").exists()
    # Checked over the records themselves. The ledger's own contract file names
    # every declared record *type*, which is how a reader knows what the ledger
    # is for -- and is not a record of one.
    for path in (trained / "ledger" / "training_run").glob("*.json"):
        payload = path.read_text(encoding="utf-8")
        assert "test_metrics" not in payload
        assert '"record_type":"test_evaluation"' not in payload


def test_generated_artifacts_stay_under_the_configured_output_root(
    trained: Path,
) -> None:
    """Nothing is written outside the root the caller named."""
    assert (trained / RUNS_DIR).is_dir()
    assert (trained / "ledger").is_dir()
    assert {path.name for path in trained.iterdir()} == {RUNS_DIR, "ledger"}


def test_the_repository_artifact_root_is_ignored_by_git() -> None:
    """Run artifacts are generated output and are never committed."""
    ignore = (repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/*" in ignore


# ---------------------------------------------------------------------------
# ml experiments
# ---------------------------------------------------------------------------


def test_experiments_lists_the_published_runs(trained: Path) -> None:
    """Identity and status, one row per immutable record."""
    result = invoke("ml", "experiments", "--output-root", str(trained))
    assert result.exit_code == 0, result.output
    assert "Experiment ledger" in result.stdout
    assert "M-010" in result.stdout
    # Column headers, since Rich truncates long cell values to fit the terminal.
    for header in ("Run", "Model", "Task", "Status", "Calibration", "Threshold"):
        assert header in result.stdout, header
    assert "immutable training_run record" in result.stdout


def test_experiments_shows_no_metric_and_no_identity(trained: Path) -> None:
    """A listing that ranked runs would be a champion selection by another name."""
    result = invoke("ml", "experiments", "--output-root", str(trained))
    output = result.stdout.lower()
    for banned in ("brier", "ece", "auc", "accuracy", "recall", "precision"):
        assert banned not in output, banned
    assert not PSEUDONYM_RE.search(result.stdout)
    assert "/home/" not in result.stdout


def test_experiments_on_an_empty_root_says_so(tmp_path: Path) -> None:
    """A ledger nobody has written to is empty, not broken."""
    result = invoke("ml", "experiments", "--output-root", str(tmp_path / "nothing"))
    assert result.exit_code == 0, result.output
    assert "holds no training runs" in result.stdout


def test_experiments_can_reconcile_an_unindexed_run(
    trained: Path, tmp_path: Path
) -> None:
    """The recovery path, driven from the CLI and appending only."""
    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(trained, copied)
    shutil.rmtree(copied / "ledger")

    listed = invoke("ml", "experiments", "--output-root", str(copied))
    assert listed.exit_code == 0
    assert "holds no training runs" in listed.stdout

    recovered = invoke("ml", "experiments", "--output-root", str(copied), "--reconcile")
    assert recovered.exit_code == 0, recovered.output
    assert "previously unindexed run" in recovered.stdout
    assert "M-010" in recovered.stdout


def test_both_commands_are_registered() -> None:
    """``--help`` advertises exactly what this build implements."""
    result = invoke("ml", "--help")
    assert result.exit_code == 0
    for command in (
        "train",
        "experiments",
        "catalog",
        "audit-features",
        "verify-manifest",
        "select",
        "freeze-champion",
        "predict",
        "validate",
        "profile",
    ):
        assert command in result.stdout, command
    for absent in ("evaluate", "compare", "explain", "drift"):
        assert absent not in result.stdout, absent
