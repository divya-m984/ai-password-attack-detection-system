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
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner, Result

from password_attack_detector.cli import app
from password_attack_detector.data.enums import CampaignStage, ScenarioType
from password_attack_detector.data.schemas import AuthEvent, GroundTruthLabel
from password_attack_detector.data.serialization import (
    write_events_parquet,
    write_labels_parquet,
)
from password_attack_detector.ml.experiments import RUNS_DIR, TRAINING_RUN_FILE
from password_attack_detector.ml.ledger import LEDGER_FILE
from tests.features.factories import make_event

runner = CliRunner()

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")

#: The fixture stream. Several campaigns, spread through it, so the validation
#: split carries more than one group and the boundary has somewhere to fall.
_CAMPAIGN_COUNT = 12
_CAMPAIGN_LENGTH = 4
_EVENT_COUNT = 160


def _repo_root() -> Path:
    """Return the repository root, located from this test file."""
    return Path(__file__).resolve().parents[2]


def _feature_config() -> dict[str, object]:
    """A CI-sized feature configuration: short windows, a small purge."""
    return {
        "windows": ["1m", "5m"],
        "cardinality_windows": ["5m"],
        "dispersion_windows": ["5m"],
        "device_session_windows": ["5m"],
        "pair_windows": ["5m"],
        "baseline": {
            "rate_reference_window": "5m",
            "min_events_per_user": 2,
            "min_events_per_source": 2,
            "response_time_min_events": 2,
        },
        "split": {
            "purge": "5m",
            "strict_isolation": True,
            "max_excluded_fraction": 0.6,
        },
    }


def _campaign_of(index: int) -> str | None:
    """Return the campaign an event index belongs to, or ``None`` if benign."""
    stride = _EVENT_COUNT // _CAMPAIGN_COUNT
    position = index % stride
    if position < _CAMPAIGN_LENGTH:
        return f"campaign-{index // stride:02d}"
    return None


def _events() -> list[AuthEvent]:
    """Return a deterministic event stream with periodic bursts of failures."""
    return [
        make_event(
            t=float(index) * 60.0,
            user=f"u{index % 5 + 1}",
            source=f"s{1 if _campaign_of(index) else index % 3 + 2}",
            device=f"d{index % 3 + 1}",
            outcome="failure" if _campaign_of(index) else "success",
            response_time_ms=40 if _campaign_of(index) else 200 + index % 50,
            country="US" if _campaign_of(index) else "GB",
            latitude=37.8 if _campaign_of(index) else 51.5,
            longitude=-122.4 if _campaign_of(index) else -0.1,
            key=str(index),
        )
        for index in range(_EVENT_COUNT)
    ]


def _labels(events: Sequence[AuthEvent]) -> list[GroundTruthLabel]:
    """Return ground truth pairing each burst with a campaign identifier.

    Two scenarios alternate across campaigns so the known-category head has more
    than one class to learn. Benign rows carry the generator's ``normal-<seed>``
    placeholder, exactly as real generator output does.
    """
    scenarios = (ScenarioType.BRUTE_FORCE, ScenarioType.PASSWORD_SPRAYING)
    labels = []
    for index, event in enumerate(events):
        campaign = _campaign_of(index)
        ordinal = 0 if campaign is None else int(campaign.split("-")[1])
        labels.append(
            GroundTruthLabel(
                event_id=event.event_id,
                campaign_id=campaign or "normal-864209",
                scenario=(
                    scenarios[ordinal % len(scenarios)]
                    if campaign is not None
                    else ScenarioType.NORMAL
                ),
                malicious=campaign is not None,
                supervised_training_eligible=True,
                generator_version="1.0.0",
                campaign_stage=None if campaign is None else CampaignStage.ACTIVE,
            )
        )
    return labels


def _invoke(*arguments: str) -> Result:
    """Run the CLI with *arguments* and return the result."""
    return runner.invoke(app, list(arguments))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a feature dataset and a drafted allowlist once for the module."""
    root = tmp_path_factory.mktemp("train-cli")
    events = _events()
    write_events_parquet(events, root / "events.parquet")
    write_labels_parquet(_labels(events), root / "labels.parquet")
    (root / "features.yaml").write_text(
        yaml.safe_dump(_feature_config()), encoding="utf-8"
    )

    built = _invoke(
        "features",
        "build",
        str(root / "events.parquet"),
        "--labels",
        str(root / "labels.parquet"),
        "--config",
        str(root / "features.yaml"),
        "-o",
        str(root / "processed"),
        "--reports-dir",
        str(root / "reports"),
    )
    assert built.exit_code == 0, built.output

    drafted = _invoke(
        "ml",
        "catalog",
        "--emit-allowlist",
        str(root / "allowlist.yaml"),
        "--feature-config",
        str(root / "features.yaml"),
    )
    assert drafted.exit_code == 0, drafted.output
    return root


def _train(workspace: Path, output_root: Path, **replace: str) -> Result:
    """Run ``ml train`` over the published workspace."""
    arguments = {
        "--features": str(workspace / "processed" / "feature_snapshots.parquet"),
        "--labels": str(workspace / "processed" / "feature_labels.parquet"),
        "--splits": str(workspace / "processed" / "feature_splits.parquet"),
        "--campaign-labels": str(workspace / "labels.parquet"),
        "--feature-manifest": str(workspace / "processed" / "feature_manifest.json"),
        "--allowlist": str(workspace / "allowlist.yaml"),
        "--feature-config": str(workspace / "features.yaml"),
        "--config": str(_repo_root() / "configs" / "ml" / "model-testing.yaml"),
        "--output-root": str(output_root),
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return _invoke("ml", "train", *flat)


@pytest.fixture(scope="module")
def trained(workspace: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train once and return the artifact root."""
    root = tmp_path_factory.mktemp("ml-artifacts")
    result = _train(workspace, root)
    assert result.exit_code == 0, result.output
    return root


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_training_completes_on_a_published_dataset(
    workspace: Path, tmp_path: Path
) -> None:
    """The whole chain, end to end, on real published artifacts."""
    result = _train(workspace, tmp_path / "artifacts")
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
    first = _train(workspace, root)
    assert first.exit_code == 0, first.output
    snapshot = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    second = _train(workspace, root)
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
    result = _train(
        workspace,
        tmp_path / "artifacts",
        **{"--labels": str(tmp_path / "absent.parquet")},
    )
    assert result.exit_code == 1
    assert "Input not found" in result.output
    assert "Traceback" not in result.output


def test_a_failed_audit_refuses_to_train(workspace: Path, tmp_path: Path) -> None:
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
    result = _train(workspace, root, **{"--feature-manifest": str(wrong)})
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
    result = _train(workspace, tmp_path / "artifacts")
    assert result.exit_code == 0
    output = result.stdout
    assert not _PSEUDONYM_RE.search(output)
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
    result = _train(workspace, trained)
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
    ignore = (_repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/*" in ignore


# ---------------------------------------------------------------------------
# ml experiments
# ---------------------------------------------------------------------------


def test_experiments_lists_the_published_runs(trained: Path) -> None:
    """Identity and status, one row per immutable record."""
    result = _invoke("ml", "experiments", "--output-root", str(trained))
    assert result.exit_code == 0, result.output
    assert "Experiment ledger" in result.stdout
    assert "M-010" in result.stdout
    # Column headers, since Rich truncates long cell values to fit the terminal.
    for header in ("Run", "Model", "Task", "Status", "Calibration", "Threshold"):
        assert header in result.stdout, header
    assert "immutable training_run record" in result.stdout


def test_experiments_shows_no_metric_and_no_identity(trained: Path) -> None:
    """A listing that ranked runs would be a champion selection by another name."""
    result = _invoke("ml", "experiments", "--output-root", str(trained))
    output = result.stdout.lower()
    for banned in ("brier", "ece", "auc", "accuracy", "recall", "precision"):
        assert banned not in output, banned
    assert not _PSEUDONYM_RE.search(result.stdout)
    assert "/home/" not in result.stdout


def test_experiments_on_an_empty_root_says_so(tmp_path: Path) -> None:
    """A ledger nobody has written to is empty, not broken."""
    result = _invoke("ml", "experiments", "--output-root", str(tmp_path / "nothing"))
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

    listed = _invoke("ml", "experiments", "--output-root", str(copied))
    assert listed.exit_code == 0
    assert "holds no training runs" in listed.stdout

    recovered = _invoke(
        "ml", "experiments", "--output-root", str(copied), "--reconcile"
    )
    assert recovered.exit_code == 0, recovered.output
    assert "previously unindexed run" in recovered.stdout
    assert "M-010" in recovered.stdout


def test_both_commands_are_registered() -> None:
    """``--help`` advertises exactly what this build implements."""
    result = _invoke("ml", "--help")
    assert result.exit_code == 0
    for command in (
        "train",
        "experiments",
        "catalog",
        "audit-features",
        "verify-manifest",
    ):
        assert command in result.stdout, command
    for absent in ("predict", "compare", "explain", "drift", "freeze"):
        assert absent not in result.stdout, absent
