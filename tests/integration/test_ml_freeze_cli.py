"""End-to-end tests for ``ml freeze-champion`` after a real selection.

The freeze is the last decision taken before the test split is opened, so the
command is tested for what it *refuses* as much as for what it writes: freezing
before selecting, freezing twice, and freezing with any kind of override. The
last of those is asserted on the command's own interface, because the safest
place for a bypass not to exist is the place a user would look for one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import Result

from password_attack_detector.ml.champion import CHAMPION_DIR, CHAMPION_LOCK_FILE
from tests.integration.ml_workspace import (
    PSEUDONYM_RE,
    build_workspace,
    invoke,
    repo_root,
    train,
)

ML_CONFIG = str(repo_root() / "configs" / "ml" / "model-testing.yaml")


def freeze(root: Path, *extra: str) -> Result:
    """Run ``ml freeze-champion`` over the artifact root."""
    return invoke(
        "ml",
        "freeze-champion",
        "--output-root",
        str(root),
        "--config",
        ML_CONFIG,
        *extra,
    )


@pytest.fixture(scope="module")
def selected(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train and select once, and return the artifact root."""
    workspace = build_workspace(tmp_path_factory.mktemp("freeze-cli"))
    root = tmp_path_factory.mktemp("freeze-artifacts")
    trained = train(workspace, root)
    assert trained.exit_code == 0, trained.output
    chosen = invoke(
        "ml",
        "select",
        "--output-root",
        str(root),
        "--config",
        ML_CONFIG,
        "--reports-dir",
        str(tmp_path_factory.mktemp("freeze-reports")),
    )
    assert chosen.exit_code == 0, chosen.output
    return root


@pytest.fixture(scope="module")
def frozen(selected: Path) -> tuple[Path, Result]:
    """Freeze once and return the artifact root with the command result."""
    result = freeze(selected)
    return (selected, result)


# ---------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------


def test_freezing_writes_a_lock_for_the_selected_champion(
    frozen: tuple[Path, Result],
) -> None:
    """One lock, in one scope, naming the model the selection chose."""
    root, result = frozen
    assert result.exit_code == 0, result.output
    locks = sorted((root / CHAMPION_DIR).rglob(CHAMPION_LOCK_FILE))
    assert len(locks) == 1
    lock = json.loads(locks[0].read_text(encoding="utf-8"))
    assert lock["task"] == "binary_malicious"
    assert lock["catalog_model_id"] == "M-010"
    assert lock["model_family"] != "prior_baseline"


def test_the_lock_names_everything_a_later_evaluation_would_load(
    frozen: tuple[Path, Result],
) -> None:
    """Model, preprocessing, operating point, and the contracts behind them."""
    root, _ = frozen
    lock = json.loads(
        next((root / CHAMPION_DIR).rglob(CHAMPION_LOCK_FILE)).read_text(
            encoding="utf-8"
        )
    )
    for field in (
        "model_content_fingerprint",
        "model_manifest_fingerprint",
        "preprocessor_fingerprint",
        "binary_threshold_fingerprint",
        "eligible_feature_list_fingerprint",
        "validation_partition_fingerprint",
        "gate_config_fingerprint",
        "dependency_contract_fingerprint",
        "serializer_id",
    ):
        assert lock[field], field


def test_the_lock_carries_no_metric(frozen: tuple[Path, Result]) -> None:
    """It says which model was chosen; the selection record says why."""
    root, _ = frozen
    lock = json.loads(
        next((root / CHAMPION_DIR).rglob(CHAMPION_LOCK_FILE)).read_text(
            encoding="utf-8"
        )
    )
    for absent in (
        "detection_rate",
        "false_positive_rate",
        "precision",
        "expected_calibration_error",
    ):
        assert absent not in lock, absent


def test_the_freeze_is_indexed_in_the_ledger(frozen: tuple[Path, Result]) -> None:
    """An immutable receipt naming the selection that justified it."""
    root, _ = frozen
    receipts = sorted((root / "ledger" / "champion_freeze").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["validation_selection_id"]
    assert receipt["selected_run_id"]
    assert receipt["scope_key"]


def test_freezing_again_writes_nothing_new(frozen: tuple[Path, Result]) -> None:
    """Idempotent, so a retried command is not a second promotion."""
    root, _ = frozen
    again = freeze(root)
    assert again.exit_code == 0, again.output
    written = next(
        line for line in again.stdout.splitlines() if "Newly written" in line
    )
    assert written.strip().endswith("no")
    assert len(sorted((root / "ledger" / "champion_freeze").glob("*.json"))) == 1
    assert len(sorted((root / CHAMPION_DIR).rglob(CHAMPION_LOCK_FILE))) == 1


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_freezing_before_selecting_freezes_nothing(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """No selection is not an empty selection, and it promotes nothing."""
    workspace = build_workspace(tmp_path_factory.mktemp("unselected-cli"))
    root = tmp_path_factory.mktemp("unselected-artifacts")
    assert train(workspace, root).exit_code == 0

    result = freeze(root)
    assert result.exit_code == 2, result.output
    assert "Nothing to freeze" in result.output
    assert not list(root.rglob(CHAMPION_LOCK_FILE))


def test_an_unknown_selection_identifier_is_refused(
    frozen: tuple[Path, Result],
) -> None:
    """Freezing names a recorded comparison or it does not proceed."""
    root, _ = frozen
    result = freeze(root, "--selection", "0" * 36)
    assert result.exit_code != 0
    assert "No binary validation selection" in result.output


def test_there_is_no_force_option() -> None:
    """Asserted on the interface, where somebody looking for one would look."""
    result = invoke("ml", "freeze-champion", "--help")
    assert result.exit_code == 0
    for absent in ("--force", "--allow", "--override", "--skip", "--test"):
        assert absent not in result.stdout, absent


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_freezing_produces_no_test_evaluation(frozen: tuple[Path, Result]) -> None:
    """The lock is what a later evaluation may run, not the evaluation."""
    root, result = frozen
    assert not (root / "ledger" / "test_evaluation").exists()
    assert not list(root.rglob("test_evaluation.json"))
    lock = next((root / CHAMPION_DIR).rglob(CHAMPION_LOCK_FILE)).read_text(
        encoding="utf-8"
    )
    for banned in ("test_metrics", "test_evaluation", "holdout"):
        assert banned not in lock.lower(), banned
    assert "no test evaluation exists" in result.stdout


def test_no_pseudonym_or_absolute_path_reaches_the_terminal(
    frozen: tuple[Path, Result],
) -> None:
    """Freeze output is identifiers, digests, and one path, displayed short."""
    _, result = frozen
    assert not PSEUDONYM_RE.search(result.output)
    assert str(Path.home()) not in result.output
