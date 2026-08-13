"""``ml verify-manifest``: exit codes, sanitised output, and what it refuses to print.

The command exists to be run against a directory somebody else produced, so two
properties matter beyond correctness. It must **never execute any part of the
artifact**, and it must **never print** a coefficient, a tree value, a training
row, or an absolute path.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from password_attack_detector.cli import app
from password_attack_detector.ml.enums import MLTask
from password_attack_detector.ml.models import (
    LogisticRegressionAdapter,
    RandomForestAdapter,
)
from password_attack_detector.ml.serialization import ARRAYS_FILE, MANIFEST_FILE
from tests.ml.models import prepare, publish

runner = CliRunner()

#: Anything shaped like an identifier or a filesystem path.
_PSEUDONYM = re.compile(r"\b(?:u|s|d|sess|usr|src|dev):?[0-9a-f]{8,}\b")


def invoke(*arguments: str) -> Result:
    """Run the CLI and return the result."""
    return runner.invoke(app, list(arguments))


@pytest.fixture
def model(tmp_path: Path) -> Path:
    """Return a published logistic model."""
    batch = prepare(count=140)
    fitted = LogisticRegressionAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    return publish(tmp_path / "model", fitted, batch.preprocessor)


@pytest.fixture
def forest(tmp_path: Path) -> Path:
    """Return a published forest, whose arrays hold many tree values."""
    batch = prepare(count=140)
    fitted = RandomForestAdapter(n_estimators=8, max_depth=4).fit(
        batch.batch, task=MLTask.BINARY_MALICIOUS
    )
    return publish(tmp_path / "forest", fitted, batch.preprocessor)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_a_valid_artifact_exits_zero(model: Path) -> None:
    """The happy path reports PASS."""
    result = invoke("ml", "verify-manifest", str(model))
    assert result.exit_code == 0, result.output
    assert "Verification PASS" in result.output


def test_a_tampered_artifact_exits_non_zero(model: Path) -> None:
    """One appended byte is enough, and the code is stable."""
    path = model / ARRAYS_FILE
    path.write_bytes(path.read_bytes() + b"\n")
    result = invoke("ml", "verify-manifest", str(model))
    assert result.exit_code == 1
    assert "CHECKSUM_MISMATCH" in result.output


def test_a_missing_directory_exits_non_zero(tmp_path: Path) -> None:
    """A path that is not there is a failure, not an empty success."""
    result = invoke("ml", "verify-manifest", str(tmp_path / "absent"))
    assert result.exit_code == 1
    assert "MODEL_DIR_MISSING" in result.output


def test_a_malformed_manifest_exits_non_zero(model: Path) -> None:
    """Unparseable JSON fails before anything is interpreted."""
    (model / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    result = invoke("ml", "verify-manifest", str(model))
    assert result.exit_code == 1
    assert "MANIFEST_INVALID" in result.output


def test_an_unexpected_file_exits_non_zero(model: Path) -> None:
    """The declared artifact set is closed."""
    (model / "extra.bin").write_bytes(b"payload")
    result = invoke("ml", "verify-manifest", str(model))
    assert result.exit_code == 1
    assert "MODEL_FILE_UNEXPECTED" in result.output


def test_a_symlinked_artifact_exits_non_zero(model: Path, tmp_path: Path) -> None:
    """A member that could point anywhere is refused."""
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}", encoding="utf-8")
    target = model / "preprocessor.json"
    target.unlink()
    target.symlink_to(elsewhere)
    result = invoke("ml", "verify-manifest", str(model))
    assert result.exit_code == 1
    assert "MODEL_FILE_SYMLINK" in result.output


# ---------------------------------------------------------------------------
# What is printed
# ---------------------------------------------------------------------------


def test_the_output_reports_identity_and_contract(model: Path) -> None:
    """Model id, family, task, schema versions, serializer, and a file count."""
    result = invoke("ml", "verify-manifest", str(model))
    payload = json.loads((model / "model.json").read_text(encoding="utf-8"))
    assert payload["model_id"] in result.output
    assert "logistic_regression" in result.output
    assert "binary_malicious" in result.output
    assert "json_linear_v1" in result.output
    assert "Files" in result.output


def test_the_output_prints_no_coefficient(model: Path) -> None:
    """A fitted parameter is model state, not something to render."""
    result = invoke("ml", "verify-manifest", str(model))
    from password_attack_detector.ml.npz import read_npz_bytes

    arrays = read_npz_bytes((model / ARRAYS_FILE).read_bytes())
    for value in arrays["coefficients"].ravel()[:8]:
        assert f"{float(value):.6f}" not in result.output


def test_the_output_prints_no_tree_value(forest: Path) -> None:
    """Aggregate tree parameters live in the archive and stay there."""
    result = invoke("ml", "verify-manifest", str(forest))
    from password_attack_detector.ml.npz import read_npz_bytes

    arrays = read_npz_bytes((forest / ARRAYS_FILE).read_bytes())
    for value in arrays["split_threshold"].ravel()[:8]:
        assert f"{float(value):.6f}" not in result.output


def test_the_output_prints_no_absolute_path(model: Path) -> None:
    """Not the model's own directory, and not a home directory."""
    result = invoke("ml", "verify-manifest", str(model))
    assert str(model.resolve()) not in result.output
    assert "/home/" not in result.output


def test_the_output_prints_no_identifier(model: Path) -> None:
    """No anchor, no campaign, no pseudonym, no training row."""
    result = invoke("ml", "verify-manifest", str(model))
    assert "anchor_event_id" not in result.output
    assert "campaign_id" not in result.output
    assert not _PSEUDONYM.search(result.output)


def test_a_failure_message_carries_no_path(model: Path) -> None:
    """A failure is often the first thing pasted into a ticket."""
    (model / MANIFEST_FILE).write_text("{oops", encoding="utf-8")
    result = invoke("ml", "verify-manifest", str(model))
    assert str(model.resolve()) not in result.output
    assert "JSONDecodeError" in result.output


def test_the_output_says_no_champion_has_been_selected(model: Path) -> None:
    """Passing verification is not a promotion, and the caveat says so."""
    result = invoke("ml", "verify-manifest", str(model))
    # The console wraps, so the phrase is matched against collapsed whitespace.
    collapsed = " ".join(result.output.lower().split())
    assert "no champion has been selected" in collapsed
    assert "no calibrator has been fitted" in collapsed


# ---------------------------------------------------------------------------
# Both entry points
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def test_the_module_entry_point_verifies(model: Path) -> None:
    """``python -m`` reaches the same command as the console script."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "password_attack_detector",
            "ml",
            "verify-manifest",
            str(model),
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Verification PASS" in completed.stdout


def test_the_module_entry_point_exits_non_zero_on_failure(model: Path) -> None:
    """And the failure propagates rather than being swallowed."""
    path = model / ARRAYS_FILE
    path.write_bytes(path.read_bytes() + b"\n")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "password_attack_detector",
            "ml",
            "verify-manifest",
            str(model),
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        check=False,
    )
    assert completed.returncode == 1


def test_the_command_appears_in_help() -> None:
    """Advertised, alongside the two commands that were already there."""
    result = invoke("ml", "--help")
    assert result.exit_code == 0
    assert "verify-manifest" in result.output
    assert "catalog" in result.output
    assert "audit-features" in result.output


def test_the_offline_command_surface_is_complete() -> None:
    """Everything the layer does is registered; nothing it must not do is.

    ``predict`` publishes what the model said without opening a label,
    ``evaluate`` scores it once against a lineage frozen beforehand, and
    ``explain`` and ``drift`` describe a model and a population without reading
    a label at all. What stays absent is anything that would make this layer
    online or self-modifying.
    """
    result = invoke("ml", "--help")
    commands = result.output.split("Commands")[-1]
    for absent in ("serve", "deploy", "retrain", "promote"):
        assert absent not in commands, absent
    for present in ("predict", "evaluate", "compare", "explain", "drift"):
        assert present in commands, present


def test_the_existing_commands_still_work() -> None:
    """The catalog command is untouched by this milestone."""
    result = invoke("ml", "catalog")
    assert result.exit_code == 0, result.output
    assert "M-010" in result.output


def test_verification_executes_nothing_from_the_artifact(model: Path) -> None:
    """A model directory is data. Rendering it must never run any of it.

    A ``__init__.py`` and a ``sitecustomize.py`` are planted in the directory;
    if verification ever imported from there the sentinel would be written. It
    is not, and the extra files make the command fail for the right reason.
    """
    sentinel = model.parent / "executed.flag"
    payload = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n"
    (model / "__init__.py").write_text(payload, encoding="utf-8")
    (model / "sitecustomize.py").write_text(payload, encoding="utf-8")
    result = invoke("ml", "verify-manifest", str(model))
    assert result.exit_code == 1
    assert "MODEL_FILE_UNEXPECTED" in result.output
    assert not sentinel.exists()
