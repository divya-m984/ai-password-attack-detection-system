"""End-to-end tests for ``ml validate`` over a real prediction publication.

``ml validate`` is not ``ml verify-manifest`` under another name, and the first
test says so: one checks a published *model* directory, the other a published
*prediction* directory, and neither accepts the other's input.

The rest is tamper detection. Each test breaks one thing in a valid publication
and asserts a non-zero exit and the stable code that names it, because a command
that reported a generic failure would leave whoever is investigating no better
off than before they ran it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import Result

from password_attack_detector.ml.prediction_manifest import (
    BINARY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    QUALITY_REPORT_JSON_FILE,
    VALIDATION_RESULT_FILE,
)
from tests.integration.ml_workspace import (
    PSEUDONYM_RE,
    build_workspace,
    freeze,
    invoke,
    predict,
)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Publish a dataset, freeze a champion, and predict once."""
    workspace = build_workspace(tmp_path_factory.mktemp("validate-workspace"))
    root = tmp_path_factory.mktemp("validate-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("validate-reports"))
    result = predict(workspace, root)
    assert result.exit_code == 0, result.output
    return (workspace, root)


@pytest.fixture
def published(frozen: tuple[Path, Path], tmp_path: Path) -> Path:
    """Return a writable copy of the artifact root, one per test."""
    _, root = frozen
    copied = tmp_path / "artifacts"
    shutil.copytree(root, copied)
    return copied


def directory_of(root: Path) -> Path:
    """Return the single published prediction directory."""
    return next((root / PREDICTIONS_DIR).iterdir())


def validate(root: Path, *extra: str) -> Result:
    """Run ``ml validate`` over *root*."""
    return invoke("ml", "validate", "--output-root", str(root), *extra)


# ---------------------------------------------------------------------------
# The valid case
# ---------------------------------------------------------------------------


def test_a_valid_publication_validates(published: Path) -> None:
    """Every check run, every mandatory one passing."""
    result = validate(published)
    assert result.exit_code == 0, result.output
    assert "Validation PASS" in result.stdout
    for code in ("M001", "M009", "M017", "M024", "M028"):
        assert code in result.stdout, code


def test_the_report_says_what_validation_is_not(published: Path) -> None:
    """Structural validity is not predictive quality, and the command says so."""
    result = validate(published)
    assert "Structural validity is not predictive quality" in result.stdout
    assert "No label was read" in result.stdout


def test_validation_is_not_model_manifest_verification(published: Path) -> None:
    """Two commands, two inputs, and neither accepts the other's."""
    model_directory = next((published / "runs").rglob("model_manifest.json")).parent
    verified = invoke("ml", "verify-manifest", str(model_directory))
    assert verified.exit_code == 0, verified.output
    assert "Model artifact" in verified.stdout
    assert "M001" not in verified.stdout

    predictions = validate(published)
    assert "Prediction validation" in predictions.stdout


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_a_missing_manifest_fails(published: Path) -> None:
    """And leaves every dependent check honestly unevaluated."""
    (directory_of(published) / PREDICTION_MANIFEST_FILE).unlink()
    result = validate(published)
    assert result.exit_code != 0
    assert "No complete prediction publication" in result.output


def test_a_tampered_manifest_fails(published: Path) -> None:
    """The seal is recomputed, not trusted."""
    path = directory_of(published) / PREDICTION_MANIFEST_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["row_count"] = payload["row_count"] + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = validate(published)
    assert result.exit_code != 0
    assert "M001" in result.stdout
    assert "Validation FAILED" in result.output


def test_a_tampered_row_table_fails(published: Path) -> None:
    """The declared digest is checked against the bytes on disk."""
    path = directory_of(published) / BINARY_PREDICTION_FILE
    path.write_bytes(path.read_bytes() + b"\x00")
    result = validate(published)
    assert result.exit_code != 0
    assert "M006" in result.stdout


def test_an_unexpected_file_fails(published: Path) -> None:
    """An extra file in a verified directory is a mistake or an attempt."""
    (directory_of(published) / "extra.json").write_text("{}", encoding="utf-8")
    result = validate(published)
    assert result.exit_code != 0
    assert "M004" in result.stdout


def test_a_tampered_quality_report_fails(published: Path) -> None:
    """The bound aggregate reports are covered by the same digests."""
    path = directory_of(published) / QUALITY_REPORT_JSON_FILE
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = validate(published)
    assert result.exit_code != 0
    assert "M006" in result.stdout


def test_a_deleted_validation_result_fails(published: Path) -> None:
    """A declared artifact that is not there is not an optional one."""
    (directory_of(published) / VALIDATION_RESULT_FILE).unlink()
    result = validate(published)
    assert result.exit_code != 0
    assert "M004" in result.stdout


# ---------------------------------------------------------------------------
# Selection and refusals
# ---------------------------------------------------------------------------


def test_validating_with_no_publications_is_refused(tmp_path: Path) -> None:
    """A validation over nothing is not a validation with no findings."""
    result = validate(tmp_path / "empty")
    assert result.exit_code != 0
    assert "No predictions have been published" in result.output


def test_naming_an_unknown_publication_is_refused(published: Path) -> None:
    """A typo is refused rather than resolved to whatever is nearby."""
    result = validate(published, "--prediction", "not-a-prediction")
    assert result.exit_code != 0
    assert "No prediction publication with that identifier" in result.output


def test_several_publications_require_one_to_be_named(
    frozen: tuple[Path, Path], published: Path
) -> None:
    """Acting on whichever came first would depend on directory order."""
    workspace, _ = frozen
    assert predict(workspace, published, split="validation").exit_code == 0
    result = validate(published)
    assert result.exit_code != 0
    assert "--prediction" in result.output

    named = validate(published, "--prediction", directory_of(published).name)
    assert named.exit_code == 0, named.output


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_the_command_takes_no_label_argument() -> None:
    """Validation reads the artifact, never the answers."""
    result = invoke("ml", "validate", "--help")
    assert result.exit_code == 0
    _, _, options = result.stdout.partition("Options")
    for absent in ("--labels", "--truth", "--test", "--force"):
        assert absent not in options, absent


def test_nothing_printed_carries_an_outcome_metric(published: Path) -> None:
    """Validation establishes integrity, never performance."""
    result = validate(published)
    lowered = result.output.lower()
    for banned in ("accuracy", "roc_auc", "brier", "confusion", "f1 "):
        assert banned not in lowered, banned


def test_no_anchor_pseudonym_or_absolute_path_reaches_the_terminal(
    published: Path,
) -> None:
    """Stable codes, counts, and declared names."""
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
    )

    result = validate(published)
    assert not PSEUDONYM_RE.search(result.output)
    assert str(Path.home()) not in result.output
    for row in read_binary_predictions(
        directory_of(published) / BINARY_PREDICTION_FILE
    )[:20]:
        assert row.anchor_event_id not in result.output


def test_validating_changes_nothing_on_disk(published: Path) -> None:
    """Reading an artifact does not modify it."""
    directory = directory_of(published)
    before = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    assert validate(published).exit_code == 0
    after = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    assert after == before


def test_the_module_entry_point_reaches_the_same_command() -> None:
    """``python -m`` and the console script are one program."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "password_attack_detector", "ml", "validate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--prediction" in completed.stdout
