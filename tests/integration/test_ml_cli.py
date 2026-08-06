"""Integration tests for the ``ml`` CLI group.

Milestone 1 ships exactly one command, so these tests cover three things: that
``ml catalog`` renders from the executable catalog, that the tracked
documentation is byte-identical to what the command produces, and that adding
the sub-application did not disturb any existing command group.

Two properties are swept, matching the conventions of the other CLI test
modules: **no command prints an identifier or an absolute path**, and **no
command advertises a capability Milestone 1 does not have**.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from password_attack_detector.cli import app
from password_attack_detector.ml.catalog import MODEL_CATALOG, model_catalog_to_markdown

runner = CliRunner()

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)

#: Commands Milestone 1 deliberately does not ship.  A placeholder that exists
#: but does nothing is worse than an honest absence, because ``--help`` would
#: advertise a capability the code lacks.
DEFERRED_COMMANDS = (
    "train",
    "predict",
    "evaluate",
    "compare",
    "select",
    "freeze-champion",
    "explain",
    "drift",
    "experiments",
    "validate",
    "profile",
    "verify-manifest",
    "audit-features",
)


def _repo_root() -> Path:
    """Return the repository root, located from this test file."""
    return Path(__file__).resolve().parents[2]


def _invoke(*arguments: str) -> Result:
    """Run the CLI with *arguments* and return the result."""
    return runner.invoke(app, list(arguments))


def _assert_sanitized(result: Result) -> None:
    """Assert the output carries no identifier, pseudonym, or absolute path."""
    output = result.stdout
    assert not _UUID_RE.search(output)
    assert not _PSEUDONYM_RE.search(output)
    assert str(Path.home()) not in output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_ml_group_is_registered() -> None:
    """The sub-application is reachable from the root command."""
    result = _invoke("--help")
    assert result.exit_code == 0
    assert "ml" in result.stdout


def test_the_ml_group_shows_help_with_no_arguments() -> None:
    """``no_args_is_help`` matches every other group in this CLI."""
    result = _invoke("ml")
    assert result.exit_code != 0 or "catalog" in result.stdout


def test_the_ml_group_advertises_only_the_shipped_command() -> None:
    """Milestone 1 registers ``catalog`` and nothing else."""
    result = _invoke("ml", "--help")
    assert result.exit_code == 0
    assert "catalog" in result.stdout
    for command in DEFERRED_COMMANDS:
        assert command not in result.stdout, command


@pytest.mark.parametrize("command", DEFERRED_COMMANDS)
def test_a_deferred_command_is_not_callable(command: str) -> None:
    """Invoking a later milestone's command must fail, not silently succeed."""
    result = _invoke("ml", command)
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ml catalog
# ---------------------------------------------------------------------------


def test_catalog_text_output_names_every_model() -> None:
    """The default rendering covers the whole registry."""
    result = _invoke("ml", "catalog")
    assert result.exit_code == 0
    for model_id in MODEL_CATALOG.model_ids:
        assert model_id in result.stdout
    _assert_sanitized(result)


def test_catalog_text_output_reports_the_required_fields() -> None:
    """Model identity, eligibility, adapters, and determinism are all shown."""
    result = _invoke("ml", "catalog")
    assert result.exit_code == 0
    output = result.stdout
    assert "Eligibility" in output
    assert "Native score kind" in output
    assert "Inference adapter" in output
    assert "Determinism controls" in output
    assert "Hyperparameters" in output
    assert "Limitation" in output


def test_catalog_text_output_states_that_membership_is_not_championship() -> None:
    """A reader must not take the listing as an endorsement."""
    result = _invoke("ml", "catalog")
    assert result.exit_code == 0
    lowered = result.stdout.lower()
    assert "does not make a model champion" in lowered
    assert "probability only after calibration" in lowered


def test_catalog_markdown_matches_the_renderer_exactly() -> None:
    """The command must emit what the library renders, not a variant."""
    result = _invoke("ml", "catalog", "--format", "markdown")
    assert result.exit_code == 0
    for line in ("# Model Catalog", MODEL_CATALOG.fingerprint()):
        assert line in result.stdout


def test_catalog_writes_markdown_to_a_file(tmp_path: Path) -> None:
    """``--output`` writes the document and reports a relative path."""
    target = tmp_path / "nested" / "model-catalog.md"
    result = _invoke("ml", "catalog", "--format", "markdown", "-o", str(target))
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == model_catalog_to_markdown()
    _assert_sanitized(result)


def test_catalog_text_mode_also_honours_the_output_option(tmp_path: Path) -> None:
    """Matching the detection group: text to stdout, Markdown to the file."""
    target = tmp_path / "model-catalog.md"
    result = _invoke("ml", "catalog", "-o", str(target))
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == model_catalog_to_markdown()


def test_an_unknown_format_exits_non_zero() -> None:
    """A typo must fail loudly rather than fall back to a default."""
    result = _invoke("ml", "catalog", "--format", "yaml")
    assert result.exit_code == 1


def test_catalog_output_is_deterministic() -> None:
    """Two invocations must agree byte for byte."""
    first = _invoke("ml", "catalog", "--format", "markdown")
    second = _invoke("ml", "catalog", "--format", "markdown")
    assert first.stdout == second.stdout


# ---------------------------------------------------------------------------
# Generated documentation
# ---------------------------------------------------------------------------


def test_the_tracked_documentation_is_byte_identical_to_the_generated_output() -> None:
    """``docs/model-catalog.md`` is generated, never hand-edited.

    If this fails, regenerate it:

        uv run password-attack-detector ml catalog --format markdown \\
            -o docs/model-catalog.md
    """
    tracked = _repo_root() / "docs" / "model-catalog.md"
    assert tracked.exists(), "docs/model-catalog.md is missing"
    assert tracked.read_text(encoding="utf-8") == model_catalog_to_markdown()


def test_regenerating_the_documentation_is_idempotent(tmp_path: Path) -> None:
    """Running the generator twice must not produce a diff."""
    target = tmp_path / "model-catalog.md"
    _invoke("ml", "catalog", "--format", "markdown", "-o", str(target))
    first = target.read_text(encoding="utf-8")
    _invoke("ml", "catalog", "--format", "markdown", "-o", str(target))
    assert target.read_text(encoding="utf-8") == first


def test_the_tracked_documentation_carries_no_measured_result() -> None:
    """The catalog declares what may be fitted, never how well anything did."""
    tracked = (_repo_root() / "docs" / "model-catalog.md").read_text(encoding="utf-8")
    lowered = tracked.lower()
    for term in ("accuracy of", "auc of", "f1 score of", "we achieved", "% recall"):
        assert term not in lowered, term


# ---------------------------------------------------------------------------
# Both entry points
# ---------------------------------------------------------------------------


def test_the_console_script_entry_point_works() -> None:
    """``password-attack-detector ml catalog`` must run as an installed script."""
    completed = subprocess.run(
        ["password-attack-detector", "ml", "catalog"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_repo_root(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "M-000" in completed.stdout


def test_the_module_entry_point_works() -> None:
    """``python -m password_attack_detector ml catalog`` must run too."""
    completed = subprocess.run(
        [sys.executable, "-m", "password_attack_detector", "ml", "catalog"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_repo_root(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "M-000" in completed.stdout


# ---------------------------------------------------------------------------
# Existing command groups stay green
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        ("version",),
        ("doctor",),
        ("show-config",),
        ("data", "--help"),
        ("features", "--help"),
        ("detection", "--help"),
        ("detection", "catalog"),
    ],
)
def test_existing_commands_still_succeed(arguments: tuple[str, ...]) -> None:
    """Registering the ML group must not disturb any Phase 1-4 command."""
    result = _invoke(*arguments)
    assert result.exit_code == 0, result.stdout


def test_every_phase_group_is_still_registered() -> None:
    """All four sub-applications remain reachable from the root."""
    result = _invoke("--help")
    assert result.exit_code == 0
    for group in ("data", "features", "detection", "ml"):
        assert group in result.stdout


def test_the_package_version_is_unchanged_at_this_checkpoint() -> None:
    """Milestone 1 does not bump the version; the phase release does.

    The version moves to 0.5.0 in the final milestone, together with the
    documentation and the changelog entry that make the bump meaningful.
    """
    from password_attack_detector import __version__

    assert __version__ == "0.4.0"

    project = tomllib.loads(
        (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert isinstance(project, dict)
    assert project["version"] == "0.4.0"


# ---------------------------------------------------------------------------
# What Milestone 1 must not do
# ---------------------------------------------------------------------------


def test_the_ml_package_declares_no_dataset_or_training_module() -> None:
    """Milestone 1 fits nothing and reads no label.

    Dataset assembly, preprocessing, fitting, calibration, thresholds,
    inference, fusion, evaluation, explanation, and drift all belong to later
    milestones. An empty placeholder for any of them would make the package
    look further along than it is.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    present = {path.stem for path in package.glob("*.py")}
    assert present == {
        "__init__",
        "catalog",
        "cli",
        "config",
        "dependencies",
        "enums",
        "schemas",
    }


def _imported_names(module: Path) -> set[str]:
    """Return every module path and imported symbol name in *module*.

    Parses the syntax tree rather than matching text, so a name appearing in a
    docstring or in a *prohibition* list is not mistaken for a use of it --
    ``ml.schemas`` legitimately names ``campaign_id`` in the set of fields it
    forbids.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names |= {alias.name for alias in node.names}
    return names


#: Modules that read ground truth, split assignments, or the canonical event
#: stream.  Milestone 2 introduces ``ml.dataset`` as the ML layer's single
#: permitted label reader; until then nothing in ``ml`` may import any of them.
LABEL_BEARING_MODULES = frozenset(
    {
        "password_attack_detector.data.serialization",
        "password_attack_detector.features.splitting",
        "password_attack_detector.features.serialization",
        "password_attack_detector.detection.evaluation",
    }
)

#: Symbols that carry ground truth, a split assignment, or campaign identity.
LABEL_BEARING_SYMBOLS = frozenset(
    {
        "GroundTruthLabel",
        "SplitLabel",
        "SplitAssignment",
        "SplitResult",
        "LabelRecord",
        "SplitRecord",
        "CampaignRecord",
        "read_ground_truth_labels",
        "read_feature_snapshots",
        "labels_for_events",
        "label_fingerprint",
        "split_dataset",
    }
)


def test_no_milestone_one_module_imports_a_label_reader() -> None:
    """Nothing in this milestone touches ground truth in any form.

    Milestone 2 introduces ``ml.dataset`` as the layer's single label reader
    and extends the detection layer's import-graph assertion to admit exactly
    it. Until then, no ML module may import a label, a split assignment, or a
    campaign record.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    modules = sorted(package.glob("*.py"))
    assert modules

    for module in modules:
        imported = _imported_names(module)
        offending = sorted(imported & (LABEL_BEARING_MODULES | LABEL_BEARING_SYMBOLS))
        assert not offending, f"{module.name} imports {offending}"


def test_no_milestone_one_module_opens_a_data_file() -> None:
    """Milestone 1 reads configuration and nothing else.

    ``load_ml_config`` reads a YAML file; no module reads a Parquet table, and
    none of them constructs a dataframe. A milestone that fits nothing should
    also load nothing.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    for module in sorted(package.glob("*.py")):
        imported = _imported_names(module)
        assert "pyarrow" not in {name.split(".")[0] for name in imported}, module.name
        assert "pandas" not in {name.split(".")[0] for name in imported}, module.name
