"""Integration tests for the ``ml`` CLI group's registration and catalog command.

This module covers ``ml catalog`` and the group's registration. The other two
commands have their own modules, because each needs different inputs and
asserts a different set of properties: ``ml audit-features`` needs published
Parquet, and ``ml verify-manifest`` needs a published model directory.

Two properties are swept, matching the conventions of the other CLI test
modules: **no command prints an identifier or an absolute path**, and **no
command advertises a capability this milestone does not have**.
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

#: Capabilities Phase 5 deliberately does not ship, named the way a command
#: would be.  The layer is offline: it serves nothing, deploys nothing,
#: retrains nothing on a monitoring finding, and promotes nothing without a
#: validation selection.  A placeholder that exists but does nothing is worse
#: than an honest absence, because ``--help`` would advertise a capability the
#: code lacks.
ABSENT_COMMANDS = (
    "serve",
    "deploy",
    "retrain",
    "promote",
    "tune",
)

#: The complete, final Phase 5 command surface.
SHIPPED_COMMANDS = (
    "catalog",
    "audit-features",
    "verify-manifest",
    "train",
    "experiments",
    "select",
    "freeze-champion",
    "predict",
    "validate",
    "profile",
    "evaluate",
    "compare",
    "explain",
    "drift",
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


def test_the_ml_group_advertises_only_the_shipped_commands() -> None:
    """Fourteen commands, and the absent capabilities must stay unregistered."""
    result = _invoke("ml", "--help")
    assert result.exit_code == 0
    for command in SHIPPED_COMMANDS:
        assert command in result.stdout, command
    # Whole words: the group's own help text legitimately says "no champion is
    # selected", and a substring search would read that as a command.
    for command in ABSENT_COMMANDS:
        assert not re.search(rf"\b{re.escape(command)}\b", result.stdout), command


def test_the_registered_command_surface_is_exactly_the_shipped_set() -> None:
    """Pinned in both directions, so a convenience command cannot slip in.

    The help text alone would only prove the shipped commands are *present*.
    Reading the registry proves nothing else is.
    """
    from password_attack_detector.ml.cli import ml_app

    registered = {
        command.name or (command.callback.__name__ if command.callback else "")
        for command in ml_app.registered_commands
    }
    assert registered == set(SHIPPED_COMMANDS)


@pytest.mark.parametrize("command", ABSENT_COMMANDS)
def test_an_absent_capability_is_not_callable(command: str) -> None:
    """This layer is offline; a command that implied otherwise must not exist."""
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


def test_the_package_version_is_the_current_release() -> None:
    """The phase release bumps the version, and every authority agrees on it.

    Phase 5's Milestones 1 through 9 deliberately left it alone; Milestone 10
    moved it to 0.5.0 together with the governance documentation that made the
    bump meaningful, and the Phase 6 release milestone moved it to 0.6.0 on the
    same terms. The runtime constant and the packaging metadata are checked
    against each other rather than each against a literal, because two literals
    can agree with a test and disagree with each other.
    """
    from password_attack_detector import __version__

    assert __version__ == "0.6.0"

    project = tomllib.loads(
        (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert isinstance(project, dict)
    assert project["version"] == "0.6.0"
    assert project["version"] == __version__


# ---------------------------------------------------------------------------
# What Milestone 8 must not do
# ---------------------------------------------------------------------------


#: The Milestone 9 modules that own fusion.  Every other module in the layer --
#: and in particular the whole Milestone 8 prediction path -- still knows nothing
#: about it: a prediction artifact publishes no fused verdict, and a fused
#: decision reaches a report only through the locked TEST evaluation.
FUSION_MODULES = frozenset(
    {"enums", "fusion", "stacking", "test_evaluation", "alerts", "comparison"}
)


def test_fusion_is_confined_to_the_milestone_nine_modules() -> None:
    """The Milestone 8 prediction path publishes no fused decision.

    Fusion arrived with Milestone 9. What must not happen is a fused verdict
    leaking into the prediction artifacts, where it would be published beside a
    frozen champion's own decision and be indistinguishable from one.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    for module in sorted(package.rglob("*.py")):
        if module.stem in FUSION_MODULES:
            continue
        source = module.read_text(encoding="utf-8")
        assert "FusionDecision" not in source, module.name
        assert "fused_flagged" not in source or module.stem in {
            "features",
            "predictions",
        }, module.name


def test_no_prediction_artifact_publishes_a_fused_verdict() -> None:
    """``fused_flagged`` stays a forbidden *input* and an unpublished output."""
    from password_attack_detector.ml.predictions import (
        ANOMALY_PREDICTION_COLUMNS,
        BINARY_PREDICTION_COLUMNS,
        CATEGORY_PREDICTION_COLUMNS,
        PROHIBITED_PREDICTION_COLUMNS,
    )

    for columns in (
        BINARY_PREDICTION_COLUMNS,
        CATEGORY_PREDICTION_COLUMNS,
        ANOMALY_PREDICTION_COLUMNS,
    ):
        assert "fused_flagged" not in columns
    assert "fused_flagged" in PROHIBITED_PREDICTION_COLUMNS


def test_no_prediction_module_writes_a_test_evaluation_record() -> None:
    """The fourth ledger record type stays reserved and unwritten.

    Parsed rather than grepped: ``prediction_manifest`` legitimately names
    ``test_evaluation`` in the set of fields it *forbids*, and a text search
    would read that prohibition as a use of the thing it prohibits.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    for name in (
        "predictions",
        "prediction_manifest",
        "prediction_publisher",
        "prediction_serialization",
        "prediction_validation",
        "quality",
    ):
        imported = _imported_names(package / f"{name}.py")
        assert "ExperimentRecordType" not in imported, name
        tree = ast.parse((package / f"{name}.py").read_text(encoding="utf-8"))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert "TEST_EVALUATION" not in attributes, name


def test_the_prediction_modules_read_no_ground_truth() -> None:
    """The label-reader allowlist is unchanged by this milestone."""
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    for name in (
        "predictions",
        "prediction_manifest",
        "prediction_publisher",
        "prediction_serialization",
        "prediction_validation",
        "quality",
    ):
        imported = _imported_names(package / f"{name}.py")
        offending = sorted(imported & (LABEL_BEARING_MODULES | LABEL_BEARING_SYMBOLS))
        assert not offending, (name, offending)


def test_the_reference_baseline_is_never_a_prediction_champion() -> None:
    """M-000 remains the comparator, and the lock refuses it outright."""
    from password_attack_detector.ml.catalog import MODEL_CATALOG

    reference = [spec for spec in MODEL_CATALOG.specs if spec.reference_baseline]
    assert [spec.model_id for spec in reference] == ["M-000"]
    assert not reference[0].champion_eligible


def test_the_unpublishable_and_experimental_families_stay_that_way() -> None:
    """M-021 has no proven serializer; M-030 is experimental and anomaly-only."""
    from password_attack_detector.ml.catalog import MODEL_CATALOG
    from password_attack_detector.ml.models import PUBLISHABLE_FAMILIES

    catalog = {spec.model_id: spec for spec in MODEL_CATALOG.specs}
    assert catalog["M-021"].family not in PUBLISHABLE_FAMILIES
    assert not catalog["M-021"].champion_eligible
    assert catalog["M-030"].experimental
    assert catalog["M-030"].anomaly_only
    assert not catalog["M-030"].champion_eligible


# ---------------------------------------------------------------------------
# What Milestone 1 must not do
# ---------------------------------------------------------------------------


def test_the_ml_package_declares_exactly_the_built_modules() -> None:
    """The final Phase 5 module inventory, pinned in both directions.

    Model adapters and serialization arrived with Milestone 4; calibration and
    threshold selection with Milestone 5; training orchestration, the immutable
    experiment ledger, and run publication with Milestone 6; champion gates,
    validation-only selection, and the champion freeze with Milestone 7; batch
    inference, the prediction artifacts, their manifest, their validation, and
    the aggregate profile with Milestone 8; the locked test evaluation, fusion,
    stacking, alerts, and the system comparison with Milestone 9; attribution,
    the reference profile, drift, and governance with Milestone 10.

    Pinned as an equality rather than a subset so an empty placeholder for a
    capability nobody built cannot make the package look further along than it
    is -- and so a module quietly added outside the reviewed inventory fails
    here rather than in review.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    present = {path.stem for path in package.glob("*.py")}
    assert present == {
        "__init__",
        "alerts",
        "calibration",
        "catalog",
        "champion",
        "cli",
        "comparison",
        "config",
        "dataset",
        "dependencies",
        "drift",
        "eligibility",
        "enums",
        "experiments",
        "explain",
        "features",
        "fusion",
        "governance",
        "metrics",
        "reference",
        "gates",
        "imbalance",
        "inference",
        "ledger",
        "manifest",
        "npz",
        "ordering",
        "partition",
        "prediction_manifest",
        "prediction_publisher",
        "prediction_serialization",
        "prediction_validation",
        "predictions",
        "preprocessing",
        "quality",
        "ranking",
        "schemas",
        "selection",
        "serialization",
        "stacking",
        "test_evaluation",
        "thresholds",
        "training",
    }


def test_the_model_package_declares_exactly_the_implemented_families() -> None:
    """One module per family, plus the contract and the closed registry."""
    package = _repo_root() / "src" / "password_attack_detector" / "ml" / "models"
    assert {path.stem for path in package.glob("*.py")} == {
        "__init__",
        "anomaly",
        "base",
        "baseline",
        "boosting",
        "forest",
        "linear",
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
#: stream.  ``ml.dataset`` is the ML layer's single permitted reader, and the
#: two-module allowlist is asserted in
#: ``tests/unit/detection/test_evaluation.py``; this module checks the narrower
#: property that *no other* ``ml`` module imports one.
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


#: The one ML module permitted to read ground truth.  Named as a constant so
#: the exemptions below cannot be widened by editing a condition.
LABEL_READER = "dataset"


def test_only_the_dataset_module_imports_a_label_reader() -> None:
    """Every other ML module receives its labels as typed arguments.

    The complete two-module allowlist -- ``detection.evaluation`` and
    ``ml.dataset`` -- is asserted in ``tests/unit/detection/test_evaluation.py``,
    in both directions. This is the ``ml``-package half of it.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    modules = sorted(package.glob("*.py"))
    assert modules

    readers = []
    for module in modules:
        imported = _imported_names(module)
        offending = sorted(imported & (LABEL_BEARING_MODULES | LABEL_BEARING_SYMBOLS))
        if offending:
            readers.append(module.stem)
    assert readers == [LABEL_READER], readers


#: The one module permitted to *write* -- and read back -- a Parquet file this
#: layer produced itself.
#:
#: A deliberately different permission from :data:`LABEL_READER`'s. The
#: prediction serializer opens no input table: it writes the rows a frozen model
#: emitted and reads them back to verify them, and there is no join for it to do
#: differently because there is nothing to join. It imports no label-bearing
#: symbol, which the sweep above asserts separately and unchanged.
PREDICTION_ARTIFACT_SERIALIZER = "prediction_serialization"


def test_only_two_modules_open_a_parquet_file() -> None:
    """Parquet is touched in exactly two places, for two unrelated reasons.

    A second module that opened an *input* table would be a second place where
    the join could be done differently, and the first thing it would need is the
    label column -- so ``ml.dataset`` remains the only reader of Phase 3 tables.
    The prediction serializer is the only *writer* of this layer's own output,
    and it never opens an input. ``cli`` is exempt as the composition root: it
    names the paths and hands them on, importing no reader of its own.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    permitted = {LABEL_READER, PREDICTION_ARTIFACT_SERIALIZER}
    for module in sorted(package.glob("*.py")):
        if module.stem in permitted:
            continue
        imported = {name.split(".")[0] for name in _imported_names(module)}
        assert "pyarrow" not in imported, module.name
        assert "pandas" not in imported, module.name


def test_the_dataset_module_is_the_one_that_reads_parquet() -> None:
    """The converse: the permitted reader must actually be the reader."""
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    imported = {
        name.split(".")[0] for name in _imported_names(package / f"{LABEL_READER}.py")
    }
    assert "pyarrow" in imported


def test_the_prediction_serializer_reads_no_label_bearing_symbol() -> None:
    """The writer's permission is narrower than the reader's, and stays narrower.

    It may open a Parquet file it wrote itself. It may not import a ground-truth
    reader, a split reader, or a campaign type -- so the exemption above cannot
    become a second route to the label table.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    imported = _imported_names(package / f"{PREDICTION_ARTIFACT_SERIALIZER}.py")
    offending = sorted(imported & (LABEL_BEARING_MODULES | LABEL_BEARING_SYMBOLS))
    assert not offending, offending


def test_only_the_model_adapters_import_an_estimator() -> None:
    """scikit-learn is confined to the six family modules, and to fitting.

    Everything outside ``ml/models`` -- the dataset, the preprocessor, the
    archive, the serializer, the manifest, the loader, the CLI -- works on
    numbers and never touches an estimator. That is what lets a published model
    be scored, verified, and loaded by a build whose scikit-learn has moved.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    for module in sorted(package.glob("*.py")):
        imported = {name.split(".")[0] for name in _imported_names(module)}
        assert "sklearn" not in imported, module.name
        assert "scipy" not in imported, module.name
        assert "joblib" not in imported, module.name


def test_no_module_imports_a_transitive_scientific_dependency() -> None:
    """Including the adapters: scipy and joblib arrive only through sklearn."""
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    for module in sorted(package.rglob("*.py")):
        imported = {name.split(".")[0] for name in _imported_names(module)}
        assert "scipy" not in imported, module.name
        assert "joblib" not in imported, module.name
        assert "threadpoolctl" not in imported, module.name


def test_the_class_weight_formula_is_the_projects_own() -> None:
    """``sklearn.utils.class_weight`` is never the authority for a shipped weight.

    Delegating would make the number in a manifest depend on a library's
    internal convention, which is exactly the kind of dependency a recorded
    fingerprint cannot express.
    """
    module = _repo_root() / "src" / "password_attack_detector" / "ml" / "imbalance.py"
    imported = _imported_names(module)
    assert not any(name.startswith("sklearn") for name in imported)
    assert "class_weight" not in imported


def test_preprocessing_reads_no_label_split_or_campaign_type() -> None:
    """The new module consumes typed frames, never a label-bearing symbol.

    The package-wide sweep in ``tests/unit/detection/test_evaluation.py`` already
    covers this; asserting it here too means the two modules Milestone 3 adds
    cannot become a quiet exemption in either place.
    """
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    for name in ("preprocessing", "imbalance"):
        imported = _imported_names(package / f"{name}.py")
        offending = sorted(imported & (LABEL_BEARING_MODULES | LABEL_BEARING_SYMBOLS))
        assert not offending, (name, offending)
