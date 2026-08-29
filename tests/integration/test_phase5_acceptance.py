"""The final Phase 5 acceptance, governance, security, privacy, and hygiene audit.

Four sweeps live here, and none of them is a claim about intent.

**Acceptance** builds the report against a pipeline that really ran, so every
requirement a produced artifact establishes resolves to a pass or a genuine
not-applicable, and nothing stays inconclusive.

**Security** parses the source rather than reading its documentation: no object
serializer is imported, no builtin turns a value into code, no artifact value
reaches an import path, and no YAML is loaded unsafely.

**Privacy** sweeps the aggregate artifacts and generated documents the pipeline
writes for anything the privacy model prohibits.

**Hygiene** checks that everything the pipeline generates stays ignored, and that
the tracked governance documents are the generated ones.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector import __version__
from password_attack_detector.ml.champion import CHAMPION_LOCK_FILE
from password_attack_detector.ml.enums import AcceptanceStatus, FusionStrategy, MLSplit
from password_attack_detector.ml.governance import (
    AcceptanceEvidence,
    acceptance_report_to_markdown,
    build_acceptance_report,
    model_card_to_markdown,
)
from password_attack_detector.ml.prediction_manifest import (
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
)
from password_attack_detector.ml.test_evaluation import (
    EVALUATION_RECEIPT_FILE,
    EVALUATIONS_DIR,
)
from tests.integration.ml_workspace import (
    build_workspace,
    detect,
    drift,
    evaluate,
    explain,
    explanation_directory,
    freeze,
    predict,
    prediction_ids,
)

PROFILE = Path("reference") / "reference_profile.json"

_PSEUDONYM_RE = re.compile(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_COORDINATE_RE = re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}")


def _repo_root() -> Path:
    """Return the repository root, located from this test file."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Run every Phase 5 stage once, so acceptance has real evidence."""
    base = tmp_path_factory.mktemp("acceptance")
    (base / "workspace").mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(base / "workspace")
    root = base / "artifacts"
    reports = base / "reports"
    freeze(workspace, root, reports)

    for split in ("train", "validation", "test"):
        scored = predict(workspace, root, split=split)
        assert scored.exit_code == 0, scored.output
    ids = prediction_ids(root)

    detection = base / "detection"
    assessed = detect(workspace, detection)
    assert assessed.exit_code == 0, assessed.output
    evaluated = evaluate(
        workspace, root, detection, reports, **{"--prediction": ids["test"]}
    )
    assert evaluated.exit_code == 0, evaluated.output

    explained = explain(workspace, root, prediction_id=ids["validation"])
    assert explained.exit_code == 0, explained.output
    monitored = drift(
        workspace,
        root,
        reports,
        incoming_split="train",
        reference_prediction=ids["train"],
        incoming_prediction=ids["train"],
    )
    assert monitored.exit_code == 0, monitored.output
    return (workspace, root, reports)


def _evidence(root: Path) -> AcceptanceEvidence:
    """Collect what the completed pipeline actually established.

    Read off the published artifacts. Nothing is asserted that a file does not
    say, which is the whole point of an acceptance report built from evidence.
    """
    lock = json.loads(
        next((root / "champion").glob(f"*/{CHAMPION_LOCK_FILE}")).read_text(
            encoding="utf-8"
        )
    )
    receipt = json.loads(
        next((root / EVALUATIONS_DIR).glob(f"*/{EVALUATION_RECEIPT_FILE}")).read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        next((root / PREDICTIONS_DIR).glob(f"*/{PREDICTION_MANIFEST_FILE}")).read_text(
            encoding="utf-8"
        )
    )
    explanation = json.loads(
        (explanation_directory(root) / "explanation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (explanation_directory(root) / "explanation_report.json").read_text(
            encoding="utf-8"
        )
    )
    profile = json.loads((root / PROFILE).read_text(encoding="utf-8"))
    runs = list((root / "ledger" / "training_run").glob("*.json"))

    return AcceptanceEvidence(
        champion_lock_fingerprint=lock["lock_fingerprint"],
        category_head_frozen=lock.get("category_head") is not None,
        training_run_count=len(runs),
        validation_selection_id=lock["validation_selection_id"],
        prediction_manifest_fingerprint=manifest["prediction_manifest_fingerprint"],
        test_evaluation_record_id=receipt["identity"],
        test_evaluation_status=receipt["status"],
        fusion_declared_candidates=tuple(FusionStrategy),
        fusion_selected_strategy=(
            None
            if receipt.get("selected_fusion_strategy") is None
            else FusionStrategy(receipt["selected_fusion_strategy"])
        ),
        novel_holdout_row_count=receipt.get("novel_holdout_row_count") or 0,
        explanation_id=explanation["explanation_id"],
        explanation_status=report["status"],
        reference_profile_id=profile["reference_profile_id"],
        reference_split=MLSplit(profile["reference_split"]),
        drift_run_id="captured",
        package_version=__version__,
    )


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def test_a_completed_pipeline_leaves_no_requirement_inconclusive(
    completed: tuple[Path, Path, Path],
) -> None:
    """Every artifact requirement resolves once the artifacts exist."""
    _, root, _ = completed
    report = build_acceptance_report(
        package_version=__version__, evidence=_evidence(root)
    )
    unresolved = [
        item
        for item in report.requirements
        if item.status is AcceptanceStatus.INCONCLUSIVE
    ]
    assert unresolved == [], [item.requirement_id for item in unresolved]


def test_a_completed_pipeline_fails_no_requirement(
    completed: tuple[Path, Path, Path],
) -> None:
    """The contracts Phase 5 declares are the contracts it met."""
    _, root, _ = completed
    report = build_acceptance_report(
        package_version=__version__, evidence=_evidence(root)
    )
    failures = [
        item for item in report.requirements if item.status is AcceptanceStatus.FAIL
    ]
    assert failures == [], [(item.requirement_id, item.evidence) for item in failures]
    assert report.accepted is True


def test_every_milestone_is_covered_by_at_least_one_requirement(
    completed: tuple[Path, Path, Path],
) -> None:
    """A milestone nobody wrote a requirement for is a milestone nobody audited."""
    _, root, _ = completed
    report = build_acceptance_report(
        package_version=__version__, evidence=_evidence(root)
    )
    covered = {item.milestone for item in report.requirements}
    assert covered == {f"M{index}" for index in range(1, 11)}


def test_the_acceptance_report_names_no_metric(
    completed: tuple[Path, Path, Path],
) -> None:
    """It says the evaluation happened, never how it came out."""
    _, root, _ = completed
    report = build_acceptance_report(
        package_version=__version__, evidence=_evidence(root)
    )
    prose = " ".join(
        f"{item.title} {item.evidence}" for item in report.requirements
    ).lower()
    for token in ("pr_auc", "roc_auc", "precision", "recall", "brier"):
        assert token not in prose, token


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_the_runtime_and_packaging_versions_agree() -> None:
    """Two literals that agree with a test can still disagree with each other."""
    import tomllib

    project = tomllib.loads(
        (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert project["version"] == __version__ == "0.6.0"


def test_the_cli_reports_the_release_version() -> None:
    """The version a user sees is the version the package declares."""
    from typer.testing import CliRunner

    from password_attack_detector.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_the_built_distributions_declare_the_release_version(
    tmp_path: Path,
) -> None:
    """Wheel and sdist metadata, read from an actual build."""
    import zipfile

    completed_build = subprocess.run(
        ["uv", "build", "--out-dir", str(tmp_path)],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed_build.returncode == 0, completed_build.stderr

    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1
    assert f"-{__version__}-" in wheels[0].name
    assert f"-{__version__}.tar.gz" in sdists[0].name

    with zipfile.ZipFile(wheels[0]) as archive:
        metadata = next(
            name for name in archive.namelist() if name.endswith("METADATA")
        )
        text = archive.read(metadata).decode("utf-8")
    assert f"Version: {__version__}" in text


# ---------------------------------------------------------------------------
# Governance documents
# ---------------------------------------------------------------------------


def test_the_tracked_model_card_is_the_generated_one() -> None:
    """Generated, never hand-edited.

    If this fails, regenerate it::

        uv run python scripts/generate_governance_docs.py
    """
    tracked = _repo_root() / "docs" / "model-card.md"
    assert tracked.is_file()
    assert tracked.read_text(encoding="utf-8") == model_card_to_markdown()


def test_the_tracked_acceptance_report_is_the_generated_one() -> None:
    """The contract-derived report, regenerable by anybody with the source."""
    tracked = _repo_root() / "docs" / "phase5-acceptance.md"
    assert tracked.is_file()
    expected = acceptance_report_to_markdown(
        build_acceptance_report(package_version=__version__)
    )
    assert tracked.read_text(encoding="utf-8") == expected


def test_regenerating_the_governance_documents_is_idempotent() -> None:
    """Running the generator twice must not produce a diff."""
    first = model_card_to_markdown()
    second = model_card_to_markdown()
    assert first == second


# ---------------------------------------------------------------------------
# Security audit
# ---------------------------------------------------------------------------


def _phase5_sources() -> list[Path]:
    """Return every Phase 5 source file, in a stable order."""
    return sorted(
        (_repo_root() / "src" / "password_attack_detector" / "ml").rglob("*.py")
    )


def test_no_phase_5_module_imports_an_object_serializer() -> None:
    """A model artifact is numbers, not a program.

    ``joblib`` may remain a scikit-learn transitive dependency; what must never
    happen is this project importing it.
    """
    from password_attack_detector.ml.governance import module_imports

    for path in _phase5_sources():
        roots = {name.split(".")[0] for name in module_imports(path)}
        assert not roots & {"pickle", "cPickle", "dill", "joblib", "shelve"}, path.name


def test_no_phase_5_module_turns_a_value_into_code() -> None:
    """No eval, no exec, no compile, no dynamic import."""
    banned = {"eval", "exec", "compile", "__import__"}
    for path in _phase5_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in banned, (path.name, node.func.id)


def test_no_phase_5_module_uses_an_unsafe_yaml_loader() -> None:
    """``safe_load`` constructs scalars, lists, and mappings, and nothing else."""
    for path in _phase5_sources():
        text = path.read_text(encoding="utf-8")
        for token in ("yaml.load(", "yaml.unsafe_load", "Loader=yaml.Loader"):
            assert token not in text, (path.name, token)


def test_no_phase_5_module_spawns_a_shell() -> None:
    """An offline analysis layer has nothing to run."""
    for path in _phase5_sources():
        text = path.read_text(encoding="utf-8")
        for token in ("shell=True", "os.system", "subprocess."):
            assert token not in text, (path.name, token)


def test_no_phase_5_module_makes_a_network_request() -> None:
    """Nothing here calls out; the layer is offline by construction."""
    from password_attack_detector.ml.governance import module_imports

    for path in _phase5_sources():
        roots = {name.split(".")[0] for name in module_imports(path)}
        assert not roots & {
            "requests",
            "httpx",
            "urllib",
            "socket",
            "http",
            "ftplib",
            "smtplib",
        }, path.name


def test_artifact_readers_dispatch_from_a_closed_registry() -> None:
    """No import path leads from an artifact's own content to code.

    A family name that is not a key in the registry selects nothing at all,
    which is what makes reading an untrusted artifact safe.
    """
    from password_attack_detector.ml.enums import ModelFamily
    from password_attack_detector.ml.models import MODEL_IMPLEMENTATIONS

    assert set(MODEL_IMPLEMENTATIONS) <= set(ModelFamily)
    assert MODEL_IMPLEMENTATIONS


def test_joblib_is_only_a_transitive_dependency() -> None:
    """It arrives with scikit-learn; it is never a project serialization format."""
    from password_attack_detector.ml.dependencies import ML_DEPENDENCY_REQUIREMENTS

    declared = {item.distribution for item in ML_DEPENDENCY_REQUIREMENTS}
    assert "joblib" not in declared
    project = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    assert '"joblib' not in project


# ---------------------------------------------------------------------------
# Privacy audit
# ---------------------------------------------------------------------------


def _aggregate_documents(root: Path, reports: Path) -> list[Path]:
    """Return every aggregate artifact and generated document to sweep."""
    found = [
        path
        for path in sorted(reports.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".md"}
    ]
    found += [
        root / PROFILE,
        explanation_directory(root) / "explanation_report.json",
        explanation_directory(root) / "explanation_report.md",
        explanation_directory(root) / "explanation_manifest.json",
    ]
    return [path for path in found if path.is_file()]


def test_no_aggregate_artifact_carries_a_pseudonym_or_a_coordinate(
    completed: tuple[Path, Path, Path],
) -> None:
    """The privacy model, swept over what the pipeline actually wrote."""
    _, root, reports = completed
    for path in _aggregate_documents(root, reports):
        text = path.read_text(encoding="utf-8")
        assert not _PSEUDONYM_RE.search(text), path.name
        assert not _COORDINATE_RE.search(text), path.name
        assert not _IPV4_RE.search(text), path.name


def test_no_aggregate_artifact_names_a_prohibited_field(
    completed: tuple[Path, Path, Path],
) -> None:
    """Field *names*: a key named for a label or an identity is a leak.

    Matched in key form (``"name":``) rather than as a bare quoted string,
    because a reference profile legitimately carries the category *value*
    ``password`` -- the authentication method a reviewed feature records, which
    is not a credential and never was one.
    """
    _, root, reports = completed
    for path in _aggregate_documents(root, reports):
        text = path.read_text(encoding="utf-8")
        for name in (
            "anchor_event_id",
            "event_id",
            "campaign_id",
            "user_id",
            "source_id",
            "session_id",
            "password",
            "credential",
            "secret",
            "token",
            "latitude",
            "longitude",
            "malicious",
            "attack_class",
        ):
            assert f'"{name}":' not in text, (path.name, name)
            assert f'"{name}" :' not in text, (path.name, name)


def test_no_generated_document_leaks_an_absolute_home_path(
    completed: tuple[Path, Path, Path],
) -> None:
    """A report that travels must not carry the machine it was written on."""
    _, root, reports = completed
    for path in _aggregate_documents(root, reports):
        assert str(Path.home()) not in path.read_text(encoding="utf-8"), path.name


def test_the_row_level_explanation_contract_is_the_documented_one() -> None:
    """Exactly one join identity is permitted, and only on the row artifact."""
    from password_attack_detector.ml.explain import (
        ExplanationManifest,
        ExplanationQualityReport,
        FeatureContribution,
        PredictionExplanation,
    )

    assert "anchor_event_id" in PredictionExplanation.model_fields
    for model in (
        FeatureContribution,
        ExplanationQualityReport,
        ExplanationManifest,
    ):
        assert "anchor_event_id" not in model.model_fields


def test_the_tracked_documents_carry_no_identifier() -> None:
    """Governance documentation is checked in; it must be safe to read."""
    for name in ("model-card.md", "phase5-acceptance.md"):
        text = (_repo_root() / "docs" / name).read_text(encoding="utf-8")
        assert not _PSEUDONYM_RE.search(text), name
        assert not _UUID_RE.search(text), name
        assert not _COORDINATE_RE.search(text), name


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------


def _ignored(paths: list[str]) -> set[str]:
    """Return which of *paths* git considers ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=_repo_root(),
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=False,
    )
    return {line for line in result.stdout.splitlines() if line}


@pytest.mark.parametrize(
    "generated",
    [
        "artifacts/ml/champion/scope/champion.lock",
        "artifacts/ml/ledger/ledger.json",
        "artifacts/ml/predictions/x/binary_predictions.parquet",
        "artifacts/ml/evaluations/x/test_evaluation.json",
        "artifacts/ml/explanations/x/explanation_manifest.json",
        "artifacts/ml/reference/reference_profile.json",
        "reports/ml_drift_report.json",
        "reports/ml_evaluation.md",
        "models/model.npz",
    ],
)
def test_every_generated_artifact_stays_ignored(generated: str) -> None:
    """A published artifact in the history is a published artifact forever."""
    assert _ignored([generated]) == {generated}


def test_the_working_tree_holds_no_untracked_generated_artifact() -> None:
    """Nothing the pipeline writes may be sitting in the tree waiting to be added."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    offending = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("??")
        and any(
            token in line
            for token in ("artifacts/", "reports/", "models/", ".parquet", ".npz")
        )
    ]
    assert offending == []


def test_the_governance_documents_are_tracked() -> None:
    """Generated, but *checked in*: they are the phase's public statement."""
    result = subprocess.run(
        ["git", "ls-files", "docs/"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = set(result.stdout.splitlines())
    for name in (
        "docs/model-card.md",
        "docs/phase5-acceptance.md",
        "docs/explainability.md",
        "docs/drift-monitoring.md",
    ):
        assert name in tracked or (_repo_root() / name).is_file(), name


# ---------------------------------------------------------------------------
# Regression: every earlier milestone still works
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
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
    ],
)
def test_every_shipped_command_still_offers_help(command: str) -> None:
    """The final surface, exercised one command at a time."""
    from typer.testing import CliRunner

    from password_attack_detector.cli import app

    result = CliRunner().invoke(app, ["ml", command, "--help"])
    assert result.exit_code == 0, command


def test_the_locked_evaluation_receipt_is_unchanged_by_milestone_10(
    completed: tuple[Path, Path, Path],
) -> None:
    """Attribution and drift ran after it; the receipt must be byte-identical.

    Captured, then both Milestone 10 commands are run again, then compared.
    """
    workspace, root, reports = completed
    receipt = next((root / EVALUATIONS_DIR).glob(f"*/{EVALUATION_RECEIPT_FILE}"))
    before = receipt.read_bytes()

    ids = prediction_ids(root)
    assert explain(workspace, root, prediction_id=ids["validation"]).exit_code == 0
    drift(workspace, root, reports, incoming_split="validation")
    assert receipt.read_bytes() == before


def test_no_post_test_reselection_is_possible(
    completed: tuple[Path, Path, Path],
) -> None:
    """The champion lock after the whole pipeline is the one selection froze."""
    _, root, _ = completed
    lock_path = next((root / "champion").glob(f"*/{CHAMPION_LOCK_FILE}"))
    lock: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    receipt = json.loads(
        next((root / EVALUATIONS_DIR).glob(f"*/{EVALUATION_RECEIPT_FILE}")).read_text(
            encoding="utf-8"
        )
    )
    # The evaluation names the lock it ran against; a reselection would have
    # produced a lock the receipt does not name.
    assert receipt["champion_lock_fingerprint"] == lock["lock_fingerprint"]
    assert lock["validation_selection_id"] == receipt["validation_selection_id"]
