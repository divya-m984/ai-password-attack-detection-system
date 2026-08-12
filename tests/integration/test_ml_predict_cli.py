"""End-to-end tests for ``ml predict`` over a champion the pipeline really froze.

Nothing is mocked and nothing is hand-built: the dataset is generated, published
through ``features build``, trained through ``ml train``, selected through
``ml select``, frozen through ``ml freeze-champion``, and only then predicted
from. A prediction suite that fed the command a fixture object graph would pass
against artifact shapes the pipeline never produces.

The property this file exists to assert is the **test-label firewall**. The
command scores the test split, and the suite checks -- three separate ways --
that it does so without ever reading a test label: the option does not exist,
rewriting every label leaves the published bytes identical, and nothing the
command writes or prints carries an outcome figure.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from password_attack_detector.ml.prediction_manifest import (
    BINARY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    QUALITY_REPORT_JSON_FILE,
    QUALITY_REPORT_MD_FILE,
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
    """Publish a dataset and freeze a champion over it, once for the module."""
    workspace = build_workspace(tmp_path_factory.mktemp("predict-workspace"))
    root = tmp_path_factory.mktemp("predict-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("predict-reports"))
    return (workspace, root)


@pytest.fixture
def prepared(frozen: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    """Return a writable copy of the frozen experiment, one per test."""
    workspace, root = frozen
    copied = tmp_path / "artifacts"
    shutil.copytree(root, copied)
    return (workspace, copied)


def _declared_options(help_text: str) -> str:
    """Return only the options panel of a ``--help`` rendering.

    The command's own prose legitimately explains that there is no label option,
    and a substring search over the whole page would read that sentence as the
    option it denies.
    """
    _, _, options = help_text.partition("Options")
    return options


def published_directory(root: Path) -> Path:
    """Return the single published prediction directory."""
    directories = sorted((root / PREDICTIONS_DIR).iterdir())
    assert len(directories) == 1, directories
    return directories[0]


# ---------------------------------------------------------------------------
# The publication
# ---------------------------------------------------------------------------


def test_predicting_the_test_split_publishes_a_complete_artifact(
    prepared: tuple[Path, Path],
) -> None:
    """Exit zero, every declared file present, and validation passing."""
    workspace, root = prepared
    result = predict(workspace, root)
    assert result.exit_code == 0, result.output
    assert "Rows scored" in result.stdout
    assert "pass" in result.stdout

    directory = published_directory(root)
    present = {item.name for item in directory.iterdir()}
    assert BINARY_PREDICTION_FILE in present
    assert VALIDATION_RESULT_FILE in present
    assert QUALITY_REPORT_JSON_FILE in present
    assert QUALITY_REPORT_MD_FILE in present
    assert PREDICTION_MANIFEST_FILE in present


def test_the_manifest_names_the_frozen_champion(
    prepared: tuple[Path, Path],
) -> None:
    """Every prediction is attributable to the lock that produced it."""
    workspace, root = prepared
    assert predict(workspace, root).exit_code == 0
    manifest = json.loads(
        (published_directory(root) / PREDICTION_MANIFEST_FILE).read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads(
        next((root / "champion").rglob("champion.lock")).read_text(encoding="utf-8")
    )
    assert manifest["lineage"]["champion_lock_fingerprint"] == lock["lock_fingerprint"]
    assert manifest["lineage"]["model_id"] == lock["model_id"]
    assert manifest["scope"] == "test"
    assert manifest["scope_role"]["role"] == "supervised_prediction"


def test_predicting_twice_publishes_once(prepared: tuple[Path, Path]) -> None:
    """The same evidence is the same publication."""
    workspace, root = prepared
    first = predict(workspace, root)
    second = predict(workspace, root)
    assert first.exit_code == 0 and second.exit_code == 0
    assert len(list((root / PREDICTIONS_DIR).iterdir())) == 1


def test_each_scope_is_a_separate_publication(prepared: tuple[Path, Path]) -> None:
    """Two splits, two artifacts, never one merged table.

    The novel-anomaly holdout is asserted separately, in the publisher suite:
    this CI-sized workspace carries no holdout rows, and a command that refused
    an empty scope is the correct behaviour rather than a gap in coverage.
    """
    workspace, root = prepared
    assert predict(workspace, root, split="test").exit_code == 0
    assert predict(workspace, root, split="validation").exit_code == 0
    published = sorted((root / PREDICTIONS_DIR).iterdir())
    assert len(published) == 2
    scopes = {
        json.loads((item / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8"))[
            "scope"
        ]
        for item in published
    }
    assert scopes == {"test", "validation"}


def test_a_scope_with_no_rows_is_refused(prepared: tuple[Path, Path]) -> None:
    """Nothing to score is a finding, not an empty publication."""
    workspace, root = prepared
    result = predict(workspace, root, split="novel_anomaly_holdout")
    assert result.exit_code != 0
    assert "nothing to score" in result.output


def test_predicting_writes_no_ledger_record(prepared: tuple[Path, Path]) -> None:
    """Prediction identity belongs to the manifest, not the experiment ledger."""
    workspace, root = prepared
    before = sorted(path.name for path in (root / "ledger").rglob("*.json"))
    assert predict(workspace, root).exit_code == 0
    after = sorted(path.name for path in (root / "ledger").rglob("*.json"))
    assert after == before
    assert not (root / "ledger" / "test_evaluation").exists()


def test_predicting_leaves_the_frozen_champion_untouched(
    prepared: tuple[Path, Path],
) -> None:
    """The lock is read; it is never written."""
    workspace, root = prepared
    lock = next((root / "champion").rglob("champion.lock"))
    before = lock.read_bytes()
    assert predict(workspace, root).exit_code == 0
    assert lock.read_bytes() == before


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_predicting_without_a_frozen_champion_is_refused(
    frozen: tuple[Path, Path], tmp_path: Path
) -> None:
    """A test prediction requires a lock; there is no default model."""
    workspace, root = frozen
    stripped = tmp_path / "no-champion"
    shutil.copytree(root, stripped)
    shutil.rmtree(stripped / "champion")
    result = predict(workspace, stripped)
    assert result.exit_code != 0
    assert "champion" in result.output.lower()


def test_a_tampered_lock_refuses_the_prediction(
    prepared: tuple[Path, Path],
) -> None:
    """The lock is checked, not believed."""
    workspace, root = prepared
    lock = next((root / "champion").rglob("champion.lock"))
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["catalog_model_id"] = "M-999"
    lock.write_text(json.dumps(payload), encoding="utf-8")
    result = predict(workspace, root)
    assert result.exit_code != 0
    assert not (root / PREDICTIONS_DIR).exists() or not list(
        (root / PREDICTIONS_DIR).iterdir()
    )


def test_naming_a_supervised_run_as_the_anomaly_probe_is_refused(
    prepared: tuple[Path, Path],
) -> None:
    """The experimental artifact comes from its own lineage or from nowhere.

    ``--anomaly-run`` is the only way an anomaly artifact is ever published: it
    is never read out of ``champion.lock``. This configuration trains no
    experimental probe, so the option is exercised through its refusal -- the
    end-to-end scoring path is covered in the unit suite, where the fixture does
    publish one.
    """
    workspace, root = prepared
    lock = json.loads(
        next((root / "champion").rglob("champion.lock")).read_text(encoding="utf-8")
    )
    result = predict(workspace, root, **{"--anomaly-run": lock["training_run_id"]})
    assert result.exit_code != 0
    assert "anomaly run" in result.output


def test_an_unknown_split_is_refused(prepared: tuple[Path, Path]) -> None:
    """Only the splits the enum declares may be scored."""
    workspace, root = prepared
    result = predict(workspace, root, split="production")
    assert result.exit_code != 0
    assert "Unknown split" in result.output


def test_excluded_rows_are_refused(prepared: tuple[Path, Path]) -> None:
    """Excluded is excluded from every stage, prediction included."""
    workspace, root = prepared
    result = predict(workspace, root, split="excluded")
    assert result.exit_code != 0
    assert "Excluded rows" in result.output


def test_a_feature_table_from_another_contract_is_refused(
    prepared: tuple[Path, Path], tmp_path: Path
) -> None:
    """A model scored against a different feature contract is not scored."""
    workspace, root = prepared
    manifest = tmp_path / "feature_manifest.json"
    payload = json.loads(
        (workspace / "processed" / "feature_manifest.json").read_text(encoding="utf-8")
    )
    payload["feature_catalog_fingerprint"] = "a" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = predict(workspace, root, **{"--feature-manifest": str(manifest)})
    assert result.exit_code != 0
    assert "feature contract" in result.output


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_the_command_takes_no_label_argument() -> None:
    """No labels, no campaign labels, no bypass of any kind."""
    result = invoke("ml", "predict", "--help")
    assert result.exit_code == 0
    options = _declared_options(result.stdout)
    for absent in (
        "--labels",
        "--campaign-labels",
        "--force",
        "--ignore-lock",
        "--model-id",
        "--model-path",
        "--allow-test",
    ):
        assert absent not in options, absent
    for present in ("--features", "--splits", "--split", "--allowlist"):
        assert present in options, present


def test_rewriting_every_label_changes_no_published_byte(
    prepared: tuple[Path, Path], tmp_path: Path
) -> None:
    """The strongest form of the firewall: the labels were never opened.

    The Phase 3 label table is rewritten with every outcome inverted, and the
    same prediction is run again. If a single byte moved, something read it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    workspace, root = prepared
    assert predict(workspace, root).exit_code == 0
    directory = published_directory(root)
    before = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }

    altered = tmp_path / "workspace"
    shutil.copytree(workspace, altered)
    labels_path = altered / "processed" / "feature_labels.parquet"
    table = pq.read_table(labels_path)
    flipped = table.set_column(
        table.schema.get_field_index("malicious"),
        "malicious",
        pa.array(
            [not value for value in table.column("malicious").to_pylist()],
            type=pa.bool_(),
        ),
    )
    pq.write_table(flipped, labels_path)

    second = tmp_path / "second"
    shutil.copytree(root, second)
    shutil.rmtree(second / PREDICTIONS_DIR)
    assert predict(altered, second).exit_code == 0

    after = {
        item.name: item.read_bytes()
        for item in published_directory(second).iterdir()
        if item.is_file()
    }
    assert after == before


def test_nothing_written_or_printed_carries_an_outcome_metric(
    prepared: tuple[Path, Path],
) -> None:
    """Not in the manifest, not in the report, not on the terminal."""
    workspace, root = prepared
    result = predict(workspace, root)
    assert result.exit_code == 0
    directory = published_directory(root)
    written = "".join(
        path.read_text(encoding="utf-8").lower()
        for path in (
            directory / PREDICTION_MANIFEST_FILE,
            directory / VALIDATION_RESULT_FILE,
            directory / QUALITY_REPORT_JSON_FILE,
        )
    )
    for banned in (
        "test_metrics",
        "test_evaluation",
        "accuracy",
        "roc_auc",
        "brier",
        "confusion_matrix",
        "label_fingerprint",
    ):
        assert banned not in written, banned
        assert banned not in result.output.lower(), banned


def test_no_anchor_pseudonym_or_absolute_path_reaches_the_terminal(
    prepared: tuple[Path, Path],
) -> None:
    """Output is an identity, counts, a model name, and a verdict."""
    workspace, root = prepared
    result = predict(workspace, root)
    assert result.exit_code == 0
    assert not PSEUDONYM_RE.search(result.output)
    assert str(Path.home()) not in result.output
    assert "/tmp/" not in result.output


def test_no_anchor_identifier_appears_in_an_aggregate_artifact(
    prepared: tuple[Path, Path],
) -> None:
    """The join keys stay in the Parquet the next milestone will join on."""
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
    )

    workspace, root = prepared
    assert predict(workspace, root).exit_code == 0
    directory = published_directory(root)
    rows = read_binary_predictions(directory / BINARY_PREDICTION_FILE)
    aggregate = "".join(
        path.read_text(encoding="utf-8")
        for path in (
            directory / PREDICTION_MANIFEST_FILE,
            directory / VALIDATION_RESULT_FILE,
            directory / QUALITY_REPORT_JSON_FILE,
            directory / QUALITY_REPORT_MD_FILE,
        )
    )
    for row in rows[:20]:
        assert row.anchor_event_id not in aggregate


def test_the_module_entry_point_reaches_the_same_command() -> None:
    """``python -m`` and the console script are one program."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "password_attack_detector", "ml", "predict", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--split" in completed.stdout
    assert "--labels" not in _declared_options(completed.stdout)


def test_the_published_rows_never_contain_a_prohibited_column(
    prepared: tuple[Path, Path],
) -> None:
    """The privacy sweep, run over the artifact rather than over the schema."""
    import pyarrow.parquet as pq

    from password_attack_detector.ml.predictions import PROHIBITED_PREDICTION_COLUMNS

    workspace, root = prepared
    assert predict(workspace, root).exit_code == 0
    directory = published_directory(root)
    for path in directory.glob("*.parquet"):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        assert not names & PROHIBITED_PREDICTION_COLUMNS, path.name


def test_a_publication_survives_being_moved_to_another_root(
    prepared: tuple[Path, Path], tmp_path: Path
) -> None:
    """Nothing in a publication depends on where it was written."""
    workspace, root = prepared
    assert predict(workspace, root).exit_code == 0
    source = published_directory(root)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(source, elsewhere / PREDICTIONS_DIR / source.name)
    validated = invoke("ml", "validate", "--output-root", str(elsewhere))
    assert validated.exit_code == 0, validated.output
    assert "Validation PASS" in validated.stdout
