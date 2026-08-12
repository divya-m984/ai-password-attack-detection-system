"""End-to-end tests for ``ml evaluate`` over a pipeline that really ran.

Nothing is mocked. The dataset is generated, published through ``features
build``, trained through ``ml train``, selected through ``ml select``, frozen
through ``ml freeze-champion``, scored through ``ml predict``, assessed through
``detection run``, and only then evaluated. An evaluation suite built on
fixture objects would pass against artifact shapes the pipeline never produces.

Two properties are swept across the file. **The evaluation is locked**: it
verifies a frozen lineage before it opens a label, refuses when any part of that
lineage disagrees, and publishes a receipt that is immutable and idempotent.
And **the comparison is fair**: rule-only, ML-only, and hybrid are measured over
one identical population, a system that could not be measured says so rather
than being dropped, and nothing declares a winner.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import Result

from password_attack_detector.ml.test_evaluation import (
    CATEGORY_EVALUATION_JSON,
    EVALUATION_RECEIPT_FILE,
    EVALUATIONS_DIR,
    ML_EVALUATION_JSON,
    ML_EVALUATION_MD,
    SYSTEM_COMPARISON_JSON,
    SYSTEM_COMPARISON_MD,
)
from tests.integration.ml_workspace import (
    PSEUDONYM_RE,
    build_workspace,
    detect,
    evaluate,
    freeze,
    invoke,
    predict,
    prediction_ids,
)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Run the whole pipeline once: freeze, predict TEST, and assess with rules."""
    workspace = build_workspace(tmp_path_factory.mktemp("evaluate-workspace"))
    root = tmp_path_factory.mktemp("evaluate-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("evaluate-reports"))

    scored = predict(workspace, root, split="test")
    assert scored.exit_code == 0, scored.output
    # Validation is scored too, so the pre-TEST fusion stage has the
    # validation-B evidence every candidate is judged on.
    on_validation = predict(workspace, root, split="validation")
    assert on_validation.exit_code == 0, on_validation.output

    detection = tmp_path_factory.mktemp("evaluate-detection")
    assessed = detect(workspace, detection)
    assert assessed.exit_code == 0, assessed.output
    return (workspace, root, detection)


@pytest.fixture
def prepared(
    pipeline: tuple[Path, Path, Path], tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    """Return a writable copy of the pipeline, one per test."""
    workspace, root, detection = pipeline
    copied = tmp_path / "artifacts"
    shutil.copytree(root, copied)
    return (workspace, copied, detection, tmp_path / "reports")


def _published(root: Path) -> Path:
    """Return the single published evaluation directory."""
    directories = sorted((root / EVALUATIONS_DIR).iterdir())
    assert len(directories) == 1, directories
    return directories[0]


def _run(prepared: tuple[Path, Path, Path, Path], **replace: str) -> Result:
    """Evaluate the prepared pipeline, with the fusion stage wired in."""
    workspace, root, detection, reports = prepared
    published = prediction_ids(root)
    arguments = {
        "--prediction": published["test"],
        "--validation-prediction": published["validation"],
    }
    arguments.update(replace)
    return evaluate(workspace, root, detection, reports, **arguments)


# ---------------------------------------------------------------------------
# The evaluation itself
# ---------------------------------------------------------------------------


def test_evaluating_publishes_a_complete_receipt(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """One command produces a receipt and every report the milestone declares."""
    result = _run(prepared)
    assert result.exit_code == 0, result.output

    directory = _published(prepared[1])
    assert (directory / EVALUATION_RECEIPT_FILE).is_file()
    for report in (
        ML_EVALUATION_JSON,
        ML_EVALUATION_MD,
        SYSTEM_COMPARISON_JSON,
        SYSTEM_COMPARISON_MD,
    ):
        assert (directory / report).is_file(), report


def test_the_reports_are_written_where_they_were_asked_for(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """``--reports-dir`` receives the rendered documents."""
    result = _run(prepared)
    assert result.exit_code == 0, result.output
    reports = prepared[3]
    assert (reports / ML_EVALUATION_MD).is_file()
    assert (reports / SYSTEM_COMPARISON_MD).is_file()


def test_the_category_head_is_evaluated_as_downstream_triage(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """Its denominator is the applicable population, not every scored row."""
    result = _run(prepared)
    assert result.exit_code == 0, result.output
    directory = _published(prepared[1])
    if not (directory / CATEGORY_EVALUATION_JSON).is_file():
        pytest.skip("no category head was frozen for this configuration")
    payload = json.loads(
        (directory / CATEGORY_EVALUATION_JSON).read_text(encoding="utf-8")
    )
    category = payload["category_evaluation"]
    assert (
        category["applicable_row_count"]
        + category["not_applicable_count"]
        + category["unknown_count"]
        == category["scored_row_count"]
    )
    # "not applicable" and "unknown" are different findings, and collapsing
    # them would turn "the binary head said benign" into "the category head
    # could not decide".
    assert category["not_applicable_count"] >= 0
    assert category["unknown_count"] >= 0


def test_evaluating_twice_publishes_once(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """An identical evaluation is not republished, and nothing is rewritten."""
    first = _run(prepared)
    assert first.exit_code == 0, first.output
    directory = _published(prepared[1])
    before = (directory / EVALUATION_RECEIPT_FILE).read_bytes()

    second = _run(prepared)
    assert second.exit_code == 0, second.output
    assert _published(prepared[1]) == directory
    assert (directory / EVALUATION_RECEIPT_FILE).read_bytes() == before


def test_the_evaluation_appends_exactly_one_ledger_record(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """Two identical evaluations leave one immutable record, not two."""
    from password_attack_detector.ml.ledger import ExperimentLedger

    assert _run(prepared).exit_code == 0
    assert _run(prepared).exit_code == 0
    ledger = ExperimentLedger(prepared[1] / "ledger")
    assert len(ledger.test_evaluations()) == 1


# ---------------------------------------------------------------------------
# The fair comparison
# ---------------------------------------------------------------------------


def test_every_system_is_measured_over_one_identical_population(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """The comparison names one population and every arm sits on it."""
    assert _run(prepared).exit_code == 0
    directory = _published(prepared[1])
    payload = json.loads(
        (directory / SYSTEM_COMPARISON_JSON).read_text(encoding="utf-8")
    )
    comparison = payload["comparison"]
    assert comparison["row_count"] > 0
    assert (
        comparison["positive_count"] + comparison["negative_count"]
        == comparison["row_count"]
    )
    systems = {entry["system"] for entry in comparison["systems"]}
    assert {"rule_only", "ml_only"} <= systems


def test_a_hybrid_that_was_never_established_says_so(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """No fusion selection means a stated absence, never a fabricated arm."""
    assert _run(prepared).exit_code == 0
    directory = _published(prepared[1])
    payload = json.loads(
        (directory / SYSTEM_COMPARISON_JSON).read_text(encoding="utf-8")
    )
    comparison = payload["comparison"]
    systems = {entry["system"] for entry in comparison["systems"]}
    if "hybrid" not in systems:
        assert comparison["hybrid_unavailable_reason"]


def test_the_rule_arm_carries_no_discrimination_metric(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """A 0-100 severity is not a ranking score, and is not reported as one."""
    assert _run(prepared).exit_code == 0
    directory = _published(prepared[1])
    payload = json.loads(
        (directory / SYSTEM_COMPARISON_JSON).read_text(encoding="utf-8")
    )
    rule = next(
        entry
        for entry in payload["comparison"]["systems"]
        if entry["system"] == "rule_only"
    )
    assert rule["continuous_score_available"] is False
    assert rule["metrics"]["pr_auc"] is None
    assert rule["metrics"]["pr_auc_unavailable_reason"]


def test_nothing_published_declares_a_winner(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """Which system to run is an operational decision this artifact cannot make."""
    assert _run(prepared).exit_code == 0
    directory = _published(prepared[1])
    for name in (SYSTEM_COMPARISON_JSON, SYSTEM_COMPARISON_MD):
        text = (directory / name).read_text(encoding="utf-8").lower()
        assert "winner" not in text
        assert "best system" not in text


def test_every_report_carries_the_synthetic_caveat(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """A reader must not take a synthetic figure as a production claim."""
    assert _run(prepared).exit_code == 0
    directory = _published(prepared[1])
    for name in (ML_EVALUATION_MD, SYSTEM_COMPARISON_MD):
        text = (directory / name).read_text(encoding="utf-8").lower()
        assert "synthetic" in text


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_evaluating_a_non_test_publication_is_refused(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """This command evaluates the locked test population and nothing else."""
    workspace, root, _detection, _reports = prepared
    scored = predict(workspace, root, split="train")
    assert scored.exit_code == 0, scored.output
    from password_attack_detector.ml.prediction_manifest import (
        PREDICTION_MANIFEST_FILE,
        PREDICTIONS_DIR,
    )

    published = sorted(
        item.name
        for item in (root / PREDICTIONS_DIR).iterdir()
        if (item / PREDICTION_MANIFEST_FILE).is_file()
    )
    for identifier in published:
        result = _run(prepared, **{"--prediction": identifier})
        if result.exit_code != 0:
            assert "TEST split" in result.output
            return
    pytest.fail("a non-test publication was accepted")


def test_a_tampered_publication_is_refused(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """Evaluation is not a way past manifest, checksum, or lineage checks."""
    from password_attack_detector.ml.prediction_manifest import (
        PREDICTION_MANIFEST_FILE,
        PREDICTIONS_DIR,
    )

    root = prepared[1]
    directory = next(
        item
        for item in sorted((root / PREDICTIONS_DIR).iterdir())
        if (item / PREDICTION_MANIFEST_FILE).is_file()
    )
    manifest = directory / PREDICTION_MANIFEST_FILE
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("prediction", "prediction ", 1),
        encoding="utf-8",
    )
    result = _run(prepared)
    assert result.exit_code == 1
    assert not (root / EVALUATIONS_DIR).exists() or not list(
        (root / EVALUATIONS_DIR).iterdir()
    )


def test_a_missing_risk_assessment_table_is_refused(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """The rule arm comes from a published run, and its absence is fatal."""
    result = _run(
        prepared, **{"--risk-assessments": str(prepared[2] / "absent.parquet")}
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_evaluating_without_a_frozen_champion_is_refused(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """Nothing is evaluated against a lineage that was never frozen."""
    workspace, root, detection, reports = prepared
    for name in ("champion.lock", "champion"):
        target = root / name
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    result = evaluate(workspace, root, detection, reports)
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Disclosure
# ---------------------------------------------------------------------------


def test_the_command_prints_no_identifier_or_absolute_path(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """Output is metrics and metadata, never an event, entity, or home path."""
    result = _run(prepared)
    assert result.exit_code == 0, result.output
    assert not PSEUDONYM_RE.search(result.stdout)
    assert str(Path.home()) not in result.stdout


def test_no_report_carries_a_pseudonym_or_a_raw_identifier(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """A published evaluation is aggregate; it names no subject."""
    assert _run(prepared).exit_code == 0
    directory = _published(prepared[1])
    for path in sorted(directory.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert not PSEUDONYM_RE.search(text), path.name


def test_the_help_states_that_this_is_the_one_command_reading_test_labels() -> None:
    """The firewall is documented where an operator will actually see it."""
    result = invoke("ml", "evaluate", "--help")
    assert result.exit_code == 0
    lowered = result.stdout.lower()
    assert "test ground truth" in lowered
    assert "frozen" in lowered


# ---------------------------------------------------------------------------
# The fusion stage
# ---------------------------------------------------------------------------


def test_all_three_strategies_enter_the_selection(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """STACKED competes on its merits, not on whether the CLI wired it in.

    The point of this test is the *third* row. OR_GATE and AND_GATE are pure
    functions of the validation evidence and were never at risk; STACKED needs
    genuine out-of-fold TRAIN refits, and an orchestration that skipped them
    would report it unavailable while looking exactly like this one.
    """
    result = _run(prepared)
    assert result.exit_code == 0, result.output
    for strategy in ("or_gate", "and_gate", "stacked"):
        assert strategy in result.stdout, strategy
    assert "out-of-fold folds" in result.stdout
    assert "not wired" not in result.stdout


def test_stacked_is_not_unavailable_for_an_architectural_reason(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """On a fixture with the campaign structure to support it, it is built."""
    result = _run(prepared)
    assert result.exit_code == 0, result.output
    for excuse in (
        "out_of_fold_evidence_not_supplied",
        "refits nothing",
        "does not wire",
    ):
        assert excuse not in result.stdout, excuse


def test_the_receipt_binds_the_frozen_fusion_selection(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """The evaluation records which strategy was frozen, and under what identity."""
    from password_attack_detector.ml.ledger import ExperimentLedger

    assert _run(prepared).exit_code == 0
    ledger = ExperimentLedger(prepared[1] / "ledger")
    record = ledger.test_evaluations()[0]
    assert record.fusion_selection_fingerprint is not None
    assert record.selected_fusion_strategy is not None


def test_the_hybrid_arm_is_measured_on_the_common_population(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """A selected strategy produces a real third arm, on the same rows."""
    assert _run(prepared).exit_code == 0
    directory = _published(prepared[1])
    payload = json.loads(
        (directory / SYSTEM_COMPARISON_JSON).read_text(encoding="utf-8")
    )
    comparison = payload["comparison"]
    hybrid = [e for e in comparison["systems"] if e["system"] == "hybrid"]
    assert hybrid, "the selected strategy produced no hybrid arm"
    assert hybrid[0]["metrics"]["row_count"] == comparison["row_count"]


def test_without_validation_evidence_no_strategy_is_substituted(
    prepared: tuple[Path, Path, Path, Path],
) -> None:
    """Absent evidence is a stated absence, never a default to OR_GATE."""
    workspace, root, detection, reports = prepared
    published = prediction_ids(root)
    result = evaluate(
        workspace,
        root,
        detection,
        reports,
        **{"--prediction": published["test"]},
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(
        (_published(root) / SYSTEM_COMPARISON_JSON).read_text(encoding="utf-8")
    )
    comparison = payload["comparison"]
    assert "hybrid" not in {e["system"] for e in comparison["systems"]}
    assert comparison["hybrid_unavailable_reason"]
