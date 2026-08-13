"""End-to-end tests for ``ml select`` over runs a real ``ml train`` published.

Nothing is mocked and nothing is hand-built: the dataset is generated, published
through ``features build``, trained through ``ml train``, and only then selected
from. A selection suite that fed the command a fixture object graph would pass
against artifact shapes the pipeline never produces.

Two things are asserted throughout. The command reaches a *decision* on this
micro-dataset -- a champion for the binary task, no head for the category task,
each with its reasons recorded -- and the command touches no test evidence:
there is no option to hand it any, nothing it writes mentions any, and the
absences are checked rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import Result

from password_attack_detector.ml.selection import SELECTION_DIR, SELECTION_FILE
from tests.integration.ml_workspace import (
    PSEUDONYM_RE,
    build_workspace,
    invoke,
    repo_root,
    train,
)

#: The configuration the runs were trained under. Selecting under a different
#: one is possible and is a different question; the suite asks the same one.
ML_CONFIG = str(repo_root() / "configs" / "ml" / "model-testing.yaml")


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a dataset and train every configured candidate once."""
    workspace = build_workspace(tmp_path_factory.mktemp("select-cli"))
    root = tmp_path_factory.mktemp("select-artifacts")
    result = train(workspace, root)
    assert result.exit_code == 0, result.output
    return root


def select(root: Path, reports: Path, **replace: str) -> Result:
    """Run ``ml select`` over the artifact root."""
    arguments = {
        "--output-root": str(root),
        "--config": ML_CONFIG,
        "--reports-dir": str(reports),
    }
    arguments.update(replace)
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return invoke("ml", "select", *flat)


@pytest.fixture(scope="module")
def selected(
    trained: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path, Result]:
    """Select once and return the artifact root, the reports, and the result."""
    reports = tmp_path_factory.mktemp("select-reports")
    result = select(trained, reports)
    return (trained, reports, result)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def test_selection_reaches_a_decision_on_published_runs(
    selected: tuple[Path, Path, Result],
) -> None:
    """Exit zero means a champion was selected, not merely that nothing broke."""
    _, _, result = selected
    assert result.exit_code == 0, result.output
    assert "Selected" in result.stdout
    assert "binary champion candidate" in result.stdout


def test_every_candidate_and_every_gate_is_printed(
    selected: tuple[Path, Path, Result],
) -> None:
    """Including the candidates that were blocked, and by what."""
    _, _, result = selected
    assert "Candidates" in result.stdout
    assert "Gates" in result.stdout
    for model in ("M-001", "M-010"):
        assert model in result.stdout, model


def test_the_negative_outcome_is_reported_alongside_the_positive(
    selected: tuple[Path, Path, Result],
) -> None:
    """The category head found nothing here, and the command says so."""
    _, reports, _ = selected
    report = json.loads((reports / "ml_gates.json").read_text(encoding="utf-8"))
    assert report["binary_malicious"]["status"] == "eligible"
    assert report["attack_category"]["status"] != "eligible"
    assert report["attack_category"]["selected_run_id"] is None


def test_the_reference_baseline_is_the_comparator_not_a_candidate(
    selected: tuple[Path, Path, Result],
) -> None:
    """M-000 is compared against and never appears among those compared."""
    _, reports, _ = selected
    binary = json.loads((reports / "ml_gates.json").read_text(encoding="utf-8"))[
        "binary_malicious"
    ]
    judged = {item["catalog_model_id"] for item in binary["candidates"]}
    assert "M-000" not in judged
    assert binary["reference_run_id"] is not None
    assert binary["selected_run_id"] != binary["reference_run_id"]


def test_a_baseline_without_an_operating_point_is_still_the_comparator(
    selected: tuple[Path, Path, Result],
) -> None:
    """M-000 finds no feasible threshold here, and still decides the gate.

    A constant score has no operating point under any ceiling worth
    configuring, so its run stops at ``threshold_unavailable``. Its exact
    ranking evidence is published all the same, and every candidate's
    improvement gate is *decided* against it rather than left unresolved.
    """
    root, reports, _ = selected
    binary = json.loads((reports / "ml_gates.json").read_text(encoding="utf-8"))[
        "binary_malicious"
    ]
    reference = root / "runs" / str(binary["reference_run_id"])
    record = json.loads((reference / "training_run.json").read_text(encoding="utf-8"))
    assert record["catalog_model_id"] == "M-000"
    assert record["status"] == "threshold_unavailable"

    evidence = json.loads(
        (reference / "ranking" / "validation_b_ranking.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["metric_name"] == "pr_auc"
    assert evidence["integration"] == "stepwise"
    assert evidence["score_kind"] == "decision_score"
    # A constant scorer: one score level, and PR-AUC is the positive prevalence.
    assert evidence["distinct_score_count"] == 1
    assert evidence["pr_auc"] == round(
        evidence["positive_count"] / evidence["row_count"], 9
    )

    for candidate in binary["candidates"]:
        gate = next(
            item
            for item in candidate["gates"]
            if item["gate_id"] == "baseline_pr_auc_gain"
        )
        assert gate["status"] in {"pass", "fail"}, gate
        assert "sampled" not in gate["reason"]


def test_the_selection_record_names_the_metric_it_decided_by(
    selected: tuple[Path, Path, Result],
) -> None:
    """Pinned in the immutable record, not left to whichever code was linked."""
    root, _, _ = selected
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / SELECTION_DIR).rglob(SELECTION_FILE))
    ]
    binary = next(item for item in records if item["task"] == "binary_malicious")
    category = next(item for item in records if item["task"] == "attack_category")
    assert binary["discrimination_metric"] == "pr_auc"
    assert binary["discrimination_integration"] == "stepwise"
    assert binary["discrimination_score_kind"] == "decision_score"
    # The category head has no reference comparison, so it declares no metric
    # rather than declaring one it never used.
    assert category["discrimination_metric"] is None
    assert category["reference_run_id"] is None


def test_the_selection_is_published_and_indexed(
    selected: tuple[Path, Path, Result],
) -> None:
    """A record on disk and a receipt in the ledger, for both tasks."""
    root, _, _ = selected
    published = sorted((root / SELECTION_DIR).iterdir())
    assert len(published) == 2
    assert all((item / SELECTION_FILE).is_file() for item in published)
    indexed = sorted((root / "ledger" / "validation_selection").glob("*.json"))
    assert len(indexed) == 2


def test_reports_are_written_where_they_were_asked_for(
    selected: tuple[Path, Path, Result],
) -> None:
    """Both renderings, and both name the limit of what was measured."""
    _, reports, _ = selected
    markdown = (reports / "ml_gates.md").read_text(encoding="utf-8")
    assert (reports / "ml_gates.json").is_file()
    assert "Validation-only" in markdown
    assert "no figure here describes performance on unseen data" in markdown


def test_selecting_again_records_no_second_finding(
    trained: Path, tmp_path: Path
) -> None:
    """The same evidence and the same criteria are the same selection."""
    before = len(list((trained / "ledger" / "validation_selection").glob("*.json")))
    result = select(trained, tmp_path / "again")
    after = len(list((trained / "ledger" / "validation_selection").glob("*.json")))
    assert result.exit_code == 0, result.output
    assert after == before


# ---------------------------------------------------------------------------
# The negative paths
# ---------------------------------------------------------------------------


def test_selecting_over_nothing_is_refused(tmp_path: Path) -> None:
    """A selection over no runs is not a selection with no champion."""
    result = select(tmp_path / "empty", tmp_path / "reports")
    assert result.exit_code != 0
    assert "No published training runs" in result.output


def test_an_unresolvable_selection_exits_two_not_one(
    trained: Path, tmp_path: Path
) -> None:
    """A finding is not a command failure, and is not reported as one.

    The development configuration's gates are far stricter than a 160-event
    micro-dataset can satisfy, which is exactly the state this exit code exists
    to name: no champion, no error, and a record of why.
    """
    result = select(
        trained,
        tmp_path / "strict-reports",
        **{"--config": str(repo_root() / "configs" / "ml" / "model-development.yaml")},
    )
    assert result.exit_code == 2, result.output
    assert "No champion selected" in result.output
    assert "never promoted" in result.output


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_the_command_takes_no_data_path_at_all() -> None:
    """No features, no labels, no splits: there is nowhere to hand it a row."""
    result = invoke("ml", "select", "--help")
    assert result.exit_code == 0
    for absent in (
        "--features",
        "--labels",
        "--splits",
        "--test",
        "--allow-test",
        "--force",
    ):
        assert absent not in result.stdout, absent


def test_nothing_written_or_printed_mentions_test_evidence(
    selected: tuple[Path, Path, Result],
) -> None:
    """Not in the record, not in the report, not on the terminal.

    The disclaimer sentence naming the holdout is the one place either word may
    appear, and it appears to say the holdout was *not* read -- so the check is
    for the fields a test figure would have to arrive in, not for the word.
    """
    root, reports, result = selected
    written = (reports / "ml_gates.json").read_text(encoding="utf-8")
    written += (reports / "ml_gates.md").read_text(encoding="utf-8")
    for record in sorted((root / SELECTION_DIR).rglob(SELECTION_FILE)):
        written += record.read_text(encoding="utf-8")
    for banned in (
        "test_metrics",
        "test_evaluation",
        "test_score",
        "holdout_metrics",
        "novel_anomaly_metrics",
    ):
        assert banned not in written.lower(), banned
        assert banned not in result.output.lower(), banned

    records = "".join(
        record.read_text(encoding="utf-8")
        for record in sorted((root / SELECTION_DIR).rglob(SELECTION_FILE))
    ).lower()
    assert "holdout" not in records
    assert "test" not in json.dumps(
        [
            list(json.loads(record.read_text(encoding="utf-8")))
            for record in sorted((root / SELECTION_DIR).rglob(SELECTION_FILE))
        ]
    )


def test_selection_writes_no_champion_lock(selected: tuple[Path, Path, Result]) -> None:
    """Selecting is not freezing; the lock is a separate, later decision."""
    root, _, _ = selected
    assert not list(root.rglob("champion.lock"))
    assert not (root / "ledger" / "champion_freeze").exists()


def test_no_pseudonym_or_absolute_path_reaches_the_terminal(
    selected: tuple[Path, Path, Result],
) -> None:
    """Selection output is model identifiers, verdicts, counts, and codes."""
    _, _, result = selected
    assert not PSEUDONYM_RE.search(result.output)
    assert str(Path.home()) not in result.output
    assert "/tmp/" not in result.output
