"""End-to-end tests for ``ml compare`` over an evaluation the pipeline published.

``ml compare`` is a *reader*. Everything it prints was measured once by
``ml evaluate``, published immutably, and is reproduced here verbatim. The
property this file exists to assert is that the command adds nothing: it opens
no label, computes no metric, cannot run without a published evaluation, and
does not name a winner among the systems it reports.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import Result

from password_attack_detector.ml.test_evaluation import (
    EVALUATIONS_DIR,
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
)


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the whole pipeline once and publish a TEST evaluation from it."""
    workspace = build_workspace(tmp_path_factory.mktemp("compare-workspace"))
    root = tmp_path_factory.mktemp("compare-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("compare-reports"))

    scored = predict(workspace, root, split="test")
    assert scored.exit_code == 0, scored.output

    detection = tmp_path_factory.mktemp("compare-detection")
    assessed = detect(workspace, detection)
    assert assessed.exit_code == 0, assessed.output

    reports = tmp_path_factory.mktemp("compare-evaluation-reports")
    result = evaluate(workspace, root, detection, reports)
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def prepared(evaluated: Path, tmp_path: Path) -> Path:
    """Return a writable copy of the published evaluation, one per test."""
    copied = tmp_path / "artifacts"
    shutil.copytree(evaluated, copied)
    return copied


def _compare(root: Path, *extra: str) -> Result:
    """Run ``ml compare`` against *root*."""
    return invoke("ml", "compare", "--output-root", str(root), *extra)


def _published(root: Path) -> Path:
    """Return the single published evaluation directory."""
    directories = sorted((root / EVALUATIONS_DIR).iterdir())
    assert len(directories) == 1, directories
    return directories[0]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_compare_reports_the_published_comparison(prepared: Path) -> None:
    """Every system in the receipt appears in the report."""
    result = _compare(prepared)
    assert result.exit_code == 0, result.output
    assert "rule_only" in result.stdout
    assert "ml_only" in result.stdout


def test_compare_reports_the_population_it_was_measured_on(prepared: Path) -> None:
    """One population, stated, so a reader cannot assume it differed per system."""
    result = _compare(prepared)
    assert result.exit_code == 0, result.output
    payload = json.loads(
        (_published(prepared) / SYSTEM_COMPARISON_JSON).read_text(encoding="utf-8")
    )
    assert f"{payload['comparison']['row_count']:,}" in result.stdout


def test_the_markdown_is_the_published_document_verbatim(prepared: Path) -> None:
    """Nothing is re-rendered; the receipt is the record."""
    result = _compare(prepared, "--format", "markdown")
    assert result.exit_code == 0, result.output
    published = (_published(prepared) / SYSTEM_COMPARISON_MD).read_text(
        encoding="utf-8"
    )
    # The console wraps, so compare on the distinctive lines rather than bytes.
    for line in published.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            assert stripped.lstrip("# ") in result.stdout


def test_compare_writes_the_document_where_it_was_asked_for(
    prepared: Path, tmp_path: Path
) -> None:
    """``--output`` receives the published Markdown."""
    target = tmp_path / "nested" / "system_comparison.md"
    result = _compare(prepared, "--output", str(target))
    assert result.exit_code == 0, result.output
    assert target.read_text(encoding="utf-8") == (
        _published(prepared) / SYSTEM_COMPARISON_MD
    ).read_text(encoding="utf-8")


def test_compare_is_deterministic(prepared: Path) -> None:
    """Two runs of a reader over an immutable artifact must agree."""
    first = _compare(prepared, "--format", "markdown")
    second = _compare(prepared, "--format", "markdown")
    assert first.stdout == second.stdout


def test_compare_declares_no_winner(prepared: Path) -> None:
    """Which system to run is an operational decision this command cannot make."""
    result = _compare(prepared)
    assert result.exit_code == 0, result.output
    lowered = result.stdout.lower()
    assert "no winner is declared" in lowered
    # The disclaimer is the only place the word may appear.
    assert lowered.count("winner") == 1
    assert "best system" not in lowered
    assert "recommended system" not in lowered


def test_compare_states_that_the_ground_truth_is_synthetic(prepared: Path) -> None:
    """A reader must not take a synthetic figure as a production claim."""
    result = _compare(prepared)
    assert result.exit_code == 0, result.output
    assert "synthetic" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_compare_without_a_published_evaluation_is_refused(tmp_path: Path) -> None:
    """Nothing here recomputes a comparison that was never published."""
    result = _compare(tmp_path / "empty")
    assert result.exit_code == 1
    assert "ml evaluate" in result.output


def test_an_unknown_evaluation_identifier_is_refused(prepared: Path) -> None:
    """A typo must fail loudly rather than fall back to whatever is present."""
    result = _compare(prepared, "--evaluation", "not-an-evaluation")
    assert result.exit_code == 1


def test_an_unknown_format_is_refused(prepared: Path) -> None:
    """A typo must not silently fall back to a default rendering."""
    result = _compare(prepared, "--format", "yaml")
    assert result.exit_code == 1


def test_an_evaluation_without_a_comparison_is_refused(prepared: Path) -> None:
    """A missing report is stated, never reconstructed."""
    (_published(prepared) / SYSTEM_COMPARISON_JSON).unlink()
    result = _compare(prepared)
    assert result.exit_code == 1
    assert "recomputes" in result.output


# ---------------------------------------------------------------------------
# The reader boundary
# ---------------------------------------------------------------------------


def test_compare_takes_no_label_argument() -> None:
    """The command has no way to be handed ground truth, by option or by path."""
    result = invoke("ml", "compare", "--help")
    assert result.exit_code == 0
    _, _, options = result.stdout.partition("Options")
    for forbidden in ("--labels", "--splits", "--features", "--risk-assessments"):
        assert forbidden not in options, forbidden


def test_compare_reads_no_parquet(prepared: Path) -> None:
    """It reads the published JSON and Markdown, and nothing else.

    Asserted by removing every table under the root: a command that still
    reports afterwards demonstrably read none of them.
    """
    for table in sorted(prepared.rglob("*.parquet")):
        table.unlink()
    result = _compare(prepared)
    assert result.exit_code == 0, result.output
    assert "rule_only" in result.stdout


def test_compare_prints_no_identifier_or_absolute_path(prepared: Path) -> None:
    """Output is aggregate; it names no subject and no home directory."""
    result = _compare(prepared)
    assert result.exit_code == 0, result.output
    assert not PSEUDONYM_RE.search(result.stdout)
    assert str(Path.home()) not in result.stdout
