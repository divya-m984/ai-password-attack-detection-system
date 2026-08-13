"""End-to-end tests for ``ml profile`` over a real prediction publication.

The report is rebuilt from the publication alone -- the rows, the manifest, and a
fresh validation pass -- which is what makes it checkable rather than merely
informative, and the suite asserts that by rebuilding it twice and comparing
bytes.

The other half is the boundary. Everything ``ml profile`` reports describes the
*distribution of what the model said*; nothing describes whether it was right,
because establishing that needs labels this milestone never opens. The suite
sweeps the JSON, the Markdown, and the terminal for every figure that would.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import Result

from password_attack_detector.ml.prediction_manifest import (
    BINARY_PREDICTION_FILE,
    CATEGORY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    QUALITY_REPORT_JSON_FILE,
    QUALITY_REPORT_MD_FILE,
)
from tests.integration.ml_workspace import (
    PSEUDONYM_RE,
    build_workspace,
    freeze,
    invoke,
    predict,
)

#: Every figure a profile is forbidden to report, in the spellings a reader
#: would search for.
OUTCOME_TERMS = (
    "accuracy",
    "roc_auc",
    "pr_auc",
    "brier",
    "expected_calibration_error",
    "confusion_matrix",
    "true_positive",
    "false_negative",
)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Publish a dataset, freeze a champion, and predict once."""
    workspace = build_workspace(tmp_path_factory.mktemp("profile-workspace"))
    root = tmp_path_factory.mktemp("profile-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("profile-reports"))
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


def profile(root: Path, reports: Path, *extra: str) -> Result:
    """Run ``ml profile`` over *root*."""
    return invoke(
        "ml",
        "profile",
        "--output-root",
        str(root),
        "--reports-dir",
        str(reports),
        *extra,
    )


def directory_of(root: Path) -> Path:
    """Return the single published prediction directory."""
    return next((root / PREDICTIONS_DIR).iterdir())


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_profiling_writes_both_renderings(published: Path, tmp_path: Path) -> None:
    """JSON for a machine, Markdown for a person, from one report."""
    reports = tmp_path / "reports"
    result = profile(published, reports)
    assert result.exit_code == 0, result.output
    assert (reports / QUALITY_REPORT_JSON_FILE).is_file()
    assert (reports / QUALITY_REPORT_MD_FILE).is_file()


def test_the_report_describes_the_distribution_of_the_output(
    published: Path, tmp_path: Path
) -> None:
    """Counts, a threshold, a range, and whether a probability exists."""
    reports = tmp_path / "reports"
    assert profile(published, reports).exit_code == 0
    report = json.loads(
        (reports / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8")
    )
    binary = report["binary"]
    assert binary["total_rows"] > 0
    assert binary["flagged_count"] + binary["unflagged_count"] == binary["total_rows"]
    assert binary["decision_threshold"] is not None
    assert binary["score_kind"] in {"decision_score", "calibrated_probability"}
    assert report["validation_status"] == "pass"


def test_the_report_is_rebuilt_from_the_publication_alone(
    published: Path, tmp_path: Path
) -> None:
    """Two runs, byte for byte, with nothing observational in between."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    assert profile(published, first).exit_code == 0
    assert profile(published, second).exit_code == 0
    assert (first / QUALITY_REPORT_JSON_FILE).read_bytes() == (
        second / QUALITY_REPORT_JSON_FILE
    ).read_bytes()
    assert (first / QUALITY_REPORT_MD_FILE).read_bytes() == (
        second / QUALITY_REPORT_MD_FILE
    ).read_bytes()


def test_the_written_report_describes_the_same_publication(
    published: Path, tmp_path: Path
) -> None:
    """Profiling reproduces the distribution the publication already carries.

    Not byte-identical to the published report, and deliberately so: the
    publisher's copy records the *staged* validation, which ran before a
    manifest existed to check, while profiling runs the full published check
    set. The distributions are the same because they are derived from the same
    rows; the validation summary is stronger because more was checked.
    """
    reports = tmp_path / "reports"
    assert profile(published, reports).exit_code == 0
    rebuilt = json.loads(
        (reports / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8")
    )
    stored = json.loads(
        (directory_of(published) / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8")
    )
    assert rebuilt["prediction_id"] == stored["prediction_id"]
    assert (
        rebuilt["prediction_content_fingerprint"]
        == (stored["prediction_content_fingerprint"])
    )
    assert rebuilt["binary"] == stored["binary"]
    assert rebuilt["category"] == stored["category"]
    assert rebuilt["validation_check_count"] >= stored["validation_check_count"]


def test_the_markdown_rendering_can_be_printed(published: Path, tmp_path: Path) -> None:
    """``--format markdown`` prints the same document it writes."""
    reports = tmp_path / "reports"
    result = profile(published, reports, "--format", "markdown")
    assert result.exit_code == 0
    assert "# ML prediction profile" in result.stdout
    assert "## Binary predictions" in result.stdout


def test_an_unknown_format_is_refused(published: Path, tmp_path: Path) -> None:
    """Two renderings, and no third."""
    result = profile(published, tmp_path / "reports", "--format", "yaml")
    assert result.exit_code != 0
    assert "Unknown format" in result.output


def test_a_category_distribution_is_reported_when_a_head_was_frozen(
    published: Path, tmp_path: Path
) -> None:
    """Per class, plus abstentions, or an explicit absence."""
    reports = tmp_path / "reports"
    assert profile(published, reports).exit_code == 0
    report = json.loads(
        (reports / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8")
    )
    markdown = (reports / QUALITY_REPORT_MD_FILE).read_text(encoding="utf-8")
    if report["category"] is None:
        assert "No category head was frozen." in markdown
    else:
        category = report["category"]
        assert (
            category["known_count"] + category["unknown_count"]
            == category["applicable_row_count"]
        )
        assert "Abstention threshold" in markdown


# ---------------------------------------------------------------------------
# Unavailable is not zero
# ---------------------------------------------------------------------------


def test_an_unavailable_figure_is_rendered_as_unavailable(
    published: Path, tmp_path: Path
) -> None:
    """A quantity nothing could produce is never rendered as a measurement."""
    reports = tmp_path / "reports"
    assert profile(published, reports).exit_code == 0
    report = json.loads(
        (reports / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8")
    )
    markdown = (reports / QUALITY_REPORT_MD_FILE).read_text(encoding="utf-8")
    binary = report["binary"]
    if not binary["calibrated_probability_available"]:
        assert binary["probability_mean"] is None
        assert "| Probability mean | unavailable |" in markdown
    else:
        assert binary["probability_mean"] is not None
        assert binary["null_probability_count"] == 0


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_the_command_takes_no_label_argument() -> None:
    """A profile is derived from the publication and nothing else."""
    result = invoke("ml", "profile", "--help")
    assert result.exit_code == 0
    _, _, options = result.stdout.partition("Options")
    for absent in ("--labels", "--truth", "--features", "--splits"):
        assert absent not in options, absent


@pytest.mark.parametrize("term", OUTCOME_TERMS)
def test_no_outcome_figure_is_reported(
    published: Path, tmp_path: Path, term: str
) -> None:
    """Except in the paragraph that says none of them is computable."""
    reports = tmp_path / "reports"
    result = profile(published, reports)
    assert result.exit_code == 0
    assert (
        term
        not in (reports / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8").lower()
    ), term

    # The one place any of these words may appear is the sentence saying none of
    # them is computable, so the check is against everything before it.
    printed, _, printed_disclaimer = result.output.lower().partition(
        "no label was read"
    )
    assert term not in printed, term
    assert printed_disclaimer

    markdown = (reports / QUALITY_REPORT_MD_FILE).read_text(encoding="utf-8").lower()
    head, _, disclaimer = markdown.partition("## what this report is not")
    assert term not in head, term
    assert disclaimer


def test_the_markdown_states_the_limits_of_what_it_measured(
    published: Path, tmp_path: Path
) -> None:
    """Named in the artifact, not left to a reviewer's memory."""
    reports = tmp_path / "reports"
    assert profile(published, reports).exit_code == 0
    markdown = (reports / QUALITY_REPORT_MD_FILE).read_text(encoding="utf-8")
    assert "Structural validity is not predictive quality" in markdown
    assert "No label was read" in markdown
    assert "synthetic" in markdown


def test_no_anchor_pseudonym_or_absolute_path_is_reported(
    published: Path, tmp_path: Path
) -> None:
    """Aggregate output only, in every rendering."""
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
    )

    reports = tmp_path / "reports"
    result = profile(published, reports)
    assert result.exit_code == 0
    written = (reports / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8")
    written += (reports / QUALITY_REPORT_MD_FILE).read_text(encoding="utf-8")
    assert not PSEUDONYM_RE.search(written)
    assert not PSEUDONYM_RE.search(result.output)
    assert str(Path.home()) not in result.output
    for row in read_binary_predictions(
        directory_of(published) / BINARY_PREDICTION_FILE
    )[:20]:
        assert row.anchor_event_id not in written
        assert row.anchor_event_id not in result.output


def test_profiling_with_no_publications_is_refused(tmp_path: Path) -> None:
    """A profile over nothing is not a profile of an empty distribution."""
    result = profile(tmp_path / "empty", tmp_path / "reports")
    assert result.exit_code != 0
    assert "No predictions have been published" in result.output


def test_profiling_changes_nothing_on_disk(published: Path, tmp_path: Path) -> None:
    """Reading an artifact does not modify it."""
    directory = directory_of(published)
    before = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    assert profile(published, tmp_path / "reports").exit_code == 0
    after = {
        item.name: item.read_bytes() for item in directory.iterdir() if item.is_file()
    }
    assert after == before


def test_the_module_entry_point_reaches_the_same_command() -> None:
    """``python -m`` and the console script are one program."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "password_attack_detector", "ml", "profile", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--reports-dir" in completed.stdout


# ---------------------------------------------------------------------------
# Profiling refuses an invalid publication, and writes nothing
# ---------------------------------------------------------------------------


def assert_refused(result: Result, reports: Path, code: str) -> None:
    """Assert the profile was refused, named its reason, and wrote nothing."""
    assert result.exit_code != 0
    assert "Refusing to profile an invalid publication" in result.output
    assert code in result.output, code
    assert not (reports / QUALITY_REPORT_JSON_FILE).exists()
    assert not (reports / QUALITY_REPORT_MD_FILE).exists()


def test_checksum_tampering_produces_no_profile(
    published: Path, tmp_path: Path
) -> None:
    """Profiling is not a way past the manifest's own integrity evidence."""
    path = directory_of(published) / QUALITY_REPORT_JSON_FILE
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    reports = tmp_path / "reports"
    assert_refused(profile(published, reports), reports, "M006")


def test_a_row_level_semantic_contradiction_produces_no_profile(
    published: Path, tmp_path: Path
) -> None:
    """Rows that no longer recompute the manifest's content are not summarised."""
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
        write_binary_predictions,
    )

    rows = list(
        read_binary_predictions(directory_of(published) / BINARY_PREDICTION_FILE)
    )
    write_binary_predictions(
        rows[:-1], directory_of(published) / BINARY_PREDICTION_FILE
    )
    reports = tmp_path / "reports"
    assert_refused(profile(published, reports), reports, "M009")


def test_a_tampered_champion_lineage_produces_no_profile(
    published: Path, tmp_path: Path
) -> None:
    """A manifest whose champion lineage was edited fails its own seal."""
    path = directory_of(published) / PREDICTION_MANIFEST_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["lineage"]["champion_lock_fingerprint"] = "a" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    reports = tmp_path / "reports"
    assert_refused(profile(published, reports), reports, "M001")


def test_a_malformed_category_payload_produces_no_profile(
    published: Path, tmp_path: Path
) -> None:
    """A class map that will not parse is refused before any summary exists.

    Skipped when this configuration froze no category head -- the same refusal
    is proved against a real category artifact in the validation suite, where
    the fixture does freeze one.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from password_attack_detector.ml.prediction_serialization import (
        CATEGORY_PREDICTION_SCHEMA,
    )

    path = directory_of(published) / CATEGORY_PREDICTION_FILE
    if not path.is_file():
        pytest.skip("this publication carries no category head")
    table = pq.read_table(path)
    broken = table.set_column(
        table.schema.get_field_index("category_scores_json"),
        "category_scores_json",
        pa.array(["{not json" for _ in range(table.num_rows)], type=pa.string()),
    )
    pq.write_table(broken.cast(CATEGORY_PREDICTION_SCHEMA), path)
    reports = tmp_path / "reports"
    result = profile(published, reports)
    assert result.exit_code != 0
    assert not (reports / QUALITY_REPORT_JSON_FILE).exists()
    assert not (reports / QUALITY_REPORT_MD_FILE).exists()


def test_a_refused_profile_does_not_overwrite_an_earlier_one(
    published: Path, tmp_path: Path
) -> None:
    """A failed attempt leaves the last good report exactly as it was."""
    reports = tmp_path / "reports"
    assert profile(published, reports).exit_code == 0
    before = {
        name: (reports / name).read_bytes()
        for name in (QUALITY_REPORT_JSON_FILE, QUALITY_REPORT_MD_FILE)
    }

    path = directory_of(published) / QUALITY_REPORT_JSON_FILE
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert profile(published, reports).exit_code != 0

    after = {
        name: (reports / name).read_bytes()
        for name in (QUALITY_REPORT_JSON_FILE, QUALITY_REPORT_MD_FILE)
    }
    assert after == before


def test_a_refusal_prints_only_sanitized_codes(published: Path, tmp_path: Path) -> None:
    """Stable codes and a pointer to ``ml validate``; never a row or a path."""
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
    )

    rows = read_binary_predictions(directory_of(published) / BINARY_PREDICTION_FILE)
    path = directory_of(published) / QUALITY_REPORT_JSON_FILE
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = profile(published, tmp_path / "reports")
    assert result.exit_code != 0
    assert "ml validate" in result.output
    assert not PSEUDONYM_RE.search(result.output)
    assert str(Path.home()) not in result.output
    for row in rows[:20]:
        assert row.anchor_event_id not in result.output


def test_the_category_profile_denominator_is_the_applicable_population(
    published: Path, tmp_path: Path
) -> None:
    """The report divides by the rows routed to triage, not by every row."""
    reports = tmp_path / "reports"
    assert profile(published, reports).exit_code == 0
    report = json.loads(
        (reports / QUALITY_REPORT_JSON_FILE).read_text(encoding="utf-8")
    )
    category = report["category"]
    if category is None:
        pytest.skip("this publication carries no category head")
    assert category["binary_row_count"] == report["binary"]["total_rows"]
    assert category["applicable_row_count"] == report["binary"]["flagged_count"]
    assert category["not_applicable_count"] == report["binary"]["unflagged_count"]
    assert (
        category["known_count"] + category["unknown_count"]
        == (category["applicable_row_count"])
    )

    markdown = (reports / QUALITY_REPORT_MD_FILE).read_text(encoding="utf-8")
    assert "Not applicable (binary did not flag)" in markdown
    assert "Abstention rate (of applicable)" in markdown
