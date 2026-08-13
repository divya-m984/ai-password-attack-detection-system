"""End-to-end tests for ``ml drift`` over a pipeline that really ran.

Nothing is mocked. The reference profile is captured from the training rows of a
real published feature dataset, and the comparison runs against real published
predictions.

Two properties are swept across the file. **The reference is immutable** -- it is
written once, reused byte for byte, and refused when the inputs would capture a
different one. And **monitoring changes nothing** -- the whole artifact root is
hashed either side of a run, so a write to a model, a lock, a ledger, a
selection, or an evaluation would fail here.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from tests.integration.ml_workspace import (
    ML_CONFIG,
    PSEUDONYM_RE,
    build_workspace,
    drift,
    freeze,
    predict,
    prediction_ids,
)

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)

#: Where the single frozen profile lives under an artifact root.
PROFILE = Path("reference") / "reference_profile.json"


@pytest.fixture(scope="module")
def pipeline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, dict[str, str]]:
    """Freeze a champion and publish train and validation predictions, once."""
    workspace = build_workspace(tmp_path_factory.mktemp("drift-workspace"))
    root = tmp_path_factory.mktemp("drift-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("drift-reports"))
    for split in ("train", "validation"):
        scored = predict(workspace, root, split=split)
        assert scored.exit_code == 0, scored.output
    return (workspace, root, prediction_ids(root))


@pytest.fixture
def scratch(
    pipeline: tuple[Path, Path, dict[str, str]], tmp_path: Path
) -> tuple[Path, Path, Path, dict[str, str]]:
    """A private copy of the artifact root, so a run cannot affect another test."""
    workspace, root, ids = pipeline
    copy = tmp_path / "artifacts"
    shutil.copytree(root, copy)
    return (workspace, copy, tmp_path / "reports", ids)


def _state(root: Path) -> dict[str, str]:
    """Return a digest of every artifact drift must not touch."""
    captured: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "reference" in path.parts:
            continue
        captured[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return captured


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_comparing_the_training_population_against_itself_reports_no_drift(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """The strongest available control: the reference is its own incoming set.

    Exit code zero as well as status ``no_drift`` -- the command signals a
    finding through its exit code, so a run that found nothing must not.
    """
    workspace, root, reports, _ = scratch
    result = drift(workspace, root, reports, incoming_split="train")
    assert result.exit_code == 0, result.output
    assert "no_drift" in result.output


def test_the_command_writes_a_report_and_a_profile_rendering(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Three documents, and the JSON carries both the manifest and the report."""
    workspace, root, reports, _ = scratch
    assert drift(workspace, root, reports, incoming_split="train").exit_code == 0
    payload = json.loads((reports / "ml_drift_report.json").read_text(encoding="utf-8"))
    assert set(payload) == {"drift_manifest", "drift_report", "drift_schema_version"}
    assert (reports / "ml_drift_report.md").is_file()
    assert (reports / "ml_reference_profile.md").is_file()


def test_a_later_split_is_compared_and_the_finding_is_reported(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Chronological splits genuinely differ; the exit code carries the finding.

    A non-zero exit here is the command working, not failing: the report exists,
    the statuses are real, and the caveat says what a finding is and is not.
    """
    workspace, root, reports, _ = scratch
    result = drift(workspace, root, reports, incoming_split="validation")
    assert result.exit_code == 1
    assert "Drift reported" in result.output
    assert "not an instruction to retrain" in result.output
    assert (reports / "ml_drift_report.json").is_file()


def test_prediction_drift_is_reported_separately_from_feature_drift(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Two aggregates, never blended into one verdict."""
    workspace, root, reports, ids = scratch
    result = drift(
        workspace,
        root,
        reports,
        incoming_split="train",
        reference_prediction=ids["train"],
        incoming_prediction=ids["train"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((reports / "ml_drift_report.json").read_text(encoding="utf-8"))[
        "drift_report"
    ]
    assert report["feature_status"] == "no_drift"
    assert report["prediction_status"] == "no_drift"
    assert report["predictions"]
    quantities = {item["quantity"] for item in report["predictions"]}
    assert "flagged_malicious_rate" in quantities
    assert "decision_score" in quantities


def test_a_shifted_prediction_population_moves_the_prediction_status(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """The output distribution is compared on its own terms."""
    workspace, root, reports, ids = scratch
    result = drift(
        workspace,
        root,
        reports,
        incoming_split="validation",
        reference_prediction=ids["train"],
        incoming_prediction=ids["validation"],
    )
    assert result.exit_code == 1
    report = json.loads((reports / "ml_drift_report.json").read_text(encoding="utf-8"))[
        "drift_report"
    ]
    assert report["prediction_status"] in {"drift_warning", "drift_detected"}


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_the_reference_profile_is_written_once_and_reused(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """A second run must reuse the frozen bytes, not recapture them."""
    workspace, root, reports, _ = scratch
    assert drift(workspace, root, reports, incoming_split="train").exit_code == 0
    first = (root / PROFILE).read_bytes()
    assert drift(workspace, root, reports, incoming_split="validation").exit_code == 1
    assert (root / PROFILE).read_bytes() == first


def test_recapturing_a_different_reference_is_refused(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Rebaselining a monitor onto its own newest input is how drift disappears.

    The second run is given a genuinely different capture -- a coarser
    partition -- and must refuse rather than overwrite.
    """
    workspace, root, reports, _ = scratch
    assert drift(workspace, root, reports, incoming_split="train").exit_code == 0
    frozen = (root / PROFILE).read_bytes()

    payload = yaml.safe_load(Path(ML_CONFIG).read_text(encoding="utf-8"))
    payload["drift"]["quantile_count"] = 8
    altered = reports.parent / "coarser.yaml"
    altered.parent.mkdir(parents=True, exist_ok=True)
    altered.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = drift(
        workspace,
        root,
        reports,
        incoming_split="train",
        **{"--config": str(altered)},
    )
    assert result.exit_code != 0
    assert "immutable" in result.output
    assert (root / PROFILE).read_bytes() == frozen


def test_the_captured_profile_names_the_training_split(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """The reviewed source, recorded on the artifact rather than assumed."""
    workspace, root, reports, _ = scratch
    assert drift(workspace, root, reports, incoming_split="train").exit_code == 0
    profile = json.loads((root / PROFILE).read_text(encoding="utf-8"))
    assert profile["reference_split"] == "train"
    assert profile["reference_prediction_id"] is None


# ---------------------------------------------------------------------------
# What it must not do
# ---------------------------------------------------------------------------


def test_the_command_takes_no_labels_option(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Drift needs no ground truth, so there is nowhere to supply one."""
    workspace, root, reports, _ = scratch
    result = drift(
        workspace,
        root,
        reports,
        incoming_split="train",
        **{"--labels": str(workspace / "processed" / "feature_labels.parquet")},
    )
    assert result.exit_code != 0


def test_monitoring_modifies_no_frozen_artifact(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Nothing outside the reference directory may change.

    Hashed rather than asserted: a write to the model, the lock, the ledger, the
    selection, or a publication would show up here whatever the code intended.
    """
    workspace, root, reports, ids = scratch
    before = _state(root)
    drift(
        workspace,
        root,
        reports,
        incoming_split="validation",
        reference_prediction=ids["train"],
        incoming_prediction=ids["validation"],
    )
    assert _state(root) == before


def test_a_one_sided_prediction_comparison_is_refused(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """A baseline against nothing, or nothing against a baseline, measures neither."""
    workspace, root, reports, ids = scratch
    result = drift(
        workspace,
        root,
        reports,
        incoming_split="train",
        incoming_prediction=ids["train"],
    )
    assert result.exit_code != 0
    assert "both a reference publication" in result.output


def test_the_output_carries_no_identifier_or_absolute_path(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """The same sweep every other command in this group is held to."""
    workspace, root, reports, _ = scratch
    result = drift(workspace, root, reports, incoming_split="train")
    assert result.exit_code == 0
    assert not PSEUDONYM_RE.search(result.output)
    assert not _UUID_RE.search(result.output)
    assert str(Path.home()) not in result.output


def test_no_report_carries_a_row_or_an_expected_mass(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """Aggregate only: the rendered documents must not restate the distribution."""
    workspace, root, reports, _ = scratch
    assert drift(workspace, root, reports, incoming_split="train").exit_code == 0
    for name in ("ml_drift_report.md", "ml_reference_profile.md"):
        text = (reports / name).read_text(encoding="utf-8")
        assert "anchor_event_id" not in text
        assert "reference_proportion" not in text
        assert not PSEUDONYM_RE.search(text)


def test_the_rendered_report_states_that_nothing_retrained(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """The most important sentence in the document, asserted as text."""
    workspace, root, reports, _ = scratch
    assert drift(workspace, root, reports, incoming_split="train").exit_code == 0
    lowered = (reports / "ml_drift_report.md").read_text(encoding="utf-8").lower()
    assert "no retraining happened" in lowered
    assert "monitoring evidence, **not** model correctness" in lowered


def test_a_disabled_configuration_refuses_rather_than_running(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """A capability the reviewed configuration turns off is not run anyway."""
    workspace, root, reports, _ = scratch
    payload = yaml.safe_load(Path(ML_CONFIG).read_text(encoding="utf-8"))
    payload["drift"]["enabled"] = False
    disabled = reports.parent / "disabled.yaml"
    disabled.parent.mkdir(parents=True, exist_ok=True)
    disabled.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = drift(
        workspace,
        root,
        reports,
        incoming_split="train",
        **{"--config": str(disabled)},
    )
    assert result.exit_code != 0
    assert "disabled" in result.output


def test_an_unknown_format_exits_non_zero(
    scratch: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    """A typo must fail loudly rather than fall back to a default."""
    workspace, root, reports, _ = scratch
    result = drift(
        workspace, root, reports, incoming_split="train", **{"--format": "yaml"}
    )
    assert result.exit_code == 1
