"""End-to-end tests for ``ml explain`` over a pipeline that really ran.

Nothing is mocked. The dataset is generated, published through ``features
build``, trained, selected, frozen, scored, and only then explained. An
attribution suite built on fixture objects would pass against artifact shapes
the pipeline never produces.

Two properties are swept across the file. **Attribution changes nothing** --
every frozen artifact is captured before the command runs and compared after it,
so a write anywhere in the model, the lock, or the publication fails here. And
**the command discloses only what it was asked to** -- no anchor, no feature
vector, no coefficient, no pseudonym, no absolute path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from password_attack_detector.ml.champion import CHAMPION_LOCK_FILE
from tests.integration.ml_workspace import (
    PSEUDONYM_RE,
    build_workspace,
    explain,
    explanation_directory,
    freeze,
    predict,
    prediction_ids,
)

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    """Freeze a champion and publish validation predictions, once."""
    workspace = build_workspace(tmp_path_factory.mktemp("explain-workspace"))
    root = tmp_path_factory.mktemp("explain-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("explain-reports"))
    scored = predict(workspace, root, split="validation")
    assert scored.exit_code == 0, scored.output
    return (workspace, root, prediction_ids(root)["validation"])


def _frozen_state(root: Path) -> dict[str, str]:
    """Return a digest of every frozen artifact attribution must not touch."""
    import hashlib

    captured: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "explanations" in path.parts:
            continue
        captured[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return captured


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_explaining_a_published_validation_prediction_succeeds(
    pipeline: tuple[Path, Path, str],
) -> None:
    """The whole path works against artifacts the pipeline really produced."""
    workspace, root, prediction = pipeline
    result = explain(workspace, root, prediction_id=prediction)
    assert result.exit_code == 0, result.output
    assert "exact" in result.output
    assert "Descriptive model attribution only" in result.output


def test_the_command_publishes_a_manifest_a_report_and_a_rendering(
    pipeline: tuple[Path, Path, str],
) -> None:
    """Three artifacts, and the manifest binds the other two by digest."""
    workspace, root, prediction = pipeline
    assert explain(workspace, root, prediction_id=prediction).exit_code == 0
    directory = explanation_directory(root)
    manifest = json.loads(
        (directory / "explanation_manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (directory / "explanation_report.json").read_text(encoding="utf-8")
    )
    assert (directory / "explanation_report.md").is_file()
    assert (
        manifest["explanation_report_fingerprint"]
        == report["explanation_report_fingerprint"]
    )


def test_the_manifest_binds_the_champion_and_the_publication(
    pipeline: tuple[Path, Path, str],
) -> None:
    """An explanation is only meaningful against one model and one input."""
    workspace, root, prediction = pipeline
    assert explain(workspace, root, prediction_id=prediction).exit_code == 0
    manifest = json.loads(
        (explanation_directory(root) / "explanation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads(
        (root / "champion")
        .glob(f"*/{CHAMPION_LOCK_FILE}")
        .__next__()
        .read_text(encoding="utf-8")
    )
    assert manifest["champion_lock_fingerprint"] == lock["lock_fingerprint"]
    assert manifest["prediction_id"] == prediction
    assert manifest["scope"] == "validation"


def test_no_row_explanation_is_written_by_default(
    pipeline: tuple[Path, Path, str],
) -> None:
    """The configured bound is zero, so the file must not exist at all."""
    workspace, root, prediction = pipeline
    assert explain(workspace, root, prediction_id=prediction).exit_code == 0
    assert not (explanation_directory(root) / "local_explanations.json").exists()


def test_the_explanation_is_byte_identical_across_runs(
    pipeline: tuple[Path, Path, str],
) -> None:
    """Same champion, same rows, same configuration -- same artifact."""
    workspace, root, prediction = pipeline
    assert explain(workspace, root, prediction_id=prediction).exit_code == 0
    first = (explanation_directory(root) / "explanation_manifest.json").read_bytes()
    assert explain(workspace, root, prediction_id=prediction).exit_code == 0
    assert (
        explanation_directory(root) / "explanation_manifest.json"
    ).read_bytes() == first


# ---------------------------------------------------------------------------
# What it must not do
# ---------------------------------------------------------------------------


def test_the_command_takes_no_labels_option(
    pipeline: tuple[Path, Path, str],
) -> None:
    """The firewall stated as a signature: there is nowhere to put one."""
    workspace, root, prediction = pipeline
    result = explain(
        workspace,
        root,
        prediction_id=prediction,
        **{"--labels": str(workspace / "processed" / "feature_labels.parquet")},
    )
    assert result.exit_code != 0


def test_attribution_modifies_no_frozen_artifact(
    pipeline: tuple[Path, Path, str],
) -> None:
    """Every byte outside the explanation directory must survive unchanged.

    Not a claim about what the code intends: the whole artifact root is hashed
    before and after, so a write to the model, the lock, the ledger, the
    selection, or the publication would show up here.
    """
    workspace, root, prediction = pipeline
    before = _frozen_state(root)
    assert explain(workspace, root, prediction_id=prediction).exit_code == 0
    assert _frozen_state(root) == before


@pytest.mark.parametrize("split", ["test", "novel_anomaly_holdout"])
def test_the_evaluation_populations_cannot_be_explained(
    pipeline: tuple[Path, Path, str], split: str
) -> None:
    """Refused by the command, and refused again by the library behind it."""
    workspace, root, prediction = pipeline
    result = explain(workspace, root, prediction_id=prediction, split=split)
    assert result.exit_code != 0


def test_a_publication_from_another_split_is_refused(
    pipeline: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """An attribution filed against another population's predictions is wrong."""
    workspace, root, _ = pipeline
    scored = predict(workspace, root, split="train")
    assert scored.exit_code == 0, scored.output
    other = prediction_ids(root)["train"]

    result = explain(workspace, root, prediction_id=other, split="validation")
    assert result.exit_code != 0
    assert "scores" in result.output or "not the rows" in result.output


def test_a_tampered_publication_is_refused(
    pipeline: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """Explaining an invalid publication would read like explaining a sound one."""
    import shutil

    workspace, root, prediction = pipeline
    copy = tmp_path / "artifacts"
    shutil.copytree(root, copy)
    target = copy / "predictions" / prediction / "prediction_manifest.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["row_count"] = payload["row_count"] + 1
    target.write_text(json.dumps(payload), encoding="utf-8")

    result = explain(workspace, copy, prediction_id=prediction)
    assert result.exit_code != 0


def test_the_output_carries_no_identifier_or_absolute_path(
    pipeline: tuple[Path, Path, str],
) -> None:
    """The same sweep every other command in this group is held to."""
    workspace, root, prediction = pipeline
    result = explain(workspace, root, prediction_id=prediction)
    assert result.exit_code == 0
    assert not PSEUDONYM_RE.search(result.output)
    assert not _UUID_RE.search(result.output)
    assert str(Path.home()) not in result.output


def test_the_output_prints_no_feature_vector(
    pipeline: tuple[Path, Path, str],
) -> None:
    """Column names are approved; the values behind them are not."""
    workspace, root, prediction = pipeline
    result = explain(workspace, root, prediction_id=prediction)
    assert result.exit_code == 0
    assert "transformed_value" not in result.output
    assert "coefficient" not in result.output.lower()


def test_the_markdown_rendering_states_the_causal_disclaimer(
    pipeline: tuple[Path, Path, str],
) -> None:
    """The document travels; the caveat must travel with it."""
    workspace, root, prediction = pipeline
    result = explain(
        workspace, root, prediction_id=prediction, **{"--format": "markdown"}
    )
    assert result.exit_code == 0
    lowered = result.output.lower()
    assert "descriptive, not causal" in lowered
    assert "no label was read" in lowered


def test_an_unknown_format_exits_non_zero(
    pipeline: tuple[Path, Path, str],
) -> None:
    """A typo must fail loudly rather than fall back to a default."""
    workspace, root, prediction = pipeline
    result = explain(workspace, root, prediction_id=prediction, **{"--format": "yaml"})
    assert result.exit_code == 1


def test_a_disabled_configuration_refuses_rather_than_running(
    pipeline: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """A capability the reviewed configuration turns off is not run anyway."""
    import yaml

    workspace, root, prediction = pipeline
    from tests.integration.ml_workspace import ML_CONFIG

    payload = yaml.safe_load(Path(ML_CONFIG).read_text(encoding="utf-8"))
    payload["explain"]["enabled"] = False
    disabled = tmp_path / "disabled.yaml"
    disabled.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = explain(
        workspace, root, prediction_id=prediction, **{"--config": str(disabled)}
    )
    assert result.exit_code != 0
    assert "disabled" in result.output
