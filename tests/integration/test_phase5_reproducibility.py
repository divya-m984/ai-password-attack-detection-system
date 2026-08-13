"""The final Phase 5 reproducibility audit: M1 through M10, twice, from scratch.

The whole pipeline is rebuilt in two **independent temporary roots** and every
authoritative identity is compared. Nothing here is mocked and nothing is
carried between the two runs: each builds its own dataset, its own feature
publication, its own runs, its own champion, its own predictions, its own locked
evaluation, its own explanation, and its own reference profile.

Two properties are swept across the file. **Semantic identity does not depend on
where or when** -- the two roots sit at different paths, are built at different
wall-clock times, and must still agree fingerprint for fingerprint. And **a
semantic input that changes must move exactly the identities downstream of it**
-- the positive controls prove the lineage is live rather than accidentally
constant, and the negative controls prove that inputs a stage must not see
cannot reach it.

The 720-hour development workflow is never run here. A contract test that needs a
month of traffic to express itself is testing the generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.ml.champion import CHAMPION_LOCK_FILE
from password_attack_detector.ml.ledger import LEDGER_FILE
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
    invoke,
    predict,
    prediction_ids,
    train,
)

#: Where the frozen reference profile lives under an artifact root.
PROFILE = Path("reference") / "reference_profile.json"


def _workspace(base: Path, **kwargs: Any) -> Path:
    """Return a published workspace under *base*, creating the directory first."""
    target = base / "workspace"
    target.mkdir(parents=True, exist_ok=True)
    return build_workspace(target, **kwargs)


def _lock(root: Path) -> dict[str, Any]:
    """Return the single frozen champion lock under *root*."""
    payload = json.loads(
        next((root / "champion").glob(f"*/{CHAMPION_LOCK_FILE}")).read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(payload, dict)
    return payload


def _pipeline(base: Path) -> dict[str, Any]:
    """Build every Phase 5 artifact under *base* and return their identities.

    One function, run twice in two roots. Writing the two runs separately would
    let them diverge in a way the comparison could not see.
    """
    workspace = _workspace(base)
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

    return _identities(workspace, root, reports)


def _identities(workspace: Path, root: Path, reports: Path) -> dict[str, Any]:
    """Return every authoritative identity the pipeline produced.

    Deliberately *semantic*: no path, no timestamp, no host, and no directory
    name. Two runs in two places must agree on every value here.
    """
    manifest = json.loads(
        (workspace / "processed" / "feature_manifest.json").read_text(encoding="utf-8")
    )
    # The ledger index names its record *types*; each record lives in a
    # directory of its own. Reading the identities from the files rather than
    # from the index is what makes this a check on the records themselves.
    ledger = sorted(
        (path.parent.name, path.stem) for path in (root / "ledger").glob("*/*.json")
    )
    assert (root / "ledger" / LEDGER_FILE).is_file()
    lock = _lock(root)
    predictions = {
        json.loads((item / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8"))[
            "scope"
        ]: json.loads((item / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8"))
        for item in sorted((root / PREDICTIONS_DIR).iterdir())
        if (item / PREDICTION_MANIFEST_FILE).is_file()
    }
    receipt = json.loads(
        next((root / EVALUATIONS_DIR).glob(f"*/{EVALUATION_RECEIPT_FILE}")).read_text(
            encoding="utf-8"
        )
    )
    explanation = json.loads(
        (explanation_directory(root) / "explanation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    explanation_report = json.loads(
        (explanation_directory(root) / "explanation_report.json").read_text(
            encoding="utf-8"
        )
    )
    profile = json.loads((root / PROFILE).read_text(encoding="utf-8"))
    monitoring = json.loads(
        (reports / "ml_drift_report.json").read_text(encoding="utf-8")
    )

    return {
        # M2/M3: the data and feature contract
        "feature_content_fingerprint": manifest.get("content_fingerprint"),
        "eligible_features": lock["eligible_feature_list_fingerprint"],
        "allowlist": lock["allowlist_fingerprint"],
        "feature_catalog": lock["feature_catalog_fingerprint"],
        "preprocessor": lock["preprocessor_fingerprint"],
        # M4/M5/M6: model content, calibration, thresholds, the ledger
        "model_content": lock["model_content_fingerprint"],
        "model_manifest": lock["model_manifest_fingerprint"],
        "calibration_state": lock["calibration_state_fingerprint"],
        "binary_threshold": lock["binary_threshold_fingerprint"],
        "class_weight": lock["class_weight_fingerprint"],
        "ledger_records": ledger,
        # M7: selection and the freeze
        "training_run_id": lock["training_run_id"],
        "validation_selection_id": lock["validation_selection_id"],
        "validation_selection": lock["validation_selection_fingerprint"],
        "champion_lock": lock["lock_fingerprint"],
        "champion_scope_key": lock["scope_key"],
        # M8: prediction identity
        "predictions": {
            scope: (
                payload["prediction_id"],
                payload["prediction_content_fingerprint"],
                payload["prediction_manifest_fingerprint"],
                payload["inference_input_fingerprint"],
            )
            for scope, payload in predictions.items()
        },
        # M9: the locked evaluation and the fusion selection frozen before it
        "evaluation_identity": receipt["identity"],
        "evaluation_fingerprint": receipt["record_fingerprint"],
        "evaluation_population": receipt["evaluation_population_fingerprint"],
        "test_label_fingerprint": receipt["test_label_fingerprint"],
        "comparison_fingerprint": receipt["comparison_fingerprint"],
        "fusion_selection": receipt.get("fusion_selection_fingerprint"),
        "selected_strategy": receipt.get("selected_fusion_strategy"),
        # M10: attribution, the reference profile, and drift
        "explanation_id": explanation["explanation_id"],
        "explanation_manifest": explanation["explanation_manifest_fingerprint"],
        "explanation_report": explanation_report["explanation_report_fingerprint"],
        "reference_profile_id": profile["reference_profile_id"],
        "reference_profile": profile["reference_profile_fingerprint"],
        "reference_population": profile["reference_population_fingerprint"],
        "drift_run_id": monitoring["drift_manifest"]["drift_run_id"],
        "drift_report": monitoring["drift_report"]["drift_report_fingerprint"],
    }


@pytest.fixture(scope="module")
def twins(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], ...]:
    """Build the whole pipeline twice, in two unrelated directories."""
    first = _pipeline(tmp_path_factory.mktemp("repro-one"))
    second = _pipeline(tmp_path_factory.mktemp("repro-two"))
    return (first, second)


# ---------------------------------------------------------------------------
# M1 -> M10 semantic equality
# ---------------------------------------------------------------------------


def test_every_authoritative_identity_agrees_across_two_roots(
    twins: tuple[dict[str, Any], ...],
) -> None:
    """The audit in one assertion: nothing may depend on where it was built."""
    first, second = twins
    assert first == second


@pytest.mark.parametrize(
    "key",
    [
        "eligible_features",
        "allowlist",
        "feature_catalog",
        "preprocessor",
        "model_content",
        "model_manifest",
        "calibration_state",
        "binary_threshold",
        "class_weight",
        "training_run_id",
        "validation_selection_id",
        "validation_selection",
        "champion_lock",
        "champion_scope_key",
        "evaluation_identity",
        "evaluation_fingerprint",
        "evaluation_population",
        "test_label_fingerprint",
        "comparison_fingerprint",
        "explanation_id",
        "explanation_manifest",
        "explanation_report",
        "reference_profile_id",
        "reference_profile",
        "reference_population",
        "drift_run_id",
        "drift_report",
    ],
)
def test_each_stage_identity_agrees_individually(
    twins: tuple[dict[str, Any], ...], key: str
) -> None:
    """Named one by one, so a failure says which milestone moved."""
    first, second = twins
    assert first[key] == second[key], key


def test_the_ledger_records_agree(twins: tuple[dict[str, Any], ...]) -> None:
    """Every run, selection, freeze, and evaluation identity is derived."""
    first, second = twins
    assert first["ledger_records"] == second["ledger_records"]
    assert first["ledger_records"]


def test_every_prediction_identity_agrees(
    twins: tuple[dict[str, Any], ...],
) -> None:
    """Three scopes, four identities each, all content-derived."""
    first, second = twins
    assert set(first["predictions"]) == {"train", "validation", "test"}
    assert first["predictions"] == second["predictions"]


#: Identities a run may legitimately not have.  A champion fitted without class
#: weights carries no weight fingerprint, an uncalibrated one carries no
#: calibrator, and a validation-B fusion stage in which no strategy qualified
#: selects none -- all measured negatives rather than missing values.
OPTIONAL_IDENTITIES = frozenset(
    {"class_weight", "calibration_state", "fusion_selection", "selected_strategy"}
)


def test_no_required_identity_is_empty(twins: tuple[dict[str, Any], ...]) -> None:
    """Two runs that both produced nothing would agree trivially.

    The optional identities are excluded by name rather than by emptiness, so a
    stage that started returning nothing could not quietly join them.
    """
    first, _ = twins
    for key, value in first.items():
        if key in OPTIONAL_IDENTITIES:
            continue
        assert value not in (None, "", [], {}), key


def test_the_wall_clock_takes_no_part(twins: tuple[dict[str, Any], ...]) -> None:
    """The two runs happened at different times, and nothing recorded either.

    Asserted structurally rather than by waiting: no authoritative identity may
    be a timestamp, so a stage that started recording one would make the two
    dictionaries differ -- which the equality above already catches.
    """
    first, second = twins
    assert first["champion_lock"] == second["champion_lock"]
    lock_fields = json.dumps(first)
    for token in ("created_at", "timestamp", "generated_at", "hostname"):
        assert token not in lock_fields, token


# ---------------------------------------------------------------------------
# Positive controls: a semantic input moves its own lineage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perturbed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A pipeline built from a genuinely different training feature stream.

    One training event's response time is moved. That is a value the feature
    layer aggregates and the model is fitted on, so the design matrix is
    genuinely different -- and the feature contract, the catalog, and the
    allowlist are all untouched, which is what makes this a control on the
    *data* rather than on the configuration.
    """
    from tests.integration.ml_workspace import events

    base = tmp_path_factory.mktemp("repro-perturbed")
    stream = events()
    original = stream[3].response_time_ms
    assert original is not None
    stream[3] = stream[3].model_copy(update={"response_time_ms": original + 5_000})

    workspace = _workspace(base, stream=stream)
    root = base / "artifacts"
    freeze(workspace, root, base / "reports")
    return _lock(root)


def test_a_changed_training_feature_moves_the_downstream_lineage(
    twins: tuple[dict[str, Any], ...], perturbed: dict[str, Any]
) -> None:
    """The lineage is live: a real input change is visible at every later stage.

    Without this, two runs agreeing would be equally consistent with a pipeline
    whose fingerprints were constants.
    """
    first, _ = twins
    assert perturbed["preprocessor_fingerprint"] != first["preprocessor"]
    assert perturbed["model_content_fingerprint"] != first["model_content"]
    assert perturbed["lock_fingerprint"] != first["champion_lock"]


def test_a_narrowed_feature_configuration_moves_the_feature_contract(
    tmp_path: Path, twins: tuple[dict[str, Any], ...]
) -> None:
    """Narrowing which reviewed feature groups a run may fit on is semantic.

    The reviewed allowlist is unchanged -- every catalog feature is still
    reviewed, so the eligibility audit still passes -- and the *configuration*
    narrows what that review admits. The resolved feature list, the
    preprocessor, the model, and the lock must all move.
    """
    import yaml

    from tests.integration.ml_workspace import ML_CONFIG

    first, _ = twins
    workspace = _workspace(tmp_path)

    payload = yaml.safe_load(Path(ML_CONFIG).read_text(encoding="utf-8"))
    payload["preprocessing"]["include_feature_groups"] = [
        "current_context",
        "source_history",
        "user_history",
    ]
    narrowed = tmp_path / "narrowed.yaml"
    narrowed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    root = tmp_path / "artifacts"
    trained = train(workspace, root, **{"--config": str(narrowed)})
    assert trained.exit_code == 0, trained.output
    selected = invoke(
        "ml",
        "select",
        "--output-root",
        str(root),
        "--config",
        str(narrowed),
        "--reports-dir",
        str(tmp_path / "reports"),
    )
    assert selected.exit_code == 0, selected.output
    frozen = invoke(
        "ml", "freeze-champion", "--output-root", str(root), "--config", str(narrowed)
    )
    assert frozen.exit_code == 0, frozen.output

    lock = _lock(root)
    assert lock["eligible_feature_list_fingerprint"] != first["eligible_features"]
    assert lock["preprocessor_fingerprint"] != first["preprocessor"]
    assert lock["model_content_fingerprint"] != first["model_content"]
    assert lock["lock_fingerprint"] != first["champion_lock"]


# ---------------------------------------------------------------------------
# Negative controls: a forbidden input cannot reach an earlier stage
# ---------------------------------------------------------------------------


def test_incoming_monitoring_data_cannot_move_any_frozen_identity(
    tmp_path: Path,
) -> None:
    """A later population is compared, never absorbed.

    Two drift runs against genuinely different incoming populations, with every
    frozen identity captured either side of both.
    """
    import hashlib
    import shutil

    workspace = _workspace(tmp_path)
    root = tmp_path / "artifacts"
    freeze(workspace, root, tmp_path / "reports")
    for split in ("train", "validation"):
        assert predict(workspace, root, split=split).exit_code == 0

    def _state() -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "reference" not in path.parts
        }

    before = _state()
    assert (
        drift(workspace, root, tmp_path / "r1", incoming_split="train").exit_code == 0
    )
    drift(workspace, root, tmp_path / "r2", incoming_split="validation")
    assert _state() == before

    # And the profile itself, captured from TRAIN, is untouched by either.
    captured = (root / PROFILE).read_bytes()
    drift(workspace, root, tmp_path / "r3", incoming_split="validation")
    assert (root / PROFILE).read_bytes() == captured

    shutil.rmtree(tmp_path / "r1", ignore_errors=True)


def test_attribution_cannot_move_any_frozen_identity(tmp_path: Path) -> None:
    """An explanation is a description; it has nowhere to write back to."""
    import hashlib

    workspace = _workspace(tmp_path)
    root = tmp_path / "artifacts"
    freeze(workspace, root, tmp_path / "reports")
    assert predict(workspace, root, split="validation").exit_code == 0
    prediction = prediction_ids(root)["validation"]

    def _state() -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "explanations" not in path.parts
        }

    before = _state()
    assert explain(workspace, root, prediction_id=prediction).exit_code == 0
    assert _state() == before


def test_the_test_split_cannot_reach_training_or_the_champion(
    twins: tuple[dict[str, Any], ...], tmp_path: Path
) -> None:
    """Scoring TEST is safe precisely because nothing upstream can see it.

    A pipeline that never scores TEST at all must produce the identical
    champion, the identical preprocessor, and the identical model as one that
    does -- which is a stronger statement than checking that ``predict`` takes no
    ``--labels``.
    """
    first, _ = twins
    workspace = _workspace(tmp_path)
    root = tmp_path / "artifacts"
    freeze(workspace, root, tmp_path / "reports")

    lock = _lock(root)
    assert lock["lock_fingerprint"] == first["champion_lock"]
    assert lock["preprocessor_fingerprint"] == first["preprocessor"]
    assert lock["model_content_fingerprint"] == first["model_content"]
    assert lock["binary_threshold_fingerprint"] == first["binary_threshold"]


def test_source_row_order_does_not_affect_any_identity(tmp_path: Path) -> None:
    """Canonical ordering is applied before anything is fitted or fingerprinted.

    The feature table is rewritten in reverse physical order and the whole
    pipeline is rebuilt from it. Physical file order is not a semantic input, so
    nothing may move.
    """
    import pyarrow.parquet as pq

    first_root = tmp_path / "ordered"
    workspace = _workspace(first_root)
    freeze(workspace, first_root / "artifacts", first_root / "reports")
    ordered = _lock(first_root / "artifacts")

    second_root = tmp_path / "reversed"
    reversed_workspace = _workspace(second_root)
    for name in ("feature_snapshots", "feature_labels", "feature_splits"):
        path = reversed_workspace / "processed" / f"{name}.parquet"
        table = pq.read_table(path)
        pq.write_table(table.take(list(reversed(range(table.num_rows)))), path)
    freeze(reversed_workspace, second_root / "artifacts", second_root / "reports")
    shuffled = _lock(second_root / "artifacts")

    assert shuffled["lock_fingerprint"] == ordered["lock_fingerprint"]
    assert shuffled["preprocessor_fingerprint"] == ordered["preprocessor_fingerprint"]
    assert shuffled["model_content_fingerprint"] == ordered["model_content_fingerprint"]
