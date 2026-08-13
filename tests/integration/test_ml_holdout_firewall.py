"""The novel-anomaly holdout reaches no supervised decision. Proven, not asserted.

The main firewall suite can only *skip* this property: the ordinary CI stream
generates no novel anomalies, so mutating them would mutate nothing. A skipped
firewall test is indistinguishable from a firewall that does not exist, so this
module builds a workspace that genuinely carries a holdout population and then
tries to move supervised state with it.

The fixture reassigns two early campaigns to ``NOVEL_ANOMALY_HOLDOUT``, giving
eight real holdout rows. Nothing is weakened to accommodate them -- the
campaigns are early precisely because the splits are chronological, so carving
out late ones would starve validation and TEST of positives and the eligibility
audit would refuse the run. Every gate, threshold rule, and out-of-fold rule is
the one the pipeline already applies.

The experiment has two halves, and both are necessary:

* **positive control** -- mutating the holdout must visibly change the
  holdout's own identity, or the negative half proves nothing;
* **the firewall** -- every supervised quantity, from the out-of-fold fold
  boundaries through to the published evaluation receipt, must be unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from password_attack_detector.ml.enums import MLSplit
from tests.integration.ml_workspace import (
    build_workspace,
    detect,
    evaluate,
    freeze,
    predict,
    prediction_ids,
)

#: The column mutated on the feature side. Numeric, present for every row, and
#: read by the preprocessor -- so if a holdout row ever entered a fit
#: population, moving it would move a fitted quantity.
MUTATED_FEATURE = "user_attempt_count__5m"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Build a holdout-bearing workspace and freeze the supervised state on it."""
    workspace = build_workspace(
        tmp_path_factory.mktemp("holdout-workspace"), with_holdout=True
    )
    root = tmp_path_factory.mktemp("holdout-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("holdout-reports"))
    for split in ("test", "validation"):
        scored = predict(workspace, root, split=split)
        assert scored.exit_code == 0, scored.output
    detection = tmp_path_factory.mktemp("holdout-detection")
    assessed = detect(workspace, detection)
    assert assessed.exit_code == 0, assessed.output
    return (workspace, root, detection)


@pytest.fixture
def prepared(
    pipeline: tuple[Path, Path, Path], tmp_path: Path
) -> tuple[Path, Path, Path]:
    """Return a writable copy of the whole pipeline, one per test."""
    workspace, root, detection = pipeline
    copied_workspace = tmp_path / "workspace"
    copied_root = tmp_path / "artifacts"
    shutil.copytree(workspace, copied_workspace)
    shutil.copytree(root, copied_root)
    return (copied_workspace, copied_root, detection)


# ---------------------------------------------------------------------------
# Reading the state under test
# ---------------------------------------------------------------------------


def _loaded(prepared: tuple[Path, Path, Path]) -> dict[str, Any]:
    """Return the assembled dataset and everything the fusion stage needs."""
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import load_feature_config
    from password_attack_detector.ml.config import load_ml_config
    from password_attack_detector.ml.dataset import load_ml_dataset
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )
    from tests.integration.ml_workspace import ML_CONFIG

    workspace, _root, _detection = prepared
    config = load_ml_config(Path(ML_CONFIG))
    catalog = build_catalog(load_feature_config(workspace / "features.yaml"))
    allowlist = load_feature_allowlist(workspace / "allowlist.yaml")
    eligible = resolve_eligible_features(
        catalog,
        allowlist,
        include_leakage_classes=config.preprocessing.include_leakage_classes,
        include_feature_groups=config.preprocessing.include_feature_groups,
        feature_schema_version=config.required_feature_schema_version,
    )
    dataset = load_ml_dataset(
        features_path=workspace / "processed" / "feature_snapshots.parquet",
        labels_path=workspace / "processed" / "feature_labels.parquet",
        splits_path=workspace / "processed" / "feature_splits.parquet",
        eligible=eligible,
        campaign_labels_path=workspace / "labels.parquet",
        feature_catalog_fingerprint=catalog.fingerprint(),
    )
    return {
        "config": config,
        "catalog": catalog,
        "allowlist": allowlist,
        "eligible": eligible,
        "dataset": dataset,
    }


def _scoped_fingerprints(dataset: Any, split: MLSplit) -> dict[str, str]:
    """Return the label and population digests scoped to one split."""
    from password_attack_detector.ml.cli import _outcomes_for
    from password_attack_detector.ml.stacking import no_fusion_evidence_proof
    from password_attack_detector.ml.test_evaluation import (
        test_label_fingerprint,
        test_population_fingerprint,
    )

    outcomes = _outcomes_for(dataset, split, no_fusion_evidence_proof())
    return {
        "labels": test_label_fingerprint(outcomes),
        "population": test_population_fingerprint(outcomes),
        "row_count": str(len(outcomes)),
        "malicious_count": str(sum(1 for item in outcomes if item.malicious)),
    }


def _supervised_state(prepared: tuple[Path, Path, Path]) -> dict[str, Any]:
    """Return every supervised quantity a holdout row must not be able to move."""
    from password_attack_detector.detection.serialization import RISK_FILE
    from password_attack_detector.ml.cli import (
        _fusion_selection,
        _rule_configuration_fingerprint,
        _rule_evidence,
    )
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.prediction_manifest import (
        PREDICTION_MANIFEST_FILE,
        PREDICTIONS_DIR,
        PredictionManifest,
    )
    from password_attack_detector.ml.predictions import FrozenChampion
    from tests.integration.ml_workspace import write_rule_config

    _workspace, root, detection = prepared
    loaded = _loaded(prepared)
    config = loaded["config"]
    ledger = ExperimentLedger(root / "ledger")
    champion = FrozenChampion.load(root, ledger=ledger, scope_key=None, config=config)
    write_rule_config(detection / "rules.yaml")
    published = prediction_ids(root)

    preparation = _fusion_selection(
        root=root,
        dataset=loaded["dataset"],
        catalog=loaded["catalog"],
        eligible=loaded["eligible"],
        allowlist=loaded["allowlist"],
        champion=champion,
        validation_directory=root / PREDICTIONS_DIR / published["validation"],
        validation_risk_path=None,
        rule=_rule_evidence(detection / RISK_FILE, "risk assessments"),
        rule_configuration_fingerprint=_rule_configuration_fingerprint(
            detection / "rules.yaml"
        ),
        config=config,
    )
    selection = preparation.selection
    assert selection is not None
    manifest = PredictionManifest.from_json(
        (
            root / PREDICTIONS_DIR / published["test"] / PREDICTION_MANIFEST_FILE
        ).read_text(encoding="utf-8")
    )
    return {
        # -- out-of-fold lineage ------------------------------------------
        "oof_fold_definition": preparation.out_of_fold.fold_definition_fingerprint,
        "oof_fold_count": preparation.out_of_fold.fold_count,
        "oof_anchors": preparation.out_of_fold.anchor_event_ids,
        "oof_scores": preparation.out_of_fold.scores,
        "oof_malicious": preparation.out_of_fold.malicious,
        "oof_fold_assignments": preparation.out_of_fold.fold_assignments,
        "oof_evidence": selection.oof_evidence_fingerprint,
        # -- the stacker ---------------------------------------------------
        "stacked_state": (
            None
            if preparation.stacked_state is None
            else preparation.stacked_state.state_fingerprint
        ),
        "stacked_unavailable": preparation.construction.unavailable_reason,
        "base_recipe": selection.base_model_recipe_fingerprint,
        # -- all three candidates on validation-B --------------------------
        "candidates": {
            str(item.strategy): {
                "eligible": item.eligible,
                "blocking_reasons": tuple(item.blocking_reasons),
                "metrics": (
                    None if item.metrics is None else item.metrics.model_dump_json()
                ),
            }
            for item in selection.candidates
        },
        "validation_evidence": selection.validation_evidence_fingerprint,
        "validation_rows": selection.validation_row_count,
        # -- the frozen selection ------------------------------------------
        "selection_fingerprint": selection.selection_fingerprint,
        "selected_strategy": str(selection.selected_strategy),
        "selection_status": str(selection.status),
        # -- the frozen champion and the TEST prediction --------------------
        "champion_lock": champion.lock.lock_fingerprint,
        "champion_freeze_record": champion.freeze_record_id,
        "prediction_id": manifest.prediction_id,
        "prediction_fingerprint": manifest.prediction_manifest_fingerprint,
        "prediction_content": manifest.prediction_content_fingerprint,
        # -- the supervised TEST ground truth itself ------------------------
        "test_scoped": _scoped_fingerprints(loaded["dataset"], MLSplit.TEST),
    }


def _evaluate(prepared: tuple[Path, Path, Path]) -> tuple[Any, bytes]:
    """Publish an evaluation and return its record with the receipt bytes."""
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.test_evaluation import (
        EVALUATION_RECEIPT_FILE,
        EVALUATIONS_DIR,
    )

    workspace, root, detection = prepared
    published = prediction_ids(root)
    result = evaluate(
        workspace,
        root,
        detection,
        root / "reports",
        **{
            "--prediction": published["test"],
            "--validation-prediction": published["validation"],
        },
    )
    assert result.exit_code == 0, result.output
    record = ExperimentLedger(root / "ledger").test_evaluations()[0]
    directory = next(iter(sorted((root / EVALUATIONS_DIR).iterdir())))
    return (record, (directory / EVALUATION_RECEIPT_FILE).read_bytes())


# ---------------------------------------------------------------------------
# The mutation
# ---------------------------------------------------------------------------


def _mutate_holdout(workspace: Path) -> int:
    """Rewrite every novel-anomaly holdout row, on both the label and feature side.

    Deliberately more than a flipped boolean. The outcome, the attack class, the
    supervised-eligibility flag, and a numeric feature the preprocessor reads
    are all moved, so a holdout row that leaked into *any* fitted or fingerprinted
    supervised quantity would move it.
    """
    processed = workspace / "processed"
    splits = pq.read_table(processed / "feature_splits.parquet").to_pylist()
    holdout = {
        str(row["event_id"])
        for row in splits
        if str(row["split"]) == str(MLSplit.NOVEL_ANOMALY_HOLDOUT)
    }
    assert holdout, "this fixture must carry a real holdout population"

    label_table = pq.read_table(processed / "feature_labels.parquet")
    label_rows = label_table.to_pylist()
    for row in label_rows:
        if str(row["event_id"]) in holdout:
            row["malicious"] = not row["malicious"]
            row["attack_class"] = "novel_anomaly_holdout"
            row["supervised_training_eligible"] = False
    pq.write_table(
        pa.Table.from_pylist(label_rows, schema=label_table.schema),
        processed / "feature_labels.parquet",
    )

    feature_table = pq.read_table(processed / "feature_snapshots.parquet")
    if MUTATED_FEATURE in feature_table.column_names:
        feature_rows = feature_table.to_pylist()
        for row in feature_rows:
            if str(row["anchor_event_id"]) in holdout:
                current = row.get(MUTATED_FEATURE)
                row[MUTATED_FEATURE] = 0 if current is None else current + 999
        pq.write_table(
            pa.Table.from_pylist(feature_rows, schema=feature_table.schema),
            processed / "feature_snapshots.parquet",
        )
    return len(holdout)


# ---------------------------------------------------------------------------
# The fixture is real
# ---------------------------------------------------------------------------


def test_the_fixture_carries_a_genuine_holdout_population(
    prepared: tuple[Path, Path, Path],
) -> None:
    """Everything below is vacuous without this."""
    dataset = _loaded(prepared)["dataset"]
    holdout = dataset.for_split(MLSplit.NOVEL_ANOMALY_HOLDOUT)
    assert holdout.row_count > 0
    # And it is genuinely excluded from supervised fitting, by the flag that
    # routes it there rather than by a filter this test applied.
    assert not any(holdout.supervised_training_eligible)
    for split in (MLSplit.TRAIN, MLSplit.VALIDATION, MLSplit.TEST):
        assert dataset.for_split(split).row_count > 0, split


def test_the_holdout_is_processed_rather_than_dropped_by_the_loader(
    prepared: tuple[Path, Path, Path],
) -> None:
    """The rows exist in the assembled dataset, on their own experimental path."""
    dataset = _loaded(prepared)["dataset"]
    holdout = dataset.for_split(MLSplit.NOVEL_ANOMALY_HOLDOUT)
    supervised: set[str] = set()
    for split in (MLSplit.TRAIN, MLSplit.VALIDATION, MLSplit.TEST):
        supervised |= {
            anchor.anchor_event_id for anchor in dataset.for_split(split).anchors
        }
    holdout_anchors = {anchor.anchor_event_id for anchor in holdout.anchors}
    assert holdout_anchors
    assert not (holdout_anchors & supervised)


# ---------------------------------------------------------------------------
# The positive control
# ---------------------------------------------------------------------------


def test_mutating_the_holdout_changes_the_holdout_itself(
    prepared: tuple[Path, Path, Path],
) -> None:
    """Without this, every negative assertion below could pass on a no-op."""
    before = _scoped_fingerprints(
        _loaded(prepared)["dataset"], MLSplit.NOVEL_ANOMALY_HOLDOUT
    )
    assert _mutate_holdout(prepared[0]) > 0
    after = _scoped_fingerprints(
        _loaded(prepared)["dataset"], MLSplit.NOVEL_ANOMALY_HOLDOUT
    )
    assert after["labels"] != before["labels"]
    assert after["malicious_count"] != before["malicious_count"]
    # The population digest covers *which rows*, not what they were labelled,
    # so it is expected to hold: the same anchors were relabelled, not
    # replaced. That is the distinction the two digests exist to draw.
    assert after["population"] == before["population"]
    assert after["row_count"] == before["row_count"]


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_mutating_the_holdout_moves_no_supervised_state(
    prepared: tuple[Path, Path, Path],
) -> None:
    """The whole claim, in one comparison.

    Out-of-fold folds and evidence, the stacker, all three candidates'
    validation-B verdicts, the frozen selection, the champion, the TEST
    prediction identity, and the scoped TEST ground truth -- none of it moves.
    """
    before = _supervised_state(prepared)
    assert _mutate_holdout(prepared[0]) > 0
    after = _supervised_state(prepared)
    assert after == before


@pytest.mark.parametrize(
    "quantity",
    [
        "oof_fold_definition",
        "oof_anchors",
        "oof_fold_assignments",
        "oof_evidence",
        "stacked_state",
        "candidates",
        "validation_evidence",
        "selection_fingerprint",
        "selected_strategy",
        "champion_lock",
        "prediction_id",
        "prediction_content",
        "test_scoped",
    ],
)
def test_each_supervised_quantity_individually_survives_the_mutation(
    prepared: tuple[Path, Path, Path], quantity: str
) -> None:
    """Named one at a time, so a failure says which invariant broke."""
    before = _supervised_state(prepared)[quantity]
    _mutate_holdout(prepared[0])
    after = _supervised_state(prepared)[quantity]
    assert after == before


def test_the_published_evaluation_is_unchanged_by_the_mutation(
    prepared: tuple[Path, Path, Path],
) -> None:
    """The receipt is byte-identical, and re-publishing writes nothing new.

    The strongest statement available: not merely that the numbers agree, but
    that the evaluation is recognised as *the same evaluation* and the immutable
    record is not appended to.
    """
    from password_attack_detector.ml.ledger import ExperimentLedger

    record, receipt = _evaluate(prepared)
    assert _mutate_holdout(prepared[0]) > 0
    again, receipt_again = _evaluate(prepared)

    assert receipt_again == receipt
    assert again.record_fingerprint == record.record_fingerprint
    assert again.identity == record.identity
    assert again.test_label_fingerprint == record.test_label_fingerprint
    assert (
        again.evaluation_population_fingerprint
        == record.evaluation_population_fingerprint
    )
    assert again.fusion_selection_fingerprint == record.fusion_selection_fingerprint
    assert again.selected_fusion_strategy == record.selected_fusion_strategy
    assert len(ExperimentLedger(prepared[1] / "ledger").test_evaluations()) == 1


def test_the_supervised_reports_are_unchanged_by_the_mutation(
    prepared: tuple[Path, Path, Path],
) -> None:
    """The rendered comparison a reader actually sees does not move either."""
    from password_attack_detector.ml.test_evaluation import (
        EVALUATIONS_DIR,
        ML_EVALUATION_JSON,
        SYSTEM_COMPARISON_JSON,
    )

    _evaluate(prepared)
    directory = next(iter(sorted((prepared[1] / EVALUATIONS_DIR).iterdir())))
    before = {
        name: json.loads((directory / name).read_text(encoding="utf-8"))
        for name in (ML_EVALUATION_JSON, SYSTEM_COMPARISON_JSON)
    }
    _mutate_holdout(prepared[0])
    _evaluate(prepared)
    after = {
        name: json.loads((directory / name).read_text(encoding="utf-8"))
        for name in (ML_EVALUATION_JSON, SYSTEM_COMPARISON_JSON)
    }
    assert after == before
