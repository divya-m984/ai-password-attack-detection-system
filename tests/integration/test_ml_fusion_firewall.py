"""The behavioural firewall around the pre-TEST fusion stage.

Milestone 9's whole ordering rests on one claim: the hybrid strategy is chosen
before any TEST label is readable, so the test set cannot influence which system
gets reported. This module tries to break that claim empirically rather than
trusting the signatures.

The method is perturbation. Hold the pipeline fixed, rewrite one split's ground
truth, re-run the fusion stage, and compare fingerprints:

* rewriting **TEST** must change nothing about fusion -- not the folds, not the
  out-of-fold evidence, not the stacker, not a candidate's verdict, not the
  selection identity;
* rewriting the **novel-anomaly holdout** must change nothing either;
* rewriting **TRAIN** may change the out-of-fold evidence and the stacker, since
  that is the population they are fitted on;
* rewriting **validation** may change the *selection*, but must not refit the
  stacker -- the stacker is fitted on TRAIN and merely judged on validation-B.

A test that only asserted the last three would pass on an implementation with no
firewall at all, which is why the first two are the ones that matter.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from password_attack_detector.ml.enums import FusionStrategy
from tests.integration.ml_workspace import (
    build_workspace,
    detect,
    freeze,
    predict,
    prediction_ids,
)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Freeze a champion, score TEST and validation, and assess with the rules."""
    workspace = build_workspace(tmp_path_factory.mktemp("firewall-workspace"))
    root = tmp_path_factory.mktemp("firewall-artifacts")
    freeze(workspace, root, tmp_path_factory.mktemp("firewall-reports"))
    for split in ("test", "validation"):
        scored = predict(workspace, root, split=split)
        assert scored.exit_code == 0, scored.output
    detection = tmp_path_factory.mktemp("firewall-detection")
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
# Driving the fusion stage directly
# ---------------------------------------------------------------------------


def _fusion(prepared: tuple[Path, Path, Path]) -> Any:
    """Run exactly the orchestration ``ml evaluate`` runs before TEST.

    Deliberately the real function rather than a reimplementation: a firewall
    test that exercised its own copy of the wiring would prove nothing about
    the command.
    """
    from password_attack_detector.detection.serialization import RISK_FILE
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import load_feature_config
    from password_attack_detector.ml.cli import (
        _fusion_selection,
        _rule_configuration_fingerprint,
        _rule_evidence,
    )
    from password_attack_detector.ml.config import load_ml_config
    from password_attack_detector.ml.dataset import load_ml_dataset
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.predictions import FrozenChampion
    from tests.integration.ml_workspace import ML_CONFIG, write_rule_config

    workspace, root, detection = prepared
    config = load_ml_config(Path(ML_CONFIG))
    feature_config = load_feature_config(workspace / "features.yaml")
    catalog = build_catalog(feature_config)
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
    ledger = ExperimentLedger(root / "ledger")
    champion = FrozenChampion.load(root, ledger=ledger, scope_key=None, config=config)
    rule = _rule_evidence(detection / RISK_FILE, "risk assessments")
    write_rule_config(detection / "rules.yaml")
    from password_attack_detector.ml.prediction_manifest import PREDICTIONS_DIR

    published = prediction_ids(root)
    return _fusion_selection(
        root=root,
        dataset=dataset,
        catalog=catalog,
        eligible=eligible,
        allowlist=allowlist,
        champion=champion,
        validation_directory=root / PREDICTIONS_DIR / published["validation"],
        validation_risk_path=None,
        rule=rule,
        rule_configuration_fingerprint=_rule_configuration_fingerprint(
            detection / "rules.yaml"
        ),
        config=config,
    )


def _state(preparation: Any) -> dict[str, Any]:
    """Return every quantity a TEST label must not be able to move."""
    selection = preparation.selection
    return {
        "fold_definition": preparation.out_of_fold.fold_definition_fingerprint,
        "fold_count": preparation.out_of_fold.fold_count,
        "oof_anchors": preparation.out_of_fold.anchor_event_ids,
        "oof_scores": preparation.out_of_fold.scores,
        "oof_folds": preparation.out_of_fold.fold_assignments,
        "stacked": (
            None
            if preparation.stacked_state is None
            else preparation.stacked_state.state_fingerprint
        ),
        "stacked_unavailable": preparation.construction.unavailable_reason,
        "candidates": {
            str(item.strategy): (item.eligible, tuple(item.blocking_reasons))
            for item in selection.candidates
        },
        "selected": str(selection.selected_strategy),
        "status": str(selection.status),
        "selection_fingerprint": selection.selection_fingerprint,
        "validation_evidence": selection.validation_evidence_fingerprint,
        "oof_evidence": selection.oof_evidence_fingerprint,
        "base_recipe": selection.base_model_recipe_fingerprint,
    }


def _rewrite_labels(workspace: Path, *, splits: set[str]) -> int:
    """Invert every label belonging to *splits*, in place.

    Radical rather than subtle on purpose: if any fusion quantity depends on
    these rows at all, inverting them will move it.
    """
    processed = workspace / "processed"
    split_rows = pq.read_table(processed / "feature_splits.parquet").to_pylist()
    assignment = {str(row["event_id"]): str(row["split"]) for row in split_rows}

    table = pq.read_table(processed / "feature_labels.parquet")
    rows = table.to_pylist()
    touched = 0
    for row in rows:
        if assignment.get(str(row["event_id"])) in splits:
            row["malicious"] = not row["malicious"]
            row["attack_class"] = (
                "credential_stuffing" if row["malicious"] else "normal"
            )
            touched += 1
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        processed / "feature_labels.parquet",
    )
    return touched


def _require_rows(touched: int, splits: set[str]) -> None:
    """Skip rather than pass vacuously when a split carries no rows.

    A perturbation test over an empty split asserts nothing, and a green tick
    for it would be worse than an honest skip.
    """
    if not touched:
        pytest.skip(f"this fixture has no {'/'.join(sorted(splits))} rows to perturb")


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


def test_the_stacker_is_genuinely_built_in_this_fixture(
    prepared: tuple[Path, Path, Path],
) -> None:
    """Everything below is vacuous if STACKED never gets constructed."""
    preparation = _fusion(prepared)
    assert preparation.stacked_available, preparation.construction.unavailable_reason
    assert preparation.out_of_fold.available
    assert set(preparation.proof.declared_candidates) == set(FusionStrategy)


def test_rewriting_every_test_label_changes_no_fusion_quantity(
    prepared: tuple[Path, Path, Path],
) -> None:
    """The firewall, stated as an experiment rather than as a signature."""
    before = _state(_fusion(prepared))
    _require_rows(_rewrite_labels(prepared[0], splits={"test"}), {"test"})
    after = _state(_fusion(prepared))
    assert after == before


def test_rewriting_the_novel_holdout_changes_no_fusion_quantity(
    prepared: tuple[Path, Path, Path],
) -> None:
    """The holdout is experimental and reaches no supervised decision."""
    before = _state(_fusion(prepared))
    _require_rows(
        _rewrite_labels(prepared[0], splits={"novel_anomaly_holdout"}),
        {"novel_anomaly_holdout"},
    )
    after = _state(_fusion(prepared))
    assert after == before


def test_rewriting_test_and_holdout_together_changes_nothing(
    prepared: tuple[Path, Path, Path],
) -> None:
    """Neither alone nor together."""
    before = _state(_fusion(prepared))
    _require_rows(
        _rewrite_labels(prepared[0], splits={"test", "novel_anomaly_holdout"}),
        {"test", "novel_anomaly_holdout"},
    )
    after = _state(_fusion(prepared))
    assert after == before


def test_rewriting_train_labels_moves_the_out_of_fold_evidence(
    prepared: tuple[Path, Path, Path],
) -> None:
    """The positive control.

    Without this the three tests above would also pass on an implementation
    that ignored labels entirely.
    """
    before = _state(_fusion(prepared))
    _require_rows(_rewrite_labels(prepared[0], splits={"train"}), {"train"})
    after = _state(_fusion(prepared))
    assert after["oof_evidence"] != before["oof_evidence"]
    assert after["selection_fingerprint"] != before["selection_fingerprint"]


def test_rewriting_validation_labels_does_not_refit_the_stacker(
    prepared: tuple[Path, Path, Path],
) -> None:
    """Validation judges the stacker. It never fits it.

    The selection may legitimately change -- that is what validation-B is for --
    but the fitted state must not, because no validation row was in its fit
    population.
    """
    before = _state(_fusion(prepared))
    _require_rows(_rewrite_labels(prepared[0], splits={"validation"}), {"validation"})
    after = _state(_fusion(prepared))
    assert after["stacked"] == before["stacked"]
    assert after["oof_evidence"] == before["oof_evidence"]
    assert after["fold_definition"] == before["fold_definition"]
    assert after["validation_evidence"] != before["validation_evidence"]


# ---------------------------------------------------------------------------
# The reader gate
# ---------------------------------------------------------------------------


def test_the_test_reader_refuses_without_a_freeze_proof() -> None:
    """The ordering is enforced by the types, not by the order of the lines."""
    from password_attack_detector.exceptions import ModelNotReadyError
    from password_attack_detector.ml.cli import _outcomes_for
    from password_attack_detector.ml.enums import MLSplit

    for forgery in (None, object(), "frozen", {"outcome": "selected"}):
        with pytest.raises(ModelNotReadyError, match="frozen fusion selection"):
            _outcomes_for(object(), MLSplit.TEST, forgery)


def test_a_freeze_proof_cannot_be_constructed_by_hand() -> None:
    """A caller cannot manufacture the token that unlocks the reader."""
    from password_attack_detector.exceptions import ModelNotReadyError
    from password_attack_detector.ml.stacking import FusionFreezeProof

    with pytest.raises(ModelNotReadyError, match="prepare_fusion_selection"):
        FusionFreezeProof(
            token=object(),
            outcome="selected",
            declared_candidates=tuple(FusionStrategy),
            candidate_status={},
            fusion_selection_fingerprint=None,
        )


def test_a_proof_must_name_the_whole_candidate_universe(
    prepared: tuple[Path, Path, Path],
) -> None:
    """A strategy dropped from the record is one nobody can see was skipped."""
    from password_attack_detector.exceptions import ModelNotReadyError
    from password_attack_detector.ml import stacking

    with pytest.raises(ModelNotReadyError, match="whole declared candidate"):
        stacking.FusionFreezeProof(
            token=stacking._FREEZE_TOKEN,
            outcome="selected",
            declared_candidates=(FusionStrategy.OR_GATE,),
            candidate_status={},
            fusion_selection_fingerprint=None,
        )


def test_the_test_reader_is_never_reached_when_the_fusion_stage_fails(
    prepared: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spy on the reader, and a fusion stage arranged to fail before it.

    Asserts the property the ordering exists for: an evaluation that cannot
    freeze a fusion selection stops, and stops *before* a TEST label is turned
    into a value anything downstream could see.
    """
    from password_attack_detector.exceptions import ModelTrainingError
    from password_attack_detector.ml import cli, stacking
    from tests.integration.ml_workspace import ML_CONFIG, evaluate

    opened: list[str] = []
    original = cli._outcomes_for

    def spy(dataset: Any, split: Any, proof: Any) -> Any:
        opened.append(str(split))
        return original(dataset, split, proof)

    def explode(**_: Any) -> Any:
        raise ModelTrainingError("fold construction refused for this population")

    monkeypatch.setattr(cli, "_outcomes_for", spy)
    monkeypatch.setattr(stacking, "prepare_fusion_selection", explode)
    monkeypatch.setattr(cli, "prepare_fusion_selection", explode, raising=False)

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
            "--config": ML_CONFIG,
        },
    )
    assert result.exit_code != 0
    assert not opened, f"a TEST label was read after the fusion stage failed: {opened}"


# ---------------------------------------------------------------------------
# When STACKED is legitimately unavailable
# ---------------------------------------------------------------------------


def _config_with(tmp_path: Path, **fusion: object) -> Path:
    """Return the testing ML configuration with *fusion* keys overridden."""
    import yaml

    from tests.integration.ml_workspace import ML_CONFIG

    loaded = yaml.safe_load(Path(ML_CONFIG).read_text(encoding="utf-8"))
    loaded.setdefault("fusion", {}).update(fusion)
    target = tmp_path / "ml-fusion.yaml"
    target.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    return target


def test_more_folds_than_campaigns_types_stacked_unavailable(
    prepared: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A real data reason, carried as a reason rather than as a silent absence.

    Folds are cut at campaign boundaries, so a fold count above the campaign
    count cannot be satisfied without splitting one. The stacker is then not
    available -- and that is the honest answer, not a bug.
    """
    from password_attack_detector.ml.enums import FusionStrategy as Strategy
    from tests.integration.ml_workspace import evaluate

    config = _config_with(tmp_path, stacked_fold_count=20)
    workspace, root, detection = prepared
    published = prediction_ids(root)
    result = evaluate(
        workspace,
        root,
        detection,
        tmp_path / "reports",
        **{
            "--prediction": published["test"],
            "--validation-prediction": published["validation"],
            "--config": str(config),
        },
    )
    assert result.exit_code == 0, result.output
    # The candidate is still declared, still shown, and carries its reason.
    assert str(Strategy.STACKED) in result.stdout
    assert str(Strategy.OR_GATE) in result.stdout
    assert str(Strategy.AND_GATE) in result.stdout


def test_an_unavailable_stacker_does_not_default_the_selection(
    prepared: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """The gates still compete on their merits; OR_GATE is not assumed."""
    from password_attack_detector.ml.config import load_ml_config
    from password_attack_detector.ml.enums import FusionStrategy as Strategy

    config = load_ml_config(_config_with(tmp_path, stacked_fold_count=20))
    preparation = _fusion_with_config(prepared, config)
    assert not preparation.stacked_available
    reason = preparation.construction.unavailable_reason
    assert reason, "an unavailable stacker must carry a reason"
    assert "not wired" not in reason
    assert "refits nothing" not in reason
    assert set(preparation.proof.declared_candidates) == set(Strategy)
    # A selection was still reached from the remaining candidates.
    assert preparation.selection is not None


def _fusion_with_config(prepared: tuple[Path, Path, Path], config: Any) -> Any:
    """Run the fusion stage under an explicit configuration."""
    from password_attack_detector.detection.serialization import RISK_FILE
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import load_feature_config
    from password_attack_detector.ml.cli import (
        _fusion_selection,
        _rule_configuration_fingerprint,
        _rule_evidence,
    )
    from password_attack_detector.ml.dataset import load_ml_dataset
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.prediction_manifest import PREDICTIONS_DIR
    from password_attack_detector.ml.predictions import FrozenChampion
    from tests.integration.ml_workspace import write_rule_config

    workspace, root, detection = prepared
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
    ledger = ExperimentLedger(root / "ledger")
    champion = FrozenChampion.load(root, ledger=ledger, scope_key=None, config=config)
    write_rule_config(detection / "rules.yaml")
    published = prediction_ids(root)
    return _fusion_selection(
        root=root,
        dataset=dataset,
        catalog=catalog,
        eligible=eligible,
        allowlist=allowlist,
        champion=champion,
        validation_directory=root / PREDICTIONS_DIR / published["validation"],
        validation_risk_path=None,
        rule=_rule_evidence(detection / RISK_FILE, "risk assessments"),
        rule_configuration_fingerprint=_rule_configuration_fingerprint(
            detection / "rules.yaml"
        ),
        config=config,
    )
