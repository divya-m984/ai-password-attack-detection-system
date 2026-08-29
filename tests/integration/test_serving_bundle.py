"""Materializing the frozen STACKED hybrid, and serving it.

The fixture runs the real pipeline once -- publish a feature dataset, train,
select, freeze, predict, detect, evaluate -- and that pipeline's own selection on
this workspace is ``stacked``.  Nothing here is hand-assembled: a test that
proved a reconstruction against a hand-built selection would be proving that two
pieces of test code agree.

What the assertions are actually about:

* the strategy Phase 5 selected can be materialized and then served end to end;
* the reconstructed stacker recomputes the fingerprint Phase 5 sealed;
* one changed semantic upstream input causes a refusal rather than a new state;
* mutating the TEST labels cannot change the materialized state;
* API startup loads and verifies, and never fits;
* a missing or unverifiable stacked artifact fails closed;
* no gate is ever substituted for the selected stacker;
* a live request is scored without being labelled with any split, and gets the
  same numbers the published-split path would give.
"""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from fastapi.testclient import TestClient

from password_attack_detector.api.app import create_app
from password_attack_detector.api.config import APISettings
from password_attack_detector.deployment.bundle import (
    BUNDLE_MANIFEST_FILE,
    STACKED_STATE_FILE,
    bundle_directory,
    load_serving_bundle,
)
from password_attack_detector.ml.enums import FusionStrategy
from tests.api.factories import brute_force_window
from tests.integration.ml_workspace import (
    ML_CONFIG,
    build_workspace,
    detect,
    evaluate,
    freeze,
    materialize,
    predict,
    prediction_ids,
    write_rule_config,
)


@dataclass(frozen=True, slots=True)
class Deployment:
    """Everything the pipeline produced, so a test can re-run one stage of it."""

    base: Path
    workspace: Path
    root: Path
    detection: Path
    reports: Path
    validation_prediction: str

    @property
    def settings(self) -> APISettings:
        """Return serving settings pointing at this deployment."""
        return APISettings(
            artifact_root=self.root,
            allowlist_path=self.workspace / "allowlist.yaml",
            feature_config_path=self.workspace / "features.yaml",
            ml_config_path=Path(ML_CONFIG),
            detection_config_path=self.detection / "rules.yaml",
        )

    @property
    def frozen_selection(self) -> dict[str, Any]:
        """Return the frozen ``FusionSelection`` the locked evaluation published."""
        report = next((self.root / "evaluations").glob("*/system_comparison.json"))
        payload: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
        selection: dict[str, Any] = payload["fusion"]
        return selection

    @property
    def scope_key(self) -> str:
        """Return the frozen champion's scope key."""
        lock = next((self.root / "champion").glob("*/champion.lock"))
        return str(lock.parent.name)


@pytest.fixture(scope="module")
def deployed(tmp_path_factory: pytest.TempPathFactory) -> Deployment:
    """Run the whole Phase 5 pipeline once and materialize its hybrid."""
    base = tmp_path_factory.mktemp("serving-bundle")
    (base / "workspace").mkdir(parents=True, exist_ok=True)
    workspace = build_workspace(base / "workspace")
    root = base / "artifacts"
    reports = base / "reports"
    freeze(workspace, root, reports)

    for split in ("train", "validation", "test"):
        scored = predict(workspace, root, split=split)
        assert scored.exit_code == 0, scored.output

    detection = base / "detection"
    assessed = detect(workspace, detection)
    assert assessed.exit_code == 0, assessed.output

    ids = prediction_ids(root)
    evaluated = evaluate(
        workspace,
        root,
        detection,
        reports,
        **{"--prediction": ids["test"], "--validation-prediction": ids["validation"]},
    )
    assert evaluated.exit_code == 0, evaluated.output

    published = materialize(
        workspace, root, detection, validation_prediction=ids["validation"]
    )
    assert published.exit_code == 0, published.output

    return Deployment(
        base=base,
        workspace=workspace,
        root=root,
        detection=detection,
        reports=reports,
        validation_prediction=ids["validation"],
    )


@pytest.fixture()
def client(deployed: Deployment) -> Any:
    """A client bound to the materialized stacked deployment."""
    with TestClient(create_app(settings=deployed.settings)) as connected:
        yield connected


def _fresh_root(deployed: Deployment, destination: Path) -> Path:
    """Return a private copy of the artifact root with no bundle published.

    Copied rather than mutated in place, and stripped of the bundle, so a
    re-materialization is a genuine first publication rather than an idempotent
    no-op that would hide a changed fingerprint.
    """
    root = destination / "artifacts"
    shutil.copytree(deployed.root, root)
    if (root / SERVING_BUNDLE_ROOT).exists():
        shutil.rmtree(root / SERVING_BUNDLE_ROOT)
    return root


SERVING_BUNDLE_ROOT = "serving"


# ---------------------------------------------------------------------------
# The frozen selection this whole suite rests on
# ---------------------------------------------------------------------------


def test_the_pipeline_selects_a_stacked_hybrid(deployed: Deployment) -> None:
    """Grounding: the interesting case is the one the real pipeline produced.

    If this ever stops holding, every assertion below is testing a strategy that
    needs no fitted artifact, and the suite should be read as vacuous rather than
    passing.
    """
    selection = deployed.frozen_selection
    assert selection["status"] == "selected"
    assert selection["selected_strategy"] == "stacked"
    assert len(selection["stacked_state_fingerprint"]) == 64


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def test_a_stacked_selection_is_materialized_and_verifies(
    deployed: Deployment,
) -> None:
    """The published bundle loads under the same verification serving performs."""
    bundle = load_serving_bundle(deployed.root, scope_key=deployed.scope_key)
    assert bundle.strategy is FusionStrategy.STACKED
    assert bundle.stacked_state is not None
    assert bundle.selection.selected_strategy is FusionStrategy.STACKED


def test_the_reconstructed_stacked_fingerprint_is_the_frozen_one(
    deployed: Deployment,
) -> None:
    """The load-bearing claim: this stacker is the stacker validation chose."""
    bundle = load_serving_bundle(deployed.root, scope_key=deployed.scope_key)
    frozen = deployed.frozen_selection
    assert bundle.stacked_state is not None
    assert bundle.stacked_state.state_fingerprint == frozen["stacked_state_fingerprint"]
    assert (
        bundle.manifest.fusion_selection_fingerprint == frozen["selection_fingerprint"]
    )
    # Recomputed from the state's own content, not merely read off it.
    assert (
        bundle.stacked_state.recomputed_fingerprint()
        == frozen["stacked_state_fingerprint"]
    )


def test_the_bundle_carries_the_whole_frozen_pipeline_lineage(
    deployed: Deployment,
) -> None:
    """Model, preprocessor, calibrator, threshold, and fusion, all named."""
    manifest = load_serving_bundle(deployed.root, scope_key=deployed.scope_key).manifest
    lock = json.loads(
        (deployed.root / "champion" / deployed.scope_key / "champion.lock").read_text(
            encoding="utf-8"
        )
    )
    assert manifest.champion_lock_fingerprint == lock["lock_fingerprint"]
    assert manifest.model_content_fingerprint == lock["model_content_fingerprint"]
    assert manifest.preprocessor_fingerprint == lock["preprocessor_fingerprint"]
    assert (
        manifest.calibration_state_fingerprint
        == (lock["calibration_state_fingerprint"])
    )
    assert manifest.binary_threshold_fingerprint == lock["binary_threshold_fingerprint"]
    assert (
        manifest.eligible_feature_list_fingerprint
        == (lock["eligible_feature_list_fingerprint"])
    )
    assert (
        manifest.dependency_contract_fingerprint
        == (lock["dependency_contract_fingerprint"])
    )
    for value in (
        manifest.fusion_config_fingerprint,
        manifest.rule_configuration_fingerprint,
        manifest.validation_evidence_fingerprint,
        manifest.oof_fold_definition_fingerprint,
        manifest.oof_evidence_fingerprint,
        manifest.base_model_recipe_fingerprint,
    ):
        assert value is not None and len(value) == 64


def test_the_bundle_carries_no_metric_and_no_threshold(deployed: Deployment) -> None:
    """A deployment artifact describes what to run, never how well it did."""
    manifest = load_serving_bundle(deployed.root, scope_key=deployed.scope_key).manifest
    body = manifest.to_dict()
    for forbidden in (
        "f1",
        "recall",
        "precision",
        "test_metrics",
        "decision_threshold",
        "fusion_threshold",
        "fallback_strategy",
    ):
        assert forbidden not in body


def test_materializing_the_same_lineage_twice_writes_nothing_new(
    deployed: Deployment,
) -> None:
    """Deterministic serialization makes re-publication a detectable no-op."""
    directory = bundle_directory(deployed.root, scope_key=deployed.scope_key)
    before = {item.name: item.read_bytes() for item in sorted(directory.iterdir())}
    again = materialize(
        deployed.workspace,
        deployed.root,
        deployed.detection,
        validation_prediction=deployed.validation_prediction,
    )
    assert again.exit_code == 0, again.output
    assert "Already published" in again.output
    after = {item.name: item.read_bytes() for item in sorted(directory.iterdir())}
    assert after == before


# ---------------------------------------------------------------------------
# Refusals: one changed semantic input, and nothing is written
# ---------------------------------------------------------------------------


def test_a_changed_rule_configuration_is_refused(
    deployed: Deployment, tmp_path: Path
) -> None:
    """The rule arm is part of the selection's identity, so changing it refuses."""
    root = _fresh_root(deployed, tmp_path)
    refused = materialize(
        deployed.workspace,
        root,
        deployed.detection,
        validation_prediction=deployed.validation_prediction,
        **{"--rule-config": str(write_rule_config(tmp_path / "rules.yaml"))},
    )
    assert refused.exit_code == 2, refused.output
    assert "rule configuration" in refused.output
    assert not (root / SERVING_BUNDLE_ROOT).exists()


def test_a_changed_fold_count_causes_a_fingerprint_mismatch(
    deployed: Deployment, tmp_path: Path
) -> None:
    """Cut the out-of-fold folds differently and the stacker is a different one.

    The single clearest demonstration of the property this contract exists for:
    the inputs still *look* right, the fit still succeeds, and the resulting state
    is refused because its digest is not the one Phase 5 sealed.
    """
    root = _fresh_root(deployed, tmp_path)
    config = yaml.safe_load(Path(ML_CONFIG).read_text(encoding="utf-8"))
    config["fusion"]["stacked_fold_count"] = 3
    altered = tmp_path / "ml-altered.yaml"
    altered.write_text(yaml.safe_dump(config), encoding="utf-8")

    refused = materialize(
        deployed.workspace,
        root,
        deployed.detection,
        validation_prediction=deployed.validation_prediction,
        **{"--config": str(altered)},
    )
    assert refused.exit_code == 2, refused.output
    assert "Fingerprints agree" in refused.output
    assert "NO" in refused.output
    assert not (root / SERVING_BUNDLE_ROOT).exists()


def test_a_report_edited_after_publication_is_refused(
    deployed: Deployment, tmp_path: Path
) -> None:
    """The chain from sealed receipt to fusion selection has no unverified link.

    The frozen selection is read out of the evaluation's own report, and the
    report is checked against the digest the *receipt* recorded. Editing the
    report -- even into something that parses -- makes it unreadable rather than
    authoritative.
    """
    root = _fresh_root(deployed, tmp_path)
    report = next((root / "evaluations").glob("*/system_comparison.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["fusion"]["selected_strategy"] = "or_gate"
    report.write_text(json.dumps(payload), encoding="utf-8")

    refused = materialize(
        deployed.workspace,
        root,
        deployed.detection,
        validation_prediction=deployed.validation_prediction,
    )
    assert refused.exit_code != 0, refused.output
    assert not (root / SERVING_BUNDLE_ROOT).exists()


def test_a_mutated_test_label_cannot_change_the_materialized_state(
    deployed: Deployment, tmp_path: Path
) -> None:
    """The reconstruction is a function of TRAIN and validation rows, and only those.

    Every TEST row's outcome is inverted in the label table, the bundle is
    materialized again from scratch, and the stacker comes out byte-identical --
    which is what "no TEST-derived fitting" means when it is checked rather than
    asserted.
    """
    root = _fresh_root(deployed, tmp_path)
    labels = _mutated_test_labels(deployed, tmp_path)

    again = materialize(
        deployed.workspace,
        root,
        deployed.detection,
        validation_prediction=deployed.validation_prediction,
        **{"--labels": str(labels)},
    )
    assert again.exit_code == 0, again.output

    original = load_serving_bundle(deployed.root, scope_key=deployed.scope_key)
    rebuilt = load_serving_bundle(root, scope_key=deployed.scope_key)
    assert original.stacked_state is not None
    assert rebuilt.stacked_state is not None
    assert rebuilt.stacked_state.to_json() == original.stacked_state.to_json()
    assert rebuilt.manifest.to_json() == original.manifest.to_json()


def _mutated_test_labels(deployed: Deployment, destination: Path) -> Path:
    """Write a label table whose every TEST outcome is inverted."""
    processed = deployed.workspace / "processed"
    splits = pq.read_table(processed / "feature_splits.parquet").to_pylist()
    test_events = {
        str(row["event_id"]) for row in splits if str(row["split"]) == "test"
    }
    assert test_events, "the fixture must publish a non-empty TEST split"

    table = pq.read_table(processed / "feature_labels.parquet")
    rows = table.to_pylist()
    flipped = 0
    for row in rows:
        if str(row["event_id"]) in test_events:
            row["malicious"] = not bool(row["malicious"])
            flipped += 1
    assert flipped == len(test_events)

    target = destination / "feature_labels_mutated.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target)
    return target


# ---------------------------------------------------------------------------
# Serving: the selected stacked path executes
# ---------------------------------------------------------------------------


def test_a_materialized_stacked_hybrid_is_ready_and_active(client: Any) -> None:
    """End to end: the frozen stacker is loaded, required, and running."""
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    status = client.get("/api/v1/system/status").json()
    assert status["hybrid_detection_enabled"] is True
    assert status["hybrid_required"] is True
    assert status["frozen_fusion_strategy"] == "stacked"
    assert status["fusion_strategy"] == "stacked"


def test_the_served_verdict_is_the_bundled_meta_learners_own(
    client: Any, deployed: Deployment
) -> None:
    """The stacked fusion path is executed, not merely reported as selected.

    The expected verdict is recomputed here from the bundled state applied to the
    response's own model probability and rule decision.  If the service were
    substituting a gate, or applying some other stacker, this would disagree.
    """
    state = load_serving_bundle(
        deployed.root, scope_key=deployed.scope_key
    ).stacked_state
    assert state is not None

    body = client.post(
        "/api/v1/detect/batch", json={"events": brute_force_window(14)}
    ).json()
    assert body["anchors"]
    for anchor in body["anchors"]:
        ml_score = (
            anchor["ml"]["probability"]
            if anchor["ml"]["probability"] is not None
            else anchor["ml"]["decision_score"]
        )
        expected = state.probability(
            ml_score=ml_score, rule_flagged=anchor["rule"]["flagged"]
        )
        assert anchor["hybrid"]["strategy"] == "stacked"
        assert anchor["hybrid"]["flagged"] is (expected >= state.decision_threshold)


def test_the_hybrid_verdict_is_not_a_blend_of_the_two_scales(client: Any) -> None:
    """A stacked verdict is a boolean; the meta-probability is not published."""
    anchor = client.post(
        "/api/v1/detect", json={"events": brute_force_window(8)}
    ).json()["anchor"]
    assert set(anchor["hybrid"]) == {
        "available",
        "unavailable_reason",
        "flagged",
        "strategy",
    }
    for forbidden in ("stacked_probability", "combined_score", "fused_score"):
        assert forbidden not in anchor["hybrid"]


# ---------------------------------------------------------------------------
# Startup loads and verifies.  It never fits.
# ---------------------------------------------------------------------------


def test_api_startup_performs_no_fit(
    deployed: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavioural proof: the fitting entry points are booby-trapped, and unused.

    Both places a stacker could be fitted are replaced with functions that fail
    the test if called.  Startup then resolves a *ready* stacked hybrid anyway,
    because it read a published artifact.
    """
    from password_attack_detector.ml import fusion as fusion_module
    from password_attack_detector.ml import stacking as stacking_module

    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("serving startup fitted a stacker")

    monkeypatch.setattr(fusion_module, "fit_stacked_fusion", _refuse)
    monkeypatch.setattr(stacking_module, "build_stacked_state", _refuse)
    monkeypatch.setattr(stacking_module, "prepare_fusion_selection", _refuse)

    with TestClient(create_app(settings=deployed.settings)) as client:
        body = client.get("/api/v1/system/status").json()
        assert body["fusion_strategy"] == "stacked"
        assert body["hybrid_detection_enabled"] is True


def test_the_serving_module_cannot_reach_a_fit_or_a_publisher() -> None:
    """Structural proof: no fitting name is in the serving namespace at all."""
    from password_attack_detector.api import services

    namespace = vars(services)
    for forbidden in (
        "fit_stacked_fusion",
        "prepare_fusion_selection",
        "reconstruct_stacked_state",
        "materialize_serving_bundle",
        "write_serving_bundle",
        "select_fusion_strategy",
        "freeze_champion",
    ):
        assert forbidden not in namespace

    tree = ast.parse(Path(services.__file__ or "").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "materialize_serving_bundle" not in imported
    assert "fit_stacked_fusion" not in imported


# ---------------------------------------------------------------------------
# Fail closed, and never fall back
# ---------------------------------------------------------------------------


def test_a_missing_stacked_artifact_fails_closed(
    deployed: Deployment, tmp_path: Path
) -> None:
    """The declared payload is gone: that is an unusable bundle, not a gate."""
    root = _fresh_root(deployed, tmp_path)
    shutil.copytree(
        bundle_directory(deployed.root, scope_key=deployed.scope_key),
        bundle_directory(root, scope_key=deployed.scope_key),
    )
    (bundle_directory(root, scope_key=deployed.scope_key) / STACKED_STATE_FILE).unlink()

    settings = deployed.settings.model_copy(update={"artifact_root": root})
    with TestClient(create_app(settings=settings)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        reasons = {
            item["component"]: item["reason"] for item in ready.json()["components"]
        }
        assert reasons["fusion"] == "serving_bundle_unverifiable"


def test_a_bundle_from_another_lineage_is_refused(
    deployed: Deployment, tmp_path: Path
) -> None:
    """A manifest naming a different champion is not this deployment's hybrid."""
    root = _fresh_root(deployed, tmp_path)
    source = bundle_directory(deployed.root, scope_key=deployed.scope_key)
    target = bundle_directory(root, scope_key=deployed.scope_key)
    shutil.copytree(source, target)
    manifest = json.loads((target / BUNDLE_MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["champion_lock_fingerprint"] = "0" * 64
    (target / BUNDLE_MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    settings = deployed.settings.model_copy(update={"artifact_root": root})
    with TestClient(create_app(settings=settings)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        reasons = {
            item["component"]: item["reason"] for item in ready.json()["components"]
        }
        # The manifest is sealed, so editing a lineage field breaks its own digest
        # before the lineage check is even reached. Either refusal is correct; what
        # must never happen is the bundle being used.
        assert reasons["fusion"] in {
            "serving_bundle_unverifiable",
            "serving_bundle_lineage_mismatch",
        }


def test_no_gate_is_ever_substituted_for_the_selected_stacker(
    deployed: Deployment, tmp_path: Path
) -> None:
    """With nothing to load, the hybrid is absent -- never downgraded to a gate."""
    root = _fresh_root(deployed, tmp_path)
    settings = deployed.settings.model_copy(update={"artifact_root": root})
    with TestClient(create_app(settings=settings)) as client:
        status = client.get("/api/v1/system/status").json()
        assert status["frozen_fusion_strategy"] == "stacked"
        assert status["fusion_strategy"] is None
        assert status["hybrid_detection_enabled"] is False
        assert status["hybrid_required"] is True
        assert status["fusion_unavailable_reason"] == "serving_bundle_not_published"
        assert status["stacked_state_fingerprint"] is None


def test_a_bundle_root_elsewhere_relocates_and_nothing_else(
    deployed: Deployment, tmp_path: Path
) -> None:
    """Artifact location may vary; scientific identity may not.

    The bundle is moved to an unrelated directory and pointed at by configuration.
    The same hybrid runs, because the setting says *where* and the frozen
    selection says *what*.
    """
    root = _fresh_root(deployed, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(
        bundle_directory(deployed.root, scope_key=deployed.scope_key),
        bundle_directory(elsewhere, scope_key=deployed.scope_key),
    )
    settings = deployed.settings.model_copy(
        update={"artifact_root": root, "serving_bundle_root": elsewhere}
    )
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/api/v1/system/status").json()["fusion_strategy"] == (
            "stacked"
        )


# ---------------------------------------------------------------------------
# The live serving scope
# ---------------------------------------------------------------------------


def test_a_live_request_is_scored_without_being_given_a_split(
    deployed: Deployment,
) -> None:
    """The serving batch names ``live_serving`` and carries no split at all."""
    import inspect

    from password_attack_detector.ml.dataset import (
        ServingFrame,
        assemble_serving_batch,
    )
    from password_attack_detector.ml.enums import MLSplit, ServingScope

    parameters = set(inspect.signature(assemble_serving_batch).parameters)
    assert "splits" not in parameters
    assert "scope" not in parameters
    assert "split" not in ServingFrame.__annotations__

    batch = _serving_batch(deployed)
    assert batch.scope is ServingScope.LIVE
    assert not isinstance(batch.scope, MLSplit)
    assert str(batch.scope) not in {str(item) for item in MLSplit}
    assert not hasattr(batch, "split")
    assert not hasattr(batch.frame, "split")


def test_serving_and_the_published_split_path_score_identically(
    deployed: Deployment,
) -> None:
    """Same feature state, same frozen champion, same numbers.

    The two paths share one implementation of the decision, and this is the
    assertion that keeps it that way: a serving-only threshold, calibrator, or
    score-kind choice would show up here as a difference.
    """
    from password_attack_detector.ml.dataset import SplitRow, assemble_inference_dataset
    from password_attack_detector.ml.enums import MLSplit
    from password_attack_detector.ml.predictions import (
        predict_binary,
        predict_serving_binary,
    )

    rows, champion, eligible = _scoring_inputs(deployed)
    live = predict_serving_binary(champion, _serving_batch(deployed))
    published = predict_binary(
        champion,
        assemble_inference_dataset(
            feature_rows=rows,
            splits=[
                SplitRow(event_id=str(row["anchor_event_id"]), split=str(MLSplit.TEST))
                for row in rows
            ],
            eligible=eligible,
            scope=MLSplit.TEST,
        ),
    )
    assert [row.model_dump() for row in live] == [row.model_dump() for row in published]


def _scoring_inputs(deployed: Deployment) -> tuple[list[dict[str, Any]], Any, Any]:
    """Return real feature rows, the frozen champion, and the feature contract."""
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import load_feature_config
    from password_attack_detector.ml.config import load_ml_config
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.predictions import FrozenChampion

    config = load_ml_config(Path(ML_CONFIG))
    catalog = build_catalog(load_feature_config(deployed.workspace / "features.yaml"))
    allowlist = load_feature_allowlist(deployed.workspace / "allowlist.yaml")
    eligible = resolve_eligible_features(
        catalog,
        allowlist,
        include_leakage_classes=config.preprocessing.include_leakage_classes,
        include_feature_groups=config.preprocessing.include_feature_groups,
        feature_schema_version=config.required_feature_schema_version,
    )
    champion = FrozenChampion.load(
        deployed.root,
        ledger=ExperimentLedger(deployed.root / "ledger"),
        scope_key=None,
        config=config,
    )
    # Read as plain mappings, exactly as the dataset loader does: a DataFrame
    # view would substitute pandas null semantics for the catalog's.
    table = pq.read_table(
        deployed.workspace / "processed" / "feature_snapshots.parquet"
    )
    rows = list(table.to_pylist())[:12]
    return (rows, champion, eligible)


def _serving_batch(deployed: Deployment) -> Any:
    """Return a serving batch over real feature rows."""
    from password_attack_detector.ml.dataset import assemble_serving_batch

    rows, _champion, eligible = _scoring_inputs(deployed)
    return assemble_serving_batch(feature_rows=rows, eligible=eligible)
