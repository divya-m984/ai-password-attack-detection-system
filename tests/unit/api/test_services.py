"""Tests for the serving orchestration.

Three groups:

* **Startup.**  ``build_runtime`` never raises, every failure names a stable
  reason code, and a missing required component leaves readiness false.
* **Documents.**  The operational payloads report what the runtime actually
  resolved, and say nothing about where it found it.
* **Detection.**  The refusals a request meets before any artifact is touched,
  and the fusion arm applied to one anchor's evidence.

The ML arm is exercised end to end against a real frozen champion in
``tests/integration/test_api_detection.py``; building one costs a full training
pipeline, and repeating it here would test the fixture rather than this module.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from password_attack_detector import __version__
from password_attack_detector.api.config import APISettings
from password_attack_detector.api.errors import APIError, ErrorCode
from password_attack_detector.api.schemas import (
    ComponentState,
    DetectionBatchRequest,
    DetectionWindowRequest,
    ReadinessState,
)
from password_attack_detector.api.services import (
    COMPONENT_FEATURE_CONTRACT,
    COMPONENT_FUSION,
    COMPONENT_ML_CHAMPION,
    COMPONENT_MODEL_ARTIFACTS,
    COMPONENT_REPLAY,
    COMPONENT_RULE_ENGINE,
    SERVING_SCOPE,
    FusionRuntime,
    RuntimeState,
    build_runtime,
    detect_batch,
    detect_single,
    hybrid_layer,
    model_info_document,
    readiness_document,
    rule_catalog_document,
    system_status_document,
    version_document,
)
from password_attack_detector.data.privacy import PseudonymService
from password_attack_detector.detection.enums import AttackCategory, Severity
from password_attack_detector.detection.schemas import RiskAssessment
from password_attack_detector.ml.enums import (
    FusionStrategy,
    MLSplit,
    ScoreKind,
    ServingScope,
)
from password_attack_detector.ml.fusion import (
    MetaRow,
    StackedFusionState,
    fit_stacked_fusion,
)
from password_attack_detector.ml.predictions import BinaryPrediction
from tests.api.factories import brute_force_window, normal_window


@pytest.fixture()
def rule_only() -> RuntimeState:
    """A runtime serving the rule layer alone: no artifacts, no model."""
    return build_runtime(APISettings(require_ml_champion=False))


def _states(runtime: RuntimeState) -> dict[str, ComponentState]:
    """Return each component's state, keyed by component name."""
    return {item.component: item.state for item in runtime.components}


def _reasons(runtime: RuntimeState) -> dict[str, str | None]:
    """Return each component's reason code, keyed by component name."""
    return {item.component: item.reason for item in runtime.components}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def test_a_rule_only_runtime_is_ready(rule_only: RuntimeState) -> None:
    """With the model layer switched off, the rule layer alone can serve."""
    assert rule_only.ready is True
    states = _states(rule_only)
    assert states[COMPONENT_FEATURE_CONTRACT] is ComponentState.READY
    assert states[COMPONENT_RULE_ENGINE] is ComponentState.READY
    assert states[COMPONENT_ML_CHAMPION] is ComponentState.DISABLED


def test_every_declared_component_is_reported(rule_only: RuntimeState) -> None:
    """A readiness document that omits a component hides a failure mode."""
    assert set(_states(rule_only)) == {
        COMPONENT_FEATURE_CONTRACT,
        COMPONENT_RULE_ENGINE,
        COMPONENT_MODEL_ARTIFACTS,
        COMPONENT_ML_CHAMPION,
        COMPONENT_FUSION,
        COMPONENT_REPLAY,
    }


def test_a_required_model_with_no_artifact_root_is_not_ready() -> None:
    """A deployment that requires a champion and names no root cannot serve."""
    runtime = build_runtime(APISettings(require_ml_champion=True))
    assert runtime.ready is False
    assert (
        _reasons(runtime)[COMPONENT_MODEL_ARTIFACTS] == "artifact_root_not_configured"
    )
    assert _states(runtime)[COMPONENT_ML_CHAMPION] is ComponentState.UNAVAILABLE


def test_an_absent_artifact_root_is_reported_rather_than_raised(
    tmp_path: Path,
) -> None:
    """Startup records the failure; it does not crash the process."""
    runtime = build_runtime(
        APISettings(require_ml_champion=True, artifact_root=tmp_path / "nowhere")
    )
    assert runtime.ready is False
    assert _reasons(runtime)[COMPONENT_MODEL_ARTIFACTS] == "artifact_root_not_found"


def test_an_artifact_root_with_no_frozen_champion_is_named_as_such(
    tmp_path: Path,
) -> None:
    """'Nothing was ever frozen here' is a different finding from 'it broke'."""
    (tmp_path / "runs").mkdir()
    runtime = build_runtime(
        APISettings(require_ml_champion=True, artifact_root=tmp_path)
    )
    assert _reasons(runtime)[COMPONENT_MODEL_ARTIFACTS] == "no_champion_frozen"


def test_an_unreadable_feature_configuration_is_reported(tmp_path: Path) -> None:
    """The whole runtime is defined against the feature contract, so it goes first."""
    broken = tmp_path / "features.yaml"
    broken.write_text("windows: [not-a-window]\n", encoding="utf-8")
    runtime = build_runtime(
        APISettings(require_ml_champion=False, feature_config_path=broken)
    )
    assert runtime.ready is False
    assert _reasons(runtime)[COMPONENT_FEATURE_CONTRACT] == "feature_config_unreadable"


def test_a_missing_feature_configuration_file_is_reported(tmp_path: Path) -> None:
    """A path that names nothing is a configuration failure, not a crash."""
    runtime = build_runtime(
        APISettings(
            require_ml_champion=False, feature_config_path=tmp_path / "absent.yaml"
        )
    )
    assert _reasons(runtime)[COMPONENT_FEATURE_CONTRACT] == "feature_config_unreadable"


def test_an_unreadable_rule_configuration_is_reported(tmp_path: Path) -> None:
    """A broken rule configuration disables detection rather than degrading it."""
    broken = tmp_path / "rules.yaml"
    broken.write_text("rules: {PAD-BF-001: {parameters: {window: nonsense}}}\n")
    runtime = build_runtime(
        APISettings(require_ml_champion=False, detection_config_path=broken)
    )
    assert runtime.ready is False
    assert _reasons(runtime)[COMPONENT_RULE_ENGINE] == "detection_config_unreadable"


def test_a_broken_feature_contract_takes_the_rule_layer_with_it(
    tmp_path: Path,
) -> None:
    """Rules are declared against the catalog, so an absent catalog is fatal to them."""
    broken = tmp_path / "features.yaml"
    broken.write_text("windows: [not-a-window]\n", encoding="utf-8")
    runtime = build_runtime(
        APISettings(require_ml_champion=False, feature_config_path=broken)
    )
    assert _reasons(runtime)[COMPONENT_RULE_ENGINE] == "feature_contract_unavailable"


def _frozen_looking_root(tmp_path: Path) -> Path:
    """Return a root that *looks* frozen: a champion directory holding a lock.

    Structurally complete and semantically empty, which is exactly the state the
    ``model_artifacts`` and ``ml_champion`` components exist to tell apart.
    """
    scope = tmp_path / "champion" / ("a" * 64)
    scope.mkdir(parents=True)
    (scope / "champion.lock").write_text("{}", encoding="utf-8")
    return tmp_path


def _reviewed_allowlist() -> Path:
    """Return the repository's own reviewed feature allowlist."""
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "ml"
        / "features-allowlist-v1.yaml"
    )


def test_a_lock_that_does_not_verify_is_distinguished_from_an_absent_one(
    tmp_path: Path,
) -> None:
    """Artifacts present, champion unusable: two components, two findings.

    A lock file that is not a lock gets no benefit of the doubt anywhere in the
    verification chain, and the failure is reported without quoting what was
    wrong with it.
    """
    runtime = build_runtime(
        APISettings(
            require_ml_champion=True,
            artifact_root=_frozen_looking_root(tmp_path),
            allowlist_path=_reviewed_allowlist(),
        )
    )
    assert _states(runtime)[COMPONENT_MODEL_ARTIFACTS] is ComponentState.READY
    assert _reasons(runtime)[COMPONENT_ML_CHAMPION] == "champion_verification_failed"
    assert runtime.ready is False


def test_a_champion_with_no_allowlist_is_refused(tmp_path: Path) -> None:
    """The reviewed feature contract is not optional, so its absence is a refusal."""
    runtime = build_runtime(
        APISettings(
            require_ml_champion=True, artifact_root=_frozen_looking_root(tmp_path)
        )
    )
    assert _reasons(runtime)[COMPONENT_ML_CHAMPION] == "allowlist_not_configured"


def test_an_unreadable_ml_configuration_is_reported(tmp_path: Path) -> None:
    """The configuration the champion was produced under has to be loadable."""
    broken = tmp_path / "ml.yaml"
    broken.write_text("preprocessing: {include_leakage_classes: [nonsense]}\n")
    runtime = build_runtime(
        APISettings(
            require_ml_champion=True,
            artifact_root=_frozen_looking_root(tmp_path),
            allowlist_path=_reviewed_allowlist(),
            ml_config_path=broken,
        )
    )
    assert _reasons(runtime)[COMPONENT_ML_CHAMPION] == "ml_config_unreadable"


def test_an_unusable_allowlist_is_reported(tmp_path: Path) -> None:
    """A file that is not a reviewed allowlist cannot resolve a feature set."""
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text("just: a mapping\n", encoding="utf-8")
    runtime = build_runtime(
        APISettings(
            require_ml_champion=True,
            artifact_root=_frozen_looking_root(tmp_path),
            allowlist_path=allowlist,
        )
    )
    assert _reasons(runtime)[COMPONENT_ML_CHAMPION] == "allowlist_unusable"


def test_the_fusion_component_follows_the_model_it_depends_on(
    tmp_path: Path,
) -> None:
    """With no champion there is nothing to fuse a rule verdict with."""
    runtime = build_runtime(
        APISettings(
            require_ml_champion=True, artifact_root=_frozen_looking_root(tmp_path)
        )
    )
    assert _reasons(runtime)[COMPONENT_FUSION] == "ml_champion_unavailable"
    assert runtime.fusion.available is False


def test_no_reason_code_carries_a_path(tmp_path: Path) -> None:
    """A readiness document is read by whoever can reach the port."""
    runtime = build_runtime(
        APISettings(
            require_ml_champion=True,
            artifact_root=tmp_path / "secret-place",
            allowlist_path=tmp_path / "secret-allowlist.yaml",
        )
    )
    for reason in _reasons(runtime).values():
        assert reason is None or "/" not in reason
        assert reason is None or str(tmp_path) not in reason


def test_the_serving_scope_is_one_fixed_value() -> None:
    """Live rows are all scored in one scope; a caller cannot select another."""
    assert isinstance(SERVING_SCOPE, ServingScope)
    assert SERVING_SCOPE is ServingScope.LIVE
    assert len(list(ServingScope)) == 1


def test_the_serving_scope_is_not_a_split() -> None:
    """A live request is not an experimental population and cannot claim to be."""
    assert not isinstance(SERVING_SCOPE, MLSplit)
    assert str(SERVING_SCOPE) not in {str(item) for item in MLSplit}


def test_the_serving_module_cannot_reach_a_split_label() -> None:
    """The guard that keeps a split label out of the request path.

    Asserted over the parsed module rather than over its text, so the guard's own
    docstring naming what it forbids does not count as using it.
    """
    import ast

    from password_attack_detector.api import services as module

    namespace = vars(module)
    for forbidden in ("MLSplit", "SplitRow", "assemble_inference_dataset"):
        assert forbidden not in namespace

    tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "MLSplit" not in used
    assert not {"TRAIN", "VALIDATION", "TEST", "NOVEL_ANOMALY_HOLDOUT"} & used


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def test_the_readiness_document_mirrors_the_runtime(rule_only: RuntimeState) -> None:
    """The document is derived from the components, not asserted beside them."""
    document = readiness_document(rule_only)
    assert document.status is ReadinessState.READY
    assert document.version == __version__
    assert len(document.components) == len(rule_only.components)


def test_the_version_document_reports_the_release() -> None:
    """Milestone 1 ships against 0.5.0 and changes no contract version."""
    document = version_document()
    assert document.package_version == __version__ == "0.5.0"
    assert document.api_schema_version == "1.0.0"
    assert document.event_schema_version == "1.0.0"
    assert document.detection_schema_version == "1.0.0"


def test_the_system_status_reports_the_layers_actually_running(
    rule_only: RuntimeState,
) -> None:
    """Enabled means loaded, not configured."""
    document = system_status_document(rule_only)
    assert document.rule_detection_enabled is True
    assert document.ml_detection_enabled is False
    assert document.hybrid_detection_enabled is False
    assert document.enabled_rule_count == document.registered_rule_count
    assert document.champion_model_family is None


def test_the_model_document_says_why_there_is_no_model(
    rule_only: RuntimeState,
) -> None:
    """An absent model reports the component's own reason, not a generic one."""
    document = model_info_document(rule_only)
    assert document.available is False
    assert document.unavailable_reason == "ml_champion_disabled"
    assert document.model_id is None
    assert document.decision_threshold is None


def test_the_rule_catalog_is_public_safe(rule_only: RuntimeState) -> None:
    """Identity and description; no thresholds and no feature names."""
    document = rule_catalog_document(rule_only)
    assert document.rule_count == 9
    assert document.enabled_rule_count == 9
    for rule in document.rules:
        assert rule.rule_id.startswith("PAD-")
        fields = set(type(rule).model_fields)
        assert fields & {"parameters", "required_features", "thresholds"} == set()


def test_the_rule_catalog_reports_nothing_enabled_without_a_rule_layer(
    tmp_path: Path,
) -> None:
    """A catalog is still published when the engine failed; nothing is enabled."""
    broken = tmp_path / "features.yaml"
    broken.write_text("windows: [not-a-window]\n", encoding="utf-8")
    runtime = build_runtime(
        APISettings(require_ml_champion=False, feature_config_path=broken)
    )
    document = rule_catalog_document(runtime)
    assert document.rule_count == 9
    assert document.enabled_rule_count == 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_an_unready_runtime_refuses_detection(tmp_path: Path) -> None:
    """The refusal happens before an event is converted or a feature computed."""
    runtime = build_runtime(APISettings(require_ml_champion=True))
    request = DetectionWindowRequest.model_validate({"events": brute_force_window(3)})
    with pytest.raises(APIError) as caught:
        detect_single(runtime, request)
    assert caught.value.code is ErrorCode.RUNTIME_NOT_READY
    assert caught.value.status_code == 503


def test_a_window_over_the_deployment_ceiling_is_refused() -> None:
    """The per-deployment limit, enforced by the service rather than the parser."""
    runtime = build_runtime(APISettings(require_ml_champion=False, max_batch_events=5))
    request = DetectionBatchRequest.model_validate({"events": brute_force_window(9)})
    with pytest.raises(APIError) as caught:
        detect_batch(runtime, request)
    assert caught.value.code is ErrorCode.BATCH_LIMIT_EXCEEDED
    assert caught.value.detail == {"event_count": 9, "max_batch_events": 5}


def test_the_single_anchor_endpoint_refuses_a_multi_anchor_selection(
    rule_only: RuntimeState,
) -> None:
    """One anchor is the contract; answering two would silently drop one."""
    events = brute_force_window(3)
    request = DetectionWindowRequest.model_validate(
        {
            "events": events,
            "anchor_selection": "explicit",
            "anchor_event_ids": [events[0]["event_id"], events[1]["event_id"]],
        }
    )
    with pytest.raises(APIError) as caught:
        detect_single(rule_only, request)
    assert caught.value.code is ErrorCode.ANCHOR_SELECTION_ERROR


def test_a_source_address_without_a_key_is_refused(rule_only: RuntimeState) -> None:
    """No key means no way to process the address under the privacy contract."""
    runtime = dataclasses.replace(rule_only, pseudonymizer=None)
    body = [
        {k: v for k, v in item.items() if k != "source_id"}
        | {"source_ip": "203.0.113.5"}
        for item in brute_force_window(2)
    ]
    request = DetectionWindowRequest.model_validate({"events": body})
    with pytest.raises(APIError) as caught:
        detect_single(runtime, request)
    assert caught.value.code is ErrorCode.PSEUDONYMIZATION_UNAVAILABLE


def test_a_source_address_is_pseudonymized_and_dropped(
    rule_only: RuntimeState,
) -> None:
    """The address is turned into a source pseudonym and never carried further.

    The same keyed HMAC the ingestion adapters use, so a live request and a
    historical dataset produce the same pseudonym for the same address.
    """
    service = PseudonymService("b" * 64)
    runtime = dataclasses.replace(rule_only, pseudonymizer=service)
    body = [
        {k: v for k, v in item.items() if k != "source_id"}
        | {"source_ip": "198.51.100.4"}
        for item in brute_force_window(30)
    ]
    request = DetectionWindowRequest.model_validate({"events": body})
    response = detect_single(runtime, request)
    rendered = response.model_dump_json()
    assert "198.51.100.4" not in rendered
    assert service.pseudonymize("source", "198.51.100.4") not in rendered
    # The pseudonym really was used: the burst is one source, so the
    # source-scoped rules see it exactly as they would a supplied source_id.
    assert response.anchor.rule.flagged is True


def test_a_brute_force_window_flags_the_rule_layer(rule_only: RuntimeState) -> None:
    """The scenario the rules were written for, scored through the whole adapter."""
    request = DetectionWindowRequest.model_validate({"events": brute_force_window(30)})
    response = detect_single(rule_only, request)
    assert response.anchor.rule.flagged is True
    assert response.anchor.rule.risk_score > 0.0
    assert "PAD-BF-001" in response.anchor.rule.fired_rule_ids
    assert response.anchor.rule.evidence


def test_ordinary_traffic_does_not_flag(rule_only: RuntimeState) -> None:
    """Well-spaced successes from one identity are not an attack."""
    request = DetectionWindowRequest.model_validate({"events": normal_window(10)})
    response = detect_single(rule_only, request)
    assert response.anchor.rule.flagged is False
    assert response.anchor.rule.risk_score == 0.0
    assert response.anchor.severity is Severity.LOW


def test_the_batch_response_is_in_canonical_order(rule_only: RuntimeState) -> None:
    """Anchor order comes from the data, not from the order they were requested."""
    events = brute_force_window(6)
    forwards = DetectionBatchRequest.model_validate(
        {
            "events": events,
            "anchor_selection": "explicit",
            "anchor_event_ids": [item["event_id"] for item in events],
        }
    )
    backwards = DetectionBatchRequest.model_validate(
        {
            "events": events,
            "anchor_selection": "explicit",
            "anchor_event_ids": [item["event_id"] for item in reversed(events)],
        }
    )
    assert detect_batch(rule_only, forwards) == detect_batch(rule_only, backwards)


def test_scoring_the_same_window_twice_gives_the_same_answer(
    rule_only: RuntimeState,
) -> None:
    """No wall clock, no random state, no per-process identity reaches a verdict."""
    request = DetectionBatchRequest.model_validate({"events": brute_force_window(12)})
    assert detect_batch(rule_only, request) == detect_batch(rule_only, request)


def test_a_window_scores_independently_of_another(rule_only: RuntimeState) -> None:
    """Two requests must not see one another's history through a shared engine."""
    quiet = DetectionWindowRequest.model_validate({"events": normal_window(4)})
    first = detect_single(rule_only, quiet)
    detect_single(
        rule_only,
        DetectionWindowRequest.model_validate({"events": brute_force_window(30)}),
    )
    assert detect_single(rule_only, quiet) == first


def test_the_model_layer_is_unavailable_with_its_reason(
    rule_only: RuntimeState,
) -> None:
    """A rule-only deployment says so on every anchor rather than staying silent."""
    request = DetectionWindowRequest.model_validate({"events": brute_force_window(3)})
    response = detect_single(rule_only, request)
    assert response.anchor.ml.available is False
    assert response.anchor.ml.unavailable_reason == "ml_champion_disabled"
    assert response.anchor.ml.flagged is None
    assert response.anchor.hybrid.available is False


# ---------------------------------------------------------------------------
# The fusion arm
# ---------------------------------------------------------------------------


def _assessment(*, fired: int, score: float) -> RiskAssessment:
    """Return a risk assessment with the given number of fired rules."""
    if fired == 0:
        return RiskAssessment(
            anchor_event_id="anchor-1",
            anchor_event_time=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
            risk_score=0.0,
            severity=Severity.LOW,
            scoring_version="1.0.0",
        )
    return RiskAssessment(
        anchor_event_id="anchor-1",
        anchor_event_time=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
        risk_score=score,
        severity=Severity.HIGH,
        primary_attack_category=AttackCategory.BRUTE_FORCE,
        contributing_categories=(AttackCategory.BRUTE_FORCE,),
        fired_rule_count=1,
        fired_rule_ids=("PAD-BF-001",),
        scoring_version="1.0.0",
    )


def _prediction(*, flagged: bool) -> BinaryPrediction:
    """Return a binary prediction with the given verdict."""
    return BinaryPrediction(
        anchor_event_id="anchor-1",
        anchor_event_time=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
        score_kind=ScoreKind.DECISION_SCORE,
        malicious_decision_score=0.9 if flagged else 0.1,
        malicious_probability=None,
        decision_threshold=0.5,
        flagged_malicious=flagged,
    )


def _running(
    strategy: FusionStrategy, *, stacked: StackedFusionState | None = None
) -> FusionRuntime:
    """Return a runtime executing the frozen *strategy*."""
    return FusionRuntime(
        selected_strategy=strategy,
        strategy=strategy,
        stacked_state=stacked,
        required=True,
        unavailable_reason=None,
    )


def _stacked_state() -> StackedFusionState:
    """Return a fitted meta-learner, sealed exactly as a real one is.

    Fitted rather than hand-written: a hand-built state would not recompute its
    own seal, and every path this test exercises refuses one that does not.
    """
    rows = [
        MetaRow(ml_score=0.9, rule_flagged=True, malicious=True),
        MetaRow(ml_score=0.8, rule_flagged=True, malicious=True),
        MetaRow(ml_score=0.1, rule_flagged=False, malicious=False),
        MetaRow(ml_score=0.2, rule_flagged=False, malicious=False),
    ]
    return fit_stacked_fusion(rows, fold_count=2)


@pytest.mark.parametrize(
    ("strategy", "rule_fired", "ml_flagged", "expected"),
    [
        (FusionStrategy.OR_GATE, 1, False, True),
        (FusionStrategy.OR_GATE, 0, False, False),
        (FusionStrategy.AND_GATE, 1, False, False),
        (FusionStrategy.AND_GATE, 1, True, True),
    ],
)
def test_a_selected_gate_is_applied_to_both_booleans(
    rule_only: RuntimeState,
    strategy: FusionStrategy,
    rule_fired: int,
    ml_flagged: bool,
    expected: bool,
) -> None:
    """The gate combines two decisions; no score of either scale is arithmetic."""
    runtime = dataclasses.replace(rule_only, fusion=_running(strategy))
    layer = hybrid_layer(
        runtime,
        _assessment(fired=rule_fired, score=60.0),
        _prediction(flagged=ml_flagged),
    )
    assert layer.available is True
    assert layer.strategy is strategy
    assert layer.flagged is expected


def test_a_selected_stacked_hybrid_runs_the_loaded_meta_learner(
    rule_only: RuntimeState,
) -> None:
    """The state the bundle published is the state the request is fused under."""
    state = _stacked_state()
    runtime = dataclasses.replace(
        rule_only, fusion=_running(FusionStrategy.STACKED, stacked=state)
    )
    layer = hybrid_layer(
        runtime, _assessment(fired=1, score=60.0), _prediction(flagged=True)
    )
    assert layer.available is True
    assert layer.strategy is FusionStrategy.STACKED
    # The verdict is the meta-learner's own, applied to this row's evidence.
    expected = state.probability(ml_score=0.9, rule_flagged=True)
    assert layer.flagged is (expected >= state.decision_threshold)


def test_an_unselected_hybrid_is_reported_rather_than_defaulted(
    rule_only: RuntimeState,
) -> None:
    """Falling back to a gate would publish a hybrid nobody selected."""
    runtime = dataclasses.replace(
        rule_only,
        fusion=FusionRuntime(
            selected_strategy=None,
            strategy=None,
            stacked_state=None,
            required=False,
            unavailable_reason="no_fusion_selection",
        ),
    )
    layer = hybrid_layer(
        runtime, _assessment(fired=1, score=60.0), _prediction(flagged=True)
    )
    assert layer.available is False
    assert layer.unavailable_reason == "no_fusion_selection"
    assert layer.flagged is None
    assert layer.strategy is None


@pytest.mark.parametrize(
    "reason",
    [
        "serving_bundle_not_published",
        "serving_bundle_unverifiable",
        "serving_bundle_lineage_mismatch",
        "fusion_selection_conflict",
    ],
)
def test_a_stacked_selection_is_not_substituted_with_a_gate(
    rule_only: RuntimeState, reason: str
) -> None:
    """A stacker that cannot be loaded is reported, never replaced by a gate."""
    runtime = dataclasses.replace(
        rule_only,
        fusion=FusionRuntime(
            selected_strategy=FusionStrategy.STACKED,
            strategy=None,
            stacked_state=None,
            required=True,
            unavailable_reason=reason,
        ),
    )
    layer = hybrid_layer(
        runtime, _assessment(fired=1, score=60.0), _prediction(flagged=True)
    )
    assert layer.available is False
    assert layer.unavailable_reason == reason
    assert layer.strategy is None
    assert layer.flagged is None


# ---------------------------------------------------------------------------
# The runtime cannot be constructed into a state that fakes a hybrid
# ---------------------------------------------------------------------------


def test_a_runtime_cannot_execute_a_strategy_it_did_not_freeze() -> None:
    """The no-fallback rule is structural, not a convention in ``_resolve_fusion``."""
    with pytest.raises(ValueError, match="executes the frozen strategy or none"):
        FusionRuntime(
            selected_strategy=FusionStrategy.STACKED,
            strategy=FusionStrategy.OR_GATE,
            stacked_state=None,
            required=True,
            unavailable_reason=None,
        )


def test_a_stacked_runtime_cannot_run_without_its_fitted_state() -> None:
    """There is no default stacker, and no way to construct a runtime with one."""
    with pytest.raises(ValueError, match="runs its fitted meta-learner"):
        FusionRuntime(
            selected_strategy=FusionStrategy.STACKED,
            strategy=FusionStrategy.STACKED,
            stacked_state=None,
            required=True,
            unavailable_reason=None,
        )


def test_a_gate_runtime_cannot_carry_a_stacker() -> None:
    """A boolean gate has no fitted state; carrying one would be deploying it."""
    with pytest.raises(ValueError, match="runs its fitted meta-learner"):
        FusionRuntime(
            selected_strategy=FusionStrategy.OR_GATE,
            strategy=FusionStrategy.OR_GATE,
            stacked_state=_stacked_state(),
            required=True,
            unavailable_reason=None,
        )


def test_a_frozen_hybrid_is_always_a_required_component() -> None:
    """A selected hybrid that cannot run must not read as an optional extra."""
    with pytest.raises(ValueError, match="required runtime component"):
        FusionRuntime(
            selected_strategy=FusionStrategy.STACKED,
            strategy=None,
            stacked_state=None,
            required=False,
            unavailable_reason="serving_bundle_not_published",
        )


def test_an_unavailable_hybrid_must_name_a_reason() -> None:
    """Silence is not a readiness state."""
    with pytest.raises(ValueError, match="must name a stable reason"):
        FusionRuntime(
            selected_strategy=None,
            strategy=None,
            stacked_state=None,
            required=False,
            unavailable_reason=None,
        )
