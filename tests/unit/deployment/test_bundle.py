"""Tests for the serving-bundle artifact contract.

Everything here is built from **real sealed documents**: a fitted stacker from
``fit_stacked_fusion``, a selection from ``select_fusion_strategy``, and a
manifest from ``ServingBundleManifest.seal``. A hand-written document would fail
its own seal, and every path below refuses one that does -- so a test that
hand-built its inputs would be testing the refusal rather than the contract.

What the assertions are actually about:

* the manifest's shape ties a stacked bundle to its state and a gate bundle to
  none;
* loading is verification, and every tamper is a refusal;
* the state a bundle serves must be the state the *frozen selection* named;
* publication is transactional, idempotent, and never overwrites;
* nothing in the contract can carry a metric, a threshold, or a fallback.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.deployment.bundle import (
    BUNDLE_MANIFEST_FILE,
    BUNDLE_SCHEMA_VERSION,
    FUSION_SELECTION_FILE,
    SERVING_BUNDLE_DIR,
    STACKED_STATE_FILE,
    ServingBundleManifest,
    bundle_directory,
    bundle_files,
    load_serving_bundle,
    write_serving_bundle,
)
from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ExperimentPublicationError,
    ManifestVerificationError,
)
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    FusionStrategy,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.fusion import (
    FusionSelection,
    MetaRow,
    MLEvidence,
    RuleEvidence,
    StackedFusionState,
    fit_stacked_fusion,
    fuse,
    select_fusion_strategy,
)

LOCK = "b" * 64
SCOPE = "c" * 64
RULES = "a" * 64
FREEZE = "00000000-0000-5000-8000-000000000001"
EVIDENCE = "e" * 64
DIGEST = "d" * 64


#: One validation population, as ``(malicious, rule_flagged, ml_flagged)``.
#:
#: Chosen so the three strategies genuinely separate on it, which is what lets a
#: *real* selection be produced rather than a hand-built one:
#:
#: * the rule arm alone is exactly right, so a stacker that learns to follow it
#:   flags both positives and neither negative;
#: * the model arm flags one benign row, so ``OR_GATE`` inherits that false
#:   positive;
#: * the two arms never agree on a positive, so ``AND_GATE`` flags nothing.
_POPULATION: tuple[tuple[bool, bool, bool], ...] = (
    (True, True, False),
    (True, True, False),
    (False, False, True),
    (False, False, False),
    (False, False, False),
)


def _stacked_state() -> StackedFusionState:
    """Return a meta-learner fitted on :data:`_POPULATION`.

    Genuinely fitted, by the project's own deterministic gradient descent. The
    class imbalance is what moves the intercept negative, which is what makes the
    fitted state separate the two groups rather than sitting on its own decision
    threshold.
    """
    return fit_stacked_fusion(
        [
            MetaRow(ml_score=0.9, rule_flagged=rule, malicious=malicious)
            for malicious, rule, _ml in _POPULATION
        ],
        fold_count=2,
    )


def _other_stacked_state() -> StackedFusionState:
    """Return a different fitted meta-learner, for the impostor cases.

    Fitted from different evidence and a different fold count, so it is a
    genuinely different stacker rather than a copy with an edited field -- an
    edited one would fail its own seal and never reach the check under test.
    """
    return fit_stacked_fusion(
        [
            MetaRow(ml_score=0.6, rule_flagged=True, malicious=True),
            MetaRow(ml_score=0.7, rule_flagged=True, malicious=True),
            MetaRow(ml_score=0.4, rule_flagged=False, malicious=False),
            MetaRow(ml_score=0.3, rule_flagged=False, malicious=False),
        ],
        fold_count=3,
    )


def _selection(
    *, stacked: StackedFusionState | None, chosen: FusionStrategy
) -> FusionSelection:
    """Return a sealed selection that selected *chosen*, on validation evidence.

    Produced by :func:`select_fusion_strategy` itself, under gates tightened just
    enough to leave one candidate standing. The gates differ between the stacked
    and the gate case because the point of each fixture differs -- and either way
    the selection is a real one, so its fingerprint is a real fingerprint.
    """
    stacked_wanted = chosen is FusionStrategy.STACKED
    outcomes = [row[0] for row in _POPULATION]
    decisions: dict[FusionStrategy, tuple[Any, ...]] = {}
    for strategy in FusionStrategy:
        if strategy is FusionStrategy.STACKED and stacked is None:
            # Exactly what ``fuse_population`` does: an unbuildable candidate
            # enters the comparison measured on nothing.
            decisions[strategy] = ()
            continue
        decisions[strategy] = tuple(
            fuse(
                strategy,
                anchor_event_id=f"anchor-{index}",
                anchor_event_time=datetime(2026, 3, 4, 12, index, tzinfo=UTC),
                rule=RuleEvidence(flagged=rule, ordinal_risk_score=60.0),
                ml=MLEvidence(
                    flagged=ml_flagged,
                    decision_score=0.9,
                    calibrated_probability=0.9,
                    score_kind=ScoreKind.CALIBRATED_PROBABILITY,
                ),
                stacked=stacked,
            )
            for index, (_malicious, rule, ml_flagged) in enumerate(_POPULATION)
        )
    built = select_fusion_strategy(
        decisions=decisions,
        malicious=outcomes,
        unavailable_strategies=(
            {}
            if stacked is not None
            else {FusionStrategy.STACKED: "stacked_unavailable: none in this fixture"}
        ),
        stacked_unavailable_reason=(
            None if stacked is not None else "stacked_unavailable: none in this fixture"
        ),
        # Tightened for the stacked fixture so the gates are measured and found
        # wanting: AND_GATE flags nothing (below the detection floor) and OR_GATE
        # carries a false positive (above the ceiling). Loose for the gate
        # fixture, where OR_GATE wins on the declared tie-break order.
        min_detection_rate=0.9 if stacked_wanted else 0.5,
        max_false_positive_rate=0.2 if stacked_wanted else 0.5,
        min_validation_positive_rows=1,
        min_validation_benign_rows=1,
        rule_configuration_fingerprint=RULES,
        champion_lock_fingerprint=LOCK,
        champion_freeze_record_id=FREEZE,
        validation_evidence_fingerprint=EVIDENCE,
        stacked_state_fingerprint=None
        if stacked is None
        else stacked.state_fingerprint,
        oof_fold_definition_fingerprint=None if stacked is None else DIGEST,
        oof_evidence_fingerprint=None if stacked is None else DIGEST,
        base_model_recipe_fingerprint=None if stacked is None else DIGEST,
    )
    # The helper is only useful if it produced the strategy the test asked for;
    # a selection that chose something else would silently test another path.
    assert built.selected_strategy is chosen, built.selected_strategy
    return built


def _manifest(
    *, selection: FusionSelection, stacked: StackedFusionState | None
) -> ServingBundleManifest:
    """Return a sealed manifest for *selection* and its payloads."""
    import hashlib

    files = bundle_files(selection=selection, stacked_state=stacked)
    assert selection.selected_strategy is not None
    return ServingBundleManifest.seal(
        champion_scope_key=SCOPE,
        champion_lock_fingerprint=LOCK,
        champion_freeze_record_id=FREEZE,
        validation_selection_id=FREEZE,
        catalog_model_id="M-010",
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_id="model-1",
        model_content_fingerprint=DIGEST,
        preprocessor_fingerprint=DIGEST,
        calibration_method=CalibrationMethod.PLATT,
        calibration_state_fingerprint=DIGEST,
        binary_threshold_fingerprint=DIGEST,
        feature_catalog_fingerprint=DIGEST,
        allowlist_fingerprint=DIGEST,
        eligible_feature_list_fingerprint=DIGEST,
        ml_config_fingerprint=DIGEST,
        serializer_id="project-json-npz",
        serializer_version=1,
        dependency_contract_fingerprint=DIGEST,
        selected_fusion_strategy=selection.selected_strategy,
        fusion_selection_fingerprint=selection.selection_fingerprint,
        fusion_config_fingerprint=selection.fusion_config_fingerprint,
        rule_configuration_fingerprint=selection.rule_configuration_fingerprint,
        validation_evidence_fingerprint=selection.validation_evidence_fingerprint,
        stacked_state_fingerprint=None
        if stacked is None
        else stacked.state_fingerprint,
        oof_fold_definition_fingerprint=selection.oof_fold_definition_fingerprint,
        oof_evidence_fingerprint=selection.oof_evidence_fingerprint,
        base_model_recipe_fingerprint=selection.base_model_recipe_fingerprint,
        evaluation_record_id=FREEZE,
        evaluation_record_fingerprint=DIGEST,
        files=tuple(
            (name, hashlib.sha256(body.encode("utf-8")).hexdigest())
            for name, body in sorted(files.items())
        ),
    )


@pytest.fixture()
def stacked_bundle(tmp_path: Path) -> tuple[Path, StackedFusionState, FusionSelection]:
    """Publish a stacked bundle and return the root and what went into it."""
    state = _stacked_state()
    selection = _selection(stacked=state, chosen=FusionStrategy.STACKED)
    manifest = _manifest(selection=selection, stacked=state)
    write_serving_bundle(
        root=tmp_path, manifest=manifest, selection=selection, stacked_state=state
    )
    return (tmp_path, state, selection)


# ---------------------------------------------------------------------------
# The manifest's shape
# ---------------------------------------------------------------------------


def test_a_stacked_manifest_names_its_state_and_its_file() -> None:
    """The payload set and the declared fingerprint move together."""
    state = _stacked_state()
    manifest = _manifest(
        selection=_selection(stacked=state, chosen=FusionStrategy.STACKED),
        stacked=state,
    )
    assert manifest.selected_fusion_strategy is FusionStrategy.STACKED
    assert manifest.stacked_state_fingerprint == state.state_fingerprint
    assert dict(manifest.files).keys() == {FUSION_SELECTION_FILE, STACKED_STATE_FILE}
    assert manifest.bundle_schema_version == BUNDLE_SCHEMA_VERSION


def test_a_gate_manifest_carries_no_fitted_state() -> None:
    """A boolean gate has nothing fitted; publishing one would deploy it."""
    selection = _selection(stacked=None, chosen=FusionStrategy.OR_GATE)
    manifest = _manifest(selection=selection, stacked=None)
    assert manifest.selected_fusion_strategy is FusionStrategy.OR_GATE
    assert manifest.stacked_state_fingerprint is None
    assert dict(manifest.files).keys() == {FUSION_SELECTION_FILE}


def test_a_stacked_manifest_without_a_state_is_refused() -> None:
    """There is no stacked bundle whose stacker was left out."""
    state = _stacked_state()
    selection = _selection(stacked=state, chosen=FusionStrategy.STACKED)
    with pytest.raises(ValueError, match="names the fitted state it applies"):
        _manifest(selection=selection, stacked=None)


def test_a_manifest_that_declares_no_selection_file_is_refused() -> None:
    """A strategy nobody can trace to a selection is not deployable."""
    with pytest.raises(ValueError, match="always carries the frozen fusion selection"):
        ServingBundleManifest.seal(
            **_minimal_manifest_fields(),
            selected_fusion_strategy=FusionStrategy.OR_GATE,
            stacked_state_fingerprint=None,
            files=(("something_else.json", DIGEST),),
        )


def test_a_manifest_file_name_may_not_be_a_path() -> None:
    """A bundle reads files beside its manifest, never through one."""
    with pytest.raises(ValueError, match="is a path"):
        ServingBundleManifest.seal(
            **_minimal_manifest_fields(),
            selected_fusion_strategy=FusionStrategy.OR_GATE,
            stacked_state_fingerprint=None,
            files=((FUSION_SELECTION_FILE, DIGEST), ("../escape.json", DIGEST)),
        )


def _minimal_manifest_fields() -> dict[str, Any]:
    """Return every manifest field except the ones a test is varying."""
    return {
        "champion_scope_key": SCOPE,
        "champion_lock_fingerprint": LOCK,
        "champion_freeze_record_id": FREEZE,
        "validation_selection_id": FREEZE,
        "catalog_model_id": "M-010",
        "model_family": ModelFamily.LOGISTIC_REGRESSION,
        "model_id": "model-1",
        "model_content_fingerprint": DIGEST,
        "preprocessor_fingerprint": DIGEST,
        "calibration_method": CalibrationMethod.NONE,
        "calibration_state_fingerprint": None,
        "binary_threshold_fingerprint": DIGEST,
        "feature_catalog_fingerprint": DIGEST,
        "allowlist_fingerprint": DIGEST,
        "eligible_feature_list_fingerprint": DIGEST,
        "ml_config_fingerprint": DIGEST,
        "serializer_id": "project-json-npz",
        "serializer_version": 1,
        "dependency_contract_fingerprint": DIGEST,
        "fusion_selection_fingerprint": DIGEST,
        "fusion_config_fingerprint": DIGEST,
        "rule_configuration_fingerprint": RULES,
        "validation_evidence_fingerprint": EVIDENCE,
        "evaluation_record_id": FREEZE,
        "evaluation_record_fingerprint": DIGEST,
    }


def test_a_hand_edited_manifest_does_not_recompute_its_own_digest() -> None:
    """The seal is a field, so a reader detects tampering without a second source."""
    state = _stacked_state()
    manifest = _manifest(
        selection=_selection(stacked=state, chosen=FusionStrategy.STACKED),
        stacked=state,
    )
    payload = manifest.to_dict()
    payload["catalog_model_id"] = "M-999"
    with pytest.raises(Exception, match="not valid"):
        ServingBundleManifest.from_dict(payload)


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def test_publication_writes_the_manifest_last(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """A directory carrying a manifest is a complete bundle."""
    root, _state, _selection = stacked_bundle
    directory = bundle_directory(root, scope_key=SCOPE)
    assert (directory / BUNDLE_MANIFEST_FILE).is_file()
    assert (directory / FUSION_SELECTION_FILE).is_file()
    assert (directory / STACKED_STATE_FILE).is_file()
    assert (root / SERVING_BUNDLE_DIR).is_dir()


def test_every_document_ends_in_a_single_newline(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """Deterministic serialization: sorted keys, ASCII, one trailing newline."""
    root, _state, _selection = stacked_bundle
    directory = bundle_directory(root, scope_key=SCOPE)
    for name in (BUNDLE_MANIFEST_FILE, FUSION_SELECTION_FILE, STACKED_STATE_FILE):
        body = (directory / name).read_text(encoding="utf-8")
        assert body.endswith("}\n")
        assert body.count("\n") == 1
        payload = json.loads(body)
        assert list(payload) == sorted(payload)


def test_publishing_an_identical_bundle_twice_writes_nothing(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """Idempotent by content, which is what deterministic bytes buy."""
    root, state, selection = stacked_bundle
    manifest = _manifest(selection=selection, stacked=state)
    directory = bundle_directory(root, scope_key=SCOPE)
    before = {item.name: item.read_bytes() for item in sorted(directory.iterdir())}
    _target, created = write_serving_bundle(
        root=root, manifest=manifest, selection=selection, stacked_state=state
    )
    assert created is False
    after = {item.name: item.read_bytes() for item in sorted(directory.iterdir())}
    assert after == before


def test_a_different_bundle_for_one_scope_is_refused(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """A deployed hybrid is not rewritten in place, and there is no override."""
    root = stacked_bundle[0]
    gate = _selection(stacked=None, chosen=FusionStrategy.OR_GATE)
    with pytest.raises(ExperimentPublicationError, match="already published"):
        write_serving_bundle(
            root=root,
            manifest=_manifest(selection=gate, stacked=None),
            selection=gate,
            stacked_state=None,
        )


def test_a_manifest_that_does_not_declare_its_payloads_is_refused(
    tmp_path: Path,
) -> None:
    """An index that disagrees with the contents is refused, not reconciled."""
    state = _stacked_state()
    selection = _selection(stacked=state, chosen=FusionStrategy.STACKED)
    other = _other_stacked_state()
    manifest = _manifest(selection=selection, stacked=state)
    with pytest.raises(ExperimentPublicationError, match="does not declare"):
        write_serving_bundle(
            root=tmp_path,
            manifest=manifest,
            selection=selection,
            # A different state than the manifest indexed, which is exactly the
            # mismatch a publisher must not smooth over.
            stacked_state=other,
        )


def test_an_incomplete_publication_is_never_completed_in_place(
    tmp_path: Path,
) -> None:
    """A directory with no manifest is a failed publication, not a partial one."""
    state = _stacked_state()
    selection = _selection(stacked=state, chosen=FusionStrategy.STACKED)
    bundle_directory(tmp_path, scope_key=SCOPE).mkdir(parents=True)
    with pytest.raises(ExperimentPublicationError, match="without a manifest"):
        write_serving_bundle(
            root=tmp_path,
            manifest=_manifest(selection=selection, stacked=state),
            selection=selection,
            stacked_state=state,
        )


# ---------------------------------------------------------------------------
# Loading is verification
# ---------------------------------------------------------------------------


def test_a_published_bundle_loads_and_verifies(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """The happy path, and the shape a serving runtime receives."""
    root, state, selection = stacked_bundle
    bundle = load_serving_bundle(root, scope_key=SCOPE)
    assert bundle.strategy is FusionStrategy.STACKED
    assert bundle.stacked is True
    assert bundle.stacked_state is not None
    assert bundle.stacked_state.state_fingerprint == state.state_fingerprint
    assert bundle.selection.selection_fingerprint == selection.selection_fingerprint


def test_an_absent_bundle_is_reported_as_absent(tmp_path: Path) -> None:
    """ "Nothing is published here" is a different finding from "it is broken"."""
    with pytest.raises(ArtifactNotFoundError, match="no serving bundle"):
        load_serving_bundle(tmp_path, scope_key=SCOPE)


def test_a_scope_key_may_not_traverse(tmp_path: Path) -> None:
    """A bundle is never read from a path a caller composed."""
    for hostile in ("../elsewhere", "a/b", "..", ""):
        with pytest.raises(ManifestVerificationError, match="single directory name"):
            bundle_directory(tmp_path, scope_key=hostile)


def test_a_tampered_payload_is_refused(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """Each payload must digest to what the manifest recorded."""
    root, _state, _selection = stacked_bundle
    path = bundle_directory(root, scope_key=SCOPE) / STACKED_STATE_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["intercept"] = 42.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="does not digest"):
        load_serving_bundle(root, scope_key=SCOPE)


def test_an_undeclared_file_beside_the_manifest_is_refused(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """A manifest that indexes some of a directory indexes something else."""
    root, _state, _selection = stacked_bundle
    extra = bundle_directory(root, scope_key=SCOPE) / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="files and its manifest"):
        load_serving_bundle(root, scope_key=SCOPE)


def test_a_missing_declared_payload_is_refused(
    stacked_bundle: tuple[Path, StackedFusionState, FusionSelection],
) -> None:
    """The stacked case that matters most: the state is gone."""
    root, _state, _selection = stacked_bundle
    (bundle_directory(root, scope_key=SCOPE) / STACKED_STATE_FILE).unlink()
    with pytest.raises(ManifestVerificationError, match="files and its manifest"):
        load_serving_bundle(root, scope_key=SCOPE)


def test_a_state_the_frozen_selection_did_not_name_is_refused(
    tmp_path: Path,
) -> None:
    """The load-bearing check: the served stacker is the *selected* stacker.

    The bundle is published consistently -- manifest, digests, and seals all
    agree -- but around a state the frozen selection never named. It is refused,
    because a stacker that verifies is not thereby the right one.
    """
    selected = _stacked_state()
    selection = _selection(stacked=selected, chosen=FusionStrategy.STACKED)

    impostor = _other_stacked_state()
    assert impostor.state_fingerprint != selected.state_fingerprint

    # A manifest built around the impostor: internally consistent, and wrong.
    manifest = _manifest(selection=selection, stacked=impostor)
    write_serving_bundle(
        root=tmp_path,
        manifest=manifest,
        selection=selection,
        stacked_state=impostor,
    )
    with pytest.raises(ManifestVerificationError, match="the frozen selection named"):
        load_serving_bundle(tmp_path, scope_key=SCOPE)


def test_a_bundle_never_substitutes_a_strategy(tmp_path: Path) -> None:
    """A manifest deploying a strategy its selection did not select is refused."""
    gate = _selection(stacked=None, chosen=FusionStrategy.OR_GATE)
    manifest = ServingBundleManifest.seal(
        **_minimal_manifest_fields()
        | {
            "fusion_selection_fingerprint": gate.selection_fingerprint,
            "fusion_config_fingerprint": gate.fusion_config_fingerprint,
        },
        selected_fusion_strategy=FusionStrategy.AND_GATE,
        stacked_state_fingerprint=None,
        files=tuple(
            (name, __import__("hashlib").sha256(body.encode()).hexdigest())
            for name, body in sorted(
                bundle_files(selection=gate, stacked_state=None).items()
            )
        ),
    )
    write_serving_bundle(
        root=tmp_path, manifest=manifest, selection=gate, stacked_state=None
    )
    with pytest.raises(ManifestVerificationError, match="did not select the strategy"):
        load_serving_bundle(tmp_path, scope_key=SCOPE)
