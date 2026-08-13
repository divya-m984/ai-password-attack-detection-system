"""Unit tests for the Phase 5 model card and acceptance report.

Two properties are swept across the file. **No status is fabricated** -- a
requirement nothing established is `inconclusive`, an inconclusive report is not
accepted, and every structural check reads an executable contract rather than
asserting a sentence about one. And **the model card cannot go stale** -- the
vocabularies it prints are read from the enums they describe, so a card that
disagrees with the layer it documents fails here rather than in review.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from password_attack_detector import __version__
from password_attack_detector.ml.enums import (
    AcceptanceStatus,
    DriftStatus,
    ExplanationMethod,
    FusionStrategy,
    MLSplit,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.governance import (
    LABEL_READER_ALLOWLIST,
    AcceptanceEvidence,
    AcceptanceRequirement,
    Phase5AcceptanceReport,
    acceptance_report_to_markdown,
    build_acceptance_report,
    model_card_to_markdown,
    module_imports,
)
from password_attack_detector.ml.schemas import PROHIBITED_METADATA_FIELDS

#: Evidence describing a pipeline that really ran end to end, with a category
#: head that did not clear its gates and a dataset with no holdout rows -- both
#: of which are ordinary outcomes rather than failures.
COMPLETE = AcceptanceEvidence(
    champion_lock_fingerprint="a" * 64,
    champion_model_family=ModelFamily.LOGISTIC_REGRESSION,
    category_head_frozen=False,
    training_run_count=4,
    validation_selection_id="selection-1",
    prediction_manifest_fingerprint="b" * 64,
    test_evaluation_record_id="record-1",
    test_evaluation_status="completed",
    fusion_selection_fingerprint="c" * 64,
    fusion_declared_candidates=tuple(FusionStrategy),
    fusion_selected_strategy=FusionStrategy.OR_GATE,
    novel_holdout_row_count=0,
    explanation_id="explanation-1",
    explanation_status="exact",
    reference_profile_id="profile-1",
    reference_split=MLSplit.TRAIN,
    drift_run_id="drift-1",
    package_version=__version__,
)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_a_report_with_no_evidence_fabricates_no_pass() -> None:
    """Every artifact requirement must be inconclusive, not quietly passed."""
    report = build_acceptance_report(package_version=__version__)
    artifact = [
        item
        for item in report.requirements
        if item.status is AcceptanceStatus.INCONCLUSIVE
    ]
    assert artifact
    assert all("supplied" in item.evidence for item in artifact)
    assert report.failed == 0


def test_an_inconclusive_report_is_not_accepted() -> None:
    """Acceptance on the absence of a failure would accept anything unchecked."""
    report = build_acceptance_report(package_version=__version__)
    assert report.inconclusive > 0
    assert report.accepted is False


def test_complete_evidence_resolves_every_requirement() -> None:
    """Nothing may stay inconclusive once the pipeline has actually run."""
    report = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    unresolved = [
        item
        for item in report.requirements
        if item.status is AcceptanceStatus.INCONCLUSIVE
    ]
    assert unresolved == [], [item.requirement_id for item in unresolved]
    assert report.failed == 0
    assert report.accepted is True


def test_the_structural_requirements_hold_on_this_build() -> None:
    """The contracts this layer claims are the contracts it has."""
    report = build_acceptance_report(package_version=__version__)
    failures = [
        item for item in report.requirements if item.status is AcceptanceStatus.FAIL
    ]
    assert failures == [], [item.requirement_id for item in failures]


def test_the_label_reader_allowlist_is_still_exactly_two_modules() -> None:
    """Widening it would be the single most consequential change in the layer."""
    assert {"detection.evaluation", "ml.dataset"} == LABEL_READER_ALLOWLIST


def test_the_ml_layer_imports_no_object_serializer() -> None:
    """Checked by parsing imports, because prose in this module names them all."""
    from pathlib import Path

    import password_attack_detector.ml as package

    for path in sorted(Path(package.__file__).parent.rglob("*.py")):
        roots = {name.split(".")[0] for name in module_imports(path)}
        assert not roots & {"pickle", "dill", "joblib"}, path.name


def test_module_imports_reads_imports_not_prose(tmp_path: object) -> None:
    """A docstring naming ``pickle`` is not an import of it.

    The distinction is load-bearing: this module's own documentation names every
    serializer it forbids, and a substring check would report the governance
    module as the violation.
    """
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    source = tmp_path / "sample.py"
    source.write_text('"""Mentions pickle and joblib."""\nimport json\n')
    assert module_imports(source) == frozenset({"json"})


# ---------------------------------------------------------------------------
# Not-applicable is not a pass, and not a failure
# ---------------------------------------------------------------------------


def test_an_unfrozen_category_head_is_not_applicable() -> None:
    """A triage head is optional by contract; its absence accepts nothing."""
    report = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    head = next(
        item
        for item in report.requirements
        if item.requirement_id == "P5-M7-CATEGORY-HEAD"
    )
    assert head.status is AcceptanceStatus.NOT_APPLICABLE
    assert "fabricated" in head.evidence


def test_a_frozen_category_head_is_a_pass() -> None:
    """When one exists, the requirement is a real one and is met."""
    report = build_acceptance_report(
        package_version=__version__,
        evidence=replace(COMPLETE, category_head_frozen=True),
    )
    head = next(
        item
        for item in report.requirements
        if item.requirement_id == "P5-M7-CATEGORY-HEAD"
    )
    assert head.status is AcceptanceStatus.PASS


def test_an_empty_holdout_is_not_applicable_rather_than_a_failure() -> None:
    """The experimental track had nothing to evaluate, and said so."""
    report = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    holdout = next(
        item
        for item in report.requirements
        if item.requirement_id == "P5-M9-NOVEL-HOLDOUT-EXPERIMENTAL"
    )
    assert holdout.status is AcceptanceStatus.NOT_APPLICABLE


def test_a_populated_holdout_is_a_pass() -> None:
    """Rows on the experimental track make the requirement a real one."""
    report = build_acceptance_report(
        package_version=__version__,
        evidence=replace(COMPLETE, novel_holdout_row_count=8),
    )
    holdout = next(
        item
        for item in report.requirements
        if item.requirement_id == "P5-M9-NOVEL-HOLDOUT-EXPERIMENTAL"
    )
    assert holdout.status is AcceptanceStatus.PASS


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_a_partial_fusion_universe_fails_rather_than_passing_quietly() -> None:
    """Offering two of three candidates is not selecting from the universe."""
    report = build_acceptance_report(
        package_version=__version__,
        evidence=replace(
            COMPLETE,
            fusion_declared_candidates=(
                FusionStrategy.OR_GATE,
                FusionStrategy.AND_GATE,
            ),
        ),
    )
    fusion = next(
        item
        for item in report.requirements
        if item.requirement_id == "P5-M9-FUSION-SELECTED-ON-VALIDATION"
    )
    assert fusion.status is AcceptanceStatus.FAIL
    assert report.accepted is False


def test_which_strategy_won_is_not_an_acceptance_criterion() -> None:
    """Every declared strategy must pass the requirement equally.

    Turning the outcome into a criterion would make the criterion a reason to
    prefer one, which is exactly what validation-only selection exists to avoid.
    """
    statuses = set()
    for strategy in FusionStrategy:
        report = build_acceptance_report(
            package_version=__version__,
            evidence=replace(COMPLETE, fusion_selected_strategy=strategy),
        )
        statuses.add(
            next(
                item.status
                for item in report.requirements
                if item.requirement_id == "P5-M9-FUSION-SELECTED-ON-VALIDATION"
            )
        )
    assert statuses == {AcceptanceStatus.PASS}


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_the_report_is_deterministic() -> None:
    """Two builds from the same evidence must agree byte for byte."""
    first = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    second = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    assert first.to_json() == second.to_json()


def test_the_report_identity_moves_with_the_evidence() -> None:
    """A different pipeline outcome is a different report."""
    first = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    second = build_acceptance_report(
        package_version=__version__,
        evidence=replace(COMPLETE, novel_holdout_row_count=8),
    )
    assert first.acceptance_report_fingerprint != second.acceptance_report_fingerprint


def test_requirements_are_recorded_once_and_in_identifier_order() -> None:
    """A report a reader can scan, and one a diff can compare."""
    report = build_acceptance_report(package_version=__version__)
    identifiers = [item.requirement_id for item in report.requirements]
    assert identifiers == sorted(identifiers)
    assert len(set(identifiers)) == len(identifiers)


def test_the_tallies_must_match_the_requirements() -> None:
    """A hand-set count would let a report claim a standing it does not have."""
    report = build_acceptance_report(package_version=__version__)
    payload = report.to_dict()
    payload["passed"] = payload["passed"] + 1
    with pytest.raises(Exception, match="not valid"):
        Phase5AcceptanceReport.from_dict(payload)


def test_a_requirement_without_evidence_is_refused() -> None:
    """A status nobody justified is a status nobody can check."""
    with pytest.raises(ValueError, match="must not be empty"):
        AcceptanceRequirement(
            requirement_id="X",
            milestone="M1",
            title="Something",
            status=AcceptanceStatus.PASS,
            evidence="   ",
        )


def test_the_report_declares_no_prohibited_field() -> None:
    """A governance document carries counts and declared names only."""
    for model in (AcceptanceRequirement, Phase5AcceptanceReport):
        assert not set(model.model_fields) & PROHIBITED_METADATA_FIELDS


def test_the_report_carries_no_performance_figure() -> None:
    """It says whether the evaluation happened, never how it came out.

    Swept over the prose rather than over the whole payload: a SHA-256 digest is
    hexadecimal, so a naive substring search finds ``f1`` in roughly every
    fingerprint the report legitimately carries.
    """
    report = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    prose = " ".join(
        f"{item.title} {item.evidence}" for item in report.requirements
    ).lower()
    for token in ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier"):
        assert token not in prose, token


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_rendered_report_explains_what_acceptance_does_not_mean() -> None:
    """A table of passes reads as an endorsement unless it says otherwise."""
    report = build_acceptance_report(package_version=__version__, evidence=COMPLETE)
    lowered = acceptance_report_to_markdown(report).lower()
    assert "synthetic" in lowered
    assert "not a production system" in lowered
    assert "concerns a **contract**, not an outcome" in lowered


def test_the_rendered_report_is_deterministic() -> None:
    """Documentation that differs between runs is documentation nobody can diff."""
    report = build_acceptance_report(package_version=__version__)
    assert acceptance_report_to_markdown(report) == acceptance_report_to_markdown(
        report
    )


# ---------------------------------------------------------------------------
# The model card
# ---------------------------------------------------------------------------


def test_the_model_card_is_deterministic() -> None:
    """It is generated, so two renderings must agree exactly."""
    assert model_card_to_markdown() == model_card_to_markdown()


def test_the_model_card_prints_the_vocabularies_it_describes() -> None:
    """Read from the enums rather than transcribed beside them.

    A card that named four model families when the catalog declared five would
    be wrong in a way no reviewer would notice.
    """
    card = model_card_to_markdown()
    for family in ModelFamily:
        assert f"`{family!s}`" in card, family
    for kind in ScoreKind:
        assert f"`{kind!s}`" in card, kind
    for split in MLSplit:
        assert f"`{split!s}`" in card, split
    for strategy in FusionStrategy:
        assert f"`{strategy!s}`" in card, strategy
    for method in ExplanationMethod:
        assert f"`{method!s}`" in card, method
    for status in DriftStatus:
        assert f"`{status!s}`" in card, status


def test_the_model_card_states_its_prohibited_uses() -> None:
    """The section most likely to be quoted out of context is the one to pin."""
    lowered = model_card_to_markdown().lower()
    assert "## prohibited use" in lowered
    assert "automated account action" in lowered
    assert "evidence about an individual" in lowered
    assert "production control" in lowered


def test_the_model_card_refuses_to_present_synthetic_metrics_as_efficacy() -> None:
    """The claim the card must never make, asserted as text."""
    lowered = model_card_to_markdown().lower()
    assert "not evidence of real-world efficacy" in lowered
    assert "synthetic traffic generated by this repository" in lowered


def test_the_model_card_states_the_explanation_and_drift_limitations() -> None:
    """Both capabilities are easy to over-read, so both say what they are not."""
    lowered = model_card_to_markdown().lower()
    assert "descriptive, not causal" in lowered
    assert "monitoring evidence, not model correctness" in lowered
    assert "nothing retrains" in lowered


def test_the_model_card_carries_no_identifier_or_pseudonym() -> None:
    """A card describes a system, not a run of it.

    The pseudonym check uses the shape the feature layer actually emits rather
    than a bare prefix, because ``s:`` is also how the card writes "Splits:".
    """
    import re

    card = model_card_to_markdown()
    assert not re.search(r"\b(?:u|s|d|sess):[0-9a-f]{32}\b", card)
    assert not re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", card
    )
    assert not re.search(r"\b[0-9a-f]{64}\b", card)
