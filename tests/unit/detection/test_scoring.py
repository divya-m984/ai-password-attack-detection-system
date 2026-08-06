"""Tests for correlation-aware risk scoring.

The scorer's value is a set of stated properties, so most of this module proves
them rather than checking particular numbers: bounded, deterministic,
order-invariant, monotone, and idempotent within a correlation group.  The
remainder pins the severity boundaries at exact equality and at one
representable value on either side, and re-establishes that nothing about a
label, split, campaign, scope, or model can reach a score.
"""

from __future__ import annotations

import inspect
import math
from datetime import timedelta

import pytest

from password_attack_detector.detection import scoring as scoring_module
from password_attack_detector.detection.catalog import RULE_CATALOG
from password_attack_detector.detection.config import (
    DetectionConfig,
    ScoringConfig,
    SeverityThresholds,
)
from password_attack_detector.detection.enums import (
    AttackCategory,
    CorrelationGroup,
    Severity,
)
from password_attack_detector.detection.schemas import RiskAssessment
from password_attack_detector.detection.scoring import (
    SCORING_VERSION,
    ZERO_RISK_SCORE,
    RiskScorer,
    ScoringResult,
)
from tests.unit.detection import factories
from tests.unit.detection.factories import fired_detection

ANCHOR = factories.ANCHOR
WHEN = factories.WHEN

#: The three rules of the credential-guessing group, which is what makes the
#: correlation reducer testable at all.
GUESSING_RULES = ("PAD-BF-001", "PAD-BF-002", "PAD-DBF-001")


@pytest.fixture()
def scorer() -> RiskScorer:
    return RiskScorer(DetectionConfig())


def score_of(scorer: RiskScorer, *detections: object) -> float:
    """Return the risk score for one anchor built from *detections*."""
    return scorer.score_anchor(ANCHOR, WHEN, list(detections)).risk_score  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The zero case
# ---------------------------------------------------------------------------


def test_no_fired_rules_scores_exactly_zero(scorer: RiskScorer) -> None:
    assessment = scorer.score_anchor(ANCHOR, WHEN, [])
    assert assessment.risk_score == ZERO_RISK_SCORE == 0.0
    assert assessment.severity is Severity.LOW
    assert assessment.primary_attack_category is None
    assert assessment.contributing_categories == ()
    assert assessment.fired_rule_count == 0
    assert assessment.top_evidence == ()


def test_a_zero_score_is_reachable_only_with_no_fired_rule(
    scorer: RiskScorer,
) -> None:
    """Both directions, which is what lets a reader trust a zero."""
    weakest = scorer.score_anchor(ANCHOR, WHEN, [fired_detection(signal_strength=1e-9)])
    assert weakest.risk_score > 0.0
    assert weakest.risk_score >= DetectionConfig().scoring.min_fired_risk_score


def test_an_empty_batch_produces_an_empty_result(scorer: RiskScorer) -> None:
    result = scorer.score([])
    assert isinstance(result, ScoringResult)
    assert result.assessments == ()
    assert result.scored_count == 0
    assert result.zero_score_count == 0


def test_a_weak_signal_still_clears_the_fired_floor(scorer: RiskScorer) -> None:
    assessment = scorer.score_anchor(
        ANCHOR, WHEN, [fired_detection(signal_strength=0.01)]
    )
    assert assessment.risk_score == pytest.approx(
        DetectionConfig().scoring.min_fired_risk_score
    )
    assert assessment.severity is Severity.LOW


# ---------------------------------------------------------------------------
# Bounds and finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strength", [1e-12, 0.01, 0.15, 0.5, 0.999, 1.0])
def test_the_score_stays_inside_its_declared_range(
    scorer: RiskScorer, strength: float
) -> None:
    score = score_of(scorer, fired_detection(signal_strength=strength))
    assert 0.0 <= score <= 100.0
    assert not math.isnan(score)
    assert not math.isinf(score)


def test_every_rule_firing_at_full_strength_stays_bounded(
    scorer: RiskScorer,
) -> None:
    """The saturating combiner cannot be pushed past its ceiling."""
    detections = [
        fired_detection(rule_id, signal_strength=1.0)
        for rule_id in sorted(RULE_CATALOG.rule_ids)
    ]
    assessment = scorer.score_anchor(ANCHOR, WHEN, detections)
    assert 0.0 < assessment.risk_score <= 100.0
    assert assessment.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# Correlation-group behaviour
# ---------------------------------------------------------------------------


def test_correlated_rules_do_not_out_score_the_strongest_of_them(
    scorer: RiskScorer,
) -> None:
    """The whole purpose of a correlation group, stated as an equality."""
    strongest = score_of(scorer, fired_detection("PAD-BF-001", signal_strength=0.9))
    together = score_of(
        scorer,
        fired_detection("PAD-BF-001", signal_strength=0.9),
        fired_detection("PAD-BF-002", signal_strength=0.4),
        fired_detection("PAD-DBF-001", signal_strength=0.3),
    )
    assert together == strongest


def test_the_group_reducer_is_idempotent(scorer: RiskScorer) -> None:
    one = score_of(scorer, fired_detection("PAD-BF-001", signal_strength=0.7))
    duplicated = score_of(
        scorer,
        fired_detection("PAD-BF-001", signal_strength=0.7),
        fired_detection("PAD-BF-002", signal_strength=0.7),
    )
    assert duplicated == one


def test_the_group_maximum_governs_regardless_of_weight_differences(
    scorer: RiskScorer,
) -> None:
    """DBF and BF share a family weight, so the stronger signal must win."""
    weak_first = score_of(
        scorer,
        fired_detection("PAD-BF-001", signal_strength=0.2),
        fired_detection("PAD-DBF-001", signal_strength=0.95),
    )
    alone = score_of(scorer, fired_detection("PAD-DBF-001", signal_strength=0.95))
    assert weak_first == alone


def test_unrelated_groups_combine_above_either_alone(scorer: RiskScorer) -> None:
    guessing = score_of(scorer, fired_detection("PAD-BF-001", signal_strength=0.6))
    fanout = score_of(scorer, fired_detection("PAD-PS-001", signal_strength=0.6))
    both = score_of(
        scorer,
        fired_detection("PAD-BF-001", signal_strength=0.6),
        fired_detection("PAD-PS-001", signal_strength=0.6),
    )
    assert both > guessing
    assert both > fanout


def test_an_unrelated_signal_can_never_lower_the_score(scorer: RiskScorer) -> None:
    """Noisy-OR multiplies by a factor at most one, so risk only rises."""
    base = fired_detection("PAD-BF-001", signal_strength=0.8)
    baseline = score_of(scorer, base)
    for rule_id in ("PAD-PS-001", "PAD-GEO-001", "PAD-BOT-001", "PAD-MFA-001"):
        for strength in (0.01, 0.5, 1.0):
            extended = score_of(
                scorer, base, fired_detection(rule_id, signal_strength=strength)
            )
            assert extended >= baseline


def test_the_bounded_sum_reducer_rewards_breadth_within_a_group() -> None:
    scorer = RiskScorer(
        DetectionConfig(scoring=ScoringConfig(correlation_reducer="bounded_sum"))
    )
    one = score_of(scorer, fired_detection("PAD-BF-001", signal_strength=0.5))
    three = score_of(
        scorer,
        *(fired_detection(rule_id, signal_strength=0.5) for rule_id in GUESSING_RULES),
    )
    assert three > one
    assert three <= 100.0


def test_the_bounded_sum_reducer_is_order_invariant() -> None:
    scorer = RiskScorer(
        DetectionConfig(scoring=ScoringConfig(correlation_reducer="bounded_sum"))
    )
    detections = [
        fired_detection("PAD-BF-001", signal_strength=0.31),
        fired_detection("PAD-BF-002", signal_strength=0.72),
        fired_detection("PAD-DBF-001", signal_strength=0.44),
    ]
    assert score_of(scorer, *detections) == score_of(scorer, *reversed(detections))


# ---------------------------------------------------------------------------
# Determinism, ordering, and monotonicity
# ---------------------------------------------------------------------------


def test_the_same_input_scores_identically(scorer: RiskScorer) -> None:
    detections = [
        fired_detection("PAD-BF-001", signal_strength=0.42),
        fired_detection("PAD-PS-001", signal_strength=0.63),
    ]
    first = scorer.score_anchor(ANCHOR, WHEN, detections)
    second = scorer.score_anchor(ANCHOR, WHEN, detections)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_input_order_does_not_change_an_assessment(scorer: RiskScorer) -> None:
    detections = [
        fired_detection("PAD-BF-001", signal_strength=0.42),
        fired_detection("PAD-PS-001", signal_strength=0.63),
        fired_detection("PAD-GEO-001", signal_strength=0.21),
        fired_detection("PAD-MFA-001", signal_strength=0.88),
    ]
    forward = scorer.score_anchor(ANCHOR, WHEN, detections)
    backward = scorer.score_anchor(ANCHOR, WHEN, list(reversed(detections)))
    assert forward == backward


def test_batch_output_is_sorted_by_time_then_anchor(scorer: RiskScorer) -> None:
    detections = [
        fired_detection(anchor_event_id="b", anchor_event_time=WHEN),
        fired_detection(anchor_event_id="a", anchor_event_time=WHEN),
        fired_detection(
            anchor_event_id="c", anchor_event_time=WHEN - timedelta(minutes=1)
        ),
    ]
    result = scorer.score(detections)
    assert [item.anchor_event_id for item in result.assessments] == ["c", "a", "b"]
    assert scorer.score(list(reversed(detections))) == result


@pytest.mark.parametrize(
    ("weaker", "stronger"), [(0.1, 0.2), (0.3, 0.9), (0.5, 1.0), (0.75, 0.751)]
)
def test_stronger_evidence_never_lowers_the_score(
    scorer: RiskScorer, weaker: float, stronger: float
) -> None:
    low = score_of(scorer, fired_detection(signal_strength=weaker))
    high = score_of(scorer, fired_detection(signal_strength=stronger))
    assert high >= low


def test_a_strictly_stronger_signal_raises_the_score(scorer: RiskScorer) -> None:
    assert score_of(scorer, fired_detection(signal_strength=0.9)) > score_of(
        scorer, fired_detection(signal_strength=0.3)
    )


# ---------------------------------------------------------------------------
# Duplicate input
# ---------------------------------------------------------------------------


def test_a_repeated_rule_collapses_to_its_strongest_occurrence(
    scorer: RiskScorer,
) -> None:
    """A caller concatenating batches gets a stable answer, not an exception."""
    assessment = scorer.score_anchor(
        ANCHOR,
        WHEN,
        [
            fired_detection("PAD-BF-001", signal_strength=0.3),
            fired_detection("PAD-BF-001", signal_strength=0.8),
        ],
    )
    assert assessment.fired_rule_ids == ("PAD-BF-001",)
    assert assessment.fired_rule_count == 1
    assert assessment.risk_score == score_of(
        scorer, fired_detection("PAD-BF-001", signal_strength=0.8)
    )


def test_duplicate_collapsing_does_not_depend_on_arrival_order(
    scorer: RiskScorer,
) -> None:
    strong = fired_detection("PAD-BF-001", signal_strength=0.8)
    weak = fired_detection("PAD-BF-001", signal_strength=0.3)
    assert scorer.score_anchor(ANCHOR, WHEN, [strong, weak]) == scorer.score_anchor(
        ANCHOR, WHEN, [weak, strong]
    )


def test_duplicates_are_counted_in_the_batch_statistics(scorer: RiskScorer) -> None:
    result = scorer.score(
        [
            fired_detection("PAD-BF-001", signal_strength=0.3),
            fired_detection("PAD-BF-001", signal_strength=0.8),
        ]
    )
    assert result.stats.duplicate_detection_count == 1
    assert result.stats.scored_detection_count == 1


# ---------------------------------------------------------------------------
# Primary category and contributing metadata
# ---------------------------------------------------------------------------


def test_the_primary_category_is_the_strongest_contribution(
    scorer: RiskScorer,
) -> None:
    assessment = scorer.score_anchor(
        ANCHOR,
        WHEN,
        [
            fired_detection("PAD-BF-001", signal_strength=0.2),
            fired_detection("PAD-PS-001", signal_strength=0.95),
        ],
    )
    assert assessment.primary_attack_category is AttackCategory.PASSWORD_SPRAYING


def test_a_contribution_tie_breaks_on_severity_then_rule_id(
    scorer: RiskScorer,
) -> None:
    """Two rules of one family at one strength contribute identically."""
    assessment = scorer.score_anchor(
        ANCHOR,
        WHEN,
        [
            fired_detection("PAD-PS-001", signal_strength=0.5),
            fired_detection("PAD-CS-001", signal_strength=0.5),
        ],
    )
    # Same family weight, same strength, same severity, so the identifier
    # decides: PAD-CS-001 sorts before PAD-PS-001.
    assert assessment.primary_attack_category is AttackCategory.CREDENTIAL_STUFFING
    assert (
        scorer.score_anchor(
            ANCHOR,
            WHEN,
            [
                fired_detection("PAD-CS-001", signal_strength=0.5),
                fired_detection("PAD-PS-001", signal_strength=0.5),
            ],
        )
        == assessment
    )


def test_a_higher_severity_wins_a_contribution_tie(scorer: RiskScorer) -> None:
    assessment = scorer.score_anchor(
        ANCHOR,
        WHEN,
        [
            fired_detection("PAD-PS-001", signal_strength=0.5, severity=Severity.LOW),
            fired_detection(
                "PAD-CS-001", signal_strength=0.5, severity=Severity.CRITICAL
            ),
        ],
    )
    assert assessment.primary_attack_category is AttackCategory.CREDENTIAL_STUFFING


def test_contributing_metadata_is_deterministically_ordered(
    scorer: RiskScorer,
) -> None:
    assessment = scorer.score_anchor(
        ANCHOR,
        WHEN,
        [
            fired_detection("PAD-PS-001", signal_strength=0.4),
            fired_detection("PAD-BF-001", signal_strength=0.9),
            fired_detection("PAD-GEO-001", signal_strength=0.6),
        ],
    )
    assert assessment.fired_rule_ids == tuple(sorted(assessment.fired_rule_ids))
    assert assessment.contributing_categories == tuple(
        sorted(assessment.contributing_categories)
    )
    assert assessment.primary_attack_category in assessment.contributing_categories


def test_top_evidence_is_capped_by_configuration(scorer: RiskScorer) -> None:
    assessment = scorer.score_anchor(
        ANCHOR,
        WHEN,
        [
            fired_detection("PAD-BF-001", signal_strength=0.9, evidence_count=5),
            fired_detection("PAD-PS-001", signal_strength=0.8, evidence_count=5),
        ],
    )
    assert len(assessment.top_evidence) == DetectionConfig().scoring.top_evidence_count


def test_insufficient_data_counts_are_retained(scorer: RiskScorer) -> None:
    assessment = scorer.score_anchor(
        ANCHOR, WHEN, [fired_detection()], insufficient_data_count=4
    )
    assert assessment.insufficient_data_count == 4


# ---------------------------------------------------------------------------
# Severity boundaries
#
# The mapping is tested twice: directly, and through a real scored assessment
# whose configuration places a boundary exactly on the produced score.  The
# second form is what proves the scorer actually consults the thresholds.
# ---------------------------------------------------------------------------


def test_the_severity_mapping_is_inclusive_from_below() -> None:
    thresholds = SeverityThresholds()
    assert thresholds.severity_for(0.0) is Severity.LOW
    assert thresholds.severity_for(39.9999) is Severity.LOW
    assert thresholds.severity_for(40.0) is Severity.MEDIUM
    assert thresholds.severity_for(64.9999) is Severity.MEDIUM
    assert thresholds.severity_for(65.0) is Severity.HIGH
    assert thresholds.severity_for(84.9999) is Severity.HIGH
    assert thresholds.severity_for(85.0) is Severity.CRITICAL
    assert thresholds.severity_for(100.0) is Severity.CRITICAL


@pytest.mark.parametrize("boundary", ["medium", "high", "critical"])
def test_each_boundary_is_exact_at_equality_and_on_both_sides(
    boundary: str,
) -> None:
    """One representable step decides the band, in both directions."""
    reference = SeverityThresholds()
    probe = getattr(reference, boundary)
    below = {"medium": Severity.LOW, "high": Severity.MEDIUM, "critical": Severity.HIGH}
    at = {
        "medium": Severity.MEDIUM,
        "high": Severity.HIGH,
        "critical": Severity.CRITICAL,
    }

    exact = SeverityThresholds(**{boundary: probe})
    assert exact.severity_for(probe) is at[boundary]
    assert exact.severity_for(math.nextafter(probe, 0.0)) is below[boundary]
    assert exact.severity_for(math.nextafter(probe, math.inf)) is at[boundary]


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        ("medium", Severity.MEDIUM),
        ("high", Severity.HIGH),
        ("critical", Severity.CRITICAL),
    ],
)
def test_a_scored_assessment_lands_on_a_boundary_placed_at_its_own_score(
    boundary: str, expected: Severity
) -> None:
    """Configure the boundary to equal the produced score, then step it."""
    detection = fired_detection("PAD-BF-001", signal_strength=0.55)
    baseline = RiskScorer(DetectionConfig()).score_anchor(ANCHOR, WHEN, [detection])
    score = baseline.risk_score
    ladder = _ladder_at(boundary, score)

    at_boundary = RiskScorer(
        DetectionConfig(severity_thresholds=ladder(score))
    ).score_anchor(ANCHOR, WHEN, [detection])
    assert at_boundary.severity is expected

    just_above_boundary = RiskScorer(
        DetectionConfig(severity_thresholds=ladder(math.nextafter(score, math.inf)))
    ).score_anchor(ANCHOR, WHEN, [detection])
    assert just_above_boundary.severity is not expected

    just_below_boundary = RiskScorer(
        DetectionConfig(severity_thresholds=ladder(math.nextafter(score, 0.0)))
    ).score_anchor(ANCHOR, WHEN, [detection])
    assert just_below_boundary.severity is expected


def _ladder_at(boundary: str, score: float):  # type: ignore[no-untyped-def]
    """Return a builder placing *boundary* at a given score, order preserved."""
    if boundary == "medium":
        return lambda value: SeverityThresholds(
            medium=value, high=max(value + 1.0, 65.0), critical=max(value + 2.0, 85.0)
        )
    if boundary == "high":
        return lambda value: SeverityThresholds(
            medium=min(value - 1.0, 1.0), high=value, critical=max(value + 1.0, 95.0)
        )
    return lambda value: SeverityThresholds(
        medium=min(value - 2.0, 1.0), high=min(value - 1.0, 2.0), critical=value
    )


def test_a_low_severity_assessment_is_an_ordinary_finding(
    scorer: RiskScorer,
) -> None:
    """LOW is a severity, not a suppressed state."""
    assessment = scorer.score_anchor(
        ANCHOR, WHEN, [fired_detection("PAD-BOT-001", signal_strength=0.02)]
    )
    assert assessment.severity is Severity.LOW
    assert assessment.risk_score > 0.0
    assert assessment.fired_rule_count == 1


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


def test_the_assessment_records_the_configuration_fingerprint(
    scorer: RiskScorer,
) -> None:
    assessment = scorer.score_anchor(ANCHOR, WHEN, [fired_detection()])
    assert assessment.configuration_fingerprint == DetectionConfig().fingerprint()
    assert assessment.scoring_version == SCORING_VERSION


def test_a_retuned_configuration_is_distinguishable(scorer: RiskScorer) -> None:
    """Different executions stay apart without destabilising detection IDs."""
    retuned = RiskScorer(
        DetectionConfig(severity_thresholds=SeverityThresholds(medium=20.0))
    )
    baseline = scorer.score_anchor(ANCHOR, WHEN, [fired_detection()])
    changed = retuned.score_anchor(ANCHOR, WHEN, [fired_detection()])
    assert changed.configuration_fingerprint != baseline.configuration_fingerprint


def test_a_zero_score_assessment_also_records_the_fingerprint(
    scorer: RiskScorer,
) -> None:
    assessment = scorer.score_anchor(ANCHOR, WHEN, [])
    assert assessment.configuration_fingerprint == DetectionConfig().fingerprint()


def test_an_assessment_accepts_an_absent_fingerprint() -> None:
    """Hand-built assessments outside a configured run stay constructible."""
    assessment = RiskAssessment(
        anchor_event_id=ANCHOR,
        anchor_event_time=WHEN,
        risk_score=0.0,
        severity=Severity.LOW,
        scoring_version="1.0.0",
    )
    assert assessment.configuration_fingerprint == ""


def test_a_malformed_fingerprint_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256 hex digest"):
        RiskAssessment(
            anchor_event_id=ANCHOR,
            anchor_event_time=WHEN,
            risk_score=0.0,
            severity=Severity.LOW,
            scoring_version="1.0.0",
            configuration_fingerprint="not-a-digest",
        )


# ---------------------------------------------------------------------------
# The detection-time boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["scope", "entity", "label", "split", "campaign", "event_table", "truth", "model"],
)
def test_no_scorer_entry_point_accepts_a_forbidden_argument(forbidden: str) -> None:
    for callable_object in (
        RiskScorer.__init__,
        RiskScorer.score,
        RiskScorer.score_anchor,
        RiskScorer.contributions,
        RiskScorer.combine,
    ):
        parameters = inspect.signature(callable_object).parameters
        assert not any(forbidden in name.lower() for name in parameters)


def test_the_scoring_module_imports_no_label_scope_or_feature_reader() -> None:
    source = inspect.getsource(scoring_module)
    for forbidden in (
        "features.serialization",
        "features.splitting",
        "detection.alerts",
        "detection.evaluation",
        "data.schemas",
    ):
        assert forbidden not in source


def test_the_scoring_module_uses_no_dynamic_execution() -> None:
    source = inspect.getsource(scoring_module)
    for forbidden in ("eval(", "exec(", "__import__", "importlib", "subprocess"):
        assert forbidden not in source


def test_no_scoring_field_or_docstring_calls_the_score_a_probability() -> None:
    """The score orders findings; it does not estimate a likelihood."""
    forbidden = ("probability", "likelihood", "confidence")

    # No field a reader could mistake for an estimate, in any model the scorer
    # produces or consumes.
    for model in (RiskAssessment, ScoringResult, scoring_module.RuleContribution):
        fields = getattr(model, "model_fields", None) or model.__dataclass_fields__
        for name in fields:
            assert not any(term in name.lower() for term in forbidden), name

    # No public symbol either.
    for name in dir(scoring_module):
        assert not any(term in name.lower() for term in forbidden), name

    # Prose may use the words only to deny them.  Sentences are the unit here,
    # not lines: the disclaimer wraps, and a line-based check would flag its
    # own continuation.
    prose = " ".join(inspect.getsource(scoring_module).split())
    for sentence in prose.replace(";", ".").split("."):
        lowered = sentence.lower()
        if any(term in lowered for term in forbidden):
            assert {"no", "not", "never"} & set(lowered.split()), sentence


def test_scoring_is_unaffected_by_anchor_identity(scorer: RiskScorer) -> None:
    """Only the detections decide the score, never which anchor they are on."""
    first = scorer.score_anchor("anchor-a", WHEN, [fired_detection()])
    second = scorer.score_anchor("anchor-b", WHEN, [fired_detection()])
    assert first.risk_score == second.risk_score
    assert first.severity is second.severity


def test_aggregate_statistics_carry_no_identifier(scorer: RiskScorer) -> None:
    result = scorer.score(
        [
            fired_detection(anchor_event_id="anchor-0001"),
            fired_detection("PAD-PS-001", anchor_event_id="anchor-0002"),
        ]
    )
    rendered = repr((result.severity_counts, result.stats, result.zero_score_count))
    assert "anchor-" not in rendered
    assert ANCHOR not in rendered


# ---------------------------------------------------------------------------
# Batch collection from the engine's diagnostic stream
# ---------------------------------------------------------------------------


def test_evaluation_results_supply_anchors_and_history_gaps() -> None:
    from password_attack_detector.detection.engine import DetectionEngine

    catalog = factories.feature_catalog()
    config = DetectionConfig()
    engine = DetectionEngine(config, feature_catalog=catalog)
    rows = [
        factories.brute_force_row(catalog, anchor_event_id="a1"),
        factories.quiet_row(
            catalog, anchor_event_id="a2", anchor_event_time=WHEN + timedelta(minutes=1)
        ),
    ]
    result = RiskScorer(config).score(engine.run_diagnostic(rows))

    assert [item.anchor_event_id for item in result.assessments] == ["a1", "a2"]
    assert result.assessments[0].fired_rule_count == 1
    assert result.assessments[1].fired_rule_count == 0
    assert result.assessments[1].risk_score == 0.0
    assert result.assessments[0].insufficient_data_count > 0


def test_evaluated_anchors_supply_the_benign_denominator(
    scorer: RiskScorer,
) -> None:
    """An anchor nothing reached still gets a zero-score row when declared."""
    result = scorer.score(
        [fired_detection(anchor_event_id="a1")],
        evaluated_anchors={"a1": WHEN, "a2": WHEN + timedelta(minutes=1)},
    )
    assert [item.anchor_event_id for item in result.assessments] == ["a1", "a2"]
    assert result.zero_score_count == 1


def test_severity_counts_cover_every_band(scorer: RiskScorer) -> None:
    result = scorer.score([fired_detection()])
    assert set(result.severity_counts) == {str(item) for item in Severity}
    assert sum(result.severity_counts.values()) == result.scored_count


def test_group_reductions_show_the_reducer_ran(scorer: RiskScorer) -> None:
    """Three correlated rules reduce to one group, not three."""
    result = scorer.score(
        [
            fired_detection(rule_id, signal_strength=0.5, anchor_event_id="a1")
            for rule_id in GUESSING_RULES
        ]
    )
    assert result.stats.scored_detection_count == 3
    assert result.stats.group_reductions == 1


def test_contributions_expose_the_weighting(scorer: RiskScorer) -> None:
    detection = fired_detection("PAD-BF-001", signal_strength=0.5)
    (contribution,) = scorer.contributions([detection])
    assert contribution.weight == pytest.approx(0.90)
    assert contribution.contribution == pytest.approx(0.45)
    assert contribution.correlation_group is (
        CorrelationGroup.CREDENTIAL_GUESSING_SINGLE_TARGET
    )


def test_combining_no_contributions_is_the_zero_score(scorer: RiskScorer) -> None:
    assert scorer.combine([]) == ZERO_RISK_SCORE


def test_a_weaker_duplicate_is_discarded_in_a_batch(scorer: RiskScorer) -> None:
    """The strongest occurrence wins whichever order the repeats arrive in."""
    strong_first = scorer.score(
        [
            fired_detection("PAD-BF-001", signal_strength=0.8),
            fired_detection("PAD-BF-001", signal_strength=0.3),
        ]
    )
    weak_first = scorer.score(
        [
            fired_detection("PAD-BF-001", signal_strength=0.3),
            fired_detection("PAD-BF-001", signal_strength=0.8),
        ]
    )
    assert strong_first.assessments == weak_first.assessments
    assert strong_first.stats.duplicate_detection_count == 1


def test_a_non_finite_combination_is_refused() -> None:
    """A NaN here would be a defect, not something to pin to a bound."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        scoring_module._clamp_unit(math.nan)
    with pytest.raises(ValueError, match="NaN or infinite"):
        scoring_module._clamp_unit(math.inf)
