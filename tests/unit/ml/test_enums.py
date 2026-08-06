"""Tests for the ML layer's domain enumerations.

The enumerations are the layer's vocabulary, and several of them carry a
guarantee expressed by a *missing* member rather than a present one.  Those
absences are what these tests protect: an enum that quietly grows a ``TEST``
member would silently reopen the discipline the type system is enforcing.
"""

from __future__ import annotations

from enum import StrEnum

import pytest

from password_attack_detector.ml.enums import (
    PROBABILITY_SCORE_KINDS,
    SUPERVISED_TASKS,
    UNKNOWN_CATEGORY,
    CalibrationMethod,
    ChampionStatus,
    ExperimentRecordType,
    FusionStrategy,
    GateStatus,
    HyperparameterKind,
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
    ScoreKind,
    ThresholdObjective,
    ValidationPartition,
    is_probability,
)

ALL_ENUMS = (
    CalibrationMethod,
    ChampionStatus,
    ExperimentRecordType,
    FusionStrategy,
    GateStatus,
    HyperparameterKind,
    MLTask,
    ModelEligibilityStatus,
    ModelFamily,
    ScoreKind,
    ThresholdObjective,
    ValidationPartition,
)


@pytest.mark.parametrize("enum_type", ALL_ENUMS)
def test_every_member_serialises_to_its_string_value(
    enum_type: type[StrEnum],
) -> None:
    """StrEnum members must round-trip through their string value."""
    for member in enum_type:
        assert str(member) == member.value
        assert enum_type(member.value) is member


@pytest.mark.parametrize("enum_type", ALL_ENUMS)
def test_no_enum_repeats_a_value(enum_type: type[StrEnum]) -> None:
    """A repeated value would silently alias two members."""
    values = [member.value for member in enum_type]
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("enum_type", ALL_ENUMS)
def test_every_value_is_lower_snake_case(enum_type: type[StrEnum]) -> None:
    """Values reach Parquet columns and JSON reports; casing is a contract."""
    for member in enum_type:
        assert member.value == member.value.lower()
        assert " " not in member.value
        assert "-" not in member.value


# ---------------------------------------------------------------------------
# Guarantees expressed as absences
# ---------------------------------------------------------------------------


def test_validation_partition_cannot_name_the_test_split() -> None:
    """The enum has no test member, so no configuration can name one.

    This is the structural enforcement behind "thresholds are never tuned on
    test". A configuration field typed as this enum has no way to express the
    forbidden choice.
    """
    values = {member.value for member in ValidationPartition}
    assert values == {"validation_a", "validation_b"}
    for forbidden in ("test", "novel_anomaly_holdout", "holdout", "train"):
        assert forbidden not in values


def test_only_the_calibrated_kind_may_be_called_a_probability() -> None:
    """Exactly one score kind is a probability, and the helper agrees."""
    assert {ScoreKind.CALIBRATED_PROBABILITY} == PROBABILITY_SCORE_KINDS
    assert is_probability(ScoreKind.CALIBRATED_PROBABILITY)
    for kind in (
        ScoreKind.DECISION_SCORE,
        ScoreKind.CLASS_SCORE,
        ScoreKind.ANOMALY_SCORE,
    ):
        assert not is_probability(kind)


def test_no_score_kind_other_than_the_calibrated_one_says_probability() -> None:
    """A raw output must not carry the word in its own value."""
    for kind in ScoreKind:
        if is_probability(kind):
            continue
        assert "probability" not in kind.value
        assert "likelihood" not in kind.value
        assert "confidence" not in kind.value


def test_the_anomaly_task_is_not_supervised() -> None:
    """The anomaly probe is fitted without reading a label."""
    assert {MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY} == SUPERVISED_TASKS
    assert MLTask.ANOMALY not in SUPERVISED_TASKS


def test_champion_status_distinguishes_a_negative_from_an_unmeasurable() -> None:
    """ "No candidate passed" and "we could not tell" are different answers."""
    required = {
        ChampionStatus.ELIGIBLE,
        ChampionStatus.NO_ELIGIBLE_CHAMPION,
        ChampionStatus.INSUFFICIENT_VALIDATION_SUPPORT,
    }
    assert required <= set(ChampionStatus)
    assert len({member.value for member in required}) == 3


def test_gate_status_has_exactly_three_outcomes() -> None:
    """Inconclusive is a first-class outcome, not an absent pass."""
    assert {member.value for member in GateStatus} == {
        "pass",
        "fail",
        "inconclusive",
    }


def test_the_four_experiment_record_types_are_declared() -> None:
    """The ledger's shape is fixed before any of it is written."""
    assert {member.value for member in ExperimentRecordType} == {
        "training_run",
        "validation_selection",
        "champion_freeze",
        "test_evaluation",
    }


def test_the_unknown_category_sentinel_is_not_a_known_scenario() -> None:
    """Abstention must not collide with a real class name.

    The category head's class space is derived from ``ScenarioType``; if the
    sentinel collided with one of its members, an abstention would be
    indistinguishable from a confident prediction.
    """
    from password_attack_detector.data.enums import ScenarioType

    assert UNKNOWN_CATEGORY == "unknown"
    assert UNKNOWN_CATEGORY not in {scenario.value for scenario in ScenarioType}


def test_hyperparameter_kinds_exclude_durations() -> None:
    """A hyperparameter is never a window, unlike a rule parameter."""
    assert {member.value for member in HyperparameterKind} == {
        "bool",
        "int",
        "float",
        "string",
    }


def test_fusion_declares_exactly_the_three_planned_strategies() -> None:
    """All three are built and compared; none is assumed to win."""
    assert {member.value for member in FusionStrategy} == {
        "or_gate",
        "and_gate",
        "stacked",
    }


def test_threshold_objectives_are_all_constraint_shaped() -> None:
    """Each objective names what it optimises and what it holds fixed."""
    assert ThresholdObjective.MAX_RECALL_AT_MAX_FPR in ThresholdObjective
    assert ThresholdObjective.MIN_FPR_AT_MIN_RECALL in ThresholdObjective
    assert ThresholdObjective.MAX_F1 in ThresholdObjective


def test_calibration_none_is_an_ordinary_choice() -> None:
    """Shipping uncalibrated is valid; it just forbids the word probability."""
    assert CalibrationMethod.NONE in CalibrationMethod
    assert {member.value for member in CalibrationMethod} == {
        "none",
        "platt",
        "isotonic",
    }


def test_model_families_name_methods_not_verdicts() -> None:
    """A family describes an algorithm, never a conclusion about traffic."""
    verdict_terms = {
        "attack",
        "malicious",
        "threat",
        "detector",
        "suspicious",
        "anomalous",
    }
    for family in ModelFamily:
        tokens = set(family.value.split("_"))
        assert not tokens & verdict_terms, family


def test_eligibility_statuses_cover_every_reason_a_model_is_not_promotable() -> None:
    """Each non-promotable reason is nameable and distinct."""
    assert {member.value for member in ModelEligibilityStatus} == {
        "champion_eligible",
        "serializer_unproven",
        "experimental",
        "anomaly_only",
    }
