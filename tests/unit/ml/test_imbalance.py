"""Class weights: the formula, its provenance, and everything it refuses to guess.

Two properties recur. **Weights are a function of training counts and nothing
else** -- not of row order, not of the validation split, not of what an
estimator would have done by default. And **an absent class is a failure, never
a default**: a class with no training rows cannot be balanced, and a fixed
mapping that forgets one cannot be completed on its behalf.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.config import ImbalanceConfig
from password_attack_detector.ml.enums import MLTask
from password_attack_detector.ml.imbalance import (
    BINARY_CLASS_ORDER,
    IMBALANCE_SCHEMA_VERSION,
    ClassWeightState,
    compute_class_weights,
)

CATEGORY_ORDER = ("brute_force", "credential_stuffing", "password_spraying")


def balanced() -> ImbalanceConfig:
    """Return the shipped default policy."""
    return ImbalanceConfig()


def none_policy() -> ImbalanceConfig:
    """Return the policy that weights nothing."""
    return ImbalanceConfig(class_weight_policy="none")


def fixed(**weights: float) -> ImbalanceConfig:
    """Return a fixed-weight policy over *weights*."""
    return ImbalanceConfig(
        class_weight_policy="fixed",
        fixed_class_weights=tuple(sorted(weights.items())),
    )


def binary_labels(benign: int, malicious: int) -> list[str]:
    """Return a training label column with the requested support."""
    return ["benign"] * benign + ["malicious"] * malicious


def weights_for(
    labels: list[str],
    *,
    config: ImbalanceConfig | None = None,
    task: MLTask = MLTask.BINARY_MALICIOUS,
    order: tuple[str, ...] = BINARY_CLASS_ORDER,
) -> ClassWeightState:
    """Return the state for *labels* under *config*."""
    return compute_class_weights(
        labels, task=task, class_order=order, config=config or balanced()
    )


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_the_none_policy_weights_every_class_equally() -> None:
    """Nothing is done, and the counts are still recorded so it is visible."""
    state = weights_for(binary_labels(90, 10), config=none_policy())
    assert state.weights == (1.0, 1.0)
    assert state.train_row_counts == (90, 10)
    assert state.policy == "none"


def test_balanced_binary_weights_follow_the_declared_formula() -> None:
    """``n_samples / (n_classes * count)``, computed by hand here.

    100 rows, 2 classes: benign gets 100/(2*90) and malicious 100/(2*10). The
    expected values are written out rather than recomputed with the same
    expression the implementation uses, so a change to the formula fails here
    instead of agreeing with itself.
    """
    state = weights_for(binary_labels(90, 10))
    assert state.weight_of("benign") == pytest.approx(0.555555556)
    assert state.weight_of("malicious") == pytest.approx(5.0)


def test_a_balanced_weight_is_one_when_the_classes_are_even() -> None:
    """The formula degenerates correctly rather than being special-cased."""
    state = weights_for(binary_labels(25, 25))
    assert state.weights == (1.0, 1.0)


def test_balanced_multiclass_weights_follow_the_same_formula() -> None:
    """Three classes, 120 rows: 120/(3*count) each."""
    labels = ["brute_force"] * 60 + ["credential_stuffing"] * 40
    labels += ["password_spraying"] * 20
    state = weights_for(labels, task=MLTask.ATTACK_CATEGORY, order=CATEGORY_ORDER)
    assert state.weight_of("brute_force") == pytest.approx(2 / 3)
    assert state.weight_of("credential_stuffing") == pytest.approx(1.0)
    assert state.weight_of("password_spraying") == pytest.approx(2.0)


def test_fixed_weights_are_taken_from_the_configuration() -> None:
    """A reviewed constant is used verbatim, and the counts are still recorded."""
    state = weights_for(binary_labels(90, 10), config=fixed(benign=1.0, malicious=4.0))
    assert state.as_mapping() == {"benign": 1.0, "malicious": 4.0}
    assert state.train_row_counts == (90, 10)
    assert state.policy == "fixed"


def test_fixed_weights_ignore_the_prevalence_they_override() -> None:
    """The point of the policy: two very different splits give the same weights."""
    config = fixed(benign=1.0, malicious=4.0)
    lopsided = weights_for(binary_labels(990, 10), config=config)
    even = weights_for(binary_labels(50, 50), config=config)
    assert lopsided.weights == even.weights
    assert lopsided.train_row_counts != even.train_row_counts


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


def test_a_class_with_no_training_rows_fails_rather_than_dividing_by_zero() -> None:
    """A class the training split never saw is a split problem, not a weight."""
    with pytest.raises(ModelTrainingError, match="no training rows"):
        weights_for(binary_labels(40, 0))


def test_an_empty_label_column_fails() -> None:
    """Balanced weights are a ratio of counts, and there is nothing to count."""
    with pytest.raises(ModelTrainingError, match="no training rows"):
        weights_for([])


def test_an_empty_label_column_is_fine_under_a_fixed_policy() -> None:
    """Fixed weights do not read the counts, so they do not need any."""
    state = weights_for([], config=fixed(benign=1.0, malicious=3.0))
    assert state.weights == (1.0, 3.0)
    assert state.train_row_counts == (0, 0)


def test_a_class_value_outside_the_declared_order_fails() -> None:
    """A class nobody declared cannot be weighted, and is not silently dropped."""
    with pytest.raises(ModelTrainingError, match="not one of"):
        weights_for([*binary_labels(4, 4), "suspicious"])


def test_an_empty_class_order_fails() -> None:
    """Weights have to be reported in some deterministic arrangement."""
    with pytest.raises(ModelTrainingError, match="no class order"):
        weights_for(binary_labels(4, 4), order=())


def test_a_repeated_class_in_the_order_fails() -> None:
    """A duplicated class would give one class two positions in the vector."""
    with pytest.raises(ModelTrainingError, match="repeats a class"):
        weights_for(binary_labels(4, 4), order=("benign", "benign"))


def test_a_fixed_mapping_naming_an_unknown_class_fails() -> None:
    """A weight for a class that does not exist is a typo, not a policy."""
    config = fixed(benign=1.0, malicious=2.0, suspicious=3.0)
    with pytest.raises(ModelTrainingError, match="does not declare"):
        weights_for(binary_labels(4, 4), config=config)


def test_a_fixed_mapping_missing_a_required_class_fails() -> None:
    """Defaulting the class somebody forgot is the silent behaviour to avoid."""
    config = fixed(benign=1.0)
    with pytest.raises(ModelTrainingError, match="does not name"):
        weights_for(binary_labels(4, 4), config=config)


@pytest.mark.parametrize("weight", [0.0, -1.0, -0.5])
def test_a_non_positive_fixed_weight_is_refused_by_the_configuration(
    weight: float,
) -> None:
    """Zero silences a class and a negative weight inverts it; neither is a policy."""
    with pytest.raises(ValueError, match="finite and strictly positive"):
        fixed(benign=1.0, malicious=weight)


def test_a_fixed_policy_without_weights_is_refused() -> None:
    """There is no implicit default to fall back to."""
    with pytest.raises(ValueError, match="requires fixed_class_weights"):
        ImbalanceConfig(class_weight_policy="fixed")


def test_weights_configured_under_another_policy_are_refused() -> None:
    """Weights that are never applied read as though they are."""
    with pytest.raises(ValueError, match="never applied"):
        ImbalanceConfig(
            class_weight_policy="balanced", fixed_class_weights=(("benign", 1.0),)
        )


def test_resampling_cannot_be_switched_on() -> None:
    """The configuration admits one value, so no run can synthesise rows."""
    with pytest.raises(ValueError):
        ImbalanceConfig(resampling="smote")  # type: ignore[arg-type]


def test_asking_for_an_undeclared_class_weight_fails() -> None:
    """Reading a weight for a class the state does not carry is an error."""
    state = weights_for(binary_labels(4, 4))
    with pytest.raises(ModelTrainingError, match="not one of"):
        state.weight_of("suspicious")


# ---------------------------------------------------------------------------
# Determinism and provenance
# ---------------------------------------------------------------------------


def test_the_class_order_is_the_one_that_was_declared() -> None:
    """Reported in the declared arrangement, not in the order rows arrived."""
    state = weights_for(["malicious", "benign", "malicious", "benign"])
    assert state.class_order == BINARY_CLASS_ORDER
    assert [item.class_name for item in state.supports] == list(BINARY_CLASS_ORDER)


@pytest.mark.parametrize("seed", [1, 7, 42, 20260809])
def test_row_order_does_not_affect_the_weights(seed: int) -> None:
    """Counted, not scanned, so a permutation changes nothing at all."""
    labels = binary_labels(30, 7)
    baseline = weights_for(labels)
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    assert weights_for(shuffled).to_json() == baseline.to_json()
    assert weights_for(shuffled).fingerprint() == baseline.fingerprint()


def test_the_state_records_where_the_counts_came_from() -> None:
    """Provenance travels with the number, so a later reader can check it."""
    state = weights_for(binary_labels(10, 5))
    assert state.computed_from == "train"
    assert state.task is MLTask.BINARY_MALICIOUS


def test_the_state_records_the_configuration_it_was_computed_under() -> None:
    """Two policies over identical counts are distinguishable afterwards."""
    counts = binary_labels(10, 5)
    first = weights_for(counts)
    second = weights_for(counts, config=none_policy())
    assert first.imbalance_config_fingerprint != second.imbalance_config_fingerprint
    assert first.fingerprint() != second.fingerprint()


def test_different_training_counts_change_the_fingerprint() -> None:
    """Identity follows the counts, so a different training split is a different state."""
    assert (
        weights_for(binary_labels(10, 5)).fingerprint()
        != weights_for(binary_labels(11, 5)).fingerprint()
    )


def test_the_same_counts_give_the_same_fingerprint() -> None:
    """And identical inputs agree exactly."""
    assert (
        weights_for(binary_labels(10, 5)).fingerprint()
        == weights_for(binary_labels(10, 5)).fingerprint()
    )


def test_a_different_task_changes_the_fingerprint() -> None:
    """The same numbers for a different head are not the same state."""
    labels = ["brute_force"] * 6 + ["credential_stuffing"] * 6
    labels += ["password_spraying"] * 6
    primary = compute_class_weights(
        ["benign"] * 9 + ["malicious"] * 9,
        task=MLTask.BINARY_MALICIOUS,
        class_order=BINARY_CLASS_ORDER,
        config=balanced(),
    )
    triage = compute_class_weights(
        labels,
        task=MLTask.ATTACK_CATEGORY,
        class_order=CATEGORY_ORDER,
        config=balanced(),
    )
    assert primary.fingerprint() != triage.fingerprint()


def test_only_training_counts_reach_the_state() -> None:
    """Validation and test rows are the caller's business, and never arrive here.

    The signature takes one label column, so there is no parameter a validation
    count could enter through. What this pins is the consequence: passing the
    training column alone and passing it alongside a differently-distributed
    validation column are not the same call, and only the first is possible.
    """
    train_only = weights_for(binary_labels(90, 10))
    if_validation_leaked = weights_for(binary_labels(90 + 500, 10 + 1))
    assert train_only.fingerprint() != if_validation_leaked.fingerprint()
    assert train_only.train_row_counts == (90, 10)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_the_json_round_trip_is_byte_identical() -> None:
    """Deserialize then serialize returns exactly the same bytes."""
    state = weights_for(binary_labels(90, 10))
    text = state.to_json()
    assert ClassWeightState.from_json(text).to_json() == text


def test_the_round_trip_preserves_the_weights_exactly() -> None:
    """Weights are quantized when computed, so a reload cannot drift."""
    state = weights_for(binary_labels(90, 10))
    reloaded = ClassWeightState.from_json(state.to_json())
    assert reloaded.weights == state.weights
    assert reloaded.fingerprint() == state.fingerprint()


def test_the_canonical_json_has_sorted_keys() -> None:
    """Key order is fixed, so two renderings of one state cannot differ."""
    payload = json.loads(weights_for(binary_labels(4, 4)).to_json())
    assert list(payload) == sorted(payload)


def test_an_unknown_schema_version_is_refused() -> None:
    """A state from a contract this build does not implement is not loaded."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    payload["imbalance_schema_version"] = "9.9.9"
    with pytest.raises(ModelTrainingError, match="schema version"):
        ClassWeightState.from_dict(payload)


def test_an_unknown_field_is_refused() -> None:
    """Loading loosely would drop what it did not understand."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    payload["computed_on_a_tuesday"] = True
    with pytest.raises(ModelTrainingError, match="not valid"):
        ClassWeightState.from_dict(payload)


def test_a_missing_field_is_refused() -> None:
    """Half a contract is not a contract."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    del payload["class_order"]
    with pytest.raises(ModelTrainingError, match="not valid"):
        ClassWeightState.from_dict(payload)


@pytest.mark.parametrize("text", ["", "{", "[]", "null", '"a string"'])
def test_malformed_json_is_refused(text: str) -> None:
    """Neither invalid JSON nor valid JSON of the wrong shape is accepted."""
    with pytest.raises(ModelTrainingError):
        ClassWeightState.from_json(text)


def test_supports_out_of_class_order_are_refused() -> None:
    """A weight vector whose entries do not match the declared order is unusable."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    payload["supports"] = list(reversed(payload["supports"]))
    with pytest.raises(ModelTrainingError, match="not valid"):
        ClassWeightState.from_dict(payload)


def test_a_non_positive_weight_in_a_payload_is_refused() -> None:
    """The invariant is enforced on load, not only on the path that computes it."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    payload["supports"][0]["weight"] = 0.0
    with pytest.raises(ModelTrainingError, match="not valid"):
        ClassWeightState.from_dict(payload)


def test_the_schema_version_is_declared() -> None:
    """State says which contract it was written against."""
    assert (
        weights_for(binary_labels(4, 4)).imbalance_schema_version
        == IMBALANCE_SCHEMA_VERSION
    )


def test_the_state_is_frozen() -> None:
    """Nothing downstream can edit a weight after the fact."""
    state = weights_for(binary_labels(4, 4))
    with pytest.raises(Exception, match=r"frozen|immutable"):
        state.policy = "none"


PROHIBITED_SUBSTRINGS = (
    "anchor_event_id",
    "campaign_id",
    "usr_",
    "src_",
    "password",
    "token",
    "credential",
    "/home/",
)


def test_the_serialized_state_carries_nothing_prohibited() -> None:
    """Counts and weights only: no identifier, no secret, no path."""
    text = weights_for(binary_labels(90, 10)).to_json()
    for token in PROHIBITED_SUBSTRINGS:
        assert token not in text, token


def test_the_state_carries_no_timestamp() -> None:
    """Identity is semantic, not the moment the counting happened."""
    text = weights_for(binary_labels(90, 10)).to_json()
    assert "created_at" not in text
    assert "computed_at" not in text


def test_the_mapping_form_matches_the_vector_form() -> None:
    """Two views of one state, and they cannot disagree."""
    state = weights_for(binary_labels(90, 10))
    mapping: dict[str, Any] = state.as_mapping()
    assert tuple(mapping[name] for name in state.class_order) == state.weights


def test_a_payload_with_an_empty_class_order_is_refused() -> None:
    """The order invariant is checked on load, not only where weights are computed."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    payload["class_order"] = []
    payload["supports"] = []
    with pytest.raises(ModelTrainingError, match="not valid"):
        ClassWeightState.from_dict(payload)


def test_a_payload_repeating_a_class_is_refused() -> None:
    """One class cannot occupy two positions in the weight vector."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    payload["class_order"] = ["benign", "benign"]
    payload["supports"][1]["class_name"] = "benign"
    with pytest.raises(ModelTrainingError, match="not valid"):
        ClassWeightState.from_dict(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_weight_in_a_payload_is_refused(value: float) -> None:
    """A weight that is not a number was never computed by this code."""
    payload = weights_for(binary_labels(4, 4)).to_dict()
    payload["supports"][0]["weight"] = value
    with pytest.raises(ModelTrainingError, match="not valid"):
        ClassWeightState.from_dict(payload)
