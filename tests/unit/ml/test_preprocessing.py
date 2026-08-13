"""Train-only preprocessing: what it fits, what it refuses, and what it never learns.

Three properties recur, and most tests here are one of them.

**Fitted state comes from TRAIN and from nowhere else.** The strongest form of
that is behavioural, not structural: change the validation, test, and holdout
rows as violently as the fixtures allow, refit nothing, and assert the state is
byte-identical. A structural check ("no split column is read") can be satisfied
by code that leaks anyway; this cannot.

**Null is not zero.** Every nullable feature carries an indicator, an observed
zero survives untouched, and an unexpected null on a non-nullable feature is a
fault rather than something to average over.

**The vocabulary is frozen at fit.** A category training never saw is
``__unknown`` forever, a category training saw too rarely is ``__other``
forever, and neither can enlarge the serialized state after the fact.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.config import PreprocessingConfig
from password_attack_detector.ml.dataset import assemble_ml_dataset
from password_attack_detector.ml.enums import MLSplit
from password_attack_detector.ml.features import resolve_eligible_features
from password_attack_detector.ml.preprocessing import (
    MISSING_INDICATOR_SUFFIX,
    PREPROCESSING_SCHEMA_VERSION,
    FittedPreprocessor,
    fit_preprocessor,
)
from tests.ml import factories as fx

ALL_CLASSES = ("prior_only", "current_event_context", "baseline_derived")

#: One planned row: the six raw values, in ``PREPROCESSING_FEATURES`` order.
Row = tuple[Any, Any, Any, Any, Any, Any]


@dataclass(frozen=True, slots=True)
class Anchor:
    """The only two anchor fields preprocessing is allowed to see."""

    anchor_event_id: str
    anchor_event_time: datetime


@dataclass(frozen=True, slots=True)
class Frame:
    """A hand-built frame, for the checks that must bypass dataset assembly.

    ``ml.dataset`` canonicalizes on the way out, so a frame it produced can never
    be out of order. Testing that the low-level fit *rejects* disorder therefore
    needs a frame nobody sorted -- which is exactly what this is.
    """

    split: MLSplit
    feature_names: tuple[str, ...]
    anchors: tuple[Anchor, ...]
    feature_matrix: tuple[tuple[Any, ...], ...]


@pytest.fixture
def catalog() -> Any:
    """Return the six-feature catalog spanning every encoding branch."""
    return fx.preprocessing_catalog()


@pytest.fixture
def eligible(catalog: Any) -> Any:
    """Return the resolved feature contract over that catalog."""
    return resolve_eligible_features(
        catalog, fx.allowlist_for(catalog), include_leakage_classes=ALL_CLASSES
    )


@pytest.fixture
def config() -> PreprocessingConfig:
    """Return a policy sized for hand-written fixtures.

    Only the rare-category *floor* is relaxed, from twenty occurrences to two.
    Every other field keeps its production value, so these tests exercise the
    shipped policy rather than a friendlier one.
    """
    return PreprocessingConfig(min_category_frequency=2)


def frame(rows: list[Row], *, split: MLSplit = MLSplit.TRAIN, start: int = 0) -> Frame:
    """Return a canonically ordered frame over *rows*."""
    return Frame(
        split=split,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=tuple(
            Anchor(fx.anchor_id(start + index), fx.at(start + index))
            for index in range(len(rows))
        ),
        feature_matrix=tuple(tuple(row) for row in rows),
    )


def train_rows(count: int = 8) -> list[Row]:
    """Return a healthy training fixture: no nulls, two categories, both booleans."""
    return [
        (
            float(index),
            index,
            "success" if index % 2 else "failure",
            "us" if index % 2 else "gb",
            index % 3 == 0,
            True,
        )
        for index in range(count)
    ]


def fit(
    rows: list[Row], catalog: Any, eligible: Any, config: PreprocessingConfig
) -> FittedPreprocessor:
    """Fit a preprocessor over *rows*."""
    return fit_preprocessor(
        frame(rows), catalog=catalog, eligible=eligible, config=config
    )


def column(matrix: Any, name: str) -> list[float]:
    """Return one transformed column by name."""
    index = matrix.output_feature_names.index(name)
    return [row[index] for row in matrix.rows]


# ---------------------------------------------------------------------------
# Fit is train-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "split",
    [MLSplit.VALIDATION, MLSplit.TEST, MLSplit.NOVEL_ANOMALY_HOLDOUT, MLSplit.EXCLUDED],
)
def test_fitting_on_any_split_but_train_is_refused(
    split: MLSplit, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The one split a model may be fitted on is the one a preprocessor may be."""
    with pytest.raises(ModelTrainingError, match="fitted on"):
        fit_preprocessor(
            frame(train_rows(), split=split),
            catalog=catalog,
            eligible=eligible,
            config=config,
        )


def test_fitting_on_an_empty_frame_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A constant nobody measured is not an imputation."""
    with pytest.raises(ModelTrainingError, match="no rows"):
        fit([], catalog, eligible, config)


def test_transform_accepts_every_split(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Transforming test rows is the normal case; only *fitting* on them is not."""
    fitted = fit(train_rows(), catalog, eligible, config)
    matrix = fitted.transform(frame(train_rows(4), split=MLSplit.TEST))
    assert matrix.row_count == 4
    assert matrix.column_count == fitted.output_feature_count


# ---------------------------------------------------------------------------
# Numeric: null is not zero
# ---------------------------------------------------------------------------


def test_the_median_is_computed_from_train_only(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Values 0..7 have median 3.5; the transform's fill is that number."""
    fitted = fit(train_rows(8), catalog, eligible, config)
    imputation = next(
        item
        for item in fitted.numeric_imputations
        if item.feature == "user_failure_rate"
    )
    assert imputation.value == pytest.approx(3.5)
    assert imputation.policy == "median"
    assert imputation.train_observed_count == 8
    assert imputation.train_null_count == 0


def test_the_even_length_median_averages_the_two_middle_values(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The tie rule is pinned here, not inherited from whatever library is installed.

    Four observed values 0, 1, 10, 11 average to 5.5 across the middle pair. A
    lower-median convention would give 1.0 and every imputed cell in the
    repository would move.
    """
    rows: list[Row] = [
        (value, index, "success", "us", True, True)
        for index, value in enumerate([0.0, 1.0, 10.0, 11.0])
    ]
    fitted = fit(rows, catalog, eligible, config)
    imputation = next(
        item
        for item in fitted.numeric_imputations
        if item.feature == "user_failure_rate"
    )
    assert imputation.value == pytest.approx(5.5)


def test_a_nullable_numeric_feature_gets_a_missing_indicator(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The indicator is the half of the encoding that survives imputation."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert f"user_failure_rate{MISSING_INDICATOR_SUFFIX}" in fitted.output_feature_names
    assert (
        f"user_attempt_count{MISSING_INDICATOR_SUFFIX}"
        not in fitted.output_feature_names
    )


def test_an_observed_zero_is_not_treated_as_missing(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The distinction Phase 3 draws survives the whole transform.

    Row 0 observed zero. Row 1 observed nothing. Both leave the value channel at
    the same number once imputation has run -- and the indicator is what still
    tells them apart.
    """
    rows: list[Row] = [
        (0.0, 0, "success", "us", True, True),
        (None, 1, "success", "us", True, True),
        (4.0, 2, "success", "us", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    matrix = fitted.transform(frame(rows))
    assert column(matrix, f"user_failure_rate{MISSING_INDICATOR_SUFFIX}") == [
        0.0,
        1.0,
        0.0,
    ]
    unscaled = fit(
        rows,
        catalog,
        eligible,
        config.model_copy(update={"standardize_numeric_for_linear_models": False}),
    )
    values = column(unscaled.transform(frame(rows)), "user_failure_rate")
    assert values[0] == 0.0, "an observed zero must stay zero"
    assert values[1] == pytest.approx(2.0), "a null takes the train median"


def test_an_unexpected_null_on_a_non_nullable_feature_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A fault is reported, not averaged over."""
    rows: list[Row] = [
        (1.0, None, "success", "us", True, True),
        (2.0, 2, "success", "us", True, True),
    ]
    with pytest.raises(ModelTrainingError, match="non-nullable"):
        fit(rows, catalog, eligible, config)


def test_an_unexpected_null_at_transform_time_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The same rule applies to rows a fitted state is merely encoding."""
    fitted = fit(train_rows(), catalog, eligible, config)
    rows: list[Row] = [(1.0, None, "success", "us", True, True)]
    with pytest.raises(ModelTrainingError, match="non-nullable"):
        fitted.transform(frame(rows))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_training_value_is_refused(
    value: float, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """NaN breaks every equality this layer's determinism rests on."""
    rows: list[Row] = [
        (value, 0, "success", "us", True, True),
        (1.0, 1, "success", "us", True, True),
    ]
    with pytest.raises(ModelTrainingError, match="non-finite"):
        fit(rows, catalog, eligible, config)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_transform_value_is_refused(
    value: float, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Nothing non-finite enters the matrix, and nothing non-finite leaves it."""
    fitted = fit(train_rows(), catalog, eligible, config)
    rows: list[Row] = [(value, 0, "success", "us", True, True)]
    with pytest.raises(ModelTrainingError, match="non-finite"):
        fitted.transform(frame(rows))


def test_a_feature_null_throughout_train_is_handled_explicitly(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Defined behaviour, recorded in state, rather than a crash or a guess.

    The median of nothing is undefined, so the value channel is filled with zero
    and ``all_null_in_train`` says why. The indicator is constant ``1.0``, which
    is the honest encoding of "never observed" -- and failing the whole run
    instead would let one feature undefined over the training window block two
    hundred that are fine.
    """
    rows: list[Row] = [(None, index, "success", "us", True, True) for index in range(4)]
    fitted = fit(rows, catalog, eligible, config)
    imputation = next(
        item
        for item in fitted.numeric_imputations
        if item.feature == "user_failure_rate"
    )
    assert imputation.all_null_in_train is True
    assert imputation.value == 0.0
    assert imputation.train_observed_count == 0
    assert imputation.train_null_count == 4
    matrix = fitted.transform(frame(rows))
    assert column(matrix, f"user_failure_rate{MISSING_INDICATOR_SUFFIX}") == [1.0] * 4


def test_the_zero_imputation_policy_fills_with_zero(
    catalog: Any, eligible: Any
) -> None:
    """The other declared policy, and it is still recorded as the one used."""
    config = PreprocessingConfig(
        min_category_frequency=2,
        numeric_imputation="zero",
        standardize_numeric_for_linear_models=False,
    )
    rows: list[Row] = [
        (10.0, 0, "success", "us", True, True),
        (None, 1, "success", "us", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    imputation = next(
        item
        for item in fitted.numeric_imputations
        if item.feature == "user_failure_rate"
    )
    assert imputation.policy == "zero"
    assert imputation.value == 0.0
    assert column(fitted.transform(frame(rows)), "user_failure_rate") == [10.0, 0.0]


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


def test_booleans_encode_false_to_zero_and_true_to_one(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The value channel is 0/1, not a scaled number."""
    rows: list[Row] = [
        (1.0, 0, "success", "us", False, True),
        (2.0, 1, "success", "us", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    matrix = fitted.transform(frame(rows))
    assert column(matrix, "is_new_device_for_user") == [0.0, 1.0]


def test_a_missing_boolean_is_distinguishable_from_false(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The pair is the encoding, and the pair separates them.

    Both rows leave the value channel at ``0.0``: filling a missing boolean with
    a third number would invent an observation and order it below false. What
    separates them is the indicator, so the assertion is on the pair rather than
    on either column alone.
    """
    rows: list[Row] = [
        (1.0, 0, "success", "us", False, True),
        (2.0, 1, "success", "us", None, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    matrix = fitted.transform(frame(rows))
    value = column(matrix, "is_new_device_for_user")
    indicator = column(matrix, f"is_new_device_for_user{MISSING_INDICATOR_SUFFIX}")
    assert list(zip(value, indicator, strict=True)) == [(0.0, 0.0), (0.0, 1.0)]


def test_a_non_nullable_boolean_gets_no_indicator(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """An indicator that can only ever read zero is a column of noise."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert "user_in_baseline" in fitted.output_feature_names
    assert (
        f"user_in_baseline{MISSING_INDICATOR_SUFFIX}" not in fitted.output_feature_names
    )


def test_booleans_are_never_standardized(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A 0/1 channel stays on 0/1 even with standardization enabled."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert fitted.standardization_enabled is True
    scaled = {item.column for item in fitted.scaling}
    assert "is_new_device_for_user" not in scaled
    assert "user_in_baseline" not in scaled


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------


def encoding_of(fitted: FittedPreprocessor, feature: str) -> Any:
    """Return one categorical feature's fitted encoding."""
    return next(
        item for item in fitted.categorical_encodings if item.feature == feature
    )


def test_the_vocabulary_is_sorted_not_encounter_ordered(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Sorted by code point, so file order cannot reorder the columns."""
    rows: list[Row] = [
        (1.0, 0, "success", None, True, True),
        (2.0, 1, "failure", None, True, True),
        (3.0, 2, "failure", None, True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    assert encoding_of(fitted, "current_authentication_outcome").categories == (
        "failure",
        "success",
    )


def test_a_null_category_lands_in_the_missing_bucket(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Nullable categoricals get a ``__missing`` column and use it."""
    rows: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (2.0, 1, "success", "us", True, True),
        (3.0, 2, "success", None, True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    matrix = fitted.transform(frame(rows))
    assert column(matrix, "current_country_code=__missing") == [0.0, 0.0, 1.0]


def test_a_non_nullable_categorical_gets_no_missing_column(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A bucket for a state the contract forbids would never be hot."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert "current_authentication_outcome=__missing" not in fitted.output_feature_names
    assert "current_authentication_outcome=__unknown" in fitted.output_feature_names


def test_a_category_unseen_in_train_lands_in_unknown(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """And the vocabulary does not grow to accommodate it."""
    fitted = fit(train_rows(), catalog, eligible, config)
    before = encoding_of(fitted, "current_country_code").categories
    rows: list[Row] = [(1.0, 0, "success", "jp", True, True)]
    matrix = fitted.transform(frame(rows, split=MLSplit.TEST))
    assert column(matrix, "current_country_code=__unknown") == [1.0]
    assert encoding_of(fitted, "current_country_code").categories == before


def test_exactly_one_column_is_hot_in_every_block(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Including for a null and for a value training never saw.

    A block of all zeros would say "none of the above" without saying which
    none, and two hot columns would double-count the row.
    """
    fitted = fit(train_rows(), catalog, eligible, config)
    rows: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (1.0, 1, "success", None, True, True),
        (1.0, 2, "success", "jp", True, True),
    ]
    matrix = fitted.transform(frame(rows, split=MLSplit.TEST))
    block = [
        index
        for index, name in enumerate(matrix.output_feature_names)
        if name.startswith("current_country_code=")
    ]
    for row in matrix.rows:
        assert sum(row[index] for index in block) == 1.0


def test_a_rare_train_category_lands_in_other(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Below the floor in TRAIN, so it is bucketed rather than given a column."""
    rows: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (2.0, 1, "success", "us", True, True),
        (3.0, 2, "success", "gb", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    encoding = encoding_of(fitted, "current_country_code")
    assert encoding.categories == ("us",)
    assert encoding.rare_categories == ("gb",)
    assert encoding.emits_rare_bucket is True
    matrix = fitted.transform(frame(rows))
    assert column(matrix, "current_country_code=__other") == [0.0, 0.0, 1.0]


def test_the_rare_threshold_boundary_is_inclusive_at_the_floor(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A count *equal* to the floor is kept; only strictly below is bucketed."""
    rows: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (2.0, 1, "success", "us", True, True),
        (3.0, 2, "success", "gb", True, True),
        (4.0, 3, "success", "gb", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    encoding = encoding_of(fitted, "current_country_code")
    assert encoding.categories == ("gb", "us")
    assert encoding.rare_categories == ()
    assert encoding.emits_rare_bucket is False
    assert "current_country_code=__other" not in fitted.output_feature_names


def test_a_category_rare_in_train_stays_other_however_common_it_becomes(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The bucket was decided at fit, and later frequency does not reopen it."""
    rows: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (2.0, 1, "success", "us", True, True),
        (3.0, 2, "success", "gb", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    later: list[Row] = [(1.0, index, "success", "gb", True, True) for index in range(9)]
    matrix = fitted.transform(frame(later, split=MLSplit.TEST))
    assert column(matrix, "current_country_code=__other") == [1.0] * 9
    assert column(matrix, "current_country_code=__unknown") == [0.0] * 9


def test_a_category_unseen_in_train_stays_unknown_not_other(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The two buckets mean different things and are never merged.

    ``__other`` means "training saw this and it was too rare to keep";
    ``__unknown`` means "training never saw this at all". Collapsing them would
    tell a model that an unheard-of country resembles the rare ones.
    """
    rows: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (2.0, 1, "success", "us", True, True),
        (3.0, 2, "success", "gb", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    later: list[Row] = [(1.0, 0, "success", "jp", True, True)]
    matrix = fitted.transform(frame(later, split=MLSplit.TEST))
    assert column(matrix, "current_country_code=__unknown") == [1.0]
    assert column(matrix, "current_country_code=__other") == [0.0]


def test_rare_bucketing_applies_only_to_declared_features(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The policy is a reviewed list, not a rule that switches itself on."""
    rows: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (2.0, 1, "success", "us", True, True),
        (3.0, 2, "failure", "us", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    outcome = encoding_of(fitted, "current_authentication_outcome")
    assert outcome.rare_bucketing_enabled is False
    assert outcome.categories == ("failure", "success")
    assert outcome.rare_categories == ()
    assert encoding_of(fitted, "current_country_code").rare_bucketing_enabled is True


def test_a_category_colliding_with_a_reserved_bucket_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Refused, not renamed: either resolution merges an observed value with a bucket."""
    rows: list[Row] = [
        (1.0, 0, "success", "__other", True, True),
        (2.0, 1, "success", "us", True, True),
    ]
    with pytest.raises(ModelTrainingError, match="reserved"):
        fit(rows, catalog, eligible, config)


def test_any_double_underscore_category_is_refused_not_just_the_three(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The prefix is reserved as a namespace, so a near-miss fails too."""
    rows: list[Row] = [
        (1.0, 0, "success", "__nearly", True, True),
        (2.0, 1, "success", "us", True, True),
    ]
    with pytest.raises(ModelTrainingError, match="reserved"):
        fit(rows, catalog, eligible, config)


def test_a_vocabulary_above_the_ceiling_is_refused(catalog: Any, eligible: Any) -> None:
    """A categorical wide enough to identify a row is not a categorical."""
    config = PreprocessingConfig(min_category_frequency=1, max_category_cardinality=3)
    rows: list[Row] = [
        (1.0, index, f"outcome-{index}", "us", True, True) for index in range(5)
    ]
    with pytest.raises(ModelTrainingError, match="ceiling"):
        fit(rows, catalog, eligible, config)


def test_the_vocabulary_is_immutable_after_fitting(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Frozen by the type system, not by convention."""
    fitted = fit(train_rows(), catalog, eligible, config)
    encoding = encoding_of(fitted, "current_country_code")
    with pytest.raises(Exception, match=r"frozen|immutable"):
        encoding.categories = ("us", "gb", "jp")


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_scaling_statistics_come_from_train_only(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Mean and population standard deviation over the training column."""
    rows: list[Row] = [
        (value, index, "success", "us", True, True)
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0])
    ]
    fitted = fit(rows, catalog, eligible, config)
    statistic = next(
        item for item in fitted.scaling if item.column == "user_failure_rate"
    )
    assert statistic.mean == pytest.approx(2.5)
    assert statistic.scale == pytest.approx(math.sqrt(1.25))
    assert statistic.zero_variance is False


def test_a_constant_column_scales_by_one_rather_than_by_zero(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Centring already sends it to zero; any other divisor is invention or a crash."""
    rows: list[Row] = [(7.0, index, "success", "us", True, True) for index in range(4)]
    fitted = fit(rows, catalog, eligible, config)
    statistic = next(
        item for item in fitted.scaling if item.column == "user_failure_rate"
    )
    assert statistic.zero_variance is True
    assert statistic.scale == 1.0
    assert column(fitted.transform(frame(rows)), "user_failure_rate") == [0.0] * 4


def test_disabling_standardization_records_no_statistics(
    catalog: Any, eligible: Any
) -> None:
    """Off means off: no state, and the raw value passes through."""
    config = PreprocessingConfig(
        min_category_frequency=2, standardize_numeric_for_linear_models=False
    )
    rows: list[Row] = [
        (value, index, "success", "us", True, True)
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0])
    ]
    fitted = fit(rows, catalog, eligible, config)
    assert fitted.scaling == ()
    assert fitted.standardization_enabled is False
    assert column(fitted.transform(frame(rows)), "user_failure_rate") == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_indicator_and_one_hot_columns_are_never_scaled(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The configuration asks for numeric standardization, and that is all it gets."""
    fitted = fit(train_rows(), catalog, eligible, config)
    scaled = {item.column for item in fitted.scaling}
    assert scaled == {"user_failure_rate", "user_attempt_count"}
    assert not any(name.endswith(MISSING_INDICATOR_SUFFIX) for name in scaled)
    assert not any("=" in name for name in scaled)


# ---------------------------------------------------------------------------
# The output contract
# ---------------------------------------------------------------------------


def test_the_output_order_follows_the_raw_order(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Every column of a feature, in raw order, before the next feature begins."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert fitted.raw_feature_names == fx.PREPROCESSING_FEATURES
    assert fitted.output_feature_names == (
        "user_failure_rate",
        "user_failure_rate__missing",
        "user_attempt_count",
        "current_authentication_outcome=failure",
        "current_authentication_outcome=success",
        "current_authentication_outcome=__unknown",
        "current_country_code=gb",
        "current_country_code=us",
        "current_country_code=__missing",
        "current_country_code=__unknown",
        "is_new_device_for_user",
        "is_new_device_for_user__missing",
        "user_in_baseline",
    )


def test_output_names_are_unique(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """One column cannot mean two things."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert len(set(fitted.output_feature_names)) == fitted.output_feature_count


def test_the_transformed_width_matches_the_declared_order(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Every row is as wide as the contract says."""
    fitted = fit(train_rows(), catalog, eligible, config)
    matrix = fitted.transform(frame(train_rows()))
    assert matrix.output_feature_names == fitted.output_feature_names
    assert all(len(row) == fitted.output_feature_count for row in matrix.rows)


def test_a_frame_missing_a_feature_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A narrower frame is a different contract."""
    fitted = fit(train_rows(), catalog, eligible, config)
    narrow = Frame(
        split=MLSplit.TEST,
        feature_names=fx.PREPROCESSING_FEATURES[:-1],
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, 0, "success", "us", True),),
    )
    with pytest.raises(ModelTrainingError, match="missing"):
        fitted.transform(narrow)


def test_a_frame_with_an_extra_feature_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A feature nobody admitted cannot arrive by being appended to the frame."""
    fitted = fit(train_rows(), catalog, eligible, config)
    wide = Frame(
        split=MLSplit.TEST,
        feature_names=(*fx.PREPROCESSING_FEATURES, "surprise_feature"),
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, 0, "success", "us", True, True, 9.0),),
    )
    with pytest.raises(ModelTrainingError, match="unexpected"):
        fitted.transform(wide)


def test_the_same_features_in_a_different_order_are_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The right set in the wrong order scores the wrong feature silently."""
    fitted = fit(train_rows(), catalog, eligible, config)
    swapped = (
        fx.PREPROCESSING_FEATURES[1],
        fx.PREPROCESSING_FEATURES[0],
        *fx.PREPROCESSING_FEATURES[2:],
    )
    reordered = Frame(
        split=MLSplit.TEST,
        feature_names=swapped,
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((0, 1.0, "success", "us", True, True),),
    )
    with pytest.raises(ModelTrainingError, match="order"):
        fitted.transform(reordered)


def test_a_frame_disagreeing_with_the_eligible_contract_is_refused_at_fit(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Fit checks the same contract transform does."""
    wrong = Frame(
        split=MLSplit.TRAIN,
        feature_names=fx.PREPROCESSING_FEATURES[:3],
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, 0, "success"),),
    )
    with pytest.raises(ModelTrainingError, match="missing"):
        fit_preprocessor(wrong, catalog=catalog, eligible=eligible, config=config)


def test_a_feature_the_catalog_does_not_declare_is_refused(
    catalog: Any, config: PreprocessingConfig
) -> None:
    """Preprocessing cannot infer a type it was never told."""
    wider = fx.catalog([*catalog.specs, fx.spec("undeclared_elsewhere")])
    eligible = resolve_eligible_features(
        wider, fx.allowlist_for(wider), include_leakage_classes=ALL_CLASSES
    )
    narrow_frame = Frame(
        split=MLSplit.TRAIN,
        feature_names=eligible.feature_names,
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, 0, "success", "us", True, True, 2.0),),
    )
    with pytest.raises(ModelTrainingError, match="does not declare"):
        fit_preprocessor(
            narrow_frame, catalog=catalog, eligible=eligible, config=config
        )


def test_the_recorded_feature_fingerprint_is_the_eligible_contracts(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Model identity traces back to the reviewed allowlist through this field."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert fitted.eligible_feature_list_fingerprint == eligible.fingerprint()


def test_a_different_feature_contract_changes_the_fingerprint(
    catalog: Any, config: PreprocessingConfig
) -> None:
    """Narrowing the admitted features is a different preprocessor, not the same one."""
    full = resolve_eligible_features(
        catalog, fx.allowlist_for(catalog), include_leakage_classes=ALL_CLASSES
    )
    narrowed = resolve_eligible_features(
        catalog,
        fx.allowlist_for(catalog),
        include_leakage_classes=("prior_only", "current_event_context"),
    )
    wide = fit(train_rows(), catalog, full, config)
    narrow = fit_preprocessor(
        Frame(
            split=MLSplit.TRAIN,
            feature_names=narrowed.feature_names,
            anchors=tuple(Anchor(fx.anchor_id(i), fx.at(i)) for i in range(8)),
            feature_matrix=tuple(
                (float(i), i, "success" if i % 2 else "failure", "us") for i in range(8)
            ),
        ),
        catalog=catalog,
        eligible=narrowed,
        config=config,
    )
    assert wide.fingerprint() != narrow.fingerprint()
    assert (
        wide.eligible_feature_list_fingerprint
        != narrow.eligible_feature_list_fingerprint
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_the_json_round_trip_is_byte_identical(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Deserialize then serialize returns exactly the same bytes."""
    fitted = fit(train_rows(), catalog, eligible, config)
    text = fitted.to_json()
    assert FittedPreprocessor.from_json(text).to_json() == text


def test_the_round_trip_preserves_numeric_output_exactly(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A reloaded preprocessor is the same function, not an approximation.

    Statistics are quantized when they are fitted rather than only when they are
    written, so the number the transform multiplies by is the number the JSON
    carries and no drift is possible.
    """
    fitted = fit(train_rows(), catalog, eligible, config)
    reloaded = FittedPreprocessor.from_json(fitted.to_json())
    rows = train_rows(5)
    assert reloaded.transform(frame(rows)).rows == fitted.transform(frame(rows)).rows
    assert reloaded.fingerprint() == fitted.fingerprint()


def test_the_canonical_json_has_sorted_keys(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Key order is fixed, so two renderings of one state cannot differ."""
    fitted = fit(train_rows(), catalog, eligible, config)
    payload = json.loads(fitted.to_json())
    assert list(payload) == sorted(payload)


def test_the_fingerprint_is_stable_across_repeated_fits(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Same rows, same policy, same identity -- every time."""
    first = fit(train_rows(), catalog, eligible, config)
    second = fit(train_rows(), catalog, eligible, config)
    assert first.fingerprint() == second.fingerprint()
    assert first.to_json() == second.to_json()


def test_the_state_carries_no_timestamp_or_path(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Identity is semantic: not when it was fitted, not where it was written."""
    fitted = fit(train_rows(), catalog, eligible, config)
    text = fitted.to_json()
    assert "fitted_at" not in text
    assert "created_at" not in text
    assert str(datetime.now(UTC).year) not in text
    assert "/home/" not in text
    assert "/tmp" not in text


def test_a_changed_policy_changes_the_config_fingerprint(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Two preprocessors fitted on the same rows under different policies differ."""
    fitted = fit(train_rows(), catalog, eligible, config)
    other = fit(
        train_rows(),
        catalog,
        eligible,
        config.model_copy(update={"numeric_imputation": "zero"}),
    )
    assert (
        fitted.preprocessing_config_fingerprint
        != other.preprocessing_config_fingerprint
    )
    assert fitted.fingerprint() != other.fingerprint()


def test_an_unknown_schema_version_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A state from a contract this build does not implement is not loaded."""
    payload = fit(train_rows(), catalog, eligible, config).to_dict()
    payload["preprocessing_schema_version"] = "2.0.0"
    with pytest.raises(ModelTrainingError, match="schema version"):
        FittedPreprocessor.from_dict(payload)


def test_an_unknown_field_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Loading loosely would drop what it did not understand and recompute a
    fingerprint over the remainder."""
    payload = fit(train_rows(), catalog, eligible, config).to_dict()
    payload["fitted_on_a_tuesday"] = True
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_missing_field_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Half a contract is not a contract."""
    payload = fit(train_rows(), catalog, eligible, config).to_dict()
    del payload["output_feature_names"]
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


@pytest.mark.parametrize("text", ["", "{", "[]", "null", '"a string"'])
def test_malformed_json_is_refused(text: str) -> None:
    """Neither invalid JSON nor valid JSON of the wrong shape is accepted."""
    with pytest.raises(ModelTrainingError):
        FittedPreprocessor.from_json(text)


def test_the_schema_version_is_declared(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """State says which contract it was written against."""
    fitted = fit(train_rows(), catalog, eligible, config)
    assert fitted.preprocessing_schema_version == PREPROCESSING_SCHEMA_VERSION


def test_transform_never_mutates_fitted_state(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Serialize, transform a pile of strange rows, serialize again: identical."""
    fitted = fit(train_rows(), catalog, eligible, config)
    before = fitted.to_json()
    fitted.transform(frame(train_rows(3), split=MLSplit.TEST))
    fitted.transform(
        frame(
            [(None, 9, "success", "jp", None, False)],
            split=MLSplit.NOVEL_ANOMALY_HOLDOUT,
        )
    )
    assert fitted.to_json() == before


def test_the_fitted_state_is_frozen(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """No assignment path exists, so there is no historical state to overwrite."""
    fitted = fit(train_rows(), catalog, eligible, config)
    with pytest.raises(Exception, match=r"frozen|immutable"):
        fitted.train_row_count = 99


def test_refitting_returns_a_new_object(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Fitting again builds a second state rather than editing the first."""
    first = fit(train_rows(), catalog, eligible, config)
    second = fit(train_rows(4), catalog, eligible, config)
    assert first is not second
    assert first.train_row_count == 8
    assert second.train_row_count == 4


# ---------------------------------------------------------------------------
# Canonical row order
# ---------------------------------------------------------------------------


def test_the_low_level_fit_rejects_non_canonical_rows(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Dataset assembly may sort; a fitted-stage API must not sort silently.

    Sorting here would hide the fact that somebody handed the trainer rows it
    had not canonicalized, and the next stage that forgot would be the one that
    produced a wrong answer quietly.
    """
    rows = train_rows(4)
    scrambled = Frame(
        split=MLSplit.TRAIN,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=tuple(
            Anchor(fx.anchor_id(index), fx.at(index)) for index in (3, 1, 2, 0)
        ),
        feature_matrix=tuple(tuple(row) for row in rows),
    )
    with pytest.raises(ModelTrainingError, match="canonical"):
        fit_preprocessor(scrambled, catalog=catalog, eligible=eligible, config=config)


def test_the_low_level_transform_rejects_non_canonical_rows(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The same rule, so an unsorted batch cannot slip past prediction either."""
    fitted = fit(train_rows(), catalog, eligible, config)
    scrambled = Frame(
        split=MLSplit.TEST,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=(Anchor(fx.anchor_id(2), fx.at(2)), Anchor(fx.anchor_id(1), fx.at(1))),
        feature_matrix=tuple(tuple(row) for row in train_rows(2)),
    )
    with pytest.raises(ModelTrainingError, match="canonical"):
        fitted.transform(scrambled)


def test_identical_timestamps_are_broken_by_anchor_id(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A tie has one resolution, and it is the anchor identifier."""
    tied = Frame(
        split=MLSplit.TRAIN,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=tuple(Anchor(fx.anchor_id(index), fx.at(0)) for index in range(4)),
        feature_matrix=tuple(tuple(row) for row in train_rows(4)),
    )
    fitted = fit_preprocessor(tied, catalog=catalog, eligible=eligible, config=config)
    assert fitted.train_row_count == 4

    reversed_ties = Frame(
        split=MLSplit.TRAIN,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=tuple(Anchor(fx.anchor_id(index), fx.at(0)) for index in (3, 2, 1, 0)),
        feature_matrix=tuple(tuple(row) for row in train_rows(4)),
    )
    with pytest.raises(ModelTrainingError, match="canonical"):
        fit_preprocessor(
            reversed_ties, catalog=catalog, eligible=eligible, config=config
        )


def assembled_train(
    eligible: Any, rows: list[Row], order: list[int] | None = None
) -> Any:
    """Return the TRAIN split of a dataset assembled from *rows*, optionally shuffled."""
    indices = list(range(len(rows))) if order is None else order
    feature_rows = []
    for index in indices:
        record = fx.feature_row(index, names=eligible.feature_names)
        record.update(dict(zip(eligible.feature_names, rows[index], strict=True)))
        feature_rows.append(record)
    dataset = assemble_ml_dataset(
        feature_rows=feature_rows,
        labels=[fx.label_row(index) for index in indices],
        splits=[fx.split_row(index, "train") for index in indices],
        eligible=eligible,
    )
    return dataset.for_split(MLSplit.TRAIN)


@pytest.mark.parametrize("seed", [1, 5, 13, 20260809])
def test_shuffled_source_rows_produce_identical_fitted_state(
    seed: int, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Milestone 2 canonicalizes, so Milestone 3 fits the same state either way."""
    rows = train_rows(8)
    baseline = fit_preprocessor(
        assembled_train(eligible, rows),
        catalog=catalog,
        eligible=eligible,
        config=config,
    )
    order = list(range(8))
    random.Random(seed).shuffle(order)
    shuffled = fit_preprocessor(
        assembled_train(eligible, rows, order),
        catalog=catalog,
        eligible=eligible,
        config=config,
    )
    assert shuffled.to_json() == baseline.to_json()
    assert shuffled.fingerprint() == baseline.fingerprint()


def test_the_vocabulary_does_not_depend_on_encounter_order(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Sorted, so which category arrived first is not a fact the state records."""
    forward: list[Row] = [
        (1.0, 0, "success", "us", True, True),
        (2.0, 1, "success", "gb", True, True),
        (3.0, 2, "success", "us", True, True),
        (4.0, 3, "success", "gb", True, True),
    ]
    swapped: list[Row] = [
        (1.0, 0, "success", "gb", True, True),
        (2.0, 1, "success", "us", True, True),
        (3.0, 2, "success", "gb", True, True),
        (4.0, 3, "success", "us", True, True),
    ]
    first = encoding_of(fit(forward, catalog, eligible, config), "current_country_code")
    second = encoding_of(
        fit(swapped, catalog, eligible, config), "current_country_code"
    )
    assert first.categories == second.categories == ("gb", "us")


def test_the_median_does_not_depend_on_row_order(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Sorted by value, so a permutation of the same values gives the same median."""
    values = [9.0, 1.0, 5.0, 3.0]
    rows: list[Row] = [
        (value, index, "success", "us", True, True)
        for index, value in enumerate(values)
    ]
    shuffled_rows: list[Row] = [
        (value, index, "success", "us", True, True)
        for index, value in enumerate(sorted(values, reverse=True))
    ]
    first = fit(rows, catalog, eligible, config)
    second = fit(shuffled_rows, catalog, eligible, config)
    assert (
        next(
            item
            for item in first.numeric_imputations
            if item.feature == "user_failure_rate"
        ).value
        == next(
            item
            for item in second.numeric_imputations
            if item.feature == "user_failure_rate"
        ).value
    )


# ---------------------------------------------------------------------------
# Behavioural leakage
# ---------------------------------------------------------------------------


def hostile_rows(kind: str) -> list[Row]:
    """Return non-training rows designed to move any statistic that reads them."""
    if kind == "validation":
        # Wildly different numbers, far more missingness, inverted category mix.
        return [
            (None, 10_000 + index, "failure", None, None, False) for index in range(40)
        ]
    if kind == "test":
        # Categories training never saw, and a numeric scale three orders larger.
        return [
            (1e6 + index, 900_000 + index, "success", "jp", True, False)
            for index in range(40)
        ]
    # Holdout: a different distribution again, and every boolean flipped.
    return [
        (-500.0 - index, index, "failure", "de", False, False) for index in range(40)
    ]


def test_perturbing_every_non_training_split_changes_no_fitted_state(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The strongest acceptance test in this milestone.

    Fit on TRAIN once. Then transform validation, test, and holdout fixtures
    built to move anything that reads them: different numeric distributions,
    different missingness, different category frequencies, unseen categories, a
    numeric scale a million times larger, and flipped booleans. Compare the
    serialized state byte for byte.
    """
    fitted = fit(train_rows(), catalog, eligible, config)
    before = fitted.to_json()
    before_fingerprint = fitted.fingerprint()

    for kind, split in (
        ("validation", MLSplit.VALIDATION),
        ("test", MLSplit.TEST),
        ("holdout", MLSplit.NOVEL_ANOMALY_HOLDOUT),
    ):
        fitted.transform(frame(hostile_rows(kind), split=split))

    assert fitted.to_json() == before
    assert fitted.fingerprint() == before_fingerprint


def test_refitting_after_the_perturbation_reaches_the_same_state(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Not merely unmutated: a fresh fit on the same TRAIN rows agrees exactly.

    This is the part a "no assignment to self" reading would miss. It proves the
    fitted statistics are a function of the training rows alone, not of anything
    the process happened to see along the way.
    """
    first = fit(train_rows(), catalog, eligible, config)
    for kind in ("validation", "test", "holdout"):
        first.transform(frame(hostile_rows(kind), split=MLSplit.TEST))
    second = fit(train_rows(), catalog, eligible, config)
    assert second.to_json() == first.to_json()


@pytest.mark.parametrize(
    ("field", "getter"),
    [
        (
            "imputation",
            lambda state: [item.value for item in state.numeric_imputations],
        ),
        (
            "vocabulary",
            lambda state: [item.categories for item in state.categorical_encodings],
        ),
        (
            "rare buckets",
            lambda state: [
                item.rare_categories for item in state.categorical_encodings
            ],
        ),
        ("scaling", lambda state: [(i.mean, i.scale) for i in state.scaling]),
        ("output order", lambda state: list(state.output_feature_names)),
    ],
)
def test_each_fitted_component_survives_the_perturbation(
    field: str,
    getter: Any,
    catalog: Any,
    eligible: Any,
    config: PreprocessingConfig,
) -> None:
    """Named individually, so a failure says which part leaked."""
    fitted = fit(train_rows(), catalog, eligible, config)
    before = getter(fitted)
    for kind in ("validation", "test", "holdout"):
        fitted.transform(frame(hostile_rows(kind), split=MLSplit.VALIDATION))
    assert getter(fitted) == before, field


def test_only_the_transformed_output_of_changed_rows_differs(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The converse: the perturbation is real, and the transform does react to it."""
    fitted = fit(train_rows(), catalog, eligible, config)
    ordinary = fitted.transform(frame(train_rows(4), split=MLSplit.TEST))
    hostile = fitted.transform(frame(hostile_rows("test")[:4], split=MLSplit.TEST))
    assert hostile.rows != ordinary.rows
    assert hostile.output_feature_names == ordinary.output_feature_names


@pytest.mark.parametrize(
    "perturbed",
    [
        [(99.0, 0, "success", "us", True, True)],
        [(1.0, 0, "timeout", "us", True, True)],
        [(1.0, 0, "success", "jp", True, True)],
        [(None, 0, "success", "us", True, True)],
        [(1.0, 0, "success", "us", False, True)],
    ],
    ids=["numeric", "new-category", "new-country", "missingness", "boolean"],
)
def test_perturbing_train_does_change_the_fingerprint(
    perturbed: list[Row], catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The other half of the leakage test.

    A state that never changed would pass the perturbation test trivially. These
    five edits each change a training semantic, and each must move the
    fingerprint -- otherwise the first test proves only that nothing is fitted.
    """
    baseline = fit(train_rows(6), catalog, eligible, config)
    changed = fit([*train_rows(6)[1:], *perturbed], catalog, eligible, config)
    assert changed.fingerprint() != baseline.fingerprint()


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


PROHIBITED_SUBSTRINGS = (
    "anchor_event_id",
    "campaign_id",
    "usr_",
    "src_",
    "dev_",
    "password",
    "token",
    "credential",
    "latitude",
    "longitude",
    "/home/",
)


def test_the_serialized_state_carries_nothing_prohibited(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A sweep of the rendered JSON for identifiers, secrets, and paths."""
    fitted = fit(train_rows(), catalog, eligible, config)
    text = fitted.to_json()
    for token in PROHIBITED_SUBSTRINGS:
        assert token not in text, token


def test_the_state_carries_no_row_identifier(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Counts and statistics only: no anchor from the training set appears."""
    fitted = fit(train_rows(), catalog, eligible, config)
    text = fitted.to_json()
    assert all(fx.anchor_id(index) not in text for index in range(8))


def test_a_pseudonym_shaped_category_fails_the_audit(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A vocabulary is observed values, so it is refused rather than published."""
    rows: list[Row] = [
        (1.0, 0, "success", "usr_9f3a1c", True, True),
        (2.0, 1, "success", "us", True, True),
    ]
    with pytest.raises(ModelTrainingError, match="pseudonym"):
        fit(rows, catalog, eligible, config)


def test_a_coordinate_shaped_category_fails_the_audit(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A location must never travel into a model artifact."""
    rows: list[Row] = [
        (1.0, 0, "success", "51.5,-0.12", True, True),
        (2.0, 1, "success", "us", True, True),
    ]
    with pytest.raises(ModelTrainingError, match="coordinate"):
        fit(rows, catalog, eligible, config)


def test_a_path_shaped_category_fails_the_audit(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """An absolute path in a vocabulary is somebody's home directory."""
    rows: list[Row] = [
        (1.0, 0, "success", "/home/analyst/data", True, True),
        (2.0, 1, "success", "us", True, True),
    ]
    with pytest.raises(ModelTrainingError, match="path"):
        fit(rows, catalog, eligible, config)


def test_a_categorical_the_catalog_calls_sensitive_is_not_serialized(
    config: PreprocessingConfig,
) -> None:
    """Declared privacy class decides, not whether the feature was admitted.

    A feature can pass every eligibility check and still be the wrong thing to
    write a vocabulary of, because eligibility asks "may a model use this" and
    this asks "may an artifact publish these values".
    """
    sensitive = fx.catalog(
        [
            fx.spec("user_failure_rate"),
            fx.spec(
                "operational_label",
                group="current_context",
                leakage_class="current_event_context",
                dtype="string",
                window=None,
                privacy_class="operational_metadata",
            ),
        ]
    )
    eligible = resolve_eligible_features(
        sensitive, fx.allowlist_for(sensitive), include_leakage_classes=ALL_CLASSES
    )
    built = Frame(
        split=MLSplit.TRAIN,
        feature_names=eligible.feature_names,
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)), Anchor(fx.anchor_id(1), fx.at(1))),
        feature_matrix=((1.0, "alpha"), (2.0, "beta")),
    )
    with pytest.raises(ModelTrainingError, match="vocabulary"):
        fit_preprocessor(built, catalog=sensitive, eligible=eligible, config=config)


# ---------------------------------------------------------------------------
# Type and shape faults
#
# A cell of the wrong Python type is a data fault, not something to coerce.
# Coercing "3" to 3.0 would work until the day a column arrives as "3,0" and
# the model scores a silently different number.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([("3.0", 0, "success", "us", True, True)], "numeric but"),
        ([(1.0, 0, 7, "us", True, True)], "categorical but"),
        ([(1.0, 0, "success", "us", 1, True)], "boolean but"),
    ],
    ids=["numeric", "categorical", "boolean"],
)
def test_a_wrongly_typed_training_value_is_refused(
    rows: list[Row],
    match: str,
    catalog: Any,
    eligible: Any,
    config: PreprocessingConfig,
) -> None:
    """Fitting refuses a cell whose type disagrees with the catalog."""
    with pytest.raises(ModelTrainingError, match=match):
        fit([*rows, *train_rows(2)], catalog, eligible, config)


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (("3.0", 0, "success", "us", True, True), "numeric but"),
        ((1.0, 0, 7, "us", True, True), "categorical but"),
        ((1.0, 0, "success", "us", 1, True), "boolean but"),
    ],
    ids=["numeric", "categorical", "boolean"],
)
def test_a_wrongly_typed_transform_value_is_refused(
    row: Row, match: str, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """And so does encoding, so a bad batch cannot reach a model at prediction."""
    fitted = fit(train_rows(), catalog, eligible, config)
    with pytest.raises(ModelTrainingError, match=match):
        fitted.transform(frame([row], split=MLSplit.TEST))


@pytest.mark.parametrize("nullable_feature", ["boolean", "categorical"])
def test_a_null_on_a_non_nullable_boolean_or_category_is_refused(
    nullable_feature: str, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Same rule as for numbers, applied to the other two dtypes."""
    row: Row = (
        (1.0, 0, "success", "us", True, None)
        if nullable_feature == "boolean"
        else (1.0, 0, None, "us", True, True)
    )
    with pytest.raises(ModelTrainingError, match="non-nullable"):
        fit([row, *train_rows(2)], catalog, eligible, config)
    fitted = fit(train_rows(), catalog, eligible, config)
    with pytest.raises(ModelTrainingError, match="non-nullable"):
        fitted.transform(frame([row], split=MLSplit.TEST))


def test_a_training_row_of_the_wrong_width_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A short row would silently shift every value one feature to the left."""
    ragged = Frame(
        split=MLSplit.TRAIN,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, 0, "success"),),
    )
    with pytest.raises(ModelTrainingError, match="training row carries"):
        fit_preprocessor(ragged, catalog=catalog, eligible=eligible, config=config)


def test_a_transform_row_of_the_wrong_width_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The declared width is checked per row, not only per frame."""
    fitted = fit(train_rows(), catalog, eligible, config)
    ragged = Frame(
        split=MLSplit.TEST,
        feature_names=fx.PREPROCESSING_FEATURES,
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, 0, "success"),),
    )
    with pytest.raises(ModelTrainingError, match="a row carries"):
        fitted.transform(ragged)


def test_a_dtype_with_no_declared_encoding_is_refused(
    config: PreprocessingConfig,
) -> None:
    """Admitting a feature to a model means deciding what it means numerically."""
    timestamped = fx.catalog(
        [
            fx.spec("user_failure_rate"),
            fx.spec("first_seen_at", dtype="timestamp", window=None),
        ]
    )
    eligible = resolve_eligible_features(
        timestamped,
        fx.allowlist_for(timestamped),
        include_leakage_classes=ALL_CLASSES,
    )
    built = Frame(
        split=MLSplit.TRAIN,
        feature_names=eligible.feature_names,
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, datetime(2026, 3, 1, tzinfo=UTC)),),
    )
    with pytest.raises(ModelTrainingError, match="no declared encoding"):
        fit_preprocessor(built, catalog=timestamped, eligible=eligible, config=config)


def test_colliding_output_column_names_are_refused(
    config: PreprocessingConfig,
) -> None:
    """A feature named after another's indicator would make one column mean two things.

    ``user_failure_rate`` is nullable, so it emits ``user_failure_rate__missing``.
    A second feature literally called that collides, and the fit refuses rather
    than letting the second silently overwrite the first.
    """
    colliding = fx.catalog(
        [
            fx.spec("user_failure_rate", nullable=True),
            fx.spec("user_failure_rate__missing", nullable=False),
        ]
    )
    eligible = resolve_eligible_features(
        colliding, fx.allowlist_for(colliding), include_leakage_classes=ALL_CLASSES
    )
    built = Frame(
        split=MLSplit.TRAIN,
        feature_names=eligible.feature_names,
        anchors=(Anchor(fx.anchor_id(0), fx.at(0)),),
        feature_matrix=((1.0, 2.0),),
    )
    with pytest.raises(ModelTrainingError, match="colliding"):
        fit_preprocessor(built, catalog=colliding, eligible=eligible, config=config)


def test_a_two_part_category_that_is_not_a_coordinate_is_allowed(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The coordinate screen tests whether the parts are numbers, not whether
    there is a comma."""
    rows: list[Row] = [
        (1.0, 0, "success", "eu,west", True, True),
        (2.0, 1, "success", "eu,west", True, True),
    ]
    fitted = fit(rows, catalog, eligible, config)
    assert encoding_of(fitted, "current_country_code").categories == ("eu,west",)


# ---------------------------------------------------------------------------
# Hand-constructed state is validated too
#
# ``from_dict`` leans on these validators, so a payload that has been edited
# after it was written is refused by the same rules a fit obeys.
# ---------------------------------------------------------------------------


def payload_of(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> dict[str, Any]:
    """Return a healthy serialized state, ready to be corrupted."""
    return fit(train_rows(), catalog, eligible, config).to_dict()


def test_a_state_whose_scaling_names_an_unemitted_column_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A statistic for a column nobody emits could never be applied."""
    payload = payload_of(catalog, eligible, config)
    payload["scaling"][0]["column"] = "a_column_nobody_emits"
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_state_with_scaling_but_standardization_off_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Statistics that the contract says are not applied read as though they are."""
    payload = payload_of(catalog, eligible, config)
    payload["standardization_enabled"] = False
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_state_that_does_not_describe_every_raw_feature_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """An undescribed feature would have no encoding and no column."""
    payload = payload_of(catalog, eligible, config)
    payload["numeric_imputations"] = payload["numeric_imputations"][:1]
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_state_with_duplicate_output_columns_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """The uniqueness the fit enforces is enforced again on load."""
    payload = payload_of(catalog, eligible, config)
    payload["output_feature_names"] = [payload["output_feature_names"][0]] * len(
        payload["output_feature_names"]
    )
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_state_with_repeated_raw_features_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """One feature cannot occupy two positions in the raw order."""
    payload = payload_of(catalog, eligible, config)
    payload["raw_feature_names"] = [payload["raw_feature_names"][0]] * len(
        payload["raw_feature_names"]
    )
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda item: item.update(categories=["us", "gb"]), "sorted order"),
        (lambda item: item.update(categories=["us", "us"]), "repeats"),
        (lambda item: item.update(categories=["__unknown"]), "collide"),
        (
            lambda item: item.update(categories=["us"], rare_categories=["us"]),
            "keeps and buckets",
        ),
        (lambda item: item.update(unknown_label="__missing"), "not distinct"),
    ],
    ids=["unsorted", "repeated", "collision", "kept-and-bucketed", "same-label"],
)
def test_an_incoherent_vocabulary_is_refused(
    mutation: Any,
    field: str,
    catalog: Any,
    eligible: Any,
    config: PreprocessingConfig,
) -> None:
    """Each vocabulary invariant is checked on load, named individually."""
    payload = payload_of(catalog, eligible, config)
    encoding = next(
        item
        for item in payload["categorical_encodings"]
        if item["feature"] == "current_country_code"
    )
    mutation(encoding)
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_rare_bucket_without_rare_bucketing_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A ``__other`` column under a policy that never buckets is unreachable."""
    payload = payload_of(catalog, eligible, config)
    encoding = next(
        item
        for item in payload["categorical_encodings"]
        if item["feature"] == "current_authentication_outcome"
    )
    encoding["emits_rare_bucket"] = True
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_a_non_positive_scale_is_refused(
    scale: float, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Dividing by zero is the failure the constant-column branch exists to avoid."""
    payload = payload_of(catalog, eligible, config)
    payload["scaling"][0]["scale"] = scale
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_statistic_in_a_payload_is_refused(
    value: float, catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A state carrying NaN was never fitted by this code, and is not loaded."""
    payload = payload_of(catalog, eligible, config)
    payload["numeric_imputations"][0]["value"] = value
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_state_describing_no_feature_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A preprocessor over nothing is not a preprocessor."""
    payload = payload_of(catalog, eligible, config)
    payload["raw_feature_names"] = []
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_state_emitting_no_column_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """A transform that produced no columns would return an empty matrix silently."""
    payload = payload_of(catalog, eligible, config)
    payload["output_feature_names"] = []
    payload["scaling"] = []
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)


def test_a_state_scaling_one_column_twice_is_refused(
    catalog: Any, eligible: Any, config: PreprocessingConfig
) -> None:
    """Two statistics for one column leave it ambiguous which is applied."""
    payload = payload_of(catalog, eligible, config)
    payload["scaling"] = [payload["scaling"][0], payload["scaling"][0]]
    with pytest.raises(ModelTrainingError, match="not valid"):
        FittedPreprocessor.from_dict(payload)
