"""Deterministic, train-only preprocessing: raw feature rows to a numeric matrix.

Everything fitted here -- an imputation constant, a category vocabulary, a
scaling mean -- is computed from the **training split alone** and then frozen.
:func:`fit_preprocessor` refuses any split but ``train``, and the fitted object
is an immutable pydantic model, so there is no ``partial_fit`` and no attribute
a later transform could quietly update.

**What this module may see.**  The reviewed ordered feature matrix from
Milestone 2, the Phase 3 catalog describing those features, and the
preprocessing configuration.  It reads no Parquet, no label, no split
assignment, and no campaign identifier: rows arrive as a typed frame whose
protocol exposes exactly ``feature_names``, ``feature_matrix``, ``anchors``, and
``split``, and an anchor exposes only its identifier and event time -- enough to
assert canonical order, and nothing more.  The label-reader allowlist stays the
two modules Milestone 2 fixed it at.

**Null is not zero.**  Phase 3's doctrine is carried through unchanged: a null
means the quantity was *undefined* for that row -- no prior events in the
window, no fitted baseline -- and a zero means it was *observed to be zero*.
Imputing a null without saying so would hand an estimator a fabricated
observation and delete the distinction in the same step.  So every nullable
feature ships a ``<name>__missing`` indicator beside its value channel, the
value channel is filled from a train-only statistic, and observed zeros are
left exactly as they were found.

**Three synthetic buckets, never a real value.**  ``__missing`` for a null,
``__unknown`` for a category no training row contained, ``__other`` for one that
was too rare in training to fit.  Their names come from the configuration, they
all begin with ``__``, and an observed category beginning with ``__`` is refused
rather than allowed to collide.  A category that was rare in training stays
``__other`` however common it later becomes, and a category training never saw
stays ``__unknown`` however often it arrives -- because the alternative is a
vocabulary that grows at transform time, which is a fitted state that changes
after it was frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any, Final, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.features.catalog import (
    FeatureCatalog,
    FeatureDType,
    FeatureSpec,
)
from password_attack_detector.ml.config import PreprocessingConfig
from password_attack_detector.ml.enums import FIT_ELIGIBLE_SPLITS, MLSplit
from password_attack_detector.ml.features import EligibleFeatureList
from password_attack_detector.ml.ordering import AnchoredRow, assert_canonical

__all__ = [
    "CATEGORY_SEPARATOR",
    "MISSING_INDICATOR_SUFFIX",
    "PREPROCESSING_SCHEMA_VERSION",
    "PRIVACY_SAFE_PRIVACY_CLASSES",
    "RESERVED_BUCKET_PREFIX",
    "BooleanEncoding",
    "CategoricalEncoding",
    "FeatureFrame",
    "FittedPreprocessor",
    "NumericImputation",
    "ScalingStatistic",
    "TransformedMatrix",
    "fit_preprocessor",
]

#: The preprocessing state contract's own version, independent of the ML schema.
PREPROCESSING_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Appended to a nullable feature's name to make its missingness indicator.
MISSING_INDICATOR_SUFFIX: Final[str] = "__missing"

#: Placed between a categorical feature and one of its values in a column name.
#:
#: ``=`` cannot occur in a Phase 3 feature name -- they are lowercase
#: identifiers -- so ``a=b`` can only have come from feature ``a``.  Output names
#: are checked for uniqueness anyway; a separator argument is not a proof.
CATEGORY_SEPARATOR: Final[str] = "="

#: Every synthetic bucket begins with this, and no observed value may.
RESERVED_BUCKET_PREFIX: Final[str] = "__"

#: Catalog privacy classes whose category values may be written into state.
#:
#: A vocabulary is the one part of fitted preprocessing state built from
#: *observed values* rather than from counts, so it is the one part that could
#: carry an identifier out of the training data and into a serialized artifact.
#: Only features the catalog declares non-sensitive are eligible, and the audit
#: below refuses to publish anything else rather than trusting that an admitted
#: feature must have been safe.
PRIVACY_SAFE_PRIVACY_CLASSES: Final[frozenset[str]] = frozenset({"non_sensitive"})

#: Value shapes that must never appear in a serialized category vocabulary.
_IDENTIFIER_SHAPED: Final[tuple[tuple[str, str], ...]] = (
    ("usr_", "a user pseudonym"),
    ("src_", "a source pseudonym"),
    ("dev_", "a device pseudonym"),
    ("app_", "an application pseudonym"),
    ("ses_", "a session pseudonym"),
    ("/", "an absolute path"),
    ("~", "a home-directory path"),
)

_NUMERIC_DTYPES: Final[frozenset[FeatureDType]] = frozenset(
    {FeatureDType.INT64, FeatureDType.FLOAT64}
)

#: Decimal places every fitted statistic is quantized to.
#:
#: The same precision every other fingerprint in this project uses.  Quantizing
#: at *construction* rather than only at serialization is what makes the
#: round-trip guarantee exact: the number a transform multiplies by is the same
#: number the JSON carries, so a state reloaded from disk cannot drift a few
#: ulps away from the one that was fitted.
_FLOAT_PRECISION: Final[int] = 9


def _quantize(value: float) -> float:
    """Return *value* at the fixed precision fitted statistics are stored in."""
    if not math.isfinite(value):
        raise ValueError(
            "a fitted statistic must be finite; NaN and infinity are neither "
            "serializable nor meaningful as a constant"
        )
    return float(f"{value:.{_FLOAT_PRECISION}f}")


@runtime_checkable
class FeatureFrame(Protocol):
    """The rows preprocessing is allowed to see.

    Deliberately narrower than
    :class:`~password_attack_detector.ml.dataset.SplitDataset`, which satisfies
    it structurally.  Nothing here exposes a label, a campaign, or supervised
    eligibility, so this module cannot read one even by accident -- the boundary
    is in the type, not in a reviewer's memory.
    """

    @property
    def split(self) -> MLSplit: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def anchors(self) -> tuple[AnchoredRow, ...]: ...

    @property
    def feature_matrix(self) -> tuple[tuple[Any, ...], ...]: ...


# ---------------------------------------------------------------------------
# Fitted state
# ---------------------------------------------------------------------------


class NumericImputation(BaseModel):
    """The train-only constant that fills one numeric feature's nulls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    policy: str
    value: float
    nullable: bool
    #: True when training observed no value at all for this feature.
    #:
    #: The statistic is then undefined, so the value channel is filled with
    #: ``0.0`` and the indicator -- constant ``1.0`` across every training row --
    #: carries the whole story: never observed.  Failing instead would let one
    #: feature that happens to be undefined over the training window block a run
    #: whose remaining two hundred features are fine.
    all_null_in_train: bool = False
    train_observed_count: int = Field(ge=0)
    train_null_count: int = Field(ge=0)

    @field_validator("value")
    @classmethod
    def quantize_value(cls, value: float) -> float:
        """Store the constant at serialized precision, so a round trip is exact."""
        return _quantize(value)


class BooleanEncoding(BaseModel):
    """How one boolean feature becomes a numeric channel plus an indicator.

    ``False`` encodes to ``0.0`` and ``True`` to ``1.0``.  A null also fills the
    value channel with ``imputed_value``, which means the value channel alone
    cannot separate "observed false" from "not observed" -- and that is why the
    indicator is not optional for a nullable boolean.  The pair is the encoding:
    ``(0.0, 0.0)`` is an observed false and ``(0.0, 1.0)`` is a missing one, and
    a test pins exactly that.

    The alternative -- a third value such as ``-1`` in the value channel -- was
    rejected: it invents an observation that never happened and orders it below
    false, which is meaningless for a boolean and actively wrong for any model
    that treats the column as continuous.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    nullable: bool
    false_value: float = 0.0
    true_value: float = 1.0
    imputed_value: float = 0.0
    train_true_count: int = Field(ge=0)
    train_false_count: int = Field(ge=0)
    train_null_count: int = Field(ge=0)


class CategoricalEncoding(BaseModel):
    """One categorical feature's frozen vocabulary and bucket policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    nullable: bool
    #: Retained training categories, sorted by code point.  Sorted rather than
    #: encounter-ordered so two runs over the same rows in different file order
    #: produce the same column order.
    categories: tuple[str, ...]
    #: Training categories that fell below the frequency floor.  Recorded, not
    #: merely dropped, so a fitted state explains its own ``__other`` column.
    rare_categories: tuple[str, ...] = ()
    rare_bucketing_enabled: bool = False
    rare_threshold: int | None = None
    emits_rare_bucket: bool = False
    missing_label: str
    unknown_label: str
    rare_label: str
    train_null_count: int = Field(ge=0)

    @model_validator(mode="after")
    def check_vocabulary(self) -> Self:
        """Categories are unique, sorted, disjoint from the buckets, and safe."""
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"{self.feature!r} repeats a category")
        if list(self.categories) != sorted(self.categories):
            raise ValueError(f"{self.feature!r} categories are not in sorted order")
        if set(self.categories) & set(self.rare_categories):
            raise ValueError(f"{self.feature!r} keeps and buckets the same category")
        buckets = {self.missing_label, self.unknown_label, self.rare_label}
        if len(buckets) != 3:
            raise ValueError(f"{self.feature!r} bucket labels are not distinct")
        collide = sorted(set(self.categories) & buckets)
        if collide:
            raise ValueError(
                f"{self.feature!r} observed category value(s) {collide} collide "
                f"with a reserved bucket label"
            )
        if self.emits_rare_bucket and not self.rare_bucketing_enabled:
            raise ValueError(
                f"{self.feature!r} emits a rare bucket without rare bucketing"
            )
        return self


class ScalingStatistic(BaseModel):
    """One numeric column's train-only location and scale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    mean: float
    #: The divisor actually used.  Equal to the population standard deviation,
    #: except on a constant column where it is ``1.0`` -- never zero, and never
    #: an epsilon that would inflate a column of identical values into noise.
    scale: float
    zero_variance: bool = False

    @field_validator("mean", "scale")
    @classmethod
    def quantize_statistic(cls, value: float) -> float:
        """Store statistics at serialized precision, so a round trip is exact."""
        return _quantize(value)

    @model_validator(mode="after")
    def check_scale_is_positive(self) -> Self:
        """A zero scale would divide the matrix into nonsense."""
        if self.scale <= 0.0:
            raise ValueError(
                f"scale for {self.column!r} is {self.scale!r}; it must be "
                f"strictly positive, and a constant column uses 1.0"
            )
        return self


class TransformedMatrix(BaseModel):
    """A transformed design matrix and the column order it was written in."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_feature_names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]

    @property
    def row_count(self) -> int:
        """Return the number of rows."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Return the number of columns."""
        return len(self.output_feature_names)


class FittedPreprocessor(BaseModel):
    """Frozen preprocessing state: everything a transform needs, and its identity.

    Immutable by construction.  :meth:`transform` reads this state and returns a
    new matrix; there is no code path that writes back, so a validation, test, or
    holdout row cannot change what a training row fitted.  Re-fitting produces a
    *new* object rather than mutating this one, which is why the leakage test can
    compare two states byte for byte and mean it.

    What is deliberately **absent**: any anchor or event identifier, any campaign
    identifier, any label or split name, any pseudonym, any path, and any
    timestamp describing when fitting happened.  Identity here is semantic -- the
    same rows fitted twice, a year apart, in two directories, produce the same
    fingerprint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    preprocessing_schema_version: str = PREPROCESSING_SCHEMA_VERSION
    raw_feature_names: tuple[str, ...]
    eligible_feature_list_fingerprint: str
    preprocessing_config_fingerprint: str
    numeric_imputations: tuple[NumericImputation, ...]
    boolean_encodings: tuple[BooleanEncoding, ...]
    categorical_encodings: tuple[CategoricalEncoding, ...]
    scaling: tuple[ScalingStatistic, ...]
    standardization_enabled: bool
    output_feature_names: tuple[str, ...]
    train_row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        """The state describes every raw feature once and names its columns once."""
        if not self.raw_feature_names:
            raise ValueError("a preprocessor must describe at least one feature")
        if len(set(self.raw_feature_names)) != len(self.raw_feature_names):
            raise ValueError("raw_feature_names repeats a feature")
        described = (
            [item.feature for item in self.numeric_imputations]
            + [item.feature for item in self.boolean_encodings]
            + [item.feature for item in self.categorical_encodings]
        )
        if sorted(described) != sorted(self.raw_feature_names):
            missing = sorted(set(self.raw_feature_names) - set(described))
            extra = sorted(set(described) - set(self.raw_feature_names))
            raise ValueError(
                f"fitted state does not describe exactly the raw features: "
                f"{len(missing)} undescribed, {len(extra)} unexpected"
            )
        if len(set(self.output_feature_names)) != len(self.output_feature_names):
            raise ValueError("output_feature_names repeats a column")
        if not self.output_feature_names:
            raise ValueError("a preprocessor must emit at least one column")
        scaled = [item.column for item in self.scaling]
        if len(set(scaled)) != len(scaled):
            raise ValueError("scaling repeats a column")
        unknown = sorted(set(scaled) - set(self.output_feature_names))
        if unknown:
            raise ValueError(f"scaling names column(s) {unknown} that are not emitted")
        if self.scaling and not self.standardization_enabled:
            raise ValueError("scaling statistics present while standardization is off")
        return self

    @property
    def output_feature_count(self) -> int:
        """Return the number of columns a transform emits."""
        return len(self.output_feature_names)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping this state serializes to."""
        rendered = _canonical_scalars(self.model_dump(mode="python"))
        if not isinstance(rendered, dict):  # pragma: no cover - shape is fixed
            raise ModelTrainingError("preprocessor state did not render as an object")
        return rendered

    @classmethod
    def from_dict(cls, payload: Any) -> FittedPreprocessor:
        """Return the state *payload* describes, or raise.

        An unknown field, a missing field, or a schema version this code does
        not implement is a failure rather than a best effort: a preprocessor
        loaded loosely would silently drop the part it did not understand, and
        the resulting matrix would be wrong in a way no fingerprint could catch
        because the fingerprint would be recomputed from the truncated state.
        """
        if not isinstance(payload, dict):
            raise ModelTrainingError(
                f"preprocessor state must be a JSON object, got "
                f"{type(payload).__name__}"
            )
        version = payload.get("preprocessing_schema_version")
        if version != PREPROCESSING_SCHEMA_VERSION:
            raise ModelTrainingError(
                f"preprocessor state declares schema version {version!r}; this "
                f"build implements {PREPROCESSING_SCHEMA_VERSION!r}"
            )
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise ModelTrainingError(
                f"preprocessor state is not valid ({type(exc).__name__})"
            ) from None

    def to_json(self) -> str:
        """Return canonical JSON: sorted keys, ASCII, no incidental whitespace."""
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, text: str) -> FittedPreprocessor:
        """Return the state *text* encodes, or raise."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelTrainingError(
                f"preprocessor state is not valid JSON ({type(exc).__name__})"
            ) from None
        return cls.from_dict(payload)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic content that gives this state its identity.

        The whole state, because every field of it changes what a transform
        does.  There is nothing here to exclude: no path, no timestamp, no
        machine detail ever enters the model in the first place.
        """
        return self.to_dict()

    def fingerprint(self) -> str:
        """Return the SHA-256 digest of this state's canonical rendering."""
        canonical = json.dumps(
            self.fingerprint_data(), sort_keys=True, ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -- transform ----------------------------------------------------------

    def transform(self, frame: FeatureFrame) -> TransformedMatrix:
        """Return *frame* encoded by this frozen state.

        Accepts any split: transforming validation, test, or holdout rows is the
        normal case, and none of it changes a fitted statistic -- there is no
        assignment to ``self`` anywhere below.  What it does check is that the
        frame is the shape this state was fitted for: the same raw features, in
        the same order, canonically sorted.

        Raises:
            ModelTrainingError: on a feature-order mismatch, a missing or extra
                feature, a non-canonical row order, an unexpected null, or a
                non-finite value.
        """
        _require_feature_order(tuple(frame.feature_names), self.raw_feature_names)
        assert_canonical(tuple(frame.anchors), stage="preprocessing transform")

        numeric = {item.feature: item for item in self.numeric_imputations}
        boolean = {item.feature: item for item in self.boolean_encodings}
        categorical = {item.feature: item for item in self.categorical_encodings}
        scaling = {item.column: item for item in self.scaling}

        rows: list[tuple[float, ...]] = []
        for values in frame.feature_matrix:
            if len(values) != len(self.raw_feature_names):
                raise ModelTrainingError(
                    f"a row carries {len(values)} value(s) for "
                    f"{len(self.raw_feature_names)} feature(s)"
                )
            encoded: list[float] = []
            for name, raw in zip(self.raw_feature_names, values, strict=True):
                if name in numeric:
                    encoded += _encode_numeric(numeric[name], raw)
                elif name in boolean:
                    encoded += _encode_boolean(boolean[name], raw)
                else:
                    encoded += _encode_categorical(categorical[name], raw)
            if len(encoded) != len(self.output_feature_names):
                raise ModelTrainingError(
                    f"encoding produced {len(encoded)} column(s) for "
                    f"{len(self.output_feature_names)} declared column(s)"
                )
            if scaling:
                encoded = [
                    (
                        (value - statistic.mean) / statistic.scale
                        if (statistic := scaling.get(column)) is not None
                        else value
                    )
                    for column, value in zip(
                        self.output_feature_names, encoded, strict=True
                    )
                ]
            for column, value in zip(self.output_feature_names, encoded, strict=True):
                if not math.isfinite(value):
                    raise ModelTrainingError(
                        f"transform produced a non-finite value in column "
                        f"{column!r}; the matrix must be finite everywhere"
                    )
            rows.append(tuple(encoded))

        return TransformedMatrix(
            output_feature_names=self.output_feature_names, rows=tuple(rows)
        )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_numeric(state: NumericImputation, raw: Any) -> list[float]:
    """Return the value channel and, when nullable, the missingness indicator."""
    if raw is None:
        if not state.nullable:
            raise ModelTrainingError(
                f"feature {state.feature!r} is declared non-nullable but a row "
                f"carries no value; an unexpected null is a data fault, not "
                f"something to impute over"
            )
        return [state.value, 1.0]
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ModelTrainingError(
            f"feature {state.feature!r} is numeric but a row carries "
            f"{type(raw).__name__}"
        )
    value = float(raw)
    if not math.isfinite(value):
        raise ModelTrainingError(
            f"feature {state.feature!r} carries a non-finite value; a missing "
            f"observation must be null, which means undefined, not NaN"
        )
    return [value, 0.0] if state.nullable else [value]


def _encode_boolean(state: BooleanEncoding, raw: Any) -> list[float]:
    """Return the value channel and, when nullable, the missingness indicator."""
    if raw is None:
        if not state.nullable:
            raise ModelTrainingError(
                f"feature {state.feature!r} is declared non-nullable but a row "
                f"carries no value"
            )
        return [state.imputed_value, 1.0]
    if not isinstance(raw, bool):
        raise ModelTrainingError(
            f"feature {state.feature!r} is boolean but a row carries "
            f"{type(raw).__name__}"
        )
    value = state.true_value if raw else state.false_value
    return [value, 0.0] if state.nullable else [value]


def _encode_categorical(state: CategoricalEncoding, raw: Any) -> list[float]:
    """Return the one-hot block for one categorical value.

    Exactly one column is hot in every block, including for a null and for a
    value training never saw.  A block of all zeros would say "none of the
    above" without saying which none.
    """
    columns = _categorical_labels(state)
    if raw is None:
        if not state.nullable:
            raise ModelTrainingError(
                f"feature {state.feature!r} is declared non-nullable but a row "
                f"carries no value"
            )
        hot = state.missing_label
    else:
        if not isinstance(raw, str):
            raise ModelTrainingError(
                f"feature {state.feature!r} is categorical but a row carries "
                f"{type(raw).__name__}"
            )
        text = raw
        if text in state.categories:
            hot = text
        elif text in state.rare_categories and state.emits_rare_bucket:
            hot = state.rare_label
        else:
            # Unseen in training, or rare in training with no bucket emitted.
            # Either way it is not a category this state can represent, and
            # __unknown is the honest column for that -- never __other, which
            # means "seen in training and too rare to keep".
            hot = state.unknown_label
    return [1.0 if column == hot else 0.0 for column in columns]


def _categorical_labels(state: CategoricalEncoding) -> tuple[str, ...]:
    """Return the ordered labels one categorical feature expands into."""
    labels = list(state.categories)
    if state.emits_rare_bucket:
        labels.append(state.rare_label)
    if state.nullable:
        labels.append(state.missing_label)
    labels.append(state.unknown_label)
    return tuple(labels)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_preprocessor(
    frame: FeatureFrame,
    *,
    catalog: FeatureCatalog,
    eligible: EligibleFeatureList,
    config: PreprocessingConfig,
) -> FittedPreprocessor:
    """Fit preprocessing state on canonical training rows.

    Args:
        frame: the training rows, canonically ordered.
        catalog: the Phase 3 catalog describing each raw feature's type,
            nullability, and privacy class.
        eligible: the reviewed feature contract, which fixes the raw order and
            supplies the fingerprint the fitted state records.
        config: the preprocessing policy.

    Raises:
        ModelTrainingError: when handed a split other than ``train``, a
            non-canonical row order, a feature order disagreeing with
            *eligible*, a feature the catalog does not declare, a category
            colliding with a reserved bucket, a vocabulary above the configured
            ceiling, an unexpected null, a non-finite value, or a categorical
            feature whose values are not safe to serialize.
    """
    if frame.split not in FIT_ELIGIBLE_SPLITS:
        raise ModelTrainingError(
            f"preprocessing may be fitted on "
            f"{sorted(str(item) for item in FIT_ELIGIBLE_SPLITS)} only; this "
            f"frame is {str(frame.split)!r}. Fitting on any other split would "
            f"put information from the rows a model is judged on into the model"
        )
    _require_feature_order(tuple(frame.feature_names), eligible.feature_names)
    assert_canonical(tuple(frame.anchors), stage="preprocessing fit")

    if not frame.feature_matrix:
        raise ModelTrainingError(
            "the training frame carries no rows; a preprocessor fitted on "
            "nothing would impute every null with a constant nobody measured"
        )

    names = eligible.feature_names
    specs = {spec.name: spec for spec in catalog.specs}
    missing = [name for name in names if name not in specs]
    if missing:
        raise ModelTrainingError(
            f"the catalog does not declare {len(missing)} admitted feature(s), "
            f"including {sorted(missing)[:5]}; preprocessing cannot infer a "
            f"type it was not told"
        )

    columns = _columns_by_feature(names, frame.feature_matrix)

    numeric: list[NumericImputation] = []
    boolean: list[BooleanEncoding] = []
    categorical: list[CategoricalEncoding] = []
    output: list[str] = []

    for name in names:
        spec = specs[name]
        values = columns[name]
        if spec.dtype in _NUMERIC_DTYPES:
            state = _fit_numeric(spec, values, config)
            numeric.append(state)
            output.append(name)
            if spec.nullable:
                output.append(f"{name}{MISSING_INDICATOR_SUFFIX}")
        elif spec.dtype is FeatureDType.BOOL:
            bool_state = _fit_boolean(spec, values)
            boolean.append(bool_state)
            output.append(name)
            if spec.nullable:
                output.append(f"{name}{MISSING_INDICATOR_SUFFIX}")
        elif spec.dtype is FeatureDType.STRING:
            cat_state = _fit_categorical(spec, values, config)
            categorical.append(cat_state)
            output += [
                f"{name}{CATEGORY_SEPARATOR}{label}"
                for label in _categorical_labels(cat_state)
            ]
        else:
            raise ModelTrainingError(
                f"feature {name!r} has dtype {str(spec.dtype)!r}, which has no "
                f"declared encoding; admitting it to a model requires deciding "
                f"what it means numerically first"
            )

    _require_unique_columns(output)

    scaling = (
        _fit_scaling(
            output_names=tuple(output),
            numeric_value_columns=tuple(
                item.feature for item in numeric if not _is_indicator_like(specs, item)
            ),
            rows=_encode_all(
                numeric, boolean, categorical, names, frame.feature_matrix
            ),
        )
        if config.standardize_numeric_for_linear_models
        else ()
    )

    return FittedPreprocessor(
        raw_feature_names=names,
        eligible_feature_list_fingerprint=eligible.fingerprint(),
        preprocessing_config_fingerprint=_config_fingerprint(config),
        numeric_imputations=tuple(numeric),
        boolean_encodings=tuple(boolean),
        categorical_encodings=tuple(categorical),
        scaling=scaling,
        standardization_enabled=config.standardize_numeric_for_linear_models,
        output_feature_names=tuple(output),
        train_row_count=len(frame.feature_matrix),
    )


def _is_indicator_like(specs: dict[str, FeatureSpec], item: NumericImputation) -> bool:
    """Return whether a numeric feature is really a 0/1 flag.

    Never true today -- Phase 3 declares its flags as ``bool`` -- but stated
    rather than assumed, because the rule that matters is "indicators are not
    standardized", not "booleans are not standardized".
    """
    spec = specs[item.feature]
    return spec.value_range == (0.0, 1.0) and spec.dtype is FeatureDType.INT64


def _columns_by_feature(
    names: tuple[str, ...], matrix: Sequence[Sequence[Any]]
) -> dict[str, list[Any]]:
    """Return each feature's training values, in canonical row order."""
    columns: dict[str, list[Any]] = {name: [] for name in names}
    for values in matrix:
        if len(values) != len(names):
            raise ModelTrainingError(
                f"a training row carries {len(values)} value(s) for "
                f"{len(names)} feature(s)"
            )
        for name, value in zip(names, values, strict=True):
            columns[name].append(value)
    return columns


def _fit_numeric(
    spec: FeatureSpec, values: Sequence[Any], config: PreprocessingConfig
) -> NumericImputation:
    """Return the imputation constant for one numeric feature."""
    observed: list[float] = []
    nulls = 0
    for raw in values:
        if raw is None:
            if not spec.nullable:
                raise ModelTrainingError(
                    f"feature {spec.name!r} is declared non-nullable but a "
                    f"training row carries no value"
                )
            nulls += 1
            continue
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ModelTrainingError(
                f"feature {spec.name!r} is numeric but a training row carries "
                f"{type(raw).__name__}"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ModelTrainingError(
                f"feature {spec.name!r} carries a non-finite training value; a "
                f"missing observation must be null, which means undefined"
            )
        observed.append(value)

    all_null = spec.nullable and not observed
    if config.numeric_imputation == "zero" or all_null:
        constant = 0.0
    else:
        constant = _median(observed)
    return NumericImputation(
        feature=spec.name,
        policy=config.numeric_imputation,
        value=constant,
        nullable=spec.nullable,
        all_null_in_train=all_null,
        train_observed_count=len(observed),
        train_null_count=nulls,
    )


def _median(values: Sequence[float]) -> float:
    """Return the median of *values*, with the even-length rule pinned.

    Odd length gives the middle order statistic.  Even length averages the two
    middle ones -- the ordinary convention, written out here rather than
    delegated, so a library changing its tie rule cannot silently move every
    imputation constant in the repository.  Sorting is by value, so the result
    does not depend on the order rows arrived in.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _fit_boolean(spec: FeatureSpec, values: Sequence[Any]) -> BooleanEncoding:
    """Return the encoding for one boolean feature."""
    trues = falses = nulls = 0
    for raw in values:
        if raw is None:
            if not spec.nullable:
                raise ModelTrainingError(
                    f"feature {spec.name!r} is declared non-nullable but a "
                    f"training row carries no value"
                )
            nulls += 1
        elif isinstance(raw, bool):
            trues += raw
            falses += not raw
        else:
            raise ModelTrainingError(
                f"feature {spec.name!r} is boolean but a training row carries "
                f"{type(raw).__name__}"
            )
    return BooleanEncoding(
        feature=spec.name,
        nullable=spec.nullable,
        train_true_count=trues,
        train_false_count=falses,
        train_null_count=nulls,
    )


def _fit_categorical(
    spec: FeatureSpec, values: Sequence[Any], config: PreprocessingConfig
) -> CategoricalEncoding:
    """Return the frozen vocabulary and bucket policy for one categorical feature."""
    _require_serializable_vocabulary(spec)

    counts: dict[str, int] = {}
    nulls = 0
    for raw in values:
        if raw is None:
            if not spec.nullable:
                raise ModelTrainingError(
                    f"feature {spec.name!r} is declared non-nullable but a "
                    f"training row carries no value"
                )
            nulls += 1
            continue
        if not isinstance(raw, str):
            raise ModelTrainingError(
                f"feature {spec.name!r} is categorical but a training row "
                f"carries {type(raw).__name__}"
            )
        _require_safe_category(spec, raw)
        counts[raw] = counts.get(raw, 0) + 1

    bucketing = spec.name in config.rare_category_features
    threshold = config.min_category_frequency if bucketing else None
    rare = (
        tuple(sorted(name for name, count in counts.items() if count < threshold))
        if threshold is not None
        else ()
    )
    kept = tuple(sorted(name for name in counts if name not in set(rare)))

    if len(kept) > config.max_category_cardinality:
        raise ModelTrainingError(
            f"feature {spec.name!r} keeps {len(kept)} training categories, above "
            f"the configured ceiling of {config.max_category_cardinality}; a "
            f"vocabulary that wide is an identifier in disguise"
        )

    return CategoricalEncoding(
        feature=spec.name,
        nullable=spec.nullable,
        categories=kept,
        rare_categories=rare,
        rare_bucketing_enabled=bucketing,
        rare_threshold=threshold,
        emits_rare_bucket=bool(rare),
        missing_label=config.missing_category_label,
        unknown_label=config.unknown_category_label,
        rare_label=config.rare_category_label,
        train_null_count=nulls,
    )


def _require_serializable_vocabulary(spec: FeatureSpec) -> None:
    """Raise unless the catalog declares this feature's values safe to publish."""
    if spec.privacy_class not in PRIVACY_SAFE_PRIVACY_CLASSES:
        raise ModelTrainingError(
            f"categorical feature {spec.name!r} is declared "
            f"{spec.privacy_class!r}, and only "
            f"{sorted(PRIVACY_SAFE_PRIVACY_CLASSES)} may have a vocabulary "
            f"written into fitted state; a vocabulary is observed values, so "
            f"publishing this one would publish them"
        )


def _require_safe_category(spec: FeatureSpec, value: str) -> None:
    """Raise on a category that must not enter serialized state.

    Two rejections, and they are different failures.  A value beginning with
    ``__`` would collide with a synthetic bucket and make ``__other`` ambiguous.
    A value shaped like a pseudonym, a path, or a coordinate is an identifier
    that an admitted feature has no business carrying -- the audit fails rather
    than publishing the vocabulary, because the alternative is a model artifact
    that quietly contains the training population.
    """
    if value.startswith(RESERVED_BUCKET_PREFIX):
        raise ModelTrainingError(
            f"feature {spec.name!r} carries a training category beginning with "
            f"{RESERVED_BUCKET_PREFIX!r}, which is reserved for the synthetic "
            f"missing, unknown, and rare buckets; the collision is refused "
            f"rather than resolved, because either resolution would silently "
            f"merge an observed value with a bucket"
        )
    for prefix, what in _IDENTIFIER_SHAPED:
        if value.startswith(prefix):
            raise ModelTrainingError(
                f"feature {spec.name!r} carries a training category that looks "
                f"like {what}; a category vocabulary is serialized into the "
                f"model artifact, and an identifier must never travel there"
            )
    if _looks_like_a_coordinate(value):
        raise ModelTrainingError(
            f"feature {spec.name!r} carries a training category that looks like "
            f"a coordinate pair; a category vocabulary is serialized into the "
            f"model artifact, and a location must never travel there"
        )


def _looks_like_a_coordinate(value: str) -> bool:
    """Return whether *value* reads as a ``lat,lon`` pair."""
    parts = value.split(",")
    if len(parts) != 2:
        return False
    try:
        latitude, longitude = (float(part.strip()) for part in parts)
    except ValueError:
        return False
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def _encode_all(
    numeric: Sequence[NumericImputation],
    boolean: Sequence[BooleanEncoding],
    categorical: Sequence[CategoricalEncoding],
    names: tuple[str, ...],
    matrix: Sequence[Sequence[Any]],
) -> list[list[float]]:
    """Return the unscaled encoding of every training row.

    Used only to fit the scaling statistics, which need the encoded columns
    rather than the raw ones.
    """
    numeric_by = {item.feature: item for item in numeric}
    boolean_by = {item.feature: item for item in boolean}
    categorical_by = {item.feature: item for item in categorical}
    rows: list[list[float]] = []
    for values in matrix:
        encoded: list[float] = []
        for name, raw in zip(names, values, strict=True):
            if name in numeric_by:
                encoded += _encode_numeric(numeric_by[name], raw)
            elif name in boolean_by:
                encoded += _encode_boolean(boolean_by[name], raw)
            else:
                encoded += _encode_categorical(categorical_by[name], raw)
        rows.append(encoded)
    return rows


def _fit_scaling(
    *,
    output_names: tuple[str, ...],
    numeric_value_columns: tuple[str, ...],
    rows: Sequence[Sequence[float]],
) -> tuple[ScalingStatistic, ...]:
    """Return train-only location and scale for the numeric value channels.

    Only those channels.  Missingness indicators and one-hot columns are already
    on ``{0, 1}``; standardizing them would turn "this value was absent" into a
    number whose meaning depends on how often it was absent in training, and
    would leave a constant indicator dividing by nothing.  The configuration
    does not ask for it, so it does not happen.

    The divisor is the **population** standard deviation, and a constant column
    gets ``1.0``: subtracting the mean already sends it to zero, and any other
    divisor would either be a division by zero or an arbitrary inflation of a
    column that carries no variation at all.
    """
    scalable = set(numeric_value_columns)
    statistics: list[ScalingStatistic] = []
    for index, column in enumerate(output_names):
        if column not in scalable:
            continue
        values = [row[index] for row in rows]
        if not values:
            statistics.append(
                ScalingStatistic(column=column, mean=0.0, scale=1.0, zero_variance=True)
            )
            continue
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        # Judged at the precision the scale is *stored* at. A deviation of 1e-12
        # is zero once quantized, and dividing by a stored zero is the failure
        # this branch exists to prevent -- so the test is applied to the number
        # that will actually be used, not to the one before rounding.
        deviation = _quantize(math.sqrt(variance))
        zero_variance = deviation <= 0.0
        statistics.append(
            ScalingStatistic(
                column=column,
                mean=mean,
                scale=1.0 if zero_variance else deviation,
                zero_variance=zero_variance,
            )
        )
    return tuple(statistics)


# ---------------------------------------------------------------------------
# Shared checks and rendering
# ---------------------------------------------------------------------------


def _require_feature_order(actual: tuple[str, ...], expected: tuple[str, ...]) -> None:
    """Raise unless *actual* is exactly *expected*, in order.

    Order, not membership.  A matrix whose columns are the right set in the
    wrong order produces a model that scores the wrong feature and reports
    nothing wrong at all.
    """
    if actual == expected:
        return
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ModelTrainingError(
            f"feature set disagrees with the fitted contract: {len(missing)} "
            f"missing, including {missing[:5]}; {len(extra)} unexpected, "
            f"including {extra[:5]}"
        )
    raise ModelTrainingError(
        "feature order disagrees with the fitted contract; the same features in "
        "a different order encode a different matrix"
    )


def _require_unique_columns(columns: Sequence[str]) -> None:
    """Raise when two encoded columns would share a name."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for column in columns:
        if column in seen:
            duplicates.add(column)
        seen.add(column)
    if duplicates:
        raise ModelTrainingError(
            f"encoding produced {len(duplicates)} colliding output column "
            f"name(s), including {sorted(duplicates)[:5]}; one column cannot "
            f"mean two things"
        )


def _config_fingerprint(config: PreprocessingConfig) -> str:
    """Return the digest of the preprocessing policy this state was fitted under."""
    canonical = json.dumps(config.fingerprint_data(), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _canonical_scalars(payload: Any) -> Any:
    """Return *payload* with every float rendered stably and finitely.

    Floats are formatted at nine decimals, matching every other fingerprint in
    this project, so a digest does not move with the last bit of a repr.  A
    non-finite value raises here rather than serializing: ``NaN`` round-trips
    through JSON as ``NaN``, which is not JSON, and infinity is not a statistic.
    """
    if isinstance(payload, dict):
        return {key: _canonical_scalars(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [_canonical_scalars(value) for value in payload]
    if isinstance(payload, bool) or payload is None or isinstance(payload, int | str):
        return payload
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ModelTrainingError(
                "preprocessor state carries a non-finite number; it cannot be "
                "serialized, and it should never have been fitted"
            )
        return float(f"{payload:.9f}")
    return str(payload)
