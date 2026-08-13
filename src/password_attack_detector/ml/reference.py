"""The frozen reference profile a later population is compared against.

A monitor needs something to be a monitor *of*.  This module captures that
something once, from the approved pre-test population, and seals it: the bin
edges, the category vocabulary, the class space, the expected proportions, and
every lineage fingerprint that says which model and which feature contract the
profile belongs to.

**The reference is TRAIN, and only TRAIN.**  The reviewed configuration declares
``drift.reference_source: train`` and
:data:`~password_attack_detector.ml.enums.DRIFT_REFERENCE_ELIGIBLE_SPLITS` says
the same thing where a split is a type rather than a string.  Test and the
novel-anomaly holdout are absent for the reason they are absent everywhere else.
Validation is absent for a narrower one: it fitted the calibrator and chose the
operating point, so a baseline drawn from it would be a baseline drawn from rows
the frozen champion was already tuned against.

**Incoming data never redefines the partition.**  Every bin edge, every
category, every class, and every expected proportion is fixed here, at capture
time, from reference rows alone.  Nothing in :mod:`password_attack_detector.ml.drift`
writes back, and there is no field on a profile that could be written to: the
profile is frozen, sealed, and re-derived from its own content on every read.

**Out-of-range is a bin, not a discard.**  Numeric partitions carry *interior*
edges only, so the lowest and highest bins are open-ended and a value beyond
anything training saw lands in one of them rather than being dropped.  Nulls get
their own bin, unseen categories get their own bin, and a reference feature that
is constant gets a deliberately degenerate two-way partition rather than a
quantile grid that would collapse to nothing.

**A profile is monitoring evidence, not a model.**  Nothing here is fitted in
the learning sense, nothing here is a parameter of the champion, and nothing
here can change what the champion does.  It is a description of a population,
kept so that a later population can be described the same way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ModelNotReadyError
from password_attack_detector.ml.calibration import SealedModel, digest, quantize
from password_attack_detector.ml.enums import (
    DRIFT_REFERENCE_ELIGIBLE_SPLITS,
    UNKNOWN_CATEGORY,
    MLSplit,
    PredictionDriftQuantity,
    ReferenceFeatureKind,
    ScoreKind,
    is_probability,
)
from password_attack_detector.ml.schemas import Sha256Hex, prohibited_metadata_fields

__all__ = [
    "NULL_BIN",
    "OTHER_BIN",
    "REFERENCE_SCHEMA_VERSION",
    "UNKNOWN_BIN",
    "FeatureReference",
    "MLReferenceProfile",
    "PredictionReference",
    "ReferenceBin",
    "build_reference_profile",
    "feature_reference_index",
    "numeric_bin_index",
    "numeric_bin_labels",
    "prediction_reference_index",
    "quantile_edges",
    "reference_profile_to_markdown",
]

#: Version of the reference-profile contract.
REFERENCE_SCHEMA_VERSION: Final = "1.0.0"

#: The bin a missing observation lands in.  Present on every feature partition,
#: including one whose reference rows were never null: a bin that only appears
#: once something lands in it is a bin the incoming data defined.
NULL_BIN: Final = "__null__"

#: The bin a value the reference partition cannot represent lands in.
UNKNOWN_BIN: Final = "__unknown__"

#: The bin the frozen preprocessor's rare-category bucket maps onto.
OTHER_BIN: Final = "__other__"

#: Labels reserved by this module.  A reference vocabulary may not collide with
#: one, because a category literally named ``__null__`` would make an observed
#: value indistinguishable from a missing one.
_RESERVED_BINS: Final[frozenset[str]] = frozenset({NULL_BIN, UNKNOWN_BIN, OTHER_BIN})


class ReferenceBin(BaseModel):
    """One cell of a frozen partition, and the reference mass that fell in it.

    ``lower`` and ``upper`` are present for a numeric interval and absent for a
    categorical or boolean cell.  An interval is half-open on the left and closed
    on the right -- ``(lower, upper]`` -- with ``None`` standing for an unbounded
    side, which is what makes the outermost cells able to hold a value the
    reference population never reached.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell: str
    lower: float | None = None
    upper: float | None = None
    reference_count: int = Field(ge=0)
    reference_proportion: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_bin(self) -> Self:
        """The interval is ordered where it exists, and the mass is finite."""
        if not self.cell.strip():
            raise ValueError("a bin carries a label")
        if not math.isfinite(self.reference_proportion):
            raise ValueError("reference_proportion must be finite")
        for value, name in ((self.lower, "lower"), (self.upper, "upper")):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite where it is present")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("a numeric bin's lower edge sits below its upper edge")
        return self


class FeatureReference(BaseModel):
    """One raw feature's frozen partition and expected distribution.

    The feature *kind* comes from the frozen preprocessor rather than from the
    reference values, so a profile cannot decide a column is categorical when the
    model was fitted treating it as numeric.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    kind: ReferenceFeatureKind
    #: How this partition was constructed.  ``quantile`` is the ordinary numeric
    #: case; ``constant`` and ``all_null`` are the degenerate ones, recorded so a
    #: reader can tell a two-cell partition from a twenty-cell one that happens
    #: to be nearly empty.
    partition_kind: str
    bins: tuple[ReferenceBin, ...]
    reference_row_count: int = Field(ge=0)
    reference_null_count: int = Field(ge=0)
    reference_null_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_partition(self) -> Self:
        """A partition names each cell once and always carries a null cell."""
        labels = [item.cell for item in self.bins]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{self.feature!r} repeats a bin label")
        if NULL_BIN not in labels:
            raise ValueError(
                f"{self.feature!r} declares no null bin; a missing observation "
                f"must have somewhere deterministic to land"
            )
        if len(labels) < 2:
            raise ValueError(
                f"{self.feature!r} declares a single cell; nothing can move "
                f"within a partition that has no other cell to move to"
            )
        if self.reference_null_count > self.reference_row_count:
            raise ValueError(f"{self.feature!r} counts more nulls than rows")
        return self


class PredictionReference(BaseModel):
    """One published prediction quantity's frozen partition and distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quantity: PredictionDriftQuantity
    partition_kind: str
    bins: tuple[ReferenceBin, ...]
    reference_row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def check_partition(self) -> Self:
        """A partition names each cell once and offers somewhere else to go."""
        labels = [item.cell for item in self.bins]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{str(self.quantity)!r} repeats a bin label")
        if len(labels) < 2:
            raise ValueError(f"{str(self.quantity)!r} declares a single cell")
        return self


class MLReferenceProfile(SealedModel):
    """The immutable baseline a later population is compared against.

    Sealed, so a profile edited after capture is refused rather than believed,
    and carrying no path, no host, and no timestamp, so two captures of the same
    reference population in two directories produce identical bytes.
    """

    fingerprint_field: ClassVar[str] = "reference_profile_fingerprint"
    schema_version_field: ClassVar[str] = "reference_schema_version"
    schema_version: ClassVar[str] = REFERENCE_SCHEMA_VERSION
    record_label: ClassVar[str] = "reference profile"

    reference_schema_version: str = REFERENCE_SCHEMA_VERSION

    #: Derived from the content below, so the identity moves when -- and only
    #: when -- something that changes a comparison moves.
    reference_profile_id: str

    reference_split: MLSplit
    reference_row_count: int = Field(ge=1)

    champion_lock_fingerprint: Sha256Hex
    champion_scope_key: Sha256Hex
    catalog_model_id: str
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    feature_catalog_fingerprint: Sha256Hex
    allowlist_fingerprint: Sha256Hex
    eligible_feature_list_fingerprint: Sha256Hex
    required_feature_schema_version: str

    #: The digest of exactly the reference rows the feature partitions were cut
    #: from, taken by the inference loader from the feature and split tables
    #: alone.  A label cannot reach it, because no label was read.
    reference_population_fingerprint: Sha256Hex
    #: The publication the prediction partitions were cut from, when one was
    #: supplied.  ``None`` leaves the profile feature-only, which is a smaller
    #: profile rather than a broken one.
    reference_prediction_id: str | None = None
    reference_prediction_manifest_fingerprint: Sha256Hex | None = None
    reference_score_kind: ScoreKind | None = None

    drift_config_fingerprint: Sha256Hex
    quantile_count: int = Field(ge=2)

    features: tuple[FeatureReference, ...]
    predictions: tuple[PredictionReference, ...] = ()

    reference_profile_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_profile(self) -> Self:
        """The source split is eligible and the prediction lineage is whole."""
        if self.reference_split not in DRIFT_REFERENCE_ELIGIBLE_SPLITS:
            raise ValueError(
                f"{str(self.reference_split)!r} may not be a drift reference; a "
                f"baseline is captured from the training population, never from "
                f"the locked evaluation population or the rows that chose the "
                f"operating point"
            )
        if not self.features:
            raise ValueError("a reference profile describes at least one feature")
        names = [item.feature for item in self.features]
        if len(set(names)) != len(names):
            raise ValueError("a reference profile describes each feature once")
        if names != sorted(names):
            raise ValueError("features must be given in sorted order")
        quantities = [item.quantity for item in self.predictions]
        if len(set(quantities)) != len(quantities):
            raise ValueError("a reference profile describes each quantity once")
        present = (
            self.reference_prediction_id,
            self.reference_prediction_manifest_fingerprint,
            self.reference_score_kind,
        )
        flags = [value is not None for value in present]
        if any(flags) and not all(flags):
            raise ValueError(
                "a prediction reference is named in full or not at all; a "
                "partial lineage is a lineage nobody can check"
            )
        if bool(self.predictions) != all(flags):
            raise ValueError(
                "prediction partitions and the publication they were cut from "
                "are declared together, or neither is"
            )
        return self


def _assert_no_outcome_field() -> None:
    """Refuse at import a schema here that declares a prohibited field name."""
    for model in (
        ReferenceBin,
        FeatureReference,
        PredictionReference,
        MLReferenceProfile,
    ):
        offending = prohibited_metadata_fields(set(model.model_fields))
        if offending:
            raise AssertionError(
                f"{model.__name__} declares prohibited field(s) {list(offending)}"
            )


_assert_no_outcome_field()


# ---------------------------------------------------------------------------
# Partition construction
# ---------------------------------------------------------------------------


def _proportions(counts: Sequence[int], total: int) -> tuple[float, ...]:
    """Return each count's share of *total*, or all-zero when there is no total."""
    if total <= 0:
        return tuple(0.0 for _ in counts)
    return tuple(quantize(count / total) for count in counts)


def quantile_edges(values: Sequence[float], *, count: int) -> tuple[float, ...]:
    """Return the interior edges of a *count*-way quantile partition.

    Interior only: the outermost cells stay open-ended so a later value below
    the reference minimum or above its maximum lands in a real bin instead of
    being silently dropped.  Duplicates are removed, because a heavily tied
    column produces repeated quantiles and a zero-width cell is not a cell.

    Computed with a plain sorted-index rule rather than an interpolating
    quantile: an edge that depends on a library's interpolation convention is an
    edge that can move between releases, and these are frozen for years.
    """
    if count < 2:
        raise ModelNotReadyError("a quantile partition needs at least two cells")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return ()
    edges: list[float] = []
    for index in range(1, count):
        position = (index * len(ordered)) // count
        if position <= 0 or position >= len(ordered):
            continue
        edges.append(quantize(ordered[position]))
    unique: list[float] = []
    for edge in edges:
        if not unique or edge != unique[-1]:
            unique.append(edge)
    return tuple(unique)


def numeric_bin_labels(edges: Sequence[float]) -> tuple[str, ...]:
    """Return the ordered labels an interior edge list partitions into."""
    return tuple(f"bin_{index:02d}" for index in range(len(edges) + 1))


def numeric_bin_index(value: float, edges: Sequence[float]) -> int:
    """Return the cell *value* falls in under ``(lower, upper]`` semantics.

    A value equal to an edge belongs to the cell below it, which is the same
    right-closed convention the exact PR-AUC uses for score levels.  Stating it
    once and reusing it is what keeps two artifacts in this project from
    disagreeing about a boundary.
    """
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def _numeric_reference(
    feature: str, values: Sequence[Any], *, quantile_count: int
) -> FeatureReference:
    """Return the frozen partition for one numeric feature."""
    observed = [float(value) for value in values if value is not None]
    nulls = len(values) - len(observed)

    if not observed:
        # Nothing was ever observed, so there is no distribution to cut. The
        # partition still has to exist, because a later population that *does*
        # observe values has drifted and must be able to say so.
        bins = _labelled_bins([(NULL_BIN, nulls), (UNKNOWN_BIN, 0)], total=len(values))
        return FeatureReference(
            feature=feature,
            kind=ReferenceFeatureKind.NUMERIC,
            partition_kind="all_null",
            bins=bins,
            reference_row_count=len(values),
            reference_null_count=nulls,
            reference_null_rate=quantize(_rate(nulls, len(values))),
        )

    edges = quantile_edges(observed, count=quantile_count)
    if len(set(observed)) < 2 or not edges:
        # Every observed value is identical, so a quantile grid has nothing to
        # cut.  Deduplicated edges would leave a single interior edge equal to
        # that value, and the resulting partition could not tell a later value
        # *below* it from the constant itself.  A two-way "is it still that
        # value" partition is degenerate on purpose and does detect that.
        constant = quantize(observed[0])
        same = sum(1 for value in observed if value == constant)
        cells = [
            (f"equals_{constant!r}", same),
            (UNKNOWN_BIN, len(observed) - same),
            (NULL_BIN, nulls),
        ]
        return FeatureReference(
            feature=feature,
            kind=ReferenceFeatureKind.NUMERIC,
            partition_kind="constant",
            bins=_labelled_bins(cells, total=len(values)),
            reference_row_count=len(values),
            reference_null_count=nulls,
            reference_null_rate=quantize(_rate(nulls, len(values))),
        )

    labels = numeric_bin_labels(edges)
    counts = [0] * len(labels)
    for value in observed:
        counts[numeric_bin_index(value, edges)] += 1
    proportions = _proportions([*counts, nulls], len(values))
    intervals = [
        (
            None if index == 0 else edges[index - 1],
            None if index == len(edges) else edges[index],
        )
        for index in range(len(labels))
    ]
    bins = (
        *(
            ReferenceBin(
                cell=cell,
                lower=lower,
                upper=upper,
                reference_count=count,
                reference_proportion=proportion,
            )
            for cell, (lower, upper), count, proportion in zip(
                labels, intervals, counts, proportions[:-1], strict=True
            )
        ),
        ReferenceBin(
            cell=NULL_BIN,
            reference_count=nulls,
            reference_proportion=proportions[-1],
        ),
    )
    return FeatureReference(
        feature=feature,
        kind=ReferenceFeatureKind.NUMERIC,
        partition_kind="quantile",
        bins=bins,
        reference_row_count=len(values),
        reference_null_count=nulls,
        reference_null_rate=quantize(_rate(nulls, len(values))),
    )


def _boolean_reference(feature: str, values: Sequence[Any]) -> FeatureReference:
    """Return the frozen three-cell partition for one boolean feature."""
    true_count = sum(1 for value in values if value is True)
    false_count = sum(1 for value in values if value is False)
    nulls = len(values) - true_count - false_count
    return FeatureReference(
        feature=feature,
        kind=ReferenceFeatureKind.BOOLEAN,
        partition_kind="boolean",
        bins=_labelled_bins(
            [("false", false_count), ("true", true_count), (NULL_BIN, nulls)],
            total=len(values),
        ),
        reference_row_count=len(values),
        reference_null_count=nulls,
        reference_null_rate=quantize(_rate(nulls, len(values))),
    )


def _categorical_reference(
    feature: str, values: Sequence[Any], encoding: Any
) -> FeatureReference:
    """Return the frozen partition for one categorical feature.

    The vocabulary is the frozen preprocessor's, not the reference data's.  A
    category the preprocessor retained but the reference rows never carried is
    still a cell -- with zero expected mass -- because the model has a column for
    it, and a later population that starts using it has drifted.
    """
    vocabulary = tuple(encoding.categories)
    counts = dict.fromkeys(vocabulary, 0)
    other = 0
    unknown = 0
    nulls = 0
    rare = set(encoding.rare_categories)
    for value in values:
        if value is None:
            nulls += 1
        elif value in counts:
            counts[value] += 1
        elif value in rare and encoding.emits_rare_bucket:
            other += 1
        else:
            unknown += 1
    cells: list[tuple[str, int]] = [(name, counts[name]) for name in vocabulary]
    if encoding.emits_rare_bucket:
        cells.append((OTHER_BIN, other))
    cells += [(UNKNOWN_BIN, unknown), (NULL_BIN, nulls)]
    return FeatureReference(
        feature=feature,
        kind=ReferenceFeatureKind.CATEGORICAL,
        partition_kind="vocabulary",
        bins=_labelled_bins(cells, total=len(values)),
        reference_row_count=len(values),
        reference_null_count=nulls,
        reference_null_rate=quantize(_rate(nulls, len(values))),
    )


def _labelled_bins(
    cells: Sequence[tuple[str, int]], *, total: int
) -> tuple[ReferenceBin, ...]:
    """Return bins for labelled counts, sharing them over *total*."""
    proportions = _proportions([count for _, count in cells], total)
    return tuple(
        ReferenceBin(cell=cell, reference_count=count, reference_proportion=proportion)
        for (cell, count), proportion in zip(cells, proportions, strict=True)
    )


def _rate(numerator: int, denominator: int) -> float:
    """Return a share, treating an empty population as zero rather than failing."""
    return 0.0 if denominator <= 0 else numerator / denominator


def _score_reference(
    quantity: PredictionDriftQuantity,
    values: Sequence[float],
    *,
    quantile_count: int,
) -> PredictionReference:
    """Return the frozen quantile partition for one continuous score."""
    edges = quantile_edges(values, count=quantile_count)
    if not edges:
        constant = float(values[0]) if values else 0.0
        same = sum(1 for value in values if float(value) == constant)
        return PredictionReference(
            quantity=quantity,
            partition_kind="constant",
            bins=_labelled_bins(
                [(f"equals_{constant!r}", same), (UNKNOWN_BIN, len(values) - same)],
                total=len(values),
            ),
            reference_row_count=len(values),
        )
    labels = numeric_bin_labels(edges)
    counts = [0] * len(labels)
    for value in values:
        counts[numeric_bin_index(float(value), edges)] += 1
    proportions = _proportions(counts, len(values))
    intervals = [
        (
            None if index == 0 else edges[index - 1],
            None if index == len(edges) else edges[index],
        )
        for index in range(len(labels))
    ]
    return PredictionReference(
        quantity=quantity,
        partition_kind="quantile",
        bins=tuple(
            ReferenceBin(
                cell=cell,
                lower=lower,
                upper=upper,
                reference_count=count,
                reference_proportion=proportion,
            )
            for cell, (lower, upper), count, proportion in zip(
                labels, intervals, counts, proportions, strict=True
            )
        ),
        reference_row_count=len(values),
    )


def _rate_reference(
    quantity: PredictionDriftQuantity,
    *,
    positive_label: str,
    negative_label: str,
    positives: int,
    total: int,
) -> PredictionReference:
    """Return the frozen two-cell partition behind a rate."""
    return PredictionReference(
        quantity=quantity,
        partition_kind="rate",
        bins=_labelled_bins(
            [(negative_label, total - positives), (positive_label, positives)],
            total=total,
        ),
        reference_row_count=total,
    )


def _class_reference(
    class_order: Sequence[str], predicted: Sequence[str]
) -> PredictionReference:
    """Return the frozen partition over the category head's own class space."""
    space = [*sorted(class_order), UNKNOWN_CATEGORY]
    counts = dict.fromkeys(space, 0)
    for value in predicted:
        counts[value] = counts.get(value, 0) + 1
    unknown_labels = sorted(set(counts) - set(space))
    if unknown_labels:
        raise ModelNotReadyError(
            f"the reference publication predicted {len(unknown_labels)} class(es) "
            f"the frozen head does not declare; the class space is the head's, "
            f"not the data's"
        )
    return PredictionReference(
        quantity=PredictionDriftQuantity.CATEGORY_PREDICTED_CLASS,
        partition_kind="class_space",
        bins=_labelled_bins(
            [(name, counts[name]) for name in space], total=len(predicted)
        ),
        reference_row_count=len(predicted),
    )


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def build_reference_profile(
    *,
    lock: Any,
    preprocessor: Any,
    reference: Any,
    reference_split: MLSplit,
    required_feature_schema_version: str,
    drift_config: Any,
    drift_config_fingerprint: str,
    prediction_manifest: Any | None = None,
    binary: Sequence[Any] = (),
    category: Sequence[Any] | None = None,
    anomaly: Sequence[Any] | None = None,
) -> MLReferenceProfile:
    """Capture the immutable baseline from the approved reference population.

    Args:
        lock: the frozen :class:`~password_attack_detector.ml.champion.ChampionLock`.
        preprocessor: that champion's frozen preprocessor, which decides how each
            raw feature is partitioned.
        reference: the loaded
            :class:`~password_attack_detector.ml.dataset.InferenceDataset` for the
            reference split.  Label-free by construction.
        reference_split: the split those rows came from.  Anything outside
            :data:`~password_attack_detector.ml.enums.DRIFT_REFERENCE_ELIGIBLE_SPLITS`
            is refused.
        required_feature_schema_version: the feature contract version bound in.
        drift_config: the reviewed drift configuration.
        drift_config_fingerprint: that configuration's digest.
        prediction_manifest: the reference publication, when prediction
            partitions are to be captured.
        binary: the reference publication's binary rows.
        category: its category rows, when a head was frozen.
        anomaly: its experimental anomaly rows, when the probe ran.

    Raises:
        ModelNotReadyError: on an ineligible split, a feature/preprocessor
            mismatch, a publication produced by a different champion, or a
            reference population the configuration considers too small.
    """
    if reference_split not in DRIFT_REFERENCE_ELIGIBLE_SPLITS:
        raise ModelNotReadyError(
            f"a drift reference may not be captured from {str(reference_split)!r}; "
            f"the reviewed source is the training population"
        )
    if reference.scope is not reference_split:
        raise ModelNotReadyError(
            "the loaded reference population is not the split it was requested for"
        )
    if preprocessor.fingerprint() != lock.preprocessor_fingerprint:
        raise ModelNotReadyError(
            "the supplied preprocessor is not the one the champion was frozen "
            "with; a profile cut with a different encoding would describe a "
            "matrix this model never sees"
        )

    frame = reference.frame
    rows = frame.row_count
    if rows < drift_config.min_reference_rows:
        raise ModelNotReadyError(
            f"the reference population carries {rows} row(s), below the "
            f"configured floor of {drift_config.min_reference_rows}; a baseline "
            f"cut from fewer rows would make every later comparison look "
            f"conclusive"
        )

    numeric = {item.feature: item for item in preprocessor.numeric_imputations}
    boolean = {item.feature: item for item in preprocessor.boolean_encodings}
    categorical = {item.feature: item for item in preprocessor.categorical_encodings}
    described = set(numeric) | set(boolean) | set(categorical)
    if set(frame.feature_names) != described:
        raise ModelNotReadyError(
            "the reference population and the frozen preprocessor describe "
            "different feature sets"
        )
    for encoding in categorical.values():
        collide = sorted(set(encoding.categories) & _RESERVED_BINS)
        if collide:
            raise ModelNotReadyError(
                f"feature {encoding.feature!r} carries category value(s) "
                f"{collide} that collide with a reserved partition label"
            )

    columns: dict[str, list[Any]] = {name: [] for name in frame.feature_names}
    for values in frame.feature_matrix:
        for name, value in zip(frame.feature_names, values, strict=True):
            columns[name].append(value)

    captured = [
        (
            _numeric_reference(
                name, columns[name], quantile_count=drift_config.quantile_count
            )
            if name in numeric
            else (
                _boolean_reference(name, columns[name])
                if name in boolean
                else _categorical_reference(name, columns[name], categorical[name])
            )
        )
        for name in frame.feature_names
    ]
    captured.sort(key=lambda item: item.feature)
    features = tuple(captured)

    predictions: tuple[PredictionReference, ...] = ()
    prediction_id: str | None = None
    manifest_fingerprint: str | None = None
    score_kind: ScoreKind | None = None
    if prediction_manifest is not None:
        prediction_id, manifest_fingerprint, score_kind, predictions = (
            _prediction_references(
                lock=lock,
                manifest=prediction_manifest,
                binary=binary,
                category=category,
                anomaly=anomaly,
                quantile_count=drift_config.quantile_count,
            )
        )

    identity = _profile_identity(
        lock=lock,
        reference=reference,
        reference_split=reference_split,
        drift_config_fingerprint=drift_config_fingerprint,
        features=features,
        predictions=predictions,
        manifest_fingerprint=manifest_fingerprint,
    )
    return MLReferenceProfile.seal(
        reference_profile_id=identity,
        reference_split=reference_split,
        reference_row_count=rows,
        champion_lock_fingerprint=lock.lock_fingerprint,
        champion_scope_key=lock.scope_key,
        catalog_model_id=lock.catalog_model_id,
        model_id=lock.model_id,
        model_content_fingerprint=lock.model_content_fingerprint,
        preprocessor_fingerprint=lock.preprocessor_fingerprint,
        feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
        allowlist_fingerprint=lock.allowlist_fingerprint,
        eligible_feature_list_fingerprint=lock.eligible_feature_list_fingerprint,
        required_feature_schema_version=required_feature_schema_version,
        reference_population_fingerprint=reference.inference_input_fingerprint,
        reference_prediction_id=prediction_id,
        reference_prediction_manifest_fingerprint=manifest_fingerprint,
        reference_score_kind=score_kind,
        drift_config_fingerprint=drift_config_fingerprint,
        quantile_count=drift_config.quantile_count,
        features=features,
        predictions=predictions,
    )


def _prediction_references(
    *,
    lock: Any,
    manifest: Any,
    binary: Sequence[Any],
    category: Sequence[Any] | None,
    anomaly: Sequence[Any] | None,
    quantile_count: int,
) -> tuple[str, str, ScoreKind, tuple[PredictionReference, ...]]:
    """Return the prediction partitions cut from the reference publication."""
    if manifest.lineage.champion_lock_fingerprint != lock.lock_fingerprint:
        raise ModelNotReadyError(
            "the reference publication was produced by a different champion "
            "than the one being profiled"
        )
    if manifest.scope not in DRIFT_REFERENCE_ELIGIBLE_SPLITS:
        raise ModelNotReadyError(
            f"the reference publication scores {str(manifest.scope)!r}; a "
            f"prediction baseline is captured from the training population"
        )
    if not binary:
        raise ModelNotReadyError(
            "the reference publication carries no binary rows to profile"
        )

    kind = manifest.lineage.binary_score_kind
    references: list[PredictionReference] = [
        _rate_reference(
            PredictionDriftQuantity.FLAGGED_MALICIOUS_RATE,
            positive_label="flagged",
            negative_label="not_flagged",
            positives=sum(1 for row in binary if row.flagged_malicious),
            total=len(binary),
        ),
        _score_reference(
            PredictionDriftQuantity.DECISION_SCORE,
            [row.malicious_decision_score for row in binary],
            quantile_count=quantile_count,
        ),
    ]
    if is_probability(kind):
        references.append(
            _score_reference(
                PredictionDriftQuantity.CALIBRATED_PROBABILITY,
                [
                    row.malicious_probability
                    for row in binary
                    if row.malicious_probability is not None
                ],
                quantile_count=quantile_count,
            )
        )
    if category:
        class_order = manifest.lineage.category_class_order or ()
        predicted = [row.predicted_category for row in category]
        references.append(_class_reference(class_order, predicted))
        references.append(
            _rate_reference(
                PredictionDriftQuantity.CATEGORY_UNKNOWN_RATE,
                positive_label="unknown",
                negative_label="known",
                positives=sum(1 for value in predicted if value == UNKNOWN_CATEGORY),
                total=len(predicted),
            )
        )
    if anomaly:
        references.append(
            _score_reference(
                PredictionDriftQuantity.ANOMALY_SCORE,
                [row.anomaly_score for row in anomaly],
                quantile_count=quantile_count,
            )
        )
    references.sort(key=lambda item: str(item.quantity))
    return (
        manifest.prediction_id,
        manifest.prediction_manifest_fingerprint,
        kind,
        tuple(references),
    )


def _profile_identity(
    *,
    lock: Any,
    reference: Any,
    reference_split: MLSplit,
    drift_config_fingerprint: str,
    features: Sequence[FeatureReference],
    predictions: Sequence[PredictionReference],
    manifest_fingerprint: str | None,
) -> str:
    """Return the content-derived identity of one captured profile."""
    return digest(
        {
            "champion_lock_fingerprint": lock.lock_fingerprint,
            "drift_config_fingerprint": drift_config_fingerprint,
            "features": [item.model_dump(mode="json") for item in features],
            "predictions": [item.model_dump(mode="json") for item in predictions],
            "reference_population_fingerprint": (reference.inference_input_fingerprint),
            "reference_prediction_manifest_fingerprint": manifest_fingerprint,
            "reference_schema_version": REFERENCE_SCHEMA_VERSION,
            "reference_split": str(reference_split),
        }
    )


def feature_reference_index(
    profile: MLReferenceProfile,
) -> Mapping[str, FeatureReference]:
    """Return the profile's feature partitions keyed by feature name."""
    return {item.feature: item for item in profile.features}


def prediction_reference_index(
    profile: MLReferenceProfile,
) -> Mapping[PredictionDriftQuantity, PredictionReference]:
    """Return the profile's prediction partitions keyed by quantity."""
    return {item.quantity: item for item in profile.predictions}


def reference_profile_to_markdown(profile: MLReferenceProfile) -> str:
    """Render the profile's identity and shape, without its expected masses.

    Bin *labels* and counts are the profile's content and stay in the JSON
    artifact.  What a human reads is the identity, the lineage, and the shape:
    a table of two hundred numeric cells is not review material, and rendering
    it would put the reference distribution into a document that gets pasted
    around.
    """
    lines = [
        "# ML reference profile",
        "",
        "The immutable baseline drift is measured against. Captured from the "
        "training population only, before any evaluation, and frozen: incoming "
        "data never redefines a bin edge, a category, or an expected share.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Reference schema | {profile.reference_schema_version} |",
        f"| Profile id | `{profile.reference_profile_id}` |",
        f"| Profile fingerprint | `{profile.reference_profile_fingerprint}` |",
        f"| Reference split | {profile.reference_split} |",
        f"| Reference rows | {profile.reference_row_count:,} |",
        f"| Population fingerprint | `{profile.reference_population_fingerprint}` |",
        f"| Champion lock | `{profile.champion_lock_fingerprint}` |",
        f"| Catalog model | {profile.catalog_model_id} |",
        f"| Preprocessor | `{profile.preprocessor_fingerprint}` |",
        f"| Allowlist | `{profile.allowlist_fingerprint}` |",
        f"| Eligible features | `{profile.eligible_feature_list_fingerprint}` |",
        f"| Feature catalog | `{profile.feature_catalog_fingerprint}` |",
        f"| Drift config | `{profile.drift_config_fingerprint}` |",
        f"| Quantile cells | {profile.quantile_count:,} |",
        f"| Profiled features | {len(profile.features):,} |",
        f"| Profiled prediction quantities | {len(profile.predictions):,} |",
        f"| Reference prediction | "
        f"{profile.reference_prediction_id or 'not captured'} |",
        f"| Reference score kind | {profile.reference_score_kind or 'not captured'} |",
        "",
        "## Partition shape",
        "",
        "| Feature | Kind | Partition | Cells | Null rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in profile.features:
        lines.append(
            f"| `{item.feature}` | {item.kind} | {item.partition_kind} | "
            f"{len(item.bins):,} | {item.reference_null_rate:.6f} |"
        )
    lines += [
        "",
        "## Prediction partitions",
        "",
        "| Quantity | Partition | Cells | Reference rows |",
        "| --- | --- | --- | --- |",
    ]
    for prediction in profile.predictions:
        lines.append(
            f"| {prediction.quantity} | {prediction.partition_kind} | "
            f"{len(prediction.bins):,} | {prediction.reference_row_count:,} |"
        )
    lines += [
        "",
        "A reference profile is a description of a population. It is not a "
        "fitted parameter of the champion, it cannot change what the champion "
        "does, and no label was read to build it.",
        "",
    ]
    return "\n".join(lines)
