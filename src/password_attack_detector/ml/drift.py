"""Comparing a later population against the frozen reference profile.

Drift is **monitoring evidence, not model correctness**.  Everything measured
here is computed without a single label, so nothing in this module can say a
model got worse -- only that the rows it is being shown, or the decisions it is
making about them, no longer look like the population it was baselined on.  Those
are different findings, and one is not weak evidence for the other: a feature
distribution can move because an organisation onboarded a new office, and a
flagged rate can move because an attack actually happened.

**Feature drift and prediction drift stay apart.**  Two result types, two
sections of the report, two aggregate statuses.  A single blended number would
average a shift in the inputs against a shift in the outputs and say neither.

**Nothing here retrains, promotes, or changes anything.**  There is no code path
from a drift finding to a model, a threshold, a fusion strategy, a champion lock,
or an evaluation record.  This module imports nothing that can write one, opens
no artifact for writing, and returns a report.  A finding may be a reason for a
human to investigate; it is never an action.

**The partition is the reference's, always.**  Every bin edge, category, class,
and expected proportion is read from the frozen profile.  Incoming values are
*assigned* to cells and never define one: an unseen category lands in the
reference's unknown cell, a value beyond the reference range lands in an
open-ended outer cell, and a null lands in the null cell.  Nothing is discarded
for being unexpected, because an observation nobody counted is the one worth
counting.

**Insufficient is not stable, and unavailable is not zero.**  A comparison over
too few rows on either side reports
:attr:`~password_attack_detector.ml.enums.DriftStatus.INCONCLUSIVE`; a quantity
that does not exist on one side reports
:attr:`~password_attack_detector.ml.enums.DriftStatus.UNAVAILABLE`.  Neither is
``NO_DRIFT``, because a monitor that reports stability when it measured nothing
is worse than no monitor.

**One thresholded measure.**  The population stability index over a frozen
partition is the only measure the reviewed configuration declares warn and alert
values for, and a second thresholded family would need thresholds nobody has
chosen.  Null rates and unknown-category rates are still reported -- as fields on
the result, and as dedicated cells inside the very partition the index is taken
over, so a shift in either moves the number rather than hiding beside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ModelNotReadyError
from password_attack_detector.ml.calibration import SealedModel, digest, quantize
from password_attack_detector.ml.enums import (
    UNKNOWN_CATEGORY,
    DriftMetric,
    DriftStatus,
    MLSplit,
    PredictionDriftQuantity,
    ReferenceFeatureKind,
    is_probability,
)
from password_attack_detector.ml.reference import (
    NULL_BIN,
    OTHER_BIN,
    UNKNOWN_BIN,
    FeatureReference,
    MLReferenceProfile,
    PredictionReference,
    numeric_bin_index,
)
from password_attack_detector.ml.schemas import Sha256Hex, prohibited_metadata_fields

__all__ = [
    "DRIFT_SCHEMA_VERSION",
    "PSI_PROPORTION_FLOOR",
    "DriftManifest",
    "FeatureDriftResult",
    "MLDriftReport",
    "PredictionDriftResult",
    "build_drift_manifest",
    "compare_feature_population",
    "compare_prediction_population",
    "drift_report_to_markdown",
    "population_stability_index",
]

#: Version of the drift artifact contract.
DRIFT_SCHEMA_VERSION: Final = "1.0.0"

#: The share a zero proportion is floored to before the index is summed.
#:
#: A bin the reference never filled makes the ratio inside the logarithm
#: infinite, and dropping such bins would make an entirely new category read as
#: perfect stability.  Flooring at a fixed, declared constant keeps the term
#: finite, keeps it large, and -- because the constant does not depend on either
#: population's size -- keeps two runs over the same data equal.
PSI_PROPORTION_FLOOR: Final[float] = 1e-6

#: Stable reason codes.  A caller may branch on these; they do not change with
#: a message rewrite.
REASON_BELOW_WARN: Final = "psi_below_warn_threshold"
REASON_AT_WARN: Final = "psi_at_or_above_warn_threshold"
REASON_AT_ALERT: Final = "psi_at_or_above_alert_threshold"
REASON_REFERENCE_SUPPORT: Final = "insufficient_reference_support"
REASON_INCOMING_SUPPORT: Final = "insufficient_incoming_support"
REASON_ABSENT_INCOMING: Final = "quantity_absent_from_incoming"
REASON_ABSENT_REFERENCE: Final = "quantity_absent_from_reference"


def population_stability_index(
    reference: Sequence[float], incoming: Sequence[float]
) -> float:
    """Return the population stability index between two aligned proportion vectors.

    ``sum((incoming - reference) * ln(incoming / reference))`` over the cells of
    a partition both vectors describe, with each share floored at
    :data:`PSI_PROPORTION_FLOOR` first.  The floor is applied and the vectors are
    deliberately **not** renormalised afterwards: renormalising would make a
    cell's contribution depend on how many other cells happened to be empty.

    Raises:
        ModelNotReadyError: if the two vectors describe different partitions.
    """
    if len(reference) != len(incoming):
        raise ModelNotReadyError(
            "the reference and incoming distributions describe different "
            "partitions; an index summed over mismatched cells is meaningless"
        )
    total = 0.0
    for expected, observed in zip(reference, incoming, strict=True):
        floor_expected = max(float(expected), PSI_PROPORTION_FLOOR)
        floor_observed = max(float(observed), PSI_PROPORTION_FLOOR)
        total += (floor_observed - floor_expected) * math.log(
            floor_observed / floor_expected
        )
    return quantize(total)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class _DriftResult(BaseModel):
    """Fields every drift result carries, whatever it describes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: DriftMetric = DriftMetric.POPULATION_STABILITY_INDEX
    #: How many rows the reference partition was cut from.
    reference_support: int = Field(ge=0)
    #: How many rows were assigned to it.
    incoming_support: int = Field(ge=0)
    #: The index, or ``None`` when nothing could be computed.  Never a zero
    #: standing in for an absence.
    observed_value: float | None = None
    warn_threshold: float | None = None
    alert_threshold: float | None = None
    status: DriftStatus
    reason_code: str

    @model_validator(mode="after")
    def check_result(self) -> Self:
        """A measured status carries a value; a refusal carries none."""
        measured = self.status in {
            DriftStatus.NO_DRIFT,
            DriftStatus.DRIFT_WARNING,
            DriftStatus.DRIFT_DETECTED,
        }
        if measured and self.observed_value is None:
            raise ValueError(
                "a measured drift status names the value it was measured at"
            )
        if not measured and self.observed_value is not None:
            raise ValueError(
                "an inconclusive or unavailable result reports no value; a "
                "number beside a refusal reads as a measurement"
            )
        if self.observed_value is not None and not math.isfinite(self.observed_value):
            raise ValueError("observed_value must be finite")
        if measured and (self.warn_threshold is None or self.alert_threshold is None):
            raise ValueError(
                "a thresholded status names the thresholds it was decided against"
            )
        if not self.reason_code.strip():
            raise ValueError("every drift result carries a stable reason code")
        return self


class FeatureDriftResult(_DriftResult):
    """One raw feature's comparison against its frozen partition.

    ``null_rate_delta`` and ``unknown_rate_delta`` are reported beside the index
    rather than thresholded on their own: the null and unknown cells are part of
    the partition the index is summed over, so a shift in either already moves
    the thresholded number.
    """

    feature: str
    kind: ReferenceFeatureKind
    partition_kind: str
    reference_null_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    incoming_null_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    null_rate_delta: float | None = None
    #: The share of incoming values the reference vocabulary cannot represent.
    #: ``None`` for a numeric or boolean feature, which has no vocabulary.
    reference_unknown_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    incoming_unknown_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    unknown_rate_delta: float | None = None

    @model_validator(mode="after")
    def check_feature_result(self) -> Self:
        """The deltas agree with the rates they are differences of."""
        if not self.feature.strip():
            raise ValueError("a feature drift result names its feature")
        for reference, incoming, delta, what in (
            (
                self.reference_null_rate,
                self.incoming_null_rate,
                self.null_rate_delta,
                "null",
            ),
            (
                self.reference_unknown_rate,
                self.incoming_unknown_rate,
                self.unknown_rate_delta,
                "unknown",
            ),
        ):
            present = [value is not None for value in (reference, incoming, delta)]
            if any(present) and not all(present):
                raise ValueError(
                    f"the {what} rates and their delta are reported together"
                )
            if all(present):
                assert reference is not None and incoming is not None
                assert delta is not None
                if abs((incoming - reference) - delta) > 1e-9:
                    raise ValueError(
                        f"the reported {what}-rate delta is not the difference "
                        f"of the reported rates"
                    )
        if (
            self.kind is not ReferenceFeatureKind.CATEGORICAL
            and self.reference_unknown_rate is not None
        ):
            raise ValueError(
                "only a categorical feature has a vocabulary to fall outside of"
            )
        return self


class PredictionDriftResult(_DriftResult):
    """One published prediction quantity's comparison against its partition."""

    quantity: PredictionDriftQuantity
    partition_kind: str
    #: The rate behind a two-cell partition, on each side.  ``None`` for a
    #: quantile partition, which is a distribution rather than a rate.
    reference_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    incoming_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    rate_delta: float | None = None

    @model_validator(mode="after")
    def check_prediction_result(self) -> Self:
        """A reported rate delta is the difference of the reported rates."""
        present = [
            value is not None
            for value in (self.reference_rate, self.incoming_rate, self.rate_delta)
        ]
        if any(present) and not all(present):
            raise ValueError("the rates and their delta are reported together")
        if all(present):
            assert self.reference_rate is not None and self.incoming_rate is not None
            assert self.rate_delta is not None
            if abs((self.incoming_rate - self.reference_rate) - self.rate_delta) > 1e-9:
                raise ValueError(
                    "the reported rate delta is not the difference of the rates"
                )
        return self


class MLDriftReport(SealedModel):
    """The aggregate drift finding: statuses and counts, no rows.

    Two aggregate statuses, kept apart on purpose.  There is no combined verdict,
    because a feature shift and an output shift are different findings and a
    reader who was handed one number would have to guess which they had.
    """

    fingerprint_field: ClassVar[str] = "drift_report_fingerprint"
    schema_version_field: ClassVar[str] = "drift_schema_version"
    schema_version: ClassVar[str] = DRIFT_SCHEMA_VERSION
    record_label: ClassVar[str] = "drift report"

    drift_schema_version: str = DRIFT_SCHEMA_VERSION

    reference_profile_id: str
    reference_profile_fingerprint: Sha256Hex
    incoming_scope: MLSplit
    incoming_row_count: int = Field(ge=0)
    incoming_population_fingerprint: Sha256Hex
    incoming_prediction_id: str | None = None
    incoming_prediction_manifest_fingerprint: Sha256Hex | None = None

    warn_threshold: float
    alert_threshold: float
    min_support: int = Field(ge=1)

    feature_status: DriftStatus
    prediction_status: DriftStatus
    features: tuple[FeatureDriftResult, ...]
    predictions: tuple[PredictionDriftResult, ...] = ()

    drift_report_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_report(self) -> Self:
        """Each subject appears once, in a deterministic order, and the aggregates hold."""
        names = [item.feature for item in self.features]
        if len(set(names)) != len(names):
            raise ValueError("a drift report describes each feature once")
        if names != sorted(names):
            raise ValueError("feature results must be given in sorted order")
        quantities = [str(item.quantity) for item in self.predictions]
        if len(set(quantities)) != len(quantities):
            raise ValueError("a drift report describes each quantity once")
        if quantities != sorted(quantities):
            raise ValueError("prediction results must be given in sorted order")
        if self.warn_threshold >= self.alert_threshold:
            raise ValueError("the warning threshold sits below the alert threshold")
        expected = aggregate_status([item.status for item in self.features])
        if self.feature_status is not expected:
            raise ValueError(
                "the aggregate feature status is not the one its results imply"
            )
        expected_predictions = aggregate_status(
            [item.status for item in self.predictions]
        )
        if self.prediction_status is not expected_predictions:
            raise ValueError(
                "the aggregate prediction status is not the one its results imply"
            )
        return self

    @property
    def drifted_feature_count(self) -> int:
        """Return how many features reported a warning or an alert."""
        return sum(
            1
            for item in self.features
            if item.status in {DriftStatus.DRIFT_WARNING, DriftStatus.DRIFT_DETECTED}
        )


class DriftManifest(SealedModel):
    """What one drift run compared, and every frozen thing it is bound to."""

    fingerprint_field: ClassVar[str] = "drift_manifest_fingerprint"
    schema_version_field: ClassVar[str] = "drift_schema_version"
    schema_version: ClassVar[str] = DRIFT_SCHEMA_VERSION
    record_label: ClassVar[str] = "drift manifest"

    drift_schema_version: str = DRIFT_SCHEMA_VERSION

    drift_run_id: str
    reference_profile_id: str
    reference_profile_fingerprint: Sha256Hex
    champion_lock_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    eligible_feature_list_fingerprint: Sha256Hex
    allowlist_fingerprint: Sha256Hex
    feature_catalog_fingerprint: Sha256Hex
    drift_config_fingerprint: Sha256Hex

    incoming_scope: MLSplit
    incoming_population_fingerprint: Sha256Hex
    incoming_prediction_id: str | None = None
    incoming_prediction_manifest_fingerprint: Sha256Hex | None = None

    feature_status: DriftStatus
    prediction_status: DriftStatus
    drift_report_fingerprint: Sha256Hex

    drift_manifest_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_manifest(self) -> Self:
        """A named incoming publication is named in full or not at all."""
        present = [
            value is not None
            for value in (
                self.incoming_prediction_id,
                self.incoming_prediction_manifest_fingerprint,
            )
        ]
        if any(present) and not all(present):
            raise ValueError("an incoming publication is named in full or not at all")
        return self


def aggregate_status(statuses: Sequence[DriftStatus]) -> DriftStatus:
    """Return the aggregate of *statuses*, worst-first and never optimistic.

    An empty list is :attr:`DriftStatus.UNAVAILABLE` rather than
    :attr:`DriftStatus.NO_DRIFT`: nothing was compared, so nothing was found
    stable.  ``INCONCLUSIVE`` outranks ``NO_DRIFT`` for the same reason a skipped
    audit check is not a passed one.
    """
    order = (
        DriftStatus.DRIFT_DETECTED,
        DriftStatus.DRIFT_WARNING,
        DriftStatus.INCONCLUSIVE,
        DriftStatus.UNAVAILABLE,
        DriftStatus.NO_DRIFT,
    )
    if not statuses:
        return DriftStatus.UNAVAILABLE
    present = set(statuses)
    for candidate in order:
        if candidate in present:
            return candidate
    raise ModelNotReadyError(  # pragma: no cover - the order covers the enum
        "an unrecognised drift status reached the aggregate"
    )


def _assert_no_outcome_field() -> None:
    """Refuse at import a schema here that declares a prohibited field name."""
    for model in (
        FeatureDriftResult,
        PredictionDriftResult,
        MLDriftReport,
        DriftManifest,
    ):
        offending = prohibited_metadata_fields(set(model.model_fields))
        if offending:
            raise AssertionError(
                f"{model.__name__} declares prohibited field(s) {list(offending)}"
            )


def _assert_one_thresholded_metric() -> None:
    """Refuse at import a second metric the configuration declares no threshold for."""
    if len(list(DriftMetric)) != 1:
        raise AssertionError(
            "DriftMetric grew a member; the reviewed configuration declares warn "
            "and alert values for the population stability index and for nothing "
            "else, so a second metric would be reported against thresholds "
            "nobody chose"
        )


_assert_no_outcome_field()
_assert_one_thresholded_metric()


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def _numeric_edges(reference: FeatureReference) -> tuple[float, ...]:
    """Return the interior edges a quantile partition was cut at."""
    return tuple(
        item.upper
        for item in reference.bins
        if item.cell != NULL_BIN and item.upper is not None
    )


def _assign_numeric(
    reference: FeatureReference, values: Sequence[Any]
) -> tuple[Mapping[str, int], int]:
    """Return incoming counts per reference cell, and the null count.

    Out-of-range values are *not* discarded: the outermost cells are open-ended,
    so a value below everything the reference saw lands in the first cell and one
    above everything it saw lands in the last.
    """
    labels = [item.cell for item in reference.bins]
    counts = dict.fromkeys(labels, 0)
    nulls = 0
    if reference.partition_kind == "quantile":
        edges = _numeric_edges(reference)
        ordinary = [name for name in labels if name != NULL_BIN]
        for value in values:
            if value is None:
                nulls += 1
                continue
            counts[ordinary[numeric_bin_index(float(value), edges)]] += 1
    elif reference.partition_kind == "constant":
        constant = next(
            item.cell for item in reference.bins if item.cell.startswith("equals_")
        )
        target = float(constant.removeprefix("equals_"))
        for value in values:
            if value is None:
                nulls += 1
            elif quantize(float(value)) == target:
                counts[constant] += 1
            else:
                counts[UNKNOWN_BIN] += 1
    else:  # all_null
        for value in values:
            if value is None:
                nulls += 1
            else:
                counts[UNKNOWN_BIN] += 1
    counts[NULL_BIN] = nulls
    return counts, nulls


def _assign_boolean(
    reference: FeatureReference, values: Sequence[Any]
) -> tuple[Mapping[str, int], int]:
    """Return incoming counts per boolean cell, and the null count."""
    counts = dict.fromkeys((item.cell for item in reference.bins), 0)
    nulls = 0
    for value in values:
        if value is True:
            counts["true"] += 1
        elif value is False:
            counts["false"] += 1
        else:
            nulls += 1
    counts[NULL_BIN] = nulls
    return counts, nulls


def _assign_categorical(
    reference: FeatureReference, values: Sequence[Any]
) -> tuple[Mapping[str, int], int, int]:
    """Return incoming counts per cell, the null count, and the unknown count.

    A category the reference vocabulary does not carry lands in
    :data:`~password_attack_detector.ml.reference.UNKNOWN_BIN` rather than being
    dropped or having a cell invented for it.  Inventing one would let the
    incoming population define the partition it is being measured against.
    """
    labels = [item.cell for item in reference.bins]
    counts = dict.fromkeys(labels, 0)
    known = {name for name in labels if name not in {NULL_BIN, UNKNOWN_BIN, OTHER_BIN}}
    nulls = 0
    unknown = 0
    for value in values:
        if value is None:
            nulls += 1
        elif value in known:
            counts[str(value)] += 1
        else:
            unknown += 1
    counts[NULL_BIN] = nulls
    counts[UNKNOWN_BIN] = unknown
    return counts, nulls, unknown


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _decide(index: float, *, warn: float, alert: float) -> tuple[DriftStatus, str]:
    """Return the status and reason the index implies under the thresholds."""
    if index >= alert:
        return DriftStatus.DRIFT_DETECTED, REASON_AT_ALERT
    if index >= warn:
        return DriftStatus.DRIFT_WARNING, REASON_AT_WARN
    return DriftStatus.NO_DRIFT, REASON_BELOW_WARN


def _shares(
    reference: Sequence[Any], counts: Mapping[str, int], total: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the aligned reference and incoming proportion vectors."""
    expected = tuple(item.reference_proportion for item in reference)
    observed = tuple(
        (0.0 if total <= 0 else counts.get(item.cell, 0) / total) for item in reference
    )
    return expected, observed


def compare_feature_population(
    *,
    profile: MLReferenceProfile,
    frame: Any,
    warn_threshold: float,
    alert_threshold: float,
    min_support: int,
) -> tuple[FeatureDriftResult, ...]:
    """Compare an incoming feature frame against the frozen feature partitions.

    Every feature the profile describes gets a result, including one the incoming
    frame does not carry: an absent column is
    :attr:`~password_attack_detector.ml.enums.DriftStatus.UNAVAILABLE` with a
    reason, not a silent omission that would shrink the report and read as
    stability.

    The profile is only ever read.  There is no assignment to it here and no
    field on it that could be assigned to.
    """
    columns: dict[str, list[Any]] = {name: [] for name in frame.feature_names}
    for values in frame.feature_matrix:
        for name, value in zip(frame.feature_names, values, strict=True):
            columns[name].append(value)

    results: list[FeatureDriftResult] = []
    for reference in profile.features:
        values = columns.get(reference.feature)
        if values is None:
            results.append(_absent_feature(reference, REASON_ABSENT_INCOMING))
            continue
        results.append(
            _compare_one_feature(
                reference,
                values,
                warn=warn_threshold,
                alert=alert_threshold,
                min_support=min_support,
            )
        )
    results.sort(key=lambda item: item.feature)
    return tuple(results)


def _absent_feature(reference: FeatureReference, reason: str) -> FeatureDriftResult:
    """Return the typed refusal for a feature that could not be compared."""
    return FeatureDriftResult(
        feature=reference.feature,
        kind=reference.kind,
        partition_kind=reference.partition_kind,
        reference_support=reference.reference_row_count,
        incoming_support=0,
        observed_value=None,
        warn_threshold=None,
        alert_threshold=None,
        status=DriftStatus.UNAVAILABLE,
        reason_code=reason,
    )


def _compare_one_feature(
    reference: FeatureReference,
    values: Sequence[Any],
    *,
    warn: float,
    alert: float,
    min_support: int,
) -> FeatureDriftResult:
    """Return one feature's comparison against its frozen partition."""
    unknown: int | None = None
    if reference.kind is ReferenceFeatureKind.NUMERIC:
        counts, nulls = _assign_numeric(reference, values)
    elif reference.kind is ReferenceFeatureKind.BOOLEAN:
        counts, nulls = _assign_boolean(reference, values)
    else:
        counts, nulls, unknown = _assign_categorical(reference, values)

    total = len(values)
    incoming_null_rate = quantize(0.0 if total == 0 else nulls / total)
    reference_unknown_rate: float | None = None
    incoming_unknown_rate: float | None = None
    unknown_delta: float | None = None
    if reference.kind is ReferenceFeatureKind.CATEGORICAL:
        expected_unknown = next(
            (
                item.reference_proportion
                for item in reference.bins
                if item.cell == UNKNOWN_BIN
            ),
            0.0,
        )
        reference_unknown_rate = quantize(expected_unknown)
        incoming_unknown_rate = quantize(0.0 if total == 0 else (unknown or 0) / total)
        unknown_delta = quantize(incoming_unknown_rate - reference_unknown_rate)

    def result(
        *,
        observed_value: float | None,
        warn_threshold: float | None,
        alert_threshold: float | None,
        status: DriftStatus,
        reason_code: str,
    ) -> FeatureDriftResult:
        """Return the result, with the rates this feature carries either way."""
        return FeatureDriftResult(
            feature=reference.feature,
            kind=reference.kind,
            partition_kind=reference.partition_kind,
            reference_support=reference.reference_row_count,
            incoming_support=total,
            reference_null_rate=reference.reference_null_rate,
            incoming_null_rate=incoming_null_rate,
            null_rate_delta=quantize(
                incoming_null_rate - reference.reference_null_rate
            ),
            reference_unknown_rate=reference_unknown_rate,
            incoming_unknown_rate=incoming_unknown_rate,
            unknown_rate_delta=unknown_delta,
            observed_value=observed_value,
            warn_threshold=warn_threshold,
            alert_threshold=alert_threshold,
            status=status,
            reason_code=reason_code,
        )

    if reference.reference_row_count < min_support:
        return result(
            observed_value=None,
            warn_threshold=None,
            alert_threshold=None,
            status=DriftStatus.INCONCLUSIVE,
            reason_code=REASON_REFERENCE_SUPPORT,
        )
    if total < min_support:
        return result(
            observed_value=None,
            warn_threshold=None,
            alert_threshold=None,
            status=DriftStatus.INCONCLUSIVE,
            reason_code=REASON_INCOMING_SUPPORT,
        )

    expected, observed = _shares(reference.bins, counts, total)
    index = population_stability_index(expected, observed)
    status, reason = _decide(index, warn=warn, alert=alert)
    return result(
        observed_value=index,
        warn_threshold=warn,
        alert_threshold=alert,
        status=status,
        reason_code=reason,
    )


def compare_prediction_population(
    *,
    profile: MLReferenceProfile,
    manifest: Any,
    binary: Sequence[Any],
    category: Sequence[Any] | None,
    anomaly: Sequence[Any] | None,
    warn_threshold: float,
    alert_threshold: float,
    min_support: int,
) -> tuple[PredictionDriftResult, ...]:
    """Compare an incoming publication against the frozen prediction partitions.

    Only quantities the reference profile actually captured are compared.  A
    quantity the incoming publication carries and the reference does not is
    reported as :attr:`~password_attack_detector.ml.enums.DriftStatus.UNAVAILABLE`
    against the reference, never measured against a partition invented on the
    spot.
    """
    from password_attack_detector.ml.reference import prediction_reference_index

    captured = prediction_reference_index(profile)
    observed = _incoming_prediction_counts(
        manifest=manifest, binary=binary, category=category, anomaly=anomaly
    )

    results: list[PredictionDriftResult] = []
    for quantity, reference in captured.items():
        entry = observed.get(quantity)
        if entry is None:
            results.append(
                PredictionDriftResult(
                    quantity=quantity,
                    partition_kind=reference.partition_kind,
                    reference_support=reference.reference_row_count,
                    incoming_support=0,
                    observed_value=None,
                    warn_threshold=None,
                    alert_threshold=None,
                    status=DriftStatus.UNAVAILABLE,
                    reason_code=REASON_ABSENT_INCOMING,
                )
            )
            continue
        results.append(
            _compare_one_quantity(
                reference,
                entry,
                warn=warn_threshold,
                alert=alert_threshold,
                min_support=min_support,
            )
        )
    for quantity in sorted(set(observed) - set(captured), key=str):
        entry = observed[quantity]
        results.append(
            PredictionDriftResult(
                quantity=quantity,
                partition_kind="uncaptured",
                reference_support=0,
                incoming_support=entry[1],
                observed_value=None,
                warn_threshold=None,
                alert_threshold=None,
                status=DriftStatus.UNAVAILABLE,
                reason_code=REASON_ABSENT_REFERENCE,
            )
        )
    results.sort(key=lambda item: str(item.quantity))
    return tuple(results)


def _incoming_prediction_counts(
    *,
    manifest: Any,
    binary: Sequence[Any],
    category: Sequence[Any] | None,
    anomaly: Sequence[Any] | None,
) -> Mapping[PredictionDriftQuantity, tuple[Sequence[Any], int, str]]:
    """Return the incoming values behind each prediction quantity.

    The third element names how the values must be assigned: ``rate`` for a
    boolean tally, ``value`` for a continuous score, ``class`` for a predicted
    label.
    """
    entries: dict[PredictionDriftQuantity, tuple[Sequence[Any], int, str]] = {
        PredictionDriftQuantity.FLAGGED_MALICIOUS_RATE: (
            [row.flagged_malicious for row in binary],
            len(binary),
            "rate",
        ),
        PredictionDriftQuantity.DECISION_SCORE: (
            [row.malicious_decision_score for row in binary],
            len(binary),
            "value",
        ),
    }
    if is_probability(manifest.lineage.binary_score_kind):
        probabilities = [
            row.malicious_probability
            for row in binary
            if row.malicious_probability is not None
        ]
        entries[PredictionDriftQuantity.CALIBRATED_PROBABILITY] = (
            probabilities,
            len(probabilities),
            "value",
        )
    if category:
        predicted = [row.predicted_category for row in category]
        entries[PredictionDriftQuantity.CATEGORY_PREDICTED_CLASS] = (
            predicted,
            len(predicted),
            "class",
        )
        entries[PredictionDriftQuantity.CATEGORY_UNKNOWN_RATE] = (
            [value == UNKNOWN_CATEGORY for value in predicted],
            len(predicted),
            "rate",
        )
    if anomaly:
        entries[PredictionDriftQuantity.ANOMALY_SCORE] = (
            [row.anomaly_score for row in anomaly],
            len(anomaly),
            "value",
        )
    return entries


def _compare_one_quantity(
    reference: PredictionReference,
    entry: tuple[Sequence[Any], int, str],
    *,
    warn: float,
    alert: float,
    min_support: int,
) -> PredictionDriftResult:
    """Return one prediction quantity's comparison against its partition."""
    values, total, mode = entry
    labels = [item.cell for item in reference.bins]
    counts = dict.fromkeys(labels, 0)

    reference_rate: float | None = None
    incoming_rate: float | None = None
    rate_delta: float | None = None

    if mode == "rate":
        positive = labels[1]
        negative = labels[0]
        hits = sum(1 for value in values if bool(value))
        counts[positive] = hits
        counts[negative] = total - hits
        reference_rate = next(
            item.reference_proportion
            for item in reference.bins
            if item.cell == positive
        )
        incoming_rate = quantize(0.0 if total == 0 else hits / total)
        rate_delta = quantize(incoming_rate - reference_rate)
    elif mode == "class":
        for value in values:
            key = str(value)
            counts[key if key in counts else UNKNOWN_BIN] = (
                counts.get(key if key in counts else UNKNOWN_BIN, 0) + 1
            )
    elif reference.partition_kind == "quantile":
        edges = tuple(item.upper for item in reference.bins if item.upper is not None)
        for value in values:
            counts[labels[numeric_bin_index(float(value), edges)]] += 1
    else:  # a constant reference score
        constant = next(
            item.cell for item in reference.bins if item.cell.startswith("equals_")
        )
        target = float(constant.removeprefix("equals_"))
        for value in values:
            if quantize(float(value)) == target:
                counts[constant] += 1
            else:
                counts[UNKNOWN_BIN] += 1

    def result(
        *,
        observed_value: float | None,
        warn_threshold: float | None,
        alert_threshold: float | None,
        status: DriftStatus,
        reason_code: str,
    ) -> PredictionDriftResult:
        """Return the result, with the rates this quantity carries either way."""
        return PredictionDriftResult(
            quantity=reference.quantity,
            partition_kind=reference.partition_kind,
            reference_support=reference.reference_row_count,
            incoming_support=total,
            reference_rate=reference_rate,
            incoming_rate=incoming_rate,
            rate_delta=rate_delta,
            observed_value=observed_value,
            warn_threshold=warn_threshold,
            alert_threshold=alert_threshold,
            status=status,
            reason_code=reason_code,
        )

    if reference.reference_row_count < min_support:
        return result(
            observed_value=None,
            warn_threshold=None,
            alert_threshold=None,
            status=DriftStatus.INCONCLUSIVE,
            reason_code=REASON_REFERENCE_SUPPORT,
        )
    if total < min_support:
        return result(
            observed_value=None,
            warn_threshold=None,
            alert_threshold=None,
            status=DriftStatus.INCONCLUSIVE,
            reason_code=REASON_INCOMING_SUPPORT,
        )

    expected, observed = _shares(reference.bins, counts, total)
    index = population_stability_index(expected, observed)
    status, reason = _decide(index, warn=warn, alert=alert)
    return result(
        observed_value=index,
        warn_threshold=warn,
        alert_threshold=alert,
        status=status,
        reason_code=reason,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_drift_report(
    *,
    profile: MLReferenceProfile,
    incoming: Any,
    features: Sequence[FeatureDriftResult],
    predictions: Sequence[PredictionDriftResult],
    incoming_manifest: Any | None,
    warn_threshold: float,
    alert_threshold: float,
    min_support: int,
) -> MLDriftReport:
    """Return the sealed report one comparison produced."""
    return MLDriftReport.seal(
        reference_profile_id=profile.reference_profile_id,
        reference_profile_fingerprint=profile.reference_profile_fingerprint,
        incoming_scope=incoming.scope,
        incoming_row_count=incoming.row_count,
        incoming_population_fingerprint=incoming.inference_input_fingerprint,
        incoming_prediction_id=(
            None if incoming_manifest is None else incoming_manifest.prediction_id
        ),
        incoming_prediction_manifest_fingerprint=(
            None
            if incoming_manifest is None
            else incoming_manifest.prediction_manifest_fingerprint
        ),
        warn_threshold=warn_threshold,
        alert_threshold=alert_threshold,
        min_support=min_support,
        feature_status=aggregate_status([item.status for item in features]),
        prediction_status=aggregate_status([item.status for item in predictions]),
        features=tuple(features),
        predictions=tuple(predictions),
    )


def build_drift_manifest(
    *, profile: MLReferenceProfile, report: MLDriftReport
) -> DriftManifest:
    """Return the manifest identifying one drift run.

    The identity covers the reference profile, the incoming population, and the
    report -- and nothing about where or when the run happened, so the same
    comparison in two directories produces the same bytes.
    """
    identity = digest(
        {
            "drift_report_fingerprint": report.drift_report_fingerprint,
            "drift_schema_version": DRIFT_SCHEMA_VERSION,
            "incoming_population_fingerprint": (report.incoming_population_fingerprint),
            "incoming_prediction_manifest_fingerprint": (
                report.incoming_prediction_manifest_fingerprint
            ),
            "reference_profile_fingerprint": (profile.reference_profile_fingerprint),
        }
    )
    return DriftManifest.seal(
        drift_run_id=identity,
        reference_profile_id=profile.reference_profile_id,
        reference_profile_fingerprint=profile.reference_profile_fingerprint,
        champion_lock_fingerprint=profile.champion_lock_fingerprint,
        preprocessor_fingerprint=profile.preprocessor_fingerprint,
        eligible_feature_list_fingerprint=(profile.eligible_feature_list_fingerprint),
        allowlist_fingerprint=profile.allowlist_fingerprint,
        feature_catalog_fingerprint=profile.feature_catalog_fingerprint,
        drift_config_fingerprint=profile.drift_config_fingerprint,
        incoming_scope=report.incoming_scope,
        incoming_population_fingerprint=report.incoming_population_fingerprint,
        incoming_prediction_id=report.incoming_prediction_id,
        incoming_prediction_manifest_fingerprint=(
            report.incoming_prediction_manifest_fingerprint
        ),
        feature_status=report.feature_status,
        prediction_status=report.prediction_status,
        drift_report_fingerprint=report.drift_report_fingerprint,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(value: float | None) -> str:
    """Render a number, or say the quantity is unavailable rather than zero."""
    return "unavailable" if value is None else f"{value:.6f}"


def drift_report_to_markdown(report: MLDriftReport, manifest: DriftManifest) -> str:
    """Render the drift finding as deterministic Markdown.

    Aggregate only: statuses, indices, supports, and identity. No row, no anchor,
    no feature value, and no bin-level mass -- a table of expected shares is the
    reference distribution, and this document travels.
    """
    lines = [
        "# ML drift report",
        "",
        "Monitoring evidence, **not** model correctness. Every number here was "
        "computed without reading a label, so nothing in this document says a "
        "model became less accurate -- only that a population no longer looks "
        "like the one the champion was baselined on.",
        "",
        "**No retraining happened, and none is triggered by this report.** A "
        "finding may be a reason for a human to investigate. It is never an "
        "action, and nothing in this repository turns one into one.",
        "",
        "## Identity",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Drift schema | {report.drift_schema_version} |",
        f"| Drift run id | `{manifest.drift_run_id}` |",
        f"| Reference profile id | `{report.reference_profile_id}` |",
        f"| Reference profile | `{report.reference_profile_fingerprint}` |",
        f"| Champion lock | `{manifest.champion_lock_fingerprint}` |",
        f"| Preprocessor | `{manifest.preprocessor_fingerprint}` |",
        f"| Drift config | `{manifest.drift_config_fingerprint}` |",
        f"| Incoming scope | {report.incoming_scope} |",
        f"| Incoming rows | {report.incoming_row_count:,} |",
        f"| Incoming population | `{report.incoming_population_fingerprint}` |",
        f"| Incoming prediction | {report.incoming_prediction_id or 'not compared'} |",
        f"| Report fingerprint | `{report.drift_report_fingerprint}` |",
        "",
        "## Aggregate",
        "",
        "| Subject | Status |",
        "| --- | --- |",
        f"| Feature drift | {report.feature_status} |",
        f"| Prediction drift | {report.prediction_status} |",
        "",
        f"Thresholds: warn at {report.warn_threshold:.6f}, alert at "
        f"{report.alert_threshold:.6f}, minimum support "
        f"{report.min_support:,} rows on each side.",
        "",
        "## Feature drift",
        "",
        "| Feature | Kind | Metric | Ref rows | In rows | Value | Null Δ | "
        "Unknown Δ | Status | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.features:
        lines.append(
            f"| `{item.feature}` | {item.kind} | {item.metric} | "
            f"{item.reference_support:,} | {item.incoming_support:,} | "
            f"{_cell(item.observed_value)} | {_cell(item.null_rate_delta)} | "
            f"{_cell(item.unknown_rate_delta)} | {item.status} | "
            f"{item.reason_code} |"
        )
    lines += [
        "",
        "## Prediction drift",
        "",
        "Kept separate from feature drift throughout. A shift in what the model "
        "is shown and a shift in what it says are different findings.",
        "",
        "| Quantity | Metric | Ref rows | In rows | Value | Rate Δ | Status | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for prediction in report.predictions:
        lines.append(
            f"| {prediction.quantity} | {prediction.metric} | "
            f"{prediction.reference_support:,} | "
            f"{prediction.incoming_support:,} | "
            f"{_cell(prediction.observed_value)} | "
            f"{_cell(prediction.rate_delta)} | {prediction.status} | "
            f"{prediction.reason_code} |"
        )
    lines += [
        "",
        "## Semantics",
        "",
        "- `inconclusive` means the comparison had too few rows on one side to "
        "be evidence. It is **not** `no_drift`.",
        "- `unavailable` means the quantity does not exist on one side. It is "
        "**not** a measurement of zero.",
        "- Bin edges, categories, classes, and expected shares are the frozen "
        "reference's. Incoming data was assigned to them and never redefined "
        "one; values beyond the reference range landed in an open-ended outer "
        "cell rather than being discarded.",
        "- The index is descriptive. A high value on a synthetic population "
        "says the two synthetic populations differ, and nothing about real "
        "authentication traffic.",
        "",
    ]
    return "\n".join(lines)
