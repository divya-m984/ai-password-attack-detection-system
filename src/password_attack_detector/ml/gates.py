"""Champion gates: pass, fail, or admit that nobody could tell.

A gate answers one question about one candidate, on validation-B evidence that
Milestone 6 already froze. It answers with **three** outcomes rather than two,
and the third is the reason this module exists: a measurement nobody had enough
data to make is not a passed check, and a project that lets it become one will
eventually promote a model on the strength of eleven benign rows.

**Every gate reports its arithmetic.** A rate arrives with its numerator, its
denominator, and a Wilson interval, so a reader can see not only what was
measured but how firmly. A rate whose denominator is empty is ``None`` and
carries :attr:`~password_attack_detector.ml.enums.GateStatus.INCONCLUSIVE`;
it is never rendered as zero, because "no false positives out of none" and "no
false positives out of nine thousand" are different facts and only one of them
is a false-positive rate.

**The resolution rule.** A false-positive ceiling of 1% cannot be *held* by a
sample of fifty benign rows -- the coarsest rate such a sample can express is
2%, so the only way to appear under the ceiling is to flag nothing at all. The
ceiling has then not been tested, and the gate says ``inconclusive`` rather than
reporting a compliance nobody measured.

**Wilson intervals are published, not applied.** The configured ceiling is a
constraint on the rate, so the gate compares the rate -- under the resolution
precondition above -- and publishes the interval beside it. Requiring the
interval's upper bound to clear the ceiling would be a *stricter* criterion
than the one written down, and inventing an acceptance criterion during
selection is exactly what a predeclared gate set exists to prevent.

Nothing here reads a row, a label, or a split. Gates consume frozen artifacts:
a threshold selection, an out-of-sample calibration report, a catalog entry.
"""

from __future__ import annotations

import math
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.ml.calibration import CalibrationReport, digest
from password_attack_detector.ml.config import ChampionGateConfig, MLConfig
from password_attack_detector.ml.enums import (
    CalibrationEvaluationKind,
    GateStatus,
    MetricStatus,
    SelectionStatus,
    ValidationPartition,
)
from password_attack_detector.ml.ranking import (
    DISCRIMINATION_SCORE_KIND,
    PR_AUC_INTEGRATION,
    RANKING_METRIC_NAME,
    RankingEvidence,
)
from password_attack_detector.ml.schemas import SupportRequirement
from password_attack_detector.ml.thresholds import (
    CategoryAbstentionSelection,
    ThresholdSelection,
)

__all__ = [
    "BINARY_GATE_IDS",
    "CATEGORY_GATE_IDS",
    "GATE_SCHEMA_VERSION",
    "BinaryGateInputs",
    "CategoryGateInputs",
    "GateEvidence",
    "RateEvidence",
    "evaluate_binary_gates",
    "evaluate_category_gates",
    "gate_config_fingerprint",
    "wilson_interval",
]

#: The gate contract's own version, independent of every artifact schema.
GATE_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Every mandatory binary gate, in evaluation order.
#:
#: A closed, ordered tuple. Ordered so two selections report their gates in the
#: same sequence; closed so a configured gate cannot quietly go missing -- a
#: test asserts every candidate result carries exactly these.
BINARY_GATE_IDS: Final[tuple[str, ...]] = (
    "validation_support",
    "operating_threshold",
    "false_positive_ceiling",
    "detection_rate_floor",
    "baseline_pr_auc_gain",
    "calibration_quality",
    "serializer_eligibility",
    "lineage_compatibility",
)

#: Every mandatory category gate, in evaluation order.
CATEGORY_GATE_IDS: Final[tuple[str, ...]] = (
    "category_validation_support",
    "category_abstention_point",
    "category_precision_floor",
    "category_class_support",
    "lineage_compatibility",
)

#: The metric name the improvement gate publishes. Derived from the ranking
#: contract rather than spelled again here, so the name in a gate verdict and
#: the arithmetic that produced it cannot drift apart.
_GAIN_METRIC: Final[str] = f"{RANKING_METRIC_NAME}_gain"

#: Quantised precision for every published rate and bound, matching the rest of
#: the layer so a digest does not move with the last bit of a repr.
_PRECISION: Final[int] = 9


def _quantize(value: float) -> float:
    """Return *value* at the precision published rates are stored in."""
    return float(f"{value:.{_PRECISION}f}")


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Return the Wilson score interval for a proportion, or ``None``.

    Wilson rather than the normal approximation. At the counts a validation
    half produces -- two false positives out of sixty, none out of two hundred --
    the normal interval runs past zero, and it is narrowest exactly where the
    data is thinnest. Wilson stays inside ``[0, 1]`` and widens honestly.

    Returns ``None`` for an empty denominator: an interval around a proportion
    of nothing is not a wide interval, it is not an interval.
    """
    if trials <= 0:
        return None
    z = _z_for(confidence)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4 * trials**2))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


#: Standard-normal quantiles for the confidence levels this project uses.
#:
#: A small table rather than an inverse-CDF implementation: ``scipy`` is not a
#: direct dependency here, three levels cover every configured value, and a
#: hand-rolled approximation would be a numerical routine nobody reviewed
#: sitting underneath a promotion decision.
_Z_SCORES: Final[dict[str, float]] = {
    "0.900000000": 1.6448536269514722,
    "0.950000000": 1.959963984540054,
    "0.990000000": 2.5758293035489004,
}


def _z_for(confidence: float) -> float:
    """Return the two-sided normal quantile for *confidence*.

    Raises:
        ValueError: for a level this project has not tabulated. Interpolating
            would produce an interval whose width nobody could check.
    """
    key = f"{confidence:.{_PRECISION}f}"
    z = _Z_SCORES.get(key)
    if z is None:
        raise ValueError(
            f"confidence level {confidence!r} is not one of the tabulated "
            f"levels {sorted(_Z_SCORES)}; a gate must not rest on an "
            f"interpolated quantile"
        )
    return z


class RateEvidence(BaseModel):
    """A rate, the counts behind it, and how firmly it is known.

    Published rather than summarised. A gate that reported only "0.008" would
    be indistinguishable whether it came from four benign rows or four thousand,
    and the difference is the whole question.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    #: ``None`` exactly when the denominator is empty. Never zero: an absent
    #: measurement and a measured zero are different findings.
    value: float | None
    interval_lower: float | None
    interval_upper: float | None
    confidence: float

    @model_validator(mode="after")
    def check_evidence(self) -> Self:
        """The rate and its interval exist exactly when the denominator does."""
        if self.numerator > self.denominator:
            raise ValueError("a rate cannot count more successes than trials")
        empty = self.denominator == 0
        if empty != (self.value is None):
            raise ValueError(
                "a rate exists exactly when its denominator does; an empty "
                "denominator is unavailable, not a measured zero"
            )
        if empty != (self.interval_lower is None):
            raise ValueError("an interval exists exactly when the rate does")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("an interval has both bounds or neither")
        if (
            self.interval_lower is not None
            and self.interval_upper is not None
            and self.interval_lower > self.interval_upper
        ):
            raise ValueError("the interval bounds are inverted")
        for name in ("value", "interval_lower", "interval_upper"):
            number = getattr(self, name)
            if number is not None and not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
        return self

    @classmethod
    def of(cls, numerator: int, denominator: int, *, confidence: float) -> RateEvidence:
        """Return the evidence for *numerator* over *denominator*."""
        interval = wilson_interval(numerator, denominator, confidence=confidence)
        return cls(
            numerator=numerator,
            denominator=denominator,
            value=None if denominator == 0 else _quantize(numerator / denominator),
            interval_lower=None if interval is None else _quantize(interval[0]),
            interval_upper=None if interval is None else _quantize(interval[1]),
            confidence=_quantize(confidence),
        )


class GateEvidence(BaseModel):
    """One gate's verdict on one candidate, with everything behind it.

    ``INCONCLUSIVE`` is not a soft pass and not a soft fail. It says the check
    could not be made, and for a mandatory gate it blocks promotion exactly as a
    failure does -- which is the only interpretation that keeps a thin
    validation half from looking like a clean bill of health.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: str
    metric: str
    status: GateStatus
    mandatory: bool
    #: A stable code, never prose. Reasons are compared, grouped, and asserted
    #: on; a sentence would be reworded and every comparison would break.
    reason: str
    observed: float | None = None
    required: float | None = None
    constraint: str
    rate: RateEvidence | None = None
    support_required: int | None = None
    support_observed: int | None = None

    @model_validator(mode="after")
    def check_gate(self) -> Self:
        """A passed or failed gate observed something; every number is finite."""
        if not self.gate_id or not self.reason:
            raise ValueError("a gate names itself and its reason")
        if self.status is not GateStatus.INCONCLUSIVE and self.observed is None:
            raise ValueError(
                f"gate {self.gate_id!r} reports {self.status} without an "
                f"observation; a verdict with nothing behind it is not one"
            )
        for name in ("observed", "required"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"gate {name} must be finite")
        if self.support_observed is not None and self.support_observed < 0:
            raise ValueError("observed support cannot be negative")
        return self

    @property
    def blocking(self) -> bool:
        """Return whether this gate stops promotion.

        A mandatory gate blocks unless it passed. Inconclusive blocks: the
        alternative is promoting on a check nobody completed.
        """
        return self.mandatory and self.status is not GateStatus.PASS


def gate_config_fingerprint(config: MLConfig) -> str:
    """Return the digest of everything that decides how gates behave.

    The gate thresholds, the support floors, and the ranking policy together.
    Two selections carried out under different acceptance criteria are different
    selections even over identical candidates, and this is what makes them
    distinguishable rather than merely differently-argued.
    """
    return digest(
        {
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "discrimination_metric": RANKING_METRIC_NAME,
            "discrimination_integration": PR_AUC_INTEGRATION,
            "discrimination_score_kind": str(DISCRIMINATION_SCORE_KIND),
            "gates": config.gates.fingerprint_data(),
            "support": config.support.fingerprint_data(),
            "selection": config.selection.fingerprint_data(),
            "category": config.category.fingerprint_data(),
            "binary_gate_ids": list(BINARY_GATE_IDS),
            "category_gate_ids": list(CATEGORY_GATE_IDS),
        }
    )


# ---------------------------------------------------------------------------
# Binary gates
# ---------------------------------------------------------------------------


class BinaryGateInputs(BaseModel):
    """Everything one binary candidate is judged on, all of it already frozen.

    Deliberately not a dataset. Every field is an artifact Milestone 6 published
    and verified, so a gate cannot reach a row even by mistake -- and there is no
    field here a test split could arrive in.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_model_id: str
    threshold: ThresholdSelection | None
    calibration_quality: CalibrationReport | None
    #: Whether this candidate's configured protocol requires a calibrator at
    #: all. A reference baseline's does not; an ordinary candidate's does when
    #: the configuration names a method.
    calibration_required: bool
    #: The reviewed catalog's verdict on the family, copied rather than decided.
    champion_eligible: bool
    publishable: bool
    #: Whether the published model artifact passed the Milestone 4 verifier.
    artifact_verified: bool
    #: This candidate's exact validation-B discrimination evidence, built from
    #: every distinct score level. ``None`` when the run published none, which
    #: is an absence of evidence rather than a score of zero.
    ranking: RankingEvidence | None
    #: The comparator's evidence, measured the same way at the same scoring
    #: stage, or ``None`` when no usable reference could be established.
    baseline_ranking: RankingEvidence | None
    baseline_unavailable_reason: str | None = None
    #: Whether this candidate's lineage matches the selection scope in full.
    lineage_matches: bool
    lineage_mismatch_reason: str | None = None


def evaluate_binary_gates(
    inputs: BinaryGateInputs, *, config: MLConfig
) -> tuple[GateEvidence, ...]:
    """Return every mandatory binary gate's verdict, in declared order.

    Every gate in :data:`BINARY_GATE_IDS` is evaluated and returned, always. A
    gate that could not be measured reports ``inconclusive``; none is skipped,
    because a missing gate in a report reads as a gate that passed.
    """
    gates = config.gates
    support = config.support
    confidence = config.selection.rate_interval_confidence
    selection = inputs.threshold

    results = [
        _validation_support_gate(selection, support=support, confidence=confidence),
        _operating_threshold_gate(selection),
        _false_positive_gate(selection, gates=gates, confidence=confidence),
        _detection_rate_gate(selection, gates=gates, confidence=confidence),
        _baseline_gain_gate(inputs, gates=gates),
        _calibration_gate(inputs, gates=gates),
        _serializer_gate(inputs),
        _lineage_gate(inputs.lineage_matches, inputs.lineage_mismatch_reason),
    ]
    assert tuple(item.gate_id for item in results) == BINARY_GATE_IDS
    return tuple(results)


def _validation_support_gate(
    selection: ThresholdSelection | None,
    *,
    support: SupportRequirement,
    confidence: float,
) -> GateEvidence:
    """The floor beneath every other gate: enough rows of each class to measure."""
    required = min(
        support.min_validation_positive_rows, support.min_validation_benign_rows
    )
    if selection is None:
        return GateEvidence(
            gate_id="validation_support",
            metric="validation_b_class_support",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="threshold_selection_absent",
            constraint=(
                f"validation-B carries at least "
                f"{support.min_validation_positive_rows} malicious and "
                f"{support.min_validation_benign_rows} benign rows"
            ),
            support_required=required,
        )
    positives = selection.malicious_row_count
    benign = selection.benign_row_count
    adequate = (
        positives >= support.min_validation_positive_rows
        and benign >= support.min_validation_benign_rows
    )
    return GateEvidence(
        gate_id="validation_support",
        metric="validation_b_class_support",
        status=GateStatus.PASS if adequate else GateStatus.INCONCLUSIVE,
        mandatory=True,
        reason="support_adequate" if adequate else "min_validation_class_rows",
        observed=float(min(positives, benign)),
        required=float(required),
        constraint=(
            f"validation-B carries at least "
            f"{support.min_validation_positive_rows} malicious and "
            f"{support.min_validation_benign_rows} benign rows"
        ),
        rate=RateEvidence.of(positives, positives + benign, confidence=confidence),
        support_required=required,
        support_observed=min(positives, benign),
    )


def _operating_threshold_gate(
    selection: ThresholdSelection | None,
) -> GateEvidence:
    """A model with no operating point cannot be deployed, whatever it scores."""
    if selection is None:
        return GateEvidence(
            gate_id="operating_threshold",
            metric="threshold_selection_status",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="threshold_selection_absent",
            constraint="a binary operating point was selected on validation-B",
        )
    selected = selection.status is SelectionStatus.SELECTED
    if selected:
        status, reason = GateStatus.PASS, "threshold_selected"
    elif selection.status is SelectionStatus.NO_FEASIBLE_THRESHOLD:
        # A measured negative: the support was adequate and no candidate
        # threshold satisfied the constraint.
        status, reason = GateStatus.FAIL, "no_feasible_threshold"
    else:
        status, reason = GateStatus.INCONCLUSIVE, "insufficient_validation_support"
    return GateEvidence(
        gate_id="operating_threshold",
        metric="threshold_selection_status",
        status=status,
        mandatory=True,
        reason=reason,
        observed=1.0 if selected else 0.0,
        required=1.0,
        constraint="a binary operating point was selected on validation-B",
    )


def _false_positive_gate(
    selection: ThresholdSelection | None,
    *,
    gates: ChampionGateConfig,
    confidence: float,
) -> GateEvidence:
    """The ceiling, and whether the sample could resolve it at all."""
    ceiling = gates.max_false_positive_rate
    constraint = f"false-positive rate at or under {ceiling:.4f}"
    if selection is None or selection.status is not SelectionStatus.SELECTED:
        return GateEvidence(
            gate_id="false_positive_ceiling",
            metric="false_positive_rate",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="operating_point_unavailable",
            required=_quantize(ceiling),
            constraint=constraint,
        )
    benign = selection.benign_row_count
    flagged = selection.benign_flagged_count or 0
    evidence = RateEvidence.of(flagged, benign, confidence=confidence)

    # The resolution rule. With N benign rows the smallest non-zero rate
    # observable is 1/N; if the ceiling sits below that, the only way to appear
    # compliant is to flag nothing, and the ceiling has not been tested.
    resolvable = benign * ceiling >= 1.0
    if evidence.value is None:
        status, reason = GateStatus.INCONCLUSIVE, "no_benign_validation_rows"
    elif not resolvable:
        status, reason = GateStatus.INCONCLUSIVE, "false_positive_rate_resolution"
    elif evidence.value <= ceiling:
        status, reason = GateStatus.PASS, "within_ceiling"
    else:
        status, reason = GateStatus.FAIL, "above_ceiling"
    return GateEvidence(
        gate_id="false_positive_ceiling",
        metric="false_positive_rate",
        status=status,
        mandatory=True,
        reason=reason,
        observed=evidence.value,
        required=_quantize(ceiling),
        constraint=constraint,
        rate=evidence,
        support_required=math.ceil(1.0 / ceiling),
        support_observed=benign,
    )


def _detection_rate_gate(
    selection: ThresholdSelection | None,
    *,
    gates: ChampionGateConfig,
    confidence: float,
) -> GateEvidence:
    """The floor: a detector that holds the ceiling by detecting nothing is not one."""
    floor = gates.min_detection_rate
    constraint = f"detection rate at or above {floor:.4f}"
    if selection is None or selection.status is not SelectionStatus.SELECTED:
        return GateEvidence(
            gate_id="detection_rate_floor",
            metric="detection_rate",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="operating_point_unavailable",
            required=_quantize(floor),
            constraint=constraint,
        )
    positives = selection.malicious_row_count
    detected = selection.malicious_flagged_count or 0
    evidence = RateEvidence.of(detected, positives, confidence=confidence)
    if evidence.value is None:
        status, reason = GateStatus.INCONCLUSIVE, "no_malicious_validation_rows"
    elif evidence.value >= floor:
        status, reason = GateStatus.PASS, "above_floor"
    else:
        status, reason = GateStatus.FAIL, "below_floor"
    return GateEvidence(
        gate_id="detection_rate_floor",
        metric="detection_rate",
        status=status,
        mandatory=True,
        reason=reason,
        observed=evidence.value,
        required=_quantize(floor),
        constraint=constraint,
        rate=evidence,
        support_observed=positives,
    )


def _baseline_gain_gate(
    inputs: BinaryGateInputs, *, gates: ChampionGateConfig
) -> GateEvidence:
    """Improvement over the mandatory comparator, measured exactly or not at all.

    The metric is the one the reviewed configuration names --
    ``min_pr_auc_gain_over_baseline`` -- so what is compared here is PR-AUC
    under the declared step-wise convention, computed over **every distinct
    validation-B score level** for both models. It is not average precision
    over the operating-point search grid wearing PR-AUC's name, and it does not
    become approximate because the threshold search was bounded: the two
    quantities are produced by different code paths on purpose, and only the
    exact one may satisfy this gate.

    **Both sides are measured at one scoring stage by construction.** The
    ranking contract refuses evidence at any stage but the declared one, so the
    gate cannot be handed a calibrated score for one model and a raw score for
    the next -- there is no such record to hand it. The stage is named in the
    constraint so a reader of the verdict does not have to know that.

    The gate is never waived. If the reference baseline cannot be established,
    or either side's exact evidence is missing, the comparison is inconclusive
    and the candidate is blocked -- promoting a model because nothing was
    available to measure it against is precisely the outcome a mandatory
    comparator exists to prevent, and promoting one on an approximation would
    make the criterion depend on a performance setting.
    """
    required = gates.min_pr_auc_gain_over_baseline
    constraint = (
        f"exact {RANKING_METRIC_NAME} ({PR_AUC_INTEGRATION} integration over "
        f"every distinct {DISCRIMINATION_SCORE_KIND!s} level) at least "
        f"{required:.4f} above the {gates.baseline_family!s} reference baseline"
    )
    candidate = inputs.ranking
    reference = inputs.baseline_ranking
    if reference is None:
        return GateEvidence(
            gate_id="baseline_pr_auc_gain",
            metric=_GAIN_METRIC,
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason=inputs.baseline_unavailable_reason
            or "reference_ranking_evidence_unavailable",
            required=_quantize(required),
            constraint=constraint,
        )
    if candidate is None:
        return GateEvidence(
            gate_id="baseline_pr_auc_gain",
            metric=_GAIN_METRIC,
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="ranking_evidence_unavailable",
            required=_quantize(required),
            constraint=constraint,
        )
    gain = _quantize(candidate.pr_auc - reference.pr_auc)
    passed = gain >= required
    return GateEvidence(
        gate_id="baseline_pr_auc_gain",
        metric=_GAIN_METRIC,
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        mandatory=True,
        reason="beats_reference_baseline" if passed else "no_gain_over_baseline",
        observed=gain,
        required=_quantize(required),
        constraint=constraint,
        support_observed=candidate.positive_count,
    )


def _calibration_gate(
    inputs: BinaryGateInputs, *, gates: ChampionGateConfig
) -> GateEvidence:
    """Calibration quality, and only from evidence that may serve as evidence.

    An in-sample validation-A diagnostic is refused outright. It measures a
    calibrator on the rows that fitted it, so it says nothing about
    generalisation -- and a gate that accepted it would be satisfied by a
    calibrator that had merely memorised.
    """
    ceiling = gates.max_expected_calibration_error
    constraint = f"out-of-sample expected calibration error at or under {ceiling:.4f}"
    report = inputs.calibration_quality

    if not inputs.calibration_required:
        # A reference baseline emits a constant score and is not calibrated by
        # contract. The gate is not applicable rather than failed, and it stays
        # mandatory so its verdict is still recorded.
        return GateEvidence(
            gate_id="calibration_quality",
            metric="expected_calibration_error",
            status=GateStatus.PASS,
            mandatory=True,
            reason="calibration_not_applicable",
            observed=0.0,
            required=_quantize(ceiling),
            constraint=constraint,
        )
    if report is None:
        return GateEvidence(
            gate_id="calibration_quality",
            metric="expected_calibration_error",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="calibration_evidence_absent",
            required=_quantize(ceiling),
            constraint=constraint,
        )
    if (
        report.evaluation_kind is not CalibrationEvaluationKind.OUT_OF_SAMPLE_VALIDATION
        or report.source_partition is not ValidationPartition.VALIDATION_B
    ):
        return GateEvidence(
            gate_id="calibration_quality",
            metric="expected_calibration_error",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="in_sample_diagnostic_is_not_evidence",
            required=_quantize(ceiling),
            constraint=constraint,
        )
    if (
        report.status is not MetricStatus.MEASURED
        or report.expected_calibration_error is None
    ):
        return GateEvidence(
            gate_id="calibration_quality",
            metric="expected_calibration_error",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="calibration_support_inadequate",
            required=_quantize(ceiling),
            constraint=constraint,
            support_required=report.min_calibration_rows,
            support_observed=report.row_count,
        )
    observed = report.expected_calibration_error
    passed = observed <= ceiling and report.admissible_as_champion_evidence
    return GateEvidence(
        gate_id="calibration_quality",
        metric="expected_calibration_error",
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        mandatory=True,
        reason="within_calibration_ceiling" if passed else "above_calibration_ceiling",
        observed=_quantize(observed),
        required=_quantize(ceiling),
        constraint=constraint,
        support_required=report.min_calibration_rows,
        support_observed=report.row_count,
    )


def _serializer_gate(inputs: BinaryGateInputs) -> GateEvidence:
    """A model that cannot be stored and reloaded exactly is never promoted.

    Three conditions, all from the reviewed catalog and the Milestone 4
    verifier: the family is champion-eligible, its serializer is proven enough
    to publish, and the artifact it produced verified.
    """
    eligible = (
        inputs.champion_eligible and inputs.publishable and inputs.artifact_verified
    )
    if eligible:
        reason = "serializer_and_artifact_verified"
    elif not inputs.champion_eligible:
        reason = "family_not_champion_eligible"
    elif not inputs.publishable:
        reason = "serializer_unproven"
    else:
        reason = "artifact_verification_failed"
    return GateEvidence(
        gate_id="serializer_eligibility",
        metric="serializer_and_artifact",
        status=GateStatus.PASS if eligible else GateStatus.FAIL,
        mandatory=True,
        reason=reason,
        observed=1.0 if eligible else 0.0,
        required=1.0,
        constraint=(
            "the family is champion-eligible, its serializer is proven, and its "
            "published artifact verified"
        ),
    )


def _lineage_gate(matches: bool, reason: str | None) -> GateEvidence:
    """Every candidate must describe the same experiment as the scope it is in."""
    return GateEvidence(
        gate_id="lineage_compatibility",
        metric="selection_scope_lineage",
        status=GateStatus.PASS if matches else GateStatus.FAIL,
        mandatory=True,
        reason="lineage_matches" if matches else (reason or "lineage_mismatch"),
        observed=1.0 if matches else 0.0,
        required=1.0,
        constraint=(
            "the candidate's configuration, catalog, feature contract, and "
            "validation partition match the selection scope"
        ),
    )


# ---------------------------------------------------------------------------
# Category gates
# ---------------------------------------------------------------------------


class CategoryGateInputs(BaseModel):
    """Everything one category candidate is judged on, all of it already frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_model_id: str
    abstention: CategoryAbstentionSelection | None
    champion_eligible: bool
    publishable: bool
    artifact_verified: bool
    lineage_matches: bool
    lineage_mismatch_reason: str | None = None


def evaluate_category_gates(
    inputs: CategoryGateInputs, *, config: MLConfig
) -> tuple[GateEvidence, ...]:
    """Return every mandatory category gate's verdict, in declared order."""
    abstention = inputs.abstention
    confidence = config.selection.rate_interval_confidence
    minimum_rows = max(
        config.category.min_rows_per_category, config.support.min_rows_per_category
    )
    results = [
        _category_support_gate(
            abstention, minimum=config.category.min_known_malicious_rows
        ),
        _category_abstention_gate(abstention),
        _category_precision_gate(
            abstention,
            minimum=config.category.min_known_category_precision,
            confidence=confidence,
        ),
        _category_class_support_gate(abstention, minimum=minimum_rows),
        _lineage_gate(inputs.lineage_matches, inputs.lineage_mismatch_reason),
    ]
    assert tuple(item.gate_id for item in results) == CATEGORY_GATE_IDS
    return tuple(results)


def _category_support_gate(
    abstention: CategoryAbstentionSelection | None, *, minimum: int
) -> GateEvidence:
    """Enough known-malicious validation-B rows for coverage to mean anything."""
    constraint = f"at least {minimum} known-malicious validation-B rows"
    if abstention is None:
        return GateEvidence(
            gate_id="category_validation_support",
            metric="known_malicious_row_count",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="abstention_selection_absent",
            required=float(minimum),
            constraint=constraint,
            support_required=minimum,
        )
    observed = abstention.known_malicious_row_count
    adequate = observed >= minimum
    return GateEvidence(
        gate_id="category_validation_support",
        metric="known_malicious_row_count",
        status=GateStatus.PASS if adequate else GateStatus.INCONCLUSIVE,
        mandatory=True,
        reason="support_adequate" if adequate else "min_known_malicious_rows",
        observed=float(observed),
        required=float(minimum),
        constraint=constraint,
        support_required=minimum,
        support_observed=observed,
    )


def _category_abstention_gate(
    abstention: CategoryAbstentionSelection | None,
) -> GateEvidence:
    """The abstention point must have been *chosen*, not fallen back to.

    A predeclared conservative fallback is a usable threshold and an honest one,
    and it is not a measurement. A head promoted on a threshold nobody selected
    would be a head whose coverage nobody established.
    """
    constraint = "an abstention threshold was selected from validation-B"
    if abstention is None:
        return GateEvidence(
            gate_id="category_abstention_point",
            metric="abstention_data_selected",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="abstention_selection_absent",
            constraint=constraint,
        )
    selected = (
        abstention.status is SelectionStatus.SELECTED and abstention.data_selected
    )
    if selected:
        status, reason = GateStatus.PASS, "abstention_selected"
    elif abstention.status is SelectionStatus.NO_FEASIBLE_THRESHOLD:
        status, reason = GateStatus.FAIL, "no_feasible_abstention_threshold"
    else:
        status, reason = GateStatus.INCONCLUSIVE, "insufficient_validation_support"
    return GateEvidence(
        gate_id="category_abstention_point",
        metric="abstention_data_selected",
        status=status,
        mandatory=True,
        reason=reason,
        observed=1.0 if selected else 0.0,
        required=1.0,
        constraint=constraint,
    )


def _category_precision_gate(
    abstention: CategoryAbstentionSelection | None,
    *,
    minimum: float,
    confidence: float,
) -> GateEvidence:
    """Known-category precision at the selected abstention point.

    The floor applied is the stricter of the one the artifact was selected
    under and the one currently configured.  A head selected under a looser
    floor is not promoted by the floor it happened to be produced with; the
    selection artifact records its own criterion, and the gate is the criterion
    in force now.
    """
    if abstention is None:
        return GateEvidence(
            gate_id="category_precision_floor",
            metric="known_category_precision",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="abstention_selection_absent",
            required=_quantize(minimum),
            constraint="known-category precision clears the configured floor",
        )
    floor = max(minimum, abstention.min_known_category_precision)
    constraint = f"known-category precision at or above {floor:.4f}"
    if (
        abstention.known_category_precision is None
        or abstention.covered_count is None
        or abstention.correct_count is None
    ):
        return GateEvidence(
            gate_id="category_precision_floor",
            metric="known_category_precision",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="precision_unmeasured",
            required=_quantize(floor),
            constraint=constraint,
        )
    evidence = RateEvidence.of(
        abstention.correct_count, abstention.covered_count, confidence=confidence
    )
    observed = abstention.known_category_precision
    passed = observed >= floor
    return GateEvidence(
        gate_id="category_precision_floor",
        metric="known_category_precision",
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        mandatory=True,
        reason="above_precision_floor" if passed else "below_precision_floor",
        observed=_quantize(observed),
        required=_quantize(floor),
        constraint=constraint,
        rate=evidence,
        support_observed=abstention.covered_count,
    )


def _category_class_support_gate(
    abstention: CategoryAbstentionSelection | None, *, minimum: int
) -> GateEvidence:
    """Every known class needs enough rows for its own share to be measurable.

    Checked per class, not in aggregate. A head that is excellent at the two
    common categories and has seen three rows of the third has not been
    evaluated on the third, and an aggregate precision would hide that.
    """
    constraint = f"every known class carries at least {minimum} validation-B rows"
    if abstention is None:
        return GateEvidence(
            gate_id="category_class_support",
            metric="min_rows_per_category",
            status=GateStatus.INCONCLUSIVE,
            mandatory=True,
            reason="abstention_selection_absent",
            required=float(minimum),
            constraint=constraint,
            support_required=minimum,
        )
    counts = [item.row_count for item in abstention.class_support]
    thinnest = min(counts) if counts else 0
    adequate = bool(counts) and thinnest >= minimum
    return GateEvidence(
        gate_id="category_class_support",
        metric="min_rows_per_category",
        status=GateStatus.PASS if adequate else GateStatus.INCONCLUSIVE,
        mandatory=True,
        reason="class_support_adequate" if adequate else "min_rows_per_category",
        observed=float(thinnest),
        required=float(minimum),
        constraint=constraint,
        support_required=minimum,
        support_observed=thinnest,
    )


def _assert_gate_ids_are_unique() -> None:
    """Fail at import if a gate identifier is declared twice."""
    for ids in (BINARY_GATE_IDS, CATEGORY_GATE_IDS):
        if len(set(ids)) != len(ids):
            raise ValueError(f"a gate identifier is declared more than once in {ids}")


_assert_gate_ids_are_unique()
