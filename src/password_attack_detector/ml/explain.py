"""Deterministic model attribution: what the frozen champion read, and how much.

This module answers one narrow question -- *which transformed columns moved this
model's decision, and by how much* -- and refuses every wider one.

**Attribution is description, not causation.**  A contribution here says how a
fitted function decomposes over the columns it was handed.  It does not say that
changing the underlying behaviour would change an attacker's success, that the
feature caused the outcome, or that the model is right.  The vocabulary is kept
deliberately flat: ``contribution``, never ``importance``, ``impact``,
``driver``, or ``because``.

**Exact or unavailable.**  Three champion-eligible families decompose exactly
against the published artifact alone, and each reconstructs the model's own
decision quantity to within a declared tolerance:

* logistic regression -- ``value * coefficient`` per column, intercept apart;
* the random forest -- the decision-path decomposition, each split credited with
  the change it makes to the node's stored class distribution, averaged over
  trees, against the ensemble-mean root value as the baseline;
* the single-feature threshold baseline -- one reviewed column carries the whole
  step, and every other column contributes exactly zero because the model does
  not read it.

A family outside that set gets :attr:`~password_attack_detector.ml.enums.ExplanationStatus.UNAVAILABLE`
and a reason code.  There is no approximate path, no surrogate model, and no
sampling: a per-feature number that cannot be reconstructed reads exactly like
one that can once it is in a table.

**Nothing here is fitted, and nothing here reads a label.**  The inputs are a
verified :class:`~password_attack_detector.ml.inference.InferenceModel`, a
transformed matrix built by that model's own frozen preprocessor, and the
configured bound on how much detail to emit.  There is no parameter for ground
truth, no parameter for a split assignment beyond the eligible-split check, and
no code path that writes back to any frozen state.

**No estimator, no private attribute.**  Every method below reads the arrays the
Milestone 4 serializer already published -- ``coefficients``, ``intercept``, the
flat forest node tables -- through the same public shapes inference reads them
through.  Nothing imports scikit-learn, and nothing reaches into a ``Tree``
object.

**Raw score, calibration, and threshold are three separate things.**  The
decomposition is of the *decision function*: the logit for a linear model, the
mean leaf distribution for a forest, the step for the threshold baseline.  It is
not a decomposition of the calibrated probability, and this module never
describes one as a sum of contributions.  The calibrated probability and the
frozen operating point are carried alongside as context, clearly labelled, and
never summed.

**Privacy.**  A contribution names a transformed column, which is an approved
engineered feature name.  It carries a *value* only when the reviewed
configuration turns that on, because a feature value can be a country code.
Row-level explanations carry the anchor join identity their contract requires and
nothing else -- no pseudonym, no campaign, no coordinate, no raw feature row --
and the aggregate report carries no anchor at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.exceptions import ModelNotReadyError
from password_attack_detector.ml.calibration import SealedModel, digest, quantize
from password_attack_detector.ml.enums import (
    EXACT_EXPLANATION_METHODS,
    EXPLANATION_ELIGIBLE_SPLITS,
    ExplanationMethod,
    ExplanationStatus,
    MLSplit,
    ModelFamily,
    ScoreKind,
)
from password_attack_detector.ml.schemas import Sha256Hex, prohibited_metadata_fields

__all__ = [
    "EXPLANATION_SCHEMA_VERSION",
    "RECONSTRUCTION_TOLERANCE",
    "ExplanationManifest",
    "ExplanationQualityReport",
    "FeatureContribution",
    "GlobalContribution",
    "PredictionExplanation",
    "build_explanation_manifest",
    "explain_predictions",
    "explanation_report_to_markdown",
    "global_sensitivity",
    "local_contributions",
    "local_explanation_fingerprint",
    "method_for_family",
    "stored_reconstruction_tolerance",
]

#: Version of the explanation artifact contract.
EXPLANATION_SCHEMA_VERSION: Final = "1.0.0"

#: How far a reconstructed decision quantity may sit from the model's own score
#: before the explanation is refused.
#:
#: Declared rather than discovered.  Both exact methods accumulate in float64
#: over at most a few hundred terms, so the residual is float noise; a residual
#: larger than this means the decomposition and the scorer disagree about what
#: the model does, which is a defect rather than a rounding artifact.
RECONSTRUCTION_TOLERANCE: Final[float] = 1e-9


def stored_reconstruction_tolerance(term_count: int) -> float:
    """Return the residual a *stored* decomposition of *term_count* terms may carry.

    Contributions are written at the layer's serialized precision, so each one
    carries up to half a unit in its last place and the sum carries up to
    ``term_count`` halves of it.  The bound is stated as a function rather than
    as a second constant so it cannot drift from
    :data:`RECONSTRUCTION_TOLERANCE`, and so a reader can see that it grows with
    the matrix rather than with how wrong the arithmetic was allowed to be.
    """
    return RECONSTRUCTION_TOLERANCE * (max(term_count, 1) + 2)


#: The families this build decomposes exactly, and the method each one uses.
_METHOD_BY_FAMILY: Final[Mapping[ModelFamily, ExplanationMethod]] = {
    ModelFamily.LOGISTIC_REGRESSION: ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION,
    ModelFamily.RANDOM_FOREST: ExplanationMethod.TREE_PATH_CONTRIBUTION,
    ModelFamily.SINGLE_FEATURE_THRESHOLD: (
        ExplanationMethod.SINGLE_FEATURE_STEP_CONTRIBUTION
    ),
}

#: Stable reason codes a caller may branch on.
_UNSUPPORTED_FAMILY: Final = "unsupported_model_family"
_NOT_BINARY: Final = "not_a_binary_head"
_NO_ROWS: Final = "no_rows_in_scope"

#: scikit-learn's sentinel for "this node has no child".
_LEAF: Final[int] = -1


def method_for_family(family: ModelFamily) -> ExplanationMethod | None:
    """Return the exact method for *family*, or ``None`` when there is none.

    The registry is closed and the lookup total: a family this build cannot
    decompose selects nothing rather than falling through to a generic path.
    """
    return _METHOD_BY_FAMILY.get(family)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    """One transformed column's signed contribution to one row's decision value.

    ``transformed_feature`` is a column the reviewed allowlist admitted and the
    frozen preprocessor emitted.  ``transformed_value`` is present only when the
    configuration explicitly opted into disclosing values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transformed_feature: str
    contribution: float
    #: The transformed value the contribution was computed from.  ``None``
    #: unless ``explain.include_feature_values`` is on: a feature value can be a
    #: country code, and widening disclosure is a deliberate act.
    transformed_value: float | None = None

    @model_validator(mode="after")
    def check_finite(self) -> Self:
        """A contribution is a finite number or it is not a contribution."""
        if not math.isfinite(self.contribution):
            raise ValueError("contribution must be finite")
        if self.transformed_value is not None and not math.isfinite(
            self.transformed_value
        ):
            raise ValueError("transformed_value must be finite")
        if not self.transformed_feature.strip():
            raise ValueError("a contribution names the column it is attributed to")
        return self


class GlobalContribution(BaseModel):
    """One transformed column's aggregate sensitivity across the scored rows.

    Deliberately *not* a decomposition.  This is the mean absolute change in the
    model's decision score when the column is replaced by a deterministically
    permuted copy of itself: a statement about how much the fitted function
    depends on the column over this population, with no claim that the parts sum
    to anything.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transformed_feature: str
    mean_absolute_score_change: float = Field(ge=0.0)
    #: The mean absolute magnitude of the exact local contributions, when the
    #: family has an exact method.  ``None`` for a family that does not, and
    #: never a zero standing in for one.
    mean_absolute_contribution: float | None = None

    @model_validator(mode="after")
    def check_finite(self) -> Self:
        """Both quantities are finite where present."""
        if not math.isfinite(self.mean_absolute_score_change):
            raise ValueError("mean_absolute_score_change must be finite")
        if self.mean_absolute_contribution is not None:
            if not math.isfinite(self.mean_absolute_contribution):
                raise ValueError("mean_absolute_contribution must be finite")
            if self.mean_absolute_contribution < 0.0:
                raise ValueError("a mean absolute magnitude is non-negative")
        return self


class PredictionExplanation(BaseModel):
    """One row's exact decomposition, and the reconstruction that checks it.

    Carries ``anchor_event_id`` -- the minimum join identity a row-level artifact
    needs to be attributable to the prediction it explains, and the same identity
    the prediction row already carries.  Nothing else identifying: no pseudonym,
    no campaign, no coordinate, no raw feature row, and no label.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_event_id: str
    method: ExplanationMethod
    #: What the contributions sum *to*: the linear logit, the forest's mean leaf
    #: score, or the threshold baseline's step.  Never a calibrated probability.
    decision_value: float
    #: The additive constant the contributions are measured against: a linear
    #: intercept, a forest's ensemble-mean root value, or zero.
    baseline_value: float
    contributions: tuple[FeatureContribution, ...]
    #: ``decision_value - (baseline_value + sum(contributions))``.  Reported
    #: rather than asserted away, so a reader can see it was checked.
    reconstruction_residual: float

    @model_validator(mode="after")
    def check_reconstruction(self) -> Self:
        """The parts add up, in the declared tolerance, or this is not a decomposition."""
        if not self.anchor_event_id.strip():
            raise ValueError("an explanation names the row it explains")
        if self.method not in EXACT_EXPLANATION_METHODS:
            raise ValueError(
                f"{str(self.method)!r} is a global method; it decomposes nothing "
                f"and may not be recorded as a row explanation"
            )
        for value, name in (
            (self.decision_value, "decision_value"),
            (self.baseline_value, "baseline_value"),
            (self.reconstruction_residual, "reconstruction_residual"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        names = [item.transformed_feature for item in self.contributions]
        if len(set(names)) != len(names):
            raise ValueError("an explanation credits each column at most once")
        # The *arithmetic* tolerance is checked against unrounded values before
        # a row is built. What is checked here is the reconstruction after
        # storage, where every term has been rounded to the layer's serialized
        # precision -- so the admissible error grows with the number of terms
        # rather than staying at one term's worth of it. Holding this to the
        # single-term tolerance would refuse a correct decomposition purely for
        # having two hundred columns.
        allowed = stored_reconstruction_tolerance(len(self.contributions))
        total = self.baseline_value + math.fsum(
            item.contribution for item in self.contributions
        )
        if abs(self.decision_value - total) > allowed:
            raise ValueError(
                f"the contributions do not reconstruct the decision value "
                f"within {allowed:g}; an attribution that does not add up is "
                f"refused rather than published"
            )
        if abs(self.reconstruction_residual) > RECONSTRUCTION_TOLERANCE:
            raise ValueError("the recorded residual exceeds the declared tolerance")
        return self


class ExplanationQualityReport(SealedModel):
    """The aggregate attribution summary: column names, magnitudes, no rows.

    Every number here describes the *scored population as a whole*.  There is no
    anchor identifier anywhere in it, no feature value, and no outcome: this
    report says which columns the frozen champion's decisions moved with, not
    whether any of those decisions was correct.
    """

    fingerprint_field: ClassVar[str] = "explanation_report_fingerprint"
    schema_version_field: ClassVar[str] = "explanation_schema_version"
    schema_version: ClassVar[str] = EXPLANATION_SCHEMA_VERSION
    record_label: ClassVar[str] = "explanation report"

    explanation_schema_version: str = EXPLANATION_SCHEMA_VERSION

    status: ExplanationStatus
    #: Stable code naming why no exact decomposition was produced.  ``None``
    #: exactly when the status is :attr:`ExplanationStatus.EXACT`.
    unavailable_reason: str | None = None

    #: The exact local method, absent when there is none for this family.
    method: ExplanationMethod | None
    #: The global method, always present: it is model-agnostic.
    global_method: ExplanationMethod = ExplanationMethod.PERMUTATION_SCORE_SENSITIVITY

    model_family: ModelFamily
    scope: MLSplit
    score_kind: ScoreKind
    explained_row_count: int = Field(ge=0)
    transformed_feature_count: int = Field(ge=1)

    top_contributions: tuple[GlobalContribution, ...]
    #: How many columns the population's exact contributions were never non-zero
    #: for.  ``None`` when there is no exact method, never a zero standing in.
    unused_column_count: int | None = Field(default=None, ge=0)

    permutation_repeats: int = Field(ge=1)
    permutation_seed: int = Field(ge=0)
    #: The largest reconstruction residual observed over every explained row.
    #: ``None`` when nothing was decomposed.
    max_reconstruction_residual: float | None = None

    explanation_report_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_report(self) -> Self:
        """Status, method, and reason agree, and the summary carries no row."""
        exact = self.status is ExplanationStatus.EXACT
        if exact != (self.method is not None):
            raise ValueError(
                "an exact explanation names its method, and an unavailable one "
                "names none"
            )
        if exact == (self.unavailable_reason is not None):
            raise ValueError(
                "an unavailable explanation states a reason, and an exact one "
                "has none to state"
            )
        if self.method is not None and self.method not in EXACT_EXPLANATION_METHODS:
            raise ValueError(f"{str(self.method)!r} is not an exact local method")
        if exact and self.max_reconstruction_residual is None:
            raise ValueError(
                "an exact explanation reports the residual it reconstructed to"
            )
        if not exact:
            if self.max_reconstruction_residual is not None:
                raise ValueError(
                    "nothing was decomposed, so there is no residual to report"
                )
            if self.unused_column_count is not None:
                raise ValueError(
                    "an unused-column count describes an exact decomposition"
                )
        if self.max_reconstruction_residual is not None and (
            not math.isfinite(self.max_reconstruction_residual)
            or self.max_reconstruction_residual > RECONSTRUCTION_TOLERANCE
        ):
            raise ValueError("the reported residual exceeds the declared tolerance")
        names = [item.transformed_feature for item in self.top_contributions]
        if len(set(names)) != len(names):
            raise ValueError("the summary names each column at most once")
        if len(names) > self.transformed_feature_count:
            raise ValueError(
                "the summary names more columns than the model was fitted on"
            )
        return self


class ExplanationManifest(SealedModel):
    """What one explanation run explained, and every frozen thing it is bound to.

    The lineage is carried once here rather than once per row, and it names the
    champion, the preprocessor, and the prediction publication together: an
    explanation is only meaningful against the exact model and the exact input
    that produced the predictions it describes.
    """

    fingerprint_field: ClassVar[str] = "explanation_manifest_fingerprint"
    schema_version_field: ClassVar[str] = "explanation_schema_version"
    schema_version: ClassVar[str] = EXPLANATION_SCHEMA_VERSION
    record_label: ClassVar[str] = "explanation manifest"

    explanation_schema_version: str = EXPLANATION_SCHEMA_VERSION

    #: Derived from the content below, so two runs over the same publication in
    #: two directories produce the same identity.
    explanation_id: str

    champion_lock_fingerprint: Sha256Hex
    champion_scope_key: Sha256Hex
    catalog_model_id: str
    model_id: str
    model_content_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    eligible_feature_list_fingerprint: Sha256Hex
    allowlist_fingerprint: Sha256Hex
    feature_catalog_fingerprint: Sha256Hex

    prediction_id: str
    prediction_manifest_fingerprint: Sha256Hex
    inference_input_fingerprint: Sha256Hex
    scope: MLSplit

    method: ExplanationMethod | None
    global_method: ExplanationMethod
    explain_config_fingerprint: Sha256Hex
    explained_row_count: int = Field(ge=0)
    local_explanation_count: int = Field(ge=0)
    include_feature_values: bool

    explanation_report_fingerprint: Sha256Hex
    #: Digest over every emitted row explanation in canonical order.  ``None``
    #: when none were emitted -- the configured bound defaults to zero.
    local_explanation_fingerprint: Sha256Hex | None = None

    explanation_manifest_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_manifest(self) -> Self:
        """The scope is explainable and the local artifact is named or absent."""
        if self.scope not in EXPLANATION_ELIGIBLE_SPLITS:
            raise ValueError(
                f"scope {str(self.scope)!r} may not be explained; attribution "
                f"runs over training or validation rows, never over the locked "
                f"evaluation population"
            )
        if (self.local_explanation_count > 0) != (
            self.local_explanation_fingerprint is not None
        ):
            raise ValueError(
                "emitted row explanations are bound by digest, and an empty set "
                "binds nothing"
            )
        if self.local_explanation_count > self.explained_row_count:
            raise ValueError("more rows were explained individually than were scored")
        return self


def _assert_no_outcome_field() -> None:
    """Refuse at import a schema here that declares a prohibited field name.

    The same structural guard the quality report carries.  An explanation is a
    statement about a model, so a field named for a label, a campaign, a
    pseudonym, or a split assignment could only ever be a mistake.
    """
    for model in (
        FeatureContribution,
        GlobalContribution,
        PredictionExplanation,
        ExplanationQualityReport,
        ExplanationManifest,
    ):
        # ``anchor_event_id`` is the one join identity a row-level explanation
        # is contractually allowed, and it is checked separately: it may appear
        # on ``PredictionExplanation`` and nowhere else in this module.
        declared = set(model.model_fields) - {"anchor_event_id"}
        offending = prohibited_metadata_fields(declared)
        if offending:
            raise AssertionError(
                f"{model.__name__} declares prohibited field(s) {list(offending)}"
            )
    for model in (
        FeatureContribution,
        GlobalContribution,
        ExplanationQualityReport,
        ExplanationManifest,
    ):
        if "anchor_event_id" in model.model_fields:
            raise AssertionError(
                f"{model.__name__} carries a row identity; only a row-level "
                f"explanation may, and an aggregate report may never"
            )


_assert_no_outcome_field()


# ---------------------------------------------------------------------------
# Exact local decomposition
# ---------------------------------------------------------------------------


def _positive_index(class_order: Sequence[str]) -> int:
    """Return the column of the score the operating point is applied to."""
    if len(class_order) != 2:
        raise ModelNotReadyError(
            "an exact local decomposition is defined for the binary head; a "
            "head with a different class count decomposes into a different "
            "quantity and is not published as one"
        )
    return 1


def _linear_contributions(
    fitted: Any, matrix: Sequence[Sequence[float]]
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return per-row ``value * coefficient`` and the per-row intercept.

    The binary decision value scikit-learn computes is
    ``X @ coef_[0] + intercept_[0]``, which is exactly what the published
    ``coefficients`` and ``intercept`` arrays hold and what
    :mod:`~password_attack_detector.ml.models.linear` scores from.  Reading the
    same arrays is what makes the sum reconstruct the score rather than
    approximate it.
    """
    coefficients = np.asarray(fitted.arrays["coefficients"], dtype=np.float64)
    intercept = np.asarray(fitted.arrays["intercept"], dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[0] < 1:
        raise ModelNotReadyError(
            "the stored coefficient array does not describe a binary head"
        )
    weights = coefficients[0]
    if len(weights) != len(fitted.transformed_feature_names):
        raise ModelNotReadyError(
            "the stored coefficients and the transformed column order disagree"
        )
    design = np.asarray(matrix, dtype=np.float64).reshape(len(matrix), len(weights))
    products = design * weights
    constant = float(intercept[0])
    return (
        tuple(tuple(float(cell) for cell in row) for row in products),
        tuple(constant for _ in range(len(matrix))),
    )


def _forest_contributions(
    fitted: Any, matrix: Sequence[Sequence[float]]
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return the decision-path decomposition of the forest's positive score.

    For one tree, walking root to leaf, each split is credited with the change
    it makes to the node's stored positive-class share.  Telescoping over the
    path leaves ``leaf - root``; averaging over trees leaves
    ``mean_leaf - mean_root``, and ``mean_leaf`` is exactly what
    :func:`~password_attack_detector.ml.models.forest.traverse_forest` returns.
    So the contributions plus the ensemble-mean root value reconstruct the
    forest's own score, and the equality is arithmetic rather than empirical.

    The comparison is made at ``float32`` against a ``float64`` threshold --
    the estimator's own contract, reproduced here for the same reason the
    scorer reproduces it: a row sitting on a cut takes a different branch
    otherwise.
    """
    from password_attack_detector.ml.models.forest import TREE_INPUT_DTYPE

    arrays = fitted.arrays
    offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
    left = np.asarray(arrays["children_left"], dtype=np.int64)
    right = np.asarray(arrays["children_right"], dtype=np.int64)
    feature = np.asarray(arrays["split_feature"], dtype=np.int64)
    threshold = np.asarray(arrays["split_threshold"], dtype=np.float64)
    values = np.asarray(arrays["leaf_value"], dtype=np.float64)

    column_count = len(fitted.transformed_feature_names)
    positive = _positive_index(fitted.class_order)
    tree_count = len(offsets) - 1
    if tree_count < 1:
        raise ModelNotReadyError("the stored ensemble declares no tree")

    compared = np.asarray(matrix, dtype=np.dtype(TREE_INPUT_DTYPE)).reshape(
        len(matrix), column_count
    )
    baseline = float(
        np.mean([values[int(offsets[tree]), positive] for tree in range(tree_count)])
    )

    per_row: list[tuple[float, ...]] = []
    for row_index in range(len(matrix)):
        credit = np.zeros(column_count, dtype=np.float64)
        for tree in range(tree_count):
            node = int(offsets[tree])
            stop = int(offsets[tree + 1])
            steps = 0
            while left[node] != _LEAF:
                steps += 1
                if steps > stop - int(offsets[tree]):
                    raise ModelNotReadyError(
                        "traversal exceeded the tree's node count; the stored "
                        "children describe a cycle"
                    )
                column = int(feature[node])
                if not 0 <= column < column_count:
                    raise ModelNotReadyError(
                        "a stored split names a feature outside the matrix"
                    )
                # The stored children are *tree-local* indices; the node arrays
                # are concatenated across the ensemble. Adding the tree's own
                # offset is what the scorer does, and omitting it would walk
                # into a neighbouring tree and still produce plausible numbers.
                local = (
                    int(left[node])
                    if compared[row_index, column] <= threshold[node]
                    else int(right[node])
                )
                child = int(offsets[tree]) + local
                if not int(offsets[tree]) <= child < stop:
                    raise ModelNotReadyError(
                        "a stored child index falls outside its own tree"
                    )
                credit[column] += float(values[child, positive]) - float(
                    values[node, positive]
                )
                node = child
        per_row.append(tuple(float(cell / tree_count) for cell in credit))
    return tuple(per_row), tuple(baseline for _ in range(len(matrix)))


def _threshold_contributions(
    fitted: Any, matrix: Sequence[Sequence[float]]
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return the step attribution for the single-feature threshold baseline.

    The model reads one column and votes ``1.0`` or ``0.0``.  That column is
    therefore credited with the whole vote and every other column with exactly
    zero -- not as an approximation, but because the fitted function is constant
    in each of them.  The baseline is ``0.0``: an unflagged row is the model
    saying nothing rather than the model saying something that cancels.
    """
    column_count = len(fitted.transformed_feature_names)
    index = int(fitted.parameters["feature_index"])
    cut = float(np.asarray(fitted.arrays["threshold"], dtype=np.float64)[0])
    direction = str(fitted.parameters["direction"])
    if not 0 <= index < column_count:
        raise ModelNotReadyError(
            "the stored threshold names a column outside the matrix"
        )

    per_row: list[tuple[float, ...]] = []
    for row in matrix:
        value = float(row[index])
        flagged = value > cut if direction == "above" else value < cut
        credit = [0.0] * column_count
        credit[index] = 1.0 if flagged else 0.0
        per_row.append(tuple(credit))
    return tuple(per_row), tuple(0.0 for _ in range(len(matrix)))


def local_contributions(
    model: Any, matrix: Sequence[Sequence[float]]
) -> tuple[
    ExplanationMethod,
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return the method, per-row contributions, baselines, and decision values.

    The decision value is the quantity the contributions reconstruct, taken from
    the model's own scorer rather than recomputed here: the whole point of the
    residual check is that two independent paths agree.

    Raises:
        ModelNotReadyError: when *model*'s family has no exact decomposition, or
            when its stored arrays do not describe the head this expects.
    """
    fitted = model.fitted
    method = method_for_family(fitted.family)
    if method is None:
        raise ModelNotReadyError(
            f"model family {str(fitted.family)!r} has no exact local "
            f"decomposition in this build; an approximate one is not published "
            f"in its place"
        )
    positive = _positive_index(fitted.class_order)

    if method is ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION:
        contributions, baselines = _linear_contributions(fitted, matrix)
        # The published sigmoid is not what the coefficients sum to; the logit
        # is. Recovering it from the reported score keeps the two paths
        # independent while still comparing like with like.
        scored = model.score(matrix, fitted.transformed_feature_names)
        decisions = tuple(_logit(float(row[positive])) for row in scored)
    elif method is ExplanationMethod.TREE_PATH_CONTRIBUTION:
        contributions, baselines = _forest_contributions(fitted, matrix)
        scored = model.score(matrix, fitted.transformed_feature_names)
        decisions = tuple(float(row[positive]) for row in scored)
    else:
        contributions, baselines = _threshold_contributions(fitted, matrix)
        scored = model.score(matrix, fitted.transformed_feature_names)
        decisions = tuple(float(row[positive]) for row in scored)

    return method, contributions, baselines, decisions


def _logit(probability: float) -> float:
    """Return the log-odds the linear head's sigmoid was taken of.

    ``score`` publishes ``sigmoid(z)``; the coefficients sum to ``z``.  Inverting
    is exact in float64 across the range the sigmoid actually produces, and a
    saturated score -- which no finite ``z`` reaches -- is refused rather than
    turned into an infinity that would then fail the residual check anyway.
    """
    if not 0.0 < probability < 1.0:
        raise ModelNotReadyError(
            "the linear head reported a saturated score; its log-odds is not "
            "finite, so no decomposition of it can be checked"
        )
    return float(math.log(probability / (1.0 - probability)))


# ---------------------------------------------------------------------------
# Global sensitivity
# ---------------------------------------------------------------------------


def global_sensitivity(
    model: Any,
    matrix: Sequence[Sequence[float]],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, ...]:
    """Return each column's mean absolute decision-score change under permutation.

    Model-agnostic, label-free, and deterministic: the permutation for repeat
    *r* of column *j* is drawn from a generator seeded by ``(seed, r, j)``, so
    two runs of this function over the same matrix produce the same numbers and
    the order the columns are visited in cannot change any of them.

    This is **not** a decomposition and is never reported as one.  It says how
    much the fitted function's output moves when a column stops carrying its own
    values -- a property of the model over this population, with no claim that
    the per-column numbers sum to the score or to each other.

    A permutation-importance measured against a *metric* would need labels.
    This one is measured against the model's own output, which is why it is
    available here at all.
    """
    if repeats < 1:
        raise ModelNotReadyError("permutation sensitivity needs at least one repeat")
    columns = tuple(model.fitted.transformed_feature_names)
    positive = _positive_index(model.fitted.class_order)
    design = np.asarray(matrix, dtype=np.float64).reshape(len(matrix), len(columns))
    if len(design) == 0:
        return tuple(0.0 for _ in columns)

    base = np.asarray(
        [row[positive] for row in model.score(matrix, columns)], dtype=np.float64
    )

    sensitivities: list[float] = []
    for index in range(len(columns)):
        total = 0.0
        for repeat in range(repeats):
            generator = np.random.default_rng([seed, repeat, index])
            order = generator.permutation(len(design))
            shuffled = design.copy()
            shuffled[:, index] = design[order, index]
            moved = np.asarray(
                [
                    row[positive]
                    for row in model.score(
                        tuple(tuple(float(c) for c in r) for r in shuffled), columns
                    )
                ],
                dtype=np.float64,
            )
            total += float(np.mean(np.abs(moved - base)))
        sensitivities.append(quantize(total / repeats))
    return tuple(sensitivities)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def explain_predictions(
    *,
    model: Any,
    matrix: Sequence[Sequence[float]],
    anchor_event_ids: Sequence[str],
    scope: MLSplit,
    score_kind: ScoreKind,
    top_k_features: int,
    permutation_repeats: int,
    permutation_seed: int,
    max_local_explanations: int,
    include_feature_values: bool,
) -> tuple[ExplanationQualityReport, tuple[PredictionExplanation, ...]]:
    """Attribute a frozen model's decisions over an already-transformed matrix.

    Nothing here fits, and nothing here is handed a label: the arguments are a
    verified model, the matrix its own preprocessor produced, the anchors those
    rows carry, and the reviewed bounds on how much to emit.

    The global sensitivity is always computed -- it is model-agnostic.  The exact
    local decomposition is computed when the family has one, and its absence is
    reported as a typed status with a reason rather than filled in with an
    approximation.

    Args:
        model: a verified :class:`~password_attack_detector.ml.inference.InferenceModel`.
        matrix: rows in the model's own transformed column order.
        anchor_event_ids: the join identity of each row, in matrix order.
        scope: the split the rows came from.  Test and the novel-anomaly holdout
            are refused.
        score_kind: what the frozen operating point was applied to.  Recorded
            for context and never summed with a contribution.
        top_k_features: how many columns the aggregate summary names.
        permutation_repeats: repeats per column for the global measure.
        permutation_seed: the run seed the permutations are derived from.
        max_local_explanations: the bound on emitted row explanations.
        include_feature_values: whether contributions may carry their value.

    Raises:
        ModelNotReadyError: on an ineligible scope, a row/anchor length
            mismatch, or a stored artifact this build cannot decompose the way
            its family declares.
    """
    if scope not in EXPLANATION_ELIGIBLE_SPLITS:
        raise ModelNotReadyError(
            f"attribution over {str(scope)!r} is refused; explanations are "
            f"computed on training or validation rows, never on the locked "
            f"evaluation population"
        )
    if len(anchor_event_ids) != len(matrix):
        raise ModelNotReadyError(
            "the anchors and the transformed matrix describe different row counts"
        )

    columns = tuple(model.fitted.transformed_feature_names)
    family = model.fitted.family

    if len(model.fitted.class_order) != 2:
        return (
            _unavailable_report(
                family=family,
                scope=scope,
                score_kind=score_kind,
                columns=columns,
                row_count=len(matrix),
                reason=_NOT_BINARY,
                repeats=permutation_repeats,
                seed=permutation_seed,
                sensitivities=(),
                top_k=top_k_features,
            ),
            (),
        )
    if not matrix:
        return (
            _unavailable_report(
                family=family,
                scope=scope,
                score_kind=score_kind,
                columns=columns,
                row_count=0,
                reason=_NO_ROWS,
                repeats=permutation_repeats,
                seed=permutation_seed,
                sensitivities=tuple(0.0 for _ in columns),
                top_k=top_k_features,
            ),
            (),
        )

    sensitivities = global_sensitivity(
        model, matrix, repeats=permutation_repeats, seed=permutation_seed
    )

    if method_for_family(family) is None:
        return (
            _unavailable_report(
                family=family,
                scope=scope,
                score_kind=score_kind,
                columns=columns,
                row_count=len(matrix),
                reason=_UNSUPPORTED_FAMILY,
                repeats=permutation_repeats,
                seed=permutation_seed,
                sensitivities=sensitivities,
                top_k=top_k_features,
            ),
            (),
        )

    method, contributions, baselines, decisions = local_contributions(model, matrix)

    magnitudes = [0.0] * len(columns)
    residuals: list[float] = []
    for row, baseline, decision in zip(
        contributions, baselines, decisions, strict=True
    ):
        for index, value in enumerate(row):
            magnitudes[index] += abs(value)
        residuals.append(decision - (baseline + math.fsum(row)))

    worst = max((abs(value) for value in residuals), default=0.0)
    if worst > RECONSTRUCTION_TOLERANCE:
        raise ModelNotReadyError(
            f"the {str(method)!r} decomposition disagrees with the model's own "
            f"score by {worst:g}, above the declared tolerance "
            f"{RECONSTRUCTION_TOLERANCE:g}; the attribution is refused rather "
            f"than published with a caveat"
        )

    means = tuple(quantize(total / len(matrix)) for total in magnitudes)
    unused = sum(1 for value in means if value == 0.0)

    explained = _row_explanations(
        method=method,
        columns=columns,
        contributions=contributions,
        baselines=baselines,
        decisions=decisions,
        residuals=residuals,
        matrix=matrix,
        anchor_event_ids=anchor_event_ids,
        limit=max_local_explanations,
        include_feature_values=include_feature_values,
    )

    report = ExplanationQualityReport.seal(
        status=ExplanationStatus.EXACT,
        unavailable_reason=None,
        method=method,
        model_family=family,
        scope=scope,
        score_kind=score_kind,
        explained_row_count=len(matrix),
        transformed_feature_count=len(columns),
        top_contributions=_top(columns, sensitivities, means, limit=top_k_features),
        unused_column_count=unused,
        permutation_repeats=permutation_repeats,
        permutation_seed=permutation_seed,
        max_reconstruction_residual=quantize(worst),
    )
    return report, explained


def _unavailable_report(
    *,
    family: ModelFamily,
    scope: MLSplit,
    score_kind: ScoreKind,
    columns: Sequence[str],
    row_count: int,
    reason: str,
    repeats: int,
    seed: int,
    sensitivities: Sequence[float],
    top_k: int,
) -> ExplanationQualityReport:
    """Return the report for a run that produced no exact decomposition."""
    return ExplanationQualityReport.seal(
        status=ExplanationStatus.UNAVAILABLE,
        unavailable_reason=reason,
        method=None,
        model_family=family,
        scope=scope,
        score_kind=score_kind,
        explained_row_count=row_count,
        transformed_feature_count=len(columns),
        top_contributions=_top(columns, sensitivities, None, limit=top_k),
        unused_column_count=None,
        permutation_repeats=repeats,
        permutation_seed=seed,
        max_reconstruction_residual=None,
    )


def _top(
    columns: Sequence[str],
    sensitivities: Sequence[float],
    magnitudes: Sequence[float] | None,
    *,
    limit: int,
) -> tuple[GlobalContribution, ...]:
    """Return the *limit* most sensitive columns, ties broken by name.

    Sorting by name within a tie is what makes the summary reproducible: two
    columns a model is equally insensitive to would otherwise appear in
    whichever order the matrix happened to declare them.
    """
    if not sensitivities:
        return ()
    rows = [
        GlobalContribution(
            transformed_feature=name,
            mean_absolute_score_change=quantize(float(sensitivity)),
            mean_absolute_contribution=(
                None if magnitudes is None else quantize(float(magnitudes[index]))
            ),
        )
        for index, (name, sensitivity) in enumerate(
            zip(columns, sensitivities, strict=True)
        )
    ]
    rows.sort(
        key=lambda item: (-item.mean_absolute_score_change, item.transformed_feature)
    )
    return tuple(rows[:limit])


def _row_explanations(
    *,
    method: ExplanationMethod,
    columns: Sequence[str],
    contributions: Sequence[Sequence[float]],
    baselines: Sequence[float],
    decisions: Sequence[float],
    residuals: Sequence[float],
    matrix: Sequence[Sequence[float]],
    anchor_event_ids: Sequence[str],
    limit: int,
    include_feature_values: bool,
) -> tuple[PredictionExplanation, ...]:
    """Return up to *limit* row explanations, in canonical anchor order.

    Which rows are emitted must not depend on scoring order, so the bound is
    applied after sorting by anchor rather than to whatever the matrix happened
    to hold first.
    """
    if limit <= 0:
        return ()
    order = sorted(range(len(matrix)), key=lambda index: anchor_event_ids[index])
    chosen = order[:limit]
    return tuple(
        PredictionExplanation(
            anchor_event_id=anchor_event_ids[index],
            method=method,
            decision_value=quantize(decisions[index]),
            baseline_value=quantize(baselines[index]),
            contributions=tuple(
                FeatureContribution(
                    transformed_feature=name,
                    contribution=quantize(contributions[index][position]),
                    transformed_value=(
                        quantize(float(matrix[index][position]))
                        if include_feature_values
                        else None
                    ),
                )
                for position, name in enumerate(columns)
            ),
            reconstruction_residual=quantize(residuals[index]),
        )
        for index in chosen
    )


def local_explanation_fingerprint(
    explanations: Sequence[PredictionExplanation],
) -> str:
    """Return the digest binding a manifest to the row explanations it declares."""
    return digest(
        [
            item.model_dump(mode="json")
            for item in sorted(explanations, key=lambda row: row.anchor_event_id)
        ]
    )


def build_explanation_manifest(
    *,
    lock: Any,
    prediction_manifest: Any,
    report: ExplanationQualityReport,
    explanations: Sequence[PredictionExplanation],
    explain_config_fingerprint: str,
    include_feature_values: bool,
) -> ExplanationManifest:
    """Return the manifest identifying one explanation run.

    The identity is derived from content alone -- the champion, the publication,
    the method, the configuration, and the digests of what was produced -- so two
    runs over the same publication in two directories agree byte for byte.
    """
    lineage = prediction_manifest.lineage
    if lineage.champion_lock_fingerprint != lock.lock_fingerprint:
        raise ModelNotReadyError(
            "the publication was produced by a different champion than the one "
            "supplied; an explanation of one model's output attributed to "
            "another model's coefficients would be wrong in a way no digest "
            "would catch"
        )
    local_fingerprint = (
        local_explanation_fingerprint(explanations) if explanations else None
    )
    identity = digest(
        {
            "champion_lock_fingerprint": lock.lock_fingerprint,
            "explain_config_fingerprint": explain_config_fingerprint,
            "explanation_report_fingerprint": (report.explanation_report_fingerprint),
            "explanation_schema_version": EXPLANATION_SCHEMA_VERSION,
            "local_explanation_fingerprint": local_fingerprint,
            "prediction_manifest_fingerprint": (
                prediction_manifest.prediction_manifest_fingerprint
            ),
        }
    )
    return ExplanationManifest.seal(
        explanation_id=identity,
        champion_lock_fingerprint=lock.lock_fingerprint,
        champion_scope_key=lock.scope_key,
        catalog_model_id=lock.catalog_model_id,
        model_id=lock.model_id,
        model_content_fingerprint=lock.model_content_fingerprint,
        preprocessor_fingerprint=lock.preprocessor_fingerprint,
        eligible_feature_list_fingerprint=lock.eligible_feature_list_fingerprint,
        allowlist_fingerprint=lock.allowlist_fingerprint,
        feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
        prediction_id=prediction_manifest.prediction_id,
        prediction_manifest_fingerprint=(
            prediction_manifest.prediction_manifest_fingerprint
        ),
        inference_input_fingerprint=prediction_manifest.inference_input_fingerprint,
        scope=prediction_manifest.scope,
        method=report.method,
        global_method=report.global_method,
        explain_config_fingerprint=explain_config_fingerprint,
        explained_row_count=report.explained_row_count,
        local_explanation_count=len(explanations),
        include_feature_values=include_feature_values,
        explanation_report_fingerprint=report.explanation_report_fingerprint,
        local_explanation_fingerprint=local_fingerprint,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(value: float | int | None) -> str:
    """Render a number, or say the quantity is unavailable rather than zero."""
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.6f}"


def explanation_report_to_markdown(
    report: ExplanationQualityReport, manifest: ExplanationManifest
) -> str:
    """Render the aggregate attribution summary as deterministic Markdown.

    No anchor identifier, no feature value, and no outcome. The caveats are part
    of the document rather than a footnote a reader can skip: a table of column
    names beside numbers reads as a causal ranking unless it says otherwise.
    """
    lines = [
        "# ML explanation report",
        "",
        "Deterministic model attribution over a frozen champion's own output.",
        "**Descriptive, not causal.** A contribution says how the fitted "
        "function decomposes over the columns it was handed; it does not say "
        "that the behaviour behind a column caused anything, and it is not "
        "evidence that the model is right.",
        "",
        "## Identity",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Explanation schema | {report.explanation_schema_version} |",
        f"| Explanation id | `{manifest.explanation_id}` |",
        f"| Champion lock | `{manifest.champion_lock_fingerprint}` |",
        f"| Catalog model | {manifest.catalog_model_id} |",
        f"| Model family | {report.model_family} |",
        f"| Prediction id | `{manifest.prediction_id}` |",
        f"| Prediction manifest | `{manifest.prediction_manifest_fingerprint}` |",
        f"| Preprocessor | `{manifest.preprocessor_fingerprint}` |",
        f"| Scope | {report.scope} |",
        f"| Report fingerprint | `{report.explanation_report_fingerprint}` |",
        "",
        "## Method",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Status | {report.status} |",
        f"| Local method | {report.method or 'unavailable'} |",
        f"| Unavailable reason | {report.unavailable_reason or 'not applicable'} |",
        f"| Global method | {report.global_method} |",
        f"| Score kind | {report.score_kind} |",
        f"| Explained rows | {report.explained_row_count:,} |",
        f"| Transformed columns | {report.transformed_feature_count:,} |",
        f"| Row explanations emitted | {manifest.local_explanation_count:,} |",
        f"| Feature values disclosed | "
        f"{'yes' if manifest.include_feature_values else 'no'} |",
        f"| Permutation repeats | {report.permutation_repeats:,} |",
        f"| Permutation seed | {report.permutation_seed:,} |",
        f"| Max reconstruction residual | "
        f"{_cell(report.max_reconstruction_residual)} |",
        f"| Unused columns | {_cell(report.unused_column_count)} |",
        "",
        "## Columns by sensitivity",
        "",
        "`Mean |score change|` is the model-agnostic permutation measure and "
        "decomposes nothing. `Mean |contribution|` is the magnitude of the "
        "exact local decomposition, where the family has one.",
        "",
        "| Transformed feature | Mean \\|score change\\| | Mean \\|contribution\\| |",
        "| --- | --- | --- |",
    ]
    for item in report.top_contributions:
        lines.append(
            f"| `{item.transformed_feature}` | "
            f"{_cell(item.mean_absolute_score_change)} | "
            f"{_cell(item.mean_absolute_contribution)} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- Attribution describes this fitted model on this population. It does "
        "not transfer to another model, another split, or real traffic.",
        "- The decomposition is of the **raw decision quantity**. Calibration "
        "and the frozen decision threshold are separate transformations "
        "applied afterwards, and no contribution here is a share of a "
        "calibrated probability.",
        "- The global measure reports how much the output moves when a column "
        "is scrambled. Correlated columns can mask each other, and a small "
        "number is not evidence a column is unused.",
        "- No label was read, so nothing here says whether any decision was correct.",
        "",
    ]
    return "\n".join(lines)
