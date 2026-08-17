"""Domain enumerations for the machine-learning detection layer.

Every enum uses :class:`~enum.StrEnum` so it serialises to its string value in
Pydantic models, Parquet columns, JSON reports, and fingerprints -- the same
contract ``data/enums.py`` and ``detection/enums.py`` already follow.

Three design points are load-bearing:

* :class:`ScoreKind` separates a **calibrated probability** from every other
  kind of number a model can emit.  The word "probability" is reserved for
  ``CALIBRATED_PROBABILITY`` alone; a raw estimator output is a
  ``DECISION_SCORE``, a per-class output is a ``CLASS_SCORE``, and an anomaly
  detector emits an ``ANOMALY_SCORE``.  None of the latter three is a
  probability, a likelihood, or a confidence, and :func:`is_probability`
  is the single place that decides.
* :class:`ValidationPartition` has **no** test member and **no** holdout
  member.  Calibration and threshold selection read a partition of the
  validation split and nothing else; the absence of the member is the
  enforcement, exactly as ``detection run`` enforces label isolation by having
  no ``--labels`` option.
* :data:`UNKNOWN_CATEGORY` is a first-class category outcome.  A row the
  category head cannot assign is labelled ``unknown`` rather than pushed into
  the nearest known class, so novel behaviour is never laundered into a
  category somebody wrote a rule for.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "DRIFT_REFERENCE_ELIGIBLE_SPLITS",
    "EXACT_EXPLANATION_METHODS",
    "EXPLANATION_ELIGIBLE_SPLITS",
    "FIT_ELIGIBLE_SPLITS",
    "PROBABILITY_SCORE_KINDS",
    "SUPERVISED_TASKS",
    "UNKNOWN_CATEGORY",
    "AcceptanceStatus",
    "AnomalyThresholdMethod",
    "AuditCheckStatus",
    "AuditStatus",
    "CalibrationEvaluationKind",
    "CalibrationMethod",
    "CalibrationStatus",
    "ChampionStatus",
    "ComparisonSystem",
    "DriftMetric",
    "DriftStatus",
    "ExperimentRecordType",
    "ExplanationMethod",
    "ExplanationStatus",
    "FeatureDecisionPoint",
    "FusionStrategy",
    "GateStatus",
    "HyperparameterKind",
    "MLSplit",
    "MLTask",
    "MetricStatus",
    "ModelEligibilityStatus",
    "ModelFamily",
    "PredictionDriftQuantity",
    "ReferenceFeatureKind",
    "ScoreKind",
    "SelectionStatus",
    "ServingScope",
    "TestEvaluationStatus",
    "ThresholdObjective",
    "TrainingRunStatus",
    "ValidationPartition",
    "ValidationPartitionStatus",
    "is_probability",
]


class MLTask(StrEnum):
    """What a model predicts.

    ``BINARY_MALICIOUS`` is the primary task and the only one whose champion is
    compared against the Phase 4 rule engine.  ``ATTACK_CATEGORY`` is a triage
    head fitted on known-malicious rows only.  ``ANOMALY`` is unsupervised and
    is never a supervised champion.
    """

    BINARY_MALICIOUS = "binary_malicious"
    ATTACK_CATEGORY = "attack_category"
    ANOMALY = "anomaly"


#: Tasks fitted against ground-truth labels.  ``ANOMALY`` is deliberately
#: absent: it is fitted on benign training rows without reading a label.
SUPERVISED_TASKS: Final[frozenset[MLTask]] = frozenset(
    {MLTask.BINARY_MALICIOUS, MLTask.ATTACK_CATEGORY}
)


class ModelFamily(StrEnum):
    """The learning algorithm behind a model specification.

    Family names describe the *method*, never a verdict.  There is no
    ``attack_detector`` family, and there never will be.
    """

    PRIOR_BASELINE = "prior_baseline"
    SINGLE_FEATURE_THRESHOLD = "single_feature_threshold"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    HISTOGRAM_GRADIENT_BOOSTING = "histogram_gradient_boosting"
    ISOLATION_FOREST = "isolation_forest"


class ScoreKind(StrEnum):
    """What kind of number a model emits.

    Only ``CALIBRATED_PROBABILITY`` may be described as a probability, and only
    after a calibrator has been fitted *and* its calibration error measured.
    Everything else is an ordered magnitude: useful for ranking, meaningless as
    a likelihood.
    """

    #: A value in ``[0, 1]`` produced by a fitted, calibration-evaluated
    #: calibrator.  The only score kind the word "probability" may describe.
    CALIBRATED_PROBABILITY = "calibrated_probability"
    #: A raw binary estimator output.  Ordered, bounded by the estimator's own
    #: contract, and **not** a probability.
    DECISION_SCORE = "decision_score"
    #: A raw per-class estimator output for the multiclass category head.
    CLASS_SCORE = "class_score"
    #: An unsupervised outlier magnitude.  Lower means more anomalous under the
    #: scikit-learn convention; it is never a probability.
    ANOMALY_SCORE = "anomaly_score"


#: The score kinds the word "probability" is permitted to describe.  Exactly
#: one member, and a test asserts it stays that way.
PROBABILITY_SCORE_KINDS: Final[frozenset[ScoreKind]] = frozenset(
    {ScoreKind.CALIBRATED_PROBABILITY}
)


def is_probability(kind: ScoreKind) -> bool:
    """Return whether *kind* may be described as a probability.

    The single decision point for probability terminology across the ML layer.
    Schemas, reports, and column naming all route through it rather than
    re-deriving the rule, so loosening it is one visible edit rather than a
    dozen invisible ones.
    """
    return kind in PROBABILITY_SCORE_KINDS


class CalibrationMethod(StrEnum):
    """How a raw decision score is mapped onto a calibrated probability.

    ``NONE`` is an ordinary, valid choice: a model may ship uncalibrated, in
    which case its output stays a :attr:`ScoreKind.DECISION_SCORE` and no
    report may call it a probability.
    """

    NONE = "none"
    #: Logistic regression fitted on the one-dimensional decision score.
    PLATT = "platt"
    #: Monotone piecewise-constant fit; more flexible, needs more data.
    ISOTONIC = "isotonic"


class CalibrationStatus(StrEnum):
    """Whether a calibrator was fitted, and why not when it was not.

    ``NOT_CALIBRATED`` is an ordinary, valid outcome: a configuration that names
    :attr:`CalibrationMethod.NONE` asked for no calibrator, and got none.  The
    other two are refusals, and neither may be read as a fit: a state object is
    produced only for :attr:`FITTED`, so nothing downstream can describe an
    output as a probability on the strength of a failed fit.
    """

    #: A calibrator was fitted from validation-A and its contract validated.
    FITTED = "fitted"
    #: The configuration asked for no calibrator.  Scores stay uncalibrated.
    NOT_CALIBRATED = "not_calibrated"
    #: Validation-A did not carry enough rows, enough of each class, or enough
    #: distinct scores for the configured method to mean anything.
    INSUFFICIENT_CALIBRATION_SUPPORT = "insufficient_calibration_support"
    #: The solver ran and did not converge, or produced a non-finite parameter.
    #: Distinct from insufficient support: the data was adequate and the fit
    #: still failed, which is a different thing to investigate.
    CONVERGENCE_FAILED = "convergence_failed"


class CalibrationEvaluationKind(StrEnum):
    """What a calibration report is evidence *of*.

    The distinction is the whole point.  A calibrator fitted on validation-A and
    then measured on validation-A has been asked to describe the rows it was
    shaped by, and it will do well at that whether or not it generalises.  That
    number is a useful diagnostic -- a wildly bad one means the fit went wrong --
    and it is not evidence about calibration quality.

    Only :attr:`OUT_OF_SAMPLE_VALIDATION` may be offered to a later champion
    gate, and a report says which it is rather than leaving a reader to work it
    out from the partition it names.
    """

    #: The frozen calibrator measured on the validation-A rows that fitted it.
    #: Inspectable, useful for spotting a pathological fit, and **never**
    #: admissible as champion calibration-quality evidence.
    IN_SAMPLE_FIT_DIAGNOSTIC = "in_sample_fit_diagnostic"
    #: The frozen calibrator measured on validation-B, which it never saw.  The
    #: authoritative calibration-quality evidence for later model selection.
    OUT_OF_SAMPLE_VALIDATION = "out_of_sample_validation"


class MetricStatus(StrEnum):
    """Whether an aggregate measurement means anything.

    Three states, and the distinction between the last two is load-bearing.
    ``UNAVAILABLE`` means the quantity is not defined at all -- an empty
    denominator, a bin nothing landed in.  ``INSUFFICIENT_SUPPORT`` means it is
    defined and was computed, but over too few rows to be evidence.  Collapsing
    either into a number would let a reliability bin holding three rows read
    exactly like one holding three thousand.
    """

    MEASURED = "measured"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    UNAVAILABLE = "unavailable"


class TrainingRunStatus(StrEnum):
    """How far one configured candidate got, and where it stopped.

    A configured candidate that cannot be trained does **not** disappear.  It
    produces a run outcome carrying the status below and the requirements it
    failed, because a candidate list that silently shrinks is a comparison
    nobody can audit: a family missing from a later report should be
    distinguishable from a family that was never configured.

    Only :attr:`COMPLETED` means every artifact a later selection needs is
    present.  Every other member names a specific missing piece, and none of
    them is a failure of the run as a whole -- the orchestration completes and
    reports them.
    """

    #: Fitted, published, and carrying every artifact its task requires.
    COMPLETED = "completed"
    #: The family cannot be trained under this configuration at all: the
    #: catalog does not admit it for the task, or a required reviewed setting
    #: was not supplied.
    UNAVAILABLE = "unavailable"
    #: The model fitted, and the published artifact failed verification.
    FAILED_VALIDATION = "failed_validation"
    #: A support floor was not met, so nothing downstream could mean anything.
    INSUFFICIENT_SUPPORT = "insufficient_support"
    #: Configured calibration could not be fitted on validation-A.
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"
    #: No operating point could be selected on validation-B.
    THRESHOLD_UNAVAILABLE = "threshold_unavailable"
    #: The family has no proven serializer, so a fitted model may be compared
    #: in process but never stored.  M-021 is the standing example.
    SERIALIZER_UNAVAILABLE = "serializer_unavailable"

    @property
    def complete(self) -> bool:
        """Return whether this run produced every artifact its task requires."""
        return self is TrainingRunStatus.COMPLETED


class SelectionStatus(StrEnum):
    """The outcome of selecting an operating point from a validation partition.

    ``NO_FEASIBLE_THRESHOLD`` is a *measured* negative: the support was adequate
    and no candidate satisfied the mandatory constraint.
    ``INSUFFICIENT_VALIDATION_SUPPORT`` is the absence of a measurement: the
    partition could not resolve the constraint in the first place.  Neither is a
    success, and neither may be answered with a best-available threshold.
    """

    SELECTED = "selected"
    INSUFFICIENT_VALIDATION_SUPPORT = "insufficient_validation_support"
    NO_FEASIBLE_THRESHOLD = "no_feasible_threshold"


class AnomalyThresholdMethod(StrEnum):
    """Where the experimental anomaly probe's flag threshold comes from.

    Two members, and both name a *benign* source.  The probe reads no
    supervised target, so a threshold chosen against malicious outcomes would
    turn an unsupervised measurement into a weakly supervised one without
    saying so.  Neither member names the test split or the novel-anomaly
    holdout, because the holdout is what the probe is measured against and a
    threshold tuned on it would be measuring itself.
    """

    #: A quantile of the benign TRAIN score distribution.  No validation row is
    #: consulted at all.
    TRAIN_BENIGN_QUANTILE = "train_benign_quantile"
    #: The largest threshold holding the benign flag rate on validation-A at or
    #: under a configured target.  Validation-A only, benign rows only.
    VALIDATION_A_BENIGN_FPR = "validation_a_benign_fpr"


class ThresholdObjective(StrEnum):
    """The rule used to pick a decision threshold from validation data.

    Every objective is evaluated on a validation partition.  None of them can
    read the test split -- see :class:`ValidationPartition`.
    """

    #: Maximise the malicious detection rate subject to a false-positive
    #: ceiling.  The default: a SOC's binding constraint is analyst time.
    MAX_RECALL_AT_MAX_FPR = "max_recall_at_max_fpr"
    #: Maximise F1 with no explicit operating constraint.
    MAX_F1 = "max_f1"
    #: Minimise the false-positive rate subject to a detection-rate floor.
    MIN_FPR_AT_MIN_RECALL = "min_fpr_at_min_recall"


class ValidationPartition(StrEnum):
    """Which partition of the validation split a fitted quantity came from.

    Calibration reads :attr:`VALIDATION_A`; threshold selection reads
    :attr:`VALIDATION_B`.  Splitting them stops one set of rows from both
    fitting the calibrator and choosing the operating point.

    There is **no** ``TEST`` member and **no** ``NOVEL_ANOMALY_HOLDOUT``
    member.  Neither split can be named as a source of a fitted quantity
    because neither has a name to give.
    """

    VALIDATION_A = "validation_a"
    VALIDATION_B = "validation_b"


class FusionStrategy(StrEnum):
    """How a rule verdict and a model score are combined into one decision.

    None of the three performs arithmetic across the two quantities: the Phase
    4 risk score is an ordinal magnitude on 0-100 and the ML output is a
    calibrated probability or a decision score.  The gates combine *booleans*;
    the stacked strategy consumes both as separately named inputs to a fitted
    model.
    """

    #: Flag when the rule engine fired **or** the model cleared its threshold.
    OR_GATE = "or_gate"
    #: Flag when the rule engine fired **and** the model cleared its threshold.
    AND_GATE = "and_gate"
    #: A meta-model over the model score and typed rule features.
    STACKED = "stacked"


class ComparisonSystem(StrEnum):
    """Which detection system produced a decision in the final comparison.

    Three named systems, compared over one identical event universe.  The
    hybrid is present only when a validation-only fusion selection actually
    chose one: there is no fallback member for "whichever hybrid we defaulted
    to", because there is no default hybrid.
    """

    #: The frozen Phase 4 rule engine, unmodified.
    RULE_ONLY = "rule_only"
    #: The frozen Milestone 7 champion, applied through its Milestone 8
    #: predictions.
    ML_ONLY = "ml_only"
    #: The validation-selected fusion of the two.
    HYBRID = "hybrid"


class TestEvaluationStatus(StrEnum):
    """Whether a locked TEST evaluation could be carried out, and what it found.

    ``COMPLETED`` means the frozen lineage verified, the labels were opened
    once, and the metrics were computed under the contract declared before they
    were.  The other two are refusals *before* any label was read -- which is
    the only useful place to refuse.
    """

    COMPLETED = "completed"
    #: The frozen lineage was incomplete or inconsistent, so nothing was opened.
    LINEAGE_INCOMPLETE = "lineage_incomplete"
    #: The TEST population carried too little of a class for the metrics to be
    #: measurements rather than anecdotes.
    INSUFFICIENT_TEST_SUPPORT = "insufficient_test_support"


class GateStatus(StrEnum):
    """Outcome of one champion-selection gate.

    ``INCONCLUSIVE`` is not a pass.  A gate whose inputs were unavailable --
    an empty denominator, a class with no support -- reports
    ``INCONCLUSIVE`` and blocks promotion, mirroring the Phase 3 leakage
    auditor's rule that a skipped check is never a passed check.
    """

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class ChampionStatus(StrEnum):
    """Whether a champion could be selected, and why not when it could not."""

    #: A model cleared every gate and may be frozen as champion.
    ELIGIBLE = "eligible"
    #: The model was evaluated and failed at least one gate.
    NOT_ELIGIBLE = "not_eligible"
    #: Every candidate was evaluated and none cleared the gates.
    NO_ELIGIBLE_CHAMPION = "no_eligible_champion"
    #: Selection could not be decided: the validation split did not carry
    #: enough support for the gates to mean anything.  Distinct from
    #: ``NO_ELIGIBLE_CHAMPION``, which is a measured negative.
    INSUFFICIENT_VALIDATION_SUPPORT = "insufficient_validation_support"


class ModelEligibilityStatus(StrEnum):
    """A catalog entry's standing with respect to becoming champion.

    Catalog membership alone never makes a model champion-ready.  A family
    reaches :attr:`CHAMPION_ELIGIBLE` only once it has a stable serializer and
    a round-trip-exact inference adapter.
    """

    #: Serializer and inference adapter are declared and provable.
    CHAMPION_ELIGIBLE = "champion_eligible"
    #: The mandatory comparator every candidate is measured against.
    #:
    #: Fully implemented, fully publishable, and permanently unpromotable --
    #: which is the point rather than a limitation.  A candidate qualifies by
    #: beating this model on the validation gate, and a model cannot
    #: meaningfully beat itself; admitting it to the contest would also give
    #: selection an automatic fallback, and
    #: :attr:`~password_attack_detector.ml.enums.ChampionStatus.NO_ELIGIBLE_CHAMPION`
    #: has to stay reachable when every real candidate fails.
    REFERENCE_BASELINE = "reference_baseline"
    #: Evaluable, but not promotable: the serializer contract is unproven.
    SERIALIZER_UNPROVEN = "serializer_unproven"
    #: Evaluable as an experimental signal only.
    EXPERIMENTAL = "experimental"
    #: Unsupervised; scores the novel-anomaly probe and nothing else.
    ANOMALY_ONLY = "anomaly_only"


class ExperimentRecordType(StrEnum):
    """The kind of immutable record appended to the experiment ledger.

    The four members are declared here so the ledger's shape is fixed before
    any of it is written.  A record is append-only and immutable: a training
    run never grows a test metric later, because a test evaluation is a
    *separate* record written after the champion is frozen.
    """

    TRAINING_RUN = "training_run"
    VALIDATION_SELECTION = "validation_selection"
    CHAMPION_FREEZE = "champion_freeze"
    TEST_EVALUATION = "test_evaluation"


class HyperparameterKind(StrEnum):
    """Logical type of a declared model hyperparameter.

    Deliberately narrower than the detection layer's ``ParameterKind``: a
    hyperparameter is never a duration, so there is no ``WINDOW`` member.
    """

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"


class MLSplit(StrEnum):
    """Where a row may be used, as the ML layer names it.

    A deliberate **mirror** of ``features.splitting.SplitLabel`` rather than an
    import of it.  ``SplitLabel`` lives in the module that also carries
    ``SplitAssignment`` and ``split_dataset``, and the ML layer's import-graph
    rule admits exactly one module -- :mod:`password_attack_detector.ml.dataset`
    -- to read from there.  Every other ML module still has to reason about
    splits, so it reasons about this enum instead.

    Mirroring risks divergence, so the divergence is made loud rather than
    prevented by convention: ``ml.dataset``, the one module that sees both,
    asserts at import that the two enums carry identical values.  Renaming a
    Phase 3 split member therefore fails the build here rather than quietly
    producing an ML layer that files rows under a split that no longer exists.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    NOVEL_ANOMALY_HOLDOUT = "novel_anomaly_holdout"
    EXCLUDED = "excluded"


#: The only split a model may be fitted on.
#:
#: One member, and the narrowness is the point.  Validation partitions fit
#: calibrators and choose operating points; test and holdout rows are read once,
#: after everything is frozen; excluded rows are read never.
FIT_ELIGIBLE_SPLITS: Final[frozenset[MLSplit]] = frozenset({MLSplit.TRAIN})


class ServingScope(StrEnum):
    """What a live row scored by the deployed service *is*.

    Deliberately **not** a member of :class:`MLSplit`, and deliberately not the
    same type.  A split is an experimental assignment: it says which population
    a row belongs to, and every scientific artifact in this layer -- a
    threshold, a calibrator, an evaluation receipt, a prediction manifest -- is
    scoped by one.  A request arriving at the serving API belongs to no such
    population.  It was not partitioned, it carries no outcome, nothing will be
    measured on it, and no receipt will name it.

    Filing live traffic under ``MLSplit.TEST`` because the scoring path
    happened to want a split would state something false in the one field a
    reader trusts to be true, and it would put live rows in the same scope as
    the locked evaluation population.  So live inference gets its own scope
    with its own vocabulary, and the two types cannot be substituted for one
    another even by mistake.

    One member, and it stays that way: serving rows are all in one scope,
    always, so a caller cannot select a different scoring population.
    """

    LIVE = "live_serving"


def _assert_serving_is_not_a_split() -> None:
    """Fail at import if the serving scope ever collides with a split label."""
    shared = {item.value for item in ServingScope} & {item.value for item in MLSplit}
    if shared:
        raise ValueError(
            f"the serving scope shares value(s) {sorted(shared)} with a split "
            f"label; live traffic is not an experimental population and must "
            f"not be able to name itself as one"
        )


_assert_serving_is_not_a_split()


class FeatureDecisionPoint(StrEnum):
    """When a feature's value becomes available to a decision.

    Recorded per admitted feature in the reviewed allowlist, and checked against
    the catalog rather than trusted: a reviewer who writes ``post_event`` beside
    a baseline-derived feature is describing a system that does not exist, and
    the mismatch is rejected rather than absorbed.
    """

    #: Computable from the anchor event and the history preceding it, with no
    #: fitted state beyond the event stream itself.
    POST_EVENT = "post_event"
    #: Requires a behavioural baseline fitted on an approved reference
    #: interval.  Available only where that baseline exists.
    REQUIRES_FITTED_BASELINE = "requires_fitted_baseline"


class AuditCheckStatus(StrEnum):
    """Outcome of one named eligibility-audit check.

    ``SKIPPED`` is not ``PASS``.  A check whose input was not supplied reports
    what it is -- unevaluated -- and the overall audit fails, mirroring the
    Phase 3 leakage auditor's rule.  Reporting an unevaluated check as passed
    would make an audit look like evidence for something nobody measured.
    """

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class AuditStatus(StrEnum):
    """Aggregate outcome of an eligibility audit.

    Two members.  There is no ``WARNING``: a leakage finding is not a matter of
    degree, and a third status would invite a run to proceed on one.
    """

    PASS = "pass"
    FAIL = "fail"


class ValidationPartitionStatus(StrEnum):
    """Whether the validation split could be partitioned meaningfully.

    ``INSUFFICIENT_VALIDATION_SUPPORT`` is a typed outcome, not an exception
    and not a fallback.  The alternative -- halving the rows anyway -- would
    hand back two partitions that look usable and are not.
    """

    PARTITIONED = "partitioned"
    INSUFFICIENT_VALIDATION_SUPPORT = "insufficient_validation_support"


#: The category emitted when the multiclass head cannot assign a known class.
#:
#: A first-class outcome, not a failure marker.  Forcing an unrecognised row
#: into the nearest known category would report a confident answer the model
#: does not have, and would quietly convert novel behaviour into a class
#: somebody already wrote a rule for.
UNKNOWN_CATEGORY: Final = "unknown"


class ExplanationMethod(StrEnum):
    """How one attribution number was produced.

    Every member is deterministic and implemented in this project against the
    published artifact alone: no estimator is reconstructed, no private
    scikit-learn attribute is read, and no sampling is involved.  A method that
    could only be described as "approximately additive" is not here, because a
    per-feature number nobody can reconstruct is indistinguishable from one that
    was made up.

    The first three are *exact local* decompositions -- the contributions sum
    back to the model's own decision quantity, and a test asserts it.  The
    fourth is a *global* sensitivity measure and deliberately does not
    decompose anything: it says how much the model's output moves when a column
    is scrambled, which is a statement about the model, not about a row.
    """

    #: ``contribution_j = transformed_value_j * coefficient_j``, with the
    #: intercept recorded separately.  Sums to the linear model's logit.
    LINEAR_LOGIT_CONTRIBUTION = "linear_logit_contribution"
    #: The decision-path decomposition of a tree ensemble: each split on the
    #: root-to-leaf path is credited with the change it makes to the node's
    #: stored class distribution, averaged over trees.  Sums to the forest's
    #: own score minus the ensemble-mean root value.
    TREE_PATH_CONTRIBUTION = "tree_path_contribution"
    #: The single reviewed column a threshold baseline cuts on carries the whole
    #: decision; every other column contributes exactly zero because the model
    #: does not read it.
    SINGLE_FEATURE_STEP_CONTRIBUTION = "single_feature_step_contribution"
    #: Mean absolute change in the model's decision score when one transformed
    #: column is replaced by a deterministically permuted copy of itself.
    #: Label-free, model-agnostic, and seeded from the run configuration.
    PERMUTATION_SCORE_SENSITIVITY = "permutation_score_sensitivity"


#: The methods whose contributions reconstruct the model's decision quantity.
#:
#: Membership is what entitles a report to claim a reconstruction residual.  A
#: method outside this set may still be published, but it may not describe
#: itself as a decomposition of anything.
EXACT_EXPLANATION_METHODS: Final[frozenset[ExplanationMethod]] = frozenset(
    {
        ExplanationMethod.LINEAR_LOGIT_CONTRIBUTION,
        ExplanationMethod.TREE_PATH_CONTRIBUTION,
        ExplanationMethod.SINGLE_FEATURE_STEP_CONTRIBUTION,
    }
)


class ExplanationStatus(StrEnum):
    """Whether an exact attribution could be produced for a model family.

    Two members, and the absence of a third is deliberate.  There is no
    ``APPROXIMATE``: a family whose local decomposition this build cannot
    compute exactly reports :attr:`UNAVAILABLE` and a reason, because an
    approximate per-feature number reads exactly like an exact one once it is
    in a table.
    """

    EXACT = "exact"
    UNAVAILABLE = "unavailable"


#: Splits an explanation may be computed over.
#:
#: Test and the novel-anomaly holdout are absent, and the absence is the
#: enforcement.  Attribution describes what a frozen model does to a row; doing
#: it over the locked evaluation population would put a per-row artifact derived
#: from test beside the one evaluation that was allowed to read it.
EXPLANATION_ELIGIBLE_SPLITS: Final[frozenset[MLSplit]] = frozenset(
    {MLSplit.TRAIN, MLSplit.VALIDATION}
)


class ReferenceFeatureKind(StrEnum):
    """How a reference profile partitions one raw feature's values.

    Mirrors the three encodings the frozen preprocessor already distinguishes,
    so the profile cannot invent a fourth interpretation of a column the model
    was fitted under.
    """

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class DriftMetric(StrEnum):
    """The measure a drift result reports.

    One member, and a test asserts it stays that way.  The population stability
    index over a partition frozen from the reference is the only measure the
    reviewed configuration declares thresholds for, and a second thresholded
    family would need warn and alert values nobody has chosen.  Null rates and
    unknown-category rates are still reported -- as fields on the result, and as
    dedicated bins inside the very partition this index is taken over, so a
    shift in either moves this number rather than hiding beside it.
    """

    POPULATION_STABILITY_INDEX = "population_stability_index"


class DriftStatus(StrEnum):
    """What one drift result means.

    ``INCONCLUSIVE`` and ``UNAVAILABLE`` are both refusals, and neither is
    ``NO_DRIFT``.  A comparison over too few rows has not established stability,
    and a quantity that does not exist has not been measured as zero -- reporting
    either as "no drift" would make a monitor that never fires look like a system
    that never moved.
    """

    #: Measured, with adequate support, below the configured warning threshold.
    NO_DRIFT = "no_drift"
    #: Measured at or above the warning threshold and below the alert threshold.
    DRIFT_WARNING = "drift_warning"
    #: Measured at or above the alert threshold.
    DRIFT_DETECTED = "drift_detected"
    #: Computable, but over too few reference or incoming rows to be evidence.
    INCONCLUSIVE = "inconclusive"
    #: Not computable at all: the quantity does not exist on one side.
    UNAVAILABLE = "unavailable"


class PredictionDriftQuantity(StrEnum):
    """Which published prediction quantity a drift result describes.

    Kept apart from feature drift throughout.  A feature distribution moving and
    a model's output distribution moving are different findings with different
    remedies, and a report that summed them would say neither.
    """

    FLAGGED_MALICIOUS_RATE = "flagged_malicious_rate"
    DECISION_SCORE = "decision_score"
    CALIBRATED_PROBABILITY = "calibrated_probability"
    CATEGORY_PREDICTED_CLASS = "category_predicted_class"
    CATEGORY_UNKNOWN_RATE = "category_unknown_rate"
    #: The experimental anomaly track, reported under its own quantity so it can
    #: never be read as part of the supervised result.
    ANOMALY_SCORE = "anomaly_score"


#: The splits a drift reference profile may be captured from.
#:
#: One member.  The reviewed configuration declares ``reference_source: train``
#: and this is the same statement made where a split is a type rather than a
#: string.  Test and the novel-anomaly holdout are absent for the reason they
#: are absent everywhere else; validation is absent because it fitted the
#: calibrator and chose the operating point, so a monitor baselined on it would
#: be baselined on rows the frozen champion was tuned against.
DRIFT_REFERENCE_ELIGIBLE_SPLITS: Final[frozenset[MLSplit]] = frozenset({MLSplit.TRAIN})


class AcceptanceStatus(StrEnum):
    """The standing of one Phase 5 requirement in the final acceptance report.

    ``INCONCLUSIVE`` exists so that a requirement whose evidence the repository
    cannot produce is not quietly recorded as met.  ``NOT_APPLICABLE`` exists so
    that a requirement which genuinely does not apply is not recorded as a
    failure.  Neither is a pass, and nothing in the report derives a pass from
    the absence of a failure.
    """

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"
