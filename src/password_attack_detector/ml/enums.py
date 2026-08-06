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
    "PROBABILITY_SCORE_KINDS",
    "SUPERVISED_TASKS",
    "UNKNOWN_CATEGORY",
    "CalibrationMethod",
    "ChampionStatus",
    "ExperimentRecordType",
    "FusionStrategy",
    "GateStatus",
    "HyperparameterKind",
    "MLTask",
    "ModelEligibilityStatus",
    "ModelFamily",
    "ScoreKind",
    "ThresholdObjective",
    "ValidationPartition",
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


#: The category emitted when the multiclass head cannot assign a known class.
#:
#: A first-class outcome, not a failure marker.  Forcing an unrecognised row
#: into the nearest known category would report a confident answer the model
#: does not have, and would quietly convert novel behaviour into a class
#: somebody already wrote a rule for.
UNKNOWN_CATEGORY: Final = "unknown"
