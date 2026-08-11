"""Training orchestration: composition, not algorithms.

Everything this module does has already been implemented somewhere else. The
dataset is assembled by :mod:`~password_attack_detector.ml.dataset`, the row
order by :mod:`~password_attack_detector.ml.ordering`, the design matrix by
:mod:`~password_attack_detector.ml.preprocessing`, the weights by
:mod:`~password_attack_detector.ml.imbalance`, the fit by an adapter in
``ml/models``, the calibrator by :mod:`~password_attack_detector.ml.calibration`,
and the operating point by :mod:`~password_attack_detector.ml.thresholds`. What
happens here is that they are called **in one order, on one set of frozen
inputs**, and the result is recorded.

That is deliberately all it is. A trainer that reimplemented a weight formula
"for convenience" would be a second formula, and the two would agree until the
day they did not.

**Three tracks, kept apart.**

===================  ==============================  ==========================
track                fitted on                       operating point
===================  ==============================  ==========================
binary supervised    TRAIN, supervised-eligible      threshold on validation-B
category supervised  TRAIN, known-malicious only     abstention on validation-B
anomaly experimental TRAIN, benign only, no target   benign quantile or
                                                     validation-A benign rate
===================  ==============================  ==========================

A category head never inherits the binary threshold; an anomaly run never
carries a calibrator; a binary run never invents an abstention artifact. Each
run publishes only what its own task means.

**What this module may see.** Typed objects, already assembled. It opens no
Parquet file and imports no label reader: the label-reader allowlist is exactly
``{detection.evaluation, ml.dataset}`` and Milestone 6 does not widen it. A
caller hands over an :class:`~password_attack_detector.ml.dataset.MLDataset`
that somebody with permission to read labels already built.

**Nothing here ranks anything.** No candidate is preferred, no champion is
chosen, and no run record carries a ``champion`` field. Candidates are
enumerated from the reviewed configuration, trained under identical frozen
contracts, and reported with their statuses. Selection is Milestone 7's, and it
will read these artifacts rather than re-derive them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.features.catalog import FeatureCatalog
from password_attack_detector.ml.calibration import (
    BinaryScoreSample,
    CalibrationOutcome,
    CalibrationReport,
    CalibrationState,
    ScoreSampleSource,
    apply_calibration,
    diagnose_calibration_fit,
    evaluate_calibration_quality,
    fit_calibration,
)
from password_attack_detector.ml.catalog import MODEL_CATALOG, ModelCatalog, ModelSpec
from password_attack_detector.ml.config import MLConfig
from password_attack_detector.ml.dataset import (
    KNOWN_CATEGORY_CLASSES,
    AnchorMetadata,
    MLDataset,
    SplitDataset,
)
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    CalibrationStatus,
    MLSplit,
    MLTask,
    ModelFamily,
    ScoreKind,
    SelectionStatus,
    TrainingRunStatus,
    ValidationPartition,
)
from password_attack_detector.ml.features import EligibleFeatureList
from password_attack_detector.ml.imbalance import (
    BINARY_CLASS_ORDER,
    ClassWeightState,
    compute_class_weights,
)
from password_attack_detector.ml.models import (
    PUBLISHABLE_FAMILIES,
    FittedModel,
    TrainingBatch,
    adapter_class_for,
)
from password_attack_detector.ml.partition import ValidationPartitionResult
from password_attack_detector.ml.preprocessing import (
    FittedPreprocessor,
    fit_preprocessor,
)
from password_attack_detector.ml.thresholds import (
    AnomalyScoreSample,
    AnomalyThresholdSelection,
    CategoryAbstentionSelection,
    CategoryScoreSample,
    ThresholdSelection,
    select_anomaly_threshold,
    select_binary_threshold,
    select_category_abstention,
)

__all__ = [
    "CandidateSpec",
    "ReadableLineage",
    "TrainingContext",
    "TrainingRunOutcome",
    "enumerate_candidates",
    "train_all",
    "train_candidate",
]

#: Catalog hyperparameters that are **not** adapter constructor arguments.
#:
#: Named rather than filtered by shape, because dropping a declared
#: hyperparameter silently is exactly the behaviour this project refuses. Every
#: remaining hyperparameter must be a constructor argument, and one that is not
#: fails loudly -- so a catalog entry added without a corresponding adapter
#: parameter is caught here rather than ignored.
#:
#: ``class_weight`` is honoured by the imbalance policy and reaches the fit as
#: per-row sample weights; passing it to the estimator as well would apply it
#: twice. ``selection_metric`` describes how a reviewer *chose* M-001's column,
#: which is a decision recorded in configuration rather than an argument to a
#: fit -- see :attr:`~password_attack_detector.ml.config.MLConfig.single_feature_baseline_column`.
NON_CONSTRUCTOR_HYPERPARAMETERS: Final[frozenset[str]] = frozenset(
    {"class_weight", "selection_metric"}
)

#: The order tasks are enumerated in. Fixed so two runs over one configuration
#: produce candidates in the same sequence, whatever order a set happened to
#: iterate in.
TASK_ORDER: Final[tuple[MLTask, ...]] = (
    MLTask.BINARY_MALICIOUS,
    MLTask.ATTACK_CATEGORY,
    MLTask.ANOMALY,
)


def _digest(payload: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON rendering of *payload*."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ReadableLineage:
    """Three role-scoped provenance digests over the rows one track may read.

    Separate rather than combined so a moved fingerprint names its own cause:
    a changed feature value moves :attr:`training_data`, a changed label moves
    :attr:`labels`, and a row entering or leaving a readable role moves
    :attr:`split`.
    """

    training_data: str
    labels: str
    split: str


@dataclass(frozen=True, slots=True)
class _Frame:
    """A minimal :class:`FeatureFrame` over a chosen subset of rows.

    Carries no label, no campaign, and no supervised-eligibility flag: the
    preprocessing protocol has nowhere to put one, which is what keeps that
    module unable to read ground truth even by accident. Selecting *which* rows
    is this module's job and happens before a frame is built.
    """

    split: MLSplit
    feature_names: tuple[str, ...]
    anchors: tuple[AnchorMetadata, ...]
    feature_matrix: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class _Population:
    """One selected set of rows, with the labels that go with it.

    Row selection is where a training population is decided, so it is one
    small, inspectable object rather than a filter expression repeated at four
    call sites.
    """

    frame: _Frame
    malicious: tuple[bool, ...]
    known_category: tuple[str | None, ...]

    @property
    def row_count(self) -> int:
        """Return the number of rows."""
        return len(self.frame.anchors)

    @property
    def positive_count(self) -> int:
        """Return the number of malicious rows."""
        return sum(1 for value in self.malicious if value)

    @property
    def benign_count(self) -> int:
        """Return the number of benign rows."""
        return self.row_count - self.positive_count

    def binary_targets(self) -> tuple[str, ...]:
        """Return the binary class name per row, in canonical row order."""
        return tuple(
            BINARY_CLASS_ORDER[1] if value else BINARY_CLASS_ORDER[0]
            for value in self.malicious
        )

    def category_targets(self) -> tuple[str, ...]:
        """Return the known-category name per row, or raise if any is absent."""
        if any(value is None for value in self.known_category):
            raise ModelTrainingError(
                "a category population carries a row with no known category; "
                "selection should have excluded it"
            )
        return tuple(str(value) for value in self.known_category)


def _select(
    dataset: SplitDataset,
    *,
    keep: Sequence[bool],
    split: MLSplit,
) -> _Population:
    """Return the rows of *dataset* where *keep* is true, order preserved.

    Order is preserved rather than restored: the dataset arrives canonically
    ordered and a filter cannot disturb that, so every downstream
    ``assert_canonical`` still holds without anything being re-sorted.
    """
    indices = [index for index, flag in enumerate(keep) if flag]
    return _Population(
        frame=_Frame(
            split=split,
            feature_names=dataset.feature_names,
            anchors=tuple(dataset.anchors[index] for index in indices),
            feature_matrix=tuple(dataset.feature_matrix[index] for index in indices),
        ),
        malicious=tuple(dataset.malicious[index] for index in indices),
        known_category=tuple(dataset.known_category[index] for index in indices),
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One predeclared thing to train: a task, a family, and its settings.

    Enumerated from the reviewed configuration and the executable catalog, and
    from nothing else. There is no search here -- no grid, no sampling, no
    optimiser, and no criterion evaluated on held-out data to decide what to try
    next. A candidate exists because somebody wrote it down.
    """

    task: MLTask
    family: ModelFamily
    catalog_model_id: str
    hyperparameters: Mapping[str, bool | int | float | str]
    candidate_fingerprint: str

    @property
    def label(self) -> str:
        """Return a short, identity-free label for a terminal or a log."""
        return f"{self.catalog_model_id}/{self.task}"


def _candidate_fingerprint(
    *, task: MLTask, spec: ModelSpec, hyperparameters: Mapping[str, Any]
) -> str:
    """Return the digest that distinguishes one candidate from another.

    Covers the task, the catalog entry, and the *effective* hyperparameters --
    what would actually be fitted, rather than only the overrides somebody
    happened to write down. Two configurations that differ in a default they
    both accepted are the same candidate, and a digest that said otherwise
    would make an identical run look new.
    """
    return _digest(
        {
            "task": str(task),
            "catalog_model_id": spec.model_id,
            "model_version": spec.model_version,
            "family": str(spec.family),
            "hyperparameters": {
                name: _scalar(hyperparameters[name]) for name in sorted(hyperparameters)
            },
        }
    )


def _canonical(payload: Any) -> str:
    """Return the one JSON rendering the lineage digests are taken over."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _cell(value: object) -> Any:
    """Render one feature cell into a JSON-stable scalar.

    Floats are formatted at nine decimals, matching every other fingerprint in
    this project, so a digest does not move with the last bit of a repr.
    """
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, bool | int | str) or value is None:
        return value
    return str(value)


def _scalar(value: object) -> Any:
    """Render one hyperparameter into a JSON-stable scalar."""
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, bool | int | str) or value is None:
        return value
    return str(value)


def enumerate_candidates(
    config: MLConfig, *, catalog: ModelCatalog = MODEL_CATALOG
) -> tuple[CandidateSpec, ...]:
    """Return every candidate this configuration declares, deterministically.

    Ordered by task and then by catalog identifier, so the sequence depends on
    the configuration and the catalog and on nothing incidental. No wall clock,
    no set iteration order, and no filesystem listing takes part.

    A family that the catalog does not admit for a task simply produces no
    candidate for it -- that is the catalog doing its job. A family that *is*
    admitted but cannot be trained still produces a candidate, so its refusal
    is recorded as an outcome rather than as an absence.
    """
    candidates: list[CandidateSpec] = []
    for task in TASK_ORDER:
        if task is MLTask.ATTACK_CATEGORY and not config.category.enabled:
            continue
        if task is MLTask.ANOMALY and not config.anomaly.enabled:
            continue
        for family in sorted(config.enabled_model_families, key=str):
            spec = catalog.for_family(family)
            if task not in spec.supported_tasks:
                continue
            if task is MLTask.ATTACK_CATEGORY and not spec.multiclass_capable:
                continue
            hyperparameters = config.hyperparameters_for(family, catalog)
            candidates.append(
                CandidateSpec(
                    task=task,
                    family=family,
                    catalog_model_id=spec.model_id,
                    hyperparameters=dict(hyperparameters),
                    candidate_fingerprint=_candidate_fingerprint(
                        task=task, spec=spec, hyperparameters=hyperparameters
                    ),
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (TASK_ORDER.index(item.task), item.catalog_model_id),
        )
    )


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingContext:
    """Everything frozen before any candidate is trained.

    Assembled once and shared by every candidate, which is the point: two
    candidates that saw different preprocessing, different weights, or a
    different validation partition were not compared, they were merely both
    run.

    The preprocessor is fitted on the **supervised training population** and
    fitted on the rows that task is permitted to learn from. A fitted
    preprocessor is not a formatting step: it learns imputation constants,
    category vocabularies, rare-value buckets, and scaling statistics, so its
    fitting population is part of the learned pipeline. An anomaly probe whose
    medians came from malicious rows has read the labels it was supposed to be
    blind to, and a category head whose vocabulary came from benign traffic has
    been shaped by rows it will never be asked about.

    The reviewed *raw* feature allowlist is shared; the fitted state is not.
    Two families fitted for the same task reuse the same state, because their
    preprocessing configuration is identical and refitting it would produce a
    second identity for one pipeline.
    """

    dataset: MLDataset
    config: MLConfig
    eligible: EligibleFeatureList
    feature_catalog: FeatureCatalog
    partition: ValidationPartitionResult
    allowlist_fingerprint: str
    #: The binary track's state, fitted eagerly: the binary task is the primary
    #: one and a run that cannot build its population has nothing to do.
    preprocessor: FittedPreprocessor
    catalog: ModelCatalog = MODEL_CATALOG
    #: Task-specific fitted states, built on first use and reused after.
    #: Fitted lazily because the category and anomaly populations can be empty
    #: or too thin, and that is a per-track outcome rather than a reason to
    #: stop the whole run.
    _preprocessors: dict[MLTask, FittedPreprocessor] = field(default_factory=dict)

    @classmethod
    def prepare(
        cls,
        dataset: MLDataset,
        *,
        config: MLConfig,
        eligible: EligibleFeatureList,
        feature_catalog: FeatureCatalog,
        partition: ValidationPartitionResult,
        allowlist_fingerprint: str,
        catalog: ModelCatalog = MODEL_CATALOG,
    ) -> TrainingContext:
        """Fit the binary track's preprocessing state and return the context.

        The other two tracks fit their own when they are trained -- see
        :meth:`preprocessor_for`.

        Raises:
            ModelTrainingError: if the validation partition is unusable, or the
                supervised training population is empty. Both are conditions no
                candidate could recover from, so they stop the run rather than
                producing a directory full of identical failures.
        """
        if not partition.usable:
            raise ModelTrainingError(
                f"the validation split could not be partitioned "
                f"[{partition.status!s}]: {list(partition.failing_requirements)}. "
                f"Calibration reads validation-A and operating points read "
                f"validation-B; without both halves there is nothing to train "
                f"against"
            )
        train = dataset.for_split(MLSplit.TRAIN)
        population = _select(
            train,
            keep=train.supervised_training_eligible,
            split=MLSplit.TRAIN,
        )
        if population.row_count == 0:
            raise ModelTrainingError(
                "the training split carries no supervised-eligible rows; there "
                "is nothing to fit preprocessing on"
            )
        preprocessor = fit_preprocessor(
            population.frame,
            catalog=feature_catalog,
            eligible=eligible,
            config=config.preprocessing,
        )
        context = cls(
            dataset=dataset,
            config=config,
            eligible=eligible,
            feature_catalog=feature_catalog,
            partition=partition,
            allowlist_fingerprint=allowlist_fingerprint,
            preprocessor=preprocessor,
            catalog=catalog,
        )
        context._preprocessors[MLTask.BINARY_MALICIOUS] = preprocessor
        return context

    # -- per-track preprocessing -------------------------------------------

    def training_population(self, task: MLTask) -> _Population:
        """Return the TRAIN rows *task* is permitted to learn from."""
        match task:
            case MLTask.BINARY_MALICIOUS:
                return self.supervised_train()
            case MLTask.ATTACK_CATEGORY:
                return self.category_train()
            case MLTask.ANOMALY:
                return self.benign_train()

    def preprocessor_for(self, task: MLTask) -> FittedPreprocessor:
        """Return the preprocessing state fitted on *task*'s own population.

        Cached per task, so two families fitted for the same task share one
        fitted state and one fingerprint rather than producing two identical
        ones.

        Raises:
            ModelTrainingError: if the task's training population is empty, or
                if preprocessing cannot be fitted on it. The caller turns that
                into a per-candidate status rather than letting it stop the run.
        """
        cached = self._preprocessors.get(task)
        if cached is not None:
            return cached
        population = self.training_population(task)
        if population.row_count == 0:
            raise ModelTrainingError(
                f"task {task!s} has no training rows of its own; preprocessing "
                f"fitted on another task's population would learn statistics "
                f"from rows this task may not read"
            )
        fitted = fit_preprocessor(
            population.frame,
            catalog=self.feature_catalog,
            eligible=self.eligible,
            config=self.config.preprocessing,
        )
        self._preprocessors[task] = fitted
        return fitted

    # -- populations --------------------------------------------------------

    def supervised_train(self) -> _Population:
        """Return TRAIN rows eligible for supervised fitting."""
        train = self.dataset.for_split(MLSplit.TRAIN)
        return _select(
            train, keep=train.supervised_training_eligible, split=MLSplit.TRAIN
        )

    def category_train(self) -> _Population:
        """Return TRAIN rows carrying a known malicious category.

        Malicious rows with a category the Phase 2 scenario contract declares,
        and nothing else. Benign rows are excluded because the head answers
        "which attack", not "whether"; novel-anomaly rows are excluded because
        they carry no known category to be right about -- and they are in a
        different split entirely.
        """
        train = self.dataset.for_split(MLSplit.TRAIN)
        keep = [
            eligible and malicious and category is not None
            for eligible, malicious, category in zip(
                train.supervised_training_eligible,
                train.malicious,
                train.known_category,
                strict=True,
            )
        ]
        return _select(train, keep=keep, split=MLSplit.TRAIN)

    def benign_train(self) -> _Population:
        """Return benign TRAIN rows, for the unsupervised probe.

        Which rows are benign is decided here, where labels are legitimately
        readable. The estimator itself receives a design matrix and no target.
        """
        train = self.dataset.for_split(MLSplit.TRAIN)
        keep = [not malicious for malicious in train.malicious]
        return _select(train, keep=keep, split=MLSplit.TRAIN)

    def validation(self, half: ValidationPartition) -> _Population:
        """Return the validation rows assigned to *half*."""
        validation = self.dataset.for_split(MLSplit.VALIDATION)
        assignment = self.partition.assignment
        keep = [
            assignment.get(anchor.anchor_event_id) is half
            for anchor in validation.anchors
        ]
        return _select(validation, keep=keep, split=MLSplit.VALIDATION)

    def validation_category(self, half: ValidationPartition) -> _Population:
        """Return known-malicious validation rows assigned to *half*."""
        validation = self.dataset.for_split(MLSplit.VALIDATION)
        assignment = self.partition.assignment
        keep = [
            assignment.get(anchor.anchor_event_id) is half
            and malicious
            and category is not None
            for anchor, malicious, category in zip(
                validation.anchors,
                validation.malicious,
                validation.known_category,
                strict=True,
            )
        ]
        return _select(validation, keep=keep, split=MLSplit.VALIDATION)

    def validation_benign(self, half: ValidationPartition) -> _Population:
        """Return benign validation rows assigned to *half*."""
        validation = self.dataset.for_split(MLSplit.VALIDATION)
        assignment = self.partition.assignment
        keep = [
            assignment.get(anchor.anchor_event_id) is half and not malicious
            for anchor, malicious in zip(
                validation.anchors, validation.malicious, strict=True
            )
        ]
        return _select(validation, keep=keep, split=MLSplit.VALIDATION)

    # -- derived fingerprints ----------------------------------------------

    def readable_roles(
        self, task: MLTask, *, anomaly_reads_validation_a: bool = False
    ) -> tuple[tuple[str, _Population], ...]:
        """Return the named row populations *task* is permitted to read.

        The scope is per task, not per run, because the three tracks read
        genuinely different rows. A digest over "everything a run touched"
        would move the category head's identity when a benign training row
        changed, which is a row that head never sees.

        Roles are named and ordered, so the digests below are reproducible and
        a reader can tell which population a change came from.
        """
        match task:
            case MLTask.BINARY_MALICIOUS:
                return (
                    ("train", self.supervised_train()),
                    ("validation_a", self.validation(ValidationPartition.VALIDATION_A)),
                    ("validation_b", self.validation(ValidationPartition.VALIDATION_B)),
                )
            case MLTask.ATTACK_CATEGORY:
                # No validation-A: the category head fits no calibrator, so it
                # never reads the calibration half.
                return (
                    ("train", self.category_train()),
                    (
                        "validation_b",
                        self.validation_category(ValidationPartition.VALIDATION_B),
                    ),
                )
            case MLTask.ANOMALY:
                roles: list[tuple[str, _Population]] = [("train", self.benign_train())]
                if anomaly_reads_validation_a:
                    roles.append(
                        (
                            "validation_a",
                            self.validation_benign(ValidationPartition.VALIDATION_A),
                        )
                    )
                return tuple(roles)

    def readable_lineage(
        self, task: MLTask, *, anomaly_reads_validation_a: bool = False
    ) -> ReadableLineage:
        """Return the three role-scoped provenance digests for *task*.

        **Not** the dataset's own ``training_data_fingerprint``,
        ``label_fingerprint``, or ``split_fingerprint``, and the difference is
        the whole point: those cover every row the dataset holds, test split and
        novel-anomaly holdout included. A run identity built from them would
        move whenever somebody added a test row -- the split firewall leaking
        through the identifier rather than through the fit.

        Three digests rather than one, because collapsing them would tell a
        reader *that* the lineage changed and never *what*: a feature value, a
        label, or a row's split membership are three different findings with
        three different investigations behind them.

        Row identity is used **inside** each digest and published by none of
        them. Pairing a label to the row that carries it is what makes the label
        digest move when two rows swap labels; publishing the pairing would put
        anchor identifiers in a record, so only the digest leaves.
        """
        roles = self.readable_roles(
            task, anomaly_reads_validation_a=anomaly_reads_validation_a
        )
        data: list[str] = []
        labels: list[str] = []
        membership: list[str] = []
        for role, population in roles:
            for index, anchor in enumerate(population.frame.anchors):
                identity = anchor.anchor_event_id
                data.append(
                    _canonical(
                        [
                            role,
                            identity,
                            [
                                _cell(value)
                                for value in population.frame.feature_matrix[index]
                            ],
                        ]
                    )
                )
                labels.append(
                    _canonical(
                        [
                            role,
                            identity,
                            anchor.supervised_training_eligible,
                            population.malicious[index],
                            population.known_category[index],
                        ]
                    )
                )
                membership.append(
                    _canonical([role, identity, anchor.supervised_training_eligible])
                )

        scope = {
            "task": str(task),
            "roles": [role for role, _ in roles],
        }
        return ReadableLineage(
            training_data=_digest(
                {
                    **scope,
                    "feature_names": list(self.dataset.feature_names),
                    "eligible_feature_list_fingerprint": (
                        self.dataset.eligible_feature_list_fingerprint
                    ),
                    "rows": sorted(data),
                }
            ),
            labels=_digest(
                {
                    **scope,
                    "known_category_classes": list(self.dataset.known_category_classes),
                    "rows": sorted(labels),
                }
            ),
            split=_digest(
                {
                    **scope,
                    # The parent partition digest covers the campaign grouping,
                    # the boundary, and the support policy that placed every
                    # validation row in the half it is in.
                    "validation_partition_fingerprint": self.partition.fingerprint,
                    "rows": sorted(membership),
                }
            ),
        )

    def dependency_contract_fingerprint(self) -> str:
        """Return a digest over the declared dependency ranges."""
        return _digest(
            [
                {
                    "distribution": requirement.distribution,
                    "minimum_version": requirement.minimum_version,
                    "below_version": requirement.below_version,
                }
                for requirement in sorted(
                    self.config.dependency_requirements,
                    key=lambda item: item.distribution,
                )
            ]
        )


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRunOutcome:
    """What one candidate produced, whether or not it produced everything.

    A candidate that stopped early still returns one of these. The status names
    where it stopped and ``failing_requirements`` names what it was missing, so
    a family absent from a later comparison is distinguishable from a family
    that was never configured.

    There is no ``champion`` field, and there will not be one: Milestone 6 does
    not rank anything.
    """

    candidate: CandidateSpec
    status: TrainingRunStatus
    failing_requirements: tuple[str, ...] = ()
    fitted: FittedModel | None = None
    #: The state this model's matrix was built by -- its own task's, never
    #: another's. Carried here so publication cannot pair a model with a
    #: preprocessor it was not fitted against.
    preprocessor: FittedPreprocessor | None = None
    class_weights: ClassWeightState | None = None
    calibration: CalibrationOutcome | None = None
    calibration_state: CalibrationState | None = None
    calibration_diagnostic: CalibrationReport | None = None
    calibration_quality: CalibrationReport | None = None
    binary_threshold: ThresholdSelection | None = None
    category_abstention: CategoryAbstentionSelection | None = None
    anomaly_threshold: AnomalyThresholdSelection | None = None
    train_row_count: int = 0

    @property
    def complete(self) -> bool:
        """Return whether every artifact this task requires is present."""
        return self.status is TrainingRunStatus.COMPLETED

    @property
    def publishable(self) -> bool:
        """Return whether there is a fitted model worth writing to disk."""
        return self.fitted is not None


def _unavailable(
    candidate: CandidateSpec, status: TrainingRunStatus, *requirements: str
) -> TrainingRunOutcome:
    """Return an outcome that stopped before producing a model."""
    return TrainingRunOutcome(
        candidate=candidate, status=status, failing_requirements=tuple(requirements)
    )


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


def _construct_adapter(candidate: CandidateSpec, *, config: MLConfig) -> Any:
    """Return the adapter for *candidate*, configured from the catalog.

    Closed-registry dispatch: the family selects between implementations this
    build already contains, and a name that is not a key selects nothing.

    Every declared hyperparameter must be either a constructor argument or a
    named exception in :data:`NON_CONSTRUCTOR_HYPERPARAMETERS`. Anything else
    raises, because a hyperparameter a reviewer declared and the fit quietly
    ignored is worse than one that was never declared.

    Raises:
        ModelTrainingError: on a hyperparameter the adapter cannot accept, or a
            threshold baseline with no reviewed column configured.
    """
    adapter_class = adapter_class_for(candidate.family)
    accepted = set(inspect.signature(adapter_class).parameters)
    arguments: dict[str, Any] = {}
    for name, value in candidate.hyperparameters.items():
        if name in NON_CONSTRUCTOR_HYPERPARAMETERS:
            continue
        if name not in accepted:
            raise ModelTrainingError(
                f"the catalog declares hyperparameter {name!r} for "
                f"{candidate.catalog_model_id}, and its adapter does not accept "
                f"it; a declared setting that no fit reads is a contract that "
                f"is not being honoured"
            )
        arguments[name] = value

    if candidate.family is ModelFamily.SINGLE_FEATURE_THRESHOLD:
        column = config.single_feature_baseline_column
        if not column:
            raise ModelTrainingError(
                "the threshold baseline needs a reviewed transformed column, "
                "and none is configured; scanning for the best one would be a "
                "model-selection procedure reported as a baseline"
            )
        arguments["feature"] = column

    return adapter_class(**arguments)


# ---------------------------------------------------------------------------
# Scoring validation rows
# ---------------------------------------------------------------------------


def _score(
    preprocessor: FittedPreprocessor,
    fitted: FittedModel,
    adapter: Any,
    population: _Population,
) -> tuple[tuple[float, ...], ...]:
    """Return per-class scores for *population* under the frozen model.

    Transformed by the state the model was fitted against, never by another
    track's: the same rows encoded by a different vocabulary are a different
    matrix, and the adapter would refuse it -- but only after the mistake had
    already been made.
    """
    matrix = preprocessor.transform(population.frame)
    scored: tuple[tuple[float, ...], ...] = adapter.score(
        fitted, matrix.rows, matrix.output_feature_names
    )
    return scored


def _binary_sample(
    context: TrainingContext,
    preprocessor: FittedPreprocessor,
    fitted: FittedModel,
    adapter: Any,
    population: _Population,
    half: ValidationPartition,
) -> BinaryScoreSample:
    """Return the typed decision-score sample for one validation half.

    The malicious-class column is taken by *name* from the fitted class order,
    never by position: a model whose classes were ordered differently would
    otherwise have its benign score read as its malicious one, and every number
    downstream would be exactly inverted.
    """
    index = fitted.class_order.index(BINARY_CLASS_ORDER[1])
    scores = tuple(
        float(row[index]) for row in _score(preprocessor, fitted, adapter, population)
    )
    return BinaryScoreSample(
        source=ScoreSampleSource(
            split=MLSplit.VALIDATION,
            partition=half,
            source_fingerprint=context.partition.fingerprint,
        ),
        anchors=population.frame.anchors,
        scores=scores,
        malicious=population.malicious,
        score_kind=ScoreKind.DECISION_SCORE,
        model_id=_model_id(fitted),
        model_content_fingerprint=fitted.content_fingerprint(),
        preprocessor_fingerprint=fitted.preprocessor_fingerprint,
    )


def _model_id(fitted: FittedModel) -> str:
    """Return the derived model identifier for *fitted*."""
    from password_attack_detector.ml.serialization import model_id_for

    return model_id_for(
        fitted.content_fingerprint(), task=fitted.task, family=fitted.family
    )


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


def train_candidate(
    candidate: CandidateSpec, *, context: TrainingContext
) -> TrainingRunOutcome:
    """Train one candidate and return what it produced.

    Dispatches on the task rather than branching inside one long function,
    because the three tracks genuinely differ: what they are fitted on, what
    they are calibrated against, and what operating point they need are three
    different answers, and a single code path with three sets of conditionals
    is where a category head quietly acquires a binary threshold.
    """
    spec = context.catalog.for_family(candidate.family)
    if candidate.family not in PUBLISHABLE_FAMILIES:
        return _unavailable(
            candidate, TrainingRunStatus.SERIALIZER_UNAVAILABLE, "publishable_family"
        )
    match candidate.task:
        case MLTask.BINARY_MALICIOUS:
            return _train_binary(candidate, context=context, spec=spec)
        case MLTask.ATTACK_CATEGORY:
            return _train_category(candidate, context=context)
        case MLTask.ANOMALY:
            return _train_anomaly(candidate, context=context)


def _fit(
    candidate: CandidateSpec,
    *,
    context: TrainingContext,
    preprocessor: FittedPreprocessor,
    population: _Population,
    targets: tuple[str, ...],
    class_order: tuple[str, ...],
    weights: ClassWeightState | None,
) -> tuple[Any, FittedModel]:
    """Return the adapter and the model it fitted on *population*."""
    matrix = preprocessor.transform(population.frame)
    batch = TrainingBatch(
        split=MLSplit.TRAIN,
        anchors=population.frame.anchors,
        transformed_feature_names=matrix.output_feature_names,
        matrix=matrix.rows,
        targets=targets,
        class_order=class_order,
        preprocessor=preprocessor,
        class_weights=weights,
        seed=context.config.seed,
    )
    adapter = _construct_adapter(candidate, config=context.config)
    return adapter, adapter.fit(batch, task=candidate.task)


def _calibration_applies(spec: ModelSpec, config: MLConfig) -> bool:
    """Return whether a calibrator is part of *spec*'s contract for this run.

    Two families of answer, and only one of them is about configuration.

    A configuration naming :attr:`CalibrationMethod.NONE` asked for no
    calibrator, and none is fitted. Separately, a **reference baseline** is
    never calibrated whatever the configuration says: it emits the training
    class prior for every row, so there is no score variation for a calibrator
    to map, and the Milestone 5 methods both require distinct scores.

    Declaring that rather than letting the fit fail matters, because the
    reference baseline is the mandatory comparator. A comparator that could
    never reach a completed run would leave every candidate measured against an
    absence -- and the alternatives, faking a calibrator for a constant or
    relaxing the support rules until one was accepted, would each make the
    comparison less trustworthy rather than more.
    """
    if spec.reference_baseline:
        return False
    return config.calibration.method is not CalibrationMethod.NONE


def _train_binary(
    candidate: CandidateSpec, *, context: TrainingContext, spec: ModelSpec
) -> TrainingRunOutcome:
    """Train the primary binary task, then calibrate and choose an operating point."""
    support = context.config.support
    population = context.supervised_train()
    if population.positive_count < support.min_train_positive_rows:
        return _unavailable(
            candidate,
            TrainingRunStatus.INSUFFICIENT_SUPPORT,
            "min_train_positive_rows",
        )
    if population.positive_count == 0 or population.benign_count == 0:
        return _unavailable(
            candidate, TrainingRunStatus.INSUFFICIENT_SUPPORT, "both_classes_required"
        )

    targets = population.binary_targets()
    weights = (
        None
        if context.config.imbalance.class_weight_policy == "none"
        else compute_class_weights(
            list(targets),
            task=MLTask.BINARY_MALICIOUS,
            class_order=BINARY_CLASS_ORDER,
            config=context.config.imbalance,
        )
    )
    try:
        preprocessor = context.preprocessor_for(MLTask.BINARY_MALICIOUS)
        adapter, fitted = _fit(
            candidate,
            context=context,
            preprocessor=preprocessor,
            population=population,
            targets=targets,
            class_order=BINARY_CLASS_ORDER,
            weights=weights,
        )
    except ModelTrainingError:
        return _unavailable(candidate, TrainingRunStatus.UNAVAILABLE, "model_fit")

    partial = TrainingRunOutcome(
        candidate=candidate,
        status=TrainingRunStatus.COMPLETED,
        fitted=fitted,
        preprocessor=preprocessor,
        class_weights=weights,
        train_row_count=population.row_count,
    )

    validation_a = context.validation(ValidationPartition.VALIDATION_A)
    validation_b = context.validation(ValidationPartition.VALIDATION_B)
    ml_config_fingerprint = context.config.fingerprint()

    calibration_outcome: CalibrationOutcome | None = None
    calibration_state: CalibrationState | None = None
    diagnostic: CalibrationReport | None = None
    quality: CalibrationReport | None = None

    if not _calibration_applies(spec, context.config):
        # The mandatory comparator is not calibrated, by contract rather than by
        # accident. M-000 emits the training class prior for every row, so there
        # is no score variation for a calibrator to map, and both methods
        # require distinct scores. Faking one, or relaxing the support rules
        # until Platt accepted a constant, would make the comparator's number
        # mean less rather than more -- so the run keeps its ``decision_score``
        # and chooses an operating point on it under the Milestone 5 raw-score
        # contract. This is not a failure: `NOT_CALIBRATED` is an ordinary
        # outcome, and a reference run reaches COMPLETED without a calibrator.
        calibration_outcome = CalibrationOutcome(
            status=CalibrationStatus.NOT_CALIBRATED,
            method=CalibrationMethod.NONE,
            row_count=validation_a.row_count,
            positive_count=validation_a.positive_count,
            negative_count=validation_a.benign_count,
            distinct_score_count=0,
        )
        operating_sample = _binary_sample(
            context,
            preprocessor,
            fitted,
            adapter,
            validation_b,
            ValidationPartition.VALIDATION_B,
        )
    elif context.config.calibration.method is not CalibrationMethod.NONE:
        if not spec.calibration_compatible:
            return _replace(
                partial,
                status=TrainingRunStatus.CALIBRATION_UNAVAILABLE,
                failing_requirements=("calibration_compatible_family",),
            )
        sample_a = _binary_sample(
            context,
            preprocessor,
            fitted,
            adapter,
            validation_a,
            ValidationPartition.VALIDATION_A,
        )
        calibration_outcome = fit_calibration(
            sample_a,
            config=context.config.calibration,
            ml_config_fingerprint=ml_config_fingerprint,
        )
        if calibration_outcome.status is not CalibrationStatus.FITTED:
            return _replace(
                partial,
                status=TrainingRunStatus.CALIBRATION_UNAVAILABLE,
                failing_requirements=calibration_outcome.failing_requirements,
                calibration=calibration_outcome,
            )
        calibration_state = calibration_outcome.require_state()
        diagnostic = diagnose_calibration_fit(
            sample_a,
            state=calibration_state,
            config=context.config.calibration,
            ml_config_fingerprint=ml_config_fingerprint,
            raw_score_semantics=fitted.score_semantics,
        )
        sample_b = _binary_sample(
            context,
            preprocessor,
            fitted,
            adapter,
            validation_b,
            ValidationPartition.VALIDATION_B,
        )
        quality = evaluate_calibration_quality(
            sample_b,
            state=calibration_state,
            config=context.config.calibration,
            ml_config_fingerprint=ml_config_fingerprint,
            raw_score_semantics=fitted.score_semantics,
        )
        operating_sample = apply_calibration(
            calibration_state, sample_b, ml_config_fingerprint=ml_config_fingerprint
        )
    else:
        operating_sample = _binary_sample(
            context,
            preprocessor,
            fitted,
            adapter,
            validation_b,
            ValidationPartition.VALIDATION_B,
        )

    selection = select_binary_threshold(
        operating_sample,
        config=context.config.thresholds,
        support=support,
        ml_config_fingerprint=ml_config_fingerprint,
        calibration=calibration_state,
    )
    outcome = _replace(
        partial,
        calibration=calibration_outcome,
        calibration_state=calibration_state,
        calibration_diagnostic=diagnostic,
        calibration_quality=quality,
        binary_threshold=selection,
    )
    if selection.status is not SelectionStatus.SELECTED:
        return _replace(
            outcome,
            status=TrainingRunStatus.THRESHOLD_UNAVAILABLE,
            failing_requirements=selection.failing_requirements
            or (str(selection.status),),
        )
    return outcome


def _train_category(
    candidate: CandidateSpec, *, context: TrainingContext
) -> TrainingRunOutcome:
    """Train the known-malicious category head and choose its abstention point."""
    population = context.category_train()
    class_order = KNOWN_CATEGORY_CLASSES
    counts = dict.fromkeys(class_order, 0)
    for value in population.known_category:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    represented = tuple(name for name in class_order if counts[name] > 0)
    if len(represented) < 2:
        return _unavailable(
            candidate,
            TrainingRunStatus.INSUFFICIENT_SUPPORT,
            "min_represented_categories",
        )
    thin = [
        name
        for name in represented
        if counts[name] < context.config.category.min_rows_per_category
    ]
    if thin:
        return _unavailable(
            candidate, TrainingRunStatus.INSUFFICIENT_SUPPORT, "min_rows_per_category"
        )

    targets = population.category_targets()
    weights = (
        None
        if context.config.imbalance.class_weight_policy == "none"
        else compute_class_weights(
            list(targets),
            task=MLTask.ATTACK_CATEGORY,
            class_order=represented,
            config=context.config.imbalance,
        )
    )
    try:
        # Fitted on known-malicious training rows alone. Benign traffic must not
        # reach this state: an imputation median, a country vocabulary, or a
        # rare-value bucket learned from rows the head will never be asked about
        # is a statistic from the wrong population baked into the pipeline.
        preprocessor = context.preprocessor_for(MLTask.ATTACK_CATEGORY)
        adapter, fitted = _fit(
            candidate,
            context=context,
            preprocessor=preprocessor,
            population=population,
            targets=targets,
            class_order=represented,
            weights=weights,
        )
    except ModelTrainingError:
        return _unavailable(candidate, TrainingRunStatus.UNAVAILABLE, "model_fit")

    partial = TrainingRunOutcome(
        candidate=candidate,
        status=TrainingRunStatus.COMPLETED,
        fitted=fitted,
        preprocessor=preprocessor,
        class_weights=weights,
        train_row_count=population.row_count,
    )

    validation_b = context.validation_category(ValidationPartition.VALIDATION_B)
    if validation_b.row_count == 0:
        return _replace(
            partial,
            status=TrainingRunStatus.THRESHOLD_UNAVAILABLE,
            failing_requirements=("known_malicious_validation_rows",),
        )
    scores = _score(preprocessor, fitted, adapter, validation_b)
    sample = CategoryScoreSample(
        source=ScoreSampleSource(
            split=MLSplit.VALIDATION,
            partition=ValidationPartition.VALIDATION_B,
            source_fingerprint=context.partition.fingerprint,
        ),
        anchors=validation_b.frame.anchors,
        class_order=represented,
        class_scores=scores,
        true_category=validation_b.category_targets(),
        malicious=validation_b.malicious,
        score_kind=ScoreKind.CLASS_SCORE,
        model_id=_model_id(fitted),
        model_content_fingerprint=fitted.content_fingerprint(),
        preprocessor_fingerprint=fitted.preprocessor_fingerprint,
    )
    abstention = select_category_abstention(
        sample,
        config=context.config.category,
        support=context.config.support,
        ml_config_fingerprint=context.config.fingerprint(),
    )
    # A conservative fallback is a usable threshold, so the run is complete: it
    # published everything the category task needs. That the threshold was not
    # chosen from data is recorded on the selection itself, permanently, and is
    # not restated as a run failure.
    return _replace(partial, category_abstention=abstention)


def _train_anomaly(
    candidate: CandidateSpec, *, context: TrainingContext
) -> TrainingRunOutcome:
    """Fit the experimental probe on benign training rows and threshold it."""
    population = context.benign_train()
    if population.row_count < context.config.anomaly.min_fit_rows:
        return _unavailable(
            candidate, TrainingRunStatus.INSUFFICIENT_SUPPORT, "min_fit_rows"
        )
    try:
        # Fitted on benign training rows alone, and so is its preprocessing. A
        # median or a vocabulary learned from malicious rows would let the
        # probe read, through its own encoder, the labels it is supposed to be
        # blind to -- and the whole point of an unsupervised comparator is that
        # it never saw them.
        preprocessor = context.preprocessor_for(MLTask.ANOMALY)
        adapter, fitted = _fit(
            candidate,
            context=context,
            preprocessor=preprocessor,
            population=population,
            targets=(),
            class_order=(),
            weights=None,
        )
    except ModelTrainingError:
        return _unavailable(candidate, TrainingRunStatus.UNAVAILABLE, "model_fit")

    partial = TrainingRunOutcome(
        candidate=candidate,
        status=TrainingRunStatus.COMPLETED,
        fitted=fitted,
        preprocessor=preprocessor,
        train_row_count=population.row_count,
    )

    method = context.config.anomaly.threshold_method
    if str(method) == "train_benign_quantile":
        source_population = population
        source = ScoreSampleSource(
            split=MLSplit.TRAIN,
            partition=None,
            # The benign training rows this probe may read, not every row the
            # dataset holds. The dataset-wide digest covers the test split and
            # the holdout, and a threshold provenance carrying it would move
            # whenever they did.
            source_fingerprint=context.readable_lineage(MLTask.ANOMALY).training_data,
        )
    else:
        source_population = context.validation_benign(ValidationPartition.VALIDATION_A)
        source = ScoreSampleSource(
            split=MLSplit.VALIDATION,
            partition=ValidationPartition.VALIDATION_A,
            source_fingerprint=context.partition.fingerprint,
        )
    if source_population.row_count == 0:
        return _replace(
            partial,
            status=TrainingRunStatus.THRESHOLD_UNAVAILABLE,
            failing_requirements=("benign_threshold_rows",),
        )

    scored = _score(preprocessor, fitted, adapter, source_population)
    sample = AnomalyScoreSample(
        source=source,
        anchors=source_population.frame.anchors,
        scores=tuple(float(row[0]) for row in scored),
        malicious=source_population.malicious,
        score_kind=ScoreKind.ANOMALY_SCORE,
        model_id=_model_id(fitted),
        model_content_fingerprint=fitted.content_fingerprint(),
        preprocessor_fingerprint=fitted.preprocessor_fingerprint,
    )
    selection = select_anomaly_threshold(
        sample,
        config=context.config.anomaly,
        support=context.config.support,
        ml_config_fingerprint=context.config.fingerprint(),
    )
    outcome = _replace(partial, anomaly_threshold=selection)
    if selection.status is not SelectionStatus.SELECTED:
        return _replace(
            outcome,
            status=TrainingRunStatus.THRESHOLD_UNAVAILABLE,
            failing_requirements=selection.failing_requirements,
        )
    return outcome


def _replace(outcome: TrainingRunOutcome, **changes: Any) -> TrainingRunOutcome:
    """Return *outcome* with *changes* applied."""
    from dataclasses import replace

    return replace(outcome, **changes)


def train_all(context: TrainingContext) -> tuple[TrainingRunOutcome, ...]:
    """Train every candidate the configuration declares, in enumeration order.

    Every candidate is attempted. One that cannot be trained produces an
    outcome naming why and the next one is attempted anyway: a run that stopped
    at the first refusal would report a comparison of whichever candidates
    happened to come first alphabetically.
    """
    return tuple(
        train_candidate(candidate, context=context)
        for candidate in enumerate_candidates(context.config, catalog=context.catalog)
    )
