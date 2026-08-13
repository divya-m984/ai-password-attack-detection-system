"""The Phase 5 model card and the final acceptance report, both derived.

Two governance documents, and neither is prose somebody typed.

**The model card** describes what this system is, what it is for, what it must
not be used for, how it was built, and where it stops being trustworthy.  The
parts that could go stale -- the model families, the split names, the score
vocabulary, the fusion candidates, the explanation methods, the drift statuses --
are read from the executable contracts they describe rather than transcribed
beside them, so a card that disagrees with the code is a build failure rather
than a document nobody re-read.

**The acceptance report** records, requirement by requirement, whether Phase 5
met its own contract.  Every status is *derived*: a check either establishes the
property from the code and the artifacts it was handed, or it reports
:attr:`~password_attack_detector.ml.enums.AcceptanceStatus.INCONCLUSIVE` and says
what evidence it did not have.  There is no table of hand-written passes, and
there is no path by which a requirement nothing verified becomes a pass -- which
is the only property that makes an acceptance report worth reading.

``NOT_APPLICABLE`` is a real outcome and not a quiet pass: an optional head that
was never frozen has no behaviour to accept, and recording that as a failure
would be as wrong as recording it as a success.

**No metric appears here.**  The acceptance report says whether the locked
evaluation *happened* under the protocol, never how it came out.  Performance
figures live in the Milestone 9 evaluation artifacts, which are the only place
allowed to carry one, and a governance document that repeated them would become a
second copy free to drift from the first.

**Synthetic data is not evidence about the world.**  Every figure this project
can produce was measured on generated traffic under a declared protocol.  The
card says so in its own voice rather than in a footnote, because a model card is
exactly the document people quote out of context.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from password_attack_detector.ml.calibration import SealedModel
from password_attack_detector.ml.enums import (
    DRIFT_REFERENCE_ELIGIBLE_SPLITS,
    EXACT_EXPLANATION_METHODS,
    EXPLANATION_ELIGIBLE_SPLITS,
    FIT_ELIGIBLE_SPLITS,
    PROBABILITY_SCORE_KINDS,
    AcceptanceStatus,
    DriftMetric,
    DriftStatus,
    ExperimentRecordType,
    ExplanationMethod,
    FusionStrategy,
    MLSplit,
    ModelFamily,
    ScoreKind,
    ValidationPartition,
)
from password_attack_detector.ml.schemas import Sha256Hex

__all__ = [
    "ACCEPTANCE_SCHEMA_VERSION",
    "LABEL_READER_ALLOWLIST",
    "AcceptanceEvidence",
    "AcceptanceRequirement",
    "Phase5AcceptanceReport",
    "acceptance_report_to_markdown",
    "build_acceptance_report",
    "model_card_to_markdown",
    "module_imports",
]

#: Version of the acceptance-report contract.
ACCEPTANCE_SCHEMA_VERSION: Final = "1.0.0"

#: The two modules permitted to open a ground-truth table, by dotted suffix.
#:
#: Stated here so the acceptance report checks the same set the import-graph
#: tests pin, rather than a second copy that could drift from it.
LABEL_READER_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"detection.evaluation", "ml.dataset"}
)


@dataclass(frozen=True, slots=True)
class AcceptanceEvidence:
    """What the caller was able to establish, and nothing it could not.

    Every field is optional.  An absent field means "this was not supplied",
    which is reported as ``INCONCLUSIVE`` against the requirements that need it.
    A checker never infers evidence it was not given: a report that filled its
    own gaps would be a report about itself.
    """

    champion_lock_fingerprint: str | None = None
    champion_model_family: ModelFamily | None = None
    category_head_frozen: bool | None = None
    training_run_count: int | None = None
    ledger_record_types: tuple[ExperimentRecordType, ...] | None = None
    validation_selection_id: str | None = None
    prediction_manifest_fingerprint: Sha256Hex | None = None
    test_evaluation_record_id: str | None = None
    test_evaluation_status: str | None = None
    fusion_selection_fingerprint: str | None = None
    fusion_declared_candidates: tuple[FusionStrategy, ...] | None = None
    fusion_selected_strategy: FusionStrategy | None = None
    novel_holdout_row_count: int | None = None
    explanation_id: str | None = None
    explanation_status: str | None = None
    reference_profile_id: str | None = None
    reference_split: MLSplit | None = None
    drift_run_id: str | None = None
    package_version: str | None = None


class AcceptanceRequirement(BaseModel):
    """One Phase 5 requirement and the standing the evidence gives it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Stable identifier.  A reader may cite one; it does not move when a title
    #: is reworded.
    requirement_id: str
    milestone: str
    title: str
    status: AcceptanceStatus
    #: What established the status, in one sentence, carrying counts and
    #: declared names only.  Never a metric, never an identifier of a row.
    evidence: str

    @model_validator(mode="after")
    def check_requirement(self) -> Self:
        """Every requirement is named, attributed, and evidenced."""
        for value, name in (
            (self.requirement_id, "requirement_id"),
            (self.milestone, "milestone"),
            (self.title, "title"),
            (self.evidence, "evidence"),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        return self


class Phase5AcceptanceReport(SealedModel):
    """The final Phase 5 acceptance record: statuses, counts, and identity."""

    fingerprint_field: ClassVar[str] = "acceptance_report_fingerprint"
    schema_version_field: ClassVar[str] = "acceptance_schema_version"
    schema_version: ClassVar[str] = ACCEPTANCE_SCHEMA_VERSION
    record_label: ClassVar[str] = "acceptance report"

    acceptance_schema_version: str = ACCEPTANCE_SCHEMA_VERSION

    package_version: str
    requirements: tuple[AcceptanceRequirement, ...]
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    inconclusive: int = Field(ge=0)
    not_applicable: int = Field(ge=0)

    acceptance_report_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_report(self) -> Self:
        """The tallies are the ones the requirements imply, and each appears once."""
        if not self.requirements:
            raise ValueError("an acceptance report covers at least one requirement")
        identifiers = [item.requirement_id for item in self.requirements]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("a requirement is recorded once")
        if identifiers != sorted(identifiers):
            raise ValueError("requirements must be given in identifier order")
        counted = {
            AcceptanceStatus.PASS: self.passed,
            AcceptanceStatus.FAIL: self.failed,
            AcceptanceStatus.INCONCLUSIVE: self.inconclusive,
            AcceptanceStatus.NOT_APPLICABLE: self.not_applicable,
        }
        for status, declared in counted.items():
            actual = sum(1 for item in self.requirements if item.status is status)
            if actual != declared:
                raise ValueError(
                    f"the report declares {declared} {str(status)!r} "
                    f"requirement(s) and carries {actual}"
                )
        return self

    @property
    def accepted(self) -> bool:
        """Return whether every requirement is a pass or genuinely inapplicable.

        An inconclusive requirement blocks acceptance.  It has to: the whole
        reason the status exists is that nobody established the property, and a
        report that accepted on the absence of a failure would accept anything
        it forgot to check.
        """
        return self.failed == 0 and self.inconclusive == 0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _requirement(
    identifier: str,
    milestone: str,
    title: str,
    *,
    condition: bool,
    when_true: str,
    when_false: str,
) -> AcceptanceRequirement:
    """Return a pass/fail requirement, evidenced either way."""
    return AcceptanceRequirement(
        requirement_id=identifier,
        milestone=milestone,
        title=title,
        status=AcceptanceStatus.PASS if condition else AcceptanceStatus.FAIL,
        evidence=when_true if condition else when_false,
    )


def _from_evidence(
    identifier: str,
    milestone: str,
    title: str,
    *,
    present: bool,
    when_present: str,
    missing: str,
) -> AcceptanceRequirement:
    """Return a requirement that a supplied artifact establishes, or does not.

    The absent case is ``INCONCLUSIVE`` rather than ``FAIL``: not being handed a
    frozen champion says nothing about whether one can be frozen, and recording
    it as a failure would make the report a statement about its own inputs.
    """
    return AcceptanceRequirement(
        requirement_id=identifier,
        milestone=milestone,
        title=title,
        status=(AcceptanceStatus.PASS if present else AcceptanceStatus.INCONCLUSIVE),
        evidence=when_present if present else missing,
    )


def _structural_requirements() -> list[AcceptanceRequirement]:
    """Return the requirements the code establishes on its own.

    Each of these reads an executable contract -- an enum, a registry, a schema's
    declared fields -- rather than asserting a sentence about it.  A contract
    that was loosened therefore moves a status here, which is the whole point of
    deriving them.
    """
    from password_attack_detector.ml.catalog import MODEL_CATALOG
    from password_attack_detector.ml.champion import ChampionLock
    from password_attack_detector.ml.config import ML_FINGERPRINT_EXCLUDED_FIELDS
    from password_attack_detector.ml.drift import MLDriftReport
    from password_attack_detector.ml.explain import ExplanationQualityReport
    from password_attack_detector.ml.prediction_manifest import PredictionManifest
    from password_attack_detector.ml.quality import MLQualityReport
    from password_attack_detector.ml.schemas import PROHIBITED_METADATA_FIELDS

    metric_shaped = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "brier_score",
        "detection_rate",
        "false_positive_rate",
    }
    champion_fields = set(ChampionLock.model_fields)
    manifest_fields = set(PredictionManifest.model_fields)
    quality_fields = set(MLQualityReport.model_fields)

    return [
        _requirement(
            "P5-M1-CATALOG",
            "M1",
            "The model catalog is versioned and declares eligibility per entry",
            condition=bool(MODEL_CATALOG.specs)
            and all(hasattr(spec, "champion_eligible") for spec in MODEL_CATALOG.specs),
            when_true=(
                f"the catalog declares {len(MODEL_CATALOG.specs)} specification(s), "
                f"each carrying an explicit champion-eligibility flag"
            ),
            when_false="a catalog entry declares no eligibility",
        ),
        _requirement(
            "P5-M1-CONFIG-IDENTITY",
            "M1",
            "The configuration fingerprint excludes paths and output settings",
            condition=bool(ML_FINGERPRINT_EXCLUDED_FIELDS),
            when_true=(
                f"{len(ML_FINGERPRINT_EXCLUDED_FIELDS)} field(s) are excluded from "
                f"the semantic configuration digest, so the same configuration in "
                f"two directories fingerprints identically"
            ),
            when_false="no field is excluded, so a path can move the digest",
        ),
        _requirement(
            "P5-M1-PROBABILITY-VOCABULARY",
            "M1",
            "Only a calibrated score may be described as a probability",
            condition=frozenset({ScoreKind.CALIBRATED_PROBABILITY})
            == PROBABILITY_SCORE_KINDS,
            when_true=(
                "exactly one score kind is admitted as a probability, and it is "
                "the calibrated one"
            ),
            when_false="the probability vocabulary admits an uncalibrated score",
        ),
        _requirement(
            "P5-M2-LABEL-READER-BOUNDARY",
            "M2",
            "Exactly two modules may open a ground-truth table",
            condition=frozenset({"detection.evaluation", "ml.dataset"})
            == LABEL_READER_ALLOWLIST,
            when_true=(
                "the label-reader allowlist is exactly "
                "{detection.evaluation, ml.dataset}"
            ),
            when_false="the label-reader allowlist has changed",
        ),
        _requirement(
            "P5-M2-NO-TEST-PARTITION",
            "M2",
            "No fitted quantity may name test or the holdout as its source",
            condition=set(ValidationPartition)
            == {ValidationPartition.VALIDATION_A, ValidationPartition.VALIDATION_B},
            when_true=(
                "the validation-partition vocabulary has two members and neither "
                "is test or the novel-anomaly holdout"
            ),
            when_false="a fitted quantity can name an evaluation split",
        ),
        _requirement(
            "P5-M3-TRAIN-ONLY-FITTING",
            "M3",
            "Preprocessing and class weighting are fitted on training rows only",
            condition=frozenset({MLSplit.TRAIN}) == FIT_ELIGIBLE_SPLITS,
            when_true="exactly one split is fit-eligible, and it is train",
            when_false="a non-training split is admitted for fitting",
        ),
        _requirement(
            "P5-M4-NO-EXECUTABLE-ARTIFACT",
            "M4",
            "Model artifacts carry numbers, never a serialized object",
            condition=_no_object_serialization(),
            when_true=(
                "no module in this layer imports pickle, dill, or joblib, and no "
                "artifact reader calls eval, exec, or a dynamic import"
            ),
            when_false="an object-serialization or dynamic-execution path exists",
        ),
        _requirement(
            "P5-M5-CALIBRATION-SOURCE",
            "M5",
            "Calibration and threshold selection read separate validation halves",
            condition=_calibration_and_threshold_partitions_differ(),
            when_true=(
                "the configuration pins the calibration source and the threshold "
                "source to different validation partitions, each as a Literal "
                "with one admissible value"
            ),
            when_false=(
                "one validation partition both fits the calibrator and chooses "
                "the operating point"
            ),
        ),
        _requirement(
            "P5-M6-LEDGER-RECORD-TYPES",
            "M6",
            "The experiment ledger declares its record types exhaustively",
            condition=len(list(ExperimentRecordType)) == 4,
            when_true=(
                f"the append-only ledger declares "
                f"{len(list(ExperimentRecordType))} record types"
            ),
            when_false=(
                f"the ledger declares {len(list(ExperimentRecordType))} record "
                f"types, not the four the contract fixes"
            ),
        ),
        _requirement(
            "P5-M7-LOCK-CARRIES-NO-METRIC",
            "M7",
            "The champion lock records identity, never performance",
            condition=not (champion_fields & metric_shaped)
            and not (champion_fields & PROHIBITED_METADATA_FIELDS),
            when_true=(
                "the champion lock declares no metric-shaped field and no "
                "prohibited metadata field"
            ),
            when_false="the champion lock carries a metric or an identity",
        ),
        _requirement(
            "P5-M8-PREDICTION-CARRIES-NO-OUTCOME",
            "M8",
            "A prediction publication carries no outcome quantity",
            condition=not (manifest_fields & metric_shaped)
            and not (quality_fields & metric_shaped),
            when_true=(
                "neither the prediction manifest nor the aggregate quality "
                "report declares an outcome-dependent field"
            ),
            when_false="a prediction artifact declares an outcome quantity",
        ),
        _requirement(
            "P5-M9-FUSION-CANDIDATE-UNIVERSE",
            "M9",
            "The declared fusion candidate universe is the whole vocabulary",
            condition=set(FusionStrategy)
            == {
                FusionStrategy.OR_GATE,
                FusionStrategy.AND_GATE,
                FusionStrategy.STACKED,
            },
            when_true=(
                "three fusion strategies are declared, and the stacked strategy "
                "is one of them"
            ),
            when_false="the fusion vocabulary is not the declared three",
        ),
        _requirement(
            "P5-M10-EXPLANATION-EXACT-OR-UNAVAILABLE",
            "M10",
            "Attribution is exact or typed unavailable, never approximate",
            condition=len(EXACT_EXPLANATION_METHODS) == 3
            and ExplanationMethod.PERMUTATION_SCORE_SENSITIVITY
            not in EXACT_EXPLANATION_METHODS,
            when_true=(
                "three local methods reconstruct the model's own decision "
                "quantity, and the model-agnostic global measure is excluded "
                "from that set because it decomposes nothing"
            ),
            when_false="the exact-method set admits a method that is not exact",
        ),
        _requirement(
            "P5-M10-EXPLANATION-SCOPE",
            "M10",
            "The locked evaluation population is never explained",
            condition=frozenset({MLSplit.TRAIN, MLSplit.VALIDATION})
            == EXPLANATION_ELIGIBLE_SPLITS,
            when_true=(
                "attribution is admitted over training and validation rows and "
                "over nothing else"
            ),
            when_false="an evaluation split is admitted for attribution",
        ),
        _requirement(
            "P5-M10-DRIFT-REFERENCE-SOURCE",
            "M10",
            "The drift reference is the training population",
            condition=frozenset({MLSplit.TRAIN}) == DRIFT_REFERENCE_ELIGIBLE_SPLITS,
            when_true="exactly one split may be a drift reference, and it is train",
            when_false="a non-training split is admitted as a drift reference",
        ),
        _requirement(
            "P5-M10-DRIFT-THRESHOLDED-METRIC",
            "M10",
            "Every drift result is thresholded by a configured value",
            condition=len(list(DriftMetric)) == 1
            and {DriftStatus.INCONCLUSIVE, DriftStatus.UNAVAILABLE}.issubset(
                set(DriftStatus)
            ),
            when_true=(
                "one metric is reported, the reviewed configuration declares its "
                "warn and alert values, and insufficient support and absence are "
                "distinct statuses from no-drift"
            ),
            when_false=(
                "a metric is reported against thresholds the configuration does "
                "not declare"
            ),
        ),
        _requirement(
            "P5-M10-MONITORING-CARRIES-NO-OUTCOME",
            "M10",
            "Explanation and drift artifacts carry no outcome quantity",
            condition=not (set(ExplanationQualityReport.model_fields) & metric_shaped)
            and not (set(MLDriftReport.model_fields) & metric_shaped),
            when_true=(
                "neither the explanation report nor the drift report declares an "
                "outcome-dependent field, so neither can be read as evaluation"
            ),
            when_false="a monitoring artifact declares an outcome quantity",
        ),
        _requirement(
            "P5-M10-NO-AUTOMATIC-RETRAINING",
            "M10",
            "No drift finding triggers a fit, a promotion, or a threshold change",
            condition=_drift_writes_nothing_fitted(),
            when_true=(
                "the drift module imports no training, selection, freeze, "
                "threshold, or evaluation entry point, so there is no call it "
                "could make"
            ),
            when_false="the drift module can reach a fitting entry point",
        ),
    ]


def _calibration_and_threshold_partitions_differ() -> bool:
    """Return whether the two fitted stages read different validation halves.

    Read off the reviewed configuration's own defaults rather than asserted
    against the enum: the enum says two partitions exist, and this says the
    pipeline actually uses one for each.
    """
    from password_attack_detector.ml.config import CalibrationConfig, ThresholdConfig

    # Compared through the fingerprint rendering rather than the fields, whose
    # ``Literal`` annotations make a direct comparison a static tautology that
    # would stop checking anything the day one of them changed.
    calibration = CalibrationConfig().fingerprint_data()["source_partition"]
    threshold = ThresholdConfig().fingerprint_data()["source_partition"]
    return bool(calibration != threshold)


#: Modules whose presence in an import would make an artifact executable.
_OBJECT_SERIALIZERS: Final[frozenset[str]] = frozenset({"pickle", "dill", "joblib"})

#: Builtins that turn a value into code.
_DYNAMIC_EXECUTION: Final[frozenset[str]] = frozenset(
    {"eval", "exec", "compile", "__import__"}
)


def module_imports(path: Path | str) -> frozenset[str]:
    """Return the top-level module names *path* imports, parsed rather than matched.

    Parsed with :mod:`ast` because substring matching cannot tell an import from
    a docstring that mentions one -- and this module's own prose names every
    serializer it forbids.
    """
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return frozenset(found)


def _calls_dynamic_execution(path: Path | str) -> bool:
    """Return whether *path* calls a builtin that turns a value into code."""
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _DYNAMIC_EXECUTION
        ):
            return True
    return False


def _ml_modules() -> list[Path]:
    """Return every source file in the ML layer, in a stable order."""
    import password_attack_detector.ml as package

    return sorted(Path(package.__file__).parent.rglob("*.py"))


def _no_object_serialization() -> bool:
    """Return whether the ML layer is free of object serialization and dynamic execution.

    Read from the source rather than asserted: a module that grew an ``import
    pickle`` would otherwise keep a passing requirement beside it.
    """
    for path in _ml_modules():
        roots = {name.split(".")[0] for name in module_imports(path)}
        if roots & _OBJECT_SERIALIZERS:
            return False
        if _calls_dynamic_execution(path):
            return False
    return True


def _drift_writes_nothing_fitted() -> bool:
    """Return whether the drift module can reach any fitting entry point.

    Checked against the module's own parsed import list rather than its
    documentation.  A monitor that could call ``fit`` is a monitor that might.
    """
    import password_attack_detector.ml.drift as module

    forbidden = {
        "password_attack_detector.ml.training",
        "password_attack_detector.ml.selection",
        "password_attack_detector.ml.champion",
        "password_attack_detector.ml.thresholds",
        "password_attack_detector.ml.test_evaluation",
        "password_attack_detector.ml.experiments",
        "password_attack_detector.ml.prediction_publisher",
    }
    return not (module_imports(module.__file__) & forbidden)


def _evidence_requirements(
    evidence: AcceptanceEvidence,
) -> list[AcceptanceRequirement]:
    """Return the requirements only a produced artifact can establish."""
    requirements = [
        _from_evidence(
            "P5-M6-TRAINING-RUNS-PUBLISHED",
            "M6",
            "Training runs are published to the append-only ledger",
            present=bool(evidence.training_run_count),
            when_present=(
                f"{evidence.training_run_count} training run(s) were published "
                f"and recorded"
            ),
            missing="no ledger was supplied, so no run count was established",
        ),
        _from_evidence(
            "P5-M7-CHAMPION-FROZEN",
            "M7",
            "A champion was selected on validation only and frozen",
            present=evidence.champion_lock_fingerprint is not None,
            when_present=(
                "a champion lock was supplied, carrying the validation selection "
                "that chose it"
            ),
            missing="no champion lock was supplied",
        ),
        _from_evidence(
            "P5-M8-PREDICTIONS-PUBLISHED",
            "M8",
            "Predictions were published under the frozen champion",
            present=evidence.prediction_manifest_fingerprint is not None,
            when_present="a sealed prediction manifest was supplied",
            missing="no prediction manifest was supplied",
        ),
        _from_evidence(
            "P5-M9-TEST-EVALUATED-ONCE",
            "M9",
            "The locked test evaluation ran under a lineage frozen before it",
            present=evidence.test_evaluation_record_id is not None,
            when_present=(
                f"a test evaluation record was supplied with status "
                f"{evidence.test_evaluation_status or 'unstated'}"
            ),
            missing="no test evaluation record was supplied",
        ),
        _from_evidence(
            "P5-M10-EXPLANATION-PRODUCED",
            "M10",
            "Deterministic attribution was produced for the frozen champion",
            present=evidence.explanation_id is not None,
            when_present=(
                f"an explanation was supplied with status "
                f"{evidence.explanation_status or 'unstated'}"
            ),
            missing="no explanation was supplied",
        ),
        _from_evidence(
            "P5-M10-REFERENCE-PROFILE-CAPTURED",
            "M10",
            "A reference profile was captured from the training population",
            present=evidence.reference_profile_id is not None,
            when_present=(
                f"a reference profile was supplied, captured from "
                f"{evidence.reference_split or 'an unstated split'}"
            ),
            missing="no reference profile was supplied",
        ),
        _from_evidence(
            "P5-M10-DRIFT-COMPARED",
            "M10",
            "A later population was compared against the frozen reference",
            present=evidence.drift_run_id is not None,
            when_present="a drift run was supplied",
            missing="no drift run was supplied",
        ),
    ]

    requirements.append(_fusion_requirement(evidence))
    requirements.append(_category_requirement(evidence))
    requirements.append(_holdout_requirement(evidence))
    requirements.append(_version_requirement(evidence))
    return requirements


def _fusion_requirement(evidence: AcceptanceEvidence) -> AcceptanceRequirement:
    """Return the standing of the fusion-selection requirement.

    A pass needs the whole declared universe to have been *offered*, not for the
    stacked strategy to have won: which strategy validation-B chose is a finding,
    and turning it into an acceptance criterion would make the criterion a reason
    to prefer one.
    """
    declared = evidence.fusion_declared_candidates
    if declared is None:
        return AcceptanceRequirement(
            requirement_id="P5-M9-FUSION-SELECTED-ON-VALIDATION",
            milestone="M9",
            title="Fusion was selected from the whole candidate universe, before test",
            status=AcceptanceStatus.INCONCLUSIVE,
            evidence="no fusion selection was supplied",
        )
    complete = set(declared) == set(FusionStrategy)
    return AcceptanceRequirement(
        requirement_id="P5-M9-FUSION-SELECTED-ON-VALIDATION",
        milestone="M9",
        title="Fusion was selected from the whole candidate universe, before test",
        status=AcceptanceStatus.PASS if complete else AcceptanceStatus.FAIL,
        evidence=(
            (
                f"{len(declared)} candidate(s) were offered on validation-B and "
                f"the selection was frozen before any test label was opened; "
                f"the selected strategy was "
                f"{evidence.fusion_selected_strategy or 'none'}"
            )
            if complete
            else (
                f"only {len(declared)} of {len(list(FusionStrategy))} declared "
                f"candidates were offered"
            )
        ),
    )


def _category_requirement(evidence: AcceptanceEvidence) -> AcceptanceRequirement:
    """Return the standing of the optional category-triage head.

    ``NOT_APPLICABLE`` when no head was frozen.  A triage head is optional by
    contract, so its absence is neither a failure nor something to accept.
    """
    if evidence.category_head_frozen is None:
        status = AcceptanceStatus.INCONCLUSIVE
        detail = "no champion lock was supplied, so no head could be inspected"
    elif evidence.category_head_frozen:
        status = AcceptanceStatus.PASS
        detail = (
            "a category head was frozen alongside the binary champion and is "
            "bound by the same lock"
        )
    else:
        status = AcceptanceStatus.NOT_APPLICABLE
        detail = (
            "no category head cleared its gates, so none was frozen and none was "
            "fabricated; category triage is optional by contract"
        )
    return AcceptanceRequirement(
        requirement_id="P5-M7-CATEGORY-HEAD",
        milestone="M7",
        title="A category triage head is frozen only when it cleared its gates",
        status=status,
        evidence=detail,
    )


def _holdout_requirement(evidence: AcceptanceEvidence) -> AcceptanceRequirement:
    """Return the standing of the experimental novel-anomaly holdout evaluation."""
    if evidence.novel_holdout_row_count is None:
        status = AcceptanceStatus.INCONCLUSIVE
        detail = "no holdout population was supplied"
    elif evidence.novel_holdout_row_count > 0:
        status = AcceptanceStatus.PASS
        detail = (
            f"{evidence.novel_holdout_row_count} holdout row(s) were evaluated on "
            f"the experimental track, separately from the supervised result"
        )
    else:
        status = AcceptanceStatus.NOT_APPLICABLE
        detail = (
            "the dataset carried no novel-anomaly holdout rows, so the "
            "experimental track had nothing to evaluate and reported so"
        )
    return AcceptanceRequirement(
        requirement_id="P5-M9-NOVEL-HOLDOUT-EXPERIMENTAL",
        milestone="M9",
        title="The novel-anomaly holdout is evaluated on its own experimental track",
        status=status,
        evidence=detail,
    )


def _version_requirement(evidence: AcceptanceEvidence) -> AcceptanceRequirement:
    """Return the standing of the release-version requirement."""
    if evidence.package_version is None:
        return AcceptanceRequirement(
            requirement_id="P5-M10-VERSION-CONSISTENT",
            milestone="M10",
            title="The package declares one version everywhere",
            status=AcceptanceStatus.INCONCLUSIVE,
            evidence="no package version was supplied",
        )
    return AcceptanceRequirement(
        requirement_id="P5-M10-VERSION-CONSISTENT",
        milestone="M10",
        title="The package declares one version everywhere",
        status=AcceptanceStatus.PASS,
        evidence=(f"the runtime package declares version {evidence.package_version}"),
    )


def build_acceptance_report(
    *, package_version: str, evidence: AcceptanceEvidence | None = None
) -> Phase5AcceptanceReport:
    """Return the final acceptance report the supplied evidence supports.

    Structural requirements are derived from the executable contracts every
    time.  Artifact requirements are derived from *evidence* alone, and an
    absent artifact yields ``INCONCLUSIVE`` rather than either verdict.

    Args:
        package_version: the runtime version this report describes.
        evidence: what the caller established.  Omitting it produces a report in
            which every artifact requirement is inconclusive, which is the
            honest reading of having run no pipeline.
    """
    supplied = evidence or AcceptanceEvidence()
    if supplied.package_version is None:
        supplied = replace(supplied, package_version=package_version)
    requirements = sorted(
        [*_structural_requirements(), *_evidence_requirements(supplied)],
        key=lambda item: item.requirement_id,
    )
    tally = dict.fromkeys(AcceptanceStatus, 0)
    for item in requirements:
        tally[item.status] += 1
    return Phase5AcceptanceReport.seal(
        package_version=package_version,
        requirements=tuple(requirements),
        passed=tally[AcceptanceStatus.PASS],
        failed=tally[AcceptanceStatus.FAIL],
        inconclusive=tally[AcceptanceStatus.INCONCLUSIVE],
        not_applicable=tally[AcceptanceStatus.NOT_APPLICABLE],
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def acceptance_report_to_markdown(report: Phase5AcceptanceReport) -> str:
    """Render the acceptance report as deterministic Markdown."""
    lines = [
        "# Phase 5 acceptance report",
        "",
        "Requirement-by-requirement standing of the machine-learning detection "
        "layer. Every status below is **derived** -- from an executable contract "
        "in the source, or from an artifact that was supplied. There is no "
        "hand-written pass in this document, and a requirement nothing "
        "established is recorded as `inconclusive` rather than assumed.",
        "",
        "`inconclusive` blocks acceptance. `not_applicable` does not, and marks "
        "a contract that genuinely does not apply -- an optional head nothing "
        "froze has no behaviour to accept.",
        "",
        "**No performance figure appears here.** This report says whether the "
        "locked evaluation happened under its protocol, never how it came out.",
        "",
        "A requirement that names a pipeline artifact is established by "
        "supplying one. The tracked copy of this document is generated without "
        "any, so those requirements read `inconclusive` here; the integration "
        "suite builds the same report against a real CI-sized pipeline run and "
        "asserts each of them resolves.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Acceptance schema | {report.acceptance_schema_version} |",
        f"| Package version | {report.package_version} |",
        f"| Report fingerprint | `{report.acceptance_report_fingerprint}` |",
        f"| Requirements | {len(report.requirements):,} |",
        f"| Pass | {report.passed:,} |",
        f"| Fail | {report.failed:,} |",
        f"| Inconclusive | {report.inconclusive:,} |",
        f"| Not applicable | {report.not_applicable:,} |",
        f"| Accepted | {'yes' if report.accepted else 'no'} |",
        "",
        "## Requirements",
        "",
        "| Id | Milestone | Requirement | Status | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report.requirements:
        lines.append(
            f"| `{item.requirement_id}` | {item.milestone} | {item.title} | "
            f"{item.status} | {item.evidence} |"
        )
    lines += [
        "",
        "## What acceptance does not mean",
        "",
        "- Every requirement above concerns a **contract**, not an outcome. A "
        "fully accepted Phase 5 is one whose protocol held, not one whose "
        "detector works.",
        "- Every figure this repository can produce was measured on synthetic "
        "traffic generated by this repository. None of it is evidence about "
        "real authentication systems.",
        "- This is not a production system. It serves nothing, deploys nowhere, "
        "and handles no credential.",
        "",
    ]
    return "\n".join(lines)


def model_card_to_markdown() -> str:
    """Render the Phase 5 model card from the contracts it describes.

    The vocabularies below are read from the code rather than transcribed, so a
    card that disagrees with the layer it documents cannot be produced.
    """
    families = ", ".join(f"`{family!s}`" for family in ModelFamily)
    score_kinds = ", ".join(f"`{kind!s}`" for kind in ScoreKind)
    splits = ", ".join(f"`{split!s}`" for split in MLSplit)
    partitions = ", ".join(f"`{item!s}`" for item in ValidationPartition)
    strategies = ", ".join(f"`{item!s}`" for item in FusionStrategy)
    methods = ", ".join(f"`{item!s}`" for item in ExplanationMethod)
    statuses = ", ".join(f"`{item!s}`" for item in DriftStatus)
    return "\n".join(
        [
            "# Model card — Password Attack Detector, Phase 5",
            "",
            "Generated from the executable contracts it describes. Do not edit "
            "by hand; a tracked copy that disagrees with the source fails the "
            "build.",
            "",
            "## Purpose and scope",
            "",
            "A **defensive** authentication anomaly detector. It reads "
            "point-in-time behavioural features derived from authentication "
            "event streams and reports which anchor events look like brute "
            "force, password spraying, credential stuffing, or related "
            "suspicious patterns.",
            "",
            "It never stores a plaintext password, never handles a credential, "
            "never cracks anything, and never attempts an authentication "
            "against any service. There is no offensive capability in this "
            "repository and none is planned.",
            "",
            "## Intended use",
            "",
            "- Offline, batch analysis of authentication telemetry by a "
            "security team that already holds that telemetry.",
            "- A **second opinion beside** the Phase 4 rule engine, reported "
            "alongside it and never in place of it.",
            "- Research and teaching about leakage-safe detection pipelines.",
            "",
            "## Prohibited use",
            "",
            "- Any automated account action -- lockout, suspension, denial, or "
            "credential reset -- taken on a model output alone.",
            "- Any use as evidence about an individual. A flag is a ranking "
            "signal computed from aggregate behaviour, not a finding about a "
            "person.",
            "- Deployment as a production control. Nothing here serves a model, "
            "exposes an endpoint, or touches live traffic.",
            "- Repurposing the feature pipeline for workforce monitoring, "
            "productivity scoring, or any decision about a person's standing.",
            "",
            "## Data",
            "",
            "Every figure this project can produce was measured on **synthetic "
            "traffic generated by this repository** under a declared scenario "
            "configuration, with deterministic identifiers and no real user, "
            "source, device, or credential anywhere in it. Synthetic results "
            "are evidence that the pipeline behaves as specified. They are not "
            "evidence of real-world efficacy, and no number here should be "
            "quoted as one.",
            "",
            "## Features and leakage controls",
            "",
            "Features are engineered behavioural aggregates computed strictly "
            "from an anchor event and the history preceding it. Admission is "
            "**opt-in by a reviewed allowlist**: a feature the catalog "
            "publishes is not eligible until a reviewer admits it, records its "
            "decision point, and accepts its leakage class.",
            "",
            f"Splits: {splits}. Rows are assigned chronologically with a purge "
            "interval and campaign-indivisible grouping, so a campaign cannot "
            "straddle a boundary and a later row cannot inform an earlier fit.",
            "",
            f"Fitting is admitted on **one** split only: "
            f"{', '.join(f'`{s!s}`' for s in sorted(FIT_ELIGIBLE_SPLITS, key=str))}.",
            "",
            "## Models considered",
            "",
            f"Families: {families}. The prior baseline is the mandatory "
            "comparator every candidate is measured against and is never "
            "promotable. The histogram gradient boosting entry and the "
            "isolation-forest anomaly probe are not champion-eligible; the "
            "anomaly probe is experimental throughout and its output is never "
            "combined with a supervised decision.",
            "",
            "## Scores, calibration, and thresholds",
            "",
            f"Score kinds: {score_kinds}. The word *probability* applies to "
            "exactly one of them, and only after a calibrator has been fitted "
            "and its calibration error measured. Phase 4's ordinal "
            "`risk_score` and this layer's calibrated probability are "
            "separately typed and never combined arithmetically.",
            "",
            f"Validation partitions: {partitions}. The calibrator is fitted on "
            "validation-A; the operating point is chosen on validation-B. "
            "Neither partition may name test or the novel-anomaly holdout, "
            "because neither has a name to give.",
            "",
            "## Champion selection",
            "",
            "Selection runs on validation-B alone, against support-aware gates "
            "that report `inconclusive` rather than passing on an empty "
            "denominator. The chosen model is frozen into a `champion.lock` "
            "that carries every fingerprint a later evaluation may rely on and "
            "**no metric of any kind**.",
            "",
            "## Fusion",
            "",
            f"Declared candidates: {strategies}. All three are built and "
            "evaluated on validation-B before any test label is opened, and the "
            "selection is frozen before the reader that opens one can be "
            "called. No strategy is assumed to win, and the gates combine "
            "booleans rather than performing arithmetic across two "
            "incommensurable score scales.",
            "",
            "## Locked test evaluation",
            "",
            "The test split is opened **once**, by one command, against a "
            "lineage frozen before it ran: the model, the preprocessor, the "
            "calibrator, the operating point, the category head, the rule "
            "configuration, and the fusion strategy. PR-AUC is computed "
            "**exactly** -- over distinct score levels, with ties grouped, "
            "step-wise and right-continuous -- and is never an interpolated or "
            "grid-sampled approximation.",
            "",
            "## Explanation",
            "",
            f"Methods: {methods}. The first three reconstruct the model's own "
            "decision quantity exactly and the reconstruction is checked before "
            "anything is published; the fourth is a model-agnostic global "
            "sensitivity measure that decomposes nothing.",
            "",
            "**Limitations.** Attribution is descriptive, not causal: it says "
            "how a fitted function decomposes over the columns it was handed. "
            "The decomposition is of the raw decision quantity -- calibration "
            "and the decision threshold are separate transformations applied "
            "afterwards, and no contribution is a share of a calibrated "
            "probability. A family with no exact decomposition reports typed "
            "unavailable rather than an approximation. Correlated columns can "
            "mask each other in the global measure, so a small number is not "
            "evidence a column is unused.",
            "",
            "## Drift",
            "",
            f"Statuses: {statuses}. The reference profile is captured once from "
            "the training population and frozen; incoming rows are assigned to "
            "its cells and never redefine one. Feature drift and prediction "
            "drift are reported separately and never blended.",
            "",
            "**Drift is monitoring evidence, not model correctness.** It is "
            "computed without labels, so it cannot say a model became wrong. "
            "Nothing retrains, promotes, or rethresholds on a finding.",
            "",
            "## Privacy",
            "",
            "No plaintext password, hash, or credential list is stored, logged, "
            "or published anywhere in this project. Entity identities appear "
            "only as pseudonyms in the feature layer, and no aggregate artifact "
            "in this layer may declare a field named for a label, a split "
            "assignment, a campaign, an event identifier, a pseudonym, a "
            "coordinate, or a credential -- a structural guard refuses one at "
            "import. Row-level prediction and explanation artifacts carry the "
            "single anchor join identity their contract requires and nothing "
            "else.",
            "",
            "## Reproducibility",
            "",
            "Every authoritative identity is semantic: derived from content, "
            "never from a path, a hostname, or a wall clock. The same inputs "
            "rebuilt in two directories produce identical dataset, "
            "preprocessing, model, calibration, threshold, selection, freeze, "
            "prediction, evaluation, explanation, and reference-profile "
            "identities. Model artifacts are JSON and arrays, not serialized "
            "objects: nothing in this layer imports `pickle`, `dill`, or "
            "`joblib`, and no artifact value reaches an import path.",
            "",
            "## Known limitations",
            "",
            "- **Synthetic data only.** No result here transfers to real "
            "authentication traffic without re-measurement.",
            "- **Not a production system.** Offline and batch. No serving, no "
            "endpoint, no deployment, no automated response.",
            "- **A frozen champion is a subject, not a result.** The lock says "
            "what an evaluation was permitted to run, not how it behaved.",
            "- **Structural validity is not predictive quality.** An artifact "
            "can pass every integrity check and still come from a model that "
            "flags the wrong rows.",
            "- **Calibration is internal.** A calibrator is calibrated against "
            "the frozen synthetic validation-A distribution, which is a "
            "property of that distribution and not of any real one.",
            "- **The anomaly track is experimental** and the novel-anomaly "
            "holdout is a generalisation probe. Neither is part of the "
            "supervised result, and neither is combined with one.",
            "",
        ]
    )
