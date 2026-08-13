"""The closed model registry: every family, named in code, discovered never.

:data:`MODEL_IMPLEMENTATIONS` is written out by hand.  There is no entry-point
scan, no module walk, no ``importlib`` call, and no path where a string from an
artifact selects a class.  A model directory names a family; the family is
looked up in this mapping; a name that is not a key fails.  That is the whole
dispatch, and it is deliberately boring -- dynamic dispatch on artifact content
is how a data file becomes code execution.

The registry is checked against the Milestone 1 catalog in both directions by
:func:`assert_registry_matches_catalog`, called at import.  A catalog entry with
no implementation and an implementation with no catalog entry are both failures,
as is a disagreement about a serializer identifier, an inference adapter
identifier, the set of supported tasks, or **champion eligibility**.  The
catalog is documentation that generates a document; this makes it documentation
that cannot drift.

Eligibility is the agreement worth naming.  ``M-000`` is fully implemented,
fully publishable, and permanently *not* champion-eligible: it is the reference
every candidate is measured against, and a model cannot beat itself.  Stating
that in the catalog and again on the adapter, and requiring the two to agree,
is what stops either being edited on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.catalog import MODEL_CATALOG, ModelSpec
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.models.anomaly import IsolationForestAdapter
from password_attack_detector.ml.models.base import (
    FittedModel,
    ModelAdapter,
    TrainingBatch,
    score_semantics_for,
)
from password_attack_detector.ml.models.baseline import (
    PriorBaselineAdapter,
    SingleFeatureThresholdAdapter,
)
from password_attack_detector.ml.models.boosting import (
    CompatibilityProbe,
    HistogramBoostingAdapter,
    probe_compatibility,
)
from password_attack_detector.ml.models.forest import RandomForestAdapter
from password_attack_detector.ml.models.linear import LogisticRegressionAdapter

__all__ = [
    "CHAMPION_ELIGIBLE_MODEL_IDS",
    "MODEL_IMPLEMENTATIONS",
    "PUBLISHABLE_FAMILIES",
    "REFERENCE_BASELINE_MODEL_IDS",
    "CompatibilityProbe",
    "FittedModel",
    "HistogramBoostingAdapter",
    "IsolationForestAdapter",
    "LogisticRegressionAdapter",
    "ModelAdapter",
    "PriorBaselineAdapter",
    "RandomForestAdapter",
    "SingleFeatureThresholdAdapter",
    "TrainingBatch",
    "adapter_class_for",
    "assert_registry_matches_catalog",
    "catalog_spec_for",
    "probe_compatibility",
    "score_semantics_for",
]

#: Every implemented family, by enum member. Closed, ordered, and hand-written.
#:
#: The value is the adapter *class*, never an instance and never an import path:
#: constructing one takes hyperparameters, and a shared instance would make two
#: concurrently fitted models share configuration.
MODEL_IMPLEMENTATIONS: Final[Mapping[ModelFamily, type]] = {
    ModelFamily.PRIOR_BASELINE: PriorBaselineAdapter,
    ModelFamily.SINGLE_FEATURE_THRESHOLD: SingleFeatureThresholdAdapter,
    ModelFamily.LOGISTIC_REGRESSION: LogisticRegressionAdapter,
    ModelFamily.RANDOM_FOREST: RandomForestAdapter,
    ModelFamily.HISTOGRAM_GRADIENT_BOOSTING: HistogramBoostingAdapter,
    ModelFamily.ISOLATION_FOREST: IsolationForestAdapter,
}

#: Families whose fitted models may be written as an artifact.
#:
#: Histogram boosting is absent: its serializer would rest on undocumented
#: estimator internals, so it may be fitted and compared in process but never
#: stored. See :mod:`password_attack_detector.ml.models.boosting`.
PUBLISHABLE_FAMILIES: Final[frozenset[ModelFamily]] = frozenset(
    family
    for family, adapter in MODEL_IMPLEMENTATIONS.items()
    if bool(getattr(adapter, "publishable", False))
)


def adapter_class_for(family: ModelFamily | str) -> type:
    """Return the adapter class for *family*, or raise.

    Accepts the enum member or its string value, because a family arriving from
    a stored artifact is a string. The lookup is a dictionary access against a
    closed mapping -- an unrecognised value simply has no entry.
    """
    if isinstance(family, str):
        try:
            resolved = ModelFamily(family)
        except ValueError:
            raise ModelTrainingError(
                f"model family {family!r} is not one this build implements; "
                f"dispatch is a closed registry, so an unrecognised family is "
                f"refused rather than resolved"
            ) from None
    else:
        resolved = family
    adapter = MODEL_IMPLEMENTATIONS.get(resolved)
    if adapter is None:
        raise ModelTrainingError(
            f"model family {str(resolved)!r} is declared but not implemented"
        )
    return adapter


def catalog_spec_for(family: ModelFamily) -> ModelSpec:
    """Return the catalog entry declaring *family*, or raise."""
    for spec in MODEL_CATALOG.specs:
        if spec.family is family:
            return spec
    raise ModelTrainingError(f"the model catalog declares no family {str(family)!r}")


def assert_registry_matches_catalog() -> None:
    """Fail at import if the registry and the catalog have drifted apart.

    Each agreement has its own failure mode:

    * a catalog family with no implementation would advertise a model nobody
      can fit;
    * an implementation with no catalog entry would ship a model nobody
      reviewed;
    * a serializer or inference-adapter identifier mismatch would make an
      artifact claim a reader it was not written by;
    * a supported-task mismatch would let a family be fitted for a task its
      declaration says it cannot do;
    * a catalog model identifier mismatch would break the manifest's link back
      to the reviewed entry;
    * a champion-eligibility or reference-baseline mismatch would let an
      adapter mark its own fitted models promotable while the reviewed entry
      says they are not -- which is exactly how a reference baseline becomes an
      accidental fallback champion.
    """
    declared = {spec.family for spec in MODEL_CATALOG.specs}
    implemented = set(MODEL_IMPLEMENTATIONS)

    unimplemented = sorted(str(item) for item in declared - implemented)
    if unimplemented:
        raise ModelTrainingError(
            f"the model catalog declares family(ies) {unimplemented} with no "
            f"implementation"
        )
    undeclared = sorted(str(item) for item in implemented - declared)
    if undeclared:
        raise ModelTrainingError(
            f"family(ies) {undeclared} are implemented but not declared in the "
            f"model catalog; an unreviewed model family cannot ship"
        )

    for spec in MODEL_CATALOG.specs:
        adapter = MODEL_IMPLEMENTATIONS[spec.family]
        for attribute, expected in (
            ("catalog_model_id", spec.model_id),
            ("serializer_id", spec.serializer_id),
            ("inference_adapter_id", spec.inference_adapter_id),
            # Eligibility is checked here as well as declared in the catalog, so
            # an adapter cannot mark its own fitted models promotable while the
            # reviewed entry says otherwise. M-000 is the case that makes this
            # load-bearing: it is fully implemented and permanently ineligible,
            # and the two facts have to be stated in one place each and agree.
            ("champion_eligible", spec.champion_eligible),
            ("reference_baseline", spec.reference_baseline),
        ):
            actual = getattr(adapter, attribute, None)
            if actual != expected:
                raise ModelTrainingError(
                    f"{spec.model_id} declares {attribute}={expected!r} but its "
                    f"implementation reports {actual!r}"
                )
        supported: tuple[MLTask, ...] = getattr(adapter, "supported_tasks", ())
        if set(supported) - set(spec.supported_tasks):
            raise ModelTrainingError(
                f"{spec.model_id} implements task(s) its catalog entry does not declare"
            )
        if not supported:
            raise ModelTrainingError(f"{spec.model_id} implements no task at all")
        if spec.reference_baseline and spec.champion_eligible:
            raise ModelTrainingError(
                f"{spec.model_id} is the reference baseline and cannot be "
                f"champion eligible; a candidate qualifies by beating it"
            )


#: The supervised families a champion may be selected from.
#:
#: Derived from the catalog rather than written down, so it cannot drift from
#: the reviewed entries -- and a test pins the resulting set, so a family that
#: quietly became eligible fails rather than joining the contest.
#:
#: Absent, each for its own reason: **M-000** is the reference every candidate
#: is measured against, **M-021** is gated on an undocumented estimator
#: interface, and **M-030** is an unsupervised anomaly model with no supervised
#: class to be champion of.
CHAMPION_ELIGIBLE_MODEL_IDS: Final[frozenset[str]] = frozenset(
    spec.model_id for spec in MODEL_CATALOG.specs if spec.champion_eligible
)

#: Families that are fully implemented and permanently unpromotable.
REFERENCE_BASELINE_MODEL_IDS: Final[frozenset[str]] = frozenset(
    spec.model_id for spec in MODEL_CATALOG.specs if spec.reference_baseline
)


assert_registry_matches_catalog()
