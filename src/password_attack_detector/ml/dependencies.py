"""Dependency policy and resolved-version recording for the ML layer.

scikit-learn is the **only** new direct runtime dependency Phase 5 adds.  Its
range is bounded on both sides:

    scikit-learn>=1.9.0,<1.10

An open upper bound would let an unreviewed minor-series upgrade change an
estimator's internal layout -- and therefore a serializer contract -- without
anything failing.  The bound is the review gate.

``scipy``, ``joblib``, ``threadpoolctl``, and ``narwhals`` arrive transitively
through scikit-learn.  Project source **never imports them directly**: they are
not declared dependencies, so importing one would create an undeclared coupling
that a future scikit-learn release could break without warning.
:data:`FORBIDDEN_DIRECT_IMPORTS` names them and a test walks every ``ml``
module's syntax tree to enforce it.

The resolved versions are recorded rather than assumed.  Every model manifest
and every experiment-ledger record carries the exact versions the fit ran
under, because "scikit-learn 1.9" is not enough information to reproduce a
model and "whatever was installed" is not information at all.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Final

from password_attack_detector.ml.schemas import DependencyRequirement, version_tuple

__all__ = [
    "FORBIDDEN_DIRECT_IMPORTS",
    "ML_DEPENDENCY_REQUIREMENTS",
    "PROHIBITED_ML_DISTRIBUTIONS",
    "RECORDED_DISTRIBUTIONS",
    "SKLEARN_BELOW_VERSION",
    "SKLEARN_MINIMUM_VERSION",
    "SKLEARN_REQUIREMENT",
    "collect_dependency_versions",
    "installed_version",
    "sklearn_compatible",
]

#: Inclusive lower bound of the reviewed scikit-learn range.
SKLEARN_MINIMUM_VERSION: Final = "1.9.0"

#: Exclusive upper bound.  Deliberately a minor-series boundary: scikit-learn
#: reorganises estimator internals between minor releases, and this layer reads
#: estimator attributes to serialise a model canonically.
SKLEARN_BELOW_VERSION: Final = "1.10"

SKLEARN_REQUIREMENT: Final = DependencyRequirement(
    distribution="scikit-learn",
    minimum_version=SKLEARN_MINIMUM_VERSION,
    below_version=SKLEARN_BELOW_VERSION,
    reason=(
        "Model serializers read documented estimator attributes whose layout "
        "is stable within a minor series. A wider range would let an "
        "unreviewed upgrade change a serializer contract silently."
    ),
)

#: Every bounded requirement this layer declares.  One entry today; the tuple
#: exists so a second never has to change a caller's shape.
ML_DEPENDENCY_REQUIREMENTS: Final[tuple[DependencyRequirement, ...]] = (
    SKLEARN_REQUIREMENT,
)

#: Distributions whose exact resolved version is recorded in every model
#: manifest and experiment record.  ``python`` is first because a minor
#: interpreter change can alter results even with every wheel pinned.
#:
#: ``narwhals`` is included because scikit-learn 1.9 pulls it in: an
#: undocumented transitive is exactly the thing a reproduction attempt three
#: months later needs written down.
RECORDED_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "python",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "joblib",
    "threadpoolctl",
    "narwhals",
)

#: Transitive distributions project source must never import directly.
#:
#: Each arrives only because scikit-learn requires it. Importing one would
#: create a dependency this project has not declared, reviewed, or bounded.
FORBIDDEN_DIRECT_IMPORTS: Final[frozenset[str]] = frozenset(
    {"scipy", "joblib", "threadpoolctl", "narwhals"}
)

#: Machine-learning frameworks Phase 5 deliberately does not add.
#:
#: Each was considered and rejected: a training framework, a gradient-boosting
#: library, an interchange format, a tuner, a resampler, an explainer, or a
#: plotting stack.  A test asserts none of them is installed as a project
#: dependency, so adding one is a visible decision rather than a silent drift.
PROHIBITED_ML_DISTRIBUTIONS: Final[frozenset[str]] = frozenset(
    {
        "mlflow",
        "xgboost",
        "lightgbm",
        "catboost",
        "onnx",
        "onnxruntime",
        "skl2onnx",
        "skops",
        "optuna",
        "imbalanced-learn",
        "shap",
        "matplotlib",
        "seaborn",
        "torch",
        "tensorflow",
        "keras",
        "dvc",
        "wandb",
    }
)


def installed_version(distribution: str) -> str | None:
    """Return the installed version of *distribution*, or ``None`` if absent.

    ``python`` is special-cased to the running interpreter's release, which is
    not a distribution but is the single most load-bearing version there is.
    """
    if distribution == "python":
        return ".".join(str(part) for part in sys.version_info[:3])
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_dependency_versions() -> dict[str, str]:
    """Return the resolved version of every recorded distribution.

    Keys are :data:`RECORDED_DISTRIBUTIONS` in declaration order.  A
    distribution that is not installed is omitted rather than recorded as an
    empty string, so a manifest never claims a version it did not observe.

    The result is a plain ``dict`` because that is what a manifest and a ledger
    record both need; ordering is insertion order and therefore stable.
    """
    resolved: dict[str, str] = {}
    for distribution in RECORDED_DISTRIBUTIONS:
        found = installed_version(distribution)
        if found is not None:
            resolved[distribution] = found
    return resolved


def sklearn_compatible(installed: str | None = None) -> bool:
    """Return whether the installed scikit-learn lies inside the reviewed range.

    Called before a serialised model is restored: a model exported under one
    minor series must not be silently reconstructed under another, because the
    estimator attributes the serializer recorded may no longer mean the same
    thing.

    Args:
        installed: Version to test.  Omit it (or pass ``None``) to test the
            release currently installed.  When scikit-learn is not installed at
            all there is nothing to test and the answer is ``False``, as it is
            for a version string that cannot be parsed.
    """
    resolved = installed if installed is not None else installed_version("scikit-learn")
    if resolved is None:
        return False
    try:
        version_tuple(resolved)
    except ValueError:
        return False
    return SKLEARN_REQUIREMENT.contains(resolved)
