"""Tests for the ML layer's dependency policy.

These are the tests that keep Milestone 1's dependency audit true over time.
They assert three separate things:

* **What was added.** scikit-learn is the only new direct runtime dependency,
  its range is bounded on both sides, and the installed release lies inside it.
* **What was not added.** No training framework, gradient-boosting library,
  interchange format, tuner, resampler, explainer, or plotting stack.
* **What is not imported.** The transitive arrivals -- scipy, joblib,
  threadpoolctl, narwhals -- are never imported by project source, checked by
  walking every module's syntax tree rather than by grepping.

The estimator-attribute checks fit deliberately tiny models. They exist to
confirm the attributes the serializers will read actually exist on the resolved
release, so a future upgrade fails here with a clear message rather than deep
inside Milestone 4.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.ml.dependencies import (
    FORBIDDEN_DIRECT_IMPORTS,
    ML_DEPENDENCY_REQUIREMENTS,
    PROHIBITED_ML_DISTRIBUTIONS,
    RECORDED_DISTRIBUTIONS,
    SKLEARN_BELOW_VERSION,
    SKLEARN_MINIMUM_VERSION,
    SKLEARN_REQUIREMENT,
    collect_dependency_versions,
    installed_version,
    sklearn_compatible,
)
from password_attack_detector.ml.schemas import version_tuple

#: Runtime dependencies Phase 5 inherited from Phases 1-4.
PHASE_FOUR_DEPENDENCIES = {
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "pydantic-settings",
    "pyyaml",
    "rich",
    "structlog",
    "typer",
}


def _repo_root() -> Path:
    """Return the repository root, located from this test file."""
    return Path(__file__).resolve().parents[3]


def _pyproject() -> dict[str, object]:
    """Return the parsed project metadata."""
    return tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))


def _declared_dependencies() -> dict[str, str]:
    """Return declared runtime dependencies as a name-to-specifier mapping."""
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    raw = project["dependencies"]
    assert isinstance(raw, list)
    declared: dict[str, str] = {}
    for entry in raw:
        text = str(entry)
        name = text
        for separator in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
            index = text.find(separator)
            if index != -1:
                name = min(name, text[:index], key=len)
        declared[name.strip().lower()] = text
    return declared


# ---------------------------------------------------------------------------
# What was added
# ---------------------------------------------------------------------------


def test_scikit_learn_is_the_only_new_direct_dependency() -> None:
    """Phase 5 adds exactly one runtime dependency, and this is it."""
    declared = set(_declared_dependencies())
    assert declared - PHASE_FOUR_DEPENDENCIES == {"scikit-learn"}


def test_the_declared_range_is_bounded_on_both_sides() -> None:
    """An open upper bound would let a minor upgrade change a serializer.

    The whole point of the bound is that a scikit-learn minor release may
    reorganise estimator internals. Without an upper bound, that would happen
    silently on the next resolve.
    """
    specifier = _declared_dependencies()["scikit-learn"]
    assert ">=" in specifier, specifier
    assert "<" in specifier.replace("<=", ""), specifier
    assert specifier == (
        f"scikit-learn>={SKLEARN_MINIMUM_VERSION},<{SKLEARN_BELOW_VERSION}"
    )


def test_the_declared_range_matches_the_module_constant() -> None:
    """pyproject.toml and the code must not drift apart."""
    assert SKLEARN_REQUIREMENT.specifier == _declared_dependencies()["scikit-learn"]


def test_the_upper_bound_is_a_minor_series_boundary() -> None:
    """A patch-level bound would churn; a major-level bound would be useless."""
    minimum = version_tuple(SKLEARN_MINIMUM_VERSION)
    below = version_tuple(SKLEARN_BELOW_VERSION)
    assert below[0] == minimum[0]
    assert below[1] == minimum[1] + 1


def test_the_installed_release_lies_inside_the_declared_range() -> None:
    """The audit's central claim: what is installed is what was reviewed."""
    resolved = installed_version("scikit-learn")
    assert resolved is not None
    assert SKLEARN_REQUIREMENT.contains(resolved), resolved
    assert sklearn_compatible()


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("1.9.0", True),
        ("1.9.7", True),
        ("1.10.0", False),
        ("1.8.9", False),
        ("2.0.0", False),
        ("not-a-version", False),
    ],
)
def test_compatibility_check_respects_the_bound(candidate: str, expected: bool) -> None:
    """The check is half-open and refuses anything it cannot parse."""
    assert sklearn_compatible(candidate) is expected


def test_compatibility_check_defaults_to_the_installed_release() -> None:
    """Omitting the argument tests what is actually installed."""
    assert sklearn_compatible(None) is sklearn_compatible(
        installed_version("scikit-learn")
    )


def test_an_absent_scikit_learn_is_never_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing installed there is nothing to reconstruct a model with."""
    monkeypatch.setattr(
        "password_attack_detector.ml.dependencies.installed_version",
        lambda distribution: None,
    )
    assert sklearn_compatible() is False


def test_every_recorded_distribution_resolves_to_a_version() -> None:
    """A manifest must record what actually ran, not a placeholder."""
    resolved = collect_dependency_versions()
    assert set(resolved) == set(RECORDED_DISTRIBUTIONS)
    for distribution, value in resolved.items():
        assert value, distribution
        assert version_tuple(value)


def test_the_transitive_arrivals_are_recorded() -> None:
    """Reproducing a fit needs the transitive versions written down."""
    resolved = collect_dependency_versions()
    for distribution in ("scikit-learn", "scipy", "joblib", "threadpoolctl"):
        assert distribution in resolved


def test_the_python_version_is_recorded_and_is_the_pinned_series() -> None:
    """A minor interpreter change can move results even with wheels pinned."""
    resolved = collect_dependency_versions()
    assert version_tuple(resolved["python"])[:2] == (3, 12)


def test_an_absent_distribution_is_omitted_rather_than_invented() -> None:
    """A manifest must never claim a version it did not observe."""
    assert installed_version("a-distribution-that-does-not-exist") is None


def test_one_bounded_requirement_is_declared() -> None:
    """Today there is exactly one; the tuple shape allows a second."""
    assert ML_DEPENDENCY_REQUIREMENTS == (SKLEARN_REQUIREMENT,)


# ---------------------------------------------------------------------------
# What was not added
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", sorted(PROHIBITED_ML_DISTRIBUTIONS))
def test_no_prohibited_ml_framework_is_declared(distribution: str) -> None:
    """Adding one of these must be a visible decision, never silent drift."""
    assert distribution not in _declared_dependencies()


@pytest.mark.parametrize("distribution", sorted(PROHIBITED_ML_DISTRIBUTIONS))
def test_no_prohibited_ml_framework_is_installed(distribution: str) -> None:
    """A framework in the environment but not the manifest is worse, not better."""
    assert installed_version(distribution) is None


def test_the_lockfile_carries_no_prohibited_framework() -> None:
    """The lock is the real dependency set; check it directly."""
    lock = (_repo_root() / "uv.lock").read_text(encoding="utf-8")
    locked = {
        line.split('"')[1] for line in lock.splitlines() if line.startswith("name = ")
    }
    assert not locked & PROHIBITED_ML_DISTRIBUTIONS


def test_the_lockfile_resolves_scikit_learn_inside_the_declared_range() -> None:
    """The locked release is the reviewed one."""
    lock = (_repo_root() / "uv.lock").read_text(encoding="utf-8")
    lines = lock.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == 'name = "scikit-learn"':
            locked = lines[index + 1].split('"')[1]
            assert SKLEARN_REQUIREMENT.contains(locked), locked
            return
    pytest.fail("scikit-learn is absent from uv.lock")


# ---------------------------------------------------------------------------
# What is not imported
# ---------------------------------------------------------------------------


def _ml_modules() -> list[Path]:
    """Return every source file in the ML package."""
    package = _repo_root() / "src" / "password_attack_detector" / "ml"
    modules = sorted(package.rglob("*.py"))
    assert modules, "the ML package should contain source modules"
    return modules


def _imported_roots(module: Path) -> set[str]:
    """Return the top-level package name of every import in *module*."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("module", _ml_modules(), ids=lambda path: path.name)
def test_no_ml_module_imports_a_transitive_dependency(module: Path) -> None:
    """scipy, joblib, threadpoolctl, and narwhals arrive only via scikit-learn.

    Importing one directly would create a dependency this project has not
    declared, reviewed, or bounded -- and which a future scikit-learn release
    could drop without warning.
    """
    offending = sorted(_imported_roots(module) & FORBIDDEN_DIRECT_IMPORTS)
    assert not offending, f"{module.name} imports {offending}"


def test_the_whole_source_tree_avoids_the_transitive_dependencies() -> None:
    """The rule is project-wide, not merely a convention inside ``ml``."""
    source = _repo_root() / "src" / "password_attack_detector"
    for module in sorted(source.rglob("*.py")):
        offending = sorted(_imported_roots(module) & FORBIDDEN_DIRECT_IMPORTS)
        assert not offending, f"{module.name} imports {offending}"


def test_no_prohibited_framework_is_imported_anywhere_in_source() -> None:
    """An import of an undeclared framework would fail only at run time."""
    source = _repo_root() / "src" / "password_attack_detector"
    prohibited_roots = {
        distribution.replace("-", "_") for distribution in PROHIBITED_ML_DISTRIBUTIONS
    }
    for module in sorted(source.rglob("*.py")):
        offending = sorted(_imported_roots(module) & prohibited_roots)
        assert not offending, f"{module.name} imports {offending}"


# ---------------------------------------------------------------------------
# Estimator attribute compatibility
# ---------------------------------------------------------------------------


def _tiny_binary_problem() -> tuple[Any, Any]:
    """Return a deliberately tiny separable problem for attribute probing."""
    import numpy as np

    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 3))
    labels = (features[:, 0] > 0).astype(int)
    return features, labels


def test_logistic_regression_exposes_the_attributes_its_serializer_reads() -> None:
    """``coef_``, ``intercept_``, and ``classes_`` are documented attributes."""
    from sklearn.linear_model import LogisticRegression

    features, labels = _tiny_binary_problem()
    model = LogisticRegression(max_iter=200, random_state=0).fit(features, labels)
    for attribute in ("coef_", "intercept_", "classes_", "n_features_in_"):
        assert hasattr(model, attribute), attribute
    assert model.coef_.shape == (1, 3)
    assert model.intercept_.shape == (1,)


def test_random_forest_exposes_the_tree_arrays_its_serializer_reads() -> None:
    """The per-tree arrays are the documented way to read a fitted tree."""
    from sklearn.ensemble import RandomForestClassifier

    features, labels = _tiny_binary_problem()
    model = RandomForestClassifier(
        n_estimators=3, max_depth=3, random_state=0, n_jobs=1
    ).fit(features, labels)
    assert len(model.estimators_) == 3
    tree = model.estimators_[0].tree_
    for attribute in (
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "value",
    ):
        assert hasattr(tree, attribute), attribute
    assert model.n_outputs_ == 1
    assert list(model.classes_) == [0, 1]


def test_isotonic_regression_exposes_its_knots() -> None:
    """Calibration is serialised from the fitted knots, not from a pickle."""
    from sklearn.isotonic import IsotonicRegression

    features, labels = _tiny_binary_problem()
    model = IsotonicRegression(out_of_bounds="clip").fit(
        features[:, 0], labels.astype(float)
    )
    assert model.X_thresholds_.shape == model.y_thresholds_.shape
    for attribute in ("X_min_", "X_max_", "increasing_"):
        assert hasattr(model, attribute), attribute


def test_isolation_forest_exposes_the_attributes_its_adapter_reads() -> None:
    """The anomaly adapter needs the trees, their features, and the offset."""
    from sklearn.ensemble import IsolationForest

    features, _ = _tiny_binary_problem()
    model = IsolationForest(n_estimators=3, random_state=0, n_jobs=1).fit(features)
    for attribute in (
        "estimators_",
        "estimators_features_",
        "max_samples_",
        "offset_",
        "n_features_in_",
    ):
        assert hasattr(model, attribute), attribute
    assert len(model.estimators_) == len(model.estimators_features_)


def test_histogram_gradient_boosting_still_needs_private_attributes() -> None:
    """The reason M-021 is gated, asserted rather than assumed.

    If a future release exposes the fitted trees publicly, this test fails and
    the gate can be revisited deliberately. If the private layout changes
    instead, the serializer work in Milestone 4 finds out here first.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    features, labels = _tiny_binary_problem()
    model = HistGradientBoostingClassifier(
        max_iter=2, random_state=0, early_stopping=False
    ).fit(features, labels)

    assert hasattr(model, "_predictors")
    assert hasattr(model, "_baseline_prediction")
    nodes = model._predictors[0][0].nodes
    assert nodes.dtype.names is not None
    for field in ("value", "feature_idx", "num_threshold", "left", "right", "is_leaf"):
        assert field in nodes.dtype.names, field


def test_the_catalog_declares_only_attributes_that_actually_exist() -> None:
    """Every declared public attribute is present on the resolved release.

    The catalog's ``public_estimator_attributes`` is a promise that Milestone 4
    will be able to read them. This checks the promise against reality now,
    while the cost of finding out is a one-line failure.
    """
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        IsolationForest,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    from password_attack_detector.ml.catalog import MODEL_CATALOG
    from password_attack_detector.ml.enums import ModelFamily

    features, labels = _tiny_binary_problem()
    fitted: dict[ModelFamily, object] = {
        ModelFamily.LOGISTIC_REGRESSION: LogisticRegression(
            max_iter=200, random_state=0
        ).fit(features, labels),
        ModelFamily.RANDOM_FOREST: RandomForestClassifier(
            n_estimators=2, max_depth=2, random_state=0, n_jobs=1
        ).fit(features, labels),
        ModelFamily.HISTOGRAM_GRADIENT_BOOSTING: HistGradientBoostingClassifier(
            max_iter=2, random_state=0, early_stopping=False
        ).fit(features, labels),
        ModelFamily.ISOLATION_FOREST: IsolationForest(
            n_estimators=2, random_state=0, n_jobs=1
        ).fit(features),
    }
    for family, estimator in fitted.items():
        spec = MODEL_CATALOG.for_family(family)
        for attribute in (
            *spec.public_estimator_attributes,
            *spec.private_estimator_attributes,
        ):
            assert hasattr(estimator, attribute), f"{family}.{attribute}"
