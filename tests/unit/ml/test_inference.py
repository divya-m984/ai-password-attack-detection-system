"""Loading a model artifact: fail closed, verify first, construct last.

The order matters as much as the checks. Every test here breaks one link and
asserts that loading fails *before* anything is constructed -- and the registry
tests assert that the only path from a string in a file to executable code is a
dictionary lookup against a hand-written mapping.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ManifestVerificationError,
    ModelNotReadyError,
)
from password_attack_detector.ml.catalog import MODEL_CATALOG
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.inference import InferenceModel, ModelCompatibility
from password_attack_detector.ml.models import (
    MODEL_IMPLEMENTATIONS,
    PUBLISHABLE_FAMILIES,
    LogisticRegressionAdapter,
    RandomForestAdapter,
    SingleFeatureThresholdAdapter,
    adapter_class_for,
    assert_registry_matches_catalog,
)
from password_attack_detector.ml.serialization import (
    ARRAYS_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE,
)
from tests.ml.models import prepare, publish


@pytest.fixture
def batch() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=140)


@pytest.fixture
def published(batch: Any, tmp_path: Path) -> Path:
    """Return a freshly published logistic model."""
    fitted = LogisticRegressionAdapter().fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    return publish(tmp_path / "model", fitted, batch.preprocessor)


def rewrite_manifest(directory: Path, **changes: Any) -> None:
    """Apply *changes* to the manifest payload and rewrite it."""
    payload = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    payload.update(changes)
    (directory / MANIFEST_FILE).write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# The registry is closed
# ---------------------------------------------------------------------------


def test_the_registry_and_the_catalog_agree_one_to_one() -> None:
    """Checked at import; asserted again here so the check itself is covered."""
    assert_registry_matches_catalog()
    assert {spec.family for spec in MODEL_CATALOG.specs} == set(MODEL_IMPLEMENTATIONS)


@pytest.mark.parametrize("spec", MODEL_CATALOG.specs, ids=lambda s: s.model_id)
def test_every_catalog_entry_has_exactly_one_implementation(spec: Any) -> None:
    """Identifiers agree in both directions, per family."""
    adapter: Any = MODEL_IMPLEMENTATIONS[spec.family]
    assert adapter.catalog_model_id == spec.model_id
    assert adapter.serializer_id == spec.serializer_id
    assert adapter.inference_adapter_id == spec.inference_adapter_id
    assert set(adapter.supported_tasks) <= set(spec.supported_tasks)


def test_every_serializer_id_is_unique() -> None:
    """Two families sharing a serializer identifier would make artifacts ambiguous."""
    adapters: list[Any] = list(MODEL_IMPLEMENTATIONS.values())
    identifiers = [adapter.serializer_id for adapter in adapters]
    assert len(set(identifiers)) == len(identifiers)


def test_every_inference_adapter_id_is_unique() -> None:
    """Same reasoning, for the reader half of the contract."""
    adapters: list[Any] = list(MODEL_IMPLEMENTATIONS.values())
    identifiers = [adapter.inference_adapter_id for adapter in adapters]
    assert len(set(identifiers)) == len(identifiers)


def test_the_unpublishable_family_is_recorded_as_such() -> None:
    """A gated family is registered with an explicit availability state."""
    assert ModelFamily.HISTOGRAM_GRADIENT_BOOSTING in MODEL_IMPLEMENTATIONS
    assert ModelFamily.HISTOGRAM_GRADIENT_BOOSTING not in PUBLISHABLE_FAMILIES


def test_an_unrecognised_family_selects_nothing() -> None:
    """Dispatch is a dictionary lookup against a closed mapping."""
    from password_attack_detector.exceptions import ModelTrainingError

    with pytest.raises(ModelTrainingError, match="closed registry"):
        adapter_class_for("something_nobody_implemented")


def test_the_registry_module_performs_no_dynamic_import() -> None:
    """No ``importlib``, no ``__import__``, no ``eval``, no ``exec``.

    Walked as a syntax tree rather than grepped, so a name appearing in a
    docstring -- and this module's docstring names all four -- is not mistaken
    for a use of it.
    """
    from password_attack_detector.ml import inference, models

    for module in (models, inference):
        source = module.__file__
        assert source is not None
        tree = ast.parse(Path(source).read_text(encoding="utf-8"))
        called: set[str] = set()
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "eval" not in called, module.__name__
        assert "exec" not in called, module.__name__
        assert "__import__" not in called, module.__name__
        assert "import_module" not in called, module.__name__
        assert "importlib" not in imported, module.__name__
        assert "pickle" not in imported, module.__name__


def test_no_model_module_imports_pickle_or_joblib() -> None:
    """Neither is authoritative, so neither is imported at all."""
    root = Path(__file__).resolve().parents[3] / "src" / "password_attack_detector"
    for module in sorted((root / "ml").rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert "pickle" not in names, module.name
        assert "joblib" not in names, module.name
        assert "dill" not in names, module.name


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_valid_artifact_loads(published: Path) -> None:
    """And exposes identity, family, task, and score columns."""
    loaded = InferenceModel.load(published)
    assert loaded.family is ModelFamily.LOGISTIC_REGRESSION
    assert loaded.task is MLTask.BINARY_MALICIOUS
    assert loaded.score_columns == ("benign", "malicious")
    assert loaded.model_id


def test_a_loaded_model_scores_identically_to_the_fitted_one(
    batch: Any, published: Path
) -> None:
    """The round-trip the whole artifact format exists to make possible."""
    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    loaded = InferenceModel.load(published)
    assert loaded.score(
        batch.matrix.rows, batch.matrix.output_feature_names
    ) == adapter.score(fitted, batch.matrix.rows, batch.matrix.output_feature_names)


def test_a_loaded_model_can_transform_and_score(batch: Any, published: Path) -> None:
    """The stored preprocessor and the stored model travel together."""
    loaded = InferenceModel.load(published)
    scored = loaded.transform_and_score(batch.frame)
    assert len(scored) == len(batch.matrix.rows)


def test_a_supplied_contract_is_checked(batch: Any, published: Path) -> None:
    """Matching expectations load; a mismatched one does not."""
    InferenceModel.load(
        published,
        compatibility=ModelCompatibility(
            preprocessor_fingerprint=batch.preprocessor.fingerprint(),
            eligible_feature_list_fingerprint=(
                batch.preprocessor.eligible_feature_list_fingerprint
            ),
            transformed_feature_names=batch.matrix.output_feature_names,
        ),
    )
    with pytest.raises(ModelNotReadyError, match="preprocessor fingerprint"):
        InferenceModel.load(
            published,
            compatibility=ModelCompatibility(preprocessor_fingerprint="f" * 64),
        )


def test_a_mismatched_transformed_order_is_refused(batch: Any, published: Path) -> None:
    """The right columns in the wrong order are a different matrix."""
    swapped = (
        batch.matrix.output_feature_names[1],
        batch.matrix.output_feature_names[0],
        *batch.matrix.output_feature_names[2:],
    )
    with pytest.raises(ModelNotReadyError, match="different order"):
        InferenceModel.load(
            published,
            compatibility=ModelCompatibility(transformed_feature_names=swapped),
        )


def test_the_threshold_baseline_is_reconstructed_from_the_document(
    batch: Any, tmp_path: Path
) -> None:
    """Its bound column comes from the verified artifact, not from a caller."""
    adapter = SingleFeatureThresholdAdapter(feature="user_failure_rate")
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    directory = publish(tmp_path / "threshold", fitted, batch.preprocessor)
    loaded = InferenceModel.load(directory)
    assert loaded.adapter.feature == "user_failure_rate"


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    """Nothing is constructed for a path that is not there."""
    with pytest.raises(ArtifactNotFoundError):
        InferenceModel.load(tmp_path / "nothing")


@pytest.mark.parametrize("name", [MODEL_FILE, ARRAYS_FILE, PREPROCESSOR_FILE])
def test_a_tampered_file_is_refused(published: Path, name: str) -> None:
    """Integrity precedes interpretation."""
    path = published / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ManifestVerificationError, match="verification failed"):
        InferenceModel.load(published)


def test_an_unexpected_file_is_refused(published: Path) -> None:
    """The declared file set is closed."""
    (published / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="MODEL_FILE_UNEXPECTED"):
        InferenceModel.load(published)


def test_a_malformed_manifest_is_refused(published: Path) -> None:
    """Loading stops at the manifest, long before an adapter exists."""
    (published / MANIFEST_FILE).write_text("{broken", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="MANIFEST_INVALID"):
        InferenceModel.load(published)


def test_a_serializer_mismatch_is_refused(published: Path, tmp_path: Path) -> None:
    """A reader that cannot read this writer says so rather than guessing.

    Both the document and the manifest are edited so the pair still agree with
    each other; the failure is that no implementation claims that serializer.
    """
    for name, key in ((MODEL_FILE, "serializer_id"), (MANIFEST_FILE, "serializer_id")):
        payload = json.loads((published / name).read_text(encoding="utf-8"))
        payload[key] = "json_from_the_future_v9"
        (published / name).write_text(json.dumps(payload), encoding="utf-8")
    _resign(published)
    with pytest.raises(ModelNotReadyError, match="serializer"):
        InferenceModel.load(published)


def test_a_serializer_version_mismatch_is_refused(published: Path) -> None:
    """A version bump is a new contract, not a compatible one."""
    for name in (MODEL_FILE, MANIFEST_FILE):
        payload = json.loads((published / name).read_text(encoding="utf-8"))
        payload["serializer_version"] = 99
        (published / name).write_text(json.dumps(payload), encoding="utf-8")
    _resign(published)
    with pytest.raises(ModelNotReadyError, match="serializer version"):
        InferenceModel.load(published)


def test_an_unimplemented_family_is_refused(
    published: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family name selects between implementations or selects nothing."""
    from password_attack_detector.ml import models as registry

    reduced = {
        family: adapter
        for family, adapter in registry.MODEL_IMPLEMENTATIONS.items()
        if family is not ModelFamily.LOGISTIC_REGRESSION
    }
    monkeypatch.setattr(registry, "MODEL_IMPLEMENTATIONS", reduced)
    with pytest.raises(ModelNotReadyError, match="not implemented"):
        InferenceModel.load(published)


def test_an_out_of_range_recorded_runtime_is_refused(published: Path) -> None:
    """Estimator internals move between minor series."""
    rewrite_manifest(published, scikit_learn_version="1.7.0")
    _resign(published)
    with pytest.raises(ModelNotReadyError, match="outside the reviewed range"):
        InferenceModel.load(published)


def test_an_unpromotable_model_is_refused_when_a_champion_is_required(
    batch: Any, tmp_path: Path
) -> None:
    """Requiring promotability is a caller's choice, and it is honoured."""
    from password_attack_detector.ml.models import IsolationForestAdapter

    unlabelled = prepare(count=120, task=MLTask.ANOMALY)
    fitted = IsolationForestAdapter(n_estimators=8, max_samples=32).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    directory = publish(tmp_path / "anomaly", fitted, unlabelled.preprocessor)
    with pytest.raises(ModelNotReadyError, match="champion-eligible"):
        InferenceModel.load(directory, require_champion_eligible=True)


def test_a_substituted_preprocessor_is_refused(batch: Any, published: Path) -> None:
    """A model scored by a differently built matrix is not this model."""
    other = prepare(count=90, seed=99)
    (published / PREPROCESSOR_FILE).write_text(
        other.preprocessor.to_json(), encoding="utf-8"
    )
    _resign(published)
    with pytest.raises(ModelNotReadyError, match="not the one this model"):
        InferenceModel.load(published)


def test_verification_precedes_construction(
    published: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No adapter is constructed for an artifact that fails a check.

    Enforced by making every adapter constructor raise: the load must fail with
    a verification error rather than with the constructor's own.
    """
    from password_attack_detector.ml import models as registry

    class Exploding:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("an adapter was constructed before verification")

    monkeypatch.setattr(
        registry,
        "MODEL_IMPLEMENTATIONS",
        dict.fromkeys(registry.MODEL_IMPLEMENTATIONS, Exploding),
    )
    path = published / ARRAYS_FILE
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ManifestVerificationError):
        InferenceModel.load(published)


def test_every_publishable_family_round_trips(batch: Any, tmp_path: Path) -> None:
    """One test covering the whole set, so a new family cannot be forgotten."""
    unlabelled = prepare(count=120, task=MLTask.ANOMALY)
    cases: dict[ModelFamily, tuple[Any, MLTask, dict[str, Any]]] = {
        ModelFamily.PRIOR_BASELINE: (batch, MLTask.BINARY_MALICIOUS, {}),
        ModelFamily.SINGLE_FEATURE_THRESHOLD: (
            batch,
            MLTask.BINARY_MALICIOUS,
            {"feature": "user_failure_rate"},
        ),
        ModelFamily.LOGISTIC_REGRESSION: (batch, MLTask.BINARY_MALICIOUS, {}),
        ModelFamily.RANDOM_FOREST: (
            batch,
            MLTask.BINARY_MALICIOUS,
            {"n_estimators": 8, "max_depth": 4},
        ),
        ModelFamily.ISOLATION_FOREST: (
            unlabelled,
            MLTask.ANOMALY,
            {"n_estimators": 8, "max_samples": 32},
        ),
    }
    assert set(cases) == PUBLISHABLE_FAMILIES

    for family, (prepared, task, kwargs) in cases.items():
        adapter = adapter_class_for(family)(**kwargs)
        fitted = adapter.fit(prepared.batch, task=task)
        directory = publish(tmp_path / str(family), fitted, prepared.preprocessor)
        loaded = InferenceModel.load(directory)
        assert loaded.score(
            prepared.matrix.rows, prepared.matrix.output_feature_names
        ) == adapter.score(
            fitted, prepared.matrix.rows, prepared.matrix.output_feature_names
        ), str(family)


def test_a_model_loads_from_a_different_directory(batch: Any, tmp_path: Path) -> None:
    """Publishing the same model twice yields two directories that load alike."""
    adapter = RandomForestAdapter(n_estimators=8, max_depth=4)
    fitted = adapter.fit(batch.batch, task=MLTask.BINARY_MALICIOUS)
    left = publish(tmp_path / "left", fitted, batch.preprocessor)
    right = publish(
        tmp_path / "deeply" / "nested" / "right", fitted, batch.preprocessor
    )
    first, second = InferenceModel.load(left), InferenceModel.load(right)
    assert first.model_id == second.model_id
    assert first.score(
        batch.matrix.rows, batch.matrix.output_feature_names
    ) == second.score(batch.matrix.rows, batch.matrix.output_feature_names)


def _resign(directory: Path) -> None:
    """Recompute the manifest's own digests after an edit.

    Used by tests that need to get *past* the integrity layer to exercise a
    later check. Rewriting the digests is exactly what a determined tamperer
    would do, which is why the checks after this point exist.
    """
    import hashlib

    payload = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    for entry in payload["files"]:
        content = (directory / entry["relative_path"]).read_bytes()
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        entry["size_bytes"] = len(content)
    (directory / MANIFEST_FILE).write_text(json.dumps(payload), encoding="utf-8")
