"""The uncalibrated-score contract, pinned across every family and artifact.

Milestone 4 fits no calibrator, so **nothing it produces is a probability**. A
binary head emits a `decision_score`, a category head a `class_score`, the
anomaly head an `anomaly_score`, and every published manifest records
`calibration_status: not_fitted`.

The temptation this guards against is specific. A logistic sigmoid and a forest
vote both produce numbers in `[0, 1]` that look exactly like probabilities, and
the parity tests in this suite legitimately compare them against scikit-learn's
`predict_proba` -- because that *is* the estimator output being reproduced.
Reproducing a library's arithmetic is not the same as inheriting its
vocabulary, and the assertions here are what keep the two apart until a
calibrator has actually been fitted and measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import (
    ManifestVerificationError,
    ModelSerializationError,
)
from password_attack_detector.ml.catalog import MODEL_CATALOG
from password_attack_detector.ml.enums import (
    PROBABILITY_SCORE_KINDS,
    CalibrationMethod,
    MLTask,
    ScoreKind,
)
from password_attack_detector.ml.manifest import read_model_manifest
from password_attack_detector.ml.models import (
    HistogramBoostingAdapter,
    IsolationForestAdapter,
    LogisticRegressionAdapter,
    PriorBaselineAdapter,
    RandomForestAdapter,
    SingleFeatureThresholdAdapter,
)
from password_attack_detector.ml.serialization import (
    MANIFEST_FILE,
    MODEL_FILE,
    build_model_document,
    read_model_document,
)
from tests.ml.models import prepare, publish

#: Names that must not appear on a Milestone 4 model output.
FORBIDDEN_SCORE_NAMES = (
    "calibrated_probability",
    "malicious_probability",
    "attack_probability",
    "probability",
    "likelihood",
    "confidence",
)


@pytest.fixture
def binary() -> Any:
    """Return a prepared binary training batch."""
    return prepare(count=140)


def fitted_families(binary: Any) -> list[tuple[str, Any, Any]]:
    """Return one fitted model per family, with the adapter that produced it."""
    triage = prepare(count=120, task=MLTask.ATTACK_CATEGORY)
    unlabelled = prepare(count=120, task=MLTask.ANOMALY)
    cases: list[tuple[str, Any, Any]] = []
    for label, adapter, prepared, task in (
        ("M-000 binary", PriorBaselineAdapter(), binary, MLTask.BINARY_MALICIOUS),
        ("M-000 category", PriorBaselineAdapter(), triage, MLTask.ATTACK_CATEGORY),
        (
            "M-001 binary",
            SingleFeatureThresholdAdapter(feature="user_failure_rate"),
            binary,
            MLTask.BINARY_MALICIOUS,
        ),
        ("M-010 binary", LogisticRegressionAdapter(), binary, MLTask.BINARY_MALICIOUS),
        (
            "M-010 category",
            LogisticRegressionAdapter(),
            triage,
            MLTask.ATTACK_CATEGORY,
        ),
        (
            "M-020 binary",
            RandomForestAdapter(n_estimators=8, max_depth=4),
            binary,
            MLTask.BINARY_MALICIOUS,
        ),
        (
            "M-020 category",
            RandomForestAdapter(n_estimators=8, max_depth=4),
            triage,
            MLTask.ATTACK_CATEGORY,
        ),
        (
            "M-021 binary",
            HistogramBoostingAdapter(max_iter=5),
            binary,
            MLTask.BINARY_MALICIOUS,
        ),
        (
            "M-030 anomaly",
            IsolationForestAdapter(n_estimators=8, max_samples=32),
            unlabelled,
            MLTask.ANOMALY,
        ),
    ):
        cases.append((label, adapter.fit(prepared.batch, task=task), prepared))
    return cases


# ---------------------------------------------------------------------------
# Every family, every task
# ---------------------------------------------------------------------------


def test_every_fitted_model_declares_an_uncalibrated_score_kind(
    binary: Any,
) -> None:
    """Nine fitted models across six families, and none is a probability."""
    expected = {
        MLTask.BINARY_MALICIOUS: ScoreKind.DECISION_SCORE,
        MLTask.ATTACK_CATEGORY: ScoreKind.CLASS_SCORE,
        MLTask.ANOMALY: ScoreKind.ANOMALY_SCORE,
    }
    cases = fitted_families(binary)
    assert len(cases) == 9
    for label, fitted, _ in cases:
        semantics = fitted.score_semantics
        assert semantics.score_kind is expected[fitted.task], label
        assert semantics.score_kind not in PROBABILITY_SCORE_KINDS, label
        assert semantics.calibration_method is CalibrationMethod.NONE, label


def test_no_fitted_model_uses_a_probability_word_anywhere(binary: Any) -> None:
    """Swept over the whole canonical content, not only the score kind.

    A hyperparameter, a parameter, or a description could each smuggle the word
    in, so the assertion is over the rendered payload.
    """
    for label, fitted, _ in fitted_families(binary):
        rendered = json.dumps(fitted.content_payload()).lower()
        rendered += fitted.score_semantics.description.lower()
        for banned in FORBIDDEN_SCORE_NAMES:
            assert banned not in rendered, f"{label}: {banned}"


def test_the_catalog_declares_no_calibrated_native_score_kind() -> None:
    """A family's *native* output is never a probability either."""
    for spec in MODEL_CATALOG.specs:
        assert spec.native_score_kind not in PROBABILITY_SCORE_KINDS, spec.model_id


def test_the_anomaly_family_emits_only_the_anomaly_vocabulary() -> None:
    """No decision score, no class score, and certainly no probability."""
    unlabelled = prepare(count=100, task=MLTask.ANOMALY)
    fitted = IsolationForestAdapter(n_estimators=8, max_samples=32).fit(
        unlabelled.batch, task=MLTask.ANOMALY
    )
    assert fitted.score_semantics.score_kind is ScoreKind.ANOMALY_SCORE
    assert fitted.score_columns == ("anomaly_score",)


# ---------------------------------------------------------------------------
# Published artifacts
# ---------------------------------------------------------------------------


def test_every_published_artifact_records_calibration_as_not_fitted(
    binary: Any, tmp_path: Path
) -> None:
    """Both the document and the manifest say so, and they must agree."""
    for index, (label, fitted, prepared) in enumerate(fitted_families(binary)):
        if fitted.family.value == "histogram_gradient_boosting":
            continue  # unpublishable by contract; covered separately
        directory = publish(tmp_path / f"model-{index}", fitted, prepared.preprocessor)
        document = read_model_document(
            json.loads((directory / MODEL_FILE).read_text(encoding="utf-8"))
        )
        manifest = read_model_manifest(
            json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        )
        assert document.calibration_status == "not_fitted", label
        assert manifest.calibration_status == "not_fitted", label
        assert document.score_semantics.score_kind not in PROBABILITY_SCORE_KINDS


def test_a_published_artifact_names_no_probability_column(
    binary: Any, tmp_path: Path
) -> None:
    """A privacy-style sweep, applied to vocabulary instead of identity."""
    fitted = LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    directory = publish(tmp_path / "linear", fitted, binary.preprocessor)
    for name in (MODEL_FILE, MANIFEST_FILE):
        text = (directory / name).read_text(encoding="utf-8").lower()
        for banned in FORBIDDEN_SCORE_NAMES:
            assert banned not in text, f"{name}: {banned}"


def test_an_artifact_cannot_claim_a_calibrated_kind_while_uncalibrated(
    binary: Any,
) -> None:
    """The central regression assertion of this correction.

    A document that declared a calibrated probability while recording
    ``calibration_status: not_fitted`` would be claiming a measurement nobody
    made. Two independent guards refuse it: ``ScoreSemantics`` requires a fitted
    calibration method for a calibrated kind, and the document requires
    ``not_fitted`` at this contract version. Both are asserted, because either
    alone could be relaxed.
    """
    fitted = LogisticRegressionAdapter().fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    payload = build_model_document(fitted).to_dict()

    # A calibrated score kind with no calibrator: refused by ScoreSemantics.
    calibrated = dict(payload)
    calibrated["score_semantics"] = {
        **payload["score_semantics"],
        "score_kind": "calibrated_probability",
    }
    with pytest.raises(ModelSerializationError, match="not a valid model document"):
        read_model_document(calibrated)

    # A calibrated score kind *and* a calibration method, but the document still
    # records not_fitted -- so the claim contradicts the artifact's own status.
    both = dict(payload)
    both["score_semantics"] = {
        **payload["score_semantics"],
        "score_kind": "calibrated_probability",
        "calibration_method": "platt",
    }
    both["calibration_status"] = "platt"
    with pytest.raises(ModelSerializationError, match="not a valid model document"):
        read_model_document(both)


def test_a_manifest_cannot_claim_calibration_while_the_document_does_not(
    binary: Any, tmp_path: Path
) -> None:
    """The two files must agree, and neither may claim it alone."""
    from password_attack_detector.ml.manifest import verify_model_artifact

    fitted = RandomForestAdapter(n_estimators=6, max_depth=3).fit(
        binary.batch, task=MLTask.BINARY_MALICIOUS
    )
    directory = publish(tmp_path / "forest", fitted, binary.preprocessor)
    payload = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    payload["calibration_status"] = "isotonic"
    (directory / MANIFEST_FILE).write_text(json.dumps(payload), encoding="utf-8")

    outcome = verify_model_artifact(directory)
    assert outcome.passed is False
    assert outcome.error_code == "MANIFEST_INVALID"

    with pytest.raises(ManifestVerificationError):
        read_model_manifest(payload)


def test_a_calibrated_score_semantics_still_needs_a_calibrator() -> None:
    """The general rule the artifact guards rest on, asserted directly."""
    from pydantic import ValidationError

    from password_attack_detector.ml.schemas import ScoreSemantics

    with pytest.raises(ValidationError, match="requires a fitted calibration"):
        ScoreSemantics(
            score_kind=ScoreKind.CALIBRATED_PROBABILITY,
            calibration_method=CalibrationMethod.NONE,
            lower_bound=0.0,
            upper_bound=1.0,
            description="A calibrated probability of the malicious class.",
        )


@pytest.mark.parametrize(
    "kind",
    [ScoreKind.DECISION_SCORE, ScoreKind.CLASS_SCORE, ScoreKind.ANOMALY_SCORE],
)
def test_an_uncalibrated_kind_may_not_describe_itself_as_a_probability(
    kind: ScoreKind,
) -> None:
    """Enforced by the type, so no adapter can talk its way around it."""
    from pydantic import ValidationError

    from password_attack_detector.ml.schemas import ScoreSemantics

    with pytest.raises(ValidationError, match="must not use the word"):
        ScoreSemantics(
            score_kind=kind,
            calibration_method=CalibrationMethod.NONE,
            lower_bound=0.0,
            upper_bound=1.0,
            description="An estimated probability that the row is malicious.",
        )


def test_reproducing_predict_proba_does_not_change_the_vocabulary(
    binary: Any,
) -> None:
    """The parity comparison is arithmetic; the contract is semantics.

    This suite's parity tests compare project-owned output against
    ``predict_proba`` because that is the estimator output being reproduced.
    That comparison is internal, and it must leave the external score semantics
    exactly where they were -- which is what this asserts, on the same model
    the parity test uses.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    adapter = LogisticRegressionAdapter()
    fitted = adapter.fit(binary.batch, task=MLTask.BINARY_MALICIOUS)
    estimator = LogisticRegression(
        solver="lbfgs", C=1.0, max_iter=1000, tol=1e-4, random_state=42
    )
    estimator.fit(
        binary.batch.design(),
        binary.batch.encoded_targets(),
        sample_weight=binary.batch.sample_weights(),
    )
    mine = np.asarray(
        adapter.score(fitted, binary.matrix.rows, binary.matrix.output_feature_names)
    )
    assert (
        np.max(np.abs(mine - estimator.predict_proba(binary.batch.design()))) <= 1e-12
    )

    # Numerically identical to a library's "proba", and still not called one.
    assert fitted.score_semantics.score_kind is ScoreKind.DECISION_SCORE
    assert fitted.score_semantics.calibration_method is CalibrationMethod.NONE
    assert "probability" not in fitted.score_semantics.description.lower()
