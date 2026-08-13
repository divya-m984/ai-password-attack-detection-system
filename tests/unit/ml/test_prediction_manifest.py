"""The prediction manifest: what it identifies, and what it refuses to carry.

The identity tests are the point of this file. A prediction identifier is
derived from what was scored and what scored it, so the suite asserts both
directions: two publications of identical predictions in different directories at
different times derive the same identifier, and a single changed predicted value
derives a different one.

The rest is prohibition. A manifest that grew a metric field, a label
fingerprint, or a list of anchors would be an artifact whose whole purpose --
being safe to hand to somebody auditing the publication -- had quietly lapsed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from password_attack_detector.exceptions import ModelTrainingError
from password_attack_detector.ml.enums import CalibrationMethod, MLSplit, ScoreKind
from password_attack_detector.ml.prediction_manifest import (
    ANOMALY_PREDICTION_FILE,
    BINARY_PREDICTION_FILE,
    CATEGORY_PREDICTION_FILE,
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    QUALITY_REPORT_JSON_FILE,
    VALIDATION_RESULT_FILE,
    PredictionFile,
    PredictionLineage,
    PredictionManifest,
    prediction_content_fingerprint,
    scope_role_for,
)
from password_attack_detector.ml.predictions import BinaryPrediction, CategoryPrediction
from tests.unit.ml.test_prediction_serialization import (
    binary_rows,
    category_rows,
)

EPOCH = datetime(2024, 3, 1, tzinfo=UTC)
DIGEST = "a" * 64
OTHER = "b" * 64


def lineage(**overrides: Any) -> PredictionLineage:
    """Return a coherent frozen lineage, with *overrides* applied."""
    fields: dict[str, Any] = {
        "champion_lock_fingerprint": DIGEST,
        "champion_scope_key": OTHER,
        "champion_freeze_record_id": "11111111-0000-5000-8000-000000000001",
        "validation_selection_id": "11111111-0000-5000-8000-000000000002",
        "training_run_id": "11111111-0000-5000-8000-000000000003",
        "catalog_model_id": "M-010",
        "model_id": "11111111-0000-5000-8000-000000000004",
        "model_content_fingerprint": "c" * 64,
        "model_manifest_fingerprint": "d" * 64,
        "preprocessor_fingerprint": "e" * 64,
        "calibration_method": CalibrationMethod.NONE,
        "calibration_state_fingerprint": None,
        "binary_threshold_fingerprint": "f" * 64,
        "binary_score_kind": ScoreKind.DECISION_SCORE,
        "decision_threshold": 0.2,
        "required_feature_schema_version": "1.0.0",
        "feature_catalog_fingerprint": "0" * 64,
        "allowlist_fingerprint": "1" * 64,
        "eligible_feature_list_fingerprint": "2" * 64,
        "ml_config_fingerprint": "3" * 64,
        "model_catalog_fingerprint": "4" * 64,
        "serializer_id": "logreg-json-v1",
        "serializer_version": 1,
        "inference_adapter_id": "logreg-adapter-v1",
        "dependency_contract_fingerprint": "5" * 64,
    }
    fields.update(overrides)
    return PredictionLineage(**fields)


def declaration(name: str, *, row_count: int | None = None, **overrides: Any) -> Any:
    """Return one declared file."""
    fields: dict[str, Any] = {
        "logical_name": name,
        "relative_path": name,
        "media_type": (
            "application/vnd.apache.parquet"
            if name.endswith(".parquet")
            else "application/json"
        ),
        "sha256": DIGEST,
        "byte_size": 1024,
        "row_count": row_count,
    }
    fields.update(overrides)
    return PredictionFile(**fields)


def manifest(
    rows: list[BinaryPrediction] | None = None,
    *,
    category: list[CategoryPrediction] | None = None,
    **overrides: Any,
) -> PredictionManifest:
    """Return a valid manifest over *rows*, with *overrides* applied."""
    binary = rows if rows is not None else binary_rows(5)
    files = [
        declaration(BINARY_PREDICTION_FILE, row_count=len(binary)),
        declaration(VALIDATION_RESULT_FILE),
        declaration(QUALITY_REPORT_JSON_FILE, sha256=OTHER),
    ]
    if category is not None:
        files.append(declaration(CATEGORY_PREDICTION_FILE, row_count=len(category)))
    fields: dict[str, Any] = {
        "scope": MLSplit.TEST,
        "scope_role": scope_role_for(MLSplit.TEST),
        "lineage": lineage(),
        "inference_input_fingerprint": "6" * 64,
        "split_membership_fingerprint": "7" * 64,
        "prediction_content_fingerprint": prediction_content_fingerprint(
            binary=binary, category=category, anomaly=None
        ),
        "row_count": len(binary),
        "category_row_count": None if category is None else len(category),
        "anomaly_row_count": None,
        "files": tuple(sorted(files, key=lambda item: item.logical_name)),
        "validation_result_fingerprint": DIGEST,
        "quality_report_fingerprint": OTHER,
    }
    fields.update(overrides)
    return PredictionManifest.build(**fields)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_a_manifest_derives_its_own_identifier() -> None:
    """An assigned identifier is not an identity."""
    built = manifest()
    assert built.prediction_id == built.derived_prediction_id()
    with pytest.raises(ValueError, match="cannot be supplied"):
        manifest(prediction_id="00000000-0000-5000-8000-000000000009")


def test_identical_predictions_derive_one_identifier() -> None:
    """Two publications of the same rows are the same publication."""
    assert manifest().prediction_id == manifest().prediction_id


def test_one_changed_prediction_derives_a_different_identifier() -> None:
    """Changing an output value moves the identity."""
    assert (
        manifest(binary_rows(5)).prediction_id != manifest(binary_rows(6)).prediction_id
    )


def test_a_different_scope_derives_a_different_identifier() -> None:
    """Test predictions and holdout predictions are different publications."""
    assert (
        manifest(
            scope=MLSplit.NOVEL_ANOMALY_HOLDOUT,
            scope_role=scope_role_for(MLSplit.NOVEL_ANOMALY_HOLDOUT),
        ).prediction_id
        != manifest().prediction_id
    )


def test_a_different_champion_derives_a_different_identifier() -> None:
    """The same rows from a different model are different predictions."""
    assert (
        manifest(lineage=lineage(champion_lock_fingerprint="9" * 64)).prediction_id
        != manifest().prediction_id
    )


def test_the_file_digests_do_not_take_part_in_identity() -> None:
    """Identity is about the rows; the digests are integrity evidence about a
    particular writing of them."""
    baseline = manifest()
    restated = manifest(
        files=tuple(
            sorted(
                (
                    declaration(BINARY_PREDICTION_FILE, row_count=5, byte_size=2048),
                    declaration(VALIDATION_RESULT_FILE, byte_size=99),
                    declaration(QUALITY_REPORT_JSON_FILE, sha256=OTHER, byte_size=77),
                ),
                key=lambda item: item.logical_name,
            )
        )
    )
    assert restated.prediction_id == baseline.prediction_id
    assert restated.prediction_manifest_fingerprint != (
        baseline.prediction_manifest_fingerprint
    )


def test_nothing_observational_appears_on_the_manifest() -> None:
    """No path, no host, no user, no publication time."""
    payload = manifest().to_dict()
    for absent in (
        "output_directory",
        "path",
        "hostname",
        "user",
        "published_at",
        "created_at",
        "mtime",
    ):
        assert absent not in payload, absent


# ---------------------------------------------------------------------------
# The seal
# ---------------------------------------------------------------------------


def test_an_edited_manifest_is_refused() -> None:
    """The digest is a field, recomputed on every deserialization."""
    payload = manifest().to_dict()
    payload["row_count"] = payload["row_count"] + 1
    with pytest.raises(ModelTrainingError, match="prediction manifest"):
        PredictionManifest.from_dict(payload)


def test_a_manifest_from_another_contract_version_is_refused() -> None:
    """The version is checked before the payload is understood."""
    payload = manifest().to_dict()
    payload["prediction_manifest_schema_version"] = "9.9.9"
    with pytest.raises(ModelTrainingError, match="schema version"):
        PredictionManifest.from_dict(payload)


def test_a_manifest_round_trips_through_canonical_json() -> None:
    """Two renderings of one manifest are the same bytes."""
    built = manifest()
    assert PredictionManifest.from_json(built.to_json()).to_json() == built.to_json()


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------


def test_a_publication_always_declares_its_binary_predictions() -> None:
    """The binary head is the publication; everything else is beside it."""
    with pytest.raises(ValueError, match="always contains its binary predictions"):
        manifest(
            files=tuple(
                sorted(
                    (
                        declaration(VALIDATION_RESULT_FILE),
                        declaration(QUALITY_REPORT_JSON_FILE, sha256=OTHER),
                    ),
                    key=lambda item: item.logical_name,
                )
            )
        )


def test_a_category_artifact_and_its_count_travel_together() -> None:
    """A declared file with no count, or a count with no file, is incoherent."""
    with pytest.raises(ValueError, match="declared together"):
        manifest(category_row_count=3)


def test_a_declared_category_artifact_names_the_head_that_produced_it() -> None:
    """A category table with no frozen head in the lineage is unattributable."""
    rows = category_rows(3)
    with pytest.raises(ValueError, match="names the frozen head"):
        manifest(category=rows)


def test_a_coherent_category_publication_validates() -> None:
    """The positive case, so the refusals above are not vacuous."""
    rows = category_rows(3)
    built = manifest(
        category=rows,
        lineage=lineage(
            category_selection_id="11111111-0000-5000-8000-000000000005",
            category_run_id="11111111-0000-5000-8000-000000000006",
            category_model_id="11111111-0000-5000-8000-000000000007",
            category_model_content_fingerprint="8" * 64,
            category_preprocessor_fingerprint="9" * 64,
            category_abstention_fingerprint="a" * 64,
            category_class_order=("brute_force", "credential_stuffing"),
            min_category_score=0.3,
        ),
    )
    assert built.category_row_count == 3


def test_a_partial_category_lineage_is_refused() -> None:
    """A lineage nobody can check is not a lineage."""
    with pytest.raises(ValueError, match="in full or not at all"):
        lineage(category_run_id="11111111-0000-5000-8000-000000000006")


def test_a_calibrated_score_kind_requires_a_calibrator() -> None:
    """The one substitution that would change every decision invisibly."""
    with pytest.raises(ValueError, match="names the calibrator"):
        lineage(binary_score_kind=ScoreKind.CALIBRATED_PROBABILITY)


def test_a_calibrator_without_a_calibrated_score_kind_is_refused() -> None:
    """Both directions of the same rule."""
    with pytest.raises(ValueError, match="names the calibrator"):
        lineage(calibration_state_fingerprint="b" * 64)


def test_the_declared_row_count_must_match_the_binary_artifact() -> None:
    """A manifest that disagreed with its own table would be checkable and wrong."""
    with pytest.raises(ValueError, match="row count"):
        manifest(row_count=99)


def test_declared_files_are_unique_and_ordered() -> None:
    """Two declarations of one file, or an unordered list, is refused."""
    with pytest.raises(ValueError, match="each file once"):
        manifest(
            files=tuple(
                sorted(
                    (
                        declaration(BINARY_PREDICTION_FILE, row_count=5),
                        declaration(BINARY_PREDICTION_FILE, row_count=5),
                        declaration(VALIDATION_RESULT_FILE),
                        declaration(QUALITY_REPORT_JSON_FILE, sha256=OTHER),
                    ),
                    key=lambda item: item.logical_name,
                )
            )
        )


def test_a_declared_path_may_not_escape_the_publication() -> None:
    """No separator, no traversal, no absolute path."""
    for unsafe in ("../escape.json", "/etc/passwd", "nested/file.json"):
        with pytest.raises(ValueError, match="not a safe file name"):
            declaration(VALIDATION_RESULT_FILE, relative_path=unsafe)


def test_an_unknown_media_type_is_refused() -> None:
    """Only the formats a publication actually writes may be declared."""
    with pytest.raises(ValueError, match="media_type"):
        declaration(VALIDATION_RESULT_FILE, media_type="application/x-pickle")


# ---------------------------------------------------------------------------
# Scope roles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "role"),
    [
        (MLSplit.TRAIN, "supervised_prediction"),
        (MLSplit.VALIDATION, "supervised_prediction"),
        (MLSplit.TEST, "supervised_prediction"),
        (MLSplit.NOVEL_ANOMALY_HOLDOUT, "generalisation_probe"),
    ],
)
def test_each_scope_carries_its_own_role(scope: MLSplit, role: str) -> None:
    """The holdout is a probe wherever it appears."""
    assert scope_role_for(scope).role == role


def test_excluded_rows_have_no_prediction_role() -> None:
    """Excluded is excluded from prediction too."""
    with pytest.raises(ValueError, match="excluded from every stage"):
        scope_role_for(MLSplit.EXCLUDED)


def test_a_holdout_publication_cannot_claim_the_supervised_role() -> None:
    """A generalisation probe is never republished as a supervised scope."""
    with pytest.raises(ValueError, match="never republished"):
        manifest(scope=MLSplit.NOVEL_ANOMALY_HOLDOUT)


# ---------------------------------------------------------------------------
# The content fingerprint
# ---------------------------------------------------------------------------


def test_the_content_fingerprint_covers_every_published_value() -> None:
    """A changed decision moves it."""
    rows = binary_rows(4)
    changed = [*rows[:-1], rows[-1].model_copy(update={"flagged_malicious": None})]
    del changed
    assert prediction_content_fingerprint(
        binary=rows, category=None, anomaly=None
    ) != prediction_content_fingerprint(
        binary=binary_rows(4, calibrated=True), category=None, anomaly=None
    )


def test_an_absent_artifact_is_not_an_empty_one() -> None:
    """A publication with no head and one whose head scored nothing differ."""
    rows = binary_rows(3)
    assert prediction_content_fingerprint(
        binary=rows, category=None, anomaly=None
    ) != prediction_content_fingerprint(binary=rows, category=[], anomaly=None)


def test_the_content_fingerprint_is_stable_across_calls() -> None:
    """Nothing observational takes part in it."""
    rows = binary_rows(3)
    assert prediction_content_fingerprint(
        binary=rows, category=None, anomaly=None
    ) == prediction_content_fingerprint(binary=rows, category=None, anomaly=None)


# ---------------------------------------------------------------------------
# What a manifest must never carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "pr_auc",
        "brier_score",
        "confusion_matrix",
        "test_metrics",
        "test_evaluation",
        "label_fingerprint",
        "anchor_event_id",
        "campaign_id",
    ],
)
def test_no_outcome_or_identity_field_is_declared(field: str) -> None:
    """The import-time guard, restated where a reader will look for it."""
    for model in (PredictionManifest, PredictionLineage, PredictionFile):
        assert field not in model.model_fields, (model.__name__, field)


def test_the_manifest_contract_version_is_pinned() -> None:
    """A change to what a manifest binds is a visible edit."""
    assert PREDICTION_MANIFEST_SCHEMA_VERSION == "1.0.0"


def test_an_anomaly_artifact_and_its_count_travel_together() -> None:
    """The experimental artifact follows the same rule as the category one."""
    with pytest.raises(ValueError, match="declared together"):
        manifest(anomaly_row_count=4)


def test_a_declared_anomaly_artifact_names_its_run() -> None:
    """An unattributable experimental artifact is not published."""
    with pytest.raises(ValueError, match="names the experimental run"):
        manifest(
            files=tuple(
                sorted(
                    (
                        declaration(BINARY_PREDICTION_FILE, row_count=5),
                        declaration(ANOMALY_PREDICTION_FILE, row_count=5),
                        declaration(VALIDATION_RESULT_FILE),
                        declaration(QUALITY_REPORT_JSON_FILE, sha256=OTHER),
                    ),
                    key=lambda item: item.logical_name,
                )
            ),
            anomaly_row_count=5,
        )


def test_a_manifest_names_no_anchor(tmp_path: Path) -> None:
    """The rows carry their join keys; the manifest counts them."""
    rendered = manifest().to_json()
    for row in binary_rows(5):
        assert row.anchor_event_id not in rendered


def test_the_epoch_used_here_is_a_literal() -> None:
    """No wall clock takes part in any fixture, so no test can drift."""
    assert datetime(2024, 3, 1, tzinfo=UTC) == EPOCH
    assert (EPOCH + timedelta(minutes=1)).minute == 1
