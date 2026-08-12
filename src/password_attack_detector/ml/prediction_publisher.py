"""Publishing predictions: staged, re-read, validated, promoted, manifest last.

A prediction directory is a claim that a named frozen champion produced exactly
these rows.  A half-written one is a claim nobody can check -- and worse, one
that looks checkable -- so a publication is built somewhere else entirely and
moved into place only once it is complete and has verified itself.

The order, and why each step is where it is:

1.  the frozen champion is verified before anything is scored, by
    :meth:`~password_attack_detector.ml.predictions.FrozenChampion.load`;
2.  the canonical inference input is built by the label-free loader;
3.  predictions are computed in memory, where every row validates itself against
    the frozen predicate as it is constructed;
4.  the row artifacts are written into a temporary **sibling** directory --
    a sibling because ``rename`` is atomic only within a filesystem, and a
    cross-device move degrades into a copy that can be interrupted halfway;
5.  every artifact is **re-read from disk** and revalidated, which is what proves
    the file is the rows rather than something that merely looked like them in
    memory;
6.  the aggregate profile is derived from those re-read rows;
7.  the manifest is written **last**, so its presence means everything it covers
    is already there;
8.  the complete staged publication is validated as a whole, exactly as
    ``ml validate`` would validate it after promotion;
9.  files and the directory are flushed, then the directory is promoted
    atomically;
10. the staging directory is removed on every exit path.

**Nothing is ever overwritten.**  A prediction identifier is derived from the
predictions themselves, so a second publication at the same identifier is either
the same publication again -- confirmed byte for byte and left exactly as it was
-- or a contradiction, which is refused.  There is no overwrite flag, because
the only thing it could do is destroy the earlier evidence.

**Corruption is not idempotency.**  A destination that exists but does not
validate is a fault, and the publisher fails closed rather than quietly
"completing" it in place: an incomplete publication finished by a later run is a
publication whose two halves came from two runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from password_attack_detector.exceptions import ExperimentPublicationError
from password_attack_detector.ml.dataset import InferenceDataset
from password_attack_detector.ml.enums import AuditStatus, MLSplit
from password_attack_detector.ml.prediction_manifest import (
    ANOMALY_PREDICTION_FILE,
    BINARY_PREDICTION_FILE,
    CATEGORY_PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    PREDICTIONS_DIR,
    QUALITY_REPORT_JSON_FILE,
    QUALITY_REPORT_MD_FILE,
    VALIDATION_RESULT_FILE,
    PredictionFile,
    PredictionLineage,
    PredictionManifest,
    prediction_content_fingerprint,
    scope_role_for,
)
from password_attack_detector.ml.prediction_serialization import (
    read_anomaly_scores,
    read_binary_predictions,
    read_category_predictions,
    write_anomaly_scores,
    write_binary_predictions,
    write_category_predictions,
)
from password_attack_detector.ml.prediction_validation import (
    MLValidationResult,
    validate_publication,
    validate_staged_predictions,
)
from password_attack_detector.ml.predictions import (
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
    ExperimentalAnomalyRun,
    FrozenCategoryModel,
    FrozenChampion,
    predict_anomaly,
    predict_binary,
    predict_category,
)
from password_attack_detector.ml.quality import (
    MLQualityReport,
    build_quality_report,
    quality_report_to_markdown,
)

__all__ = [
    "PredictionPublication",
    "publish_predictions",
]

_PARQUET_MEDIA_TYPE: Final[str] = "application/vnd.apache.parquet"
_JSON_MEDIA_TYPE: Final[str] = "application/json"
_MARKDOWN_MEDIA_TYPE: Final[str] = "text/markdown"


class PredictionPublication(BaseModel):
    """What one publication did."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str
    prediction_content_fingerprint: str
    scope: MLSplit
    scope_role: str
    row_count: int
    category_row_count: int | None
    anomaly_row_count: int | None
    catalog_model_id: str
    model_id: str
    #: True when this call wrote the publication; false when an identical one was
    #: already published and was left exactly as it was.
    created: bool
    validation_status: AuditStatus
    validation_failures: tuple[str, ...]

    @property
    def directory_name(self) -> str:
        """Return the publication's directory name under ``predictions/``."""
        return self.prediction_id


@dataclass(frozen=True, slots=True)
class _Staged:
    """Everything one staged publication produced, before promotion."""

    binary: tuple[BinaryPrediction, ...]
    category: tuple[CategoryPrediction, ...] | None
    anomaly: tuple[AnomalyScore, ...] | None
    validation: MLValidationResult
    quality: MLQualityReport
    manifest: PredictionManifest


def _refuse(message: str) -> None:
    """Refuse a publication, naming the state that made it impossible."""
    raise ExperimentPublicationError(message)


def _digest(payload: bytes) -> str:
    """Return the SHA-256 hex digest of *payload*."""
    return hashlib.sha256(payload).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    """Write *payload* to *path* and flush it to the device."""
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(directory: Path) -> None:
    """Flush *directory*'s own entries, so a promotion survives a crash."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _declared(
    directory: Path, logical_name: str, media_type: str, row_count: int | None
) -> PredictionFile:
    """Return the manifest declaration for one staged file."""
    path = directory / logical_name
    payload = path.read_bytes()
    return PredictionFile(
        logical_name=logical_name,
        relative_path=logical_name,
        media_type=media_type,
        sha256=_digest(payload),
        byte_size=len(payload),
        row_count=row_count,
    )


def _lineage(
    champion: FrozenChampion,
    *,
    probe: ExperimentalAnomalyRun | None,
    category: FrozenCategoryModel | None,
) -> PredictionLineage:
    """Return the frozen lineage every row in this publication is attributable to.

    *category* is the head whose assignments this publication actually carries,
    which is not always the head the lock froze: a caller may publish binary
    predictions alone. The lineage describes what produced *these* rows, so a
    head that scored nothing here is absent from it rather than named beside an
    artifact that does not exist.
    """
    lock = champion.lock
    head = category
    return PredictionLineage(
        champion_lock_fingerprint=lock.lock_fingerprint,
        champion_scope_key=lock.scope_key,
        champion_freeze_record_id=champion.freeze_record_id,
        validation_selection_id=lock.validation_selection_id,
        training_run_id=lock.training_run_id,
        catalog_model_id=lock.catalog_model_id,
        model_id=lock.model_id,
        model_content_fingerprint=lock.model_content_fingerprint,
        model_manifest_fingerprint=lock.model_manifest_fingerprint,
        preprocessor_fingerprint=lock.preprocessor_fingerprint,
        calibration_method=lock.calibration_method,
        calibration_state_fingerprint=lock.calibration_state_fingerprint,
        binary_threshold_fingerprint=lock.binary_threshold_fingerprint,
        binary_score_kind=champion.score_kind,
        decision_threshold=champion.decision_threshold,
        category_selection_id=(
            None
            if head is None or lock.category_head is None
            else lock.category_head.validation_selection_id
        ),
        category_run_id=None if head is None else head.run_id,
        category_model_id=None if head is None else head.model.model_id,
        category_model_content_fingerprint=(
            None if head is None else head.model.document.model_content_fingerprint
        ),
        category_preprocessor_fingerprint=(
            None if head is None else head.model.document.preprocessor_fingerprint
        ),
        category_abstention_fingerprint=(
            None if head is None else head.abstention.selection_fingerprint
        ),
        category_class_order=None if head is None else head.class_order,
        min_category_score=None if head is None else head.min_category_score,
        anomaly_run_id=None if probe is None else probe.run_id,
        anomaly_model_id=None if probe is None else probe.model.model_id,
        anomaly_model_content_fingerprint=(
            None if probe is None else probe.model.document.model_content_fingerprint
        ),
        anomaly_preprocessor_fingerprint=(
            None if probe is None else probe.model.document.preprocessor_fingerprint
        ),
        anomaly_threshold_fingerprint=(
            None
            if probe is None or probe.threshold is None
            else probe.threshold.selection_fingerprint
        ),
        required_feature_schema_version=(
            champion.binary.manifest.required_feature_schema_version
        ),
        feature_catalog_fingerprint=lock.feature_catalog_fingerprint,
        allowlist_fingerprint=lock.allowlist_fingerprint,
        eligible_feature_list_fingerprint=lock.eligible_feature_list_fingerprint,
        ml_config_fingerprint=lock.ml_config_fingerprint,
        model_catalog_fingerprint=lock.model_catalog_fingerprint,
        serializer_id=lock.serializer_id,
        serializer_version=lock.serializer_version,
        inference_adapter_id=lock.inference_adapter_id,
        dependency_contract_fingerprint=lock.dependency_contract_fingerprint,
    )


def _stage(
    staging: Path,
    *,
    champion: FrozenChampion,
    dataset: InferenceDataset,
    probe: ExperimentalAnomalyRun | None,
    include_category: bool,
) -> _Staged:
    """Score, write, re-read, validate, and describe one publication.

    Everything below happens inside *staging*.  Nothing outside it is created,
    touched, or removed, so a failure at any point leaves the destination and
    every previously published artifact exactly as they were.
    """
    binary = predict_binary(champion, dataset)
    # Category triage runs *after* the binary decision and only over the rows it
    # flagged. An empty result is published as an empty table rather than
    # dropped: "nothing was routed to triage" is a finding, and an absent
    # artifact would say instead that no head was frozen.
    category = (
        predict_category(champion, dataset, binary=binary)
        if include_category and champion.category is not None
        else None
    )
    anomaly = predict_anomaly(probe, dataset) if probe is not None else None

    write_binary_predictions(binary, staging / BINARY_PREDICTION_FILE)
    if category is not None:
        write_category_predictions(category, staging / CATEGORY_PREDICTION_FILE)
    if anomaly is not None:
        write_anomaly_scores(anomaly, staging / ANOMALY_PREDICTION_FILE)

    # Re-read from the bytes on disk. What is validated, fingerprinted, and
    # profiled below is the file, not the objects that produced it.
    stored_binary = read_binary_predictions(staging / BINARY_PREDICTION_FILE)
    stored_category = (
        None
        if category is None
        else read_category_predictions(staging / CATEGORY_PREDICTION_FILE)
    )
    stored_anomaly = (
        None
        if anomaly is None
        else read_anomaly_scores(staging / ANOMALY_PREDICTION_FILE)
    )
    if (
        stored_binary != binary
        or stored_category != category
        or (stored_anomaly != anomaly)
    ):
        _refuse(
            "a staged prediction artifact does not read back as the rows that "
            "were written; the destination and every published artifact are "
            "unchanged"
        )

    validation = validate_staged_predictions(
        binary=stored_binary,
        category=stored_category,
        anomaly=stored_anomaly,
        scope=dataset.scope,
        binary_schema_matches=True,
    )
    if not validation.passed:
        _refuse(
            f"the staged predictions failed validation "
            f"({', '.join(validation.failures)}); nothing was promoted"
        )
    _write_bytes(
        staging / VALIDATION_RESULT_FILE, (validation.to_json() + "\n").encode()
    )

    manifest = _build_manifest(
        staging,
        champion=champion,
        dataset=dataset,
        probe=probe,
        binary=stored_binary,
        category=stored_category,
        anomaly=stored_anomaly,
        quality=None,
    )
    quality = build_quality_report(
        manifest=manifest,
        validation=validation,
        binary=stored_binary,
        category=stored_category,
        anomaly=stored_anomaly,
    )
    _write_bytes(
        staging / QUALITY_REPORT_JSON_FILE,
        (json.dumps(quality.to_dict(), indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_bytes(
        staging / QUALITY_REPORT_MD_FILE,
        quality_report_to_markdown(quality).encode(),
    )

    # Rebuilt now that every file it declares exists. The identity is unchanged
    # by the rebuild -- it is derived from the rows and the lineage, neither of
    # which the reports touch -- so the two manifests differ only in the file
    # declarations, which is exactly what the second pass is for.
    final = _build_manifest(
        staging,
        champion=champion,
        dataset=dataset,
        probe=probe,
        binary=stored_binary,
        category=stored_category,
        anomaly=stored_anomaly,
        quality=quality,
    )
    if final.prediction_id != manifest.prediction_id:
        _refuse(
            "the prediction identity moved when the aggregate reports were "
            "added; identity must depend on the predictions alone"
        )
    _write_bytes(staging / PREDICTION_MANIFEST_FILE, (final.to_json() + "\n").encode())
    return _Staged(
        binary=stored_binary,
        category=stored_category,
        anomaly=stored_anomaly,
        validation=validation,
        quality=quality,
        manifest=final,
    )


def _build_manifest(
    staging: Path,
    *,
    champion: FrozenChampion,
    dataset: InferenceDataset,
    probe: ExperimentalAnomalyRun | None,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    anomaly: Sequence[AnomalyScore] | None,
    quality: MLQualityReport | None,
) -> PredictionManifest:
    """Return the manifest describing what is currently staged.

    The validation result and the quality report are bound by the **digest of
    the bytes on disk**, not by the objects that produced them: a report swapped
    for one describing different predictions has different bytes, and that is
    the thing worth catching.
    """
    files = [
        _declared(staging, BINARY_PREDICTION_FILE, _PARQUET_MEDIA_TYPE, len(binary))
    ]
    if category is not None:
        files.append(
            _declared(
                staging,
                CATEGORY_PREDICTION_FILE,
                _PARQUET_MEDIA_TYPE,
                len(category),
            )
        )
    if anomaly is not None:
        files.append(
            _declared(
                staging, ANOMALY_PREDICTION_FILE, _PARQUET_MEDIA_TYPE, len(anomaly)
            )
        )
    files.append(_declared(staging, VALIDATION_RESULT_FILE, _JSON_MEDIA_TYPE, None))
    if quality is not None:
        files.append(
            _declared(staging, QUALITY_REPORT_JSON_FILE, _JSON_MEDIA_TYPE, None)
        )
        files.append(
            _declared(staging, QUALITY_REPORT_MD_FILE, _MARKDOWN_MEDIA_TYPE, None)
        )

    quality_declaration = next(
        (item for item in files if item.logical_name == QUALITY_REPORT_JSON_FILE), None
    )
    validation_declaration = next(
        item for item in files if item.logical_name == VALIDATION_RESULT_FILE
    )
    return PredictionManifest.build(
        scope=dataset.scope,
        scope_role=scope_role_for(dataset.scope),
        lineage=_lineage(
            champion,
            probe=probe,
            category=None if category is None else champion.category,
        ),
        inference_input_fingerprint=dataset.inference_input_fingerprint,
        split_membership_fingerprint=dataset.split_membership_fingerprint,
        prediction_content_fingerprint=prediction_content_fingerprint(
            binary=binary, category=category, anomaly=anomaly
        ),
        row_count=len(binary),
        category_row_count=None if category is None else len(category),
        anomaly_row_count=None if anomaly is None else len(anomaly),
        files=tuple(sorted(files, key=lambda item: item.logical_name)),
        validation_result_fingerprint=validation_declaration.sha256,
        quality_report_fingerprint=(
            None if quality_declaration is None else quality_declaration.sha256
        ),
    )


def publish_predictions(
    *,
    champion: FrozenChampion,
    dataset: InferenceDataset,
    root: Path,
    probe: ExperimentalAnomalyRun | None = None,
    include_category: bool = True,
) -> PredictionPublication:
    """Score *dataset* under *champion* and publish the result, transactionally.

    Args:
        champion: the verified frozen champion.  There is no parameter for a
            model path or a model identifier: predictions attributable to
            "whichever model was in this folder" are attributable to nothing.
        dataset: the label-free inference input, already canonically ordered.
        root: the artifact root; publications are written under
            ``<root>/predictions/<prediction_id>``.
        probe: a verified experimental anomaly run whose scores are published
            beside the supervised ones, in their own artifact.  Never read from
            the champion lock, and never able to change a supervised decision.
        include_category: publish the frozen category head's assignments when
            one was frozen.  A head that was not frozen is never fabricated.

    Raises:
        ExperimentPublicationError: if staging, validation, or promotion fails,
            or if a different publication already exists at this identity.  The
            destination and every previously published artifact are untouched.
    """
    predictions_root = Path(root) / PREDICTIONS_DIR
    predictions_root.mkdir(parents=True, exist_ok=True)
    staging = predictions_root / f".staging-{dataset.inference_input_fingerprint[:16]}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    promoted = False
    try:
        staged = _stage(
            staging,
            champion=champion,
            dataset=dataset,
            probe=probe,
            include_category=include_category,
        )
        # The whole staged publication, checked exactly as a later reader will
        # check the promoted one. A publication that would fail 'ml validate'
        # the moment it landed is not promoted.
        outcome = validate_publication(staging)
        if not outcome.passed:
            _refuse(
                f"the staged publication did not validate as a whole "
                f"({', '.join(outcome.failures)}); nothing was promoted"
            )

        target = predictions_root / staged.manifest.prediction_id
        if target.exists():
            stored = _require_identical(target, staged.manifest)
            return _publication(staged, champion=champion, created=False, stored=stored)

        _sync_directory(staging)
        staging.rename(target)
        promoted = True
        _sync_directory(predictions_root)
    except ExperimentPublicationError:
        raise
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the prediction publication could not be staged "
            f"({type(exc).__name__}); the destination and every published "
            f"artifact are unchanged"
        ) from None
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)

    return _publication(staged, champion=champion, created=True, stored=staged.manifest)


def _publication(
    staged: _Staged,
    *,
    champion: FrozenChampion,
    created: bool,
    stored: PredictionManifest,
) -> PredictionPublication:
    """Return the result record for one publication."""
    return PredictionPublication(
        prediction_id=stored.prediction_id,
        prediction_content_fingerprint=stored.prediction_content_fingerprint,
        scope=stored.scope,
        scope_role=stored.scope_role.role,
        row_count=stored.row_count,
        category_row_count=stored.category_row_count,
        anomaly_row_count=stored.anomaly_row_count,
        catalog_model_id=champion.lock.catalog_model_id,
        model_id=champion.lock.model_id,
        created=created,
        validation_status=staged.validation.status,
        validation_failures=staged.validation.failures,
    )


def _require_identical(
    target: Path, manifest: PredictionManifest
) -> PredictionManifest:
    """Return the published manifest at *target*, or refuse.

    Compared on canonical bytes rather than on the identifier alone.  Two
    publications sharing an identifier and disagreeing about their content is
    the case worth catching, and it is the only one an identifier check would
    miss.
    """
    path = target / PREDICTION_MANIFEST_FILE
    if not path.is_file():
        _refuse(
            "a prediction directory already exists at this identity without a "
            "manifest; an incomplete publication is never completed in place, "
            "because its two halves would come from two runs"
        )
    try:
        stored = PredictionManifest.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the published prediction manifest is not readable "
            f"({type(exc).__name__}); a corrupt publication is a fault, not an "
            f"idempotent republication"
        ) from None
    if stored.to_json() != manifest.to_json():
        _refuse(
            "a different prediction publication already exists at this "
            "identity; a published prediction is evidence and is never "
            "overwritten, merged, or regenerated in place"
        )
    outcome = validate_publication(target)
    if not outcome.passed:
        _refuse(
            f"the existing publication at this identity does not validate "
            f"({', '.join(outcome.failures)}); a corrupt destination is a fault "
            f"rather than an idempotent republication"
        )
    return stored
