"""The manifest a prediction publication is identified and checked by.

A prediction manifest answers two questions that a directory of Parquet files
cannot answer about itself: **which frozen state produced these rows**, and
**are these still those rows**.  It is written last, after every artifact it
covers, so its presence means the publication is complete.

**Identity is semantic.**  ``prediction_id`` is derived from what was scored and
what scored it -- the frozen champion's whole lineage, the resolved feature
contract, the inference input, the scope, and the exact content of the rows.  It
is derived from nothing observational: not the output directory, not the
checkout path, not the hostname, not the user, not a publication time, and not a
filesystem timestamp.  Two publications of the same predictions in two
directories at two times therefore carry the same identifier and byte-identical
manifests, and a publication whose *rows* differ carries a different one.

**There is no metric here of any kind.**  Not an accuracy, not a precision, not
a confusion matrix, and above all no test figure -- a prediction publication is
produced without opening a label, so there is nothing outcome-dependent for it
to record.  It carries no label fingerprint either: fingerprinting the labels
would require reading them.

**There is no identifier list.**  The rows carry their own join keys; the
manifest counts them.  A manifest that enumerated anchors would republish the
identity of every scored event in a file whose whole purpose is to be read by
somebody auditing the publication.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from typing import Any, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from password_attack_detector.ml.calibration import SealedModel, canonical_json
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    MLSplit,
    ScoreKind,
    is_probability,
)
from password_attack_detector.ml.predictions import (
    PREDICTION_SCHEMA_VERSION,
    AnomalyScore,
    BinaryPrediction,
    CategoryPrediction,
)
from password_attack_detector.ml.schemas import Sha256Hex

__all__ = [
    "ANOMALY_PREDICTION_FILE",
    "BINARY_PREDICTION_FILE",
    "CATEGORY_PREDICTION_FILE",
    "PREDICTIONS_DIR",
    "PREDICTION_MANIFEST_FILE",
    "PREDICTION_MANIFEST_SCHEMA_VERSION",
    "QUALITY_REPORT_JSON_FILE",
    "QUALITY_REPORT_MD_FILE",
    "VALIDATION_RESULT_FILE",
    "PredictionFile",
    "PredictionLineage",
    "PredictionManifest",
    "PredictionScopeRole",
    "prediction_content_fingerprint",
    "scope_role_for",
]

#: The manifest contract's own version.
PREDICTION_MANIFEST_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Where publications live under the artifact root.
PREDICTIONS_DIR: Final[str] = "predictions"

#: The files one publication may contain.  Exactly these names, and a
#: publication carrying anything else is refused rather than tolerated: an
#: unexpected file in a verified directory is either a mistake or an attempt.
BINARY_PREDICTION_FILE: Final[str] = "binary_predictions.parquet"
CATEGORY_PREDICTION_FILE: Final[str] = "category_predictions.parquet"
ANOMALY_PREDICTION_FILE: Final[str] = "anomaly_scores.parquet"
VALIDATION_RESULT_FILE: Final[str] = "prediction_validation.json"
QUALITY_REPORT_JSON_FILE: Final[str] = "ml_quality.json"
QUALITY_REPORT_MD_FILE: Final[str] = "ml_quality.md"

#: The manifest, written last.
PREDICTION_MANIFEST_FILE: Final[str] = "prediction_manifest.json"

#: Namespace for derived prediction identifiers.  Fixed and distinct from the
#: experiment-record namespace, so a prediction identifier can never collide
#: with a run, selection, or freeze identifier however similar their content.
_NS_PREDICTION: Final[uuid.UUID] = uuid.UUID("5b7f2a10-3c94-5e6d-9a21-7c4b8e0d1f33")

_RELATIVE_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"application/json", "application/vnd.apache.parquet", "text/markdown"}
)


class PredictionScopeRole(BaseModel):
    """A typed marker separating a supervised scope from a generalisation probe.

    Declared as a model rather than an enum member on :class:`MLSplit` because
    it is a property of *this publication*, not of the split: the novel-anomaly
    holdout is a probe wherever it appears, and merging its rows into a
    supervised prediction artifact is exactly what this field exists to make
    impossible to do quietly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str

    @field_validator("role")
    @classmethod
    def check_role(cls, value: str) -> str:
        """Two roles, and no third."""
        if value not in {"supervised_prediction", "generalisation_probe"}:
            raise ValueError(f"unknown prediction scope role {value!r}")
        return value


def scope_role_for(scope: MLSplit) -> PredictionScopeRole:
    """Return the role predictions for *scope* are published under.

    The novel-anomaly holdout is a generalisation probe and is never a
    supervised prediction scope: it exists to be scored by a model that has
    never seen anything like it, and a report combining it with the test split
    would answer neither question.
    """
    if scope is MLSplit.EXCLUDED:
        raise ValueError(
            "excluded rows are excluded from every stage of this layer, "
            "prediction included"
        )
    if scope is MLSplit.NOVEL_ANOMALY_HOLDOUT:
        return PredictionScopeRole(role="generalisation_probe")
    return PredictionScopeRole(role="supervised_prediction")


class PredictionFile(BaseModel):
    """One file a prediction publication declares, and its integrity evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_name: str
    relative_path: str
    media_type: str
    sha256: Sha256Hex
    byte_size: int = Field(ge=0)
    #: Present for a row-oriented artifact, absent for a report.  ``None`` means
    #: "this file has no rows", never "zero rows".
    row_count: int | None = Field(default=None, ge=0)

    @field_validator("relative_path")
    @classmethod
    def check_path(cls, value: str) -> str:
        """A declared path is a bare file name in the publication directory.

        No separator, no traversal, no absolute path, and therefore nothing a
        publication can name outside itself.
        """
        if not _RELATIVE_PATH_RE.match(value):
            raise ValueError(f"relative_path {value!r} is not a safe file name")
        return value

    @field_validator("media_type")
    @classmethod
    def check_media_type(cls, value: str) -> str:
        """Only the formats a publication actually writes may be declared."""
        if value not in _MEDIA_TYPES:
            raise ValueError(
                f"media_type {value!r} must be one of {sorted(_MEDIA_TYPES)}"
            )
        return value


class PredictionLineage(BaseModel):
    """Every frozen thing a prediction is attributable to, named by fingerprint.

    Carried once on the manifest rather than once per row.  A hundred thousand
    copies of the same digest would be a hundred thousand chances for one of
    them to disagree, and the per-row copy that disagreed would be the one
    somebody read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    champion_lock_fingerprint: Sha256Hex
    champion_scope_key: Sha256Hex
    champion_freeze_record_id: str
    validation_selection_id: str

    training_run_id: str
    catalog_model_id: str
    model_id: str
    model_content_fingerprint: Sha256Hex
    model_manifest_fingerprint: Sha256Hex
    preprocessor_fingerprint: Sha256Hex
    calibration_method: CalibrationMethod
    calibration_state_fingerprint: Sha256Hex | None
    binary_threshold_fingerprint: Sha256Hex
    #: The score kind the frozen threshold was selected against, and the value
    #: it was frozen at.  Both on the manifest so a reader can check the rows
    #: against the operating point without opening the run that chose it.
    binary_score_kind: ScoreKind
    decision_threshold: float

    category_selection_id: str | None = None
    category_run_id: str | None = None
    category_model_id: str | None = None
    category_model_content_fingerprint: Sha256Hex | None = None
    category_preprocessor_fingerprint: Sha256Hex | None = None
    category_abstention_fingerprint: Sha256Hex | None = None
    category_class_order: tuple[str, ...] | None = None
    min_category_score: float | None = None

    anomaly_run_id: str | None = None
    anomaly_model_id: str | None = None
    anomaly_model_content_fingerprint: Sha256Hex | None = None
    anomaly_preprocessor_fingerprint: Sha256Hex | None = None
    anomaly_threshold_fingerprint: Sha256Hex | None = None

    required_feature_schema_version: str
    feature_catalog_fingerprint: Sha256Hex
    allowlist_fingerprint: Sha256Hex
    eligible_feature_list_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    model_catalog_fingerprint: Sha256Hex

    serializer_id: str
    serializer_version: int = Field(ge=1)
    inference_adapter_id: str
    dependency_contract_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_lineage(self) -> Self:
        """A calibrated threshold names a calibrator, and a head names all of itself."""
        if is_probability(self.binary_score_kind) != (
            self.calibration_state_fingerprint is not None
        ):
            raise ValueError(
                "a calibrated probability names the calibrator that produced it, "
                "and nothing else may"
            )
        if is_probability(self.binary_score_kind) != (
            self.calibration_method is not CalibrationMethod.NONE
        ):
            raise ValueError(
                "the calibration method and the score kind must agree about "
                "whether a calibrator was applied"
            )
        category = (
            self.category_selection_id,
            self.category_run_id,
            self.category_model_id,
            self.category_model_content_fingerprint,
            self.category_preprocessor_fingerprint,
            self.category_abstention_fingerprint,
            self.category_class_order,
            self.min_category_score,
        )
        present = [value is not None for value in category]
        if any(present) and not all(present):
            raise ValueError(
                "a frozen category head is named in full or not at all; a "
                "partial lineage is a lineage nobody can check"
            )
        if self.category_class_order is not None:
            if len(self.category_class_order) < 2:
                raise ValueError("a category head distinguishes at least two classes")
            if tuple(sorted(self.category_class_order)) != self.category_class_order:
                raise ValueError("category_class_order is not the deterministic order")
        anomaly = (
            self.anomaly_run_id,
            self.anomaly_model_id,
            self.anomaly_model_content_fingerprint,
            self.anomaly_preprocessor_fingerprint,
        )
        anomaly_present = [value is not None for value in anomaly]
        if any(anomaly_present) and not all(anomaly_present):
            raise ValueError(
                "an experimental anomaly lineage is named in full or not at all"
            )
        if (
            self.anomaly_threshold_fingerprint is not None
            and self.anomaly_run_id is None
        ):
            raise ValueError(
                "an anomaly threshold without an anomaly run is not a lineage"
            )
        return self


def prediction_content_fingerprint(
    *,
    binary: Sequence[BinaryPrediction],
    category: Sequence[CategoryPrediction] | None,
    anomaly: Sequence[AnomalyScore] | None,
) -> str:
    """Return the digest of exactly what a publication predicts.

    Over the rows as they will be published, in the canonical order they will be
    published in.  Changing a predicted value, a decision, a threshold, or a row's
    presence moves this digest; writing the same predictions to a different
    directory does not, because nothing about the directory takes part.

    The absent artifacts are represented as ``None`` rather than as empty lists:
    a publication with no category head and one whose head predicted nothing are
    different publications, and a digest that could not tell them apart would let
    the first be republished as the second.
    """
    payload = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "binary": [
            {
                "anchor_event_id": row.anchor_event_id,
                "anchor_event_time": row.anchor_event_time.isoformat(),
                "score_kind": str(row.score_kind),
                "malicious_decision_score": row.malicious_decision_score,
                "malicious_probability": row.malicious_probability,
                "decision_threshold": row.decision_threshold,
                "flagged_malicious": row.flagged_malicious,
            }
            for row in binary
        ],
        "category": (
            None
            if category is None
            else [
                {
                    "anchor_event_id": row.anchor_event_id,
                    "anchor_event_time": row.anchor_event_time.isoformat(),
                    "predicted_scenario": row.predicted_scenario,
                    "category_scores_json": row.category_scores_json,
                    "max_category_score": row.max_category_score,
                    "min_category_score": row.min_category_score,
                }
                for row in category
            ]
        ),
        "anomaly": (
            None
            if anomaly is None
            else [
                {
                    "anchor_event_id": row.anchor_event_id,
                    "anchor_event_time": row.anchor_event_time.isoformat(),
                    "anomaly_score": row.anomaly_score,
                    "anomaly_threshold": row.anomaly_threshold,
                    "flagged_anomalous": row.flagged_anomalous,
                    "experimental": row.experimental,
                    "influences_champion_selection": (
                        row.influences_champion_selection
                    ),
                }
                for row in anomaly
            ]
        ),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


class PredictionManifest(SealedModel):
    """What one prediction publication contains, and what produced it.

    Sealed: the digest is a field, recomputed on every construction and on every
    deserialization, so a manifest edited after publication is refused rather
    than believed.
    """

    fingerprint_field: ClassVar[str] = "prediction_manifest_fingerprint"
    schema_version_field: ClassVar[str] = "prediction_manifest_schema_version"
    schema_version: ClassVar[str] = PREDICTION_MANIFEST_SCHEMA_VERSION
    record_label: ClassVar[str] = "prediction manifest"

    prediction_manifest_schema_version: str = PREDICTION_MANIFEST_SCHEMA_VERSION
    prediction_schema_version: str = PREDICTION_SCHEMA_VERSION

    prediction_id: str
    prediction_content_fingerprint: Sha256Hex

    scope: MLSplit
    scope_role: PredictionScopeRole
    lineage: PredictionLineage

    #: The digest of exactly the rows that were scored, computed by the
    #: inference loader from the feature and split tables alone.
    inference_input_fingerprint: Sha256Hex
    split_membership_fingerprint: Sha256Hex

    row_count: int = Field(ge=1)
    #: ``None`` when no head was frozen, never ``0``.  A publication with no
    #: category artifact and one whose head scored nothing are different states.
    category_row_count: int | None = Field(default=None, ge=0)
    anomaly_row_count: int | None = Field(default=None, ge=0)

    files: tuple[PredictionFile, ...]

    #: The aggregate artifacts published beside the rows, bound by digest so a
    #: report cannot be swapped for one describing different predictions.
    validation_result_fingerprint: Sha256Hex | None = None
    quality_report_fingerprint: Sha256Hex | None = None

    prediction_manifest_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_manifest(self) -> Self:
        """A manifest describes one publication, coherently and completely."""
        if self.prediction_schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError(
                f"the manifest declares prediction schema version "
                f"{self.prediction_schema_version!r}; this build implements "
                f"{PREDICTION_SCHEMA_VERSION!r}"
            )
        if self.scope_role != scope_role_for(self.scope):
            raise ValueError(
                "the declared scope role is not the one this scope carries; a "
                "generalisation probe is never republished as a supervised scope"
            )
        names = [item.logical_name for item in self.files]
        if len(set(names)) != len(names):
            raise ValueError("a manifest declares each file once")
        if names != sorted(names):
            raise ValueError("declared files must be given in logical-name order")
        paths = [item.relative_path for item in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("two declared files share a path")
        declared = {item.logical_name for item in self.files}
        if BINARY_PREDICTION_FILE not in declared:
            raise ValueError(
                "a prediction publication always contains its binary predictions"
            )
        if (CATEGORY_PREDICTION_FILE in declared) != (
            self.category_row_count is not None
        ):
            raise ValueError(
                "a category artifact and a category row count are declared "
                "together, or neither is"
            )
        if (ANOMALY_PREDICTION_FILE in declared) != (
            self.anomaly_row_count is not None
        ):
            raise ValueError(
                "an anomaly artifact and an anomaly row count are declared "
                "together, or neither is"
            )
        if (self.category_row_count is None) != (self.lineage.category_run_id is None):
            raise ValueError(
                "a published category artifact names the frozen head that produced it"
            )
        if self.category_row_count is not None and (
            self.category_row_count > self.row_count
        ):
            raise ValueError(
                "more rows were categorised than were scored; category triage "
                "covers the binary-positive rows and cannot exceed them"
            )
        if (self.anomaly_row_count is None) != (self.lineage.anomaly_run_id is None):
            raise ValueError(
                "a published anomaly artifact names the experimental run that "
                "produced it"
            )
        binary_file = next(
            item for item in self.files if item.logical_name == BINARY_PREDICTION_FILE
        )
        if binary_file.row_count != self.row_count:
            raise ValueError(
                "the manifest's row count and its binary artifact's row count disagree"
            )
        if self.prediction_id != self.derived_prediction_id():
            raise ValueError(
                "the prediction identifier is not the one this manifest's "
                "content derives; an assigned identifier is not an identity"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        """Return the canonical semantics ``prediction_id`` is derived from.

        Deliberately **not** the whole manifest.  The per-file digests and byte
        sizes are integrity evidence about a particular writing of the rows; the
        identity is about the rows themselves, so re-publishing identical
        predictions through a writer whose page layout changed keeps the same
        identifier and produces a different manifest digest.
        """
        return {
            "prediction_manifest_schema_version": (
                self.prediction_manifest_schema_version
            ),
            "prediction_schema_version": self.prediction_schema_version,
            "scope": str(self.scope),
            "scope_role": self.scope_role.role,
            "lineage": self.lineage.model_dump(mode="json"),
            "inference_input_fingerprint": self.inference_input_fingerprint,
            "split_membership_fingerprint": self.split_membership_fingerprint,
            "prediction_content_fingerprint": self.prediction_content_fingerprint,
            "row_count": self.row_count,
            "category_row_count": self.category_row_count,
            "anomaly_row_count": self.anomaly_row_count,
        }

    def derived_prediction_id(self) -> str:
        """Return the identifier this manifest's semantics derive."""
        canonical = json.dumps(
            self.identity_payload(), sort_keys=True, ensure_ascii=True
        )
        return str(uuid.uuid5(_NS_PREDICTION, canonical))

    @classmethod
    def build(cls, **fields: Any) -> PredictionManifest:
        """Return a sealed manifest whose identifier its own content derives.

        The only supported way to build one.  A caller cannot supply
        ``prediction_id``: an identifier that could be assigned would let two
        different publications claim to be the same publication.
        """
        if "prediction_id" in fields:
            raise ValueError(
                "prediction_id is derived from the manifest's own content and "
                "cannot be supplied"
            )
        probe = cls.model_construct(
            **fields,
            prediction_id="00000000-0000-5000-8000-000000000000",
            prediction_manifest_fingerprint="0" * 64,
        )
        return cls.seal(**fields, prediction_id=probe.derived_prediction_id())


def _assert_manifest_carries_no_outcome_field() -> None:
    """Fail at import if the manifest grows a field that could hold a metric.

    A structural guard rather than a review note.  ``test_evaluation`` is a
    later milestone's record, and the way this manifest would stop being
    label-free is one plausible-sounding field at a time.
    """
    forbidden = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "true_positive_rate",
        "pr_auc",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "confusion_matrix",
        "test_metrics",
        "test_evaluation",
        "label_fingerprint",
        "labels",
        "y_true",
        "anchor_event_id",
        "anchor_event_ids",
        "campaign_id",
    }
    for model in (PredictionManifest, PredictionLineage, PredictionFile):
        offending = sorted(set(model.model_fields) & forbidden)
        if offending:
            raise ValueError(
                f"{model.__name__} declares field(s) a prediction manifest must "
                f"never carry: {offending}"
            )


_assert_manifest_carries_no_outcome_field()
