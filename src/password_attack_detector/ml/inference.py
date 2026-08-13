"""Loading a model artifact: fail closed, verify everything, construct last.

A model directory is **untrusted data**.  It may have been copied from another
machine, edited by hand, produced by a different build, or written by somebody
who wanted this process to do something for them.  So nothing is constructed
until everything has been checked, and the checks run in an order where each
one only assumes what the previous ones established.

The full order, and what each step protects against:

===  ===================================  ==========================================
1    directory exists                     a typo pointing at nothing
2    exactly the declared files           a smuggled extra file
3    no symbolic links                    a member pointing outside the directory
4    size ceilings                        a decompression or allocation bomb
5    manifest parses, supported version   a manifest from a contract this build
                                          does not implement
6    every recorded digest matches        a tampered or truncated file
7    ``model.json`` parses                a malformed or over-permissive document
8    manifest and document agree          an artifact assembled from two models
9    archive holds the declared arrays    a swapped or padded archive
10   model identity recomputes            an artifact whose identifier was assigned
11   serializer id and version supported  a reader that cannot read this writer
12   inference adapter id supported       an adapter identifier nobody implements
13   family is in the closed registry     a family name selecting arbitrary code
14   catalog entry agrees                 an artifact contradicting the reviewed
                                          catalog
15   runtime scikit-learn in range        internals that may have moved
16   feature contract fingerprints match  a model scored against a different
                                          feature set
17   preprocessor fingerprint matches     a matrix built by different rules
18   transformed order matches            the right columns in the wrong order
19   class order valid                    a score vector with no meaning
===  ===================================  ==========================================

Only then is the adapter constructed, and it is constructed by **dictionary
lookup in a closed registry**.  No ``importlib``, no ``eval``, no ``exec``, no
attribute path from artifact content, and no pickle: the strings in a model
directory select between implementations this build already contains, or they
select nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ManifestVerificationError,
    ModelNotReadyError,
)
from password_attack_detector.ml.dependencies import (
    installed_version,
    sklearn_compatible,
)
from password_attack_detector.ml.enums import MLTask, ModelFamily
from password_attack_detector.ml.manifest import (
    ModelManifest,
    read_model_manifest,
    verify_model_artifact,
)
from password_attack_detector.ml.models.base import FittedModel
from password_attack_detector.ml.npz import read_npz_bytes
from password_attack_detector.ml.preprocessing import FittedPreprocessor
from password_attack_detector.ml.schemas import ML_SCHEMA_VERSION
from password_attack_detector.ml.serialization import (
    ARRAYS_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    ModelDocument,
    read_model_document,
)

__all__ = ["InferenceModel", "ModelCompatibility"]


@dataclass(frozen=True, slots=True)
class ModelCompatibility:
    """The runtime contract a caller may require a model to have been fitted under.

    Every field is optional and every supplied field is checked. Omitting one
    means "do not check this", which is honest -- a verification that silently
    skipped an absent expectation while reporting success would be worse than
    one that says it checked nothing.
    """

    eligible_feature_list_fingerprint: str | None = None
    preprocessor_fingerprint: str | None = None
    transformed_feature_names: tuple[str, ...] | None = None
    feature_catalog_fingerprint: str | None = None
    allowlist_fingerprint: str | None = None
    required_feature_schema_version: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceModel:
    """A verified model, ready to score, with no estimator anywhere in it."""

    manifest: ModelManifest
    document: ModelDocument
    fitted: FittedModel
    preprocessor: FittedPreprocessor
    adapter: Any

    @property
    def model_id(self) -> str:
        """Return the derived model identifier."""
        return self.document.model_id

    @property
    def family(self) -> ModelFamily:
        """Return the model family."""
        return self.document.model_family

    @property
    def task(self) -> MLTask:
        """Return the task this model was fitted for."""
        return self.document.task

    @property
    def score_columns(self) -> tuple[str, ...]:
        """Return the columns :meth:`score` emits, in order."""
        return self.fitted.score_columns

    def score(
        self, rows: Sequence[Sequence[float]], columns: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        """Score an already-transformed matrix.

        Takes a transformed matrix rather than raw feature rows: transforming is
        the preprocessor's job, the preprocessor is right here on this object,
        and a scoring path that also transformed would make it possible to score
        a matrix built by a different one.
        """
        scored: tuple[tuple[float, ...], ...] = self.adapter.score(
            self.fitted, rows, columns
        )
        return scored

    def transform_and_score(self, frame: Any) -> tuple[tuple[float, ...], ...]:
        """Transform *frame* with this model's own preprocessor, then score it."""
        matrix = self.preprocessor.transform(frame)
        return self.score(matrix.rows, matrix.output_feature_names)

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        compatibility: ModelCompatibility | None = None,
        require_champion_eligible: bool = False,
    ) -> InferenceModel:
        """Verify *directory* completely, then construct an adapter for it.

        Args:
            directory: the published model directory.
            compatibility: the runtime feature and preprocessing contract the
                model must have been fitted under. Omit to skip those checks.
            require_champion_eligible: refuse a model the catalog does not
                consider promotable.

        Raises:
            ArtifactNotFoundError: the directory does not exist.
            ManifestVerificationError: any structural or integrity check failed.
            ModelNotReadyError: the artifact is intact but this runtime cannot
                honour it -- an unsupported serializer, an unimplemented family,
                an out-of-range scikit-learn, or a contract mismatch.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ArtifactNotFoundError(
                "the model directory does not exist or is not a directory"
            )

        # Steps 1-10. Structural, integrity, and internal-agreement checks. The
        # outcome carries a stable code and no path.
        outcome = verify_model_artifact(directory)
        if not outcome.passed:
            raise ManifestVerificationError(
                f"model artifact verification failed [{outcome.error_code}]: "
                f"{outcome.error_detail}"
            )

        manifest = read_model_manifest(
            json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        )
        document = read_model_document(
            json.loads((directory / MODEL_FILE).read_text(encoding="utf-8"))
        )

        # Step 11-12. A reader that cannot read this writer must say so rather
        # than interpret the bytes optimistically.
        from password_attack_detector.ml.models import (
            MODEL_IMPLEMENTATIONS,
            adapter_class_for,
            catalog_spec_for,
        )

        if document.ml_schema_version != ML_SCHEMA_VERSION:
            raise ModelNotReadyError(
                f"the artifact declares ML schema version "
                f"{document.ml_schema_version!r}; this build implements "
                f"{ML_SCHEMA_VERSION!r}"
            )

        # Step 13. Closed-registry dispatch. A family name that is not a key
        # here selects nothing at all; there is no import path from artifact
        # content to code.
        if document.model_family not in MODEL_IMPLEMENTATIONS:
            raise ModelNotReadyError(
                f"model family {str(document.model_family)!r} is not implemented "
                f"by this build"
            )
        adapter_class = adapter_class_for(document.model_family)

        if document.serializer_id != getattr(adapter_class, "serializer_id", None):
            raise ModelNotReadyError(
                f"the artifact was written by serializer "
                f"{document.serializer_id!r}, which is not the one this build's "
                f"{str(document.model_family)!r} implementation reads"
            )
        if document.serializer_version != getattr(
            adapter_class, "serializer_version", None
        ):
            raise ModelNotReadyError(
                f"the artifact declares serializer version "
                f"{document.serializer_version}, which this build does not read"
            )
        if document.inference_adapter_id != getattr(
            adapter_class, "inference_adapter_id", None
        ):
            raise ModelNotReadyError(
                f"the artifact names inference adapter "
                f"{document.inference_adapter_id!r}, which this build does not "
                f"implement"
            )

        # Step 14. The reviewed catalog is the authority on what a family is.
        spec = catalog_spec_for(document.model_family)
        if spec.model_id != document.catalog_model_id:
            raise ModelNotReadyError(
                "the artifact names a catalog entry that does not describe its family"
            )
        if document.task not in spec.supported_tasks:
            raise ModelNotReadyError(
                f"the catalog does not declare task {str(document.task)!r} for "
                f"this family"
            )
        if require_champion_eligible and not (
            document.champion_eligible and spec.champion_eligible
        ):
            raise ModelNotReadyError(
                "the artifact is not champion-eligible; a family whose "
                "serializer contract rests on undocumented internals is never "
                "promoted, however well it scores"
            )

        # Step 15. Estimator internals move between minor series, and this
        # artifact's arrays were extracted from one of them.
        if spec.requires_sklearn:
            resolved = manifest.scikit_learn_version
            if resolved is not None and not sklearn_compatible(resolved):
                raise ModelNotReadyError(
                    "the artifact was produced under a scikit-learn release "
                    "outside the reviewed range"
                )
            if not sklearn_compatible(installed_version("scikit-learn")):
                raise ModelNotReadyError(
                    "the installed scikit-learn lies outside the range this "
                    "artifact declares compatibility with"
                )

        # Steps 16-18. What the model was fitted against must be what it is
        # about to be used with.
        preprocessor = FittedPreprocessor.from_json(
            (directory / PREPROCESSOR_FILE).read_text(encoding="utf-8")
        )
        if preprocessor.fingerprint() != document.preprocessor_fingerprint:
            raise ModelNotReadyError(
                "the stored preprocessor is not the one this model was fitted against"
            )
        if preprocessor.output_feature_names != document.transformed_feature_names:
            raise ModelNotReadyError(
                "the stored preprocessor emits a different transformed column "
                "order from the one the model was fitted on"
            )

        expected = compatibility or ModelCompatibility()
        _require_match(
            "eligible feature list fingerprint",
            expected.eligible_feature_list_fingerprint,
            document.eligible_feature_list_fingerprint,
        )
        _require_match(
            "preprocessor fingerprint",
            expected.preprocessor_fingerprint,
            document.preprocessor_fingerprint,
        )
        _require_match(
            "feature catalog fingerprint",
            expected.feature_catalog_fingerprint,
            manifest.feature_catalog_fingerprint,
        )
        _require_match(
            "allowlist fingerprint",
            expected.allowlist_fingerprint,
            manifest.allowlist_fingerprint,
        )
        _require_match(
            "required feature schema version",
            expected.required_feature_schema_version,
            manifest.required_feature_schema_version,
        )
        if (
            expected.transformed_feature_names is not None
            and tuple(expected.transformed_feature_names)
            != document.transformed_feature_names
        ):
            raise ModelNotReadyError(
                "the runtime transformed feature order disagrees with the one "
                "this model was fitted on; the same columns in a different "
                "order are a different matrix"
            )

        # Step 19. A score vector is only meaningful beside its class order.
        if document.task is MLTask.ANOMALY:
            if document.class_order:
                raise ModelNotReadyError(
                    "an anomaly model declares classes, which it cannot have"
                )
        elif len(document.class_order) < 2:
            raise ModelNotReadyError(
                "a supervised model declares fewer than two classes"
            )

        arrays = read_npz_bytes((directory / ARRAYS_FILE).read_bytes())
        fitted = FittedModel(
            serializer_id=document.serializer_id,
            serializer_version=document.serializer_version,
            inference_adapter_id=document.inference_adapter_id,
            catalog_model_id=document.catalog_model_id,
            family=document.model_family,
            task=document.task,
            class_order=document.class_order,
            raw_feature_names=document.raw_feature_names,
            transformed_feature_names=document.transformed_feature_names,
            hyperparameters=document.hyperparameters,
            parameters=document.parameters,
            arrays=arrays,
            score_semantics=document.score_semantics,
            train_row_count=document.train_row_count,
            preprocessor_fingerprint=document.preprocessor_fingerprint,
            eligible_feature_list_fingerprint=(
                document.eligible_feature_list_fingerprint
            ),
            class_weight_fingerprint=document.class_weight_fingerprint,
            champion_eligible=document.champion_eligible,
            experimental=document.experimental,
        )
        if fitted.content_fingerprint() != document.model_content_fingerprint:
            raise ManifestVerificationError(
                "the reconstructed model content does not reproduce the "
                "recorded fingerprint"
            )

        return cls(
            manifest=manifest,
            document=document,
            fitted=fitted,
            preprocessor=preprocessor,
            adapter=_construct(adapter_class, document),
        )


def _construct(adapter_class: type, document: ModelDocument) -> Any:
    """Return an adapter instance for scoring.

    Constructed with no arguments wherever the family allows it: scoring reads
    the artifact, not the constructor. The one exception is the threshold
    baseline, whose bound column is part of what it *is*, and that value comes
    from the verified document rather than from anything a caller supplied.
    """
    from password_attack_detector.ml.models import SingleFeatureThresholdAdapter

    if adapter_class is SingleFeatureThresholdAdapter:
        feature = document.parameters.get("feature")
        if not isinstance(feature, str) or feature not in (
            document.transformed_feature_names
        ):
            raise ModelNotReadyError(
                "the threshold baseline names a column that is not one of its "
                "own transformed features"
            )
        return SingleFeatureThresholdAdapter(feature=feature)
    return adapter_class()


def _require_match(what: str, expected: str | None, actual: str | None) -> None:
    """Raise when a supplied expectation disagrees with the artifact."""
    if expected is None:
        return
    if expected != actual:
        raise ModelNotReadyError(
            f"the artifact's {what} disagrees with the runtime contract it was "
            f"asked to satisfy"
        )
