"""The serving-bundle artifact contract.

A bundle is what a deployment reads to execute the hybrid strategy validation
selected before TEST.  It is deliberately **thin**: it does not carry a model, a
preprocessor, a calibrator, or a threshold, because those already live in the
champion directory and duplicating them would create a second copy able to
disagree with the first.  What it carries is the one thing Phase 5 leaves
unpublished -- the fitted stacked meta-learner -- plus the lineage needed to
prove that this state belongs to *this* champion, this rule configuration, and
this frozen selection.

Three files, and the manifest is written last::

    <artifact_root>/serving/<champion_scope_key>/
        fusion_selection.json      the frozen FusionSelection, verbatim
        fusion_stacked_state.json  the reconstructed StackedFusionState (STACKED only)
        serving_bundle.json        the sealed manifest: lineage + file digests

**Everything is verified on load, and a failure is a refusal.**  The manifest
recomputes its own digest; each payload file must digest to what the manifest
records; the selection and the state each recompute their own seal; the state's
fingerprint must equal the one the selection named; and the manifest's declared
strategy must equal the selection's.  A bundle that fails any of these is
unusable rather than partially believed -- there is no repair path and no
override, because a partially trusted hybrid is indistinguishable from a sound
one in every response it would produce.

**Deterministic serialization.**  Every document is canonical JSON: sorted keys,
ASCII, no incidental whitespace, one trailing newline.  Two materializations of
one frozen selection write byte-identical bundles, which is what makes
publishing idempotent and re-publication detectable.

**No TEST quantity and no re-decision is representable here.**  There is no field
for a metric, a threshold, a candidate comparison, or a fallback strategy, and
:func:`_assert_no_test_or_override_field` fails at import if one appears.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Self

from pydantic import model_validator

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ExperimentPublicationError,
    ManifestVerificationError,
)
from password_attack_detector.ml.calibration import SealedModel
from password_attack_detector.ml.enums import (
    CalibrationMethod,
    FusionStrategy,
    ModelFamily,
)
from password_attack_detector.ml.fusion import (
    FUSION_SCHEMA_VERSION,
    FusionSelection,
    StackedFusionState,
)
from password_attack_detector.ml.schemas import ML_SCHEMA_VERSION, Sha256Hex

__all__ = [
    "BUNDLE_MANIFEST_FILE",
    "BUNDLE_SCHEMA_VERSION",
    "FUSION_SELECTION_FILE",
    "SERVING_BUNDLE_DIR",
    "STACKED_STATE_FILE",
    "ServingBundle",
    "ServingBundleManifest",
    "bundle_directory",
    "load_serving_bundle",
    "write_serving_bundle",
]

#: The bundle contract's own version, independent of every scientific contract
#: it references.  What a deployment needs to load can change without the fusion
#: contract changing, and the reverse.
BUNDLE_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The directory, under the artifact root, holding one bundle per champion scope.
SERVING_BUNDLE_DIR: Final[str] = "serving"

#: The sealed manifest.  Written last, so a directory carrying one is complete.
BUNDLE_MANIFEST_FILE: Final[str] = "serving_bundle.json"

#: The frozen fusion selection, copied verbatim out of the locked evaluation.
FUSION_SELECTION_FILE: Final[str] = "fusion_selection.json"

#: The reconstructed meta-learner.  Present for ``STACKED`` and for nothing else.
STACKED_STATE_FILE: Final[str] = "fusion_stacked_state.json"

#: Field names that would make a bundle a place to re-decide something.
_PROHIBITED_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "decision_threshold",
        "threshold",
        "fusion_threshold",
        "stacked_decision_threshold",
        "fallback_strategy",
        "default_strategy",
        "min_detection_rate",
        "max_false_positive_rate",
        "test_f1",
        "test_metrics",
        "test_label_fingerprint",
        "test_precision",
        "test_recall",
        "f1",
        "precision",
        "recall",
        "override",
    }
)


class ServingBundleManifest(SealedModel):
    """What one deployment bundle contains, and whose frozen state it belongs to.

    Sealed and fingerprinted like every other record in this project, and for the
    same reason: the digest is a *field*, so a reader can tell a tampered payload
    from an intact one without being handed the expected value separately.

    Its identity binds the whole deployable state -- the champion lineage, the
    rule configuration, the fusion selection, the reconstructed stacker, and the
    digests of the files beside it.  A changed champion, a changed rule
    configuration, or a changed stacker is therefore a *different* bundle rather
    than the same one with different contents.

    There is no metric here, no threshold, and no candidate comparison. A bundle
    describes what to execute, never how well it did: a deployment artifact
    carrying a TEST figure is one somebody eventually deploys *because of* the
    figure.
    """

    fingerprint_field: ClassVar[str] = "manifest_fingerprint"
    schema_version_field: ClassVar[str] = "bundle_schema_version"
    schema_version: ClassVar[str] = BUNDLE_SCHEMA_VERSION
    record_label: ClassVar[str] = "serving bundle manifest"

    bundle_schema_version: str = BUNDLE_SCHEMA_VERSION
    fusion_schema_version: str = FUSION_SCHEMA_VERSION
    ml_schema_version: str = ML_SCHEMA_VERSION

    # -- which frozen champion this bundle is deployable beside ---------------
    champion_scope_key: Sha256Hex
    champion_lock_fingerprint: Sha256Hex
    champion_freeze_record_id: str
    validation_selection_id: str
    catalog_model_id: str
    model_family: ModelFamily
    model_id: str
    model_content_fingerprint: Sha256Hex

    # -- the frozen decision pipeline, named rather than copied ---------------
    preprocessor_fingerprint: Sha256Hex
    calibration_method: CalibrationMethod
    calibration_state_fingerprint: Sha256Hex | None
    binary_threshold_fingerprint: Sha256Hex
    feature_catalog_fingerprint: Sha256Hex
    allowlist_fingerprint: Sha256Hex
    eligible_feature_list_fingerprint: Sha256Hex
    ml_config_fingerprint: Sha256Hex
    serializer_id: str
    serializer_version: int
    dependency_contract_fingerprint: Sha256Hex

    # -- the frozen hybrid ----------------------------------------------------
    #: The strategy validation selected.  Never ``None``: a bundle exists to make
    #: a *selected* hybrid executable, and there is nothing to deploy when none
    #: was selected.
    selected_fusion_strategy: FusionStrategy
    fusion_selection_fingerprint: Sha256Hex
    fusion_config_fingerprint: Sha256Hex
    rule_configuration_fingerprint: Sha256Hex
    validation_evidence_fingerprint: Sha256Hex
    #: Present exactly for ``STACKED``.  The digest Phase 5 sealed, which the
    #: reconstructed state had to recompute before this manifest was written.
    stacked_state_fingerprint: Sha256Hex | None = None
    oof_fold_definition_fingerprint: Sha256Hex | None = None
    oof_evidence_fingerprint: Sha256Hex | None = None
    base_model_recipe_fingerprint: Sha256Hex | None = None

    # -- where the frozen selection was read from ----------------------------
    #: The locked TEST evaluation whose sealed receipt named this selection. The
    #: bundle is downstream of it in provenance and upstream of nothing.
    evaluation_record_id: str
    evaluation_record_fingerprint: Sha256Hex

    #: Every payload file beside this manifest, and its digest.
    files: tuple[tuple[str, str], ...]

    manifest_fingerprint: Sha256Hex

    @model_validator(mode="after")
    def check_manifest(self) -> Self:
        """A stacked bundle carries its state; a gate bundle carries none."""
        declared = dict(self.files)
        if FUSION_SELECTION_FILE not in declared:
            raise ValueError(
                "a serving bundle always carries the frozen fusion selection; a "
                "bundle that named a strategy without it would be a strategy "
                "nobody could trace to a selection"
            )
        stacked = self.selected_fusion_strategy is FusionStrategy.STACKED
        if stacked != (self.stacked_state_fingerprint is not None):
            raise ValueError(
                "a stacked bundle names the fitted state it applies, and a "
                "boolean gate names none; there is no fitted gate"
            )
        if stacked != (STACKED_STATE_FILE in declared):
            raise ValueError(
                "a stacked bundle carries its fitted state as a file, and a "
                "boolean gate carries no state file; a gate bundle with one "
                "would be deploying a stacker nobody selected"
            )
        if len(declared) != len(self.files):
            raise ValueError("each bundle file is declared once")
        for name, digest_value in self.files:
            if not name or "/" in name or "\\" in name or name.startswith("."):
                raise ValueError(
                    f"a bundle file name is a plain file beside the manifest; "
                    f"{name!r} is a path"
                )
            if len(digest_value) != 64:
                raise ValueError("a bundle file digest is a SHA-256 hex digest")
        return self


@dataclass(frozen=True, slots=True)
class ServingBundle:
    """A loaded, fully verified bundle: what to execute, and what it belongs to."""

    manifest: ServingBundleManifest
    selection: FusionSelection
    #: The reconstructed meta-learner, for ``STACKED`` and for nothing else.
    stacked_state: StackedFusionState | None

    @property
    def strategy(self) -> FusionStrategy:
        """Return the frozen strategy this bundle makes executable."""
        return self.manifest.selected_fusion_strategy

    @property
    def stacked(self) -> bool:
        """Return whether this bundle deploys a fitted meta-learner."""
        return self.strategy is FusionStrategy.STACKED


def bundle_directory(root: Path, *, scope_key: str) -> Path:
    """Return the bundle directory for one champion scope under *root*."""
    if not scope_key or "/" in scope_key or "\\" in scope_key or ".." in scope_key:
        raise ManifestVerificationError(
            "a champion scope key is a single directory name; a bundle is never "
            "read from a path a caller composed"
        )
    return Path(root) / SERVING_BUNDLE_DIR / scope_key


def _canonical(document: SealedModel) -> str:
    """Return the exact bytes one sealed document is stored as."""
    return document.to_json() + "\n"


def _digest_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text* as stored."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bundle_files(
    *, selection: FusionSelection, stacked_state: StackedFusionState | None
) -> dict[str, str]:
    """Return every payload file a bundle carries, keyed by name.

    The single place the payload set is decided, so the manifest's declaration
    and the bytes on disk cannot be assembled from two different opinions.
    """
    files = {FUSION_SELECTION_FILE: _canonical(selection)}
    if stacked_state is not None:
        files[STACKED_STATE_FILE] = _canonical(stacked_state)
    return files


def write_serving_bundle(
    *,
    root: Path,
    manifest: ServingBundleManifest,
    selection: FusionSelection,
    stacked_state: StackedFusionState | None,
) -> tuple[Path, bool]:
    """Publish *manifest* and its payloads transactionally.  Manifest last.

    Staged into a sibling directory, verified by being read back, then promoted
    with a single atomic rename.  The ordering matters for the reason it matters
    everywhere else in this project: a manifest asserting payloads that are not
    there is worse than payloads nobody has indexed yet, because the first reads
    as a complete bundle and the second reads as no bundle at all.

    Returns:
        The bundle directory, and whether this call created it.  Publishing an
        identical bundle twice writes nothing the second time.

    Raises:
        ExperimentPublicationError: when staging or promotion fails, or when a
            *different* bundle already occupies this scope.  Nothing is
            overwritten and there is no ``--force``.
    """
    payloads = bundle_files(selection=selection, stacked_state=stacked_state)
    declared = dict(manifest.files)
    if declared != {name: _digest_text(body) for name, body in payloads.items()}:
        raise ExperimentPublicationError(
            "the manifest does not declare the payloads it is being published "
            "with; a bundle whose index and contents disagree is refused rather "
            "than reconciled"
        )

    target = bundle_directory(root, scope_key=manifest.champion_scope_key)
    if target.exists():
        _require_identical_manifest(target, manifest)
        return (target, False)

    staging = target.parent / f".staging-{manifest.champion_scope_key}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    promoted = False
    try:
        for name, body in sorted(payloads.items()):
            (staging / name).write_text(body, encoding="utf-8")
        (staging / BUNDLE_MANIFEST_FILE).write_text(
            _canonical(manifest), encoding="utf-8"
        )
        _verify_staged(staging, manifest=manifest, payloads=payloads)
        staging.rename(target)
        promoted = True
    except ExperimentPublicationError:
        raise
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the serving bundle could not be staged ({type(exc).__name__}); the "
            f"destination is unchanged"
        ) from None
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)
    return (target, True)


def _verify_staged(
    staging: Path, *, manifest: ServingBundleManifest, payloads: dict[str, str]
) -> None:
    """Raise unless the staged bundle reads back as exactly itself."""
    for name, body in payloads.items():
        stored = (staging / name).read_text(encoding="utf-8")
        if stored != body:
            raise ExperimentPublicationError(
                "a staged bundle payload does not read back as itself"
            )
        if dict(manifest.files).get(name) != _digest_text(stored):
            raise ExperimentPublicationError(
                "a staged payload's digest is not the one the manifest records"
            )
    reloaded = ServingBundleManifest.from_json(
        (staging / BUNDLE_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    if reloaded.to_json() != manifest.to_json():
        raise ExperimentPublicationError(
            "the staged bundle manifest does not read back as itself"
        )


def _require_identical_manifest(
    target: Path, manifest: ServingBundleManifest
) -> ServingBundleManifest:
    """Return the stored manifest when it is byte-identical, or raise."""
    path = target / BUNDLE_MANIFEST_FILE
    if not path.is_file():
        raise ExperimentPublicationError(
            "a serving bundle directory already exists here without a manifest; "
            "an incomplete publication is never completed in place"
        )
    try:
        stored = ServingBundleManifest.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentPublicationError(
            f"the published serving bundle manifest is not readable "
            f"({type(exc).__name__})"
        ) from None
    if stored.to_json() != manifest.to_json():
        raise ExperimentPublicationError(
            "a different serving bundle is already published for this champion "
            "scope; a deployed hybrid is not rewritten in place"
        )
    return stored


def load_serving_bundle(root: Path, *, scope_key: str) -> ServingBundle:
    """Read and fully verify the bundle published for one champion scope.

    Every check is performed before anything is returned, and each only assumes
    what the previous one established:

    1. the directory and the manifest exist;
    2. the manifest parses at this contract version and recomputes its own seal;
    3. every payload the manifest declares is present, and no undeclared payload
       sits beside them;
    4. each payload digests to exactly what the manifest records;
    5. the fusion selection recomputes its own seal, and its fingerprint is the
       one the manifest names;
    6. the selection actually selected the strategy the manifest declares;
    7. for ``STACKED``, the state recomputes its own seal and its fingerprint is
       the one *the selection* named -- not merely the one the manifest did.

    Step 7 is the load-bearing one.  It is what makes a served stacker provably
    the stacker validation chose, rather than a stacker somebody published.

    Raises:
        ArtifactNotFoundError: when no bundle is published for this scope.
        ManifestVerificationError: on any verification failure above.
    """
    directory = bundle_directory(root, scope_key=scope_key)
    path = directory / BUNDLE_MANIFEST_FILE
    if not directory.is_dir() or not path.is_file():
        raise ArtifactNotFoundError(
            "no serving bundle is published for this champion scope"
        )

    try:
        manifest = ServingBundleManifest.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManifestVerificationError(
            f"the serving bundle manifest is not valid ({type(exc).__name__})"
        ) from None

    declared = dict(manifest.files)
    present = {
        item.name
        for item in directory.iterdir()
        if item.is_file() and item.name != BUNDLE_MANIFEST_FILE
    }
    if present != set(declared):
        raise ManifestVerificationError(
            "the serving bundle's files and its manifest disagree; an undeclared "
            "or missing payload makes the manifest an index of something else"
        )
    bodies: dict[str, str] = {}
    for name, recorded in sorted(declared.items()):
        body = (directory / name).read_text(encoding="utf-8")
        if _digest_text(body) != recorded:
            raise ManifestVerificationError(
                f"a serving bundle payload does not digest to the value the "
                f"manifest records for {name!r}"
            )
        bodies[name] = body

    try:
        selection = FusionSelection.from_json(bodies[FUSION_SELECTION_FILE])
    except Exception as exc:
        raise ManifestVerificationError(
            f"the bundled fusion selection is not valid ({type(exc).__name__})"
        ) from None
    if selection.selection_fingerprint != manifest.fusion_selection_fingerprint:
        raise ManifestVerificationError(
            "the bundled fusion selection is not the one the manifest names"
        )
    if selection.selected_strategy is not manifest.selected_fusion_strategy:
        raise ManifestVerificationError(
            "the bundled selection did not select the strategy this bundle "
            "deploys; a bundle never substitutes a strategy for the selected one"
        )

    stacked: StackedFusionState | None = None
    if manifest.selected_fusion_strategy is FusionStrategy.STACKED:
        try:
            stacked = StackedFusionState.from_json(bodies[STACKED_STATE_FILE])
        except Exception as exc:
            raise ManifestVerificationError(
                f"the bundled stacked state is not valid ({type(exc).__name__})"
            ) from None
        if stacked.state_fingerprint != selection.stacked_state_fingerprint:
            raise ManifestVerificationError(
                "the bundled stacked state is not the state the frozen selection "
                "named; a stacker that is not the selected one is not the hybrid "
                "this deployment claims to run"
            )
        if stacked.state_fingerprint != manifest.stacked_state_fingerprint:
            raise ManifestVerificationError(
                "the bundled stacked state is not the one the manifest names"
            )

    return ServingBundle(manifest=manifest, selection=selection, stacked_state=stacked)


def _assert_no_test_or_override_field() -> None:
    """Fail at import if the bundle grows somewhere to re-decide or to boast.

    A deployment artifact with a threshold field is a deployment artifact
    somebody sets a threshold in, and one with a TEST metric is one somebody
    deploys because of the metric.  Both names are refused rather than reviewed
    for.
    """
    offending = sorted(
        set(ServingBundleManifest.model_fields) & _PROHIBITED_MANIFEST_FIELDS
    )
    if offending:
        raise ValueError(
            f"ServingBundleManifest declares field(s) {offending}; a serving "
            f"bundle names frozen state and neither restates a decision nor "
            f"carries a result"
        )


_assert_no_test_or_override_field()
